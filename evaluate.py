"""
Quality of the speech denoiser: PESQ, STOI and ESTOI, and the composite
measures VoiceBank-DEMAND papers report (CSIG, CBAK, COVL and segmental SNR,
see denoiser/metrics.py). Scores the noisy input and the denoised output
against the clean reference, so the improvement the model brings is visible
next to the baseline.

  .venv/bin/python evaluate.py                       # score <train.output_dir>/samples/*/
  .venv/bin/python evaluate.py --model runs/default/model.pt --num-clips 100

With --model, the clean and noisy scores don't depend on the model, so they
are cached per clip in runs/eval_cache.json and only computed once.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq, PesqError
from pystoi import stoi

from denoiser.config import add_config_args, get_device, load_config, resolve_path
from denoiser.data import load_pairs, split_files
from denoiser.metrics import composite
from denoiser.model import describe_model, load_checkpoint
from denoiser.pipeline import denoise
from denoiser.snn.stats import CostMeter, describe_cost

SAMPLE_RATE = 16000
CACHE_PATH = "runs/eval_cache.json"


def compute_stoi(clean, est, sr=SAMPLE_RATE, extended=False):
    """STOI (0..1, higher is better). extended=True gives ESTOI."""
    length = min(len(clean), len(est))
    return float(stoi(clean[:length], est[:length], sr, extended=extended))


def compute_pesq(clean, est, sr=SAMPLE_RATE, mode="wb"):
    """PESQ (-0.5..4.5, higher is better). 'wb' needs 16 kHz, 'nb' 8 or 16 kHz.
    Returns nan when PESQ can't score the clip (e.g. no speech detected)."""
    length = min(len(clean), len(est))
    try:
        return float(pesq(sr, clean[:length], est[:length], mode))
    except PesqError:
        return math.nan


METRICS = ("pesq", "stoi", "estoi", "csig", "cbak", "covl", "ssnr")


def score_pair(clean, est, sr=SAMPLE_RATE):
    """Every measure in METRICS, of `est` against the reference `clean`."""
    pesq_wb = compute_pesq(clean, est, sr)
    return {
        "pesq": pesq_wb,
        "stoi": compute_stoi(clean, est, sr),
        "estoi": compute_stoi(clean, est, sr, extended=True),
        **composite(clean, est, pesq_wb, sr),
    }


def evaluate_estimates(items, sr=SAMPLE_RATE):
    """items: iterable of (clean, {name: estimate}) float waveforms.
    Returns {name: {metric: mean}} in the order the names first appear."""
    scores = {}
    for clean, estimates in items:
        for name, est in estimates.items():
            scores.setdefault(name, []).append(score_pair(clean, est, sr))
    return mean_scores(scores)


def mean_scores(scores):
    """{name: [per-clip scores]} -> {name: {metric: mean}}"""
    return {
        name: {m: float(np.nanmean([r[m] for r in rows])) for m in rows[0]}
        for name, rows in scores.items()
    }


def evaluate_triples(triples, sr=SAMPLE_RATE):
    """triples: iterable of (clean, noisy, denoised) float waveforms.
    "clean" scores the reference against itself: the metric ceiling."""
    return evaluate_estimates(
        ((clean, {"clean": clean, "noisy": noisy, "denoised": denoised})
         for clean, noisy, denoised in triples), sr)


def load_samples_dir(samples_dir):
    """Yield (clean, noisy, denoised) from the folders save_samples writes."""
    for d in sorted(p for p in Path(samples_dir).iterdir() if p.is_dir()):
        clean, _ = sf.read(str(d / "clean.wav"), dtype="float32")
        noisy, _ = sf.read(str(d / "noisy.wav"), dtype="float32")
        denoised, _ = sf.read(str(d / "denoised.wav"), dtype="float32")
        yield clean, noisy, denoised


def evaluate_model(denoise_fn, pairs, cached=(), after_noisy=lambda: None):
    """Also runs the clean clip through the model ("clean->model"), which
    shows how much the model distorts speech that has no noise to remove.
    cached: per-clip {"clean", "noisy"} scores from an earlier run, used for
    the first len(cached) clips. after_noisy() is called after each noisy clip
    has gone through the model. Returns (results, per-clip clean/noisy scores)."""
    scores, fixed = {}, []
    for i, (noisy, clean) in enumerate(pairs):
        row = cached[i] if i < len(cached) else {
            "clean": score_pair(clean, clean), "noisy": score_pair(clean, noisy)}
        fixed.append(row)
        denoised = denoise_fn(noisy)
        after_noisy()
        row = {**row, "denoised": score_pair(clean, denoised),
               "clean->model": score_pair(clean, denoise_fn(clean))}
        for name, s in row.items():
            scores.setdefault(name, []).append(s)
    return mean_scores(scores), fixed


def cache_key(files, sample_rate):
    """Changes when the test files or the measures do, so a stale cache isn't used."""
    return [sample_rate, list(METRICS)] + [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]


def load_cache(key, cache_path=CACHE_PATH):
    try:
        data = json.loads(resolve_path(cache_path).read_text())
    except (OSError, ValueError):
        return []
    return data["clips"] if data.get("key") == key else []


def save_cache(key, clips, cache_path=CACHE_PATH):
    path = resolve_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "clips": clips}))


def print_results(results):
    print(f"{'':14s}" + "".join(f"{m.upper():>8s}" for m in METRICS))
    for kind, m in results.items():
        print(f"{kind:14s}" + "".join(f"{m[name]:8.3f}" for name in METRICS))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-dir", default=None,
                    help="Score saved clean/noisy/denoised triples (default mode; "
                         "defaults to <train.output_dir>/samples).")
    ap.add_argument("--model", default="runs/default/model.pt",
                    help="Instead, run this model.pt over test clips and score it.")
    ap.add_argument("--num-clips", type=int, default=824)
    ap.add_argument("--split", default="test", choices=("test", "val"),
                    help="val: for choosing between models, so the test set stays untouched.")
    ap.add_argument("--every", type=int, default=1, help="Score each Nth clip of the split only.")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set)

    if args.model:
        device = get_device(cfg)
        model, model_cfg = load_checkpoint(args.model, cfg, device)
        print(f"Evaluating {args.model} on up to {args.num_clips} {args.split} clips"
              + (f", each {args.every}th" if args.every > 1 else "") + "...")
        print(f"  {describe_model(model_cfg)}")
        files, sr = split_files(cfg, args.split), cfg["data"]["sample_rate"]
        key = cache_key(files, sr) + ([args.every] if args.every > 1 else [])
        cache = CACHE_PATH if (args.split, args.every) == ("test", 1) else \
            CACHE_PATH.replace(".json", f"_{args.split}_{args.every}.json")
        cached = load_cache(key, cache)
        meter = CostMeter(model)  # counts a spiking model's spikes on the noisy clips
        results, fixed = evaluate_model(lambda wav: denoise(model, wav, model_cfg, device),
                                        load_pairs(files, args.num_clips, sr, every=args.every), cached, meter.add)
        if len(fixed) > len(cached):
            save_cache(key, fixed, cache)
        print(f"  cost: {describe_cost(meter.cost(), model_cfg)}")
    else:
        samples_dir = args.samples_dir or resolve_path(cfg["train"]["output_dir"]) / "samples"
        print(f"Evaluating samples in {samples_dir}...")
        results = evaluate_triples(load_samples_dir(samples_dir))
    print_results(results)


if __name__ == "__main__":
    main()
