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

The scores are computed in worker processes (--workers, default: one per CPU
core, at most MAX_WORKERS), so with --model they run while the model works on
the next clips.
"""
import argparse
import json
import math
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
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
MAX_WORKERS = 8  # default limit: on Windows every worker loads its own copy of PyTorch

Wave = np.ndarray
Scores = dict[str, float]  # metric -> score


def compute_stoi(clean: Wave, est: Wave, sr: int = SAMPLE_RATE, extended: bool = False) -> float:
    """STOI (0..1, higher is better). extended=True gives ESTOI."""
    length = min(len(clean), len(est))
    return float(stoi(clean[:length], est[:length], sr, extended=extended))


def compute_pesq(clean: Wave, est: Wave, sr: int = SAMPLE_RATE, mode: str = "wb") -> float:
    """PESQ (-0.5..4.5, higher is better). 'wb' needs 16 kHz, 'nb' 8 or 16 kHz.
    Returns nan when PESQ can't score the clip (e.g. no speech detected)."""
    length = min(len(clean), len(est))
    try:
        return float(pesq(sr, clean[:length], est[:length], mode))
    except PesqError:
        return math.nan


METRICS = ("pesq", "stoi", "estoi", "csig", "cbak", "covl", "ssnr")


def score_pair(clean: Wave, est: Wave, sr: int = SAMPLE_RATE) -> Scores:
    """Every measure in METRICS, of `est` against the reference `clean`."""
    pesq_wb = compute_pesq(clean, est, sr)
    return {
        "pesq": pesq_wb,
        "stoi": compute_stoi(clean, est, sr),
        "estoi": compute_stoi(clean, est, sr, extended=True),
        **composite(clean, est, pesq_wb, sr),
    }


def scorer(workers: int | None = None) -> ProcessPoolExecutor:
    """A pool of `workers` processes for score_pair (None: one per CPU core, at
    most MAX_WORKERS). The workers are started fresh, never forked from this
    process, which may have started the GPU and PyTorch's threads."""
    methods = multiprocessing.get_all_start_methods()
    method = next((m for m in ("forkserver", "spawn") if m in methods), None)
    if method is None:
        raise RuntimeError(f"Need the 'forkserver' or 'spawn' start method, this system has {methods}")
    max_workers = (
        min(os.cpu_count() or 1, MAX_WORKERS)
        if workers is None
        else workers
    )
    return ProcessPoolExecutor(max_workers, mp_context=multiprocessing.get_context(method))


def resolved(rows: list[dict[str, Scores | Future]]) -> list[dict[str, Scores]]:
    """[{name: scores or Future of them}] -> [{name: scores}]"""
    return [
        {
            name: value.result() if isinstance(value, Future) else value
            for name, value in row.items()
        }
        for row in rows
    ]


def evaluate_estimates(
    items: Iterable[tuple[Wave, dict[str, Wave]]],
    sr: int = SAMPLE_RATE,
    workers: int | None = None,
) -> dict[str, Scores]:
    """items: (clean, {name: estimate}) per clip.
    Returns {name: {metric: mean}} in the order the names first appear."""
    with scorer(workers) as pool:
        rows = [
            {
                name: pool.submit(score_pair, clean, est, sr)
                for name, est in estimates.items()
            }
            for clean, estimates in items
        ]
        return mean_scores(by_name(resolved(rows)))


def by_name(rows: list[dict[str, Scores]]) -> dict[str, list[Scores]]:
    """[{name: scores}] per clip -> {name: [scores of each clip]}"""
    scores = {}
    for row in rows:
        for name, s in row.items():
            scores.setdefault(name, []).append(s)
    return scores


def mean_scores(scores: dict[str, list[Scores]]) -> dict[str, Scores]:
    """{name: [per-clip scores]} -> {name: {metric: mean}}"""
    return {
        name: {m: float(np.nanmean([r[m] for r in rows])) for m in rows[0]}
        for name, rows in scores.items()
    }


def evaluate_triples(
    triples: Iterable[tuple[Wave, Wave, Wave]],
    sr: int = SAMPLE_RATE,
    workers: int | None = None,
) -> dict[str, Scores]:
    """triples: (clean, noisy, denoised) per clip.
    "clean" scores the reference against itself: the metric ceiling."""
    return evaluate_estimates(
        (
            (clean, {
                "clean": clean,
                "noisy": noisy,
                "denoised": denoised,
            })
            for clean, noisy, denoised in triples
        ),
        sr,
        workers,
    )


def load_samples_dir(samples_dir: str | Path) -> Iterator[tuple[Wave, Wave, Wave]]:
    """Yield (clean, noisy, denoised) from the folders save_samples writes."""
    for d in sorted(p for p in Path(samples_dir).iterdir() if p.is_dir()):
        clean, _ = sf.read(str(d / "clean.wav"), dtype="float32")
        noisy, _ = sf.read(str(d / "noisy.wav"), dtype="float32")
        denoised, _ = sf.read(str(d / "denoised.wav"), dtype="float32")
        yield clean, noisy, denoised


def evaluate_model(
    denoise_fn: Callable[[Wave], Wave],
    pairs: Iterable[tuple[Wave, Wave]],
    cached: Sequence[dict[str, Scores]] = (),
    after_noisy: Callable[[], None] = lambda: None,
    workers: int | None = None,
) -> tuple[dict[str, Scores], list[dict[str, Scores]]]:
    """Also runs the clean clip through the model ("clean->model"), which
    shows how much the model distorts speech that has no noise to remove.
    cached: per-clip {"clean", "noisy"} scores from an earlier run, used for
    the first len(cached) clips. after_noisy() is called after each noisy clip
    has gone through the model. Returns (results, per-clip clean/noisy scores)."""
    rows = []
    with scorer(workers) as pool:
        for i, (noisy, clean) in enumerate(pairs):
            if i < len(cached):
                row = dict(cached[i])
            else:
                row = {
                    "clean": pool.submit(score_pair, clean, clean),
                    "noisy": pool.submit(score_pair, clean, noisy),
                }
            denoised = denoise_fn(noisy)
            after_noisy()
            row["denoised"] = pool.submit(score_pair, clean, denoised)
            row["clean->model"] = pool.submit(score_pair, clean, denoise_fn(clean))
            rows.append(row)
        resolved_rows = resolved(rows)
    fixed = [
        {
            "clean": row["clean"],
            "noisy": row["noisy"],
        }
        for row in resolved_rows
    ]
    return mean_scores(by_name(resolved_rows)), fixed


def cache_key(files: Sequence[Path], sample_rate: int) -> list:
    """Changes when the test files or the measures do, so a stale cache isn't used."""
    return [sample_rate, list(METRICS)] + [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]


def load_cache(key: list, cache_path: str = CACHE_PATH) -> list[dict[str, Scores]]:
    try:
        data = json.loads(resolve_path(cache_path).read_text())
    except (OSError, ValueError):
        return []
    return data["clips"] if data.get("key") == key else []


def save_cache(key: list, clips: list[dict[str, Scores]], cache_path: str = CACHE_PATH) -> None:
    path = resolve_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "clips": clips}))


def print_results(results: dict[str, Scores]) -> None:
    print(f"{'':14s}" + "".join(f"{m.upper():>8s}" for m in METRICS))
    for kind, m in results.items():
        print(f"{kind:14s}" + "".join(f"{m[name]:8.3f}" for name in METRICS))


def main() -> None:
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
    ap.add_argument("--workers", type=int, default=None,
                    help=f"Processes that compute the scores (default: one per CPU core, at most {MAX_WORKERS}).")
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
                                        load_pairs(files, args.num_clips, sr, every=args.every), cached, meter.add,
                                        args.workers)
        if len(fixed) > len(cached):
            save_cache(key, fixed, cache)
        print(f"  cost: {describe_cost(meter.cost(), model_cfg)}")
    else:
        samples_dir = args.samples_dir or resolve_path(cfg["train"]["output_dir"]) / "samples"
        print(f"Evaluating samples in {samples_dir}...")
        results = evaluate_triples(load_samples_dir(samples_dir), workers=args.workers)
    print_results(results)


if __name__ == "__main__":
    main()
