"""
DNSMOS P.835 (Reddy et al., ICASSP 2022; Microsoft DNS Challenge): a network trained on human ratings that scores a
clip without a clean reference: SIG (speech quality), BAK (background noise), OVRL (overall), each on a 1-5 MOS
scale. For comparing outputs only, never as a training loss: a network would learn to fool it.

Model: sig_bak_ovr.onnx from github.com/microsoft/DNS-Challenge (DNSMOS/DNSMOS, CC BY 4.0), at data.dnsmos_model
(see docs/datasets.md). Scoring as their dnsmos_local.py: 16 kHz, clips shorter than 9.01 s repeated to that
length, 9.01 s windows every second, the raw scores mapped by their (non-personalised) polynomials, the mean over
the windows.

  python -m denoiser.dnsmos runs/x/model.pt [--split test] [--num-clips N]   clean, noisy and denoised clips
  python -m denoiser.dnsmos runs/x/model.pt --dns4-dev [--every N]          DNS Challenge 4 dev test set

The DNS-4 dev test set (in data.dns4_dir; python -m denoiser.dns4) is 921 noisy clips without clean references:
new noises, other microphones, reverberation. It shows how a model does away from VoiceBank-DEMAND.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
INPUT_LENGTH = 9.01  # seconds per window
DEFAULT_MODEL = "data/dnsmos/sig_bak_ovr.onnx"  # data.dnsmos_model
POLY = {"SIG": np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
        "BAK": np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
        "OVRL": np.poly1d([-0.06766283, 1.11546468, 0.04602535])}
KEYS = ("SIG", "BAK", "OVRL")

Scores = dict[str, float]


def model_path(cfg: dict) -> str:
    """data.dnsmos_model (or train.dnsmos_model, where a run's settings still have it)."""
    return cfg["train"].get("dnsmos_model") or cfg["data"].get("dnsmos_model") or DEFAULT_MODEL


class DNSMOS:
    def __init__(self, path: str | Path = DEFAULT_MODEL) -> None:
        import onnxruntime as ort

        from .config import resolve_path
        p = resolve_path(path)
        if not p.exists():
            raise SystemExit(f"DNSMOS model not found: {p} (see docs/datasets.md)")
        self.session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])

    def __call__(self, audio: np.ndarray) -> Scores:
        """audio: 1-D float array at 16 kHz -> {"SIG", "BAK", "OVRL"}."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        n = int(INPUT_LENGTH * SAMPLE_RATE)
        while len(audio) < n:
            audio = np.append(audio, audio)
        hops = int(np.floor(len(audio) / SAMPLE_RATE) - INPUT_LENGTH) + 1
        windows = [audio[i * SAMPLE_RATE: i * SAMPLE_RATE + n] for i in range(hops)]
        windows = [w for w in windows if len(w) == n]
        raw = np.array([self.session.run(None, {"input_1": w[None]})[0][0] for w in windows])  # (windows, 3)
        return {k: float(np.mean(POLY[k](raw[:, j]))) for j, k in enumerate(KEYS)}


def mean_scores(rows: list[Scores]) -> Scores:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def print_table(rows: dict[str, list[Scores]]) -> None:
    print(f"{'':14s}{'SIG':>7s}{'BAK':>7s}{'OVRL':>7s}   ({len(next(iter(rows.values())))} clips)")
    for name, r in rows.items():
        m = mean_scores(r)
        print(f"{name:14s}{m['SIG']:7.3f}{m['BAK']:7.3f}{m['OVRL']:7.3f}")


def dev_clips(dns4_dir: Path, every: int = 1) -> Iterator[np.ndarray]:
    """Every `every`th clip of the DNS-4 dev test set, as 16 kHz float32."""
    import soundfile as sf

    from .dns4 import DEV_TESTSET

    files = sorted((dns4_dir / DEV_TESTSET).glob("*.wav"))[::every]
    if not files:
        raise SystemExit(f"no DNS-4 dev test clips in {dns4_dir / DEV_TESTSET} (set data.dns4_dir; docs/datasets.md)")
    for f in files:
        wav, rate = sf.read(f, dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{f}: {rate} Hz, expected {SAMPLE_RATE}")
        yield wav.mean(axis=1)


def main() -> None:
    import torch

    from .config import add_config_args, get_device, load_config, resolve_path
    from .data import load_pairs, split_files
    from .model import describe_model, load_checkpoint
    from .pipeline import enhance

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--num-clips", type=int, default=0, help="0 = all")
    ap.add_argument("--every", type=int, default=1, help="score each Nth clip only")
    ap.add_argument("--dns4-dev", action="store_true", help="the DNS-4 dev test set instead of --split")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set)
    device = get_device(cfg)
    model, mcfg = load_checkpoint(args.model, cfg, device)
    model.eval()
    score = DNSMOS(model_path(cfg))

    def denoise(wav: np.ndarray) -> np.ndarray:
        return enhance(model, torch.from_numpy(wav)[None].to(device), mcfg)[0].cpu().numpy()

    rows: dict[str, list[Scores]]
    with torch.no_grad():
        if args.dns4_dev:
            print(f"DNSMOS P.835 of {args.model} on the DNS-4 dev test set\n  {describe_model(mcfg)}")
            rows = {"noisy": [], "denoised": []}
            for noisy in dev_clips(resolve_path(cfg["data"]["dns4_dir"]), args.every):
                rows["noisy"].append(score(noisy))
                rows["denoised"].append(score(denoise(noisy)))
            print_table(rows)
            worse = np.mean([d["OVRL"] < n["OVRL"] for n, d in zip(rows["noisy"], rows["denoised"])])
            print(f"  OVRL lower than the noisy input's on {worse * 100:.0f}% of clips")
            return
        print(f"DNSMOS P.835 of {args.model} on {args.split}\n  {describe_model(mcfg)}")
        rows = {"clean": [], "noisy": [], "denoised": []}
        for noisy, clean in load_pairs(split_files(cfg, args.split), args.num_clips, cfg["data"]["sample_rate"],
                                       every=args.every):
            for name, wav in (("clean", clean), ("noisy", noisy), ("denoised", denoise(noisy))):
                rows[name].append(score(wav))
        print_table(rows)


if __name__ == "__main__":
    main()
