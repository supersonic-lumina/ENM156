"""
Checks the composite measures of denoiser/metrics.py on made-up voiced sound,
without a dataset:
  - a clip against itself gets the best scores: LLR and WSS 0, SSNR at its
    35 dB cap, CSIG / CBAK / COVL at 5
  - every measure gets worse as more noise is added
  - LLR and WSS don't depend on the estimate's volume
  - an estimate with silent stretches gives finite numbers, not nan
  - a nan PESQ gives nan CSIG / CBAK / COVL but still an SSNR
On the VoiceBank-DEMAND test set, the noisy input scores CSIG 3.346,
CBAK 2.443, COVL 2.632, SSNR 1.680, as published (3.35, 2.44, 2.63, 1.68).

  .venv/bin/python tests/test_composite.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.metrics import composite, llr, ssnr, wss  # noqa: E402

SR = 16000
failures = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def voiced(seconds, rng):
    """Harmonics of a gliding pitch, switched on and off like syllables."""
    t = np.arange(int(seconds * SR)) / SR
    pitch = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 20))
    syllables = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(float)
    return 0.1 * x * syllables + 1e-3 * rng.standard_normal(len(t))


def main():
    rng = np.random.default_rng(0)
    clean = voiced(3.0, rng)
    noise = rng.standard_normal(len(clean)) * 0.05

    same = composite(clean, clean, pesq=4.64)
    print(f"     itself {same}")
    check(llr(clean, clean, SR) == 0 and wss(clean, clean, SR) == 0, "a clip against itself: LLR and WSS are 0")
    check(same == {"csig": 5.0, "cbak": 5.0, "covl": 5.0, "ssnr": 35.0},
          "a clip against itself: CSIG, CBAK, COVL 5 and SSNR 35 dB")

    light, heavy = clean + 0.2 * noise, clean + 2 * noise
    raw = [(llr(clean, x, SR), wss(clean, x, SR), ssnr(clean, x, SR)) for x in (light, heavy)]
    print(f"     light noise LLR {raw[0][0]:.3f} WSS {raw[0][1]:.1f} SSNR {raw[0][2]:.1f}; "
          f"heavy noise LLR {raw[1][0]:.3f} WSS {raw[1][1]:.1f} SSNR {raw[1][2]:.1f}")
    check(raw[0][0] < raw[1][0] and raw[0][1] < raw[1][1] and raw[0][2] > raw[1][2],
          "LLR and WSS rise and SSNR falls as noise is added")
    a, b = composite(clean, light, pesq=3.0), composite(clean, heavy, pesq=3.0)
    check(all(a[m] > b[m] for m in a), "at the same PESQ, CSIG, CBAK, COVL and SSNR fall as noise is added")

    check(math.isclose(llr(clean, 3 * light, SR), raw[0][0], rel_tol=1e-9)
          and math.isclose(wss(clean, 3 * light, SR), raw[0][1], rel_tol=1e-9),
          "LLR and WSS don't depend on the estimate's volume")

    gappy = light.copy()
    gappy[:SR // 2] = 0.0
    gappy[SR:SR + SR // 4] = 0.0
    check(all(math.isfinite(v) for v in composite(clean, gappy, pesq=2.0).values()),
          "an estimate with silent stretches gives finite scores")

    missing = composite(clean, light, pesq=math.nan)
    check(all(math.isnan(missing[m]) for m in ("csig", "cbak", "covl")) and math.isfinite(missing["ssnr"]),
          "a nan PESQ gives nan CSIG, CBAK and COVL, and still an SSNR")

    print("\nall passed" if not failures else f"\n{len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
