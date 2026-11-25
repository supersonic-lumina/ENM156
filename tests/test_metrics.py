"""
Checks the quality measures on made-up speech, without a dataset:
  - evaluate.py's PESQ / STOI / ESTOI: a clean clip scores near the top
    against itself, and lower the more noise is added
  - denoiser/pesq_loss.py: its FFT filter gives what torchaudio's lfilter
    gives, and the differentiable PESQ ranks clean above noisy and has a
    finite gradient
These catch a broken install or an API change of pesq, pystoi or torch-pesq.

  .venv/bin/python tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torchaudio.functional import lfilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from denoiser.pesq_loss import fft_lfilter, make_pesq_loss  # noqa: E402
from evaluate import score_pair  # noqa: E402
from test_scripts import noise, speech  # noqa: E402  (tests/ is on the path: this file's folder)

SR = 16000
failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def main() -> None:
    rng = np.random.default_rng(0)
    clean = speech(3.0, SR, rng)
    noises = noise(3.0, SR, rng)
    light, heavy = clean + 0.05 * noises, clean + 2 * noises

    perfect, slight, bad = score_pair(clean, clean), score_pair(clean, light), score_pair(clean, heavy)
    print(f"     clean {perfect}\n     light {slight}\n     heavy {bad}")
    check(perfect["pesq"] > 4.0 and perfect["stoi"] > 0.99 and perfect["estoi"] > 0.99,
          "a clean clip scores near the top against itself")
    check(all(perfect[m] > slight[m] > bad[m] for m in ("pesq", "stoi", "estoi")),
          "PESQ, STOI and ESTOI drop as noise is added")

    x = torch.from_numpy(heavy).unsqueeze(0)
    a, b = torch.tensor([1.0, -1.6, 0.68]), torch.tensor([0.2, 0.1, 0.0])  # a stable IIR filter
    check(torch.allclose(fft_lfilter(x, a, b), lfilter(x, a, b, clamp=False), atol=1e-4),
          "fft_lfilter gives what torchaudio's lfilter gives")

    loss_fn = make_pesq_loss("cpu", SR)
    ref = torch.from_numpy(clean).unsqueeze(0)
    deg = torch.from_numpy(heavy).unsqueeze(0).requires_grad_()
    with torch.no_grad():
        mos_clean, mos_noisy = loss_fn.mos(ref, ref).item(), loss_fn.mos(ref, deg).item()
    print(f"     differentiable PESQ: clean {mos_clean:.2f}, heavy noise {mos_noisy:.2f}")
    check(mos_clean > mos_noisy, "the differentiable PESQ ranks clean above noisy")
    loss_fn(ref, deg).sum().backward()
    check(deg.grad is not None and bool(torch.isfinite(deg.grad).all()) and deg.grad.abs().sum() > 0,
          "the PESQ loss has a finite, non-zero gradient")

    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
