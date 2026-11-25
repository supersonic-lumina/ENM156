"""
Checks train_pesq.py's pesq.pesq_weight: the default (1) gives the same loss
as before, 0 trains on the other terms only (MSE-only training with
mse_weight = 1, stoi_weight = 0), and a configuration with every weight 0
is refused. Runs on random audio with a small TCN on the CPU.

  .venv/bin/python tests/test_pesq_weight.py
"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.config import load_config, merge  # noqa: E402
from denoiser.model import make_model  # noqa: E402
from train_pesq import CombinedLoss  # noqa: E402

SETTINGS = {"stft": {"n_fft": 256, "hop": 64, "long_factor": 0, "causal": True},
            "model": {"type": "tcn", "output": "complex_mask", "filterbank": ""},
            "tcn": {"channels": 8, "kernel": 3, "dilations": [1, 2]}}
failures = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def config(**pesq):
    cfg = merge(copy.deepcopy(load_config()), copy.deepcopy(SETTINGS))
    cfg["pesq"].update(pesq)
    return cfg


def loss(cfg, noisy, clean):
    torch.manual_seed(0)
    model = make_model(cfg)
    total, terms = CombinedLoss(cfg, torch.device("cpu"), 16000)(model, noisy, clean)
    total.backward()
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    return total.detach(), terms, grads


def main():
    torch.manual_seed(1)
    clean = 0.1 * torch.randn(2, 16000)
    noisy = clean + 0.05 * torch.randn(2, 16000)

    both, terms, _ = loss(config(pesq_weight=1, mse_weight=1, stoi_weight=0), noisy, clean)
    mse_only, mse_terms, grads = loss(config(pesq_weight=0, mse_weight=1, stoi_weight=0), noisy, clean)
    scale = config()["pesq"]["mse_scale"]
    check(torch.equal(mse_only, mse_terms["mse"] / scale), "pesq_weight = 0: the loss is the MSE term alone")
    check(grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads),
          "pesq_weight = 0: the model still gets gradients")
    check("pesq" in mse_terms and torch.isfinite(mse_terms["pesq"]), "pesq_weight = 0: PESQ is still estimated for the log")
    check(torch.equal(mse_terms["mse"], terms["mse"]), "same MSE term with and without the PESQ loss")

    # The default reproduces the loss as it was: PESQ loss + MSE / mse_scale
    torch.manual_seed(0)
    model = make_model(config(pesq_weight=1, mse_weight=1, stoi_weight=0))
    crit = CombinedLoss(config(pesq_weight=1, mse_weight=1, stoi_weight=0), torch.device("cpu"), 16000)
    from denoiser.pipeline import run_model
    with torch.no_grad():
        est, _, _ = run_model(model, noisy, crit.cfg)
        old = crit.pesq(clean, est).mean() + terms["mse"] / scale
    check(torch.allclose(both, old, rtol=0, atol=1e-6), f"pesq_weight = 1: the same loss as before ({both:.6f} vs {old:.6f})")

    try:
        CombinedLoss(config(pesq_weight=0, mse_weight=0, stoi_weight=0), torch.device("cpu"), 16000)
        check(False, "every weight 0 is refused")
    except SystemExit:
        check(True, "every weight 0 is refused")

    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
