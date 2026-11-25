"""
Checks train_pesq.py's pesq.pesq_weight: the default (1) gives the same loss
as before, the weight scales the PESQ term, 0 trains on the other terms only
(MSE-only training with mse_weight = 1, stoi_weight = 0) while PESQ is still
logged, and negative or all-zero weights are refused. Runs on random audio
with a small TCN on the CPU.

  .venv/bin/python tests/test_pesq_weight.py
"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.config import load_config, merge  # noqa: E402
from denoiser.model import make_model  # noqa: E402
from denoiser.pipeline import run_model  # noqa: E402
from train_pesq import CombinedLoss  # noqa: E402

SETTINGS = {"stft": {"n_fft": 256, "hop": 64, "long_factor": 0, "causal": True},
            "model": {"type": "tcn", "output": "complex_mask", "filterbank": ""},
            "tcn": {"channels": 8, "kernel": 3, "dilations": [1, 2]}}
CPU = torch.device("cpu")
failures = []


def check(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def config(**pesq: float) -> dict:
    """config.toml with the small TCN, the STOI term off, and `pesq` settings."""
    cfg = merge(copy.deepcopy(load_config()), copy.deepcopy(SETTINGS))
    cfg["pesq"].update({"stoi_weight": 0, **pesq})
    return cfg


def model_for(cfg: dict) -> torch.nn.Module:
    torch.manual_seed(0)  # the same weights every time
    return make_model(cfg)


def run_loss(noisy: torch.Tensor, clean: torch.Tensor, **pesq: float):
    """The loss, its terms and the gradients of a fresh model."""
    cfg = config(**pesq)
    model = model_for(cfg)
    total, terms = CombinedLoss(cfg, CPU, 16000)(model, noisy, clean)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return total.detach(), terms, grads


def refused(**pesq: float) -> bool:
    try:
        CombinedLoss(config(**pesq), CPU, 16000)
    except SystemExit:
        return True
    return False


def main():
    torch.manual_seed(1)
    clean = 0.1 * torch.randn(2, 16000)
    noisy = clean + 0.05 * torch.randn(2, 16000)
    mse_scale = config()["pesq"]["mse_scale"]

    # The PESQ loss of the test model on its own, for the expected totals
    cfg = config()
    crit = CombinedLoss(cfg, CPU, 16000)
    with torch.no_grad():
        est, _, _ = run_model(model_for(cfg), noisy, cfg)
        pesq_term = crit.pesq(clean, est).mean()

    default, terms, _ = run_loss(noisy, clean)
    old = pesq_term + terms["mse"] / mse_scale  # the loss before pesq_weight existed
    check(torch.equal(default, old), f"default pesq_weight = 1: the same loss as before ({default:.6f})")

    double, _, _ = run_loss(noisy, clean, pesq_weight=2)
    check(torch.equal(double, 2 * pesq_term + terms["mse"] / mse_scale), "pesq_weight = 2 doubles the PESQ term")

    mse_only, mse_terms, grads = run_loss(noisy, clean, pesq_weight=0)
    check(torch.equal(mse_only, mse_terms["mse"] / mse_scale), "pesq_weight = 0: the loss is the MSE term alone")
    check(len(grads) > 0 and all(torch.isfinite(g).all() for g in grads), "pesq_weight = 0: finite gradients")
    check(any(g.abs().sum() > 0 for g in grads), "pesq_weight = 0: the model still learns")
    check(torch.isfinite(mse_terms["pesq"]).item(), "pesq_weight = 0: PESQ is still estimated for the log")

    check(refused(pesq_weight=0, mse_weight=0), "every weight 0 is refused")
    check(refused(pesq_weight=-1), "a negative weight is refused")

    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
