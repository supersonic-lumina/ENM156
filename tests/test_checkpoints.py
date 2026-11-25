"""
Checks the checkpoints a training run leaves (denoiser.model.CheckpointSaver),
with made-up validation scores and a model whose weights say which epoch
they are from:
  - best.pt / best_epoch_NNN.pt hold the best epochs, epoch_NNN.pt every one
  - a score that isn't a number (nan) is never the best, even in epoch 1
  - when the last epoch is also kept, model.pt is the same file, not a copy,
    and saving over one of the two names later leaves the other as it was
  - every checkpoint says which epoch and score it holds, and loads as before

  .venv/bin/python tests/test_checkpoints.py
"""
import contextlib
import io
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.config import load_config  # noqa: E402
from denoiser.model import CheckpointSaver, load_checkpoint, make_model, save_checkpoint  # noqa: E402

failures = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def run(cfg, model, mode, scores, higher=True):
    """Trains for len(scores) made-up epochs; returns the folder and {file: epoch it holds}."""
    out = Path(tempfile.mkdtemp())
    saver = CheckpointSaver(mode, out, higher, metric="val")
    with contextlib.redirect_stdout(io.StringIO()):
        for epoch, score in enumerate(scores, 1):
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(epoch)
            saver.epoch_done(model, cfg, epoch, score)
        saver.finish(model, cfg, out / "model.pt", len(scores), scores[-1])
    held = {}
    for f in sorted(out.glob("*.pt")):
        state = torch.load(f, map_location="cpu", weights_only=False)
        held[f.name] = int(next(iter(state["state_dict"].values())).flatten()[0])
    return out, held


def main():
    cfg = load_config([])
    model = make_model(cfg)

    _, held = run(cfg, model, 2, [3.0, 3.2, 3.1])
    check(held == {"best.pt": 2, "model.pt": 3}, f"best.pt is the best epoch, model.pt the last: {held}")

    _, held = run(cfg, model, 2, [math.nan, 3.0, 3.1])
    check(held["best.pt"] == 3, f"a nan score in epoch 1 doesn't stay the best: {held}")
    _, held = run(cfg, model, 2, [math.nan, 0.5, 0.4], higher=False)
    check(held["best.pt"] == 3, f"... also when lower is better: {held}")

    _, held = run(cfg, model, 3, [3.0, math.nan, 3.1, 3.2, 2.9])
    check(held == {"best_epoch_001.pt": 1, "best_epoch_003.pt": 3, "best_epoch_004.pt": 4, "model.pt": 5},
          f"the 3 best are kept, never a nan epoch: {held}")

    out, held = run(cfg, model, 2, [3.0, 3.1, 3.2])
    best, final = out / "best.pt", out / "model.pt"
    check(held == {"best.pt": 3, "model.pt": 3} and best.samefile(final),
          "last epoch the best: model.pt is the same file as best.pt, not a second copy")
    info = torch.load(best, map_location="cpu", weights_only=False)["info"]
    check(info == {"epoch": 3, "val": 3.2}, f"a checkpoint says which epoch and score it holds: {info}")
    loaded, _ = load_checkpoint(best, cfg, torch.device("cpu"))
    check(int(next(iter(loaded.state_dict().values())).flatten()[0]) == 3, "a checkpoint with info loads as before")
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(9)
    save_checkpoint(model, cfg, best)
    _, after = None, torch.load(final, map_location="cpu", weights_only=False)["state_dict"]
    check(int(next(iter(after.values())).flatten()[0]) == 3,
          "saving over best.pt later leaves model.pt as it was")

    _, held = run(cfg, model, 1, [3.0, 3.1])
    check(held == {"epoch_001.pt": 1, "epoch_002.pt": 2, "model.pt": 2}, f"every epoch kept: {held}")

    out = Path(tempfile.mkdtemp())
    (out / "best_epoch_050.pt").write_bytes(b"")
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        CheckpointSaver(3, out, True)
    check("best_epoch_050.pt" in text.getvalue(), "checkpoints left from an earlier run in the folder are pointed out")

    print("\nall passed" if not failures else f"\n{len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
