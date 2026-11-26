"""
Checks denoiser/dnsmos.py's windowing and score mapping against what DNS-Challenge's dnsmos_local.py does, with a
stand-in for the ONNX model (the real one isn't in the repository): a clip shorter than 9.01 s is repeated to that
length by doubling it, every clip is cut into 9.01 s windows every second, and each score is the mean of the
mapped raw scores. Also checks where model_path finds the model: data.dnsmos_model, or train.dnsmos_model where a
run's settings have it.

  .venv/bin/python tests/test_dnsmos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.dnsmos import DEFAULT_MODEL, DNSMOS, KEYS, POLY, SAMPLE_RATE, model_path  # noqa: E402

WINDOW = int(9.01 * SAMPLE_RATE)
failed = False


def check(ok: bool, what: str) -> None:
    global failed
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    failed |= not ok


class FakeSession:
    """Records the windows it's given; its raw scores are 1 + the window's first sample, 2, 3."""

    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []

    def run(self, _outputs: None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds["input_1"]
        self.windows.append(x[0])
        return [np.array([[1 + x[0, 0], 2.0, 3.0]], dtype=np.float32)]


def scorer() -> tuple[DNSMOS, FakeSession]:
    dnsmos = DNSMOS.__new__(DNSMOS)  # without loading the ONNX model
    dnsmos.session = FakeSession()
    return dnsmos, dnsmos.session


def main() -> None:
    dnsmos, session = scorer()
    short = np.arange(3 * SAMPLE_RATE, dtype=np.float32) / 1e6
    scores = dnsmos(short)
    # 3 s doubled to 6, then 12 s: windows at 0, 1 and 2 s
    check(len(session.windows) == 3 and all(len(w) == WINDOW for w in session.windows),
          f"a 3 s clip, doubled to 12 s, gives three 9.01 s windows (got {len(session.windows)})")
    first = session.windows[0]
    check(all(np.array_equal(first[i * len(short):(i + 1) * len(short)], short) for i in range(3)),
          "made by repeating the clip")
    check(set(scores) == set(KEYS) and abs(scores["BAK"] - POLY["BAK"](2.0)) < 1e-6,
          "the raw scores are mapped by the polynomials")

    dnsmos, session = scorer()
    ramp = np.repeat(np.arange(20, dtype=np.float32), SAMPLE_RATE) / 100  # each second's own value
    scores = dnsmos(ramp)
    starts = [w[0] * 100 for w in session.windows]
    check(len(session.windows) == 11 and starts == list(range(11)),
          f"a 20 s clip gives windows starting every second, 0..10 s (got {len(session.windows)})")
    check(all(len(w) == WINDOW for w in session.windows), "every window is 9.01 s")
    expected = np.mean([POLY["SIG"](1 + s / 100) for s in range(11)])
    check(abs(scores["SIG"] - expected) < 1e-5, "each score is the mean over the windows")

    cfg = {"train": {}, "data": {"dnsmos_model": "a.onnx"}}
    check(model_path(cfg) == "a.onnx", "model_path reads data.dnsmos_model")
    check(model_path({"train": {"dnsmos_model": "b.onnx"}, "data": cfg["data"]}) == "b.onnx",
          "a run's train.dnsmos_model still wins")
    check(model_path({"train": {}, "data": {}}) == DEFAULT_MODEL, "and the default without either")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
