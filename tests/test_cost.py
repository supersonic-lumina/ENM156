"""
Checks the whole-pipeline cost estimate of denoiser/snn/stats.py: the STFT
and features around the network, the [energy] table, and registered cost
terms, for every model type.

  .venv/bin/python tests/test_cost.py
"""
import copy
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.config import ROOT, merge, read_toml  # noqa: E402
from denoiser.model import make_model  # noqa: E402
from denoiser.pipeline import features  # noqa: E402
from denoiser.snn.stats import (ENERGY_PJ, PJ_PER_MAC, WEIGHT_BITS, CostMeter, cost_lines,  # noqa: E402
                                macs_per_frame, pipeline_ops, register_cost_term, unregister_cost_term)


def close(a, b):
    return abs(a - b) < 1e-6 * max(1, abs(b))


def config(**sections):
    """config.toml's defaults (not config.local.toml, so the numbers don't
    depend on personal settings), with small networks and `sections` merged in."""
    cfg = read_toml(ROOT / "config.toml")
    merge(cfg, {"stft": {"n_fft": 128, "hop": 80, "long_factor": 0, "causal": True},
                "model": {"type": "mlp", "hidden": [32], "output": "mask", "filterbank": ""},
                "tcn": {"channels": 16, "kernel": 3, "dilations": [1, 2]},
                "snn": {"hidden": [24, 16], "threshold": 0.1}})
    return merge(cfg, copy.deepcopy(sections))


def measured(model, cfg):
    """A CostMeter that has seen one clip (a spiking model needs its spikes)."""
    meter = CostMeter(model.eval())
    with torch.no_grad():
        model(features(torch.randn(1, 4000), cfg)[1])
    meter.add()
    return meter


def main():
    torch.manual_seed(0)
    checks = {}

    # config.toml's [energy] reproduces the defaults, which are the old constants
    checks["config.toml energy = defaults"] = read_toml(ROOT / "config.toml")["energy"] == ENERGY_PJ
    checks["mac default unchanged"] = PJ_PER_MAC == 4.6 and ENERGY_PJ["synop"] == 0.9

    # STFT & features by hand, mask output, no long window: N = 128, F = 65, hop 80
    cfg = config()
    parts = pipeline_ops(cfg)
    checks["analysis by hand"] = close(parts["analysis"], 128 + 128 * 7)  # window + real FFT
    checks["features by hand"] = close(parts["features"], 65 * (3 + 1))  # |X|, log1p
    checks["output by hand"] = close(parts["output"], 65 * 2)  # gain times re and im
    checks["synthesis by hand"] = close(parts["synthesis"], 128 * 7 + 128 + 128 + 80)  # iFFT, window, OLA, norm

    # doubling n_fft: the front/back end grows as counted, the network's cost doesn't change
    wide = config(stft={"n_fft": 256})
    model = make_model(cfg)
    small, big = CostMeter(model).cost(cfg), CostMeter(model).cost(wide)
    by_hand = (256 + 256 * 8) + 129 * 4 + 129 * 2 + (256 * 8 + 2 * 256 + 80)
    checks["n_fft 256 by hand"] = close(big["pipeline"]["ops"], by_hand)
    checks["n_fft doubled: front/back end grows"] = big["pipeline"]["ops"] > 2 * small["pipeline"]["ops"]
    checks["n_fft doubled: network unchanged"] = (big["network_pj"] == small["network_pj"]
                                                  and big["macs"] == small["macs"])
    checks["network energy = old estimate"] = close(small["network_pj"], macs_per_frame(model) * 4.6)
    checks["no settings: network only, as before"] = (CostMeter(model).cost()["pj"] == small["network_pj"]
                                                      and CostMeter(model).cost()["pipeline"] is None)
    checks["total = network + pipeline"] = close(small["pj"], small["network_pj"] + small["pipeline"]["pj"])

    # a long window adds its own window, FFT, power, pooling and log1p
    long = pipeline_ops(config(stft={"long_factor": 10}))
    checks["long window"] = (close(long["analysis"] - parts["analysis"], 1280 + 1280 * math.log2(1280))
                             and close(long["features"] - parts["features"], 3 * 641 + 3 * 65))

    # output types
    magnitude = pipeline_ops(config(model={"output": "magnitude"}))
    checks["magnitude output"] = close(magnitude["output"], 4 * 65)
    complex_ = pipeline_ops(config(model={"output": "complex_mask", "mag_input": True, "mask_bound": 1.0}))
    checks["complex_mask output"] = close(complex_["output"], 18 * 65)
    checks["complex_mask features"] = close(complex_["features"], 65 * (3 + 1 + 3))
    unbounded = pipeline_ops(config(model={"output": "complex_mask", "mag_input": False, "mask_bound": 0}))
    checks["complex_mask unbounded"] = (close(unbounded["output"], 10 * 65)
                                        and close(unbounded["features"], 65 * 6))

    # the [energy] table overrides
    cheap = config(energy={"fft_mac": 1.0, "mac": 2.0, "memory_bit": 0.5})
    cost = CostMeter(model).cost(cheap)
    macs = macs_per_frame(model)
    checks["energy: fft_mac"] = close(cost["pipeline"]["pj"], cost["pipeline"]["ops"])
    checks["energy: mac and memory_bit"] = close(cost["network_pj"], macs * 2.0 + macs * WEIGHT_BITS * 0.5)
    try:
        CostMeter(model).cost(config(energy={"mac": "a lot"}))
        checks["energy: not a number refused"] = False
    except SystemExit:
        checks["energy: not a number refused"] = True

    # a registered cost term is printed and added to the total; it sees the spikes of a spiking model
    seen = {}

    def dummy(model, cfg, per_frame_spikes):
        seen[cfg["model"]["type"]] = per_frame_spikes
        return {"ops": 100, "pj": 7.0}

    register_cost_term("dummy", dummy)
    try:
        for kind in ("mlp", "tcn", "snn"):
            cfg = config(model={"type": kind})
            meter = measured(make_model(cfg), cfg)
            cost = meter.cost(cfg)
            labels = [label for label, _ in cost_lines(cost, cfg)]
            checks[f"{kind}: lines"] = labels == ["network", "STFT & features", "dummy", "total"]
            checks[f"{kind}: dummy in total"] = close(cost["pj"], cost["network_pj"] + cost["pipeline"]["pj"] + 7.0)
            checks[f"{kind}: same STFT & features"] = close(cost["pipeline"]["ops"], sum(parts.values()))
        checks["term gets spikes of snn only"] = (seen["mlp"] is None and seen["tcn"] is None
                                                  and len(seen["snn"]) == 2)
    finally:
        unregister_cost_term("dummy")
    cfg = config(model={"type": "snn"})
    meter = measured(make_model(cfg), cfg)
    cost = meter.cost(cfg)
    checks["unregistered term is gone"] = "dummy" not in cost["terms"]
    # spiking network: neuron updates counted on their own, the energy as before (updates at the MAC's energy)
    net = meter.model
    checks["snn neuron updates"] = cost["neuron_updates"] == (24 + 16 + net.n_out) * net.steps_per_frame
    checks["snn energy as before"] = close(cost["network_pj"], (cost["macs"] + cost["neuron_updates"]) * 4.6
                                           + cost["synops"] * 0.9)

    for label, text in cost_lines(cost, cfg):
        print(f"     {label}: {text}")
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
