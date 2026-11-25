"""
Checks the operation counts of denoiser/snn/stats.py against counts made by
hand on tiny networks.

  .venv/bin/python tests/test_snn_stats.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.nets import MLP, TCN  # noqa: E402
from denoiser.snn.nets import LIFNet  # noqa: E402
from denoiser.snn.stats import PJ_PER_MAC, PJ_PER_SYNOP, CostMeter, fan_out, macs_per_frame  # noqa: E402


def close(a, b):
    return abs(a - b) < 1e-6 * max(1, abs(b))


def main():
    torch.manual_seed(0)
    checks = {}

    # conventional networks: one MAC per weight
    checks["mlp MACs"] = macs_per_frame(MLP(10, 4, hidden=(8, 6))) == 10 * 8 + 8 * 6 + 6 * 4
    # tcn: 1x1 in (10*5), one block (5*5*3 dilated + 5*5 mix), 1x1 out (5*4)
    tcn = TCN(10, 4, channels=5, kernel=3, dilations=(2,))
    checks["tcn MACs"] = macs_per_frame(tcn) == 10 * 5 + 5 * 5 * 3 + 5 * 5 + 5 * 4
    checks["tcn cost"] = close(CostMeter(tcn).cost()["pj"], 170 * PJ_PER_MAC)

    # spiking: a spike of layer 1 reaches the 6 neurons of layer 2, one of layer 2 the 4 outputs;
    # with recurrence also the neurons of its own layer
    checks["fan-out"] = fan_out(LIFNet(10, 4, hidden=(8, 6))) == [6, 4]
    checks["fan-out recurrent"] = fan_out(LIFNet(10, 4, hidden=(8, 6), recurrent=True)) == [6 + 8, 4 + 6]

    for steps in (1, 3):
        net = LIFNet(10, 4, hidden=(8, 6), threshold=0.1, steps_per_frame=steps).eval()
        meter = CostMeter(net)
        with torch.no_grad():
            for _ in range(2):  # two passes of 3 clips x 7 frames
                net(torch.randn(3, 7, 10))
                meter.add()
        cost = meter.cost()
        frames = 2 * 3 * 7
        spikes = meter.spikes
        checks[f"{steps} step(s): frames"] = meter.frames == frames
        checks[f"{steps} step(s): network spikes"] = spikes[0] > 0 and spikes[1] > 0
        # by hand: input layer 10*8 MACs once per frame, an update per neuron (8 + 6 + 4) per step
        checks[f"{steps} step(s): MACs"] = cost["macs"] == 10 * 8
        checks[f"{steps} step(s): neuron updates"] = cost["neuron_updates"] == steps * 18
        synops = (spikes[0] * 6 + spikes[1] * 4) / frames
        checks[f"{steps} step(s): synops"] = close(cost["synops"], synops)
        checks[f"{steps} step(s): rates"] = (close(cost["rates"][0], spikes[0] / (frames * steps * 8))
                                             and close(cost["rates"][1], spikes[1] / (frames * steps * 6)))
        # the same energy as when neuron updates were counted as MACs
        checks[f"{steps} step(s): energy"] = close(cost["pj"], (10 * 8 + steps * 18) * PJ_PER_MAC + synops * PJ_PER_SYNOP)
        # the rate the loss sees is the mean over all neurons of the last pass
        total = sum(net.stats["spikes"]) / (net.stats["frames"] * steps * 14)
        checks[f"{steps} step(s): spike_rate"] = close(net.spike_rate.item(), total)

    # pruned input weights: no current through them, no MACs counted, no gradient step moves them back
    net = LIFNet(10, 4, hidden=(8, 6), threshold=0.1)
    net.input_mask[:, :5] = 0  # inputs 0-4 removed
    x = torch.randn(2, 7, 10)
    changed = x.clone()
    changed[..., :5] += 1
    with torch.no_grad():
        checks["pruned inputs are ignored"] = torch.equal(net(x), net(changed))
        meter = CostMeter(net)
        meter.add()
    checks["pruned MACs"] = meter.cost()["macs"] == 5 * 8
    net(x).sum().backward()
    checks["pruned weights get no gradient"] = not net.inputs[0].weight.grad[:, :5].any()
    fresh = LIFNet(10, 4, hidden=(8, 6))
    fresh.load_state_dict(net.state_dict())
    checks["mask is saved"] = torch.equal(fresh.input_mask, net.input_mask)

    # skip connections: one more addition per neuron of layers 2.. and time step
    plain, skip = LIFNet(10, 4, hidden=(8, 8, 8), threshold=0.1), LIFNet(10, 4, hidden=(8, 8, 8), threshold=0.1, skip=True)
    with torch.no_grad():
        skip.load_state_dict(plain.state_dict())
        skip(x)
        meter = CostMeter(skip)
        meter.add()
    by_spikes = sum(s * f for s, f in zip(meter.spikes, fan_out(skip))) / meter.frames
    checks["skip additions"] = close(meter.cost()["synops"], by_spikes + 16)

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
