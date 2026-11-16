"""
What a network costs to run, counted in operations, and a rough energy
estimate from them. Nothing is measured: the counts come from the layer
sizes and, for a spiking network, from how many spikes it actually sent.

  MAC     one multiply-accumulate: a weight times a value, added up. A
          conventional layer does one per weight per frame.
  synop   one synaptic operation: a spike arriving at a weight, which is
          just added (no multiply). Only happens when a neuron spikes.

Energy per operation is the usual estimate for 45 nm chips (Horowitz 2014,
32-bit float): 4.6 pJ per MAC, 0.9 pJ per synop. It is a convention for
comparing networks, not a measurement of any hardware.
"""
import torch.nn as nn

from .nets import SpikingModel

PJ_PER_MAC = 4.6
PJ_PER_SYNOP = 0.9


def macs_per_frame(model):
    """MACs of a conventional network per frame: one per weight of its
    linear and convolution layers (normalisation and activations left out)."""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d)))


def fan_out(model):
    """For each spiking layer of a LIFNet: the weights one spike arrives at."""
    targets = [*model.hidden[1:], model.n_out]
    own = model.hidden if model.recurrent is not None else [0] * len(model.hidden)
    return [t + o for t, o in zip(targets, own)]


class CostMeter:
    """Adds up the spikes of every forward pass of a spiking model, so the
    cost is that of the clips it actually ran on. For other models there is
    nothing to add up: their cost per frame is fixed."""

    def __init__(self, model):
        self.model = model
        self.frames, self.spikes = 0, None

    def add(self):
        """Call after each forward pass."""
        stats = getattr(self.model, "stats", None)
        if stats is None:
            return
        self.frames += stats["frames"]
        self.spikes = [a + b for a, b in zip(self.spikes or [0] * len(stats["spikes"]), stats["spikes"])]

    def cost(self):
        """Per frame: {"macs", "synops", "rates" (spikes per neuron per time
        step, per layer), "pj" (energy estimate)}."""
        model = self.model
        if not isinstance(model, SpikingModel):
            macs = macs_per_frame(model)
            return {"macs": macs, "synops": 0, "rates": [], "pj": macs * PJ_PER_MAC}
        if not self.frames:
            raise ValueError("no forward pass to count spikes from yet")
        per_frame = [s / self.frames for s in self.spikes]
        steps = model.steps_per_frame
        # analog input into the first layer (its weights that aren't pruned), once
        # per frame; then one multiply per neuron and time step for its leak
        macs = int(model.input_mask.sum()) + steps * (sum(model.hidden) + model.n_out)
        synops = sum(s * f for s, f in zip(per_frame, fan_out(model)))
        if model.skip:  # the current passed on to the next layer: one addition per neuron and time step
            synops += steps * sum(h for i, h in enumerate(model.hidden) if model.has_skip(i))
        return {"macs": macs, "synops": synops,
                "rates": [s / (h * steps) for s, h in zip(per_frame, model.hidden)],
                "pj": macs * PJ_PER_MAC + synops * PJ_PER_SYNOP}


def describe_cost(cost, cfg):
    """One printable line: operations and estimated power when run live."""
    fps = cfg["data"]["sample_rate"] / cfg["stft"]["hop"]
    text = f"{cost['macs'] * fps / 1e6:.1f} M MACs/s"
    if cost["rates"]:
        rates = ", ".join(f"{r:.3f}" for r in cost["rates"])
        text += f" + {cost['synops'] * fps / 1e6:.2f} M synops/s (spikes per neuron per step: {rates})"
    return f"{text}; about {cost['pj'] * fps / 1e6:.0f} uW at 45 nm (estimate from operation counts)"
