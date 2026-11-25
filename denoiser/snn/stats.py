"""
What the denoiser costs to run, counted in operations per frame, and a rough
energy estimate from them. Nothing is measured: the counts come from the
settings and layer sizes and, for a spiking network, from how many spikes it
actually sent. Every model type (mlp, tcn, snn) is counted the same way.

The whole pipeline is counted, in parts that are reported separately:

  network          the model itself (macs_per_frame, or the spikes a
                   CostMeter counted). The number older logs show.
  STFT & features  everything around the network (pipeline_ops): the STFT
                   of the input, the features the model gets, applying its
                   output to the noisy spectrum, the inverse STFT
  registered terms anything added with register_cost_term (see below)

Operation types and their energy, in pJ per operation (ENERGY_PJ; override
them in the [energy] section of config.toml):

  mac            one multiply-accumulate of the network: a weight times a
                 value, added up. A conventional layer does one per weight
                 per frame.
  synop          one synaptic operation: a spike arriving at a weight, which
                 is just added (no multiply). Only happens when a neuron spikes.
  neuron_update  one update of a spiking neuron's state in one time step
                 (its leak: a multiply, and the incoming current added).
                 Counted for every neuron of the spiking layers and of the
                 readout layer, every time step. A conventional network's
                 activations are not counted, as before.
  fft_mac        one real multiply-add of the signal processing around the
                 network: the FFTs, windows, features, applying the output.
                 Other elementary operations there (sqrt, log1p, pow, tanh,
                 a division) count as one each, which flatters them.
  memory_bit     one bit read from memory: each weight a MAC or a spike's
                 synop uses is read once, as a 32-bit float. 0 by default,
                 so memory is left out unless you set it.

The defaults are the usual estimate for 45 nm chips (Horowitz 2014, 32-bit
float: 3.7 pJ multiply + 0.9 pJ add = 4.6 pJ per multiply-add, 0.9 pJ per
add) and give the network the same energy this module estimated before
neuron updates were counted on their own. They are a convention for
comparing networks with each other, not a measurement of any hardware.

Extending: register_cost_term(name, fn) adds a cost term without editing
this file. fn(model, cfg, per_frame_spikes) returns {"ops": ..., "pj": ...}
per frame (or None to leave it out for this model); it is printed on its
own line and added to the total. cfg is the model's settings, including
the [energy] section (energy_table(cfg) gives it with the defaults filled
in, and a term may read keys of its own from it); per_frame_spikes is a
spiking network's spikes per frame of each layer, or None for other networks.
"""
import math

import torch.nn as nn

from ..pipeline import filterbank, hop, is_complex, mag_input, n_freq
from .nets import SpikingModel

ENERGY_PJ = {"mac": 4.6, "synop": 0.9, "neuron_update": 4.6, "fft_mac": 4.6, "memory_bit": 0.0}
PJ_PER_MAC = ENERGY_PJ["mac"]  # the names older code uses
PJ_PER_SYNOP = ENERGY_PJ["synop"]
WEIGHT_BITS = 32  # a 32-bit float weight, as the energy figures assume

_cost_terms = {}  # name -> fn, in the order they were registered


def register_cost_term(name, fn):
    """Adds a cost term to every cost estimate made with settings from here
    on (CostMeter.cost(cfg)). fn(model, cfg, per_frame_spikes) -> {"ops":
    operations per frame, "pj": energy per frame in pJ}, or None to leave it
    out for this model. Registering a name again replaces its function."""
    _cost_terms[name] = fn


def unregister_cost_term(name):
    _cost_terms.pop(name, None)


def energy_table(cfg=None):
    """pJ per operation type: ENERGY_PJ, overridden by cfg["energy"]."""
    table = dict(ENERGY_PJ)
    for key, value in ((cfg or {}).get("energy") or {}).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SystemExit(f"energy.{key} must be a number (pJ per operation), got {value!r}")
        table[key] = float(value)
    return table


def macs_per_frame(model):
    """MACs of a conventional network per frame: one per weight of its
    linear and convolution layers (normalisation and activations left out)."""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d)))


def fan_out(model):
    """For each spiking layer of a LIFNet: the weights one spike arrives at."""
    targets = [*model.hidden[1:], model.n_out]
    own = model.hidden if model.recurrent is not None else [0] * len(model.hidden)
    return [t + o for t, o in zip(targets, own)]


def fft_ops(n):
    """Real multiply-adds of one real FFT (or inverse real FFT) of n points,
    counted as n * log2(n): a radix-2 complex FFT has (n / 2) * log2(n)
    butterflies of 4 real multiply-adds each (2 n log2 n), and a real input
    halves that (a complex FFT of n / 2 points plus a twiddle pass). The
    same formula is used when n is not a power of two (e.g. a long window of
    1280 points), where an FFT really costs somewhat more."""
    return n * math.log2(n) if n > 1 else 0.0


def pipeline_ops(cfg):
    """Operations per frame outside the network, in fft_mac units:
    {"analysis", "features", "output", "synthesis"}. With N = stft.n_fft,
    F = N // 2 + 1 bins and a long window of L = N * stft.long_factor:

      analysis   window N + real FFT of N (fft_ops); with a long window also
                 L + real FFT of L
      features   |X| = sqrt(re^2 + im^2): 3 per bin; log1p: 1 per bin if the
                 model gets the log magnitude; complex_mask: compression,
                 |X| ** (compress - 1) times re and im, 3 per bin; long window:
                 power 2 and pooling 1 per long bin, then sqrt, scale and
                 log1p 3 per bin
      output     magnitude: expm1, gain = estimate / |X|, times re and im:
                 4 per bin; mask: the gain times re and im: 2 per bin;
                 complex_mask: the complex multiply 4, decompression 6
                 (|Y| 3, pow, times re and im) and, with mask_bound, the
                 soft limit 8 (|m| 3, tanh, divide, scale, times re and im)
      synthesis  inverse real FFT of N, window N, overlap-add N additions,
                 normalisation by the window overlap: hop multiplies

    With model.filterbank the learned filterbank replaces the STFT: its
    encoder and decoder count one multiply-add per weight per frame
    (filters * kernel each), plus relu 1, compression 2 (sign * |z| ** c)
    and the mask 1 per filter, and kernel additions of overlap-add."""
    bank = filterbank(cfg)
    if bank is not None:
        filters, kernel = bank.enc.out_channels, bank.kernel
        return {"analysis": filters * kernel + (filters if bank.relu else 0),
                "features": 2 * filters,
                "output": filters,
                "synthesis": filters * kernel + kernel}
    stft, model = cfg["stft"], cfg["model"]
    n, factor, bins = stft["n_fft"], stft["long_factor"], n_freq(cfg)
    analysis = n + fft_ops(n)
    features = 3 * bins
    if not is_complex(cfg) or mag_input(cfg):
        features += bins  # log1p
    if is_complex(cfg):
        features += 3 * bins
    if factor:
        long_n = n * factor
        analysis += long_n + fft_ops(long_n)
        features += 3 * (long_n // 2 + 1) + 3 * bins
    output = {"magnitude": 4 * bins, "mask": 2 * bins}.get(model["output"])
    if output is None:  # complex_mask
        output = (4 + 6 + (8 if model["mask_bound"] else 0)) * bins
    synthesis = fft_ops(n) + 2 * n + stft["hop"]
    return {"analysis": analysis, "features": features, "output": output, "synthesis": synthesis}


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

    def network_ops(self):
        """The network's operations per frame: {"macs", "neuron_updates",
        "synops", "weights_read", "rates" (spikes per neuron per time step,
        per layer), "per_frame_spikes" (per layer; None if not spiking)}."""
        model = self.model
        if not isinstance(model, SpikingModel):
            macs = macs_per_frame(model)
            return {"macs": macs, "neuron_updates": 0, "synops": 0, "weights_read": macs,
                    "rates": [], "per_frame_spikes": None}
        if not self.frames:
            raise ValueError("no forward pass to count spikes from yet")
        per_frame = [s / self.frames for s in self.spikes]
        steps = model.steps_per_frame
        # analog input into the first layer (its weights that aren't pruned), once per frame
        macs = int(model.input_mask.sum())
        # every neuron's state, of the spiking layers and the readout, once per time step
        neuron_updates = steps * (sum(model.hidden) + model.n_out)
        spike_synops = sum(s * f for s, f in zip(per_frame, fan_out(model)))
        synops = spike_synops
        if model.skip:  # the current passed on to the next layer: one addition per neuron and time step
            synops += steps * sum(h for i, h in enumerate(model.hidden) if model.has_skip(i))
        return {"macs": macs, "neuron_updates": neuron_updates, "synops": synops,
                "weights_read": macs + spike_synops,
                "rates": [s / (h * steps) for s, h in zip(per_frame, model.hidden)],
                "per_frame_spikes": per_frame}

    def cost(self, cfg=None):
        """Per frame: the network's operations (network_ops) and
          memory_bits  bits of weights the network reads
          network_pj   the network's energy estimate (all "pj" used to be)
          pipeline     {"ops", "pj", "parts": pipeline_ops(cfg)}: STFT & features
          terms        {name: {"ops", "pj"}}: the registered cost terms
          pj           the total energy estimate
        cfg: the model's settings, which also give the [energy] table.
        Without them only the network is counted, at the default energies."""
        ops = self.network_ops()
        pj = energy_table(cfg)
        bits = ops["weights_read"] * WEIGHT_BITS
        network_pj = (ops["macs"] * pj["mac"] + ops["neuron_updates"] * pj["neuron_update"]
                      + ops["synops"] * pj["synop"] + bits * pj["memory_bit"])
        pipeline, terms = None, {}
        if cfg is not None:
            parts = pipeline_ops(cfg)
            total_ops = sum(parts.values())
            pipeline = {"ops": total_ops, "pj": total_ops * pj["fft_mac"], "parts": parts}
            for name, fn in _cost_terms.items():
                term = fn(self.model, cfg, ops["per_frame_spikes"])
                if term is not None:
                    terms[name] = {"ops": term["ops"], "pj": term["pj"]}
        total = network_pj + (pipeline["pj"] if pipeline else 0) + sum(t["pj"] for t in terms.values())
        return {**ops, "memory_bits": bits, "network_pj": network_pj, "pipeline": pipeline,
                "terms": terms, "pj": total}


def frames_per_second(cfg):
    return cfg["data"]["sample_rate"] / hop(cfg)


def describe_cost(cost, cfg):
    """The network's line: operations and estimated power when run live."""
    fps = frames_per_second(cfg)
    text = f"{cost['macs'] * fps / 1e6:.1f} M MACs/s"
    if cost["neuron_updates"]:
        text += f" + {cost['neuron_updates'] * fps / 1e6:.2f} M neuron updates/s"
    if cost["rates"]:
        rates = ", ".join(f"{r:.3f}" for r in cost["rates"])
        text += f" + {cost['synops'] * fps / 1e6:.2f} M synops/s (spikes per neuron per step: {rates})"
    return f"{text}; about {cost['network_pj'] * fps / 1e6:.0f} uW"


def cost_lines(cost, cfg):
    """[(label, text)] of a cost: the network, the STFT & features, each
    registered term and the total, as operations and power when run live."""
    fps = frames_per_second(cfg)
    lines = [("network", describe_cost(cost, cfg))]
    if cost["pipeline"]:
        p = cost["pipeline"]
        parts = ", ".join(f"{k} {v * fps / 1e6:.2f}" for k, v in p["parts"].items())
        lines.append(("STFT & features", f"{p['ops'] * fps / 1e6:.2f} M ops/s ({parts}); "
                                         f"about {p['pj'] * fps / 1e6:.0f} uW"))
    for name, term in cost["terms"].items():
        lines.append((name, f"{term['ops'] * fps / 1e6:.2f} M ops/s; about {term['pj'] * fps / 1e6:.0f} uW"))
    lines.append(("total", f"about {cost['pj'] * fps / 1e6:.0f} uW (from operation counts at the [energy] "
                           "pJ per operation, 45 nm by default: a convention, not a measurement)"))
    return lines
