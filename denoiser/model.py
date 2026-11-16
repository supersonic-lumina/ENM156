"""Building the network the settings ask for, and checkpoints that
remember the settings they were trained with."""
import copy
import math

import torch

from .config import MODEL_SECTIONS, merge, resolve_path
from .nets import MLP, TCN
from .filterbank import describe_filterbank
from .snn.nets import LIFNet
from .pipeline import check_stft, filterbank, hop, io_sizes, is_complex, n_freq, out_activation

MODEL_TYPES = ("mlp", "tcn", "snn")


def model_type(cfg):
    kind = cfg["model"]["type"]
    if kind not in MODEL_TYPES:
        raise SystemExit(f"model.type must be one of {MODEL_TYPES}, got {kind!r}")
    return kind


def make_model(cfg):
    check_stft(cfg)
    n_in, n_out = io_sizes(cfg)
    act = out_activation(cfg)
    if model_type(cfg) == "tcn":
        tcn = cfg["tcn"]
        model = TCN(n_in, n_out, tcn["channels"], tcn["kernel"], tuple(tcn["dilations"]), act)
        last = model.out
    elif model_type(cfg) == "snn":
        snn = cfg["snn"]
        model = LIFNet(n_in, n_out, tuple(snn["hidden"]), snn["recurrent"], snn["decay"], snn["readout_decay"],
                       snn["learn_decay"], snn["threshold"], snn["surrogate_slope"], snn["steps_per_frame"], act,
                       snn["skip"], snn["norm"], snn["init_gain"], snn["spiking"])
        last = model
    else:
        model = MLP(n_in, n_out, tuple(cfg["model"]["hidden"]), act)
        last = model.net[-2]
    if is_complex(cfg):
        # Start out close to a mask of 1 + 0j, i.e. passing the noisy audio
        # through: a mask near 0 would give near-silent output, where the
        # PESQ loss isn't finite
        with torch.no_grad():
            last.bias[:n_freq(cfg)] = 1
            last.bias[n_freq(cfg):] = 0
    return model


def describe_net(cfg):
    n_in, n_out = io_sizes(cfg)
    if model_type(cfg) == "tcn":
        tcn = cfg["tcn"]
        history = (tcn["kernel"] - 1) * sum(tcn["dilations"])
        ms = history * hop(cfg) / cfg["data"]["sample_rate"] * 1000
        return (f"tcn {n_in} -> {tcn['channels']} channels x {len(tcn['dilations'])} blocks "
                f"(dilations {tcn['dilations']}) -> {n_out}, sees {history + 1} frames "
                f"({ms:.0f} ms of history)")
    if model_type(cfg) == "snn":
        snn = cfg["snn"]
        return (f"snn {n_in} -> {' -> '.join(map(str, snn['hidden']))} {'spiking' if snn['spiking'] else 'smooth (not spiking)'} neurons"
                f"{' (recurrent)' if snn['recurrent'] else ''}{', skip connections' if snn['skip'] else ''}"
                f"{', normalised' if snn['norm'] else ''} -> {n_out}, "
                f"{snn['steps_per_frame']} time step(s) per frame")
    return f"mlp {n_in} -> {' -> '.join(map(str, cfg['model']['hidden']))} -> {n_out}"


def describe_model(cfg):
    stft = cfg["stft"]
    sr = cfg["data"]["sample_rate"]
    bank = filterbank(cfg)
    if bank is not None:
        fb = {"filters": bank.enc.out_channels, "kernel": bank.kernel, "stride": bank.stride, "relu": bank.relu}
        return (f"{cfg['model']['output']} {describe_net(cfg)}; frozen "
                f"{describe_filterbank(fb, sr)} from {cfg['model']['filterbank']}")
    long = (f", long window {stft['n_fft'] * stft['long_factor']} samples "
            f"({stft['n_fft'] * stft['long_factor'] / sr * 1000:.0f} ms)" if stft["long_factor"] else "")
    return (f"{cfg['model']['output']} {describe_net(cfg)}; "
            f"{'causal' if stft['causal'] else 'centred'} frames of {stft['n_fft']} samples "
            f"({stft['n_fft'] / sr * 1000:.0f} ms), hop {stft['hop']}{long}")


def save_checkpoint(model, cfg, path):
    torch.save({"config": {s: cfg[s] for s in MODEL_SECTIONS},
                "state_dict": model.state_dict()}, path)


class CheckpointSaver:
    """Saves checkpoints during training, after each epoch's validation,
    besides the final model.pt. train.save_checkpoints picks which:
      0 = none, 1 = every epoch (epoch_NNN.pt), 2 = the best so far (best.pt),
      3 = the 3 best (best_epoch_NNN.pt; the worst of them is deleted)
    `higher_is_better` says which way the validation score goes."""

    MODES = (0, 1, 2, 3)

    def __init__(self, mode, out_dir, higher_is_better, keep=3):
        if mode not in self.MODES:
            raise SystemExit(f"train.save_checkpoints must be one of {self.MODES}, got {mode!r}")
        self.mode, self.out_dir, self.keep = mode, out_dir, keep
        self.sign = 1 if higher_is_better else -1
        self.best = []  # (score, path) of the checkpoints kept, best first

    def epoch_done(self, model, cfg, epoch, score):
        if self.mode == 1:
            self._save(model, cfg, self.out_dir / f"epoch_{epoch:03d}.pt")
        elif self.mode == 2:
            if not self.best or self.sign * score > self.sign * self.best[0][0]:
                self.best = [(score, self.out_dir / "best.pt")]
                self._save(model, cfg, self.best[0][1])
        elif self.mode == 3:
            if len(self.best) < self.keep or self.sign * score > self.sign * self.best[-1][0]:
                path = self.out_dir / f"best_epoch_{epoch:03d}.pt"
                self._save(model, cfg, path)
                self.best.append((score, path))
                self.best.sort(key=lambda b: -self.sign * b[0])
                for _, old in self.best[self.keep:]:
                    old.unlink(missing_ok=True)
                del self.best[self.keep:]

    def _save(self, model, cfg, path):
        save_checkpoint(model, cfg, path)
        print(f"  saved {path.name}")


class LrSchedule:
    """The lr of each training step (batch), from the [train] settings:
    first warmup_steps rising linearly to lr, then
      "step": StepLR, lr times lr_gamma every lr_step epochs (lr_step = 0: constant)
      "wsd":  warmup-stable-decay, lr until the last decay_share of the steps,
              then a cosine down to lr_end at the last step"""

    KINDS = ("step", "wsd")

    def __init__(self, tr, steps_per_epoch):
        if tr["lr_schedule"] not in self.KINDS:
            raise SystemExit(f"train.lr_schedule must be one of {self.KINDS}, got {tr['lr_schedule']!r}")
        self.tr, self.steps_per_epoch = tr, steps_per_epoch
        self.total = tr["epochs"] * steps_per_epoch
        self.decay_start = self.total - round(self.total * tr["decay_share"])

    def lr(self, step):
        tr = self.tr
        if step < tr["warmup_steps"]:
            return tr["lr"] * (step + 1) / tr["warmup_steps"]
        if tr["lr_schedule"] == "step":
            epoch = step // self.steps_per_epoch
            return tr["lr"] * tr["lr_gamma"] ** (epoch // tr["lr_step"] if tr["lr_step"] else 0)
        if step < self.decay_start:
            return tr["lr"]
        t = (step + 1 - self.decay_start) / (self.total - self.decay_start)  # just above 0 .. 1
        return tr["lr_end"] + (tr["lr"] - tr["lr_end"]) * 0.5 * (1 + math.cos(math.pi * t))

    def set(self, opt, step):
        """Sets step's lr on the optimizer and returns it."""
        lr = self.lr(step)
        for group in opt.param_groups:
            group["lr"] = lr
        return lr


def load_checkpoint(path, cfg, device):
    """Rebuilds the model from a checkpoint. Its stored settings replace the
    stft/model settings of cfg; returns (model, cfg used)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = copy.deepcopy(cfg)
    if "state_dict" in checkpoint:
        stored = checkpoint["config"]
        stored["model"].setdefault("type", "mlp")  # saved before there was a choice
        # complex_mask checkpoints from before these settings: real/imag input only, no limit
        stored["model"].setdefault("mag_input", False)
        stored["model"].setdefault("mask_bound", 0)
        stored["model"].setdefault("filterbank", "")  # saved before it could replace the STFT
        merge(cfg, stored)
        state = checkpoint["state_dict"]
    else:
        state = legacy_settings(dict(checkpoint), cfg)
    model = make_model(cfg)
    model.load_state_dict(state)
    return model.to(device), cfg


def initial_model(cfg, device, path):
    """The model a training run starts from: a fresh one if path is "", else
    that checkpoint, whose stft/model settings then replace cfg's (the
    features have to match the ones it was trained on). Returns (model, cfg)."""
    if not path:
        return make_model(cfg).to(device), cfg
    print(f"Starting from {path} (its stft/model settings replace the configured ones)")
    return load_checkpoint(resolve_path(path), cfg, device)


def legacy_settings(state, cfg):
    """Checkpoints from the old main*.py scripts only hold weights. Their
    settings are inferred from the layer sizes (all used centred frames);
    the hop isn't stored anywhere, so it is taken from cfg."""
    weights = [w for name, w in state.items() if name.endswith(".weight")]
    n_in, n_out = weights[0].shape[1], weights[-1].shape[0]
    long_factor = int(state.pop("long_factor", 0))  # main_memory.py stored it as a buffer
    if long_factor:
        output, n_freq = "magnitude", n_out
    elif n_in == n_out and n_out % 2 == 0:
        output, n_freq = "complex_mask", n_out // 2
    else:
        output, n_freq = "magnitude", n_out
    merge(cfg, {
        "stft": {"n_fft": 2 * (n_freq - 1), "long_factor": long_factor, "causal": False},
        "model": {"type": "mlp", "output": output, "hidden": [w.shape[0] for w in weights[:-1]],
                  "mag_input": False, "mask_bound": 0},
    })
    print(f"  old checkpoint without settings: assuming hop {cfg['stft']['hop']} "
          f"(pass --set stft.hop=... if it was trained with another)")
    return state
