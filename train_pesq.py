"""
Trains the same denoiser as train.py, but with a PESQ loss on whole clips
instead of MSE on frames; optionally mixed with train.py's MSE and a STOI
loss (pesq.mse_weight, pesq.stoi_weight; see config.toml). Each batch is a
piece of pesq.crop_seconds from a random start in each of a few clips;
clips shorter than that are left out, so every batch has the same shape.
The model runs on every frame, the output is turned back into audio and
scored with a differentiable PESQ estimate against the clean piece. Model and STFT settings come from
config.toml, like train.py; clips per batch is pesq.batch_size.

  .venv/bin/python train_pesq.py --set train.output_dir=runs/pesq
  .venv/bin/python train_pesq.py --set pesq.init_from=runs/x/model.pt --set train.lr=1e-4 \
      --set train.epochs=5 --set train.output_dir=runs/x_pesq     # fine-tune a trained model
"""
import argparse
import random

import torch

from denoiser.config import (add_config_args, describe_device, get_device, load_config,
                             log_output, print_config, resolve_path)
from denoiser.data import (load_crops, load_pairs, random_crops, sample_ladder, split_files,
                           swap_in_clean)
from denoiser.model import CheckpointSaver, LrSchedule, describe_model, initial_model, save_checkpoint
from denoiser.features import analysis
from denoiser.pesq_loss import make_pesq_loss
from denoiser.pipeline import enhance, filterbank, loss_fn, run_model, save_samples, target_frames


def whole_clip(pair, device, step):
    """One validation clip as (1, samples) tensors, cut to a multiple of
    `step` samples: fewer different lengths for the GPU to tune for."""
    noisy, clean = pair
    length = len(noisy) // step * step or len(noisy)  # a clip shorter than one step is kept whole
    return (torch.from_numpy(noisy[:length])[None].to(device),
            torch.from_numpy(clean[:length])[None].to(device))


def loss_to_mos(loss, factor):
    """torch-pesq's per-clip loss -> its PESQ (MOS) estimate, as PesqLoss.mos
    computes it from the same distances: loss = factor * raw distance."""
    mos = 4.5 - loss / factor
    return 0.999 + 4 / (1 + torch.exp(-1.3669 * mos + 3.8224))


class CombinedLoss:
    """PESQ loss, plus the MSE and STOI terms when their weight in [pesq] is
    above 0. Each term is divided by its typical size (the *_scale settings),
    so the weights say how much a term counts next to the PESQ loss (~1).
    Returns the total and, for printing, the estimated PESQ, the MSE and the
    STOI (higher PESQ and STOI are better, lower MSE)."""

    def __init__(self, cfg, device, sample_rate):
        self.cfg, self.p = cfg, cfg["pesq"]
        self.pesq = make_pesq_loss(device, sample_rate)
        if self.p["mse_weight"] and filterbank(cfg) is not None:
            raise SystemExit("pesq.mse_weight must be 0 with model.filterbank: the MSE term is on STFT frames")
        self.stoi = None
        if self.p["stoi_weight"]:
            from torch_stoi import NegSTOILoss  # only needed for the STOI term
            self.stoi = NegSTOILoss(sample_rate=sample_rate).to(device)

    def __call__(self, model, noisy, clean):
        est, pred, x = run_model(model, noisy, self.cfg)
        pesq_loss = self.pesq(clean, est)  # per clip
        total = pesq_loss.mean()
        terms = {"pesq": loss_to_mos(pesq_loss.detach(), self.pesq.factor).mean()}
        if self.p["mse_weight"]:
            stft = self.cfg["stft"]
            y = target_frames(analysis(clean, stft["n_fft"], stft), self.cfg)
            terms["mse"] = loss_fn(pred, x, y, self.cfg)
            total = total + self.p["mse_weight"] * terms["mse"] / self.p["mse_scale"]
        if self.stoi is not None:
            terms["stoi"] = -self.stoi(est, clean).mean()
            total = total + self.p["stoi_weight"] * (1 - terms["stoi"]) / self.p["stoi_scale"]
        if getattr(model, "spike_rate", None) is not None:  # spiking model: spikes per neuron per step
            terms["rate"] = model.spike_rate
            total = total + self.cfg["snn"]["rate_weight"] * terms["rate"]
        return total, terms


def mean_mos(pesq_loss, pairs, cfg, device, step, model=None):
    """Mean PESQ estimate over the clips, one clip at a time. Without a model
    it scores the noisy input itself (the baseline to beat)."""
    total = 0.0
    with torch.no_grad():
        for pair in pairs:
            noisy, clean = whole_clip(pair, device, step)
            est = enhance(model, noisy, cfg) if model is not None else noisy
            total += pesq_loss.mos(clean, est).item()
    return total / len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    cfg = load_config(ap.parse_args().set)
    tr, sr = cfg["train"], cfg["data"]["sample_rate"]
    out_dir = resolve_path(tr["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_output(out_dir / "log.txt", cfg["pesq"]["init_from"])

    if tr["seed"] >= 0:
        random.seed(tr["seed"])
        torch.manual_seed(tr["seed"])
    device = get_device(cfg)
    print(f"Using device: {describe_device(device)}")
    rate_weight = cfg["snn"]["rate_weight"]
    model, cfg = initial_model(cfg, device, cfg["pesq"]["init_from"])
    cfg["snn"]["rate_weight"] = rate_weight  # a setting of the loss, not of the checkpoint's model
    print_config(cfg, ("stft", "model", *(("snn",) if cfg["model"]["type"] == "snn" else ()), "train", "pesq"))
    print(describe_model(cfg))

    print(f"Loading up to {tr['max_train'] or 'all'} train and {tr['max_val'] or 'all'} val clips...")
    crop = round(cfg["pesq"]["crop_seconds"] * sr)
    cache = cfg["data"]["audio_cache"]
    train = load_crops(split_files(cfg, "train"), tr["max_train"], sr, crop, tr["train_every"],
                       resolve_path(cache) if cache else None)
    val = list(load_pairs(split_files(cfg, "val"), tr["max_val"], sr, every=tr["val_every"]))
    hours = sum(len(n) for n, _ in train) / sr / 3600
    print(f"  train clips: {len(train)} of at least {cfg['pesq']['crop_seconds']} s "
          f"({hours:.1f} h of audio)   val clips: {len(val)}")

    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    criterion = CombinedLoss(cfg, device, sr)
    step = round(cfg["pesq"]["crop_step"] * sr)
    saver = CheckpointSaver(tr["save_checkpoints"], out_dir, higher_is_better=True)

    print(f"noisy val_PESQ (baseline): {mean_mos(criterion.pesq, val, cfg, device, step):.3f}")
    size = cfg["pesq"]["batch_size"]
    batches = range(0, len(train) - size + 1, size)  # the last partial batch is left out: one shape
    schedule, n_steps = LrSchedule(tr, len(batches)), 0
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        total_loss, term_sums, skipped = 0.0, {}, 0
        random.shuffle(train)
        for i in batches:
            noisy, clean = random_crops(train[i:i + size], crop, device)
            noisy = swap_in_clean(noisy, clean, tr["clean_share"])
            lr = schedule.set(opt, n_steps)  # skipped batches count too, so the run's length is fixed
            n_steps += 1
            opt.zero_grad()
            loss, terms = criterion(model, noisy, clean)
            if not torch.isfinite(loss):  # e.g. an all-silent output
                skipped += 1
                continue
            loss.backward()
            opt.step()
            total_loss += loss.item()
            for name, value in terms.items():
                term_sums[name] = term_sums.get(name, 0.0) + value.item()
        n_done = max(len(batches) - skipped, 1)
        train_loss = total_loss / n_done
        terms_text = "  ".join(f"{name}={value / n_done:.4f}" for name, value in term_sums.items())

        model.eval()
        val_pesq = mean_mos(criterion.pesq, val, cfg, device, step, model)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} ({terms_text})  val_PESQ={val_pesq:.3f}  "
              f"lr={lr:.1e}"
              + (f"  (skipped {skipped} non-finite batches)" if skipped else ""))
        saver.epoch_done(model, cfg, epoch, val_pesq)

    save_checkpoint(model, cfg, out_dir / "model.pt")
    print(f"Saved model to {out_dir / 'model.pt'}")

    if tr["num_samples"]:
        save_samples(model, sample_ladder(cfg), cfg, device, out_dir)


if __name__ == "__main__":
    main()
