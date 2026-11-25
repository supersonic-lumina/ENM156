"""
Step 1 of two-step training: learns the encoder/decoder of [filterbank]
(denoiser/filterbank.py) so that the ideal mask on its latent,
|E(clean)| / (|E(clean)| + |E(noise)|), applied to E(noisy) and decoded,
gives back the clean speech. The loss is the SNR of that output, plus a
differentiable PESQ loss when filterbank.pesq_weight is above 0, on random
fixed-length crops. No denoiser is trained here: the result says how good a
mask on this latent could get, and is the representation a denoiser would
then learn to mask (step 2). Validation shows the same ideal mask on the
causal STFT of [stft] next to it.

  .venv/bin/python train_filterbank.py
  .venv/bin/python train_filterbank.py --set filterbank.filters=512 --set filterbank.output_dir=runs/fb512
  .venv/bin/python train_filterbank.py --set filterbank.pesq_weight=1 --set filterbank.output_dir=runs/fb_pesq
"""
import argparse
import random

import torch

from denoiser.config import (add_config_args, describe_device, get_device, load_config,
                             log_output, print_config, resolve_path, save_run_config)
from denoiser.data import first_crops, load_crops, random_crops, split_files
from denoiser.filterbank import (describe_filterbank, make_filterbank, save_filterbank, snr,
                                 stft_oracle)


class OracleLoss:
    """-SNR / 10, plus pesq_weight * the PESQ loss of train_pesq.py (both ~1).
    Returns the per-batch loss and, for printing, the mean SNR and PESQ estimate."""

    def __init__(self, fb, device, sample_rate):
        self.weight, self.pesq = fb["pesq_weight"], None
        if self.weight:
            from denoiser.pesq_loss import make_pesq_loss  # only needed for the PESQ term
            self.pesq = make_pesq_loss(device, sample_rate)

    def __call__(self, est, clean):
        snrs = snr(est, clean)
        loss, terms = -snrs.mean() / 10, {"SNR": snrs.mean().item()}
        if self.pesq is not None:
            pesq_loss = self.pesq(clean, est)
            loss = loss + self.weight * pesq_loss.mean()
            terms["PESQ"] = self.pesq.mos(clean, est.detach()).mean().item()
        return loss, terms


def describe(terms):
    return "  ".join(f"{name}={value:.2f}" if name == "SNR" else f"{name}={value:.3f}"
                     for name, value in terms.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    cfg = load_config(ap.parse_args().set)
    fb, sr = cfg["filterbank"], cfg["data"]["sample_rate"]
    out_dir = resolve_path(fb["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_output(out_dir / "log.txt")
    save_run_config(cfg, out_dir / "config.toml")

    device = get_device(cfg)
    print(f"Using device: {describe_device(device)}")
    print_config(cfg, ("filterbank",))
    print(describe_filterbank(fb, sr))
    bank = make_filterbank(fb).to(device)
    print(f"  {sum(p.numel() for p in bank.parameters()) / 1e3:.1f} k parameters")

    crop = int(fb["crop_seconds"] * sr)
    print(f"Loading up to {fb['max_train'] or 'all'} train and {fb['max_val'] or 'all'} val clips...")
    train = load_crops(split_files(cfg, "train"), fb["max_train"], sr, crop)
    val = first_crops(load_crops(split_files(cfg, "val"), fb["max_val"], sr, crop), crop, device)
    print(f"  {len(train)} train and {len(val[0])} val clips of at least {fb['crop_seconds']} s")

    stft = cfg["stft"]
    criterion = OracleLoss(fb, device, sr)
    print("loss = -SNR / 10" + (f" + {fb['pesq_weight']} * PESQ loss" if criterion.pesq else ""))
    with torch.no_grad():
        _, noisy_terms = criterion(val[0], val[1])
        _, stft_terms = criterion(stft_oracle(*val, stft["n_fft"], stft["hop"]), val[1])
    print(f"val: noisy {describe(noisy_terms)}, ideal mask on the STFT "
          f"({stft['n_fft']}/{stft['hop']}) {describe(stft_terms)}")

    opt = torch.optim.Adam(bank.parameters(), lr=fb["lr"])
    # per-epoch factor that takes lr to lr_end by the last epoch
    gamma = (fb["lr_end"] / fb["lr"]) ** (1 / max(fb["epochs"] - 1, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    best = None
    for epoch in range(1, fb["epochs"] + 1):
        bank.train()
        random.shuffle(train)
        sums, steps = {}, 0
        for i in range(0, len(train) - fb["batch_size"] + 1, fb["batch_size"]):
            noisy, clean = random_crops(train[i:i + fb["batch_size"]], crop, device)
            loss, terms = criterion(bank.oracle(noisy, clean), clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + value
            steps += 1

        bank.eval()
        with torch.no_grad():
            val_loss, val_terms = criterion(bank.oracle(*val), val[1])
        train_text = describe({name: total / steps for name, total in sums.items()})
        print(f"epoch {epoch:3d}  train: {train_text}  val: {describe(val_terms)}  "
              f"val_loss={val_loss.item():.4f}  lr={opt.param_groups[0]['lr']:.1e}")
        if best is None or val_loss.item() < best:
            best = val_loss.item()
            save_filterbank(bank, fb, out_dir / "filterbank.pt")
            print("  saved filterbank.pt")
        scheduler.step()
    print(f"Best val loss {best:.4f}, in {out_dir / 'filterbank.pt'}")


if __name__ == "__main__":
    main()
