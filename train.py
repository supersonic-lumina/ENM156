"""
Trains the speech denoiser on VoiceBank-DEMAND-16k with MSE on frames.
Each training example is a crop of train.crop_frames consecutive frames
from one clip, plus the earlier frames the model looks back at (its
context); crop_frames = 1 with the MLP is plain shuffled single frames.
What goes in and out of the model is set in config.toml (see
denoiser/pipeline.py); personal settings go in config.local.toml.

  .venv/bin/python train.py
  .venv/bin/python train.py --set stft.long_factor=0 --set train.output_dir=runs/short_only
  .venv/bin/python train.py --set model.type=tcn --set train.crop_frames=200
"""
import argparse

import torch

from denoiser.config import (add_config_args, describe_device, get_device, load_config,
                             log_output, print_config, resolve_path, save_run_config)
from denoiser.data import (epoch_crops, gather_segments, load_pairs, release_freed_memory,
                           sample_ladder, split_files, stack_waves, swap_in_clean)
from denoiser.model import CheckpointSaver, LrSchedule, describe_model, initial_model, save_checkpoint
from denoiser.pipeline import loss_fn, save_samples, segment_frames


def load_split(files, cfg, max_pairs, audio_device):
    """All clips of a split as Waves on `audio_device`. The frames are made
    from the audio batch by batch: stored, they'd take ~8x the memory."""
    return stack_waves(load_pairs(files, max_pairs, cfg["data"]["sample_rate"]), cfg, audio_device)


def crop_batch(waves, clips, first, crop, history, cfg, device, clean_share=0):
    """Input frames of the crops with `history` earlier frames in front,
    (B, history + crop, n_in), and the target frames of the crops alone.
    clean_share: see data.swap_in_clean."""
    start = first - history
    noisy, clean = gather_segments(waves, clips, start, history + crop, cfg, device)
    noisy = swap_in_clean(noisy, clean, clean_share)
    x, y = segment_frames(noisy, clean, start, cfg)
    return x, y[:, history:]


def evaluate_loss(model, waves, cfg, device):
    """Mean loss over all frames, one whole clip at a time, as at test time."""
    total, zero = 0.0, torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        for i, n in enumerate(waves.frames.tolist()):
            x, y = crop_batch(waves, torch.tensor([i]), zero, n, 0, cfg, device)
            total += loss_fn(model(x), x, y, cfg).item() * n
    return total / int(waves.frames.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    cfg = load_config(ap.parse_args().set)
    tr = cfg["train"]
    out_dir = resolve_path(tr["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_output(out_dir / "log.txt", tr["init_from"])

    device, audio_device = get_device(cfg), get_device(cfg, "audio_device")
    print(f"Using device: {describe_device(device)}, training audio on {audio_device}")
    model, cfg = initial_model(cfg, device, tr["init_from"])  # before loading: it can set the stft
    save_run_config(cfg, out_dir / "config.toml")  # after initial_model: with init_from, its model settings
    print_config(cfg, ("stft", "model", "train"))
    print(describe_model(cfg))

    print(f"Loading up to {tr['max_train'] or 'all'} train pairs...")
    train = load_split(split_files(cfg, "train"), cfg, tr["max_train"], audio_device)
    print(f"Loading up to {tr['max_val'] or 'all'} val pairs...")
    val = load_split(split_files(cfg, "val"), cfg, tr["max_val"], audio_device)
    release_freed_memory()
    print(f"  train: {len(train.frames)} clips, {int(train.frames.sum())} frames   "
          f"val: {len(val.frames)} clips, {int(val.frames.sum())} frames")

    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])

    saver = CheckpointSaver(tr["save_checkpoints"], out_dir, higher_is_better=False)
    crop, history = tr["crop_frames"], model.context - 1
    crops_per_batch = max(1, tr["batch_size"] // crop)
    print(f"  crops of {crop} frames + {history} frames of history, {crops_per_batch} per batch")
    steps_per_epoch = -(-int((train.frames // crop).sum()) // crops_per_batch)  # one per batch
    schedule, step = LrSchedule(tr, steps_per_epoch), 0
    for epoch in range(1, tr["epochs"] + 1):
        clips, firsts = epoch_crops(train.frames, crop)
        total_loss = 0.0
        for i in range(0, len(clips), crops_per_batch):
            c, f = clips[i:i + crops_per_batch], firsts[i:i + crops_per_batch]
            x, y = crop_batch(train, c, f, crop, history, cfg, device, tr["clean_share"])
            lr = schedule.set(opt, step)
            step += 1
            opt.zero_grad()
            # one output per crop frame; the history frames are only looked at
            loss = loss_fn(model.forward_valid(x), x[:, history:], y, cfg)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(c)
        train_loss = total_loss / len(clips)

        model.eval()
        val_loss = evaluate_loss(model, val, cfg, device)
        model.train()
        print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"lr={lr:.1e}")
        saver.epoch_done(model, cfg, epoch, val_loss)

    save_checkpoint(model, cfg, out_dir / "model.pt")
    print(f"Saved model to {out_dir / 'model.pt'}")

    if tr["num_samples"]:
        save_samples(model, sample_ladder(cfg), cfg, device, out_dir)


if __name__ == "__main__":
    main()
