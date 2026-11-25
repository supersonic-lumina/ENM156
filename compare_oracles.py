"""
How good an ideal (oracle) mask can get on each representation: causal
STFTs (by default the one of [stft] and a 2 ms one with the learned
filterbank's default framing), and any filterbanks trained with
train_filterbank.py. Every row is scored on the
output waveform against the clean clip, so representations of any kind
compare directly:

  SNR / SI-SDR   of the whole clip (dB)
  pause, speech  how much lower the error is than the noise in pause and
                 speech frames (dB); speech = clean frames within 30 dB of
                 the clip's loudest (as the speech-vs-pause measurement)
  PESQ, STOI, ESTOI

These are upper bounds: what a denoiser predicting the mask perfectly would
reach, not what one will. Runs on the CPU: every clip has its own length,
and on the GPU each new length is tuned again.

  .venv/bin/python compare_oracles.py --filterbank runs/filterbank/filterbank.pt
  .venv/bin/python compare_oracles.py --stft 128/32 --filterbank runs/fb_k128_s32/filterbank.pt
"""
import argparse

import numpy as np
import torch

from denoiser.config import add_config_args, load_config, resolve_path
from denoiser.data import load_pairs, split_files
from denoiser.filterbank import load_filterbank, snr, stft_oracle
from evaluate import score_pair

SEGMENT = 512  # samples per frame for the speech / pause split


def si_sdr(est, ref):
    ref_part = (est * ref).sum() / ref.pow(2).sum().clamp(min=1e-8) * ref
    return snr(est, ref_part)


def speech_samples(clean):
    """True for the samples of speech frames (clean energy within 30 dB of the loudest frame)."""
    n = len(clean) // SEGMENT * SEGMENT
    energy = clean[:n].reshape(-1, SEGMENT).pow(2).sum(-1)
    db = 10 * torch.log10(energy + 1e-10)
    speech = (db > db.max() - 30).repeat_interleave(SEGMENT)
    return torch.cat([speech, speech.new_zeros(len(clean) - n)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filterbank", action="append", default=[],
                    help="A filterbank.pt to compare (repeatable).")
    ap.add_argument("--stft", action="append", default=[], metavar="N_FFT/HOP",
                    help="A causal STFT to compare (repeatable); default: [stft]'s and 32/16.")
    ap.add_argument("--num-clips", type=int, default=824)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set)
    sr, stft = cfg["data"]["sample_rate"], cfg["stft"]
    device = torch.device("cpu")

    methods = {"noisy": lambda noisy, clean: noisy}
    sizes = args.stft or [f"{stft['n_fft']}/{stft['hop']}", "32/16"]
    for n_fft, hop in (map(int, size.split("/")) for size in sizes):
        for form in ("ratio", "clip"):
            methods[f"STFT {n_fft}/{hop} {form}"] = (
                lambda noisy, clean, n_fft=n_fft, hop=hop, form=form:
                stft_oracle(noisy, clean, n_fft, hop, form))
    for path in args.filterbank:
        bank = load_filterbank(resolve_path(path), device).eval()
        methods[f"learned {path}"] = bank.oracle

    print(f"Ideal masks on {args.num_clips} test clips (ratio = |S| / (|S| + |N|), "
          f"clip = min(|S| / |X|, 1))")
    rows = {name: {"snr": [], "si_sdr": [], "pesq": [], "stoi": [], "estoi": []} for name in methods}
    energy = {name: np.zeros(4) for name in methods}  # noise pause, error pause, noise speech, error speech
    with torch.no_grad():
        for noisy, clean in load_pairs(split_files(cfg, "test"), args.num_clips, sr):
            noisy, clean = torch.from_numpy(noisy), torch.from_numpy(clean)
            speech = speech_samples(clean)
            noise = (noisy - clean).pow(2)
            for name, method in methods.items():
                est = method(noisy[None], clean[None])[0]
                error = (est - clean).pow(2)
                r = rows[name]
                r["snr"].append(snr(est, clean).item())
                r["si_sdr"].append(si_sdr(est, clean).item())
                for metric, value in score_pair(clean.numpy(), est.numpy(), sr).items():
                    r.setdefault(metric, []).append(value)
                energy[name] += [noise[~speech].sum().item(), error[~speech].sum().item(),
                                 noise[speech].sum().item(), error[speech].sum().item()]

    print(f"{'':42s}{'SNR':>7s}{'SI-SDR':>8s}{'pause':>7s}{'speech':>8s}"
          f"{'PESQ':>7s}{'STOI':>7s}{'ESTOI':>7s}"
          f"{'CSIG':>7s}{'CBAK':>7s}{'COVL':>7s}{'SSNR':>7s}")
    for name, r in rows.items():
        e = energy[name]
        pause, speech = (10 * np.log10(e[0] / max(e[1], 1e-12)),
                         10 * np.log10(e[2] / max(e[3], 1e-12)))
        print(f"{name:42s}{np.mean(r['snr']):7.2f}{np.mean(r['si_sdr']):8.2f}{pause:7.1f}"
              f"{speech:8.1f}{np.nanmean(r['pesq']):7.3f}{np.mean(r['stoi']):7.3f}"
              f"{np.mean(r['estoi']):7.3f}{np.nanmean(r['csig']):7.3f}"
              f"{np.nanmean(r['cbak']):7.3f}{np.nanmean(r['covl']):7.3f}"
              f"{np.mean(r['ssnr']):7.3f}")


if __name__ == "__main__":
    main()
