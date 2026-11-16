"""Loading VoiceBank-DEMAND clips from the parquet files, fixed-length
waveform crops, and memory helpers."""
import ctypes
import hashlib
import io
import os
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

from .config import resolve_path
from .features import pad_for_frames, widest_window


def split_files(cfg, split):
    """Sorted parquet files of a split ("train", "val" or "test"), from
    data.<split>_dir, or data.voicebank_dir when that is "". VoiceBank has no
    validation split, so there "val" falls back to its test files."""
    data = cfg["data"]
    folder = resolve_path(data[f"{split}_dir"] or data["voicebank_dir"])
    files = sorted(folder.glob(f"{split}-*.parquet"))
    if not files and split == "val" and not data["val_dir"]:
        files = sorted(folder.glob("test-*.parquet"))
    if not files:
        raise SystemExit(f"No {split}-*.parquet files in {folder} "
                         f"(set data.voicebank_dir or data.{split}_dir in config.local.toml)")
    return files


def load_pairs(files, max_pairs, sample_rate, dtype="float32", every=1):
    """Yield (noisy, clean) waveforms (float32, or the files' 16-bit values with dtype="int16") one clip at a time, each pair
    cut to the shorter length; a blank or 0 `max_pairs` reads every clip.
    Files are read a few clips at a time: a whole file would take ~1.4 GB of
    RAM while it's decoded. every = N keeps each Nth clip only: the files are
    sorted by speaker, so this thins the set out evenly, unlike max_pairs."""
    count, seen = 0, -1
    for f in files:
        for batch in pq.ParquetFile(str(f)).iter_batches(batch_size=64, columns=["noisy", "clean"]):
            for noisy, clean in zip(batch.column("noisy"), batch.column("clean")):
                if max_pairs and count >= max_pairs:
                    return
                seen += 1
                if seen % every:
                    continue
                n, nsr = sf.read(io.BytesIO(noisy.as_py()["bytes"]), dtype=dtype)
                c, csr = sf.read(io.BytesIO(clean.as_py()["bytes"]), dtype=dtype)
                assert csr == sample_rate and nsr == sample_rate
                length = min(len(n), len(c))
                yield n[:length], c[:length]
                count += 1
        release_freed_memory()


SAMPLE_SNR_RANGE = (15, -6)  # dB, first and last sample clip


def sample_ladder(cfg):
    """The clips of the sample output, [(name, noisy, clean)]: the same
    validation clips for every model (evenly spread over the split, so over
    its speakers and noises), each remixed to a fixed SNR: train.num_samples
    clips in even steps over SAMPLE_SNR_RANGE (8 clips = 3 dB steps). Sorted
    by name they go from least to most noisy.
    The SNR is speech power over noise power of the whole clip; a mix that
    would clip is turned down, together with its clean reference."""
    files = split_files(cfg, "val")
    total = sum(pq.ParquetFile(str(f)).metadata.num_rows for f in files)
    n, (first, last) = cfg["train"]["num_samples"], SAMPLE_SNR_RANGE
    snrs = [round(first + (last - first) * i / max(n - 1, 1), 1) for i in range(n)]
    pairs = load_pairs(files, 0, cfg["data"]["sample_rate"], every=max(total // max(n, 1), 1))
    ladder = []
    for i, (snr, (noisy, clean)) in enumerate(zip(snrs, pairs)):
        noise = noisy - clean
        gain = ((clean ** 2).mean() / ((noise ** 2).mean() * 10 ** (snr / 10) + 1e-12)) ** 0.5
        mix, name = clean + gain * noise, f"{i:02d}_snr_{snr:+g}dB"
        scale = min(1.0, 0.99 / max(abs(mix).max(), 1e-9))
        ladder.append((name, (mix * scale).astype("float32"), (clean * scale).astype("float32")))
    return ladder


def cached_pairs(files, sample_rate, cache_dir):
    """Every clip of the files as (noisy, clean) int16 arrays that live in two
    files in cache_dir instead of in this process: the arrays are views of
    those files (memory-mapped), so runs side by side share one copy in RAM,
    and the system can drop and re-read parts when RAM is short. The first
    run decodes the clips and writes the files; a run that starts meanwhile
    waits for it."""
    stamp = [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files]
    base = Path(cache_dir) / hashlib.sha1(repr((stamp, sample_rate)).encode()).hexdigest()[:16]
    names = {part: Path(f"{base}.{part}") for part in ("noisy.i16", "clean.i16", "lengths.npy")}
    base.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{base}.lock", "w") as lock:
        try:
            import fcntl  # not on Windows: there two first runs may both write, each its own files
            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:
            pass
        if not names["lengths.npy"].exists():
            print(f"  writing the decoded audio to {base}.* (once; later runs share it)", flush=True)
            temp = {part: Path(f"{base}.{os.getpid()}.{part}") for part in names}
            lengths = []
            with open(temp["noisy.i16"], "wb") as noisy_file, open(temp["clean.i16"], "wb") as clean_file:
                for noisy, clean in load_pairs(files, 0, sample_rate, "int16"):
                    noisy_file.write(noisy.tobytes())
                    clean_file.write(clean.tobytes())
                    lengths.append(len(noisy))
            with open(temp["lengths.npy"], "wb") as f:
                np.save(f, np.array(lengths, dtype=np.int64))
            for part in names:  # the lengths last: they mark the cache as complete
                os.replace(temp[part], names[part])
    lengths = np.load(names["lengths.npy"])
    ends = np.cumsum(lengths)
    # "c": pages are shared with the file until written to, which nothing does
    noisy, clean = (np.asarray(np.memmap(names[part], dtype=np.int16, mode="c")) for part in ("noisy.i16", "clean.i16"))
    return [(noisy[end - length:end], clean[end - length:end]) for end, length in zip(ends, lengths)]


def load_crops(files, max_clips, sample_rate, crop, every=1, cache_dir=None):
    """Clips at least `crop` samples long, as (noisy, clean) int16 arrays: the
    16-bit values of the files, half the RAM of float32. random_crops and
    first_crops turn their pieces into floats. With a cache_dir the audio is
    shared between runs (cached_pairs) instead of loaded into this one."""
    if cache_dir:
        pairs = cached_pairs(files, sample_rate, cache_dir)[::every]
        pairs = pairs[:max_clips] if max_clips else pairs
    else:
        pairs = load_pairs(files, max_clips, sample_rate, "int16", every)
    return [(n, c) for n, c in pairs if len(n) >= crop]


def to_float(int16_wav, device):
    return int16_wav.to(device).float() / 32768


def random_crops(pairs, crop, device):
    """A `crop`-sample piece of each pair from a random start, stacked to (B, crop)
    tensors. Every batch has the same shape, so the GPU's convolution library
    (MIOpen) only searches algorithms for it once, in the first batch."""
    starts = [random.randint(0, len(n) - crop) for n, _ in pairs]
    noisy = torch.stack([torch.from_numpy(n[s:s + crop]) for (n, _), s in zip(pairs, starts)])
    clean = torch.stack([torch.from_numpy(c[s:s + crop]) for (_, c), s in zip(pairs, starts)])
    return to_float(noisy, device), to_float(clean, device)


def first_crops(pairs, crop, device):
    """The first `crop` samples of every pair: the same validation set each epoch."""
    noisy = torch.stack([torch.from_numpy(n[:crop]) for n, _ in pairs])
    clean = torch.stack([torch.from_numpy(c[:crop]) for _, c in pairs])
    return to_float(noisy, device), to_float(clean, device)


def stack_and_free(chunks, device, trim_every=256):
    """torch.cat equivalent that doesn't hold the chunks and the result at
    the same time: each chunk is copied into a preallocated tensor and freed
    right away. The big allocation is mmap'd, so its pages only become
    resident as they're written, while the freed chunks are handed back to
    the OS every `trim_every` chunks. Peak stays ~1x the data instead of 2x."""
    total = sum(c.shape[0] for c in chunks)
    out = torch.empty((total, *chunks[0].shape[1:]), dtype=chunks[0].dtype, device=device)
    chunks.reverse()  # pop() from the end keeps the original order
    pos = 0
    while chunks:
        c = chunks.pop()
        out[pos:pos + c.shape[0]] = c
        pos += c.shape[0]
        del c
        if len(chunks) % trim_every == 0:
            release_freed_memory()
    return out


def release_freed_memory():
    """The per-clip chunks freed after loading are small heap allocations
    that glibc keeps reserved instead of returning to the OS, so RSS stays at
    the loading peak. malloc_trim hands those pages back; pyarrow has its own
    allocator, which keeps the parquet read buffers until told otherwise."""
    pa.default_memory_pool().release_unused()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc


def swap_in_clean(noisy, clean, share):
    """Each row's noisy input replaced by its clean target with probability
    `share`: clean-to-clean examples, so the model learns that "no noise"
    happens and leaves clean speech alone. Rows are (B, ...) tensors."""
    if not share:
        return noisy
    swap = torch.rand(noisy.shape[0], device=noisy.device) < share
    return torch.where(swap.view(-1, *[1] * (noisy.dim() - 1)), clean, noisy)


class Waves(NamedTuple):
    """Clips stored for frame-free training: each clip padded the way
    analysis() pads it for the widest window (so frame t of every window
    starts at sample t * hop of the padded clip), as int16, end to end."""
    noisy: torch.Tensor  # (samples,) int16
    clean: torch.Tensor
    starts: torch.Tensor  # (clips,) where each padded clip starts
    sizes: torch.Tensor  # (clips,) padded length
    frames: torch.Tensor  # (clips,) frames per clip, as analysis() gives


def stack_waves(pairs, cfg, device):
    """(noisy, clean) float32 pairs -> Waves on `device`. int16 is lossless
    for the 16-bit WAVs in the parquet files and half the size of float32;
    it is also far smaller than the frames themselves (257 values per hop
    of 128 samples, for input and target)."""
    stft = cfg["stft"]
    wide = widest_window(stft)
    noisy_chunks, clean_chunks = [], []
    for noisy, clean in pairs:
        for wav, chunks in ((noisy, noisy_chunks), (clean, clean_chunks)):
            padded = pad_for_frames(torch.from_numpy(wav), wide, stft)
            chunks.append((padded * 32768).round().clamp(-32768, 32767).to(torch.int16))
    sizes = torch.tensor([len(c) for c in noisy_chunks])
    return Waves(stack_and_free(noisy_chunks, device), stack_and_free(clean_chunks, device),
                 torch.cumsum(sizes, 0) - sizes, sizes, (sizes - wide) // stft["hop"] + 1)


def epoch_crops(frames, crop_frames):
    """One epoch of training crops, in random order. frames: frames per clip.
    Every clip is cut into as many crops of crop_frames as fit, from a random
    offset, so the few frames left over change between epochs (crop_frames =
    1: every frame once). Returns the clip of each crop and its first frame."""
    counts = frames // crop_frames
    spare = frames - counts * crop_frames
    offsets = (torch.rand(len(frames)) * (spare + 1)).long()
    clips = torch.repeat_interleave(torch.arange(len(frames)), counts)
    nth = torch.arange(int(counts.sum())) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    perm = torch.randperm(len(clips))
    return clips[perm], (offsets[clips] + nth * crop_frames)[perm]


def gather_segments(waves, clips, first, n_frames, cfg, device):
    """The (B, samples) noisy and clean float32 waveform pieces that frames
    first .. first + n_frames - 1 of the clips are cut from. `first` can be
    negative: samples before the padded clip are zeros."""
    hop = cfg["stft"]["hop"]
    length = (n_frames - 1) * hop + widest_window(cfg["stft"])
    clips, first = clips.to(waves.noisy.device), first.to(waves.noisy.device)
    pos = first[:, None] * hop + torch.arange(length, device=waves.noisy.device)
    inside = (pos >= 0) & (pos < waves.sizes.to(pos.device)[clips, None])
    idx = (waves.starts.to(pos.device)[clips, None] + pos).clamp(0, len(waves.noisy) - 1)
    return tuple((wav[idx] * inside).to(device).float() / 32768 for wav in (waves.noisy, waves.clean))
