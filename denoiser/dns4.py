"""
The Microsoft DNS Challenge 4 (ICASSP 2022) fullband data, which the Intel N-DNS Challenge also uses (source:
microsoft/DNS-Challenge download-dns-challenge-4.sh; CC BY 4.0): 127 .tar.bz2 archives, 582.4 GB, listed with their
sizes in dns4_blobs.txt. See docs/datasets.md.

  download ARCH_DIR      fetches every archive into ARCH_DIR, --parallel at a time (default 20: the server gives each
                         connection only ~1.3 MB/s). Resumable: a finished archive (the right size) is skipped and a
                         partial .part file is continued. Log: ARCH_DIR/download.log.
  extract ARCH_DIR       unpacks every finished archive into data.dns4_dir (or --out), resampling each WAV to
                         16 kHz 16-bit PCM (about a third of the 48 kHz size: ~300 GB in all). Each file is written
                         under a temporary name and renamed, and <out>/.done/<archive> marks a finished archive, so
                         it can simply be run again. --watch keeps checking for newly downloaded archives until the
                         download is done, so both can run at once. Log: <out>/extract.log.
  synthesize OUT_DIR     makes noisy/clean training and validation mixtures from the unpacked data, as the N-DNS
                         Challenge does (see Mixtures below), and writes them as train-*.parquet and val-*.parquet
                         for data.train_dir / data.val_dir. Resumable: finished shards are kept.
  stop DIR               stops a download, extract or synthesize running in DIR (the folder it writes to) and
                         everything it started; partial files stay and are continued next time.

  nohup .venv/bin/python -m denoiser.dns4 download /mnt/hdd/dns4/archives > /dev/null 2>&1 &
  nohup .venv/bin/python -m denoiser.dns4 extract /mnt/hdd/dns4/archives --watch > /dev/null 2>&1 &
  nohup .venv/bin/python -m denoiser.dns4 synthesize /mnt/hdd/dns4/mixtures_500h > /dev/null 2>&1 &
  .venv/bin/python -m denoiser.dns4 stop /mnt/hdd/dns4/archives

Mixtures (defaults as the N-DNS Challenge's noisyspeech_synthesizer): each clip is --seconds long. Its speech is
random clean files (any language or speaker) joined with 0.2 s of silence between them; files that are mostly
silent or clipped are skipped. Its noise is random noise files joined end to end. The noise is scaled to an SNR
drawn uniformly from --snr (over the speech frames, as in make_mixtures.py), then the mixture and its clean
reference together to a level drawn uniformly from --level (dBFS RMS of the mixture), turned down further if the
mixture would clip. About 1 % of the clean and of the noise files (chosen by a hash of their path) are kept for
validation, so validation speech and noise never occur in training; speakers can still overlap, since the file
names don't name speakers in every language. Testing stays on VoiceBank-DEMAND (data.test_dir).

Needs Linux, macOS or WSL (process groups and file locks).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import signal
import subprocess
import sys
import tarfile
import time
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import IO

import numpy as np

from .config import add_config_args, load_config, resolve_path

URL = "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
BLOBS = Path(__file__).with_name("dns4_blobs.txt")
SAMPLE_RATE = 16000
# The blob store's default anonymous API version answers an open-ended range (what curl -C - sends to resume)
# with the whole file; this version answers it with the rest.
CURL = ["curl", "-sS", "--fail", "-L", "-C", "-", "-H", "x-ms-version: 2020-04-08", "--retry", "10",
        "--retry-delay", "30", "--speed-limit", "100000", "--speed-time", "120"]
TRIES = 6
RETRY_WAIT = 60  # seconds between tries of one archive
WATCH_WAIT = 300  # seconds between checks for newly downloaded archives

CLEAN_DIR = "datasets_fullband/clean_fullband"  # in data.dns4_dir
NOISE_DIR = "datasets_fullband/noise_fullband"
DEV_TESTSET = "datasets_fullband/dev_testset/noisy_testclips"  # the dev test set's noisy clips (no references)
SHARD = 1000  # clips per parquet file
GAP_SECONDS = 0.2  # silence between joined clean files
# A clean file is used if at least MIN_ACTIVE_SHARE of its 10 ms frames are speech (within make_mixtures.ACTIVE_DB
# of its loudest frame and louder than SILENCE_DB dBFS) and at most MAX_CLIPPED_SHARE of its samples at full scale.
MIN_ACTIVE_SHARE = 0.6
SILENCE_DB = -60
MAX_CLIPPED_SHARE = 1e-4
VAL_PERCENT = 1  # share of clean and noise files kept for validation

Log = Callable[[str], None]


# ---------------------------------------------------------------- shared


def blobs() -> list[tuple[int, str]]:
    """(size in bytes, blob path) of every archive."""
    rows = (line.split() for line in BLOBS.read_text().splitlines() if line.strip())
    return [(int(size), blob) for size, blob in rows]


def is_finished(arch: Path, size: int, blob: str) -> bool:
    path = arch / blob
    return path.is_file() and path.stat().st_size == size


def open_log(path: Path) -> Log:
    stream = open(path, "a", buffering=1)  # line-buffered: each message is on disk at once

    def log(msg: str) -> None:
        stream.write(f"{time.strftime('%m-%d %H:%M')} {msg}\n")
    return log


def claim(folder: Path, job: str) -> IO[str]:
    """Makes sure only one `job` runs in `folder`: holds <folder>/.<job>.lock (returned; keep it open) and writes
    <folder>/<job>.pid, the id of this process's group, so `stop` can end it and everything it started. Started
    detached (nohup ... &, no terminal on stdin), the process first gets a group of its own; in a terminal it
    stays in the shell's, so Ctrl-C keeps working."""
    import fcntl

    if not sys.stdin.isatty() and os.getpgrp() != os.getpid():
        os.setpgrp()
    lock = open(folder / f".{job}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another {job} is running in {folder}")
    (folder / f"{job}.pid").write_text(f"{os.getpgrp()}\n")
    return lock


def is_running(folder: Path, job: str) -> bool:
    """Whether a `job` holds its lock in `folder` (a stale pid file alone doesn't count)."""
    import fcntl

    lock_path = folder / f".{job}.lock"
    if not lock_path.exists():
        return False
    with open(lock_path) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def stop(folder: Path) -> None:
    for job in ("download", "extract", "synthesize"):
        pid_file = folder / f"{job}.pid"
        if not pid_file.exists():
            continue
        if is_running(folder, job):  # else the group id may belong to an unrelated process by now
            os.killpg(int(pid_file.read_text()), signal.SIGTERM)
            print(f"stopped the {job} in {folder}")
        else:
            print(f"no {job} running in {folder}")
        pid_file.unlink()


# ---------------------------------------------------------------- download


def fetch(arch: Path, size: int, blob: str, log: Log) -> bool:
    """Downloads one archive (continuing a .part file); True once it's complete."""
    path = arch / blob
    part = path.with_name(path.name + ".part")
    if is_finished(arch, size, blob):
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, TRIES + 1):
        if part.exists() and part.stat().st_size > size:
            part.unlink()
        rc = subprocess.run([*CURL, "-o", str(part), f"{URL}/{blob}"], capture_output=True).returncode
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            part.rename(path)
            log(f"done {blob}")
            return True
        if rc == 33:  # the server refused to resume: start again from 0
            part.unlink(missing_ok=True)
        log(f"retry {attempt} {blob} (curl {rc}, {have} of {size} bytes)")
        if attempt < TRIES:
            time.sleep(RETRY_WAIT)
    log(f"FAILED {blob}")
    return False


def download(arch: Path, parallel: int) -> int:
    arch.mkdir(parents=True, exist_ok=True)
    lock = claim(arch, "download")  # noqa: F841 (held until exit)
    log = open_log(arch / "download.log")
    todo = blobs()
    log(f"start, {len(todo)} archives, {parallel} at a time")
    with ThreadPoolExecutor(parallel) as pool:
        n_ok = sum(pool.map(lambda row: fetch(arch, row[0], row[1], log), todo))
    log(f"all done: {n_ok} of {len(todo)} archives" + ("" if n_ok == len(todo) else " (see FAILED above)"))
    return 0 if n_ok == len(todo) else 1


# ---------------------------------------------------------------- extract


def is_safe(name: str) -> bool:
    """A relative archive path that stays inside the output folder."""
    return not name.startswith("/") and ".." not in name.split("/") and name.strip("/") != ""


def write_atomic(path: Path, write: Callable[[Path], object]) -> None:
    """write(temporary path), then rename it to `path`: a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def resample_wav(data: bytes, rate: int) -> np.ndarray:
    """WAV file bytes -> float32 samples at `rate`, (samples,) or (samples, channels), within 16-bit range."""
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, file_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if file_rate != rate:
        g = gcd(file_rate, rate)
        wav = resample_poly(wav, rate // g, file_rate // g, axis=0).astype(np.float32)
    wav = np.clip(wav, -1.0, 32767 / 32768)
    return wav[:, 0] if wav.shape[1] == 1 else wav


def unpack(archive: Path, out: Path, rate: int = SAMPLE_RATE) -> str:
    """Unpacks one archive into `out` in its own layout (datasets_fullband/...), WAVs resampled to `rate`. Entries
    that aren't plain files or directories, or whose path isn't safe, are skipped. Returns a line for the log."""
    import soundfile as sf

    done = out / ".done" / archive.name
    if done.exists():
        return f"{archive.name}: already done"
    n_files = n_skipped = 0
    with tarfile.open(archive, "r|bz2") as tar:  # streaming: one sequential pass over the archive
        for member in tar:
            if not is_safe(member.name) or not (member.isfile() or member.isdir()):
                n_skipped += 1
                continue
            if member.isdir():
                continue
            target = out / member.name
            data = tar.extractfile(member).read()
            if member.name.lower().endswith(".wav"):
                wav = resample_wav(data, rate)
                write_atomic(target, lambda p: sf.write(str(p), wav, rate, subtype="PCM_16", format="WAV"))
            else:
                write_atomic(target, lambda p: p.write_bytes(data))
            n_files += 1
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(f"{n_files} files, {n_skipped} entries skipped\n")
    return f"{archive.name}: {n_files} files, {n_skipped} skipped"


def unpack_logged(archive: Path, out: Path) -> str:
    """unpack, with a failure (a corrupt archive, a full disk) as a log line instead of ending the whole run."""
    try:
        return unpack(archive, out)
    except Exception as e:  # noqa: BLE001
        return f"FAILED {archive.name}: {type(e).__name__}: {e}"


def is_download_over(arch: Path) -> bool:
    log = arch / "download.log"
    lines = log.read_text().splitlines() if log.exists() else []
    return bool(lines) and "all done" in lines[-1]


def extract(arch: Path, out: Path, parallel: int, watch: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    lock = claim(out, "extract")  # noqa: F841
    log = open_log(out / "extract.log")
    archives = blobs()

    def is_done(blob: str) -> bool:
        return (out / ".done" / Path(blob).name).exists()

    def ready() -> list[Path]:
        return [arch / blob for size, blob in archives if is_finished(arch, size, blob) and not is_done(blob)]

    failed: set[Path] = set()
    with ProcessPoolExecutor(parallel, initializer=os.nice, initargs=(15,)) as pool:
        while True:
            todo = [a for a in ready() if a not in failed]
            for archive, msg in zip(todo, pool.map(unpack_logged, todo, [out] * len(todo))):
                log(msg)
                if msg.startswith("FAILED"):
                    failed.add(archive)
            n_done = sum(is_done(blob) for _, blob in archives)
            if n_done == len(archives):
                log(f"all {n_done} archives extracted")
                return 0
            waiting = [a for a in ready() if a not in failed]
            if waiting:
                continue
            if not watch or is_download_over(arch):
                log(f"{n_done} of {len(archives)} archives extracted; {len(failed)} failed, "
                    f"the rest aren't downloaded")
                return 1
            time.sleep(WATCH_WAIT)


# ---------------------------------------------------------------- synthesize


@dataclass(frozen=True)
class Shard:
    """One parquet file of mixtures: everything needed to make it, so shards are independent and the same settings
    give the same shard on any machine with the same unpacked data."""
    split: str  # "train" or "val"
    index: int
    count: int  # shards in the split
    clips: int
    out: Path
    root: Path  # data.dns4_dir
    seconds: float
    snr: tuple[float, float]
    level: tuple[float, float]
    seed: int

    @property
    def path(self) -> Path:
        return self.out / f"{self.split}-{self.index:05d}-of-{self.count:05d}.parquet"


def plan_shards(n_clips: int, **settings) -> list[Shard]:
    """The shards of one split: SHARD clips each, the last one the rest."""
    count = max(1, -(-n_clips // SHARD))
    return [Shard(index=i, count=count, clips=min(SHARD, n_clips - i * SHARD), **settings) for i in range(count)]


def is_val(path: str) -> bool:
    """Whether a source file is kept for validation: the same answer on every machine and every run."""
    return zlib.crc32(path.encode()) % 100 < VAL_PERCENT


def wav_files(folder: Path) -> list[str]:
    """Every .wav under `folder`, as sorted paths relative to it."""
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*.wav"))


def read_mono(path: Path) -> np.ndarray:
    import soundfile as sf

    wav, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: {rate} Hz, expected {SAMPLE_RATE} (run extract first)")
    return wav.mean(axis=1)


def is_usable_speech(wav: np.ndarray) -> bool:
    """Mostly speech (MIN_ACTIVE_SHARE of its frames active) and not clipped."""
    from .make_mixtures import ACTIVE_DB, frame_db

    if len(wav) < SAMPLE_RATE // 2:
        return False
    db, _ = frame_db(wav, SAMPLE_RATE)
    active = np.mean((db > db.max() - ACTIVE_DB) & (db > SILENCE_DB))
    clipped = np.mean(np.abs(wav) >= 0.99)
    return bool(active >= MIN_ACTIVE_SHARE and clipped <= MAX_CLIPPED_SHARE)


def has_sound(wav: np.ndarray) -> bool:
    """A noise file that isn't digital silence (which no gain could bring to an SNR)."""
    return bool((wav ** 2).mean() > 1e-9)


def joined(files: Sequence[str], folder: Path, length: int, rng: random.Random, gap: int,
           usable: Callable[[np.ndarray], bool]) -> tuple[np.ndarray, list[str]]:
    """Random usable files from `files` joined, `gap` samples of silence between them, cut to `length` samples.
    Returns the audio and the files used."""
    pieces: list[np.ndarray] = []
    used: list[str] = []
    filled = rejected = 0
    while filled < length:
        name = files[rng.randrange(len(files))]
        wav = read_mono(folder / name)
        if not usable(wav):
            rejected += 1
            if rejected > 1000:
                raise ValueError(f"over 1000 unusable files for one clip in {folder}")
            continue
        if pieces and gap:
            pieces.append(np.zeros(gap, dtype=np.float32))
            filled += gap
        pieces.append(wav)
        used.append(name)
        filled += len(wav)
    return np.concatenate(pieces)[:length], used


def mix_at(clean: np.ndarray, noise: np.ndarray, snr: float, level: float) -> tuple[np.ndarray, np.ndarray, float]:
    """noise scaled to `snr` dB below the speech frames of `clean`, then mixture and clean reference together to
    `level` dBFS RMS of the mixture, turned down further if the mixture would clip. Returns (clean, noisy, the
    clip-protection gain: 1 = none)."""
    from .make_mixtures import active_power

    noise_gain = np.sqrt(active_power(clean, SAMPLE_RATE) / ((noise ** 2).mean() * 10 ** (snr / 10) + 1e-12))
    noisy = clean + noise_gain * noise
    gain = 10 ** (level / 20) / np.sqrt((noisy ** 2).mean())
    protect = min(1.0, 0.99 / (np.abs(noisy).max() * gain))
    gain *= protect
    return (clean * gain).astype(np.float32), (noisy * gain).astype(np.float32), float(protect)


# The source files of each split, set once in each worker process (start_worker): they're too big (~1.1 M paths)
# to send along with every shard.
SOURCES: dict[str, tuple[list[str], list[str]]] = {}


def start_worker(sources: dict[str, tuple[list[str], list[str]]], niceness: int) -> None:
    SOURCES.update(sources)
    os.nice(niceness)


def write_shard(shard: Shard) -> str:
    """Makes one parquet file of mixtures (in a worker process). Returns a line for the log."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .make_mixtures import wav_bytes

    clean_files, noise_files = SOURCES[shard.split]
    clean_dir, noise_dir = shard.root / CLEAN_DIR, shard.root / NOISE_DIR
    length, gap = round(shard.seconds * SAMPLE_RATE), round(GAP_SECONDS * SAMPLE_RATE)
    rng = random.Random(f"{shard.seed}-{shard.split}-{shard.index}")
    rows, n_protected = [], 0
    for n in range(shard.clips):
        clean, speech_used = joined(clean_files, clean_dir, length, rng, gap, is_usable_speech)
        noise, noise_used = joined(noise_files, noise_dir, length, rng, 0, has_sound)
        snr, level = rng.uniform(*shard.snr), rng.uniform(*shard.level)
        clean, noisy, protect = mix_at(clean, noise, snr, level)
        n_protected += protect < 1
        clip_id = f"{shard.split}_{shard.index:05d}_{n:04d}"
        rows.append({"id": clip_id,
                     "clean": {"bytes": wav_bytes(clean, SAMPLE_RATE), "path": f"{clip_id}.wav"},
                     "noisy": {"bytes": wav_bytes(noisy, SAMPLE_RATE), "path": f"{clip_id}.wav"},
                     "snr": snr, "level": level, "gain": protect,
                     "speech": ",".join(speech_used), "noise": ",".join(noise_used)})
    write_atomic(shard.path, lambda p: pq.write_table(pa.Table.from_pylist(rows), p))
    return f"wrote {shard.path.name}: {shard.clips} clips, {n_protected} turned down to avoid clipping"


def synthesize(out: Path, root: Path, hours: float, val_hours: float, seconds: float, snr: tuple[float, float],
               level: tuple[float, float], seed: int, parallel: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    lock = claim(out, "synthesize")  # noqa: F841
    log = open_log(out / "synthesize.log")
    # Shards of other settings in the same folder would silently mix two data sets: the settings are kept in
    # settings.json, and a run with other ones is refused.
    chosen = {"hours": hours, "val_hours": val_hours, "seconds": seconds, "snr": list(snr), "level": list(level),
              "seed": seed}
    settings_file = out / "settings.json"
    if settings_file.exists() and json.loads(settings_file.read_text()) != chosen:
        raise SystemExit(f"{out} holds mixtures made with other settings ({settings_file.read_text().strip()}): "
                         f"use another folder, or the same settings")
    if not settings_file.exists() and any(out.glob("*.parquet")):
        raise SystemExit(f"{out} holds parquet files that synthesize didn't make: use another folder")
    clips = {"train": round(hours * 3600 / seconds), "val": round(val_hours * 3600 / seconds)}
    if min(clips.values()) < 1:
        raise SystemExit(f"--hours and --val-hours must each give at least one clip of {seconds:g} s")
    settings_file.write_text(json.dumps(chosen) + "\n")
    common = {"out": out, "root": root, "seconds": seconds, "snr": snr, "level": level, "seed": seed}
    shards = {split: plan_shards(n, split=split, **common) for split, n in clips.items()}
    planned = {s.path for split_shards in shards.values() for s in split_shards}
    for folder in (root / CLEAN_DIR, root / NOISE_DIR):
        if not folder.is_dir():
            raise SystemExit(f"no {folder}: run extract first, or set data.dns4_dir")

    print(f"listing the files in {root} (a few minutes on an HDD) ...", flush=True)
    clean, noise = wav_files(root / CLEAN_DIR), wav_files(root / NOISE_DIR)
    sources = {split: ([f for f in clean if is_val(f) == (split == "val")],
                       [f for f in noise if is_val(f) == (split == "val")]) for split in shards}
    for split, split_shards in shards.items():
        log(f"{split}: {sum(s.clips for s in split_shards)} clips of {seconds:g} s from {len(sources[split][0])} "
            f"clean and {len(sources[split][1])} noise files; SNR {snr[0]:g}..{snr[1]:g} dB, "
            f"level {level[0]:g}..{level[1]:g} dBFS")
    todo = [s for split_shards in shards.values() for s in split_shards if not s.path.exists()]
    log(f"{len(todo)} of {len(planned)} shards to write, {parallel} at a time")
    try:
        with ProcessPoolExecutor(parallel, initializer=start_worker, initargs=(sources, 10)) as pool:
            for msg in pool.map(write_shard, todo):
                log(msg)
    except Exception as e:  # noqa: BLE001 (stderr is usually /dev/null here: the log is where it's seen)
        log(f"FAILED: {type(e).__name__}: {e} (finished shards are kept; run again to continue)")
        return 1
    log("all done")
    return 0


# ---------------------------------------------------------------- command line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch the archives")
    d.add_argument("arch_dir", type=Path)
    d.add_argument("--parallel", type=int, default=20, help="downloads at a time")
    e = sub.add_parser("extract", help="unpack the archives at 16 kHz")
    e.add_argument("arch_dir", type=Path)
    e.add_argument("--out", type=Path, help="default: data.dns4_dir")
    e.add_argument("--parallel", type=int, default=4, help="archives at a time (an HDD saturates at about 4)")
    e.add_argument("--watch", action="store_true", help="wait for archives still downloading")
    add_config_args(e)
    s = sub.add_parser("synthesize", help="make train/val mixtures (parquet)")
    s.add_argument("out_dir", type=Path)
    s.add_argument("--hours", type=float, default=500, help="training audio (N-DNS: 500 h, ~115 GB)")
    s.add_argument("--val-hours", type=float, default=0.5)
    s.add_argument("--seconds", type=float, default=30, help="clip length")
    s.add_argument("--snr", type=float, nargs=2, default=(-5, 20), metavar=("MIN", "MAX"), help="dB")
    s.add_argument("--level", type=float, nargs=2, default=(-35, -15), metavar=("MIN", "MAX"), help="dBFS")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--parallel", type=int, default=4, help="shards at a time")
    add_config_args(s)
    t = sub.add_parser("stop", help="stop a running download, extract or synthesize")
    t.add_argument("dir", type=Path)
    args = ap.parse_args()

    if args.cmd == "download":
        return download(args.arch_dir, args.parallel)
    if args.cmd == "stop":
        stop(args.dir)
        return 0
    dns4_dir = resolve_path(load_config(args.set)["data"]["dns4_dir"])
    if args.cmd == "extract":
        return extract(args.arch_dir, args.out or dns4_dir, args.parallel, args.watch)
    if args.seconds <= 0 or args.snr[0] > args.snr[1] or args.level[0] > args.level[1] or args.level[1] > 0:
        raise SystemExit("need --seconds > 0, MIN <= MAX for --snr and --level, and --level at most 0 dBFS")
    return synthesize(args.out_dir, dns4_dir, args.hours, args.val_hours, args.seconds, tuple(args.snr),
                      tuple(args.level), args.seed, args.parallel)


if __name__ == "__main__":
    sys.exit(main())
