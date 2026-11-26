"""
Checks denoiser/dns4.py on small made-up data, without network access (the download itself isn't run):

  extract     a 48 kHz WAV comes out at 16 kHz with the same length and tone, other files are copied, unsafe paths
              are skipped, only finished archives are unpacked and marked done, --watch stops once the download is
              over, and a corrupt archive is logged as FAILED instead of ending the run
  synthesize  writes train and val shards that the training code (denoiser.data) reads, at the SNR and level in
              their metadata, validation clips only from validation files, the same shards again for the same seed,
              nothing new when run a second time, and refuses a folder holding mixtures of other settings

  .venv/bin/python tests/test_dns4.py
"""
from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from denoiser import dns4  # noqa: E402
from denoiser.data import load_pairs, split_files  # noqa: E402
from denoiser.make_mixtures import active_power  # noqa: E402

SR = dns4.SAMPLE_RATE
failed = False


def check(ok: bool, what: str) -> None:
    global failed
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    failed |= not ok


def add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def wav_data(wav: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def test_extract(tmp: Path) -> None:
    arch, out = tmp / "archives", tmp / "out"
    (arch / "noise_fullband").mkdir(parents=True)
    t = np.arange(48000) / 48000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    good = arch / "noise_fullband" / "a.tar.bz2"
    with tarfile.open(good, "w:bz2") as tar:
        add(tar, "datasets_fullband/noise_fullband/tone.wav", wav_data(tone, 48000))
        add(tar, "datasets_fullband/noise_fullband/notes.txt", b"hello")
        add(tar, "../escape.txt", b"bad")
    corrupt = arch / "c.tar.bz2"
    corrupt.write_bytes(b"not a bzip2 file")
    (arch / "b.tar.bz2").write_bytes(b"x" * 10)  # still downloading: shorter than its listed size
    blobs = tmp / "blobs.txt"
    blobs.write_text(f"{good.stat().st_size} noise_fullband/a.tar.bz2\n100 b.tar.bz2\n"
                     f"{corrupt.stat().st_size} c.tar.bz2\n")
    dns4.BLOBS = blobs  # the worker processes only need it through the paths they're given

    check(dns4.extract(arch, out, parallel=1, watch=False) == 1, "extract reports the archives it couldn't unpack")
    x, sr = sf.read(out / "datasets_fullband/noise_fullband/tone.wav", dtype="float32")
    check(sr == SR and len(x) == SR, f"the WAV is resampled to 16 kHz, 1 s (got {sr} Hz, {len(x)} samples)")
    check(np.max(np.abs(x[100:-100] - tone[::3][100:-100])) < 2e-3, "the tone survives resampling")
    check((out / "datasets_fullband/noise_fullband/notes.txt").read_bytes() == b"hello", "other files are copied")
    check(not (tmp / "escape.txt").exists(), "'..' paths are skipped")
    check(sorted(p.name for p in (out / ".done").iterdir()) == ["a.tar.bz2"],
          "only the finished, readable archive is marked done")
    log = (out / "extract.log").read_text()
    check("FAILED c.tar.bz2" in log, "the corrupt archive is logged as FAILED")
    check(not list(out.rglob("*.tmp")), "no temporary files are left")
    check(dns4.unpack(good, out).endswith("already done"), "a done archive is skipped")
    (arch / "download.log").write_text("11-26 00:12 all done: 2 of 3 archives\n")
    check(dns4.extract(arch, out, parallel=1, watch=True) == 1,
          "--watch stops once the download is over and nothing more can be unpacked")


def make_sources(root: Path) -> None:
    """Speech-like clean files (tone bursts, mostly active) and noise files, with some of each in validation."""
    rng = np.random.default_rng(0)
    clean_dir, noise_dir = root / dns4.CLEAN_DIR, root / dns4.NOISE_DIR
    for base, sub_dir, n_val, n_train in ((clean_dir, "read_speech/", 3, 12), (noise_dir, "", 3, 8)):
        (base / sub_dir).mkdir(parents=True, exist_ok=True)
        wanted, i = {True: n_val, False: n_train}, 0
        while wanted[True] or wanted[False]:
            name = f"{sub_dir}f{i:04d}.wav"  # as synthesize lists it: relative to clean_fullband / noise_fullband
            i += 1
            val = dns4.is_val(name)
            if not wanted[val]:
                continue
            wanted[val] -= 1
            n = int(SR * rng.uniform(0.8, 2.0))
            if base == noise_dir:
                wav = 0.1 * rng.standard_normal(n)
            else:
                t = np.arange(n) / SR
                wav = 0.3 * np.sin(2 * np.pi * rng.uniform(150, 300) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
            sf.write(base / name, wav.astype(np.float32), SR, subtype="PCM_16")
    sf.write(clean_dir / "read_speech" / "silent.wav", np.zeros(SR, np.float32), SR, subtype="PCM_16")


def synth(out: Path, root: Path, seed: int = 0, seconds: float = 3.0) -> int:
    return dns4.synthesize(out, root, hours=12 * 3 / 3600, val_hours=4 * 3 / 3600, seconds=seconds,
                           snr=(-5, 20), level=(-35, -15), seed=seed, parallel=2)


def test_synthesize(tmp: Path) -> None:
    root, out = tmp / "dns4", tmp / "mix"
    make_sources(root)
    dns4.SHARD = 5  # several shards per split (the workers only see each shard's clip count)
    check(synth(out, root) == 0, "synthesize finishes")
    names = sorted(p.name for p in out.glob("*.parquet"))
    expected = [f"train-{i:05d}-of-00003.parquet" for i in range(3)] + ["val-00000-of-00001.parquet"]
    check(names == expected, f"3 train shards (5, 5, 2 clips) and 1 val shard are written ({names})")
    cfg = {"data": {"voicebank_dir": "", "train_dir": str(out), "val_dir": str(out), "test_dir": ""}}
    train = list(load_pairs(split_files(cfg, "train"), 0, SR))
    val = list(load_pairs(split_files(cfg, "val"), 0, SR))
    check(len(train) == 12 and len(val) == 4, f"the training loader reads 12 train and 4 val clips "
                                              f"(got {len(train)}, {len(val)})")
    check(all(len(n) == len(c) == 3 * SR for n, c in train + val), "every clip is 3 s, noisy and clean alike")

    meta = pq.read_table(split_files(cfg, "train")).to_pylist()
    errors = []
    for (noisy, clean), row in zip(train, meta):
        snr = 10 * np.log10(active_power(clean, SR) / np.mean((noisy - clean) ** 2))
        level = 10 * np.log10(np.mean(noisy ** 2))
        errors.append((abs(snr - row["snr"]), abs(level - row["level"]) if row["gain"] == 1 else 0))
    snr_err, level_err = np.max(errors, axis=0)
    check(snr_err < 0.2, f"each clip's SNR matches its metadata (max error {snr_err:.3f} dB)")
    check(level_err < 0.2, f"each clip's level matches its metadata (max error {level_err:.3f} dB)")
    check(all(-5 <= r["snr"] <= 20 and -35 <= r["level"] <= -15 for r in meta), "SNRs and levels are in range")
    val_meta = pq.read_table(split_files(cfg, "val")).to_pylist()
    used_val = [f for r in val_meta for f in r["speech"].split(",")]
    used_train = [f for r in meta for f in r["speech"].split(",")]
    check(all(dns4.is_val(f) for f in used_val) and not any(dns4.is_val(f) for f in used_train),
          "validation speech only from validation files, training speech never from them")
    check(not any("silent" in f for f in used_val + used_train), "the silent clean file is never used")

    before = {p.name: p.stat().st_mtime_ns for p in out.glob("*.parquet")}
    check(synth(out, root) == 0 and before == {p.name: p.stat().st_mtime_ns for p in out.glob("*.parquet")},
          "a second run keeps the finished shards")
    again = tmp / "mix_again"
    synth(again, root)
    check(all((again / n).read_bytes() == (out / n).read_bytes() for n in before), "the same seed gives the same shards")
    for change, kwargs in (("clip length", {"seconds": 2.0}), ("seed", {"seed": 1})):
        try:
            synth(out, root, **kwargs)
            check(False, f"a folder with mixtures of another {change} is refused")
        except SystemExit:
            check(True, f"a folder with mixtures of another {change} is refused")


def test_stop(tmp: Path) -> None:
    folder = tmp / "job"
    folder.mkdir()
    (folder / "download.pid").write_text("999999\n")  # stale: no download holds the lock
    dns4.stop(folder)
    check(not (folder / "download.pid").exists(), "stop ignores (and removes) a stale pid file")
    sleeper = subprocess.Popen([sys.executable, "-c", "import sys, time; sys.path.insert(0, sys.argv[1]); "
                                "from denoiser import dns4; from pathlib import Path; "
                                "lock = dns4.claim(Path(sys.argv[2]), 'extract'); time.sleep(60)",
                                str(ROOT), str(folder)], stdin=subprocess.DEVNULL)
    for _ in range(100):
        if dns4.is_running(folder, "extract"):
            break
        time.sleep(0.05)
    dns4.stop(folder)
    check(sleeper.wait(timeout=10) != 0, "stop ends a running job's process group")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for test in (test_extract, test_synthesize, test_stop):
            sub = Path(tmp) / test.__name__
            sub.mkdir()
            test(sub)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
