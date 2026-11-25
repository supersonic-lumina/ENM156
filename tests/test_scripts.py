"""
Smoke test of the scripts, without a dataset: makes a tiny fake VoiceBank-DEMAND
(and fake VCTK / DEMAND / VoiceBank log files) in a temp folder, then runs each
script end to end on it with tiny settings, on the CPU:
  - denoiser.make_mixtures, both recipes
  - train.py, train_pesq.py (a spiking model, on make_mixtures' output),
    train_filterbank.py
  - evaluate.py on a model and on saved samples, make_samples.py, prune.py,
    compare_oracles.py
This only checks that the scripts run and write what they should, not that
anything trains well. config.local.toml is still read, so every setting that
matters here is set with --set.

  .venv/bin/python tests/test_scripts.py
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 16000

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


# ---------------------------------------------------------------- fake data


def speech(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-like audio: harmonic syllables with pauses, and quiet ends, so
    make_mixtures finds where the speech starts and stops."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = rng.uniform(100, 220)
    voice = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 8))
    syllables = np.clip(np.sin(2 * np.pi * rng.uniform(3, 5) * t), 0, None)
    edges = (t > 0.15) & (t < seconds - 0.15)
    wav = 0.1 * voice * syllables * edges + 1e-4 * rng.standard_normal(len(t))
    return wav.astype(np.float32)


def noise(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    return (0.03 * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def write_parquet(path: Path, clips: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """VoiceBank-DEMAND's layout: id, and clean/noisy as {bytes, path} of a WAV."""
    rows = [{"id": clip_id,
             "clean": {"bytes": wav_bytes(clean, SR), "path": f"{clip_id}.wav"},
             "noisy": {"bytes": wav_bytes(noisy, SR), "path": f"{clip_id}.wav"}}
            for clip_id, clean, noisy in clips]
    pq.write_table(pa.Table.from_pylist(rows), path)


def make_data(data: Path) -> None:
    rng = np.random.default_rng(0)

    # VoiceBank-DEMAND: 6 training clips of two speakers and two noise types, 4 test clips
    voicebank = data / "voicebank"
    voicebank.mkdir(parents=True)
    train, log = [], []
    for i in range(6):
        clip_id, kind = f"p22{5 + i % 2}_{i:03d}", ("cafe", "car")[i % 2]
        clean = speech(2.5, SR, rng)
        train.append((clip_id, clean, clean + noise(2.5, SR, rng)))
        log.append(f"{clip_id} {kind} 5")
    test = []
    for i in range(4):
        clean = speech(2.5, SR, rng)
        test.append((f"p232_{i:03d}", clean, clean + noise(2.5, SR, rng)))
    write_parquet(voicebank / "train-00000-of-00001.parquet", train)
    write_parquet(voicebank / "test-00000-of-00001.parquet", test)

    logs = data / "logfiles"
    logs.mkdir()
    (logs / "log_trainset_28spk.txt").write_text("\n".join(log) + "\n")
    (logs / "log_testset.txt").write_text("\n".join(f"{c} bus 5" for c, _, _ in test) + "\n")

    # VCTK: the two VoiceBank training speakers and four others, 48 kHz FLAC
    vctk = data / "vctk"
    speakers = {"p225": "F", "p226": "M", "p300": "F", "p301": "M", "p302": "F", "p303": "M"}
    info = ["ID  AGE  GENDER  ACCENTS  REGION"]
    for speaker, gender in speakers.items():
        info.append(f"{speaker}  23  {gender}  English  Somewhere")
        folder = vctk / "wav48_silence_trimmed" / speaker
        folder.mkdir(parents=True)
        for n in range(3):
            sf.write(folder / f"{speaker}_{n:03d}_mic1.flac", speech(1.5, 48000, rng), 48000)
    (vctk / "speaker-info.txt").write_text("\n".join(info) + "\n")

    # DEMAND: one environment to use, one of VoiceBank's test set that must be skipped;
    # 30 s, so the 10 % kept for validation is longer than a clip
    for env in ("DKITCHEN", "TBUS"):
        (data / "demand" / env).mkdir(parents=True)
        sf.write(data / "demand" / env / "ch01.wav", noise(30, SR, rng), SR)


# ---------------------------------------------------------------- running


def toml(value: str) -> str:
    return json.dumps(value)  # a TOML string, so paths with ':' or spaces stay whole


def run(what: str, args: list[str], cwd: Path) -> bool:
    """Runs `python <args>` (from cwd, with the repository importable) and
    checks it exits 0; on failure, prints the end of its output."""
    start = time.time()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=600)
    ok = proc.returncode == 0
    check(ok, f"{what} runs ({time.time() - start:.0f} s)")
    if not ok:
        print("\n".join(("    " + line) for line in (proc.stdout + proc.stderr).splitlines()[-30:]))
    return ok


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data, runs = tmp / "data", tmp / "runs"
        make_data(data)

        common = [  # every run: the fake data, the CPU, nothing shared with a real setup
            "--set", f"data.voicebank_dir={toml((data / 'voicebank').as_posix())}",
            "--set", "data.train_dir=\"\"", "--set", "data.val_dir=\"\"", "--set", "data.test_dir=\"\"",
            "--set", "data.device=\"cpu\"", "--set", "data.audio_device=\"\"",
            "--set", "data.audio_cache=\"\"",
            "--set", f"mix.vctk_dir={toml((data / 'vctk').as_posix())}",
            "--set", f"mix.demand_dir={toml((data / 'demand').as_posix())}",
            "--set", f"mix.voicebank_logs={toml((data / 'logfiles').as_posix())}",
            "--set", "model.filterbank=\"\"", "--set", "train.init_from=\"\"", "--set", "pesq.init_from=\"\"",
        ]
        tiny = ["--set", "train.epochs=1", "--set", "train.max_train=4", "--set", "train.max_val=2",
                "--set", "train.num_samples=2", "--set", "train.save_checkpoints=2"]

        # make_mixtures: both recipes
        mix = ["--set", "mix.seconds=2.0", "--set", "mix.val_speakers=2",
               "--set", "mix.voicebank_passes=1", "--set", "mix.vctk_passes=1"]
        for recipe in ("voicebank", "vctk_demand"):
            out = data / f"mixed_{recipe}"
            if run(f"make_mixtures {recipe}",
                   ["-m", "denoiser.make_mixtures", recipe, str(out), *common, *mix], tmp):
                check(any(out.glob("train-*.parquet")) and any(out.glob("val-*.parquet")),
                      f"make_mixtures {recipe} writes train and val parquet files")

        # train.py: the default MLP, small
        mlp = runs / "mlp"
        if run("train.py", [str(ROOT / "train.py"), *common, *tiny,
                            "--set", "model.hidden=[32]", "--set", "train.batch_size=64",
                            "--set", f"train.output_dir={toml(mlp.as_posix())}"], tmp):
            check((mlp / "model.pt").is_file() and (mlp / "log.txt").is_file()
                  and len(list((mlp / "samples").iterdir())) == 2,
                  "train.py writes model.pt, log.txt and the samples")

        # train_pesq.py: a small spiking network, on make_mixtures' output
        snn = runs / "snn"
        mixed = (data / "mixed_vctk_demand").as_posix()
        if run("train_pesq.py (snn)", [str(ROOT / "train_pesq.py"), *common, *tiny,
                                       "--set", f"data.train_dir={toml(mixed)}",
                                       "--set", f"data.val_dir={toml(mixed)}",
                                       "--set", "model.type=\"snn\"", "--set", "snn.hidden=[16, 16]",
                                       "--set", "pesq.batch_size=2", "--set", "pesq.crop_seconds=1.0",
                                       "--set", "train.seed=0",
                                       "--set", f"train.output_dir={toml(snn.as_posix())}"], tmp):
            check((snn / "model.pt").is_file(), "train_pesq.py writes model.pt")

        # train_filterbank.py
        fb = runs / "filterbank"
        if run("train_filterbank.py", [str(ROOT / "train_filterbank.py"), *common,
                                       "--set", "filterbank.filters=16", "--set", "filterbank.epochs=1",
                                       "--set", "filterbank.batch_size=2", "--set", "filterbank.crop_seconds=1.0",
                                       "--set", "filterbank.max_train=4", "--set", "filterbank.max_val=2",
                                       "--set", f"filterbank.output_dir={toml(fb.as_posix())}"], tmp):
            check((fb / "filterbank.pt").is_file(), "train_filterbank.py writes filterbank.pt")

        # evaluation and the tools that use a trained model
        if (mlp / "model.pt").is_file():
            run("evaluate.py --model", [str(ROOT / "evaluate.py"), *common, "--model", str(mlp / "model.pt"),
                                        "--num-clips", "2"], tmp)
            run("evaluate.py on samples", [str(ROOT / "evaluate.py"), *common, "--model", "",
                                           "--samples-dir", str(mlp / "samples")], tmp)
            run("make_samples.py", [str(ROOT / "make_samples.py"), *common, "--set", "train.num_samples=2",
                                    str(mlp / "model.pt")], tmp)
        if (snn / "model.pt").is_file():
            if run("prune.py", [str(ROOT / "prune.py"), *common, str(snn / "model.pt"),
                                str(runs / "pruned.pt"), "--keep", "0.5"], tmp):
                check((runs / "pruned.pt").is_file(), "prune.py writes the pruned checkpoint")
        if (fb / "filterbank.pt").is_file():
            run("compare_oracles.py", [str(ROOT / "compare_oracles.py"), *common, "--num-clips", "2",
                                       "--filterbank", str(fb / "filterbank.pt")], tmp)

    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
