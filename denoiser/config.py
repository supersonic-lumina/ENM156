"""
Settings come from config.toml (shared defaults), then config.local.toml
(personal, git-ignored), then the command line: --config FILE (a TOML file of
settings, e.g. one from experiments/ or a run's own config.toml) and --set
section.key=value, applied in the order given.

Every training run writes the settings it used to config.toml in its output
folder (save_run_config), and its log starts with the exact command, so a run
can be repeated or shared with --config runs/<run>/config.toml.
"""
import json
import shlex
import sys
import tomllib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = ROOT / "config.local.toml"
MODEL_SECTIONS = ("stft", "model", "tcn", "snn")  # what a checkpoint stores: needed to rebuild the model


def merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_set(item):
    """'train.epochs=5' -> {'train': {'epochs': 5}}. Values are read as TOML
    (numbers, true/false, lists), anything else as a plain string."""
    key, _, value = item.partition("=")
    section, _, name = key.strip().partition(".")
    if not name:
        raise SystemExit(f"--set needs section.key=value, got {item!r}")
    try:
        value = tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        value = value.strip()
    return {section: {name: value}}


CONFIG_FILE_MARK = "@"  # how --config files are told apart from --set items in the same list


def read_toml(path: Path) -> dict[str, object]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(overrides: Sequence[str] = ()) -> dict[str, object]:
    """The settings: config.toml, config.local.toml, then `overrides` in order:
    --set items ("section.key=value") and --config files ("@" + path)."""
    cfg = read_toml(ROOT / "config.toml")
    if LOCAL_CONFIG.exists():
        merge(cfg, read_toml(LOCAL_CONFIG))
    for item in overrides:
        if item.startswith(CONFIG_FILE_MARK):
            file = item[len(CONFIG_FILE_MARK):].strip()
            if not file:
                raise SystemExit("--config: missing file path")
            path = resolve_path(file)
            if not path.is_file():
                raise SystemExit(f"--config: no such file {path}")
            merge(cfg, read_toml(path))
        else:
            merge(cfg, parse_set(item))
    return cfg


def add_config_args(ap):
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="Override a setting for this run, e.g. --set train.epochs=5 (repeatable).")
    ap.add_argument("--config", action="append", dest="set", metavar="FILE",
                    type=lambda path: CONFIG_FILE_MARK + path,
                    help="Read settings from a TOML file, e.g. experiments/snn-1x256.toml or a run's "
                         "config.toml (repeatable; applied in order with --set).")


# Machine-specific settings (paths, GPU), saved only as comments in a run's config.toml.
MACHINE_SETTINGS = {
    "data": ("voicebank_dir", "train_dir", "val_dir", "test_dir", "audio_cache",
             "device", "audio_device", "vram_fraction"),
    "mix": ("vctk_dir", "demand_dir", "voicebank_logs"),
}
# Also saved only as comments, so loading a run's settings can't overwrite that run.
OUTPUT_SETTINGS = {"train": ("output_dir",), "filterbank": ("output_dir",)}


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    if isinstance(value, list):
        return f"[{', '.join(toml_value(v) for v in value)}]"
    raise TypeError(f"can't write {value!r} ({type(value).__name__}) as a TOML value")


def save_run_config(cfg: dict[str, object], path: Path) -> None:
    """Writes the settings of a run to `path` as TOML that --config can load:
    every section and key, except the machine and output settings above,
    which are listed as comments instead."""
    left_out = dict(MACHINE_SETTINGS)
    for section, keys in OUTPUT_SETTINGS.items():
        left_out[section] = left_out.get(section, ()) + keys

    lines = [
        f"# Settings of this run, written {datetime.now():%Y-%m-%d %H:%M} by:",
        f"#   {command_line()}",
        "# Repeat it with --config <this file> --set <section>.output_dir=<a new folder>.",
        "# Left out because they belong to the machine or the run folder (set your own):",
    ]
    for section, keys in left_out.items():
        values = cfg.get(section)
        if not isinstance(values, dict):
            continue
        for key in keys:
            if key in values:
                lines.append(f"#   {section}.{key} = {toml_value(home_as_tilde(values[key]))}")

    for section, values in cfg.items():
        if not isinstance(values, dict):
            raise TypeError(f"setting {section!r} is not in a [section]")
        lines.extend(["", f"[{section}]"])
        for key, value in values.items():
            if key not in left_out.get(section, ()):
                lines.append(f"{key} = {toml_value(value)}")
    Path(path).write_text("\n".join(lines) + "\n")


def home_as_tilde(value: object) -> object:
    """A path under the home folder as ~/..., so a shared config.toml doesn't carry your user name."""
    if isinstance(value, str) and Path(value).is_relative_to(Path.home()):
        rel = Path(value).relative_to(Path.home()).as_posix()
        return "~" if rel == "." else f"~/{rel}"
    return value


def command_line() -> str:
    """The command this process was started with, ready to paste."""
    return shlex.join([Path(sys.executable).name, *sys.argv])


def print_config(cfg, sections):
    for section in sections:
        print(f"  [{section}] " + json.dumps(cfg[section])[1:-1])


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class _Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def log_output(path, init_from=""):
    """Copy everything printed from here on (stdout and stderr) to path,
    replacing an older log there, like model.pt is replaced. The log starts
    with the command the run was started with. A run that
    starts from a checkpoint (init_from) begins its log with the log.txt
    next to that checkpoint, so the log covers the model's whole training;
    if that log can't be read, the new log starts on its own. Each run's part
    of the log begins with a ===== line saying when it started."""
    previous, problem = "", ""
    if init_from:
        try:  # read before opening path: it can be the same file
            previous = (resolve_path(init_from).parent / "log.txt").read_text()
        except (OSError, UnicodeDecodeError) as e:
            problem = f"{type(e).__name__}: {e}"
    file = open(path, "w", buffering=1)  # line-buffered: a crash still leaves the log
    now = f"{datetime.now():%Y-%m-%d %H:%M}"
    if previous:
        file.write(previous.rstrip("\n") + f"\n\n===== continued from {init_from} at {now} =====\n\n")
    else:
        file.write(f"===== started at {now} =====\n\n")
    file.write(f"command: {command_line()}\n\n")
    sys.stdout, sys.stderr = _Tee(sys.stdout, file), _Tee(sys.stderr, file)
    if problem:
        print(f"Couldn't add the log of {init_from}'s run ({problem}); this log starts here")


def get_device(cfg, key="device"):
    """The torch device data.<key> names. A GPU is capped at data.vram_fraction
    of its memory, so going over it is an out-of-memory error instead of the
    driver moving memory out to RAM, which stalls the GPU and the desktop."""
    name = cfg["data"][key] or cfg["data"]["device"]
    device = torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and cfg["data"]["vram_fraction"] < 1:
        torch.cuda.set_per_process_memory_fraction(cfg["data"]["vram_fraction"], device.index or 0)
    return device


def describe_device(device):
    return f"{device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
