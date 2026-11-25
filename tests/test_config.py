"""
Checks the settings files, without a dataset:
  - --config and --set apply in the order given; a missing file is an error
  - a run's config.toml loads back to the same settings, except the machine
    and output settings, which it lists as comments instead
  - every file in experiments/ loads and only sets keys config.toml knows
  - the log starts with the command

  .venv/bin/python tests/test_config.py
"""
import argparse
import copy
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from denoiser.config import (MACHINE_SETTINGS, OUTPUT_SETTINGS, add_config_args, load_config,  # noqa: E402
                             read_toml, save_run_config)

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def parse(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    return load_config(ap.parse_args(argv).set)


def refused(fn: Callable[[], object]) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = tmp / "a.toml"
        a.write_text("[train]\nepochs = 7\nlr = 0.5\n")
        cfg = parse(["--config", str(a), "--set", "train.epochs=3"])
        check(cfg["train"]["epochs"] == 3 and cfg["train"]["lr"] == 0.5, "--set after --config wins")
        cfg = parse(["--set", "train.epochs=3", "--config", str(a)])
        check(cfg["train"]["epochs"] == 7, "--config after --set wins")
        check(refused(lambda: parse(["--config", str(tmp / "missing.toml")])), "a missing --config file is an error")
        check(refused(lambda: parse(["--config", ""])), "an empty --config path is an error")

        # a run's config.toml: same settings back, minus machine and output settings
        cfg = parse(["--set", "train.epochs=3", "--set", "snn.hidden=[64, 32]", "--set", "model.type=\"snn\""])
        cfg["data"]["voicebank_dir"] = "/somewhere/only/on/my/machine"
        cfg["data"]["train_dir"] = str(Path.home() / "datasets" / "mixtures")
        saved = tmp / "config.toml"
        save_run_config(cfg, saved)
        text = saved.read_text()
        loaded = tomllib.loads(text)
        expected = copy.deepcopy(cfg)
        for section, keys in {**MACHINE_SETTINGS, **OUTPUT_SETTINGS}.items():
            for key in keys:
                expected.get(section, {}).pop(key, None)
        check(loaded == expected, "a run's config.toml has every other setting, unchanged")
        check('#   data.train_dir = "~/datasets/mixtures"' in text and str(Path.home()) not in text,
              "paths under the home folder are written as ~/...")
        reloaded = parse(["--config", str(saved)])
        check(reloaded["train"]["epochs"] == 3 and reloaded["snn"]["hidden"] == [64, 32]
              and reloaded["data"]["voicebank_dir"] == load_config()["data"]["voicebank_dir"],
              "--config with a run's config.toml repeats its settings, keeps your own data folders")

    defaults = read_toml(ROOT / "config.toml")
    for path in sorted((ROOT / "experiments").glob("*.toml")):
        unknown = [f"{s}.{k}" for s, keys in read_toml(path).items() for k in keys if k not in defaults.get(s, {})]
        check(not unknown and not refused(lambda: parse(["--config", str(path)])),
              f"experiments/{path.name} loads" + (f" (unknown keys {unknown})" if unknown else ""))

    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
