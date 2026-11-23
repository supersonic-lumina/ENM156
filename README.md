# Speech denoiser

A small causal network that removes noise from speech, trained on
VoiceBank-DEMAND-16k: a frame-by-frame MLP, a TCN that also looks at
the past half second, or a spiking network (SNN) that remembers the past in
its neurons (`model.type`). Frames are causal by default, so the same model can
run live (`denoiser/stream.py`) with a latency of one short window.

## Contents
 
- [Setup](#setup)
- [Running](#running)
  - [Training](#training)
  - [Settings files](#settings-files)
  - [Evaluation](#evaluation)
  - [Making training data](#making-training-data)
  - [Tests](#tests)
- [Checkpoints and data](#checkpoints-and-data)
- [Layout](#layout)
- [Contributing](#contributing)
- [License](#license)

## Setup

Pick your system.

| System | Guide |
|-:|:-|
| **Linux** | [docs/setup-linux.md](docs/setup-linux.md) |
| **Windows** | [docs/setup-windows.md](docs/setup-windows.md) |

Where to download the datasets is in [docs/datasets.md](docs/datasets.md).

If you already have Python 3.11 or newer, and the dataset, the short version is:
 
```bash
git clone https://github.com/supersonic-lumina/ENM156.git
```
```bash
cd ENM156
```
```bash
python3 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements.txt
```
```bash
cp config.local.example.toml config.local.toml   # then set data.voicebank_dir in it
```
 
`config.local.toml` is yours and is git-ignored: put anything specific to your machine there,
such as a smaller `train.batch_size`/`train.max_train`, or `data.audio_device = "cpu"` if the
training audio doesn't fit in your VRAM.

## Running

All commands are run from the repository root. `.venv/bin/python` is the Python of the environment you created during setup.

### Training

```bash
.venv/bin/python train.py                                   # settings from config.toml
.venv/bin/python train.py --set stft.long_factor=0 --set train.output_dir=runs/short_only
.venv/bin/python train.py --set model.type=tcn --set model.output=mask --set train.crop_frames=200
.venv/bin/python train_pesq.py --set train.output_dir=runs/pesq
.venv/bin/python train_pesq.py --set pesq.init_from=runs/default/model.pt --set train.lr=1e-4 --set train.output_dir=runs/default_pesq
.venv/bin/python train_pesq.py --set model.type=snn --set pesq.init_from= --set train.output_dir=runs/snn   # spiking network
.venv/bin/python train_filterbank.py                        # learned encoder/decoder (step 1)
```

### Settings files

Instead of many `--set` flags, settings can come from a TOML file with `--config`; `--config`
and `--set` apply in the order given, so the later one wins:

```bash
.venv/bin/python train_pesq.py --config experiments/snn-1x256.toml --set train.output_dir=runs/snn_1x256
.venv/bin/python train_pesq.py --config runs/snn_1x256/config.toml --set train.epochs=5 --set train.output_dir=runs/snn_short
```

Every training run writes the settings it used to `config.toml` in its output folder, and its
`log.txt` starts with the exact command. To repeat or share a run, pass its `config.toml` with
`--config`. Settings that belong to one machine (dataset folders, device) and the output
folder are left out of it as comments, so someone else's file keeps your own paths.
`experiments/` holds the settings of runs worth repeating.

### Evaluation

```bash
.venv/bin/python evaluate.py --model runs/default/model.pt  # PESQ / STOI on test clips
.venv/bin/python compare_oracles.py --filterbank runs/filterbank/filterbank.pt
```

With `--model`, `evaluate.py` also estimates what the denoiser would cost to
run live, from operation counts per frame (`denoiser/snn/stats.py`), for every
model type:

```
  cost, network:         6.7 M MACs/s + 0.17 M neuron updates/s + 0.57 M synops/s (spikes per neuron per step: ...); about 32 uW
  cost, STFT & features: 3.85 M ops/s (analysis 3.10, features 0.48, output 0.03, synthesis 0.25); about 18 uW
  cost, total:           about 50 uW (...)
```

`network` is the model alone, the number older logs show. `STFT & features`
is everything around it: the STFT (window and real FFT, also of the long
window), the model's input features, applying its output to the noisy
spectrum, and the inverse STFT with overlap-add; a real FFT of N points is
counted as N·log2(N) multiply-adds. The pJ per operation type (MAC, synaptic
operation, neuron update, FFT multiply-add, memory bit) are set in `[energy]`
in config.toml. Their defaults are the usual 45 nm figures (Horowitz 2014):
a convention for comparing networks with each other, not a measurement.
Other code can add its own cost terms with
`denoiser.snn.stats.register_cost_term(name, fn)`; they are printed and
added to the total the same way.

### Making training data

```bash
.venv/bin/python -m denoiser.make_mixtures voicebank /path/to/mixed_8s   # 8 s train/val mixtures
```

### Tests

The tests don't need the dataset. Run them all with:

```bash
for t in tests/test_*.py; do echo "== $t"; .venv/bin/python "$t" || break; done
```

Or one at a time:
 
```bash
.venv/bin/python tests/test_stream.py         # live output == offline output
.venv/bin/python tests/test_crops.py          # training crops == whole clips
.venv/bin/python tests/test_filterbank.py     # learned filterbank framing
.venv/bin/python tests/test_snn_neurons.py    # spiking neurons on hand-made inputs
.venv/bin/python tests/test_snn_stream.py     # spiking network: live == offline == frame by frame
.venv/bin/python tests/test_snn_stats.py      # operation counts against counts by hand
.venv/bin/python tests/test_cost.py           # whole-pipeline cost, [energy], registered cost terms
.venv/bin/python tests/test_config.py         # --config / --set order, a run's config.toml
```

## Checkpoints and data

Each checkpoint stores the `stft` and `model` settings it was trained with,
so `evaluate.py` and the live denoiser rebuild it without extra flags.
Besides the final `model.pt`, `train.save_checkpoints` can keep every
epoch's model, the best one so far, or the 3 best (see config.toml).
Everything a training run prints is also saved to `log.txt` next to them.

Training, validation and testing read `train-*`, `val-*` and `test-*.parquet`
from `data.train_dir`, `data.val_dir` and `data.test_dir`; any left at `""`
use VoiceBank-DEMAND (`data.voicebank_dir`), which validates on its test set.
`denoiser/make_mixtures.py` makes 8 s train and val mixtures to point them at:
VoiceBank's speech with its own noise (`voicebank`), or VCTK speech with
DEMAND noise (`vctk_demand`). Both keep VoiceBank's test speakers and noise
types out, and validate on VCTK speakers that VoiceBank doesn't use (`[mix]`).

## Layout

| | |
|---:|:---|
| `config.toml` | shared default settings, documented inline |
| `requirements.txt` | Python dependencies |
| `docs/` | other documentation |
| `experiments/` | settings files of runs worth repeating, for `--config` |
| `denoiser/config.py` | settings loading: config.toml → config.local.toml → `--config` / `--set`; a run's config.toml |
| `denoiser/data.py` | reading clips from the parquet files, training crops |
| `denoiser/make_mixtures.py` | 8 s train/val mixtures from VoiceBank, VCTK and DEMAND |
| `denoiser/features.py` | causal / centred STFT, long-window pooling, compression |
| `denoiser/pipeline.py` | model inputs, targets, loss and output for each setting |
| `denoiser/nets.py` | the networks (MLP, TCN) |
| `denoiser/snn/neurons.py` | spiking neurons (leaky integrate-and-fire, leaky readout) |
| `denoiser/snn/nets.py` | the spiking networks; settings in `[snn]` |
| `denoiser/snn/stats.py` | operations per frame of the network and the STFT around it, and the energy estimate `evaluate.py` prints (all model types; pJ per operation in `[energy]`) |
| `denoiser/model.py` | building the configured network, checkpoints |
| `denoiser/filterbank.py` | learned encoder/decoder that can replace the STFT (not used by the denoiser yet) |
| `denoiser/stream.py` | live, chunk-by-chunk denoising |
| `denoiser/pesq_loss.py` | differentiable PESQ loss for `train_pesq.py` |
| `legacy/` | the original single-file experiments |

## Contributing

We work with issues, short-lived branches, and reviewed pull requests. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow

## License

Licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
