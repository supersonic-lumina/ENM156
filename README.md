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

### Evaluation

```bash
.venv/bin/python evaluate.py --model runs/default/model.pt  # PESQ / STOI on test clips
.venv/bin/python compare_oracles.py --filterbank runs/filterbank/filterbank.pt
```

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
| `denoiser/config.py` | settings loading: config.toml → config.local.toml → `--set` |
| `denoiser/data.py` | reading clips from the parquet files, training crops |
| `denoiser/make_mixtures.py` | 8 s train/val mixtures from VoiceBank, VCTK and DEMAND |
| `denoiser/features.py` | causal / centred STFT, long-window pooling, compression |
| `denoiser/pipeline.py` | model inputs, targets, loss and output for each setting |
| `denoiser/nets.py` | the networks (MLP, TCN) |
| `denoiser/snn/neurons.py` | spiking neurons (leaky integrate-and-fire, leaky readout) |
| `denoiser/snn/nets.py` | the spiking networks; settings in `[snn]` |
| `denoiser/snn/stats.py` | operations per frame and the energy estimate `evaluate.py` prints |
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
