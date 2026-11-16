# Setup on Linux
 
> **On Windows?** Do the [Windows (WSL) setup](setup-windows.md) first. It gets you a Linux
> shell, and then you come back here and continue from step 1.
>
> Back to the [README](../README.md).
 
These steps use Ubuntu/Debian commands (`apt`). On other distributions, install the same
packages with your package manager.
 
## Contents
 
1. [Install prerequisites](#1-install-prerequisites)
2. [Get the code](#2-get-the-code)
3. [Create the environment and install dependencies](#3-create-the-environment-and-install-dependencies)
4. [Get the dataset](#4-get-the-dataset)
5. [Create your local config](#5-create-your-local-config)
6. [Check that everything works](#6-check-that-everything-works)
Also: [Troubleshooting](#troubleshooting) · [Next steps](#next-steps)
 
## 1. Install prerequisites
 
```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-dev build-essential
```
 
`build-essential` and `python3-dev` are needed because the `pesq` package compiles a small
C extension when you install it.
 
Check your Python version. **It must be 3.11 or newer**:
 
```bash
python3 --version
```
 
Ubuntu 24.04 ships Python 3.12, which is fine. Ubuntu 22.04 ships 3.10, which is too old: use a
newer Ubuntu release, or install a newer Python (for example with
[`uv`](https://docs.astral.sh/uv/) or the deadsnakes PPA).
 
## 2. Get the code
 
Cloning needs no login, because the repository is public:
 
```bash
git clone https://github.com/supersonic-lumina/ENM156.git
cd ENM156
```
 
To push branches and open pull requests you need to be signed in to GitHub. The easiest way is
the GitHub CLI:
 
```bash
sudo apt install -y gh
gh auth login
```
 
Recommended answers:
 
| Question | Answer |
|---|---|
| What account do you want to log into? | GitHub.com |
| Preferred protocol for Git operations? | HTTPS |
| Authenticate Git with your GitHub credentials? | Yes |
| How would you like to authenticate? | Login with a web browser |
 
You'll get a one-time code to enter at <https://github.com/login/device>.
 
Set the name and email that appear on your commits:
 
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
 
## 3. Create the environment and install dependencies
 
From the repository folder:
 
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
 
This downloads several GB (PyTorch with CUDA support is large), so the first install takes a
while. You don't need to "activate" the environment: the commands in this project call
`.venv/bin/python` directly.
 
**GPU:** pip's default PyTorch build for Linux already includes CUDA support. You only need a
working NVIDIA driver. If you need a specific CUDA version, install `torch` and `torchaudio`
from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run the `pip install`
command above.
 
## 4. Get the dataset
 
Training and evaluation use **VoiceBank-DEMAND-16k** as parquet files.
 
Where to download it, and the other datasets the mixtures need, is in [datasets.md](datasets.md).
 
The default location is `data/VoiceBank-DEMAND-16k/data` inside the repository (the path is
relative to `config.toml`). It should contain files named `train-*.parquet` and
`test-*.parquet`. If you keep the dataset elsewhere, you set the path in the next step.
 
## 5. Create your local config
 
```bash
cp config.local.example.toml config.local.toml
```
 
Open `config.local.toml` and set `data.voicebank_dir` to the folder from step 4. This file is
yours: it is git-ignored, so anything specific to your machine goes here, such as:
 
- a smaller `train.batch_size` or `train.max_train` if you run out of memory
- `data.audio_device = "cpu"` if the training audio doesn't fit in your GPU memory
- `data.device = "cpu"` to train without a GPU
The shared defaults live in `config.toml` and are documented inline. Don't put personal
settings there.
 
## 6. Check that everything works
 
Run the tests. They don't need the dataset:
 
```bash
for t in tests/test_*.py; do echo "== $t"; .venv/bin/python "$t" || break; done
```
 
Each script exits with an error if its check fails, and the loop stops at the first failure.
 
Check whether PyTorch sees your GPU (optional):
 
```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```
 
`True` means training will use the GPU. `False` means it will fall back to the CPU, which works
but is much slower.
 
## Troubleshooting
 
| Problem | Likely fix |
|---|---|
| `pip` fails while building `pesq` | Install the build tools: `sudo apt install build-essential python3-dev` |
| `SyntaxError` or `No module named tomllib` | Your Python is older than 3.11. See step 1. |
| `OSError: sndfile library not found` | `sudo apt install libsndfile1` |
| `torch.cuda.is_available()` prints `False` | Check the NVIDIA driver with `nvidia-smi`. On WSL, see the [Windows guide](setup-windows.md#5-gpu-support-optional). |
| Out of memory while loading data or training | In `config.local.toml`, lower `train.max_train` or `train.batch_size`, or set `data.audio_device = "cpu"` |
| `No train-*.parquet files in ...` | `data.voicebank_dir` points at the wrong folder. See steps 4 and 5. |
| `Hash Sum mismatch` during `apt install` or `apt update` | The download is being corrupted on the way, which is common on WSL. Switch to HTTPS with the command below the table, then run `sudo apt clean`, `sudo apt update`, and install again. |

Switching `apt` to HTTPS:

```bash
sudo sed -i.bak 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources
```
 
## Next steps
 
- Train and evaluate: [README → Running](../README.md#running)
- Contribute a change: [CONTRIBUTING.md](../CONTRIBUTING.md)
 
