# Datasets

Where every dataset this repo uses comes from, and how to get it into
`data/`, where the default `config.toml` paths already point. `data/` is
git-ignored. If you keep the datasets somewhere else, set the same keys in
`config.local.toml` instead.

| Dataset | Needed for | Source | Download size | Config key |
|---|---|---|---|---|
| VoiceBank-DEMAND-16k (parquet) | everything: training, validation and the test set | [Hugging Face](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) | ~2.2 GB | `data.voicebank_dir` |
| VoiceBank-DEMAND log files | `make_mixtures.py` (both recipes) | [Edinburgh DataShare 10283/2791](https://datashare.ed.ac.uk/handle/10283/2791) | < 1 MB | `mix.voicebank_logs` |
| VCTK 0.92 | `make_mixtures.py vctk_demand` and the validation speakers of both recipes | [Edinburgh DataShare 10283/3443](https://datashare.ed.ac.uk/handle/10283/3443) | ~11 GB zip, ~12 GB unpacked | `mix.vctk_dir` |
| DEMAND, 16 kHz | `make_mixtures.py vctk_demand` | [Zenodo 1227121](https://zenodo.org/records/1227121) | ~1.7 GB zips, ~4 GB unpacked | `mix.demand_dir` |
| DNS Challenge 4 (fullband) | training mixtures like the Intel N-DNS Challenge's; its dev test set (`denoiser.dnsmos --dns4-dev`) | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) | 582 GB archives, ~300 GB at 16 kHz | `data.dns4_dir` |
| DNSMOS P.835 model | `denoiser.dnsmos`: quality scores without a clean reference | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS) | 1.2 MB | `data.dnsmos_model` |

If you only want to train and evaluate on VoiceBank-DEMAND as it is, the
first one is all you need. The other three are only for making the 8 s
mixtures with `denoiser/make_mixtures.py`.

The finished layout:

```
data/
├── VoiceBank-DEMAND-16k/
│   ├── data/          train-0000N-of-00005.parquet, test-00000-of-00001.parquet
│   └── logfiles/      log_trainset_28spk.txt, log_testset.txt, ...
├── VCTK-Corpus-0.92/
│   ├── speaker-info.txt
│   └── wav48_silence_trimmed/   p225/ ... (one folder of .flac files per speaker)
└── DEMAND-16k/
    ├── DKITCHEN/ch01.wav ... ch16.wav
    └── ...        (one folder per environment)
```

All commands below are run from the repo root.

## VoiceBank-DEMAND-16k

The Valentini-Botinhao et al. VoiceBank-DEMAND set (28 training speakers,
11572 clips; 2 test speakers, 824 clips), resampled to 16 kHz and packed
as parquet files with the columns `id`, `clean` and `noisy` (WAV bytes).
Only the `data/*.parquet` files are needed:

```bash
.venv/bin/pip install huggingface_hub
```

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('JacobLinCool/VoiceBank-DEMAND-16k', repo_type='dataset',
                  allow_patterns=['data/*.parquet'], local_dir='data/VoiceBank-DEMAND-16k')
"
```

This leaves the shards in `data/VoiceBank-DEMAND-16k/data/`, which is the
default `data.voicebank_dir`. (`huggingface_hub` also leaves a small
`.cache/` folder next to them; it can be deleted.)

## VoiceBank-DEMAND log files

`make_mixtures.py` reads which speakers are in VoiceBank's training and test
sets, and which noise type each training clip has, from the log files of
the original 48 kHz release on Edinburgh DataShare. Only `logfiles.zip` is
needed, not the audio:

```bash
mkdir -p data/VoiceBank-DEMAND-16k/logfiles
```

```bash
curl -L -o data/VoiceBank-DEMAND-16k/logfiles/logfiles.zip https://datashare.ed.ac.uk/bitstream/handle/10283/2791/logfiles.zip
```

```bash
unzip -j data/VoiceBank-DEMAND-16k/logfiles/logfiles.zip -x '__MACOSX/*' -d data/VoiceBank-DEMAND-16k/logfiles
```

## VCTK 0.92

The CSTR VCTK corpus, version 0.92: 110 speakers at 48 kHz, two microphones
per utterance (`mix.vctk_mic`). `make_mixtures.py` resamples it to 16 kHz
itself. It is one ~11 GB zip, so give it time:

```bash
mkdir -p data/VCTK-Corpus-0.92
```

```bash
curl -L -C - -o data/VCTK-Corpus-0.92/VCTK-Corpus-0.92.zip https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
```

```bash
unzip -d data/VCTK-Corpus-0.92 data/VCTK-Corpus-0.92/VCTK-Corpus-0.92.zip
```

(`-C -` resumes the download if it was cut off.) Check that
`data/VCTK-Corpus-0.92/speaker-info.txt` and
`data/VCTK-Corpus-0.92/wav48_silence_trimmed/` exist; the zip can be
deleted afterwards.

## DEMAND (16 kHz)

The DEMAND multichannel noise recordings: one zip per environment, each
with 16 channels of about 5 minutes; `make_mixtures.py` uses `ch01.wav`.
Get the `*_16k.zip` files (Zenodo also has `*_48k.zip`, which are not
needed). SCAFE has no 16 kHz zip, which doesn't matter: it is one of
VoiceBank's test noises, so `make_mixtures.py` leaves it out anyway, as it
does TBUS, DLIVING, OOFFICE and SPSQUARE.

```bash
mkdir -p data/DEMAND-16k
```

```bash
for env in DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO; do curl -L -o data/DEMAND-16k/${env}_16k.zip "https://zenodo.org/records/1227121/files/${env}_16k.zip?download=1" && unzip -o -d data/DEMAND-16k data/DEMAND-16k/${env}_16k.zip; done
```

Each zip unpacks into its own folder (`data/DEMAND-16k/DKITCHEN/ch01.wav`,
...). The zips can be deleted afterwards.

## Making the mixtures

With everything above in place:

```bash
.venv/bin/python -m denoiser.make_mixtures vctk_demand data/VCTK-DEMAND-16k-8s
```

```bash
.venv/bin/python -m denoiser.make_mixtures voicebank data/VoiceBank-DEMAND-16k/mixed_8s
```

and point `data.train_dir` / `data.val_dir` at the output folder (in
`config.local.toml` or with `--set`). The `vctk_demand` mixtures take about
9 GB, the `voicebank` ones (3 passes) about 7 GB. The test set always stays
VoiceBank-DEMAND's own.

## DNSMOS P.835

The no-reference quality model (`sig_bak_ovr.onnx`) that `denoiser/dnsmos.py`
runs:

```bash
mkdir -p data/dnsmos
```

```bash
curl -L -o data/dnsmos/sig_bak_ovr.onnx https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
```

`python -m denoiser.dnsmos runs/x/model.pt` scores the clean, noisy and
denoised test clips; with `--dns4-dev` it scores the noisy and denoised clips
of the DNS-4 dev test set (below), which has no clean references, and how
often the model makes a clip worse. DNSMOS is for comparing outputs, never a
training loss: a network learns to fool it.

## DNS Challenge 4

The ICASSP 2022 DNS Challenge's fullband data, which the Intel N-DNS
Challenge also trains on: 127 `.tar.bz2` archives (582 GB) of clean speech in
several languages, noise, room impulse responses and a dev test set of noisy
clips without clean references, all at 48 kHz. `denoiser/dns4.py` downloads
them and unpacks them at 16 kHz, about 300 GB in all:

| Folder in `datasets_fullband/` | Files | 16 kHz size |
|---|---|---|
| `clean_fullband/` (read English, German, French, Spanish, Italian, Russian, emotional and singing speech, VCTK) | 1.12 M | 277 GB |
| `noise_fullband/` (AudioSet, Freesound and DEMAND clips) | 64 k | 20 GB |
| `impulse_responses/` | 60 k | 2 GB |
| `dev_testset/` | 921 | 0.6 GB |

The archives don't need to stay on a fast disk once unpacked; a spare HDD is
fine for both. The server gives each connection only about 1.3 MB/s, so the
download runs 20 at a time (`--parallel`): about 9 hours at 18 MB/s.
Unpacking writes over a million small files and is limited by the disk: 4
at a time (`--parallel`) took about 6 hours on a 7200 rpm HDD. The two can
run at once (`--watch` keeps unpacking archives as they arrive). Both resume
where they stopped when run again.

```bash
nohup .venv/bin/python -m denoiser.dns4 download /mnt/hdd/dns4/archives > /dev/null 2>&1 &
```

```bash
nohup .venv/bin/python -m denoiser.dns4 extract /mnt/hdd/dns4/archives --watch > /dev/null 2>&1 &
```

The second unpacks into `data.dns4_dir` (`--out` for another folder); set
`data.dns4_dir` in `config.local.toml` if that isn't `data/dns4`. Progress
is in `download.log` in the archive folder and `extract.log` in the output
folder; `python -m denoiser.dns4 stop <folder>` stops either.

### Training on DNS-4

The unpacked data is clean speech and noise, not pairs. `synthesize` mixes
them into train and val parquet files, as the N-DNS Challenge does: 30 s
clips of random clean files, noise at −5 to 20 dB SNR, the mixture at −35 to
−15 dBFS. About 1 % of the clean and noise files (chosen by a hash of their
path) are kept for validation only. See `python -m denoiser.dns4 --help` for
the details and settings.

```bash
nohup .venv/bin/python -m denoiser.dns4 synthesize data/dns4/mixtures_500h > /dev/null 2>&1 &
```

500 h (the N-DNS size, `--hours`) takes about 115 GB and roughly an hour
with 4 workers (`--parallel`). It is resumable, and the same `--seed` gives
the same mixtures on any machine. Then, in `config.local.toml`:

```toml
[data]
train_dir = "data/dns4/mixtures_500h"
val_dir = "data/dns4/mixtures_500h"
```

and train as usual. Two settings to change for DNS-4 runs:

- `train.val_every = 1`. The default 14 thins out VoiceBank's 824 short
  validation clips; DNS-4's validation set is 60 clips of 30 s (0.5 h,
  `--val-hours`).
- `data.audio_cache` on a fast disk with room for the decoded audio (as much
  as the parquet files, about 115 GB for 500 h). Without a cache
  `train_pesq.py` holds all the training audio in RAM.

Testing stays on VoiceBank-DEMAND's test set (`data.test_dir`).

## Licences and citations

- **VoiceBank-DEMAND**: C. Valentini-Botinhao, *Noisy speech database for
  training speech enhancement algorithms and TTS models*, University of
  Edinburgh, 2017. CC BY 4.0.
- **VCTK 0.92**: J. Yamagishi, C. Veaux, K. MacDonald, *CSTR VCTK Corpus:
  English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version
  0.92)*, University of Edinburgh, 2019. CC BY 4.0.
- **DEMAND**: J. Thiemann, N. Ito, E. Vincent, *DEMAND: a collection of
  multi-channel recordings of acoustic noise in diverse environments*,
  2013. CC BY 4.0.
- **DNS Challenge 4 and DNSMOS**: H. Dubey et al., *ICASSP 2022 Deep Noise
  Suppression Challenge*, ICASSP 2022; C. K. A. Reddy, V. Gopal, R. Cutler,
  *DNSMOS P.835: a non-intrusive perceptual objective speech quality metric
  to evaluate noise suppressors*, ICASSP 2022. The data is CC BY 4.0 (each
  part keeps its source's licence; see the challenge repository).

Check each source page for the current terms before redistributing anything.
