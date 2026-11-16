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

Check each source page for the current terms before redistributing anything.
