# GitHub Collaboration Notes

This repository is prepared for code collaboration, not as a full artifact dump.

## What is intentionally not tracked

- `models/ner_model/`: trained model weights are too large for normal GitHub.
- `rrf/`, `prescribe/rrf/`: raw RxNorm release dumps are huge.
- `.agent_runs/`, `.kaggle_download/`, `output_model*/`: local run artifacts.
- `input_turn*/`, `output_turn*.zip`: leaderboard/test inputs and submissions.

The committed `data/terminology/*.csv` files are enough for inference-time entity linking. Raw RRF files are only needed when rebuilding the RxNorm-derived CSVs.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

To run inference, place a model export in:

```text
models/ner_model/
  config.json
  model.pt
  tokenizer.json
  tokenizer_config.json
```

The model can be produced from Kaggle with:

```powershell
.\venv\Scripts\python.exe scripts\run_all.py all --aug-multiplier 1 --assertion-docs 30
```

The Kaggle notebook now exports the best holdout checkpoint, not simply the
last epoch. `run_all.py all` installs the downloaded export into
`models/ner_model/` automatically when the Kaggle run finishes. If you only
run the Kaggle notebook manually, unzip/copy the exported files into the same
folder.

## Inference

For a private/test input folder:

```powershell
.\venv\Scripts\python.exe scripts\run_all.py submit --input input_turn2 --pred output_model_turn2 --out output_turn2.zip
```

`run_all.py submit` uses the private-safe post-process defaults:

- `--drop-short-noise`
- `--add-terminology-entities`
- `--add-public-phrase-entities`

Do not enable `--drop-tuned-noise` for source submissions unless the goal is only to tune against a known leaderboard batch.

## Source Bundle

When the organizers need a rerunnable source artifact, build it from the local machine that has `models/ner_model/`:

```powershell
.\venv\Scripts\python.exe scripts\package_source.py --dry-run
.\venv\Scripts\python.exe scripts\package_source.py --out source_bundle.zip
```
