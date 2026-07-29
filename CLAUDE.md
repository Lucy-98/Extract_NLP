# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **ĐỌC TRƯỚC TIÊN:** [docs/problem_statement.md](docs/problem_statement.md) là **đề bài gốc của BTC**
> — nguồn sự thật cho luật thi, format nộp, metric, và các ràng buộc (đặc biệt: **model ≤ 9B params,
> không dùng API ngoài lúc inference**). Bất kỳ mô tả nào bên dưới hoặc trong README mà mâu thuẫn với
> file đó thì file đó đúng. Đọc nó trước khi bắt đầu bất kỳ việc gì trong repo này.

## Project

ViettelRace AI Race 2026, Đề 2: extract and normalize medical concepts from free-form Vietnamese
clinical text (doctor's notes, discharge summaries, lab results, EHR excerpts). For each
`input/{id}.txt`, produce `output/{id}.json` — a list of entities, each with a character span, a
type, contextual assertions, and (for diagnoses/drugs) standard-code candidates.

- **Types**: `CHẨN_ĐOÁN` (diagnosis), `TRIỆU_CHỨNG` (symptom), `THUỐC` (drug),
  `TÊN_XÉT_NGHIỆM` (lab test name), `KẾT_QUẢ_XÉT_NGHIỆM` (lab result value+unit).
- **`assertions`** (list, only on `CHẨN_ĐOÁN`/`TRIỆU_CHỨNG`/`THUỐC`): `isNegated`, `isHistorical`,
  `isFamily`. Empty list if none apply.
- **`candidates`** (list of code strings, only on `CHẨN_ĐOÁN`/`THUỐC`): ICD-10 codes for diagnoses,
  RxNorm codes for drugs. RxNorm is dose+form specific (e.g. clonazepam 0.5mg vs 1.5mg are different
  codes) — never map by ingredient name alone.
- **Score**: `0.3 · text_score(WER over entity text) + 0.3 · J_assertion + 0.4 · J_candidates`.
  `scripts/check_submission.py --truth <dir>` computes a type-aware local proxy: it prefixes text
  tokens with entity type so correct text + wrong type gets zero text credit, and assertion/candidate
  Jaccard keeps duplicate mentions distinct by occurrence index. Treat it as a closer simulator, not
  a replacement for the hidden scorer.

**Critical constraint driving all architecture decisions here**: the organizers rebuild the top ~15
teams' source code and rerun it on a private test set. Anything that doesn't perform real inference
at run time (regex/wordlist keyed off specific file contents) won't generalize and risks
disqualification. This is why the repo has a `legacy/` (rule-based, deprecated) and a current model
pipeline, and why `output/` is treated as *training data*, not as the artifact to keep hand-tuning.

## Status (2026-07-29)

Best turn-2 score whose checkpoint still exists: **35.7087** (`xlm-roberta-large`, in
`models/ner_model/`). 36.3160 scored higher but its export was never downloaded and Kaggle no longer
serves it — see `docs/score_history.md` for the full run table and `worklog.md` for why each
configuration was tried. Two things that table settles, so they are not re-litigated:

- **Qwen pseudo-label dose is an inverted U**, not a monotone gain (0 → 34.71, ~800 → 36.32,
  1208 → 35.19, 2213 → 31.44). More or better teacher labels is not the lever it looks like.
- **The turn-1 holdout is not a proxy for turn-2.** Runs have set holdout records while scoring
  below runs with worse holdout. Turn-2 is patient-education *articles*; turn-1 is doctor's notes.

Nothing in the 34–36 band has been run twice, so differences that size are not separable from seed
noise. Do not spend GPU chasing them.

## Pipeline

For private-test/submission recreation, use the offline path only:

```bash
python scripts/run_all.py submit --input input_turn2 --pred output_model_turn2 --out output_turn2.zip
```

This does not need `output/` public labels and does not train. Use the training path below only
when the public-label `output/` folder is available and you intentionally want to rebuild the model:

```bash
python scripts/build_rxnorm_rrf_index.py   # rrf/ + prescribe/rrf/ (local, gitignored) -> data/terminology/rxnorm_full.csv,
                                            # rxnorm_drug_names.csv -- only needs rerunning if the RRF dump changes,
                                            # not part of every iteration (see Architecture below)
python scripts/prepare_ner_dataset.py      # output/ (hand-curated) -> data/ner_dataset/{train,holdout,all}.jsonl
python scripts/build_terminology_index.py  # output/ + legacy dicts -> data/terminology/{drugs,diagnoses}.csv
python scripts/augment_ner_dataset.py      # train.jsonl + terminology -> train_augmented.jsonl
# train on Kaggle/Colab GPU: notebooks/train_ner_assertion_model.ipynb (see "Training on Kaggle" below)
# unzip the downloaded ner_model_export.zip into models/ner_model/
python scripts/run_pipeline.py             # model + terminology index -> output_model/*.json
python scripts/check_submission.py --pred output_model --input input --truth output
python scripts/package_submission.py       # output_model/*.json -> output.zip with output/{id}.json entries
python scripts/package_source.py --dry-run # verify source bundle contents before writing source_bundle.zip
```

The `run_all.py` stages are `prepare`, `train`, `infer`, `package`, `submit`, and `all`. `submit` is
the offline submission path above. `all` runs `prepare -> train -> infer`. `prepare` also copies `train_augmented.jsonl` into
`kaggle_upload/dataset/train.jsonl`, matching the filename the Kaggle notebook reads. `train`
versions that Kaggle dataset by default, pushes `kaggle_upload/kernel` with
`--accelerator NvidiaTeslaT4`, polls until it finishes, and unzips the result into
`models/ner_model/` — it needs the `kaggle` package + `~/.kaggle/kaggle.json`, and consumes your
weekly Kaggle GPU quota every time it runs, so don't loop it. `infer` runs `run_pipeline.py`, then
validates with `--truth output` only when that folder exists, and packages `output.zip`.

`output/` (current best hand-tuned submission, see `docs/score_history.md`) is never overwritten by
the pipeline — `run_pipeline.py` always writes to `output_model/` so the two can be diffed before
promoting one over the other.

`requirements.txt` pins the torch/transformers versions verified to work for local inference
(`run_pipeline.py`); the Kaggle kernel uses its own preinstalled versions instead (printed each run
in cell 1 of the notebook) — verified compatible as of 2026-07-20, re-verify with the `infer` stage
above if you deliberately upgrade either side.

Every script under `scripts/` except `run_pipeline.py` is Python-stdlib-only by design (no
torch/transformers needed) so they run anywhere, including machines used only for local iteration.
This includes `run_pipeline.py`'s RxNorm fallback: it reads the *committed derived* CSVs under
`data/terminology/`, never the raw `rrf/`/`prescribe/` dumps, so the private-rerun environment never
needs those ~1.8GB of gitignored files present.
All of them do `sys.stdout.reconfigure(encoding="utf-8")` — required on Windows, where the default
console codepage (cp932/cp1252) cannot print Vietnamese text and will crash otherwise.

## Architecture

- **`legacy/scripts/`** — the original regex/wordlist extractor (`generate_outputs.py` and friends).
  Kept for audit only. Its diagnosis/symptom lists are phrases copied verbatim from the 100 public
  input files, so it cannot generalize to unseen text; its assertion heuristic is hard-disabled.
  **Never use this to produce a submission.**
- **`scripts/prepare_ner_dataset.py`** — turns the hand-curated `output/*.json` (refined over many
  scored leaderboard rounds, see `docs/score_history.md`) into weak-supervision training data for a
  small token-classification model, instead of treating it as a fixed answer key. Fixed 85/15
  train/holdout split, seed 13. `--folds N` additionally writes N disjoint `fold{k}_{train,
  holdout}.jsonl` (purely additive, never touches the default `train.jsonl`/`holdout.jsonl`) — a way
  to spot-check whether the default split's holdout score is a fluke of that one partition, by
  training on a fold or two, without committing to full N-way cross-validation (each fold still
  costs a full Kaggle run to actually evaluate).
- **`scripts/build_rxnorm_rrf_index.py`** — derives two committed CSVs from the official RxNorm RRF
  release (unzipped locally under `rrf/` + `prescribe/rrf/`, gitignored, ~1.8GB combined — re-download
  from NLM instead of committing): `data/terminology/rxnorm_full.csv` (~512k `text -> RXCUI` rows,
  built from `RXNCONSO.RRF` + `RXNATOMARCHIVE.RRF`, so it covers Remapped/retired RxNorm concepts —
  see `fetch_rxnorm.py`'s docstring for why that matters and the confirmed 360047 worked example) and
  `data/terminology/rxnorm_drug_names.csv` (~11.2k clean, currently-prescribable ingredient/brand
  names, from the separate smaller "Prescribable Content" RRF release, for `augment_ner_dataset.py`'s
  synthetic pool). `RxNormOfflineIndex` (exact match, then first-token + dose/form-token-overlap
  fallback) is the class `run_pipeline.py` and `augment_ner_dataset.py` both import. Only needs
  rerunning when the RRF dump changes — the derived CSVs are what everything else reads.
- **`scripts/augment_ner_dataset.py`** — synthetic data generation required by the task statement
  ("dùng giải pháp ngoài lời giải chính để tạo thêm dữ liệu"), stdlib-only. Three techniques: (1)
  entity-text substitution within real sentences, shifting downstream offsets, to teach the model
  the BIO *pattern* rather than the ~100 strings it has seen — for `THUỐC` this now draws from
  ~11.2k real RxNorm ingredient/brand names (`rxnorm_drug_names.csv`), not just the corpus's own ~109,
  since generalizing to drug names the private test set introduces is exactly the gap this exists to
  close; (2) templated negation/historical/family clauses around currently-unmarked entity mentions,
  concatenated into multi-entity synthetic docs. Never touches `holdout.jsonl` — synthetic rows only
  ever land in `train_augmented.jsonl`.
- **`scripts/build_terminology_index.py`** — entity linking (ICD/RxNorm) is deliberately a separate
  lookup problem from NER, not something the model predicts. Builds `data/terminology/{drugs,
  diagnoses}.csv` (`text,candidate,source` columns) by mining `(text, type) -> candidates` pairs
  already present in curated `output/`, falling back to small hardcoded legacy dicts. `conflicts.txt`
  lists texts that map to different codes depending on context (e.g. "loét") — these need model
  context, not a lookup table, to resolve. `TerminologyMatcher` (exact-normalized-text + difflib
  fuzzy fallback) is the class both `run_pipeline.py` and `augment_ner_dataset.py` import.
- **`notebooks/train_ner_assertion_model.ipynb`** (mirrored in `kaggle_upload/kernel/`, must be kept
  in sync manually) — fine-tunes `xlm-roberta-base` with two heads on one shared encoder: BIO tagging
  (11 labels: O + B-/I- x 5 types) and per-token multi-label assertion classification (loss only on
  tokens belonging to CHẨN_ĐOÁN/TRIỆU_CHỨNG/THUỐC). xlm-roberta was chosen specifically because its
  fast tokenizer gives `offset_mapping` directly on raw Vietnamese text — no word-segmentation step
  (as PhoBERT would need) between model output and the `position` field the schema requires.
- **`scripts/run_pipeline.py`** — loads the exported model from `models/ner_model/` (needs
  `model.pt` + `config.json` + tokenizer files), builds the XLM-R base architecture locally instead
  of calling Hugging Face Hub, runs inference over `input/*.txt`, decodes BIO spans + assertion
  probabilities (threshold 0.5), links `CHẨN_ĐOÁN`/`THUỐC` spans through
  `TerminologyMatcher`, falling back to `RxNormOfflineFallback` (offline RxNorm RRF index, see
  `build_rxnorm_rrf_index.py`) for `THUỐC` mentions `drugs.csv` doesn't cover, writes
  `output_model/*.json`. It probes the prediction folder before loading the model so OneDrive
  placeholder/reparse-point write failures fail fast. Post-processing flags, all measured rather
  than assumed (see `worklog.md` 2026-07-27 part 4 and 2026-07-29):
  - `--add-terminology-entities` injects lexicon terms found in the text. Worth **+8.9 to +12.1**
    at realistic recall (76.6% standalone precision on held-out turn-1 folds); only turns negative
    once the model is near-perfect. Keep it on.
  - `--add-public-phrase-entities` only *expands* a span that already contains a same-type model
    entity, so it cannot invent entities. Safe by construction.
  - `--propagate-repeats` copies each detected mention onto its other whole-word occurrences,
    because gold is occurrence-based (94.1% of occurrences of an annotated string carry a label on
    turn-1). Real effect measured on actual turn-2 predictions is **+5.5% entities**, not the +9
    *points* an earlier simulation claimed — that simulation deleted gold entities independently at
    random, which fabricates per-occurrence gaps. Real recall errors are correlated across
    occurrences: the model gets a known term nearly everywhere and misses an unknown one
    everywhere, and a wholly missed term has nothing to propagate from.
- **`scripts/package_submission.py`** — validates a prediction folder and writes the required
  `output.zip` with exactly `output/{1..100}.json` entries. Defaults to `output_model/`.
- **`scripts/package_source.py`** — packages code, notebooks, derived data, public input/output labels,
  and `models/ner_model` for the BTC source-code rerun. Raw RRF dumps, venvs, cached Kaggle outputs,
  `.agent_runs`, and generated `output_model/` are excluded.
- **`scripts/check_submission.py`** — shared validator for both pipelines: schema/span sanity checks
  (span text matches input slice, valid types/assertions, no duplicate/overlapping entities) plus,
  given `--truth`, a local re-implementation of the real scoring formula for fast iteration without
  touching the leaderboard.
- **`scripts/fetch_icd.py`** — pulls ICD-10 codes from the WHO ICD-API; `--crawl-all` built
  `data/terminology/icd10_full.csv` (11,243 codes, English titles only — WHO's ICD-10 free-text
  search doesn't work, only ICD-11's does, so this table needs a Vietnamese→English bridge before
  it's usable for entity linking; not yet built).
- **`scripts/fetch_rxnorm.py`** — **superseded 2026-07-21** by `build_rxnorm_rrf_index.py` now that
  the official RxNorm RRF release (needs a UMLS account) has actually been obtained; kept only as a
  manual one-off live-lookup tool, no longer part of the default pipeline. Originally used the free
  public RxNav REST API as a substitute for the RRF release. Its docstring documents the discovery
  that motivated the RRF integration: none of RxNav's search endpoints can return a concept whose
  RxNorm status is `"Remapped"` — confirmed against the task statement's own worked example
  (`Chlorpheniramine 0.4 MG/ML / .../ Oral Solution` → RxNorm 360047) by querying rxcui 360047's exact
  canonical name and getting zero results. `RXNATOMARCHIVE.RRF` (in the now-available full release)
  reaches it directly — see `build_rxnorm_rrf_index.py`'s `--verify` flag, which checks this exact
  example on every rebuild.

## Training on Kaggle

The kernel is pushed from `kaggle_upload/kernel/` (separate from `notebooks/` — keep both copies in
sync when editing). On Windows, the `kaggle` CLI crashes with a `cp932` codec error when its own
output contains Vietnamese text unless UTF-8 I/O is forced first:

```bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
python -m kaggle kernels push -p kaggle_upload/kernel --accelerator NvidiaTeslaT4
python -m kaggle kernels status lucylng/viettelrace-ner-assertion-train
python -m kaggle kernels output lucylng/viettelrace-ner-assertion-train -p <dest>
```

Always pass `--accelerator NvidiaTeslaT4` explicitly. Without it, Kaggle may assign an older P100
(compute capability sm_60), which the preinstalled torch wheel no longer has kernel images for —
this crashed a full training run partway through epoch 1 with `CUDA error: no kernel image is
available for execution on the device` (fixed 2026-07-20: the notebook now also runs a real
forward+backward probe on a dummy batch right after building the model, before the training loop
starts, and falls back to CPU if the probe fails — but requesting T4 up front avoids needing that
fallback at all).

As of 2026-07-20, training has succeeded end-to-end on T4 (20 epochs, holdout WER 0.41 /
J_assertion 0.29 on the pre-augmentation 85-file train set) and `scripts/run_pipeline.py` has
produced `output_model/*.json` from it for the first time. `output/` (score 41.591, Run 8, hand-
tuned) has not yet been replaced — compare `output_model/` against it via `check_submission.py
--truth output` before deciding whether to promote it.

Dated history of what changed and why (CUDA fix, augmentation, fetch_rxnorm's Remapped-concept
discovery, etc.) lives in `worklog.md`, not here or in README.md — check it for the most recent
status before assuming anything above is still current.

## Reference

- `worklog.md` — dated engineering changelog (what changed, why, and when).
- `docs/score_history.md` — score history and deltas across manual rounds (Run 1–8).
- `data/terminology/conflicts.txt` — diagnosis texts with context-dependent ICD codes.
- `README.md` — same pipeline, more narrative detail on the legacy→model migration rationale.
