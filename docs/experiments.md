# Experiment ledger

Every prediction folder under `experiments/` and every zip under `submissions/` is recorded here
with the exact command that produced it and, once known, its real leaderboard score. Folder names
are labels, not documentation — before this file existed the repo had twelve `output_model_turn2_*`
folders and nobody could say which config produced `output_turn2_best.zip`.

Add a row **before** you submit, fill in the score **the same day** you get it.

## Scored

| Date | Score | text | J_assert | J_cand | What it was |
| --- | ---: | ---: | ---: | ---: | --- |
| 07-24 | **34.388** | 35.720 | 39.557 | 29.514 | `xlm-roberta-base` + submit recipe, curated linking only |
| 07-26 | **36.32** | 38.340 | 43.590 | 29.340 | `xlm-roberta-large` + Qwen2.5-7B distillation, built entirely inside a Kaggle kernel — **artifacts lost, not reproducible locally** |
| 07-31 | **33.679** | 35.944 | 38.084 | 28.676 | `archive/submissions/output_turn2_icd.zip` — base model + ICD fallback + dose/form RxNorm |
| 08-01 | **34.303** | 36.057 | 38.297 | **29.992** | `submissions/v5_recoded.zip` — `--no-icd-fallback` + 22 deterministic code fixes |
| 08-01 | 33.644 | 36.057 | **36.101** | 29.992 | `submissions/v7_assert_union.zip` — v5 + rule-based assertion post-processing |

### What the 08-01 result settled

Clean A/B against 33.679 (same code, same model, differing only in `--no-icd-fallback` and the 22
code fixes): `text +0.113`, `J_assertion +0.212`, **`J_candidates +1.316`**, `final +0.624`.

`J_candidates = 29.9917` is the **highest ever recorded in this repo** — above the turn-1 hand-tuned
29.98 and the turn-2 baseline 29.514. Both halves of the change are confirmed: reverting the ICD
fallback recovered the predicted ~0.84, and the deterministic code fixes added ~0.48 on top.

It still trails the 07-24 baseline (34.388) by **0.086**, entirely from `J_assertion` (38.297 vs
39.557). That is an entity-set difference, not a linking one: the current code emits 2898 entities
against that run's 2972 (241 lost, 167 gained). Note the 07-24 run *also* carried an ICD fallback —
it emitted `'đột biến'`→Q95, `'6PD'`→D55.0, `'Bại não'`→G80 — just at 3-character granularity.
0.086 is inside the noise of that churn and is **not worth chasing**; the 2.0-point gap to the 36.32
distilled model is where the value is.

### What the 07-31 result settled

`33.679` vs `34.388` is a clean A/B: the same `xlm-roberta-base` in `models/ner_model/`, differing
only in linking. Diffing the two prediction folders gives **78 entities added, 0 removed, 6 drug
codes changed** — every one of the 78 a `CHẨN_ĐOÁN` that survived `filter_noisy_entities` *only*
because the ICD fallback found it a code. Their quality is visibly poor (`'6PD'`→D55.0, a fragment
of "G6PD"; `'đột biến'`→Q95.9; `'nhiễm trùng'`→A31.9; `'bệnh lây truyền'`→A56.2).

If ~65% of those 78 are wrong, `J_candidates` moves 29.514 → ~28.7. Observed **28.676**. So the
entire drop is explained by the 78 rescued entities, and the 6 drug-code changes are neutral.
Conclusion: keep the dose/form work, revert the ICD fallback, and never let a linking fallback
decide whether an entity survives a noise filter.

The offline evidence had said the opposite (holdout `J_candidates` 0.5909 → 0.7424). That eval fed
the linker **ground-truth spans**, which measures the ceiling of linking, not the pipeline. Also
note `models/ner_model/config.json` carries `"train_holdout_overlap": true` and a holdout WER of
0.006 — the holdout is contaminated. **No offline number in this repo has predictive validity.**

### Rule-based assertion post-processing is falsified (08-01)

The cleanest isolation in the repo: `postprocess.py` only edits `assertions`, and `text_score`
(36.057) and `J_candidates` (29.9917) came back **identical to the digit**. The whole delta is
attributable: `J_assertion` **38.297 → 36.101 (−2.196)**, final −0.659.

Why it failed, and the number that should have stopped me:

| | isFamily rate | isHistorical rate |
| --- | ---: | ---: |
| turn-1 truth (hand-labelled) | **0.9%** (16/1689) | — |
| v5 (model, untouched) | **0.85%** (19/2243) | 16.3% |
| v6 (`--family-gate`) | 0.27% (6) | 18.5% |
| v7 (+ `--consistency union`) | 0.31% (7) | 25.4% |

**The model's assertion head is already calibrated** — its isFamily rate matches the only truth
rate we can verify, to within 0.05pp. `--family-gate` cut a correct rate by two thirds, and
`--consistency union` was built on my hypothesis that isHistorical was under-predicted (16.3% vs
turn-1's 28%). That hypothesis was wrong: turn-2 is patient Q&A prose, not discharge summaries —
the same measurement that showed only 31/100 turn-2 docs carry a section header should have
predicted this, and I pushed the lever anyway.

Adding a wrong assertion is **doubly costly** under Jaccard: the entity's correct `__EMPTY__` item
is removed *and* a wrong item is added. That is why 262 edits cost 2.2 points.

Conclusion: `J_assertion` has real headroom (38.3 now, 43.6 with the distilled model, 50.3 with
hand labels) but **rules are not the way to it — a better model is**. Distillation delivered +4.03
on assertions; every rule tried has been negative. Do not submit `v6_assert`; its `--family-gate`
alone breaks the one rate known to be right.

## The real bottleneck (measured 2026-07-31)

Inverting `J = k/(2−k)` where `k` = fraction of emitted codes that are correct:

| Run | WER | `J_cand` | implied code accuracy |
| --- | ---: | ---: | ---: |
| turn-1 hand-tuned (Run 8) | 0.517 | 0.2998 | **46.1%** |
| turn-2 baseline | 0.643 | 0.2951 | **45.6%** |
| turn-2 distilled | 0.617 | 0.2934 | **45.4%** |

Three configurations spanning WER 0.52–0.64 — including human hand-labelling — all land on 45–46%.
The cleanest A/B in the repo (34.388 → 36.32, same test set, same tables, better model) gives
`text +2.62`, `J_assertion +4.03`, `J_candidates` **−0.17**.

Meanwhile the curated table already covers **93.2%** of diagnosis mentions by *exact* match
(214/252 distinct texts, 744/798 mentions). So linking is not coverage-limited; it is
**correctness**-limited. `diagnoses.csv` is mined from `output/`, the turn-1 submission that scored
`J_candidates` 29.98 — the table memorises our own wrong answers, and every variant reproduces them.

**Unit of work: ~350 distinct strings, not 15k concepts and not a new inference-path model.**
Procedure in [linking_recode.md](linking_recode.md).

### Stage 0 applied — deterministic code audit (no GPU)

`scripts/recode_terminology.py --audit` found two classes of guaranteed-zero code in
`diagnoses.csv`, 22 rows / **37 mentions** (5.0% of covered diagnosis mentions):

- **7 rows**: 3-char category where the catalog has subcodes — `I48`→`I48.9`, `E14`→`E14.9`,
  `R65`→`R65.9`, `N04`→`N04.9`, `K25`→`K25.9`. The other 19 3-char codes (`I10`, `N19`, `J91`,
  `J47`) are *terminal* in the BYT catalog and were correctly left alone.
- **15 rows**: code absent from the BYT catalog. Some are ICD-10-**CM** (`S06.4X9A`, `G31.84`,
  `L89.94`, `I73.89`); the rest are valid WHO ICD-10 that BYT omits (`I31.4`, `E87.6`, `I49.1`,
  `K58.9`, `N40.0`, `C64.9`). `nearest_catalog_code()` maps all 15 onto a catalog ancestor.

Applied and verified: `experiments/v5_recoded` has **2898 entities, identical to v1** (entity set
invariant held), 37 mentions changed code, 0 schema errors. Backups at `data/terminology/*.csv.bak`.

**Follow-up 08-01 — the audit had a hole.** Re-running `--audit` on a clean table correctly reported
zero (it is idempotent), but one bad code had never been reported at all: `J91`
("tràn dịch màng phổi", **6 mentions**). The control flow did `continue` after the 3-character
branch, so a 3-char code that was *neither* promotable *nor* present in the catalog was skipped
silently. BYT omits J91 entirely — it is a WHO dagger/asterisk code — and the answer is **J90**
("Tràn dịch màng phổi, không phân loại mục khác"). `nearest_catalog_code()` cannot help here because
nothing shorter than 3 characters is a code, so a catalog-title match now runs as a last resort and
is written with `needs_review=1`. Applied as `experiments/v8_recoded_j90`: 2898 entities, 6 codes
changed, 0 errors. Two other bugs fixed in the same pass: re-applying an already-applied proposal
overwrote `.csv.bak` with the fixed table (now only backs up when something changes), and the README
used a `<mới>` placeholder that reads as a real path.

## Pending

| Folder / zip | Command | Rationale | Score |
| --- | --- | --- | ---: |
| `v5_recoded` | `recode_terminology.py --proposed recode_autofix.csv` then rerun the v1 recipe | 22 deterministic code fixes, 37 mentions; entity set verified unchanged at 2898 | |
| `v1_revert_icd` | `run_pipeline.py --input input_turn2 --pred experiments/v1_revert_icd --no-icd-fallback --drop-short-noise --add-terminology-entities --add-public-phrase-entities` | Revert the falsified fallback, keep dose/form. Expect ≈34.4 | |
| `v2_assert` | `postprocess.py --pred experiments/v1_revert_icd --out experiments/v2_assert --sections --negex --family-gate` | Attribute-only assertion levers: 53 section marks, 6 negex, 18 isFamily removed | |
| `v3_assert_union` | `… --sections --negex --family-gate --consistency union` | Also propagates assertions across repeated mentions: isHistorical 366 → 569. The model marks 16% of assertable entities isHistorical while turn-1 truth carries 28% | |
| `v4_hedge_all` | `… --sections --negex --family-gate --hedge-icd all` | Adds the `.9` sibling as a 2nd ICD candidate on 293 diagnoses. **Lowest confidence**: payoff needs p₂ > J/(1+J) = 0.223, estimated 0.11–0.20 | |

Suggested order: **v5 → v1 → v3 → v2.** v5 is v1 plus the deterministic code fixes, so it strictly
dominates v1 unless the 3-char promotion is wrong — and it answers the recoding hypothesis first,
which is the one that decides where the remaining effort goes. v1 next as the clean revert baseline.
v3 carries the largest assertion swing; v2 isolates section/negex if v3 disappoints. v4 last, and
only if a submission slot is going spare — its payoff needs p₂ > 0.223 and the estimate is 0.11–0.20.

## How much of ICD / RxNorm the pipeline actually uses (measured 2026-07-31)

On `experiments/v1_revert_icd` (the reverted turn-2 run):

| Resource | Rows available | Distinct codes emitted | Utilisation |
| --- | ---: | ---: | ---: |
| `icd10_vi.csv` (BYT official) | 15,144 | 161 | **1.06%** |
| `rxnorm_full.csv` | 517,991 | 72 | **0.014%** |
| `diagnoses.csv` (curated, mined from `output/`) | 321 | — | drives 94.1% of answers |
| `drugs.csv` (curated) | 159 | — | — |

Provenance of the 798 surviving `CHẨN_ĐOÁN`: **exact 751 (94.1%)**, containment 45 (5.6%),
difflib fuzzy 2 (0.3%). And the 86 rows in `diagnoses.csv` tagged `icd10_alias` are a hardcoded
dict in `build_terminology_index.py` — **not** the BYT catalog. With `--no-icd-fallback` on,
`icd10_vi.csv` is unused entirely.

Combined with `filter_noisy_entities` deleting any `CHẨN_ĐOÁN`/`THUỐC` without a code, this means
**a diagnosis can only reach the submission if its exact normalized text is one of ~321 turn-1
strings.** Dropping `--add-terminology-entities` moves diagnoses 798 → 712, so the table generates
86 spans (11%) and the model supplies the rest — but the survivors are still gated on that table.

### Three ways to exploit the vocabularies, all measured and rejected

Reproduce with `python scripts/expand_terminology.py --report`.

1. **ICD whole-title exact rows.** 11,033 titles pass chapter/length/4-char gates; exactly **35**
   occur verbatim across all 200 documents. Official catalog language is not clinical language —
   "bệnh tả do vi khuẩn vibrio cholerae 01, típ sinh học cholerae" is a BYT title, not something a
   doctor writes. Near-zero recall.
2. **ICD token-subset matching.** Already tried and scored: 34.388 → **33.679**. This is the tier
   that produces `'nhiễm trùng'`→A31.9 over 15k titles.
3. **RxNorm name expansion.** 10,400 clean names, 98 occur in corpus, 60 already in `drugs.csv`.
   Of the 38 remaining, 9 brand names are **already emitted correctly** by the existing RxNorm
   fallback (medrol, mucinex, toprol, zestril, …), `vitamin b12`/`vitamin k` are correctly typed
   `TÊN_XÉT_NGHIỆM`, and much of the rest are lab analytes (creatinine, glucose, cholesterol,
   lactate, magnesium, fibrinogen). Putting analytes in `drugs.csv` would type them `THUỐC` — and
   a wrong type is **counted twice with zero on all three metrics**. Real upside: ~10 mentions
   across 200 documents.

**What would actually work** is a paraphrase layer between clinical phrasing and the 15k catalog:
LLM-generated Vietnamese synonyms/abbreviations per title (~15–20 each → ~250k pairs), trained into
a bi-encoder. That needs no task labels — the vocabulary is its own supervision. It is a GPU job,
not a lookup-table job, and it is the only identified path to moving `J_candidates` off ~29.

## Rules

- One variable per submission where the budget allows. All four variants above share the same
  entity set (2898), so any score difference between them is purely assertions/candidates.
- **Never change the entity set and an attribute in the same submission.** That is what made the
  07-31 result take three weeks to interpret.
- Attribute-only changes (assertions, candidates on existing entities) are the low-risk lane.
  Entity-set changes (spans, thresholds, noise filters) move the denominator of all three metrics
  at once and have gone negative every time they were tried: 31.89, 33.679.
