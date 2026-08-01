# Recoding the linking tables

The offline procedure that rebuilds the `candidate` column of
`data/terminology/{diagnoses,drugs}.csv`. This is the highest-leverage change
available in the repo. Read [experiments.md](experiments.md) first.

## Why this and not a retrieval layer in the pipeline

Measured on `experiments/v1_revert_icd` (2026-07-31):

| | distinct texts | exact hit in curated table | mentions covered |
| --- | ---: | ---: | ---: |
| CHẨN_ĐOÁN | 252 | 214 (84.9%) | **744/798 = 93.2%** |
| THUỐC | 94 | 64 (68.1%) | 172/216 = 79.6% |

Coverage is 93.2%. But inverting `J = k/(2−k)` on the real leaderboard gives:

```
turn-1 hand-tuned   J_cand 0.2998  ->  ~46.1% of codes correct
turn-2 baseline     J_cand 0.2951  ->  ~45.6%
turn-2 distilled    J_cand 0.2934  ->  ~45.4%
```

Three configurations with WER from 0.52 to 0.64 — wildly different NER quality —
all land on 45–46% code accuracy. Confirmed by the cleanest A/B in the repo
(34.388 → 36.32, same test set, same tables, better model): `text +2.62`,
`J_assertion +4.03`, `J_candidates **−0.17**`.

So the pipeline already *finds* a candidate for 93% of mentions. It finds the
*wrong* one half the time. A bi-encoder in the inference path would improve
recall, which is not the problem. The fix belongs in the table.

The cause is circular: `diagnoses.csv` is mined from `output/`, which is our own
turn-1 submission — the one that scored `J_candidates` **29.98**. The table is a
memorised copy of our own wrong answers.

**Corollary that shapes the whole design:** the unit of work is ~350 distinct
strings, not 15,144 catalog concepts. At that scale a human can review the diff.

## Stage 0 — deterministic audit (no GPU, no model)

```bash
python scripts/extract_mentions.py                       # -> recode_worklist.csv
python scripts/recode_terminology.py --audit             # -> recode_autofix.csv
python scripts/recode_terminology.py --proposed data/terminology/recode_autofix.csv --dry-run
python scripts/recode_terminology.py --proposed data/terminology/recode_autofix.csv
```

Fixes two classes of guaranteed-zero code, 22 rows / 37 mentions (5.0% of covered
diagnosis mentions):

- **3-char categories that have subcodes.** `I48` → `I48.9`, `E14` → `E14.9`.
  Categories that are *terminal* in the BYT catalog (`I10`, `J47`, `N40`, `L97`,
  `C64`) are left alone — 19 of the 26 3-char codes are correct as they stand.
- **Codes absent from the BYT catalog.** 15 rows. Some are ICD-10-**CM**
  (`S06.4X9A`, `G31.84`, `L89.94`, `I73.89`); the rest are valid WHO ICD-10 that
  BYT omits (`I31.4`, `E87.6`, `I49.1`, `K58.9`, `N40.0`, `C64.9`). Both map onto
  the nearest catalog ancestor.

Expected: **+0.6 … +1.25 final.** Already applied; `.csv.bak` backups sit next to
each table.

## Stage 1 — paraphrase corpus (Kaggle GPU, ~2h)

For each BYT title outside chapters U/V/W/X/Y/Z (12,730 titles), have
Qwen2.5-7B-Instruct emit 15–20 clinical Vietnamese renderings. Prompt shape:

```
Tiêu đề danh mục ICD-10 của Bộ Y tế: "{name_vi}"  (mã {code})
Liệt kê 15-20 cách một bác sĩ Việt Nam thực sự VIẾT khái niệm này trong bệnh án.
Bao gồm: tên thông tục, viết tắt, dạng không dấu, thuật ngữ tiếng Anh xen kẽ,
lỗi chính tả thường gặp. Mỗi dòng một cách viết, không đánh số, không giải thích.
TUYỆT ĐỐI không đổi sang một bệnh khác.
```

Then filter mechanically — the model will drift:

- drop a variant sharing **no** content token with the title (catches
  "viêm phổi" → "viêm phế quản")
- drop variants longer than 1.5× the title or shorter than 3 characters
- deduplicate after normalisation

Expect ~200k surviving pairs. Write `data/terminology/synonyms/pairs.jsonl` as
`{"code":..., "title":..., "variant":...}`.

## Stage 2 — bi-encoder (Kaggle GPU, ~1h)

Fine-tune `xlm-roberta-base` (or `mDeBERTa-v3-base`) with InfoNCE:

```
L = -log  exp(sim(E(variant), E(title⁺))/τ)
         ─────────────────────────────────────────────
         Σ over title⁺ and K hard negatives, same form
τ = 0.05,  K = 16,  2-3 epochs,  batch 128
```

**Hard negatives must come from the same ICD chapter.** A negative from another
chapter is trivially separable and contributes no gradient; mine them with BM25
top-20 restricted to `code[0] == positive_code[0]`.

## Stage 3 — propose codes for the worklist

Encode all 15,144 titles once (15,144 × 768 × 4B = **46 MB**, so a plain numpy
dot product is fine — no FAISS needed). For each row of `recode_worklist.csv`:

1. Simplify the query with Qwen ("sốt cao liên tục 3 ngày" → "sốt"). This changes
   only the *query*; the `text` column and the emitted span stay untouched.
2. Retrieve top-5 titles.
3. Ask Qwen to pick one, **with the 5 titles and their 5 codes in the prompt** so
   it selects rather than generates.

Write `data/terminology/recode_proposed.csv`:
`text,type,proposed_candidate,proposed_title,score`.

## Stage 4 — apply, with the guards

```bash
python scripts/recode_terminology.py --proposed data/terminology/recode_proposed.csv --dry-run
```

`recode_terminology.py` enforces four invariants, each for a reason that cost
points before:

1. **`text` may not change.** `--add-terminology-entities` generates spans *from*
   that column; editing it moves the entity set — the lane that scored 31.89 and
   33.679.
2. **Every code must exist** in `icd10_vi.csv` / `rxnorm_full.csv`. This is what
   stops an LLM-proposed code from reaching a submission.
3. **Codes must sit at a level the catalog carries** (`.9` promotion, terminal
   categories accepted).
4. **No row may end up with no code** — `filter_noisy_entities` deletes such
   entities entirely, taking their text and assertions with them.

Then regenerate with the *same* recipe as the baseline and diff:

```bash
python scripts/run_pipeline.py --input input_turn2 --pred experiments/v6_recoded_bienc \
    --no-icd-fallback --drop-short-noise --add-terminology-entities --add-public-phrase-entities
python scripts/package_submission.py --pred experiments/v6_recoded_bienc --input input_turn2 \
    --out submissions/v6_recoded_bienc.zip
```

Entity count must stay **2898**. If it moved, invariant 1 leaked — stop.

## Expected gain

| code accuracy `k` | `J_cand` | Δ vs 29.5 | **Δfinal** |
| ---: | ---: | ---: | ---: |
| 45.6% (now) | 29.5 | — | — |
| 55% | 37.9 | +8.4 | **+3.4** |
| 65% | 48.1 | +18.6 | **+7.4** |
| 75% | 60.0 | +30.5 | **+12.2** |

Realistic target **k = 55–65% ⇒ +3.4 … +7.4 final**.

## Falsification

If the recoded table scores **below** 29.5, the hypothesis is wrong — the old
table's 46% was not beatable this way. Restore and stop:

```bash
git checkout data/terminology/diagnoses.csv data/terminology/drugs.csv
```

Do **not** re-tune the retrieval offline. Every offline eval in this repo has
failed to transfer: `models/ner_model/config.json` carries
`train_holdout_overlap: true` with a holdout WER of 0.006, and the ICD linking
eval that predicted `J_candidates` 0.59 → 0.74 delivered 29.51 → 28.68 in
reality.
