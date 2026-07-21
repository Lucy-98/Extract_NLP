# Worklog

Dated engineering changelog for this repo. Newest entries at the top. For the manual-round scoring
history (Run 1–8, legacy hand-tuned `output/`), see `docs/score_history.md` — that file is
score-only and unaffected by this one.

## 2026-07-21 (part 4) — .git was empty; reinitialized

- `git status` started failing with "not a git repository": `.git/` had only an `info/` subdirectory,
  no `HEAD`/`objects`/`refs`/`config` -- confirms the OneDrive-desync suspicion flagged (but not
  acted on) in part 2 below was real, not just a hunch. User confirmed this working copy was never
  pushed to a remote, so no history was recoverable -- ran `git init` and made a fresh initial commit
  of the current tree instead of trying to reconstruct the old one.
- Added `venv/`, `.env`, `.kaggle_download/` to `.gitignore` before the first `git add` -- none of the
  three were covered by the pre-existing rules (which only handled `models/`, `output_model/`,
  `.agent_runs/`, `kaggle_upload/`, `/rrf/`, `/prescribe/`). Verified nothing matching those patterns,
  or any `.pt`/`.zip`, ended up staged; largest tracked file is `data/terminology/rxnorm_full.csv`
  (38MB, expected -- it's the ~512k-row derived RxNorm index, see `build_rxnorm_rrf_index.py`).

## 2026-07-21 (part 3) — merge_fragmented_entities: whitespace/conjunction extension tried, reverted

- Investigated why `output_model` scores far worse on the 15 holdout files (combined local metric
  ~0.573) than on the 85 train files (~0.988) that raised concern about promoting the trained model
  over hand-tuned `output/` (leaderboard 41.591). Root-caused several of the worst holdout files
  (6, 22, 32, 37, 60, 70, 71, 76, 87) by diffing entities against `output/` truth: most gaps are the
  model under-extending a single span (e.g. "viêm gan" vs truth "viêm gan virus") -- not fixable by
  postprocessing since there's no second fragment to merge with, only retraining helps there.
- Tried extending `merge_fragmented_entities()` (which already bridges mid-word BIO splits) to also
  bridge two adjacent same-type CANDIDATE_TYPES mentions separated by whitespace (fixes file 37's
  "Insulin" + "glargine" -> "Insulin glargine") or by a Vietnamese conjunction ("và"/"hoặc"/"hay")
  (fixes file 6's "Nghẽn tắc" + "hẹp động mạch cảnh" -> one CHẨN_ĐOÁN via "và").
- **Reverted the conjunction half**: "và" is the normal way this corpus enumerates two *distinct*
  mentions, not just a mid-phrase joiner -- confirmed by a real false merge it caused in file 32,
  fusing "guaifenesin" and "furosemide 40 mg" (two different real drugs) into one bogus entity. The
  `same_known_terms` guard didn't catch it because the model's own span for the second drug was
  already truncated (missing "đường uống"), so its terminology lookup came back empty and the guard
  never fired -- it only protects when *both* fragments independently resolve.
- **Reverted the whitespace half too**, after finding it has the same failure mode in miniature: it
  fused a spurious false-positive "ho" (CHẨN_ĐOÁN, empty candidates) into the real diagnosis mention
  sitting right after it in file 32, for the identical same_known_terms-can't-fire-on-an-empty-side
  reason. Net effect across the 15 holdout files was a wash (combined 0.5734 -> 0.5725, i.e. slightly
  negative) -- the file 37 gain was offset by the new file 32 regression. Given no clear win and a
  demonstrated new failure mode, kept `merge_fragmented_entities` at the original mid-word-only scope.
- Conclusion for next session: the boundary-trim and false-positive gaps seen on holdout are a model
  quality/training-data issue, not something safely fixable by more inference-time postprocessing
  heuristics without a much more principled way to distinguish "genuine fragment" from "adjacent
  distinct mention" than exact terminology-lookup match (which fails whenever either side's span is
  itself imperfect -- the common case). Retraining (more epochs, boundary-focused augmentation, or a
  training objective that penalizes under-extension) is the more promising direction.

## 2026-07-21 (part 2) — real leaderboard regression investigation

- **Real leaderboard score dropped after the RxNorm RRF integration below: 40.32290**
  (WER 53.1232, J_assertion 48.9858, J_candidates 28.9102) vs the prior deployed-model submission
  (v13, 2026-07-20: 40.5885, WER 52.7321, J_assertion 48.9599, J_candidates 29.3004). User asked for
  root-cause analysis (`models/ner_model/` was **not** retrained between these two submissions — same
  checkpoint, confirmed by mtimes/hash below — so this is a pure pipeline-code regression, not a
  worse model).
- **Found and fixed a real bug in `RxNormOfflineIndex.lookup()`** (`scripts/build_rxnorm_rrf_index.py`):
  the "don't return a bare-ingredient guess" guard read `if best_overlap == 0 and rest:` — backwards.
  `rest` (the mention's tokens after the first word) is *empty* for a single-word query, which makes
  `and rest` short-circuit `False`, so the guard **never fired for exactly the single-word queries it
  was written to block** — the most common miss shape. Confirmed empirically pre-fix:
  `lookup("methicillin")` and `lookup("Enterococcus")` (a bacterium name, not a drug) both returned a
  "best guess" RXCUI. Fixed to `if not rest or best_overlap == 0:`. Verified post-fix: both now
  correctly return `[]`; dose-aware multi-token queries (`"furosemide 40 mg"` etc.) and the 360047
  `--verify` case are unaffected.
  - Audited every THUỐC entity where the fallback fired on the 100-file corpus (9 total, pre-fix):
    every single one was attached to a **pre-existing, unrelated model span error** (over-extension —
    `"aspirin 325mg hằng ngày"` vs truth `"aspirin 325mg"`; outright false positives — `"methicillin"`,
    `"Enterococcus"` predicted where truth has no such entity at all; fragmentation — `"Insulin"` +
    `"glargine"` predicted separately where truth has one `"Insulin glargine"` entity). Per
    `check_submission.py`'s Jaccard formula (confirmed by re-reading the original `RxNavFallback`
    docstring this file's `RxNormOfflineFallback` replaced): any extra candidate on a mention whose
    *text* doesn't exactly match a truth mention only enlarges the union with zero chance of
    intersecting, i.e. **guessing wrong is strictly worse than guessing empty** whenever there's no
    truth to match. The bug made the always-on, always-successful (no network/timeout gate) offline
    fallback add exactly this kind of pure-downside guess for every span the model already got wrong
    — explaining the observed `J_candidates` drop (29.30 → 28.91).
  - Post-fix, 2 of the 9 firings (bare single-token misses: `Enterococcus`, `glargine`) are now
    suppressed. The remaining ~7 (exact matches on real drug words that happen to sit on a
    false-positive/fragmented entity, or genuine dose-aware fuzzy matches on an over/under-extended
    span) are **inherent to having any text-matching fallback at all** — the old live `RxNavFallback`
    would carry the identical risk if/whenever it had network access; not fixable at the
    candidate-linking layer without disabling the fallback's entire reason for existing (recall on
    genuinely novel private-test drugs).
- **Could not fully explain the WER component (52.7321 → 53.1232) from this bug.** Candidate linking
  runs strictly after entity decoding (`link_candidates()` only ever sets `ent["candidates"]`) and
  cannot change predicted entity text/spans by construction — verified by direct A/B testing: reverted
  the fix, regenerated `output_model` end-to-end, and diffed byte-for-byte against the fixed version —
  **only the `candidates` field differed, in exactly the 2 files (37, 87) with fallback-affected
  entities; zero span/text/assertion differences anywhere.** So this bug cannot be the WER cause.
- **However, that same A/B test surfaced something else, unrelated to the bug fix, worth flagging**:
  the very first `run_pipeline.py` execution this session (the one whose `output_model` got zipped and
  submitted for 40.32290) produced entity counts/warnings that a `diff -rq`-confirmed-deterministic
  re-run *of the exact same code* no longer reproduced a few minutes later (2287 entities / 4 warnings
  vs a stable 2286 entities / 5 warnings from every run after). Since decode code was untouched
  throughout and torch inference on this machine is otherwise proven deterministic (`model.eval()`,
  three independent same-code re-runs all byte-identical), the only remaining explanation is that
  `models/ner_model/model.pt` (1.1GB) itself was in a different state for that first run than for every
  run since. `config.json`/`model.pt` mtimes (14:52:38 / 14:55:13) sit suspiciously inside this exact
  session window despite neither file being touched by any command run here. **Working hypothesis:
  OneDrive (this repo lives under `OneDrive\Máy tính\ViettelRace`) re-synced/rehydrated the large
  model file mid-session** — consistent with the *other* OneDrive-shaped anomaly found this session:
  `.git/` on this machine has no objects/refs, only `info/exclude`, i.e. it looks stripped/desynced
  too. `model.pt` md5 `1051f7de5045503855ab2bc61955e05f` recorded here as a going-forward integrity
  checkpoint — no prior hash exists to compare against, since none was captured before this session.
  **Recommendation, not yet acted on**: move this working copy out of OneDrive sync (or exclude this
  folder from Files-On-Demand) before trusting further local inference runs or real submissions;
  re-verify `model.pt` against the original Kaggle kernel output if still available.
- **Current state**: `output_model/` on disk now reflects three consecutive, mutually-identical,
  fix-applied, hash-verified-model runs — local proxy `final_score 0.926839` (WER 0.045452,
  J_assertion 0.912307, J_candidates 0.916957), consistent with pre-session baseline. Worth
  resubmitting to see whether the real score recovers toward the ~40.5–40.8 range now that both the
  candidate-linking bug is fixed and the model file has read consistently across repeated local runs.

## 2026-07-21

- **Integrated the official RxNorm RRF release, replacing the RxNav API dependency.** UMLS
  Metathesaurus registration (previously blocked, see 2026-07-20 entry below) went through; user
  downloaded both `RxNorm_full_07062026.zip` and the smaller "Prescribable Content" release, unzipped
  to `rrf/` and `prescribe/rrf/` respectively (~1.8GB combined — gitignored, not committed;
  `.gitignore` updated with a note on where to re-download from). Repo hygiene: the zips had also
  dumped their bundled Oracle/MySQL loader SQL scripts directly into `scripts/` (`scripts/mysql/`,
  `scripts/oracle/`) alongside our actual pipeline scripts — moved to `rrf/mysql/`, `rrf/oracle/`
  (unused; we parse the RRF text files directly with Python, no DB load needed).
  - **New `scripts/build_rxnorm_rrf_index.py`** derives two committed CSVs: `data/terminology/
    rxnorm_full.csv` (517,991 rows / 511,646 distinct texts, from `RXNCONSO.RRF`'s 362,409 SAB=RXNORM
    rows + `RXNATOMARCHIVE.RRF`'s 373,484 rows, 38MB) and `data/terminology/rxnorm_drug_names.csv`
    (11,201 clean ingredient/brand names from the Prescribable Content release's `RXNCONSO.RRF`,
    filtered to short IN/PIN/BN entries so obscure research chemicals from the full release's ~33k
    IN-level names don't leak into synthetic training data).
  - **Closes the confirmed RxNav hard limit** documented in `fetch_rxnorm.py`'s docstring: RxNav's
    search endpoints cannot return any concept with RxNorm status `"Remapped"`, so the task
    statement's own worked example (`Chlorpheniramine 0.4 MG/ML / Dextromethorphan 6 MG/ML /
    Guaifenesin 40 MG/ML / Pseudoephedrine 6 MG/ML Oral Solution` → RxNorm 360047) was structurally
    unreachable through the old live-API approach. `RXNATOMARCHIVE.RRF` (RxNorm's own historical-atom
    table, which turned out to be 100% SAB=RXNORM already, no extra filtering needed) has it directly
    — verified via `--verify`, now a standing regression check on every rebuild.
  - **`run_pipeline.py`**: replaced the network-dependent `RxNavFallback` (live `quick_lookup` call,
    disabled after first failure) with `RxNormOfflineFallback`, backed by the new `RxNormOfflineIndex`
    class (exact-normalized-text match, then a first-token + dose/form-token-overlap fallback —
    O(matching entries), not a full 512k-row scan). Strictly better for the private-rerun-environment
    risk `CLAUDE.md` flags: no network call, no latency, no dependency on connectivity the organizers'
    environment may or may not have. `--no-rxnav-fallback` CLI flag renamed `--no-rxnorm-fallback`.
  - **`augment_ner_dataset.py`**: `THUỐC` entity-substitution pool now also draws from the 11.2k-name
    `rxnorm_drug_names.csv`, not just the corpus's own ~109 drug strings + `drugs.csv` — directly
    targets generalization to drug names the private test set introduces that never appeared in
    `output/`. Added `ChainedDrugMatcher` (curated `drugs.csv` first, offline RxNorm index second) so
    synthetic entities pulled from the new pool still get a real `candidates` value in the written
    JSONL, even though (confirmed by reading `train_ner_assertion_model.ipynb`) the training loop
    never actually reads `candidates` — only BIO tags and assertions feed the loss; entity linking is
    a wholly separate lookup step run after inference. That confirmation is itself useful: it means
    augmentation's `candidates` field only matters for artifact correctness, not training quality, so
    this fix was cosmetic/correctness, not a training-signal change.
  - **`fetch_rxnorm.py`** marked superseded in its own docstring and in `CLAUDE.md`; kept as a manual
    one-off live-lookup tool, removed from the default pipeline.
  - Re-ran the full local pipeline against the still-deployed (pre-this-change) model to confirm no
    regression: `prepare_ner_dataset.py` (unchanged, 2223 entities), `build_terminology_index.py`
    (drugs.csv: 126 rows, diagnoses.csv: 235 rows — unchanged, as expected, since corpus THUỐC texts
    already had 100% coverage), `augment_ner_dataset.py` (230 synthetic docs, confirmed the new
    ~11.2k-name pool is actually being drawn from -- e.g. `Ryaltris`, `Vectibix`, `Orencia` substituted
    in with real RxNorm-derived candidates via `ChainedDrugMatcher`, none of which were in the old
    109-text corpus pool), `run_pipeline.py` (100 files, no exceptions, `RxNormOfflineFallback` loads
    fine), `check_submission.py --truth output`: `final_score 0.926990` (0 errors, 4 pre-existing
    fragmentation warnings) — matches the 0.926 this same model scored before this change (see
    2026-07-20 entry), confirming the swap from `RxNavFallback` to `RxNormOfflineFallback` is
    behavior-neutral on the known corpus, as expected (the offline index only fires for THUỐC texts
    `drugs.csv` doesn't already cover, which today's known corpus doesn't have). **The actual payoff
    -- better generalization to unseen drug names -- only shows up after retraining on the new
    `train_augmented.jsonl`; not done yet, since that costs a Kaggle GPU run** (see CLAUDE.md: don't
    loop training). Also noticed `.git` in this working copy has no objects/refs (only
    `info/exclude`) -- pre-existing, not caused by this session, likely a OneDrive sync artifact on
    this path; flagged to the user rather than touched.

## 2026-07-20

- **First real leaderboard submission of a model-based (`output_model/`) zip: 35.67130**
  (WER 59.1407, J_assertion 41.8891, J_candidates 27.1168), submitted 17:09 — below hand-tuned
  `output/`'s last real score of 41.59120 (Run 8). Root cause, confirmed by diffing
  `output_model/` against `output/` entity-by-entity: `check_submission.py`'s `J_assertion`/
  `J_candidates` key on **exact entity text**, not position overlap, so any span-boundary miss
  (a comma-modifier clause dropped, a compound phrase split at "và"/"kèm") zeroes out that
  entity's contribution to all three metrics at once, not just WER. This is a real precision/
  recall limitation of the 20-epoch model, not something fixable in decode logic — checked
  whether a decode-time rule to bridge same-type entities separated by "và"/"kèm" would help,
  but `output/`'s own labels contain ~100 counterexamples (e.g. "ho và mệt mỏi", "sốt và đau cơ",
  "tylenol và advil" are each two separate entities), so that rule would net-hurt and was not
  added.
  - **Fixed one confirmed regression in `merge_fragmented_entities()` (`scripts/run_pipeline.py`)**:
    it was gluing two touching same-type CANDIDATE_TYPES fragments together unconditionally,
    which is correct for tokenizer artifacts (`"g"+"gine"` -> `"glargine"`) but wrong when the
    input text itself concatenates two distinct drugs with no separator (`"ciproflagyl"` in
    `output/64.json` is truth-labeled as two entities, "cipro" RxNorm 203563 + "flagyl" RxNorm
    202866, glued into one useless span with no candidates by the old logic). Fix: before
    merging, look both fragments up in the `TerminologyMatcher`; if both resolve to distinct
    non-empty candidates on their own, they're known real terms — leave separate. Verified fix
    on the corpus: local proxy `J_candidates` 0.7445 -> 0.7596, `final_score` 0.7740 -> 0.7819
    (0 errors/warnings, no regressions).
- **Retrain with the `MAX_LENGTH=512` fix (below) confirmed the diagnosis: real leaderboard score
  35.671 -> 40.769** (WER 53.4271, J_assertion 49.137, J_candidates 30.1391), submitted 17:54,
  right below hand-tuned `output/`'s 41.591. `models/ner_model/` now holds this retrained export
  (`config.json` confirms `max_length: 512`); Kaggle log in `.kaggle_download/` shows holdout WER
  0.3186 / J_assertion 0.3518 (up from the pre-fix run's 0.4054 / 0.3049).
- **Real leaderboard result for the sliding-window inference change: 40.769 -> 40.828 (+0.06),
  essentially flat** (WER 53.4271 -> 53.3781, J_assertion 49.137 -> 49.183, J_candidates
  30.1391 -> 30.2151), submitted ~18:xx — nowhere near the local proxy's predicted jump (final
  proxy 0.7819 -> 0.8804, i.e. +9.85 on the same 0-100 scale). Root cause, found by checking which
  of the 21 long (>512-token) documents the windowing affects fall in train vs. holdout: **20/21 are
  in the 85-doc train split; only docs 32 and 70 are in the 15-doc holdout split.** The local proxy
  compares `output_model/` against `output/` directly, and `output/` *is* the source of every
  training label -- so for those 20 train-split docs, "recovering" their previously-truncated tail
  mostly means the model correctly reproducing labels it was already trained on (loss had converged
  to ~0.007 by epoch 20), not new correct predictions on genuinely unseen text. That inflates the
  full-corpus local proxy without reflecting real generalization, since the real leaderboard grades
  against a truth the model never had access to, and 40/2223 corpus entities recoverable this way is
  too small a lever to move the aggregate much once the memorization credit is discounted. Silver
  lining: the model's real score (40.828) is now within 0.76 points of hand-tuned `output/`'s
  best-ever real score (41.591, Run 8) — via genuine inference, not memorized answers, which was
  the actual goal (see CLAUDE.md's disqualification-risk constraint). **Lesson for future local
  iteration on this repo: don't trust the full-100-file proxy against `output/` for changes that
  specifically affect train-split documents or that let the model recall memorized labels -- prefer
  the notebook's own holdout-only metric (15 files never used in training) as the cleaner
  generalization signal, even though it's also only measuring agreement with `output/`, not real
  truth.** The training-side windowing (below) is less exposed to this confound since it changes
  what the model *learns*, not just what it can recall, but should still be sanity-checked against
  holdout specifically after the next retrain, not the full corpus.
- **Added sliding-window inference to `scripts/run_pipeline.py`** to close the remaining gap: even
  at 512 tokens, 21/100 corpus documents are still longer than that (xlm-roberta-base's hard
  position-embedding limit), losing ~10.7% of entities from their tail. `run_inference()` now
  tokenizes the full document once (untruncated, offsets only), slices it into overlapping
  512-token windows (64-token stride -- longer than any entity mention in the corpus), runs the
  model per window, and merges results: `dedupe_entities()` collapses exact-duplicate predictions
  from the overlap region, `resolve_overlapping_entities()` handles the case where a mention sits
  right at one window's cut and comes out truncated in that window while a neighboring window
  (which contains it away from any edge) predicts it whole -- prefers whichever version didn't
  touch a synthetic window boundary. Combined with the `ciproflagyl` merge-guard fix above, local
  proxy (`output_model/` vs `output/`, not real truth) went WER 0.1972 -> 0.0557, `final_score`
  0.7819 -> 0.8804, and the "overlapping entities" warnings the naive windowing introduced (7 -> 0
  after `resolve_overlapping_entities`) are resolved. Pure inference-time change, no retraining
  needed -- already validated against the retrained model above. Regenerated `output_model/`;
  worth a resubmission before the next Kaggle run, since it costs no GPU quota and the local proxy
  suggests a meaningfully higher real score than 40.769.
- **Found and fixed a real (if unproven on this corpus) bug candidate for the v13 regression:
  `dedupe_entities` was unioning assertions across duplicate predictions from two overlapping
  windows.** A token near a window's synthetic edge has strictly less bidirectional context than
  the same token in a neighboring window's interior, so an edge-adjacent duplicate's assertion
  probabilities are less trustworthy than an interior duplicate's -- but the union kept BOTH
  versions' assertions regardless, silently adding a false-positive label even when the
  higher-context interior prediction correctly had none. This failure mode doesn't exist in
  single-window decoding (only one prediction per entity, nothing to union). Fixed: prefer the
  non-edge duplicate's assertions outright; only union when both duplicates are equally
  (un)trustworthy (both edge or both interior). Verified 0 regressions on the local proxy
  (identical J_assertion 0.9104 before/after) -- expected, since this exact conflict apparently
  doesn't occur among the known 100 files' predictions, so this fix is a defensible correctness
  improvement but **not confirmed** to be what caused the real-leaderboard drop below. Included in
  the regenerated `output_model_submission.zip`.
- **Real leaderboard regression from kernel v13: 40.828 -> 40.5885** (WER 52.7321, J_assertion
  48.9599, J_candidates 29.3004) despite the genuinely-better holdout numbers below. Isolated the
  RxNav fallback as a possible cause and ruled it out: disabling it (`--no-rxnav-fallback`) moved
  the local full-corpus J_candidates proxy by only 0.0021 (0.9154 -> 0.9175), far too small to
  explain the real ~0.9-point drop, and it can't touch J_assertion at all (which also dropped) --
  candidate linking is orthogonal to assertion prediction. Best remaining explanation: the
  15-file holdout split is too small and high-variance to reliably predict full-100-file real
  performance; kernel v13's genuine holdout gain doesn't necessarily generalize to the other 85
  files scored on the real leaderboard. Tried to roll back to kernel v12's exact weights (the
  40.828 version) to confirm/recover -- **not recoverable**: `kaggle kernels output <slug>/12`
  silently ignores the version suffix and returns the *latest* run's output regardless (confirmed
  via MD5: the "v12" and "v13" downloads were byte-identical). `models/ner_model/` was already
  overwritten with v13 before this was discovered, with no backup taken first -- a real process
  gap (**lesson: snapshot `models/ner_model/` before overwriting it with a new retrain's export,
  since Kaggle's kernel-output endpoint cannot be used to fetch a past version's artifacts after
  the fact**). Net effect: currently deployed model (v13) real-scores below both v12 (40.828) and
  hand-tuned `output/` (41.591) -- recommended falling back to `output/` for any near-term
  submission while this is investigated further, rather than resubmitting v13 as-is.
- **Retrained with training-side windowing (kernel v13): genuine holdout improvement, not
  memorization.** Log confirms `windowed 315 docs -> 375 training examples (+60 from long docs)`
  fired this time. Holdout (15 files never trained on) WER 0.3186 -> **0.2955**, J_assertion
  0.3518 -> **0.4425** -- a real jump, cross-checked two ways: the notebook's own eval cell, and
  independently by deploying the export to `models/ner_model/` and running the actual
  `scripts/run_pipeline.py` restricted to just the 15 holdout ids against `output/`
  (WER 0.2827, J_assertion 0.4234 -- close enough to the notebook's numbers, given
  run_pipeline.py's extra postprocessing, to trust both). Full-100-file proxy also jumped
  (final_score 0.880 -> 0.926) but per the memorization lesson above, treat that full-corpus number
  as inflated and prefer the holdout-only read. J_candidates on the holdout subset (0.541) is
  **not** a clean generalization signal either way -- `data/terminology/*.csv` is mined from ALL
  100 files including these 15, so it already "knows" their answers regardless of model quality.
  Deployed model verified end-to-end (`torch.load` succeeds, 203 state_dict keys, 1.1GB) after two
  corrupted downloads (`IncompleteRead`, 0-byte `model.pt`) on a flaky connection -- always verify
  size/load before trusting a downloaded export, don't assume "COMPLETE" kernel status implies a
  good local copy. `output_model_submission.zip` regenerated with this model; worth resubmitting.
- **Added training-side sliding-window document splitting** (`window_records()` in both notebook
  mirrors' `code-dataset` cell), closing the same 512-token gap on the training side that the
  inference-side windowing above closes at submission time -- without this, the model was never
  trained on the tail of the 19/85 (train split) documents still longer than 512 tokens, only ever
  seeing document beginnings. `window_records()` splits a long doc into overlapping 512-token
  windows (64-token stride, same scheme as inference), keeping an entity in a window only when its
  full span fits inside it (never truncating one into a corrupt BIO label -- it lands whole in a
  neighboring overlapping window instead). Also updated the notebook's own holdout-eval cell
  (`predict_records`) to do the same sliding-window inference + merge as `run_pipeline.py`, so the
  printed holdout WER/J_assertion actually reflects what a real submission does for long documents,
  instead of the old single-truncated-window estimate. Added `STRIDE` to `code-setup` and `"stride"`
  to the exported `config.json` so `run_pipeline.py` picks up the same value automatically.
  Offline-validated `window_records` against the real `data/ner_dataset/train.jsonl` (no GPU
  needed): 19/85 docs get windowed, 85 -> 105 training examples, 0 bad spans, and **0/1953 original
  entities left uncovered by every window** -- confirms the model will now see every entity in
  training regardless of document length. Not yet retrained with this change.
- **Found and fixed the likely biggest root cause of the low holdout/leaderboard scores:
  `MAX_LENGTH = 320` in `train_ner_assertion_model.ipynb` (both `notebooks/` and
  `kaggle_upload/kernel/` mirrors) was silently truncating both training and inference input.**
  Tokenized every `input/*.txt` with the trained tokenizer and counted `output/*.json` entities
  starting past each cutoff: 46/100 docs exceed 320 tokens (median doc is already 316 tokens),
  which drops 640/2223 (28.8%) of ground-truth entities entirely -- the model never saw them
  during training and structurally cannot predict them at inference, no matter how well it
  otherwise trained. `xlm-roberta-base`'s position embeddings cap at 512 tokens (the max reachable
  without a sliding-window/chunking rewrite of both the notebook and `run_pipeline.py`); raising
  `MAX_LENGTH` to 512 cuts entity loss to 237/2223 (10.7%). Changed in both notebook mirrors,
  confirmed still byte-identical after the edit. Not yet retrained against this fix -- next
  training run should use it before drawing conclusions from any further holdout/leaderboard
  numbers, since every past run (including today's 35.671 leaderboard submission) was trained and
  evaluated with nearly 30% of entities unreachable.
- **Augmented-dataset retrain completed; `run_all.py train` fixed for kaggle CLI 2.x.** The
  `kaggle kernels output` step succeeded (kernel version 9, `status: COMPLETE`) but `run_all.py`
  raised `ner_model_export.zip not found in kernel output`. Cause: kaggle CLI 2.2.3 auto-extracts a
  single-zip kernel output instead of leaving it under its original name — it saves the outer bundle
  as `output.zip` and pre-extracts the inner `ner_model_export.zip` into a same-named folder
  (`.kaggle_download/ner_model_export/{config.json,model.pt,tokenizer.json,tokenizer_config.json}`).
  `stage_train()` now checks for that extracted folder first and falls back to unzipping
  `ner_model_export.zip` if present, instead of assuming the old raw-zip layout. Manually copied this
  run's already-downloaded files into `models/ner_model/` rather than re-triggering training (avoids
  burning another Kaggle GPU quota slot).
  - Holdout results for this run (315 train files: 85 real + 230 synthetic, 20 epochs): mean WER
    0.4054, J_assertion 0.3049 — essentially flat vs. the pre-augmentation run (WER 0.41 /
    J_assertion 0.29). `scripts/run_all.py infer` against `output/` on the full 100-file corpus (85 of
    which were in training, so this number is optimistic, not a generalization estimate) gave
    text_score 0.801, J_assertion 0.786, J_candidates 0.745, final_score 0.774.
    `output/` (hand-tuned, score 41.591 on the real leaderboard scale) still has **not** been
    replaced — promote only after comparing against a real leaderboard submission or a stricter
    holdout-only check.
- **Retrain kicked off on Kaggle with the augmented dataset.** Pushed `train_augmented.jsonl` (315
  records: 85 real + 230 synthetic) as `kaggle_upload/dataset/train.jsonl`, new dataset version live
  at kaggle.com/datasets/lucylng/viettelrace-ner-dataset. `python scripts/run_all.py train` running
  in the background to retrain and pull the result into `models/ner_model/`.
  - Note: `kaggle datasets version` failed with a path error
    (`No such file or directory: '...\.kaggle/uploads\kaggle_upload/dataset_holdout.jsonl.json'`)
    when invoked from git-bash (forward-slash `-p` path mixed with the CLI's Windows temp-dir
    handling). Fixed by running the same command from PowerShell with a native backslash path
    instead — `kaggle kernels push` (used by `run_all.py train`) was unaffected either way since
    `pathlib.Path` always resolves to native backslash paths internally regardless of the invoking
    shell.
- **Fixed subword-boundary fragmentation in `scripts/run_pipeline.py` decode logic.** The model was
  splitting single words across two entities with a confidence dip on a middle subword (e.g.
  "opioid" → THUỐC "opio" + TRIỆU_CHỨNG "id"; "glargine" → THUỐC "g" + gap + THUỐC "gine"). Added
  `merge_fragmented_entities()`: two entities touching directly, or separated only by ≤4 word
  characters with no whitespace/punctuation between them, are always one mention split by decoding
  in this schema — merged regardless of type mismatch (longer fragment's type wins), assertions
  dropped if the winning type doesn't support them (TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM). Verified
  against the real fragmentation cases pulled from `output_model/`; correctly leaves genuine
  boundaries alone (e.g. `wbc` / `11.` separated by `) `). `check_submission.py` warnings on the
  100-file corpus: 21 → 13. Does **not** fix pure recall gaps (a fragment with no adjacent partner to
  merge into, e.g. `output_model/60.json`'s `'vị'` from a mostly-missed `CHẨN_ĐOÁN` phrase) or the
  `Insulin ... glargine` case (gap too large, >4 word chars) — those need more training
  data/epochs, not decode-time fixes.
- **First successful model training**, after two prior blockers fixed:
  - CUDA fix: Kaggle's preinstalled torch wheel has no kernel image for the P100 GPU (compute
    capability sm_60) that was assigned by default, crashing training ~40s into epoch 1
    (`no kernel image is available for execution on the device`). Fixed two ways: (1) always push
    with `--accelerator NvidiaTeslaT4` (T4 = sm_75, fully supported) — see `CLAUDE.md`; (2)
    `notebooks/train_ner_assertion_model.ipynb` now runs one real forward+backward pass on a dummy
    batch right after building the model, before the training loop starts, falling back to CPU if
    that probe fails (the previous probe only tested a trivial tensor add, which passed even on the
    broken GPU).
  - Result: 20 epochs on T4, no crash, holdout WER 0.4117 / J_assertion 0.2875 (pre-augmentation
    85-file train set). `scripts/run_pipeline.py` produced `output_model/*.json` end-to-end for the
    first time; validated 0 schema errors on all 100 files via `check_submission.py`.
- **Added `scripts/augment_ner_dataset.py`** — synthetic training data (entity-text substitution +
  templated assertion-cue clauses), required by the task statement's "use techniques outside the
  core solution to generate more data." 85 real docs → 315 with augmentation. Never touches
  `holdout.jsonl`.
- **Added `scripts/run_all.py`** — single entrypoint (`prepare` / `train` / `infer` / `all` stages)
  so the full pipeline doesn't depend on remembering the right command order.
- **Added `requirements.txt`** — pins torch/transformers for local inference (`run_pipeline.py`).
- **Added `--folds N` to `scripts/prepare_ner_dataset.py`** — optional, purely additive N-way split
  generation to spot-check whether the default 85/15 holdout score is a fluke of that one partition.
- **Added `scripts/fetch_rxnorm.py`** — RxNav REST API (free, no UMLS registration needed) as a
  substitute for the official RxNorm RRF release. Discovered while validating against the task
  statement's own worked example (`Chlorpheniramine 0.4 MG/ML` → RxNorm 360047): RxNav's search
  endpoints cannot return any concept with RxNorm status `"Remapped"` — confirmed by querying rxcui
  360047's exact canonical name and getting zero results; only `historystatus.json` reaches it, and
  only if the rxcui is already known. If the task's answer key was built against an older/frozen
  RxNorm snapshot, no live API text search can fully reconstruct it. Not urgent currently:
  `data/terminology/drugs.csv` already covers 100% (109/109) of distinct `THUỐC` texts in the
  100-file corpus; this only matters for drugs not yet seen.
- **Wrote `CLAUDE.md`** — architecture/commands orientation for future Claude Code sessions.
- Repo cleanup (carried over from before this session): old submission `.zip` files, `variant_*/`,
  `New folder/` removed 2026-07-17; corresponding score history preserved in `docs/score_history.md`.
