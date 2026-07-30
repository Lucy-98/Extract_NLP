# Score History

| Run | Final score | WER | J_assertion | J_candidates | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 36.29780 | 54.1554 | 47.5662 | 20.6863 | Baseline sau các cải thiện đầu |
| 2 | 40.12190 | 53.1702 | 49.1408 | 28.3267 | Thêm candidates cho thuốc/bệnh rõ ràng |
| 3 | 40.66300 | 52.5299 | 49.3668 | 29.0290 | Dọn false positive + sửa span/candidate ung thư, loét |
| 4 | 40.68250 | 52.4858 | 49.3848 | 29.0321 | Sau vòng agent/check cuối |
| 5 | 40.80730 | 52.2688 | 49.5363 | 29.0676 | Breakthrough candidate 16 file: mở rộng span + sửa type/candidate |
| 6 | 40.96090 | 52.2625 | 49.6130 | 29.3892 | Thêm recall thuốc dính chữ + tụ máu ngoài màng cứng |
| 7 | 41.08770 | 51.9423 | 49.8818 | 29.2645 | Thêm symptom miss + sửa span đau đầu gối; WER/assertion tăng, candidates giảm nhẹ |
| 8 | 41.59120 | 51.6596 | 50.3235 | 29.9800 | Thêm drug chắc + mở rộng diagnosis/symptom bị cắt |

## Delta

| From -> To | Final score | WER | J_assertion | J_candidates | Summary |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 -> 2 | +3.82410 | -0.9852 | +1.5746 | +7.6404 | Cải thiện lớn nhất đến từ J_candidates |
| 2 -> 3 | +0.54110 | -0.6403 | +0.2260 | +0.7023 | WER và candidates đều nhích lên |
| 3 -> 4 | +0.01950 | -0.0441 | +0.0180 | +0.0031 | Cải thiện rất nhẹ, chủ yếu WER/assertion |
| 4 -> 5 | +0.12480 | -0.2170 | +0.1515 | +0.0355 | Cải thiện chính từ WER và assertion, candidates tăng nhẹ |
| 5 -> 6 | +0.15360 | -0.0063 | +0.0767 | +0.3216 | Tăng rõ nhất ở J_candidates |
| 6 -> 7 | +0.12680 | -0.3202 | +0.2688 | -0.1247 | WER/assertion tăng tốt, candidates tụt nhẹ |
| 7 -> 8 | +0.50350 | -0.2827 | +0.4417 | +0.7155 | Cải thiện đều cả ba chỉ số |
| 1 -> 8 | +5.29340 | -2.4958 | +2.7573 | +9.2937 | Best mới, cân bằng WER/assertion/candidates |

## Current Best (turn 1)

- Final score: 41.59120
- WER: 51.6596
- J_assertion: 50.3235
- J_candidates: 29.9800

Lower WER is better. Higher J_assertion and J_candidates are better.

**Not reusable.** Run 8 is `output/`, hand-tuned file by file against the visible turn-1 answers
over eight scored rounds. It does not generalise and must never be submitted for a private rerun.
It survives only as training data and as the source of the terminology tables.

## Turn 2 (model pipeline, blind — no public labels)

Every row is a real leaderboard submission. Full reasoning for each is in `worklog.md`.

| Score | WER | J_assert | J_cand | Configuration | Model kept? |
| ---: | ---: | ---: | ---: | --- | --- |
| **36.3385** | **60.40** | 42.42 | 29.33 | **NEW BEST**: distill 70 docs + xlm-roberta-large + GPU T4 | **yes** |
| 36.3160 | 61.66 | 43.59 | 29.34 | distill, ~73 docs sparse (parser bug), base tables | lost |
| 35.7087 | 63.11 | 42.82 | 29.49 | distill 70 docs, assert-mask, ICD off, propagate | yes |
| 35.1865 | 62.27 | 41.61 | 28.47 | distill 100 docs @12.1/doc, ICD merge on | lost |
| 34.7142 | 62.39 | 38.79 | 29.48 | no distillation, curated labels only, large | lost |
| 34.3880 | 64.28 | 39.56 | 29.51 | pre-distillation baseline, xlm-roberta-**base** | **yes** |
| 32.7454 | 65.91 | 36.70 | 28.77 | distill 200 docs — relabeled turn-1 over its gold | lost |
| 31.4429 | 68.29 | 33.91 | 29.39 | distill 100 docs @22.1/doc (chunking on) | lost |
| 11.4736 | 86.40 | 14.01 |  7.98 | Qwen few-shot direct extraction (dead end) | n/a |

**Submit 35.7087.** It is the highest score whose checkpoint still exists, and the 0.61 gap to
36.3160 sits inside the run-to-run spread — no configuration has ever been run twice, so nothing in
the 34–36 band is separable from seed noise. 34.3880 (`models/ner_model` before the large export was
restored) is the zero-GPU fallback.

Pseudo-label dose is an inverted U, not a monotone gain: 0 entities → 34.71, ~800 → 36.32, ~1030 →
35.71, 1208 → 35.19, 2213 → 31.44.

## Active Output

- `output/` is the turn-1 Run 9 candidate, never scored; 100 files, 2223 entities, 0 errors,
  1 warning. Kept as the label set the terminology tables and training data are mined from.
- Turn-1 zip snapshots and `.agent_runs/` backups referenced by earlier revisions of this file are
  **gone** — `.agent_runs/` was removed in cleanup and `output (4).zip` was overwritten on
  2026-07-26 by a Kaggle download. Run 8 survives as `output/` itself, which is what matters.
- Scored Run 7 snapshots: `.agent_runs/run7_breakthrough_candidate_20260714_215223`, `.agent_runs/scored_run7_4108770_20260714_215511`.
- Pending Run 8 snapshot: `.agent_runs/run8_candidate_20260714_220220`.
- Scored Run 8 snapshot: `.agent_runs/scored_run8_4159120_20260714_220609`.
- Pending Run 9 snapshot: `.agent_runs/run9_candidate_20260714_221158`.
- Main intended gains over scored Run 8: expand `xơ gan mất bù`, `xẹp phổi thùy dưới phải`, `viêm túi mật thủng cấp tính`; add missed `não úng thuỷ`, `tăng nhãn áp`, `Bệnh thủy đậu/Zona`; add missed file 49 symptom phrases.
