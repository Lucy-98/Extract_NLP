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

## Current Best

- Final score: 41.59120
- WER: 51.6596
- J_assertion: 50.3235
- J_candidates: 29.9800

Lower WER is better. Higher J_assertion and J_candidates are better.

## Active Output

- Current `output/` is pending Run 9 candidate, not yet scored.
- It validates with 100 files, 2223 entities, 0 errors, 1 warning.
- It differs from stable Run 4 in 35 files: 3, 4, 11, 12, 13, 14, 15, 16, 18, 23, 24, 32, 34, 35, 37, 38, 39, 40, 46, 47, 49, 54, 56, 59, 61, 64, 71, 72, 73, 74, 88, 94, 96, 97, 100.
- Stable Run 4 is still available as `output (4).zip` and backup `.agent_runs/stable_before_breakthrough_20260714_211926`.
- Scored Run 7 snapshots: `.agent_runs/run7_breakthrough_candidate_20260714_215223`, `.agent_runs/scored_run7_4108770_20260714_215511`.
- Pending Run 8 snapshot: `.agent_runs/run8_candidate_20260714_220220`.
- Scored Run 8 snapshot: `.agent_runs/scored_run8_4159120_20260714_220609`.
- Pending Run 9 snapshot: `.agent_runs/run9_candidate_20260714_221158`.
- Main intended gains over scored Run 8: expand `xơ gan mất bù`, `xẹp phổi thùy dưới phải`, `viêm túi mật thủng cấp tính`; add missed `não úng thuỷ`, `tăng nhãn áp`, `Bệnh thủy đậu/Zona`; add missed file 49 symptom phrases.
