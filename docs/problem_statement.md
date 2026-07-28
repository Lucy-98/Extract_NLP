# Đề bài & Quy định (bản gốc BTC)

_Nguồn: đề bài chính thức của Ban Tổ chức, do người dùng cung cấp và lưu lại ngày 2026-07-24.
Đây là nguồn sự thật cho luật thi; mọi mô tả thứ cấp trong [CLAUDE.md](../CLAUDE.md) /
[README.md](../README.md) / [repo_report.md](repo_report.md) phải khớp với file này._

## Thể thức

Vòng 1: thí sinh nộp kết quả dự đoán dưới dạng JSON theo format BTC quy định. File nộp là một
`output.zip`, giải nén ra:

```
output/
    ├── 1.json     # Nhãn của bản ghi 1
    ├── 2.json     # Nhãn của bản ghi 2
    ├── …
    └── 100.json
```

**Lưu ý về source code (chống hard-code):**
- Trước khi Vòng 1 kết thúc, BTC yêu cầu **top ~15 đội** gửi trước source code để dựng lại và đánh
  giá trên **dữ liệu private test** — nhằm tránh nộp file hard-code output theo input được cung cấp.
- Source code gồm:
  - tất cả file code (data processing, training, inference, …),
  - data nhóm sử dụng,
  - **model weights**,
  - 1 file README hướng dẫn cài đặt.
- Nếu BTC **không cài đặt được** code, nhóm được liên lạc riêng để hỗ trợ trong một khoảng thời gian
  nhất định; **không hỗ trợ kịp thời sẽ bị loại**.

## Ví dụ input → output (Vòng 1)

**Input:**
> `Danh sách thuốc trước nhập viện chính xác và đầy đủ. 1. amlodipine 10 mg po daily 2. aspirin 81
> mg po daily 3. metoprolol succinate xl 50 mg po daily 4. guaifenesin ml po q6h:prn điều trị ho 5.
> nystatin oral suspension 5 ml po qid:prn điều trị đau nhức 6. acetaminophen 325-650 mg po q6h:prn
> điều trị sốt đau 7. pravastatin 40 mg po daily 8. docusate sodium 100 mg po bid điều trị táo bón
> 9. senna 8.6 mg po bid:prn điều trị táo bón 10. clonazepam 0.5 mg po qam:prn điều trị lo âu 11.
> clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ`

**Output:**
```json
[
  {"text": "amlodipine 10 mg po daily", "type": "THUỐC", "candidates": ["308135"], "assertions": ["isHistorical"], "position": [58, 83]},
  {"text": "aspirin 81 mg po daily", "type": "THUỐC", "candidates": ["243670"], "assertions": ["isHistorical"], "position": [89, 111]},
  {"text": "metoprolol succinate xl 50 mg po daily", "type": "THUỐC", "candidates": ["866436"], "assertions": ["isHistorical"], "position": [117, 155]},
  {"text": "guaifenesin ml po q6h:prn", "type": "THUỐC", "candidates": ["392085"], "assertions": ["isHistorical"], "position": [161, 186]},
  {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [196, 198]},
  {"text": "nystatin oral suspension 5 ml po qid:prn", "type": "THUỐC", "candidates": ["7597"], "assertions": ["isHistorical"], "position": [204, 244]},
  {"text": "đau nhức", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [254, 262]},
  {"text": "acetaminophen 325-650 mg po q6h:prn", "type": "THUỐC", "candidates": ["313782"], "assertions": ["isHistorical"], "position": [268, 303]},
  {"text": "sốt đau", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [313, 320]},
  {"text": "pravastatin 40 mg po daily", "type": "THUỐC", "candidates": ["904475"], "assertions": ["isHistorical"], "position": [326, 352]},
  {"text": "docusate sodium 100 mg po bid", "type": "THUỐC", "candidates": ["1099279"], "assertions": ["isHistorical"], "position": [358, 387]},
  {"text": "táo bón", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [397, 404]},
  {"text": "senna 8.6 mg po bid:prn", "type": "THUỐC", "candidates": ["312935"], "assertions": ["isHistorical"], "position": [410, 433]},
  {"text": "táo bón", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [443, 450]},
  {"text": "clonazepam 0.5 mg po qam:prn", "type": "THUỐC", "candidates": ["197527"], "assertions": ["isHistorical"], "position": [457, 485]},
  {"text": "lo âu", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [495, 500]},
  {"text": "clonazepam 1.5 mg po qhs", "type": "THUỐC", "candidates": ["197528"], "assertions": ["isHistorical"], "position": [507, 531]},
  {"text": "lo âu", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [541, 546]},
  {"text": "mất ngủ", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [547, 554]}
]
```

> Chú ý ví dụ: `clonazepam 0.5 mg po qam:prn` → RxNorm **197527**, còn `clonazepam 1.5 mg po qhs` →
> **197528**. Cùng hoạt chất, khác liều → khác mã ⇒ RxNorm phải theo **liều + dạng**, không map theo
> tên hoạt chất. Đây là ràng buộc cốt lõi của bước entity-linking.

## Metric đánh giá

Tính trên tập test theo 3 thành phần:

- **text** — Word Error Rate (WER) trên trường `text`.
- **assertions** — Jaccard similarity trên assertions (với bệnh/thuốc/triệu chứng), trung bình thành
  1 điểm `J_assertions`.
- **candidates** — Jaccard tương tự trên trường `candidates`.

**Công thức cuối:**
```
final_score = 0.3·text_score + 0.3·assertions_score + 0.4·candidates_score
```

Với `i` là 1 sample trong test, `k` là 1 candidate (khái niệm có candidates) trong sample `i`:

```
text_score       = ( Σ_{i∈test} (1 − WER(i)) ) / len(test)

assertions_score = ( Σ_{i∈test} J_assertions(i) ) / len(test)

                     Σ_{i∈test} [ J_candidates(i) · Σ_{k∈i} (len(ground_truth(k)) + 1) ]
candidates_score = ─────────────────────────────────────────────────────────────────────
                            Σ_{i∈test} Σ_{k∈i} (len(ground_truth(k)) + 1)
```

Định nghĩa Jaccard cho trường `X` của sample `i`:
```
J_X(i) = 1                                            nếu len(gt_X(i)) = 0 và len(pred_X(i)) = 0
J_X(i) = 0                                            nếu len(gt_X(i)) = 0 và len(pred_X(i)) ≠ 0
J_X(i) = |gt_X(i) ∩ pred_X(i)| / |gt_X(i) ∪ pred_X(i)|   trong các trường hợp còn lại
```

> **Lưu ý quan trọng (đoán đúng text, sai loại):** nếu đoán đúng phần `text` của khái niệm nhưng
> **sai `type`** (VD đoán `CHẨN_ĐOÁN` nhưng ground truth là `TRIỆU_CHỨNG`), khái niệm bị **tính 2
> lần** (do tạo ra 1 khái niệm mới so với ground truth) và **mỗi lần đều 0 điểm cả 3 metric**.

## Tài nguyên

- Thí sinh **tự chuẩn bị tài nguyên tính toán**.
- Với giải pháp **LLM/agent**: chỉ cho phép **self-host model, không được dùng API ngoài**. Model
  self-host **tối đa 9B params**.

## Đối chiếu với code repo (2026-07-24)

Kiểm tra nhanh mức tuân thủ hiện tại:

| Yêu cầu đề | Trạng thái repo |
| --- | --- |
| Format `output.zip` → `output/{1..100}.json` | ✅ [package_submission.py](../scripts/package_submission.py) đóng đúng layout |
| Nộp source: code + data + weights + README | ✅ [package_source.py](../scripts/package_source.py) gói `scripts/`, `data/`, `models/ner_model`, `README.md` |
| Rerun trên private test (không hard-code) | ✅ Pipeline model thật; `legacy/` regex đã bỏ; đường `run_all.py submit` không cần label public |
| RxNorm theo liều+dạng | ✅ `rxnorm_full.csv` + `RxNormOfflineIndex` giữ mã dose-specific (vd 197527 vs 197528) |
| Metric: weighted candidates + type-aware WER | ✅ [check_submission.py](../scripts/check_submission.py) khớp công thức: `candidate_weight = len(candidates)+1`, prefix type vào token WER |
| **LLM/agent: không API ngoài lúc inference** | ✅ `run_pipeline.py` hoàn toàn offline (đã bỏ RxNav API → RxNorm RRF offline). `fetch_icd.py`/`fetch_rxnorm.py` chỉ gọi API lúc build data trước, không nằm trong inference |
| **Model ≤ 9B params** | ✅ `xlm-roberta-base` (~270M) — dư địa rất lớn so với trần 9B |

Kết luận: kiến trúc hiện tại thỏa mãn toàn bộ ràng buộc đề bài. Trần **9B params** mở ra khả năng
nâng cấp encoder lớn hơn nhiều (hiện mới dùng ~270M) nếu muốn tăng chất lượng NER/assertion, miễn là
vẫn self-host offline.
