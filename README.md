# ViettelRace — Vòng 1: trích xuất thực thể y tế tiếng Việt

Bài toán: với mỗi bệnh án (`input/{id}.txt`), sinh `output/{id}.json` chứa danh sách thực thể
(`CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `THUỐC`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`), mỗi thực thể có
span ký tự, assertion (`isNegated`/`isHistorical`/`isFamily`) và với `CHẨN_ĐOÁN`/`THUỐC` là mã
ICD/RxNorm (`candidates`). Điểm: `0.3·text_score(WER) + 0.3·J_assertion + 0.4·J_candidates`.

**BTC sẽ dựng lại source code của top ~15 đội trên private test** — bất kỳ cách làm nào không
chạy suy luận thật tại thời điểm inference (regex/wordlist chép theo từng file cụ thể) sẽ không
generalize và có nguy cơ bị loại. Vì vậy pipeline hiện dùng là 1 model NER+assertion fine-tune
(`xlm-roberta-base`, train trên Kaggle/Colab) + 1 bước entity-linking (tra cứu ICD/RxNorm) tách
riêng — không phải regex/wordlist. Chi tiết lý do và lịch sử thay đổi: xem `worklog.md`.

## Cấu trúc thư mục

```
input/                        100 bệnh án gốc (BTC cung cấp, không sửa)
output/                        bản nộp hiện hành (thủ công, đã chấm 41.591 — không bị pipeline mới ghi đè)
output_model/                  (sinh ra khi chạy run_pipeline.py, gitignore, so sánh với output/ trước khi thay)

data/
  ner_dataset/                 train.jsonl / holdout.jsonl / all.jsonl / split.json (sinh bởi prepare_ner_dataset.py)
                                 train_augmented.jsonl (sinh bởi augment_ner_dataset.py)
  terminology/                 drugs.csv / diagnoses.csv / conflicts.txt (sinh bởi build_terminology_index.py)
                                icd10_full.csv — 11.243 mã ICD-10 (chapter->category, tên tiếng Anh),
                                crawl toàn bộ qua WHO ICD-API bằng scripts/fetch_icd.py --crawl-all

models/                        model đã fine-tune, tải về từ Kaggle/Colab (gitignore, trống cho tới khi bạn train)

notebooks/
  train_ner_assertion_model.ipynb   fine-tune trên Kaggle/Colab (GPU) — bản dùng để push là
                                     kaggle_upload/kernel/ (giữ đồng bộ 2 bản khi sửa)

scripts/                       pipeline đang dùng
  prepare_ner_dataset.py       input/ + output/ (đã tinh chỉnh)  -> data/ner_dataset/*.jsonl
                                 (--folds N: thêm fold{k}_{train,holdout}.jsonl để spot-check split)
  build_rxnorm_rrf_index.py     rrf/ + prescribe/rrf/ (RRF chính thức, local) -> data/terminology/
                                 rxnorm_full.csv, rxnorm_drug_names.csv (chỉ cần chạy lại khi bản RRF đổi)
  augment_ner_dataset.py        train.jsonl + terminology         -> data/ner_dataset/train_augmented.jsonl
  build_terminology_index.py   output/ đã tinh chỉnh + bảng cũ    -> data/terminology/*.csv
  fetch_icd.py                  WHO ICD-API: tra 1 mã, hoặc crawl toàn bộ cây ICD-10 -> icd10_full.csv
  fetch_rxnorm.py                [superseded 2026-07-21] RxNav API — giữ lại để tra thủ công, không
                                 còn nằm trong pipeline mặc định (xem build_rxnorm_rrf_index.py)
  run_pipeline.py               model + terminology index (+ fallback offline RxNorm RRF) -> output_model/*.json
  check_submission.py           validator + mô phỏng điểm cục bộ  (type-aware, dùng chung cho pipeline cũ/mới)
  package_submission.py         validate + đóng gói output.zip đúng layout output/{id}.json
  package_source.py             đóng gói code + data derived + model weights để BTC dựng lại
  run_all.py                    entrypoint gộp: prepare | train | infer | package | all

requirements.txt               pin torch/transformers cho run_pipeline.py (suy luận cục bộ)

legacy/scripts/                 pipeline cũ (regex/wordlist), giữ lại để tham khảo/audit — KHÔNG
                                 dùng để sinh submission nữa (overfit theo 100 file, không generalize)

docs/
  score_history.md              lịch sử điểm số qua các vòng thủ công (Run 1-8)
  output_check_report.txt       log lần chạy validator/agent gần nhất của pipeline cũ

worklog.md                      nhật ký thay đổi theo ngày (kỹ thuật, không phải điểm số)
CLAUDE.md                       tài liệu định hướng cho Claude Code khi làm việc trong repo này

.agent_runs/                    (gitignore) backup/snapshot từ các lần chạy legacy/scripts/auto_improve_agent.py
```

## Cách chạy lại từ đầu

`scripts/run_all.py` gộp toàn bộ chuỗi dưới đây thành 1 lệnh, để không phải nhớ đúng thứ tự khi lặp
lại nhiều vòng dưới áp lực thời gian thi (`python scripts/run_all.py all`, hoặc chạy riêng từng giai
đoạn `prepare` / `train` / `infer` / `package`). Chi tiết từng bước nếu muốn chạy tay:

```bash
# 1. Chuẩn bị dữ liệu train (chạy local, chỉ cần Python stdlib)
```

For `input_turn2/` or any private input folder, do not run train. Use:

```bash
python scripts/run_all.py submit --input input_turn2 --pred output_model_turn2 --out output_turn2.zip
```

`python scripts/run_all.py all --aug-multiplier 1 --assertion-docs 30` is the full public-label rebuild path:
`prepare -> train -> infer`.

```bash
python scripts/prepare_ner_dataset.py
# (tuỳ chọn) --folds 5 để thêm 5 fold train/holdout khác nhau, dùng spot-check xem điểm holdout
# mặc định (85/15, seed 13) có phải may rủi theo 1 split cụ thể không

# 1b. Xây bảng tra từ nhãn public + legacy fallback
python scripts/build_terminology_index.py

# 1c. (khuyến nghị) Sinh thêm dữ liệu train tổng hợp
python scripts/augment_ner_dataset.py

# 2. Đồng bộ train_augmented.jsonl thành kaggle_upload/dataset/train.jsonl và publish Kaggle Dataset.
#    `python scripts/run_all.py prepare` làm bước sync local này; `run_all.py train` version dataset
#    trước khi push kernel, trừ khi truyền --skip-dataset-version.
#    Sau đó chạy notebooks/train_ner_assertion_model.ipynb trên GPU, tải
#    ner_model_export.zip về, giải nén vào models/ner_model/
#    Nếu push kernel bằng Kaggle CLI, luôn ép accelerator T4 (không để hệ thống tự gán P100 --
#    xem worklog.md để biết lý do):
#      kaggle kernels push -p kaggle_upload/kernel --accelerator NvidiaTeslaT4
#    (`python scripts/run_all.py train` làm: version dataset + push + poll + tải + giải nén tự động.)

# 3. (tuỳ chọn, chỉ cần khi bản RRF trong rrf/ đổi) rebuild RxNorm derived CSV
python scripts/build_rxnorm_rrf_index.py

# 4. Suy luận trên input/ bằng model đã train (cần torch + transformers, xem requirements.txt)
python scripts/run_pipeline.py

# 5. Validate + so điểm cục bộ với baseline hiện tại
python scripts/check_submission.py --pred output_model --input input --truth output

# 6. Đóng gói file nộp đúng format BTC: output.zip -> output/{1..100}.json
python scripts/package_submission.py --pred output_model

# 7. Khi BTC yêu cầu source code, đóng gói code + data + weights
python scripts/package_source.py
```

`output/` (bản thủ công đã chấm 41.591) được giữ nguyên, không bị ghi đè — `run_pipeline.py` ghi
ra `output_model/` để so sánh trước khi quyết định thay thế bản nộp. `output.zip` mặc định được đóng
gói từ `output_model/`, không phải từ `output/`, để artifact nộp gắn với pipeline có thể dựng lại.

## Việc còn lại

- **ICD-10 đầy đủ đã có** (`data/terminology/icd10_full.csv`, 11.243 mã, tên tiếng Anh — crawl qua
  WHO ICD-API), **nhưng chưa dùng được trực tiếp để match chẩn đoán tiếng Việt**: API chỉ trả tiêu
  đề tiếng Anh, và free-text search của WHO chỉ hoạt động cho ICD-11, không phải ICD-10 — cần thêm
  một bước bắc cầu Việt→Anh (dịch, hoặc embedding đa ngôn ngữ) trước khi dùng để entity-linking;
  bước đó **chưa được cài đặt**.
- **RxNorm đầy đủ đã tích hợp (2026-07-21)**: bản RRF chính thức (`RxNorm_full_...zip` +
  "Prescribable Content") đã tải về, giải nén local vào `rrf/` + `prescribe/rrf/` (gitignore, ~1.8GB,
  tải lại tại trang RxNorm của NLM nếu cần — không commit). `scripts/build_rxnorm_rrf_index.py` build
  `data/terminology/rxnorm_full.csv` (offline, ~512k dòng `text -> RXCUI`, gồm cả
  `RXNATOMARCHIVE.RRF` để phủ các concept "Remapped"/đã gộp) và `rxnorm_drug_names.csv` (~11.2k tên
  thuốc/hoạt chất sạch để tăng đa dạng dữ liệu tổng hợp). Việc này giải quyết đúng giới hạn cũ của
  RxNav API: RxNav không tra được concept có status "Remapped" qua bất kỳ endpoint nào — ví dụ đúng
  trong đề bài (`Chlorpheniramine 0.4 MG/ML / .../ Oral Solution` → RxNorm 360047) giờ tra được offline
  (`build_rxnorm_rrf_index.py --verify`). `run_pipeline.py` dùng bảng này làm fallback (thay
  `fetch_rxnorm.py`/RxNav — không còn phụ thuộc mạng khi BTC dựng lại source trên môi trường private).
- Split train/holdout hiện tại (85/15, seed 13) là 1 lần cố định — dữ liệu quá nhỏ để tin tưởng
  tuyệt đối, coi số liệu holdout là ước lượng sơ bộ. `prepare_ner_dataset.py --folds N` sinh thêm
  N fold khác để spot-check khi cần, không bắt buộc chạy đủ N fold.
- Model hiện train trên `xlm-roberta-base` với early stopping theo holdout và export checkpoint tốt nhất,
  không lấy epoch cuối một cách mù quáng. Vẫn cần theo dõi holdout WER/J_assertion mỗi lần train lại vì dữ liệu nhỏ.
  Một số thực thể ngắn vẫn bị bỏ sót hoàn toàn (recall gap, không phải lỗi decode) — cần thêm dữ
  liệu train/epoch, không giải quyết được bằng hậu xử lý.
- `data/terminology/conflicts.txt` liệt kê các text có nhiều mã ICD khác nhau tùy ngữ cảnh (ví dụ
  "loét") — những case này về bản chất cần model hiểu ngữ cảnh, không giải được bằng bảng tra.
- `legacy/scripts/improve_from_baseline.py` không còn chạy được (baseline `New folder/output` đã bị
  xoá khi dọn dẹp repo) — giữ lại chỉ để tham khảo kỹ thuật map thuốc theo liều đã dùng trước đây.
