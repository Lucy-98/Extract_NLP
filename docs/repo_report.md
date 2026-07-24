# Báo cáo tổng quan repo ViettelRace

_Ngày lập: 2026-07-23. Đây là ảnh chụp trạng thái tại thời điểm đọc repo, không thay thế
[worklog.md](../worklog.md) (nhật ký kỹ thuật theo ngày) hay [score_history.md](score_history.md)
(lịch sử điểm). Khi hai file đó mâu thuẫn với báo cáo này, tin chúng._

## 1. Bài toán

ViettelRace AI Race 2026, **Đề 2**: trích xuất & chuẩn hóa khái niệm y tế từ bệnh án tiếng Việt tự
do (ghi chú bác sĩ, tóm tắt xuất viện, kết quả xét nghiệm, trích EHR).

- Với mỗi `input/{id}.txt` → sinh `output/{id}.json`: danh sách thực thể, mỗi thực thể có:
  - `text`, `type`, `position` `[start, end]` (span ký tự khớp đúng lát cắt input),
  - `assertions` (chỉ với `CHẨN_ĐOÁN`/`TRIỆU_CHỨNG`/`THUỐC`): `isNegated` / `isHistorical` /
    `isFamily`; rỗng nếu không có,
  - `candidates` (chỉ với `CHẨN_ĐOÁN`/`THUỐC`): ICD-10 cho chẩn đoán, **RxNorm theo liều+dạng** cho
    thuốc (clonazepam 0.5mg ≠ 1.5mg — không được map theo tên hoạt chất đơn thuần).
- **5 loại thực thể**: `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `THUỐC`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`.
- **Công thức điểm**: `0.3·text_score(1−WER) + 0.3·J_assertion + 0.4·J_candidates`.

**Ràng buộc kiến trúc cốt lõi**: BTC dựng lại source top ~15 đội và rerun trên private test. Mọi
cách không suy luận thật lúc runtime (regex/wordlist chép theo file cụ thể) sẽ không generalize và
có nguy cơ bị loại. Đây là lý do repo tách `legacy/` (rule-based, đã bỏ) khỏi model pipeline hiện
tại, và coi `output/` là **dữ liệu train**, không phải đáp án để tinh chỉnh tay tiếp.

## 2. Kiến trúc — 2 khối tách biệt

### Khối A: NER + Assertion model
`notebooks/train_ner_assertion_model.ipynb` (mirror ở `kaggle_upload/kernel/`, phải giữ đồng bộ tay).

- Nền `xlm-roberta-base`, 1 encoder chung + 2 head trên cùng encoder:
  1. **BIO tagging** (11 nhãn: O + B-/I- × 5 loại) → `text`/`type`/`position`.
  2. **Assertion multi-label** per-token, chỉ tính loss trên token thuộc CHẨN_ĐOÁN/TRIỆU_CHỨNG/THUỐC.
- Chọn XLM-R vì fast tokenizer cho `offset_mapping` trực tiếp trên text thô → không cần
  word-segmentation (như PhoBERT sẽ cần) giữa output model và field `position`.
- `MAX_LENGTH=512` + **sliding window** (stride 64) cả train lẫn inference — vá lỗi cũ
  `MAX_LENGTH=320` từng cắt cụt ~29% ground-truth entity.
- Early stopping theo holdout, export **best checkpoint** (không lấy epoch cuối). Calibrate ngưỡng
  assertion trên holdout.
- `candidates` **không** do model sinh — đó là bài entity-linking riêng.

### Khối B: Entity linking (chỉ lookup, tách khỏi model)
- `scripts/build_terminology_index.py` → `data/terminology/{drugs,diagnoses}.csv` mined từ
  `output/` + fallback dict legacy. `TerminologyMatcher` = exact-normalized-text + fuzzy (difflib).
  `conflicts.txt` liệt kê text map nhiều mã theo ngữ cảnh (vd "loét", "ung thư biểu mô tuyến").
- `scripts/build_rxnorm_rrf_index.py` → `rxnorm_full.csv` (~512k dòng text→RXCUI, gồm cả concept
  "Remapped"/retired qua `RXNATOMARCHIVE.RRF`) + `rxnorm_drug_names.csv` (~11.2k tên thuốc sạch cho
  augmentation). `RxNormOfflineIndex` = exact match rồi first-token + dose/form token-overlap.
  Xử đúng ví dụ đề bài (Chlorpheniramine... → 360047) mà RxNav API không tra được.

### Glue
- `scripts/run_pipeline.py` — **khối DUY NHẤT cần torch/transformers**. Load model từ
  `models/ner_model/`, dựng kiến trúc XLM-R cục bộ (không gọi HF Hub), inference sliding-window,
  decode BIO + assertion (threshold 0.5), link candidates qua TerminologyMatcher → fallback
  RxNormOffline. Ghi `output_model/*.json`.
- `scripts/run_all.py` — entrypoint gộp: `prepare | train | infer | package | submit | all`.
- `scripts/check_submission.py` — validator schema/span + mô phỏng điểm **type-aware** (prefix token
  bằng type để text đúng nhưng sai type = 0 điểm text; Jaccard giữ duplicate theo occurrence index).
- `scripts/package_submission.py` / `package_source.py` — đóng gói `output.zip` / `source_bundle.zip`.

**Ghi chú kỹ thuật quan trọng**: mọi script trừ `run_pipeline.py` là **stdlib-only** (chạy được ở
mọi máy, kể cả không có torch). Tất cả đều `sys.stdout.reconfigure(encoding="utf-8")` — bắt buộc
trên Windows (cp932/cp1252 không in được tiếng Việt sẽ crash).

## 3. Lệnh chạy chính

Đường suy luận private-test / nộp lại (không cần label public, không train):
```bash
python scripts/run_all.py submit --input input_turn2 --pred output_model_turn2 --out output_turn2.zip
```
Đường rebuild model đầy đủ (khi có label public `output/`):
```bash
python scripts/run_all.py all --aug-multiplier 1 --assertion-docs 30   # prepare -> train -> infer
```
`train` version Kaggle dataset, push kernel `--accelerator NvidiaTeslaT4`, poll, giải nén vào
`models/ner_model/`. Tốn quota GPU Kaggle mỗi lần — **không loop**.

## 4. Trạng thái thực tế (điểm số)

| Mốc | Điểm thật (leaderboard) | Ghi chú |
| --- | ---: | --- |
| `output/` hand-tuned, Run 8 | **41.591** | Bản nộp tốt nhất, **chưa bị thay** |
| Model v12 (genuine inference) | ~40.83 | Cách hand-tuned ~0.76 điểm |
| Model v13 | 40.59 | Regression so với v12 (holdout nhỏ, variance cao) |
| Model đầu (MAX_LENGTH=320) | 35.67 | Trước khi vá truncation |

**Bài học đã rút** (từ worklog):
- Proxy 100-file so với `output/` **bơm phồng** vì 85 file nằm trong tập train → chỉ tin
  holdout 15 file (dù nó cũng chỉ đo agreement với `output/`, không phải truth thật).
- Các gap còn lại (under-extension span, false positive, fragmentation) là **vấn đề chất lượng
  model / dữ liệu train**, KHÔNG vá được bằng postprocessing (đã thử merge theo "và"/whitespace và
  đều revert vì tạo false merge). Hướng đúng: train thêm epoch / augmentation tập trung biên span.
- `J_candidates`: đoán sai còn tệ hơn để rỗng (chỉ nới union, không giao) — đã sửa bug guard
  single-token trong `RxNormOfflineIndex.lookup()`.

## 5. Điểm cần lưu ý / việc dang dở

1. **`models/` đang TRỐNG** — chưa có model export cục bộ. Muốn chạy `run_pipeline.py` phải train
   lại trên Kaggle hoặc lấy export về `models/ner_model/` (cần `model.pt` + `config.json` +
   tokenizer files).
2. **Rủi ro OneDrive (lịch sử)** — repo từng nằm dưới OneDrive: `.git` từng desync/rỗng, `model.pt`
   1.1GB từng bị rehydrate giữa session gây kết quả không deterministic. Hiện repo ở
   `/Users/quanganh/Documents/code/ViettelRace`, git sạch, 1 commit `735fc41`. Nên xác minh
   `model.pt` bằng hash sau mỗi lần tải.
3. **ICD-10 full đã crawl** (`icd10_full.csv`, 11.243 mã) nhưng **chưa dùng được để match** — chỉ
   có title tiếng Anh, thiếu bước cầu nối Việt→Anh (dịch hoặc embedding đa ngôn ngữ).
4. **Split 85/15 seed 13 cố định** — dữ liệu quá nhỏ, coi số holdout là ước lượng sơ bộ.
   `prepare_ner_dataset.py --folds N` sinh thêm fold để spot-check (mỗi fold vẫn tốn 1 run Kaggle).
5. **`legacy/`** chỉ giữ để audit — wordlist chép verbatim từ 100 file public, không generalize.
   **Không dùng để sinh submission.**
6. **RRF thô** (`rrf/`, `prescribe/rrf/`, ~1.8GB) gitignored — chỉ cần khi rebuild CSV RxNorm. Môi
   trường private-rerun **không** cần chúng: `run_pipeline.py` chỉ đọc CSV derived đã commit.

## 6. Cây thư mục (rút gọn)

```
input/           100 bệnh án gốc (.txt)
output/          bản nộp hand-tuned tốt nhất (41.591) — KHÔNG bị pipeline ghi đè
output_model/    (gitignore) do run_pipeline.py sinh, để so với output/ trước khi promote
data/
  ner_dataset/   train/holdout/all.jsonl + train_augmented.jsonl + split.json
  terminology/   drugs.csv, diagnoses.csv, conflicts.txt, rxnorm_full.csv (38MB),
                 rxnorm_drug_names.csv, icd10_full.csv
models/          (gitignore, TRỐNG) model đã fine-tune
notebooks/       train_ner_assertion_model.ipynb
kaggle_upload/   dataset/ + kernel/ (bản push Kaggle, giữ đồng bộ với notebooks/)
scripts/         pipeline hiện dùng (run_pipeline.py cần torch; còn lại stdlib-only)
legacy/scripts/  pipeline cũ regex/wordlist — chỉ audit
docs/            score_history.md, github_collaboration.md, output_check_report.txt, repo_report.md
worklog.md       nhật ký kỹ thuật theo ngày (nguồn sự thật mới nhất)
CLAUDE.md        định hướng cho Claude Code
```
