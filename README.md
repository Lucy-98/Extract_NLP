# ViettelRace — Vòng 1: trích xuất thực thể y tế tiếng Việt

Với mỗi bệnh án `input/{id}.txt`, sinh `output/{id}.json` gồm danh sách thực thể (`CHẨN_ĐOÁN`,
`TRIỆU_CHỨNG`, `THUỐC`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`), mỗi thực thể có span ký tự,
`assertions` (`isNegated`/`isHistorical`/`isFamily`) và với `CHẨN_ĐOÁN`/`THUỐC` là mã ICD-10/RxNorm.

```
final_score = 0.3·text_score(1 − WER) + 0.3·J_assertions + 0.4·J_candidates
```

Đề bài gốc BTC: [docs/problem_statement.md](docs/problem_statement.md) — nguồn sự thật cho luật thi.
Lịch sử điểm đầy đủ: [docs/experiments.md](docs/experiments.md).

**Điểm tốt nhất hiện tại: 36.6183** (`submissions/v11_patience8.zip`).

---

# ⚡ Từ số 0 đến file nộp — 4 bước

Máy đã có `models/ner_model/` (model đã train) thì chỉ cần chừng này. **~15 phút trên CPU.**

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"

# 1. Suy luận
python scripts\run_pipeline.py --input input_turn2 --pred experiments\sub `
    --no-icd-fallback --drop-short-noise --add-terminology-entities --add-public-phrase-entities

# 2. Mở rộng span thuốc bị cắt cụt (+11.04 trên ground truth BTC)
python scripts\fix_drug_spans.py --pred experiments\sub --input input_turn2 --out experiments\sub_fixed

# 3. isHistorical từ header danh sách thuốc (+20.84 trên ground truth BTC)
python scripts\postprocess.py --pred experiments\sub_fixed --input input_turn2 --out experiments\sub_final --sections

# 4. Kiểm tra + đóng gói
python scripts\check_submission.py --pred experiments\sub_final --input input_turn2
python scripts\package_submission.py --pred experiments\sub_final --input input_turn2 --out submissions\sub.zip
```

`submissions\sub.zip` là file nộp. Bước 4 phải in **`errors: 0`** — nếu không, đừng nộp.

Đổi `input_turn2` thành thư mục input BTC giao (private test) là chạy được ngay, không sửa code.

**4 cờ ở bước 1 là công thức đã được chấm — đừng bỏ bớt.** Bỏ 3 cờ cuối từng làm điểm tụt xuống
31.89; `--no-icd-fallback` là bản revert sau kết quả 33.679.

---

# 📖 Đọc 90 giây trước khi tối ưu

## Nút thắt đã xác định bằng đo đạc

| | text | J_assert | J_cand | final |
|---|---:|---:|---:|---:|
| turn-2 baseline | 35.72 | 39.56 | 29.51 | 34.388 |
| + distillation | 38.34 | 43.59 | **29.34** | 36.32 |
| turn-1 **nhãn tay người** | 48.34 | 50.32 | **29.98** | 41.591 |
| **v11 (tốt nhất)** | 38.89 | 43.17 | **30.00** | **36.618** |

Model tốt lên 2.6 điểm text và 4.0 điểm assertion ⇒ `J_candidates` **giảm 0.17**. Người gán nhãn
tay, span tốt hơn hẳn ⇒ vẫn 29.98. Suy ngược `k = 2J/(1+J)`: **mọi cấu hình đều ~45–46% mã đúng.**

`J_candidates` (trọng số 0.4) là lever lớn duy nhất chưa được khai thác.
Kế hoạch: [docs/linking_recode.md](docs/linking_recode.md).

## Bốn bài học đã trả giá bằng điểm

1. **Số liệu offline duy nhất đáng tin là `data/corpus/gold_btc/`.** `models/ner_model/config.json`
   của model cũ ghi `train_holdout_overlap: true` với holdout WER `0.006`. Eval linking thì feed
   ground-truth span cho linker. Một eval dự báo `J_cand` 0.59 → 0.74 đã cho kết quả thật 29.51 →
   **28.68**.
2. **Đừng đổi tập thực thể một cách mù quáng.** Mọi lần làm thế đều âm: 31.89, 33.679.
   `fix_drug_spans.py` là ngoại lệ **chỉ vì** nó đo được trên gold trước khi nộp.
3. **Luật thủ công thua model.** Hậu xử lý assertion (`postprocess.py`) làm mất **0.66** — tỉ lệ
   `isFamily` của model (0.85%) khớp gần như hoàn hảo với nhãn tay turn-1 (0.9%).
4. **Mỗi lần nộp đổi một biến**, ghi vào [docs/experiments.md](docs/experiments.md) **trước** khi nộp.

---

# 1. Cài đặt

```powershell
pip install -r requirements.txt      # chỉ torch + transformers
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
```
```bash
# git-bash / WSL / Linux
export PYTHONIOENCODING=utf-8 && export PYTHONUTF8=1
```

Ép UTF-8 là **bắt buộc** trên Windows — console mặc định (cp932/cp1252) không in được tiếng Việt và
sẽ crash. Mọi script trong `scripts/` **trừ `run_pipeline.py`** đều stdlib-only.

> Repo chạy chính bằng **PowerShell**. Lệnh `python …` giống nhau ở mọi shell; lệnh shell thuần thì
> khác: `ls -la` → `Get-ChildItem`, `cp` → `Copy-Item`, `rm -rf` → `Remove-Item -Recurse -Force`,
> `grep` → `Select-String`.

# 2. Kiểm tra chất lượng trên ground truth THẬT

`data/corpus/gold_btc/` là ví dụ 19 thực thể BTC công bố trong đề bài, tái dựng đúng đến từng
offset. Đây là **eval offline duy nhất có giá trị dự báo** trong repo.

```powershell
python scripts\build_gold_btc.py --verify     # phải in 19/19; nếu không, bỏ --verify để dựng lại
```

> Dấu phân cách trong document là `\n\r\n` và offset chỉ khớp khi giữ nguyên từng byte.
> Ghi file theo cách thông thường trên Windows sẽ dịch `\n` thành `\r\n`, làm lệch **toàn bộ**
> offset mà **không báo lỗi** (bản đầu tiên: 576 ký tự, 0/19 khớp; bản đúng: 554 ký tự, 19/19).
> `build_gold_btc.py` dựng lại từ đề bài và assert cả 19 offset trước khi ghi.

```powershell
python scripts\run_pipeline.py --input data\corpus\gold_btc\input --pred experiments\gold `
    --no-icd-fallback --drop-short-noise --add-terminology-entities --add-public-phrase-entities
python scripts\check_submission.py --pred experiments\gold --input data\corpus\gold_btc\input `
    --truth data\corpus\gold_btc\truth
```

Chạy nó **trước mỗi thay đổi có đụng span hoặc mã**. Số hiện tại:

| | pipeline thô | + `fix_drug_spans` | + `--sections` |
|---|---:|---:|---:|
| text_score | 82.43 | **95.95** | 95.95 |
| J_assertion | 20.00 | 20.00 | **89.47** |
| J_candidates | 4.76 | **22.22** | 22.22 |
| **final** | 32.63 | 43.67 | **64.52** |

Ba lỗi hệ thống nó phát hiện, theo thứ tự ưu tiên:

1. ~~**Span thuốc cắt cụt 7/11 (64%)**~~ — đã sửa bằng `fix_drug_spans.py`. Đây là lỗi đắt nhất vì
   span sai làm mất **cả ba** metric của thực thể đó (assertion và candidate đều key theo
   `(text, type, occurrence)`), tức 1.0 trọng số chứ không phải 0.3.
2. ~~**`isHistorical` thiếu 11/19**~~ — đã sửa bằng `postprocess.py --sections`. Ba lỗi riêng
   biệt: `HEADER_RE` đòi dấu `:` (header thật kết thúc bằng dấu chấm), cụm từ trong list là
   *"thuốc trước **khi** nhập viện"* trong khi text thật là *"thuốc trước nhập viện"*, và quan trọng
   nhất — section không được gán cho **mọi** loại: truth đánh `isHistorical` cho 11 thuốc nhưng để
   **rỗng** cả 8 triệu chứng (chúng là chỉ định, thì hiện tại). Kết quả: `J_assertion` 20.00 →
   **89.47**, `final` 43.67 → **64.52**.
3. **RxNorm sai 3/4 ngay cả khi span đúng** — `aspirin 81 mg po daily` → truth 243670, ta đoán
   2668107. **Chưa sửa**, thuộc phạm vi [docs/linking_recode.md](docs/linking_recode.md).

> Cảnh báo diễn giải: gold chỉ có **1 document, toàn thuốc có sig** (100% mention thuốc chứa
> route/frequency). Turn-2 chỉ **3%**. Nên `fix_drug_spans.py` sửa 7 span trên gold nhưng chỉ 2 trên
> turn-2. Đừng suy ra tỉ lệ lỗi của gold cho tập khác — chỉ suy ra **cơ chế** lỗi.

# 3. Sửa bảng tra — lever lớn nhất còn lại

**Giai đoạn 0 (không cần GPU, đã áp dụng):**

```powershell
python scripts\extract_mentions.py                 # -> data\terminology\recode_worklist.csv (546 dòng)
python scripts\recode_terminology.py --audit       # -> data\terminology\recode_autofix.csv
python scripts\recode_terminology.py --proposed data\terminology\recode_autofix.csv --dry-run
python scripts\recode_terminology.py --proposed data\terminology\recode_autofix.csv
```

Đã sửa 23 mã chắc chắn bị chấm 0: mã 3 ký tự có mã con (`I48`→`I48.9`, `E14`→`E14.9`) và mã không
có trong catalog BYT (`S06.4X9A`→`S06.4`, `J91`→`J90`). Đo được: `J_candidates` 29.38 → **30.20**.

Chạy lại `--audit` sau khi áp dụng sẽ báo `0 rows` — **đúng**, công cụ idempotent.

**Giai đoạn 1–4 (cần GPU):** sinh corpus diễn giải bằng Qwen → train bi-encoder → đề xuất mã mới.
Bi-encoder **không nằm trong đường suy luận** — nó chỉ xây bảng. Chi tiết, prompt, tham số, và điều
kiện phủ định: [docs/linking_recode.md](docs/linking_recode.md). Ước tính **+3.4 … +7.4**.

Sau **mọi** thay đổi bảng, bắt buộc kiểm tra tập thực thể không đổi (turn-2 phải là **2296**):

```powershell
python scripts\check_submission.py --pred experiments\sub_fixed --input input_turn2
```

Khác con số đó nghĩa là cột `text` của bảng đã bị đụng và tập thực thể đã dịch chuyển — dừng lại.

# 4. Train lại model (Kaggle GPU, chạy tay)

Máy local quá yếu để train. Notebook train `xlm-roberta-large` + teacher `Qwen/Qwen2.5-7B-Instruct`
4-bit — công thức đã cho **36.6**.

```powershell
python scripts\prepare_ner_dataset.py
python scripts\build_terminology_index.py
python scripts\augment_ner_dataset.py
python scripts\build_kaggle_bundle.py       # -> kaggle_bundle\ (44MB), verify từng file

$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
python -m kaggle datasets version -p kaggle_bundle -m "retrain" --dir-mode zip
python -m kaggle kernels push -p kaggle_upload\kernel --accelerator NvidiaTeslaT4
python scripts\run_all.py train --skip-push --timeout 14400    # chỉ theo dõi, không push lại
```

Bốn chỗ dễ sai, mỗi chỗ đều **hỏng im lặng**:

| | |
|---|---|
| Dùng `kaggle_bundle`, **KHÔNG** `kaggle_upload\dataset` | Hai thư mục cùng slug; bản thin thiếu `input_turn2/`, `scripts/`, `data/`. Sai ⇒ distillation bị `try/except` nuốt ⇒ **~34.4 thay vì ~36.6** |
| `--dir-mode zip` | Thiếu ⇒ không upload thư mục con nào |
| `--accelerator NvidiaTeslaT4` | Thiếu ⇒ Kaggle cấp P100 (sm_60) ⇒ crash giữa epoch 1 |
| **Internet: On** trong kernel settings | Cần để tải Qwen từ HuggingFace |

**Tải model về** (`kaggle kernels output` tải nguyên khối 1.75GB và không resume được — đã đứt 3 lần):

```powershell
python scripts\fetch_kernel_output.py --only ner_model_export/ --only llm_labels.json
Copy-Item .kaggle_download\ner_model_export\* models\ner_model\ -Force
Copy-Item .kaggle_download\llm_labels.json data\ner_dataset\ -Force   # cache -> lần sau ~30 phút
```

**Luôn verify trước khi tin** — đã có 3 lần tải hỏng mà kernel vẫn báo `COMPLETE`:

```powershell
python -c "import torch; sd=torch.load('models/ner_model/model.pt', map_location='cpu'); print(len(sd),'keys')"
Get-ChildItem models\ner_model | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
# model.pt ~1100 MB = base | ~2136 MB = large (395 keys)
```

Rồi quay lại **mục ⚡** để sinh file nộp.

> Notebook có **hai bản** (`notebooks/` và `kaggle_upload/kernel/`); bản được push là bản thứ hai.
> **Sửa cả hai.** Quên đồng bộ đã từng làm tụt điểm 40.828 → 40.5885.

> `llm_labels.json` đã được commit. Nhờ nó bước Qwen (1.5–2.5h, chiếm ~80% thời gian train) được bỏ
> qua — kernel in `[distill] CACHE HIT`. Chỉ xoá file này khi `input_turn2/` hoặc prompt thay đổi.

# 5. Nộp source cho BTC

```powershell
python scripts\package_source.py --dry-run
python scripts\package_source.py
```

BTC dựng lại source của top ~15 đội trên private test; **không cài được là bị loại**. Giải nén bundle
vào thư mục trống và chạy lại **mục ⚡** từ đầu trước khi nộp.

# 6. Cấu trúc thư mục

```
input/ input_turn2/          bệnh án BTC (không sửa)
output/                      nhãn turn-1 tự gán — dữ liệu train + lexicon, KHÔNG phải đáp án
data/
  corpus/gold_btc/           ★ ground truth THẬT (19 entity từ đề bài) — eval offline duy nhất đáng tin
  terminology/
    diagnoses.csv drugs.csv  bảng tra dùng lúc suy luận  ← lever chính, xem mục 3
    icd10_vi.csv             15.144 tiêu đề BYT (chỉ dùng offline lúc mã hoá lại)
    rxnorm_full.csv          517.991 dòng RxNorm
    recode_worklist.csv      546 chuỗi cần mã hoá lại
  ner_dataset/
    llm_labels.json          cache nhãn Qwen — giữ lại, nó tiết kiệm 2h mỗi lần train
models/ner_model/            model đã train (gitignore): config.json, model.pt, tokenizer*
scripts/
  run_pipeline.py            suy luận (thứ duy nhất cần torch)
  fix_drug_spans.py          ★ mở rộng span thuốc bị cắt; --truth để chấm trên gold
  extract_mentions.py        sinh worklist mã hoá lại
  recode_terminology.py      --audit / áp dụng mã mới, có 4 invariant chặn
  postprocess.py             hậu xử lý assertion — ĐÃ BỊ BÁC BỎ, đọc docstring
  expand_terminology.py      --report: kết quả PHỦ ĐỊNH, đừng thử lại
  fetch_kernel_output.py     tải kernel output có resume
  build_kaggle_bundle.py     dựng bundle 44MB + verify
  check_submission.py        validate schema + mô phỏng điểm khi có --truth
  package_submission.py      đóng gói output.zip đúng layout
experiments/ submissions/ archive/    (gitignore) prediction, zip, artifact cũ
legacy/                      pipeline regex cũ — chỉ audit, KHÔNG sinh submission
```

# 7. Lịch sử điểm

| Ngày | Điểm | Là gì |
|---|---:|---|
| 07-24 | 34.388 | `xlm-roberta-base` + curated linking |
| 07-26 | 36.32 | large + distillation — **artifact đã mất** |
| 07-31 | 33.679 | + ICD fallback → **bị bác bỏ** |
| 08-01 | 34.303 | revert + 22 fix mã |
| 08-01 | 33.644 | + luật assertion → **bị bác bỏ** |
| 08-01 | 35.951 | large + distillation, tái lập từ source |
| 08-01 | 36.379 | + bảng sạch của repo (bỏ 46 dòng Qwen) |
| 08-02 | **36.618** | + `EARLY_STOP_PATIENCE=8` ← **tốt nhất** |

Chi tiết từng lần, kèm phân tích vì sao: [docs/experiments.md](docs/experiments.md).
