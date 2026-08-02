# ViettelRace — Vòng 1: trích xuất thực thể y tế tiếng Việt

Với mỗi bệnh án `input/{id}.txt`, sinh `output/{id}.json` gồm danh sách thực thể (`CHẨN_ĐOÁN`,
`TRIỆU_CHỨNG`, `THUỐC`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`), mỗi thực thể có span ký tự,
`assertions` (`isNegated`/`isHistorical`/`isFamily`) và với `CHẨN_ĐOÁN`/`THUỐC` là mã ICD-10/RxNorm.

```
final_score = 0.3·text_score(1 − WER) + 0.3·J_assertions + 0.4·J_candidates
```

Đề bài gốc BTC: [docs/problem_statement.md](docs/problem_statement.md) — nguồn sự thật cho mọi luật thi.

---

## Đọc 60 giây trước khi làm bất cứ gì

**Nút thắt đã được xác định bằng đo đạc, không phải phỏng đoán.**

| | text | J_assert | J_cand | final |
|---|---:|---:|---:|---:|
| turn-2 baseline | 35.72 | 39.56 | 29.51 | **34.388** |
| turn-2 + distillation | 38.34 | 43.59 | **29.34** | 36.32 |
| turn-1 nhãn tay, 8 vòng | 48.34 | 50.32 | **29.98** | 41.591 |

NER cải thiện 2.6 điểm text và assertion 4.0 điểm ⇒ `J_candidates` **giảm 0.17**. Nhãn tay của con
người, span tốt hơn hẳn ⇒ vẫn 29.98. Suy ngược `J = k/(2−k)`: cả ba cấu hình đều có
**~45–46% mã đúng**.

Nhưng bảng tra đã phủ **93.2%** mention chẩn đoán bằng khớp *chính xác*. Vậy vấn đề không phải độ
phủ — mà là **cột `candidate` sai một nửa**, vì `diagnoses.csv` được mine từ `output/`, chính là bài
nộp turn-1 đạt `J_candidates` 29.98. Bảng tra đang ghi nhớ lại chính các câu trả lời sai của mình.

**⇒ Việc đáng làm nhất trong repo này là mã hoá lại ~350 chuỗi, không phải đổi kiến trúc model.**
Xem [docs/linking_recode.md](docs/linking_recode.md).

**Ba bài học đã trả giá bằng điểm:**

1. **Không có số liệu offline nào ở đây có giá trị dự báo.** `models/ner_model/config.json` ghi
   `train_holdout_overlap: true` và holdout WER `0.006`. Eval linking thì feed ground-truth span cho
   linker. Lần eval dự báo `J_cand` 0.59 → 0.74 đã cho kết quả thật 29.51 → **28.68**.
2. **Đừng đổi tập thực thể.** Mọi lần đổi đều âm: 31.89, 33.679. Hậu xử lý *thuộc tính* mới an toàn.
3. **Mỗi lần nộp chỉ đổi một biến**, và ghi vào [docs/experiments.md](docs/experiments.md) **trước** khi nộp.

---

## 1. Cài đặt

```bash
pip install -r requirements.txt      # chỉ torch + transformers
```

Mọi script trong `scripts/` **trừ `run_pipeline.py`** đều stdlib-only.

Trên Windows luôn ép UTF-8 trước — console mặc định (cp932/cp1252) không in được tiếng Việt và crash:

```powershell
# PowerShell (mặc định trên Windows)
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
```
```bash
# git-bash / WSL / Linux
export PYTHONIOENCODING=utf-8 && export PYTHONUTF8=1
```

> **Repo này chạy chính bằng PowerShell.** Mọi lệnh `python …` bên dưới chạy y nguyên ở cả hai
> shell, nhưng các lệnh *shell thuần* thì không: `ls -la` → `Get-ChildItem`, `cp` → `Copy-Item`,
> `rm -rf` → `Remove-Item -Recurse -Force`, `grep` → `Select-String`. Chỗ nào khác biệt, tài liệu
> ghi rõ cả hai.

## 2. Chạy inference (không cần train, không cần nhãn)

Đường dùng cho private test. **~5 phút / 100 file trên CPU.**

```bash
python scripts/run_pipeline.py \
    --input input_turn2 --pred experiments/v5_recoded \
    --no-icd-fallback \
    --drop-short-noise --add-terminology-entities --add-public-phrase-entities

python scripts/check_submission.py  --pred experiments/v5_recoded --input input_turn2
python scripts/package_submission.py --pred experiments/v5_recoded --input input_turn2 \
    --out submissions/v5_recoded.zip
```

Ba flag cuối là **công thức đã chấm 34.388** — bỏ đi điểm tụt (đã thử: 31.89).
`--no-icd-fallback` là bản revert sau kết quả 33.679.

## 3. Sửa bảng tra — lever lớn nhất hiện có

**Giai đoạn 0 (không cần GPU, đã áp dụng):**

```bash
python scripts/extract_mentions.py                                          # -> recode_worklist.csv
python scripts/recode_terminology.py --audit                                # -> recode_autofix.csv
python scripts/recode_terminology.py --proposed data/terminology/recode_autofix.csv --dry-run
python scripts/recode_terminology.py --proposed data/terminology/recode_autofix.csv
```

Sửa 22 dòng / 37 mention chắc chắn bị chấm 0: mã 3 ký tự có mã con (`I48`→`I48.9`, `E14`→`E14.9`)
và mã không có trong catalog BYT (`S06.4X9A`→`S06.4`, `N40.0`→`N40`). Ước tính **+0.6 … +1.25**.

**Giai đoạn 1–4 (cần GPU):** sinh corpus diễn giải bằng Qwen → train bi-encoder → đề xuất mã mới cho
worklist. **Bi-encoder KHÔNG nằm trong đường suy luận** — nó chỉ xây bảng. Chi tiết + prompt + tham số:
[docs/linking_recode.md](docs/linking_recode.md). Ước tính **+3.4 … +7.4**.

Sau khi đổi bảng, **bắt buộc sinh lại prediction và kiểm tra tập entity không đổi** (phải đúng
**2898** trên turn-2 — nếu khác, cột `text` của bảng đã bị đụng và tập thực thể đã dịch chuyển):

```bash
python scripts/run_pipeline.py --input input_turn2 --pred experiments/v5_recoded \
    --no-icd-fallback --drop-short-noise --add-terminology-entities --add-public-phrase-entities
python scripts/check_submission.py --pred experiments/v5_recoded --input input_turn2
```

Thay `v5_recoded` bằng tên thư mục bạn muốn. `entities: 0` nghĩa là thư mục `--pred` không tồn tại
hoặc rỗng — kiểm tra lại đường dẫn, đừng chép nguyên chỗ giữ chỗ trong tài liệu.

Chạy lại `--audit` sau khi đã áp dụng sẽ báo `0 rows` — đó là **đúng**, công cụ idempotent, không
phải lỗi. Kết quả chi tiết từng lần chạy nằm ở [docs/experiments.md](docs/experiments.md).

## 4. Hậu xử lý thuộc tính

`scripts/postprocess.py` chỉ sửa `assertions`/`candidates` của thực thể **đã có**, không bao giờ
đụng span:

```bash
python scripts/postprocess.py --pred experiments/v5_recoded --input input_turn2 \
    --out experiments/v6 --sections --negex --family-gate --consistency union
```

| Flag | Tác dụng | Chạm |
|---|---|---:|
| `--sections` | `isHistorical`/`isFamily` theo section header + cue trong câu | 53 |
| `--negex` | `isNegated` theo cue phủ định VI, có scope mệnh đề, chặn `không thể loại trừ` | 6 |
| `--family-gate` | Bỏ `isFamily` khi thiếu cue gia đình (`isFamily` chỉ 0.8% nhãn) | 18 |
| `--consistency union\|majority` | Đồng bộ assertion giữa mention trùng trong cùng doc | 185 |
| `--hedge-icd all\|uncurated` | Thêm mã `.9` làm candidate thứ 2 — **không khuyến nghị** | 293 / 19 |

Ngưỡng sinh lời của candidate thứ 2 là `p₂ > J/(1+J)` = **0.223** ở `J_cand` hiện tại; xác suất mã
`.9` trúng ước tính chỉ 0.11–0.20 ⇒ hedging đang là kèo âm.

## 5. Train lại model (chạy tay trên Kaggle)

Máy local quá yếu để train. Notebook train `xlm-roberta-large` + teacher `Qwen/Qwen2.5-7B-Instruct`
4-bit (distillation) — đúng công thức đã cho **36.32**.

```bash
python scripts/prepare_ner_dataset.py
python scripts/build_terminology_index.py
python scripts/augment_ner_dataset.py
python scripts/build_kaggle_bundle.py       # -> kaggle_bundle/ (43MB), có verify từng file
```

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
python -m kaggle datasets version -p kaggle_bundle -m "retrain" --dir-mode zip
python -m kaggle kernels push -p kaggle_upload\kernel --accelerator NvidiaTeslaT4
python -m kaggle kernels status lucylng/viettelrace-ner-assertion-train
python -m kaggle kernels output lucylng/viettelrace-ner-assertion-train -p .kaggle_download
```

> ⚠️ **Phải là `kaggle_bundle`, KHÔNG phải `kaggle_upload\dataset`.** Hai thư mục cùng khai báo slug
> `lucylng/viettelrace-ner-dataset`, nhưng chỉ `kaggle_bundle/` có `input_turn2/`, `scripts/`,
> `data/terminology/`, `output/`. Version nhầm ⇒ cell distillation nằm trong `try/except` nên
> **không crash**, nó lặng lẽ bỏ Qwen và train curated-only: **~34.4 thay vì ~36.3**.

- `--dir-mode zip` bắt buộc — thiếu nó lệnh im lặng không upload thư mục con nào.
- `--accelerator NvidiaTeslaT4` bắt buộc — không có, Kaggle cấp P100 (sm_60) và training crash giữa
  epoch 1 với `CUDA error: no kernel image is available`.
- Kernel cần **Internet: On** để tải Qwen từ HuggingFace.

`kaggle kernels output` tải nguyên khối ~1.75GB và **không resume được** — nó đã đứt 2 lần
(`IncompleteRead`, `ConnectionAbortedError`). Dùng bản tải từng file có resume:

```powershell
python scripts/fetch_kernel_output.py --only ner_model_export/ --only output.zip --only llm_labels.json
Copy-Item .kaggle_download\ner_model_export\* models\ner_model\ -Force
```

**Luôn kiểm tra trước khi tin** — đã có 2 lần tải hỏng mà kernel vẫn báo `COMPLETE`:

```bash
python -c "import torch; sd=torch.load('models/ner_model/model.pt', map_location='cpu'); print(len(sd),'keys')"
```
```powershell
Get-ChildItem models\ner_model | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
# model.pt ~1100 MB = xlm-roberta-base | ~2136 MB = xlm-roberta-large
```

> Notebook có **hai bản** (`notebooks/` và `kaggle_upload/kernel/`); bản được push là bản thứ hai.
> **Sửa cả hai.** Quên đồng bộ đã từng làm tụt điểm 40.828 → 40.5885.

Hoặc để `run_all.py` làm hết (build bundle → version → push → poll → tải → giải nén):

```bash
python scripts/run_all.py train
```

## 6. Nộp source cho BTC

```bash
python scripts/package_source.py --dry-run
python scripts/package_source.py
```

BTC dựng lại source của top ~15 đội trên private test; **không cài được là bị loại**. Giải nén bundle
vào thư mục trống và chạy lại mục 2 từ đầu trước khi nộp.

## 7. Cấu trúc thư mục

```
input/ input_turn2/          bệnh án BTC (không sửa)
output/                      nhãn turn-1 tự gán — dữ liệu train + lexicon, KHÔNG phải đáp án
models/ner_model/            model fine-tune (gitignore): config.json, model.pt, tokenizer*
data/terminology/
   diagnoses.csv drugs.csv   bảng tra dùng lúc suy luận  ← lever chính, xem mục 3
   icd10_vi.csv              15.144 tiêu đề BYT (chỉ dùng offline lúc mã hoá lại)
   rxnorm_full.csv           517.991 dòng RxNorm
   recode_worklist.csv       546 chuỗi cần mã hoá lại (sinh bởi extract_mentions.py)
scripts/
   run_pipeline.py           suy luận (thứ duy nhất cần torch)
   postprocess.py            hậu xử lý thuộc tính
   extract_mentions.py       sinh worklist mã hoá lại
   recode_terminology.py     --audit / áp dụng mã mới, có 4 invariant chặn
   expand_terminology.py     --report: kết quả PHỦ ĐỊNH, đừng thử lại (xem docstring)
experiments/                 1 thư mục prediction / 1 cấu hình (gitignore)
submissions/                 zip đã đóng gói (gitignore)
archive/                     artifact các vòng cũ (gitignore)
legacy/                      pipeline regex cũ — chỉ audit, KHÔNG sinh submission
```

## 8. Bài nộp đang sẵn sàng

| Zip | Khác gì bản trước | Kỳ vọng |
|---|---|---|
| `v1_revert_icd.zip` | Revert ICD fallback đã bị falsify | ≈34.4 — chắc nhất |
| **`v5_recoded.zip`** | v1 + 22 mã sửa tất định (37 mention) | **≈35.0–35.6** |
| `v3_assert_union.zip` | v1 + lever assertion + lan truyền | swing assertion lớn nhất |
| `v2_assert.zip` | v1 + section/negex/family-gate | để tách nguyên nhân |
| `v4_hedge_all.zip` | v1 + hedging `.9` | thấp nhất, kèo âm |

**Thứ tự nộp: v5 → v1 → v3 → v2.** Trần của cả nhóm là ~36 vì đều chạy `xlm-roberta-base`; muốn
vượt 40 cần mục 3 (bảng tra) và mục 5 (train lại).
