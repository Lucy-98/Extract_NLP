# Báo cáo chi tiết các phương pháp triển khai trong repository ViettelRace

_Ngày lập: 28/07/2026 (bản sửa lần 2 — xem mục 6 "Đính chính so với bản lần 1")_
_Repository: ViettelRace (Vòng 1, Đề 2: Trích xuất & chuẩn hóa thực thể y tế tiếng Việt)_

> Nguồn sự thật, theo thứ tự ưu tiên: [`docs/problem_statement.md`](problem_statement.md) (đề bài
> gốc BTC) → [`worklog.md`](../worklog.md) (nhật ký kỹ thuật theo ngày) →
> [`docs/score_history.md`](score_history.md) (điểm) → báo cáo này. Báo cáo này là ảnh chụp trạng
> thái nên sẽ cũ đi; khi mâu thuẫn thì tin ba file kia.

---

## 1. Tổng quan bài toán & ràng buộc

### 1.1. Bài toán
Với mỗi hồ sơ bệnh án tiếng Việt tự do `input/{id}.txt`, sinh `output/{id}.json` gồm danh sách thực
thể, mỗi thực thể có:

1. **`text`** — chuỗi trích xuất.
2. **`type`** — 1 trong 5 loại: `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `THUỐC`, `TÊN_XÉT_NGHIỆM`,
   `KẾT_QUẢ_XÉT_NGHIỆM`.
3. **`position`** — `[start, end]`, span ký tự phải khớp đúng lát cắt của input.
4. **`assertions`** — chỉ với `CHẨN_ĐOÁN`/`TRIỆU_CHỨNG`/`THUỐC`: `isNegated`, `isHistorical`,
   `isFamily`; rỗng nếu không có.
5. **`candidates`** — chỉ với `CHẨN_ĐOÁN` (mã **ICD-10**) và `THUỐC` (mã **RxNorm**, chính xác theo
   liều + dạng bào chế: clonazepam 0.5mg ≠ 1.5mg).

### 1.2. Công thức điểm
```
Score = 0.3 · text_score(1 − WER) + 0.3 · J_assertion + 0.4 · J_candidates
```
`J_candidates` có trọng số cao nhất (0.4) và cũng là chỉ số thấp nhất trong mọi lần nộp (mục 4) →
đây là đòn bẩy lớn nhất còn lại.

### 1.3. Ràng buộc kỹ thuật cốt lõi từ BTC
- **BTC dựng lại source code** của top ~15 đội và chạy lại trên **private test set**.
- **Không gọi API ngoài lúc inference** (Gemini, OpenAI, RxNav REST…).
- **Mô hình suy luận ≤ 9 tỷ tham số**.
- **Phải suy luận thật**: regex/wordlist chép verbatim theo nội dung file public không tổng quát hóa
  được và có nguy cơ bị loại.

---

## 2. Các luồng triển khai trong repo

Repo có **5 luồng** đang tồn tại (luồng thứ 6 đã bị gỡ, xem mục 2.6). Chỉ luồng 1 là đường nộp:

```
     Dữ liệu public (input/ + output/, 100 file)        Dữ liệu BTC turn 2 (input_turn2/)
                    │                                                  │
                    ▼                                                  ▼
   [3] prepare_ner_dataset + augment_ner_dataset          [2] Qwen2.5-7B-Instruct 4-bit
       85 train / 15 holdout / 200 dòng augmented             (self-host trên Kaggle GPU)
                    │                                      pseudo-label + gợi ý mã, lọc ảo giác
                    └──────────────┬───────────────────────────────────┘
                                   ▼
                    [1a] Fine-tune xlm-roberta (base → large)
                         1 encoder chung + 2 head (BIO • assertion)
                                   │
                                   ▼
                    [1b] Entity linking OFFLINE (tra bảng, không phải model)
                         THUỐC     : drugs.csv    → RxNormOfflineIndex   (517,991 dòng)
                         CHẨN_ĐOÁN : diagnoses.csv → ICD10VietnameseIndex (15,144 dòng) ← MỚI 28/07
                                   │
                                   ▼
                    [4] run_all.py submit → output_model_*/ → output*.zip

     [5] legacy/ (rule-based, chỉ để audit — KHÔNG dùng để nộp)
```

---

### 2.1. Phương pháp 1 — Mô hình NER/Assertion + entity linking offline (đường nộp chính thức)

Đây là **đường duy nhất hợp lệ để nộp**: mọi thực thể đều do model sinh ra lúc chạy, không có gì khóa
theo id file.

#### a) Kiến trúc model (dual-head)
- **Backbone**: `xlm-roberta-base` (270M), đã nâng lên `xlm-roberta-large` (560M) từ 25/07. Cả hai
  đều ≪ 9B.
- **Vì sao XLM-RoBERTa**: fast tokenizer trả `offset_mapping` trực tiếp trên text thô → không cần
  bước tách từ (như PhoBERT sẽ cần), nên `position` không bị lệch.
- **Hai head trên một encoder chung**:
  1. Token classification (BIO, 11 nhãn: `O` + `B-`/`I-` × 5 loại).
  2. Multi-label assertion theo token; loss chỉ tính trên token thuộc
     `CHẨN_ĐOÁN`/`TRIỆU_CHỨNG`/`THUỐC`.
- **Sliding window** `MAX_LENGTH=512`, `stride=64`, áp dụng **cả lúc train lẫn lúc inference**. Cấu
  hình cũ `MAX_LENGTH=320` từng cắt mất **640/2223 (28.8%)** thực thể ground-truth — model không bao
  giờ nhìn thấy chúng khi train và về mặt cấu trúc không thể dự đoán chúng lúc inference.
- Tối ưu train: AMP (mixed precision), `nn.DataParallel` khi Kaggle cấp 2×T4, gradient checkpointing
  cho bản large.

#### b) Entity linking offline — tách hẳn khỏi model
Model **không** sinh mã; gán mã là bài toán tra cứu riêng, chạy offline 100%:

| Loại | Bảng | Cơ chế |
| :--- | :--- | :--- |
| `THUỐC` | `drugs.csv` (159 dòng, mined từ `output/`) | **chỉ** exact chuẩn hóa (containment bị đẩy xuống cuối, xem 2.7) |
| ↳ fallback | `rxnorm_full.csv` (**517,991** dòng) | `RxNormOfflineIndex`: exact → **mở rộng sang sản phẩm đủ liều+dạng (SCD/SBD)** → first-token overlap |
| ↳ cuối cùng | `drugs.csv` | containment + difflib |
| `CHẨN_ĐOÁN` | `diagnoses.csv` (321 dòng, mined từ `output/`) | exact chuẩn hóa → containment |
| ↳ fallback | `icd10_vi.csv` (**15,144** dòng) | `ICD10VietnameseIndex` (mục 2.5) |
| ↳ cuối cùng | `diagnoses.csv` | difflib fuzzy (ngưỡng 0.86) |

- `rxnorm_full.csv` dựng từ bản RxNorm RRF chính thức (`RXNCONSO.RRF` + `RXNATOMARCHIVE.RRF`) nên phủ
  được cả concept trạng thái **"Remapped"**/hết hiệu lực — ví dụ chính trong đề bài
  (`Chlorpheniramine 0.4 MG/ML … Oral Solution` → RxNorm `360047`) mà **không** endpoint tìm kiếm nào
  của RxNav công khai trả về được. `build_rxnorm_rrf_index.py --verify` kiểm tra đúng ví dụ này mỗi
  lần rebuild.
- `conflicts.txt` liệt kê các text map ra mã khác nhau tùy ngữ cảnh (vd "loét") — cần model hiểu ngữ
  cảnh, bảng tra không giải quyết được.

#### c) Script
- Suy luận: `scripts/run_pipeline.py` — **module duy nhất cần `torch`/`transformers`**; tự dựng kiến
  trúc XLM-R cục bộ từ `hf_config.json`, không gọi HuggingFace Hub.
- Đóng gói: `scripts/package_submission.py` (`output.zip`), `scripts/package_source.py`
  (`source_bundle.zip` để BTC dựng lại).
- Kiểm tra: `scripts/check_submission.py` — validate schema/span + mô phỏng điểm **type-aware**.

---

### 2.2. Phương pháp 2 — Chưng cất tri thức từ Qwen2.5-7B, chạy trên Kaggle (chỉ data-prep)

Điểm bị chặn bởi **thiếu nhãn + lệch phân phối** (100 file turn-1 curated so với văn phong turn-2),
không phải bởi compute:

1. Nạp `Qwen2.5-7B-Instruct` **4-bit** (bitsandbytes) **tự host trên Kaggle GPU** — không API, không
   API key, model Apache-2.0.
2. Pseudo-label toàn bộ 100 file `input_turn2/`: 5 loại thực thể + 3 assertion, kèm gợi ý mã ICD-10 /
   RxNorm.
3. **Căn vị trí & lọc ảo giác**: `locate_exact_positions` (exact → khớp bỏ qua khoảng trắng → cắt dấu
   câu); mã do LLM sinh phải **đối chiếu offline** với `icd10_full.csv` (11,243 mã) và
   `rxnorm_full.csv`, sai format hoặc không tồn tại thì loại bỏ. Mã ICD hợp lệ ghi ra
   `qwen_icd_supplement.csv`, chỉ merge vào `diagnoses.csv` cho các text **chưa có** trong bảng
   curated (giữ nguyên ánh xạ turn-1; Jaccard phạt mã thừa).
4. Gộp pseudo-label turn-2 với nhãn curated turn-1 → train `xlm-roberta-large`. Giải phóng LLM
   (`del` + `empty_cache`) trước khi train student.
5. Toàn bộ khối labeling nằm trong `try/except`: lỗi OOM/parse thì in cảnh báo và train bằng dữ liệu
   curated, không hỏng cả run.

**Tính hợp lệ**: LLM chỉ dùng ở bước **chuẩn bị dữ liệu**, không gọi API ngoài, và mô hình **được
nộp** vẫn là encoder nhỏ + bảng tra offline. Notebook: `notebooks/qwen_extract.ipynb` (bản few-shot
trích xuất trực tiếp) và các cell distillation trong `train_ner_assertion_model.ipynb`.

---

### 2.3. Phương pháp 3 — Sinh dữ liệu tổng hợp (`scripts/augment_ner_dataset.py`)

Đáp ứng yêu cầu của đề ("dùng giải pháp ngoài lời giải chính để tạo thêm dữ liệu"), thuần Python
stdlib:

1. **Thay thế thực thể trong câu thật** — tên `THUỐC` được thay bằng tên lấy từ **11,201** tên
   thuốc/hoạt chất RxNorm sạch (`rxnorm_drug_names.csv`), rồi dịch lại toàn bộ offset phía sau. Mục
   đích là dạy model **mẫu BIO**, không phải ghi nhớ ~109 tên thuốc có trong corpus — đúng khoảng
   trống mà private test sẽ khai thác.
2. **Chèn mẫu ngữ cảnh assertion** — phủ định ("không ghi nhận…"), tiền sử ("tiền sử 5 năm…"), tiền
   sử gia đình ("bố mắc…") quanh các mention chưa có assertion.
3. **Ghép văn bản đa thực thể** → `data/ner_dataset/train_augmented.jsonl` (hiện **200 dòng**).
4. **Không bao giờ chạm vào `holdout.jsonl`** — dữ liệu tổng hợp chỉ vào tập train.

---

### 2.4. Phương pháp 4 — Framework tự động hóa (`scripts/run_all.py`)

| Lệnh | Chức năng |
| :--- | :--- |
| `run_all.py prepare` | `output/` → `.jsonl`, build terminology index, sinh augmentation, đồng bộ sang `kaggle_upload/dataset/train.jsonl`. |
| `run_all.py train` | Tạo version Kaggle Dataset, push kernel `--accelerator NvidiaTeslaT4`, poll tới khi xong, giải nén export vào `models/ner_model/`. **Tốn quota GPU mỗi lần — không loop.** |
| `run_all.py infer` | `run_pipeline.py` → `output_model/`, validate (`--truth output` nếu thư mục đó tồn tại), đóng gói `output.zip`. |
| `run_all.py package` | Chỉ validate + đóng gói, không chạy lại inference. |
| `run_all.py submit` | **Đường nộp private test**: infer → validate → package, **không cần nhãn public, không train**. |
| `run_all.py all` | `prepare` → `train` → `infer` (bước `infer` đã bao gồm đóng gói). |

Lệnh nộp turn-2:
```bash
python scripts/run_all.py submit --input input_turn2 --pred output_model_turn2 --out output_turn2.zip
```

---

### 2.5. Phương pháp 5 (mới 28/07) — Chuẩn hóa ICD-10 tiếng Việt bằng danh mục Bộ Y tế

**Vấn đề**: `diagnoses.csv` được mined 100% từ nhãn public turn-1 (321 dòng). Trên văn bản chưa từng
thấy nó hỏng theo hai kiểu, cả hai đều đánh vào `J_candidates` (trọng số 0.4):
- chẩn đoán mới không có trong bảng → difflib gán **nhầm** mã của chuỗi turn-1 gần nhất;
- với Jaccard trên tập mã, **đoán sai còn tệ hơn để rỗng**.

**Cách làm** (`scripts/build_icd10_vi_index.py`, stdlib-only):
- Dựng `data/terminology/icd10_vi.csv` (**15,144 dòng** `code,name_vi,name_en`) từ danh mục song ngữ
  chính thức của Bộ Y tế (QĐ 4469/QĐ-BYT, nền WHO 2019). File thô ~10MB nằm trong
  `data/terminology/raw/` và **được gitignore** — cùng quy tắc raw-in/derived-out như `rrf/`.
- `ICD10VietnameseIndex.lookup()`: khớp tên chuẩn hóa chính xác → fallback **toàn bộ token truy vấn ⊆
  tên ICD** → **mở rộng nhóm 3 ký tự sang mã con "không xác định"** (`I26` → `I26.9`). Bước cuối này
  là bắt buộc vì đáp án chấm bằng mã 4 ký tự so khớp chuỗi chính xác: trả về đúng *nhóm* mà thiếu mã
  con thì điểm bằng 0 y hệt trả về mã sai. Không có bước đoán mò: trượt thì trả rỗng.
- Chèn vào `run_pipeline.py` theo thứ tự **curated exact/containment → ICD-vi → curated fuzzy**. Thứ
  tự này mới là điểm mấu chốt: một hit *fuzzy* của bảng curated chỉ là chuỗi gần nhất trong ~320
  chuỗi turn-1, kém tin cậy hơn một hit của từ điển ICD chính thức. Tắt bằng `--no-icd-fallback`.
- **Ba lớp chặn để giữ độ chính xác** (thêm sau khi A/B trên turn-2, mục 4.2): fallback từ-chối truy
  vấn **1 token** (trong 15k tên bệnh, một chữ "viêm"/"thiếu" bao giờ cũng khớp *một cái gì đó*);
  **loại hẳn chương U/V/W/X/Y** (nguyên nhân ngoại sinh — không có mã nào thuộc các chương này trong
  542 mã chẩn đoán turn-1); **đẩy chương O/P/Z xuống cuối** thay vì loại.

**Tác động vượt ngoài `J_candidates`**: `filter_noisy_entities` **xóa hẳn** mọi `CHẨN_ĐOÁN`/`THUỐC`
không gán được mã. Nghĩa là trước thay đổi này, một chẩn đoán trích xuất **đúng** nhưng không tra
được mã sẽ bị loại khỏi bài nộp, mất luôn cả `text` lẫn `assertions`. Vì vậy thay đổi này nâng cả ba
thành phần điểm. Kết quả đo: mục 4.2.

---

### 2.7. Phương pháp 6 (mới 28/07) — Chuẩn hóa RxNorm theo **liều + dạng bào chế**

**Phát hiện quyết định**: `RxNormOfflineIndex` **trượt 2/3 mã thuốc mà chính đề bài công bố**
(197527, 197528, 360047) — dù **cả ba mã đều có sẵn** trong `rxnorm_full.csv`. Đây là *dữ liệu đúng
duy nhất đã được kiểm chứng* về mã thuốc trong repo, và trước đó chưa từng được đem ra đánh giá.

**Nguyên nhân**: bệnh án ghi dạng bào chế bằng **ký hiệu đường dùng** (`po`) hoặc tiếng Việt
("đường uống", "đặt dưới lưỡi") — không chuỗi RxNorm nào chứa các từ đó. Vì vậy một mention có liều
chỉ với tới được **concept thành phần không có dạng bào chế**: `clonazepam 0.5 mg po qam:prn` →
SCDC **315699** thay vì SCD **197527** ("clonazepam 0.5 mg oral tablet"). Cùng *hình dạng* lỗi với
ICD ở mục 2.5: mã đúng nằm trong từ điển, nhưng ta trả về **sai cấp độ**.

**Cách sửa**:
- `strip_administration_noise()` tách mention thành `core` (hoạt chất + liều) và `hints` (dạng bào
  chế suy ra từ đường dùng/tần suất, cả Anh lẫn Việt, kể cả cụm trong ngoặc và ký hiệu `qam:prn`).
- `resolve_complete_product()` coi `core` là **tiền tố** của tên sản phẩm chính thức, ưu tiên phần
  hoàn chỉnh khớp dạng bào chế → sản phẩm đủ (SCD/SBD) → tên ngắn nhất. Dùng *tiền tố* thay vì
  *bằng nhau* chính là cách một hoạt chất đơn với tới được thuốc **phối hợp** — đúng cơ chế lấy 360047.
- `drugs.csv` đang trả lời **mù liều**: tầng containment khớp theo hoạt chất nên trả **cùng mã 197527
  cho cả hai liều clonazepam** — đúng điều đề bài cấm. Nay `link_candidates` dùng cùng cấu trúc 3
  tầng như chẩn đoán, nhưng đẩy tầng containment xuống **sau** RxNorm.

**Kết quả**: ví dụ BTC **1/3 → 2/3** (ca còn lại cần suy luận liều 1.5mg → viên 1mg, cố ý không làm);
holdout leakage-free `J` 0.7500 → **0.7778**; đồng thuận trên 109 text thuốc 0.550 → **0.606**; toàn
bộ 100 file với bảng đầy đủ **không đổi** (THUỐC 1.0000).

> ⚠️ **Đây là một canh bạc có chủ đích**: đồng thuận với `output/` ở nhóm mention **có liều** vẫn thấp
> (0.107) — **cố ý**, vì `output/` gán 74% mã thuốc ở mức hoạt chất (IN/BN), điều đề bài cấm. Bằng
> chứng hậu thuẫn: `output/` chỉ đạt `J_candidates` **29.98** thật ⇒ ~70% mã của nó sai, nên "giống
> `output/`" không phải bằng chứng đúng. Nếu một lần nộp thật làm `J_candidates` **giảm**, giả thuyết
> bị bác bỏ ⇒ tắt bằng `--no-dose-form-promotion`, đừng đi tinh chỉnh tiếp.

---

### 2.6. Các luồng đã bỏ

| Luồng | Vị trí | Trạng thái |
| :--- | :--- | :--- |
| **Rule-based / wordlist** | `legacy/scripts/` (`generate_outputs.py`…) | **Deprecated, chỉ để audit.** Wordlist chép verbatim từ 100 file public → overfit tuyệt đối, vi phạm tiêu chí "suy luận thật"; heuristic assertion đã bị tắt cứng. **Không dùng để sinh bài nộp.** |
| **Gemini Cloud API** (`scripts/run_llm_analysis.py`) | — | **Đã gỡ khỏi repo ngày 25/07/2026** (cùng `.env`, `output_llm_turn2/`). Gọi API đám mây lúc inference vi phạm luật thi; vai trò data-prep của nó đã được thay bằng Qwen self-host trên Kaggle (mục 2.2). Nêu ở đây chỉ để ghi nhận lịch sử — **script này không còn tồn tại trong repo**. |

---

## 3. So sánh các phương pháp

| Tiêu chí | PP1 (XLM-R + linking) | PP2 (Distill Qwen2.5-7B) | PP3 (Augmentation) | PP5 (ICD-vi linking) | Legacy rule-based |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Đúng luật BTC** | ✅ Đường nộp | ✅ Chỉ data-prep | ✅ | ✅ | ❌ Rủi ro bị loại |
| **Tổng quát hóa** | Cao | Cao (học domain turn-2) | Cao (mở rộng tên thuốc) | Cao (từ điển chính thức, độc lập với `output/`) | ❌ Rất thấp |
| **Chi phí tính toán** | Trung bình (270–560M) | Cao (cần GPU Kaggle) | Thấp (stdlib) | Rất thấp (tra bảng) | Rất thấp |
| **Ảnh hưởng `J_candidates`** | Nền tảng | Bổ sung mã cho text mới | Gián tiếp | **+0.15 (đo leakage-free)** | Thấp |
| **Phụ thuộc mạng** | Offline 100% | Cần mạng khi chạy Kaggle | Offline 100% | Offline 100% | Offline 100% |

---

## 4. Trạng thái điểm số thực tế

### 4.1. Điểm leaderboard đã ghi nhận

| Mốc | Điểm thật | WER | J_assertion | J_candidates | Ghi chú |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Turn 1 — `output/` hand-tuned (Run 8) | **41.591** | 51.66 | 50.32 | 29.98 | Tốt nhất turn-1, nhưng là bản chỉnh tay |
| Turn 1 — model v12 (suy luận thật) | 40.828 | 53.38 | 49.18 | 30.22 | Cách bản chỉnh tay 0.76 điểm |
| Turn 1 — model v13 | 40.589 | 52.73 | 48.96 | 29.30 | Regression; holdout 15 file quá nhỏ, phương sai cao |
| Turn 1 — model đầu (`MAX_LENGTH=320`) | 35.671 | 59.14 | 41.89 | 27.12 | Trước khi vá truncation |
| **Turn 2 — baseline `run_all.py submit`** | **34.388** | 64.28 | 39.56 | 29.51 | |
| Turn 2 — notebook all-Kaggle (bản rút gọn) | 31.894 | 65.03 | 38.53 | 24.61 | Regression: cell inline bỏ mất 3 flag + RxNorm fallback |
| **Turn 2 — sau distillation Qwen** | **36.320** | 61.66 | 43.59 | 29.34 | text +2.6, assertion +4.0, **candidates đứng yên** |

**Kết luận từ bảng này**: `J_candidates` đứng yên quanh 29–30 qua mọi lần nộp, kể cả khi model tốt lên
rõ rệt — vì nó bị chặn bởi **bảng tra**, không phải bởi model. Đó chính là lý do phương pháp 5 ra đời.

### 4.2. Đo hiệu quả của ICD-vi linking (28/07)

Cách đo **loại bỏ rò rỉ dữ liệu**: dựng lại `diagnoses.csv` **chỉ từ 85 file train**, rồi chấm trên
66 mention `CHẨN_ĐOÁN` ground-truth của 15 file holdout (giả định NER hoàn hảo, để chỉ đo phần
linking):

| Cấu hình | J_candidates | Tỉ lệ trả rỗng |
| :--- | ---: | ---: |
| Chỉ bảng curated (pipeline cũ) | 0.5909 | 0.348 |
| **+ ICD-vi fallback** | **0.7424** | 0.167 |
| Toàn bộ 100 file với bảng curated đầy đủ | 0.9852 (không đổi) | 0.000 |

Dòng cuối quan trọng: trên văn bản **đã biết**, thay đổi này **không đổi gì** — nó chỉ kích hoạt ở chỗ
bảng curated không có câu trả lời, nên không thể gây regression.

Phép đo độc lập thứ hai: chỉ riêng ICD-vi index đoán đúng top-1 cho **36.2%** trong 229 text chẩn đoán
phân biệt của corpus (danh mục BYT không dẫn xuất từ `output/` nên không rò rỉ theo chiều nào).

Chạy end-to-end với model thật trên 15 file holdout (1 lượt inference, nhiều cấu hình linking):
`final_score` 0.6848 → **0.7855** khi bật fallback với bảng curated train-only. **Lưu ý**: model đang
deploy đã được train trên nhãn turn-1 *bao gồm* 15 file này, nên chất lượng span/assertion ở đây là
ghi nhớ chứ không phải tổng quát hóa — chỉ nên đọc phần chênh lệch `J_candidates`.

**A/B trên chính tập turn-2** (chạy đủ recipe `submit` hai lần, chỉ khác `--no-icd-fallback`):
2898 → 3010 thực thể, **112 chẩn đoán được gán mã và nhờ đó không bị xóa**, không mất cái nào. Nhưng
soi 112 mention đó thì khoảng 1/3 là **mảnh vụn 1 chữ** của model mà bộ lọc no-candidate trước đây
đang xóa giúp ("vàng" → A95.9 sốt vàng, "tính" → A80.9 bại liệt). Phép đo holdout **không thể** phát
hiện lỗi này vì nó nạp text ground-truth vào bộ linking, span nhiễu không bao giờ tới. Đó là lý do
có ba lớp chặn ở mục 2.5 — chúng loại 34/112 mention rác **mà không mất gì** trên cả hai phép đo
turn-1. **Bài học**: một phép đo linking với NER hoàn hảo chỉ đo trần lý thuyết, không đo pipeline
thật; luôn A/B thay đổi linking trên một lượt inference thật trước khi chốt.

### 4.3. Bài học đã rút (từ `worklog.md`)

- **Đừng tin proxy 100 file** so với `output/`: 85 file nằm trong tập train nên số bị bơm phồng. Ưu
  tiên chỉ số holdout 15 file (dù nó cũng chỉ đo mức đồng thuận với `output/`).
- Các gap còn lại (span thiếu chữ, false positive, phân mảnh subword) là **vấn đề chất lượng
  model/dữ liệu**, không vá được bằng hậu xử lý: đã thử merge theo `"và"`/whitespace và **revert** vì
  gây false merge thật (gộp nhầm "guaifenesin" với "furosemide 40 mg").
- **Không khôi phục được weights của một version Kaggle cũ**: `kaggle kernels output <slug>/12` bỏ qua
  hậu tố version và luôn trả về run mới nhất (đã xác minh bằng MD5). ⇒ **Snapshot `models/ner_model/`
  trước khi ghi đè bằng export mới.**
- Luôn push kernel với `--accelerator NvidiaTeslaT4`: P100 (sm_60) làm crash wheel torch của Kaggle
  giữa epoch 1.

---

## 5. Cây thư mục

```
ViettelRace/
├── input/                        100 bệnh án turn 1 (.txt)
├── input_turn2/                  100 bệnh án turn 2 của BTC (gitignore)
├── output/                       nhãn public turn 1 = bản hand-tuned 41.591; dùng làm DỮ LIỆU TRAIN,
│                                 không bao giờ bị pipeline ghi đè
├── output_model*/                kết quả do run_pipeline.py sinh (gitignore)
├── data/
│   ├── ner_dataset/              train.jsonl (85) · holdout.jsonl (15) · train_augmented.jsonl (200)
│   │                             · split.json (seed 13)
│   └── terminology/              drugs.csv (159) · diagnoses.csv (321) · conflicts.txt
│                                 · rxnorm_full.csv (517,991) · rxnorm_drug_names.csv (11,201)
│                                 · icd10_full.csv (11,243, tiếng Anh — dùng để VALIDATE mã LLM)
│                                 · icd10_vi.csv (15,144, tiếng Việt — dùng để LINKING)
│                                 · raw/ (nguồn thô, gitignore)
├── models/ner_model/             weights fine-tuned (gitignore): model.pt · config.json ·
│                                 hf_config.json · tokenizer*
├── notebooks/                    train_ner_assertion_model.ipynb · qwen_extract.ipynb
├── kaggle_upload/                dataset/ · kernel/ · qwen_kernel/  (giữ đồng bộ tay với notebooks/)
├── scripts/
│   ├── prepare_ner_dataset.py    output/ → train/holdout jsonl (85/15, seed 13, --folds N)
│   ├── augment_ner_dataset.py    sinh dữ liệu tổng hợp
│   ├── build_terminology_index.py  mined drugs.csv / diagnoses.csv + TerminologyMatcher
│   ├── build_rxnorm_rrf_index.py   RRF → rxnorm_full.csv + RxNormOfflineIndex
│   ├── build_icd10_vi_index.py     danh mục BYT → icd10_vi.csv + ICD10VietnameseIndex
│   ├── fetch_icd.py / fetch_rxnorm.py  công cụ crawl một lần (KHÔNG nằm trong đường inference)
│   ├── run_pipeline.py           suy luận chính thức (module duy nhất cần torch)
│   ├── check_submission.py       validator + mô phỏng điểm type-aware
│   ├── package_submission.py     output.zip
│   ├── package_source.py         source_bundle.zip cho BTC
│   └── run_all.py                entrypoint (prepare/train/infer/package/submit/all)
├── legacy/                       pipeline cũ rule-based — chỉ audit
├── docs/                         problem_statement.md · repo_report.md · score_history.md · báo cáo này
├── worklog.md                    nhật ký kỹ thuật theo ngày (nguồn cập nhật nhất)
└── CLAUDE.md                     định hướng cho Claude Code
```

---

## 6. Đính chính so với bản lần 1 (cùng ngày 28/07)

Bản trước của báo cáo này có các lỗi sau, đã sửa ở trên:

1. **"Phương pháp 6: Gemini API (`scripts/run_llm_analysis.py`)"** — script này **đã bị gỡ khỏi repo
   từ 25/07/2026**, không còn tồn tại. Đã chuyển thành mục lịch sử 2.6.
2. **Thiếu toàn bộ lịch sử điểm turn 2** (34.388 → 31.894 → 36.320) và thiếu nhận định quan trọng
   nhất rút ra từ nó: `J_candidates` bị chặn bởi bảng tra chứ không phải bởi model.
3. **Thiếu `build_icd10_vi_index.py` / `icd10_vi.csv`** — thời điểm đó code đã có trên đĩa nhưng chưa
   được import ở đâu cả.
4. **`run_all.py all`** không có stage `package` riêng: chuỗi là `prepare → train → infer`, và `infer`
   tự đóng gói.
5. Sai số liệu: `rxnorm_full.csv` là **517,991** dòng (không phải "~512,000");
   `train_augmented.jsonl` hiện là **200** dòng (không phải 315 — con số của cấu hình augmentation
   cũ); `rxnorm_drug_names.csv` là **11,201** tên; `icd10_vi.csv` là **15,144** dòng.
6. Tài khoản Kaggle đang dùng là **`quanganh1008`** (`quanganh1008/viettelrace-ner-dataset`,
   `quanganh1008/viettelrace-ner-assertion-train`).
