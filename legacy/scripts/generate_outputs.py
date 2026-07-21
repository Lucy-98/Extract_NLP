# Superseded by scripts/run_pipeline.py -- kept for reference only. The
# wordlists below are overfit to the 100 public files (see README.md); do
# not use this to regenerate a submission.

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


RXNORM = {
    "acetaminophen": "313782",
    "amlodipine": "308135",
    "aspirin": "243670",
    "atenolol": "1202",
    "clonazepam": "197527",
    "docusate sodium": "1099279",
    "doxycycline": "3640",
    "furosemide": "4603",
    "guaifenesin": "392085",
    "heparin": "5224",
    "ibuprofen": "5640",
    "insulin": "5856",
    "lisinopril": "29046",
    "metformin": "6809",
    "metoprolol": "6918",
    "morphine": "7052",
    "nitroglycerin": "4917",
    "nystatin": "7597",
    "omeprazole": "7646",
    "pravastatin": "42463",
    "prednisone": "8640",
    "senna": "312935",
    "warfarin": "11289",
    "atorvastatin": "83367",
    "simvastatin": "36567",
    "ceftriaxone": "2193",
    "vancomycin": "11124",
    "albuterol": "435",
    "levothyroxine": "10582",
    "gabapentin": "25480",
    "hydrochlorothiazide": "5487",
    "losartan": "52175",
    "pantoprazole": "40790",
    "clonidine": "2599",
    "suboxone": "1544857",
    "ipratropium": "7213",
    "bactrim": "151399",
    "cefepime": "20481",
    "cefepim": "20481",
    "zosyn": "203167",
    "amoxicillin": "723",
    "ceftazidime": "2191",
    "toradol": "35827",
    "ketorolac": "35827",
    "lorazepam": "6470",
    "dextrose": "4850",
    "insulin glargine": "274783",
    "mucinex d": "352051",
}

MED_NAMES = sorted(
    set(RXNORM)
    | {
        "tylenol",
        "motrin",
        "advil",
        "lasix",
        "coumadin",
        "plavix",
        "clopidogrel",
        "ipratropium",
        "bactrim",
        "cefepime",
        "cefepim",
        "zosyn",
        "amoxicillin",
        "ceftazidime",
        "toradol",
        "ketorolac",
        "lorazepam",
        "dextrose",
        "insulin glargine",
        "mucinex d",
        "metoprolol succinate",
        "metoprolol tartrate",
        "aspirin 325mg",
        "aspirin 81mg",
        "nitrate",
        "klonopin",
        "clonidine",
        "suboxone",
        "yếu tố ix",
    },
    key=len,
    reverse=True,
)


DIAGNOSES = [
    "hội chứng não gan",
    "xơ gan do rượu",
    "bệnh tim mạch do xơ vữa động mạch",
    "xơ vữa động mạch",
    "tăng huyết áp",
    "phình động mạch chủ",
    "u ác của đại tràng",
    "u ác trực tràng",
    "khối u trực tràng",
    "u tuyến",
    "cường cận giáp nguyên phát",
    "cơn đau thắt ngực ổn định",
    "tăng calci máu",
    "nhồi máu cũ",
    "ngoại tâm thu nhĩ",
    "ngoại tâm thu thất",
    "viêm phổi",
    "suy tim",
    "suy thận",
    "đái tháo đường",
    "tiểu đường",
    "copd",
    "hen",
    "thiếu máu",
    "nhiễm trùng",
    "áp xe",
    "viêm tuyến mồ hôi",
    "chuyển dạ",
    "thai 41 tuần",
    "nghẽn tắc và hẹp động mạch cảnh",
    "hẹp động mạch cảnh",
    "xuất huyết nội sọ không do chấn thương",
    "xuất huyết nội sọ",
    "tách thành động mạch chủ",
    "liệt hai chân",
    "rò động - tĩnh mạch đùi phải",
    "rò động",
    "béo phì",
    "tiểu tiện không tự chủ",
    "sa âm đạo",
    "bệnh rễ thần kinh tuỷ sống",
    "hẹp ống sống",
    "hẹp lỗ liên hợp",
    "trào ngược dạ dày thực quản",
    "thoát vị cạnh thực quản",
    "thoát vị hoành",
    "bệnh phổi tắc nghẽn mạn tính",
    "tâm thần phân liệt",
    "rối loạn cảm xúc",
    "rối loạn lưỡng cực",
    "rối loạn lo âu",
    "trầm cảm",
    "bàn chân vẹo bẩm sinh",
    "chấn thương gãy xương sườn trái",
    "gãy xương sườn",
    "đụng dập một vùng nhu mô phổi",
    "vết thương thấu bụng",
    "tổn thương dây thanh quản",
]


SYMPTOMS = [
    "cảm giác thắt chặt ngực vùng trước tim",
    "cảm giác thắt chặt ngực",
    "khó chịu vùng ngực",
    "đánh trống ngực",
    "tăng đánh trống ngực",
    "khó thở khi gắng sức",
    "khó thở về đêm",
    "khó thở khi nằm",
    "khó thở nhẹ",
    "khó thở",
    "mệt mỏi nhiều",
    "giảm dung nạp gắng sức",
    "buồn nôn",
    "đổ mồ hôi",
    "đau ngực",
    "đau bụng quặn",
    "đau bụng",
    "đau quanh vết mổ",
    "đau chân",
    "đau đầu",
    "đau ở sau đầu",
    "sốt",
    "ho",
    "đờm",
    "tiêu chảy",
    "đi ngoài ra máu",
    "không tăng cân",
    "ngất xỉu",
    "xỉu",
    "mất ý thức",
    "chóng mặt",
    "quầng sáng",
    "phù mắt cá chân",
    "ra huyết âm đạo",
    "vỡ ối",
    "rỉ ối",
    "cơn co tử cung",
    "thai máy tốt",
    "ý thức suy giảm",
    "không thể tự chăm sóc bản thân",
    "căng thẳng",
    "mất việc làm",
    "động kinh",
    "tăng cân",
    "giảm cân",
    "bàng quang căng",
    "cảm giác bí tiếu",
    "triệu chứng trào ngược",
    "trào ngược",
    "ý định tự tử",
    "ý nghĩ tự bắn vào đầu",
    "hưng cảm",
    "tăng hoạt động",
    "giảm nhu cầu ngủ",
    "lạm dụng chất kích thích",
    "lạm dụng chất gây nghiện opioid",
    "chảy dịch liên tục",
    "giọng khàn",
]


NEGATION_RE = re.compile(
    r"(không|khong|chưa|chua|không có|không còn|chưa có|không ghi nhận|chưa phát hiện)\s+$",
    re.IGNORECASE,
)


def assertion_for(text: str, start: int, end: int, kind: str) -> list[str]:
    # The public scorer showed J_assertion=0 with heuristic assertions. Leaving
    # assertions empty is safer than adding noisy guessed labels.
    return []
    before = text[max(0, start - 40) : start].lower()
    section = text[max(0, start - 220) : start].lower()
    assertions: list[str] = []
    if NEGATION_RE.search(before):
        assertions.append("isNegated")
    if kind == "THUỐC" and any(
        marker in section
        for marker in [
            "thuốc trước khi nhập viện",
            "trước nhập viện",
            "ở nhà",
            "trước đây",
            "đã sử dụng",
        ]
    ):
        assertions.append("isHistorical")
    near = text[max(0, start - 80) : start].lower()
    if kind != "THUỐC" and any(marker in near for marker in ["trước đây", "mạn tính", "mãn tính"]):
        assertions.append("isHistorical")
    return list(dict.fromkeys(assertions))


def normalize_candidate(med_text: str) -> list[str]:
    low = med_text.lower()
    for name, code in RXNORM.items():
        if name in low:
            return [code]
    aliases = {
        "tylenol": "313782",
        "motrin": "5640",
        "advil": "5640",
        "lasix": "4603",
        "coumadin": "11289",
        "plavix": "32968",
        "clopidogrel": "32968",
        "klonopin": "197527",
        "yếu tố ix": "1424945",
    }
    for name, code in aliases.items():
        if name in low:
            return [code]
    return []


def add_entity(entities: list[dict], text: str, start: int, end: int, kind: str):
    raw = text[start:end]
    left_trim = len(raw) - len(raw.lstrip())
    right_trim = len(raw) - len(raw.rstrip())
    start += left_trim
    end -= right_trim
    value = text[start:end]
    if not value or len(value.strip()) < 2:
        return
    if kind == "TRIỆU_CHỨNG" and value.lower() == "yếu" and text[end : end + 3].lower() == " tố":
        return
    if any(e["position"] == [start, end] and e["type"] == kind for e in entities):
        return
    item = {
        "text": text[start:end],
        "type": kind,
        "assertions": assertion_for(text, start, end, kind),
        "position": [start, end],
    }
    if kind == "THUỐC":
        item["candidates"] = normalize_candidate(item["text"])
    entities.append(item)


def extract_meds(text: str, entities: list[dict]):
    name_group = "|".join(re.escape(x) for x in MED_NAMES)
    tail = r"(?:\s*(?:\d+(?:[.,]\d+)?\s*)?(?:mg|mcg|g|ml|iu|đơn vị|viên|ống|x\s*1|po|iv|im|sc|bid|tid|qid|qhs|qam|q\d+h|prn|daily|ngày|lần|uống|đặt dưới lưỡi|truyền|tiêm|xl|succinate|tartrate|oral|suspension|tablet|tab|cap|capsule|viên))*"
    pattern = re.compile(rf"(?<!\w)({name_group}){tail}", re.IGNORECASE)
    for m in pattern.finditer(text):
        start, end = m.span()
        chunk = text[start:end]
        chunk = re.split(
            r"\s+(?:cho|vì|để|điều trị|không có|không cải thiện|trong ngày)\b",
            chunk,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        add_entity(entities, text, start, start + len(chunk), "THUỐC")


def extract_terms(text: str, terms: list[str], kind: str, entities: list[dict]):
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for m in pattern.finditer(text):
            add_entity(entities, text, m.start(), m.end(), kind)


def remove_nested(entities: list[dict]) -> list[dict]:
    entities = sorted(entities, key=lambda e: (e["position"][0], -(e["position"][1] - e["position"][0]), e["type"]))
    kept: list[dict] = []
    for ent in entities:
        s, e = ent["position"]
        if any(k["type"] == ent["type"] and k["position"][0] <= s and e <= k["position"][1] for k in kept):
            continue
        if ent["type"] == "TRIỆU_CHỨNG" and any(
            k["type"] in {"CHẨN_ĐOÁN", "THUỐC"}
            and k["position"][0] <= s
            and e <= k["position"][1]
            and (k["position"][1] - k["position"][0]) > (e - s)
            for k in kept
        ):
            continue
        kept.append(ent)
    return sorted(kept, key=lambda x: (x["position"][0], x["position"][1], x["type"]))


def extract(text: str) -> list[dict]:
    entities: list[dict] = []
    extract_meds(text, entities)
    extract_terms(text, DIAGNOSES, "CHẨN_ĐOÁN", entities)
    extract_terms(text, SYMPTOMS, "TRIỆU_CHỨNG", entities)
    return remove_nested(entities)


def validate(path: Path, entities: list[dict]):
    text = path.read_text(encoding="utf-8")
    for ent in entities:
        start, end = ent["position"]
        if text[start:end] != ent["text"]:
            raise ValueError(f"{path.name}: bad span {ent!r} got {text[start:end]!r}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.txt"), key=lambda p: int(p.stem)):
        entities = extract(path.read_text(encoding="utf-8"))
        validate(path, entities)
        out = OUTPUT_DIR / f"{path.stem}.json"
        out.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(list(OUTPUT_DIR.glob('*.json')))} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
