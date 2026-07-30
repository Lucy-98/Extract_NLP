#!/usr/bin/env python3
"""Clinical text normalizer and hybrid candidate linking engine for ViettelRace AI Race 2026.

Provides:
  1. VietnameseRxNormNormalizer: Maps Vietnamese clinical drug phrasing (spelling, routes, forms, doses, frequencies) to RxNorm terms.
  2. HybridCandidateLinker: High-precision candidate resolution (Curated Terminology -> Dose-aware RxNorm -> High-confidence ICD-10).
  3. ContextAssertionEngine: Sentence-bounded context window detector for isNegated, isHistorical, isFamily.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Canonical drug spelling map (Vietnamese clinical terms -> English RxNorm canonical names)
EXACT_DRUG_NAMES = {
    "aspirin", "amoxicillin", "ampicillin", "paracetamol", "acetaminophen",
    "clonazepam", "diazepam", "lorazepam", "alprazolam", "midazolam", "oxazepam",
    "nitrazepam", "ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin",
    "vancomycin", "gentamicin", "tobramycin", "amikacin", "clindamycin",
    "erythromycin", "azithromycin", "clarithromycin", "nystatin", "heparin",
    "insulin", "metformin", "nifedipine", "felodipine", "amlodipine",
}

DRUG_SPELLING_MAP = {
    "amoxicilin": "amoxicillin",
    "ampicilin": "ampicillin",
    "furosemid": "furosemide",
    "omeprazol": "omeprazole",
    "pantoprazol": "pantoprazole",
    "rabeprazol": "rabeprazole",
    "esomeprazol": "esomeprazole",
    "prednisolon": "prednisolone",
    "methylprednisolon": "methylprednisolone",
    "acetylcystein": "acetylcysteine",
    "doxycyclin": "doxycycline",
    "trimetazidin": "trimetazidine",
    "alverin": "alverine",
    "simethicon": "simethicone",
}

# Route & Form mapping (Vietnamese clinical shorthand -> RxNorm English equivalents)
ROUTE_FORM_MAP = [
    (r"\bviên nén\b", "Tablet"),
    (r"\bviên bao phim\b", "Oral Tablet"),
    (r"\bviên nang\b", "Capsule"),
    (r"\bviên\b", "Tablet"),
    (r"\bdung dịch xịt\b", "Spray"),
    (r"\bdung dịch tiêm\b", "Injectable Solution"),
    (r"\bdung dịch uống\b", "Oral Solution"),
    (r"\bdung dịch\b", "Solution"),
    (r"\bthuốc xịt\b", "Spray"),
    (r"\bxịt\b", "Spray"),
    (r"\bthuốc tiêm\b", "Injectable"),
    (r"\btiêm tĩnh mạch\b", "Intravenous"),
    (r"\btiêm bắp\b", "Intramuscular"),
    (r"\btiêm\b", "Injectable"),
    (r"\bnhỏ mắt\b", "Ophthalmic Solution"),
    (r"\bnhỏ tai\b", "Otic Solution"),
    (r"\bnhỏ\b", "Drops"),
    (r"\bthuốc mỡ\b", "Ointment"),
    (r"\bkem\b", "Cream"),
    (r"\bbôi\b", "Topical"),
    (r"\buống\b", "Oral"),
    (r"\bpo\b", "Oral"),
    (r"\biv\b", "Intravenous"),
    (r"\bim\b", "Intramuscular"),
]

FREQUENCY_PATTERNS = [
    r"\b\d+\s*lần\s*/\s*ngày\b",
    r"\bngày\s*\d+\s*lần\b",
    r"\bmỗi\s*\d+\s*h(?:iờ)?\b",
    r"\bbid\b",
    r"\btid\b",
    r"\bqid\b",
    r"\b\d+\s*lần\b",
    r"\blần\s*x\s*\d+\b",
]

# Trigger words for Assertion Engine
NEGATION_TRIGGERS = [
    "không", "phủ nhận", "chưa ghi nhận", "không phát hiện", "không thấy",
    "không sốt", "bình thường", "không đau", "không ho", "âm tính", "(-)",
    "chưa phát hiện", "không ghi nhận", "không có", "loại trừ"
]

HISTORICAL_TRIGGERS = [
    "tiền sử", "trước đây", "đã từng", "đã phẫu thuật", "tiền căn",
    "thuốc trước nhập viện", "mãn tính", "cũ", "đã điều trị", "tiền sử bệnh",
    "đã mổ", "đã điều trị bằng", "tiền sử dùng", "tiền sử mắc"
]

FAMILY_TRIGGERS = [
    "bố", "mẹ", "ba", "cha", "ông", "bà", "anh", "chị", "em",
    "gia đình", "di truyền", "họ hàng", "người nhà"
]

# Precise Vietnamese diagnosis regex mappings -> ICD-10
VIETNAMESE_DIAGNOSIS_FALLBACKS = [
    (r"\b(?:đái tháo đường|tiểu đường|dthd)\b", "E11.9"),
    (r"\b(?:tăng huyết áp|tha|tang huyet ap)\b", "I10"),
    (r"\b(?:suy tim)\b", "I50.9"),
    (r"\b(?:viêm phổi|xẹp phổi)\b", "J18.9"),
    (r"\b(?:viêm phế quản|vpq)\b", "J20.9"),
    (r"\b(?:viêm dạ dày|viêm bao tử|loét dạ dày|viêm hang vị)\b", "K29.7"),
    (r"\b(?:suy thận)\b", "N18.9"),
    (r"\b(?:xơ gan)\b", "K74.6"),
    (r"\b(?:viêm gan|viêm gan b|viêm gan c)\b", "B19.9"),
    (r"\b(?:gút|gout)\b", "M10.9"),
    (r"\b(?:đột quỵ|tai biến|xuất huyết não|nhồi máu não)\b", "I64"),
    (r"\b(?:ung thư|u ác|khối u ác)\b", "C80.9"),
    (r"\b(?:sốt xuất huyết|dengue)\b", "A97.9"),
    (r"\b(?:viêm ruột thừa|vrt)\b", "K37"),
    (r"\b(?:sỏi thận|sỏi bàng quang|sỏi tiết niệu)\b", "N20.9"),
    (r"\b(?:viêm màng não)\b", "G03.9"),
    (r"\b(?:rối loạn tiền đình)\b", "H81.9"),
    (r"\b(?:thiếu máu|tan huyết)\b", "D64.9"),
    (r"\b(?:hen suyễn|hen phế quản)\b", "J45.9"),
    (r"\b(?:bệnh mạch vành|thiếu máu cơ tim)\b", "I25.9"),
]


class VietnameseRxNormNormalizer:
    """Translates Vietnamese clinical phrasing into standardized RxNorm format."""

    @staticmethod
    def clean_frequency(text: str) -> str:
        res = text.lower()
        for pat in FREQUENCY_PATTERNS:
            res = re.sub(pat, " ", res)
        return " ".join(res.split())

    @staticmethod
    def normalize_doses(text: str) -> str:
        res = text
        res = re.sub(r"(\d+(?:\.\d+)?)\s*mg\b", r"\1 MG", res, flags=re.IGNORECASE)
        res = re.sub(r"(\d+(?:\.\d+)?)\s*g\b", r"\1 GRAM", res, flags=re.IGNORECASE)
        res = re.sub(r"(\d+(?:\.\d+)?)\s*ml\b", r"\1 ML", res, flags=re.IGNORECASE)
        res = re.sub(r"(\d+(?:\.\d+)?)\s*mcg\b", r"\1 MCG", res, flags=re.IGNORECASE)
        res = re.sub(r"(\d+(?:\.\d+)?)\s*iu\b", r"\1 UNT", res, flags=re.IGNORECASE)
        res = re.sub(r"(\d+(?:\.\d+)?)\s*%\b", r"\1 %", res)
        return res

    @staticmethod
    def normalize_drug_spelling(text: str) -> str:
        words = text.split()
        out = []
        for w in words:
            wl = w.lower()
            if wl in EXACT_DRUG_NAMES:
                out.append(wl)
            elif wl in DRUG_SPELLING_MAP:
                out.append(DRUG_SPELLING_MAP[wl])
            else:
                out.append(w)
        return " ".join(out)

    @classmethod
    def normalize_drug_text(cls, text: str) -> str:
        clean = cls.clean_frequency(text)
        clean = cls.normalize_doses(clean)
        clean = cls.normalize_drug_spelling(clean)
        for pattern, english_term in ROUTE_FORM_MAP:
            clean = re.sub(pattern, f" {english_term} ", clean, flags=re.IGNORECASE)
        return " ".join(clean.split())


class ContextAssertionEngine:
    """Post-processing context scope detector for assertions bounded by sentence limits."""

    @staticmethod
    def extract_sentence_context(text: str, start: int, end: int, window_size: int = 80) -> tuple[str, str]:
        raw_left = text[max(0, start - window_size):start]
        raw_right = text[end:min(len(text), end + window_size)]

        # Stop at sentence boundaries
        left_parts = re.split(r"[\.\n;\?!]", raw_left)
        left_ctx = left_parts[-1].lower()

        right_parts = re.split(r"[\.\n;\?!]", raw_right)
        right_ctx = right_parts[0].lower()

        return left_ctx, right_ctx

    @classmethod
    def detect_assertions(
        self,
        text: str,
        start: int,
        end: int,
        model_assertions: list[str] | None = None,
    ) -> list[str]:
        assertions = set(model_assertions or [])
        left_ctx, right_ctx = self.extract_sentence_context(text, start, end)

        if any(trig in left_ctx for trig in NEGATION_TRIGGERS) or any(trig in right_ctx[:15] for trig in ["âm tính", "(-)"]):
            assertions.add("isNegated")

        if any(trig in left_ctx for trig in HISTORICAL_TRIGGERS):
            assertions.add("isHistorical")

        if any(trig in left_ctx for trig in FAMILY_TRIGGERS):
            assertions.add("isFamily")

        return sorted(assertions)


class HybridCandidateLinker:
    """High-precision candidate resolution engine."""

    def __init__(self, rxnorm_index=None, term_matcher_drug=None, term_matcher_diag=None):
        self.rxnorm_index = rxnorm_index
        self.term_matcher_drug = term_matcher_drug
        self.term_matcher_diag = term_matcher_diag

    def link_drug(self, entity_text: str) -> list[str]:
        # Layer 1: Curated drugs.csv exact & fuzzy matcher (Gold Ground-Truth index)
        if self.term_matcher_drug:
            hits = self.term_matcher_drug.lookup(entity_text)
            if hits:
                return hits

        # Layer 2: Vietnamese-to-RxNorm Translation + RxNorm Index Lookup
        norm_text = VietnameseRxNormNormalizer.normalize_drug_text(entity_text)
        if self.rxnorm_index:
            hits = self.rxnorm_index.lookup(norm_text)
            if hits:
                return hits

            hits = self.rxnorm_index.lookup(entity_text)
            if hits:
                return hits

        return []

    def link_diagnosis(self, entity_text: str) -> list[str]:
        if self.term_matcher_diag:
            hits = self.term_matcher_diag.lookup(entity_text)
            if hits:
                return hits

        clean_text = entity_text.lower()
        for pattern, icd_code in VIETNAMESE_DIAGNOSIS_FALLBACKS:
            if re.search(pattern, clean_text):
                return [icd_code]

        return []
