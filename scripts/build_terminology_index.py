#!/usr/bin/env python3
"""Build an offline drug/diagnosis terminology index for entity linking.

Why this exists: the old candidate-matching code (generate_outputs.py's
RXNORM dict, improve_from_baseline.py's DRUG_CANDIDATES, auto_improve_agent's
phrase_rules) maps by *drug/diagnosis name only*. The contest's own example
proves ground truth needs dose-specific RxNorm concepts -- "clonazepam 0.5 mg
po qam:prn" -> 197527 vs "clonazepam 1.5 mg po qhs" -> 197528, same
ingredient, different code. A name-only table can never resolve that; it
needs a real (text -> concept) lookup keyed on the fuller mention.

This script does not call any external API. It builds the index from data
already present in this repo:
  1. Every (text, type) -> candidates pair already assigned in the curated
     output/*.json (the strongest signal: it survived several scored
     leaderboard rounds).
  2. The legacy hardcoded dicts in legacy/scripts/*.py, kept as a fallback
     layer for names not seen in (1).
It also reports conflicts (same normalized text mapped to different codes
across files) so they can be reviewed by hand.

NOTE on scaling this properly: RxNorm's full concept table (RRF files) and
ICD-10-CM code lists are the correct long-term source, but RxNorm full
downloads require a free UMLS Metathesaurus license/login -- not something
this script can fetch automatically. Once you have those files, load them
into data/terminology/{drugs,diagnoses}.csv in the same 3-column format
this script writes and the matcher below will use them unchanged.

Usage:
  python scripts/build_terminology_index.py
  python scripts/build_terminology_index.py --lookup "clonazepam 0.5 mg po qam:prn"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
TERM_DIR = ROOT / "data" / "terminology"

DIAG = "CHẨN_ĐOÁN"
DRUG = "THUỐC"
CANDIDATE_TYPES = {DIAG, DRUG}

# Fallback layer: name-level (ingredient/diagnosis) codes salvaged from the
# legacy rule scripts. Used only when no exact/fuzzy match exists from the
# curated output/ data, since these are known to be too coarse for dosed
# drugs (see module docstring).
LEGACY_DRUG_NAMES: dict[str, str] = {
    "acetaminophen": "313782", "amlodipine": "308135", "aspirin": "243670",
    "atenolol": "1202", "clonazepam": "197527", "docusate sodium": "1099279",
    "doxycycline": "3640", "furosemide": "4603", "guaifenesin": "392085",
    "heparin": "5224", "ibuprofen": "5640", "insulin": "5856",
    "lisinopril": "29046", "metformin": "6809", "metoprolol": "6918",
    "morphine": "7052", "nitroglycerin": "4917", "nystatin": "7597",
    "omeprazole": "7646", "pravastatin": "42463", "prednisone": "8640",
    "senna": "312935", "warfarin": "11289", "atorvastatin": "83367",
    "simvastatin": "36567", "ceftriaxone": "2193", "vancomycin": "11124",
    "albuterol": "435", "levothyroxine": "10582", "gabapentin": "25480",
    "hydrochlorothiazide": "5487", "losartan": "52175", "pantoprazole": "40790",
    "clonidine": "2599", "suboxone": "1544857", "ipratropium": "7213",
    "bactrim": "151399", "cefepime": "20481", "cefepim": "20481",
    "zosyn": "203167", "amoxicillin": "723", "ceftazidime": "2191",
    "toradol": "35827", "ketorolac": "35827", "lorazepam": "6470",
    "dextrose": "4850", "insulin glargine": "274783", "mucinex d": "352051",
    "tylenol": "313782", "motrin": "5640", "advil": "5640", "lasix": "4603",
    "coumadin": "11289", "plavix": "32968", "clopidogrel": "32968",
    "klonopin": "197527", "yếu tố ix": "1424945",
}

LEGACY_DIAG_PHRASES: dict[str, str] = {
    "u ác trực tràng": "C20",
    "u ác của đại tràng": "C18.9",
    "ung thư phổi không tế bào nhỏ": "C34.9",
    "ung thư phổi": "C34.9",
    "ung thư vú": "C50.9",
    "ung thư biểu mô tế bào thận": "C64.9",
    "ung thư đường mật": "C24.0",
    "ung thư biểu mô tế bào mật": "C24.0",
    "viêm loét đại tràng": "K51.9",
    "tăng sản tuyến tiền liệt": "N40.0",
    "liệt hai chi dưới": "G82.2",
}


ADDITIONAL_DRUG_ALIASES: dict[str, str] = {
    # General brand/generic aliases observed in Vietnamese clinical notes.
    # These are term-level aliases, not file-id rules; they let the same
    # reproducible pipeline link unseen notes without relying only on the
    # public output-derived table.
    "acetylcysteine": "197",
    "nac": "197",
    "alverin": "17627",
    "alverine": "17627",
    "alverin citrate": "17627",
    "alverin citrate 40 mg": "17627",
    "amoxicillin clavulanate": "19711",
    "amoxicillin / clavulanate": "19711",
    "augmentin": "19711",
    "bacillus clausii": "2600050",
    "biogermin": "2600050",
    "enterogermina": "2600050",
    "berlthyrox": "10582",
    "carvedilol": "20352",
    "corticoid": "8640",
    "corticosteroid": "8640",
    "furosemid": "4603",
    "furosemid 40 mg": "315971",
    "glargine": "274783",
    "insulin glargine": "274783",
    "metoclopramide": "6915",
    "nitralmyl": "4917",
    "nitramyl": "4917",
    "omez": "7646",
    "omez 20 mg": "316408",
    "pimperan": "6915",
    "rosuvastatin": "301542",
    "simenic": "9796",
    "simethicon": "9796",
    "simethicon 100 mg": "9796",
    "simethicone": "9796",
    "trimetazidin": "10826",
    "trimetazidine": "10826",
    "vastarel": "10826",
    "vitamin b12": "476630",
}


ADDITIONAL_DIAG_PHRASES: dict[str, str] = {
    # Vietnamese/English diagnosis aliases backed by data/terminology/icd10_full.csv.
    "amyloidosis": "E85.9",
    "bệnh amyloidosis": "E85.9",
    "bệnh thoái hóa tinh bột": "E85.9",
    "rối loạn chuyển hóa tinh bột": "E85.9",
    "amyloidosis tự miễn dịch": "E85.9",
    "amyloidosis di truyền": "E85.2",
    "amyloidosis gia đình": "E85.2",
    "áp xe phổi": "J85.2",
    "bàn chân bẹt": "M21.4",
    "bệnh bàn chân bẹt": "M21.4",
    "bệnh dạ dày": "K31.9",
    "bệnh dại": "A82.9",
    "dại": "A82.9",
    "lyssavirus": "A82.9",
    "virus dại": "A82.9",
    "vi-rút dại": "A82.9",
    "bệnh mạch vành": "I25.9",
    "động mạch vành": "I25.9",
    "hẹp tắc mạch vành": "I25.9",
    "bệnh kawasaki": "M30.3",
    "kawasaki": "M30.3",
    "biến chứng mạch vành": "I25.9",
    "biến chứng động mạch vành": "I25.9",
    "béo phì": "E66.9",
    "béo phì do có nhiều nếp gấp da": "E66.9",
    "dị tật thiểu sản vành tai": "Q17.2",
    "giãn thừng tinh": "I86.1",
    "gout": "M10.9",
    "gút": "M10.9",
    "gout mạn tính": "M10.9",
    "hội chứng thận hư": "N04",
    "mày đay": "L50.9",
    "mày đay vô căn": "L50.1",
    "mày đay vô": "L50.1",
    "đay": "L50.9",
    "đay vô": "L50.1",
    "mụn": "L70.9",
    "mụn cám": "L70.0",
    "mụn đầu trắng": "L70.0",
    "mụn trứng cá": "L70.0",
    "trứng cá": "L70.0",
    "nấm bẹn": "B35.6",
    "nhiễm nấm da": "B35.9",
    "nhiễm trùng máu": "A41.9",
    "nhiễm khuẩn": "A49.9",
    "nhiễm vi khuẩn": "A49.9",
    "nhiễm virus": "B34.9",
    "nhiễm khuẩn, nhiễm virus": "B34.9",
    "nhồi máu": "I21.9",
    "phù phổi cấp": "J81",
    "rối loạn đông máu": "D68.9",
    "thrombophilia": "D68.5",
    "sâu răng": "K02.9",
    "loãng xương": "M81.9",
    "suy các cơ quan": "R65",
    "tăng áp lực động mạch phổi": "I27.2",
    "tăng bilirubin máu": "E80.6",
    "tăng men gan": "R74.0",
    "tan huyết": "D59.9",
    "thiếu máu do tan huyết": "D55.0",
    "thiếu máu tan huyết": "D59.9",
    "thiếu men": "D55.0",
    "thiếu men g6pd": "D55.0",
    "thiếu hụt g6pd": "D55.0",
    "glucose-6-phosphate dehydrogenase": "D55.0",
    "thuyên tắc phổi": "I26.9",
    "thuyên tắc phổi hai bên": "I26.9",
    "tiểu đường": "E14",
    "tiền sản giật": "O14.9",
    "nguy cơ tiền sản giật cao": "O14.9",
    "trào ngược dạ dày thực quản": "K21.9",
    "u nang tuyến vú": "N60.0",
    "u xơ tuyến vú": "N60.2",
    "ung thư": "C80.9",
    "ung thư cổ tử cung": "C53.9",
    "ung thư máu": "C95.9",
    "đa u tủy": "C90.0",
    "viêm bao tử": "K29.7",
    "viêm dạ dày": "K29.7",
    "viêm hang vị sung huyết": "K29.6",
    "viêm sung huyết hang vị dạ dày": "K29.6",
    "ổ loét trong bao tử": "K25",
    "viêm da tiếp xúc": "L25.9",
    "viêm gan virus": "B19.9",
    "viêm gan cấp tính do virus b thể": "B16.9",
    "viêm gan virus b": "B16.9",
    "viêm kết mạc": "H10.9",
    "viêm khớp dạng thấp": "M06.9",
    "viêm nha chu": "K05.6",
    "viêm phổi hoại tử": "J85.1",
    "viêm tim": "I51.4",
}


def replace_existing(tmp: Path, path: Path) -> None:
    try:
        os.replace(tmp, path)
        return
    except PermissionError:
        if path.exists():
            try:
                path.chmod(0o666)
            except OSError:
                pass
            path.unlink()
        try:
            os.replace(tmp, path)
        except PermissionError:
            path.write_bytes(tmp.read_bytes())
            tmp.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.{os.getpid()}.tmp"
    tmp.write_text(text, encoding="utf-8")
    replace_existing(tmp, path)


def norm(text: str) -> str:
    """Normalize a mention before using it as a lookup key. Two extra steps
    beyond whitespace/case, both purely cosmetic (never touch the digits
    themselves, so 0.5mg and 1.5mg stay distinct):
    - Unicode NFC: Vietnamese text can arrive pre-composed ("ệ" as one
      codepoint) or decomposed ("e" + combining marks) depending on its
      authoring tool; two visually-identical strings in different forms
      compare unequal under a plain dict/set lookup. output/'s own texts are
      already all-NFC (confirmed empirically), but a private test set's
      input files are an unknown authoring source -- normalize defensively.
    - Digit/letter spacing ("500mg" -> "500 mg"): a common free-text
      formatting inconsistency; without it, the same drug written with vs.
      without a space around its dose unit fails both the exact and (nearly)
      the fuzzy match.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(\d)([a-zA-ZÀ-ỹ])", r"\1 \2", text)
    text = re.sub(r"([a-zA-ZÀ-ỹ])(\d)", r"\1 \2", text)
    return re.sub(r"\s+", " ", text.strip().lower())


def collect_from_output() -> tuple[dict[str, dict[str, Counter[tuple[str, ...]]]], dict[str, set[str]]]:
    """(type -> normalized text -> Counter of candidate tuples seen), plus a
    separate (type -> set of normalized texts) for mentions that DO appear in
    output/ but were deliberately given no candidates. That second set
    matters for anything that treats "no local match" as license to try a
    live fallback lookup (see run_pipeline.py's RxNormOfflineFallback): a text seen here
    with candidates=[] is a known, reviewed "no code" answer, not a gap --
    falling back for it would overwrite a deliberate decision with a live
    API's best guess, which is a real regression (confirmed empirically: it
    dropped the local J_candidates proxy). Only a text absent from *both*
    sets is a genuine "never seen this before" case worth a live lookup.
    """
    seen: dict[str, dict[str, Counter[tuple[str, ...]]]] = {DIAG: defaultdict(Counter), DRUG: defaultdict(Counter)}
    no_candidate: dict[str, set[str]] = {DIAG: set(), DRUG: set()}
    for path in sorted(OUTPUT_DIR.glob("*.json")):
        try:
            entities: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for ent in entities:
            typ = ent.get("type")
            if typ not in CANDIDATE_TYPES:
                continue
            candidates = tuple(str(c) for c in (ent.get("candidates") or []))
            text_norm = norm(str(ent.get("text", "")))
            if not candidates:
                no_candidate[typ].add(text_norm)
                continue
            seen[typ][text_norm][candidates] += 1
    return seen, no_candidate


def write_lines(path: Path, lines: set[str]) -> None:
    write_text(path, "\n".join(sorted(lines)))


def write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.{os.getpid()}.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "candidate", "source"])
        writer.writerows(rows)
    replace_existing(tmp, path)


def build() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[str], dict[str, set[str]]]:
    seen, no_candidate = collect_from_output()
    conflicts: list[str] = []

    drug_rows: list[tuple[str, str, str]] = []
    for text, choices in seen[DRUG].items():
        if len(choices) > 1:
            conflicts.append(f"{DRUG} {text!r}: {dict(choices)}")
        # keep the most frequent candidate set observed
        best = max(choices.items(), key=lambda kv: kv[1])[0]
        for code in best:
            drug_rows.append((text, code, "output_curated"))
    known_drug_texts = {t for t, *_ in drug_rows}
    for name, code in LEGACY_DRUG_NAMES.items():
        if name not in known_drug_texts:
            drug_rows.append((name, code, "legacy_name_level"))
            known_drug_texts.add(name)
    for name, code in ADDITIONAL_DRUG_ALIASES.items():
        key = norm(name)
        if key not in known_drug_texts:
            drug_rows.append((key, code, "rxnorm_alias"))
            known_drug_texts.add(key)

    diag_rows: list[tuple[str, str, str]] = []
    for text, choices in seen[DIAG].items():
        if len(choices) > 1:
            conflicts.append(f"{DIAG} {text!r}: {dict(choices)}")
        best = max(choices.items(), key=lambda kv: kv[1])[0]
        for code in best:
            diag_rows.append((text, code, "output_curated"))
    known_diag_texts = {t for t, *_ in diag_rows}
    for phrase, code in LEGACY_DIAG_PHRASES.items():
        key = norm(phrase)
        if key not in known_diag_texts:
            diag_rows.append((key, code, "legacy_phrase_level"))
            known_diag_texts.add(key)
    for phrase, code in ADDITIONAL_DIAG_PHRASES.items():
        key = norm(phrase)
        if key not in known_diag_texts:
            diag_rows.append((key, code, "icd10_alias"))
            known_diag_texts.add(key)

    return drug_rows, diag_rows, conflicts, no_candidate


class TerminologyMatcher:
    """Exact-normalized-text lookup with a fuzzy fallback (difflib).

    Deliberately dependency-free so it runs identically on Kaggle/Colab and
    locally. Swap in an embedding-based matcher later (e.g. a bi-encoder
    fine-tuned on this same CSV) without changing the CSV format or callers.
    """

    def __init__(self, csv_path: Path):
        self.exact: dict[str, list[str]] = defaultdict(list)
        self.texts: list[str] = []
        self.containment_texts: list[str] = []
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = norm(row["text"])
                    if row["candidate"] not in self.exact[key]:
                        self.exact[key].append(row["candidate"])
            self.texts = list(self.exact)
            self.containment_texts = sorted(
                [t for t in self.texts if len(t) >= 5],
                key=lambda t: (-len(t), t),
            )

    def lookup_exact(self, text: str, containment: bool = True) -> list[str]:
        """Only the high-precision half of `lookup`: exact normalized match, then
        (optionally) whole-word containment of a known term. No difflib fuzzy
        fallback.

        Callers that have a second, authoritative index to consult (e.g.
        run_pipeline.py's ICD10VietnameseIndex for CHẨN_ĐOÁN) use this to order
        the two: a curated hit is trusted, but a *fuzzy* curated hit is just the
        nearest of the few hundred mined turn-1 strings and is worse than an
        official-vocabulary match, so it belongs after it, not before.

        `containment=False` additionally rejects the substring tier, which is
        actively wrong for THUỐC: "clonazepam 1.5 mg po qhs" contains the known
        term "clonazepam 0.5 mg po qam:prn"'s ingredient, so containment happily
        returns the 0.5mg code for a 1.5mg mention -- precisely the dose-blind
        mapping the task statement forbids (197527 vs 197528).
        """
        key = norm(text)
        if key in self.exact:
            return self.exact[key]
        if containment:
            for candidate_key in self.containment_texts:
                if re.search(rf"(?<!\w){re.escape(candidate_key)}(?!\w)", key):
                    return self.exact[candidate_key]
        return []

    def lookup(self, text: str, fuzzy_threshold: float = 0.86) -> list[str]:
        key = norm(text)
        exact_hit = self.lookup_exact(text)
        if exact_hit:
            return exact_hit
        best_score, best_key = 0.0, None
        for candidate_key in self.texts:
            score = SequenceMatcher(None, key, candidate_key).ratio()
            if score > best_score:
                best_score, best_key = score, candidate_key
        if best_key is not None and best_score >= fuzzy_threshold:
            return self.exact[best_key]
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", help="Print candidates for a given drug/diagnosis text and exit.")
    args = parser.parse_args()

    if args.lookup:
        drug_matcher = TerminologyMatcher(TERM_DIR / "drugs.csv")
        diag_matcher = TerminologyMatcher(TERM_DIR / "diagnoses.csv")
        print("drug match:", drug_matcher.lookup(args.lookup))
        print("diagnosis match:", diag_matcher.lookup(args.lookup))
        return 0

    drug_rows, diag_rows, conflicts, no_candidate = build()
    write_csv(TERM_DIR / "drugs.csv", sorted(set(drug_rows)))
    write_csv(TERM_DIR / "diagnoses.csv", sorted(set(diag_rows)))
    write_text(TERM_DIR / "conflicts.txt", "\n".join(conflicts))
    write_lines(TERM_DIR / "drugs_no_candidate.txt", no_candidate[DRUG])
    write_lines(TERM_DIR / "diagnoses_no_candidate.txt", no_candidate[DIAG])

    print(f"drugs.csv: {len(set(drug_rows))} rows")
    print(f"diagnoses.csv: {len(set(diag_rows))} rows")
    print(f"conflicts: {len(conflicts)} (see data/terminology/conflicts.txt)")
    print(f"drugs_no_candidate.txt: {len(no_candidate[DRUG])} texts seen with a deliberate empty candidate list")
    print(f"diagnoses_no_candidate.txt: {len(no_candidate[DIAG])} texts seen with a deliberate empty candidate list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
