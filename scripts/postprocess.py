#!/usr/bin/env python3
"""Attribute-only post-processing for a prediction folder.

FALSIFIED 2026-08-01 -- do not ship these assertion levers. `v7_assert_union`
scored 33.644 against `v5_recoded`'s 34.303, with `text_score` and
`J_candidates` returning *identical to the digit* (36.057 / 29.9917) and
`J_assertion` falling 38.297 -> 36.101. The model's assertion head is better
calibrated than these rules: its isFamily rate (0.85%) matches turn-1 truth
(0.9%) almost exactly, and `--family-gate` cut it to 0.27%. See
docs/experiments.md. Kept for the record and for the `--hedge-icd` flag; if you
reach for `--sections` / `--negex` / `--consistency` again, you need new
evidence, not new thresholds.

Every transform here changes `assertions` or `candidates` on entities that are
*already* predicted. Nothing adds, removes, or re-spans an entity.

That restriction is not stylistic. The three metrics are Jaccard/WER over the
predicted entity set, so any change to that set moves the denominator of all
three at once. The 2026-07-28 experiment did exactly that -- the ICD fallback
rescued 78 `CHẨN_ĐOÁN` from the no-candidate filter -- and the real leaderboard
went 34.388 -> 33.679, with `J_candidates` *falling* 29.514 -> 28.676 even
though raising it was the whole point. Attribute-only changes have a much
smaller blast radius and can be A/B'd one at a time.

Each lever is a separate flag so a submission can isolate one of them:

  --sections      isHistorical / isFamily from document section scope + cues
  --negex         isNegated from Vietnamese negation cues with clause scope
  --consistency   majority-vote assertions across repeated mentions in a doc
  --family-gate   drop isFamily unless a family cue is nearby
  --hedge-icd     add the ".9" unspecified sibling as a 2nd ICD candidate

Usage:
  python scripts/postprocess.py --pred experiments/v1 --input input_turn2 \
      --out experiments/v2 --sections --consistency --family-gate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DIAG = "CHẨN_ĐOÁN"
SYM = "TRIỆU_CHỨNG"
DRUG = "THUỐC"
ASSERTABLE = {DIAG, SYM, DRUG}
CODED = {DIAG, "THUỐC"}

HISTORICAL = "isHistorical"
NEGATED = "isNegated"
FAMILY = "isFamily"


# --------------------------------------------------------------------------
# Section scope
# --------------------------------------------------------------------------

# Headers observed across input_turn2 (see worklog). Kept deliberately narrow:
# "tình trạng ngay trước khi nhập viện" is the acute presentation, not history,
# so it must NOT match here even though it contains "trước khi nhập viện".
HISTORICAL_HEADERS = (
    "tiền sử",
    "tiền căn",
    "bệnh sử",
    "thuốc trước khi nhập viện",
    "bệnh lý mãn tính",
    "bệnh lý mạn tính",
    "các sự kiện trước khi nhập viện",
    "thuốc đang dùng",
    "thuốc đang sử dụng",
)
FAMILY_HEADERS = ("tiền sử gia đình",)

HEADER_RE = re.compile(r"^[ \t\-*•\d.)]*([^\n:]{3,60}):", re.MULTILINE)

# Cues that make a *sentence* historical even without a section header.
HISTORICAL_CUES = (
    "tiền sử",
    "tiền căn",
    "đã từng",
    "từng bị",
    "từng được chẩn đoán",
    "trước đây",
    "nhiều năm nay",
    "được chẩn đoán từ",
    "cách đây",
)
FAMILY_CUES = (
    "gia đình",
    "mẹ bệnh nhân",
    "bố bệnh nhân",
    "cha bệnh nhân",
    "anh trai",
    "chị gái",
    "người thân",
    "di truyền trong",
)


def find_sections(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, kind) spans, kind in {"history", "family"}."""
    marks: list[tuple[int, str]] = []
    for m in HEADER_RE.finditer(text):
        title = m.group(1).strip().lower()
        if any(h in title for h in FAMILY_HEADERS):
            kind = "family"
        elif any(h in title for h in HISTORICAL_HEADERS):
            kind = "history"
        else:
            kind = "other"
        marks.append((m.start(), kind))

    spans: list[tuple[int, int, str]] = []
    for i, (start, kind) in enumerate(marks):
        if kind == "other":
            continue
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        spans.append((start, end, kind))
    return spans


def sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    start = max(text.rfind(c, 0, pos) for c in (".", "\n", ";", "?", "!"))
    end_candidates = [text.find(c, pos) for c in (".", "\n", ";", "?", "!")]
    end_candidates = [e for e in end_candidates if e != -1]
    return (start + 1 if start != -1 else 0, min(end_candidates) if end_candidates else len(text))


def apply_sections(entities: list[dict[str, Any]], text: str) -> int:
    spans = find_sections(text)
    changed = 0
    for ent in entities:
        if ent.get("type") not in ASSERTABLE:
            continue
        pos = ent.get("position") or [0, 0]
        start = pos[0]
        assertions = list(ent.get("assertions") or [])

        kind = None
        for s_start, s_end, s_kind in spans:
            if s_start <= start < s_end:
                kind = s_kind  # later (more specific) sections win
        if kind is None:
            s_start, s_end = sentence_bounds(text, start)
            before = text[s_start:start].lower()
            if any(cue in before for cue in FAMILY_CUES):
                kind = "family"
            elif any(cue in before for cue in HISTORICAL_CUES):
                kind = "history"

        if kind == "family" and FAMILY not in assertions:
            assertions.append(FAMILY)
            changed += 1
        elif kind == "history" and HISTORICAL not in assertions:
            assertions.append(HISTORICAL)
            changed += 1

        ent["assertions"] = assertions
    return changed


# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------

# Only high-precision cues. "không" alone is deliberately included but guarded
# by PSEUDO_NEG below, because "không thể loại trừ" ("cannot rule out") is an
# assertion of possibility, not a negation.
NEG_CUES = (
    "không có",
    "không ghi nhận",
    "không thấy",
    "không phát hiện",
    "không bị",
    "không còn",
    "không sốt",
    "chưa ghi nhận",
    "chưa phát hiện",
    "chưa có",
    "loại trừ",
    "phủ định",
    "âm tính",
)
PSEUDO_NEG = (
    "không thể loại trừ",
    "không loại trừ được",
    "chưa thể loại trừ",
    "không rõ",
)
CLAUSE_BREAK = re.compile(r"[,.;:\n]|\bnhưng\b|\bsong\b|\btuy nhiên\b")
NEG_WINDOW = 60


def apply_negex(entities: list[dict[str, Any]], text: str) -> int:
    lowered = text.lower()
    changed = 0
    for ent in entities:
        if ent.get("type") not in ASSERTABLE:
            continue
        assertions = list(ent.get("assertions") or [])
        if NEGATED in assertions:
            continue
        start = (ent.get("position") or [0, 0])[0]
        window_start = max(0, start - NEG_WINDOW)
        window = lowered[window_start:start]
        if any(p in window for p in PSEUDO_NEG):
            continue
        hit = -1
        for cue in NEG_CUES:
            idx = window.rfind(cue)
            if idx > hit:
                hit = idx + len(cue)
        if hit == -1:
            continue
        # A clause break between the cue and the mention ends the scope.
        if CLAUSE_BREAK.search(window[hit:]):
            continue
        assertions.append(NEGATED)
        ent["assertions"] = assertions
        changed += 1
    return changed


# --------------------------------------------------------------------------
# Consistency and gates
# --------------------------------------------------------------------------

def apply_consistency(entities: list[dict[str, Any]], mode: str) -> int:
    """Harmonize assertions across repeated (text, type) in one document.

    `majority` favours precision (a minority mark is dropped), `union` favours
    recall (any mark propagates to every occurrence). The model currently marks
    isHistorical on 16% of assertable entities while turn-1 truth carries 28%,
    so `union` is the better prior -- but neither is verifiable without labels.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ent in entities:
        if ent.get("type") not in ASSERTABLE:
            continue
        groups.setdefault((str(ent.get("text")), str(ent.get("type"))), []).append(ent)

    changed = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        votes: dict[str, int] = {}
        for ent in members:
            for a in ent.get("assertions") or []:
                votes[a] = votes.get(a, 0) + 1
        if mode == "union":
            consensus = sorted(votes)
        else:
            consensus = sorted(a for a, n in votes.items() if n * 2 > len(members))
        for ent in members:
            if sorted(ent.get("assertions") or []) != consensus:
                ent["assertions"] = list(consensus)
                changed += 1
    return changed


def apply_family_gate(entities: list[dict[str, Any]], text: str) -> int:
    """isFamily is 0.9% of assertions; unsupported ones are EV-negative."""
    lowered = text.lower()
    spans = [s for s in find_sections(text) if s[2] == "family"]
    changed = 0
    for ent in entities:
        assertions = list(ent.get("assertions") or [])
        if FAMILY not in assertions:
            continue
        start = (ent.get("position") or [0, 0])[0]
        in_section = any(s <= start < e for s, e, _ in spans)
        s_start, s_end = sentence_bounds(text, start)
        has_cue = any(cue in lowered[s_start:s_end] for cue in FAMILY_CUES)
        if not (in_section or has_cue):
            assertions.remove(FAMILY)
            ent["assertions"] = assertions
            changed += 1
    return changed


ICD4_RE = re.compile(r"^([A-Z]\d{2})\.(\d)$")


def load_curated_texts(path: Path) -> set[str]:
    """Normalized diagnosis texts that `diagnoses.csv` resolves exactly."""
    texts: set[str] = set()
    if not path.exists():
        return texts
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("text") or "").strip().lower()
            if raw:
                texts.add(re.sub(r"\s+", " ", raw))
    return texts


def apply_hedge_icd(entities: list[dict[str, Any]], curated: set[str], scope: str) -> int:
    """Add the category's ".9" sibling as a second candidate, where uncertain.

    Adding a 2nd candidate pays when its hit probability exceeds J/(1+J); at the
    current J_candidates of 0.287 that threshold is 0.223. The most common
    residual error is a wrong *subcode*, and ".9" (unspecified) is the highest
    prior alternative, so this is the cheapest hedge available.

    Restricted to mentions the curated `diagnoses.csv` does NOT resolve exactly.
    A curated exact hit was mined from graded turn-1 labels and is the most
    reliable code in the pipeline -- hedging it only grows the union. Unverified
    against truth: ship it as its own variant and read the leaderboard.
    """
    changed = 0
    for ent in entities:
        if ent.get("type") != DIAG:
            continue
        candidates = list(ent.get("candidates") or [])
        if len(candidates) != 1:
            continue
        norm = re.sub(r"\s+", " ", str(ent.get("text", "")).strip().lower())
        if scope == "uncurated" and norm in curated:
            continue
        m = ICD4_RE.match(candidates[0])
        if not m or m.group(2) == "9":
            continue
        candidates.append(f"{m.group(1)}.9")
        ent["candidates"] = candidates
        changed += 1
    return changed


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="Input prediction folder.")
    parser.add_argument("--input", type=Path, required=True, help="Folder with the matching *.txt.")
    parser.add_argument("--out", type=Path, required=True, help="Output prediction folder.")
    parser.add_argument("--sections", action="store_true")
    parser.add_argument("--negex", action="store_true")
    parser.add_argument(
        "--consistency",
        choices=["majority", "union"],
        help="Repeated (text,type) mentions in one doc: strip to the majority assertion set, "
        "or propagate the union. Unverifiable offline -- ship as separate variants.",
    )
    parser.add_argument("--family-gate", action="store_true")
    parser.add_argument(
        "--hedge-icd",
        choices=["uncurated", "all"],
        help="Add the '.9' sibling as a 2nd ICD candidate. 'uncurated' skips mentions that "
        "diagnoses.csv resolves exactly (~19 entities); 'all' hedges every 4-char non-.9 code "
        "(~321). NOT recommended without leaderboard evidence: the payoff threshold is "
        "J/(1+J)=0.223 and the sibling's hit probability is estimated at only 0.11-0.20.",
    )
    parser.add_argument(
        "--diagnoses-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "terminology" / "diagnoses.csv",
        help="Curated table used to decide which diagnoses are too reliable to hedge.",
    )
    args = parser.parse_args()

    curated = load_curated_texts(args.diagnoses_csv) if args.hedge_icd else set()
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {"sections": 0, "negex": 0, "consistency": 0, "family_gate": 0, "hedge_icd": 0}
    files = 0

    for pred_path in sorted(args.pred.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        text_path = args.input / f"{pred_path.stem}.txt"
        if not text_path.exists():
            print(f"WARN: no input for {pred_path.name}, copied unchanged")
            entities = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            text = text_path.read_text(encoding="utf-8")
            entities = json.loads(pred_path.read_text(encoding="utf-8"))
            if args.sections:
                stats["sections"] += apply_sections(entities, text)
            if args.negex:
                stats["negex"] += apply_negex(entities, text)
            if args.family_gate:
                stats["family_gate"] += apply_family_gate(entities, text)
            if args.consistency:
                stats["consistency"] += apply_consistency(entities, args.consistency)
            if args.hedge_icd:
                stats["hedge_icd"] += apply_hedge_icd(entities, curated, args.hedge_icd)

        for ent in entities:
            if ent.get("type") in ASSERTABLE:
                ent["assertions"] = sorted(set(ent.get("assertions") or []))
        (args.out / pred_path.name).write_text(
            json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        files += 1

    print(f"Wrote {files} files to {args.out}")
    for key, value in stats.items():
        if value:
            print(f"  {key}: {value} entities changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
