#!/usr/bin/env python3
"""Build the recoding worklist: every distinct mention whose code decides the score.

Why this file exists
--------------------
Measured on `experiments/v1_revert_icd` (2026-07-31):

    CHẨN_ĐOÁN  252 distinct texts / 798 mentions -- 93.2% of mentions are an
               EXACT hit in data/terminology/diagnoses.csv
    THUỐC       94 distinct texts / 216 mentions -- 79.6% exact in drugs.csv

Yet `J_candidates = 0.2951` implies only **~45.6% of emitted codes are right**
(inverting J = k/(2-k)). Coverage is not the bottleneck; code correctness is.

The cause is circular: `diagnoses.csv` is mined from `output/`, which is our own
turn-1 submission -- the one that scored `J_candidates` **29.98**. The lookup
table is a memorised copy of our own wrong answers, and every pipeline variant
faithfully reproduces that 46% accuracy. That is why `J_candidates` did not move
across eight submissions (29.98 / 29.51 / 29.34 / 28.68).

So the unit of work is not 15,144 catalog titles and not a new retrieval layer
in the inference path. It is **a few hundred distinct strings whose `candidate`
column is wrong**. This script emits exactly that list, with the context needed
to code each one correctly, and nothing else.

The `text` column is the join key and must never change: `run_pipeline.py`'s
`--add-terminology-entities` generates spans *from* it, so editing it would
change the entity set -- the high-risk lane that has scored negative every time
it was touched (31.89, 33.679).

Usage:
  python scripts/extract_mentions.py
  python scripts/extract_mentions.py --pred experiments/v1_revert_icd --context 2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
TERM = ROOT / "data" / "terminology"

DIAG = "CHẨN_ĐOÁN"
DRUG = "THUỐC"
TABLES = {DIAG: "diagnoses.csv", DRUG: "drugs.csv"}

SENT_SPLIT = re.compile(r"(?<=[.;!?\n])\s+")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_table(path: Path) -> dict[str, list[str]]:
    codes: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return codes
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = norm(row["text"])
            if row["candidate"] not in codes[key]:
                codes[key].append(row["candidate"])
    return codes


def sentence_around(text: str, start: int, end: int, window: int) -> str:
    """The sentence containing [start, end), plus `window - 1` neighbours."""
    sentences: list[tuple[int, int, str]] = []
    offset = 0
    for part in SENT_SPLIT.split(text):
        if part:
            sentences.append((offset, offset + len(part), part))
        offset += len(part) + 1

    hit = next((i for i, (s, e, _t) in enumerate(sentences) if s <= start < e), None)
    if hit is None:
        return re.sub(r"\s+", " ", text[max(0, start - 120):end + 120]).strip()

    lo = max(0, hit - (window - 1))
    hi = min(len(sentences), hit + window)
    joined = " ".join(s[2] for s in sentences[lo:hi])
    return re.sub(r"\s+", " ", joined).strip()[:400]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, default=ROOT / "experiments" / "v1_revert_icd")
    parser.add_argument("--input", type=Path, default=ROOT / "input_turn2")
    parser.add_argument("--out", type=Path, default=TERM / "recode_worklist.csv")
    parser.add_argument("--context", type=int, default=2, help="Sentences of context to carry.")
    args = parser.parse_args()

    tables = {t: load_table(TERM / name) for t, name in TABLES.items()}

    freq: dict[tuple[str, str], int] = Counter()
    raw_forms: dict[tuple[str, str], Counter] = defaultdict(Counter)
    contexts: dict[tuple[str, str], list[str]] = defaultdict(list)

    pred_files = sorted(args.pred.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not pred_files:
        raise SystemExit(f"No predictions in {args.pred}. Run run_pipeline.py first.")

    for pred_path in pred_files:
        text_path = args.input / f"{pred_path.stem}.txt"
        text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        for ent in json.loads(pred_path.read_text(encoding="utf-8")):
            ent_type = ent.get("type")
            if ent_type not in TABLES:
                continue
            key = (norm(str(ent.get("text", ""))), ent_type)
            freq[key] += 1
            raw_forms[key][str(ent.get("text", ""))] += 1
            if text and len(contexts[key]) < 2:
                pos = ent.get("position") or [0, 0]
                snippet = sentence_around(text, pos[0], pos[1], args.context)
                if snippet and snippet not in contexts[key]:
                    contexts[key].append(snippet)

    # Table rows never seen in these predictions still answer on the private
    # test, so they are recoded too -- just without context to guide it.
    rows: list[dict[str, object]] = []
    for ent_type, codes in tables.items():
        for key_text, current in codes.items():
            key = (key_text, ent_type)
            rows.append({
                "text": key_text,
                "type": ent_type,
                "freq": freq.get(key, 0),
                "current_candidate": "|".join(current),
                "surface_form": raw_forms[key].most_common(1)[0][0] if raw_forms.get(key) else key_text,
                "in_predictions": int(key in freq),
                "context": " || ".join(contexts.get(key, [])),
            })
    for (key_text, ent_type), count in freq.items():
        if key_text in tables[ent_type]:
            continue
        key = (key_text, ent_type)
        rows.append({
            "text": key_text,
            "type": ent_type,
            "freq": count,
            "current_candidate": "",
            "surface_form": raw_forms[key].most_common(1)[0][0],
            "in_predictions": 1,
            "context": " || ".join(contexts.get(key, [])),
        })

    rows.sort(key=lambda r: (r["type"], -int(r["freq"]), str(r["text"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "type", "freq", "current_candidate", "surface_form", "in_predictions", "context"],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_type = Counter(str(r["type"]) for r in rows)
    used = Counter(str(r["type"]) for r in rows if r["in_predictions"])
    uncoded = Counter(str(r["type"]) for r in rows if not r["current_candidate"])
    print(f"Wrote {len(rows)} rows to {args.out}\n")
    print(f"{'type':<14}{'rows':>7}{'seen in preds':>16}{'no code yet':>14}")
    for ent_type in TABLES:
        print(f"{ent_type:<14}{by_type[ent_type]:>7}{used[ent_type]:>16}{uncoded[ent_type]:>14}")
    print("\nNext: docs/linking_recode.md (GPU steps), then scripts/recode_terminology.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
