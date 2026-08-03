#!/usr/bin/env python3
"""Extend truncated THUỐC spans over their dose/route/sig tail.

Why
---
Scored against the only real ground truth in this repo -- the 19-entity worked
example the organisers publish in the task statement, reconstructed into
`data/corpus/gold_btc/` -- the pipeline truncates **7 of 11 drug spans (64%)**,
always at the sig token:

    truth  senna 8.6 mg po bid:prn          pred  senna
    truth  clonazepam 0.5 mg po qam:prn     pred  clonazepam 0.5 mg po
    truth  guaifenesin ml po q6h:prn        pred  guaifenesin ml po q6
    truth  docusate sodium 100 mg po bid    pred  docusate sodium 100 mg po

This is the most expensive error class in the pipeline because it is the only
one that hits all three metrics at once: the mention's words are wrong (WER), and
because assertion and candidate items are keyed on `(text, type, occurrence)`, a
wrong span also zeroes that entity's assertion and its code. One truncated drug
costs the full 1.0 of weight, not 0.3.

The convention is fixed by the task statement's own example: a `THUỐC` span
covers ingredient + dose + route + frequency (`amlodipine 10 mg po daily`) and
**stops before the indication** (`điều trị ho` is excluded, and `ho` is a separate
TRIỆU_CHỨNG). So the fix is not a model change -- it is a deterministic walk
along a closed sig vocabulary, with hard stops at the indication.

This is an entity-set change, the lane that scored 31.89 and 33.679 when done
blind. It is only defensible because `data/corpus/gold_btc/` can now measure it
before submitting. Always run `--report` before shipping.

Usage:
  python scripts/fix_drug_spans.py --pred experiments/v11 --input input_turn2 --out experiments/v12
  python scripts/fix_drug_spans.py --pred experiments/gold_check \
      --input data/corpus/gold_btc/input --truth data/corpus/gold_btc/truth --report
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

DRUG = "THUỐC"

# Routes and frequencies as written in these records. Sig shorthand packs several
# into one token separated by ":" ("q6h:prn"), so matching is per sub-token.
ROUTES = {
    "po", "iv", "im", "sc", "sq", "sl", "pr", "ng", "pg", "top", "inh", "id",
    "ip", "it", "neb", "od", "os", "ou", "buccal", "rectal", "oral",
}
FREQUENCIES = {
    "daily", "qd", "bid", "tid", "qid", "qhs", "qam", "qpm", "qnoon", "qod",
    "qwk", "weekly", "prn", "ac", "pc", "hs", "stat", "once", "nightly",
}
UNITS = {
    "mg", "mcg", "µg", "g", "gm", "kg", "ml", "l", "unit", "units", "u", "iu",
    "meq", "mmol", "mmol/l", "%", "tab", "tabs", "cap", "caps", "puff", "puffs",
    "drop", "drops", "patch", "spray", "vial", "amp",
}
FORMS = {
    "tablet", "tablets", "capsule", "capsules", "suspension", "solution",
    "syrup", "injection", "cream", "ointment", "gel", "inhaler", "nebuliser",
    "nebulizer", "supp", "suppository", "er", "xl", "xr", "sr", "cr", "la",
    "viên", "nén", "gói", "ống", "lọ", "chai", "siro",
}
# q4h, q6h, q12h, q8h... and bare numeric sig like "q6"
QSIG_RE = re.compile(r"^q\d+h?$")
NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?$")

SIG_WORDS = ROUTES | FREQUENCIES | UNITS | FORMS

# Hard stops. The task statement's example excludes the indication from the span
# ("guaifenesin ml po q6h:prn điều trị ho" -> the drug span ends at "q6h:prn"),
# and an item number starts the next drug.
STOP_PHRASES = ("điều trị", "để điều", "chỉ định", "dùng cho", "cho bệnh")
ITEM_NUMBER_RE = re.compile(r"^\d{1,2}[.)]$")
MAX_EXTRA_TOKENS = 6


def is_sig_token(token: str) -> bool:
    cleaned = token.strip().strip(",;").lower()
    if not cleaned:
        return False
    if ITEM_NUMBER_RE.match(cleaned):
        return False
    # "q6h:prn" / "bid:prn" -- every part must be sig vocabulary
    parts = [p for p in cleaned.split(":") if p]
    if not parts:
        return False
    for part in parts:
        if part in SIG_WORDS or QSIG_RE.match(part) or NUMERIC_RE.match(part):
            continue
        return False
    return True


def extend_span(text: str, start: int, end: int) -> int:
    """Return a new end offset covering the dose/route/sig tail."""
    # 1. The span may stop mid-word ("q6" out of "q6h:prn"). Complete it ONLY when
    #    the whole word is itself sig vocabulary. Completing unconditionally looks
    #    right on the organisers' clean example and is destructive on turn-2, whose
    #    text runs words together: it produced 'atenololt' -> 'atenololtrong',
    #    'doxycycli' -> 'doxycyclinebactrim' (two different drugs merged),
    #    'morphine' -> 'morphineoral', and swallowed trailing ')' ':' '.' from
    #    'omeprazole)' / 'Torsemide:' / 'Nitramyl.'.
    if end < len(text) and not text[end].isspace():
        word_end = end
        while word_end < len(text) and not text[word_end].isspace():
            word_end += 1
        word_start = end
        while word_start > start and not text[word_start - 1].isspace():
            word_start -= 1
        if is_sig_token(text[word_start:word_end]):
            end = word_end

    # 2. Walk forward over sig tokens, never past a line break or an indication.
    consumed = 0
    while consumed < MAX_EXTRA_TOKENS:
        probe = end
        while probe < len(text) and text[probe] in " \t":
            probe += 1
        if probe >= len(text) or probe == end:
            break
        if text[end:probe].count("\n"):
            break
        tail = text[probe:probe + 40].lower()
        if any(tail.startswith(p) for p in STOP_PHRASES):
            break
        match = re.match(r"[^\s]+", text[probe:])
        if not match:
            break
        token = match.group(0)
        if not is_sig_token(token):
            break

        # A bare number only belongs to the drug when a unit follows it
        # ("senna 8.6 mg"). Without the lookahead this appends list numbering and
        # unrelated digits: 'suboxone' -> 'suboxone 3', 'zosyn' -> 'zosyn 8'.
        if NUMERIC_RE.match(token.strip(",;.")):
            rest = text[probe + len(token):].lstrip(" \t")
            nxt = re.match(r"[^\s]+", rest)
            if not nxt or nxt.group(0).strip(",;.").lower() not in UNITS | FORMS:
                break

        end = probe + len(token)
        consumed += 1

    # Trailing punctuation is never part of the mention -- consuming "bid," gave
    # 'metoprolol 25mg po bid,' and 'Vitamin 3B x 4 viên,'.
    while end > start and text[end - 1] in ",;.:/)":
        end -= 1

    return end


def fix_document(text: str, entities: list[dict[str, Any]]) -> int:
    """Extend drug spans in place; returns how many changed."""
    occupied = [
        (e["position"][0], e["position"][1])
        for e in entities
        if e.get("type") != DRUG and isinstance(e.get("position"), list)
    ]
    changed = 0
    for ent in entities:
        if ent.get("type") != DRUG or not isinstance(ent.get("position"), list):
            continue
        start, end = ent["position"]
        new_end = extend_span(text, start, end)
        if new_end <= end:
            continue
        # Never swallow another entity: the indication symptom sits right after
        # many of these ("... q6h:prn điều trị ho"), and overlapping entities are
        # a schema violation as well as a scoring loss.
        if any(start < o_end and new_end > o_start for o_start, o_end in occupied):
            continue
        ent["position"] = [start, new_end]
        ent["text"] = text[start:new_end]
        changed += 1
    return changed


def score(pred_dir: Path, truth_dir: Path, input_dir: Path) -> dict[str, float]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_submission import simulate_metrics

    return simulate_metrics(pred_dir, truth_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--truth", type=Path, help="Score before/after against this ground truth.")
    parser.add_argument("--report", action="store_true", help="Print every span change.")
    args = parser.parse_args()

    if not args.out and not (args.report or args.truth):
        raise SystemExit("Need --out, or --report/--truth for a dry look.")

    out_dir = args.out or (args.pred.parent / (args.pred.name + "_spanfix"))
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    changes: list[tuple[str, str, str]] = []
    for pred_path in sorted(args.pred.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        text_path = args.input / f"{pred_path.stem}.txt"
        entities = json.loads(pred_path.read_text(encoding="utf-8"))
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
            before = {id(e): e.get("text") for e in entities}
            n = fix_document(text, entities)
            total += n
            if n:
                for e in entities:
                    if e.get("type") == DRUG and before.get(id(e)) != e.get("text"):
                        changes.append((pred_path.name, before[id(e)], e["text"]))
        (out_dir / pred_path.name).write_text(
            json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Extended {total} THUỐC spans -> {out_dir}")
    if args.report:
        for name, old, new in changes[:40]:
            print(f"  {name:>9}  {old[:40]!r:<44} -> {new[:52]!r}")
        if len(changes) > 40:
            print(f"  ... {len(changes) - 40} more")

    if args.truth:
        before_metrics = score(args.pred, args.truth, args.input)
        after_metrics = score(out_dir, args.truth, args.input)
        print(f"\n{'metric':<16}{'before':>10}{'after':>10}{'delta':>10}")
        for key in ("text_score", "J_assertion", "J_candidates", "final_score"):
            b, a = before_metrics[key], after_metrics[key]
            print(f"{key:<16}{b*100:>10.3f}{a*100:>10.3f}{(a-b)*100:>+10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
