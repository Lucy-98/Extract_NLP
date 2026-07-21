#!/usr/bin/env python3
"""Deeper heuristic audit for ViettelRace predictions.

This does not need ground truth. It looks for patterns that often hurt the
public metrics: inconsistent candidates for the same concept, clipped spans,
very short concepts, and disease/drug entities without candidates.

Legacy tool -- still useful for auditing any output/ folder (old or new
pipeline), unlike the other scripts in this folder. Run from the repo root:
  python legacy/scripts/deep_audit_output.py --pred output --input input
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIAG = "CH\u1ea8N_\u0110O\u00c1N"
SYM = "TRI\u1ec6U_CH\u1ee8NG"
DRUG = "THU\u1ed0C"
CANDIDATE_TYPES = {DIAG, DRUG}


def norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="output")
    parser.add_argument("--input", default="input")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    pred_dir = Path(args.pred)
    input_dir = Path(args.input)

    type_counts: Counter[str] = Counter()
    missing_candidates: Counter[str] = Counter()
    candidate_totals: Counter[str] = Counter()
    candidate_by_text: dict[tuple[str, str], Counter[tuple[str, ...]]] = defaultdict(Counter)
    short_entities: list[tuple[str, int, str, str, list[int], list[str] | None, list[str]]] = []
    clipped_entities: list[tuple[str, int, str, str, list[int], str]] = []
    duplicate_texts: Counter[tuple[str, str]] = Counter()

    for path in sorted(pred_dir.glob("*.json"), key=lambda p: int(p.stem)):
        file_id = path.stem
        text = (input_dir / f"{file_id}.txt").read_text(encoding="utf-8")
        entities = read_json(path)
        per_file_texts: Counter[tuple[str, str]] = Counter()

        for idx, ent in enumerate(entities):
            ent_type = str(ent.get("type", ""))
            ent_text = str(ent.get("text", ""))
            start, end = ent.get("position", [0, 0])
            candidates = ent.get("candidates")
            assertions = ent.get("assertions") or []

            type_counts[ent_type] += 1
            per_file_texts[(ent_type, norm(ent_text))] += 1
            if ent_type in CANDIDATE_TYPES:
                candidate_totals[ent_type] += 1
                cand_tuple = tuple(str(c) for c in (candidates or []))
                candidate_by_text[(ent_type, norm(ent_text))][cand_tuple] += 1
                if not cand_tuple:
                    missing_candidates[ent_type] += 1

            if len(ent_text) <= 3:
                short_entities.append((file_id, idx, ent_type, ent_text, [start, end], candidates, assertions))

            before = text[max(0, start - 18) : start]
            after = text[end : min(len(text), end + 26)]
            if after.startswith((" cơ tim", " tá tràng", " thực quản", " đại tràng", " phổi", " gan", " thận")):
                clipped_entities.append((file_id, idx, ent_type, ent_text, [start, end], before + "[" + ent_text + "]" + after))

        for key, count in per_file_texts.items():
            if count > 1:
                duplicate_texts[key] += count

    print("== Deep audit ==")
    print("Type counts:", dict(type_counts))
    print("Candidate totals:", dict(candidate_totals))
    print("Missing candidates:", dict(missing_candidates))

    print("\n== Inconsistent or empty candidates by repeated text ==")
    shown = 0
    for (ent_type, ent_text), choices in sorted(
        candidate_by_text.items(), key=lambda item: (-sum(item[1].values()), item[0])
    ):
        if len(choices) <= 1 and () not in choices:
            continue
        print(ent_type, repr(ent_text), "count=", sum(choices.values()), "choices=", dict(choices))
        shown += 1
        if shown >= args.limit:
            break

    print("\n== Very short entities ==")
    for item in short_entities[: args.limit]:
        print(item)
    if len(short_entities) > args.limit:
        print(f"... {len(short_entities) - args.limit} more")

    print("\n== Possibly clipped entities ==")
    for item in clipped_entities[: args.limit]:
        print(item)
    if len(clipped_entities) > args.limit:
        print(f"... {len(clipped_entities) - args.limit} more")

    print("\n== Repeated same text/type inside files ==")
    for (ent_type, ent_text), count in duplicate_texts.most_common(args.limit):
        print(ent_type, repr(ent_text), count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
