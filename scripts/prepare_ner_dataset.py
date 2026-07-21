#!/usr/bin/env python3
"""Convert the current curated output/ (hand-refined over many leaderboard
rounds) into a weak-labeled training set for a fine-tuned NER + assertion
model.

Why: output/*.json is not reproducible from any script in this repo (see
generate_outputs.py's disabled assertion heuristic and the many hand-edits
tracked in .agent_runs/). Treated as a fixed submission it is an overfit
answer key for the 100 public files and will not generalize to the private
test set the contest reruns source code against. Treated as *training data*
for a model, the manual effort behind it becomes reusable signal instead of
being thrown away.

Output:
  data/ner_dataset/all.jsonl       one JSON record per input file
  data/ner_dataset/train.jsonl     85 files (deterministic split)
  data/ner_dataset/holdout.jsonl   15 files, for local generalization checks
  data/ner_dataset/split.json      {"train": [...ids], "holdout": [...ids]}

Usage:
  python scripts/prepare_ner_dataset.py
  python scripts/prepare_ner_dataset.py --holdout-frac 0.15 --seed 13
  python scripts/prepare_ner_dataset.py --train-all-labels
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DATASET_DIR = ROOT / "data" / "ner_dataset"

ASSERTION_TYPES = {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC"}
CANDIDATE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}
VALID_TYPES = ASSERTION_TYPES | {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}


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
            shutil.copy2(tmp, path)
            tmp.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"viettelrace-{os.getpid()}-{path.name}.tmp"
    tmp.write_text(text, encoding="utf-8")
    replace_existing(tmp, path)


def load_record(file_id: int) -> dict[str, Any]:
    text = (INPUT_DIR / f"{file_id}.txt").read_text(encoding="utf-8")
    entities = json.loads((OUTPUT_DIR / f"{file_id}.json").read_text(encoding="utf-8"))

    clean_entities: list[dict[str, Any]] = []
    for ent in entities:
        ent_type = ent.get("type")
        start, end = ent["position"]
        if ent_type not in VALID_TYPES:
            continue
        if text[start:end] != ent["text"]:
            raise ValueError(f"file {file_id}: span mismatch for {ent!r}")
        record = {
            "start": start,
            "end": end,
            "type": ent_type,
            "text": ent["text"],
        }
        if ent_type in ASSERTION_TYPES:
            record["assertions"] = sorted(set(ent.get("assertions") or []))
        if ent_type in CANDIDATE_TYPES:
            record["candidates"] = list(ent.get("candidates") or [])
        clean_entities.append(record)

    clean_entities.sort(key=lambda e: (e["start"], e["end"]))
    return {"id": file_id, "text": text, "entities": clean_entities}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"viettelrace-{os.getpid()}-{path.name}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    replace_existing(tmp, path)


def summarize(records: list[dict[str, Any]]) -> None:
    type_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()
    candidate_missing: Counter[str] = Counter()
    total_entities = 0
    for rec in records:
        for ent in rec["entities"]:
            type_counts[ent["type"]] += 1
            total_entities += 1
            for a in ent.get("assertions", []):
                assertion_counts[a] += 1
            if ent["type"] in CANDIDATE_TYPES and not ent.get("candidates"):
                candidate_missing[ent["type"]] += 1

    print(f"files: {len(records)}  entities: {total_entities}")
    print("by type:", dict(type_counts))
    print("assertions:", dict(assertion_counts))
    print("missing candidates:", dict(candidate_missing))


def make_folds(file_ids: list[int], n_folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """n_folds disjoint holdout partitions covering every id exactly once
    (round-robin over a shuffled order, so fold sizes differ by at most 1).
    """
    rng = random.Random(seed)
    shuffled = file_ids[:]
    rng.shuffle(shuffled)
    buckets = [shuffled[i::n_folds] for i in range(n_folds)]
    folds = []
    for k in range(n_folds):
        holdout_ids = set(buckets[k])
        train_ids = sorted(set(file_ids) - holdout_ids)
        folds.append((train_ids, sorted(holdout_ids)))
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--train-all-labels",
        action="store_true",
        help=(
            "Write every labeled public file into train.jsonl for a final model build. "
            "holdout.jsonl is still written as a reporting set, but it overlaps train "
            "and must not be treated as an unbiased validation score."
        ),
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Warn if the number of labeled training files differs from this value.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help=(
            "If >1, additionally write N disjoint train/holdout folds to "
            "data/ner_dataset/fold{k}_{train,holdout}.jsonl -- a cheaper way to sanity-check "
            "that the default 85/15 holdout score isn't a fluke of one particular split, without "
            "committing to full N-way cross-validation (each fold still costs a full Kaggle "
            "training run to actually evaluate). Purely additive: never touches the default "
            "train.jsonl/holdout.jsonl produced below."
        ),
    )
    args = parser.parse_args()

    input_ids = {int(p.stem) for p in INPUT_DIR.glob("*.txt") if p.stem.isdigit()}
    output_ids = {int(p.stem) for p in OUTPUT_DIR.glob("*.json") if p.stem.isdigit()}
    file_ids = sorted(input_ids & output_ids)
    unlabeled_input_ids = sorted(input_ids - output_ids)
    orphan_output_ids = sorted(output_ids - input_ids)
    if args.expected_count is not None and len(file_ids) != args.expected_count:
        print(f"warning: expected {args.expected_count} labeled files, found {len(file_ids)}")
    if unlabeled_input_ids:
        print(
            "warning: ignoring input files without training labels in output/: "
            f"{unlabeled_input_ids[:20]}{'...' if len(unlabeled_input_ids) > 20 else ''}"
        )
    if orphan_output_ids:
        print(
            "warning: ignoring output labels without matching input files: "
            f"{orphan_output_ids[:20]}{'...' if len(orphan_output_ids) > 20 else ''}"
        )
    if not file_ids:
        raise SystemExit("No labeled training files found. Need matching input/{id}.txt and output/{id}.json.")

    records = [load_record(i) for i in file_ids]

    rng = random.Random(args.seed)
    shuffled = file_ids[:]
    rng.shuffle(shuffled)
    n_holdout = max(1, round(len(shuffled) * args.holdout_frac))
    holdout_ids = set(shuffled[:n_holdout])
    train_ids = set(file_ids if args.train_all_labels else shuffled[n_holdout:])

    train_records = [r for r in records if r["id"] in train_ids]
    holdout_records = [r for r in records if r["id"] in holdout_ids]

    write_jsonl(DATASET_DIR / "all.jsonl", records)
    write_jsonl(DATASET_DIR / "train.jsonl", train_records)
    write_jsonl(DATASET_DIR / "holdout.jsonl", holdout_records)
    write_text(
        DATASET_DIR / "split.json",
        json.dumps(
            {
                "train": sorted(train_ids),
                "holdout": sorted(holdout_ids),
                "seed": args.seed,
                "train_all_labels": args.train_all_labels,
            },
            indent=2,
        ),
    )

    print("== all ==")
    summarize(records)
    print(f"\n== train ({len(train_records)} files) ==")
    summarize(train_records)
    print(f"\n== holdout ({len(holdout_records)} files) ==")
    summarize(holdout_records)
    print(f"\nWrote dataset to {DATASET_DIR}")

    if args.folds > 1:
        by_id = {r["id"]: r for r in records}
        folds = make_folds(file_ids, args.folds, args.seed)
        summary = []
        for k, (train_ids_k, holdout_ids_k) in enumerate(folds):
            write_jsonl(DATASET_DIR / f"fold{k}_train.jsonl", [by_id[i] for i in train_ids_k])
            write_jsonl(DATASET_DIR / f"fold{k}_holdout.jsonl", [by_id[i] for i in holdout_ids_k])
            summary.append({"fold": k, "train": train_ids_k, "holdout": holdout_ids_k})
        write_text(
            DATASET_DIR / "folds_summary.json",
            json.dumps({"n_folds": args.folds, "seed": args.seed, "folds": summary}, indent=2),
        )
        print(f"\nWrote {args.folds} folds (fold{{k}}_train.jsonl / fold{{k}}_holdout.jsonl) to {DATASET_DIR}")
        print(
            "To spot-check generalization: upload one fold's train/holdout as if they were "
            "train.jsonl/holdout.jsonl, train once, and compare its holdout score to the default "
            "split's -- large swings mean the default split's number shouldn't be trusted alone."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
