#!/usr/bin/env python3
"""Synthetic data augmentation for the NER + assertion training set.

Why this exists: the task statement explicitly requires solutions to use
techniques *outside* the core solution to generate additional training data
-- the labeled corpus here is only 100 hand-annotated documents (85 after
the train/holdout split). This script is dependency-free (stdlib only, same
constraint as the rest of scripts/) and produces two kinds of synthetic
records from data/ner_dataset/train.jsonl:

1. Entity substitution: swap one entity's text for another same-type entity
   text seen elsewhere in the corpus or in data/terminology/{diagnoses,
   drugs}.csv, shifting the offsets of every entity after it in the
   document. This directly targets the biggest generalization gap in this
   repo: the terminology tables only cover ~230 diagnosis phrasings and
   ~127 drug names, so the model needs to learn the *pattern* (this kind of
   span is a CHẨN_ĐOÁN/THUỐC/TRIỆU_CHỨNG) rather than memorizing the ~100
   strings it has seen.
2. Assertion-cue synthesis: take entity mentions that currently carry no
   assertion and wrap them in short templated negation / historical /
   family clauses ("không có ...", "tiền sử ...", "mẹ bệnh nhân có tiền sử
   ..."). Several clauses are concatenated into one synthetic multi-entity
   document (real documents always contain more than one concept -- see
   README) instead of emitting one-entity toy sentences.

data/ner_dataset/holdout.jsonl is never touched by this script -- synthetic
records only ever go into a separate train_augmented.jsonl, so the holdout
generalization estimate stays computed on real clinical text only.

Usage:
  python scripts/augment_ner_dataset.py
  python scripts/augment_ner_dataset.py --multiplier 1 --assertion-docs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "ner_dataset"
TERM_DIR = ROOT / "data" / "terminology"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_terminology_index import TerminologyMatcher, DIAG, DRUG  # noqa: E402
from build_rxnorm_rrf_index import RxNormOfflineIndex  # noqa: E402

SYM = "TRIỆU_CHỨNG"
ASSERTION_TYPES = {DIAG, SYM, DRUG}
CANDIDATE_TYPES = {DIAG, DRUG}
SUBSTITUTABLE_TYPES = ASSERTION_TYPES

NEG_TEMPLATES = [
    "không có {ent}",
    "không ghi nhận {ent}",
    "chưa phát hiện {ent}",
    "hiện tại không còn {ent}",
]
HIST_TEMPLATES = [
    "tiền sử {ent}",
    "có tiền sử {ent} nhiều năm nay",
    "tiền sử {ent}, đã điều trị ổn định",
    "trước đây từng {ent}",
]
FAMILY_TEMPLATES = [
    "mẹ bệnh nhân có tiền sử {ent}",
    "bố bệnh nhân từng bị {ent}",
    "trong gia đình có người mắc {ent}",
    "anh trai bệnh nhân cũng {ent}",
]
ASSERTION_TEMPLATES = {
    "isNegated": NEG_TEMPLATES,
    "isHistorical": HIST_TEMPLATES,
    "isFamily": FAMILY_TEMPLATES,
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
            shutil.copy2(tmp, path)
            tmp.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"viettelrace-{os.getpid()}-{path.name}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    replace_existing(tmp, path)


class ChainedDrugMatcher:
    """TerminologyMatcher (curated drugs.csv) first, RxNormOfflineIndex
    (data/terminology/rxnorm_full.csv) second. Most synthetic THUỐC
    substitutions now come from rxnorm_drug_names.csv (see build_text_pools),
    names the curated 127-row drugs.csv was never going to cover -- without
    this fallback those synthetic entities would carry candidates=[], which
    is misleading in the written JSONL even though (confirmed against
    train_ner_assertion_model.ipynb) the training loop never reads
    `candidates` at all, only BIO tags/assertions -- this only matters if the
    JSONL is inspected or reused for something else later.
    """

    def __init__(self, primary: TerminologyMatcher, fallback: RxNormOfflineIndex | None):
        self.primary = primary
        self.fallback = fallback

    def lookup(self, text: str) -> list[str]:
        return self.primary.lookup(text) or (self.fallback.lookup(text) if self.fallback else [])


def validate_record(rec: dict[str, Any]) -> None:
    text = rec["text"]
    for ent in rec["entities"]:
        start, end = ent["start"], ent["end"]
        if not (0 <= start <= end <= len(text)) or text[start:end] != ent["text"]:
            raise ValueError(f"{rec['id']}: bad span {ent!r} got {text[start:end]!r}")


def build_text_pools(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    pools: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        for e in rec["entities"]:
            if e["type"] in SUBSTITUTABLE_TYPES:
                pools[e["type"]].add(e["text"])
    for path, typ in [
        (TERM_DIR / "diagnoses.csv", DIAG),
        (TERM_DIR / "drugs.csv", DRUG),
        # ~11.2k real, currently-prescribable ingredient/brand names mined from
        # the RxNorm RRF release (scripts/build_rxnorm_rrf_index.py) -- without
        # this, THUỐC substitution only ever draws from the ~109 drug strings
        # this 100-file corpus happens to contain, so the model can memorize
        # "these specific strings are drugs" instead of learning the span
        # pattern generalizes to any drug name, including ones the private
        # test set introduces that never appear in output/ or drugs.csv.
        (TERM_DIR / "rxnorm_drug_names.csv", DRUG),
    ]:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("text"):
                        pools[typ].add(row["text"])
    return {t: sorted(v) for t, v in pools.items()}


def build_assertion_pool(records: list[dict[str, Any]]) -> list[tuple[str, str, list[str]]]:
    seen: set[tuple[str, str]] = set()
    pool: list[tuple[str, str, list[str]]] = []
    for rec in records:
        for e in rec["entities"]:
            if e["type"] not in ASSERTION_TYPES or e.get("assertions"):
                continue
            key = (e["text"], e["type"])
            if key in seen:
                continue
            seen.add(key)
            pool.append((e["text"], e["type"], e.get("candidates", [])))
    return pool


def substitute_one(
    record: dict[str, Any],
    pools: dict[str, list[str]],
    matchers: dict[str, TerminologyMatcher],
    rng: random.Random,
) -> dict[str, Any] | None:
    entities = record["entities"]
    positions = [i for i, e in enumerate(entities) if e["type"] in SUBSTITUTABLE_TYPES]
    rng.shuffle(positions)
    for i in positions:
        ent = entities[i]
        pool = [t for t in pools.get(ent["type"], []) if t.lower() != ent["text"].lower()]
        if not pool:
            continue
        new_val = rng.choice(pool)
        start, end = ent["start"], ent["end"]
        text = record["text"]
        new_text = text[:start] + new_val + text[end:]
        delta = len(new_val) - (end - start)

        new_entities = []
        for j, e in enumerate(entities):
            e2 = dict(e)
            if j == i:
                e2["start"], e2["end"], e2["text"] = start, start + len(new_val), new_val
                if e2["type"] in CANDIDATE_TYPES:
                    e2["candidates"] = matchers[e2["type"]].lookup(new_val) or e.get("candidates", [])
            elif e["start"] >= end:
                e2["start"] += delta
                e2["end"] += delta
            new_entities.append(e2)
        return {"id": f"{record['id']}-sub{i}-{new_val[:12]}", "text": new_text, "entities": new_entities}
    return None


def synth_assertion_doc(
    pool: list[tuple[str, str, list[str]]],
    matchers: dict[str, TerminologyMatcher],
    rng: random.Random,
    doc_id: str,
    n_clauses: tuple[int, int] = (2, 4),
) -> dict[str, Any] | None:
    n = rng.randint(*n_clauses)
    if len(pool) < n:
        return None
    chosen = rng.sample(pool, k=n)

    doc_text = ""
    entities: list[dict[str, Any]] = []
    for text_, type_, candidates in chosen:
        assertion = rng.choice(list(ASSERTION_TEMPLATES))
        template = rng.choice(ASSERTION_TEMPLATES[assertion])
        prefix, _, suffix = template.partition("{ent}")
        clause = prefix + text_ + suffix
        clause = clause[0].upper() + clause[1:] + "."

        if doc_text:
            doc_text += " "
        base = len(doc_text)
        doc_text += clause
        start, end = base + len(prefix), base + len(prefix) + len(text_)

        ent: dict[str, Any] = {"start": start, "end": end, "type": type_, "text": text_, "assertions": [assertion]}
        if type_ in CANDIDATE_TYPES:
            ent["candidates"] = candidates or matchers[type_].lookup(text_)
        entities.append(ent)

    return {"id": doc_id, "text": doc_text, "entities": entities}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multiplier", type=int, default=1, help="substitution variants generated per source doc")
    parser.add_argument("--assertion-docs", type=int, default=30, help="number of synthetic assertion-cue documents")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    train_path = DATASET_DIR / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(f"{train_path} not found -- run scripts/prepare_ner_dataset.py first.")

    rng = random.Random(args.seed)
    train_records = load_jsonl(train_path)
    rxnorm_csv = TERM_DIR / "rxnorm_full.csv"
    matchers = {
        DIAG: TerminologyMatcher(TERM_DIR / "diagnoses.csv"),
        DRUG: ChainedDrugMatcher(
            TerminologyMatcher(TERM_DIR / "drugs.csv"),
            RxNormOfflineIndex(rxnorm_csv) if rxnorm_csv.exists() else None,
        ),
    }
    pools = build_text_pools(train_records)

    synthetic: list[dict[str, Any]] = []

    for rec in train_records:
        seen_texts = {rec["text"]}
        made = 0
        attempts = 0
        while made < args.multiplier and attempts < args.multiplier * 6:
            attempts += 1
            variant = substitute_one(rec, pools, matchers, rng)
            if variant is None:
                break
            if variant["text"] in seen_texts:
                continue
            seen_texts.add(variant["text"])
            validate_record(variant)
            synthetic.append(variant)
            made += 1

    assertion_pool = build_assertion_pool(train_records)
    for k in range(args.assertion_docs):
        doc = synth_assertion_doc(assertion_pool, matchers, rng, doc_id=f"synth-assert-{k}")
        if doc is None:
            continue
        validate_record(doc)
        synthetic.append(doc)

    out_records = train_records + synthetic
    write_jsonl(DATASET_DIR / "train_augmented.jsonl", out_records)

    n_sub = sum(1 for r in synthetic if str(r["id"]).find("-sub") != -1)
    print(f"train.jsonl: {len(train_records)} real docs")
    print(f"synthetic: {len(synthetic)} ({n_sub} substitution, {len(synthetic) - n_sub} assertion-cue)")
    print(f"wrote {len(out_records)} total docs -> {DATASET_DIR / 'train_augmented.jsonl'}")
    print("\nTo use it: upload train_augmented.jsonl to the Kaggle Dataset in place of train.jsonl")
    print("(keep holdout.jsonl as-is -- it must stay real-only for a meaningful generalization estimate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
