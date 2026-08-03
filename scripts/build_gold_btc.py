#!/usr/bin/env python3
"""Rebuild (and verify) the only real ground truth in this repo.

The task statement publishes a complete worked example -- one document, 19
entities, with exact character offsets, types, assertions and codes. Nothing
used it until 2026-08-02. It is the only offline eval here with predictive
validity: the model holdout is contaminated (`train_holdout_overlap: true`,
holdout WER 0.006) and every other eval scores against our own labels.

Why this is a script and not just two committed files
-----------------------------------------------------
The document's separator is `\\n\\r\\n`, and the published offsets only line up
if it is preserved byte-for-byte. That does not survive normal file handling on
Windows: writing it with `Path.write_text` translates `\\n` to `\\r\\n`, which
grows each of the 11 separators by 2 characters and shifts every offset -- the
checked-in copy scored 19/19 while the working-tree copy scored 0/19, silently.

So the file is reconstructed from `docs/problem_statement.md` on demand, written
with newline translation disabled, and **every one of the 19 offsets is asserted
before anything is written**. A broken gold set now fails loudly instead of
quietly reporting wrong numbers.

Usage:
  python scripts/build_gold_btc.py            # rebuild data/corpus/gold_btc/
  python scripts/build_gold_btc.py --verify   # check the existing copy, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "problem_statement.md"
DEST = ROOT / "data" / "corpus" / "gold_btc"

# The header line, then 11 numbered drug items. Candidates are ordered so the
# first exact hit wins; '\n\r\n' is the one that reproduces all 19 offsets.
SEPARATORS = ["\n\r\n", "\n\n", "\r\n\r\n", "\n", "\r\n", " \n", "\n "]


def parse_statement() -> tuple[str, list[dict]]:
    doc = SOURCE.read_text(encoding="utf-8")
    block = re.search(r"\*\*Input:\*\*\n((?:> .*\n)+)", doc)
    payload = re.search(r"\*\*Output:\*\*\n```json\n(\[.*?\])\n```", doc, re.S)
    if not block or not payload:
        raise SystemExit(f"Could not find the worked example in {SOURCE}")

    flat = " ".join(line[2:].strip() for line in block.group(1).strip().split("\n"))
    flat = re.sub(r"\s+", " ", flat).strip("` ").strip()
    gold = json.loads(payload.group(1))

    header = "Danh sách thuốc trước nhập viện chính xác và đầy đủ."
    items = re.split(r"\s(?=\d{1,2}\.\s)", flat[len(header):].strip())

    for separator in SEPARATORS:
        text = header + separator + separator.join(items)
        if all(text[e["position"][0]:e["position"][1]] == e["text"] for e in gold):
            return text, gold

    raise SystemExit("No separator reproduces the published offsets; the statement may have changed.")


def offsets_ok(text: str, gold: list[dict]) -> tuple[int, list[str]]:
    good = 0
    problems: list[str] = []
    for entity in gold:
        start, end = entity["position"]
        actual = text[start:end]
        if actual == entity["text"]:
            good += 1
        else:
            problems.append(f"{entity['position']} expected {entity['text']!r}, got {actual!r}")
    return good, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="Check the existing files, write nothing.")
    args = parser.parse_args()

    text, gold = parse_statement()

    if args.verify:
        input_path = DEST / "input" / "1.txt"
        if not input_path.exists():
            raise SystemExit(f"{input_path} is missing; run without --verify to build it.")
        # newline="" so the bytes are read exactly as stored.
        with input_path.open(encoding="utf-8", newline="") as f:
            on_disk = f.read()
        good, problems = offsets_ok(on_disk, gold)
        print(f"{input_path}: {len(on_disk)} chars, {good}/{len(gold)} offsets match")
        for problem in problems[:5]:
            print(f"  {problem}")
        if good != len(gold):
            print("\nBroken -- rebuild with: python scripts/build_gold_btc.py")
            return 1
        return 0

    (DEST / "input").mkdir(parents=True, exist_ok=True)
    (DEST / "truth").mkdir(parents=True, exist_ok=True)
    # newline="" disables the \n -> \r\n translation that silently shifted every
    # offset the first time this was built on Windows.
    with (DEST / "input" / "1.txt").open("w", encoding="utf-8", newline="") as f:
        f.write(text)
    with (DEST / "truth" / "1.json").open("w", encoding="utf-8", newline="") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

    good, _ = offsets_ok(text, gold)
    print(f"Wrote {DEST} -- {len(text)} chars, {len(gold)} entities, {good}/{len(gold)} offsets match")
    print("\nScore the pipeline against it with:")
    print("  python scripts/run_pipeline.py --input data/corpus/gold_btc/input --pred experiments/gold \\")
    print("      --no-icd-fallback --drop-short-noise --add-terminology-entities --add-public-phrase-entities")
    print("  python scripts/check_submission.py --pred experiments/gold \\")
    print("      --input data/corpus/gold_btc/input --truth data/corpus/gold_btc/truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
