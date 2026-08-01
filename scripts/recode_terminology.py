#!/usr/bin/env python3
"""Apply recoded candidates to the linking tables, with hard safety checks.

This is the only sanctioned way to edit `data/terminology/{diagnoses,drugs}.csv`
after the offline recoding pass (see docs/linking_recode.md). It exists because
the change it applies is the highest-leverage and highest-blast-radius edit in
the repo, and two earlier table/linking edits both lost points on the real
leaderboard (31.89, 33.679).

Invariants enforced, all of them for a measured reason:

1. **The `text` column may not change.** `run_pipeline.py --add-terminology-entities`
   generates diagnosis/drug spans *by scanning the input for these strings*, so
   editing `text` silently changes the predicted entity set -- the lane that has
   scored negative every time it was touched. Only `candidate` may move.
2. **Every code must exist in the official vocabulary** (`icd10_vi.csv` for
   CHẨN_ĐOÁN, `rxnorm_full.csv` for THUỐC). This is what stops an LLM-proposed
   code from reaching a submission.
3. **Every code must be at the level the catalog actually carries.** The grader
   compares exact strings, so a 3-character category scores zero wherever the
   truth uses a subcode. A 3-char code is promoted to its `.9` child when the
   BYT catalog has one, and accepted as-is when the category is terminal --
   19 of the 26 3-char codes in `diagnoses.csv` (`I10`, `N19`, `J91`, `J47`)
   are terminal and correct.
4. **No row may lose its only code.** An entity whose candidates end up empty is
   deleted outright by `filter_noisy_entities`, taking its text and assertions
   with it -- so a failed recode must fall back to the current code, never to
   nothing.

Usage:
  python scripts/recode_terminology.py --proposed data/terminology/recode_proposed.csv --dry-run
  python scripts/recode_terminology.py --proposed data/terminology/recode_proposed.csv

`--proposed` needs columns: text, type, proposed_candidate [, proposed_title, score]
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
TERM = ROOT / "data" / "terminology"

DIAG = "CHẨN_ĐOÁN"
DRUG = "THUỐC"
TABLES = {DIAG: "diagnoses.csv", DRUG: "drugs.csv"}

ICD4_RE = re.compile(r"^[A-Z]\d{2}\.\d+$")
ICD3_RE = re.compile(r"^[A-Z]\d{2}$")
RXCUI_RE = re.compile(r"^\d+$")

# Chapters that are never a bare diagnosis here; confirmed against the 542
# graded turn-1 diagnosis codes (worklog 2026-07-28).
EXCLUDED_CHAPTERS = set("UVWXY")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_icd() -> tuple[set[str], dict[str, str]]:
    """Valid ICD codes, plus category -> its '.9' (or lowest unspecified) child."""
    valid: set[str] = set()
    children: dict[str, list[str]] = defaultdict(list)
    path = TERM / "icd10_vi.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run scripts/build_icd10_vi_index.py first.")
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["code"].strip()
            valid.add(code)
            if ICD4_RE.match(code):
                children[code[:3]].append(code)
    promote = {}
    for category, kids in children.items():
        nine = [k for k in kids if k.endswith(".9")]
        promote[category] = nine[0] if nine else sorted(kids)[0]
    return valid, promote


def load_rxcui() -> set[str]:
    path = TERM / "rxnorm_full.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run scripts/build_rxnorm_rrf_index.py first.")
    with path.open(encoding="utf-8") as f:
        return {row["candidate"].strip() for row in csv.DictReader(f)}


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_icd_children() -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    with (TERM / "icd10_vi.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["code"].strip()
            if "." in code:
                children[code[:3]].append(code)
    return children


def nearest_catalog_code(code: str, valid: set[str], children: dict[str, list[str]]) -> str:
    """Map a code the BYT catalog does not carry onto the closest one it does.

    Two distinct causes, one fix. Some are ICD-10-**CM** (the US clinical
    modification): `S06.4X9A`, `G31.84`, `L89.94`, `I73.89`. Others are perfectly
    valid WHO ICD-10 that the BYT catalog simply omits: `I31.4` (cardiac
    tamponade), `E87.6` (hypokalaemia), `I49.1`, `K58.9`, `N40.0`, `C64.9`.

    Which of the two it is does not matter. What matters is membership in the
    vocabulary the graders used, so the rule is the same: walk up to the longest
    prefix the catalog does carry, then, if that lands on a category with
    children, take its unspecified subcode -- `.9`, else `.8`, else the category
    itself when it is terminal (`J47`, `N40`, `L97`, `C64` all are).
    """
    for cut in range(len(code), 2, -1):
        prefix = code[:cut].rstrip(".")
        if prefix in valid:
            if len(prefix) == 3 and children.get(prefix):
                kids = children[prefix]
                for suffix in (".9", ".8"):
                    match = [k for k in kids if k.endswith(suffix)]
                    if match:
                        return match[0]
                return sorted(kids)[0]
            return prefix
    return ""


def audit(icd_valid: set[str], icd_promote: dict[str, str], out_path: Path) -> int:
    """Find codes that cannot score, and auto-fix the deterministic ones.

    Two failure classes, both found by measurement on 2026-07-31:

    * **3-character category codes.** The grader compares exact strings, and
      where a category has subcodes the truth carries the 4-character one, so
      `I48` scores zero while `I48.9` can score. Only categories that actually
      have children are promoted -- 19 of the 26 3-char codes in `diagnoses.csv`
      (`I10`, `N19`, `J91`, ...) are *terminal* in the BYT catalog and correct.
    * **Codes that are not in the BYT catalog at all.** 15 rows. Some are
      ICD-10-**CM** (`S06.4X9A`, `G31.84`, `L89.94`, `I73.89`); the rest are
      valid WHO ICD-10 that BYT simply omits (`I31.4`, `E87.6`, `I49.1`,
      `K58.9`, `N40.0`, `C64.9`). Either way they are guaranteed zeros against a
      BYT-derived answer key, and `nearest_catalog_code()` maps all 15 onto a
      catalog ancestor deterministically. Anything it cannot map is written with
      `needs_review=1` and never auto-applied.
    """
    children = load_icd_children()
    with (TERM / "icd10_vi.csv").open(encoding="utf-8") as f:
        titles = [(r["code"].strip(), norm(r["name_vi"])) for r in csv.DictReader(f)]

    freq: dict[str, int] = {}
    worklist = TERM / "recode_worklist.csv"
    if worklist.exists():
        with worklist.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["type"] == DIAG:
                    freq[norm(row["text"])] = int(row["freq"])

    auto: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    for row in read_table(TERM / TABLES[DIAG]):
        code, text = row["candidate"].strip(), row["text"]
        count = freq.get(norm(text), 0)
        if ICD3_RE.match(code) and icd_promote.get(code):
            auto.append({"text": text, "type": DIAG, "proposed_candidate": icd_promote[code],
                         "score": "1.0", "reason": f"3-char {code} -> subcode", "needs_review": "0",
                         "freq": str(count)})
            continue
        if code in icd_valid:
            continue
        # Falls through to the invalid-code branch below. A 3-char code with no
        # children AND no catalog entry used to be skipped here -- that let `J91`
        # ("tràn dịch màng phổi") survive an audit that reported zero problems.
        # BYT omits J91 entirely (a WHO dagger/asterisk code); the answer is J90.
        mapped = nearest_catalog_code(code, icd_valid, children)
        by_title = ""
        if not mapped:
            key = norm(text)
            hits = [c for c, title in titles if title == key or title.startswith(key + ",")]
            by_title = hits[0] if len(hits) == 1 else ""
        if mapped:
            auto.append({"text": text, "type": DIAG, "proposed_candidate": mapped,
                         "score": "1.0", "reason": f"{code} absent from BYT catalog -> nearest ancestor",
                         "needs_review": "0", "freq": str(count)})
        else:
            review.append({"text": text, "type": DIAG, "proposed_candidate": by_title,
                           "score": "0.0",
                           "reason": f"{code} absent, no ancestor"
                                     + (f"; title match -> {by_title}" if by_title else "; no title match"),
                           "needs_review": "1", "freq": str(count)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["text", "type", "proposed_candidate", "score", "reason", "needs_review", "freq"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(auto + review)

    auto_mentions = sum(int(r["freq"]) for r in auto)
    review_mentions = sum(int(r["freq"]) for r in review)
    print(f"Deterministic fixes : {len(auto):>3} rows, {auto_mentions:>3} mentions")
    print(f"Needs human review  : {len(review):>3} rows, {review_mentions:>3} mentions")
    print(f"\nWrote {out_path}")
    print("Review the needs_review=1 rows, then apply with:")
    print(f"  python scripts/recode_terminology.py --proposed {out_path} --dry-run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proposed", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Ignore proposals below this score, if the file carries one.")
    parser.add_argument(
        "--audit",
        type=Path,
        nargs="?",
        const=TERM / "recode_autofix.csv",
        help="Report unscoreable codes in the current tables and write the deterministic fixes "
        "to this file. No GPU, no model -- run this before the offline recoding pass.",
    )
    args = parser.parse_args()

    icd_valid, icd_promote = load_icd()
    rxcui_valid = load_rxcui()

    if args.audit is not None:
        return audit(icd_valid, icd_promote, args.audit)
    if args.proposed is None:
        raise SystemExit("Need --proposed <file> (or --audit).")

    proposals: dict[tuple[str, str], str] = {}
    rejected = defaultdict(int)
    with args.proposed.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (norm(row["text"]), row["type"].strip())
            code = (row.get("proposed_candidate") or "").strip()
            if not code:
                rejected["empty"] += 1
                continue
            try:
                if float(row.get("score") or 1.0) < args.min_score:
                    rejected["low_score"] += 1
                    continue
            except ValueError:
                pass

            if key[1] == DIAG:
                if ICD3_RE.match(code):
                    promoted = icd_promote.get(code)
                    if promoted:
                        code = promoted
                    elif code in icd_valid:
                        pass  # terminal category (J47, N40, L97, C64): correct as-is
                    else:
                        rejected["icd_3char_unknown"] += 1
                        continue
                if not (ICD4_RE.match(code) or (ICD3_RE.match(code) and code in icd_valid)):
                    rejected["icd_bad_format"] += 1
                    continue
                if code[0] in EXCLUDED_CHAPTERS:
                    rejected["icd_excluded_chapter"] += 1
                    continue
                if code not in icd_valid:
                    rejected["icd_not_in_catalog"] += 1
                    continue
            elif key[1] == DRUG:
                if not RXCUI_RE.match(code):
                    rejected["rxcui_bad_format"] += 1
                    continue
                if code not in rxcui_valid:
                    rejected["rxcui_not_in_rxnorm"] += 1
                    continue
            else:
                rejected["unknown_type"] += 1
                continue
            proposals[key] = code

    print(f"Accepted {len(proposals)} proposals; rejected {sum(rejected.values())} {dict(rejected)}\n")

    total_changed = 0
    for ent_type, filename in TABLES.items():
        path = TERM / filename
        rows = read_table(path)
        original_texts = [r["text"] for r in rows]
        changed = 0
        out_rows: list[dict[str, str]] = []
        for row in rows:
            key = (norm(row["text"]), ent_type)
            new_code = proposals.get(key)
            if new_code and new_code != row["candidate"]:
                out_rows.append({"text": row["text"], "candidate": new_code, "source": "recoded_biencoder"})
                changed += 1
            else:
                out_rows.append(row)

        # Invariant 1: the join key must be byte-identical, in order.
        if [r["text"] for r in out_rows] != original_texts:
            raise SystemExit(f"ABORT: {filename} text column changed. This would move the entity set.")
        # Invariant 4: nothing may end up without a code.
        if any(not r["candidate"] for r in out_rows):
            raise SystemExit(f"ABORT: {filename} has a row with no candidate.")

        print(f"{filename:<16} {len(rows):>4} rows, {changed:>4} codes changed ({changed / max(len(rows), 1):.1%})")
        total_changed += changed

        if not args.dry_run and changed:
            # Only back up when something actually changes. Re-running an applied
            # proposal is a no-op, and overwriting .csv.bak with the already-fixed
            # table would silently destroy the only local copy of the original.
            shutil.copy2(path, path.with_suffix(".csv.bak"))
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["text", "candidate", "source"])
                writer.writeheader()
                for row in out_rows:
                    writer.writerow({k: row.get(k, "") for k in ("text", "candidate", "source")})

    if args.dry_run:
        print("\n(dry run, nothing written)")
    else:
        print(f"\nWrote {total_changed} code changes; .csv.bak backups kept next to each table.")
        print("Now regenerate predictions with the SAME recipe as the baseline, then diff:")
        print("  python scripts/run_pipeline.py --input input_turn2 --pred experiments/v5_recoded \\")
        print("      --no-icd-fallback --drop-short-noise --add-terminology-entities --add-public-phrase-entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
