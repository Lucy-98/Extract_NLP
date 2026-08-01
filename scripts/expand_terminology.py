#!/usr/bin/env python3
"""Seed the linking tables from the official vocabularies -- MEASURED, REJECTED.

VERDICT (2026-07-31): do not ship this. Run `--report` to reproduce the numbers.

  ICD:    11,033 catalog titles pass the gates below; exactly **35** of them occur
          verbatim anywhere in the 200 turn-1 + turn-2 documents.
  RxNorm: 10,400 clean names; 98 occur, 60 are already in drugs.csv, and of the
          38 remaining only ~10 are drugs the pipeline is currently missing --
          the rest are lab analytes (creatinine, glucose, cholesterol, lactate,
          magnesium, fibrinogen...). Adding those to drugs.csv would tag them
          THUỐC when truth says TÊN_XÉT_NGHIỆM, and the task statement counts a
          wrong type **twice with zero on all three metrics**.

Root cause, and it is not fixable by better gating: **official catalog language
is not clinical language.** "bệnh tả do vi khuẩn vibrio cholerae 01, típ sinh
học cholerae" is a BYT title; no doctor writes it. Exact matching therefore has
near-zero recall, and the loose matching that does have recall is exactly what
scored 34.388 -> 33.679 on 2026-07-31. Closing this gap needs a paraphrase layer
(an LLM-generated synonym corpus over the 15k titles, trained into a
bi-encoder), not another lookup heuristic.

The generator below is kept only so the measurement is reproducible and so the
next person does not spend a day rediscovering this.

Why the tables look the way they do
-----------------------------------
Measured on the reverted turn-2 run (`experiments/v1_revert_icd`):

    798 CHẨN_ĐOÁN  ->    161 distinct ICD codes   (icd10_vi.csv has 15,144 rows -> 1.06% used)
    216 THUỐC      ->     72 distinct RxNorm codes (rxnorm_full.csv has 517,991 -> 0.014% used)

and of those 798 diagnoses, **94.1% were answered by the exact tier** of
`diagnoses.csv` -- a 321-row table mined from `output/`, i.e. from our own
turn-1 submission, the one that scored `J_candidates` **29.98**. The 86 rows
labelled `icd10_alias` in that table are a hardcoded dict in
`build_terminology_index.py`, not the BYT catalog.

So the pipeline is closed over ~321 turn-1 strings: `--add-terminology-entities`
generates diagnosis spans *from* the table, and the table then answers them
exactly. On unseen text that loop has almost no coverage.

How this differs from the falsified 2026-07-28 change
-----------------------------------------------------
That change used `ICD10VietnameseIndex`'s **token-subset** fallback, where a
short query always finds *some* title among 15k ("nhiễm trùng" -> A31.9,
"bệnh lây truyền" -> A56.2). It rescued 78 junk `CHẨN_ĐOÁN` from the noise
filter and the leaderboard went 34.388 -> 33.679.

This script adds **whole-title exact rows** instead. A match means the input
literally contains an official BYT diagnosis title of >= MIN_WORDS words. There
is no partial-credit tier, so the failure mode that produced A31.9 cannot occur.

It is still an entity-set change, and every entity-set change tried so far has
scored negative. Treat the output as a bet, ship it as its own variant, and
falsify it on the leaderboard -- do not re-tune the gates offline.

Usage:
  python scripts/expand_terminology.py --dry-run
  python scripts/expand_terminology.py --out-dir data/terminology_expanded
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
TERM = ROOT / "data" / "terminology"

# External causes (V-Y), special purposes (U) and "factors influencing health
# status" (Z) are never a bare diagnosis in this corpus. The 2026-07-28 audit
# confirmed not one of the 542 graded turn-1 diagnosis codes falls in V-Y or U.
EXCLUDED_CHAPTERS = set("UVWXYZ")

# A 1-2 word official title ("Bệnh tả", "Sốt") is short enough to appear inside
# unrelated prose; >= 3 words is specific enough that a literal occurrence is
# almost certainly that diagnosis. 14,959 of 15,144 titles clear this.
MIN_WORDS = 3

# The grader compares codes as exact strings and truth uses 4-character codes,
# so 3-character category rows are dropped rather than emitted as-is.
ICD4_RE = re.compile(r"^[A-Z]\d{2}\.\d+$")

MIN_DRUG_CHARS = 6


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def read_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {norm(r["text"]) for r in csv.DictReader(f)}


def expand_diagnoses(existing: set[str]) -> list[tuple[str, str, str]]:
    src = TERM / "icd10_vi.csv"
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set(existing)
    skipped = {"chapter": 0, "short": 0, "not4char": 0, "dup": 0}
    with src.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code, name = r["code"].strip(), r["name_vi"].strip()
            if code[:1] in EXCLUDED_CHAPTERS:
                skipped["chapter"] += 1
                continue
            if not ICD4_RE.match(code):
                skipped["not4char"] += 1
                continue
            if len(name.split()) < MIN_WORDS:
                skipped["short"] += 1
                continue
            key = norm(name)
            if key in seen:
                skipped["dup"] += 1
                continue
            seen.add(key)
            rows.append((key, code, "byt_icd10_title"))
    return rows, skipped


def expand_drugs(existing: set[str]) -> list[tuple[str, str, str]]:
    """RxNorm ingredient/brand names -> RXCUI, via the full offline index.

    `rxnorm_drug_names.csv` is a name list with no codes; it is currently read
    only by `augment_ner_dataset.py` to diversify synthetic training text. The
    codes come from `rxnorm_full.csv`, and only names that resolve to exactly
    one RXCUI are kept -- an ambiguous name is precisely the case the dose+form
    logic in run_pipeline.py exists to handle, and hardcoding one code here
    would pre-empt it.
    """
    names_path = TERM / "rxnorm_drug_names.csv"
    full_path = TERM / "rxnorm_full.csv"
    if not names_path.exists() or not full_path.exists():
        return [], {"missing_source": 1}

    by_text: dict[str, set[str]] = {}
    with full_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_text.setdefault(norm(r["text"]), set()).add(r["candidate" if "candidate" in r else "rxcui"])

    rows: list[tuple[str, str, str]] = []
    seen = set(existing)
    skipped = {"short": 0, "no_code": 0, "ambiguous": 0, "dup": 0}
    with names_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = norm(r["text"])
            if len(key) < MIN_DRUG_CHARS or not re.search(r"[a-z]", key):
                skipped["short"] += 1
                continue
            if key in seen:
                skipped["dup"] += 1
                continue
            codes = by_text.get(key)
            if not codes:
                skipped["no_code"] += 1
                continue
            if len(codes) > 1:
                skipped["ambiguous"] += 1
                continue
            seen.add(key)
            rows.append((key, next(iter(codes)), "rxnorm_prescribable"))
    return rows, skipped


def write_table(src: Path, extra: list[tuple[str, str, str]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["text", "candidate", "source"])
        if src.exists():
            with src.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    writer.writerow([r["text"], r["candidate"], r.get("source", "")])
        for row in extra:
            writer.writerow(row)


def report_corpus_coverage() -> int:
    """How many candidate rows would ever fire? This is what rejected the idea."""
    import glob

    corpus = norm(" ".join(
        Path(p).read_text(encoding="utf-8")
        for p in glob.glob(str(ROOT / "input" / "*.txt")) + glob.glob(str(ROOT / "input_turn2" / "*.txt"))
    ))

    eligible = 0
    icd_hits: list[tuple[str, str]] = []
    with (TERM / "icd10_vi.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code, name = r["code"].strip(), norm(r["name_vi"])
            if code[:1] in EXCLUDED_CHAPTERS or not ICD4_RE.match(code) or len(name.split()) < MIN_WORDS:
                continue
            eligible += 1
            if name in corpus:
                icd_hits.append((name, code))

    have = read_existing(TERM / "drugs.csv")
    names = [
        norm(r["text"]) for r in csv.DictReader((TERM / "rxnorm_drug_names.csv").open(encoding="utf-8"))
    ]
    clean = [x for x in names if len(x) >= MIN_DRUG_CHARS and re.fullmatch(r"[a-z][a-z0-9 \-/]+", x)]
    drug_hits = [x for x in clean if x in corpus]
    drug_new = [x for x in drug_hits if x not in have]

    print("Corpus coverage of the official vocabularies (input/ + input_turn2/, 200 docs)\n")
    print(f"  ICD titles passing the gates      : {eligible:>6}")
    print(f"    occurring verbatim in corpus    : {len(icd_hits):>6}  <- the whole upside")
    print(f"  RxNorm names passing the gates    : {len(clean):>6}")
    print(f"    occurring verbatim in corpus    : {len(drug_hits):>6}")
    print(f"    not already in drugs.csv        : {len(drug_new):>6}")
    print(f"\n  Sample ICD hits: {[c for _n, c in icd_hits[:8]]}")
    print(f"  Sample new drug names: {sorted(drug_new)[:12]}")
    print("\nVerdict: not worth an entity-set change. See this file's module docstring.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "terminology_expanded")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Reproduce the falsification: how many candidate rows actually occur in the corpus.",
    )
    args = parser.parse_args()

    if args.report:
        return report_corpus_coverage()

    diag_existing = read_existing(TERM / "diagnoses.csv")
    drug_existing = read_existing(TERM / "drugs.csv")

    diag_rows, diag_skip = expand_diagnoses(diag_existing)
    drug_rows, drug_skip = expand_drugs(drug_existing)

    print(f"diagnoses.csv  {len(diag_existing):>6} -> {len(diag_existing) + len(diag_rows):>6} "
          f"(+{len(diag_rows)} from icd10_vi.csv)")
    print(f"   skipped: {diag_skip}")
    print(f"drugs.csv      {len(drug_existing):>6} -> {len(drug_existing) + len(drug_rows):>6} "
          f"(+{len(drug_rows)} from rxnorm_drug_names.csv)")
    print(f"   skipped: {drug_skip}")

    if args.dry_run:
        print("\nSample new diagnosis rows:")
        for row in diag_rows[:8]:
            print(f"   {row[0][:60]!r:<64} {row[1]}")
        print("\nSample new drug rows:")
        for row in drug_rows[:8]:
            print(f"   {row[0][:60]!r:<64} {row[1]}")
        print("\n(dry run, nothing written)")
        return 0

    write_table(TERM / "diagnoses.csv", diag_rows, args.out_dir / "diagnoses.csv")
    write_table(TERM / "drugs.csv", drug_rows, args.out_dir / "drugs.csv")
    for name in ("conflicts.txt", "icd10_vi.csv", "icd10_full.csv", "rxnorm_full.csv", "rxnorm_drug_names.csv"):
        src = TERM / name
        if src.exists():
            (args.out_dir / name).write_bytes(src.read_bytes())
    print(f"\nWrote expanded tables to {args.out_dir}")
    print("Run the pipeline against them by pointing TERM_DIR at that folder, e.g.:")
    print(f"  VTR_TERM_DIR={args.out_dir} python scripts/run_pipeline.py --input input_turn2 ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
