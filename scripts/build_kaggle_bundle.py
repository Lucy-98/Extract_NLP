#!/usr/bin/env python3
"""Assemble the single Kaggle dataset the training/submission notebook consumes.

Why this exists
---------------
The bundle used to be built by hand from a prose checklist in the notebook, and
that checklist silently went stale: it listed
``data/terminology/{drugs,diagnoses,rxnorm_full}.csv`` and never gained
``icd10_vi.csv`` after the ICD-10 linking landed (worklog 2026-07-28). A bundle
missing that file does not fail -- ``run_pipeline.py`` warns and falls back to
curated-table-only diagnosis linking, so the Kaggle-side submission quietly
scores as if the change had never been made. Same trap for a stale ``scripts/``
copy: the kernel runs the *bundled* code, not the repo you edited.

So the checklist is executable now, and every required member is verified after
the copy rather than trusted.

stdlib-only, like every script here except run_pipeline.py.

Usage:
  python scripts/build_kaggle_bundle.py            # -> kaggle_bundle/ (gitignored)
  python scripts/build_kaggle_bundle.py --dry-run  # report what would be copied
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "kaggle_bundle"

# (source, destination-relative-to-bundle). Directories are copied whole.
MEMBERS: list[tuple[str, str]] = [
    # Kaggle requires this at the dataset root for `datasets version`; it is the
    # single source of the dataset slug, kept in kaggle_upload/dataset/ next to
    # the kernel metadata so both point at the same account.
    ("kaggle_upload/dataset/dataset-metadata.json", "dataset-metadata.json"),
    ("scripts", "scripts"),
    ("data/terminology/drugs.csv", "data/terminology/drugs.csv"),
    ("data/terminology/diagnoses.csv", "data/terminology/diagnoses.csv"),
    ("data/terminology/conflicts.txt", "data/terminology/conflicts.txt"),
    ("data/terminology/rxnorm_full.csv", "data/terminology/rxnorm_full.csv"),
    # Vietnamese ICD-10 linking (2026-07-28). Omitting this does NOT fail loudly
    # -- it silently degrades J_candidates. See the module docstring.
    ("data/terminology/icd10_vi.csv", "data/terminology/icd10_vi.csv"),
    # English ICD-10, used by the notebook to validate Qwen-generated codes.
    ("data/terminology/icd10_full.csv", "data/terminology/icd10_full.csv"),
    ("data/ner_dataset/train_augmented.jsonl", "train.jsonl"),
    ("data/ner_dataset/holdout.jsonl", "holdout.jsonl"),
    ("output", "output"),          # turn-1 curated labels: training + phrase lexicon
    ("input", "input"),            # turn-1 inputs, for offsets/lexicon building
    ("input_turn2", "input_turn2"),  # the documents actually being submitted
]

# Nothing in scripts/ needs these, and they bloat the upload.
EXCLUDED_NAMES = {"__pycache__", ".ipynb_checkpoints"}

# Files whose absence would silently change the submission rather than crash it.
CRITICAL = [
    "scripts/run_pipeline.py",
    "scripts/build_icd10_vi_index.py",
    "scripts/build_rxnorm_rrf_index.py",
    "scripts/build_terminology_index.py",
    "scripts/package_submission.py",
    "data/terminology/icd10_vi.csv",
    "data/terminology/rxnorm_full.csv",
    "data/terminology/drugs.csv",
    "data/terminology/diagnoses.csv",
]


def ignored(path: Path) -> bool:
    return any(part in EXCLUDED_NAMES for part in path.parts) or path.suffix == ".pyc"


def copy_member(src: Path, dst: Path) -> tuple[int, int]:
    """Copy a file or directory into the bundle. Returns (files, bytes)."""
    if src.is_dir():
        files = bytes_ = 0
        for child in sorted(src.rglob("*")):
            if not child.is_file() or ignored(child.relative_to(src)):
                continue
            target = dst / child.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
            files += 1
            bytes_ += target.stat().st_size
        return files, bytes_
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1, dst.stat().st_size


def verify(out_dir: Path) -> list[str]:
    """Post-copy verification. A bundle that is merely *present* is not enough:
    a stale scripts/ copy is the failure mode that cost a submission before, so
    compare content against the repo rather than just checking existence."""
    problems: list[str] = []
    for rel in CRITICAL:
        bundled = out_dir / rel
        if not bundled.exists():
            problems.append(f"missing required member: {rel}")
            continue
        source = ROOT / rel
        if source.exists() and not filecmp.cmp(source, bundled, shallow=False):
            problems.append(f"bundled copy differs from repo: {rel}")
    turn2 = out_dir / "input_turn2"
    n_turn2 = len(list(turn2.glob("*.txt"))) if turn2.is_dir() else 0
    if n_turn2 == 0:
        problems.append("input_turn2/ has no .txt files -- nothing to submit")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="List members and sizes without writing the bundle.")
    args = parser.parse_args()

    missing = [src for src, _dst in MEMBERS if not (ROOT / src).exists()]
    if missing:
        print("Cannot build bundle, missing from the repo:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        if "data/ner_dataset/train_augmented.jsonl" in missing:
            print("  (run: python scripts/run_all.py prepare)", file=sys.stderr)
        return 1

    if args.dry_run:
        total = 0
        for src, dst in MEMBERS:
            path = ROOT / src
            size = (sum(f.stat().st_size for f in path.rglob("*")
                        if f.is_file() and not ignored(f.relative_to(path)))
                    if path.is_dir() else path.stat().st_size)
            total += size
            print(f"  {src:44} -> {dst:34} {size / 1e6:8.1f} MB")
        print(f"\ntotal {total / 1e6:.1f} MB (dry run, nothing written)")
        return 0

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    files = total = 0
    for src, dst in MEMBERS:
        n, b = copy_member(ROOT / src, args.out / dst)
        files += n
        total += b
        print(f"  {src:44} -> {dst:34} {n:5} file(s) {b / 1e6:8.1f} MB")

    problems = verify(args.out)
    if problems:
        print("\nBUNDLE VERIFICATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"\nWrote {args.out} — {files} files, {total / 1e6:.1f} MB. Verified "
          f"{len(CRITICAL)} critical members match the repo.")
    print("Next: upload as a Kaggle dataset version, then push the kernel:")
    print("  python -m kaggle datasets version -p kaggle_bundle -m \"<message>\"")
    print("  python -m kaggle kernels push -p kaggle_upload/kernel --accelerator NvidiaTeslaT4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
