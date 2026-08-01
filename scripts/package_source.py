#!/usr/bin/env python3
"""Package the reproducible source bundle requested from top teams.

This is separate from `package_submission.py`: `output.zip` is only the JSON
prediction artifact, while this source bundle contains code, derived data,
and model weights needed to rerun inference on a private test folder.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INCLUDE_ROOTS = [
    "README.md",
    "CLAUDE.md",
    "requirements.txt",
    "scripts",
    "notebooks",
    "kaggle_upload/kernel",
    "kaggle_upload/dataset",
    "data",
    "models/ner_model",
    "input",
    "output",
]

EXCLUDED_PARTS = {
    ".git",
    ".agents",
    ".agent_runs",
    ".codex",
    ".claude",
    ".kaggle_download",
    ".ipynb_checkpoints",
    "__pycache__",
    "venv",
    ".venv",
    "rrf",
    "output_model",
    "run_output",
}

EXCLUDED_SUFFIXES = {".pyc", ".zip"}
REQUIRED_MODEL_FILES = {"config.json", "model.pt", "tokenizer.json", "tokenizer_config.json"}


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    if path.name == ".env":
        return True
    # Keep prescribable helper scripts if present, but never package raw RRF dumps.
    if len(rel.parts) >= 2 and rel.parts[0] == "prescribe" and rel.parts[1] == "rrf":
        return True
    # Same for the raw BYT ICD-10 catalog: only the derived icd10_vi.csv ships.
    if rel.parts[:3] == ("data", "terminology", "raw"):
        return True
    return False


def iter_files(include_roots: list[str]):
    for item in include_roots:
        path = ROOT / item
        if not path.exists():
            continue
        if path.is_file():
            if not is_excluded(path):
                yield path
            continue
        for child in path.rglob("*"):
            if child.is_file() and not is_excluded(child):
                yield child


def validate_required_files() -> None:
    model_dir = ROOT / "models" / "ner_model"
    missing = sorted(name for name in REQUIRED_MODEL_FILES if not (model_dir / name).exists())
    if missing:
        raise SystemExit(f"Cannot package source: missing model files in {model_dir}: {missing}")

    for path in [
        ROOT / "data" / "terminology" / "drugs.csv",
        ROOT / "data" / "terminology" / "diagnoses.csv",
        ROOT / "data" / "terminology" / "rxnorm_full.csv",
        ROOT / "data" / "terminology" / "icd10_vi.csv",
        ROOT / "scripts" / "run_pipeline.py",
        ROOT / "scripts" / "package_submission.py",
    ]:
        if not path.exists():
            raise SystemExit(f"Cannot package source: missing required file {path}")


def package(out_zip: Path, include_roots: list[str], dry_run: bool = False) -> None:
    validate_required_files()
    files = sorted(set(iter_files(include_roots)), key=lambda p: p.relative_to(ROOT).as_posix())
    total_bytes = sum(path.stat().st_size for path in files)
    if dry_run:
        print(f"Source bundle dry run: {len(files)} files, {total_bytes / (1024 ** 2):.1f} MiB.")
        print("Required files are present; rerun without --dry-run to write the zip.")
        return
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(ROOT).as_posix())
    print(f"Wrote {out_zip} with {len(files)} files ({total_bytes / (1024 ** 2):.1f} MiB before compression).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "source_bundle.zip")
    parser.add_argument(
        "--include",
        action="append",
        help="Additional file or directory to include, relative to repo root. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report size without writing a zip.")
    args = parser.parse_args()

    include_roots = DEFAULT_INCLUDE_ROOTS + (args.include or [])
    package(args.out, include_roots, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
