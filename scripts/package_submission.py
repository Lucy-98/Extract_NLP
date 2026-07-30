#!/usr/bin/env python3
"""Create the Round 1 `output.zip` with the exact folder layout BTC expects.

The zip entries are:

  output/1.json
  output/2.json
  ...
  output/{last_input_id}.json

By default this packages `output_model/`, the reproducible model pipeline's
prediction folder. Use `--pred output` only when intentionally packaging the
hand-tuned public submission.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_submission import get_input_ids, validate_prediction, print_report  # noqa: E402


def package(pred_dir: Path, input_dir: Path, out_zip: Path, expected_count: int | None) -> None:
    report = validate_prediction(pred_dir, input_dir, expected_count=expected_count)
    print_report(report)
    if report.errors:
        raise SystemExit("Refusing to package because validation has errors.")

    ids = list(range(1, expected_count + 1)) if expected_count is not None else get_input_ids(input_dir)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in ids:
            src = pred_dir / f"{i}.json"
            if not src.exists():
                raise SystemExit(f"Missing required prediction file: {src}")
            zf.write(src, arcname=f"output/{i}.json")

    print(f"Wrote {out_zip} with {len(ids)} files under output/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, default=ROOT / "output_model", help="Prediction folder to package.")
    parser.add_argument("--input", type=Path, default=ROOT / "input_turn2", help="Input folder used for validation.")
    parser.add_argument("--out", type=Path, default=ROOT / "output.zip", help="Zip file to write.")
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Expected number of sequential json files. Defaults to the numbered files in --input.",
    )
    args = parser.parse_args()

    package(args.pred, args.input, args.out, args.expected_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
