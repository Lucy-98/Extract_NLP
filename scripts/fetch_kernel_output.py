#!/usr/bin/env python3
"""Download Kaggle kernel output file-by-file, with HTTP range resume.

Why this exists
---------------
`kaggle kernels output` fetches the whole output in one shot and cannot resume.
The kernel's `ner_model_export.zip` is ~1.75GB, so on a flaky connection the
download dies partway and everything is lost:

    Connection broken: IncompleteRead(1679322846 bytes read, 70619516 more expected)
    Connection broken: ConnectionAbortedError(10053, ...)

This has now bitten the project three times (worklog 2026-07-20 records two
earlier corrupted downloads that still reported kernel status COMPLETE).

Two fixes, both here:

1. **Skip the zip.** The same files are also published individually under
   `ner_model_export/`, so `--only ner_model_export/` fetches the 5 real
   artifacts and never touches the 1.75GB archive.
2. **Resume.** Each file is written to `<name>.part` and re-requested with a
   `Range:` header on retry, so a broken transfer costs only the remaining
   bytes.

The worst failure mode this guards against is silent: a truncated `model.pt`
loads as garbage or 0 bytes while everything upstream reports success. Files are
only renamed into place after the server-reported length matches.

Usage:
  python scripts/fetch_kernel_output.py --list
  python scripts/fetch_kernel_output.py --only output.zip
  python scripts/fetch_kernel_output.py --only ner_model_export/ --dest .kaggle_download
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
API = "https://www.kaggle.com/api/v1/kernels/output"
DEFAULT_SLUG = "lucylng/viettelrace-ner-assertion-train"
CHUNK = 1 << 20


def credentials() -> tuple[str, str]:
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Download it from your Kaggle account settings.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["username"], data["key"]


def human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}GB"


def list_files(session: requests.Session, slug: str) -> list[dict]:
    user, kernel = slug.split("/", 1)
    files: list[dict] = []
    token = None
    while True:
        params = {"user_name": user, "kernel_slug": kernel}
        if token:
            params["page_token"] = token
        response = session.get(API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        files.extend(payload.get("files", []))
        token = payload.get("nextPageToken") if payload.get("hasNextPageToken") else None
        if not token:
            return files


def fetch(session: requests.Session, url: str, dest: Path, attempts: int = 8) -> bool:
    """Download one file, resuming from `<dest>.part` across attempts."""
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    total: int | None = None

    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                if response.status_code == 416:  # already complete
                    break
                response.raise_for_status()
                length = response.headers.get("Content-Length")
                if length is not None:
                    total = have + int(length)
                mode = "ab" if have and response.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                with part.open(mode) as handle:
                    for block in response.iter_content(CHUNK):
                        handle.write(block)
                        have += len(block)
                        if total:
                            pct = 100 * have / total
                            print(f"\r    {human(have)} / {human(total)}  {pct:5.1f}%", end="", flush=True)
            print()
            break
        except (requests.RequestException, OSError) as exc:
            got = part.stat().st_size if part.exists() else 0
            print(f"\n    attempt {attempt}/{attempts} failed at {human(got)}: {type(exc).__name__}")
            if attempt == attempts:
                return False
            time.sleep(min(5 * attempt, 30))

    if total is not None and part.stat().st_size != total:
        print(f"    SIZE MISMATCH: got {part.stat().st_size}, expected {total} -- keeping .part")
        return False
    part.replace(dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--dest", type=Path, default=ROOT / ".kaggle_download")
    parser.add_argument("--only", action="append", default=[],
                        help="Substring filter; repeatable. Default fetches the model export only.")
    parser.add_argument("--list", action="store_true", help="List output files and exit.")
    args = parser.parse_args()

    session = requests.Session()
    session.auth = credentials()

    files = list_files(session, args.slug)
    if args.list:
        for entry in files:
            print(f"  {entry['fileName']}")
        print(f"\n{len(files)} files")
        return 0

    patterns = args.only or ["ner_model_export/"]
    selected = [f for f in files if any(p in f["fileName"] for p in patterns)]
    # The all-in-one archive is exactly what cannot be resumed reliably; the same
    # bytes are available as individual files, so never pick it up implicitly.
    selected = [f for f in selected if f["fileName"] != "ner_model_export.zip" or "ner_model_export.zip" in patterns]
    if not selected:
        print(f"No output file matches {patterns}. Use --list to see what is available.")
        return 1

    print(f"Fetching {len(selected)} file(s) from {args.slug} -> {args.dest}\n")
    failed: list[str] = []
    for entry in selected:
        name = entry["fileName"]
        target = args.dest / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  skip (exists) {name}  {human(target.stat().st_size)}")
            continue
        print(f"  {name}")
        if not fetch(session, entry["url"], target):
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} file(s) incomplete: {failed}")
        print("Re-run this command -- partial .part files resume where they stopped.")
        return 1
    print("\nAll files complete. Verify before trusting the model:")
    print("  python -c \"import torch; sd=torch.load('models/ner_model/model.pt', map_location='cpu'); print(len(sd),'keys')\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())