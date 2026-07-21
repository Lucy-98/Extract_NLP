#!/usr/bin/env python3
"""Fetch ICD-10 entity data from the WHO ICD-API to validate/enrich
data/terminology/diagnoses.csv.

Setup:
  1. Register a free app at https://icd.who.int/icdapi ("Get API Access").
  2. Put credentials in a .env file at the repo root (never commit this file):
       ICD_CLIENT_ID=...
       ICD_CLIENT_SECRET=...
  3. No extra pip install -- this script only uses the stdlib (urllib).

Usage:
  python scripts/fetch_icd.py --self-test
  python scripts/fetch_icd.py --code C20
  python scripts/fetch_icd.py --enrich                # look up every code already in diagnoses.csv
  python scripts/fetch_icd.py --crawl-chapter II       # full ICD-10 chapter (roman numeral) -> CSV
  python scripts/fetch_icd.py --crawl-all              # all 22 chapters -> data/terminology/icd10_full.csv
                                                        # (slow: ~12k codes, resumable, safe to Ctrl-C and rerun)

Important limitations:
  - The API returns English titles; our diagnosis text is Vietnamese, so this
    cannot resolve Vietnamese phrase -> ICD-10 code by string matching alone.
    Use --crawl-all to build a full local English (code -> title) index, then
    bridge Vietnamese -> English via translation or a cross-lingual embedding
    before matching against it -- that bridging step is not in this script.
  - Free-text --search only works against the ICD-11 foundation layer
    (id.who.int/icd/entity/search), which returns ICD-11 entity ids, not
    ICD-10 codes -- confirmed by testing against this API in this project.
    There is no working free-text search for the ICD-10 release; --crawl-*
    (walking the chapter/block/category tree) is the only way found so far
    to get ICD-10 code+title pairs in bulk from this API.

Security note: this script never prints the client secret or the raw access
token. If either was ever pasted into a chat/log, rotate it at
https://icd.who.int/icdapi (My apps) regardless of what this script does.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
TERM_DIR = ROOT / "data" / "terminology"

TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
API_BASE = "https://id.who.int/icd/release/10/2019"

CLIENT_ID_KEYS = ["ICD_CLIENT_ID", "WHO_ICD_CLIENT_ID", "CLIENT_ID"]
CLIENT_SECRET_KEYS = ["ICD_CLIENT_SECRET", "WHO_ICD_CLIENT_SECRET", "CLIENT_SECRET"]


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    # utf-8-sig strips a leading BOM if present (common when a .env is saved
    # from Notepad/PowerShell on Windows) -- without it the first key in the
    # file silently fails to match because of an invisible leading character.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_credential(keys: list[str], env_file_values: dict[str, str]) -> str | None:
    for key in keys:
        if os.environ.get(key):
            return os.environ[key]
    for key in keys:
        if env_file_values.get(key):
            return env_file_values[key]
    return None


def fetch_token(client_id: str, client_secret: str) -> str:
    data = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "icdapi_access",
        }
    ).encode()
    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode())
    return payload["access_token"]


def api_get_url(url: str, token: str) -> Any:
    # Child links in API responses come back as "http://id.who.int/..." --
    # normalize to https, which is what the API actually requires.
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Accept-Language", "en")
    req.add_header("API-Version", "v2")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def api_get(path: str, token: str) -> Any:
    return api_get_url(f"{API_BASE}{path}", token)


def lookup_code(code: str, token: str) -> Any:
    return api_get(f"/{code}", token)


def entity_title(entity: Any) -> str:
    if not isinstance(entity, dict):
        return ""
    title_field = entity.get("title")
    if isinstance(title_field, dict):
        return title_field.get("@value", "")
    if isinstance(title_field, str):
        return title_field
    return ""


def _fetch_with_retry(url: str, token: str, retries: int = 3) -> Any | None:
    for attempt in range(retries):
        try:
            return api_get_url(url, token)
        except urllib.error.HTTPError as exc:
            print(f"  skip {url}: HTTP {exc.code}")
            return None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                print(f"  giving up on {url}: {exc}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def crawl(start_urls: list[str], token: str, delay: float = 0.05, workers: int = 8) -> list[tuple[str, str]]:
    """Breadth-first, multi-threaded walk of the ICD-10 chapter/block/category
    tree from the given entity URLs. Only classKind == "category" entities
    (both 3-char categories like C20 and 4-char subcategories like C20.0) are
    real codes and get recorded; chapters and block ranges are only used to
    reach their children. Network-bound (each request is ~1s), so this uses
    a thread pool -- the WHO API has no documented rate limit, but `delay`
    adds a small per-batch pause as a courtesy throttle.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    seen: set[str] = set(start_urls)
    frontier: list[str] = list(start_urls)
    results: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while frontier:
            futures = {executor.submit(_fetch_with_retry, url, token): url for url in frontier}
            next_frontier: list[str] = []
            for fut in as_completed(futures):
                entity = fut.result()
                if entity is None:
                    continue
                if entity.get("classKind") == "category":
                    results.append((entity.get("code", ""), entity_title(entity)))
                for child in entity.get("child", []):
                    if child not in seen:
                        seen.add(child)
                        next_frontier.append(child)
            frontier = next_frontier
            if frontier:
                time.sleep(delay)
    return results


def get_token() -> str:
    env_values = load_env_file(ENV_PATH)
    client_id = get_credential(CLIENT_ID_KEYS, env_values)
    client_secret = get_credential(CLIENT_SECRET_KEYS, env_values)
    if not client_id or not client_secret:
        raise SystemExit(
            "Missing ICD API credentials. Put ICD_CLIENT_ID and ICD_CLIENT_SECRET in "
            f"{ENV_PATH} (or export them as environment variables)."
        )
    return fetch_token(client_id, client_secret)


def http_error_snippet(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode(errors="replace")[:400]
    except Exception:  # noqa: BLE001
        return str(exc)


CHAPTERS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
    "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII",
]


def load_existing_crawl(out_path: Path) -> dict[str, str]:
    if not out_path.exists():
        return {}
    with out_path.open(encoding="utf-8") as f:
        return {row["code"]: row["title_en"] for row in csv.DictReader(f) if row.get("code")}


def append_crawl_results(out_path: Path, existing: dict[str, str], new_results: list[tuple[str, str]]) -> int:
    added = 0
    for code, title in new_results:
        if code and code not in existing:
            existing[code] = title
            added += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "title_en"])
        for code in sorted(existing):
            writer.writerow([code, existing[code]])
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="Token + one lookup, to verify setup.")
    parser.add_argument("--code", help="Look up a single ICD-10 code, e.g. C20")
    parser.add_argument("--enrich", action="store_true", help="Look up every code in diagnoses.csv, write English titles")
    parser.add_argument("--crawl-chapter", metavar="ROMAN", help="Crawl one ICD-10 chapter, e.g. II")
    parser.add_argument("--crawl-all", action="store_true", help="Crawl all 22 ICD-10 chapters (slow, resumable)")
    parser.add_argument("--out", type=Path, default=TERM_DIR / "icd10_full.csv", help="Output CSV for --crawl-*")
    parser.add_argument("--delay", type=float, default=0.15, help="Seconds between requests while crawling")
    args = parser.parse_args()

    if not any([args.self_test, args.code, args.enrich, args.crawl_chapter, args.crawl_all]):
        parser.print_help()
        return 1

    try:
        token = get_token()
    except urllib.error.HTTPError as exc:
        print(f"Token request failed: HTTP {exc.code}")
        print(http_error_snippet(exc))
        return 1
    print(f"token acquired (length={len(token)}; value not printed)")

    if args.self_test:
        try:
            entity = lookup_code("C20", token)
            print("lookup C20 ->", json.dumps(entity, ensure_ascii=False)[:400])
        except urllib.error.HTTPError as exc:
            print(f"lookup C20 failed: HTTP {exc.code}: {http_error_snippet(exc)}")
            return 1
        print("self-test OK")
        return 0

    if args.code:
        try:
            entity = lookup_code(args.code, token)
            print(json.dumps(entity, ensure_ascii=False, indent=2))
        except urllib.error.HTTPError as exc:
            print(f"lookup failed: HTTP {exc.code}: {http_error_snippet(exc)}")
            return 1

    if args.enrich:
        diag_path = TERM_DIR / "diagnoses.csv"
        if not diag_path.exists():
            raise SystemExit(f"{diag_path} not found -- run build_terminology_index.py first.")
        with diag_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        codes = sorted({r["candidate"] for r in rows})
        enriched: list[tuple[str, str]] = []
        for code in codes:
            try:
                entity = lookup_code(code, token)
                title = entity_title(entity)
            except urllib.error.HTTPError as exc:
                title = f"<lookup failed: HTTP {exc.code}>"
            enriched.append((code, title))
            time.sleep(args.delay)

        out_path = TERM_DIR / "diagnoses_icd10_titles.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate", "who_icd10_title_en"])
            writer.writerows(enriched)
        print(f"Wrote {len(enriched)} codes -> {out_path}")

    if args.crawl_chapter or args.crawl_all:
        chapters = CHAPTERS if args.crawl_all else [args.crawl_chapter]
        existing = load_existing_crawl(args.out)
        print(f"resuming: {len(existing)} codes already in {args.out}")
        for chapter in chapters:
            print(f"crawling chapter {chapter} ...")
            results = crawl([f"{API_BASE}/{chapter}"], token, delay=args.delay)
            added = append_crawl_results(args.out, existing, results)
            print(f"  chapter {chapter}: {len(results)} categories seen, {added} new, total {len(existing)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
