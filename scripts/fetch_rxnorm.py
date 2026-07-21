#!/usr/bin/env python3
"""Fetch RxNorm concept candidates from the RxNav REST API (NLM) to enrich
data/terminology/drugs.csv.

SUPERSEDED as of 2026-07-21: the full RxNorm RRF release (needs a UMLS
account, see below) has since been obtained, and
scripts/build_rxnorm_rrf_index.py builds a committed, offline, ~512k-row
text->RXCUI index from it (data/terminology/rxnorm_full.csv) -- strictly
more complete than this API (it reaches Remapped/retired concepts RxNav's
search can never return, see that script's docstring for the confirmed
360047 example) and with no network dependency, which matters for
run_pipeline.py's private-rerun environment. run_pipeline.py and
augment_ner_dataset.py both now use that offline index instead of this file.
Kept for reference and as a manual one-off lookup tool (e.g. if a future
drug text misses both drugs.csv and rxnorm_full.csv and you want a quick
live cross-check) -- not part of the default pipeline anymore.

Why RxNav instead of the full RxNorm RRF release: the RRF files require a
free UMLS Metathesaurus account/license to download, and that registration
didn't go through (see README.md "Giới hạn đã biết"). RxNav is a public REST
API over the same RxNorm data -- no registration, no API key -- and its
approximate-match endpoint is dose/form aware, which is exactly the
granularity the contest's own worked example requires: clonazepam 0.5 mg and
1.5 mg are different RxNorm concepts, so matching by ingredient name alone is
wrong by construction (see build_terminology_index.py's module docstring).

No extra pip install -- stdlib only (urllib).

Usage:
  python scripts/fetch_rxnorm.py --self-test
  python scripts/fetch_rxnorm.py --lookup "Chlorpheniramine 0.4 MG/ML"
  python scripts/fetch_rxnorm.py --enrich                  # every THUỐC text seen in the corpus
                                                             # that drugs.csv doesn't cover yet
  python scripts/fetch_rxnorm.py --enrich --all             # re-query every THUỐC text, including
                                                             # ones already in drugs.csv (validation)

Output of --enrich: data/terminology/drugs_rxnav.csv (text,candidate,source,score,matched_name,tty) --
a review layer, NOT auto-merged into drugs.csv. RxNav's approximate match is
a best guess, not ground truth -- inspect scores/matched_name, then
hand-merge rows you trust into drugs.csv (same text,candidate,source columns
build_terminology_index.py already writes/reads; drop the score/matched_name/
tty columns when you do).

IMPORTANT, discovered 2026-07-20 while validating against the contest's own
worked example ("Chlorpheniramine 0.4 MG/ML" -> RxNorm 360047): RxNav's
search endpoints (approximateTerm, drugs.json, rxcui.json name search) all
silently exclude concepts whose status is "Remapped" -- confirmed by
querying rxcui 360047's *exact* canonical name and getting zero results.
Only rxcui/{rxcui}/historystatus.json can retrieve a Remapped concept, and
that needs the rxcui already known. "Obsolete" concepts (different status)
do still show up in search. If the contest's answer key was generated
against an older/frozen RxNorm snapshot -- as this one example suggests --
no live text search against current RxNav can fully reconstruct it; this
script is a recall booster for texts drugs.csv doesn't cover yet, not a
substitute for the full RxNorm RRF release (needs a UMLS license) or for
human review of its output.
"""

from __future__ import annotations

import argparse
import csv
import json
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
TERM_DIR = ROOT / "data" / "terminology"
DATASET_DIR = ROOT / "data" / "ner_dataset"
DRUG = "THUỐC"

API_BASE = "https://rxnav.nlm.nih.gov/REST"


def api_get(path: str, params: dict[str, str], retries: int = 3, timeout: float = 20) -> Any:
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            print(f"  HTTP {exc.code} for {url}")
            return None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                print(f"  giving up on {url}: {exc}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def approximate_match(term: str, max_entries: int = 5) -> list[dict[str, str]]:
    """Dose/form-aware fuzzy match: the endpoint RxNav documents specifically
    for messy real-world drug mention strings (as opposed to exact-name
    lookup, which would miss anything not phrased exactly like RxNorm's
    canonical name).
    """
    payload = api_get("/approximateTerm.json", {"term": term, "maxEntries": str(max_entries)})
    if not payload:
        return []
    group = payload.get("approximateGroup") or {}
    return group.get("candidate") or []


def quick_lookup(term: str, max_entries: int = 3, timeout: float = 2.0) -> list[str]:
    """Single fast approximateTerm.json call, one attempt, no
    drugs_by_ingredient/historystatus enrichment -- meant as an inference-
    time safety-net fallback for THUỐC mentions the offline CSV/fuzzy
    matcher can't resolve (see run_pipeline.py's RxNavFallback), not a
    substitute for --enrich's fuller, human-reviewed pass. Kept to one
    request with a short timeout so a network-less private test environment
    fails fast instead of stalling inference.
    """
    payload = api_get(
        "/approximateTerm.json", {"term": term, "maxEntries": str(max_entries)}, retries=1, timeout=timeout
    )
    if not payload:
        return []
    group = payload.get("approximateGroup") or {}
    candidates = group.get("candidate") or []
    return [c["rxcui"] for c in candidates if c.get("rxcui")]


def rxcui_info(rxcui: str) -> dict[str, str]:
    """historystatus.json, not properties.json: the latter returns {} for any
    concept that isn't currently active (Obsolete/Remapped), which turns out
    to be common among the correct answers in this domain -- see the module
    docstring. historystatus.json returns name/tty/status for those too
    (except Remapped concepts found by rxcui, which is the one case neither
    endpoint reaches by name search at all).
    """
    payload = api_get(f"/rxcui/{rxcui}/historystatus.json", {})
    if not payload:
        return {}
    history = payload.get("rxcuiStatusHistory") or {}
    attrs = history.get("attributes") or {}
    meta = history.get("metaData") or {}
    return {"name": attrs.get("name", ""), "tty": attrs.get("tty", ""), "status": meta.get("status", "")}


# Prefer dispensable-product-level concepts (the schema wants dose+form
# specific codes) over abstract ingredient/component-level ones when
# candidates are otherwise close in score.
PRODUCT_TTY = {"SCD", "SBD", "GPCK", "BPCK"}


def drugs_by_ingredient(text: str) -> list[dict[str, str]]:
    """Secondary source: query RxNorm's /drugs.json by the first word of the
    mention (usually the ingredient name) and keep only concepts whose name
    contains every other significant token from the original text (e.g. the
    dose). Broader recall than approximateTerm for straightforward cases,
    but -- like approximateTerm -- only reaches currently searchable
    (non-Remapped) concepts.
    """
    tokens = [t.lower() for t in text.split()[1:] if len(t) > 1]
    if not tokens:
        # Nothing to filter on beyond the ingredient itself (e.g. a bare
        # "Albuterol" mention with no dose) -- every product for that
        # ingredient would match, which is noise, not a useful candidate list.
        return []
    first_word = text.split()[0]
    payload = api_get("/drugs.json", {"name": first_word})
    if not payload:
        return []
    hits: list[dict[str, str]] = []
    for group in (payload.get("drugGroup") or {}).get("conceptGroup") or []:
        tty = group.get("tty", "")
        for prop in group.get("conceptProperties") or []:
            name_lower = prop.get("name", "").lower()
            if all(tok in name_lower for tok in tokens):
                hits.append({"rxcui": prop["rxcui"], "name": prop.get("name", ""), "tty": tty, "score": ""})
    return hits


def load_covered_texts(drugs_csv: Path) -> set[str]:
    covered: set[str] = set()
    if drugs_csv.exists():
        with drugs_csv.open(encoding="utf-8") as f:
            covered = {row["text"].strip().lower() for row in csv.DictReader(f) if row.get("text")}
    return covered


def load_corpus_drug_texts() -> list[str]:
    all_path = DATASET_DIR / "all.jsonl"
    if not all_path.exists():
        return []
    texts: set[str] = set()
    with all_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for ent in rec.get("entities", []):
                if ent.get("type") == DRUG and ent.get("text"):
                    texts.add(ent["text"])
    return sorted(texts)


def gather_candidates(text: str, max_entries: int) -> list[dict[str, str]]:
    """Merge approximateTerm (fuzzy, full text) with drugs.json (ingredient
    name + substring-on-tokens, broader recall for straightforward cases),
    dedup by rxcui, fill in name/tty/status via historystatus.json for
    anything that came back without one, then sort SCD/SBD/pack-level
    (product) concepts before component/ingredient-level ones -- the schema
    wants dose+form-specific product codes, and raw text-similarity score
    doesn't know that distinction.
    """
    by_rxcui: dict[str, dict[str, str]] = {}
    for c in approximate_match(text, max_entries):
        rxcui = c.get("rxcui", "")
        if rxcui:
            by_rxcui[rxcui] = {"rxcui": rxcui, "score": str(c.get("score", "")), "name": "", "tty": ""}
    for h in drugs_by_ingredient(text):
        by_rxcui.setdefault(h["rxcui"], {"rxcui": h["rxcui"], "score": "", "name": "", "tty": ""})
        by_rxcui[h["rxcui"]]["name"] = h.get("name", "")
        by_rxcui[h["rxcui"]]["tty"] = h.get("tty", "")

    for rxcui, entry in by_rxcui.items():
        if not entry["name"]:
            info = rxcui_info(rxcui)
            entry["name"] = info.get("name", "")
            entry["tty"] = info.get("tty", "")
            entry["status"] = info.get("status", "")

    return sorted(
        by_rxcui.values(),
        key=lambda e: (e.get("tty") not in PRODUCT_TTY, -float(e["score"] or 0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="One approximate-match call, to verify connectivity.")
    parser.add_argument("--lookup", help="Approximate-match a single drug mention string and print candidates.")
    parser.add_argument("--enrich", action="store_true", help="Approximate-match THUỐC texts from the corpus.")
    parser.add_argument("--all", action="store_true", help="With --enrich: include texts already in drugs.csv too.")
    parser.add_argument("--max-entries", type=int, default=3, help="Candidates to keep per text.")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between requests.")
    parser.add_argument("--limit", type=int, help="With --enrich: only query the first N texts (for a quick dry run).")
    parser.add_argument("--out", type=Path, default=TERM_DIR / "drugs_rxnav.csv")
    args = parser.parse_args()

    if not any([args.self_test, args.lookup, args.enrich]):
        parser.print_help()
        return 1

    if args.self_test:
        candidates = approximate_match("aspirin 325 mg")
        if not candidates:
            print("self-test FAILED: no candidates returned (check network/connectivity).")
            return 1
        print(f"self-test OK: {len(candidates)} candidate(s) for 'aspirin 325 mg':")
        for c in candidates[:3]:
            print(f"  rxcui={c.get('rxcui')} score={c.get('score')} rank={c.get('rank')}")
        return 0

    if args.lookup:
        candidates = gather_candidates(args.lookup, args.max_entries)
        if not candidates:
            print("no candidates found")
            return 0
        for c in candidates:
            print(f"rxcui={c['rxcui']} tty={c.get('tty') or '?'} score={c.get('score') or '-'} name={c.get('name', '')!r}")
        return 0

    if args.enrich:
        corpus_texts = load_corpus_drug_texts()
        if not corpus_texts:
            raise SystemExit(
                f"No drug texts found in {DATASET_DIR / 'all.jsonl'} -- run "
                "scripts/prepare_ner_dataset.py first."
            )
        covered = set() if args.all else load_covered_texts(TERM_DIR / "drugs.csv")
        targets = [t for t in corpus_texts if t.strip().lower() not in covered]
        if args.limit:
            targets = targets[: args.limit]
        print(f"{len(corpus_texts)} distinct THUỐC texts in corpus, {len(targets)} to query "
              f"({'all' if args.all else 'not already in drugs.csv'}{f', limited to {args.limit}' if args.limit else ''})")

        rows: list[tuple[str, str, str, str, str, str]] = []
        for i, text in enumerate(targets, 1):
            candidates = gather_candidates(text, args.max_entries)
            if not candidates:
                print(f"  [{i}/{len(targets)}] {text!r}: no match")
            for c in candidates:
                rows.append((text, c["rxcui"], "rxnav", c.get("score", ""), c.get("name", ""), c.get("tty", "")))
            if candidates:
                best = candidates[0]
                print(f"  [{i}/{len(targets)}] {text!r} -> rxcui={best['rxcui']} tty={best.get('tty') or '?'}")
            time.sleep(args.delay)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "candidate", "source", "score", "matched_name", "tty"])
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} candidate rows for {len(targets)} texts -> {args.out}")
        print("Review this file, then hand-merge rows you trust into data/terminology/drugs.csv "
              "(drop the score/matched_name/tty columns -- build_terminology_index.py only reads "
              "text,candidate,source). Candidates are sorted product-level (SCD/SBD/pack) first, "
              "but a text search can only find Current or Obsolete concepts, never Remapped ones "
              "-- see the module docstring.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
