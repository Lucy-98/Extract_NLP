#!/usr/bin/env python3
"""Build an offline RxNorm text -> RXCUI index from the official RRF release.

Why this exists: `scripts/fetch_rxnorm.py` used the free RxNav REST API as a
substitute for the RRF release because the RRF release needs a UMLS
Metathesaurus account, and that registration hadn't gone through. It has
since gone through -- the full RxNorm release (`RxNorm_full_07062026.zip`)
is unzipped locally under `rrf/` (gitignored, ~1.3GB, re-download from
https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html; not
something to commit). This script derives a small, committable CSV from it,
the same way `build_terminology_index.py` derives `drugs.csv`/`diagnoses.csv`
from `output/`.

This closes the one confirmed hard gap in the RxNav approach (see
`fetch_rxnorm.py`'s module docstring): RxNav's search endpoints cannot return
any concept whose RxNorm status is "Remapped" -- e.g. the task statement's
own worked example, "Chlorpheniramine 0.4 MG/ML" -> RxNorm 360047, is
unreachable through live API text search. `RXNATOMARCHIVE.RRF` (RxNorm's own
historical-atom table) has it: rxaui 1564730, rxcui 360047, tty SCD, str
"Chlorpheniramine 0.4 MG/ML / Dextromethorphan 6 MG/ML / Guaifenesin 40
MG/ML / Pseudoephedrine 6 MG/ML Oral Solution" -- confirmed by `--verify`
below. Combining `RXNCONSO.RRF` (currently active RxNorm concepts) with
`RXNATOMARCHIVE.RRF` (every concept RxNorm has ever had, including ones later
merged/remapped away) gives full historical coverage no live API can match,
entirely offline -- which also removes run_pipeline.py's one remaining
network dependency (see RxNormOfflineFallback there), a real win given
CLAUDE.md's constraint that the organizers rerun submitted source on a
private test set of unknown network connectivity.

RRF column layout (both files are pipe-delimited, no header row; verified
empirically against this release rather than assumed from the NLM docs,
since column counts can drift between releases):
  RXNCONSO.RRF (19 fields incl. trailing empty from the terminal '|'):
    0 RXCUI, 1 LAT, 11 SAB, 12 TTY, 14 STR
  RXNATOMARCHIVE.RRF (17 fields incl. trailing empty):
    2 STR, 6 RXCUI, 8 LAT, 13 SAB, 14 TTY
  (RXNATOMARCHIVE.RRF in this release is 100% SAB=RXNORM already -- verified
  by counting -- so no SAB filter is needed there, only on RXNCONSO.)

Also writes `data/terminology/rxnorm_drug_names.csv` (~11.9k rows) -- a short,
clean list of real ingredient/brand names for `augment_ner_dataset.py` to draw
synthetic THUỐC substitutions from. Source: `prescribe/rrf/RXNCONSO.RRF`, the
separate "RxNorm Prescribable Content" release the user also downloaded
(gitignored, ~30MB there vs 131MB for the full RXNCONSO.RRF) -- deliberately
*not* the full release used above, because the full release's ~33k IN-level
names include obscure research chemicals ("(2-carbethoxyethyl)diethoxy
(methyl)silane") no clinician would ever write in a note; the prescribable
subset is pre-filtered to currently-prescribable drugs, which is exactly the
realism augmentation needs. (This file is a training-data-generation input,
consumed only by augment_ner_dataset.py before training -- unlike
rxnorm_full.csv, it never needs to be read by run_pipeline.py at inference
time, so there was no reason to fold it into the same derivation as the
Remapped-concept coverage above.)

Usage:
  python scripts/build_rxnorm_rrf_index.py                 # writes data/terminology/rxnorm_full.csv
  python scripts/build_rxnorm_rrf_index.py --rrf-dir X      # non-default extract location
  python scripts/build_rxnorm_rrf_index.py --verify         # rebuild + check the 360047 worked example
  python scripts/build_rxnorm_rrf_index.py --lookup "clonazepam 0.5 mg"
  python scripts/build_rxnorm_rrf_index.py --no-names       # skip the prescribable-names pass
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TERM_DIR = ROOT / "data" / "terminology"
DEFAULT_RRF_DIR = ROOT / "rrf"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_terminology_index import norm  # noqa: E402

# Preference tier when the same normalized text maps to several RXCUI/TTY
# pairs -- lower is better. Product-level (dose+form-specific, what the
# schema wants) first, then component/form, then bare ingredient/brand, then
# free-text synonyms last (still useful for fuzzy recall, just not a
# preferred *exact* match when something more specific also matched).
TTY_TIER = {
    "SCD": 0, "SBD": 0, "GPCK": 0, "BPCK": 0,
    "SCDC": 1, "SBDC": 1, "SCDF": 1, "SBDF": 1, "SCDG": 1, "SBDG": 1,
    "SCDFP": 1, "SBDFP": 1, "SCDGP": 1,
    "IN": 2, "PIN": 2, "BN": 2, "MIN": 2, "PSN": 2,
    "SY": 3, "TMSY": 3, "DF": 3, "DFG": 3, "ET": 3,
}


TIER_OF_COMPLETE_PRODUCT = 0  # SCD/SBD/GPCK/BPCK -- what the schema asks for

# --- dose+form resolution (added 2026-07-28) --------------------------------
# The task statement is explicit that a drug code is dose+form specific
# ("clonazepam 0.5 mg po qam:prn" -> 197527, "clonazepam 1.5 mg po qhs" ->
# 197528) and its worked example resolves to an SCD. Before this, the index
# scored 1/3 on those three published examples even though every target code is
# present in rxnorm_full.csv: a clinical mention writes the dose form as a route
# abbreviation ("po") or in Vietnamese ("đường uống"), which no RxNorm string
# contains, so the mention could only ever reach the form-less component concept
# (SCDC 315699 instead of SCD 197527). Bridging that vocabulary lifts it to 2/3;
# the remaining miss needs dose arithmetic (1.5 mg dispensed as 1 mg tablets),
# which is deliberately not attempted.
#
# NOTE this is a bet on the *rules* over the *public labels*: output/ codes most
# dosed mentions at ingredient level (IN/BN), which this deliberately stops
# reproducing, so agreement with output/ on dosed mentions stays low by design.
# output/ scored J_candidates 29.98 on the real board, i.e. most of its codes
# were wrong, so agreement with it is not evidence of correctness here.
# Reversible via promote_dose_form=False / run_pipeline.py --no-dose-form-promotion.

# Administration noise that appears in clinical drug mentions but never in an
# RxNorm string: dosing frequency, PRN markers, counts. Stripped before matching.
_FREQUENCY_NOISE = {
    "bid", "tid", "qid", "qd", "qod", "qhs", "qam", "qpm", "qh", "prn", "stat",
    "q2h", "q4h", "q6h", "q8h", "q12h", "q24h", "ac", "pc", "hs",
    "x", "lần", "liều", "ngày", "mỗi", "sau", "trước", "khi", "cần", "và",
    # English sig wording that shows up in the corpus' own mentions
    # ("lasix 40mg daily", "combivent nebs x3 every 20 minutes")
    "daily", "once", "twice", "thrice", "every", "other", "day", "days",
    "week", "weekly", "month", "monthly", "night", "nightly", "morning",
    "evening", "hour", "hours", "hr", "hrs", "minute", "minutes", "min",
    "at", "in", "as", "needed", "per", "then", "now", "home", "tại", "nhà",
    "hàng", "buổi", "sáng", "tối", "chiều", "đơn", "vị",
}

# Route/dose-form vocabulary -> the RxNorm dose-form word(s) it implies. The
# graded answer is a complete product ("clonazepam 0.5 mg ORAL TABLET", SCD),
# but clinicians write the form as a route abbreviation ("po") or in Vietnamese
# ("đường uống"), so without this bridge the mention can only ever reach the
# form-less component concept (SCDC) -- which is exactly how
# "clonazepam 0.5 mg po qam:prn" was resolving to 315699 instead of the task
# statement's own answer, 197527.
_ROUTE_FORM_HINTS = {
    "po": ("oral",), "uống": ("oral",), "đường": (), "miệng": ("oral",),
    "viên": ("tablet",), "nén": ("tablet",), "nang": ("capsule",),
    "iv": ("injection", "injectable"), "im": ("injection", "injectable"),
    "sc": ("injection", "injectable"), "tiêm": ("injection", "injectable"),
    "tĩnh": ("injection", "injectable"), "mạch": (), "truyền": ("injection", "injectable"),
    "sl": ("sublingual",), "lưỡi": ("sublingual",), "dưới": (),
    "hít": ("inhalation",), "xịt": ("spray", "inhalation"), "khí": ("inhalation",),
    "bôi": ("topical",), "thoa": ("topical",), "nhỏ": ("drop", "ophthalmic"),
    "đặt": (), "hậu": ("rectal",), "môn": (),
    # English form words a mention may already carry verbatim
    "oral": ("oral",), "tablet": ("tablet",), "capsule": ("capsule",),
    "solution": ("solution",), "suspension": ("suspension",), "syrup": ("syrup",),
    "injection": ("injection", "injectable"), "injectable": ("injection", "injectable"),
    "sublingual": ("sublingual",), "topical": ("topical",), "inhalation": ("inhalation",),
    "cream": ("cream",), "ointment": ("ointment",), "patch": ("patch",),
    "tab": ("tablet",), "tabs": ("tablet",), "cap": ("capsule",), "caps": ("capsule",),
    "neb": ("inhalation", "solution"), "nebs": ("inhalation", "solution"),
    "puff": ("inhalation",), "puffs": ("inhalation",), "nhồi": ("inhalation",),
}

_DOSE_RE = re.compile(r"\d\s*(?:mg|mcg|ug|g|gram|gm|ml|l|iu|unit|units|%)\b|\d+\s*/\s*\d")


def strip_administration_noise(key: str) -> tuple[str, set[str]]:
    """Split an already-normalized mention into (core, form_hints).

    `core` keeps only the ingredient/brand and dose tokens, in order, so it can
    be prefix-matched against an RxNorm product string -- including a
    *combination* product, where the mention names one ingredient and the
    official string lists all of them (the task statement's 360047 example).
    `form_hints` collects the RxNorm dose-form words implied by whatever route
    or form vocabulary was stripped out.
    """
    # Parenthetical asides are prescriber commentary, never part of the product
    # name ("prograf (dose decreased from 5mg bid to 3mg bid)"). Dropping them
    # also keeps the dose *inside* the parentheses out of the matching prefix.
    key = re.sub(r"\([^)]*\)?", " ", key)
    core: list[str] = []
    hints: set[str] = set()

    def is_admin_vocab(part: str) -> bool:
        return part in _ROUTE_FORM_HINTS or part in _FREQUENCY_NOISE

    for token in key.split():
        # Sig shorthand packs several codes into one token ("qam:prn", "po/iv"),
        # so split before cleaning punctuation -- cleaning first would fuse
        # "qam:prn" into the unrecognisable "qamprn".
        parts = [re.sub(r"[^0-9a-zà-ỹ/.%]+", "", p) for p in re.split(r"[:;,]", token)]
        expanded: list[str] = []
        for part in (p for p in parts if p):
            # "/" separates two routes ("po/iv") but joins a dose unit
            # ("mg/ml") -- only split it when both sides are route vocabulary,
            # so the unit survives into the prefix used for product matching.
            subs = [s for s in part.split("/") if s]
            if len(subs) > 1 and all(is_admin_vocab(s) for s in subs):
                expanded.extend(subs)
            else:
                expanded.append(part)
        if not expanded:
            continue
        if all(is_admin_vocab(p) for p in expanded):
            for p in expanded:
                hints.update(_ROUTE_FORM_HINTS.get(p, ()))
            continue
        core.extend(expanded)
    return " ".join(core), hints


def iter_rxnconso_rxnorm(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (rxcui, tty, str) for every SAB=RXNORM row."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 15 or fields[11] != "RXNORM":
                continue
            yield fields[0], fields[12], fields[14]


def iter_rxnatomarchive(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (rxcui, tty, str) for every archived (incl. Remapped/retired)
    RxNorm atom -- the piece live RxNav search cannot reach at all."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 15 or fields[13] != "RXNORM":
                continue
            yield fields[6], fields[14], fields[2]


def build_index(rrf_dir: Path) -> dict[str, dict[str, tuple[str, str]]]:
    """norm(text) -> {rxcui: (tty, source)}, source in {'rrf_current','rrf_archive'}."""
    index: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)

    conso = rrf_dir / "RXNCONSO.RRF"
    if not conso.exists():
        raise SystemExit(f"{conso} not found -- unzip the RxNorm full release there first.")
    n = 0
    for rxcui, tty, str_ in iter_rxnconso_rxnorm(conso):
        if not str_:
            continue
        key = norm(str_)
        index[key].setdefault(rxcui, (tty, "rrf_current"))
        n += 1
    print(f"RXNCONSO.RRF: {n} SAB=RXNORM rows")

    archive = rrf_dir / "RXNATOMARCHIVE.RRF"
    if archive.exists():
        n = 0
        for rxcui, tty, str_ in iter_rxnatomarchive(archive):
            if not str_:
                continue
            key = norm(str_)
            if rxcui not in index[key]:
                index[key][rxcui] = (tty, "rrf_archive")
            n += 1
        print(f"RXNATOMARCHIVE.RRF: {n} rows")
    else:
        print(f"(skipping {archive}, not found -- Remapped/retired concepts won't be covered)")

    return index


def write_index(index: dict[str, dict[str, tuple[str, str]]], out_path: Path) -> int:
    rows: list[tuple[str, str, str, str]] = []
    for text, by_rxcui in index.items():
        for rxcui, (tty, source) in by_rxcui.items():
            rows.append((text, rxcui, tty, source))
    rows.sort()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "candidate", "tty", "source"])
        writer.writerows(rows)
    return len(rows)


NAME_TTY = {"IN", "PIN", "BN"}


def build_prescribable_names(prescribe_rrf_dir: Path) -> list[str]:
    conso = prescribe_rrf_dir / "RXNCONSO.RRF"
    if not conso.exists():
        print(f"(skipping prescribable-names pass, {conso} not found)")
        return []
    names: dict[str, str] = {}  # lowercase -> nicest-cased original seen
    for _rxcui, tty, str_ in iter_rxnconso_rxnorm(conso):
        if tty not in NAME_TTY or not str_:
            continue
        if len(str_.split()) > 4 or len(str_) > 40:
            continue
        low = str_.lower()
        names.setdefault(low, str_)
    return sorted(names.values())


class RxNormOfflineIndex:
    """Fast offline (text -> RXCUI) lookup over the derived rxnorm_full.csv,
    for use as run_pipeline.py's private-test-set fallback in place of the
    network-dependent RxNavFallback, and as an extra candidate source when
    building drugs.csv. Exact-normalized-text first; if that misses, narrows
    to entries sharing the mention's first token (almost always the
    ingredient/brand word) before ranking -- a full difflib scan over ~700k
    rows per miss would be too slow for run_pipeline.py's per-entity calls.
    """

    def __init__(self, csv_path: Path, promote_dose_form: bool = True):
        # promote_dose_form=False restores the pre-2026-07-28 behaviour, where a
        # dosed mention could only ever reach a form-less component concept.
        # Kept switchable because this is a bet on the task statement's stated
        # rule over the style of the curated public labels -- see the module
        # docstring and run_pipeline.py's --no-dose-form-promotion.
        self.promote_dose_form = promote_dose_form
        self.exact: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.by_first_token: dict[str, list[str]] = defaultdict(list)
        if not csv_path.exists():
            return
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row["text"]
                entry = (row["candidate"], row.get("tty", ""))
                if entry not in self.exact[key]:
                    self.exact[key].append(entry)
        for key in self.exact:
            first = key.split(" ", 1)[0]
            if len(first) > 2:
                self.by_first_token[first].append(key)

    def _rank(self, entries: list[tuple[str, str]]) -> list[str]:
        return [rxcui for rxcui, _tty in sorted(entries, key=lambda e: TTY_TIER.get(e[1], 4))]

    def _best_tier(self, entries: list[tuple[str, str]]) -> int:
        return min((TTY_TIER.get(tty, 4) for _rxcui, tty in entries), default=4)

    def resolve_complete_product(self, core: str, hints: set[str], max_candidates: int = 3) -> list[str]:
        """Promote an ingredient+dose mention to the complete product (SCD/SBD).

        The mention gives ingredient + dose; the graded code additionally encodes
        the dose form. So treat the mention's `core` as a *prefix* of an official
        product string and pick among the completions, preferring the dose form
        the mention's route implied. Prefix (not equality) is what reaches a
        combination product from a single named ingredient, e.g.
        "chlorpheniramine 0.4 mg/ml" -> "chlorpheniramine 0.4 mg/ml /
        dextromethorphan 6 mg/ml / guaifenesin 40 mg/ml ... oral solution"
        (rxcui 360047, the task statement's worked example).

        Returns [] rather than a guess when nothing completes the prefix -- for
        Jaccard over codes an empty list and a wrong code score the same, and an
        empty one doesn't also drag a wrong answer into the union.
        """
        tokens = core.split()
        if not tokens:
            return []
        completions = [
            k for k in self.by_first_token.get(tokens[0], ())
            if k.startswith(core + " ")
        ]
        if not completions:
            return []
        if hints:
            completions = [k for k in completions if any(h in k for h in hints)]
            # No completion matches the stated route. Do NOT fall back to the
            # unfiltered list: that returns a product which *contradicts* the
            # mention -- "bumetanide 2mg iv" has no 2mg injectable (injectable
            # bumetanide is dosed per mL), so the only completion is
            # "bumetanide 2 mg oral tablet", an oral tablet for an IV order.
            # An empty result lets the caller keep a route-consistent answer
            # from another tier, and scores the same as a wrong code anyway.
            if not completions:
                return []
        # Prefer a complete product, then the shortest completion -- the shortest
        # suffix is the plainest form ("oral tablet" over "extended release oral
        # tablet"), which is what an unqualified mention means.
        completions.sort(key=lambda k: (self._best_tier(self.exact[k]), len(k), k))
        out: list[str] = []
        for key in completions:
            for rxcui in self._rank(self.exact[key]):
                if rxcui not in out:
                    out.append(rxcui)
            if len(out) >= max_candidates:
                break
        return out[:max_candidates]

    def lookup(self, text: str, max_candidates: int = 3) -> list[str]:
        key = norm(text)
        exact_entries = self.exact.get(key)
        core, hints = strip_administration_noise(key)
        if exact_entries is None and core and core != key:
            exact_entries = self.exact.get(core)

        # A dosed mention must resolve to a dose+form-specific code (task
        # statement: clonazepam 0.5mg 197527 vs 1.5mg 197528). If the only exact
        # hit is a form-less component (SCDC) or a loose synonym, a completed
        # product beats it; an exact hit that is already a complete product wins.
        if (self.promote_dose_form and _DOSE_RE.search(key)
                and self._best_tier(exact_entries or []) > TIER_OF_COMPLETE_PRODUCT):
            promoted = self.resolve_complete_product(core, hints, max_candidates)
            if promoted:
                return promoted
        if exact_entries:
            return self._rank(exact_entries)[:max_candidates]

        tokens = key.split()
        if not tokens:
            return []
        # Token-overlap fallback: candidate keys sharing the mention's first
        # word, ranked by how many of the mention's other tokens (usually
        # dose/form/route) also appear -- the same dose-aware intent as
        # fetch_rxnorm.py's drugs_by_ingredient, done against the offline index.
        candidates = self.by_first_token.get(tokens[0], [])
        if not candidates:
            return []
        rest = set(tokens[1:])
        scored = []
        for cand_key in candidates:
            cand_tokens = set(cand_key.split())
            overlap = len(rest & cand_tokens)
            scored.append((overlap, cand_key))
        scored.sort(key=lambda t: -t[0])
        best_overlap = scored[0][0]
        if not rest or best_overlap == 0:
            # Bare single-token query (nothing to filter on beyond the first
            # word) or no dose/form/route token matched at all -- an
            # ingredient-only hit is too weak a guess to return (same
            # reasoning as fetch_rxnorm.py's drugs_by_ingredient: bare
            # ingredient matches are noise, not signal). BUG FIXED
            # 2026-07-21: this used to read `best_overlap == 0 and rest`,
            # which is backwards -- `rest` empty (the bare-single-token case,
            # by far the most common miss) made the `and` short-circuit
            # False, so the guard never fired for exactly the queries it was
            # written to block. Confirmed empirically: pre-fix,
            # lookup("methicillin") and lookup("Enterococcus") -- a
            # bacterium name, not a drug at all -- both returned a "best
            # guess" RXCUI; this real-leaderboard-submission-visible bug is
            # the direct cause of a J_candidates regression (see worklog.md
            # 2026-07-21) once these fired on the model's own pre-existing
            # span/false-positive errors on THUỐC entities.
            return []
        out: list[str] = []
        for _overlap, cand_key in scored:
            if _overlap < best_overlap:
                break
            for rxcui in self._rank(self.exact[cand_key]):
                if rxcui not in out:
                    out.append(rxcui)
            if len(out) >= max_candidates:
                break
        return out[:max_candidates]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rrf-dir", type=Path, default=DEFAULT_RRF_DIR)
    parser.add_argument("--prescribe-dir", type=Path, default=ROOT / "prescribe" / "rrf")
    parser.add_argument("--out", type=Path, default=TERM_DIR / "rxnorm_full.csv")
    parser.add_argument("--names-out", type=Path, default=TERM_DIR / "rxnorm_drug_names.csv")
    parser.add_argument("--no-names", action="store_true", help="Skip the prescribable-names pass.")
    parser.add_argument("--lookup", help="Look up a single text against the (already built) index and exit.")
    parser.add_argument("--verify", action="store_true", help="After building, check the 360047 worked example.")
    args = parser.parse_args()

    if args.lookup:
        idx = RxNormOfflineIndex(args.out)
        print(f"{args.lookup!r} -> {idx.lookup(args.lookup)}")
        return 0

    index = build_index(args.rrf_dir)
    n_rows = write_index(index, args.out)
    print(f"wrote {n_rows} (text, candidate) rows for {len(index)} distinct normalized texts -> {args.out}")

    if not args.no_names:
        names = build_prescribable_names(args.prescribe_dir)
        if names:
            args.names_out.parent.mkdir(parents=True, exist_ok=True)
            with args.names_out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["text"])
                writer.writerows([n] for n in names)
            print(f"wrote {len(names)} prescribable drug names -> {args.names_out}")

    if args.verify:
        idx = RxNormOfflineIndex(args.out)
        target = "Chlorpheniramine 0.4 MG/ML / Dextromethorphan 6 MG/ML / Guaifenesin 40 MG/ML / Pseudoephedrine 6 MG/ML Oral Solution"
        result = idx.lookup(target)
        ok = "360047" in result
        print(f"verify: lookup({target!r}) -> {result} ({'OK' if ok else 'FAILED'}, expected 360047)")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
