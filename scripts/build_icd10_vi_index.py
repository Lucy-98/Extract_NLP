#!/usr/bin/env python3
"""Build a Vietnamese ICD-10 entity-linking index from the official Bộ Y tế
(Ministry of Health) bilingual ICD-10 catalog, and expose the matcher class
that ``run_pipeline.py`` uses to code Vietnamese ``CHẨN_ĐOÁN`` mentions that the
curated ``diagnoses.csv`` (mined only from the 100 turn-1 public files) can't
resolve correctly on unseen text.

Why this exists
---------------
``data/terminology/diagnoses.csv`` is mined entirely from the turn-1 public
``output/`` labels (322 rows). On a genuinely unseen test set it has two failure
modes, both hurting ``J_candidates`` (the highest-weighted score component, 0.4):

- a novel diagnosis with no curated entry gets a *wrong* ICD code via difflib's
  fuzzy fallback (matched to the nearest of 322 turn-1 strings), and
- "guessing wrong is strictly worse than guessing empty" for the Jaccard-over-
  codes metric (see worklog.md 2026-07-21 part 2).

This index replaces that guess with an authoritative lookup against the full
official Vietnamese ICD-10 vocabulary (~15.1k rows, WHO 2019 basis, Quyết định
4469/QĐ-BYT). Exact-normalized-name match first, then a token-subset fallback
(query tokens ⊆ an ICD title, preferring the most general / shortest code) --
the same "exact, then structured fallback, never a blind fuzzy guess" shape as
``RxNormOfflineIndex``. A matched 3-character category is then resolved to its
"unspecified" subcode (see ``unspecified_subcode``), because the graded truth
codes are 4-character ones and the scorer compares codes as exact strings.

Measured (2026-07-28, leakage-free: curated ``diagnoses.csv`` rebuilt from the
85-file train split only, scored on the 15-file holdout split's 66 ground-truth
``CHẨN_ĐOÁN`` mentions, perfect-NER assumption so only linking is measured):

    curated-only (previous pipeline)          J_candidates 0.5909
    + this index as a middle fallback         J_candidates 0.7424

with no change on the full 100-file corpus using the full committed table
(0.9852 either way), i.e. it only fires where the curated table had nothing.

Source
------
Official BYT bilingual catalog, columns: 14=3-char code, 16=3-char VI name,
17=full code (e.g. ``A00.0``), 19=WHO 2019 English name, 21=Vietnamese name
(``TÊN BỆNH``). Re-downloadable from
https://raw.githubusercontent.com/tamton23/primekg-vn-icd10-omop/HEAD/icd10_danh_muc.csv
(a machine-readable mirror of the QĐ 4469 Excel). The ~10MB raw file is
gitignored under ``data/terminology/raw/``; only the derived
``data/terminology/icd10_vi.csv`` (code,name_vi,name_en) is committed -- same
raw-in / derived-out split as build_rxnorm_rrf_index.py.

stdlib-only by design (no torch/transformers) so it runs anywhere, matching
every other script under scripts/ except run_pipeline.py.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp932/cp1252

ROOT = Path(__file__).resolve().parent.parent
TERM_DIR = ROOT / "data" / "terminology"
RAW_DEFAULT = TERM_DIR / "raw" / "icd10_danh_muc.csv"
OUT_DEFAULT = TERM_DIR / "icd10_vi.csv"

# Column indices in the BYT bilingual catalog (0-based). Verified against the
# committed source; the header cells themselves are blank due to Excel row-merge
# artifacts, so these are pinned positionally, not by header name.
COL_CODE3 = 14   # 3-character group code, e.g. "A00"
COL_NAME3_EN = 15
COL_NAME3_VI = 16
COL_CODE = 17    # specific code, e.g. "A00.0"
COL_NAME_EN = 19
COL_NAME_VI = 21

_CODE_RE = re.compile(r"^[A-Z]\d{2}(?:\.\d+)?$")


# Clinical-text vs official-ICD-title wording gaps that are systematic in
# Vietnamese and otherwise silently zero out a token-subset match. Applied
# identically to both the indexed ICD titles and the query, so both sides land
# on the same surface form. Longest phrases first so multi-word forms win.
_SYNONYMS = [
    # Sepsis: clinical Vietnamese says "nhiễm khuẩn/nhiễm trùng huyết", the BYT
    # catalog titles A40-A41 "nhiễm trùng hệ thống". Must come before the generic
    # nhiễm trùng<->nhiễm khuẩn rule below, which would otherwise leave "huyết"
    # as an unmatchable token and push the query onto an unrelated A20.x code.
    ("nhiễm khuẩn huyết", "nhiễm trùng hệ thống"),
    ("nhiễm trùng huyết", "nhiễm trùng hệ thống"),
    ("ung thư", "u ác tính"),        # clinical "cancer" vs ICD "malignant neoplasm"
    ("nhiễm trùng", "nhiễm khuẩn"),  # both used for "infection"
    ("tiểu đường", "đái tháo đường"),
    ("cao huyết áp", "tăng huyết áp"),
    (" típ ", " type "), (" typ ", " type "),  # diabetes/subtype markers
]


def norm(text: str) -> str:
    """Same normalization contract as build_terminology_index.norm (NFC,
    digit/letter spacing, lowercase, whitespace-collapsed), plus Vietnamese
    clinical<->ICD synonym folding so the token-subset matcher isn't defeated by
    surface wording differences."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(\d)([a-zA-ZÀ-ỹ])", r"\1 \2", text)
    text = re.sub(r"([a-zA-ZÀ-ỹ])(\d)", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text.strip().lower())
    for src, dst in _SYNONYMS:
        if src in text:
            text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text.strip())


def _clean(cell: str) -> str:
    return re.sub(r"\s+", " ", (cell or "").strip())


def build_rows(raw_path: Path) -> list[tuple[str, str, str]]:
    """Return sorted unique (code, name_vi, name_en) rows from the BYT catalog.

    Emits both specific 4/5-char codes (col 17/21/19) and the 3-char group codes
    (col 14/16/15). The 3-char rows matter because short clinical phrases
    ("viêm phổi", "tăng huyết áp") map to category-level ICD names far more often
    than to a fully-qualified subcode's long descriptive title.
    """
    seen: dict[str, tuple[str, str, str]] = {}
    with raw_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) <= COL_NAME_VI:
                continue
            for code_i, vi_i, en_i in ((COL_CODE, COL_NAME_VI, COL_NAME_EN),
                                       (COL_CODE3, COL_NAME3_VI, COL_NAME3_EN)):
                code = _clean(row[code_i]).upper()
                name_vi = _clean(row[vi_i])
                name_en = _clean(row[en_i]) if en_i < len(row) else ""
                if not code or not name_vi or not _CODE_RE.match(code):
                    continue
                # First Vietnamese name seen per (code, name) wins; dedupe exact.
                key = f"{code}\t{name_vi.lower()}"
                seen.setdefault(key, (code, name_vi, name_en))
    rows = sorted(seen.values(), key=lambda r: (r[0], r[1]))
    return rows


def write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name_vi", "name_en"])
        w.writerows(rows)


# Structural connectors + ultra-generic head nouns that carry no discriminative
# meaning on their own. Deliberately does NOT include clinical modifiers like
# "tăng"/"giảm"/"viêm"/"thiếu"/"cấp"/"mạn" -- those flip the diagnosis ("tăng
# kali máu" E87.5 vs "giảm kali máu" E87.6 are opposite codes), so they must
# stay as matchable tokens.
_STOPWORDS = {
    "bệnh", "hội", "chứng", "do", "và", "của", "khác", "không", "kèm", "theo",
    "được", "phân", "loại", "nơi", "hoặc", "ở", "tại", "với", "các", "một",
    "này", "đó", "the", "of", "and", "in", "to", "phần", "vùng", "thể",
}


_TOKEN_RE = re.compile(r"[0-9a-zà-ỹ]+", re.IGNORECASE)


def _tokens(normalized_name: str) -> frozenset:
    """Discriminative word tokens of an already-normalized name: punctuation
    stripped, stopwords removed. Single-character tokens are KEPT -- they are
    disease-subtype markers ("viêm gan B" -> B18/B16 vs generic "viêm gan"),
    which dropping by length would silently discard."""
    return frozenset(t for t in _TOKEN_RE.findall(normalized_name) if t not in _STOPWORDS)


# Title markers of an "unspecified" ICD subcode, used to resolve a matched
# 3-character category down to a gradeable 4-character code.
_UNSPECIFIED_MARKERS = (
    "không đặc hiệu", "không xác định", "không nói rõ", "không rõ", "tính không",
)

# Chapters that are never the answer for a bare diagnosis mention: V-Y are
# external causes of morbidity ("adverse effect of anti-infectives") and U is
# reserved for special purposes. They are full of generic words, so the
# token-subset fallback lands on them for noisy model spans -- "nhiễm trùng lợi"
# resolved to Y41.9 before this. Confirmed against the 542 graded turn-1
# diagnosis codes: not one is in these chapters.
_EXCLUDED_CHAPTERS = frozenset("UVWXY")

# Context-gated rather than excluded: O (pregnancy/childbirth), P (perinatal)
# and Z (factors influencing health status) are real answers only when the
# mention itself is obstetric/neonatal/administrative, so they are pushed last
# instead of dropped -- a plain "nhiễm khuẩn tiết niệu" must not resolve to
# O86.2 "UTI following delivery". Z appears exactly once in the turn-1 truth.
_CONTEXT_GATED_CHAPTERS = frozenset("OPZ")

# A one-word query is too ambiguous for the token-subset fallback: over 15k
# titles, a bare "viêm"/"thiếu"/"vàng" always finds *some* title that contains
# it, and the model emits exactly those fragments on noisy text. Measured on the
# turn-2 run: this drops 34 junk codes and costs nothing on either turn-1 eval
# (holdout J_candidates and the 229-text top-1 rate are both unchanged). An
# exact match against a full official title is still honoured at any length.
_MIN_FALLBACK_TOKENS = 2


class ICD10VietnameseIndex:
    """Exact-normalized-name lookup over the official BYT Vietnamese ICD-10
    vocabulary, then a token-subset fallback. Dependency-free.

    ``lookup(text)`` returns ``[code]`` (single best) or ``[]``. It never returns
    a blind fuzzy guess -- a miss degrades to empty, which the caller treats as
    "leave whatever the curated matcher decided / no candidate", per the
    guess-empty-beats-guess-wrong rule for J_candidates.
    """

    def __init__(self, csv_path: Path = OUT_DEFAULT):
        self.exact: dict[str, str] = {}
        # token -> list of (code, token_frozenset, is_three_char) for candidates
        self.by_token: dict[str, list[int]] = defaultdict(list)
        self.entries: list[tuple[str, frozenset, bool]] = []
        # 3-char category -> its subcodes, as (code, normalized_name)
        self.children: dict[str, list[tuple[str, str]]] = defaultdict(list)
        if not csv_path.exists():
            return
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                code = row["code"].strip()
                name = norm(row["name_vi"])
                if not code or not name:
                    continue
                # Exact key: shortest code wins ties (most general default).
                if name not in self.exact or len(code) < len(self.exact[name]):
                    self.exact[name] = code
                if "." in code:
                    self.children[code.split(".")[0]].append((code, name))
                toks = _tokens(name)
                if not toks:
                    continue
                idx = len(self.entries)
                self.entries.append((code, toks, "." not in code))
                for t in toks:
                    self.by_token[t].append(idx)

    def unspecified_subcode(self, category: str) -> str | None:
        """The gradeable "unspecified" child of a 3-character category, or None.

        Short clinical phrases ("thuyên tắc phổi", "suy hô hấp") match a category
        title exactly, but the graded truth codes are 4-character ones compared as
        exact strings -- returning ``I26`` where truth says ``I26.9`` scores the
        same zero as returning a wrong code. ICD-10 reserves ``.9`` for
        "unspecified" within a category, which is what an unqualified clinical
        mention means, so prefer ``.9`` and only then fall back to a title-marker
        scan. Both passes take the *shortest* code first, so a category with 5-char
        codes resolves to ``J96.9`` ("Suy hô hấp, tính không xác định") rather than
        the more specific ``J96.09``.
        """
        kids = sorted(self.children.get(category, ()), key=lambda kv: (len(kv[0]), kv[0]))
        for code, _name in kids:
            if code.endswith(".9"):
                return code
        for code, name in kids:
            if any(marker in name for marker in _UNSPECIFIED_MARKERS):
                return code
        return None

    def lookup(self, text: str) -> list[str]:
        code = self._best_code(text)
        if code is None:
            return []
        if "." not in code:
            code = self.unspecified_subcode(code) or code
        return [code]

    def _best_code(self, text: str) -> str | None:
        key = norm(text)
        if not key:
            return None
        if key in self.exact:
            return self.exact[key]
        q = _tokens(key)
        if len(q) < _MIN_FALLBACK_TOKENS:
            return None
        # Candidate entries: any that share at least one meaningful query token.
        cand_idx: set[int] = set()
        for t in q:
            cand_idx.update(self.by_token.get(t, ()))
        best = None  # (score tuple, code)
        for i in cand_idx:
            code, toks, is3 = self.entries[i]
            if code[0] in _EXCLUDED_CHAPTERS:
                continue
            covered = len(q & toks)
            if covered < len(q):
                continue  # require ALL query tokens present in the ICD title
            extra = len(toks) - covered  # how many extra words the title carries
            context_specific = code[0] in _CONTEXT_GATED_CHAPTERS
            # Prefer: non-context-specific -> 3-char category (more general
            # default; the all-tokens-required rule above already prevents a
            # too-broad parent from beating a more specific subcode) -> fewest
            # extra words -> lexically smaller code (stable).
            score = (context_specific, 0 if is3 else 1, extra, code)
            if best is None or score < best[0]:
                best = (score, code)
        return best[1] if best else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT,
                        help="BYT bilingual catalog CSV (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT,
                        help="derived icd10_vi.csv to write (default: %(default)s)")
    parser.add_argument("--lookup", help="Look up one diagnosis text and exit.")
    args = parser.parse_args()

    if args.lookup:
        idx = ICD10VietnameseIndex(args.out)
        print(f"{args.lookup!r} -> {idx.lookup(args.lookup)}")
        return 0

    if not args.raw.exists():
        print(f"raw catalog not found: {args.raw}\n"
              f"download from https://raw.githubusercontent.com/"
              f"tamton23/primekg-vn-icd10-omop/HEAD/icd10_danh_muc.csv",
              file=sys.stderr)
        return 1

    rows = build_rows(args.raw)
    write_csv(args.out, rows)
    codes = {r[0] for r in rows}
    print(f"icd10_vi.csv: {len(rows)} rows, {len(codes)} distinct codes -> {args.out}")

    # Smoke test on a few common Vietnamese diagnosis phrases.
    idx = ICD10VietnameseIndex(args.out)
    for probe in ["tăng huyết áp", "đái tháo đường", "viêm phổi", "suy tim",
                  "bệnh dại", "sốt xuất huyết", "béo phì", "trầm cảm",
                  "nhiễm khuẩn đường tiết niệu", "viêm dạ dày",
                  # holdout-verified cases for the category -> unspecified-subcode
                  # expansion and the sepsis synonym (see module docstring)
                  "thuyên tắc phổi", "suy hô hấp", "nhiễm khuẩn huyết",
                  "tăng kali máu"]:
        print(f"  {probe:32} -> {idx.lookup(probe)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
