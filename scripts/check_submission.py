#!/usr/bin/env python3
"""Validate output files and optionally simulate the competition metrics.

Usage:
  python scripts/check_submission.py --pred output --input input
  python scripts/check_submission.py --pred output --input input --truth path/to/ground_truth

Without --truth, this script performs submission sanity checks only. With
--truth, it computes an approximation of the public metric from the task
statement: WER over entity text, Jaccard over assertions, and weighted Jaccard
over candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DIAG = "CH\u1ea8N_\u0110O\u00c1N"
SYM = "TRI\u1ec6U_CH\u1ee8NG"
DRUG = "THU\u1ed0C"
TEST = "T\u00caN_X\u00c9T_NGHI\u1ec6M"
RESULT = "K\u1ebeT_QU\u1ea2_X\u00c9T_NGHI\u1ec6M"

VALID_TYPES = {DIAG, SYM, DRUG, TEST, RESULT}
ASSERTION_TYPES = {DIAG, SYM, DRUG}
VALID_ASSERTIONS = {"isNegated", "isHistorical", "isFamily"}
CANDIDATE_TYPES = {DIAG, DRUG}
EMPTY = "__EMPTY__"


@dataclass
class CheckReport:
    errors: list[str]
    warnings: list[str]
    file_count: int = 0
    entity_count: int = 0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_file_ids(folder: Path) -> list[int]:
    ids: list[int] = []
    for p in folder.glob("*.json"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return sorted(ids)


def get_input_ids(folder: Path) -> list[int]:
    ids: list[int] = []
    for p in folder.glob("*.txt"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return sorted(ids)


def is_word_char(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def validate_prediction(pred_dir: Path, input_dir: Path, expected_count: int | None = None) -> CheckReport:
    report = CheckReport(errors=[], warnings=[])

    ids = get_file_ids(pred_dir)
    expected_ids = list(range(1, expected_count + 1)) if expected_count is not None else get_input_ids(input_dir)
    expected_set = set(expected_ids)
    report.file_count = len(ids)
    if len(ids) != len(expected_ids):
        report.errors.append(f"Expected {len(expected_ids)} json files, found {len(ids)}.")

    missing = [i for i in expected_ids if i not in ids]
    if missing:
        report.errors.append(f"Missing output files: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    extra = [i for i in ids if i not in expected_set]
    if extra:
        report.warnings.append(f"Unexpected output file ids: {extra[:20]}{'...' if len(extra) > 20 else ''}")

    for i in ids:
        pred_path = pred_dir / f"{i}.json"
        input_path = input_dir / f"{i}.txt"

        if not input_path.exists():
            report.errors.append(f"{pred_path}: missing matching input file {input_path}.")
            continue

        try:
            data = load_json(pred_path)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"{pred_path}: invalid JSON: {exc}")
            continue

        if not isinstance(data, list):
            report.errors.append(f"{pred_path}: top-level JSON must be a list.")
            continue

        text = input_path.read_text(encoding="utf-8")
        seen_exact: dict[tuple[Any, ...], int] = {}
        sort_keys: list[tuple[int, int, str, str]] = []

        for idx, ent in enumerate(data):
            report.entity_count += 1
            where = f"{pred_path.name}[{idx}]"

            if not isinstance(ent, dict):
                report.errors.append(f"{where}: entity must be an object.")
                continue

            required = {"text", "type", "position", "assertions"}
            missing_fields = required - set(ent)
            if missing_fields:
                report.errors.append(f"{where}: missing fields {sorted(missing_fields)}.")
                continue

            ent_text = ent.get("text")
            ent_type = ent.get("type")
            pos = ent.get("position")
            assertions = ent.get("assertions")
            candidates = ent.get("candidates")

            if ent_type not in VALID_TYPES:
                report.errors.append(f"{where}: invalid type {ent_type!r}.")

            if not isinstance(ent_text, str):
                report.errors.append(f"{where}: text must be a string.")
                continue

            if not (
                isinstance(pos, list)
                and len(pos) == 2
                and all(isinstance(x, int) for x in pos)
            ):
                report.errors.append(f"{where}: position must be [start, end] integers.")
                continue

            start, end = pos
            sort_keys.append((start, end, str(ent_type), ent_text))
            pos_in_bounds = 0 <= start <= end <= len(text)
            if not pos_in_bounds:
                report.errors.append(f"{where}: position {pos} is outside input length {len(text)}.")
            elif text[start:end] != ent_text:
                got = text[start:end]
                report.errors.append(
                    f"{where}: span mismatch at {pos}: json={ent_text!r}, input={got!r}."
                )

            if pos_in_bounds and start < end:
                left = text[start - 1] if start > 0 else " "
                right = text[end] if end < len(text) else " "
                if len(ent_text.strip()) <= 4 and (is_word_char(left) or is_word_char(right)):
                    report.warnings.append(
                        f"{where}: very short entity {ent_text!r} may be embedded in a larger word."
                    )

            if not isinstance(assertions, list) or not all(isinstance(a, str) for a in assertions):
                report.errors.append(f"{where}: assertions must be a list of strings.")
            else:
                invalid = sorted(set(assertions) - VALID_ASSERTIONS)
                if invalid:
                    report.errors.append(f"{where}: invalid assertions {invalid}.")
                if ent_type not in ASSERTION_TYPES and assertions:
                    report.warnings.append(
                        f"{where}: assertions on non disease/drug/symptom type {ent_type!r}."
                    )

            if ent_type in CANDIDATE_TYPES and "candidates" not in ent:
                report.errors.append(f"{where}: {ent_type!r} should include candidates field.")
            if candidates is not None and (
                not isinstance(candidates, list) or not all(isinstance(c, str) for c in candidates)
            ):
                report.errors.append(f"{where}: candidates must be a list of strings if present.")

            key = (tuple(pos), ent_type, ent_text)
            seen_exact[key] = seen_exact.get(key, 0) + 1

        if sort_keys != sorted(sort_keys):
            report.warnings.append(f"{pred_path.name}: entities are not sorted by position.")

        duplicates = [k for k, count in seen_exact.items() if count > 1]
        if duplicates:
            report.warnings.append(
                f"{pred_path.name}: duplicate entities found: {duplicates[:5]}"
                f"{'...' if len(duplicates) > 5 else ''}"
            )

        sorted_entities = sorted(
            [e for e in data if isinstance(e, dict) and isinstance(e.get("position"), list)],
            key=lambda e: (e["position"][0], e["position"][1], str(e.get("type")), str(e.get("text"))),
        )
        for a, b in zip(sorted_entities, sorted_entities[1:]):
            a_start, a_end = a["position"]
            b_start, b_end = b["position"]
            if max(a_start, b_start) < min(a_end, b_end):
                report.warnings.append(
                    f"{pred_path.name}: overlapping entities {a.get('text')!r} {a['position']} "
                    f"and {b.get('text')!r} {b['position']}."
                )

    return report


def words_for_text_metric(entities: list[dict[str, Any]]) -> list[str]:
    """Token stream for the text WER proxy.

    The task statement says a concept with correct text but wrong type is
    counted twice and receives zero credit. Plain text-only WER would treat
    `TRIỆU_CHỨNG "ho"` and `CHẨN_ĐOÁN "ho"` as identical, so prefix each word
    with the entity type. A wrong type therefore becomes substitutions across
    the mention's words, producing zero text credit for that concept.
    """
    ordered = sorted(
        entities,
        key=lambda e: (
            e.get("position", [10**12, 10**12])[0],
            e.get("position", [10**12, 10**12])[1],
            str(e.get("type", "")),
            str(e.get("text", "")),
        ),
    )
    tokens: list[str] = []
    for e in ordered:
        ent_type = str(e.get("type", ""))
        words = str(e.get("text", "")).split()
        tokens.extend(f"{ent_type}\t{word}" for word in words)
    return tokens


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ai == bj else 1),
            )
        prev = cur
    return prev[-1]


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def jaccard(a: set[tuple[Any, ...]], b: set[tuple[Any, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a and b:
        return 0.0
    return len(a & b) / len(a | b)


def occurrence_keys(entities: list[dict[str, Any]], valid_types: set[str]) -> list[tuple[dict[str, Any], tuple[str, str, int]]]:
    """Return entities with stable per-sample occurrence keys.

    Sets collapse duplicate mentions such as two separate "táo bón" symptoms.
    The official output is instance-based, so keep duplicates distinct by
    assigning an occurrence index per `(text, type)` in document order.
    """
    ordered = sorted(
        entities,
        key=lambda e: (
            e.get("position", [10**12, 10**12])[0],
            e.get("position", [10**12, 10**12])[1],
            str(e.get("type", "")),
            str(e.get("text", "")),
        ),
    )
    counts: dict[tuple[str, str], int] = {}
    out: list[tuple[dict[str, Any], tuple[str, str, int]]] = []
    for e in ordered:
        ent_type = str(e.get("type", ""))
        if ent_type not in valid_types:
            continue
        base = (str(e.get("text", "")), ent_type)
        counts[base] = counts.get(base, 0) + 1
        out.append((e, (*base, counts[base])))
    return out


def assertion_items(entities: list[dict[str, Any]]) -> set[tuple[str, str, int, str]]:
    items: set[tuple[str, str, int, str]] = set()
    for e, base in occurrence_keys(entities, ASSERTION_TYPES):
        assertions = e.get("assertions") or []
        if not assertions:
            items.add((*base, EMPTY))
        else:
            for assertion in assertions:
                items.add((*base, str(assertion)))
    return items


def candidate_items(entities: list[dict[str, Any]]) -> set[tuple[str, str, int, str]]:
    items: set[tuple[str, str, int, str]] = set()
    for e, base in occurrence_keys(entities, CANDIDATE_TYPES):
        candidates = e.get("candidates") or []
        if not candidates:
            items.add((*base, EMPTY))
        else:
            for candidate in candidates:
                items.add((*base, str(candidate)))
    return items


def candidate_weight(entities: list[dict[str, Any]]) -> int:
    weight = 0
    for e in entities:
        if e.get("type") in CANDIDATE_TYPES:
            weight += len(e.get("candidates") or []) + 1
    return weight


def simulate_metrics(pred_dir: Path, truth_dir: Path) -> dict[str, float]:
    ids = sorted(set(get_file_ids(pred_dir)) | set(get_file_ids(truth_dir)))
    if not ids:
        raise ValueError("No json files found for metric simulation.")

    text_scores: list[float] = []
    assertion_scores: list[float] = []
    weighted_candidate_sum = 0.0
    weight_sum = 0
    wer_values: list[float] = []

    for i in ids:
        pred_path = pred_dir / f"{i}.json"
        truth_path = truth_dir / f"{i}.json"
        pred = load_json(pred_path) if pred_path.exists() else []
        truth = load_json(truth_path) if truth_path.exists() else []

        sample_wer = wer(words_for_text_metric(truth), words_for_text_metric(pred))
        wer_values.append(sample_wer)
        text_scores.append(1.0 - sample_wer)

        assertion_scores.append(jaccard(assertion_items(truth), assertion_items(pred)))

        sample_candidate_score = jaccard(candidate_items(truth), candidate_items(pred))
        sample_weight = candidate_weight(truth)
        weighted_candidate_sum += sample_candidate_score * sample_weight
        weight_sum += sample_weight

    text_score = sum(text_scores) / len(text_scores)
    assertions_score = sum(assertion_scores) / len(assertion_scores)
    candidates_score = weighted_candidate_sum / weight_sum if weight_sum else 1.0
    final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score

    return {
        "num_scored": float(len(ids)),
        "WER_mean": sum(wer_values) / len(wer_values),
        "text_score": text_score,
        "J_assertion": assertions_score,
        "J_candidates": candidates_score,
        "final_score": final_score,
    }


def print_report(report: CheckReport) -> None:
    print("== Submission sanity check ==")
    print(f"files: {report.file_count}")
    print(f"entities: {report.entity_count}")
    print(f"errors: {len(report.errors)}")
    print(f"warnings: {len(report.warnings)}")

    if report.errors:
        print("\nErrors:")
        for msg in report.errors[:80]:
            print(f"- {msg}")
        if len(report.errors) > 80:
            print(f"- ... {len(report.errors) - 80} more errors")

    if report.warnings:
        print("\nWarnings:")
        for msg in report.warnings[:80]:
            print(f"- {msg}")
        if len(report.warnings) > 80:
            print(f"- ... {len(report.warnings) - 80} more warnings")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ViettelRace output files before submission.")
    parser.add_argument("--pred", default="output", type=Path, help="Prediction output folder.")
    parser.add_argument("--input", default="input_turn2", type=Path, help="Input txt folder.")
    parser.add_argument("--truth", type=Path, help="Optional ground-truth folder for metric simulation.")
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Expected number of sequential json files. Defaults to the numbered files in --input.",
    )
    args = parser.parse_args()

    report = validate_prediction(args.pred, args.input, args.expected_count)
    print_report(report)

    if args.truth:
        print("\n== Metric simulation ==")
        metrics = simulate_metrics(args.pred, args.truth)
        for key, value in metrics.items():
            if key == "num_scored":
                print(f"{key}: {int(value)}")
            else:
                print(f"{key}: {value:.6f}")
        print(
            "\nNote: this is a local approximation. It is most useful when --truth points "
            "to an actual labeled folder with the same schema."
        )

    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
