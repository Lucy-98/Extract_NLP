#!/usr/bin/env python3
"""Metric-aware local improvement loop for ViettelRace outputs.

The public score cannot be reproduced without ground truth, so this agent is
conservative by default:

* Always validates schema, spans, ordering, duplicates, and overlaps.
* Learns known candidates from the current/baseline outputs and fills exact
  repeated missing candidates.
* Applies only high-confidence false-positive and span-expansion rules.
* If --truth is provided, keeps a loop iteration only when the simulated final
  metric improves.
* If --truth is omitted, keeps only sanity-safe rules that pass validation.

Superseded by scripts/run_pipeline.py (model-based extraction + terminology
linking). Kept here only as a reference fallback / for auditing the old
hand-tuned output; do not use it to regenerate a submission from scratch --
its wordlists are overfit to the 100 public files (see README.md).

Examples:
  python legacy/scripts/auto_improve_agent.py --pred output --input input
  python legacy/scripts/auto_improve_agent.py --pred output --input input --apply
  python legacy/scripts/auto_improve_agent.py --pred output --input input --truth path/to/ground_truth --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_submission import (  # noqa: E402
    ASSERTION_TYPES,
    CANDIDATE_TYPES,
    DIAG,
    DRUG,
    RESULT,
    SYM,
    simulate_metrics,
    validate_prediction,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class AgentScore:
    value: float
    errors: int
    warnings: int
    entities: int
    candidate_coverage: float
    metric: dict[str, float] | None = None


def read_entities(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_entities(path: Path, entities: list[dict[str, Any]]) -> None:
    entities.sort(key=lambda e: (e["position"][0], e["position"][1], e.get("type", ""), e.get("text", "")))
    path.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_output(src: Path, dst: Path) -> None:
    if dst.exists():
        resolved = dst.resolve()
        if ROOT.resolve() not in resolved.parents:
            raise RuntimeError(f"Refusing to delete outside workspace: {resolved}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def candidate_coverage(pred_dir: Path) -> tuple[int, int]:
    total = 0
    filled = 0
    for i in range(1, 101):
        for ent in read_entities(pred_dir / f"{i}.json"):
            if ent.get("type") in CANDIDATE_TYPES:
                total += 1
                filled += bool(ent.get("candidates"))
    return filled, total


def score_state(pred_dir: Path, input_dir: Path, truth_dir: Path | None) -> AgentScore:
    report = validate_prediction(pred_dir, input_dir)
    filled, total = candidate_coverage(pred_dir)
    coverage = filled / total if total else 1.0

    if report.errors:
        return AgentScore(
            value=-1e9,
            errors=len(report.errors),
            warnings=len(report.warnings),
            entities=report.entity_count,
            candidate_coverage=coverage,
        )

    if truth_dir:
        metrics = simulate_metrics(pred_dir, truth_dir)
        return AgentScore(
            value=metrics["final_score"],
            errors=0,
            warnings=len(report.warnings),
            entities=report.entity_count,
            candidate_coverage=coverage,
            metric=metrics,
        )

    # Proxy objective when ground truth is absent. This is intentionally simple:
    # validation health dominates, candidate coverage helps, warning count hurts.
    warning_penalty = min(len(report.warnings) * 0.001, 0.10)
    value = 0.70 + 0.25 * coverage - warning_penalty
    return AgentScore(
        value=value,
        errors=0,
        warnings=len(report.warnings),
        entities=report.entity_count,
        candidate_coverage=coverage,
    )


def collect_known_candidates(*dirs: Path) -> dict[tuple[str, str], list[str]]:
    known: dict[tuple[str, str], list[str]] = {}
    for folder in dirs:
        if not folder or not folder.exists():
            continue
        for path in folder.glob("*.json"):
            try:
                entities = read_entities(path)
            except Exception:  # noqa: BLE001
                continue
            for ent in entities:
                typ = ent.get("type")
                candidates = ent.get("candidates") or []
                if typ in CANDIDATE_TYPES and candidates:
                    known[(typ, norm_text(str(ent.get("text", ""))))] = [str(c) for c in candidates]
    return known


def apply_sanity_fixes(pred_dir: Path) -> int:
    changes = 0
    for i in range(1, 101):
        path = pred_dir / f"{i}.json"
        entities = read_entities(path)
        new_entities: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        for ent in entities:
            typ = ent.get("type")
            if typ in CANDIDATE_TYPES and "candidates" not in ent:
                ent["candidates"] = []
                changes += 1
            if typ not in ASSERTION_TYPES and ent.get("assertions"):
                ent["assertions"] = []
                changes += 1

            key = (tuple(ent.get("position", [])), typ, ent.get("text"))
            if key in seen:
                changes += 1
                continue
            seen.add(key)
            new_entities.append(ent)

        old_order = [e.get("position") for e in new_entities]
        new_entities.sort(key=lambda e: (e["position"][0], e["position"][1], e.get("type", ""), e.get("text", "")))
        if old_order != [e.get("position") for e in new_entities]:
            changes += 1
        write_entities(path, new_entities)
    return changes


def apply_learned_candidates(pred_dir: Path, known: dict[tuple[str, str], list[str]]) -> int:
    changes = 0
    for i in range(1, 101):
        path = pred_dir / f"{i}.json"
        entities = read_entities(path)
        dirty = False
        for ent in entities:
            typ = ent.get("type")
            if typ not in CANDIDATE_TYPES or ent.get("candidates"):
                continue
            candidates = known.get((typ, norm_text(str(ent.get("text", "")))))
            if candidates:
                ent["candidates"] = candidates
                changes += 1
                dirty = True
        if dirty:
            write_entities(path, entities)
    return changes


def is_inside(text: str, start: int, end: int, phrase: str) -> bool:
    left = max(0, start - len(phrase))
    right = min(len(text), end + len(phrase))
    return phrase in text[left:right]


def apply_false_positive_rules(pred_dir: Path, input_dir: Path) -> int:
    changes = 0
    remove_positions: dict[int, set[tuple[int, int]]] = {}

    for i in range(1, 101):
        text = (input_dir / f"{i}.txt").read_text(encoding="utf-8")
        entities = read_entities(pred_dir / f"{i}.json")
        for ent in entities:
            typ = ent.get("type")
            if typ not in {DIAG, SYM}:
                continue
            start, end = ent["position"]
            ent_text = str(ent.get("text", "")).lower()
            ctx = text[max(0, start - 40) : min(len(text), end + 60)].lower()

            remove = False
            if ent_text == "ban" and is_inside(text.lower(), start, end, "ban đầu"):
                remove = True
            elif ent_text == "phù" and is_inside(text.lower(), start, end, "phù hợp"):
                remove = True
            elif ent_text == "sốc" and ("sốc điện" in ctx or "bị sốc khi" in ctx):
                remove = True
            elif ent_text == "liệt" and is_inside(text.lower(), start, end, "tuyến tiền liệt"):
                remove = True
            elif ent_text == "ho" and ("- ho bệnh" in ctx or "- ho rung" in ctx or "- ho đái" in ctx):
                remove = True

            if remove:
                remove_positions.setdefault(i, set()).add((start, end))

    for i, positions in remove_positions.items():
        path = pred_dir / f"{i}.json"
        entities = read_entities(path)
        new_entities = [e for e in entities if tuple(e.get("position", [])) not in positions]
        if len(new_entities) != len(entities):
            changes += len(entities) - len(new_entities)
            write_entities(path, new_entities)
    return changes


def replace_contained_diagnosis(
    entities: list[dict[str, Any]],
    text: str,
    phrase: str,
    candidate: str,
    assertions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    changes = 0
    start = 0
    while True:
        idx = text.find(phrase, start)
        if idx < 0:
            break
        end = idx + len(phrase)
        contained = [
            ent
            for ent in entities
            if ent.get("type") == DIAG
            and idx <= ent.get("position", [-1, -1])[0]
            and ent.get("position", [-1, -1])[1] <= end
            and ent.get("position") != [idx, end]
        ]
        if contained:
            inherited = assertions
            if inherited is None:
                inherited = contained[0].get("assertions", [])
            entities = [ent for ent in entities if ent not in contained]
            if not any(ent.get("type") == DIAG and ent.get("position") == [idx, end] for ent in entities):
                entities.append(
                    {
                        "text": text[idx:end],
                        "type": DIAG,
                        "position": [idx, end],
                        "assertions": list(inherited),
                        "candidates": [candidate],
                    }
                )
            changes += len(contained)
        start = idx + 1
    return entities, changes


def apply_span_expansions(pred_dir: Path, input_dir: Path) -> int:
    # High-confidence phrase expansions seen repeatedly in this dataset. The
    # list is intentionally small; add phrases only after manual inspection.
    phrase_rules = [
        ("u \u00e1c tr\u1ef1c tr\u00e0ng", "C20"),
        ("U \u00e1c c\u1ee7a \u0111\u1ea1i tr\u00e0ng", "C18.9"),
        ("Ung th\u01b0 ph\u1ed5i kh\u00f4ng t\u1ebf b\u00e0o nh\u1ecf", "C34.9"),
        ("ung th\u01b0 ph\u1ed5i", "C34.9"),
        ("ung th\u01b0 v\u00fa", "C50.9"),
        ("ung th\u01b0 bi\u1ec3u m\u00f4 t\u1ebf b\u00e0o th\u1eadn", "C64.9"),
        ("ung th\u01b0 \u0111\u01b0\u1eddng m\u1eadt", "C24.0"),
        ("ung th\u01b0 bi\u1ec3u m\u00f4 t\u1ebf b\u00e0o m\u1eadt", "C24.0"),
        ("Vi\u00eam lo\u00e9t \u0111\u1ea1i tr\u00e0ng", "K51.9"),
        ("t\u0103ng s\u1ea3n tuy\u1ebfn ti\u1ec1n li\u1ec7t", "N40.0"),
        ("li\u1ec7t hai chi d\u01b0\u1edbi", "G82.2"),
    ]

    changes = 0
    for i in range(1, 101):
        path = pred_dir / f"{i}.json"
        text = (input_dir / f"{i}.txt").read_text(encoding="utf-8")
        entities = read_entities(path)
        file_changes = 0
        for phrase, candidate in phrase_rules:
            entities, delta = replace_contained_diagnosis(entities, text, phrase, candidate)
            file_changes += delta
        if file_changes:
            changes += file_changes
            write_entities(path, entities)
    return changes


def run_agent(args: argparse.Namespace) -> int:
    pred_dir = args.pred
    input_dir = args.input
    truth_dir = args.truth
    run_dir = ROOT / ".agent_runs" / time.strftime("auto_%Y%m%d_%H%M%S")
    work_dir = run_dir / "work_output"
    before_dir = run_dir / "before_output"
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_output(pred_dir, work_dir)
    copy_output(pred_dir, before_dir)

    known_dirs = [pred_dir]
    if args.learn_from:
        known_dirs.extend(args.learn_from)
    known = collect_known_candidates(*known_dirs)

    best_score = score_state(work_dir, input_dir, truth_dir)
    print("== Auto improve agent ==")
    print(f"work_dir: {work_dir}")
    print(f"initial_score: {best_score.value:.6f}")
    print(f"initial_entities: {best_score.entities}")
    print(f"initial_candidate_coverage: {best_score.candidate_coverage:.4f}")

    total_changes = 0
    for round_idx in range(1, args.max_rounds + 1):
        round_backup = run_dir / f"round_{round_idx}_backup"
        copy_output(work_dir, round_backup)

        changes = 0
        changes += apply_sanity_fixes(work_dir)
        changes += apply_learned_candidates(work_dir, known)
        changes += apply_false_positive_rules(work_dir, input_dir)
        changes += apply_span_expansions(work_dir, input_dir)

        new_score = score_state(work_dir, input_dir, truth_dir)
        print(
            f"round {round_idx}: changes={changes}, score={new_score.value:.6f}, "
            f"errors={new_score.errors}, warnings={new_score.warnings}, "
            f"coverage={new_score.candidate_coverage:.4f}"
        )

        if changes == 0:
            shutil.rmtree(round_backup)
            break

        keep = new_score.errors == 0
        if truth_dir:
            keep = keep and new_score.value >= best_score.value

        if not keep:
            copy_output(round_backup, work_dir)
            print(f"round {round_idx}: reverted")
            shutil.rmtree(round_backup)
            break

        total_changes += changes
        best_score = new_score
        shutil.rmtree(round_backup)

    final_report = validate_prediction(work_dir, input_dir)
    if final_report.errors:
        print("final validation failed; refusing to apply")
        for err in final_report.errors[:20]:
            print(f"- {err}")
        return 1

    print("== Final candidate ==")
    print(f"changes_kept: {total_changes}")
    print(f"entities: {final_report.entity_count}")
    print(f"warnings: {len(final_report.warnings)}")
    if best_score.metric:
        print(
            "metric: "
            f"final={best_score.metric['final_score']:.6f}, "
            f"text={best_score.metric['text_score']:.6f}, "
            f"assertion={best_score.metric['J_assertion']:.6f}, "
            f"candidates={best_score.metric['J_candidates']:.6f}"
        )

    if args.apply:
        for path in work_dir.glob("*.json"):
            shutil.copy2(path, pred_dir / path.name)
        print(f"applied_to: {pred_dir}")
        print(f"backup: {before_dir}")
    else:
        print("dry_run: no files changed in output")
        print("use --apply to copy work_output back to output")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Loop over safe output improvements and validation.")
    parser.add_argument("--pred", default=ROOT / "output", type=Path)
    parser.add_argument("--input", default=ROOT / "input", type=Path)
    parser.add_argument("--truth", type=Path, help="Optional ground-truth folder for exact local metric simulation.")
    parser.add_argument(
        "--learn-from",
        action="append",
        type=Path,
        default=[],
        help="Additional output folders to learn exact candidates from.",
    )
    parser.add_argument("--max-rounds", default=5, type=int)
    parser.add_argument("--apply", action="store_true", help="Write accepted changes back to --pred.")
    return run_agent(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
