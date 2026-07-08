"""Blinded human review and quality-first generation selection."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .evaluation import write_csv

SCORE_FIELDS = (
    "human_correctness_1_to_5",
    "human_relevance_1_to_5",
    "human_faithfulness_1_to_5",
)
SCORE_LABELS = ("correctness", "relevance", "faithfulness")


def load_judgments(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No blinded judgments found in {path}")
    missing = set(SCORE_FIELDS) - set(rows[0])
    if missing:
        raise RuntimeError(f"Judgment sheet is missing columns: {sorted(missing)}")
    return rows


def review_judgments(
    path: Path,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Review incomplete cells, atomically saving after every entered score."""
    rows = load_judgments(path)
    completed_before = sum(_row_complete(row) for row in rows)
    output_fn(
        f"Blinded generation review: {completed_before}/{len(rows)} judgments complete. "
        "Enter q at any score prompt to stop."
    )
    for index, row in enumerate(rows, start=1):
        if _row_complete(row):
            continue
        _show_judgment(row, index, len(rows), output_fn)
        for field, label in zip(SCORE_FIELDS, SCORE_LABELS, strict=True):
            if _valid_score(row.get(field, "")):
                continue
            while True:
                value = input_fn(f"{label.title()} (1-5, q to save and quit): ").strip().lower()
                if value == "q":
                    return sum(_row_complete(item) for item in rows), len(rows)
                if _valid_score(value):
                    row[field] = value
                    write_csv(path, rows, fieldnames=list(rows[0]))
                    break
                output_fn("Please enter an integer from 1 to 5, or q.")
        output_fn("Saved.\n")
    return len(rows), len(rows)


def all_scores_complete(rows: Sequence[dict[str, str]]) -> bool:
    return bool(rows) and all(_row_complete(row) for row in rows)


def load_mapping(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append({key: str(value) for key, value in json.loads(line).items()})
    if not rows:
        raise RuntimeError(f"No blind mapping found in {path}")
    return rows


def add_human_quality(
    aggregate_rows: Sequence[dict[str, Any]],
    judgments: Sequence[dict[str, str]],
    mapping: Sequence[dict[str, str]],
    *,
    group_field: str,
) -> list[dict[str, Any]]:
    """Unblind completed scores and attach human means to aggregate rows."""
    if not all_scores_complete(judgments):
        raise RuntimeError("Human review is incomplete; model identities remain blinded")
    identities = {row["blind_output_id"]: row for row in mapping}
    judgment_ids = {row["blind_output_id"] for row in judgments}
    if judgment_ids != set(identities):
        raise RuntimeError("Blind mapping does not exactly match the judgment sheet")

    grouped: dict[str, list[dict[str, str]]] = {}
    reviewer_types = {row.get("reviewer_type", "human") or "human" for row in judgments}
    if len(reviewer_types) != 1:
        raise RuntimeError("Judgment sheet mixes reviewer types")
    reviewer_type = reviewer_types.pop()
    for judgment in judgments:
        identity = identities[judgment["blind_output_id"]]
        value = identity.get(group_field)
        if value is None:
            raise RuntimeError(f"Blind mapping has no {group_field!r} field")
        grouped.setdefault(value, []).append(judgment)

    enriched: list[dict[str, Any]] = []
    for aggregate in aggregate_rows:
        group_value = str(aggregate[group_field])
        scores = grouped.get(group_value)
        if not scores:
            raise RuntimeError(f"No human scores found for {group_field}={group_value}")
        correctness = _score_mean(scores, SCORE_FIELDS[0])
        relevance = _score_mean(scores, SCORE_FIELDS[1])
        faithfulness = _score_mean(scores, SCORE_FIELDS[2])
        serious_failures = sum(int(row[SCORE_FIELDS[2]]) == 1 for row in scores)
        enriched.append(
            {
                **aggregate,
                "human_correctness": correctness,
                "human_relevance": relevance,
                "human_faithfulness": faithfulness,
                "human_mean_quality": statistics.fmean(
                    (correctness, relevance, faithfulness)
                ),
                "serious_grounding_failures": serious_failures,
                "quality_eligible": serious_failures == 0,
                "reviewer_type": reviewer_type,
                "status": "COMPLETED",
            }
        )
    return enriched


def select_quality_winner(
    rows: Sequence[dict[str, Any]],
    *,
    phase: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select by human quality, applying only phase-appropriate tie-breakers."""
    eligible = [row for row in rows if row.get("quality_eligible")]
    if not eligible:
        raise RuntimeError("No configuration is eligible: serious grounding failures were found")
    best_quality = max(float(row["human_mean_quality"]) for row in eligible)
    quality_tied = [
        row for row in eligible if float(row["human_mean_quality"]) == best_quality
    ]
    if phase == "model":
        winner = min(quality_tied, key=lambda row: float(row["latency_p50_ms"]))
    elif phase == "topk":
        winner = min(quality_tied, key=lambda row: int(row["top_k"]))
    elif phase == "prompt":
        best_behavior = max(
            float(row.get("appropriate_context_behavior_rate") or 0) for row in quality_tied
        )
        behavior_tied = [
            row
            for row in quality_tied
            if float(row.get("appropriate_context_behavior_rate") or 0) == best_behavior
        ]
        winner = min(
            behavior_tied,
            key=lambda row: (
                float(row["latency_p50_ms"]),
                float(row.get("mean_prompt_tokens") or float("inf")),
            ),
        )
    else:
        raise ValueError(f"Unknown selection phase: {phase}")
    selected = [
        {
            **row,
            "selected": row is winner,
            "selection_basis": _selection_basis(phase, row is winner),
        }
        for row in rows
    ]
    return selected, winner


def _show_judgment(
    row: dict[str, str],
    index: int,
    total: int,
    output_fn: Callable[[str], None],
) -> None:
    output_fn("=" * 80)
    output_fn(f"Blinded answer {index}/{total} ({row['blind_output_id']})")
    output_fn(f"\nQUESTION\n{row['question']}")
    output_fn("\nRETRIEVED EVIDENCE")
    evidence = json.loads(row["retrieved_evidence_json"])
    for item in evidence:
        output_fn(
            f"\n[{item.get('rank', '?')}] {item.get('parent_id', '')}\n{item.get('text', '')}"
        )
    status = row.get("generated_status", "")
    answer = row.get("generated_answer", "") or "<empty answer>"
    output_fn(f"\nGENERATED ANSWER ({status})\n{answer}\n")


def _row_complete(row: dict[str, str]) -> bool:
    return all(_valid_score(row.get(field, "")) for field in SCORE_FIELDS)


def _valid_score(value: str) -> bool:
    return value in {"1", "2", "3", "4", "5"}


def _score_mean(rows: Sequence[dict[str, str]], field: str) -> float:
    return statistics.fmean(int(row[field]) for row in rows)


def _selection_basis(phase: str, selected: bool) -> str:
    if not selected:
        return "not_selected"
    if phase == "model":
        return "highest_human_quality_then_p50_latency"
    if phase == "topk":
        return "smallest_k_at_highest_human_quality"
    return "highest_human_quality_then_refusal_behavior_latency_tokens"
