from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hh_goa_rag.generation.evaluation import write_csv
from hh_goa_rag.generation.review import (
    add_human_quality,
    load_judgments,
    review_judgments,
    select_quality_winner,
)


def judgment(blind_id: str, scores: tuple[str, str, str] = ("", "", "")) -> dict[str, str]:
    return {
        "blind_output_id": blind_id,
        "question": "प्रश्न",
        "retrieved_evidence_json": json.dumps(
            [{"rank": 1, "parent_id": "p-1", "text": "सबूत"}], ensure_ascii=False
        ),
        "generated_status": "ANSWER",
        "generated_answer": "उत्तर",
        "human_correctness_1_to_5": scores[0],
        "human_relevance_1_to_5": scores[1],
        "human_faithfulness_1_to_5": scores[2],
        "human_notes": "",
    }


def test_review_saves_each_score_and_resumes_without_revealing_model(tmp_path: Path) -> None:
    path = tmp_path / "judgments.csv"
    write_csv(path, [judgment("blind-1")])
    answers = iter(("5", "q"))
    output: list[str] = []

    completed, total = review_judgments(
        path, input_fn=lambda _: next(answers), output_fn=output.append
    )

    assert (completed, total) == (0, 1)
    saved = load_judgments(path)[0]
    assert saved["human_correctness_1_to_5"] == "5"
    assert saved["human_relevance_1_to_5"] == ""
    assert "model" not in "\n".join(output).lower()

    answers = iter(("4", "5"))
    assert review_judgments(path, input_fn=lambda _: next(answers), output_fn=output.append) == (
        1,
        1,
    )


def test_unblinding_requires_complete_scores() -> None:
    aggregates = [{"model": "a", "latency_p50_ms": 10}]
    mapping = [{"blind_output_id": "blind-1", "model": "a"}]
    with pytest.raises(RuntimeError, match="incomplete"):
        add_human_quality(aggregates, [judgment("blind-1")], mapping, group_field="model")


def test_model_selection_uses_quality_then_latency_and_rejects_serious_failure() -> None:
    rows = [
        {
            "model": "fast-low-quality",
            "human_mean_quality": 4.0,
            "quality_eligible": True,
            "latency_p50_ms": 10,
        },
        {
            "model": "slow-high-quality",
            "human_mean_quality": 5.0,
            "quality_eligible": True,
            "latency_p50_ms": 100,
        },
        {
            "model": "failed-grounding",
            "human_mean_quality": 5.0,
            "quality_eligible": False,
            "latency_p50_ms": 1,
        },
    ]
    selected, winner = select_quality_winner(rows, phase="model")
    assert winner["model"] == "slow-high-quality"
    assert next(row for row in selected if row["selected"])["model"] == "slow-high-quality"


def test_topk_selection_chooses_smallest_k_at_best_quality() -> None:
    rows = [
        {"top_k": 1, "human_mean_quality": 4.5, "quality_eligible": True},
        {"top_k": 3, "human_mean_quality": 5.0, "quality_eligible": True},
        {"top_k": 5, "human_mean_quality": 5.0, "quality_eligible": True},
    ]
    _, winner = select_quality_winner(rows, phase="topk")
    assert winner["top_k"] == 3


def test_csv_remains_readable_after_atomic_save(tmp_path: Path) -> None:
    path = tmp_path / "judgments.csv"
    write_csv(path, [judgment("blind-1", ("5", "4", "3"))])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["human_faithfulness_1_to_5"] == "3"
