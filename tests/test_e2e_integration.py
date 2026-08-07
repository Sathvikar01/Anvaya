from __future__ import annotations

from pathlib import Path

import pytest

from eval.evaluate_e2e import (
    _development_cases,
    _pending_category_rows,
    _pending_evaluation,
    _pending_latency_rows,
    _validate_manifest,
)
from hh_goa_rag.io import read_jsonl


def test_missing_recordings_produce_pending_not_synthetic_metrics() -> None:
    manifest = list(read_jsonl(Path("eval/stt_manifest.jsonl")))
    cases = _development_cases(Path("eval/eval_dataset.jsonl"))
    _validate_manifest(manifest, cases)

    rows = _pending_evaluation(manifest, cases)

    assert len(rows) == 24
    assert {row["classification"] for row in rows} == {"PENDING"}
    assert {row["status"] for row in rows} == {"PENDING_REAL_RECORDING"}
    assert all(row["total_e2e_ms"] == "" for row in rows)
    assert all(row["stt_wer"] == "" for row in rows)
    assert all(row["route_correct"] == "" for row in rows)


def test_pending_latency_and_category_tables_have_blank_metrics() -> None:
    manifest = list(read_jsonl(Path("eval/stt_manifest.jsonl")))
    latency = _pending_latency_rows()
    categories = _pending_category_rows(manifest)

    assert {row["stage"] for row in latency} == {
        "stt",
        "query_embedding",
        "vector_search",
        "guardrails",
        "generation",
        "total_e2e",
    }
    assert all(row["p50_ms"] == "" and row["cases"] == 0 for row in latency)
    assert len(categories) == 6
    assert sum(row["planned_cases"] for row in categories) == 24
    assert all(row["measured_cases"] == 0 for row in categories)


def test_manifest_validation_rejects_partial_formal_benchmark() -> None:
    manifest = list(read_jsonl(Path("eval/stt_manifest.jsonl")))
    cases = _development_cases(Path("eval/eval_dataset.jsonl"))
    with pytest.raises(RuntimeError, match="Expected 24"):
        _validate_manifest(manifest[:-1], cases)


def test_report_separates_all_evidence_classes() -> None:
    report = Path("results/e2e_recommendation.md").read_text(encoding="utf-8")
    for heading in ("## Measured", "## Development-only", "## Smoke tests", "## Pending"):
        assert heading in report
    assert "There are no formal real-voice E2E quality or latency numbers" in report
