import pytest

from hh_goa_rag.latency import (
    compress_contexts,
    extractive_is_grounded,
    latency_summary,
    select_extractive_answer,
    token_f1,
)


def test_latency_summary_interpolates_required_percentiles() -> None:
    summary = latency_summary([1, 2, 3, 4])
    assert summary["n"] == 4
    assert summary["p50_ms"] == 2.5
    assert summary["p70_ms"] == pytest.approx(3.1)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["p100_ms"] == 4


def test_extractive_answer_is_verbatim_and_cited() -> None:
    contexts = [
        {
            "parent_id": "p1",
            "score": 0.82,
            "text": "अन्य वाक्य। गोल्डस्मिथ टेक्सास एक्टर काउंटी में है।",
        }
    ]
    result = select_extractive_answer("गोल्डस्मिथ किस काउंटी में है", contexts)
    assert result.eligible
    assert result.evidence_ids == ("p1",)
    assert extractive_is_grounded(result.answer, result.evidence_ids, contexts)


def test_uncertain_extractive_answer_falls_back_without_text() -> None:
    contexts = [{"parent_id": "p1", "score": 0.5, "text": "असंबंधित सामग्री।"}]
    result = select_extractive_answer("पासपोर्ट की कीमत", contexts)
    assert not result.eligible
    assert result.answer == ""
    assert result.evidence_ids == ()


def test_compression_preserves_literal_evidence() -> None:
    contexts = [
        {
            "parent_id": "p1",
            "chunk_id": "c1",
            "score": 0.8,
            "text": "पहला तथ्य। पासपोर्ट की कीमत 72 पाउंड है।",
        }
    ]
    compressed = compress_contexts("पासपोर्ट की कीमत", contexts)
    assert compressed[0]["text"] == "पासपोर्ट की कीमत 72 पाउंड है।"
    assert compressed[0]["parent_id"] == "p1"


def test_token_f1_is_bounded() -> None:
    assert token_f1("one two", "one three") == 0.5
    assert token_f1("", "") == 1.0
