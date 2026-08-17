import pytest

from hh_goa_rag.metrics import (
    evaluate_rankings,
    evaluate_rankings_by_language,
    latency_percentiles,
    qrels_by_query,
)


def test_metrics_with_multiple_relevant_parents() -> None:
    qrels = {"q1": {"a", "b"}, "q2": {"c"}}
    rankings = {"q1": ["x", "a", "b"], "q2": ["c", "z"]}
    result = evaluate_rankings(rankings, qrels)
    assert result["recall_at_1"] == pytest.approx(0.5)
    assert result["recall_at_3"] == pytest.approx(1.0)
    assert result["mrr_at_10"] == pytest.approx(0.75)
    assert 0 < result["ndcg_at_10"] <= 1


def test_qrels_and_latency_helpers() -> None:
    rows = [
        {"query_id": "q", "passage_id": "a", "relevance": 1},
        {"query_id": "q", "passage_id": "b", "relevance": 0},
    ]
    assert qrels_by_query(rows) == {"q": {"a"}}
    stats = latency_percentiles([1, 2, 3, 4])
    assert stats["p50_ms"] == pytest.approx(2.5)
    assert stats["p100_ms"] == 4


def test_metrics_are_reported_per_language_without_query_id_collisions() -> None:
    qrels = {"hi:1": {"a"}, "ta:1": {"b"}}
    rankings = {"hi:1": ["a"], "ta:1": ["x", "b"]}
    result = evaluate_rankings_by_language(rankings, qrels)
    assert sorted(result) == ["hi", "ta"]
    assert result["hi"]["mrr_at_10"] == 1.0
    assert result["ta"]["mrr_at_10"] == 0.5
