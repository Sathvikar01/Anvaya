"""Retrieval metrics computed at the parent-passage level."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def qrels_by_query(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if float(row.get("relevance", 0)) > 0:
            result[str(row["query_id"])].add(str(row["passage_id"]))
    return dict(result)


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Macro-average Recall@K, MRR@10, and binary nDCG@10."""
    if not qrels:
        raise ValueError("qrels cannot be empty")
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query_id, relevant in qrels.items():
        ranked = list(rankings.get(query_id, ()))
        for k in ks:
            recalls[k].append(len(relevant.intersection(ranked[:k])) / len(relevant))
        first_relevant = next(
            (rank for rank, parent_id in enumerate(ranked[:10], start=1) if parent_id in relevant),
            None,
        )
        reciprocal_ranks.append(0.0 if first_relevant is None else 1.0 / first_relevant)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, parent_id in enumerate(ranked[:10], start=1)
            if parent_id in relevant
        )
        ideal_hits = min(len(relevant), 10)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / ideal_dcg)
    result = {f"recall_at_{k}": float(np.mean(values)) for k, values in recalls.items()}
    result["mrr_at_10"] = float(np.mean(reciprocal_ranks))
    result["ndcg_at_10"] = float(np.mean(ndcgs))
    result["evaluated_queries"] = float(len(qrels))
    return result


def evaluate_rankings_by_language(
    rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, set[str]]
) -> dict[str, dict[str, float]]:
    """Report the same retrieval metrics per language-prefixed query ID."""
    grouped: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for query_id, relevant in qrels.items():
        language, _, _source_id = str(query_id).partition(":")
        grouped[language or "unknown"][query_id] = relevant
    return {
        language: evaluate_rankings(rankings, language_qrels)
        for language, language_qrels in sorted(grouped.items())
    }


def latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("latencies_ms cannot be empty")
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p70_ms": float(np.percentile(values, 70)),
        "p95_ms": float(np.percentile(values, 95)),
        "p100_ms": float(np.max(values)),
    }
