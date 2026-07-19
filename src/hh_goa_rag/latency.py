"""Additive latency-ablation helpers; the frozen production pipeline does not import this module."""

from __future__ import annotations

import math
import re
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

_TOKEN = re.compile(r"[\w\u0900-\u097f]+", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?।])\s+|\n+")


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    observed = [float(value) for value in values if value is not None]
    return {
        "n": len(observed),
        "p50_ms": percentile(observed, 0.50),
        "p70_ms": percentile(observed, 0.70),
        "p95_ms": percentile(observed, 0.95),
        "p100_ms": max(observed) if observed else None,
        "mean_ms": statistics.fmean(observed) if observed else None,
    }


def tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN.finditer(text)}


def token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = tokens(candidate)
    reference_tokens = tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return float(candidate_tokens == reference_tokens)
    overlap = len(candidate_tokens & reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class ExtractiveAnswer:
    answer: str
    evidence_ids: tuple[str, ...]
    confidence: float
    latency_ms: float
    eligible: bool
    reason: str


def select_extractive_answer(
    question: str,
    contexts: Sequence[dict[str, Any]],
    *,
    min_retrieval_score: float = 0.80,
    min_lexical_overlap: float = 0.12,
) -> ExtractiveAnswer:
    """Select one verbatim evidence sentence, otherwise return no extractive answer.

    This is deliberately conservative. It never synthesizes words and therefore cannot
    introduce a fact absent from the cited passage.
    """
    started = time.perf_counter_ns()
    if not contexts:
        return ExtractiveAnswer("", (), 0.0, _elapsed_ms(started), False, "no_context")
    query_tokens = tokens(question)
    best: tuple[float, str, str] | None = None
    for context in contexts[:3]:
        parent_id = str(context.get("parent_id", ""))
        retrieval_score = float(context.get("score") or 0.0)
        for sentence in _SENTENCE.split(str(context.get("text", ""))):
            sentence = sentence.strip()
            sentence_tokens = tokens(sentence)
            if not sentence_tokens or not query_tokens:
                continue
            lexical = len(query_tokens & sentence_tokens) / len(query_tokens)
            score = 0.65 * retrieval_score + 0.35 * lexical
            if best is None or score > best[0]:
                best = (score, sentence, parent_id)
    top_score = float(contexts[0].get("score") or 0.0)
    if best is None:
        return ExtractiveAnswer("", (), 0.0, _elapsed_ms(started), False, "no_sentence")
    lexical = len(query_tokens & tokens(best[1])) / max(1, len(query_tokens))
    eligible = top_score >= min_retrieval_score and lexical >= min_lexical_overlap
    reason = "high_confidence_verbatim" if eligible else "uncertain"
    return ExtractiveAnswer(
        best[1] if eligible else "",
        (best[2],) if eligible else (),
        best[0],
        _elapsed_ms(started),
        eligible,
        reason,
    )


def compress_contexts(
    question: str, contexts: Sequence[dict[str, Any]], *, limit: int = 5
) -> list[dict[str, Any]]:
    """Keep the most query-overlapping verbatim sentence from each retrieved passage."""
    query_tokens = tokens(question)
    compressed: list[dict[str, Any]] = []
    for context in contexts[:limit]:
        sentences = [part.strip() for part in _SENTENCE.split(str(context["text"]))]
        sentences = [part for part in sentences if part]
        if not sentences:
            continue
        best = max(
            sentences,
            key=lambda text: len(query_tokens & tokens(text)) / max(1, len(query_tokens)),
        )
        compressed.append({**context, "text": best})
    return compressed


def extractive_is_grounded(
    answer: str, evidence_ids: Sequence[str], contexts: Sequence[Any]
) -> bool:
    """Verify that an extractive answer is a literal substring of its cited evidence."""
    evidence = {
        str(context.get("parent_id")): str(context.get("text", ""))
        for context in contexts
        if isinstance(context, dict)
    }
    return bool(answer and evidence_ids) and any(
        answer in evidence.get(evidence_id, "") for evidence_id in evidence_ids
    )


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1e6
