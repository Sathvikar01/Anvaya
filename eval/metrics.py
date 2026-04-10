"""Shared metrics, validation, and structured logging for Voice-RAG evaluation.

The module deliberately has no STT, embedding, retrieval, or LLM provider imports.  Evaluators
consume JSONL observations produced by an experiment and score them against the fixed dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVALUATOR_VERSION = "voice-rag-eval-v1"
ANSWER_ROUTE = "answer"
INSUFFICIENT_ROUTE = "refuse_insufficient_context"
OFF_TOPIC_ROUTE = "reject_off_topic"
UNSAFE_ROUTE = "reject_unsafe"
VALID_ROUTES = {ANSWER_ROUTE, INSUFFICIENT_ROUTE, OFF_TOPIC_ROUTE, UNSAFE_ROUTE}
REQUIRED_CATEGORIES = {
    "normal_answerable",
    "paraphrased",
    "noisy_transcription",
    "insufficient_evidence",
    "off_topic",
    "unsafe",
}
FROZEN_RETRIEVAL_STACK = {
    "model": "BAAI/bge-m3",
    "chunking_strategy": "sentence",
    "chunk_size_words": 128,
    "index_engine": "faiss",
    "index_type": "hnsw",
    "m": 32,
    "ef_construction": 200,
    "ef_search": 128,
    "top_k": 10,
    "search_oversample": 20,
    "normalization_method": "float32_l2_v1",
}
GENERATION_PASS_THRESHOLD = 0.8
_SPACE_RE = re.compile(r"\s+", re.UNICODE)


class EvaluationInputError(ValueError):
    """Raised when an experiment artifact violates the evaluation contract."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                message = f"Invalid JSON at {path}:{line_number}: {error}"
                raise EvaluationInputError(message) from error
            if not isinstance(row, dict):
                raise EvaluationInputError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_dataset(
    path: str | Path,
    *,
    split: str = "development",
    allow_sealed_test: bool = False,
) -> list[dict[str, Any]]:
    """Load and validate cases; sealed rows require a deliberate command-line opt-in."""
    rows = read_jsonl(path)
    cases = [row for row in rows if row.get("record_type", "case") == "case"]
    if not cases:
        raise EvaluationInputError("Evaluation dataset has no case records")
    identifiers: set[str] = set()
    categories: set[str] = set()
    selected_categories: set[str] = set()
    selected: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in identifiers:
            raise EvaluationInputError(f"Missing or duplicate case_id: {case_id!r}")
        identifiers.add(case_id)
        category = str(case.get("category", ""))
        categories.add(category)
        route = case.get("expected", {}).get("route")
        if category not in REQUIRED_CATEGORIES:
            raise EvaluationInputError(f"Unknown category for {case_id}: {category!r}")
        if route not in VALID_ROUTES:
            raise EvaluationInputError(f"Unknown expected route for {case_id}: {route!r}")
        case_split = str(case.get("split", "development"))
        if case_split == "sealed_test" and not allow_sealed_test:
            if split == "sealed_test":
                raise EvaluationInputError(
                    "Sealed test access requires the explicit --allow-sealed-test flag"
                )
            continue
        if case_split == split:
            selected.append(case)
            selected_categories.add(category)
    if not selected:
        raise EvaluationInputError(f"No cases found for split {split!r}")
    if split == "development" and not REQUIRED_CATEGORIES.issubset(selected_categories):
        missing = sorted(REQUIRED_CATEGORIES - selected_categories)
        raise EvaluationInputError(f"Development dataset is missing categories: {missing}")
    return selected


def index_by_case(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in indexed:
            raise EvaluationInputError(f"Missing or duplicate prediction case_id: {case_id!r}")
        indexed[case_id] = row
    return indexed


def pair_cases_and_predictions(
    cases: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Require exactly one prediction for every selected case and no unknown case IDs."""
    prediction_map = index_by_case(predictions)
    case_ids = {str(case["case_id"]) for case in cases}
    missing = sorted(case_ids - prediction_map.keys())
    unknown = sorted(prediction_map.keys() - case_ids)
    if missing or unknown:
        raise EvaluationInputError(
            f"Prediction coverage mismatch: missing={missing[:10]}, unknown={unknown[:10]}"
        )
    return [(case, prediction_map[str(case["case_id"])]) for case in cases]


def assert_frozen_retrieval_config(path: str | Path) -> dict[str, Any]:
    """Reject evaluation when the experiment changes any frozen retrieval-stack field."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    observed = {
        "model": config.get("model"),
        "chunking_strategy": config.get("chunking", {}).get("strategy"),
        "chunk_size_words": config.get("chunking", {}).get("max_words"),
        "index_engine": config.get("index", {}).get("engine"),
        "index_type": config.get("index", {}).get("index_type"),
        "m": config.get("index", {}).get("m"),
        "ef_construction": config.get("index", {}).get("ef_construction"),
        "ef_search": config.get("index", {}).get("ef_search"),
        "top_k": config.get("top_k"),
        "search_oversample": config.get("search_oversample"),
        "normalization_method": config.get("normalization_method"),
    }
    if observed != FROZEN_RETRIEVAL_STACK:
        differences = {
            key: {"expected": expected, "observed": observed.get(key)}
            for key, expected in FROZEN_RETRIEVAL_STACK.items()
            if observed.get(key) != expected
        }
        raise EvaluationInputError(f"Retrieval stack is frozen; differences: {differences}")
    return observed


def normalize_text(text: str) -> str:
    """Unicode-aware, case-insensitive normalization used only for WER tokenization."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    characters = [
        character
        if unicodedata.category(character)[0] in {"L", "N"} or character.isspace()
        else " "
        for character in normalized
    ]
    return _SPACE_RE.sub(" ", "".join(characters)).strip()


def word_error_counts(reference: str, hypothesis: str) -> dict[str, int | float]:
    """Return minimum word-level Levenshtein edits and reference-normalized WER."""
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous[hypothesis_index - 1] + (
                reference_word != hypothesis_word
            )
            deletion = previous[hypothesis_index] + 1
            insertion = current[hypothesis_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    errors = previous[-1]
    denominator = len(reference_words)
    wer = (0.0 if not hypothesis_words else 1.0) if denominator == 0 else errors / denominator
    return {"errors": errors, "reference_words": denominator, "wer": float(wer)}


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile, equivalent to NumPy's default method."""
    if not values:
        raise EvaluationInputError("Cannot calculate a percentile for an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_metrics(latencies_ms: Sequence[float]) -> dict[str, float | int]:
    if not latencies_ms:
        return {"count": 0}
    values = [float(value) for value in latencies_ms]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise EvaluationInputError("Latencies must be finite, non-negative milliseconds")
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p70_ms": percentile(values, 0.70),
        "p95_ms": percentile(values, 0.95),
        "p100_ms": max(values),
        "under_200ms_rate": sum(value < 200 for value in values) / len(values),
        "target_ms": 200,
    }


def prediction_failed(prediction: Mapping[str, Any]) -> bool:
    return prediction.get("status", "ok") != "ok"


def required_latency(prediction: Mapping[str, Any]) -> float:
    if "latency_ms" not in prediction:
        case_id = prediction.get("case_id")
        raise EvaluationInputError(
            f"Prediction {case_id} is missing latency_ms; include timeout duration"
        )
    return float(prediction["latency_ms"])


def evaluate_stt_records(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    total_errors = 0
    total_reference_words = 0
    failures = 0
    latencies: list[float] = []
    for case, prediction in pairs:
        latency = required_latency(prediction)
        latencies.append(latency)
        failed = prediction_failed(prediction) or not str(prediction.get("transcript", "")).strip()
        row: dict[str, Any] = {
            "case_id": case["case_id"],
            "category": case["category"],
            "latency_ms": latency,
            "failed": failed,
        }
        if failed:
            failures += 1
            row.update({"errors": None, "reference_words": None, "wer": None})
        else:
            counts = word_error_counts(case["stt_reference"], prediction["transcript"])
            total_errors += int(counts["errors"])
            total_reference_words += int(counts["reference_words"])
            row.update(counts)
        details.append(row)
    successful_wers = [float(row["wer"]) for row in details if row["wer"] is not None]
    summary = {
        "wer_micro": total_errors / total_reference_words if total_reference_words else None,
        "wer_macro": statistics.fmean(successful_wers) if successful_wers else None,
        "word_errors": total_errors,
        "reference_words": total_reference_words,
        "failure_rate": failures / len(pairs),
        "successful_cases": len(pairs) - failures,
        "total_cases": len(pairs),
        "latency": latency_metrics(latencies),
    }
    return summary, details


def _ranked_parent_ids(prediction: Mapping[str, Any]) -> list[str]:
    raw = prediction.get("retrieved_parent_ids", prediction.get("retrieved", []))
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        parent_id = str(item.get("parent_id")) if isinstance(item, dict) else str(item)
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            result.append(parent_id)
    return result


def evaluate_retrieval_records(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ks = (1, 3, 5, 10)
    details: list[dict[str, Any]] = []
    failures = 0
    latencies: list[float] = []
    eligible = 0
    for case, prediction in pairs:
        relevant = {str(item) for item in case["expected"].get("relevant_parent_ids", [])}
        if not relevant:
            continue
        eligible += 1
        latency = required_latency(prediction)
        latencies.append(latency)
        failed = prediction_failed(prediction)
        ranked = [] if failed else _ranked_parent_ids(prediction)[:10]
        failures += int(failed)
        recalls = {
            f"recall_at_{k}": len(relevant.intersection(ranked[:k])) / len(relevant) for k in ks
        }
        first_rank = next(
            (rank for rank, parent_id in enumerate(ranked, start=1) if parent_id in relevant), None
        )
        reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, parent_id in enumerate(ranked, start=1)
            if parent_id in relevant
        )
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1)
        )
        details.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "failed": failed,
                "latency_ms": latency,
                **recalls,
                "reciprocal_rank_at_10": reciprocal_rank,
                "ndcg_at_10": dcg / ideal_dcg,
                "first_relevant_rank": first_rank,
            }
        )
    if not eligible:
        raise EvaluationInputError("No answerable cases with retrieval qrels were found")
    summary: dict[str, Any] = {
        metric: statistics.fmean(float(row[metric]) for row in details)
        for metric in [*(f"recall_at_{k}" for k in ks), "ndcg_at_10"]
    }
    summary["mrr_at_10"] = statistics.fmean(
        float(row["reciprocal_rank_at_10"]) for row in details
    )
    summary.update(
        {
            "failure_rate": failures / eligible,
            "successful_cases": eligible - failures,
            "evaluated_queries": eligible,
            "latency": latency_metrics(latencies),
        }
    )
    return summary, details


def _score(value: Any, field: str, case_id: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise EvaluationInputError(f"{case_id}: judgment.{field} must be numeric") from error
    if not 0 <= score <= 1:
        raise EvaluationInputError(f"{case_id}: judgment.{field} must be between 0 and 1")
    return score


def generation_row(
    case: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    failed = prediction_failed(prediction) or not str(prediction.get("answer", "")).strip()
    row: dict[str, Any] = {
        "case_id": case_id,
        "category": case["category"],
        "failed": failed,
        "latency_ms": required_latency(prediction),
    }
    if failed:
        row.update(
            {
                "correctness": 0.0,
                "relevance": 0.0,
                "faithfulness": 0.0,
                "claim_count": 0,
                "unsupported_claims": 0,
                "unsupported_claim_rate": 0.0,
            }
        )
        return row
    judgment = prediction.get("judgment")
    if not isinstance(judgment, dict):
        raise EvaluationInputError(f"{case_id}: successful answers require a judgment object")
    claims = judgment.get("claims", [])
    if not isinstance(claims, list):
        raise EvaluationInputError(f"{case_id}: judgment.claims must be a list")
    unsupported = sum(
        1 for claim in claims if not isinstance(claim, dict) or claim.get("supported") is not True
    )
    row.update(
        {
            "correctness": _score(judgment.get("correctness"), "correctness", case_id),
            "relevance": _score(judgment.get("relevance"), "relevance", case_id),
            "faithfulness": _score(judgment.get("faithfulness"), "faithfulness", case_id),
            "claim_count": len(claims),
            "unsupported_claims": unsupported,
            "unsupported_claim_rate": unsupported / len(claims) if claims else 0.0,
        }
    )
    return row


def evaluate_generation_records(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answerable = [(case, pred) for case, pred in pairs if case["expected"]["route"] == ANSWER_ROUTE]
    if not answerable:
        raise EvaluationInputError("No answerable generation cases were found")
    details = [generation_row(case, prediction) for case, prediction in answerable]
    failures = sum(int(row["failed"]) for row in details)
    claim_count = sum(int(row["claim_count"]) for row in details)
    unsupported = sum(int(row["unsupported_claims"]) for row in details)
    summary = {
        metric: statistics.fmean(float(row[metric]) for row in details)
        for metric in ("correctness", "relevance", "faithfulness")
    }
    summary.update(
        {
            "unsupported_claim_rate": unsupported / claim_count if claim_count else 0.0,
            "unsupported_claims": unsupported,
            "claim_count": claim_count,
            "failure_rate": failures / len(details),
            "successful_cases": len(details) - failures,
            "evaluated_cases": len(details),
            "latency": latency_metrics([float(row["latency_ms"]) for row in details]),
        }
    )
    return summary, details


def evaluate_guardrail_records(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures = 0
    for case, prediction in pairs:
        latency = required_latency(prediction)
        latencies.append(latency)
        failed = prediction_failed(prediction)
        decision = None if failed else prediction.get("decision")
        if decision is not None and decision not in VALID_ROUTES:
            raise EvaluationInputError(f"{case['case_id']}: invalid decision {decision!r}")
        expected = case["expected"]["route"]
        failures += int(failed)
        details.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "expected_route": expected,
                "decision": decision,
                "correct": decision == expected,
                "false_refusal": expected == ANSWER_ROUTE and decision != ANSWER_ROUTE,
                "failed": failed,
                "latency_ms": latency,
            }
        )

    def accuracy(route: str) -> float | None:
        selected = [row for row in details if row["expected_route"] == route]
        return (
            sum(int(row["decision"] == route) for row in selected) / len(selected)
            if selected
            else None
        )

    answerable = [row for row in details if row["expected_route"] == ANSWER_ROUTE]
    summary = {
        "off_topic_rejection_accuracy": accuracy(OFF_TOPIC_ROUTE),
        "insufficient_context_refusal_accuracy": accuracy(INSUFFICIENT_ROUTE),
        "unsafe_rejection_accuracy": accuracy(UNSAFE_ROUTE),
        "false_refusal_rate": (
            sum(int(row["false_refusal"]) for row in answerable) / len(answerable)
            if answerable
            else None
        ),
        "exact_route_accuracy": sum(int(row["correct"]) for row in details) / len(details),
        "failure_rate": failures / len(details),
        "total_cases": len(details),
        "latency": latency_metrics(latencies),
    }
    return summary, details


def evaluate_e2e_records(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case, prediction in pairs:
        latency = required_latency(prediction)
        latencies.append(latency)
        failed = prediction_failed(prediction)
        expected_route = case["expected"]["route"]
        decision = prediction.get("decision")
        route_correct = decision == expected_route
        retrieval_hit = None
        generation_pass = None
        if expected_route == ANSWER_ROUTE and not failed:
            relevant = {str(item) for item in case["expected"].get("relevant_parent_ids", [])}
            ranked = _ranked_parent_ids(prediction)[:10]
            retrieval_hit = bool(relevant.intersection(ranked))
            citations = prediction.get("citations", prediction.get("evidence_ids", ()))
            if not isinstance(citations, (list, tuple, set)):
                citations = ()
            citation_valid = bool(citations) and set(map(str, citations)).issubset(set(ranked))
            judged = generation_row(case, prediction)
            generation_pass = (
                citation_valid
                and all(
                    float(judged[field]) >= GENERATION_PASS_THRESHOLD
                    for field in ("correctness", "relevance", "faithfulness")
                )
                and int(judged["claim_count"]) > 0
                and int(judged["unsupported_claims"]) == 0
            )
            success = not failed and route_correct and retrieval_hit and generation_pass
        else:
            success = not failed and route_correct
        stage_latencies = prediction.get("stage_latencies_ms", {})
        details.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "expected_route": expected_route,
                "decision": decision,
                "route_correct": route_correct,
                "retrieval_hit_at_10": retrieval_hit,
                "citation_valid": citation_valid,
                "generation_pass": generation_pass,
                "success": success,
                "failed": failed,
                "latency_ms": latency,
                "stage_latencies_ms": stage_latencies,
            }
        )
    success_count = sum(int(row["success"]) for row in details)
    summary = {
        "success_rate": success_count / len(details),
        "failure_rate": 1 - success_count / len(details),
        "technical_failure_rate": sum(int(row["failed"]) for row in details) / len(details),
        "successful_cases": success_count,
        "total_cases": len(details),
        "latency": latency_metrics(latencies),
    }
    return summary, details


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_evaluation_run(
    *,
    output_dir: str | Path,
    run_id: str,
    stage: str,
    dataset_path: str | Path,
    predictions_path: str | Path,
    split: str,
    metrics: Mapping[str, Any],
    details: Sequence[Mapping[str, Any]],
    system_id: str,
    retrieval_stack: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        message = "run-id must contain only letters, digits, dot, underscore, hyphen"
        raise EvaluationInputError(message)
    run_dir = Path(output_dir) / run_id
    summary_path = run_dir / f"{stage}_summary.json"
    cases_path = run_dir / f"{stage}_cases.csv"
    envelope = {
        "schema_version": "1.0",
        "evaluator_version": EVALUATOR_VERSION,
        "stage": stage,
        "run_id": run_id,
        "system_id": system_id,
        "split": split,
        "created_at_utc": utc_now(),
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
        },
        "retrieval_stack": dict(retrieval_stack) if retrieval_stack else None,
        "metrics": dict(metrics),
    }
    _atomic_json(summary_path, envelope)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in details for key in row})
    temporary = cases_path.with_suffix(cases_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in details:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    os.replace(temporary, cases_path)
    return summary_path, cases_path


def print_summary(summary_path: Path, cases_path: Path, metrics: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "cases_path": str(cases_path),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
