"""Measure the text-to-retrieval Voice-RAG path without invoking answer generation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from hh_goa_rag.generation.evaluation import percentile_summary
from hh_goa_rag.guardrails.input import route_input, validate_transcript
from hh_goa_rag.guardrails.retrieval import evidence_sufficiency
from hh_goa_rag.io import read_jsonl
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel
from hh_goa_rag.retriever import ParentFaissRetriever

ATOMIC_STAGES = (
    "stt",
    "input_validation",
    "route_check",
    "query_embedding",
    "vector_search",
    "evidence_guardrail",
    "guardrails",
    "generation",
    "grounding_validation",
    "pre_generation_total",
)


def _ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1e6


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float]) -> dict[str, Any]:
    result = percentile_summary(values)
    result["mean_ms"] = statistics.fmean(values) if values else ""
    return result


def _run_case(
    case: dict[str, Any], embedder: EmbeddingModel, retriever: ParentFaissRetriever
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    stages = {stage: 0.0 for stage in ATOMIC_STAGES}
    stages["stt"] = None
    stages["generation"] = None
    stages["grounding_validation"] = None
    transcript = str(case["stt_reference"])

    checkpoint = time.perf_counter_ns()
    validation = validate_transcript(transcript)
    stages["input_validation"] = _ms(checkpoint)
    if not validation.allow:
        stages["guardrails"] = stages["input_validation"]
        stages["pre_generation_total"] = _ms(started)
        return _row(case, validation.route.value, validation.reason_code.value, stages, [])

    checkpoint = time.perf_counter_ns()
    route = route_input(validation.normalized_transcript)
    stages["route_check"] = _ms(checkpoint)
    if not route.allow:
        stages["guardrails"] = stages["input_validation"] + stages["route_check"]
        stages["pre_generation_total"] = _ms(started)
        return _row(case, route.route.value, route.reason_code.value, stages, [])

    checkpoint = time.perf_counter_ns()
    vectors, _ = embedder.encode_queries([validation.normalized_transcript])
    stages["query_embedding"] = _ms(checkpoint)

    checkpoint = time.perf_counter_ns()
    contexts = retriever.retrieve(vectors[0])
    stages["vector_search"] = _ms(checkpoint)

    checkpoint = time.perf_counter_ns()
    sufficiency = evidence_sufficiency(contexts)
    stages["evidence_guardrail"] = _ms(checkpoint)
    stages["guardrails"] = (
        stages["input_validation"]
        + stages["route_check"]
        + stages["evidence_guardrail"]
    )
    stages["pre_generation_total"] = _ms(started)
    ids = [str(item.parent_id) for item in contexts]
    route_value = "ANSWERABLE_PRE_GENERATION" if sufficiency.sufficient else "INSUFFICIENT_CONTEXT"
    reason = "GENERATION_SKIPPED" if sufficiency.sufficient else sufficiency.reason_code.value
    return _row(case, route_value, reason, stages, ids)


def _row(
    case: dict[str, Any],
    route: str,
    reason: str,
    stages: dict[str, Any],
    retrieved_ids: list[str],
) -> dict[str, Any]:
    predicted_route = {
        "ANSWERABLE_PRE_GENERATION": "answer",
        "INSUFFICIENT_CONTEXT": "refuse_insufficient_context",
        "OFF_TOPIC": "reject_off_topic",
        "UNSAFE": "reject_unsafe",
    }[route]
    expected_route = str(case["expected"]["route"])
    relevant = {str(item) for item in case["expected"].get("relevant_parent_ids", [])}
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "route": route,
        "predicted_route": predicted_route,
        "expected_route": expected_route,
        "route_correct": predicted_route == expected_route,
        "retrieval_hit_at_10": bool(relevant.intersection(retrieved_ids)) if relevant else None,
        "reason_code": reason,
        "retrieved_count": len(retrieved_ids),
        "retrieved_ids_json": json.dumps(retrieved_ids),
        **{f"{stage}_ms": value for stage, value in stages.items()},
    }


def _latency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stage in ATOMIC_STAGES:
        values = [float(row[f"{stage}_ms"]) for row in rows if row[f"{stage}_ms"] is not None]
        summary = _summary(values)
        output.append(
            {
                "scope": "24 development text cases",
                "stage": stage,
                "measured_cases": len(values),
                "p50_ms": summary.get("p50_ms", ""),
                "p70_ms": summary.get("p70_ms", ""),
                "p95_ms": summary.get("p95_ms", ""),
                "p100_ms": summary.get("p100_ms", ""),
                "mean_ms": summary.get("mean_ms", ""),
                "status": "SKIPPED_NO_GENERATION"
                if stage in {"stt", "generation", "grounding_validation"}
                else "MEASURED",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/eval_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--index-winner",
        type=Path,
        default=Path("results/index_winner.json"),
        help="Development index selection; the sealed test index is never used by default.",
    )
    parser.add_argument(
        "--chunking-winner",
        type=Path,
        default=Path("results/chunking_winner.json"),
    )
    parser.add_argument(
        "--embedding-winner",
        type=Path,
        default=Path("results/embedding_winner.json"),
    )
    args = parser.parse_args()

    cases = [
        row
        for row in read_jsonl(args.dataset)
        if row.get("record_type") == "case" and row.get("split") == "development"
    ]
    if len(cases) != 24:
        raise RuntimeError(f"Expected 24 development cases, found {len(cases)}")

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # The cases above are development-only.  Loading final_retriever_config.json
    # here would pair development qrels with the sealed validation index and
    # make every answerable case look like a retrieval failure.
    index_winner = json.loads(args.index_winner.read_text(encoding="utf-8"))
    chunking_winner = json.loads(args.chunking_winner.read_text(encoding="utf-8"))
    embedding_winner = json.loads(args.embedding_winner.read_text(encoding="utf-8"))
    index_metrics = index_winner["metrics"]
    chunk_metrics = chunking_winner["metrics"]
    embedding_metrics = embedding_winner["metrics"]
    if index_metrics.get("split") != "dev" or chunk_metrics.get("split") != "dev":
        raise RuntimeError("Generation-free development evaluation requires development artifacts")
    config = {
        "model": embedding_winner["winner"],
        "model_cache_path": embedding_metrics["model_cache_path"],
        "index_artifact": index_metrics["index_artifact"],
        "chunk_artifact": chunk_metrics["chunk_artifact"],
        "top_k": 10,
        "search_oversample": 20,
    }
    startup_started = time.perf_counter_ns()
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    embedder = EmbeddingModel(
        MODEL_SPECS[config["model"]],
        Path(config["model_cache_path"]),
        device=device,
        max_sequence_length=512,
        dtype=dtype,
    )
    retriever = ParentFaissRetriever.load(
        config["index_artifact"],
        config["chunk_artifact"],
        top_k=int(config["top_k"]),
        oversample=int(config["search_oversample"]),
    )
    model_index_startup_ms = _ms(startup_started)
    warmup_started = time.perf_counter_ns()
    embedder.warm_up(
        "यूनाइटेड किंगडम में कौन से चार देश शामिल हैं",
        "यूनाइटेड किंगडम में इंग्लैंड, स्कॉटलैंड, वेल्स और उत्तरी आयरलैंड शामिल हैं",
        rounds=1,
    )
    warmup_ms = _ms(warmup_started)
    startup_ms = model_index_startup_ms + warmup_ms
    try:
        rows = [_run_case(case, embedder, retriever) for case in cases]
    finally:
        embedder.close()

    latency = _latency_rows(rows)
    _write_csv(args.output_dir / "e2e_no_generation_per_case.csv", rows)
    _write_csv(args.output_dir / "e2e_no_generation_latency.csv", latency)

    print("Evaluation: text path through evidence gate; answer generation disabled")
    print(
        f"Cases: {len(rows)} | device: {device} | model/index startup: "
        f"{model_index_startup_ms:.1f} ms | warm-up: {warmup_ms:.1f} ms | "
        f"total startup: {startup_ms:.1f} ms"
    )
    print("stage | measured | p50_ms | p70_ms | p95_ms | p100_ms | mean_ms | status")
    for row in latency:
        print(
            f"{row['stage']} | {row['measured_cases']} | {row['p50_ms']} | {row['p70_ms']} | "
            f"{row['p95_ms']} | {row['p100_ms']} | {row['mean_ms']} | {row['status']}"
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["route"]] = counts.get(row["route"], 0) + 1
    print(f"Routes: {json.dumps(counts, sort_keys=True)}")
    route_accuracy = sum(bool(row["route_correct"]) for row in rows) / len(rows)
    answerable = [row for row in rows if row["expected_route"] == "answer"]
    answerable_gate_rate = sum(
        row["predicted_route"] == "answer" for row in answerable
    ) / len(answerable)
    retrieval_hits = [
        row["retrieval_hit_at_10"]
        for row in answerable
        if row["retrieval_hit_at_10"] is not None
    ]
    print(f"Route accuracy: {route_accuracy:.3f}")
    print(f"Answerable pre-generation pass rate: {answerable_gate_rate:.3f}")
    if retrieval_hits:
        print(f"Answerable retrieval hit@10: {sum(retrieval_hits) / len(retrieval_hits):.3f}")
    print(f"Wrote: {args.output_dir / 'e2e_no_generation_per_case.csv'}")
    print(f"Wrote: {args.output_dir / 'e2e_no_generation_latency.csv'}")


if __name__ == "__main__":
    main()
