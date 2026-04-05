"""Run the development text E2E path with Gemini answer generation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from hh_goa_rag.generation.evaluation import percentile_summary
from hh_goa_rag.generation.gemini import (
    GeminiGeneration,
    GeminiGenerationConfig,
)
from hh_goa_rag.harness import WARMUP_PASSAGE, WARMUP_QUERY, VoiceRAGHarness
from hh_goa_rag.io import read_jsonl
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel
from hh_goa_rag.retriever import ParentFaissRetriever

EXPECTED_ROUTE = {
    "answer": "ANSWER",
    "refuse_insufficient_context": "INSUFFICIENT_CONTEXT",
    "reject_off_topic": "OFF_TOPIC",
    "reject_unsafe": "UNSAFE",
}


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


def _load_cases(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    manifest = next((row for row in rows if row.get("record_type") == "manifest"), {})
    if manifest.get("sealed_test_included") is not False:
        raise RuntimeError("Gemini text E2E must use the development-only dataset")
    cases = [
        row
        for row in rows
        if row.get("record_type") == "case" and row.get("split") == "development"
    ]
    if len(cases) != 24:
        raise RuntimeError(f"Expected 24 development cases, found {len(cases)}")
    return cases


def _load_development_stack(
    device: str,
) -> tuple[EmbeddingModel, ParentFaissRetriever, dict[str, Any]]:
    index_winner = json.loads(Path("results/index_winner.json").read_text(encoding="utf-8"))
    chunk_winner = json.loads(Path("results/chunking_winner.json").read_text(encoding="utf-8"))
    embedding_winner = json.loads(
        Path("results/embedding_winner.json").read_text(encoding="utf-8")
    )
    if index_winner["metrics"].get("split") != "dev":
        raise RuntimeError("Gemini text E2E requires the development index")
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    model = EmbeddingModel(
        MODEL_SPECS[embedding_winner["winner"]],
        Path(embedding_winner["metrics"]["model_cache_path"]),
        device=device,
        max_sequence_length=512,
        dtype=dtype,
    )
    retriever = ParentFaissRetriever.load(
        index_winner["metrics"]["index_artifact"],
        chunk_winner["metrics"]["chunk_artifact"],
        top_k=10,
        oversample=20,
    )
    return model, retriever, {
        "embedding_model": embedding_winner["winner"],
        "index": index_winner["winner"],
        "chunking": chunk_winner["winner"],
    }


def _run(
    device: str,
    api_key: str,
    output_dir: Path,
    dataset: Path,
    inter_request_delay_s: float,
    case_interval_s: float,
    model_name: str,
) -> None:
    cases = _load_cases(dataset)
    started = time.perf_counter_ns()
    embedder, retriever, stack = _load_development_stack(device)
    model_index_startup_ms = (time.perf_counter_ns() - started) / 1e6
    warmup_started = time.perf_counter_ns()
    embedder.warm_up(WARMUP_QUERY, WARMUP_PASSAGE, rounds=1)
    warmup_ms = (time.perf_counter_ns() - warmup_started) / 1e6
    generator = GeminiGeneration(
        api_key,
        config=GeminiGenerationConfig(model=model_name),
    )
    harness = VoiceRAGHarness(embedder=embedder, retriever=retriever, generator=generator)
    rows: list[dict[str, Any]] = []
    try:
        for case_index, case in enumerate(cases):
            response = harness.handle_text(case["stt_reference"])
            value = response.to_dict()
            expected_route = EXPECTED_ROUTE[case["expected"]["route"]]
            retrieved = set(value["retrieved_ids"])
            citations = set(value["citations"])
            generation = value.get("metadata", {}).get("generation", {})
            rows.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "expected_route": expected_route,
                    "predicted_route": value["route"],
                    "route_correct": value["route"] == expected_route,
                    "pipeline_success": value["route"] != "SYSTEM_ERROR",
                    "generation_status": generation.get("provider_status", "SKIPPED"),
                    "generation_error_code": generation.get("error_code", ""),
                    "generation_http_status": generation.get("http_status", ""),
                    "generation_error_message": generation.get("error_message", ""),
                    "grounding_valid": value["route"] in {"ANSWER", "INSUFFICIENT_CONTEXT"},
                    "citation_valid": (
                        bool(citations) and citations.issubset(retrieved)
                        if value["route"] == "ANSWER"
                        else value["route"] != "SYSTEM_ERROR"
                    ),
                    "reason_code": value["reason_code"],
                    "answer": value["answer"],
                    "stt_ms": value["stage_latencies_ms"].get("stt", 0.0),
                    "input_validation_ms": value["stage_latencies_ms"].get(
                        "input_validation", 0.0
                    ),
                    "route_check_ms": value["stage_latencies_ms"].get("route_check", 0.0),
                    "query_embedding_ms": value["stage_latencies_ms"].get("embedding", 0.0),
                    "vector_search_ms": value["stage_latencies_ms"].get("retrieval", 0.0),
                    "evidence_guardrail_ms": value["stage_latencies_ms"].get(
                        "evidence_guardrail", 0.0
                    ),
                    "guardrails_ms": value["stage_latencies_ms"].get("guardrails", 0.0),
                    "generation_ms": value["stage_latencies_ms"].get("generation", 0.0),
                    "grounding_validation_ms": value["stage_latencies_ms"].get(
                        "grounding_validation", 0.0
                    ),
                    "total_e2e_ms": value["total_latency_ms"],
                    "under_200_ms": value["total_latency_ms"] < 200,
                    "retrieved_ids_json": json.dumps(value["retrieved_ids"]),
                    "citations_json": json.dumps(value["citations"]),
                }
            )
            if (
                inter_request_delay_s > 0
                and generation.get("provider_status", "SKIPPED") != "SKIPPED"
            ):
                time.sleep(inter_request_delay_s)
            elif case_interval_s > 0 and case_index < len(cases) - 1:
                time.sleep(case_interval_s)
    finally:
        harness.close()
        generator.close()

    stages = (
        "input_validation_ms",
        "route_check_ms",
        "query_embedding_ms",
        "vector_search_ms",
        "guardrails_ms",
        "evidence_guardrail_ms",
        "generation_ms",
        "grounding_validation_ms",
        "total_e2e_ms",
    )
    latency_rows = []
    for field in stages:
        values = [float(row[field]) for row in rows]
        summary = percentile_summary(values)
        latency_rows.append(
            {
                "stage": field.removesuffix("_ms"),
                "cases": len(values),
                **summary,
                "mean_ms": statistics.fmean(values),
                "under_200_ms_percent": (
                    statistics.fmean(value < 200 for value in values) * 100
                    if field == "total_e2e_ms"
                    else ""
                ),
            }
        )
    summary = {
        "provider": "gemini",
        "model": model_name,
        "device": device,
        "cases": len(rows),
        "startup_model_index_ms": model_index_startup_ms,
        "warmup_ms": warmup_ms,
        "total_startup_ms": model_index_startup_ms + warmup_ms,
        "route_accuracy": statistics.fmean(row["route_correct"] for row in rows),
        "pipeline_success_rate": statistics.fmean(row["pipeline_success"] for row in rows),
        "grounding_validity_rate": statistics.fmean(row["grounding_valid"] for row in rows),
        "citation_validity_rate": statistics.fmean(row["citation_valid"] for row in rows),
        "generation_calls": sum(row["generation_status"] != "SKIPPED" for row in rows),
        "under_200_ms_percent": statistics.fmean(row["under_200_ms"] for row in rows) * 100,
        "stack": stack,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "gemini_text_e2e_per_case.csv", rows)
    _write_csv(output_dir / "gemini_text_e2e_latency.csv", latency_rows)
    (output_dir / "gemini_text_e2e_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for row in latency_rows:
        print(
            f"{row['stage']}: p50={row['p50_ms']:.3f} ms, "
            f"p95={row['p95_ms']:.3f} ms, mean={row['mean_ms']:.3f} ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", type=Path, default=Path("eval/eval_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--inter-request-delay-s", type=float, default=0.0)
    parser.add_argument(
        "--case-interval-s",
        type=float,
        default=0.0,
        help="Wait this long between every development case, including non-generation cases.",
    )
    args = parser.parse_args()
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is required")
    _run(
        device,
        api_key,
        args.output_dir,
        args.dataset,
        args.inter_request_delay_s,
        args.case_interval_s,
        args.model,
    )


if __name__ == "__main__":
    main()
