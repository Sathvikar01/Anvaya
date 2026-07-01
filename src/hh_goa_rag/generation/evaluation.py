"""Reproducible context caching, diagnostics, and reports for generation experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hh_goa_rag.io import read_jsonl, write_json, write_jsonl
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel
from hh_goa_rag.retriever import ParentFaissRetriever

from .prompts import PROMPT_VERSION
from .sarvam import (
    CALLABLE_SARVAM_MODELS,
    GenerationContext,
    SarvamGeneration,
    SarvamGenerationConfig,
)

FROZEN_STACK = {
    "model": "BAAI/bge-m3",
    "chunking_strategy": "sentence",
    "chunk_size_words": 128,
    "index_engine": "faiss",
    "index_type": "hnsw",
    "m": 32,
    "ef_construction": 200,
    "ef_search": 128,
}
MODEL_ABLATION_TOP_K = 10
MODEL_ABLATION_PROMPT = "structured_evidence_ids"
MAX_OUTPUT_TOKENS = 192
HUMAN_RUBRIC_VERSION = "generation-human-v1"


def load_answerable_cases(dataset_path: Path) -> list[dict[str, Any]]:
    cases = [
        row
        for row in read_jsonl(dataset_path)
        if row.get("record_type") == "case"
        and row.get("split") == "development"
        and row.get("expected", {}).get("route") == "answer"
    ]
    if not cases:
        raise RuntimeError("No development answerable cases were found")
    return cases


def prepare_context_cache(
    dataset_path: Path,
    cache_path: Path,
    *,
    device: str = "auto",
) -> list[dict[str, Any]]:
    """Retrieve gold queries exactly once, then validate and reuse the snapshot."""
    meta_path = cache_path.with_suffix(".meta.json")
    fingerprint = _retrieval_fingerprint(dataset_path)
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = list(read_jsonl(cache_path))
        if meta.get("fingerprint") == fingerprint and _valid_cache(rows):
            return rows
        raise RuntimeError(
            "Existing generation context cache does not match the frozen inputs; "
            "remove only cache/generation to rebuild intentionally"
        )

    cases = load_answerable_cases(dataset_path)
    index_winner = json.loads(Path("results/index_winner.json").read_text(encoding="utf-8"))
    chunk_winner = json.loads(Path("results/chunking_winner.json").read_text(encoding="utf-8"))
    final_config = json.loads(
        Path("results/final_retriever_config.json").read_text(encoding="utf-8")
    )
    _assert_frozen(index_winner)
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = ParentFaissRetriever.load(
        index_winner["metrics"]["index_artifact"],
        chunk_winner["metrics"]["chunk_artifact"],
        top_k=MODEL_ABLATION_TOP_K,
        oversample=20,
    )
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    model_name = str(final_config["model"])
    model = EmbeddingModel(
        MODEL_SPECS[model_name],
        Path(final_config["model_cache_path"]),
        device=device,
        max_sequence_length=512,
        dtype=dtype,
    )
    try:
        queries = [str(case["stt_reference"]) for case in cases]
        vectors, _ = model.encode_queries(queries)
        rows: list[dict[str, Any]] = []
        for case, query, vector in zip(cases, queries, vectors, strict=True):
            contexts = retriever.retrieve(vector)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "question": query,
                    "expected": case["expected"],
                    "contexts": [
                        {
                            "rank": rank,
                            "parent_id": item.parent_id,
                            "chunk_id": item.chunk_id,
                            "score": item.score,
                            "text": item.text,
                        }
                        for rank, item in enumerate(contexts, start=1)
                    ],
                }
            )
    finally:
        model.close()
    if not _valid_cache(rows):
        raise RuntimeError("Frozen retriever did not produce ten contexts per generation case")
    write_jsonl(cache_path, rows)
    write_json(
        meta_path,
        {
            "fingerprint": fingerprint,
            "frozen_stack": FROZEN_STACK,
            "query_field": "stt_reference",
            "top_k": MODEL_ABLATION_TOP_K,
            "case_count": len(rows),
        },
    )
    return rows


def run_model_ablation(
    context_rows: Sequence[dict[str, Any]],
    raw_path: Path,
    *,
    env_path: Path,
    models: Sequence[str] = CALLABLE_SARVAM_MODELS,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for model_name in models:
        service = SarvamGeneration.from_env(
            env_path,
            config=SarvamGenerationConfig(
                model=model_name,  # type: ignore[arg-type]
                max_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        for row in context_rows:
            contexts = [GenerationContext(**context) for context in row["contexts"]]
            result = service.generate(
                row["question"], contexts, prompt_variant=MODEL_ABLATION_PROMPT
            )
            observation = {
                **result.to_dict(),
                "case_id": row["case_id"],
                "category": row["category"],
                "question": row["question"],
                "top_k": MODEL_ABLATION_TOP_K,
                "prompt_variant": MODEL_ABLATION_PROMPT,
                "prompt_version": PROMPT_VERSION,
                "temperature": 0,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "retrieved_parent_ids": [item["parent_id"] for item in row["contexts"]],
            }
            observation["diagnostics"].update(_content_diagnostics(observation, row))
            observations.append(observation)
            write_jsonl(raw_path, observations)
    return observations


def run_configuration_ablation(
    context_rows: Sequence[dict[str, Any]],
    raw_path: Path,
    *,
    env_path: Path,
    model_name: str,
    configurations: Sequence[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Run a fixed model over explicit (Top-K, prompt) one-variable configurations."""
    service = SarvamGeneration.from_env(
        env_path,
        config=SarvamGenerationConfig(
            model=model_name,  # type: ignore[arg-type]
            max_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    observations: list[dict[str, Any]] = []
    for top_k, prompt_variant in configurations:
        if top_k not in {1, 3, 5, 10}:
            raise ValueError(f"Unsupported Top-K ablation value: {top_k}")
        for row in context_rows:
            selected = row["contexts"][:top_k]
            contexts = [GenerationContext(**context) for context in selected]
            result = service.generate(
                row["question"],
                contexts,
                prompt_variant=prompt_variant,  # type: ignore[arg-type]
            )
            observation = {
                **result.to_dict(),
                "case_id": row["case_id"],
                "category": row["category"],
                "question": row["question"],
                "top_k": top_k,
                "prompt_variant": prompt_variant,
                "prompt_version": PROMPT_VERSION,
                "temperature": 0,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "retrieved_parent_ids": [item["parent_id"] for item in selected],
            }
            diagnostic_row = {**row, "contexts": selected}
            observation["diagnostics"].update(_content_diagnostics(observation, diagnostic_row))
            observations.append(observation)
            write_jsonl(raw_path, observations)
    return observations


def aggregate_model_rows(observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in CALLABLE_SARVAM_MODELS:
        selected = [row for row in observations if row["model"] == model_name]
        if not selected:
            continue
        successful = [row for row in selected if row["status"] == "ok"]
        latencies = [float(row["latency_ms"]) for row in selected]
        ttfts = [
            float(row["time_to_first_token_ms"])
            for row in selected
            if row.get("time_to_first_token_ms") is not None
        ]
        tokens = [
            int(row["output_tokens"]) for row in selected if row.get("output_tokens") is not None
        ]
        diagnostics = [row.get("diagnostics", {}) for row in selected]
        latency = percentile_summary(latencies)
        ttft = percentile_summary(ttfts)
        output_token_distribution = percentile_summary([float(value) for value in tokens])
        prompt_tokens = [
            int(row["prompt_tokens"])
            for row in selected
            if row.get("prompt_tokens") is not None
        ]
        rows.append(
            {
                "provider": "sarvam",
                "model": model_name,
                "prompt_version": PROMPT_VERSION,
                "prompt_variant": MODEL_ABLATION_PROMPT,
                "top_k": MODEL_ABLATION_TOP_K,
                "temperature": 0,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "streaming": True,
                "cases": len(selected),
                "successes": len(successful),
                "failures": len(selected) - len(successful),
                "failure_rate": (len(selected) - len(successful)) / len(selected),
                "timeouts": sum(row.get("error_code") == "timeout" for row in selected),
                "recovered_retry_cases": sum(int(row.get("attempts", 1)) > 1 for row in selected),
                "latency_p50_ms": latency.get("p50_ms"),
                "latency_p70_ms": latency.get("p70_ms"),
                "latency_p95_ms": latency.get("p95_ms"),
                "latency_p100_ms": latency.get("p100_ms"),
                "ttft_p50_ms": ttft.get("p50_ms"),
                "ttft_p70_ms": ttft.get("p70_ms"),
                "ttft_p95_ms": ttft.get("p95_ms"),
                "ttft_p100_ms": ttft.get("p100_ms"),
                "mean_output_tokens": statistics.fmean(tokens) if tokens else None,
                "mean_prompt_tokens": statistics.fmean(prompt_tokens) if prompt_tokens else None,
                "output_tokens_p50": output_token_distribution.get("p50_ms"),
                "output_tokens_p95": output_token_distribution.get("p95_ms"),
                "output_tokens_max": output_token_distribution.get("p100_ms"),
                "schema_valid_rate": _mean_bool(diagnostics, "schema_valid"),
                "missing_citation_rate": _mean_bool(diagnostics, "missing_citation"),
                "unknown_citation_rate": _mean_nonempty(diagnostics, "unknown_evidence_ids"),
                "novel_number_diagnostic_rate": _mean_nonempty(diagnostics, "novel_numbers"),
                "insufficient_context_rate": (
                    sum(row.get("answer_status") == "INSUFFICIENT_CONTEXT" for row in successful)
                    / len(successful)
                    if successful
                    else None
                ),
                "retrieved_qrel_at_10_rate": _mean_bool(diagnostics, "qrel_retrieved"),
                "human_correctness": "PENDING_BLINDED_REVIEW",
                "human_relevance": "PENDING_BLINDED_REVIEW",
                "human_faithfulness": "PENDING_BLINDED_REVIEW",
            }
        )
    return rows


def aggregate_configuration_rows(
    observations: Sequence[dict[str, Any]],
    configurations: Sequence[tuple[int, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for top_k, prompt_variant in configurations:
        selected = [
            row
            for row in observations
            if int(row["top_k"]) == top_k and row["prompt_variant"] == prompt_variant
        ]
        if not selected:
            continue
        successful = [row for row in selected if row["status"] == "ok"]
        latency = percentile_summary([float(row["latency_ms"]) for row in selected])
        ttft = percentile_summary(
            [
                float(row["time_to_first_token_ms"])
                for row in selected
                if row.get("time_to_first_token_ms") is not None
            ]
        )
        tokens = [
            int(row["output_tokens"]) for row in selected if row.get("output_tokens") is not None
        ]
        prompt_tokens = [
            int(row["prompt_tokens"])
            for row in selected
            if row.get("prompt_tokens") is not None
        ]
        diagnostics = [row.get("diagnostics", {}) for row in selected]
        citation_validity = [
            bool(diagnostic.get("schema_valid"))
            and not diagnostic.get("missing_citation")
            and not diagnostic.get("unknown_evidence_ids")
            for diagnostic in diagnostics
        ]
        context_available = [
            row for row in successful if bool(row.get("diagnostics", {}).get("qrel_retrieved"))
        ]
        context_missing = [
            row for row in successful if not bool(row.get("diagnostics", {}).get("qrel_retrieved"))
        ]
        rows.append(
            {
                "status": "PENDING_BLINDED_REVIEW",
                "model": selected[0]["model"],
                "top_k": top_k,
                "prompt_version": PROMPT_VERSION,
                "prompt_variant": prompt_variant,
                "cases": len(selected),
                "successes": len(successful),
                "failures": len(selected) - len(successful),
                "failure_rate": (len(selected) - len(successful)) / len(selected),
                "latency_p50_ms": latency.get("p50_ms"),
                "latency_p70_ms": latency.get("p70_ms"),
                "latency_p95_ms": latency.get("p95_ms"),
                "latency_p100_ms": latency.get("p100_ms"),
                "ttft_p50_ms": ttft.get("p50_ms"),
                "ttft_p70_ms": ttft.get("p70_ms"),
                "ttft_p95_ms": ttft.get("p95_ms"),
                "ttft_p100_ms": ttft.get("p100_ms"),
                "mean_output_tokens": statistics.fmean(tokens) if tokens else None,
                "mean_prompt_tokens": statistics.fmean(prompt_tokens) if prompt_tokens else None,
                "prompt_tokens_min": min(prompt_tokens) if prompt_tokens else None,
                "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
                "schema_valid_rate": _mean_bool(diagnostics, "schema_valid"),
                "grounded_citation_validity_rate": (
                    statistics.fmean(citation_validity) if citation_validity else None
                ),
                "missing_citation_rate": _mean_bool(diagnostics, "missing_citation"),
                "unknown_citation_rate": _mean_nonempty(diagnostics, "unknown_evidence_ids"),
                "novel_number_diagnostic_rate": _mean_nonempty(diagnostics, "novel_numbers"),
                "retrieved_qrel_rate": _mean_bool(diagnostics, "qrel_retrieved"),
                "insufficient_context_rate": (
                    sum(row.get("answer_status") == "INSUFFICIENT_CONTEXT" for row in successful)
                    / len(successful)
                    if successful
                    else None
                ),
                "answer_when_context_available_rate": (
                    sum(row.get("answer_status") == "ANSWER" for row in context_available)
                    / len(context_available)
                    if context_available
                    else None
                ),
                "refusal_when_context_missing_rate": (
                    sum(
                        row.get("answer_status") == "INSUFFICIENT_CONTEXT"
                        for row in context_missing
                    )
                    / len(context_missing)
                    if context_missing
                    else None
                ),
                "appropriate_context_behavior_rate": (
                    sum(
                        (
                            row.get("answer_status") == "ANSWER"
                            if row.get("diagnostics", {}).get("qrel_retrieved")
                            else row.get("answer_status") == "INSUFFICIENT_CONTEXT"
                        )
                        for row in successful
                    )
                    / len(successful)
                    if successful
                    else None
                ),
                "human_correctness": "PENDING_BLINDED_REVIEW",
                "human_relevance": "PENDING_BLINDED_REVIEW",
                "human_faithfulness": "PENDING_BLINDED_REVIEW",
            }
        )
    return rows


def write_blinded_judgments(
    observations: Sequence[dict[str, Any]],
    context_rows: Sequence[dict[str, Any]],
    output_path: Path,
    mapping_path: Path,
    *,
    seed: int = 20260818,
    experiment_id: str = "",
) -> None:
    contexts = {row["case_id"]: row for row in context_rows}
    prior_scores: dict[str, dict[str, str]] = {}
    if output_path.exists():
        with output_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                prior_scores[row["blind_output_id"]] = {
                    key: row.get(key, "")
                    for key in (
                        "human_correctness_1_to_5",
                        "human_relevance_1_to_5",
                        "human_faithfulness_1_to_5",
                        "human_notes",
                        "reviewer_type",
                    )
                }
    blind_rows: list[dict[str, Any]] = []
    mapping: list[dict[str, str]] = []
    for observation in observations:
        case = contexts[observation["case_id"]]
        identity = (
            f"{seed}:{experiment_id}:{observation['case_id']}:{observation['model']}:"
            f"{observation['top_k']}:{observation['prompt_variant']}"
        )
        blind_id = "g-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
        mapping.append(
            {
                "blind_output_id": blind_id,
                "case_id": observation["case_id"],
                "model": observation["model"],
                "top_k": str(observation["top_k"]),
                "prompt_variant": observation["prompt_variant"],
                "experiment_id": experiment_id,
            }
        )
        blind_row = {
                "blind_output_id": blind_id,
                "rubric_version": HUMAN_RUBRIC_VERSION,
                "case_id": observation["case_id"],
                "category": observation["category"],
                "question": observation["question"],
                "reference_answer": case["expected"]["reference_answer"],
                "required_claims_json": json.dumps(
                    case["expected"]["required_claims"], ensure_ascii=False
                ),
                "retrieved_evidence_json": json.dumps(
                    case["contexts"][: int(observation["top_k"])], ensure_ascii=False
                ),
                "generated_status": observation.get("answer_status") or "ERROR",
                "generated_answer": observation.get("answer", ""),
                "generated_evidence_ids_json": json.dumps(
                    observation.get("evidence_ids", []), ensure_ascii=False
                ),
                "human_correctness_1_to_5": "",
                "human_relevance_1_to_5": "",
                "human_faithfulness_1_to_5": "",
                "human_notes": "",
                "reviewer_type": "human",
            }
        blind_row.update(prior_scores.get(blind_id, {}))
        blind_rows.append(blind_row)
    random.Random(seed).shuffle(blind_rows)
    write_csv(output_path, blind_rows)
    write_jsonl(mapping_path, mapping)


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_pending_ablation_csvs(topk_path: Path, prompt_path: Path) -> None:
    common = {
        "status": "PENDING_BLINDED_MODEL_REVIEW",
        "reason": "A winning model cannot be selected before human quality scores are entered.",
    }
    write_csv(
        topk_path,
        [{**common, "model": "", "top_k": value} for value in (1, 3, 5, 10)],
    )
    write_csv(
        prompt_path,
        [
            {**common, "model": "", "top_k": "", "prompt_variant": variant}
            for variant in (
                "strict_context_only",
                "context_only_refusal",
                "structured_evidence_ids",
            )
        ],
    )


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "p50_ms": percentile(0.50),
        "p70_ms": percentile(0.70),
        "p95_ms": percentile(0.95),
        "p100_ms": ordered[-1],
    }


def _content_diagnostics(observation: dict[str, Any], cache_row: dict[str, Any]) -> dict[str, Any]:
    context_text = " ".join(str(item["text"]) for item in cache_row["contexts"])
    answer = str(observation.get("answer", ""))
    answer_numbers = {_normalize_number(value) for value in _numbers(answer)}
    context_numbers = {_normalize_number(value) for value in _numbers(context_text)}
    relevant = set(cache_row["expected"]["relevant_parent_ids"])
    retrieved = {item["parent_id"] for item in cache_row["contexts"]}
    return {
        "novel_numbers": sorted(answer_numbers - context_numbers),
        "qrel_retrieved": bool(relevant & retrieved),
    }


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d[\d,.]*(?:%|\b)", text))


def _normalize_number(value: str) -> str:
    return value.rstrip("%,.").replace(",", "")


def _mean_bool(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    return statistics.fmean(bool(row.get(key)) for row in rows) if rows else None


def _mean_nonempty(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    return statistics.fmean(bool(row.get(key)) for row in rows) if rows else None


def _valid_cache(rows: Sequence[dict[str, Any]]) -> bool:
    return bool(rows) and all(len(row.get("contexts", [])) == MODEL_ABLATION_TOP_K for row in rows)


def _assert_frozen(index_winner: dict[str, Any]) -> None:
    observed = {
        "model": index_winner["embedding_model"],
        "chunking_strategy": index_winner["chunking_config"]["strategy"],
        "chunk_size_words": index_winner["chunking_config"]["max_words"],
        "index_engine": index_winner["backend_config"]["engine"],
        "index_type": index_winner["backend_config"]["index_type"],
        "m": index_winner["backend_config"]["m"],
        "ef_construction": index_winner["backend_config"]["ef_construction"],
        "ef_search": index_winner["backend_config"]["ef_search"],
    }
    if observed != FROZEN_STACK:
        raise RuntimeError(f"Retrieval stack changed: {observed}")


def _retrieval_fingerprint(dataset_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        dataset_path,
        Path("results/index_winner.json"),
        Path("results/chunking_winner.json"),
        Path("results/final_retriever_config.json"),
    ):
        digest.update(path.read_bytes())
    digest.update(json.dumps(FROZEN_STACK, sort_keys=True).encode())
    digest.update(b"gold=stt_reference;top_k=10")
    return digest.hexdigest()
