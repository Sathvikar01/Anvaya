"""Run additive latency and fast-path ablations without mutating the frozen pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from hh_goa_rag.generation.prompts import build_messages
from hh_goa_rag.generation.sarvam import (
    _ANSWER_SCHEMA,
    SARVAM_CHAT_ENDPOINT,
    _parse_answer,
)
from hh_goa_rag.latency import (
    compress_contexts,
    extractive_is_grounded,
    latency_summary,
    select_extractive_answer,
    token_f1,
)

CONTEXT_CACHE = Path("cache/generation/gold_contexts_top10.jsonl")
RAW_DIR = Path("cache/latency")
GENERATION_CSV = Path("results/generation_latency_ablation.csv")
FAST_PATH_CSV = Path("results/fast_path_ablation.csv")
AUTOPSY_CSV = Path("results/latency_autopsy.csv")
STT_CSV = Path("results/stt_latency_ablation.csv")
CHAT_SCHEMA = _ANSWER_SCHEMA


@dataclass(frozen=True)
class GenerationExperiment:
    name: str
    max_tokens: int = 192
    reasoning: str = "none"
    stream: bool = True
    persistent_client: bool = False
    context_mode: str = "top10"


class Trace:
    """Collect documented httpcore trace events without recording request data."""

    def __init__(self, started_ns: int) -> None:
        self.started_ns = started_ns
        self.events: dict[str, float] = {}

    def __call__(self, name: str, _info: dict[str, Any]) -> None:
        self.events[name] = (time.perf_counter_ns() - self.started_ns) / 1e6

    def duration(self, start: str, end: str) -> float | None:
        if start not in self.events or end not in self.events:
            return None
        return max(0.0, self.events[end] - self.events[start])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _contexts(row: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    contexts = list(row["contexts"])
    if mode == "top10":
        return contexts
    if mode == "top5":
        return contexts[:5]
    if mode == "top3":
        return contexts[:3]
    if mode == "compressed_top5":
        return compress_contexts(str(row["question"]), contexts, limit=5)
    raise ValueError(f"Unknown context mode: {mode}")


def _profile_generation(
    api_key: str,
    row: dict[str, Any],
    experiment: GenerationExperiment,
    shared_client: httpx.Client | None,
) -> dict[str, Any]:
    contexts = _contexts(row, experiment.context_mode)
    serialization_started = time.perf_counter_ns()
    messages = build_messages(row["question"], contexts, variant="strict_context_only")
    payload: dict[str, Any] = {
        "model": "sarvam-105b",
        "messages": messages,
        "temperature": 0,
        "max_tokens": experiment.max_tokens,
        "stream": experiment.stream,
        "response_format": CHAT_SCHEMA,
    }
    if experiment.reasoning == "none":
        payload["reasoning_effort"] = None
    elif experiment.reasoning != "provider_default":
        payload["reasoning_effort"] = experiment.reasoning
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    serialization_ms = (time.perf_counter_ns() - serialization_started) / 1e6
    client_setup_started = time.perf_counter_ns()
    client = shared_client or httpx.Client(timeout=30)
    client_setup_ms = (time.perf_counter_ns() - client_setup_started) / 1e6
    request_started = time.perf_counter_ns()
    trace = Trace(request_started)
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    content = ""
    usage: dict[str, Any] = {}
    ttft_ms: float | None = None
    first_content_ns: int | None = None
    status = "error"
    error_code = None
    http_status = None
    try:
        if experiment.stream:
            pieces: list[str] = []
            with client.stream(
                "POST",
                SARVAM_CHAT_ENDPOINT,
                headers=headers,
                content=serialized.encode("utf-8"),
                extensions={"trace": trace},
            ) as response:
                http_status = response.status_code
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    event = line[5:].strip()
                    if event == "[DONE]":
                        break
                    body = json.loads(event)
                    if body.get("usage"):
                        usage = body["usage"]
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = str(delta.get("content") or "")
                    if piece:
                        if first_content_ns is None:
                            first_content_ns = time.perf_counter_ns()
                            ttft_ms = (first_content_ns - request_started) / 1e6
                        pieces.append(piece)
            content = "".join(pieces)
        else:
            response = client.post(
                SARVAM_CHAT_ENDPOINT,
                headers=headers,
                content=serialized.encode("utf-8"),
                extensions={"trace": trace},
            )
            http_status = response.status_code
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage") or {}
            content = str(body["choices"][0]["message"].get("content") or "")
        response_complete_ns = time.perf_counter_ns()
        parse_started = time.perf_counter_ns()
        allowed = {str(context["parent_id"]) for context in contexts}
        parsed, diagnostics = _parse_answer(content, allowed)
        parsing_ms = (time.perf_counter_ns() - parse_started) / 1e6
        grounding_started = time.perf_counter_ns()
        grounded = bool(parsed and diagnostics.get("schema_valid"))
        grounding_ms = (time.perf_counter_ns() - grounding_started) / 1e6
        status = "ok" if parsed else "error"
        error_code = None if parsed else "invalid_structured_output"
        answer = str(parsed["answer"]) if parsed else ""
        evidence_ids = list(parsed["evidence_ids"]) if parsed else []
        full_ms = (response_complete_ns - request_started) / 1e6
        return {
            "configuration": experiment.name,
            "case_id": row["case_id"],
            "category": row["category"],
            "status": status,
            "http_status": http_status,
            "error_code": error_code,
            "prompt_serialization_ms": serialization_ms,
            "client_setup_ms": client_setup_ms,
            "connection_setup_ms": trace.duration(
                "connection.connect_tcp.started", "connection.start_tls.complete"
            ),
            "audio_or_request_send_ms": trace.duration(
                "http11.send_request_body.started", "http11.send_request_body.complete"
            ),
            "server_wait_ms": trace.duration(
                "http11.send_request_body.complete", "http11.receive_response_headers.complete"
            ),
            "ttft_ms": ttft_ms,
            "token_generation_ms": (
                (response_complete_ns - first_content_ns) / 1e6 if first_content_ns else None
            ),
            "full_completion_ms": full_ms,
            "usable_answer_ms": full_ms + parsing_ms + grounding_ms,
            "output_parsing_ms": parsing_ms,
            "grounding_validation_ms": grounding_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "answer": answer,
            "evidence_ids": evidence_ids,
            "reference_token_f1": token_f1(answer, row["expected"]["reference_answer"]),
            "citation_valid": bool(evidence_ids) and set(evidence_ids) <= allowed,
            "grounding_valid": grounded,
            "context_mode": experiment.context_mode,
            "max_tokens": experiment.max_tokens,
            "reasoning": experiment.reasoning,
            "stream": experiment.stream,
            "persistent_client": experiment.persistent_client,
        }
    except Exception as error:
        return {
            "configuration": experiment.name,
            "case_id": row["case_id"],
            "category": row["category"],
            "status": "error",
            "http_status": http_status,
            "error_code": type(error).__name__,
            "prompt_serialization_ms": serialization_ms,
            "client_setup_ms": client_setup_ms,
            "connection_setup_ms": trace.duration(
                "connection.connect_tcp.started", "connection.start_tls.complete"
            ),
            "full_completion_ms": (time.perf_counter_ns() - request_started) / 1e6,
            "context_mode": experiment.context_mode,
            "max_tokens": experiment.max_tokens,
            "reasoning": experiment.reasoning,
            "stream": experiment.stream,
            "persistent_client": experiment.persistent_client,
        }
    finally:
        if shared_client is None:
            client.close()


def _summary_fields(
    rows: list[dict[str, Any]],
    field: str,
    prefix: str,
    *,
    successes_only: bool = True,
) -> dict[str, Any]:
    selected = [row for row in rows if not successes_only or row["status"] == "ok"]
    summary = latency_summary(row.get(field) for row in selected)
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _aggregate_generation(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    configurations = list(dict.fromkeys(row["configuration"] for row in observations))
    for name in configurations:
        rows = [row for row in observations if row["configuration"] == name]
        successes = [row for row in rows if row["status"] == "ok"]
        first = rows[0]
        result: dict[str, Any] = {
            "configuration": name,
            "measurement_label": "ABLATION",
            "model": "sarvam-105b",
            "reasoning": first["reasoning"],
            "max_tokens": first["max_tokens"],
            "stream": first["stream"],
            "persistent_client": first["persistent_client"],
            "context_mode": first["context_mode"],
            "cases": len(rows),
            "successes": len(successes),
            "failures": len(rows) - len(successes),
            "failure_rate": 1 - len(successes) / len(rows),
            "mean_prompt_tokens": _mean(successes, "prompt_tokens"),
            "mean_output_tokens": _mean(successes, "output_tokens"),
            "mean_reference_token_f1": _mean(successes, "reference_token_f1"),
            "citation_validity_rate": _rate(successes, "citation_valid"),
            "grounding_validity_rate": _rate(successes, "grounding_valid"),
            "quality_method": "deterministic reference-token/citation/grounding proxies",
            "error_codes": ";".join(
                sorted({str(row["error_code"]) for row in rows if row.get("error_code")})
            ),
        }
        result.update(
            _summary_fields(
                rows, "full_completion_ms", "observed_request", successes_only=False
            )
        )
        for field, prefix in (
            ("prompt_serialization_ms", "serialization"),
            ("client_setup_ms", "client_setup"),
            ("connection_setup_ms", "connection"),
            ("server_wait_ms", "server_wait"),
            ("ttft_ms", "ttft"),
            ("token_generation_ms", "token_generation"),
            ("full_completion_ms", "full"),
            ("usable_answer_ms", "usable"),
            ("output_parsing_ms", "parsing"),
            ("grounding_validation_ms", "grounding"),
        ):
            result.update(_summary_fields(rows, field, prefix))
        output.append(result)
    return output


def probe_chat_model(api_key: str, model: str) -> dict[str, Any]:
    """Make a minimal live capability probe and retain no generated content."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "reasoning_effort": None,
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }
    started = time.perf_counter_ns()
    with httpx.Client(timeout=30) as client:
        response = client.post(
            SARVAM_CHAT_ENDPOINT,
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    body = response.json() if response.headers.get("content-type", "").startswith(
        "application/json"
    ) else {}
    error = body.get("error") or {}
    usage = body.get("usage") or {}
    success = response.status_code < 400
    return {
        "configuration": f"model_probe_{model}",
        "measurement_label": "ABLATION (live account capability probe)",
        "model": model,
        "reasoning": "none",
        "max_tokens": 16,
        "stream": False,
        "persistent_client": False,
        "context_mode": "minimal_probe",
        "cases": 1,
        "successes": int(success),
        "failures": int(not success),
        "failure_rate": float(not success),
        "error_codes": "" if success else str(error.get("code") or "api_error"),
        "observed_request_n": 1,
        "observed_request_p50_ms": latency_ms,
        "observed_request_p70_ms": latency_ms,
        "observed_request_p95_ms": latency_ms,
        "observed_request_p100_ms": latency_ms,
        "observed_request_mean_ms": latency_ms,
        "mean_prompt_tokens": usage.get("prompt_tokens"),
        "mean_output_tokens": usage.get("completion_tokens"),
        "quality_method": "capability probe only; no quality score",
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    return sum(bool(row.get(field)) for row in rows) / len(rows) if rows else None


def run_generation(limit: int | None) -> list[dict[str, Any]]:
    load_dotenv(".env", override=False)
    api_key = os.getenv("SARVAM_API_KEY", "")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is required")
    cases = _read_jsonl(CONTEXT_CACHE)
    if limit:
        cases = cases[:limit]
    experiments = [
        GenerationExperiment("reasoning_none_current"),
        GenerationExperiment("reasoning_provider_default", reasoning="provider_default"),
        GenerationExperiment("max_tokens_128", max_tokens=128),
        GenerationExperiment("max_tokens_64", max_tokens=64),
        GenerationExperiment("max_tokens_32", max_tokens=32),
        GenerationExperiment("non_streaming", stream=False),
        GenerationExperiment("persistent_http", persistent_client=True),
        GenerationExperiment("context_top5", context_mode="top5"),
        GenerationExperiment("context_top3", context_mode="top3"),
        GenerationExperiment("compressed_top5", context_mode="compressed_top5"),
    ]
    observations: list[dict[str, Any]] = []
    for experiment in experiments:
        shared = httpx.Client(timeout=30) if experiment.persistent_client else None
        try:
            for case in cases:
                observation = _profile_generation(api_key, case, experiment, shared)
                observations.append(observation)
                print(
                    f"generation {experiment.name} {case['case_id']}: "
                    f"{observation['status']} {observation.get('full_completion_ms', 0):.1f} ms",
                    flush=True,
                )
        finally:
            if shared:
                shared.close()
    _write_jsonl(RAW_DIR / "generation_observations.jsonl", observations)
    aggregated = _aggregate_generation(observations)
    aggregated.append(probe_chat_model(api_key, "sarvam-30b"))
    _write_csv(GENERATION_CSV, aggregated)
    return aggregated


def run_fast_path() -> list[dict[str, Any]]:
    cases = _read_jsonl(CONTEXT_CACHE)
    prompt_outputs = _read_jsonl(
        Path("results/runs/generation/20260818T123859Z/prompt_outputs.jsonl")
    )
    baseline = {
        row["case_id"]: row
        for row in prompt_outputs
        if row.get("prompt_variant") == "strict_context_only"
    }
    per_case: list[dict[str, Any]] = []
    for case in cases:
        selected = select_extractive_answer(case["question"], case["contexts"])
        grounded = extractive_is_grounded(
            selected.answer, selected.evidence_ids, case["contexts"]
        )
        relevant_ids = set(case["expected"]["relevant_parent_ids"])
        citation_relevant = bool(set(selected.evidence_ids) & relevant_ids)
        base = baseline[case["case_id"]]
        for architecture in ("extractive_high_confidence", "hybrid_router"):
            use_extract = selected.eligible
            if architecture == "extractive_high_confidence":
                answer = selected.answer
                evidence_ids = selected.evidence_ids
                latency = selected.latency_ms
                completed = use_extract
                path = "extractive" if use_extract else "abstain"
            elif use_extract:
                answer = selected.answer
                evidence_ids = selected.evidence_ids
                latency = selected.latency_ms
                completed = True
                path = "extractive"
            else:
                answer = str(base.get("answer") or "")
                evidence_ids = tuple(base.get("evidence_ids") or ())
                latency = selected.latency_ms + float(base["latency_ms"])
                completed = base.get("status") == "ok"
                path = "generator_fallback"
                grounded = bool(base.get("diagnostics", {}).get("schema_valid"))
                citation_relevant = bool(set(evidence_ids) & relevant_ids)
            per_case.append(
                {
                    "architecture": architecture,
                    "case_id": case["case_id"],
                    "completed": completed,
                    "path": path,
                    "answer_latency_ms": latency,
                    "reference_token_f1": token_f1(
                        answer, case["expected"]["reference_answer"]
                    ),
                    "grounding_valid": grounded if completed else False,
                    "citation_valid": bool(evidence_ids)
                    and set(evidence_ids)
                    <= {context["parent_id"] for context in case["contexts"]},
                    "citation_relevant": citation_relevant if completed else False,
                }
            )
    output: list[dict[str, Any]] = []
    with Path("results/generation_prompt_ablation.csv").open(encoding="utf-8") as handle:
        baseline_csv = next(
            row
            for row in csv.DictReader(handle)
            if row["prompt_variant"] == "strict_context_only"
        )
    output.append(
        {
            "configuration": "current_generative_rag",
            "measurement_label": "ABLATION (historical frozen generation run)",
            "cases": int(baseline_csv["cases"]),
            "completion_rate": 1 - float(baseline_csv["failure_rate"]),
            "extractive_route_rate": 0,
            "generator_fallback_rate": 1,
            "answer_p50_ms": float(baseline_csv["latency_p50_ms"]),
            "answer_p70_ms": float(baseline_csv["latency_p70_ms"]),
            "answer_p95_ms": float(baseline_csv["latency_p95_ms"]),
            "answer_p100_ms": float(baseline_csv["latency_p100_ms"]),
            "correctness_1_to_5": float(baseline_csv["human_correctness"]),
            "relevance_1_to_5": float(baseline_csv["human_relevance"]),
            "faithfulness_1_to_5": float(baseline_csv["human_faithfulness"]),
            "quality_method": baseline_csv["reviewer_type"],
            "grounding_validity_rate": float(baseline_csv["grounded_citation_validity_rate"]),
        }
    )
    for architecture in ("extractive_high_confidence", "hybrid_router"):
        rows = [row for row in per_case if row["architecture"] == architecture]
        complete = [row for row in rows if row["completed"]]
        latency = latency_summary(row["answer_latency_ms"] for row in complete)
        output.append(
            {
                "configuration": architecture,
                "measurement_label": "ABLATION",
                "cases": len(rows),
                "completion_rate": len(complete) / len(rows),
                "extractive_route_rate": sum(row["path"] == "extractive" for row in rows)
                / len(rows),
                "generator_fallback_rate": sum(
                    row["path"] == "generator_fallback" for row in rows
                )
                / len(rows),
                "answer_p50_ms": latency["p50_ms"],
                "answer_p70_ms": latency["p70_ms"],
                "answer_p95_ms": latency["p95_ms"],
                "answer_p100_ms": latency["p100_ms"],
                "mean_reference_token_f1": _mean(complete, "reference_token_f1"),
                "citation_validity_rate": _rate(complete, "citation_valid"),
                "citation_relevance_rate": _rate(complete, "citation_relevant"),
                "grounding_validity_rate": _rate(complete, "grounding_valid"),
                "quality_method": "deterministic verbatim/citation/reference-token proxies",
            }
        )
    _write_jsonl(RAW_DIR / "fast_path_observations.jsonl", per_case)
    _write_csv(FAST_PATH_CSV, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("generation", "fast-path", "all"), default="all"
    )
    parser.add_argument("--limit", type=int, help="Diagnostic-only case limit")
    args = parser.parse_args()
    if args.phase in {"generation", "all"}:
        run_generation(args.limit)
    if args.phase in {"fast-path", "all"}:
        run_fast_path()


if __name__ == "__main__":
    main()
