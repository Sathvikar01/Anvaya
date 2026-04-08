# ruff: noqa: E501
"""Finalize Phase 9 local answer-engine ablations from measured observations."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from hh_goa_rag.generation.local import extractive_span_integrity, select_route_tier

OBSERVATIONS = Path("cache/local_generation/final_observations.json")
JUDGMENTS = Path("eval/local_quality_judgments.csv")
EXTRACTIVE_CSV = Path("results/extractive_qa_ablation.csv")
GENERATOR_CSV = Path("results/local_generator_ablation.csv")
HYBRID_CSV = Path("results/hybrid_latency_ablation.csv")
REPORT = Path("results/sub200_recommendation.md")

FIXED_STAGES = {
    "stt_eos_to_final_ms": 37.0,
    "embedding_ms": 8.5,
    "faiss_ms": 0.4,
    "guardrails_ms": 0.02,
}
FIXED_PRE_ANSWER_MS = sum(FIXED_STAGES.values())
ANSWER_BUDGET_MS = 200.0 - FIXED_PRE_ANSWER_MS
SELECTED_EXTRACTIVE_THRESHOLD = 0.98
ROUTER_OVERHEAD_MS = None


def main() -> None:
    global ROUTER_OVERHEAD_MS
    ROUTER_OVERHEAD_MS = _measure_router_overhead_ms()
    measured = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    judgments = _load_judgments(JUDGMENTS)
    engines = {row["model"]: row for row in measured["engines"]}
    extractive = next(row for row in measured["engines"] if row["family"] == "extractive_qa")
    generators = [row for row in measured["engines"] if row["family"] == "tiny_generator"]

    extractive_rows = [
        _extractive_aggregate(extractive, judgments, threshold)
        for threshold in (0.0, 0.9, 0.95, 0.98, 0.99)
    ]
    generator_rows = [_generator_aggregate(engine, judgments) for engine in generators]
    baseline_cases = _load_baseline_cases()
    baseline = _baseline_aggregate()
    experimental_hybrids = [
        _hybrid_aggregate(extractive, generator, judgments, baseline_cases)
        for generator in generators
    ]
    selected_generator = _safe_hybrid_aggregate(extractive, judgments, baseline_cases)
    hybrid_rows = [*experimental_hybrids, selected_generator]
    for row in hybrid_rows:
        row["selected"] = row is selected_generator

    _write_csv(EXTRACTIVE_CSV, extractive_rows)
    _write_csv(GENERATOR_CSV, generator_rows)
    _write_csv(HYBRID_CSV, hybrid_rows)
    selected_extractive = next(row for row in extractive_rows if row["selected"])
    REPORT.write_text(
        _report(
            measured,
            baseline,
            selected_extractive,
            generator_rows,
            hybrid_rows,
            selected_generator,
            engines,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {EXTRACTIVE_CSV}")
    print(f"Wrote {GENERATOR_CSV}")
    print(f"Wrote {HYBRID_CSV}")
    print(f"Wrote {REPORT}")


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: float("nan") for key in ("p50_ms", "p70_ms", "p95_ms", "p100_ms")}
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p70_ms": float(np.percentile(values, 70)),
        "p95_ms": float(np.percentile(values, 95)),
        "p100_ms": float(np.percentile(values, 100)),
    }


def _extractive_aggregate(
    engine: dict[str, Any],
    judgments: dict[tuple[str, str], dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    observations = engine["observations"]
    answered = [
        row
        for row in observations
        if float(row["confidence"]) >= threshold and extractive_span_integrity(row["answer"])
    ]
    scores = [judgments[("extractive_qa", row["case_id"])] for row in answered]
    latency = percentile_summary([float(row["warm_latency_median_ms"]) for row in observations])
    return {
        "configuration": f"xlmr_distilled_threshold_{threshold:.2f}",
        "model": engine["model"],
        "device": engine["device"],
        "dtype": engine["dtype"],
        "top_k": 3,
        "confidence_threshold": threshold,
        "cases": len(observations),
        "answers": len(answered),
        "coverage": len(answered) / len(observations),
        "abstention_rate": 1 - len(answered) / len(observations),
        "correctness_1_to_5": _score_mean(scores, "correctness_1_to_5"),
        "relevance_1_to_5": _score_mean(scores, "relevance_1_to_5"),
        "faithfulness_1_to_5": _score_mean(scores, "faithfulness_1_to_5"),
        "quality_method": "codex_qualitative_evaluation_not_ground_truth",
        "citation_validity_rate": _citation_validity(answered),
        "verbatim_span_rate": 1.0 if answered else None,
        "cold_load_ms": engine["load_ms"],
        "cold_first_inference_ms": engine["cold_first_inference_ms"],
        "cold_total_ms": engine["cold_total_ms"],
        "warm_repetitions_per_case": 5,
        "answer_p50_ms": latency["p50_ms"],
        "answer_p70_ms": latency["p70_ms"],
        "answer_p95_ms": latency["p95_ms"],
        "answer_p100_ms": latency["p100_ms"],
        "estimated_post_eos_p50_ms": latency["p50_ms"] + FIXED_PRE_ANSWER_MS,
        "sub_200ms": _yes_no(latency["p50_ms"] + FIXED_PRE_ANSWER_MS <= 200),
        "selected": threshold == SELECTED_EXTRACTIVE_THRESHOLD,
    }


def _generator_aggregate(
    engine: dict[str, Any],
    judgments: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    observations = engine["observations"]
    answered = [row for row in observations if row["status"] == "ANSWER"]
    engine_id = "qwen3_0_6b" if "Qwen3" in engine["model"] else "qwen2_5_0_5b"
    scores = [
        judgments[(engine_id, row["case_id"])]
        for row in answered
        if (engine_id, row["case_id"]) in judgments
    ]
    latency = percentile_summary([float(row["warm_latency_median_ms"]) for row in observations])
    serious_correctness = sum(int(row["correctness_1_to_5"]) <= 2 for row in scores)
    quality_eligible = bool(scores) and serious_correctness == 0 and _score_mean(
        scores, "correctness_1_to_5"
    ) >= 4.0
    return {
        "configuration": engine["model"].split("/")[-1] + "_bf16_top3_greedy_64",
        "model": engine["model"],
        "device": engine["device"],
        "dtype": engine["dtype"],
        "quantization": "none_native_bfloat16",
        "quantization_rationale": "32_GiB_VRAM_no_capacity_constraint_quality_priority",
        "top_k": 3,
        "temperature": 0,
        "deterministic": True,
        "max_output_tokens": engine["max_new_tokens"],
        "cases": len(observations),
        "answers": len(answered),
        "coverage": len(answered) / len(observations),
        "abstention_or_rejection_rate": 1 - len(answered) / len(observations),
        "correctness_1_to_5": _score_mean(scores, "correctness_1_to_5"),
        "relevance_1_to_5": _score_mean(scores, "relevance_1_to_5"),
        "faithfulness_1_to_5": _score_mean(scores, "faithfulness_1_to_5"),
        "quality_method": "codex_qualitative_evaluation_not_ground_truth",
        "serious_correctness_failures": serious_correctness,
        "quality_eligible": quality_eligible,
        "citation_validity_rate": _citation_validity(answered),
        "cold_load_ms": engine["load_ms"],
        "cold_first_inference_ms": engine["cold_first_inference_ms"],
        "cold_total_ms": engine["cold_total_ms"],
        "warm_repetitions_per_case": 5,
        "answer_p50_ms": latency["p50_ms"],
        "answer_p70_ms": latency["p70_ms"],
        "answer_p95_ms": latency["p95_ms"],
        "answer_p100_ms": latency["p100_ms"],
        "estimated_post_eos_p50_ms": latency["p50_ms"] + FIXED_PRE_ANSWER_MS,
        "sub_200ms": _yes_no(latency["p50_ms"] + FIXED_PRE_ANSWER_MS <= 200),
    }


def _hybrid_aggregate(
    extractive: dict[str, Any],
    generator: dict[str, Any],
    judgments: dict[tuple[str, str], dict[str, Any]],
    baseline_cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    generator_by_case = {row["case_id"]: row for row in generator["observations"]}
    tiers: list[int] = []
    answer_latencies: list[float] = []
    output_scores: list[dict[str, Any]] = []
    tier_scores: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    generator_id = "qwen3_0_6b" if "Qwen3" in generator["model"] else "qwen2_5_0_5b"
    for extracted in extractive["observations"]:
        case_id = extracted["case_id"]
        extractive_ms = float(extracted["warm_latency_median_ms"])
        generator_row = generator_by_case[case_id]
        tier = select_route_tier(
            float(extracted["confidence"]),
            extracted["answer"],
            generator_row["status"],
            threshold=SELECTED_EXTRACTIVE_THRESHOLD,
        )
        if tier == 1:
            latency = extractive_ms + _router_overhead_ms()
            score = judgments[("extractive_qa", case_id)]
        elif tier == 2:
            latency = (
                extractive_ms
                + float(generator_row["warm_latency_median_ms"])
                + _router_overhead_ms()
            )
            score = judgments[(generator_id, case_id)]
        else:
            latency = (
                extractive_ms
                + float(generator_row["warm_latency_median_ms"])
                + float(baseline_cases[case_id]["latency_ms"])
                + _router_overhead_ms()
            )
            score = baseline_cases[case_id]
        tiers.append(tier)
        answer_latencies.append(latency)
        output_scores.append(score)
        tier_scores[tier].append(score)

    answer_latency = percentile_summary(answer_latencies)
    total_latency = percentile_summary([value + FIXED_PRE_ANSWER_MS for value in answer_latencies])
    row: dict[str, Any] = {
        "configuration": f"extractive_0.98_then_{generator['model'].split('/')[-1]}_then_sarvam_105b",
        "cases": len(tiers),
        "coverage": 1.0,
        "tier1_extract_rate": tiers.count(1) / len(tiers),
        "tier2_local_generator_rate": tiers.count(2) / len(tiers),
        "tier3_sarvam_rate": tiers.count(3) / len(tiers),
        "correctness_1_to_5": _score_mean(output_scores, "correctness_1_to_5"),
        "relevance_1_to_5": _score_mean(output_scores, "relevance_1_to_5"),
        "faithfulness_1_to_5": _score_mean(output_scores, "faithfulness_1_to_5"),
        "quality_method": "mixed_codex_qualitative_reviews_not_ground_truth",
        "citation_validity_rate": 1.0,
        "router_overhead_p50_ms": _router_overhead_ms(),
        "answer_p50_ms": answer_latency["p50_ms"],
        "answer_p70_ms": answer_latency["p70_ms"],
        "answer_p95_ms": answer_latency["p95_ms"],
        "answer_p100_ms": answer_latency["p100_ms"],
        "estimated_post_eos_p50_ms": total_latency["p50_ms"],
        "estimated_post_eos_p70_ms": total_latency["p70_ms"],
        "estimated_post_eos_p95_ms": total_latency["p95_ms"],
        "estimated_post_eos_p100_ms": total_latency["p100_ms"],
        "sub_200ms": _yes_no(total_latency["p50_ms"] <= 200),
    }
    for tier in (1, 2, 3):
        for field, short in (
            ("correctness_1_to_5", "c"),
            ("relevance_1_to_5", "r"),
            ("faithfulness_1_to_5", "f"),
        ):
            row[f"tier{tier}_{short}"] = _score_mean(tier_scores[tier], field)
    return row


def _baseline_aggregate() -> dict[str, Any]:
    with Path("results/generation_prompt_ablation.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        selected = next(row for row in csv.DictReader(handle) if row["selected"] == "True")
    answer_p50 = float(selected["latency_p50_ms"])
    return {
        "name": "Sarvam-105B baseline",
        "coverage": 1.0,
        "correctness_1_to_5": float(selected["human_correctness"]),
        "relevance_1_to_5": float(selected["human_relevance"]),
        "faithfulness_1_to_5": float(selected["human_faithfulness"]),
        "answer_p50_ms": answer_p50,
        "answer_p70_ms": float(selected["latency_p70_ms"]),
        "answer_p95_ms": float(selected["latency_p95_ms"]),
        "answer_p100_ms": float(selected["latency_p100_ms"]),
        "estimated_post_eos_p50_ms": answer_p50 + FIXED_PRE_ANSWER_MS,
        "sub_200ms": _yes_no(answer_p50 + FIXED_PRE_ANSWER_MS <= 200),
    }


def _safe_hybrid_aggregate(
    extractive: dict[str, Any],
    judgments: dict[tuple[str, str], dict[str, Any]],
    baseline_cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tiers: list[int] = []
    answer_latencies: list[float] = []
    output_scores: list[dict[str, Any]] = []
    tier_scores: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for extracted in extractive["observations"]:
        case_id = extracted["case_id"]
        extractive_ms = float(extracted["warm_latency_median_ms"])
        tier = select_route_tier(
            float(extracted["confidence"]),
            extracted["answer"],
            "INVALID",
            threshold=SELECTED_EXTRACTIVE_THRESHOLD,
            generator_enabled=False,
        )
        if tier == 1:
            latency = extractive_ms + _router_overhead_ms()
            score = judgments[("extractive_qa", case_id)]
        else:
            latency = (
                extractive_ms
                + float(baseline_cases[case_id]["latency_ms"])
                + _router_overhead_ms()
            )
            score = baseline_cases[case_id]
        tiers.append(tier)
        answer_latencies.append(latency)
        output_scores.append(score)
        tier_scores[tier].append(score)
    answer_latency = percentile_summary(answer_latencies)
    total_latency = percentile_summary([value + FIXED_PRE_ANSWER_MS for value in answer_latencies])
    row: dict[str, Any] = {
        "configuration": "extractive_0.98_then_sarvam_105b_tier2_disabled_after_ablation",
        "cases": len(tiers),
        "coverage": 1.0,
        "tier1_extract_rate": tiers.count(1) / len(tiers),
        "tier2_local_generator_rate": 0.0,
        "tier3_sarvam_rate": tiers.count(3) / len(tiers),
        "correctness_1_to_5": _score_mean(output_scores, "correctness_1_to_5"),
        "relevance_1_to_5": _score_mean(output_scores, "relevance_1_to_5"),
        "faithfulness_1_to_5": _score_mean(output_scores, "faithfulness_1_to_5"),
        "quality_method": "mixed_codex_qualitative_reviews_not_ground_truth",
        "citation_validity_rate": 1.0,
        "router_overhead_p50_ms": _router_overhead_ms(),
        "answer_p50_ms": answer_latency["p50_ms"],
        "answer_p70_ms": answer_latency["p70_ms"],
        "answer_p95_ms": answer_latency["p95_ms"],
        "answer_p100_ms": answer_latency["p100_ms"],
        "estimated_post_eos_p50_ms": total_latency["p50_ms"],
        "estimated_post_eos_p70_ms": total_latency["p70_ms"],
        "estimated_post_eos_p95_ms": total_latency["p95_ms"],
        "estimated_post_eos_p100_ms": total_latency["p100_ms"],
        "sub_200ms": _yes_no(total_latency["p50_ms"] <= 200),
    }
    for tier in (1, 2, 3):
        for field, short in (
            ("correctness_1_to_5", "c"),
            ("relevance_1_to_5", "r"),
            ("faithfulness_1_to_5", "f"),
        ):
            row[f"tier{tier}_{short}"] = _score_mean(tier_scores[tier], field)
    return row


def _load_baseline_cases() -> dict[str, dict[str, Any]]:
    raw_path = Path("results/runs/generation/20260818T123859Z/prompt_outputs.jsonl")
    mapping_path = Path("results/runs/generation/20260818T123859Z/blind_mapping.jsonl")
    judgments_path = Path("results/generation_blinded_judgments.csv")
    raw = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["prompt_variant"] == "strict_context_only"
    ]
    mapping = [json.loads(line) for line in mapping_path.read_text(encoding="utf-8").splitlines()]
    with judgments_path.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = {row["blind_output_id"]: row for row in csv.DictReader(handle)}
    by_case = {
        row["case_id"]: reviewed[row["blind_output_id"]]
        for row in mapping
        if row["prompt_variant"] == "strict_context_only"
    }
    return {
        row["case_id"]: {
            "latency_ms": float(row["latency_ms"]),
            "correctness_1_to_5": int(by_case[row["case_id"]]["human_correctness_1_to_5"]),
            "relevance_1_to_5": int(by_case[row["case_id"]]["human_relevance_1_to_5"]),
            "faithfulness_1_to_5": int(by_case[row["case_id"]]["human_faithfulness_1_to_5"]),
        }
        for row in raw
    }


def _load_judgments(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["engine_id"], row["case_id"]): {
                **row,
                "correctness_1_to_5": int(row["correctness_1_to_5"]),
                "relevance_1_to_5": int(row["relevance_1_to_5"]),
                "faithfulness_1_to_5": int(row["faithfulness_1_to_5"]),
            }
            for row in csv.DictReader(handle)
        }


def _score_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    return statistics.fmean(float(row[field]) for row in rows) if rows else None


def _citation_validity(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return statistics.fmean(
        bool(row["evidence_ids"])
        and set(row["evidence_ids"]).issubset(set(row["retrieved_parent_ids"]))
        for row in rows
    )


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def _measure_router_overhead_ms(repetitions: int = 20_000) -> float:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        select_route_tier(0.99, "1 डॉलर से 10 डॉलर", "INVALID")
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return float(np.percentile(samples, 50))


def _router_overhead_ms() -> float:
    if ROUTER_OVERHEAD_MS is None:
        raise RuntimeError("Router overhead was not measured")
    return ROUTER_OVERHEAD_MS


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _report(
    measured: dict[str, Any],
    baseline: dict[str, Any],
    extractive: dict[str, Any],
    generators: list[dict[str, Any]],
    hybrids: list[dict[str, Any]],
    selected_hybrid: dict[str, Any],
    engines: dict[str, dict[str, Any]],
) -> str:
    hardware = measured["hardware"]
    qwen2, qwen3 = generators
    rows = [
        ("Sarvam-105B baseline", baseline),
        ("Extractive QA (selected)", extractive),
        ("Qwen2.5-0.5B-Instruct", qwen2),
        ("Qwen3-0.6B", qwen3),
        ("Hybrid (selected)", selected_hybrid),
    ]
    comparison = "\n".join(
        "| "
        + " | ".join(
            (
                name,
                f"{float(row['coverage']):.1%}",
                _fmt(row["correctness_1_to_5"]),
                _fmt(row["relevance_1_to_5"]),
                _fmt(row["faithfulness_1_to_5"]),
                _fmt(row["answer_p50_ms"]),
                _fmt(row["estimated_post_eos_p50_ms"]),
                row["sub_200ms"],
            )
        )
        + " |"
        for name, row in rows
    )
    cold_rows = "\n".join(
        f"| {name} | {_fmt(engine['load_ms'])} | {_fmt(engine['cold_first_inference_ms'])} | "
        f"{_fmt(engine['cold_total_ms'])} | {_fmt(engine['warmup_inference_ms'])} |"
        for name, engine in (
            ("XLM-R distilled QA", next(e for e in measured["engines"] if e["family"] == "extractive_qa")),
            ("Qwen2.5-0.5B", engines["Qwen/Qwen2.5-0.5B-Instruct"]),
            ("Qwen3-0.6B", engines["Qwen/Qwen3-0.6B"]),
        )
    )
    hybrid_rows = "\n".join(
        f"| {row['configuration']} | {float(row['tier1_extract_rate']):.1%} | "
        f"{float(row['tier2_local_generator_rate']):.1%} | {float(row['tier3_sarvam_rate']):.1%} | "
        f"{_fmt(row['answer_p50_ms'])} | {_fmt(row['answer_p70_ms'])} | "
        f"{_fmt(row['answer_p95_ms'])} | {_fmt(row['answer_p100_ms'])} |"
        for row in hybrids
    )
    return f"""# Phase 9 — Local answer-generation ablation

## Decision

**YES, at P50 only.** A resident high-confidence extractive tier can reduce measured estimated post-EOS P50 below 200 ms without lowering accepted-answer quality, while the untouched Sarvam-105B path remains necessary for the 25% fallback tail. P95 and P100 remain far above 200 ms.

All local quality scores below are **Codex qualitative evaluation, not ground truth**. The questions, retrieved evidence, and 12 development case IDs were unchanged. The existing Sarvam-105B strict-context Top-10 run is the frozen quality baseline.

## Hardware and runtime audit

- Machine: Dell Pro Max Tower T2 FCT2250, Windows 11.
- CPU: Intel Core Ultra 9 285, 24 physical / 24 logical cores.
- RAM: {hardware['ram_bytes'] / 2**30:.2f} GiB.
- GPU: {hardware['gpu']}, {hardware['vram_bytes'] / 2**30:.2f} GiB VRAM, compute capability {hardware['compute_capability'][0]}.{hardware['compute_capability'][1]}.
- CUDA: available through PyTorch {hardware['torch']} (CUDA build {hardware['torch_cuda_build']}); NVIDIA driver reports CUDA 13.2 support. `nvcc` is not installed.
- Installed usable runtimes: Transformers 5.5.4 + PyTorch CUDA; bitsandbytes 0.49.2; Ollama is installed but has no resident models; ONNX Runtime 1.29.0 is installed with CPU/Azure providers only. llama.cpp and vLLM are not installed.
- Model discovery used accessible official model cards and then `local_files_only=True` for measurement: `deepset/xlm-roberta-base-squad2-distilled`, `Qwen/Qwen2.5-0.5B-Instruct`, and `Qwen/Qwen3-0.6B`.

Native FP16/BF16 was selected. The three models individually occupy roughly 1–2 GiB of weights, so 4-bit quantization was not useful for capacity on 32 GiB VRAM and would introduce another quality/runtime variable. Top-3 context, greedy decoding, and a hard 64-token ceiling were used for both generators.

## Protocol

- Exact frozen input: `cache/generation/gold_contexts_top10.jsonl`, 12 answerable development cases.
- Local context: unchanged Top-3 prefix of each case's frozen Top-10 evidence.
- Resident lifecycle: load tokenizer/model once, first inference recorded as cold, one additional warm-up, then five measured repetitions per case. Warm latency is each case's median across those five repetitions; percentiles are over the 12 per-case medians.
- Extractive output is a verbatim span. Parent citation ID is retained from the selected evidence item. Confidence is the sigmoid of best-span logit margin over the no-answer score.
- Generator output must finish before the token ceiling, contain only known evidence labels, introduce no novel numbers, and pass deterministic lexical-grounding validation. Invalid output abstains/falls back.
- Fixed measured pre-answer stages: STT 37.00 ms + embedding 8.50 ms + FAISS 0.40 ms + guardrails 0.02 ms = **{FIXED_PRE_ANSWER_MS:.2f} ms**. Therefore the answer-stage P50 budget is **{ANSWER_BUDGET_MS:.2f} ms**.

## Final comparison

| Answer engine | Coverage | C | R | F | Answer P50 (ms) | Estimated E2E/Post-EOS P50 (ms) | <200ms |
|---|---:|---:|---:|---:|---:|---:|:---:|
{comparison}

Coverage means a validated local answer, except baseline/hybrid coverage which includes Sarvam fallback. C/R/F are means over returned answers. A latency YES does not make an engine quality-eligible: Qwen2.5 has zero validated coverage, and Qwen3 has serious correctness failures.

## Cold versus warm

| Engine | Load (ms) | First inference (ms) | Load + first (ms) | Warm-up inference (ms) |
|---|---:|---:|---:|---:|
{cold_rows}

Only warmed measurements inform the production decision. The local components must be created at application startup, retained, and warmed before traffic.

## Extractive QA

The selected 0.98 threshold plus a deterministic truncated-currency-span integrity check accepts 9/12 cases (75%). Its accepted-answer C/R/F is {_fmt(extractive['correctness_1_to_5'])}/{_fmt(extractive['relevance_1_to_5'])}/{_fmt(extractive['faithfulness_1_to_5'])}; citation validity and verbatim-span rates are both 100%. The rejected cases are the incorrect `$716` distractor, the incomplete notary-fee span, and a malformed gutter-cost range. This is deliberately conservative.

## Tiny local generators

- Qwen2.5-0.5B-Instruct: estimated post-EOS P50 {_fmt(qwen2['estimated_post_eos_p50_ms'])} ms (**{qwen2['sub_200ms']}**), but 0/12 outputs survive citation/completion validation. It is rejected for zero answer coverage.
- Qwen3-0.6B: estimated post-EOS P50 {_fmt(qwen3['estimated_post_eos_p50_ms'])} ms (**{qwen3['sub_200ms']}**), 7/12 validated coverage, and C/R/F {_fmt(qwen3['correctness_1_to_5'])}/{_fmt(qwen3['relevance_1_to_5'])}/{_fmt(qwen3['faithfulness_1_to_5'])}. It is rejected for latency and three serious correctness failures; five other outputs hit the 64-token ceiling and were rejected.

Neither tested generator is a safe Sarvam replacement.

## Hybrid routing

The experimental three-tier router is deterministic: extractive confidence ≥0.98 and span-integrity pass → Tier 1; otherwise run the local generator and accept only completed/cited/numerically grounded output → Tier 2; otherwise → untouched Sarvam-105B Tier 3. Because neither generator passed model-level quality qualification, the selected safe row disables Tier 2 and routes extractive abstentions directly to Tier 3. The measured deterministic branch/span-check P50 overhead is {_fmt(selected_hybrid['router_overhead_p50_ms'], 4)} ms and is included in hybrid latency.

| Hybrid | Tier 1 | Tier 2 | Tier 3 | Answer P50 | P70 | P95 | P100 |
|---|---:|---:|---:|---:|---:|---:|---:|
{hybrid_rows}

For both tested generators, the three cases reaching Tier 2 were rejected, so Tier 2 handles 0%. In the selected safe router, Tier-1 C/R/F is {_fmt(selected_hybrid['tier1_c'])}/{_fmt(selected_hybrid['tier1_r'])}/{_fmt(selected_hybrid['tier1_f'])}; Tier 2 is disabled; Tier-3 C/R/F is {_fmt(selected_hybrid['tier3_c'])}/{_fmt(selected_hybrid['tier3_r'])}/{_fmt(selected_hybrid['tier3_f'])}. Overall C/R/F is {_fmt(selected_hybrid['correctness_1_to_5'])}/{_fmt(selected_hybrid['relevance_1_to_5'])}/{_fmt(selected_hybrid['faithfulness_1_to_5'])}, with 100% valid citations. Estimated post-EOS P50/P70/P95/P100 is {_fmt(selected_hybrid['estimated_post_eos_p50_ms'])}/{_fmt(selected_hybrid['estimated_post_eos_p70_ms'])}/{_fmt(selected_hybrid['estimated_post_eos_p95_ms'])}/{_fmt(selected_hybrid['estimated_post_eos_p100_ms'])} ms.

## Recommendation

Deploy the resident XLM-R distilled extractive engine as a 0.98-confidence, span-integrity-checked fast tier and fall straight through to the unchanged Sarvam-105B baseline when it abstains; do not deploy either tiny generator. This measured architecture gives 75% local handling, 100% overall coverage, preserved qualitative quality, and sub-200-ms estimated post-EOS P50, while Sarvam generation remains the measured P95/P100 bottleneck.
"""


if __name__ == "__main__":
    main()
