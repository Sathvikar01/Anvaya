"""Build the measured latency autopsy and recommendation report from ablation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from hh_goa_rag.latency import latency_summary

RESULTS = Path("results")
RAW = Path("cache/latency")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    return float(value) if value not in (None, "") else None


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _autopsy() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    startup = _jsonl(RAW / "retrieval_startup.jsonl")
    rows.append(
        _stage_row(
            "startup",
            "embedding_model_and_index_initialization",
            "cold_process",
            [float(startup[0]["startup_ms"])],
            "ABLATION",
        )
    )
    retrieval = _jsonl(RAW / "retrieval_observations.jsonl")
    first = retrieval[:1]
    warm = retrieval[1:]
    for configuration, observations in (("cold", first), ("warm", warm)):
        for field, stage in (
            ("embedding_ms", "query_embedding"),
            ("search_ms", "faiss_search"),
            ("evidence_guardrail_ms", "evidence_guardrail"),
        ):
            rows.append(
                _stage_row(
                    "retrieval",
                    stage,
                    configuration,
                    [float(row[field]) for row in observations],
                    "ABLATION",
                )
            )
    stt = _csv(RESULTS / "stt_latency_ablation.csv")
    for observation in stt:
        for prefix, stage in (
            ("connection", "connection_setup"),
            ("audio_send", "audio_upload_or_send"),
            ("vad_detection", "vad_endpoint_detection"),
            ("server_processing", "server_processing"),
            ("eos_to_final", "eos_to_final_transcript"),
        ):
            rows.append(
                {
                    "measurement_label": observation["measurement_label"],
                    "component": "stt",
                    "stage": stage,
                    "configuration": observation["configuration"],
                    "samples": observation.get(f"{prefix}_n", ""),
                    "p50_ms": observation.get(f"{prefix}_p50_ms", ""),
                    "p70_ms": observation.get(f"{prefix}_p70_ms", ""),
                    "p95_ms": observation.get(f"{prefix}_p95_ms", ""),
                    "p100_ms": observation.get(f"{prefix}_p100_ms", ""),
                    "mean_ms": observation.get(f"{prefix}_mean_ms", ""),
                    "notes": "speech duration is reported separately and excluded",
                }
            )
    generation = _csv(RESULTS / "generation_latency_ablation.csv")
    for observation in generation:
        for prefix, stage in (
            ("serialization", "prompt_serialization"),
            ("client_setup", "client_setup"),
            ("connection", "network_connection_setup"),
            ("server_wait", "server_wait_to_headers"),
            ("ttft", "time_to_first_content_token"),
            ("token_generation", "content_token_generation"),
            ("parsing", "output_parsing"),
            ("grounding", "grounding_validation"),
            ("full", "full_completion"),
        ):
            rows.append(
                {
                    "measurement_label": observation["measurement_label"],
                    "component": "generation",
                    "stage": stage,
                    "configuration": observation["configuration"],
                    "samples": observation.get("successes", ""),
                    "p50_ms": observation.get(f"{prefix}_p50_ms", ""),
                    "p70_ms": observation.get(f"{prefix}_p70_ms", ""),
                    "p95_ms": observation.get(f"{prefix}_p95_ms", ""),
                    "p100_ms": observation.get(f"{prefix}_p100_ms", ""),
                    "mean_ms": observation.get(f"{prefix}_mean_ms", ""),
                    "notes": "new development optimization sample",
                }
            )
    with (RESULTS / "latency_autopsy.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _stage_row(
    component: str,
    stage: str,
    configuration: str,
    values: list[float],
    label: str,
) -> dict[str, Any]:
    summary = latency_summary(values)
    return {
        "measurement_label": label,
        "component": component,
        "stage": stage,
        "configuration": configuration,
        "samples": summary["n"],
        "p50_ms": summary["p50_ms"],
        "p70_ms": summary["p70_ms"],
        "p95_ms": summary["p95_ms"],
        "p100_ms": summary["p100_ms"],
        "mean_ms": summary["mean_ms"],
        "notes": "",
    }


def _report() -> None:
    stt = _csv(RESULTS / "stt_latency_ablation.csv")
    generation = _csv(RESULTS / "generation_latency_ablation.csv")
    fast = _csv(RESULTS / "fast_path_ablation.csv")
    prompt = _csv(RESULTS / "generation_prompt_ablation.csv")
    historical = next(row for row in prompt if row["prompt_variant"] == "strict_context_only")
    retrieval = _jsonl(RAW / "retrieval_observations.jsonl")[1:]
    embed = latency_summary(float(row["embedding_ms"]) for row in retrieval)["p50_ms"]
    search = latency_summary(float(row["search_ms"]) for row in retrieval)["p50_ms"]
    guard = latency_summary(float(row["evidence_guardrail_ms"]) for row in retrieval)["p50_ms"]
    live_ws = next(
        row for row in stt if row["configuration"] == "websocket_live_stream_preopened"
    )
    stt_eos = _number(live_ws, "eos_to_final_p50_ms")
    baseline_generation = float(historical["latency_p50_ms"])
    waterfall_total = sum(
        value for value in (stt_eos, embed, search, guard, baseline_generation) if value
    )
    contributions = [
        ("Generation", baseline_generation),
        ("STT EOS→final", stt_eos or 0.0),
        ("Embedding", embed or 0.0),
        ("FAISS", search or 0.0),
        ("Evidence guardrail", guard or 0.0),
    ]
    lines = [
        "# Latency optimization summary",
        "",
        "## Measurement scope",
        "",
        "All new numbers are **ABLATION** measurements. They are not FORMAL VOICE E2E results. "
        "STT uses one pre-existing 12.696 s real-human smoke clip with three repeats; generation "
        "uses the first four development answerable cases. Historical reviewed generation "
        "quality remains separate.",
        "",
        "The frozen baseline was not modified. No sealed data was used and no synthetic audio or "
        "synthetic metrics were used.",
        "",
        "## Live capability audit",
        "",
        "- Account probe: `sarvam-105b` returned HTTP 200; `sarvam-30b` returned HTTP 400 "
        "`invalid_request_error`. No other provider API key is configured.",
        "- The frozen generator already sends `reasoning_effort=None`; the requested existing vs "
        "None comparison is therefore an identity, not a separate architecture.",
        "- Sarvam documents the GA WebSocket VAD frame controls and notes that the newer realtime "
        "beta supersedes it for true partial transcripts: "
        "https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/streaming-api",
        "- Current model catalog: https://docs.sarvam.ai/api/getting-started/models",
        "",
        "## Quality-preserving P50 waterfall",
        "",
        "This is a component-sum diagnostic, not a synchronized E2E percentile:",
        "",
        "```mermaid",
        "flowchart LR",
        f'  A["STT EOS→final<br/>{_fmt(stt_eos)} ms"] --> B["BGE-M3<br/>{_fmt(embed, 2)} ms"]',
        f'  B --> C["FAISS<br/>{_fmt(search, 3)} ms"]',
        f'  C --> D["Evidence gate<br/>{_fmt(guard, 3)} ms"]',
        f'  D --> E["Sarvam-105B complete<br/>{_fmt(baseline_generation)} ms"]',
        f'  E --> F["Component sum<br/>{_fmt(waterfall_total)} ms"]',
        "```",
        "",
        "| Bottleneck rank | Stage | P50 (ms) | Share of component sum |",
        "| ---: | --- | ---: | ---: |",
    ]
    for rank, (stage, value) in enumerate(sorted(contributions, key=lambda item: -item[1]), 1):
        lines.append(f"| {rank} | {stage} | {value:.3f} | {value / waterfall_total:.1%} |")
    lines.extend(["", "## STT ablation", ""])
    lines.extend(
        [
            "| Configuration | n/success | Connection P50 | EOS→final P50 | Wall P50 | Note |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in stt:
        note = "latency-only; quality reference unavailable"
        if row["configuration"] == "websocket_live_stream_persistent":
            note = "connection timeout; no winner claim"
        lines.append(
            f"| {row['configuration']} | {row['samples']}/{row['successes']} | "
            f"{_fmt(_number(row, 'connection_p50_ms'))} | "
            f"{_fmt(_number(row, 'eos_to_final_p50_ms'))} | "
            f"{_fmt(_number(row, 'wall_clock_p50_ms'))} | {note} |"
        )
    lines.extend(
        [
            "",
            "Speaking duration is deliberately excluded from avoidable compute. For paced "
            "streaming, wall time includes the 12.696 s utterance; EOS→final is the "
            "decision metric.",
            "",
            "## Generation ablation",
            "",
            "| Configuration | n | Fail | TTFT P50 | Full P50 | Tokens in/out | Quality proxy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in generation:
        quality = (
            f"F1 {_fmt(_number(row, 'mean_reference_token_f1'), 3)}, "
            f"ground {_percent(_number(row, 'grounding_validity_rate'))}"
        )
        lines.append(
            f"| {row['configuration']} | {row['cases']} | {row['failures']} | "
            f"{_fmt(_number(row, 'ttft_p50_ms'))} | {_fmt(_number(row, 'full_p50_ms'))} | "
            f"{_fmt(_number(row, 'mean_prompt_tokens'), 0)}/"
            f"{_fmt(_number(row, 'mean_output_tokens'), 0)} | {quality} |"
        )
    lines.extend(
        [
            "",
            "Historical frozen strict-context run (12 cases, reviewed C/R/F): P50/P70/P95/P100 "
            f"= {float(historical['latency_p50_ms']):.0f}/"
            f"{float(historical['latency_p70_ms']):.0f}/"
            f"{float(historical['latency_p95_ms']):.0f}/"
            f"{float(historical['latency_p100_ms']):.0f} ms; C/R/F "
            f"= {float(historical['human_correctness']):.2f}/"
            f"{float(historical['human_relevance']):.2f}/"
            f"{float(historical['human_faithfulness']):.2f}.",
            "",
            "## Fast-path ablation",
            "",
            "| Configuration | Completion | Extractive | Fallback | Answer P50 | "
            "Citation valid | Grounded |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fast:
        lines.append(
            f"| {row['configuration']} | {_percent(_number(row, 'completion_rate'))} | "
            f"{_percent(_number(row, 'extractive_route_rate'))} | "
            f"{_percent(_number(row, 'generator_fallback_rate'))} | "
            f"{_fmt(_number(row, 'answer_p50_ms'), 3)} | "
            f"{_percent(_number(row, 'citation_validity_rate'))} | "
            f"{_percent(_number(row, 'grounding_validity_rate'))} |"
        )
    lines.extend(
        _optimization_table(
            stt,
            generation,
            fast,
            embed or 0,
            search or 0,
            guard or 0,
            baseline_generation,
        )
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "### A — Best quality",
            "",
            "Keep frozen Top-10 + strict-context Sarvam-105B with reasoning disabled. Pre-open the "
            "STT WebSocket and stream during speech; pre-initialize BGE-M3/index and use HTTP "
            "pooling. This preserves the only fully reviewed quality result. Generation remains "
            "the bottleneck.",
            "",
            "### B — Best latency/quality tradeoff",
            "",
            "Use pre-opened live STT plus a conservative high-confidence extractive router, "
            "falling "
            "back to the frozen generator. At the 0.80 experimental threshold, only 1/12 cases "
            "routed extractively; that routed citation was valid and relevant. Coverage is too low "
            "to move the overall P50 materially, but the path is safe enough for further "
            "evaluation.",
            "",
            "### C — Fastest possible",
            "",
            "For high-confidence cases only, return the selected verbatim sentence with its parent "
            "citation. Its post-EOS component sum is below 200 ms, but it completed only 1/12 "
            "cases. "
            "It is not a complete replacement and must abstain/fallback for every uncertain query.",
            "",
            "## Safe parallelization and prewarming",
            "",
            "- Load BGE-M3 and the FAISS index at process startup; the measured cold "
            "initialization "
            "cost is startup-only, not a per-query dependency.",
            "- Open the STT WebSocket before speech and stream audio while the user speaks. "
            "This is "
            "the only large overlap with user time supported by the dependency graph.",
            "- Prewarm HTTP/TLS connections, but do not overlap embedding with STT finalization, "
            "retrieval with embedding, or generation with evidence gating; each consumes the prior "
            "stage's output.",
            "- No lightweight local QA checkpoint was already configured. The measured fast path "
            "therefore uses deterministic verbatim sentence selection and adds no unmeasured "
            "model.",
            "",
            "## <200 ms conclusion",
            "",
            "**The quality-preserving complete Voice-RAG architecture did not meet <200 ms.** The "
            f"best diagnostic component sum is ~{waterfall_total:.0f} ms after EOS, with "
            f"Sarvam-105B generation contributing {baseline_generation / waterfall_total:.1%}. "
            "Historical Sarvam-105B TTFT alone is above 200 ms at P50, so connection/context "
            "tweaks "
            "cannot make the normal generative path compliant. The only sub-200 ms observation is "
            "the narrow extractive path, which lacks full coverage and is not a measured complete "
            "E2E percentile.",
            "",
            "## Limitations",
            "",
            "- STT sample size is one real-human smoke clip × three repeats, without a trusted "
            "reference transcript; VAD transcript-quality differences cannot be approved.",
            "- Generation optimization uses four development cases; new quality values are "
            "deterministic proxies, not C/R/F human scores.",
            "- Component-sum totals add independently measured P50s and are not formal E2E "
            "percentiles.",
            "- Persistent WebSocket reuse failed during this run and remains unvalidated.",
            "- No formal claim is made until the 24 real recordings are available.",
        ]
    )
    (RESULTS / "latency_optimization_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _optimization_table(
    stt: list[dict[str, str]],
    generation: list[dict[str, str]],
    fast: list[dict[str, str]],
    embed: float,
    search: float,
    guard: float,
    baseline_generation: float,
) -> list[str]:
    live = next(row for row in stt if row["configuration"] == "websocket_live_stream_preopened")
    stt_p50 = _number(live, "eos_to_final_p50_ms") or 0.0
    lines = [
        "",
        "## Optimization table",
        "",
        "Totals are P50 component sums, not synchronized E2E percentiles.",
        "",
        "| Configuration | STT P50 | Embed | Search | Generation/answer P50 | Total P50 | "
        "Quality | <200 ms? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in generation:
        answer = _number(row, "full_p50_ms")
        total = stt_p50 + embed + search + guard + answer if answer is not None else None
        quality = f"proxy; failures {row['failures']}/{row['cases']}"
        lines.append(
            f"| {row['configuration']} | {stt_p50:.1f} | {embed:.2f} | {search:.3f} | "
            f"{_fmt(answer)} | {_fmt(total)} | {quality} | "
            f"{'yes*' if total is not None and total < 200 else 'no'} |"
        )
    for row in stt:
        configuration = row["configuration"]
        observed_stt = _number(row, "eos_to_final_p50_ms")
        total = (
            observed_stt + embed + search + guard + baseline_generation
            if observed_stt is not None
            else None
        )
        quality = "STT quality unscored" if observed_stt is not None else "measured failure"
        lines.append(
            f"| STT {configuration} + frozen generator | {_fmt(observed_stt)} | "
            f"{embed:.2f} | {search:.3f} | {baseline_generation:.1f} | {_fmt(total)} | "
            f"{quality} | no |"
        )
    for row in fast[1:]:
        answer = _number(row, "answer_p50_ms")
        total = stt_p50 + embed + search + guard + answer if answer is not None else None
        quality = f"completion {_percent(_number(row, 'completion_rate'))}"
        lines.append(
            f"| {row['configuration']} | {stt_p50:.1f} | {embed:.2f} | {search:.3f} | "
            f"{_fmt(answer, 3)} | {_fmt(total)} | {quality} | "
            f"{'yes*' if total is not None and total < 200 else 'no'} |"
        )
    lines.append(
        "\n`yes*` means only the independently summed post-EOS component medians are below 200 ms; "
        "it is not a formal complete-pipeline compliance result."
    )
    return lines


def main() -> None:
    _autopsy()
    _report()


if __name__ == "__main__":
    main()
