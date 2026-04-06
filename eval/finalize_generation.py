"""Unblind completed human review and finalize one generation ablation phase."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from hh_goa_rag.generation.evaluation import (
    aggregate_configuration_rows,
    aggregate_model_rows,
    write_csv,
)
from hh_goa_rag.generation.review import (
    add_human_quality,
    load_judgments,
    load_mapping,
    select_quality_winner,
)
from hh_goa_rag.io import read_jsonl

OUTPUTS = {
    "model": Path("results/generation_model_ablation.csv"),
    "topk": Path("results/generation_topk_ablation.csv"),
    "prompt": Path("results/generation_prompt_ablation.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=tuple(OUTPUTS))
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("results/generation_blinded_judgments.csv"),
    )
    args = parser.parse_args()

    observations = list(read_jsonl(args.raw))
    if not observations:
        parser.error(f"no raw observations found in {args.raw}")
    if args.phase == "model":
        aggregates = aggregate_model_rows(observations)
        group_field = "model"
    else:
        configurations = sorted(
            {(int(row["top_k"]), str(row["prompt_variant"])) for row in observations}
        )
        aggregates = aggregate_configuration_rows(observations, configurations)
        group_field = "top_k" if args.phase == "topk" else "prompt_variant"

    enriched = add_human_quality(
        aggregates,
        load_judgments(args.judgments),
        load_mapping(args.mapping),
        group_field=group_field,
    )
    selected, winner = select_quality_winner(enriched, phase=args.phase)
    write_csv(OUTPUTS[args.phase], selected)
    if args.phase == "prompt":
        write_recommendation()

    identity = winner[group_field]
    print(f"Selected {args.phase}: {identity}")
    print(
        "Qualitative means: "
        f"correctness={winner['human_correctness']:.3f}, "
        f"relevance={winner['human_relevance']:.3f}, "
        f"faithfulness={winner['human_faithfulness']:.3f}"
    )
    print(f"P50/P95 latency: {winner['latency_p50_ms']:.1f}/{winner['latency_p95_ms']:.1f} ms")


def write_recommendation() -> None:
    model = _selected_row(OUTPUTS["model"])
    topk = _selected_row(OUTPUTS["topk"])
    prompt = _selected_row(OUTPUTS["prompt"])
    model_rows = _all_rows(OUTPUTS["model"])
    topk_rows = _all_rows(OUTPUTS["topk"])
    prompt_rows = _all_rows(OUTPUTS["prompt"])
    text = f"""# Generation recommendation

## Final selection

**{model['model']} → Top-{topk['top_k']} → {prompt['prompt_variant']} → {_quality(prompt)} →
{_latency(prompt)}**

At the user's direction, Codex performed the blinded qualitative review; these are not human scores.
Model/configuration identity stayed sealed until every correctness, relevance, and faithfulness
score was persisted. Automated diagnostic heuristics were not substituted for these qualitative
scores. A faithfulness score of 1 was defined as a serious grounding failure and made a
configuration ineligible. Latency was used only as the model tie-breaker.

Top-K was evaluated only with `{model['model']}` at K = 1, 3, 5, and 10. The selected value is the
smallest K achieving the highest observed mean quality without a serious grounding failure.
The prompt comparison then held that model and Top-K fixed.

## Ablation summary

### Model

| Model | Correctness | Relevance | Faithfulness | Mean quality | P50 / P95 latency | Selected |
|---|---:|---:|---:|---:|---:|---:|
{_model_table(model_rows)}

### Top-K

| K | Correct. | Relevance | Faithful. | Mean quality | Citation valid. | P50 / P95 | Selected |
|---:|---:|---:|---:|---:|---:|---:|---:|
{_topk_table(topk_rows)}

### Prompt

| Prompt | Correct. | Relevance | Faithful. | Mean quality | Tokens | P50 / P95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|
{_prompt_table(prompt_rows)}

## Selected quality and performance

- Mean correctness: {float(prompt['human_correctness']):.3f}/5
- Mean relevance: {float(prompt['human_relevance']):.3f}/5
- Mean faithfulness: {float(prompt['human_faithfulness']):.3f}/5
- Serious grounding failures: {prompt['serious_grounding_failures']}
- Grounded citation validity: {_percent(prompt.get('grounded_citation_validity_rate'))}
- Appropriate answer/refusal behavior: {_percent(prompt.get('appropriate_context_behavior_rate'))}
- Generation latency P50/P70/P95/P100: {_latency(prompt)}
- Mean prompt/context tokens: {float(prompt['mean_prompt_tokens']):.1f}

All 12 Top-10 cases contained relevant evidence, so the prompt ablation had no missing-context case
on which to distinguish refusal behavior. All variants answered all 12 cases. In the Top-K ablation,
K=1 lacked a relevant parent for 4/12 cases and refused 0/4, yielding 66.7% appropriate context
behavior; this limitation is recorded rather than treated as a quality score.

## Latency target

The current Sarvam generation stack **does not satisfy the <200 ms complete-pipeline target**.
Its measured generation latency alone is P50 {float(prompt['latency_p50_ms']):.1f} ms and P100
{float(prompt['latency_p100_ms']):.1f} ms. Since the complete pipeline also includes STT and
retrieval, it cannot be faster than this measured generation stage under the evaluated stack.

## Scope

The frozen STT and retriever were not modified. The experiment reused the cached gold-query
retrieval snapshot. No guardrails, UI, or deployment work is included.
"""
    OUTPUTS["prompt"].with_name("generation_recommendation.md").write_text(
        text, encoding="utf-8"
    )


def _selected_row(path: Path) -> dict[str, str]:
    rows = _all_rows(path)
    selected = [row for row in rows if row.get("selected", "").lower() == "true"]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one selected row in {path}")
    return selected[0]


def _all_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _quality(row: dict[str, Any]) -> str:
    return (
        f"C/R/F {float(row['human_correctness']):.2f}/"
        f"{float(row['human_relevance']):.2f}/{float(row['human_faithfulness']):.2f}"
    )


def _latency(row: dict[str, Any]) -> str:
    return (
        f"{float(row['latency_p50_ms']):.0f}/{float(row['latency_p70_ms']):.0f}/"
        f"{float(row['latency_p95_ms']):.0f}/{float(row['latency_p100_ms']):.0f} ms"
    )


def _percent(value: Any) -> str:
    return "n/a" if value in (None, "") else f"{100 * float(value):.1f}%"


def _model_table(rows: list[dict[str, str]]) -> str:
    return "\n".join(
        f"| {row['model']} | {float(row['human_correctness']):.3f} | "
        f"{float(row['human_relevance']):.3f} | {float(row['human_faithfulness']):.3f} | "
        f"{float(row['human_mean_quality']):.3f} | {float(row['latency_p50_ms']):.0f} / "
        f"{float(row['latency_p95_ms']):.0f} ms | {row['selected']} |"
        for row in rows
    )


def _topk_table(rows: list[dict[str, str]]) -> str:
    return "\n".join(
        f"| {row['top_k']} | {float(row['human_correctness']):.3f} | "
        f"{float(row['human_relevance']):.3f} | {float(row['human_faithfulness']):.3f} | "
        f"{float(row['human_mean_quality']):.3f} | "
        f"{_percent(row['grounded_citation_validity_rate'])} | "
        f"{float(row['latency_p50_ms']):.0f} / {float(row['latency_p95_ms']):.0f} ms | "
        f"{row['selected']} |"
        for row in rows
    )


def _prompt_table(rows: list[dict[str, str]]) -> str:
    return "\n".join(
        f"| {row['prompt_variant']} | {float(row['human_correctness']):.3f} | "
        f"{float(row['human_relevance']):.3f} | {float(row['human_faithfulness']):.3f} | "
        f"{float(row['human_mean_quality']):.3f} | {float(row['mean_prompt_tokens']):.1f} | "
        f"{float(row['latency_p50_ms']):.0f} / {float(row['latency_p95_ms']):.0f} ms | "
        f"{row['selected']} |"
        for row in rows
    )


if __name__ == "__main__":
    main()
