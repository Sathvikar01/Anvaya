"""Run the gold-query generation ablation without changing frozen STT or retrieval."""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import UTC, datetime
from pathlib import Path

from hh_goa_rag.generation.evaluation import (
    aggregate_configuration_rows,
    aggregate_model_rows,
    prepare_context_cache,
    run_configuration_ablation,
    run_model_ablation,
    write_blinded_judgments,
    write_csv,
    write_pending_ablation_csvs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/eval_dataset.jsonl"))
    parser.add_argument(
        "--context-cache",
        type=Path,
        default=Path("cache/generation/gold_contexts_top10.jsonl"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--phase",
        choices=("model", "topk", "prompt"),
        default="model",
        help="Run models first; Top-K and prompt phases require a human-selected model.",
    )
    parser.add_argument("--model", choices=("sarvam-105b", "sarvam-105b-conversations"))
    parser.add_argument("--top-k", type=int, choices=(1, 3, 5, 10))
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or Path("results/runs/generation") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    contexts = prepare_context_cache(
        args.dataset,
        args.context_cache,
        device=args.device,
    )
    judgment_path = Path("results/generation_blinded_judgments.csv")
    if judgment_path.exists():
        shutil.copy2(judgment_path, run_dir / "previous_blinded_judgments.csv")
    if args.phase == "model":
        raw_path = run_dir / "model_outputs.jsonl"
        observations = run_model_ablation(contexts, raw_path, env_path=args.env_file)
        write_csv(
            Path("results/generation_model_ablation.csv"),
            aggregate_model_rows(observations),
        )
        write_pending_ablation_csvs(
            Path("results/generation_topk_ablation.csv"),
            Path("results/generation_prompt_ablation.csv"),
        )
    elif args.phase == "topk":
        if not args.model:
            parser.error("--phase topk requires the model selected by blinded human review")
        _require_selected(Path("results/generation_model_ablation.csv"), "model", args.model)
        configurations = [(value, "structured_evidence_ids") for value in (1, 3, 5, 10)]
        raw_path = run_dir / "topk_outputs.jsonl"
        observations = run_configuration_ablation(
            contexts,
            raw_path,
            env_path=args.env_file,
            model_name=args.model,
            configurations=configurations,
        )
        write_csv(
            Path("results/generation_topk_ablation.csv"),
            aggregate_configuration_rows(observations, configurations),
        )
    else:
        if not args.model or args.top_k is None:
            parser.error("--phase prompt requires the human-selected --model and --top-k")
        _require_selected(Path("results/generation_model_ablation.csv"), "model", args.model)
        _require_selected(Path("results/generation_topk_ablation.csv"), "top_k", str(args.top_k))
        configurations = [
            (args.top_k, variant)
            for variant in (
                "strict_context_only",
                "context_only_refusal",
                "structured_evidence_ids",
            )
        ]
        raw_path = run_dir / "prompt_outputs.jsonl"
        observations = run_configuration_ablation(
            contexts,
            raw_path,
            env_path=args.env_file,
            model_name=args.model,
            configurations=configurations,
        )
        write_csv(
            Path("results/generation_prompt_ablation.csv"),
            aggregate_configuration_rows(observations, configurations),
        )
    write_blinded_judgments(
        observations,
        contexts,
        judgment_path,
        run_dir / "blind_mapping.jsonl",
        experiment_id=args.phase,
    )
    print(f"Cached contexts: {args.context_cache}")
    print(f"Raw observations: {raw_path}")
    print(f"Completed phase: {args.phase}")
    print("Blinded review sheet: results/generation_blinded_judgments.csv")
    print(f"Blind mapping (do not inspect before review): {run_dir / 'blind_mapping.jsonl'}")
    if args.phase == "model":
        print("Top-K and prompt ablations are gated on completed human model review.")


def _require_selected(path: Path, field: str, requested: str) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row.get("selected", "").lower() == "true"]
    if len(selected) != 1 or selected[0].get(field) != requested:
        raise RuntimeError(f"{requested!r} is not the human-quality winner recorded in {path}")


if __name__ == "__main__":
    main()
