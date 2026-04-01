"""Benchmark resident local answer engines on the frozen 12-case generation set."""

from __future__ import annotations

import argparse
import platform
import statistics
from pathlib import Path
from typing import Any

from hh_goa_rag.generation.local import ExtractiveQAEngine, TinyGeneratorEngine
from hh_goa_rag.generation.sarvam import GenerationContext
from hh_goa_rag.io import read_jsonl, write_json

DEFAULT_EXTRACTIVE_MODEL = "deepset/xlm-roberta-base-squad2-distilled"
DEFAULT_GENERATORS = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", type=Path, default=Path("cache/generation/gold_contexts_top10.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cache/local_generation/observations.json")
    )
    parser.add_argument("--extractive-model", default=DEFAULT_EXTRACTIVE_MODEL)
    parser.add_argument("--generator", action="append", dest="generators")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--warm-repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 10:
        parser.error("--top-k must be in [1, 10]")
    if args.warm_repetitions < 1:
        parser.error("--warm-repetitions must be positive")

    cases = list(read_jsonl(args.contexts))
    if len(cases) != 12:
        raise RuntimeError(f"Expected the frozen 12 answerable cases, found {len(cases)}")
    generators = tuple(args.generators or DEFAULT_GENERATORS)
    report: dict[str, Any] = {
        "benchmark_protocol": {
            "case_count": len(cases),
            "top_k": args.top_k,
            "warm_repetitions_per_case": args.warm_repetitions,
            "warm_latency_value": "per-case median across repetitions",
            "cold_definition": "model load plus first inference, reported separately",
            "production_assumption": "tokenizer/model loaded once and kept resident",
        },
        "hardware": _hardware(),
        "engines": [],
    }

    extractive = ExtractiveQAEngine(
        args.extractive_model,
        device=args.device,
        confidence_threshold=0.0,
    )
    report["engines"].append(
        _benchmark_engine(
            extractive,
            "extractive_qa",
            cases,
            args.top_k,
            args.warm_repetitions,
            answer_kwargs={"threshold": 0.0},
        )
    )
    del extractive
    _release_cuda()

    for model_name in generators:
        generator = TinyGeneratorEngine(model_name, device=args.device, max_new_tokens=64)
        report["engines"].append(
            _benchmark_engine(
                generator,
                "tiny_generator",
                cases,
                args.top_k,
                args.warm_repetitions,
            )
        )
        del generator
        _release_cuda()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"Wrote {args.output}")


def _benchmark_engine(
    engine: Any,
    family: str,
    cases: list[dict[str, Any]],
    top_k: int,
    repetitions: int,
    *,
    answer_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = answer_kwargs or {}
    first_case = cases[0]
    first_contexts = _contexts(first_case, top_k)
    cold = engine.answer(first_case["question"], first_contexts, **kwargs)
    warmup = engine.answer(first_case["question"], first_contexts, **kwargs)
    observations: list[dict[str, Any]] = []
    for case in cases:
        contexts = _contexts(case, top_k)
        results = [
            engine.answer(case["question"], contexts, **kwargs) for _ in range(repetitions)
        ]
        canonical = results[-1]
        if any(result.answer != canonical.answer for result in results):
            raise RuntimeError(f"Non-deterministic answer for {case['case_id']}")
        observations.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "question": case["question"],
                "reference_answer": case["expected"]["reference_answer"],
                "required_claims": case["expected"]["required_claims"],
                "relevant_parent_ids": case["expected"]["relevant_parent_ids"],
                "status": canonical.status,
                "answer": canonical.answer,
                "evidence_ids": list(canonical.evidence_ids),
                "confidence": canonical.confidence,
                "raw_output": canonical.raw_output,
                "validation_error": canonical.validation_error,
                "grounding_overlap": canonical.grounding_overlap,
                "warm_latency_samples_ms": [result.latency_ms for result in results],
                "warm_latency_median_ms": statistics.median(
                    result.latency_ms for result in results
                ),
                "retrieved_parent_ids": [context.parent_id for context in contexts],
            }
        )
    return {
        "family": family,
        "model": engine.model_name,
        "device": str(engine.device),
        "dtype": str(next(engine.model.parameters()).dtype),
        "max_new_tokens": getattr(engine, "max_new_tokens", None),
        "load_ms": engine.load_ms,
        "cold_first_inference_ms": cold.latency_ms,
        "cold_total_ms": engine.load_ms + cold.latency_ms,
        "warmup_inference_ms": warmup.latency_ms,
        "observations": observations,
    }


def _contexts(case: dict[str, Any], top_k: int) -> list[GenerationContext]:
    return [GenerationContext(**row) for row in case["contexts"][:top_k]]


def _hardware() -> dict[str, Any]:
    import psutil
    import torch

    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
    }


def _release_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
