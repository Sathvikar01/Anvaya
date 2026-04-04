"""Evaluate Sarvam REST/streaming STT and its impact on the frozen retriever.

The original ``--predictions`` scoring path remains supported. ``--manifest`` runs real WAV files
through the only configured provider, Sarvam Saaras v3, and produces the phase result artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any

from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel
from hh_goa_rag.retriever import ParentFaissRetriever
from hh_goa_rag.stt.sarvam import SARVAM_MODE, SARVAM_MODEL, SarvamSTT, STTResult

try:
    from .metrics import (
        FROZEN_RETRIEVAL_STACK,
        assert_frozen_retrieval_config,
        evaluate_retrieval_records,
        evaluate_stt_records,
        latency_metrics,
        load_dataset,
        normalize_text,
        pair_cases_and_predictions,
        print_summary,
        read_jsonl,
        word_error_counts,
        write_evaluation_run,
    )
except ImportError:
    from metrics import (  # type: ignore[no-redef]
        FROZEN_RETRIEVAL_STACK,
        assert_frozen_retrieval_config,
        evaluate_retrieval_records,
        evaluate_stt_records,
        latency_metrics,
        load_dataset,
        normalize_text,
        pair_cases_and_predictions,
        print_summary,
        read_jsonl,
        word_error_counts,
        write_evaluation_run,
    )

STT_CSV_FIELDS = [
    "transport",
    "sample_id",
    "case_id",
    "category",
    "variant",
    "speaker_id",
    "audio_path",
    "reference_text",
    "transcript",
    "status",
    "error_code",
    "error_message",
    "attempts",
    "wer",
    "word_errors",
    "reference_words",
    "exact_match",
    "normalized_match",
    "latency_ms",
    "connection_latency_ms",
    "time_to_first_partial_ms",
    "time_to_first_transcript_ms",
    "end_of_speech_to_final_ms",
    "audio_duration_ms",
    "language_code",
    "request_id",
]
RETRIEVAL_CSV_FIELDS = [
    "transport",
    "metric",
    "gold_text",
    "stt_transcript",
    "absolute_degradation",
    "evaluated_queries",
]
QUALITY_METRICS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _ready_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ready = [row for row in rows if row.get("status") == "ready"]
    missing = [row["sample_id"] for row in ready if not Path(str(row["audio_path"])).is_file()]
    if missing:
        raise ValueError(f"Ready manifest rows have missing audio: {missing}")
    if not ready:
        raise ValueError(
            "No real recordings are ready. Run eval/record_audio.py; pending rows are not scored."
        )
    sample_ids = [str(row["sample_id"]) for row in ready]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id values in STT manifest")
    return ready


async def _run_live_stt(
    service: SarvamSTT,
    manifest: list[dict[str, Any]],
    transports: list[str],
) -> dict[str, list[STTResult]]:
    observations: dict[str, list[STTResult]] = {transport: [] for transport in transports}
    for transport in transports:
        for index, row in enumerate(manifest, start=1):
            print(f"[{transport} {index}/{len(manifest)}] {row['sample_id']}")
            if transport == "rest":
                result = await asyncio.to_thread(
                    service.transcribe_rest,
                    row["audio_path"],
                    language_code=row.get("language_code", "hi-IN"),
                )
            else:
                result = await service.transcribe_streaming(
                    row["audio_path"],
                    language_code=row.get("language_code", "hi-IN"),
                    speech_end_offset_ms=row.get("speech_end_ms"),
                    pace_audio=True,
                )
            observations[transport].append(result)
    return observations


def _stt_rows_and_summaries(
    manifest: list[dict[str, Any]],
    observations: dict[str, list[STTResult]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for transport, results in observations.items():
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        exact_matches = 0
        normalized_matches = 0
        for sample, result in zip(manifest, results, strict=True):
            reference = str(sample["reference_text"])
            failed = result.status != "ok" or not result.transcript.strip()
            counts = (
                {"errors": None, "reference_words": None, "wer": None}
                if failed
                else word_error_counts(reference, result.transcript)
            )
            exact = not failed and result.transcript.strip() == reference.strip()
            normalized = not failed and normalize_text(result.transcript) == normalize_text(
                reference
            )
            exact_matches += int(exact)
            normalized_matches += int(normalized)
            output_rows.append(
                {
                    "transport": transport,
                    "sample_id": sample["sample_id"],
                    "case_id": sample["case_id"],
                    "category": sample["category"],
                    "variant": sample.get("variant"),
                    "speaker_id": sample.get("speaker_id"),
                    "audio_path": sample["audio_path"],
                    "reference_text": reference,
                    "transcript": result.transcript,
                    "status": result.status,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                    "attempts": result.attempts,
                    "wer": counts["wer"],
                    "word_errors": counts["errors"],
                    "reference_words": counts["reference_words"],
                    "exact_match": exact,
                    "normalized_match": normalized,
                    "latency_ms": result.latency_ms,
                    "connection_latency_ms": result.connection_latency_ms,
                    "time_to_first_partial_ms": result.time_to_first_partial_ms,
                    "time_to_first_transcript_ms": result.time_to_first_transcript_ms,
                    "end_of_speech_to_final_ms": result.end_of_speech_to_final_ms,
                    "audio_duration_ms": result.audio_duration_ms,
                    "language_code": result.language_code,
                    "request_id": result.request_id,
                }
            )
            pairs.append(
                (
                    {
                        "case_id": sample["sample_id"],
                        "category": sample["category"],
                        "stt_reference": reference,
                    },
                    {
                        "case_id": sample["sample_id"],
                        "status": result.status,
                        "transcript": result.transcript,
                        "latency_ms": result.latency_ms,
                    },
                )
            )
        summary, _ = evaluate_stt_records(pairs)
        summary.update(
            {
                "exact_match_rate": exact_matches / len(manifest),
                "normalized_match_rate": normalized_matches / len(manifest),
                "connection_latency": _optional_latency(results, "connection_latency_ms"),
                "time_to_first_partial": _optional_latency(
                    results, "time_to_first_partial_ms"
                ),
                "time_to_first_transcript": _optional_latency(
                    results, "time_to_first_transcript_ms"
                ),
                "end_of_speech_to_final": _optional_latency(
                    results, "end_of_speech_to_final_ms"
                ),
            }
        )
        summaries[transport] = summary
    return output_rows, summaries


def _optional_latency(results: list[STTResult], field: str) -> dict[str, Any]:
    values = [float(value) for result in results if (value := getattr(result, field)) is not None]
    return latency_metrics(values)


def _load_frozen_dev_retriever(device: str) -> tuple[ParentFaissRetriever, EmbeddingModel]:
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    final_config_path = Path("results/final_retriever_config.json")
    assert_frozen_retrieval_config(final_config_path)
    final_config = json.loads(final_config_path.read_text(encoding="utf-8"))
    index_winner = json.loads(Path("results/index_winner.json").read_text(encoding="utf-8"))
    chunking_winner = json.loads(
        Path("results/chunking_winner.json").read_text(encoding="utf-8")
    )
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
    if observed != FROZEN_RETRIEVAL_STACK:
        raise RuntimeError(f"Development retriever is not the frozen stack: {observed}")
    index_path = Path(index_winner["metrics"]["index_artifact"])
    chunk_path = Path(chunking_winner["metrics"]["chunk_artifact"])
    retriever = ParentFaissRetriever.load(index_path, chunk_path, top_k=10, oversample=20)
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    model_name = str(final_config["model"])
    model = EmbeddingModel(
        MODEL_SPECS[model_name],
        Path(final_config["model_cache_path"]),
        device=device,
        max_sequence_length=512,
        dtype=dtype,
    )
    return retriever, model


def _rank_texts(
    retriever: ParentFaissRetriever,
    model: EmbeddingModel,
    texts: list[str],
) -> list[list[str]]:
    if not texts:
        return []
    vectors, _ = model.encode_queries(texts)
    return [[item.parent_id for item in retriever.retrieve(vector)] for vector in vectors]


def _retrieval_impact(
    manifest: list[dict[str, Any]],
    observations: dict[str, list[STTResult]],
    dataset_cases: list[dict[str, Any]],
    *,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    case_map = {str(case["case_id"]): case for case in dataset_cases}
    positions = [
        index
        for index, row in enumerate(manifest)
        if case_map[str(row["case_id"])]["expected"].get("relevant_parent_ids")
    ]
    if not positions:
        return [], {}
    retriever, model = _load_frozen_dev_retriever(device)
    try:
        gold_texts = [str(manifest[index]["reference_text"]) for index in positions]
        gold_rankings = _rank_texts(retriever, model, gold_texts)
        rows: list[dict[str, Any]] = []
        summaries: dict[str, dict[str, Any]] = {}
        cases = [
            {
                **case_map[str(manifest[index]["case_id"])],
                "case_id": manifest[index]["sample_id"],
            }
            for index in positions
        ]
        gold_predictions = [
            {
                "case_id": case["case_id"],
                "status": "ok",
                "retrieved_parent_ids": ranking,
                "latency_ms": 0.0,
            }
            for case, ranking in zip(cases, gold_rankings, strict=True)
        ]
        gold_summary, _ = evaluate_retrieval_records(
            list(zip(cases, gold_predictions, strict=True))
        )
        for transport, results in observations.items():
            successful_positions = [
                index
                for index in positions
                if results[index].status == "ok" and results[index].transcript
            ]
            transcript_rankings = _rank_texts(
                retriever,
                model,
                [results[index].transcript for index in successful_positions],
            )
            ranking_map = dict(zip(successful_positions, transcript_rankings, strict=True))
            predictions = [
                {
                    "case_id": manifest[index]["sample_id"],
                    "status": results[index].status,
                    "retrieved_parent_ids": ranking_map.get(index, []),
                    "latency_ms": 0.0,
                }
                for index in positions
            ]
            transcript_summary, _ = evaluate_retrieval_records(
                list(zip(cases, predictions, strict=True))
            )
            summaries[transport] = {"gold": gold_summary, "transcript": transcript_summary}
            for metric in QUALITY_METRICS:
                gold_value = float(gold_summary[metric])
                transcript_value = float(transcript_summary[metric])
                rows.append(
                    {
                        "transport": transport,
                        "metric": metric,
                        "gold_text": gold_value,
                        "stt_transcript": transcript_value,
                        "absolute_degradation": gold_value - transcript_value,
                        "evaluated_queries": len(positions),
                    }
                )
        return rows, summaries
    finally:
        model.close()


def _recommendation(stt_summaries: dict[str, dict[str, Any]]) -> tuple[str, str]:
    if not stt_summaries:
        return "unmeasured", "No real audio has been evaluated."
    if "rest" not in stt_summaries or "streaming" not in stt_summaries:
        only = next(iter(stt_summaries))
        return "incomplete", f"Only {only} was measured; both integration modes are required."
    rest = stt_summaries["rest"]
    stream = stt_summaries["streaming"]
    if rest.get("wer_micro") is None or stream.get("wer_micro") is None:
        return "incomplete", "At least one integration mode produced no successful transcripts."
    quality_eligible = (
        float(stream["wer_micro"]) <= float(rest["wer_micro"]) + 0.02
        and float(stream["failure_rate"]) <= float(rest["failure_rate"]) + 0.01
    )
    rest_p95 = float(rest["latency"].get("p95_ms", float("inf")))
    stream_p95 = float(stream["end_of_speech_to_final"].get("p95_ms", float("inf")))
    if quality_eligible and stream_p95 < rest_p95:
        return (
            "streaming",
            "Streaming preserves REST quality within tolerance and has lower P95 "
            "post-end-of-speech latency.",
        )
    return (
        "rest",
        "Streaming did not meet the fixed quality/failure tolerance and lower P95 "
        "post-end-of-speech latency condition.",
    )


def _write_recommendation(
    path: Path,
    stt_summaries: dict[str, dict[str, Any]],
    retrieval_summaries: dict[str, dict[str, Any]],
    sample_count: int,
) -> None:
    decision, reason = _recommendation(stt_summaries)
    lines = [
        "# Sarvam STT evaluation and recommendation",
        "",
        "## Fixed configuration",
        "",
        "- Provider: Sarvam AI only",
        f"- Model: `{SARVAM_MODEL}`",
        f"- Mode: `{SARVAM_MODE}`",
        "- Audio: real human speech, mono PCM16 WAV, 16 kHz",
        "- Streaming: 64 ms WAV chunks, high VAD sensitivity, VAD events and flush enabled",
        "- REST: synchronous `/speech-to-text`, files limited to 30 seconds",
        "",
        "## Measurement status",
        "",
        f"- Real audio samples evaluated: {sample_count}",
        f"- Decision: **{decision}**",
        f"- Reason: {reason}",
        "- The complete Voice-RAG pipeline has not been measured; the <200 ms end-to-end target "
        "is not claimed.",
        "",
        "## STT quality and latency",
        "",
    ]
    if not stt_summaries:
        lines.append("No measurements. Record the pending manifest samples before evaluation.")
    else:
        lines.extend(
            [
                "| transport | WER | failure | exact | normalized | P50 | P70 | P95 | "
                "P100 | EOS→final P95 | first partial P50 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for transport, summary in stt_summaries.items():
            latency = summary["latency"]
            eos = summary["end_of_speech_to_final"]
            partial = summary["time_to_first_partial"]
            lines.append(
                "| {transport} | {wer} | {failure} | {exact} | {normalized} | {p50} | "
                "{p70} | {p95} | {p100} | {eos} | {partial} |".format(
                    transport=transport,
                    wer=_fmt(summary.get("wer_micro")),
                    failure=_fmt(summary.get("failure_rate")),
                    exact=_fmt(summary.get("exact_match_rate")),
                    normalized=_fmt(summary.get("normalized_match_rate")),
                    p50=_fmt(latency.get("p50_ms")),
                    p70=_fmt(latency.get("p70_ms")),
                    p95=_fmt(latency.get("p95_ms")),
                    p100=_fmt(latency.get("p100_ms")),
                    eos=_fmt(eos.get("p95_ms")),
                    partial=_fmt(partial.get("p50_ms")),
                )
            )
    lines.extend(["", "## Frozen retrieval impact", ""])
    if not retrieval_summaries:
        lines.append("No measurements.")
    else:
        lines.extend(
            [
                "| transport | metric | gold text | STT transcript | degradation |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for transport, summary in retrieval_summaries.items():
            for metric in QUALITY_METRICS:
                gold = float(summary["gold"][metric])
                transcript = float(summary["transcript"][metric])
                lines.append(
                    f"| {transport} | {metric} | {gold:.6f} | {transcript:.6f} | "
                    f"{gold - transcript:.6f} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The Saaras v3 generally available WebSocket emits VAD and finalized transcripts, "
            "not guaranteed true interim hypotheses. First-partial latency is unavailable unless "
            "a non-final transcript is actually observed.",
            "",
            "Mode selection rule fixed before measurement: streaming is selected only when its "
            "WER is no more than 0.02 worse than REST, failure rate no more than 0.01 worse, and "
            "P95 end-of-speech-to-final latency is lower than REST P95 request latency.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _run_predictions_mode(args: argparse.Namespace) -> None:
    cases = load_dataset(
        args.dataset, split=args.split, allow_sealed_test=args.allow_sealed_test
    )
    pairs = pair_cases_and_predictions(cases, read_jsonl(args.predictions))
    metrics, details = evaluate_stt_records(pairs)
    summary_path, cases_path = write_evaluation_run(
        output_dir=args.output_dir,
        run_id=args.run_id,
        stage="stt",
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        split=args.split,
        metrics=metrics,
        details=details,
        system_id=args.system_id,
    )
    print_summary(summary_path, cases_path, metrics)


def _run_manifest_mode(args: argparse.Namespace) -> None:
    manifest = _ready_manifest_rows(args.manifest)
    transports = ["rest", "streaming"] if args.transport == "both" else [args.transport]
    service = SarvamSTT.from_env(args.env_file)
    observations = asyncio.run(_run_live_stt(service, manifest, transports))
    raw_dir = args.output_dir / args.run_id
    raw_paths: dict[str, Path] = {}
    for transport, results in observations.items():
        raw_rows = [
            {
                "sample_id": sample["sample_id"],
                "case_id": sample["case_id"],
                **result.to_dict(),
            }
            for sample, result in zip(manifest, results, strict=True)
        ]
        raw_path = raw_dir / f"sarvam_{transport}_observations.jsonl"
        _write_jsonl(raw_path, raw_rows)
        raw_paths[transport] = raw_path
    stt_rows, stt_summaries = _stt_rows_and_summaries(manifest, observations)
    for transport in transports:
        write_evaluation_run(
            output_dir=args.output_dir,
            run_id=args.run_id,
            stage=f"stt_{transport}",
            dataset_path=args.manifest,
            predictions_path=raw_paths[transport],
            split=args.split,
            metrics=stt_summaries[transport],
            details=[row for row in stt_rows if row["transport"] == transport],
            system_id=args.system_id,
        )
    dataset_cases = load_dataset(
        args.dataset, split=args.split, allow_sealed_test=args.allow_sealed_test
    )
    retrieval_rows, retrieval_summaries = _retrieval_impact(
        manifest, observations, dataset_cases, device=args.device
    )
    for transport, summary in retrieval_summaries.items():
        metrics: dict[str, Any] = {"evaluated_queries": summary["gold"]["evaluated_queries"]}
        for metric in QUALITY_METRICS:
            gold = float(summary["gold"][metric])
            transcript = float(summary["transcript"][metric])
            metrics[f"gold_{metric}"] = gold
            metrics[f"transcript_{metric}"] = transcript
            metrics[f"degradation_{metric}"] = gold - transcript
        write_evaluation_run(
            output_dir=args.output_dir,
            run_id=args.run_id,
            stage=f"stt_retrieval_impact_{transport}",
            dataset_path=args.dataset,
            predictions_path=raw_paths[transport],
            split=args.split,
            metrics=metrics,
            details=[row for row in retrieval_rows if row["transport"] == transport],
            system_id=args.system_id,
            retrieval_stack=FROZEN_RETRIEVAL_STACK,
        )
    _write_csv(args.stt_csv, stt_rows, STT_CSV_FIELDS)
    _write_csv(args.retrieval_csv, retrieval_rows, RETRIEVAL_CSV_FIELDS)
    _write_recommendation(
        args.recommendation, stt_summaries, retrieval_summaries, len(manifest)
    )
    print(
        json.dumps(
            {
                "stt_csv": str(args.stt_csv),
                "retrieval_csv": str(args.retrieval_csv),
                "recommendation": str(args.recommendation),
                "samples": len(manifest),
                "stt": stt_summaries,
                "retrieval": retrieval_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("eval/eval_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--system-id", default="sarvam-saaras-v3-transcribe")
    parser.add_argument("--split", default="development")
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--transport", choices=("rest", "streaming", "both"), default="both")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--device", default="auto", help="Embedding device for retrieval impact (default: auto)"
    )
    parser.add_argument("--stt-csv", type=Path, default=Path("results/stt_evaluation.csv"))
    parser.add_argument(
        "--retrieval-csv", type=Path, default=Path("results/stt_retrieval_impact.csv")
    )
    parser.add_argument(
        "--recommendation", type=Path, default=Path("results/stt_recommendation.md")
    )
    args = parser.parse_args()
    if args.predictions:
        _run_predictions_mode(args)
    else:
        _run_manifest_mode(args)


if __name__ == "__main__":
    main()
