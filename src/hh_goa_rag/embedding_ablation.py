"""Stage 1: fixed-chunk/fixed-index embedding model ablation."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch

from hh_goa_rag.chunking import chunk_corpus
from hh_goa_rag.config import stable_fingerprint
from hh_goa_rag.io import read_jsonl, write_json, write_jsonl
from hh_goa_rag.metrics import (
    evaluate_rankings,
    evaluate_rankings_by_language,
    qrels_by_query,
)
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel, acquire_model
from hh_goa_rag.reporting import markdown_table, write_csv
from hh_goa_rag.retrieval import build_flat_ip, search_parent_rankings


def resolve_data_dir(config: dict[str, Any], requested: str | Path | None) -> Path:
    if requested:
        path = Path(requested)
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"No manifest.json in {path}")
        return path
    root = Path(config["dataset"]["output_dir"])
    manifests = sorted(root.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime)
    if not manifests:
        raise FileNotFoundError("No prepared dataset found; run `dataset prepare` first")
    return manifests[-1].parent


def _prepare_chunks(data_dir: Path, split: str, strategy: dict[str, Any]) -> Path:
    identity = {"data": data_dir.name, "split": split, "strategy": strategy}
    path = data_dir / "chunks" / f"{stable_fingerprint(identity)}.jsonl"
    if not path.exists():
        corpus = list(read_jsonl(data_dir / f"{split}_corpus.jsonl"))
        write_jsonl(path, chunk_corpus(corpus, strategy))
    return path


def _embedding_cache_paths(
    config: dict[str, Any], data_dir: Path, split: str, chunk_path: Path, model: str, revision: str
) -> tuple[Path, Path, Path]:
    identity = {
        "dataset": data_dir.name,
        "split": split,
        "chunks": chunk_path.stem,
        "model": model,
        "revision": revision,
        "normalized": True,
        "max_length": config["embedding_ablation"]["max_sequence_length"],
        "dtype": config["embedding_ablation"]["dtype"],
    }
    root = Path(config["cache"]["embeddings"]) / stable_fingerprint(identity)
    return root / "corpus.npy", root / "queries.npy", root / "metadata.json"


def _run_one(
    config: dict[str, Any], data_dir: Path, chunk_path: Path, repository: str
) -> dict[str, Any]:
    stage = config["embedding_ablation"]
    retrieval = config["retrieval"]
    split = stage["split"]
    chunks = list(read_jsonl(chunk_path))
    queries = list(read_jsonl(data_dir / f"{split}_queries.jsonl"))
    qrels = qrels_by_query(read_jsonl(data_dir / f"{split}_qrels.jsonl"))
    model_path, revision = acquire_model(repository, config["cache"]["models"])
    corpus_path, query_path, metadata_path = _embedding_cache_paths(
        config, data_dir, split, chunk_path, repository, revision
    )
    metadata: dict[str, Any]
    if corpus_path.exists() and query_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        adapter_version = MODEL_SPECS[repository].adapter_version
        cache_valid = adapter_version is None or metadata.get("adapter_version") == adapter_version
    else:
        cache_valid = False
        metadata = {}
    if cache_valid:
        corpus_embeddings = np.load(corpus_path)
        query_embeddings = np.load(query_path)
    else:
        spec = MODEL_SPECS[repository]
        batch_size = (
            int(stage["indicbert_batch_size"])
            if spec.mean_pooling_base_encoder
            else int(stage["batch_size"])
        )
        encoder = EmbeddingModel(
            spec,
            model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_sequence_length=int(stage["max_sequence_length"]),
            dtype=stage["dtype"],
        )
        try:
            warmups = int(retrieval["warmup_queries"])
            encoder.warm_up(queries[0]["text"], chunks[0]["text"], warmups)
            corpus_embeddings, corpus_elapsed_ms = encoder.encode_corpus(
                [chunk["text"] for chunk in chunks], batch_size
            )
            query_embeddings, query_latency = encoder.encode_queries(
                [query["text"] for query in queries]
            )
        finally:
            encoder.close()
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(corpus_path, corpus_embeddings)
        np.save(query_path, query_embeddings)
        metadata = {
            "model": repository,
            "model_revision": revision,
            "model_path": str(model_path.resolve()),
            "embedding_dimension": int(corpus_embeddings.shape[1]),
            "corpus_embedding_time_ms": corpus_elapsed_ms,
            "corpus_embedding_ms_per_chunk": corpus_elapsed_ms / len(chunks),
            "query_embedding_latency": query_latency,
            "query_prefix": MODEL_SPECS[repository].query_prefix,
            "passage_prefix": MODEL_SPECS[repository].passage_prefix,
            "query_task": MODEL_SPECS[repository].query_task,
            "passage_task": MODEL_SPECS[repository].passage_task,
            "mean_pooling_base_encoder": MODEL_SPECS[repository].mean_pooling_base_encoder,
            "reset_position_ids_buffer": MODEL_SPECS[
                repository
            ].reset_position_ids_buffer,
            "adapter_version": MODEL_SPECS[repository].adapter_version,
        }
        write_json(metadata_path, metadata)

    index_identity = stable_fingerprint(
        {"embedding_cache": corpus_path.parent.name, "index": "faiss_flat_ip"}
    )
    index_path = Path(config["cache"]["indexes"]) / "embedding" / f"{index_identity}.faiss"
    index, index_stats = build_flat_ip(corpus_embeddings, index_path)
    rankings, retrieval_latency = search_parent_rankings(
        index,
        query_embeddings,
        [str(query["query_id"]) for query in queries],
        [str(chunk["parent_id"]) for chunk in chunks],
        top_k=int(retrieval["top_k"]),
        oversample=int(retrieval["search_oversample"]),
        warmup_queries=int(retrieval["warmup_queries"]),
    )
    quality = evaluate_rankings(rankings, qrels)
    language_quality = evaluate_rankings_by_language(rankings, qrels)
    del index
    result: dict[str, Any] = {
        "status": "ok",
        "model": repository,
        "model_revision": revision,
        "split": split,
        "chunking": json.dumps(retrieval["baseline_chunking"], sort_keys=True),
        "index": "faiss.IndexFlatIP",
        "normalized_embeddings": True,
        "normalization_method": retrieval["normalization_method"],
        "embedding_dimension": metadata["embedding_dimension"],
        "corpus_chunks": len(chunks),
        **quality,
        "language_metrics": json.dumps(language_quality, ensure_ascii=False, sort_keys=True),
        "language_count": len(language_quality),
        "corpus_embedding_time_ms": metadata["corpus_embedding_time_ms"],
        "corpus_embedding_ms_per_chunk": metadata["corpus_embedding_ms_per_chunk"],
        "query_embedding_mean_ms": metadata["query_embedding_latency"]["mean_ms"],
        "query_embedding_p50_ms": metadata["query_embedding_latency"]["p50_ms"],
        "query_embedding_p95_ms": metadata["query_embedding_latency"]["p95_ms"],
        "retrieval_mean_ms": retrieval_latency["mean_ms"],
        "retrieval_p50_ms": retrieval_latency["p50_ms"],
        "retrieval_p95_ms": retrieval_latency["p95_ms"],
        **index_stats,
        "model_cache_path": str(model_path),
        "embedding_cache_path": str(corpus_path.parent),
    }
    result_name = f"{repository.replace('/', '__')}.json"
    per_model_path = Path("results") / "runs" / "embedding" / result_name
    write_json(per_model_path, result)
    return result


def _select_winner(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, ...]:
        quality = tuple(float(row[metric]) for metric in metrics)
        tie_latency = -(
            float(row["query_embedding_p50_ms"]) + float(row["retrieval_p50_ms"])
        )
        return (*quality, tie_latency)

    return max(rows, key=key)


def run_embedding_ablation(
    config: dict[str, Any], *, data_dir: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = config["embedding_ablation"]
    resolved_data = resolve_data_dir(config, data_dir)
    chunk_path = _prepare_chunks(
        resolved_data, stage["split"], config["retrieval"]["baseline_chunking"]
    )
    rows = [_run_one(config, resolved_data, chunk_path, model) for model in stage["models"]]
    write_csv("results/embedding_ablation.csv", rows)
    winner = _select_winner(rows, list(stage["selection_metrics"]))
    winner_record = {
        "stage": "embedding_ablation",
        "winner": winner["model"],
        "model_revision": winner["model_revision"],
        "selection_metrics_in_priority_order": stage["selection_metrics"],
        "metrics": winner,
        "dataset_artifact": str(resolved_data),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "faiss": faiss.__version__,
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "seed": config["experiment"]["seed"],
        },
    }
    write_json("results/embedding_winner.json", winner_record)
    columns = [
        "model",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "query_embedding_p50_ms",
        "retrieval_p50_ms",
        "index_size_bytes",
    ]
    print(markdown_table(rows, columns))
    return rows, winner_record
