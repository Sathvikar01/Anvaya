"""Standalone ablation for small multilingual embedding models and vector indexes."""

from __future__ import annotations

import csv
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import chromadb
import faiss
import numpy as np
import torch

from hh_goa_rag.chunking import chunk_corpus
from hh_goa_rag.config import stable_fingerprint
from hh_goa_rag.embedding_ablation import resolve_data_dir
from hh_goa_rag.index_backends import run_chroma_local, run_faiss, run_qdrant_local
from hh_goa_rag.io import read_jsonl, write_json
from hh_goa_rag.metrics import (
    evaluate_rankings,
    evaluate_rankings_by_language,
    qrels_by_query,
)
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel, acquire_model, safe_model_name
from hh_goa_rag.reporting import markdown_table, write_csv
from hh_goa_rag.retrieval import build_flat_ip

OUTPUT_ROOT = Path("results")


def _model_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _resolve_device(stage: dict[str, Any]) -> str:
    requested = str(stage.get("device", "auto")).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("small_ablation.device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("small_ablation.device=cuda but CUDA is unavailable")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _prepare_chunks(data_dir: Path, split: str, strategy: dict[str, Any]) -> Path:
    identity = {"data": data_dir.name, "split": split, "strategy": strategy}
    path = data_dir / "chunks" / f"small-ablation-{stable_fingerprint(identity)}.jsonl"
    if not path.exists():
        corpus = list(read_jsonl(data_dir / f"{split}_corpus.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        from hh_goa_rag.io import write_jsonl

        write_jsonl(path, chunk_corpus(corpus, strategy))
    return path


def _embedding_cache(
    config: dict[str, Any],
    data_dir: Path,
    chunk_path: Path,
    model: str,
    revision: str,
) -> tuple[Path, Path, Path]:
    stage = config["small_ablation"]
    identity = {
        "study": "small-ablation-v1",
        "dataset": data_dir.name,
        "split": stage["split"],
        "chunks": chunk_path.stem,
        "model": model,
        "revision": revision,
        "max_sequence_length": stage["max_sequence_length"],
        "dtype": stage["dtype"],
        "device": stage.get("device", "auto"),
    }
    root = Path(config["cache"]["embeddings"]) / "small-ablation" / stable_fingerprint(identity)
    return root / "corpus.npy", root / "queries.npy", root / "metadata.json"


def _encode_model(
    config: dict[str, Any],
    data_dir: Path,
    chunk_path: Path,
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    stage = config["small_ablation"]
    retrieval = config["retrieval"]
    model_path, revision = acquire_model(model, config["cache"]["models"])
    corpus_path, query_path, metadata_path = _embedding_cache(
        config, data_dir, chunk_path, model, revision
    )
    if corpus_path.exists() and query_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        corpus_vectors = np.load(corpus_path)
        query_vectors = np.load(query_path)
    else:
        device = _resolve_device(stage)
        dtype = str(stage["dtype"]) if device == "cuda" else "float32"
        encoder = EmbeddingModel(
            MODEL_SPECS[model],
            model_path,
            device=device,
            max_sequence_length=int(stage["max_sequence_length"]),
            dtype=dtype,
        )
        try:
            encoder.warm_up(
                str(queries[0]["text"]),
                str(chunks[0]["text"]),
                int(retrieval["warmup_queries"]),
            )
            corpus_vectors, corpus_ms = encoder.encode_corpus(
                [str(chunk["text"]) for chunk in chunks], int(stage["batch_size"])
            )
            query_vectors, query_latency = encoder.encode_queries(
                [str(query["text"]) for query in queries]
            )
        finally:
            encoder.close()
        metadata = {
            "model": model,
            "model_revision": revision,
            "embedding_dimension": int(corpus_vectors.shape[1]),
            "corpus_embedding_time_ms": corpus_ms,
            "corpus_embedding_ms_per_chunk": corpus_ms / len(chunks),
            "query_embedding_latency": query_latency,
            "model_size_bytes": _model_size_bytes(model_path),
            "device": device,
            "dtype": dtype,
        }
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(corpus_path, corpus_vectors)
        np.save(query_path, query_vectors)
        write_json(metadata_path, metadata)
    if len(corpus_vectors) != len(chunks) or len(query_vectors) != len(queries):
        raise RuntimeError(f"Embedding cache mismatch for {model}")
    return {
        "model": model,
        "model_revision": revision,
        "model_cache_path": str(model_path),
        "embedding_cache_path": str(corpus_path.parent),
        "corpus_vectors": corpus_vectors,
        "query_vectors": query_vectors,
        "metadata": metadata,
    }


def _embedding_rows(
    config: dict[str, Any],
    data_dir: Path,
    chunk_path: Path,
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    retrieval = config["retrieval"]
    vectors_by_model: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for model in config["small_ablation"]["models"]:
        encoded = _encode_model(config, data_dir, chunk_path, chunks, queries, model)
        vectors_by_model[model] = encoded
        index_path = (
            Path(config["cache"]["indexes"])
            / "small-ablation"
            / f"embedding-{safe_model_name(model)}.faiss"
        )
        index, index_stats = build_flat_ip(encoded["corpus_vectors"], index_path)
        from hh_goa_rag.retrieval import search_parent_rankings

        rankings, retrieval_latency = search_parent_rankings(
            index,
            encoded["query_vectors"],
            [str(query["query_id"]) for query in queries],
            [str(chunk["parent_id"]) for chunk in chunks],
            top_k=int(retrieval["top_k"]),
            oversample=int(retrieval["search_oversample"]),
            warmup_queries=int(retrieval["warmup_queries"]),
        )
        quality = evaluate_rankings(rankings, qrels)
        language_quality = evaluate_rankings_by_language(rankings, qrels)
        metadata = encoded["metadata"]
        row = {
            "status": "ok",
            "model": model,
            "model_revision": encoded["model_revision"],
            "split": config["small_ablation"]["split"],
            "chunking": json.dumps(config["small_ablation"]["chunking"], sort_keys=True),
            "embedding_dimension": metadata["embedding_dimension"],
            "model_size_bytes": metadata["model_size_bytes"],
            "embedding_device": metadata["device"],
            "embedding_dtype": metadata["dtype"],
            "corpus_chunks": len(chunks),
            **quality,
            "language_metrics": json.dumps(
                language_quality, ensure_ascii=False, sort_keys=True
            ),
            "language_count": len(language_quality),
            "corpus_embedding_time_ms": metadata["corpus_embedding_time_ms"],
            "corpus_embedding_ms_per_chunk": metadata["corpus_embedding_ms_per_chunk"],
            "query_embedding_mean_ms": metadata["query_embedding_latency"]["mean_ms"],
            "query_embedding_p50_ms": metadata["query_embedding_latency"]["p50_ms"],
            "query_embedding_p70_ms": metadata["query_embedding_latency"]["p70_ms"],
            "query_embedding_p95_ms": metadata["query_embedding_latency"]["p95_ms"],
            "query_embedding_p100_ms": metadata["query_embedding_latency"]["p100_ms"],
            "retrieval_mean_ms": retrieval_latency["mean_ms"],
            "retrieval_p50_ms": retrieval_latency["p50_ms"],
            "retrieval_p70_ms": retrieval_latency["p70_ms"],
            "retrieval_p95_ms": retrieval_latency["p95_ms"],
            "retrieval_p100_ms": retrieval_latency["p100_ms"],
            **index_stats,
            "model_cache_path": encoded["model_cache_path"],
            "embedding_cache_path": encoded["embedding_cache_path"],
        }
        rows.append(row)
        write_json(OUTPUT_ROOT / "runs" / "small_embedding" / f"{safe_model_name(model)}.json", row)
        del index
    return rows, vectors_by_model


def _select(rows: list[dict[str, Any]], metrics: list[str], latency_key: str) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, ...]:
        return (*tuple(float(row[metric]) for metric in metrics), -float(row[latency_key]))

    return max(rows, key=key)


def _index_rows(
    config: dict[str, Any],
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: dict[str, set[str]],
    encoded: dict[str, Any],
    data_dir: Path,
) -> list[dict[str, Any]]:
    retrieval = config["retrieval"]
    stage = config["small_ablation"]
    vectors = encoded["corpus_vectors"]
    query_vectors = encoded["query_vectors"]
    query_ids = [str(query["query_id"]) for query in queries]
    parent_ids = [str(chunk["parent_id"]) for chunk in chunks]
    rows: list[dict[str, Any]] = []
    root = Path(config["cache"]["indexes"]) / "small-ablation" / "backends"
    for backend in stage["index_backends"]:
        name = backend["name"]
        identity = stable_fingerprint(
            {
                "study": "small-ablation-v1",
                "data": data_dir.name,
                "model": encoded["model_revision"],
                "backend": backend,
            }
        )
        artifact = root / f"{name}-{identity}"
        if backend["engine"] == "faiss":
            result = run_faiss(
                backend,
                vectors,
                query_vectors,
                query_ids,
                parent_ids,
                artifact.with_suffix(".faiss"),
                top_k=int(retrieval["top_k"]), oversample=int(retrieval["search_oversample"]),
                warmup_queries=int(retrieval["warmup_queries"]),
            )
        elif backend["engine"] == "qdrant_local":
            result = run_qdrant_local(
                backend, vectors, query_vectors, query_ids, parent_ids, artifact,
                top_k=int(retrieval["top_k"]), oversample=int(retrieval["search_oversample"]),
                warmup_queries=int(retrieval["warmup_queries"]),
            )
        elif backend["engine"] == "chroma_local":
            result = run_chroma_local(
                backend, vectors, query_vectors, query_ids, parent_ids, artifact,
                top_k=int(retrieval["top_k"]), oversample=int(retrieval["search_oversample"]),
                warmup_queries=int(retrieval["warmup_queries"]),
            )
        else:
            raise ValueError(f"Unsupported index engine: {backend['engine']}")
        row = {
            "status": "ok",
            "backend": name,
            "engine": backend["engine"],
            "backend_config": json.dumps(backend, sort_keys=True),
            "model": encoded["model"],
            "model_revision": encoded["model_revision"],
            "embedding_dimension": int(vectors.shape[1]),
            "corpus_vectors": len(vectors),
            "normalized_embeddings": True,
            "normalization_method": retrieval["normalization_method"],
            "cosine_equivalent": True,
            "index_device": "cpu",
            **evaluate_rankings(result.rankings, qrels),
            "language_metrics": json.dumps(
                evaluate_rankings_by_language(result.rankings, qrels),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "language_count": len(evaluate_rankings_by_language(result.rankings, qrels)),
            "retrieval_mean_ms": result.latency["mean_ms"],
            "retrieval_p50_ms": result.latency["p50_ms"],
            "retrieval_p70_ms": result.latency["p70_ms"],
            "retrieval_p95_ms": result.latency["p95_ms"],
            "retrieval_p100_ms": result.latency["p100_ms"],
            **result.stats,
            "seed": config["experiment"]["seed"],
            "faiss_version": faiss.__version__,
            "qdrant_client_version": version("qdrant-client"),
            "chroma_version": chromadb.__version__,
        }
        rows.append(row)
        write_json(OUTPUT_ROOT / "runs" / "small_index" / f"{name}-{identity}.json", row)
    return rows


def run_small_ablation(
    config: dict[str, Any], *, data_dir: str | Path | None = None, index_only: bool = False
) -> dict[str, Any]:
    stage = config["small_ablation"]
    resolved_data = resolve_data_dir(config, data_dir)
    split = stage["split"]
    chunk_path = _prepare_chunks(resolved_data, split, stage["chunking"])
    chunks = list(read_jsonl(chunk_path))
    queries = list(read_jsonl(resolved_data / f"{split}_queries.jsonl"))
    qrels = qrels_by_query(read_jsonl(resolved_data / f"{split}_qrels.jsonl"))
    if index_only:
        return run_small_index_only(config, resolved_data, chunks, queries, qrels)
    embedding_rows, vectors_by_model = _embedding_rows(
        config, resolved_data, chunk_path, chunks, queries, qrels
    )
    write_csv(OUTPUT_ROOT / "small_embedding_ablation.csv", embedding_rows)
    embedding_winner = _select(
        embedding_rows, list(stage["selection_metrics"]), "query_embedding_p50_ms"
    )
    index_rows: list[dict[str, Any]] = []
    for encoded in vectors_by_model.values():
        index_rows.extend(_index_rows(config, chunks, queries, qrels, encoded, resolved_data))
    write_csv(OUTPUT_ROOT / "small_index_ablation.csv", index_rows)
    index_winner = _select(index_rows, list(stage["selection_metrics"]), "retrieval_p50_ms")
    summary = {
        "study": "small_embedding_index_ablation",
        "dataset_artifact": str(resolved_data),
        "split": split,
        "chunk_artifact": str(chunk_path),
        "embedding_winner": embedding_winner["model"],
        "index_winner_model": index_winner["model"],
        "index_winner": index_winner["backend"],
        "selection_metrics_in_priority_order": stage["selection_metrics"],
        "embedding_results": str(OUTPUT_ROOT / "small_embedding_ablation.csv"),
        "index_results": str(OUTPUT_ROOT / "small_index_ablation.csv"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "faiss": faiss.__version__,
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "seed": config["experiment"]["seed"],
        },
    }
    write_json(OUTPUT_ROOT / "small_ablation_summary.json", summary)
    print("Small embedding ablation")
    print(markdown_table(embedding_rows, [
        "model", "embedding_dimension", "model_size_bytes", "recall_at_1", "recall_at_5",
        "recall_at_10", "mrr_at_10", "query_embedding_p50_ms", "retrieval_p50_ms",
    ]))
    print("\nPaired index ablation on both embeddings")
    print(markdown_table(index_rows, [
        "model", "backend", "recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10",
        "indexing_time_ms", "retrieval_p50_ms", "retrieval_p95_ms", "index_size_bytes",
    ]))
    return summary


def run_small_index_only(
    config: dict[str, Any],
    data_dir: Path,
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: dict[str, set[str]],
) -> dict[str, Any]:
    """Run only the CPU index comparison using an existing embedding snapshot."""
    stage = config["small_ablation"]
    source_path = OUTPUT_ROOT / "small_embedding_ablation.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Embedding results are required before index-only mode: {source_path}"
        )
    with source_path.open(encoding="utf-8", newline="") as handle:
        embedding_rows = list(csv.DictReader(handle))
    if not embedding_rows:
        raise RuntimeError(f"No embedding rows found in {source_path}")
    embedding_winner = _select(
        [
            {
                key: float(value) if key in stage["selection_metrics"] else value
                for key, value in row.items()
            }
            for row in embedding_rows
        ],
        list(stage["selection_metrics"]),
        "query_embedding_p50_ms",
    )
    cache_path = Path(str(embedding_winner["embedding_cache_path"]))
    corpus_path = cache_path / "corpus.npy"
    query_path = cache_path / "queries.npy"
    if not corpus_path.exists() or not query_path.exists():
        raise FileNotFoundError(f"Embedding snapshot is incomplete: {cache_path}")
    source_metadata_path = cache_path / "metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )
    encoded = {
        "model": str(embedding_winner["model"]),
        "model_revision": str(embedding_winner["model_revision"]),
        "corpus_vectors": np.load(corpus_path),
        "query_vectors": np.load(query_path),
    }
    index_rows = _index_rows(config, chunks, queries, qrels, encoded, data_dir)
    cpu_path = OUTPUT_ROOT / "small_index_ablation_cpu.csv"
    write_csv(cpu_path, index_rows)
    index_winner = _select(index_rows, list(stage["selection_metrics"]), "retrieval_p50_ms")
    summary = {
        "study": "small_index_ablation_cpu",
        "dataset_artifact": str(data_dir),
        "split": stage["split"],
        "embedding_model": encoded["model"],
        "embedding_source_cache": str(cache_path),
        "embedding_source_device": source_metadata.get(
            "device", embedding_winner.get("embedding_device", "unknown")
        ),
        "index_device": "cpu",
        "cuda_visible_to_process": torch.cuda.is_available(),
        "index_winner": index_winner["backend"],
        "results": str(cpu_path),
        "selection_metrics_in_priority_order": stage["selection_metrics"],
    }
    write_json(OUTPUT_ROOT / "small_index_ablation_cpu_summary.json", summary)
    print("CPU-only index ablation on: " + encoded["model"])
    print(markdown_table(index_rows, [
        "backend", "recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10",
        "indexing_time_ms", "retrieval_p50_ms", "retrieval_p95_ms", "index_size_bytes",
    ]))
    return summary
