"""Stage 3: vector index and local storage ablation on fixed embeddings."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import chromadb
import faiss
import numpy as np

from hh_goa_rag.config import stable_fingerprint
from hh_goa_rag.embedding_ablation import resolve_data_dir
from hh_goa_rag.index_backends import run_chroma_local, run_faiss, run_qdrant_local
from hh_goa_rag.io import read_jsonl, write_json
from hh_goa_rag.metrics import (
    evaluate_rankings,
    evaluate_rankings_by_language,
    qrels_by_query,
)
from hh_goa_rag.reporting import markdown_table, write_csv
from hh_goa_rag.retrieval import l2_normalize


def _load_winner(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required prior-stage winner is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _select_winner(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, ...]:
        quality = tuple(float(row[metric]) for metric in metrics)
        return (
            *quality,
            -float(row["retrieval_p95_ms"]),
            -float(row["retrieval_p50_ms"]),
            -float(row["index_size_bytes"]),
        )

    return max(rows, key=key)


def run_index_ablation(
    config: dict[str, Any], *, data_dir: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = config["index_ablation"]
    retrieval = config["retrieval"]
    resolved_data = resolve_data_dir(config, data_dir)
    embedding_winner = _load_winner("results/embedding_winner.json")
    chunking_winner = _load_winner("results/chunking_winner.json")
    split = stage["split"]
    chunks = list(read_jsonl(chunking_winner["metrics"]["chunk_artifact"]))
    queries = list(read_jsonl(resolved_data / f"{split}_queries.jsonl"))
    qrels = qrels_by_query(read_jsonl(resolved_data / f"{split}_qrels.jsonl"))
    corpus_vectors = np.load(
        Path(chunking_winner["metrics"]["embedding_cache_path"]) / "corpus.npy"
    )
    query_vectors = np.load(
        Path(embedding_winner["metrics"]["embedding_cache_path"]) / "queries.npy"
    )
    if len(corpus_vectors) != len(chunks) or len(query_vectors) != len(queries):
        raise RuntimeError("Fixed embedding cache does not match the evaluation artifacts")
    corpus_vectors = l2_normalize(corpus_vectors)
    query_vectors = l2_normalize(query_vectors)
    query_ids = [str(query["query_id"]) for query in queries]
    parent_ids = [str(chunk["parent_id"]) for chunk in chunks]
    identity = {
        "dataset": resolved_data.name,
        "model_revision": embedding_winner["model_revision"],
        "chunking": chunking_winner["strategy_config"],
        "vectors": chunking_winner["metrics"]["embedding_cache_path"],
        "normalization_method": retrieval["normalization_method"],
    }
    root = Path(config["cache"]["indexes"]) / "backends" / stable_fingerprint(identity)
    rows: list[dict[str, Any]] = []
    for backend in stage["backends"]:
        name = backend["name"]
        run_identity = stable_fingerprint({"identity": identity, "backend": backend})
        run_path = Path("results") / "runs" / "index" / f"{name}-{run_identity}.json"
        if run_path.exists():
            rows.append(json.loads(run_path.read_text(encoding="utf-8")))
            continue
        if backend["engine"] == "faiss":
            result = run_faiss(
                backend,
                corpus_vectors,
                query_vectors,
                query_ids,
                parent_ids,
                root / f"{name}.faiss",
                top_k=int(retrieval["top_k"]),
                oversample=int(retrieval["search_oversample"]),
                warmup_queries=int(retrieval["warmup_queries"]),
            )
        elif backend["engine"] == "qdrant_local":
            result = run_qdrant_local(
                backend,
                corpus_vectors,
                query_vectors,
                query_ids,
                parent_ids,
                root / name,
                top_k=int(retrieval["top_k"]),
                oversample=int(retrieval["search_oversample"]),
                warmup_queries=int(retrieval["warmup_queries"]),
            )
        elif backend["engine"] == "chroma_local":
            result = run_chroma_local(
                backend,
                corpus_vectors,
                query_vectors,
                query_ids,
                parent_ids,
                root / name,
                top_k=int(retrieval["top_k"]),
                oversample=int(retrieval["search_oversample"]),
                warmup_queries=int(retrieval["warmup_queries"]),
            )
        else:
            raise ValueError(f"Unsupported index engine: {backend['engine']}")
        quality = evaluate_rankings(result.rankings, qrels)
        language_quality = evaluate_rankings_by_language(result.rankings, qrels)
        row: dict[str, Any] = {
            "status": "ok",
            "backend": name,
            "engine": backend["engine"],
            "backend_config": json.dumps(backend, sort_keys=True),
            "model": embedding_winner["winner"],
            "model_revision": embedding_winner["model_revision"],
            "chunking": chunking_winner["winner"],
            "chunking_config": json.dumps(
                chunking_winner["strategy_config"], sort_keys=True
            ),
            "split": split,
            "normalized_embeddings": True,
            "normalization_method": retrieval["normalization_method"],
            "cosine_equivalent": True,
            "corpus_vectors": len(corpus_vectors),
            "embedding_dimension": int(corpus_vectors.shape[1]),
            **quality,
            "language_metrics": json.dumps(
                language_quality, ensure_ascii=False, sort_keys=True
            ),
            "language_count": len(language_quality),
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
        write_json(run_path, row)
    write_csv("results/index_ablation.csv", rows)
    winner = _select_winner(rows, list(stage["selection_metrics"]))
    winner_record = {
        "stage": "index_ablation",
        "winner": winner["backend"],
        "backend_config": json.loads(winner["backend_config"]),
        "embedding_model": embedding_winner["winner"],
        "model_revision": embedding_winner["model_revision"],
        "chunking_strategy": chunking_winner["winner"],
        "chunking_config": chunking_winner["strategy_config"],
        "selection_metrics_in_priority_order": stage["selection_metrics"],
        "latency_tiebreakers_in_priority_order": [
            "retrieval_p95_ms",
            "retrieval_p50_ms",
            "index_size_bytes",
        ],
        "metrics": winner,
        "dataset_artifact": str(resolved_data),
        "seed": config["experiment"]["seed"],
    }
    write_json("results/index_winner.json", winner_record)
    columns = [
        "backend",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "retrieval_p50_ms",
        "retrieval_p70_ms",
        "retrieval_p95_ms",
        "retrieval_p100_ms",
        "indexing_time_ms",
        "index_ram_bytes",
        "index_size_bytes",
    ]
    print(markdown_table(rows, columns))
    return rows, winner_record
