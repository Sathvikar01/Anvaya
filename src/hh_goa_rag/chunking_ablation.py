"""Stage 2: chunking ablation using the winning embedding model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hh_goa_rag.chunking import chunk_corpus, split_sentences
from hh_goa_rag.config import stable_fingerprint
from hh_goa_rag.embedding_ablation import resolve_data_dir
from hh_goa_rag.io import read_jsonl, write_json, write_jsonl
from hh_goa_rag.metrics import (
    evaluate_rankings,
    evaluate_rankings_by_language,
    qrels_by_query,
)
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel, acquire_model
from hh_goa_rag.reporting import markdown_table, write_csv
from hh_goa_rag.retrieval import build_flat_ip, search_parent_rankings


def _load_embedding_winner(path: str | Path = "results/embedding_winner.json") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Run the embedding ablation before chunking ablation")
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_vectors(
    config: dict[str, Any],
    data_dir: Path,
    corpus: list[dict[str, Any]],
    encoder: EmbeddingModel,
    model: str,
    revision: str,
) -> dict[str, np.ndarray]:
    sentence_counts = [len(split_sentences(parent["text"])) for parent in corpus]
    sentences = [
        sentence for parent in corpus for sentence in split_sentences(parent["text"])
    ]
    identity = {
        "dataset": data_dir.name,
        "model": model,
        "revision": revision,
        "segmenter": "punctuation-v1",
        "sentence_count": len(sentences),
    }
    root = Path(config["cache"]["embeddings"]) / "semantic_sentences"
    path = root / f"{stable_fingerprint(identity)}.npy"
    if path.exists():
        vectors = np.load(path)
    else:
        vectors, _elapsed_ms = encoder.encode_corpus(
            sentences, int(config["chunking_ablation"]["batch_size"])
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    if len(vectors) != len(sentences):
        raise RuntimeError("Cached semantic sentence vectors do not match the corpus")
    result: dict[str, np.ndarray] = {}
    offset = 0
    for parent, count in zip(corpus, sentence_counts, strict=True):
        result[parent["passage_id"]] = vectors[offset : offset + count]
        offset += count
    return result


def _prepare_strategy_chunks(
    config: dict[str, Any],
    data_dir: Path,
    corpus: list[dict[str, Any]],
    strategy: dict[str, Any],
    encoder: EmbeddingModel,
    model: str,
    revision: str,
) -> Path:
    identity = {"dataset": data_dir.name, "split": "dev", "strategy": strategy}
    path = data_dir / "chunks" / f"chunking-{strategy['name']}-{stable_fingerprint(identity)}.jsonl"
    if path.exists():
        return path
    semantic_vectors = None
    if strategy["strategy"] == "semantic":
        semantic_vectors = _semantic_vectors(
            config, data_dir, corpus, encoder, model, revision
        )
    chunks = chunk_corpus(corpus, strategy, semantic_vectors=semantic_vectors)
    write_jsonl(path, chunks)
    return path


def _corpus_embeddings(
    config: dict[str, Any],
    data_dir: Path,
    chunk_path: Path,
    chunks: list[dict[str, Any]],
    encoder: EmbeddingModel,
    model: str,
    revision: str,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    identity = {
        "dataset": data_dir.name,
        "chunks": chunk_path.stem,
        "model": model,
        "revision": revision,
        "normalized": True,
        "max_length": config["embedding_ablation"]["max_sequence_length"],
    }
    root = Path(config["cache"]["embeddings"]) / "chunking" / stable_fingerprint(identity)
    vector_path = root / "corpus.npy"
    metadata_path = root / "metadata.json"
    if vector_path.exists() and metadata_path.exists():
        vectors = np.load(vector_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        vectors, elapsed_ms = encoder.encode_corpus(
            [chunk["text"] for chunk in chunks],
            int(config["chunking_ablation"]["batch_size"]),
        )
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vector_path, vectors)
        metadata = {
            "model": model,
            "model_revision": revision,
            "chunks": len(chunks),
            "embedding_dimension": int(vectors.shape[1]),
            "corpus_embedding_time_ms": elapsed_ms,
            "corpus_embedding_ms_per_chunk": elapsed_ms / len(chunks),
            "cache_path": str(root),
        }
        write_json(metadata_path, metadata)
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Embedding cache mismatch for {chunk_path}")
    return vectors, metadata


def _select_winner(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, ...]:
        quality = tuple(float(row[metric]) for metric in metrics)
        return (*quality, -float(row["retrieval_p50_ms"]))

    return max(rows, key=key)


def run_chunking_ablation(
    config: dict[str, Any], *, data_dir: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = config["chunking_ablation"]
    retrieval = config["retrieval"]
    resolved_data = resolve_data_dir(config, data_dir)
    embedding_winner = _load_embedding_winner()
    repository = embedding_winner["winner"]
    model_path, revision = acquire_model(repository, config["cache"]["models"])
    if revision != embedding_winner["model_revision"]:
        raise RuntimeError("Embedding winner revision no longer matches the cached result")
    encoder = EmbeddingModel(
        MODEL_SPECS[repository],
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_sequence_length=int(config["embedding_ablation"]["max_sequence_length"]),
        dtype=config["embedding_ablation"]["dtype"],
    )
    split = stage["split"]
    corpus = list(read_jsonl(resolved_data / f"{split}_corpus.jsonl"))
    queries = list(read_jsonl(resolved_data / f"{split}_queries.jsonl"))
    qrels = qrels_by_query(read_jsonl(resolved_data / f"{split}_qrels.jsonl"))
    query_cache = Path(embedding_winner["metrics"]["embedding_cache_path"]) / "queries.npy"
    query_embeddings = np.load(query_cache)
    rows: list[dict[str, Any]] = []
    try:
        encoder.warm_up(
            queries[0]["text"], corpus[0]["text"], int(retrieval["warmup_queries"])
        )
        for strategy in stage["strategies"]:
            chunk_path = _prepare_strategy_chunks(
                config, resolved_data, corpus, strategy, encoder, repository, revision
            )
            chunks = list(read_jsonl(chunk_path))
            embeddings, embedding_stats = _corpus_embeddings(
                config,
                resolved_data,
                chunk_path,
                chunks,
                encoder,
                repository,
                revision,
            )
            index_identity = stable_fingerprint(
                {"chunks": chunk_path.stem, "model_revision": revision, "index": "flat_ip"}
            )
            index_path = (
                Path(config["cache"]["indexes"])
                / "chunking"
                / f"{strategy['name']}-{index_identity}.faiss"
            )
            index, index_stats = build_flat_ip(embeddings, index_path)
            rankings, latency = search_parent_rankings(
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
            row: dict[str, Any] = {
                "status": "ok",
                "strategy": strategy["name"],
                "strategy_config": json.dumps(strategy, sort_keys=True),
                "model": repository,
                "model_revision": revision,
                "split": split,
                "index": "faiss.IndexFlatIP",
                "normalized_embeddings": True,
                "normalization_method": retrieval["normalization_method"],
                "corpus_chunks": len(chunks),
                "avg_chunks_per_parent": len(chunks) / len(corpus),
                **quality,
                "language_metrics": json.dumps(
                    language_quality, ensure_ascii=False, sort_keys=True
                ),
                "language_count": len(language_quality),
                "corpus_embedding_time_ms": embedding_stats["corpus_embedding_time_ms"],
                "corpus_embedding_ms_per_chunk": embedding_stats[
                    "corpus_embedding_ms_per_chunk"
                ],
                "query_embedding_p50_ms": embedding_winner["metrics"][
                    "query_embedding_p50_ms"
                ],
                "retrieval_mean_ms": latency["mean_ms"],
                "retrieval_p50_ms": latency["p50_ms"],
                "retrieval_p95_ms": latency["p95_ms"],
                **index_stats,
                "chunk_artifact": str(chunk_path),
                "embedding_cache_path": embedding_stats["cache_path"],
            }
            rows.append(row)
            write_json(
                Path("results") / "runs" / "chunking" / f"{strategy['name']}.json", row
            )
    finally:
        encoder.close()
    write_csv("results/chunking_ablation.csv", rows)
    winner = _select_winner(rows, list(stage["selection_metrics"]))
    winner_record = {
        "stage": "chunking_ablation",
        "winner": winner["strategy"],
        "strategy_config": json.loads(winner["strategy_config"]),
        "embedding_model": repository,
        "model_revision": revision,
        "selection_metrics_in_priority_order": stage["selection_metrics"],
        "metrics": winner,
        "dataset_artifact": str(resolved_data),
        "seed": config["experiment"]["seed"],
    }
    write_json("results/chunking_winner.json", winner_record)
    columns = [
        "strategy",
        "corpus_chunks",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "retrieval_p50_ms",
        "index_size_bytes",
    ]
    print(markdown_table(rows, columns))
    return rows, winner_record
