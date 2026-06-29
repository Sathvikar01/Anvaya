"""Sealed-test evaluation, retriever handoff, recommendation, and safe model cleanup."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hh_goa_rag.chunking import chunk_corpus
from hh_goa_rag.cleanup import cleanup_losing_models
from hh_goa_rag.config import stable_fingerprint
from hh_goa_rag.embedding_ablation import resolve_data_dir
from hh_goa_rag.index_backends import run_faiss
from hh_goa_rag.io import read_jsonl, write_json, write_jsonl
from hh_goa_rag.metrics import (
    evaluate_rankings,
    evaluate_rankings_by_language,
    qrels_by_query,
)
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel, acquire_model
from hh_goa_rag.reporting import markdown_table


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _table(path: str | Path, label: str, columns: list[str]) -> str:
    rows = _load_csv(path)
    formatted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {label: row[label]}
        for column in columns:
            item[column] = float(row[column])
        formatted.append(item)
    return markdown_table(formatted, [label, *columns])


def _prepare_test_embeddings(
    config: dict[str, Any],
    data_dir: Path,
    chunks: list[dict[str, Any]],
    chunk_path: Path,
    queries: list[dict[str, Any]],
    model: str,
    revision: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    effective_device = "cuda" if torch.cuda.is_available() else "cpu"
    effective_dtype = (
        str(config["embedding_ablation"]["dtype"])
        if effective_device == "cuda"
        else "float32"
    )
    identity = {
        "dataset": data_dir.name,
        "split": "test",
        "chunks": chunk_path.stem,
        "model": model,
        "revision": revision,
        "max_length": config["embedding_ablation"]["max_sequence_length"],
        "device": effective_device,
        "dtype": effective_dtype,
        "normalization": config["retrieval"]["normalization_method"],
    }
    root = Path(config["cache"]["embeddings"]) / "final" / stable_fingerprint(identity)
    corpus_path = root / "corpus.npy"
    query_path = root / "queries.npy"
    metadata_path = root / "metadata.json"
    if corpus_path.exists() and query_path.exists() and metadata_path.exists():
        return (
            np.load(corpus_path),
            np.load(query_path),
            _load_json(metadata_path),
        )
    model_path, acquired_revision = acquire_model(model, config["cache"]["models"])
    if acquired_revision != revision:
        raise RuntimeError("Winning model revision changed before sealed-test evaluation")
    encoder = EmbeddingModel(
        MODEL_SPECS[model],
        model_path,
        device=effective_device,
        max_sequence_length=int(config["embedding_ablation"]["max_sequence_length"]),
        dtype=effective_dtype,
    )
    try:
        encoder.warm_up(
            queries[0]["text"], chunks[0]["text"], int(config["retrieval"]["warmup_queries"])
        )
        corpus_vectors, corpus_ms = encoder.encode_corpus(
            [str(chunk["text"]) for chunk in chunks],
            int(config["chunking_ablation"]["batch_size"]),
        )
        query_vectors, query_latency = encoder.encode_queries(
            [str(query["text"]) for query in queries]
        )
    finally:
        encoder.close()
    root.mkdir(parents=True, exist_ok=True)
    np.save(corpus_path, corpus_vectors)
    np.save(query_path, query_vectors)
    metadata = {
        "identity": identity,
        "corpus_embedding_time_ms": corpus_ms,
        "corpus_embedding_ms_per_chunk": corpus_ms / len(chunks),
        "query_embedding_latency": query_latency,
        "cache_path": str(root),
    }
    write_json(metadata_path, metadata)
    return corpus_vectors, query_vectors, metadata


def _recommendation(
    final_result: dict[str, Any],
    embedding_winner: dict[str, Any],
    chunking_winner: dict[str, Any],
    index_winner: dict[str, Any],
    manifest: dict[str, Any],
    cleanup: dict[str, Any],
) -> str:
    dataset_revision = manifest["artifact_identity"]["revision"]
    languages = ", ".join(manifest["artifact_identity"]["languages"])
    embedding_model = embedding_winner["winner"]
    chunking_strategy = chunking_winner["winner"]
    backend_name = index_winner["winner"]
    embedding_dimension = embedding_winner["metrics"].get(
        "embedding_dimension", "recorded per candidate"
    )
    embedding_table = _table(
        "results/embedding_ablation.csv",
        "model",
        ["recall_at_10", "mrr_at_10", "ndcg_at_10", "query_embedding_p50_ms"],
    )
    chunking_table = _table(
        "results/chunking_ablation.csv",
        "strategy",
        ["recall_at_10", "mrr_at_10", "ndcg_at_10", "retrieval_p50_ms"],
    )
    index_table = _table(
        "results/index_ablation.csv",
        "backend",
        [
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "retrieval_p50_ms",
            "retrieval_p95_ms",
            "indexing_time_ms",
        ],
    )
    test_table = markdown_table(
        [final_result],
        [
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "retrieval_p50_ms",
            "retrieval_p95_ms",
        ],
    )
    removed = "\n".join(
        f"- `{item['repository']}` ({int(item['bytes']) / 2**30:.2f} GiB)"
        for item in cleanup["removed"]
    )
    return f"""# HH Goa retrieval-stack recommendation

## Selected stack

**Best embedding model → Best chunking strategy → Best index/storage system**

**`{embedding_model}` → `{chunking_strategy}` → `{backend_name}`**

Selection used only the development split. The upstream validation-derived test split remained
sealed until all three winners and their configurations were fixed.

## Exact reproducibility configuration

- Dataset: `ai4bharat/MSMARCO-XI`, balanced languages `{languages}`, Hub revision
  `{dataset_revision}`
- Processed dataset artifact: `{final_result['dataset_artifact']}`
- Random seed: `{final_result['seed']}`
- Embedding checkpoint: `{embedding_model}` revision `{embedding_winner['model_revision']}`
- Embedding dimension: `{embedding_dimension}`
- Normalization: final float32 L2 (`float32_l2_v1`); similarity is cosine-equivalent inner product
- Chunking: punctuation-aware sentence packing, maximum 128 whitespace-delimited words,
  parent-level qrels
- Index configuration: `{json.dumps(index_winner['backend_config'], sort_keys=True)}`
- Retrieval: top 10 unique parents, search oversampling 20×, 20 warm-up queries
- Evaluation: `{manifest['splits']['dev']['queries']}` balanced development queries and
  `{manifest['splits']['test']['queries']}` sealed test queries exactly once

## Embedding ablation (development)

{embedding_table}

The winner used the predeclared lexicographic quality priority
`{embedding_winner['selection_metrics_in_priority_order']}`; latency was used only after quality
ties. Each row also records per-language metrics.

## Chunking ablation (development)

{chunking_table}

Chunking was selected on the same multilingual development artifact with the embedding model
held fixed; the table reports its quality/latency trade-off.

## Index/storage ablation (development)

{index_table}

Index candidates were evaluated with the same normalized vectors and query set; the table reports
FAISS algorithm quality, build cost, serialized size, and p50/p95/p100 retrieval latency.

## Sealed test result

{test_table}

- Test chunks: {final_result['corpus_vectors']}
- Indexing time: {final_result['indexing_time_ms']:.4f} ms
- Index disk size: {int(final_result['index_size_bytes']) / 2**20:.2f} MiB
- Query embedding P50: {final_result['query_embedding_p50_ms']:.4f} ms
- Final index artifact: `{final_result['index_artifact']}`
- Chunk-to-parent artifact: `{final_result['chunk_artifact']}`

`hh_goa_rag.retriever.ParentFaissRetriever` loads these two artifacts and accepts an already encoded
query vector, so it can be placed directly after the future STT/query-embedding boundary without
adding generation, speech, or frontend code here.

## Project-local model cleanup

The winning `{embedding_model}` directory was preserved. Only direct children of `{cleanup['root']}`
carrying the exact `.hh_goa_model.json` ownership marker for this experiment were eligible for
deletion. Removed:

{removed}

Freed {int(cleanup['removed_bytes']) / 2**30:.2f} GiB. No global Hugging Face cache, model outside
this repository, processed dataset, embeddings, index, or winning model was deleted.

## Scope and caveats

These findings are retrieval-only and do not evaluate STT errors, RAG generation, frontend behavior,
or a production Goa/Konkani document corpus. The current Hub revision has a validation-only Telugu
file, which is treated as a zero-shot holdout. Re-run the same leakage-safe protocol when the target
corpus and real voice-derived queries become available.
"""


def run_finalization(
    config: dict[str, Any], *, data_dir: str | Path | None = None
) -> dict[str, Any]:
    data = resolve_data_dir(config, data_dir)
    embedding_winner = _load_json("results/embedding_winner.json")
    chunking_winner = _load_json("results/chunking_winner.json")
    index_winner = _load_json("results/index_winner.json")
    identity = {
        "dataset": data.name,
        "embedding": embedding_winner["winner"],
        "model_revision": embedding_winner["model_revision"],
        "chunking": chunking_winner["strategy_config"],
        "index": index_winner["backend_config"],
        "normalization": config["retrieval"]["normalization_method"],
    }
    fingerprint = stable_fingerprint(identity)
    final_path = Path("results/final_test.json")
    if final_path.exists():
        result = _load_json(final_path)
        if result.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError(
                "Sealed test was already evaluated for a different configuration; refusing leakage"
            )
    else:
        corpus = list(read_jsonl(data / "test_corpus.jsonl"))
        queries = list(read_jsonl(data / "test_queries.jsonl"))
        qrels = qrels_by_query(read_jsonl(data / "test_qrels.jsonl"))
        strategy = chunking_winner["strategy_config"]
        chunk_path = data / "chunks" / f"final-test-{fingerprint}.jsonl"
        if not chunk_path.exists():
            write_jsonl(chunk_path, chunk_corpus(corpus, strategy))
        chunks = list(read_jsonl(chunk_path))
        corpus_vectors, query_vectors, embedding_stats = _prepare_test_embeddings(
            config,
            data,
            chunks,
            chunk_path,
            queries,
            embedding_winner["winner"],
            embedding_winner["model_revision"],
        )
        index_path = Path(config["cache"]["indexes"]) / "final" / fingerprint / "retriever.faiss"
        backend = run_faiss(
            index_winner["backend_config"],
            corpus_vectors,
            query_vectors,
            [str(query["query_id"]) for query in queries],
            [str(chunk["parent_id"]) for chunk in chunks],
            index_path,
            top_k=int(config["retrieval"]["top_k"]),
            oversample=int(config["retrieval"]["search_oversample"]),
            warmup_queries=int(config["retrieval"]["warmup_queries"]),
        )
        quality = evaluate_rankings(backend.rankings, qrels)
        language_quality = evaluate_rankings_by_language(backend.rankings, qrels)
        result = {
            "status": "ok",
            "sealed_test_evaluation": True,
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "experiment_fingerprint": fingerprint,
            "dataset_artifact": str(data),
            "split": "test",
            "seed": config["experiment"]["seed"],
            "model": embedding_winner["winner"],
            "model_revision": embedding_winner["model_revision"],
            "chunking": chunking_winner["winner"],
            "chunking_config": strategy,
            "backend": index_winner["winner"],
            "backend_config": index_winner["backend_config"],
            "normalization_method": config["retrieval"]["normalization_method"],
            "corpus_vectors": len(corpus_vectors),
            **quality,
            "language_metrics": language_quality,
            "language_count": len(language_quality),
            "query_embedding_p50_ms": embedding_stats["query_embedding_latency"]["p50_ms"],
            "corpus_embedding_time_ms": embedding_stats["corpus_embedding_time_ms"],
            "retrieval_mean_ms": backend.latency["mean_ms"],
            "retrieval_p50_ms": backend.latency["p50_ms"],
            "retrieval_p70_ms": backend.latency["p70_ms"],
            "retrieval_p95_ms": backend.latency["p95_ms"],
            "retrieval_p100_ms": backend.latency["p100_ms"],
            **backend.stats,
            "chunk_artifact": str(chunk_path),
            "embedding_cache_path": embedding_stats["cache_path"],
        }
        write_json(final_path, result)
    cleanup_path = Path("results/model_cleanup.json")
    if cleanup_path.exists():
        cleanup = _load_json(cleanup_path)
        if cleanup.get("winner") != embedding_winner["winner"]:
            raise RuntimeError("Stored model cleanup report belongs to another winner")
        expected_candidates = sorted(config["embedding_ablation"]["models"])
        if sorted(cleanup.get("candidates", [])) != expected_candidates:
            raise RuntimeError("Stored model cleanup report belongs to another candidate set")
    else:
        cleanup = cleanup_losing_models(
            config["cache"]["models"],
            winner=embedding_winner["winner"],
            candidates=list(config["embedding_ablation"]["models"]),
        )
        write_json(cleanup_path, cleanup)
    retriever_config = {
        "class": "hh_goa_rag.retriever.ParentFaissRetriever",
        "model": embedding_winner["winner"],
        "model_revision": embedding_winner["model_revision"],
        "model_cache_path": embedding_winner["metrics"]["model_cache_path"],
        "chunking": chunking_winner["strategy_config"],
        "index": index_winner["backend_config"],
        "index_artifact": result["index_artifact"],
        "chunk_artifact": result["chunk_artifact"],
        "normalization_method": config["retrieval"]["normalization_method"],
        "top_k": config["retrieval"]["top_k"],
        "search_oversample": config["retrieval"]["search_oversample"],
    }
    write_json("results/final_retriever_config.json", retriever_config)
    manifest = _load_json(data / "manifest.json")
    recommendation = _recommendation(
        result, embedding_winner, chunking_winner, index_winner, manifest, cleanup
    )
    Path("results/final_recommendation.md").write_text(
        recommendation, encoding="utf-8", newline="\n"
    )
    terminal_markdown = recommendation.replace("→", "->").replace("×", "x")
    print(terminal_markdown)
    return result
