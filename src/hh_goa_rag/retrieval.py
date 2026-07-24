"""Shared exact cosine retrieval and parent-level result mapping."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import faiss
import numpy as np
import psutil

from hh_goa_rag.metrics import latency_percentiles


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Return contiguous float32 unit vectors, including after low-precision inference."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero embedding")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def build_flat_ip(
    embeddings: np.ndarray, index_path: str | Path
) -> tuple[faiss.IndexFlatIP, dict[str, int | float]]:
    vectors = l2_normalize(embeddings)
    gc.collect()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter_ns()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    indexing_ms = (time.perf_counter_ns() - started) / 1e6
    rss_after = process.memory_info().rss
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    return index, {
        "indexing_time_ms": indexing_ms,
        "index_size_bytes": index_path.stat().st_size,
        "index_serialized_bytes": index_path.stat().st_size,
        "index_ram_bytes": max(0, rss_after - rss_before),
        "index_vector_bytes": int(vectors.nbytes),
        "process_rss_delta_bytes": max(0, rss_after - rss_before),
    }


def search_parent_rankings(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    chunk_parent_ids: list[str],
    *,
    top_k: int,
    oversample: int,
    warmup_queries: int,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    search_k = min(index.ntotal, max(top_k, top_k * oversample))
    queries = l2_normalize(query_embeddings)
    for query in queries[: min(warmup_queries, len(queries))]:
        index.search(query.reshape(1, -1), search_k)
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query_id, query in zip(query_ids, queries, strict=True):
        started = time.perf_counter_ns()
        _scores, positions = index.search(query.reshape(1, -1), search_k)
        latencies.append((time.perf_counter_ns() - started) / 1e6)
        unique_parents: list[str] = []
        seen: set[str] = set()
        for position in positions[0]:
            if position < 0:
                continue
            parent_id = chunk_parent_ids[int(position)]
            if parent_id not in seen:
                seen.add(parent_id)
                unique_parents.append(parent_id)
            if len(unique_parents) == top_k:
                break
        rankings[query_id] = unique_parents
    return rankings, latency_percentiles(latencies)
