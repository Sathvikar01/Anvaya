"""Comparable local vector-index builders and parent-level search runners."""

from __future__ import annotations

import gc
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import psutil

from hh_goa_rag.metrics import latency_percentiles
from hh_goa_rag.retrieval import search_parent_rankings


@dataclass
class BackendRun:
    rankings: dict[str, list[str]]
    latency: dict[str, float]
    stats: dict[str, int | float | str]


def directory_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def _unique_parents(candidates: list[str], top_k: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for parent_id in candidates:
        if parent_id not in seen:
            seen.add(parent_id)
            result.append(parent_id)
        if len(result) == top_k:
            break
    return result


def _rss() -> int:
    gc.collect()
    return psutil.Process().memory_info().rss


def run_faiss(
    config: dict[str, Any],
    vectors: np.ndarray,
    queries: np.ndarray,
    query_ids: list[str],
    parent_ids: list[str],
    path: str | Path,
    *,
    top_k: int,
    oversample: int,
    warmup_queries: int,
) -> BackendRun:
    corpus = np.ascontiguousarray(vectors, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rss_before = _rss()
    started = time.perf_counter_ns()
    index_type = config["index_type"]
    if index_type == "flat_ip":
        index: faiss.Index = faiss.IndexFlatIP(corpus.shape[1])
    elif index_type == "flat_l2":
        # Exact flat L2 is the second flat baseline. With unit-normalized
        # vectors, ranking is mathematically equivalent to inner product.
        index = faiss.IndexFlatL2(corpus.shape[1])
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(
            corpus.shape[1], int(config["m"]), faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = int(config["ef_construction"])
        index.hnsw.efSearch = int(config["ef_search"])
    elif index_type == "ivf_flat":
        quantizer = faiss.IndexFlatIP(corpus.shape[1])
        index = faiss.IndexIVFFlat(
            quantizer, corpus.shape[1], int(config["nlist"]), faiss.METRIC_INNER_PRODUCT
        )
        index.train(corpus)
        index.nprobe = int(config["nprobe"])
    elif index_type == "ivf_pq":
        quantizer = faiss.IndexFlatIP(corpus.shape[1])
        index = faiss.IndexIVFPQ(
            quantizer,
            corpus.shape[1],
            int(config["nlist"]),
            int(config["pq_m"]),
            int(config["pq_nbits"]),
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(corpus)
        index.nprobe = int(config["nprobe"])
    else:
        raise ValueError(f"Unsupported FAISS index type: {index_type}")
    index.add(corpus)
    faiss.write_index(index, str(path))
    indexing_ms = (time.perf_counter_ns() - started) / 1e6
    rss_after = _rss()
    rankings, latency = search_parent_rankings(
        index,
        queries,
        query_ids,
        parent_ids,
        top_k=top_k,
        oversample=oversample,
        warmup_queries=warmup_queries,
    )
    size = path.stat().st_size
    return BackendRun(
        rankings,
        latency,
        {
            "indexing_time_ms": indexing_ms,
            "index_size_bytes": size,
            "index_serialized_bytes": size,
            "index_ram_bytes": max(0, rss_after - rss_before),
            "index_vector_bytes": int(corpus.nbytes),
            "process_rss_delta_bytes": max(0, rss_after - rss_before),
            "index_artifact": str(path),
        },
    )


def run_qdrant_local(
    config: dict[str, Any],
    vectors: np.ndarray,
    queries: np.ndarray,
    query_ids: list[str],
    parent_ids: list[str],
    path: str | Path,
    *,
    top_k: int,
    oversample: int,
    warmup_queries: int,
) -> BackendRun:
    from qdrant_client import QdrantClient, models

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    collection_name = "hh_goa_chunks"
    client = QdrantClient(path=str(path))
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    rss_before = _rss()
    started = time.perf_counter_ns()
    client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(
            size=int(vectors.shape[1]), distance=models.Distance.COSINE
        ),
    )
    client.upload_collection(
        collection_name,
        vectors=np.ascontiguousarray(vectors, dtype=np.float32),
        payload=({"parent_id": parent_id} for parent_id in parent_ids),
        ids=range(len(parent_ids)),
        batch_size=256,
        parallel=1,
        wait=True,
    )
    indexing_ms = (time.perf_counter_ns() - started) / 1e6
    rss_after = _rss()
    search_k = min(len(parent_ids), max(top_k, top_k * oversample))

    def search(query: np.ndarray) -> list[str]:
        response = client.query_points(
            collection_name,
            query=query,
            limit=search_k,
            with_payload=["parent_id"],
            with_vectors=False,
            search_params=models.SearchParams(exact=True)
            if config.get("search_mode") == "exact"
            else None,
        )
        return [str(point.payload["parent_id"]) for point in response.points]

    for query in queries[: min(warmup_queries, len(queries))]:
        search(query)
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query_id, query in zip(query_ids, queries, strict=True):
        started = time.perf_counter_ns()
        candidates = search(query)
        latencies.append((time.perf_counter_ns() - started) / 1e6)
        rankings[query_id] = _unique_parents(candidates, top_k)
    client.close()
    size = directory_size(path)
    return BackendRun(
        rankings,
        latency_percentiles(latencies),
        {
            "indexing_time_ms": indexing_ms,
            "index_size_bytes": size,
            "index_serialized_bytes": size,
            "index_ram_bytes": max(0, rss_after - rss_before),
            "index_vector_bytes": int(np.asarray(vectors, dtype=np.float32).nbytes),
            "process_rss_delta_bytes": max(0, rss_after - rss_before),
            "index_artifact": str(path),
        },
    )


def run_chroma_local(
    config: dict[str, Any],
    vectors: np.ndarray,
    queries: np.ndarray,
    query_ids: list[str],
    parent_ids: list[str],
    path: str | Path,
    *,
    top_k: int,
    oversample: int,
    warmup_queries: int,
) -> BackendRun:
    import chromadb
    from chromadb.config import Settings

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    collection_name = "hh_goa_chunks"
    with suppress(chromadb.errors.NotFoundError):
        client.delete_collection(collection_name)
    rss_before = _rss()
    started = time.perf_counter_ns()
    collection = client.create_collection(
        collection_name,
        configuration={
            "hnsw": {
                "space": "cosine",
                "ef_construction": int(config["ef_construction"]),
                "ef_search": int(config["ef_search"]),
                "max_neighbors": int(config["m"]),
                "num_threads": int(config["num_threads"]),
            }
        },
        embedding_function=None,
    )
    batch_size = min(5000, client.get_max_batch_size())
    for offset in range(0, len(parent_ids), batch_size):
        stop = min(offset + batch_size, len(parent_ids))
        collection.add(
            ids=[str(position) for position in range(offset, stop)],
            embeddings=np.ascontiguousarray(vectors[offset:stop], dtype=np.float32),
            metadatas=[{"parent_id": parent_id} for parent_id in parent_ids[offset:stop]],
        )
    indexing_ms = (time.perf_counter_ns() - started) / 1e6
    rss_after = _rss()
    search_k = min(len(parent_ids), max(top_k, top_k * oversample))

    def search(query: np.ndarray) -> list[str]:
        result = collection.query(
            query_embeddings=query.reshape(1, -1),
            n_results=search_k,
            include=["metadatas"],
        )
        metadata = result["metadatas"][0]
        return [str(item["parent_id"]) for item in metadata]

    for query in queries[: min(warmup_queries, len(queries))]:
        search(query)
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query_id, query in zip(query_ids, queries, strict=True):
        started = time.perf_counter_ns()
        candidates = search(query)
        latencies.append((time.perf_counter_ns() - started) / 1e6)
        rankings[query_id] = _unique_parents(candidates, top_k)
    size = directory_size(path)
    return BackendRun(
        rankings,
        latency_percentiles(latencies),
        {
            "indexing_time_ms": indexing_ms,
            "index_size_bytes": size,
            "index_serialized_bytes": size,
            "index_ram_bytes": max(0, rss_after - rss_before),
            "index_vector_bytes": int(np.asarray(vectors, dtype=np.float32).nbytes),
            "process_rss_delta_bytes": max(0, rss_after - rss_before),
            "index_artifact": str(path),
        },
    )
