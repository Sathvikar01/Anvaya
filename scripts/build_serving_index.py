"""Build the multilingual BGE-M3 + FAISS HNSW serving artifact from both KB splits."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import torch

from hh_goa_rag.chunking import chunk_corpus
from hh_goa_rag.harness import FROZEN_RETRIEVAL, FROZEN_TOP_K
from hh_goa_rag.io import read_jsonl, write_json, write_jsonl
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel

ROOT = Path("data/processed/2c0dcd7c6bc9f61e")
MODEL_CACHE = Path("cache/models/BAAI__bge-m3--5617a9f61b02")
CHUNK_PATH = ROOT / "chunks/serving-combined-fixed-128-bge-m3.jsonl"
INDEX_PATH = Path("cache/indexes/serving-combined/faiss_hnsw.faiss")
CONFIG_PATH = Path("results/final_retriever_config.json")


def main() -> None:
    corpus = list(read_jsonl(ROOT / "dev_corpus.jsonl")) + list(
        read_jsonl(ROOT / "test_corpus.jsonl")
    )
    chunks = chunk_corpus(corpus, {"strategy": "fixed_words", "size": 128, "overlap": 0})
    write_jsonl(CHUNK_PATH, chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    encoder = EmbeddingModel(
        MODEL_SPECS[FROZEN_RETRIEVAL["model"]],
        MODEL_CACHE,
        device=device,
        max_sequence_length=512,
        dtype=dtype,
    )
    try:
        encoder.warm_up(
            "query: पीले पत्थर के पार्क केबिन की लागत",
            "passage: कैंपिंग केबिन की कीमतें 25 से 44 डॉलर के बीच होती हैं।",
            rounds=1,
        )
        vectors, embedding_ms = encoder.encode_corpus(
            [str(chunk["text"]) for chunk in chunks], batch_size=32
        )
    finally:
        encoder.close()

    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    write_json(
        CONFIG_PATH,
        {
            "class": "hh_goa_rag.retriever.ParentFaissRetriever",
            "model": FROZEN_RETRIEVAL["model"],
            "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
            "model_cache_path": str(MODEL_CACHE),
            "chunk_artifact": str(CHUNK_PATH),
            "index_artifact": str(INDEX_PATH),
            "chunking": {
                "name": "fixed_words",
                "strategy": "fixed_words",
                "max_words": 128,
            },
            "index": {
                "name": "faiss_hnsw",
                "engine": "faiss",
                "index_type": "hnsw",
                "m": 32,
                "ef_construction": 200,
                "ef_search": 128,
            },
            "normalization_method": "float32_l2_v1",
            "search_oversample": 20,
            "top_k": FROZEN_TOP_K,
            "serving_corpus": "dev_plus_test",
            "corpus_chunks": len(chunks),
            "corpus_embedding_ms": embedding_ms,
        },
    )
    print(
        json.dumps(
            {
                "chunks": len(chunks),
                "index_vectors": index.ntotal,
                "embedding_ms": embedding_ms,
                "device": device,
                "chunk_artifact": str(CHUNK_PATH),
                "index_artifact": str(INDEX_PATH),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
