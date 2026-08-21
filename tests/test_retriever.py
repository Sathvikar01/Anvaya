from pathlib import Path

import faiss
import numpy as np

from hh_goa_rag.io import write_jsonl
from hh_goa_rag.retriever import ParentFaissRetriever


def test_persisted_retriever_returns_unique_parents(tmp_path: Path) -> None:
    vectors = np.asarray([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    index = faiss.IndexHNSWFlat(2, 4, faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)
    index_path = tmp_path / "index.faiss"
    faiss.write_index(index, str(index_path))
    chunk_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        chunk_path,
        [
            {"chunk_id": "c1", "parent_id": "p1", "text": "first"},
            {"chunk_id": "c2", "parent_id": "p1", "text": "duplicate parent"},
            {"chunk_id": "c3", "parent_id": "p2", "text": "second"},
        ],
    )
    retriever = ParentFaissRetriever.load(index_path, chunk_path, top_k=2, oversample=2)
    result = retriever.retrieve(np.asarray([1, 0], dtype=np.float32))
    assert [item.parent_id for item in result] == ["p1", "p2"]


def test_flat_ip2_normalizes_l2_distance_to_cosine_score(tmp_path: Path) -> None:
    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    index_path = tmp_path / "flat-ip2.faiss"
    faiss.write_index(index, str(index_path))
    chunk_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        chunk_path,
        [
            {"chunk_id": "c1", "parent_id": "p1", "text": "first"},
            {"chunk_id": "c2", "parent_id": "p2", "text": "second"},
        ],
    )

    result = ParentFaissRetriever.load(index_path, chunk_path, top_k=2).retrieve(
        np.asarray([1, 0], dtype=np.float32)
    )

    assert [item.parent_id for item in result] == ["p1", "p2"]
    assert result[0].score == 1.0
    assert result[1].score == 0.0


def test_retriever_prefers_substantive_chunk_for_parent(tmp_path: Path) -> None:
    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1, 0], [0.99, 0.01]], dtype=np.float32))
    index_path = tmp_path / "flat-ip2.faiss"
    faiss.write_index(index, str(index_path))
    chunk_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        chunk_path,
        [
            {"chunk_id": "tail", "parent_id": "p1", "text": "है।"},
            {
                "chunk_id": "body",
                "parent_id": "p1",
                "text": "सिरियस एक्सएम एक ऑनलाइन समाचार वेबसाइट है।",
            },
        ],
    )

    result = ParentFaissRetriever.load(index_path, chunk_path, top_k=1).retrieve(
        np.asarray([1, 0], dtype=np.float32)
    )

    assert result[0].chunk_id == "body"
