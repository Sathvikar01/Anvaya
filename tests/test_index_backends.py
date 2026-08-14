from pathlib import Path

import numpy as np
import pytest

from hh_goa_rag.index_backends import run_chroma_local, run_faiss, run_qdrant_local


@pytest.mark.parametrize(
    "config",
    [
        {"index_type": "flat_ip"},
        {"index_type": "hnsw", "m": 8, "ef_construction": 40, "ef_search": 20},
        {"index_type": "ivf_flat", "nlist": 2, "nprobe": 2},
    ],
)
def test_faiss_backends_return_parent_rankings(
    tmp_path: Path, config: dict[str, int | str]
) -> None:
    corpus = np.asarray([[1, 0], [0.99, 0.01], [0, 1], [0.1, 0.9]], dtype=np.float32)
    result = run_faiss(
        config,
        corpus,
        np.asarray([[1, 0]], dtype=np.float32),
        ["q"],
        ["p1", "p1", "p2", "p2"],
        tmp_path / f"{config['index_type']}.faiss",
        top_k=2,
        oversample=2,
        warmup_queries=1,
    )
    assert result.rankings == {"q": ["p1", "p2"]}
    assert result.stats["index_size_bytes"] > 0
    assert result.latency["p100_ms"] >= 0


@pytest.mark.parametrize(
    ("runner", "config"),
    [
        (
            run_qdrant_local,
            {"search_mode": "exact"},
        ),
        (
            run_chroma_local,
            {"m": 8, "ef_construction": 40, "ef_search": 20, "num_threads": 1},
        ),
    ],
)
def test_local_vector_stores_use_cosine_and_parent_payloads(
    tmp_path: Path, runner: object, config: dict[str, int]
) -> None:
    corpus = np.asarray([[1, 0], [0.99, 0.01], [0, 1], [0.1, 0.9]], dtype=np.float32)
    result = runner(
        config,
        corpus,
        np.asarray([[1, 0]], dtype=np.float32),
        ["q"],
        ["p1", "p1", "p2", "p2"],
        tmp_path / "store",
        top_k=2,
        oversample=2,
        warmup_queries=1,
    )
    assert result.rankings == {"q": ["p1", "p2"]}
    assert result.stats["index_size_bytes"] > 0
