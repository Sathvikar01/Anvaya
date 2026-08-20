from pathlib import Path

import numpy as np

from hh_goa_rag.retrieval import build_flat_ip, l2_normalize, search_parent_rankings


def test_flat_ip_search_deduplicates_parent_chunks(tmp_path: Path) -> None:
    corpus = np.asarray([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    index, stats = build_flat_ip(corpus, tmp_path / "index.faiss")
    rankings, latency = search_parent_rankings(
        index,
        np.asarray([[1, 0]], dtype=np.float32),
        ["q"],
        ["p1", "p1", "p2"],
        top_k=2,
        oversample=2,
        warmup_queries=1,
    )
    assert rankings == {"q": ["p1", "p2"]}
    assert stats["index_size_bytes"] > 0
    assert latency["p100_ms"] >= 0


def test_float32_normalization_corrects_low_precision_norms() -> None:
    vectors = np.asarray([[2.0, 0.0], [0.0, 0.25]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-6)
