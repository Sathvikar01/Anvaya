import numpy as np

from hh_goa_rag.chunking import (
    chunk_corpus,
    fixed_word_chunks,
    semantic_chunks,
    sentence_chunks,
    split_sentences,
)


def test_fixed_word_chunks_support_overlap() -> None:
    assert fixed_word_chunks("a b c d e", size=3, overlap=1) == ["a b c", "c d e", "e"]


def test_chunk_ids_are_stable_and_keep_parent() -> None:
    corpus = [{"passage_id": "p1", "text": "one two three"}]
    strategy = {"strategy": "fixed_words", "size": 2, "overlap": 0}
    first = chunk_corpus(corpus, strategy)
    second = chunk_corpus(corpus, strategy)
    assert first == second
    assert [chunk["parent_id"] for chunk in first] == ["p1", "p1"]


def test_hindi_sentence_segmentation_and_packing() -> None:
    text = "यह पहला वाक्य है। यह दूसरा है! तीसरा वाक्य?"
    assert len(split_sentences(text)) == 3
    assert len(sentence_chunks(text, max_words=4)) == 3


def test_semantic_chunks_break_on_low_similarity() -> None:
    text = "one two. three four. five six."
    vectors = np.asarray([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    chunks = semantic_chunks(
        text, vectors, min_words=2, max_words=10, similarity_threshold=0.5
    )
    assert chunks == ["one two. three four.", "five six."]


def test_parent_child_includes_both_granularities() -> None:
    corpus = [{"passage_id": "p1", "text": "one two three four five"}]
    strategy = {
        "name": "parent_child",
        "strategy": "parent_child",
        "child_size": 3,
        "child_overlap": 1,
        "include_parent": True,
    }
    chunks = chunk_corpus(corpus, strategy)
    assert chunks[0]["granularity"] == "parent"
    assert {chunk["granularity"] for chunk in chunks[1:]} == {"child"}
