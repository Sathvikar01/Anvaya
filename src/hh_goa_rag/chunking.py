"""Deterministic passage chunking with stable parent/child identities."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np

from hh_goa_rag.config import stable_fingerprint

WORD_PATTERN = re.compile(r"\S+", re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])\s+", re.UNICODE)


def fixed_word_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be non-negative and smaller than size")
    words = WORD_PATTERN.findall(text)
    if not words:
        return []
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def sentence_chunks(text: str, max_words: int) -> list[str]:
    """Pack punctuation-delimited sentences without exceeding a word budget."""
    units: list[str] = []
    for sentence in split_sentences(text):
        units.extend(fixed_word_chunks(sentence, max_words))
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in units:
        count = len(WORD_PATTERN.findall(sentence))
        if current and current_words + count > max_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += count
    if current:
        chunks.append(" ".join(current))
    return chunks


def semantic_chunks(
    text: str,
    sentence_vectors: np.ndarray,
    *,
    min_words: int,
    max_words: int,
    similarity_threshold: float,
) -> list[str]:
    """Group adjacent sentences while their normalized embedding similarity stays high."""
    sentences = split_sentences(text)
    if len(sentences) != len(sentence_vectors):
        raise ValueError("sentence vector count does not match segmented text")
    if not sentences:
        return []
    chunks: list[str] = []
    current = sentences[0]
    current_words = len(WORD_PATTERN.findall(current))
    for index, sentence in enumerate(sentences[1:], start=1):
        sentence_words = len(WORD_PATTERN.findall(sentence))
        similarity = float(np.dot(sentence_vectors[index - 1], sentence_vectors[index]))
        semantic_break = current_words >= min_words and similarity < similarity_threshold
        size_break = current_words + sentence_words > max_words
        if semantic_break or size_break:
            chunks.extend(fixed_word_chunks(current, max_words))
            current, current_words = sentence, sentence_words
        else:
            current = f"{current} {sentence}"
            current_words += sentence_words
    chunks.extend(fixed_word_chunks(current, max_words))
    return chunks


def chunk_corpus(
    corpus: Iterable[dict[str, Any]],
    strategy: dict[str, Any],
    *,
    semantic_vectors: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Apply a configured chunker and retain stable parent passage IDs."""
    name = strategy["strategy"]
    chunks: list[dict[str, Any]] = []
    for parent in corpus:
        granularities: list[tuple[str, str]]
        if name == "fixed_words":
            texts = fixed_word_chunks(
                parent["text"], int(strategy["size"]), int(strategy.get("overlap", 0))
            )
            granularities = [("chunk", text) for text in texts]
        elif name == "sentence":
            granularities = [
                ("sentence_group", text)
                for text in sentence_chunks(parent["text"], int(strategy["max_words"]))
            ]
        elif name == "semantic":
            if semantic_vectors is None or parent["passage_id"] not in semantic_vectors:
                raise ValueError("semantic chunking requires vectors for every parent passage")
            texts = semantic_chunks(
                parent["text"],
                semantic_vectors[parent["passage_id"]],
                min_words=int(strategy["min_words"]),
                max_words=int(strategy["max_words"]),
                similarity_threshold=float(strategy["similarity_threshold"]),
            )
            granularities = [("semantic_group", text) for text in texts]
        elif name == "parent_child":
            children = fixed_word_chunks(
                parent["text"],
                int(strategy["child_size"]),
                int(strategy["child_overlap"]),
            )
            granularities = [("child", text) for text in children]
            if bool(strategy.get("include_parent", True)):
                granularities.insert(0, ("parent", parent["text"]))
        else:
            raise ValueError(f"Unsupported chunking strategy: {name}")
        for position, (granularity, text) in enumerate(granularities):
            identity = {
                "parent_id": parent["passage_id"],
                "position": position,
                "granularity": granularity,
                "strategy": strategy,
                "text": text,
            }
            chunk = {
                "chunk_id": f"c-{stable_fingerprint(identity, 24)}",
                "parent_id": parent["passage_id"],
                "position": position,
                "granularity": granularity,
                "text": text,
            }
            if parent.get("language"):
                chunk["language"] = str(parent["language"])
            chunks.append(chunk)
    return chunks
