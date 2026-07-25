"""Reusable parent-level FAISS retriever for the later Voice → STT → RAG pipeline."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from hh_goa_rag.io import read_jsonl
from hh_goa_rag.retrieval import l2_normalize


@dataclass(frozen=True)
class RetrievedParent:
    parent_id: str
    score: float
    chunk_id: str
    text: str
    language: str | None = None
    normalized_text: str = field(default="", repr=False, compare=False)


_LANGUAGE_ALIASES = {"od": "or"}
# UI languages without their own MSMARCO-XI content pivot to the
# linguistically closest available partition instead of searching every
# script in the corpus and surfacing random-language passages.
_LANGUAGE_PIVOTS = {
    "kok": ("mr", "hi"),
    "ks": ("ur", "hi"),
    "sd": ("ur", "hi"),
    "mai": ("hi",),
}
_DENSE_WEIGHT = 0.40
_LEXICAL_WEIGHT = 1.0 - _DENSE_WEIGHT
_RAW_CHUNK_MULTIPLIER = 5


class ParentFaissRetriever:
    """Load a persisted FAISS index and return unique parent passages."""

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[dict[str, str]],
        *,
        top_k: int = 10,
        oversample: int = 20,
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunk mapping have different lengths")
        self.index = index
        self.chunks = chunks
        self.top_k = top_k
        self.oversample = oversample
        self._normalized_texts = [_normalize_text(str(chunk["text"])) for chunk in chunks]
        self._language_indexes = self._build_language_indexes()

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        chunk_path: str | Path,
        *,
        top_k: int = 10,
        oversample: int = 20,
    ) -> ParentFaissRetriever:
        return cls(
            faiss.read_index(str(index_path)),
            list(read_jsonl(chunk_path)),
            top_k=top_k,
            oversample=oversample,
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        *,
        query_text: str | None = None,
        language_code: str | None = None,
    ) -> list[RetrievedParent]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query = l2_normalize(query)
        language = _language_key(language_code)
        language_index = self._language_indexes.get(language) if language else None
        pivot_note: str | None = None
        if language_index is None and language:
            # UI languages without corpus coverage (Konkani, Kashmiri, Sindhi)
            # pivot to their linguistically closest available partition instead
            # of searching the full multilingual index and surfacing random
            # scripts. The pivot is recorded so responses can disclose it.
            for pivot in _LANGUAGE_PIVOTS.get(language, ()):  # ordered preference
                candidate = self._language_indexes.get(pivot)
                if candidate is not None:
                    language_index = candidate
                    pivot_note = f"{language}->{pivot}"
                    break
        active_index: faiss.Index = self.index
        source_positions: np.ndarray | None = None
        if language_index is not None:
            active_index, source_positions = language_index
        hybrid_fallback = bool(query_text) and language_index is None
        search_k = min(
            active_index.ntotal,
            self.top_k
            * self.oversample
            * (_RAW_CHUNK_MULTIPLIER if hybrid_fallback else 1),
        )
        scores, positions = active_index.search(query, search_k)
        best_by_parent: dict[str, RetrievedParent] = {}
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            source_position = (
                int(source_positions[int(position)])
                if source_positions is not None
                else int(position)
            )
            chunk = self.chunks[source_position]
            parent_id = str(chunk["parent_id"])
            # `faiss_flat_ip2` is the exact IndexFlatL2 baseline. Because the
            # stored/query vectors are unit-normalized, squared L2 distance is
            # equivalent to cosine similarity via cosine = 1 - distance / 2.
            normalized_score = (
                1.0 - float(score) / 2.0
                if self.index.metric_type == faiss.METRIC_L2
                else float(score)
            )
            candidate = RetrievedParent(
                parent_id=parent_id,
                score=normalized_score,
                chunk_id=str(chunk["chunk_id"]),
                text=str(chunk["text"]),
                language=str(chunk["language"]) if chunk.get("language") else None,
                normalized_text=self._normalized_texts[source_position],
            )
            current = best_by_parent.get(parent_id)
            if current is None or _prefer_chunk(candidate, current):
                best_by_parent[parent_id] = candidate
        candidates = sorted(best_by_parent.values(), key=lambda item: item.score, reverse=True)[
            : self.top_k * self.oversample
        ]
        if hybrid_fallback and len(candidates) > self.top_k:
            candidates = _hybrid_rerank(query_text, candidates)
        return candidates[: self.top_k]

    def _build_language_indexes(self) -> dict[str, tuple[faiss.Index, np.ndarray]]:
        """Build tiny exact sub-indexes when language metadata is available."""

        positions_by_language: dict[str, list[int]] = {}
        for position, chunk in enumerate(self.chunks):
            language = str(chunk.get("language") or "").strip().lower()
            if language:
                positions_by_language.setdefault(language, []).append(position)
        if not positions_by_language:
            return {}
        try:
            vectors = np.empty((self.index.ntotal, self.index.d), dtype=np.float32)
            self.index.reconstruct_n(0, self.index.ntotal, vectors)
        except (RuntimeError, TypeError):
            return {}
        result: dict[str, tuple[faiss.Index, np.ndarray]] = {}
        for language, positions in positions_by_language.items():
            source_positions = np.asarray(positions, dtype=np.int64)
            subindex = faiss.IndexFlatIP(self.index.d)
            subindex.add(np.ascontiguousarray(vectors[source_positions], dtype=np.float32))
            result[language] = (subindex, source_positions)
        return result


_CHUNK_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_text(value: str) -> str:
    return " ".join(_CHUNK_TOKEN_RE.findall(unicodedata.normalize("NFKC", value).casefold()))


def _trigrams(value: str) -> set[str]:
    normalized = _normalize_text(value)
    return {
        normalized[index : index + 3]
        for index in range(max(0, len(normalized) - 2))
        if normalized[index : index + 3].strip()
    }


def _language_key(language_code: str | None) -> str | None:
    if not language_code:
        return None
    key = language_code.split("-", 1)[0].strip().lower()
    return _LANGUAGE_ALIASES.get(key, key) or None


def _hybrid_rerank(
    query_text: str, candidates: list[RetrievedParent]
) -> list[RetrievedParent]:
    query_trigrams = _trigrams(query_text)
    if not query_trigrams:
        return candidates
    dense_scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float32)
    span = float(dense_scores.max() - dense_scores.min())
    if span > 0:
        dense_scores = (dense_scores - dense_scores.min()) / span
    else:
        dense_scores.fill(1.0)
    lexical_scores = np.asarray(
        [
            sum(
                trigram in (candidate.normalized_text or _normalize_text(candidate.text))
                for trigram in query_trigrams
            )
            / len(query_trigrams)
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    combined = _DENSE_WEIGHT * dense_scores + _LEXICAL_WEIGHT * lexical_scores
    order = np.argsort(-combined, kind="stable")
    return [candidates[int(index)] for index in order]


def _prefer_chunk(candidate: RetrievedParent, current: RetrievedParent) -> bool:
    """Avoid returning a semantically empty trailing chunk for a parent.

    Fixed-size chunking can leave a final fragment such as ``"है।"``.  That
    fragment can win vector similarity while containing none of the evidence
    needed by the answer generator.  Prefer a substantive candidate for the
    same parent, then use similarity as the tie-breaker.
    """

    candidate_substantive = len(_CHUNK_TOKEN_RE.findall(candidate.text)) >= 5
    current_substantive = len(_CHUNK_TOKEN_RE.findall(current.text)) >= 5
    if candidate_substantive != current_substantive:
        return candidate_substantive
    return candidate.score > current.score
