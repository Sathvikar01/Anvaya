"""Development-selected retrieval sufficiency signals for the frozen Top-10 retriever."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .types import ReasonCode

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:  # pragma: no cover - optional outside the web app extra
    sanscript = None
    transliterate = None

TOP_SCORE_THRESHOLD = 0.67
CONSISTENCY_RESCUE_FLOOR = 0.64
TOP_TWO_MAX_GAP = 0.005
TOP_TO_FIFTH_MIN_SPREAD = 0.12
QUERY_CORROBORATION_FLOOR = 0.50
QUERY_CORROBORATION_MEAN_FLOOR = 0.50
QUERY_CORROBORATION_CONTEXTS = 2
# Above this dense-similarity floor the evidence is plausible enough that the
# grounded generator - which can still answer INSUFFICIENT_CONTEXT under the
# exact output schema - should judge sufficiency instead of refusing here.
BORDERLINE_REVIEW_FLOOR = 0.45
_QUERY_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "और",
    "का",
    "की",
    "के",
    "को",
    "में",
    "से",
    "है",
    "हैं",
    "यह",
    "एक",
}


@dataclass(frozen=True)
class RetrievalSignals:
    sufficient: bool
    reason_code: ReasonCode | None
    top_score: float | None
    top_two_gap: float | None
    top_to_fifth_spread: float | None
    top_three_mean: float | None
    decision_rule: str

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "sufficient": self.sufficient,
            "reason_code": self.reason_code.value if self.reason_code else None,
            "top_score": self.top_score,
            "top_two_gap": self.top_two_gap,
            "top_to_fifth_spread": self.top_to_fifth_spread,
            "top_three_mean": self.top_three_mean,
            "decision_rule": self.decision_rule,
        }


def evidence_sufficiency(
    contexts: Sequence[Any],
    *,
    query: str | None = None,
    language_code: str | None = None,
    top_score_threshold: float = TOP_SCORE_THRESHOLD,
) -> RetrievalSignals:
    # Hybrid retrieval order is not necessarily descending by dense score.
    # Keep the development-selected semantic thresholds on their original
    # signal distribution by computing them from sorted dense similarities.
    scores = sorted((float(_field(context, "score")) for context in contexts), reverse=True)
    if not scores:
        return RetrievalSignals(False, ReasonCode.RETRIEVAL_EMPTY, None, None, None, None, "empty")
    top = scores[0]
    gap = top - scores[1] if len(scores) >= 2 else None
    spread = top - scores[4] if len(scores) >= 5 else None
    mean3 = sum(scores[:3]) / min(3, len(scores))
    overlap_count = _query_overlap_count(query, contexts) if query is not None else 0
    requested_language = _language_key(language_code)
    if requested_language and requested_language != "hi":
        # The frozen 0.67 threshold was selected on the original Hindi demo
        # distribution. Dense similarities are not calibrated across scripts,
        # so applying that number to another requested language creates false
        # refusals before the multilingual generator can assess the evidence.
        return RetrievalSignals(
            True,
            None,
            top,
            gap,
            spread,
            mean3,
            "multilingual_generation_review",
        )
    if query is not None and overlap_count == 0:
        if top >= BORDERLINE_REVIEW_FLOOR:
            return RetrievalSignals(
                True,
                None,
                top,
                gap,
                spread,
                mean3,
                "borderline_generation_review",
            )
        return RetrievalSignals(
            False,
            ReasonCode.RETRIEVAL_LOW_CONFIDENCE,
            top,
            gap,
            spread,
            mean3,
            "no_query_term_overlap",
        )
    if top >= top_score_threshold:
        return RetrievalSignals(True, None, top, gap, spread, mean3, "top_score")
    corroborated = top >= CONSISTENCY_RESCUE_FLOOR and gap is not None and gap <= TOP_TWO_MAX_GAP
    if corroborated:
        return RetrievalSignals(True, None, top, gap, spread, mean3, "top_two_corroboration")
    dominant = (
        top >= CONSISTENCY_RESCUE_FLOOR and spread is not None and spread >= TOP_TO_FIFTH_MIN_SPREAD
    )
    if dominant:
        return RetrievalSignals(True, None, top, gap, spread, mean3, "top_to_fifth_spread")
    multilingual_corroborated = (
        query is not None
        and overlap_count is not None
        and overlap_count >= QUERY_CORROBORATION_CONTEXTS
        and top >= QUERY_CORROBORATION_FLOOR
        and mean3 >= QUERY_CORROBORATION_MEAN_FLOOR
    )
    if multilingual_corroborated:
        return RetrievalSignals(True, None, top, gap, spread, mean3, "query_term_corroboration")
    if top >= BORDERLINE_REVIEW_FLOOR:
        return RetrievalSignals(
            True,
            None,
            top,
            gap,
            spread,
            mean3,
            "borderline_generation_review",
        )
    return RetrievalSignals(
        False,
        ReasonCode.RETRIEVAL_LOW_CONFIDENCE,
        top,
        gap,
        spread,
        mean3,
        "below_threshold_without_consistency_rescue",
    )


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name)


def _query_has_evidence_overlap(query: str, contexts: Sequence[Any]) -> bool:
    return bool(_query_overlap_count(query, contexts) or 0)


def _query_overlap_count(query: str | None, contexts: Sequence[Any]) -> int | None:
    if query is None:
        return None

    def tokens(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return {
            token
            for token in _QUERY_TOKEN_RE.findall(normalized)
            if token not in _QUERY_STOPWORDS and len(token) > 1
        }

    query_terms = tokens(query)
    if not query_terms:
        return None

    def context_matches(context: Any) -> bool:
        evidence_terms = tokens(str(_field(context, "text")))
        if query_terms & evidence_terms:
            return True

        # Hindi/Indic KB text often joins or separates the same named entity
        # differently (for example, ``सिरियस एक्सएम`` vs ``सिरियसएक्सएम``).
        # Compare a compact form as a secondary check without weakening the
        # primary token-overlap guardrail.
        compact_query = "".join(query_terms)
        compact_evidence = "".join(evidence_terms)
        if len(compact_query) >= 4 and compact_query in compact_evidence:
            return True

        if transliterate is None or sanscript is None:
            return False
        try:
            romanized = transliterate(
                str(_field(context, "text")),
                sanscript.DEVANAGARI,
                sanscript.ITRANS,
            )
        except (TypeError, ValueError):
            return False

        romanized_terms = tokens(romanized)
        for query_term in query_terms:
            if len(query_term) < 5:
                continue
            query_skeleton = _phonetic_skeleton(query_term)
            if len(query_skeleton) < 3:
                continue
            for evidence_term in romanized_terms:
                evidence_skeleton = _phonetic_skeleton(evidence_term)
                if len(evidence_skeleton) < 3:
                    continue
                if SequenceMatcher(None, query_skeleton, evidence_skeleton).ratio() >= 0.66:
                    return True
        return False

    return sum(context_matches(context) for context in contexts[:5])


def _phonetic_skeleton(value: str) -> str:
    """Keep consonant shapes for cross-script transliterated entity matching."""

    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value).casefold()
        if character.isalpha() and character not in "aeiouy"
    )


def language_key(language_code: str | None) -> str | None:
    if not language_code:
        return None
    key = language_code.split("-", 1)[0].strip().lower()
    return {"od": "or"}.get(key, key) or None


_language_key = language_key
