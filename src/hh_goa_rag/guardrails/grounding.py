"""Deterministic validation of schema-constrained generation output."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import ReasonCode, Route

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_GROUNDING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
    "और", "का", "की", "के", "को", "में", "से", "है", "हैं", "यह", "एक",
}


@dataclass(frozen=True)
class GroundingDecision:
    valid: bool
    route: Route
    reason_code: ReasonCode
    answer: str
    citations: tuple[str, ...]


def validate_generation(result: Any, contexts: Sequence[Any]) -> GroundingDecision:
    allowed_ids = {str(_field(context, "parent_id", "")) for context in contexts}
    transport_status = str(_field(result, "status", "error"))
    if transport_status != "ok":
        error_code = str(_field(result, "error_code", ""))
        reason = (
            ReasonCode.GENERATION_SCHEMA_INVALID
            if error_code == "invalid_structured_output"
            else ReasonCode.GENERATION_PROVIDER_ERROR
        )
        return GroundingDecision(False, Route.SYSTEM_ERROR, reason, "", ())

    answer_status = str(_field(result, "answer_status", ""))
    answer = str(_field(result, "answer", "") or "").strip()
    evidence_ids = tuple(dict.fromkeys(str(item) for item in _field(result, "evidence_ids", ())))
    raw_output = str(_field(result, "raw_output", "") or "")
    if raw_output:
        raw = _parse_raw(raw_output)
        if raw is None or (
            raw["status"] != answer_status
            or raw["answer"].strip() != answer
            or tuple(dict.fromkeys(raw["evidence_ids"])) != evidence_ids
        ):
            return GroundingDecision(
                False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_SCHEMA_INVALID, "", ()
            )
    diagnostics = _field(result, "diagnostics", {}) or {}
    if diagnostics.get("schema_valid") is False:
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_SCHEMA_INVALID, "", ()
        )

    unknown = set(evidence_ids) - allowed_ids
    if unknown:
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_UNKNOWN_CITATION, "", ()
        )
    if answer_status == "INSUFFICIENT_CONTEXT":
        if answer or evidence_ids:
            return GroundingDecision(
                False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_INVALID_REFUSAL, "", ()
            )
        return GroundingDecision(
            True,
            Route.INSUFFICIENT_CONTEXT,
            ReasonCode.MODEL_INSUFFICIENT_CONTEXT,
            "",
            (),
        )
    if answer_status != "ANSWER":
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_SCHEMA_INVALID, "", ()
        )
    if not answer:
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_EMPTY_ANSWER, "", ()
        )
    if not evidence_ids:
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_MISSING_CITATION, "", ()
        )
    cited_text = " ".join(
        str(_field(context, "text", ""))
        for context in contexts
        if str(_field(context, "parent_id", "")) in set(evidence_ids)
    )
    if not _answer_overlaps_evidence(answer, cited_text):
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_UNGROUNDED_ANSWER, "", ()
        )
    return GroundingDecision(
        True, Route.ANSWER, ReasonCode.ANSWER_GROUNDED, answer, evidence_ids
    )


def _answer_overlaps_evidence(answer: str, evidence: str, *, minimum: float = 0.2) -> bool:
    """Require informative answer tokens to occur in the cited evidence.

    This is intentionally a conservative lexical gate. It is not a substitute for
    human/entailment evaluation, but it prevents a valid citation ID from acting as
    a blanket approval for an unrelated answer.

    When the answer and its evidence are written in different scripts - an
    English paraphrase grounded on Hindi evidence, say - lexical overlap is
    meaningless (only transliterated names survive). In that cross-lingual
    case the gate switches to number conservation: every figure in the answer
    must exist in the cited evidence, which is language-independent and still
    catches fabricated statistics while allowing faithful translation.
    """
    answer_script = _dominant_script(answer)
    evidence_script = _dominant_script(evidence)
    cross_lingual = (
        answer_script != "none"
        and evidence_script != "none"
        and answer_script != evidence_script
    )
    if cross_lingual:
        return not _introduces_novel_numbers(answer, evidence)

    def tokens(value: str) -> set[str]:
        return {
            token
            for token in _lexical_tokens(value)
            if token not in _GROUNDING_STOPWORDS
        }

    answer_tokens = tokens(answer)
    evidence_tokens = tokens(evidence)
    if not answer_tokens or not evidence_tokens:
        return False
    overlap = len(answer_tokens & evidence_tokens) / len(answer_tokens)
    return overlap >= minimum


# Coarse Unicode-block buckets: enough to tell "answer script differs from
# evidence script" without full language identification.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "indic"),
    (0x0980, 0x09FF, "indic"),
    (0x0A00, 0x0A7F, "indic"),
    (0x0A80, 0x0AFF, "indic"),
    (0x0B00, 0x0B7F, "indic"),
    (0x0B80, 0x0BFF, "indic"),
    (0x0C00, 0x0C7F, "indic"),
    (0x0C80, 0x0CFF, "indic"),
    (0x0D00, 0x0D7F, "indic"),
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
)


def _dominant_script(text: str) -> str:
    counts: dict[str, int] = {"latin": 0, "indic": 0, "arabic": 0}
    for character in text:
        code = ord(character)
        for low, high, name in _SCRIPT_RANGES:
            if low <= code <= high:
                counts[name] += 1
                break
        else:
            if character.isascii() and character.isalpha():
                counts["latin"] += 1
    dominant = max(counts, key=lambda name: counts[name])
    return dominant if counts[dominant] > 0 else "none"


_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
_DIGIT_TRANSLATION = str.maketrans("०१२३४५६७८٩٠١٢٣٤٥٦٧٨", "0123456789012345678")


def _lexical_tokens(value: str) -> set[str]:
    """Word tokens that keep Indic vowel signs attached.

    ``\\w`` excludes combining marks (Mn), which silently shatters Devanagari
    words into consonant fragments; tokenize on alnum-or-mark runs instead.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens: set[str] = set()
    buffer: list[str] = []
    for character in normalized:
        if character.isalnum() or unicodedata.category(character).startswith("M"):
            buffer.append(character)
        elif buffer:
            tokens.add("".join(buffer))
            buffer = []
    if buffer:
        tokens.add("".join(buffer))
    return tokens


def _introduces_novel_numbers(answer: str, evidence: str) -> bool:
    """True when the answer states figures absent from the cited evidence."""
    normalized_answer = unicodedata.normalize("NFKC", answer).translate(_DIGIT_TRANSLATION)
    normalized_evidence = unicodedata.normalize("NFKC", evidence).translate(_DIGIT_TRANSLATION)
    evidence_numbers = {
        number.rstrip(".,")
        for number in _NUMBER_PATTERN.findall(normalized_evidence)
    }
    answer_numbers = {
        number.rstrip(".,")
        for number in _NUMBER_PATTERN.findall(normalized_answer)
    }
    return bool(answer_numbers - evidence_numbers)


def _parse_raw(raw_output: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    valid = (
        isinstance(value, dict)
        and set(value) == {"status", "answer", "evidence_ids"}
        and value["status"] in {"ANSWER", "INSUFFICIENT_CONTEXT"}
        and isinstance(value["answer"], str)
        and isinstance(value["evidence_ids"], list)
        and all(isinstance(item, str) for item in value["evidence_ids"])
    )
    return value if valid else None


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
