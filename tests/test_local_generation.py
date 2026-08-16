from __future__ import annotations

import pytest

from hh_goa_rag.generation.local import (
    extractive_span_integrity,
    grounding_overlap,
    select_route_tier,
    validate_generated_answer,
)
from hh_goa_rag.generation.sarvam import GenerationContext


@pytest.fixture
def contexts() -> list[GenerationContext]:
    return [
        GenerationContext(
            rank=1,
            parent_id="p-source-one",
            chunk_id="c-one",
            score=0.9,
            text="मस्तिष्क के सीटी स्कैन की राष्ट्रीय औसत लागत 1,200 डॉलर है।",
        ),
        GenerationContext(
            rank=2,
            parent_id="p-source-two",
            chunk_id="c-two",
            score=0.8,
            text="कीमत स्थान और बीमा के अनुसार बदल सकती है।",
        ),
    ]


def test_generated_answer_maps_citation_to_original_parent(
    contexts: list[GenerationContext],
) -> None:
    result = validate_generated_answer(
        "राष्ट्रीय औसत लागत 1,200 डॉलर है। [E1]",
        contexts,
        confidence=0.8,
        latency_ms=12.0,
    )

    assert result.status == "ANSWER"
    assert result.evidence_ids == ("p-source-one",)
    assert result.answer == "राष्ट्रीय औसत लागत 1,200 डॉलर है।"


def test_generated_answer_does_not_strip_uttar_prefix_from_uttari(
    contexts: list[GenerationContext],
) -> None:
    northern_context = [
        GenerationContext(
            rank=1,
            parent_id="p-uk",
            chunk_id="c-uk",
            score=0.9,
            text="इंग्लैंड, स्कॉटलैंड, वेल्स और उत्तरी आयरलैंड देश हैं।",
        )
    ]
    result = validate_generated_answer(
        "[E1] इंग्लैंड, स्कॉटलैंड, वेल्स और उत्तरी आयरलैंड।",
        northern_context,
        confidence=0.8,
        latency_ms=12.0,
    )

    assert result.answer == "इंग्लैंड, स्कॉटलैंड, वेल्स और उत्तरी आयरलैंड।"


def test_generated_answer_rejects_novel_number(contexts: list[GenerationContext]) -> None:
    result = validate_generated_answer(
        "राष्ट्रीय औसत लागत 9,999 डॉलर है। [E1]",
        contexts,
        confidence=0.8,
        latency_ms=12.0,
    )

    assert result.status == "INVALID"
    assert result.validation_error.startswith("novel_numbers:")


def test_generated_answer_rejects_missing_or_unknown_citation(
    contexts: list[GenerationContext],
) -> None:
    missing = validate_generated_answer(
        "राष्ट्रीय औसत लागत 1,200 डॉलर है।",
        contexts,
        confidence=0.8,
        latency_ms=12.0,
    )
    unknown = validate_generated_answer(
        "राष्ट्रीय औसत लागत 1,200 डॉलर है। [E3]",
        contexts,
        confidence=0.8,
        latency_ms=12.0,
    )

    assert missing.validation_error == "missing_citation"
    assert unknown.validation_error == "unknown_citation"


def test_generated_answer_prefers_abstention(contexts: list[GenerationContext]) -> None:
    result = validate_generated_answer(
        "INSUFFICIENT_CONTEXT",
        contexts,
        confidence=0.9,
        latency_ms=12.0,
    )

    assert result.status == "INSUFFICIENT_CONTEXT"
    assert not result.answered


def test_grounding_overlap_ignores_common_stopwords() -> None:
    overlap = grounding_overlap("कार्बोनिक एसिड बनता है", "यह कार्बोनिक एसिड बनाता है")

    assert overlap == pytest.approx(2 / 3)


def test_extractive_integrity_rejects_truncated_currency_range() -> None:
    assert not extractive_span_integrity("प्रति रेखीय फुट डॉलर से 10 डॉलर तक")
    assert extractive_span_integrity("प्रति रेखीय फुट 1 डॉलर से 10 डॉलर तक")


def test_confidence_router_is_conservative_and_deterministic() -> None:
    assert select_route_tier(0.99, "कार्बोनिक एसिड", "INVALID") == 1
    assert select_route_tier(0.5, "गलत span", "ANSWER") == 2
    assert select_route_tier(0.5, "गलत span", "INVALID") == 3
    assert select_route_tier(
        0.5, "गलत span", "ANSWER", generator_enabled=False
    ) == 3
