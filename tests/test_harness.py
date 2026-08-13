from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hh_goa_rag.guardrails import ReasonCode, Route
from hh_goa_rag.guardrails.types import STAGE_NAMES
from hh_goa_rag.harness import (
    FROZEN_PROMPT,
    FROZEN_TOP_K,
    VoiceRAGHarness,
)


@dataclass(frozen=True)
class Context:
    parent_id: str
    chunk_id: str
    text: str
    score: float


class FakeEmbedder:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def encode_queries(self, texts: list[str]) -> tuple[list[str], dict[str, float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return texts, {}


class FakeRetriever:
    def __init__(self, score: float = 0.8) -> None:
        self.calls = 0
        self.contexts = [
            Context(
                f"p-{index}",
                f"c-{index}",
                (
                    f"evidence {index}"
                    if index == 1
                    else f"evidence {index} गोल्डस्मिथ टेक्सास किस काउंटी में है"
                ),
                score - index / 100,
            )
            for index in range(1, FROZEN_TOP_K + 3)
        ]

    def retrieve(self, _: Any) -> list[Context]:
        self.calls += 1
        return self.contexts


class FakeGenerator:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.calls = 0
        self.result = result

    def generate(
        self, question: str, contexts: list[Any], *, prompt_variant: str
    ) -> dict[str, object]:
        self.calls += 1
        assert prompt_variant == FROZEN_PROMPT
        assert len(contexts) == FROZEN_TOP_K
        if self.result is not None:
            return self.result
        raw = {
            "status": "ANSWER",
            "answer": "evidence 1",
            "evidence_ids": ["p-1"],
        }
        return {
            "status": "ok",
            "answer_status": "ANSWER",
            "answer": "evidence 1",
            "evidence_ids": ["p-1"],
            "raw_output": json.dumps(raw),
            "diagnostics": {"schema_valid": True},
        }


def test_harness_returns_structured_grounded_answer() -> None:
    harness = VoiceRAGHarness(
        embedder=FakeEmbedder(), retriever=FakeRetriever(), generator=FakeGenerator()
    )
    response = harness.handle_text("गोल्डस्मिथ टेक्सास किस काउंटी में है")
    value = response.to_dict()
    assert response.route == Route.ANSWER
    assert response.answer == "evidence 1"
    assert response.citations == ("p-1",)
    assert len(response.retrieved_ids) == FROZEN_TOP_K
    assert set(STAGE_NAMES).issubset(value["stage_latencies_ms"])
    assert {"query_embedding", "vector_search", "guardrails", "total_end_to_end"}.issubset(
        value["stage_latencies_ms"]
    )
    assert value["metadata"]["grounding"]["valid"] is True
    assert len(value["metadata"]["retrieved"]) == FROZEN_TOP_K
    assert value["metadata"]["retrieved"][0]["chunk_id"] == "c-1"
    assert value["metadata"]["retrieved"][0]["text"] == "evidence 1"
    assert value["total_latency_ms"] >= 0


def test_audio_harness_reports_only_actual_pipeline_stages() -> None:
    observed: list[str] = []
    harness = VoiceRAGHarness(
        embedder=FakeEmbedder(),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        stt=FakeSTT("ok", transcript="गोल्डस्मिथ टेक्सास किस काउंटी में है"),
    )

    response = harness.handle_audio("query.wav", on_stage=observed.append)

    assert response.route == Route.ANSWER
    assert observed == [
        "Transcribing",
        "Checking query",
        "Retrieving evidence",
        "Generating answer",
        "Validating grounding",
    ]


def test_input_route_short_circuits_expensive_components() -> None:
    embedder = FakeEmbedder()
    retriever = FakeRetriever()
    generator = FakeGenerator()
    harness = VoiceRAGHarness(embedder=embedder, retriever=retriever, generator=generator)
    response = harness.handle_text("मेरे लिए बारिश पर एक प्रेम कविता लिखो")
    assert response.route == Route.OFF_TOPIC
    assert embedder.calls == retriever.calls == generator.calls == 0


def test_low_retrieval_confidence_skips_generation() -> None:
    generator = FakeGenerator()
    harness = VoiceRAGHarness(
        embedder=FakeEmbedder(), retriever=FakeRetriever(score=0.55), generator=generator
    )
    response = harness.handle_text("मस्तिष्क के सीटी स्कैन की कीमत क्या है")
    assert response.route == Route.INSUFFICIENT_CONTEXT
    assert response.reason_code == ReasonCode.RETRIEVAL_LOW_CONFIDENCE
    assert generator.calls == 0


def test_invalid_model_output_falls_back_to_verbatim_kb_evidence() -> None:
    generator = FakeGenerator(
        {
            "status": "ok",
            "answer_status": "ANSWER",
            "answer": "outside model knowledge",
            "evidence_ids": ["p-1"],
            "raw_output": json.dumps(
                {
                    "status": "ANSWER",
                    "answer": "outside model knowledge",
                    "evidence_ids": ["p-1"],
                }
            ),
            "diagnostics": {"schema_valid": True},
        }
    )
    harness = VoiceRAGHarness(
        embedder=FakeEmbedder(), retriever=FakeRetriever(), generator=generator
    )

    response = harness.handle_text("गोल्डस्मिथ टेक्सास किस काउंटी में है")

    assert response.route == Route.ANSWER
    assert response.answer == "evidence 1"
    assert response.citations == ("p-1",)
    assert response.metadata["generation"]["fallback"] == "extractive_kb"


def test_component_exception_fails_closed() -> None:
    harness = VoiceRAGHarness(
        embedder=FakeEmbedder(fail=True),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )
    response = harness.handle_text("मस्तिष्क के सीटी स्कैन की कीमत क्या है")
    assert response.route == Route.SYSTEM_ERROR
    assert response.reason_code == ReasonCode.SYSTEM_COMPONENT_ERROR


@dataclass
class FakeSTT:
    status: str
    transcript: str = ""
    error_code: str | None = None

    def transcribe_rest(self, _: object, *, language_code: str = "hi-IN") -> FakeSTT:
        return self


def test_audio_stt_failure_has_structured_route() -> None:
    harness = VoiceRAGHarness(
        embedder=FakeEmbedder(),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        stt=FakeSTT("error"),
    )
    response = harness.handle_audio("missing.wav")
    assert response.route == Route.STT_FAILURE
    assert response.reason_code == ReasonCode.STT_PROVIDER_ERROR


def test_invalid_audio_has_specific_structured_reason() -> None:
    harness = VoiceRAGHarness(
        embedder=FakeEmbedder(),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        stt=FakeSTT("error", error_code="invalid_audio"),
    )
    response = harness.handle_audio("empty.wav")
    assert response.route == Route.STT_FAILURE
    assert response.reason_code == ReasonCode.STT_INVALID_AUDIO
