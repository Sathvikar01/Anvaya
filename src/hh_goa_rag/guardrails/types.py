"""Shared structured types for deterministic Voice-RAG routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Route(StrEnum):
    ANSWER = "ANSWER"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    OFF_TOPIC = "OFF_TOPIC"
    UNSAFE = "UNSAFE"
    STT_FAILURE = "STT_FAILURE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class ReasonCode(StrEnum):
    ANSWER_GROUNDED = "ANSWER_GROUNDED"
    TRANSCRIPT_EMPTY = "TRANSCRIPT_EMPTY"
    TRANSCRIPT_INVALID_TYPE = "TRANSCRIPT_INVALID_TYPE"
    TRANSCRIPT_TOO_LONG = "TRANSCRIPT_TOO_LONG"
    TRANSCRIPT_LOW_INFORMATION = "TRANSCRIPT_LOW_INFORMATION"
    STT_INVALID_AUDIO = "STT_INVALID_AUDIO"
    STT_PROVIDER_ERROR = "STT_PROVIDER_ERROR"
    UNSAFE_PHYSICAL_HARM = "UNSAFE_PHYSICAL_HARM"
    UNSAFE_WEAPONS = "UNSAFE_WEAPONS"
    UNSAFE_CREDENTIAL_THEFT = "UNSAFE_CREDENTIAL_THEFT"
    UNSAFE_HATE = "UNSAFE_HATE"
    UNSAFE_PROMPT_INJECTION = "UNSAFE_PROMPT_INJECTION"
    OFF_TOPIC_CREATIVE_WRITING = "OFF_TOPIC_CREATIVE_WRITING"
    OFF_TOPIC_LIVE_INFORMATION = "OFF_TOPIC_LIVE_INFORMATION"
    OFF_TOPIC_TRANSACTION = "OFF_TOPIC_TRANSACTION"
    OFF_TOPIC_RECIPE = "OFF_TOPIC_RECIPE"
    RETRIEVAL_EMPTY = "RETRIEVAL_EMPTY"
    RETRIEVAL_LOW_CONFIDENCE = "RETRIEVAL_LOW_CONFIDENCE"
    MODEL_INSUFFICIENT_CONTEXT = "MODEL_INSUFFICIENT_CONTEXT"
    GENERATION_PROVIDER_ERROR = "GENERATION_PROVIDER_ERROR"
    GENERATION_SCHEMA_INVALID = "GENERATION_SCHEMA_INVALID"
    GENERATION_UNKNOWN_CITATION = "GENERATION_UNKNOWN_CITATION"
    GENERATION_MISSING_CITATION = "GENERATION_MISSING_CITATION"
    GENERATION_EMPTY_ANSWER = "GENERATION_EMPTY_ANSWER"
    GENERATION_UNGROUNDED_ANSWER = "GENERATION_UNGROUNDED_ANSWER"
    GENERATION_INVALID_REFUSAL = "GENERATION_INVALID_REFUSAL"
    DEADLINE_BUDGET_EXHAUSTED = "DEADLINE_BUDGET_EXHAUSTED"
    SYSTEM_COMPONENT_ERROR = "SYSTEM_COMPONENT_ERROR"


STAGE_NAMES = (
    "stt",
    "input_validation",
    "route_check",
    "embedding",
    "retrieval",
    "evidence_guardrail",
    "generation",
    "grounding_validation",
)


@dataclass(frozen=True)
class GuardrailResponse:
    route: Route
    answer: str = ""
    retrieved_ids: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    reason_code: ReasonCode | None = None
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    transcript: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["route"] = self.route.value
        value["reason_code"] = self.reason_code.value if self.reason_code else None
        value["retrieved_ids"] = list(self.retrieved_ids)
        value["citations"] = list(self.citations)
        return value
