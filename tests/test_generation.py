from __future__ import annotations

import json

import httpx
import pytest

from hh_goa_rag.generation.groq import GroqGeneration, GroqGenerationConfig
from hh_goa_rag.generation.prompts import PROMPT_VARIANTS, build_messages
from hh_goa_rag.generation.sarvam import (
    GenerationContext,
    SarvamGeneration,
    SarvamGenerationConfig,
)


def context() -> GenerationContext:
    return GenerationContext(
        parent_id="p-1",
        chunk_id="c-1",
        text="गोवा भारत में है।",
        rank=1,
        score=0.9,
    )


def stream_client_factory(payload: dict[str, object]):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["api-subscription-key"] == "secret"
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["reasoning_effort"] is None
        content = json.dumps(payload, ensure_ascii=False)
        chunks = [
            {"choices": [{"delta": {"content": content}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            },
        ]
        sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        sse += "data: [DONE]\n\n"
        return httpx.Response(200, text=sse)

    transport = httpx.MockTransport(handler)
    return lambda **_: httpx.Client(transport=transport)


def test_generation_returns_grounded_structured_result() -> None:
    service = SarvamGeneration(
        "secret",
        config=SarvamGenerationConfig(),
        client_factory=stream_client_factory(
            {"status": "ANSWER", "answer": "गोवा भारत में है।", "evidence_ids": ["p-1"]}
        ),
    )
    result = service.generate("गोवा कहाँ है?", [context()])
    assert result.status == "ok"
    assert result.answer_status == "ANSWER"
    assert result.evidence_ids == ("p-1",)
    assert result.output_tokens == 8
    assert result.time_to_first_token_ms is not None
    assert result.diagnostics["schema_valid"] is True


def test_generation_rejects_unknown_provenance() -> None:
    service = SarvamGeneration(
        "secret",
        client_factory=stream_client_factory(
            {"status": "ANSWER", "answer": "उत्तर", "evidence_ids": ["not-retrieved"]}
        ),
    )
    result = service.generate("प्रश्न", [context()])
    assert result.status == "error"
    assert result.error_code == "invalid_structured_output"
    assert result.diagnostics["unknown_evidence_ids"] == ["not-retrieved"]


@pytest.mark.parametrize("variant", PROMPT_VARIANTS)
def test_all_prompt_variants_preserve_parent_and_chunk_ids(variant: str) -> None:
    messages = build_messages("प्रश्न", [context()], variant=variant)  # type: ignore[arg-type]
    assert 'parent_id="p-1"' in messages[1]["content"]
    assert 'chunk_id="c-1"' in messages[1]["content"]


def test_config_freezes_ablation_parameters() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SarvamGenerationConfig(temperature=0.2)
    with pytest.raises(ValueError, match="Unsupported"):
        SarvamGenerationConfig(model="invented")  # type: ignore[arg-type]


def groq_stream_client_factory(payload: dict[str, object]):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["include_reasoning"] is False
        assert body["reasoning_effort"] == "low"
        assert body["response_format"] == {"type": "json_object"}
        content = json.dumps(payload, ensure_ascii=False)
        chunks = [
            {"choices": [{"delta": {"content": content}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                    "completion_time": 0.05,
                    "total_time": 0.06,
                },
            },
        ]
        sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        sse += "data: [DONE]\n\n"
        return httpx.Response(200, text=sse)

    transport = httpx.MockTransport(handler)
    return lambda **_: httpx.Client(transport=transport)


def test_groq_generation_returns_provider_metrics() -> None:
    service = GroqGeneration(
        "secret",
        config=GroqGenerationConfig(),
        client_factory=groq_stream_client_factory(
            {"status": "ANSWER", "answer": "गोवा भारत में है।", "evidence_ids": ["p-1"]}
        ),
    )
    result = service.generate("गोवा कहाँ है?", [context()])
    assert result.status == "ok"
    assert result.provider == "groq"
    assert result.output_tokens == 10
    assert result.tokens_per_second == 200
    assert result.provider_latency_ms == 60
    assert result.time_to_first_token_ms is not None


def test_groq_warm_up_reuses_client_and_checks_model() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={"data": [{"id": "openai/gpt-oss-20b"}]},
        )

    transport = httpx.MockTransport(handler)
    service = GroqGeneration(
        "secret",
        client_factory=lambda **_: httpx.Client(transport=transport),
    )
    service.warm_up()
    service.close()
    assert calls == ["/openai/v1/models"]
