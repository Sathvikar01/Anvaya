from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from hh_goa_rag.guardrails import ReasonCode, Route
from hh_goa_rag.guardrails.types import GuardrailResponse
from hh_goa_rag.web import AppSettings, create_app


class FakeWebHarness:
    def __init__(self) -> None:
        self.closed = False

    def handle_audio(
        self, audio_path: Path, *, language_code: str = "hi-IN", on_stage: Any
    ) -> GuardrailResponse:
        with wave.open(str(audio_path), "rb") as handle:
            assert handle.getframerate() == 16_000
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getnframes() == 1600
        for stage in (
            "Transcribing",
            "Checking query",
            "Retrieving evidence",
            "Generating answer",
            "Validating grounding",
        ):
            on_stage(stage)
        return GuardrailResponse(
            route=Route.ANSWER,
            answer="grounded answer",
            retrieved_ids=("p-1",),
            citations=("p-1",),
            reason_code=ReasonCode.ANSWER_GROUNDED,
            transcript="test transcript",
            stage_latencies_ms={"stt": 1.0, "generation": 2.0},
            total_latency_ms=3.0,
            metadata={
                "retrieved": [
                    {
                        "rank": 1,
                        "parent_id": "p-1",
                        "chunk_id": "c-1",
                        "score": 0.81,
                        "text": "supporting passage",
                    }
                ]
            },
        )

    def handle_text(
        self, text: str, *, language_code: str = "hi-IN", on_stage: Any
    ) -> GuardrailResponse:
        assert text == "typed question"
        for stage in (
            "Checking query",
            "Retrieving evidence",
            "Generating answer",
            "Validating grounding",
        ):
            on_stage(stage)
        return GuardrailResponse(
            route=Route.ANSWER,
            answer="text grounded answer",
            retrieved_ids=("p-1",),
            citations=("p-1",),
            reason_code=ReasonCode.ANSWER_GROUNDED,
            transcript=text,
            stage_latencies_ms={"generation": 2.0},
            total_latency_ms=2.0,
            metadata={"retrieved": []},
        )

    def close(self) -> None:
        self.closed = True


def _wav_bytes(*, sample_rate: int = 16_000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x01\x00" * 1600 * channels)
    return buffer.getvalue()


def _test_app(harness: FakeWebHarness) -> Any:
    return create_app(
        harness_factory=lambda _settings: harness,
        settings=AppSettings(Path("unused.json"), Path("unused.env"), "cpu"),
    )


def test_audio_endpoint_calls_harness_and_returns_structured_output() -> None:
    harness = FakeWebHarness()
    with TestClient(_test_app(harness)) as client:
        response = client.post(
            "/api/query/audio",
            headers={"X-Request-ID": "request-1234"},
            files={"audio": ("query.wav", _wav_bytes(), "audio/wav")},
        )
        progress = client.get("/api/query/status/request-1234")

    assert response.status_code == 200
    assert response.json()["route"] == "ANSWER"
    assert response.json()["metadata"]["retrieved"][0]["text"] == "supporting passage"
    assert progress.status_code == 200
    assert progress.json()["complete"] is True
    assert progress.json()["history"] == [
        "Transcribing",
        "Checking query",
        "Retrieving evidence",
        "Generating answer",
        "Validating grounding",
        "Complete",
    ]
    assert harness.closed is True


def test_audio_endpoint_rejects_non_frozen_audio_format() -> None:
    harness = FakeWebHarness()
    with TestClient(_test_app(harness)) as client:
        response = client.post(
            "/api/query/audio",
            headers={"X-Request-ID": "request-5678"},
            files={"audio": ("wrong.wav", _wav_bytes(sample_rate=44_100), "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Audio must be 16 kHz, mono, PCM16 WAV"


def test_text_endpoint_calls_harness_and_returns_structured_output() -> None:
    harness = FakeWebHarness()
    with TestClient(_test_app(harness)) as client:
        response = client.post(
            "/api/query/text",
            headers={"X-Request-ID": "request-text-1"},
            json={"text": "typed question"},
        )
        progress = client.get("/api/query/status/request-text-1")

    assert response.status_code == 200
    assert response.json()["route"] == "ANSWER"
    assert response.json()["transcript"] == "typed question"
    assert progress.status_code == 200
    assert progress.json()["complete"] is True
    assert progress.json()["history"] == [
        "Checking query",
        "Retrieving evidence",
        "Generating answer",
        "Validating grounding",
        "Complete",
    ]


def test_text_endpoint_validates_payload() -> None:
    harness = FakeWebHarness()
    with TestClient(_test_app(harness)) as client:
        missing_text = client.post("/api/query/text", json={})
        too_long = client.post("/api/query/text", json={"text": "x" * 2_001})

    assert missing_text.status_code == 422
    assert missing_text.json()["detail"] == "Text query must be a string"
    assert too_long.status_code == 422
    assert "2,000" in too_long.json()["detail"]


def test_endpoints_validate_and_forward_language_code() -> None:
    harness = FakeWebHarness()
    with TestClient(_test_app(harness)) as client:
        response = client.post(
            "/api/query/text",
            json={"text": "typed question", "language_code": "en-IN"},
        )
        invalid = client.post(
            "/api/query/text",
            json={"text": "typed question", "language_code": "xx-IN"},
        )

    assert response.status_code == 200
    assert invalid.status_code == 422


def test_health_and_frontend_are_served() -> None:
    harness = FakeWebHarness()
    with TestClient(_test_app(harness)) as client:
        health = client.get("/health")
        page = client.get("/")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert page.status_code == 200
    assert "Anvaya" in page.text
