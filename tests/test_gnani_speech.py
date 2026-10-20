import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hh_goa_rag.stt.gnani import GNANI_STT_LANGUAGES, GnaniSTT, GnaniSTTConfig
from hh_goa_rag.stt.providers import FallbackSTT, build_stt_from_env, resolve_stt_provider
from hh_goa_rag.stt.sarvam import SarvamSTT, STTResult
from hh_goa_rag.tts import build_tts_from_env
from hh_goa_rag.tts.gnani import GNANI_TTS_LANGUAGES, GnaniTTS, GnaniTTSConfig, TTSResult
from hh_goa_rag.web import AppSettings, create_app

LIVE_KEY = os.getenv("GNANI_API_KEY", "").strip()


def _wav(path: Path, *, duration_ms: int = 128, sample_rate: int = 16_000) -> None:
    samples = np.full(round(sample_rate * duration_ms / 1000), 500, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


class _HTTPError(Exception):
    def __init__(self, status_code: int, body: str = "{}") -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class _FakeSTTClient:
    def __init__(self, responses: list[object], calls: list[dict]) -> None:
        self._responses = responses
        self._calls = calls

    def transcribe(self, audio: str, **kwargs: Any) -> object:
        self._calls.append({"audio": audio, **kwargs})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _gnani(responses: list[object], calls: list[dict], delays: list[float]) -> GnaniSTT:
    return GnaniSTT(
        "secret",
        config=GnaniSTTConfig(backoff_base_s=0.25, backoff_max_s=0.25),
        client_factory=lambda **_kwargs: _FakeSTTClient(responses, calls),
        sleep=delays.append,
    )


def test_gnani_stt_language_table_matches_installed_sdk() -> None:
    gnani_stt = pytest.importorskip("gnani.stt")
    assert set(gnani_stt.SUPPORTED_LANGUAGES) == GNANI_STT_LANGUAGES
    gnani_tts = pytest.importorskip("gnani.tts")
    assert set(gnani_tts.SUPPORTED_TTS_LANGUAGES) - {"auto"} == GNANI_TTS_LANGUAGES


def test_gnani_stt_retries_retryable_errors_and_returns_neutral_result(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    _wav(audio)
    calls: list[dict] = []
    delays: list[float] = []
    service = _gnani(
        [
            _HTTPError(503, '{"success": false}'),
            {"success": True, "request_id": "req-1", "transcript": " नमस्ते "},
        ],
        calls,
        delays,
    )
    result = service.transcribe_rest(audio, language_code="hi-IN")
    assert result.status == "ok"
    assert result.provider == "gnani"
    assert result.transcript == "नमस्ते"
    assert result.request_id == "req-1"
    assert result.attempts == 2
    assert delays == [0.25]
    assert calls[-1] == {"audio": str(audio), "language_code": "hi-IN", "format": "verbatim"}


def test_gnani_stt_does_not_retry_auth_or_unsupported_language(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    _wav(audio)
    calls: list[dict] = []
    service = _gnani([_HTTPError(403, '{"error": {"message": "no credits"}}')], calls, [])
    result = service.transcribe_rest(audio, language_code="en-IN")
    assert result.status == "error"
    assert result.error_code == "authentication_error"
    assert result.attempts == 1

    unsupported = service.transcribe_rest(audio, language_code="od-IN")
    assert unsupported.error_code == "unsupported_language"
    assert len(calls) == 1  # no network call for an unsupported language


def test_gnani_stt_rejects_bad_audio_before_calling_api(tmp_path: Path) -> None:
    calls: list[dict] = []
    service = _gnani([], calls, [])
    result = service.transcribe_rest(tmp_path / "missing.wav", language_code="hi-IN")
    assert result.error_code == "invalid_audio"
    assert calls == []


class _StaticSTT:
    def __init__(self, result: STTResult) -> None:
        self.result = result
        self.calls = 0

    def transcribe_rest(self, audio_path: Any, *, language_code: str = "hi-IN") -> STTResult:
        self.calls += 1
        return self.result


def _result(provider: str, status: str, error_code: str | None = None) -> STTResult:
    return STTResult(
        provider=provider,  # type: ignore[arg-type]
        transport="rest",
        model="m",
        mode="x",
        status=status,  # type: ignore[arg-type]
        transcript="text" if status == "ok" else "",
        latency_ms=1.0,
        attempts=1,
        error_code=error_code,
    )


def test_fallback_stt_uses_sarvam_only_when_gnani_fails() -> None:
    sarvam = _StaticSTT(_result("sarvam", "ok"))
    ok = FallbackSTT(_StaticSTT(_result("gnani", "ok")), sarvam)
    assert ok.transcribe_rest("a.wav").provider == "gnani"
    assert sarvam.calls == 0

    failed = FallbackSTT(_StaticSTT(_result("gnani", "error", "gnani_unavailable")), sarvam)
    recovered = failed.transcribe_rest("a.wav", language_code="od-IN")
    assert recovered.provider == "sarvam"
    assert recovered.events[-1]["fallback_from"] == "gnani"
    assert recovered.events[-1]["fallback_reason"] == "gnani_unavailable"

    bad_audio = FallbackSTT(_StaticSTT(_result("gnani", "error", "invalid_audio")), sarvam)
    assert bad_audio.transcribe_rest("a.wav").error_code == "invalid_audio"
    assert sarvam.calls == 1


def test_provider_selection_defaults_to_gnani_and_keeps_sarvam_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = tmp_path / "missing.env"
    monkeypatch.delenv("HH_RAG_STT_PROVIDER", raising=False)
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setenv("SARVAM_API_KEY", "sarvam-secret")
    assert resolve_stt_provider(env) == "sarvam"
    assert isinstance(build_stt_from_env(env), SarvamSTT)

    monkeypatch.setenv("GNANI_API_KEY", "gnani-secret")
    assert resolve_stt_provider(env) == "gnani+sarvam"
    service = build_stt_from_env(env)
    assert isinstance(service, FallbackSTT)
    assert isinstance(service.primary, GnaniSTT)
    assert isinstance(service.secondary, SarvamSTT)

    monkeypatch.delenv("SARVAM_API_KEY")
    assert isinstance(build_stt_from_env(env), GnaniSTT)

    monkeypatch.setenv("HH_RAG_STT_PROVIDER", "sarvam")
    with pytest.raises(ValueError, match="SARVAM_API_KEY"):
        resolve_stt_provider(env)
    monkeypatch.setenv("HH_RAG_STT_PROVIDER", "whisper")
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_stt_provider(env)


class _FakeTTSClient:
    def __init__(self, responses: list[object], calls: list[dict]) -> None:
        self._responses = responses
        self._calls = calls

    def synthesize(self, text: str, **kwargs: Any) -> object:
        self._calls.append({"text": text, **kwargs})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _tts(responses: list[object], calls: list[dict], **config: Any) -> GnaniTTS:
    return GnaniTTS(
        "secret",
        config=GnaniTTSConfig(**config),
        client_factory=lambda **_kwargs: _FakeTTSClient(responses, calls),
        audio_config_factory=lambda **kwargs: kwargs,
        sleep=lambda _seconds: None,
    )


def test_gnani_tts_sends_timbre_v25_controls_and_returns_audio() -> None:
    calls: list[dict] = []
    service = _tts([_HTTPError(429), b"RIFF-audio"], calls)
    result = service.synthesize("  Answer   text ", language_code="kn-IN")
    assert result.status == "ok"
    assert result.audio == b"RIFF-audio"
    assert result.mime_type == "audio/wav"
    assert result.attempts == 2
    assert "audio" not in result.to_dict()
    assert calls[-1] == {
        "text": "Answer text",
        "voice": "Kaveri",
        "model": "timbre-v2.5",
        "language": "kn-IN",
        "speed": 1.0,
        "audio_config": {"sample_rate": 24_000, "encoding": "linear_pcm", "container": "wav"},
    }


def test_gnani_tts_v20_omits_language_and_validates_input() -> None:
    calls: list[dict] = []
    service = _tts([b"audio"], calls, model="timbre-v2.0", voice="Pranav")
    assert service.synthesize("hello", language_code="od-IN").status == "ok"
    assert "language" not in calls[-1] and "speed" not in calls[-1]

    v25 = _tts([], calls)
    assert v25.synthesize("", language_code="hi-IN").error_code == "empty_text"
    assert v25.synthesize("x" * 1001, language_code="hi-IN").error_code == "text_too_long"
    assert v25.synthesize("hello", language_code="od-IN").error_code == "unsupported_language"


def test_tts_is_disabled_without_key_or_when_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = tmp_path / "missing.env"
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.delenv("HH_RAG_TTS_PROVIDER", raising=False)
    assert build_tts_from_env(env) is None
    monkeypatch.setenv("GNANI_API_KEY", "gnani-secret")
    assert isinstance(build_tts_from_env(env), GnaniTTS)
    monkeypatch.setenv("HH_RAG_TTS_PROVIDER", "off")
    assert build_tts_from_env(env) is None


class _Harness:
    def close(self) -> None:
        return None


class _WebTTS:
    def synthesize(self, text: str, *, language_code: str = "hi-IN") -> TTSResult:
        if language_code == "od-IN":
            return TTSResult(
                provider="gnani",
                model="timbre-v2.5",
                voice="Kaveri",
                status="error",
                latency_ms=1.0,
                attempts=0,
                mime_type="audio/wav",
                error_code="unsupported_language",
            )
        return TTSResult(
            provider="gnani",
            model="timbre-v2.5",
            voice="Kaveri",
            status="ok",
            latency_ms=12.3,
            attempts=1,
            mime_type="audio/wav",
            audio=b"RIFF" + text.encode(),
        )


def _app(tts: Any) -> Any:
    return create_app(
        harness_factory=lambda _settings: _Harness(),
        tts_factory=lambda _settings: tts,
        settings=AppSettings(Path("unused.json"), Path("unused.env"), "cpu"),
    )


def test_tts_endpoint_streams_audio_and_reports_availability() -> None:
    with TestClient(_app(_WebTTS())) as client:
        assert client.get("/health").json()["tts_available"] is True
        response = client.post("/api/tts", json={"text": "answer", "language_code": "en-IN"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-tts-provider"] == "gnani"
        assert response.content == b"RIFFanswer"
        unsupported = client.post("/api/tts", json={"text": "x", "language_code": "od-IN"})
        assert unsupported.status_code == 422
        assert unsupported.json()["error_code"] == "unsupported_language"
        assert client.post("/api/tts", json={"text": "  "}).status_code == 422


def test_tts_endpoint_is_unavailable_without_provider() -> None:
    with TestClient(_app(None)) as client:
        assert client.get("/health").json()["tts_available"] is False
        assert client.post("/api/tts", json={"text": "answer"}).status_code == 503


@pytest.mark.skipif(not LIVE_KEY, reason="GNANI_API_KEY is not set; live Gnani smoke skipped")
def test_live_gnani_timbre_to_prisma_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("gnani")
    tts = GnaniTTS(LIVE_KEY, config=GnaniTTSConfig(sample_rate_hz=16_000))
    spoken = tts.synthesize("Goa is famous for its beaches.", language_code="en-IN")
    assert spoken.status == "ok", spoken.to_dict()
    audio = tmp_path / "round_trip.wav"
    audio.write_bytes(spoken.audio)
    heard = GnaniSTT(LIVE_KEY).transcribe_rest(audio, language_code="en-IN")
    assert heard.status == "ok", heard.to_dict()
    assert "goa" in heard.transcript.casefold()
