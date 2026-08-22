import asyncio
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sarvamai.core.api_error import ApiError

from eval.evaluate_stt import _ready_manifest_rows
from eval.record_audio import detect_speech_end_ms
from hh_goa_rag.stt.sarvam import SarvamSTT, SarvamSTTConfig, _merge_transcripts


def _wav(path: Path, *, duration_ms: int = 128) -> None:
    samples = np.full(round(16_000 * duration_ms / 1000), 500, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


class _RestClient:
    def __init__(self, responses: list[object], calls: list[dict]) -> None:
        self._responses = responses
        self._calls = calls
        self.speech_to_text = self

    def transcribe(self, **kwargs):
        self._calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        return None


def test_rest_is_fixed_and_retries_only_retryable_errors(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    _wav(audio)
    calls: list[dict] = []
    responses: list[object] = [
        ApiError(status_code=429, body={"error": "busy"}),
        SimpleNamespace(
            transcript="नमस्ते",
            language_code="hi-IN",
            language_probability=0.99,
            request_id="request-1",
        ),
    ]
    delays: list[float] = []

    def factory(**_kwargs):
        return _RestClient(responses, calls)

    service = SarvamSTT(
        "secret",
        config=SarvamSTTConfig(backoff_base_s=0.25, backoff_max_s=0.25),
        rest_client_factory=factory,
        sleep=delays.append,
    )
    result = service.transcribe_rest(audio)
    assert result.status == "ok"
    assert result.transcript == "नमस्ते"
    assert result.attempts == 2
    assert delays == [0.25]
    assert calls[-1]["model"] == "saaras:v3"
    assert calls[-1]["mode"] == "transcribe"


def test_rest_does_not_retry_auth_and_handles_bad_audio(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    _wav(audio)
    calls: list[dict] = []

    def factory(**_kwargs):
        return _RestClient([ApiError(status_code=403, body={"error": "bad key"})], calls)

    service = SarvamSTT("secret", rest_client_factory=factory)
    auth = service.transcribe_rest(audio)
    assert auth.status == "error"
    assert auth.error_code == "authentication_error"
    assert auth.attempts == 1

    malformed = tmp_path / "empty.wav"
    malformed.write_bytes(b"")
    invalid = service.transcribe_rest(malformed)
    assert invalid.status == "error"
    assert invalid.error_code == "invalid_audio"
    assert invalid.attempts == 0


class _Socket:
    def __init__(self) -> None:
        self.flushed = asyncio.Event()
        self.delivered = False
        self.sent: list[dict] = []

    async def transcribe(self, **kwargs) -> None:
        self.sent.append(kwargs)

    async def flush(self) -> None:
        self.flushed.set()

    async def recv(self):
        await self.flushed.wait()
        if self.delivered:
            await asyncio.Event().wait()
        self.delivered = True
        return SimpleNamespace(
            type="data",
            data=SimpleNamespace(
                transcript="स्ट्रीम परिणाम",
                request_id="stream-1",
                language_code="hi-IN",
                language_probability=0.98,
            ),
        )


class _Connect:
    def __init__(self, socket: _Socket) -> None:
        self.socket = socket

    async def __aenter__(self) -> _Socket:
        return self.socket

    async def __aexit__(self, *_args) -> None:
        return None


class _StreamingClient:
    def __init__(self, socket: _Socket) -> None:
        self.socket = socket
        self.speech_to_text_streaming = self
        self.connect_args: dict = {}

    def connect(self, **kwargs):
        self.connect_args = kwargs
        return _Connect(self.socket)

    async def close(self) -> None:
        return None


def test_streaming_reports_finalization_and_no_fake_partial(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    _wav(audio)
    socket = _Socket()
    client = _StreamingClient(socket)
    service = SarvamSTT(
        "secret",
        config=SarvamSTTConfig(post_final_grace_s=0.001),
        async_client_factory=lambda **_kwargs: client,
    )
    result = asyncio.run(
        service.transcribe_streaming(audio, speech_end_offset_ms=100, pace_audio=False)
    )
    assert result.status == "ok"
    assert result.transcript == "स्ट्रीम परिणाम"
    assert result.time_to_first_partial_ms is None
    assert result.time_to_first_transcript_ms is not None
    assert result.end_of_speech_to_final_ms is not None
    assert client.connect_args["model"] == "saaras:v3"
    assert client.connect_args["mode"] == "transcribe"
    assert socket.sent


def test_manifest_refuses_pending_or_missing_audio(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    pending = {
        "sample_id": "s1",
        "audio_path": str(tmp_path / "missing.wav"),
        "status": "pending",
    }
    manifest.write_text(json.dumps(pending) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No real recordings"):
        _ready_manifest_rows(manifest)
    pending["status"] = "ready"
    manifest.write_text(json.dumps(pending) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing audio"):
        _ready_manifest_rows(manifest)


def test_speech_end_detection() -> None:
    samples = np.zeros(16_000, dtype=np.int16)
    samples[1600:3200] = 10_000
    assert detect_speech_end_ms(samples) == pytest.approx(200.0)
    assert detect_speech_end_ms(np.zeros(1600, dtype=np.int16)) is None


def test_model_and_mode_cannot_change() -> None:
    with pytest.raises(ValueError, match="frozen"):
        SarvamSTTConfig(model="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frozen"):
        SarvamSTTConfig(mode="translate")  # type: ignore[arg-type]


def test_streaming_segments_remove_boundary_overlap() -> None:
    segments = [
        "It was built",
        "Built by Shah Jahan for his wife",
        "wife Mumtaz Mahal.",
    ]
    assert _merge_transcripts(segments) == "It was built by Shah Jahan for his wife Mumtaz Mahal."
