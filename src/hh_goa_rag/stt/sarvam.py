"""Sarvam-only speech-to-text service with REST and WebSocket transports.

The provider/model/mode are intentionally fixed.  REST exists for reproducible file benchmarking
and debugging; the WebSocket path exposes streaming-specific timings for the future voice pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import re
import time
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, SarvamAI
from sarvamai.core.api_error import ApiError

SARVAM_PROVIDER = "sarvam"
SARVAM_MODEL = "saaras:v3"
SARVAM_MODE = "transcribe"
RETRYABLE_HTTP_STATUSES = {429, 500, 503}
SUPPORTED_REST_SUFFIXES = {
    ".aac",
    ".aiff",
    ".amr",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}
MAX_AUDIO_BYTES = 2 * 1024 * 1024
MAX_AUDIO_DURATION_MS = 30_000


@dataclass(frozen=True)
class SarvamSTTConfig:
    """Immutable Saaras v3 transcription configuration."""

    model: Literal["saaras:v3"] = SARVAM_MODEL
    mode: Literal["transcribe"] = SARVAM_MODE
    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    streaming_chunk_ms: int = 64
    high_vad_sensitivity: bool = True
    vad_signals: bool = True
    flush_signal: bool = True
    timeout_s: float = 20.0
    max_attempts: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 2.0
    post_final_grace_s: float = 0.25
    max_audio_bytes: int = MAX_AUDIO_BYTES
    max_audio_duration_ms: int = MAX_AUDIO_DURATION_MS

    def __post_init__(self) -> None:
        if self.model != SARVAM_MODEL or self.mode != SARVAM_MODE:
            raise ValueError("Sarvam STT is frozen to saaras:v3 with mode='transcribe'")
        if self.sample_rate_hz != 16_000:
            raise ValueError("Evaluation audio is frozen to 16 kHz")
        if self.channels != 1 or self.sample_width_bytes != 2:
            raise ValueError("Evaluation audio must be mono 16-bit PCM")
        if not 32 <= self.streaming_chunk_ms <= 1000:
            raise ValueError("streaming_chunk_ms must be between 32 and 1000")
        if self.timeout_s <= 0 or self.max_attempts <= 0:
            raise ValueError("timeout_s and max_attempts must be positive")
        if self.max_audio_bytes <= 0 or self.max_audio_duration_ms <= 0:
            raise ValueError("Audio limits must be positive")
        if not 0 <= self.backoff_base_s <= self.backoff_max_s:
            raise ValueError("Invalid bounded-backoff configuration")


@dataclass(frozen=True)
class AudioInfo:
    path: str
    bytes: int
    duration_ms: float | None
    sample_rate_hz: int | None
    channels: int | None
    sample_width_bytes: int | None
    frame_count: int | None


@dataclass(frozen=True)
class STTResult:
    """Provider-neutral, JSON-serializable transcription observation."""

    provider: Literal["sarvam"]
    transport: Literal["rest", "streaming"]
    model: Literal["saaras:v3"]
    mode: Literal["transcribe"]
    status: Literal["ok", "error"]
    transcript: str
    latency_ms: float
    attempts: int
    language_code: str | None = None
    language_probability: float | None = None
    request_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    retryable: bool = False
    connection_latency_ms: float | None = None
    time_to_first_partial_ms: float | None = None
    time_to_first_transcript_ms: float | None = None
    end_of_speech_to_final_ms: float | None = None
    audio_duration_ms: float | None = None
    sample_rate_hz: int | None = None
    partial_transcripts: tuple[str, ...] = field(default_factory=tuple)
    events: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioValidationError(ValueError):
    pass


class _StreamingAPIError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SarvamSTT:
    """Saaras v3 service supporting synchronous REST and GA WebSocket streaming."""

    def __init__(
        self,
        api_key: str,
        *,
        config: SarvamSTTConfig | None = None,
        rest_client_factory: Callable[..., Any] = SarvamAI,
        async_client_factory: Callable[..., Any] = AsyncSarvamAI,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SARVAM_API_KEY is missing; set it in the ignored .env file")
        self._api_key = api_key.strip()
        self.config = config or SarvamSTTConfig()
        self._rest_client_factory = rest_client_factory
        self._async_client_factory = async_client_factory
        self._sleep = sleep
        self._async_sleep = async_sleep

    @classmethod
    def from_env(
        cls,
        env_path: str | Path = ".env",
        *,
        config: SarvamSTTConfig | None = None,
    ) -> SarvamSTT:
        load_dotenv(dotenv_path=Path(env_path), override=False)
        return cls(os.getenv("SARVAM_API_KEY", ""), config=config)

    def transcribe_rest(
        self,
        audio_path: str | Path,
        *,
        language_code: str = "hi-IN",
    ) -> STTResult:
        """Transcribe a complete file through Sarvam REST with bounded retries."""
        operation_started = time.perf_counter_ns()
        try:
            audio = inspect_audio(audio_path, require_streaming_wav=False)
            if audio.bytes > self.config.max_audio_bytes:
                raise AudioValidationError("REST transcription audio exceeds the byte limit")
            if (
                audio.duration_ms is not None
                and audio.duration_ms > self.config.max_audio_duration_ms
            ):
                raise AudioValidationError("REST transcription accepts at most 30 seconds")
        except (AudioValidationError, OSError) as error:
            return self._error_result(
                "rest", operation_started, 0, "invalid_audio", str(error), audio=None
            )

        last_error: BaseException | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            client = self._rest_client_factory(
                api_subscription_key=self._api_key,
                timeout=self.config.timeout_s,
            )
            try:
                with Path(audio.path).open("rb") as handle:
                    response = client.speech_to_text.transcribe(
                        file=handle,
                        model=self.config.model,
                        mode=self.config.mode,
                        language_code=language_code,
                    )
                transcript = str(getattr(response, "transcript", "") or "").strip()
                if not transcript:
                    return self._error_result(
                        "rest",
                        operation_started,
                        attempt,
                        "empty_transcript",
                        "Sarvam returned an empty transcript",
                        audio=audio,
                    )
                return STTResult(
                    provider=SARVAM_PROVIDER,
                    transport="rest",
                    model=SARVAM_MODEL,
                    mode=SARVAM_MODE,
                    status="ok",
                    transcript=transcript,
                    latency_ms=_elapsed_ms(operation_started),
                    attempts=attempt,
                    language_code=getattr(response, "language_code", None),
                    language_probability=getattr(response, "language_probability", None),
                    request_id=getattr(response, "request_id", None),
                    audio_duration_ms=audio.duration_ms,
                    sample_rate_hz=audio.sample_rate_hz,
                )
            except Exception as error:
                last_error = error
                status = _http_status(error)
                if not _retryable(error) or attempt == self.config.max_attempts:
                    code, message = _classify_error(error)
                    return self._error_result(
                        "rest",
                        operation_started,
                        attempt,
                        code,
                        message,
                        audio=audio,
                        http_status=status,
                        retryable=_retryable(error),
                    )
                self._sleep(self._backoff(attempt))
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        raise AssertionError(f"unreachable retry state: {last_error}")

    async def transcribe_streaming(
        self,
        audio_path: str | Path,
        *,
        language_code: str = "hi-IN",
        speech_end_offset_ms: float | None = None,
        pace_audio: bool = True,
    ) -> STTResult:
        """Replay a WAV through the WebSocket at microphone pace and measure finalization."""
        operation_started = time.perf_counter_ns()
        try:
            audio = inspect_audio(audio_path, require_streaming_wav=True)
            if audio.bytes > self.config.max_audio_bytes:
                raise AudioValidationError("Streaming audio exceeds the byte limit")
            if audio.duration_ms > self.config.max_audio_duration_ms:
                raise AudioValidationError("Streaming transcription accepts at most 30 seconds")
            wav_chunks = _wav_chunks(Path(audio.path), self.config.streaming_chunk_ms)
        except (AudioValidationError, OSError, wave.Error) as error:
            return self._error_result(
                "streaming", operation_started, 0, "invalid_audio", str(error), audio=None
            )

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await asyncio.wait_for(
                    self._stream_once(
                        wav_chunks,
                        audio,
                        language_code=language_code,
                        speech_end_offset_ms=speech_end_offset_ms,
                        pace_audio=pace_audio,
                        attempt=attempt,
                        operation_started=operation_started,
                    ),
                    timeout=self.config.timeout_s + (audio.duration_ms or 0) / 1000,
                )
                return result
            except Exception as error:
                status = _http_status(error)
                if not _retryable(error) or attempt == self.config.max_attempts:
                    code, message = _classify_error(error)
                    return self._error_result(
                        "streaming",
                        operation_started,
                        attempt,
                        code,
                        message,
                        audio=audio,
                        http_status=status,
                        retryable=_retryable(error),
                    )
                await self._async_sleep(self._backoff(attempt))
        raise AssertionError("unreachable retry state")

    async def _stream_once(
        self,
        wav_chunks: list[tuple[bytes, float]],
        audio: AudioInfo,
        *,
        language_code: str,
        speech_end_offset_ms: float | None,
        pace_audio: bool,
        attempt: int,
        operation_started: int,
    ) -> STTResult:
        client = self._async_client_factory(
            api_subscription_key=self._api_key,
            timeout=self.config.timeout_s,
        )
        connect_started = time.perf_counter_ns()
        connect = client.speech_to_text_streaming.connect(
            language_code=language_code,
            model=self.config.model,
            mode=self.config.mode,
            sample_rate=str(self.config.sample_rate_hz),
            high_vad_sensitivity=str(self.config.high_vad_sensitivity).lower(),
            vad_signals=str(self.config.vad_signals).lower(),
            flush_signal=str(self.config.flush_signal).lower(),
            input_audio_codec="wav",
        )
        async with connect as socket:
            connection_latency_ms = _elapsed_ms(connect_started)
            messages: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()

            async def receive() -> None:
                try:
                    while True:
                        message = await socket.recv()
                        await messages.put((time.perf_counter_ns(), message))
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    await messages.put((time.perf_counter_ns(), error))

            receiver = asyncio.create_task(receive())
            first_audio_ns = time.perf_counter_ns()
            try:
                for chunk, duration_ms in wav_chunks:
                    await socket.transcribe(
                        audio=base64.b64encode(chunk).decode("ascii"),
                        encoding="audio/wav",
                        sample_rate=self.config.sample_rate_hz,
                    )
                    if pace_audio:
                        await self._async_sleep(duration_ms / 1000)
                audio_sent_ns = time.perf_counter_ns()
                await socket.flush()
                observed = await self._collect_stream_messages(messages, audio_sent_ns)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
                close = getattr(client, "close", None)
                if callable(close):
                    maybe_awaitable = close()
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable

        transcripts = observed["transcripts"]
        transcript = _merge_transcripts([item[1] for item in transcripts if item[1]])
        if not transcript:
            raise _StreamingAPIError("empty_transcript", "Sarvam returned no final transcript")
        final_ns = transcripts[-1][0]
        first_transcript_ns = transcripts[0][0]
        eos_offset = (
            float(speech_end_offset_ms)
            if speech_end_offset_ms is not None
            else float(audio.duration_ms or 0)
        )
        eos_ns = first_audio_ns + int(eos_offset * 1e6)
        partials = observed["partials"]
        return STTResult(
            provider=SARVAM_PROVIDER,
            transport="streaming",
            model=SARVAM_MODEL,
            mode=SARVAM_MODE,
            status="ok",
            transcript=transcript,
            latency_ms=_elapsed_ms(operation_started, final_ns),
            attempts=attempt,
            language_code=observed["language_code"],
            language_probability=observed["language_probability"],
            request_id=observed["request_id"],
            connection_latency_ms=connection_latency_ms,
            time_to_first_partial_ms=(
                (partials[0][0] - first_audio_ns) / 1e6 if partials else None
            ),
            time_to_first_transcript_ms=(first_transcript_ns - first_audio_ns) / 1e6,
            end_of_speech_to_final_ms=max(0.0, (final_ns - eos_ns) / 1e6),
            audio_duration_ms=audio.duration_ms,
            sample_rate_hz=audio.sample_rate_hz,
            partial_transcripts=tuple(item[1] for item in partials),
            events=tuple(observed["events"]),
        )

    async def _collect_stream_messages(
        self,
        messages: asyncio.Queue[tuple[int, Any]],
        audio_sent_ns: int,
    ) -> dict[str, Any]:
        transcripts: list[tuple[int, str]] = []
        partials: list[tuple[int, str]] = []
        events: list[dict[str, Any]] = []
        request_id = None
        language_code = None
        language_probability = None
        received_final = False
        while True:
            timeout = (
                self.config.post_final_grace_s
                if received_final
                else self.config.timeout_s
            )
            try:
                received_ns, message = await asyncio.wait_for(messages.get(), timeout=timeout)
            except TimeoutError:
                if received_final:
                    break
                raise
            if isinstance(message, Exception):
                raise message
            message_type = getattr(message, "type", None)
            data = getattr(message, "data", None)
            if message_type == "error":
                raise _StreamingAPIError(
                    str(getattr(data, "code", "websocket_api_error")),
                    str(getattr(data, "error", "Sarvam streaming API error")),
                )
            if message_type == "events":
                events.append(
                    {
                        "signal_type": getattr(data, "signal_type", None),
                        "event_type": getattr(data, "event_type", None),
                        "received_ms_after_audio_start": None,
                    }
                )
                continue
            if message_type != "data":
                continue
            text = str(getattr(data, "transcript", "") or "").strip()
            is_final = getattr(data, "is_final", True)
            if is_final is False:
                partials.append((received_ns, text))
            elif text:
                transcripts.append((received_ns, text))
                received_final = True
            request_id = getattr(data, "request_id", request_id)
            language_code = getattr(data, "language_code", language_code)
            language_probability = getattr(data, "language_probability", language_probability)
            if received_ns >= audio_sent_ns and text and is_final is not False:
                received_final = True
        return {
            "transcripts": transcripts,
            "partials": partials,
            "events": events,
            "request_id": request_id,
            "language_code": language_code,
            "language_probability": language_probability,
        }

    def _backoff(self, attempt: int) -> float:
        return min(
            self.config.backoff_base_s * (2 ** max(0, attempt - 1)),
            self.config.backoff_max_s,
        )

    def _error_result(
        self,
        transport: Literal["rest", "streaming"],
        started_ns: int,
        attempts: int,
        code: str,
        message: str,
        *,
        audio: AudioInfo | None,
        http_status: int | None = None,
        retryable: bool = False,
    ) -> STTResult:
        return STTResult(
            provider=SARVAM_PROVIDER,
            transport=transport,
            model=SARVAM_MODEL,
            mode=SARVAM_MODE,
            status="error",
            transcript="",
            latency_ms=_elapsed_ms(started_ns),
            attempts=attempts,
            error_code=code,
            error_message=message,
            http_status=http_status,
            retryable=retryable,
            audio_duration_ms=audio.duration_ms if audio else None,
            sample_rate_hz=audio.sample_rate_hz if audio else None,
        )


def inspect_audio(path: str | Path, *, require_streaming_wav: bool) -> AudioInfo:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioValidationError(f"Audio file does not exist: {audio_path}")
    size = audio_path.stat().st_size
    if size == 0:
        raise AudioValidationError("Audio file is empty")
    suffix = audio_path.suffix.lower()
    if require_streaming_wav and suffix != ".wav":
        raise AudioValidationError("Sarvam streaming evaluation requires WAV input")
    if not require_streaming_wav and suffix not in SUPPORTED_REST_SUFFIXES:
        raise AudioValidationError(f"Unsupported REST audio extension: {suffix}")
    if suffix != ".wav":
        return AudioInfo(str(audio_path), size, None, None, None, None, None)
    try:
        with wave.open(str(audio_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            compression = handle.getcomptype()
    except (EOFError, wave.Error) as error:
        raise AudioValidationError(f"Malformed WAV audio: {error}") from error
    if frames <= 0:
        raise AudioValidationError("WAV audio contains no frames")
    if require_streaming_wav:
        if compression != "NONE":
            raise AudioValidationError("Streaming WAV must be uncompressed PCM")
        if channels != 1 or sample_width != 2 or sample_rate != 16_000:
            raise AudioValidationError("Streaming WAV must be mono PCM16 at 16 kHz")
    duration_ms = frames / sample_rate * 1000
    if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise AudioValidationError("Audio duration is invalid")
    return AudioInfo(
        str(audio_path), size, duration_ms, sample_rate, channels, sample_width, frames
    )


def _wav_chunks(path: Path, chunk_ms: int) -> list[tuple[bytes, float]]:
    chunks: list[tuple[bytes, float]] = []
    with wave.open(str(path), "rb") as source:
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frames_per_chunk = max(1, round(sample_rate * chunk_ms / 1000))
        while frames := source.readframes(frames_per_chunk):
            frame_count = len(frames) // (channels * sample_width)
            output = io.BytesIO()
            with wave.open(output, "wb") as target:
                target.setnchannels(channels)
                target.setsampwidth(sample_width)
                target.setframerate(sample_rate)
                target.writeframes(frames)
            chunks.append((output.getvalue(), frame_count / sample_rate * 1000))
    if not chunks:
        raise AudioValidationError("WAV audio contains no streamable frames")
    return chunks


def _elapsed_ms(started_ns: int, ended_ns: int | None = None) -> float:
    return ((ended_ns or time.perf_counter_ns()) - started_ns) / 1e6


def _merge_transcripts(segments: list[str]) -> str:
    """Join finalized VAD segments while removing exact boundary-word overlap."""
    merged: list[str] = []
    for segment in segments:
        incoming = segment.strip().split()
        if not incoming:
            continue
        overlap = 0
        limit = min(12, len(merged), len(incoming))
        for size in range(limit, 0, -1):
            left = [_comparison_token(token) for token in merged[-size:]]
            right = [_comparison_token(token) for token in incoming[:size]]
            if left == right:
                overlap = size
                break
        merged.extend(incoming[overlap:])
    return " ".join(merged).strip()


def _comparison_token(token: str) -> str:
    return re.sub(r"[^\w]+", "", token.casefold(), flags=re.UNICODE)


def _http_status(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _retryable(error: BaseException) -> bool:
    status = _http_status(error)
    if status in RETRYABLE_HTTP_STATUSES:
        return True
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        return True
    if isinstance(error, httpx.TransportError):
        return True
    close_code = getattr(error, "code", None)
    return close_code in {1001, 1006, 1011}


def _classify_error(error: BaseException) -> tuple[str, str]:
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        return "timeout", "Sarvam transcription timed out"
    if isinstance(error, _StreamingAPIError):
        return error.code, str(error)
    status = _http_status(error)
    if status == 403:
        return "authentication_error", "Sarvam rejected the API key or access"
    if status == 429:
        return "rate_limit_or_quota", "Sarvam rate limit or quota was exceeded"
    if status in {400, 413, 422}:
        return "invalid_request", _safe_error_message(error)
    if status in {500, 503}:
        return "sarvam_unavailable", _safe_error_message(error)
    if isinstance(error, httpx.TransportError):
        return "transport_error", str(error)
    if isinstance(error, ApiError):
        return "api_error", _safe_error_message(error)
    return "websocket_or_client_error", str(error)


def _safe_error_message(error: BaseException) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", body.get("message", body.get("detail")))
        if detail:
            return str(detail)[:500]
    return str(error)[:500]
