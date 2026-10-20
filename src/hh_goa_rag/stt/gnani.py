"""Gnani Prisma speech-to-text service built on the official ``gnani-vachana`` SDK.

The SDK's ``GnaniSTTClient.transcribe(audio, language_code=..., format=...)`` posts one clip to
``https://api.vachana.ai/stt/v3`` and returns ``{"success", "request_id", "transcript", ...}``.
This wrapper keeps the pipeline contract identical to :class:`SarvamSTT`: ``transcribe_rest``
returns a provider-neutral :class:`STTResult` and never raises for provider or audio errors.

Reference: https://pypi.org/project/gnani-vachana/ and
https://docs.gnani.ai/api/STT/speech-to-text
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from hh_goa_rag.stt.sarvam import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_DURATION_MS,
    AudioInfo,
    AudioValidationError,
    STTResult,
    inspect_audio,
)

GNANI_PROVIDER = "gnani"
# Gnani's REST docs name the default STT model "Prisma v2.5"; the SDK exposes no model switch.
GNANI_STT_MODEL = "prisma-v2.5"
GNANI_API_KEY_ENV = "GNANI_API_KEY"
# Mirrors gnani.stt.SUPPORTED_LANGUAGES (gnani-vachana 0.8.1) so the language gate works even
# before the SDK is imported. Batch STT differs; this service only uses REST.
GNANI_STT_LANGUAGES = frozenset(
    {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"}
)
# Extensions accepted by the SDK's REST client (gnani.stt.client.SUPPORTED_EXTENSIONS).
GNANI_REST_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GnaniSTTConfig:
    """Immutable Prisma REST transcription configuration."""

    format: Literal["verbatim", "transcribe"] = "verbatim"
    timeout_s: float = 20.0
    max_attempts: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 2.0
    max_audio_bytes: int = MAX_AUDIO_BYTES
    # The API accepts up to 60 s (ideal <= 30 s); the pipeline keeps its 30 s product limit.
    max_audio_duration_ms: int = MAX_AUDIO_DURATION_MS

    def __post_init__(self) -> None:
        if self.format not in {"verbatim", "transcribe"}:
            raise ValueError("format must be 'verbatim' or 'transcribe'")
        if self.timeout_s <= 0 or self.max_attempts <= 0:
            raise ValueError("timeout_s and max_attempts must be positive")
        if self.max_audio_bytes <= 0 or self.max_audio_duration_ms <= 0:
            raise ValueError("Audio limits must be positive")
        if not 0 <= self.backoff_base_s <= self.backoff_max_s:
            raise ValueError("Invalid bounded-backoff configuration")


def _default_client_factory(**kwargs: Any) -> Any:
    from gnani.stt import GnaniSTTClient

    return GnaniSTTClient(**kwargs)


class GnaniSTT:
    """Gnani Prisma REST service exposing the same ``transcribe_rest`` contract as Sarvam."""

    def __init__(
        self,
        api_key: str,
        *,
        config: GnaniSTTConfig | None = None,
        client_factory: Callable[..., Any] = _default_client_factory,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError(f"{GNANI_API_KEY_ENV} is missing; set it in the ignored .env file")
        self._api_key = api_key.strip()
        self.config = config or GnaniSTTConfig()
        self._client_factory = client_factory
        self._sleep = sleep

    @classmethod
    def from_env(
        cls,
        env_path: str | Path = ".env",
        *,
        config: GnaniSTTConfig | None = None,
    ) -> GnaniSTT:
        load_dotenv(dotenv_path=Path(env_path), override=False)
        return cls(os.getenv(GNANI_API_KEY_ENV, ""), config=config)

    @staticmethod
    def supports_language(language_code: str | None) -> bool:
        return language_code in GNANI_STT_LANGUAGES

    def transcribe_rest(
        self,
        audio_path: str | Path,
        *,
        language_code: str = "hi-IN",
    ) -> STTResult:
        """Transcribe one complete clip through Prisma REST with bounded retries."""
        operation_started = time.perf_counter_ns()
        if not self.supports_language(language_code):
            return self._error_result(
                operation_started,
                0,
                "unsupported_language",
                f"Gnani Prisma does not support language_code={language_code!r}",
                audio=None,
            )
        try:
            audio = inspect_audio(audio_path, require_streaming_wav=False)
            if Path(audio.path).suffix.lower() not in GNANI_REST_SUFFIXES:
                raise AudioValidationError("Unsupported audio extension for Gnani Prisma REST")
            if audio.bytes > self.config.max_audio_bytes:
                raise AudioValidationError("REST transcription audio exceeds the byte limit")
            if (
                audio.duration_ms is not None
                and audio.duration_ms > self.config.max_audio_duration_ms
            ):
                raise AudioValidationError("REST transcription accepts at most 30 seconds")
        except (AudioValidationError, OSError) as error:
            return self._error_result(
                operation_started, 0, "invalid_audio", str(error), audio=None
            )

        try:
            client = self._client_factory(
                api_key=self._api_key, timeout=int(max(1, round(self.config.timeout_s)))
            )
        except Exception as error:
            code, message = _classify_error(error)
            return self._error_result(operation_started, 0, code, message, audio=audio)

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = client.transcribe(
                    audio.path,
                    language_code=language_code,
                    format=self.config.format,
                )
                transcript = str(_get(response, "transcript", "") or "").strip()
                if _get(response, "success", True) is False or not transcript:
                    return self._error_result(
                        operation_started,
                        attempt,
                        "empty_transcript",
                        "Gnani returned an empty transcript",
                        audio=audio,
                    )
                return STTResult(
                    provider=GNANI_PROVIDER,
                    transport="rest",
                    model=GNANI_STT_MODEL,
                    mode=self.config.format,
                    status="ok",
                    transcript=transcript,
                    latency_ms=_elapsed_ms(operation_started),
                    attempts=attempt,
                    language_code=language_code,
                    request_id=_get(response, "request_id", None),
                    audio_duration_ms=audio.duration_ms,
                    sample_rate_hz=audio.sample_rate_hz,
                )
            except Exception as error:
                retryable = _retryable(error)
                if not retryable or attempt == self.config.max_attempts:
                    code, message = _classify_error(error)
                    return self._error_result(
                        operation_started,
                        attempt,
                        code,
                        message,
                        audio=audio,
                        http_status=_http_status(error),
                        retryable=retryable,
                    )
                self._sleep(self._backoff(attempt))
        raise AssertionError("unreachable retry state")

    def _backoff(self, attempt: int) -> float:
        return min(
            self.config.backoff_base_s * (2 ** max(0, attempt - 1)),
            self.config.backoff_max_s,
        )

    def _error_result(
        self,
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
            provider=GNANI_PROVIDER,
            transport="rest",
            model=GNANI_STT_MODEL,
            mode=self.config.format,
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


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1e6


def _http_status(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _is_requests_error(error: BaseException, name: str) -> bool:
    try:
        import requests
    except ImportError:  # pragma: no cover - requests ships with gnani-vachana
        return False
    return isinstance(error, getattr(requests.exceptions, name))


def _retryable(error: BaseException) -> bool:
    if _http_status(error) in RETRYABLE_HTTP_STATUSES:
        return True
    if isinstance(error, TimeoutError):
        return True
    return _is_requests_error(error, "Timeout") or _is_requests_error(error, "ConnectionError")


def _classify_error(error: BaseException) -> tuple[str, str]:
    name = type(error).__name__
    if name == "AuthenticationError":
        return "authentication_error", "Gnani API key is missing or was rejected"
    if name == "InvalidAudioError":
        return "invalid_audio", str(error)[:500]
    if isinstance(error, ValueError) and "language_code" in str(error):
        return "unsupported_language", str(error)[:500]
    if isinstance(error, TimeoutError) or _is_requests_error(error, "Timeout"):
        return "timeout", "Gnani transcription timed out"
    if _is_requests_error(error, "ConnectionError"):
        return "transport_error", str(error)[:500]
    status = _http_status(error)
    if status in {401, 403}:
        return "authentication_error", "Gnani rejected the API key, organization, or credits"
    if status == 429:
        return "rate_limit_or_quota", "Gnani rate limit or quota was exceeded"
    if status in {400, 413, 422}:
        return "invalid_request", _safe_error_message(error)
    if status is not None and status >= 500:
        return "gnani_unavailable", _safe_error_message(error)
    return "client_error", str(error)[:500]


def _safe_error_message(error: BaseException) -> str:
    """Extract ``error.message`` from Gnani's JSON error body without echoing headers."""
    body = getattr(error, "body", None)
    if isinstance(body, str):
        try:
            import json

            parsed = json.loads(body)
        except ValueError:
            return body[:500]
        if isinstance(parsed, dict):
            detail = parsed.get("error", parsed.get("message", parsed.get("detail")))
            if isinstance(detail, dict):
                detail = detail.get("message", detail)
            if detail:
                return str(detail)[:500]
    return str(error)[:500]
