"""Gnani Timbre text-to-speech service built on the official ``gnani-vachana`` SDK.

Uses ``GnaniTTSClient.synthesize(text, voice=..., model=..., language=..., speed=...,
audio_config=AudioConfig(...))`` which returns the complete audio as bytes.
``language`` and ``speed`` are only sent for ``timbre-v2.5``, as the SDK requires.

Reference: https://pypi.org/project/gnani-vachana/ and
https://docs.gnani.ai/api/TTS/tts-inference
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

GNANI_PROVIDER = "gnani"
GNANI_API_KEY_ENV = "GNANI_API_KEY"
DEFAULT_TTS_MODEL = "timbre-v2.5"
DEFAULT_TTS_VOICE = "Kaveri"
# timbre-v2.5 language codes (gnani.tts.SUPPORTED_TTS_LANGUAGES, gnani-vachana 0.8.1).
GNANI_TTS_LANGUAGES = frozenset(
    {
        "hi-IN",
        "en-IN",
        "ta-IN",
        "te-IN",
        "kn-IN",
        "ml-IN",
        "mr-IN",
        "bn-IN",
        "gu-IN",
        "pa-IN",
    }
)
TTS_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
MAX_TTS_TEXT_CHARS = 1_000


@dataclass(frozen=True)
class GnaniTTSConfig:
    """Immutable Timbre synthesis configuration."""

    model: Literal["timbre-v2.0", "timbre-v2.5"] = DEFAULT_TTS_MODEL
    voice: str = DEFAULT_TTS_VOICE
    speed: float = 1.0
    sample_rate_hz: int = 24_000
    container: Literal["wav", "mp3"] = "wav"
    timeout_s: float = 20.0
    max_attempts: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 1.0
    max_text_chars: int = MAX_TTS_TEXT_CHARS

    def __post_init__(self) -> None:
        if self.model not in {"timbre-v2.0", "timbre-v2.5"}:
            raise ValueError("model must be 'timbre-v2.0' or 'timbre-v2.5'")
        if not self.voice.strip():
            raise ValueError("voice must be a Timbre voice name")
        if not 0.85 <= self.speed <= 1.15:
            raise ValueError("speed must be between 0.85 and 1.15")
        if self.sample_rate_hz not in TTS_SAMPLE_RATES:
            raise ValueError(f"sample_rate_hz must be one of {TTS_SAMPLE_RATES}")
        if self.container not in {"wav", "mp3"}:
            raise ValueError("container must be 'wav' or 'mp3'")
        if self.timeout_s <= 0 or self.max_attempts <= 0 or self.max_text_chars <= 0:
            raise ValueError("timeout_s, max_attempts and max_text_chars must be positive")
        if not 0 <= self.backoff_base_s <= self.backoff_max_s:
            raise ValueError("Invalid bounded-backoff configuration")

    @property
    def mime_type(self) -> str:
        return "audio/wav" if self.container == "wav" else "audio/mpeg"


@dataclass(frozen=True)
class TTSResult:
    """Provider-neutral synthesis observation. ``audio`` is excluded from ``to_dict``."""

    provider: Literal["gnani"]
    model: str
    voice: str
    status: Literal["ok", "error"]
    latency_ms: float
    attempts: int
    mime_type: str
    language_code: str | None = None
    characters: int = 0
    audio_bytes: int = 0
    error_code: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    retryable: bool = False
    audio: bytes = field(default=b"", repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("audio", None)
        return payload


def _default_client_factory(**kwargs: Any) -> Any:
    from gnani.tts import GnaniTTSClient

    return GnaniTTSClient(**kwargs)


def _default_audio_config_factory(**kwargs: Any) -> Any:
    from gnani.tts import AudioConfig

    return AudioConfig(**kwargs)


class GnaniTTS:
    """Synchronous Timbre REST synthesis with bounded retries."""

    def __init__(
        self,
        api_key: str,
        *,
        config: GnaniTTSConfig | None = None,
        client_factory: Callable[..., Any] = _default_client_factory,
        audio_config_factory: Callable[..., Any] = _default_audio_config_factory,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError(f"{GNANI_API_KEY_ENV} is missing; set it in the ignored .env file")
        self._api_key = api_key.strip()
        self.config = config or GnaniTTSConfig()
        self._client_factory = client_factory
        self._audio_config_factory = audio_config_factory
        self._sleep = sleep
        self._client: Any | None = None

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> GnaniTTS:
        load_dotenv(dotenv_path=Path(env_path), override=False)
        config = GnaniTTSConfig(
            model=os.getenv("GNANI_TTS_MODEL", DEFAULT_TTS_MODEL).strip(),  # type: ignore[arg-type]
            voice=os.getenv("GNANI_TTS_VOICE", DEFAULT_TTS_VOICE).strip(),
            speed=float(os.getenv("GNANI_TTS_SPEED", "1.0")),
        )
        return cls(os.getenv(GNANI_API_KEY_ENV, ""), config=config)

    @staticmethod
    def supports_language(language_code: str | None) -> bool:
        return language_code in GNANI_TTS_LANGUAGES

    def synthesize(self, text: str, *, language_code: str = "hi-IN") -> TTSResult:
        started = time.perf_counter_ns()
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return self._error(started, 0, "empty_text", "Nothing to synthesize", language_code)
        if len(cleaned) > self.config.max_text_chars:
            return self._error(
                started,
                0,
                "text_too_long",
                f"Text exceeds {self.config.max_text_chars} characters",
                language_code,
            )
        if self.config.model == "timbre-v2.5" and not self.supports_language(language_code):
            return self._error(
                started,
                0,
                "unsupported_language",
                f"Gnani Timbre does not support language_code={language_code!r}",
                language_code,
            )
        kwargs: dict[str, Any] = {
            "voice": self.config.voice,
            "model": self.config.model,
            "audio_config": self._audio_config_factory(
                sample_rate=self.config.sample_rate_hz,
                encoding="linear_pcm",
                container=self.config.container,
            ),
        }
        if self.config.model == "timbre-v2.5":
            kwargs["language"] = language_code
            kwargs["speed"] = self.config.speed

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                audio = self._get_client().synthesize(cleaned, **kwargs)
                if not audio:
                    return self._error(
                        started, attempt, "empty_audio", "Gnani returned no audio", language_code
                    )
                return TTSResult(
                    provider=GNANI_PROVIDER,
                    model=self.config.model,
                    voice=self.config.voice,
                    status="ok",
                    latency_ms=_elapsed_ms(started),
                    attempts=attempt,
                    mime_type=self.config.mime_type,
                    language_code=language_code,
                    characters=len(cleaned),
                    audio_bytes=len(audio),
                    audio=bytes(audio),
                )
            except Exception as error:
                retryable = _retryable(error)
                if not retryable or attempt == self.config.max_attempts:
                    code, message = _classify_error(error)
                    return self._error(
                        started,
                        attempt,
                        code,
                        message,
                        language_code,
                        http_status=_http_status(error),
                        retryable=retryable,
                    )
                self._sleep(
                    min(
                        self.config.backoff_base_s * (2 ** (attempt - 1)),
                        self.config.backoff_max_s,
                    )
                )
        raise AssertionError("unreachable retry state")

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(
                api_key=self._api_key, timeout=int(max(1, round(self.config.timeout_s)))
            )
        return self._client

    def _error(
        self,
        started: int,
        attempts: int,
        code: str,
        message: str,
        language_code: str | None,
        *,
        http_status: int | None = None,
        retryable: bool = False,
    ) -> TTSResult:
        return TTSResult(
            provider=GNANI_PROVIDER,
            model=self.config.model,
            voice=self.config.voice,
            status="error",
            latency_ms=_elapsed_ms(started),
            attempts=attempts,
            mime_type=self.config.mime_type,
            language_code=language_code,
            error_code=code,
            error_message=message,
            http_status=http_status,
            retryable=retryable,
        )


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
    if type(error).__name__ == "AuthenticationError":
        return "authentication_error", "Gnani API key is missing or was rejected"
    if isinstance(error, ValueError):
        # SDK-side validation: unknown voice/model, language on timbre-v2.0, bad audio config.
        return "invalid_request", str(error)[:500]
    if isinstance(error, TimeoutError) or _is_requests_error(error, "Timeout"):
        return "timeout", "Gnani synthesis timed out"
    if _is_requests_error(error, "ConnectionError"):
        return "transport_error", str(error)[:500]
    status = _http_status(error)
    if status in {401, 403}:
        return "authentication_error", "Gnani rejected the API key, organization, or credits"
    if status == 429:
        return "rate_limit_or_quota", "Gnani rate limit or quota was exceeded"
    if status is not None and 400 <= status < 500:
        return "invalid_request", str(getattr(error, "body", error))[:500]
    if status is not None and status >= 500:
        return "gnani_unavailable", str(getattr(error, "body", error))[:500]
    return "client_error", str(error)[:500]
