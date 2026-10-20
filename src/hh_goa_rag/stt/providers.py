"""STT provider selection: Gnani Prisma first, Sarvam Saaras v3 as the resilient fallback.

``HH_RAG_STT_PROVIDER`` picks the primary provider:

* ``gnani`` (default) - Gnani Prisma via ``gnani-vachana``. When ``SARVAM_API_KEY`` is also set,
  Sarvam answers any request Gnani cannot (unsupported language, provider outage, auth error).
  When ``GNANI_API_KEY`` is missing, the pipeline falls back to Sarvam alone so an existing
  deployment keeps working unchanged.
* ``sarvam`` - the original frozen Saaras v3 path, with no Gnani calls.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from hh_goa_rag.stt.sarvam import SarvamSTT, STTResult

LOGGER = logging.getLogger(__name__)

STT_PROVIDER_ENV = "HH_RAG_STT_PROVIDER"
DEFAULT_STT_PROVIDER = "gnani"
SUPPORTED_STT_PROVIDERS = {"gnani", "sarvam"}
# Errors that the secondary provider cannot fix: bad audio is bad for every provider.
NON_RECOVERABLE_ERRORS = {"invalid_audio"}


class FallbackSTT:
    """Try the primary provider, then the secondary one when the primary did not succeed."""

    def __init__(self, primary: Any, secondary: Any) -> None:
        self.primary = primary
        self.secondary = secondary

    def transcribe_rest(self, audio_path: str | Path, *, language_code: str = "hi-IN") -> Any:
        try:
            first = self.primary.transcribe_rest(audio_path, language_code=language_code)
        except Exception:
            LOGGER.exception("Primary STT provider raised; using fallback provider")
            first = None
        if first is not None and (
            _field(first, "status") == "ok"
            or _field(first, "error_code") in NON_RECOVERABLE_ERRORS
        ):
            return first
        if first is not None:
            LOGGER.warning(
                "Primary STT provider failed (%s); using fallback provider",
                _field(first, "error_code"),
            )
        second = self.secondary.transcribe_rest(audio_path, language_code=language_code)
        if first is not None and isinstance(second, STTResult):
            fallback_event = {
                "fallback_from": _field(first, "provider"),
                "fallback_reason": _field(first, "error_code"),
                "primary_latency_ms": _field(first, "latency_ms"),
            }
            second = replace(second, events=(*second.events, fallback_event))
        return second


def resolve_stt_provider(env_path: str | Path = ".env") -> str:
    """Return the provider that will actually serve STT, without building clients."""
    load_dotenv(dotenv_path=Path(env_path), override=False)
    requested = os.getenv(STT_PROVIDER_ENV, DEFAULT_STT_PROVIDER).strip().lower()
    if requested not in SUPPORTED_STT_PROVIDERS:
        raise ValueError(f"Unsupported {STT_PROVIDER_ENV}: {requested}")
    has_gnani = bool(os.getenv("GNANI_API_KEY", "").strip())
    has_sarvam = bool(os.getenv("SARVAM_API_KEY", "").strip())
    if requested == "gnani":
        if has_gnani:
            return "gnani+sarvam" if has_sarvam else "gnani"
        if has_sarvam:
            return "sarvam"
        raise ValueError("GNANI_API_KEY is missing (and no SARVAM_API_KEY fallback is set)")
    if not has_sarvam:
        raise ValueError("SARVAM_API_KEY is missing")
    return "sarvam"


def build_stt_from_env(env_path: str | Path = ".env") -> Any:
    """Construct the configured STT service with the old Sarvam path as fallback."""
    provider = resolve_stt_provider(env_path)
    if provider == "sarvam":
        if os.getenv(STT_PROVIDER_ENV, DEFAULT_STT_PROVIDER).strip().lower() == "gnani":
            LOGGER.warning("GNANI_API_KEY is not set; serving speech-to-text with Sarvam")
        return SarvamSTT.from_env(env_path)
    from hh_goa_rag.stt.gnani import GnaniSTT

    gnani = GnaniSTT.from_env(env_path)
    if provider == "gnani+sarvam":
        return FallbackSTT(gnani, SarvamSTT.from_env(env_path))
    return gnani


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
