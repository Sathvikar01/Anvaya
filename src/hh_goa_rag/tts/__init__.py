"""Text-to-speech services for reading grounded answers aloud.

``HH_RAG_TTS_PROVIDER`` selects the provider: ``gnani`` (default, Timbre via ``gnani-vachana``)
or ``off``. Without ``GNANI_API_KEY`` the stage is disabled and the demo stays text-only, which
was the behaviour before TTS existed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from hh_goa_rag.tts.gnani import GnaniTTS, GnaniTTSConfig, TTSResult

LOGGER = logging.getLogger(__name__)

TTS_PROVIDER_ENV = "HH_RAG_TTS_PROVIDER"
DEFAULT_TTS_PROVIDER = "gnani"


def build_tts_from_env(env_path: str | Path = ".env") -> Any | None:
    """Return the configured TTS service, or ``None`` when TTS is disabled or unconfigured."""
    load_dotenv(dotenv_path=Path(env_path), override=False)
    provider = os.getenv(TTS_PROVIDER_ENV, DEFAULT_TTS_PROVIDER).strip().lower()
    if provider in {"off", "none", "disabled", ""}:
        return None
    if provider != "gnani":
        raise ValueError(f"Unsupported {TTS_PROVIDER_ENV}: {provider}")
    if not os.getenv("GNANI_API_KEY", "").strip():
        LOGGER.info("GNANI_API_KEY is not set; text-to-speech is disabled")
        return None
    return GnaniTTS.from_env(env_path)


__all__ = ["GnaniTTS", "GnaniTTSConfig", "TTSResult", "build_tts_from_env"]
