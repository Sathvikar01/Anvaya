"""Speech-to-text services for the Voice-RAG pipeline."""

from hh_goa_rag.stt.gnani import GnaniSTT, GnaniSTTConfig
from hh_goa_rag.stt.providers import FallbackSTT, build_stt_from_env, resolve_stt_provider
from hh_goa_rag.stt.sarvam import SarvamSTT, SarvamSTTConfig, STTResult

__all__ = [
    "FallbackSTT",
    "GnaniSTT",
    "GnaniSTTConfig",
    "STTResult",
    "SarvamSTT",
    "SarvamSTTConfig",
    "build_stt_from_env",
    "resolve_stt_provider",
]
