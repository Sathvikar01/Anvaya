"""Speech-to-text services for the Voice-RAG pipeline."""

from hh_goa_rag.stt.sarvam import SarvamSTT, SarvamSTTConfig, STTResult

__all__ = ["STTResult", "SarvamSTT", "SarvamSTTConfig"]
