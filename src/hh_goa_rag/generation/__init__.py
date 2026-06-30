"""Grounded answer-generation services for the frozen HH Goa retriever."""

from .fallback import FallbackGeneration
from .gemini import GEMINI_MODEL, GeminiGeneration, GeminiGenerationConfig
from .groq import CALLABLE_GROQ_MODELS, GROQ_MODEL, GroqGeneration, GroqGenerationConfig
from .prompts import PROMPT_VARIANTS, build_messages
from .qwen_gguf import (
    QWEN_GGUF_FILENAME,
    QWEN_GGUF_MODEL,
    QWEN_GGUF_REPOSITORY,
    QwenGGUFGeneration,
    QwenGGUFGenerationConfig,
    resolve_gguf_model,
)
from .sarvam import (
    CALLABLE_SARVAM_MODELS,
    GenerationContext,
    GenerationResult,
    SarvamGeneration,
    SarvamGenerationConfig,
)

__all__ = [
    "CALLABLE_SARVAM_MODELS",
    "FallbackGeneration",
    "GEMINI_MODEL",
    "GeminiGeneration",
    "GeminiGenerationConfig",
    "CALLABLE_GROQ_MODELS",
    "GROQ_MODEL",
    "GroqGeneration",
    "GroqGenerationConfig",
    "QWEN_GGUF_FILENAME",
    "QWEN_GGUF_MODEL",
    "QWEN_GGUF_REPOSITORY",
    "QwenGGUFGeneration",
    "QwenGGUFGenerationConfig",
    "resolve_gguf_model",
    "PROMPT_VARIANTS",
    "GenerationContext",
    "GenerationResult",
    "SarvamGeneration",
    "SarvamGenerationConfig",
    "build_messages",
]
