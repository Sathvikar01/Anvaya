"""Gemini structured-output adapter for grounded answer generation."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from .prompts import PromptVariant, build_messages
from .sarvam import GenerationContext, _parse_answer

GEMINI_GENERATE_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_MODEL = "gemini-3.7-flash"
SUPPORTED_GEMINI_MODELS = ("gemini-3.7-flash", "gemini-3.5-flash-lite")
_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ANSWER", "INSUFFICIENT_CONTEXT"]},
        "answer": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "answer", "evidence_ids"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class GeminiGenerationConfig:
    model: str = GEMINI_MODEL
    max_output_tokens: int = 512
    timeout_s: float = 45.0

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_GEMINI_MODELS:
            raise ValueError(f"Unsupported or unverified Gemini model: {self.model}")
        if self.max_output_tokens <= 0 or self.timeout_s <= 0:
            raise ValueError("max_output_tokens and timeout_s must be positive")


class GeminiGeneration:
    """Call Gemini with JSON schema output and return the harness-compatible shape."""

    def __init__(
        self,
        api_key: str,
        *,
        config: GeminiGenerationConfig | None = None,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is missing")
        self._api_key = api_key.strip()
        self.config = config or GeminiGenerationConfig()
        self._client = client_factory(timeout=self.config.timeout_s)

    def close(self) -> None:
        self._client.close()

    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext | dict[str, Any]],
        *,
        prompt_variant: PromptVariant = "structured_evidence_ids",
        language_code: str | None = None,
    ) -> dict[str, Any]:
        messages = build_messages(
            question,
            contexts,
            variant=prompt_variant,
            language_code=language_code,
        )
        allowed_ids = {str(_field(context, "parent_id", "")) for context in contexts}
        body = {
            "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": [
                {"role": "user", "parts": [{"text": messages[1]["content"]}]}
            ],
            "generationConfig": {
                "maxOutputTokens": self.config.max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": _ANSWER_SCHEMA,
            },
        }
        endpoint = GEMINI_GENERATE_ENDPOINT.format(model=self.config.model)
        started = time.perf_counter_ns()
        try:
            response = self._client.post(
                endpoint,
                headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
                json=body,
            )
            status = response.status_code
            response.raise_for_status()
            payload = response.json()
            raw_output = _response_text(payload)
            parsed, diagnostics = _parse_answer(raw_output, allowed_ids)
            if parsed is None:
                return _result(
                    status="error",
                    latency_ms=_elapsed_ms(started),
                    raw_output=raw_output,
                    error_code="invalid_structured_output",
                    error_message=str(diagnostics.get("parse_error", "invalid output")),
                    diagnostics=diagnostics,
                    http_status=status,
                    usage=payload.get("usageMetadata", {}),
                    model=self.config.model,
                )
            return _result(
                status="ok",
                latency_ms=_elapsed_ms(started),
                raw_output=raw_output,
                answer_status=parsed["status"],
                answer=parsed["answer"],
                evidence_ids=tuple(parsed["evidence_ids"]),
                diagnostics=diagnostics,
                http_status=status,
                usage=payload.get("usageMetadata", {}),
                model=self.config.model,
            )
        except Exception as error:
            return _result(
                status="error",
                latency_ms=_elapsed_ms(started),
                error_code="http_error" if isinstance(error, httpx.HTTPError) else "provider_error",
                error_message=_safe_error(error),
                http_status=getattr(error, "response", None).status_code
                if getattr(error, "response", None) is not None
                else None,
                model=self.config.model,
            )


def _response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts)
    if not text.strip():
        raise ValueError("Gemini returned an empty candidate")
    return text.strip()


def _result(
    *,
    status: str,
    latency_ms: float,
    raw_output: str = "",
    answer_status: str | None = None,
    answer: str = "",
    evidence_ids: tuple[str, ...] = (),
    error_code: str | None = None,
    error_message: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    http_status: int | None = None,
    usage: dict[str, Any] | None = None,
    model: str = GEMINI_MODEL,
) -> dict[str, Any]:
    usage = usage or {}
    return {
        "provider": "gemini",
        "model": model,
        "status": status,
        "answer_status": answer_status,
        "answer": answer,
        "evidence_ids": evidence_ids,
        "raw_output": raw_output,
        "latency_ms": latency_ms,
        "time_to_first_token_ms": None,
        "prompt_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
        "finish_reason": None,
        "attempts": 1,
        "error_code": error_code,
        "error_message": error_message,
        "http_status": http_status,
        "diagnostics": diagnostics or {},
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1e6


def _safe_error(error: BaseException) -> str:
    message = str(error).replace("\n", " ").strip()
    return message[:500] or error.__class__.__name__
