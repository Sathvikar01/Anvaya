"""Sarvam chat-completion adapter with structured grounding and streaming timings."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv

from .prompts import PromptVariant, build_messages

SARVAM_CHAT_ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"
CALLABLE_SARVAM_MODELS = ("sarvam-105b", "sarvam-105b-conversations")
RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}

_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_rag_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["ANSWER", "INSUFFICIENT_CONTEXT"],
                },
                "answer": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "answer", "evidence_ids"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class GenerationContext:
    parent_id: str
    chunk_id: str
    text: str
    rank: int
    score: float | None = None
    language: str | None = None


@dataclass(frozen=True)
class SarvamGenerationConfig:
    model: Literal["sarvam-105b", "sarvam-105b-conversations"] = "sarvam-105b"
    temperature: float = 0.0
    max_tokens: int = 192
    reasoning_effort: None = None
    timeout_s: float = 30.0
    max_attempts: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 2.0
    stream: bool = True

    def __post_init__(self) -> None:
        if self.model not in CALLABLE_SARVAM_MODELS:
            raise ValueError(f"Unsupported or unverified Sarvam model: {self.model}")
        if self.temperature != 0:
            raise ValueError("Generation ablation temperature is fixed at 0")
        if self.reasoning_effort is not None:
            raise ValueError("Reasoning is disabled to isolate low-latency answer generation")
        if self.max_tokens <= 0 or self.timeout_s <= 0 or self.max_attempts <= 0:
            raise ValueError("max_tokens, timeout_s, and max_attempts must be positive")
        if not 0 <= self.backoff_base_s <= self.backoff_max_s:
            raise ValueError("Invalid bounded-backoff configuration")


@dataclass(frozen=True)
class GenerationResult:
    provider: Literal["sarvam", "groq", "qwen-gguf"]
    model: str
    status: Literal["ok", "error"]
    answer_status: Literal["ANSWER", "INSUFFICIENT_CONTEXT"] | None
    answer: str
    evidence_ids: tuple[str, ...]
    raw_output: str
    latency_ms: float
    time_to_first_token_ms: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    attempts: int
    error_code: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    tokens_per_second: float | None = None
    provider_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _HTTPFailure(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class SarvamGeneration:
    """Generate schema-constrained answers while retaining retrieved provenance."""

    def __init__(
        self,
        api_key: str,
        *,
        config: SarvamGenerationConfig | None = None,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SARVAM_API_KEY is missing; set it in the ignored .env file")
        self._api_key = api_key.strip()
        self.config = config or SarvamGenerationConfig()
        self._client_factory = client_factory
        self._sleep = sleep

    @classmethod
    def from_env(
        cls,
        env_path: str | Path = ".env",
        *,
        config: SarvamGenerationConfig | None = None,
    ) -> SarvamGeneration:
        load_dotenv(dotenv_path=Path(env_path), override=False)
        return cls(os.getenv("SARVAM_API_KEY", ""), config=config)

    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext | dict[str, Any]],
        *,
        prompt_variant: PromptVariant = "structured_evidence_ids",
        language_code: str | None = None,
    ) -> GenerationResult:
        messages = build_messages(
            question,
            contexts,
            variant=prompt_variant,
            language_code=language_code,
        )
        allowed_ids = {str(_field(context, "parent_id")) for context in contexts}
        started = time.perf_counter_ns()
        last_error: BaseException | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                observation = self._request(messages)
                parsed, diagnostics = _parse_answer(observation["content"], allowed_ids)
                if parsed is None:
                    # A truncated JSON payload (finish_reason="length") is a
                    # transient decode artifact, not a policy failure: retry
                    # within the attempt budget before failing the tier.
                    parse_failure = (observation, diagnostics)
                    last_error = RuntimeError(str(diagnostics.get("parse_error", "invalid JSON")))
                    if attempt < self.config.max_attempts:
                        self._sleep(self._backoff(attempt))
                        continue
                    return self._error_result(
                        started,
                        attempt,
                        "invalid_structured_output",
                        str(diagnostics["parse_error"]),
                        raw_output=observation["content"],
                        ttft_ms=observation["ttft_ms"],
                        usage=observation["usage"],
                        finish_reason=observation["finish_reason"],
                        diagnostics=diagnostics,
                    )
                return GenerationResult(
                    provider="sarvam",
                    model=self.config.model,
                    status="ok",
                    answer_status=parsed["status"],
                    answer=parsed["answer"],
                    evidence_ids=tuple(parsed["evidence_ids"]),
                    raw_output=observation["content"],
                    latency_ms=_elapsed_ms(started),
                    time_to_first_token_ms=observation["ttft_ms"],
                    prompt_tokens=_usage_value(observation["usage"], "prompt_tokens"),
                    output_tokens=_usage_value(observation["usage"], "completion_tokens"),
                    total_tokens=_usage_value(observation["usage"], "total_tokens"),
                    finish_reason=observation["finish_reason"],
                    attempts=attempt,
                    diagnostics=diagnostics,
                )
            except Exception as error:
                last_error = error
                status = error.status if isinstance(error, _HTTPFailure) else None
                retryable = status in RETRYABLE_HTTP_STATUSES or isinstance(
                    error, (httpx.TimeoutException, httpx.TransportError)
                )
                if not retryable or attempt == self.config.max_attempts:
                    return self._error_result(
                        started,
                        attempt,
                        _error_code(error),
                        _safe_error(error),
                        http_status=status,
                    )
                self._sleep(self._backoff(attempt))
        raise AssertionError(f"Unreachable retry state: {last_error}")

    def _request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "reasoning_effort": self.config.reasoning_effort,
            "max_tokens": self.config.max_tokens,
            "stream": self.config.stream,
            "response_format": _ANSWER_SCHEMA,
        }
        headers = {"api-subscription-key": self._api_key, "Content-Type": "application/json"}
        client = self._client_factory(timeout=self.config.timeout_s)
        try:
            if self.config.stream:
                return self._stream_request(client, payload, headers)
            response = client.post(SARVAM_CHAT_ENDPOINT, headers=headers, json=payload)
            _raise_for_status(response)
            body = response.json()
            choice = body["choices"][0]
            return {
                "content": str(choice["message"].get("content") or ""),
                "ttft_ms": None,
                "usage": body.get("usage") or {},
                "finish_reason": choice.get("finish_reason"),
            }
        finally:
            client.close()

    def _stream_request(
        self,
        client: httpx.Client,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        request_started = time.perf_counter_ns()
        pieces: list[str] = []
        first_token_ms: float | None = None
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        with client.stream("POST", SARVAM_CHAT_ENDPOINT, headers=headers, json=payload) as response:
            _raise_for_status(response)
            for event in _sse_events(response.iter_lines()):
                if event == "[DONE]":
                    break
                body = json.loads(event)
                if body.get("usage"):
                    usage = body["usage"]
                choices = body.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                content = str((choice.get("delta") or {}).get("content") or "")
                if content:
                    if first_token_ms is None:
                        first_token_ms = _elapsed_ms(request_started)
                    pieces.append(content)
        return {
            "content": "".join(pieces),
            "ttft_ms": first_token_ms,
            "usage": usage,
            "finish_reason": finish_reason,
        }

    def _backoff(self, attempt: int) -> float:
        raw = min(self.config.backoff_base_s * (2 ** (attempt - 1)), self.config.backoff_max_s)
        return min(raw + random.uniform(0, raw * 0.1), self.config.backoff_max_s)

    def _error_result(
        self,
        started: int,
        attempts: int,
        code: str,
        message: str,
        *,
        raw_output: str = "",
        ttft_ms: float | None = None,
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        http_status: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> GenerationResult:
        usage = usage or {}
        return GenerationResult(
            provider="sarvam",
            model=self.config.model,
            status="error",
            answer_status=None,
            answer="",
            evidence_ids=(),
            raw_output=raw_output,
            latency_ms=_elapsed_ms(started),
            time_to_first_token_ms=ttft_ms,
            prompt_tokens=_usage_value(usage, "prompt_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens"),
            total_tokens=_usage_value(usage, "total_tokens"),
            finish_reason=finish_reason,
            attempts=attempts,
            error_code=code,
            error_message=message,
            http_status=http_status,
            diagnostics=diagnostics or {},
        )


def _parse_answer(
    raw_output: str,
    allowed_ids: set[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "schema_valid": False,
        "missing_citation": False,
        "unknown_evidence_ids": [],
        "answer_without_context": False,
        "novel_numbers": [],
    }
    try:
        value = json.loads(raw_output.strip())
        if not isinstance(value, dict) or set(value) != {"status", "answer", "evidence_ids"}:
            raise ValueError("Expected exactly status, answer, and evidence_ids")
        status = value["status"]
        answer = value["answer"]
        evidence_ids = value["evidence_ids"]
        if status not in {"ANSWER", "INSUFFICIENT_CONTEXT"}:
            raise ValueError("Invalid answer status")
        if not isinstance(answer, str) or not isinstance(evidence_ids, list):
            raise ValueError("answer must be a string and evidence_ids must be an array")
        if any(not isinstance(item, str) for item in evidence_ids):
            raise ValueError("Every evidence ID must be a string")
        evidence_ids = list(dict.fromkeys(evidence_ids))
        unknown = sorted(set(evidence_ids) - allowed_ids)
        diagnostics["unknown_evidence_ids"] = unknown
        diagnostics["missing_citation"] = status == "ANSWER" and not evidence_ids
        diagnostics["answer_without_context"] = status == "ANSWER" and not allowed_ids
        if status == "INSUFFICIENT_CONTEXT" and (answer.strip() or evidence_ids):
            raise ValueError("INSUFFICIENT_CONTEXT requires an empty answer and no evidence IDs")
        if status == "ANSWER" and (not answer.strip() or not evidence_ids or unknown):
            raise ValueError("ANSWER requires text and valid retrieved evidence IDs")
        diagnostics["schema_valid"] = True
        parsed = {"status": status, "answer": answer.strip(), "evidence_ids": evidence_ids}
        return parsed, diagnostics
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        diagnostics["parse_error"] = str(error)
        return None, diagnostics


def _sse_events(lines: Iterator[str]) -> Iterator[str]:
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            yield line[5:].strip()


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        body = response.read().decode(errors="replace")
    except Exception:
        body = ""
    raise _HTTPFailure(response.status_code, body[:500] or f"HTTP {response.status_code}")


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name, "")
    return getattr(value, name, "")


def _usage_value(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return int(value) if value is not None else None


def _elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1e6


def _error_code(error: BaseException) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.TransportError):
        return "transport_error"
    if isinstance(error, _HTTPFailure):
        return "rate_limited" if error.status == 429 else "api_error"
    return "unexpected_error"


def _safe_error(error: BaseException) -> str:
    text = str(error).replace("\r", " ").replace("\n", " ").strip()
    return text[:500] or type(error).__name__
