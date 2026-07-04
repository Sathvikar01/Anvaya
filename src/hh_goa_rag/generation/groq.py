"""Groq chat-completion adapter for grounded answer generation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from .prompts import PromptVariant, build_messages
from .sarvam import (
    RETRYABLE_HTTP_STATUSES,
    GenerationContext,
    GenerationResult,
    _elapsed_ms,
    _error_code,
    _field,
    _HTTPFailure,
    _parse_answer,
    _safe_error,
    _sse_events,
    _usage_value,
)

GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"
GROQ_MODEL = "openai/gpt-oss-20b"
CALLABLE_GROQ_MODELS = (
    GROQ_MODEL,
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
)


def _http_client_factory(**kwargs: Any) -> httpx.Client:
    kwargs.setdefault("http2", True)
    kwargs.setdefault("trust_env", False)
    kwargs.setdefault(
        "limits",
        httpx.Limits(
            # A single pooled connection turns one abandoned/slow request into
            # head-of-line blocking for every later request on this client.
            # Keep a few slots so a timed-out attempt cannot starve the tier.
            max_connections=4,
            max_keepalive_connections=2,
            keepalive_expiry=30.0,
        ),
    )
    return httpx.Client(**kwargs)


@dataclass(frozen=True)
class GroqGenerationConfig:
    model: str = GROQ_MODEL
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_s: float = 30.0
    max_attempts: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 2.0
    stream: bool = True

    def __post_init__(self) -> None:
        if self.model not in CALLABLE_GROQ_MODELS:
            raise ValueError(f"Unsupported or unverified Groq model: {self.model}")
        if self.temperature != 0:
            raise ValueError("Generation temperature is fixed at 0")
        if self.max_tokens <= 0 or self.timeout_s <= 0 or self.max_attempts <= 0:
            raise ValueError("max_tokens, timeout_s, and max_attempts must be positive")
        if not 0 <= self.backoff_base_s <= self.backoff_max_s:
            raise ValueError("Invalid bounded-backoff configuration")


class GroqGeneration:
    """Generate schema-shaped JSON answers using Groq's OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        *,
        config: GroqGenerationConfig | None = None,
        client_factory: Callable[..., httpx.Client] = _http_client_factory,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GROQ_API_KEY is missing; set it in the ignored .env file")
        self._api_key = api_key.strip()
        self.config = config or GroqGenerationConfig()
        self._client_factory = client_factory
        self._sleep = sleep
        self._client = self._client_factory(timeout=self.config.timeout_s)
        self._closed = False

    @classmethod
    def from_env(
        cls,
        env_path: str | Path = ".env",
        *,
        config: GroqGenerationConfig | None = None,
    ) -> GroqGeneration:
        load_dotenv(dotenv_path=Path(env_path), override=False)
        return cls(os.getenv("GROQ_API_KEY", ""), config=config)

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def warm_up(self) -> None:
        """Open the persistent route and verify the configured model before serving traffic."""
        response = self._client.get(
            GROQ_MODELS_ENDPOINT,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        _raise_for_status(response)
        models = response.json().get("data") or []
        if not any(str(model.get("id")) == self.config.model for model in models):
            raise RuntimeError(f"Configured Groq model is unavailable: {self.config.model}")

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
                    return self._result(
                        started,
                        attempt,
                        status="error",
                        code="invalid_structured_output",
                        message=str(diagnostics["parse_error"]),
                        raw_output=observation["content"],
                        ttft_ms=observation["ttft_ms"],
                        usage=observation["usage"],
                        finish_reason=observation["finish_reason"],
                        diagnostics=diagnostics,
                    )
                return self._result(
                    started,
                    attempt,
                    status="ok",
                    answer_status=parsed["status"],
                    answer=parsed["answer"],
                    evidence_ids=tuple(parsed["evidence_ids"]),
                    raw_output=observation["content"],
                    ttft_ms=observation["ttft_ms"],
                    usage=observation["usage"],
                    finish_reason=observation["finish_reason"],
                    diagnostics=diagnostics,
                )
            except Exception as error:
                last_error = error
                status = error.status if hasattr(error, "status") else None
                retryable = status in RETRYABLE_HTTP_STATUSES or isinstance(
                    error, (httpx.TimeoutException, httpx.TransportError)
                )
                if not retryable or attempt == self.config.max_attempts:
                    return self._result(
                        started,
                        attempt,
                        status="error",
                        code=_error_code(error),
                        message=_safe_error(error),
                        http_status=status,
                    )
                self._sleep(self._backoff(attempt))
        raise AssertionError(f"Unreachable retry state: {last_error}")

    def _request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        try:
            return self._send_request(messages)
        except _HTTPFailure as failure:
            # Provider APIs evolve: an unknown/renamed optional parameter
            # surfaces as HTTP 400. Retry once with a minimal payload before
            # giving up so the tier keeps working across API changes.
            if failure.status != 400:
                raise
            print(
                f"Groq rejected the request (HTTP 400); retrying without "
                f"optional parameters: {_safe_error(failure)}",
                flush=True,
            )
            return self._send_request(messages, include_optional_params=False)

    def _send_request(
        self,
        messages: list[dict[str, str]],
        *,
        include_optional_params: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
            "stream": self.config.stream,
            "response_format": {"type": "json_object"},
        }
        if self.config.stream:
            payload["stream_options"] = {"include_usage": True}
        if include_optional_params:
            if self.config.model.startswith("openai/gpt-oss-"):
                payload["include_reasoning"] = False
                payload["reasoning_effort"] = "low"
            elif self.config.model.startswith("qwen/"):
                payload["reasoning_effort"] = "none"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Groq-Beta": "inference-metrics",
        }
        if self.config.stream:
            return self._stream_request(self._client, payload, headers)
        response = self._client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        _raise_for_status(response)
        body = response.json()
        choice = body["choices"][0]
        return {
            "content": str(choice["message"].get("content") or ""),
            "ttft_ms": None,
            "usage": body.get("usage") or {},
            "finish_reason": choice.get("finish_reason"),
        }

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
        with client.stream("POST", GROQ_CHAT_ENDPOINT, headers=headers, json=payload) as response:
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
        return min(self.config.backoff_base_s * (2 ** (attempt - 1)), self.config.backoff_max_s)

    def _result(
        self,
        started: int,
        attempts: int,
        *,
        status: str,
        code: str | None = None,
        message: str | None = None,
        raw_output: str = "",
        answer_status: str | None = None,
        answer: str = "",
        evidence_ids: tuple[str, ...] = (),
        ttft_ms: float | None = None,
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        http_status: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> GenerationResult:
        usage = usage or {}
        output_tokens = _usage_value(usage, "completion_tokens")
        return GenerationResult(
            provider="groq",
            model=self.config.model,
            status=status,  # type: ignore[arg-type]
            answer_status=answer_status,  # type: ignore[arg-type]
            answer=answer,
            evidence_ids=evidence_ids,
            raw_output=raw_output,
            latency_ms=_elapsed_ms(started),
            time_to_first_token_ms=ttft_ms,
            prompt_tokens=_usage_value(usage, "prompt_tokens"),
            output_tokens=output_tokens,
            total_tokens=_usage_value(usage, "total_tokens"),
            finish_reason=finish_reason,
            attempts=attempts,
            error_code=code,
            error_message=message,
            http_status=http_status,
            diagnostics=diagnostics or {},
            tokens_per_second=_tokens_per_second(usage, output_tokens),
            provider_latency_ms=_usage_ms(usage, "total_time"),
        )


def _tokens_per_second(usage: dict[str, Any], output_tokens: int | None) -> float | None:
    completion_time = usage.get("completion_time")
    if output_tokens is None or completion_time is None:
        return None
    try:
        seconds = float(completion_time)
    except (TypeError, ValueError):
        return None
    return output_tokens / seconds if seconds > 0 else None


def _usage_ms(usage: dict[str, Any], key: str) -> float | None:
    value = usage.get(key)
    if value is None:
        return None
    try:
        return float(value) * 1000
    except (TypeError, ValueError):
        return None


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        body = response.read().decode(errors="replace")
    except Exception:
        body = ""
    raise _HTTPFailure(response.status_code, body[:500] or f"HTTP {response.status_code}")
