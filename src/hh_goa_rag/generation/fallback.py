"""Resilient multi-tier answer generation for the Voice-RAG harness.

The demo serves every grounded answer through an ordered chain of generator
tiers. Each tier receives the identical structured request and must return
the exact ``{status, answer, evidence_ids}`` schema, so downstream grounding
validation is unchanged no matter which tier served the answer:

1. API tier(s): Groq ``openai/gpt-oss-20b`` and/or Sarvam when their keys are
   configured - low-latency generative coverage.
2. Local tier: resident Qwen3.5 0.8B Q4 GGUF via llama.cpp - keeps the demo
   answering even with no network egress, at reduced speed.

Every tier attempt is individually time-bounded so one hung provider cannot
exhaust the whole request budget. The serving tier and any failures are
recorded in the response ``diagnostics``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from .prompts import PromptVariant
from .sarvam import GenerationContext, GenerationResult

DEFAULT_TIER_TIMEOUT_S = 12.0
# Hard ceiling for the whole chain. Whatever happens inside a provider, the
# caller gets a structured answer - grounded answer, honest refusal, or
# fail-closed error - within this budget.
DEFAULT_CHAIN_BUDGET_S = 18.0


class FallbackGeneration:
    """Serve answers through an ordered, time-bounded chain of generators."""

    def __init__(self, *, tiers: list[Any]) -> None:
        if not tiers:
            raise ValueError("FallbackGeneration requires at least one tier")
        self.tiers = tiers
        self.model_name = getattr(tiers[0], "model_name", "")
        self.tier_timeout_s = _tier_timeout()
        self.chain_budget_s = _chain_budget()

    @classmethod
    def from_env(cls, *, config: Any = None) -> "FallbackGeneration | None":
        """Build the generation chain from configured secrets and settings."""
        tiers = _build_chain(config)
        if not tiers:
            return None
        return cls(tiers=tiers)

    def warm_up(self) -> None:
        for tier in self.tiers:
            warm = getattr(tier, "warm_up", None)
            if callable(warm):
                try:
                    warm()
                except Exception as error:  # noqa: BLE001 - warm-up never blocks serving
                    print(
                        f"{type(tier).__name__} warm-up failed: {error!r}", flush=True
                    )

    def close(self) -> None:
        for tier in self.tiers:
            close = getattr(tier, "close", None)
            if callable(close):
                close()

    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext],
        *,
        prompt_variant: PromptVariant = "structured_evidence_ids",
        language_code: str | None = None,
    ) -> GenerationResult:
        attempts: list[dict[str, Any]] = []
        last_result: GenerationResult | None = None
        chain_started = time.perf_counter_ns()
        for index, tier in enumerate(self.tiers):
            remaining_s = self.chain_budget_s - (time.perf_counter_ns() - chain_started) / 1e9
            if remaining_s < 3.0:
                attempts.append(
                    {
                        "tier": type(tier).__name__,
                        "provider": None,
                        "status": None,
                        "error": f"skipped: chain budget exhausted ({remaining_s:.1f}s left)",
                        "latency_ms": 0.0,
                    }
                )
                continue
            tier_started = time.perf_counter_ns()
            result, error = _call_tier(
                tier,
                question,
                contexts,
                prompt_variant=prompt_variant,
                language_code=language_code,
                timeout_s=min(self.tier_timeout_s, remaining_s),
            )
            elapsed_ms = (time.perf_counter_ns() - tier_started) / 1_000_000
            entry = {
                "tier": type(tier).__name__,
                "provider": str(getattr(result, "provider", "") or "") or None,
                "status": str(getattr(result, "status", "")) or None,
                "error": error,
                "latency_ms": round(elapsed_ms, 3),
            }
            attempts.append(entry)
            if result is not None and getattr(result, "status", "") == "ok":
                diagnostics = dict(getattr(result, "diagnostics", {}) or {})
                diagnostics["generation_chain"] = {
                    "attempts": attempts,
                    "served_by": type(tier).__name__,
                    "tier_index": index,
                }
                return _with_diagnostics(result, diagnostics)
            if result is not None:
                provider_message = str(getattr(result, "error_message", "") or "").strip()
                if provider_message and not entry["error"]:
                    entry["error"] = provider_message[:300]
                entry["error_code"] = getattr(result, "error_code", None)
            last_result = result if result is not None else last_result
            print(
                f"Generation tier {type(tier).__name__} failed "
                f"({error or 'provider error'}) after {elapsed_ms:.0f} ms; "
                f"trying next tier",
                flush=True,
            )

        # Every tier failed: surface a structured provider error.
        base = last_result
        if base is not None:
            diagnostics = dict(getattr(base, "diagnostics", {}) or {})
            diagnostics["generation_chain"] = {"attempts": attempts, "served_by": None}
            return _with_diagnostics(base, diagnostics)
        return GenerationResult(
            provider="chain",
            model="",
            status="error",
            answer_status=None,
            answer="",
            evidence_ids=(),
            raw_output="",
            latency_ms=0.0,
            time_to_first_token_ms=None,
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            finish_reason=None,
            attempts=len(attempts),
            error_code="all_generation_tiers_failed",
            error_message="; ".join(
                str(entry.get("error") or "unknown") for entry in attempts
            )[:500],
            diagnostics={"generation_chain": {"attempts": attempts, "served_by": None}},
        )


def _call_tier(
    tier: Any,
    question: str,
    contexts: Sequence[GenerationContext],
    *,
    prompt_variant: PromptVariant,
    language_code: str | None,
    timeout_s: float,
) -> tuple[GenerationResult | None, str | None]:
    """Run one tier under a wall-clock bound; never raises."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hh-gen-tier")
    try:
        future = executor.submit(
            tier.generate,
            question,
            contexts,
            prompt_variant=prompt_variant,
            language_code=language_code,
        )
        try:
            return future.result(timeout=timeout_s), None
        except FuturesTimeoutError:
            future.cancel()
            return (
                None,
                f"tier exceeded {timeout_s:.0f}s wall clock",
            )
        except Exception as error:  # noqa: BLE001 - degrade to next tier
            return None, f"{type(error).__name__}: {error}"[:300]
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _with_diagnostics(
    result: GenerationResult, diagnostics: dict[str, Any]
) -> GenerationResult:
    return GenerationResult(
        provider=result.provider,
        model=result.model,
        status=result.status,
        answer_status=result.answer_status,
        answer=result.answer,
        evidence_ids=result.evidence_ids,
        raw_output=result.raw_output,
        latency_ms=result.latency_ms,
        time_to_first_token_ms=result.time_to_first_token_ms,
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        finish_reason=result.finish_reason,
        attempts=result.attempts,
        error_code=result.error_code,
        error_message=result.error_message,
        http_status=result.http_status,
        diagnostics=diagnostics,
        tokens_per_second=result.tokens_per_second,
        provider_latency_ms=result.provider_latency_ms,
    )


def _build_chain(config: Any = None) -> list[Any]:
    """Resolve the ordered tier list; empty means caller falls back to pure local."""
    chain_spec = os.getenv("HH_RAG_GENERATION_CHAIN", "auto").strip().lower()
    if chain_spec in {"off", "none", "disabled"}:
        return []

    has_groq = bool(os.getenv("GROQ_API_KEY", "").strip())
    has_sarvam = bool(os.getenv("SARVAM_API_KEY", "").strip())
    max_tokens = int(getattr(config, "max_tokens", 128) or 128)

    names: list[str]
    if chain_spec == "auto":
        # Sarvam first: consistent latency and no observed rate limiting under
        # sustained demo load; Groq answers faster but throttles on the free
        # tier, so it is the second option. Local GGUF is the offline net.
        names = ["sarvam" if has_sarvam else "", "groq" if has_groq else "", "local"]
        names = [name for name in names if name]
    else:
        names = [part.strip() for part in chain_spec.split(",") if part.strip()]

    tiers: list[Any] = []
    for name in names:
        try:
            if name == "groq":
                if not has_groq:
                    raise RuntimeError("GROQ_API_KEY missing")
                from .groq import GroqGeneration, GroqGenerationConfig

                tiers.append(
                    GroqGeneration.from_env(
                        config=GroqGenerationConfig(
                            max_tokens=max_tokens,
                            timeout_s=10.0,
                            max_attempts=2,
                        )
                    )
                )
            elif name == "sarvam":
                if not has_sarvam:
                    raise RuntimeError("SARVAM_API_KEY missing")
                from .sarvam import SarvamGeneration, SarvamGenerationConfig

                tiers.append(
                    SarvamGeneration.from_env(
                        config=SarvamGenerationConfig(
                            # Headroom above the local tier's 128 so the
                            # schema-constrained JSON is never cut mid-string.
                            max_tokens=max(256, max_tokens),
                            timeout_s=12.0,
                            max_attempts=3,
                        )
                    )
                )
            elif name == "local":
                from .qwen_gguf import QwenGGUFGeneration

                tiers.append(QwenGGUFGeneration.from_env(config=config))
            else:
                raise RuntimeError(f"unknown chain tier: {name}")
        except Exception as error:  # noqa: BLE001 - skip unavailable tiers
            print(f"Generation tier '{name}' unavailable: {error!r}", flush=True)
    return tiers


def _tier_timeout() -> float:
    raw = os.getenv("HH_RAG_TIER_TIMEOUT_S", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_TIER_TIMEOUT_S
    except ValueError:
        value = DEFAULT_TIER_TIMEOUT_S
    return max(1.0, value)


def _chain_budget() -> float:
    raw = os.getenv("HH_RAG_GENERATION_BUDGET_S", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_CHAIN_BUDGET_S
    except ValueError:
        value = DEFAULT_CHAIN_BUDGET_S
    return max(3.0, value)
