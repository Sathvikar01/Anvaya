"""Structured orchestration for the fully frozen Voice-RAG stack."""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from hh_goa_rag.generation import (
    QWEN_GGUF_MODEL,
    GenerationContext,
    QwenGGUFGeneration,
    QwenGGUFGenerationConfig,
)
from hh_goa_rag.guardrails import (
    GuardrailResponse,
    ReasonCode,
    Route,
    evidence_sufficiency,
    language_key,
    route_input,
    validate_generation,
    validate_transcript,
)
from hh_goa_rag.guardrails.types import STAGE_NAMES
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel, safe_model_name
from hh_goa_rag.retriever import ParentFaissRetriever
from hh_goa_rag.stt.sarvam import SarvamSTT

LOGGER = logging.getLogger("hh_goa_rag.harness")

FROZEN_GENERATION_MODEL = QWEN_GGUF_MODEL
FROZEN_TOP_K = 10
FROZEN_PROMPT = "strict_context_only"
FROZEN_MAX_OUTPUT_TOKENS = 128
FROZEN_GENERATION_CONTEXTS = 3
FROZEN_FAST_TIER_MODEL = "deepset/xlm-roberta-base-squad2-distilled"
FROZEN_FAST_TIER_THRESHOLD = 0.98
FAST_TIER_CONTEXTS = 3
FROZEN_RECOVERY_PROMPT = "structured_evidence_ids"
DEFAULT_MIN_GENERATION_BUDGET_MS = 60.0
# Minimum remaining budget required before the resident extractive tier is
# attempted under a strict deadline; below this the request fails fast with an
# honest DEADLINE_BUDGET_EXHAUSTED refusal so P100 stays inside the cap. The
# value covers a worst-case warm extractive pass (~135 ms observed) plus
# serialization margin inside the 200 ms cap.
DEFAULT_FAST_TIER_MIN_BUDGET_MS = 140.0
DEFAULT_GENERATION_BUDGET_ESTIMATE_MS = 800.0
# Outer wall clock for the full generate->validate(+recover) sequence. The
# chain budgets nest inside this so a single request can never run two full
# generation passes back-to-back into the tens of seconds.
DEFAULT_GENERATION_TIMEOUT_S = 22.0
DEFAULT_RECOVERY_MAX_FIRST_ATTEMPT_MS = 6_000.0
RECOVERABLE_GENERATION_REASONS = frozenset(
    {
        ReasonCode.GENERATION_SCHEMA_INVALID,
        ReasonCode.GENERATION_UNGROUNDED_ANSWER,
        ReasonCode.GENERATION_MISSING_CITATION,
        ReasonCode.GENERATION_UNKNOWN_CITATION,
        ReasonCode.GENERATION_EMPTY_ANSWER,
    }
)
WARMUP_QUERY = "यूनाइटेड किंगडम में कौन से चार देश शामिल हैं"
WARMUP_PASSAGE = "यूनाइटेड किंगडम में इंग्लैंड, स्कॉटलैंड, वेल्स और उत्तरी आयरलैंड शामिल हैं"
FROZEN_RETRIEVAL = {
    "model": "BAAI/bge-m3",
    "chunking_strategy": "fixed_words",
    "chunk_size_words": 128,
    "index_engine": "faiss",
    "index_type": "hnsw",
    "m": 32,
    "ef_construction": 200,
    "ef_search": 128,
    "search_oversample": 20,
    "normalization_method": "float32_l2_v1",
    "top_k": FROZEN_TOP_K,
}


class Embedder(Protocol):
    def encode_queries(self, texts: list[str]) -> tuple[Any, Any]: ...


class Retriever(Protocol):
    def retrieve(self, query_embedding: Any) -> list[Any]: ...


class Generator(Protocol):
    def generate(
        self,
        question: str,
        contexts: list[Any],
        *,
        prompt_variant: str,
        language_code: str | None = None,
    ) -> Any: ...


class FastTier(Protocol):
    def answer(self, question: str, contexts: list[Any]) -> Any: ...


CACHEABLE_ROUTES = frozenset(
    {
        Route.ANSWER,
        Route.INSUFFICIENT_CONTEXT,
        Route.OFF_TOPIC,
        Route.UNSAFE,
    }
)


class ResponseCache:
    """Thread-safe LRU+TTL cache for deterministic harness responses.

    Generation is temperature-0 and every route is deterministic, so identical
    (transcript, language) inputs produce identical responses. Hits are always
    labeled ``cache_hit`` in metadata so demo timings stay honest. Formal
    benchmarks keep this disabled by default.
    """

    def __init__(self, *, max_entries: int = 256, ttl_s: float = 600.0) -> None:
        if max_entries <= 0 or ttl_s <= 0:
            raise ValueError("max_entries and ttl_s must be positive")
        self._entries: OrderedDict[tuple[str, str], tuple[float, GuardrailResponse]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._ttl_s = ttl_s

    @staticmethod
    def key(transcript: str, language_code: str | None) -> tuple[str, str]:
        normalized = " ".join(transcript.split()) if isinstance(transcript, str) else ""
        return normalized, (language_code or "").strip().lower()

    def get(self, key: tuple[str, str]) -> GuardrailResponse | None:
        if not key[0]:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, response = entry
            if now - stored_at > self._ttl_s:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return response

    def put(self, key: tuple[str, str], response: GuardrailResponse) -> None:
        if not key[0] or response.route not in CACHEABLE_ROUTES:
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), response)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


class VoiceRAGHarness:
    """Route text/audio through frozen services and deterministic guardrails."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
        stt: Any | None = None,
        orchestrator: str = "native",
        fast_tier: FastTier | None = None,
        allow_fallback: bool = True,
        fast_tier_thresholds: dict[str, float] | None = None,
        response_cache: ResponseCache | None = None,
        deadline_ms: float | None = None,
        generation_budget_estimate_ms: float = DEFAULT_GENERATION_BUDGET_ESTIMATE_MS,
        min_generation_budget_ms: float = DEFAULT_MIN_GENERATION_BUDGET_MS,
        generation_timeout_s: float = DEFAULT_GENERATION_TIMEOUT_S,
        recovery_max_first_attempt_ms: float = DEFAULT_RECOVERY_MAX_FIRST_ATTEMPT_MS,
        fast_tier_min_budget_ms: float = DEFAULT_FAST_TIER_MIN_BUDGET_MS,
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.stt = stt
        self.fast_tier = fast_tier
        self.allow_fallback = allow_fallback
        if not allow_fallback and fast_tier is None:
            raise ValueError("fast-tier-only mode requires an enabled fast tier")
        if orchestrator not in {"native", "langgraph"}:
            raise ValueError("orchestrator must be 'native' or 'langgraph'")
        self.orchestrator = orchestrator
        self.fast_tier_thresholds = _validate_fast_tier_thresholds(fast_tier_thresholds)
        self.response_cache = response_cache
        if deadline_ms is not None and deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive when provided")
        if generation_budget_estimate_ms <= 0 or min_generation_budget_ms <= 0:
            raise ValueError("generation budget values must be positive")
        if generation_timeout_s <= 0:
            raise ValueError("generation_timeout_s must be positive")
        if recovery_max_first_attempt_ms <= 0:
            raise ValueError("recovery_max_first_attempt_ms must be positive")
        if fast_tier_min_budget_ms <= 0:
            raise ValueError("fast_tier_min_budget_ms must be positive")
        self.deadline_ms = deadline_ms
        self.generation_budget_estimate_ms = generation_budget_estimate_ms
        self.min_generation_budget_ms = min_generation_budget_ms
        self.generation_timeout_s = generation_timeout_s
        self.recovery_max_first_attempt_ms = recovery_max_first_attempt_ms
        self.fast_tier_min_budget_ms = fast_tier_min_budget_ms
        self._graph = None
        if orchestrator == "langgraph":
            from hh_goa_rag.orchestration import build_langgraph

            self._graph = build_langgraph(self._handle_transcript)

    @classmethod
    def from_frozen_artifacts(
        cls,
        *,
        retriever_config_path: str | Path = "results/final_retriever_config.json",
        env_path: str | Path = ".env",
        device: str = "auto",
        include_stt: bool = True,
    ) -> VoiceRAGHarness:
        config = json.loads(Path(retriever_config_path).read_text(encoding="utf-8"))
        _assert_frozen_config(config)
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16" if device.startswith("cuda") else "float32"
        embedder = EmbeddingModel(
            MODEL_SPECS[FROZEN_RETRIEVAL["model"]],
            _portable_path(config["model_cache_path"]),
            device=device,
            max_sequence_length=512,
            dtype=dtype,
        )
        retriever = ParentFaissRetriever.load(
            _portable_path(config["index_artifact"]),
            _portable_path(config["chunk_artifact"]),
            top_k=FROZEN_TOP_K,
            oversample=int(config["search_oversample"]),
        )
        fallback_mode = os.getenv("HH_RAG_FALLBACK", "generative").strip().lower()
        if fallback_mode not in {"generative", "off", "fast_tier_only"}:
            raise ValueError(f"Unsupported HH_RAG_FALLBACK: {fallback_mode}")
        allow_fallback = fallback_mode == "generative"
        generator_name = os.getenv("HH_RAG_GENERATOR", "resilient").strip().lower()
        if not allow_fallback:
            # Extractive-only serving never invokes a generator: skip the
            # multi-hundred-MB load, its boot warm-up, and API clients.
            generator = None
        elif generator_name == "sarvam":
            from hh_goa_rag.generation import SarvamGeneration

            generator = SarvamGeneration.from_env(env_path)
        elif generator_name == "qwen_gguf":
            from hh_goa_rag.generation import resolve_gguf_model

            generator = QwenGGUFGeneration.from_env(
                config=QwenGGUFGenerationConfig(
                    model=resolve_gguf_model(),
                    max_tokens=FROZEN_MAX_OUTPUT_TOKENS,
                )
            )
        elif generator_name in {"resilient", "fallback", "auto"}:
            # Local GGUF first; identical structured request retried against
            # an API generator only when the local runtime is unavailable.
            from hh_goa_rag.generation import FallbackGeneration, resolve_gguf_model

            resolved_generator = FallbackGeneration.from_env(
                config=QwenGGUFGenerationConfig(
                    model=resolve_gguf_model(),
                    max_tokens=FROZEN_MAX_OUTPUT_TOKENS,
                )
            )
            if resolved_generator is None:
                generator = QwenGGUFGeneration.from_env(
                    config=QwenGGUFGenerationConfig(
                        model=resolve_gguf_model(),
                        max_tokens=FROZEN_MAX_OUTPUT_TOKENS,
                    )
                )
            else:
                generator = resolved_generator
        else:
            raise ValueError(f"Unsupported HH_RAG_GENERATOR: {generator_name}")
        stt = SarvamSTT.from_env(env_path) if include_stt else None
        orchestrator = os.getenv("HH_RAG_ORCHESTRATOR", "native").strip().lower()
        fast_tier_thresholds = _parse_fast_tier_threshold_env()
        response_cache = _parse_response_cache_env()
        deadline_ms, budget_estimate_ms, min_budget_ms = _parse_deadline_env()
        generation_timeout_s = _parse_generation_timeout_env()
        recovery_max_first_attempt_ms = _parse_recovery_first_attempt_env()
        fast_tier_min_budget_ms = _parse_fast_tier_min_budget_env()
        if generator is not None:
            warm_up = getattr(generator, "warm_up", None)
            if callable(warm_up):
                warm_up()
        embedder.warm_up(WARMUP_QUERY, WARMUP_PASSAGE, rounds=1)
        fast_tier = _build_fast_tier(device)
        return cls(
            embedder=embedder,
            retriever=retriever,
            generator=generator,
            stt=stt,
            orchestrator=orchestrator,
            fast_tier=fast_tier,
            allow_fallback=allow_fallback,
            fast_tier_thresholds=fast_tier_thresholds,
            response_cache=response_cache,
            deadline_ms=deadline_ms,
            generation_budget_estimate_ms=budget_estimate_ms,
            min_generation_budget_ms=min_budget_ms,
            generation_timeout_s=generation_timeout_s,
            recovery_max_first_attempt_ms=recovery_max_first_attempt_ms,
            fast_tier_min_budget_ms=fast_tier_min_budget_ms,
        )

    def handle_text(
        self,
        transcript: object,
        *,
        language_code: str | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> GuardrailResponse:
        return self._dispatch_transcript(
            transcript,
            operation_started=time.perf_counter_ns(),
            base_metadata=(
                {
                    "input_language": language_code,
                    "requested_output_language": language_code,
                }
                if language_code is not None
                else None
            ),
            on_stage=on_stage,
        )

    def handle_audio(
        self,
        audio_path: str | Path,
        *,
        language_code: str = "hi-IN",
        on_stage: Callable[[str], None] | None = None,
    ) -> GuardrailResponse:
        operation_started = time.perf_counter_ns()
        stages = _empty_stages()
        if self.stt is None:
            return _response(
                Route.SYSTEM_ERROR,
                ReasonCode.SYSTEM_COMPONENT_ERROR,
                stages,
                operation_started,
            )
        _notify_stage(on_stage, "Transcribing")
        started = time.perf_counter_ns()
        try:
            result = self.stt.transcribe_rest(audio_path, language_code=language_code)
        except Exception:
            stages["stt"] = _elapsed_ms(started)
            return _response(
                Route.SYSTEM_ERROR,
                ReasonCode.SYSTEM_COMPONENT_ERROR,
                stages,
                operation_started,
            )
        stages["stt"] = _elapsed_ms(started)
        stt_metadata = {
            "provider": _field(result, "provider", "sarvam"),
            "model": _field(result, "model", "saaras:v3"),
            "status": _field(result, "status", "error"),
            "error_code": _field(result, "error_code", None),
            "requested_language": language_code,
        }
        if _field(result, "status", "error") != "ok":
            stt_reason = (
                ReasonCode.STT_INVALID_AUDIO
                if _field(result, "error_code", None) == "invalid_audio"
                else ReasonCode.STT_PROVIDER_ERROR
            )
            return _response(
                Route.STT_FAILURE,
                stt_reason,
                stages,
                operation_started,
                metadata={"stt": stt_metadata},
            )
        # The strict-latency deadline governs the RAG pipeline itself
        # (retrieval -> grounding -> final output) as specified by the task;
        # network speech-to-text is a separate upstream stage and is reported
        # through `stage_latencies_ms.stt`. The dispatch clock therefore starts
        # once the transcript is available.
        rag_started = time.perf_counter_ns()
        return self._dispatch_transcript(
            _field(result, "transcript", ""),
            operation_started=rag_started,
            stages=stages,
            base_metadata={
                "stt": stt_metadata,
                "input_language": language_code,
                "requested_output_language": language_code,
            },
            on_stage=on_stage,
        )

    def _dispatch_transcript(
        self,
        transcript: object,
        *,
        operation_started: int,
        stages: dict[str, float] | None = None,
        base_metadata: dict[str, Any] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> GuardrailResponse:
        cache_key = None
        if self.response_cache is not None and isinstance(transcript, str):
            cache_key = ResponseCache.key(
                transcript, base_metadata.get("input_language") if base_metadata else None
            )
            lookup_started = time.perf_counter_ns()
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                return _cache_hit_response(
                    cached,
                    base_metadata or {},
                    lookup_ms=_elapsed_ms(lookup_started),
                )
        if self._graph is None:
            response = self._handle_transcript(
                transcript,
                operation_started=operation_started,
                stages=stages,
                base_metadata=base_metadata,
                on_stage=on_stage,
            )
        else:
            result = self._graph.invoke(
                {
                    "transcript": transcript,
                    "operation_started": operation_started,
                    "stages": stages,
                    "base_metadata": base_metadata,
                    "on_stage": on_stage,
                }
            )
            response = result["response"]
        response.metadata.setdefault("orchestrator", self.orchestrator)
        if cache_key is not None and self.response_cache is not None:
            self.response_cache.put(cache_key, response)
        return response

    def close(self) -> None:
        close = getattr(self.embedder, "close", None)
        if callable(close):
            close()
        close = getattr(self.generator, "close", None)
        if callable(close):
            close()
        close = getattr(self.fast_tier, "close", None)
        if callable(close):
            close()

    def _handle_transcript(
        self,
        transcript: object,
        *,
        operation_started: int,
        stages: dict[str, float] | None = None,
        base_metadata: dict[str, Any] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> GuardrailResponse:
        stages = stages or _empty_stages()
        metadata = dict(base_metadata or {})
        decision_trace: list[dict[str, Any]] = []
        metadata["decision_trace"] = decision_trace
        try:
            _notify_stage(on_stage, "Checking query")
            started = time.perf_counter_ns()
            validation = validate_transcript(transcript)
            stages["input_validation"] = _elapsed_ms(started)
            decision_trace.append(
                {
                    "stage": "input_validation",
                    "allow": validation.allow,
                    "route": validation.route.value if validation.route else None,
                    "reason_code": (
                        validation.reason_code.value if validation.reason_code else None
                    ),
                }
            )
            if not validation.allow:
                return _response(
                    validation.route or Route.STT_FAILURE,
                    validation.reason_code,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    metadata=metadata,
                )

            started = time.perf_counter_ns()
            route = route_input(validation.normalized_transcript)
            stages["route_check"] = _elapsed_ms(started)
            decision_trace.append(
                {
                    "stage": "route_check",
                    "allow": route.allow,
                    "route": route.route.value if route.route else None,
                    "reason_code": route.reason_code.value if route.reason_code else None,
                }
            )
            if not route.allow:
                return _response(
                    route.route or Route.OFF_TOPIC,
                    route.reason_code,
                    stages,
                    operation_started,
                    transcript=route.normalized_transcript,
                    metadata=metadata,
                )

            _notify_stage(on_stage, "Retrieving evidence")
            started = time.perf_counter_ns()
            vectors, _ = self.embedder.encode_queries([validation.normalized_transcript])
            stages["embedding"] = _elapsed_ms(started)
            started = time.perf_counter_ns()
            contexts = _retrieve_evidence(
                self.retriever,
                vectors[0],
                query_text=validation.normalized_transcript,
                language_code=metadata.get("input_language"),
            )[:FROZEN_TOP_K]
            stages["retrieval"] = _elapsed_ms(started)
            retrieved_ids = tuple(str(_field(item, "parent_id", "")) for item in contexts)
            metadata["retrieved"] = [
                {
                    "rank": index,
                    "parent_id": str(_field(item, "parent_id", "")),
                    "chunk_id": str(_field(item, "chunk_id", "")),
                    "score": float(_field(item, "score", 0.0)),
                    "text": str(_field(item, "text", "")),
                }
                for index, item in enumerate(contexts, start=1)
            ]

            started = time.perf_counter_ns()
            sufficiency = evidence_sufficiency(
                contexts,
                query=validation.normalized_transcript,
                language_code=metadata.get("input_language"),
            )
            stages["evidence_guardrail"] = _elapsed_ms(started)
            metadata["evidence_decision"] = sufficiency.to_dict()
            decision_trace.append(
                {
                    "stage": "evidence_guardrail",
                    "allow": sufficiency.sufficient,
                    "route": None if sufficiency.sufficient else Route.INSUFFICIENT_CONTEXT.value,
                    "reason_code": (
                        sufficiency.reason_code.value if sufficiency.reason_code else None
                    ),
                    "decision_rule": sufficiency.decision_rule,
                }
            )
            if not sufficiency.sufficient:
                return _response(
                    Route.INSUFFICIENT_CONTEXT,
                    sufficiency.reason_code,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    retrieved_ids=retrieved_ids,
                    metadata=metadata,
                )

            generation_contexts = [
                GenerationContext(
                    parent_id=str(_field(item, "parent_id", "")),
                    chunk_id=str(_field(item, "chunk_id", "")),
                    text=str(_field(item, "text", "")),
                    rank=index,
                    score=float(_field(item, "score", 0.0)),
                    language=_field(item, "language", None),
                )
                for index, item in enumerate(contexts, start=1)
            ]
            metadata["generation_context_ids"] = [
                context.parent_id for context in generation_contexts[:FROZEN_GENERATION_CONTEXTS]
            ]
            _notify_stage(on_stage, "Generating answer")
            answer_started = time.perf_counter_ns()

            def deadline_budget_remaining_ms() -> float:
                assert self.deadline_ms is not None
                return self.deadline_ms - _elapsed_ms(operation_started)

            if (
                self.deadline_ms is not None
                and deadline_budget_remaining_ms() < self.fast_tier_min_budget_ms
            ):
                # Strict-latency mode: not enough budget left even for the
                # resident extractive tier, so fail fast with an honest,
                # clearly-labeled refusal instead of overshooting the cap.
                remaining_ms = max(deadline_budget_remaining_ms(), 0.0)
                stages["generation"] = _elapsed_ms(answer_started)
                metadata["generation"] = {
                    "provider": "skipped-deadline",
                    "model": None,
                    "provider_status": "ok",
                    "answer_status": "INSUFFICIENT_CONTEXT",
                    "error_code": None,
                    "error_message": None,
                    "http_status": None,
                    "latency_ms": None,
                    "time_to_first_token_ms": None,
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "tokens_per_second": None,
                    "provider_latency_ms": None,
                    "diagnostics": {
                        "remaining_budget_ms": round(remaining_ms, 3),
                        "fast_tier_min_budget_ms": self.fast_tier_min_budget_ms,
                    },
                    "runtime": "deadline-guarded",
                    "answer_mode": "deadline_refusal",
                }
                metadata["grounding"] = {
                    "valid": True,
                    "route": Route.INSUFFICIENT_CONTEXT.value,
                    "reason_code": ReasonCode.DEADLINE_BUDGET_EXHAUSTED.value,
                }
                decision_trace.append(
                    {
                        "stage": "deadline_guard",
                        "allow": False,
                        "route": Route.INSUFFICIENT_CONTEXT.value,
                        "reason_code": ReasonCode.DEADLINE_BUDGET_EXHAUSTED.value,
                        "remaining_budget_ms": round(remaining_ms, 3),
                    }
                )
                return _response(
                    Route.INSUFFICIENT_CONTEXT,
                    ReasonCode.DEADLINE_BUDGET_EXHAUSTED,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    retrieved_ids=retrieved_ids,
                    metadata=metadata,
                )

            if self.fast_tier is not None:
                fast_outcome: dict[str, Any] | None = _run_fast_tier(
                    self.fast_tier,
                    validation.normalized_transcript,
                    generation_contexts[:FAST_TIER_CONTEXTS],
                    language_code=metadata.get("input_language"),
                    thresholds=self.fast_tier_thresholds,
                )
            else:
                fast_outcome = None
            if fast_outcome is not None:
                metadata["extractive_tier"] = fast_outcome
                decision_trace.append(
                    {
                        "stage": "extractive_tier",
                        "allow": bool(fast_outcome.get("accepted")),
                        "reason_code": fast_outcome.get("validation_error"),
                        "confidence": fast_outcome.get("confidence"),
                    }
                )
            if fast_outcome is not None and fast_outcome.get("accepted"):
                _notify_stage(on_stage, "Validating grounding")
                started = time.perf_counter_ns()
                stages["generation"] = _elapsed_ms(answer_started)
                stages["grounding_validation"] = _elapsed_ms(started)
                metadata["generation"] = {
                    "provider": "local-extractive",
                    "model": _field(self.fast_tier, "model_name", FROZEN_FAST_TIER_MODEL),
                    "provider_status": "ok",
                    "answer_status": "ANSWER",
                    "error_code": None,
                    "error_message": None,
                    "http_status": None,
                    "latency_ms": fast_outcome.get("latency_ms"),
                    "time_to_first_token_ms": None,
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "tokens_per_second": None,
                    "provider_latency_ms": None,
                    "diagnostics": {"confidence": fast_outcome.get("confidence")},
                }
                metadata["latency_checkpoints"] = {}
                metadata["generation"]["runtime"] = "resident-extractive"
                metadata["generation"]["answer_mode"] = "extractive_fast_tier"
                metadata["grounding"] = {
                    "valid": True,
                    "route": Route.ANSWER.value,
                    "reason_code": ReasonCode.ANSWER_GROUNDED.value,
                }
                decision_trace.append(
                    {
                        "stage": "grounding_validation",
                        "allow": True,
                        "route": Route.ANSWER.value,
                        "reason_code": ReasonCode.ANSWER_GROUNDED.value,
                    }
                )
                return _response(
                    Route.ANSWER,
                    ReasonCode.ANSWER_GROUNDED,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    answer=str(fast_outcome.get("answer", "")),
                    retrieved_ids=retrieved_ids,
                    citations=(str(fast_outcome["citation"]),),
                    metadata=metadata,
                )
            if not self.allow_fallback:
                stages["generation"] = _elapsed_ms(answer_started)
                metadata["generation"] = {
                    "provider": "disabled",
                    "model": None,
                    "provider_status": "ok",
                    "answer_status": "INSUFFICIENT_CONTEXT",
                    "error_code": None,
                    "error_message": None,
                    "http_status": None,
                    "latency_ms": None,
                    "time_to_first_token_ms": None,
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "tokens_per_second": None,
                    "provider_latency_ms": None,
                    "diagnostics": {},
                    "runtime": "fast-tier-only",
                    "answer_mode": "fast_tier_only",
                }
                metadata["grounding"] = {
                    "valid": True,
                    "route": Route.INSUFFICIENT_CONTEXT.value,
                    "reason_code": ReasonCode.MODEL_INSUFFICIENT_CONTEXT.value,
                }
                decision_trace.append(
                    {
                        "stage": "fallback_disabled",
                        "allow": False,
                        "route": Route.INSUFFICIENT_CONTEXT.value,
                        "reason_code": ReasonCode.MODEL_INSUFFICIENT_CONTEXT.value,
                        "fast_tier_status": str(fast_outcome.get("status", "")),
                    }
                )
                return _response(
                    Route.INSUFFICIENT_CONTEXT,
                    ReasonCode.MODEL_INSUFFICIENT_CONTEXT,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    retrieved_ids=retrieved_ids,
                    metadata=metadata,
                )
            if self.deadline_ms is not None:
                remaining_ms = self.deadline_ms - _elapsed_ms(operation_started)
                if remaining_ms < max(
                    self.min_generation_budget_ms, self.generation_budget_estimate_ms
                ):
                    stages["generation"] = _elapsed_ms(answer_started)
                    metadata["generation"] = {
                        "provider": "skipped-deadline",
                        "model": None,
                        "provider_status": "ok",
                        "answer_status": "INSUFFICIENT_CONTEXT",
                        "error_code": None,
                        "error_message": None,
                        "http_status": None,
                        "latency_ms": None,
                        "time_to_first_token_ms": None,
                        "prompt_tokens": None,
                        "output_tokens": None,
                        "tokens_per_second": None,
                        "provider_latency_ms": None,
                        "diagnostics": {
                            "remaining_budget_ms": round(remaining_ms, 3),
                            "generation_budget_estimate_ms": (
                                self.generation_budget_estimate_ms
                            ),
                        },
                        "runtime": "deadline-guarded",
                        "answer_mode": "deadline_refusal",
                    }
                    metadata["grounding"] = {
                        "valid": True,
                        "route": Route.INSUFFICIENT_CONTEXT.value,
                        "reason_code": ReasonCode.DEADLINE_BUDGET_EXHAUSTED.value,
                    }
                    decision_trace.append(
                        {
                            "stage": "deadline_guard",
                            "allow": False,
                            "route": Route.INSUFFICIENT_CONTEXT.value,
                            "reason_code": ReasonCode.DEADLINE_BUDGET_EXHAUSTED.value,
                            "remaining_budget_ms": round(remaining_ms, 3),
                        }
                    )
                    return _response(
                        Route.INSUFFICIENT_CONTEXT,
                        ReasonCode.DEADLINE_BUDGET_EXHAUSTED,
                        stages,
                        operation_started,
                        transcript=validation.normalized_transcript,
                        retrieved_ids=retrieved_ids,
                        metadata=metadata,
                    )
            generated = _generate_with_timeout(
                self.generator,
                validation.normalized_transcript,
                generation_contexts,
                prompt_variant=FROZEN_PROMPT,
                language_code=metadata.get("requested_output_language"),
                timeout_s=self.generation_timeout_s,
            )
            stages["generation"] = _elapsed_ms(answer_started)
            _notify_stage(on_stage, "Validating grounding")
            started = time.perf_counter_ns()
            grounding = validate_generation(generated, generation_contexts)
            stages["grounding_validation"] = _elapsed_ms(started)
            if (
                self.deadline_ms is not None
                and not grounding.valid
                and _elapsed_ms(operation_started) >= self.deadline_ms
            ):
                metadata["generation_recovery"] = {
                    "attempted": False,
                    "success": False,
                    "skipped_reason": ReasonCode.DEADLINE_BUDGET_EXHAUSTED.value,
                }
                metadata["grounding"] = {
                    "valid": True,
                    "route": Route.INSUFFICIENT_CONTEXT.value,
                    "reason_code": ReasonCode.DEADLINE_BUDGET_EXHAUSTED.value,
                }
                decision_trace.append(
                    {
                        "stage": "deadline_guard",
                        "allow": False,
                        "route": Route.INSUFFICIENT_CONTEXT.value,
                        "reason_code": ReasonCode.DEADLINE_BUDGET_EXHAUSTED.value,
                    }
                )
                return _response(
                    Route.INSUFFICIENT_CONTEXT,
                    ReasonCode.DEADLINE_BUDGET_EXHAUSTED,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    retrieved_ids=retrieved_ids,
                    metadata=metadata,
                )
            recovery: dict[str, Any] = {"attempted": False, "success": False}
            first_attempt_ms = float(stages.get("generation", 0.0))
            if (
                not grounding.valid
                and grounding.reason_code in RECOVERABLE_GENERATION_REASONS
                and first_attempt_ms <= self.recovery_max_first_attempt_ms
            ):
                recovery_contexts = generation_contexts[:FAST_TIER_CONTEXTS]
                recovered = _generate_with_timeout(
                    self.generator,
                    validation.normalized_transcript,
                    recovery_contexts,
                    prompt_variant=FROZEN_RECOVERY_PROMPT,
                    language_code=metadata.get("requested_output_language"),
                    timeout_s=self.generation_timeout_s,
                )
                recovered_grounding = validate_generation(recovered, recovery_contexts)
                recovery = {
                    "attempted": True,
                    "prompt_variant": FROZEN_RECOVERY_PROMPT,
                    "contexts": len(recovery_contexts),
                    "success": recovered_grounding.valid,
                    "reason_code": recovered_grounding.reason_code.value,
                }
                stages["generation"] = _elapsed_ms(answer_started)
                if recovered_grounding.valid:
                    generated = recovered
                    grounding = recovered_grounding
                    generation_contexts = recovery_contexts
                    metadata["generation_context_ids"] = [
                        context.parent_id for context in recovery_contexts
                    ]
            elif (
                not grounding.valid
                and grounding.reason_code in RECOVERABLE_GENERATION_REASONS
            ):
                recovery = {
                    "attempted": False,
                    "success": False,
                    "skipped_reason": "first_attempt_too_slow",
                    "first_attempt_ms": round(first_attempt_ms, 3),
                }
            metadata["generation_recovery"] = recovery
            metadata["generation"] = {
                "provider": _field(generated, "provider", "qwen-gguf"),
                "model": _field(generated, "model", FROZEN_GENERATION_MODEL),
                "provider_status": _field(generated, "status", "error"),
                "answer_status": _field(generated, "answer_status", None),
                "error_code": _field(generated, "error_code", None),
                "error_message": _field(generated, "error_message", None),
                "http_status": _field(generated, "http_status", None),
                "latency_ms": _field(generated, "latency_ms", None),
                "time_to_first_token_ms": _field(generated, "time_to_first_token_ms", None),
                "prompt_tokens": _field(generated, "prompt_tokens", None),
                "output_tokens": _field(generated, "output_tokens", None),
                "tokens_per_second": _field(generated, "tokens_per_second", None),
                "provider_latency_ms": _field(generated, "provider_latency_ms", None),
                "diagnostics": _field(generated, "diagnostics", {}),
            }
            metadata["latency_checkpoints"] = (
                metadata["generation"].get("diagnostics", {}).get("checkpoints", {})
            )
            metadata["generation"]["runtime"] = metadata["latency_checkpoints"].get("qwen_runtime")
            metadata["generation"]["answer_mode"] = "model_generated"
            metadata["grounding"] = {
                "valid": grounding.valid,
                "route": grounding.route.value,
                "reason_code": grounding.reason_code.value,
            }
            decision_trace.append(
                {
                    "stage": "grounding_validation",
                    "allow": grounding.valid,
                    "route": grounding.route.value,
                    "reason_code": grounding.reason_code.value,
                }
            )
            if grounding.valid and _answer_echoes_query(
                grounding.answer, validation.normalized_transcript
            ):
                # Small SLMs occasionally parrot the question back as the
                # answer; lexical grounding can pass such echoes when the
                # retrieved passage shares phrasing. Reject them explicitly.
                metadata["grounding"] = {
                    "valid": False,
                    "route": Route.INSUFFICIENT_CONTEXT.value,
                    "reason_code": ReasonCode.MODEL_INSUFFICIENT_CONTEXT.value,
                    "note": "answer_echoed_query",
                }
                decision_trace.append(
                    {
                        "stage": "echo_guard",
                        "allow": False,
                        "route": Route.INSUFFICIENT_CONTEXT.value,
                        "reason_code": ReasonCode.MODEL_INSUFFICIENT_CONTEXT.value,
                    }
                )
                return _response(
                    Route.INSUFFICIENT_CONTEXT,
                    ReasonCode.MODEL_INSUFFICIENT_CONTEXT,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    retrieved_ids=retrieved_ids,
                    metadata=metadata,
                )
            return _response(
                grounding.route,
                grounding.reason_code,
                stages,
                operation_started,
                transcript=validation.normalized_transcript,
                answer=grounding.answer,
                retrieved_ids=retrieved_ids,
                citations=grounding.citations,
                metadata=metadata,
            )
        except Exception as error:
            # Never leak a stack trace to clients, but the operator must be
            # able to see which component failed and why in the Space logs.
            LOGGER.exception(
                "Voice-RAG pipeline error",
                extra={"component": type(error).__name__},
            )
            metadata["system_error"] = type(error).__name__
            return _response(
                Route.SYSTEM_ERROR,
                ReasonCode.SYSTEM_COMPONENT_ERROR,
                stages,
                operation_started,
                transcript=transcript if isinstance(transcript, str) else "",
                metadata=metadata,
            )


def _assert_frozen_config(config: dict[str, Any]) -> None:
    chunking = config.get("chunking", {})
    index = config.get("index", {})
    observed = {
        "model": config.get("model"),
        "chunking_strategy": chunking.get("strategy"),
        "chunk_size_words": chunking.get("max_words", chunking.get("size")),
        "index_engine": index.get("engine"),
        "index_type": index.get("index_type"),
        "m": index.get("m"),
        "ef_construction": index.get("ef_construction"),
        "ef_search": index.get("ef_search"),
        "search_oversample": config.get("search_oversample"),
        "normalization_method": config.get("normalization_method"),
        "top_k": config.get("top_k"),
    }
    if observed != FROZEN_RETRIEVAL:
        raise RuntimeError(f"Frozen retrieval configuration changed: {observed}")


def _response(
    route: Route,
    reason_code: ReasonCode | None,
    stages: dict[str, float],
    operation_started: int,
    *,
    transcript: str = "",
    answer: str = "",
    retrieved_ids: tuple[str, ...] = (),
    citations: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> GuardrailResponse:
    total_latency = _elapsed_ms(operation_started)
    stage_values = {name: float(stages.get(name, 0.0)) for name in STAGE_NAMES}
    stage_values.update(
        {
            "query_embedding": stage_values["embedding"],
            "vector_search": stage_values["retrieval"],
            "guardrails": sum(
                stage_values[name]
                for name in (
                    "input_validation",
                    "route_check",
                    "evidence_guardrail",
                    "grounding_validation",
                )
            ),
            "total_end_to_end": total_latency,
        }
    )
    # Coverage-first latency honesty: the 200 ms target is reported against
    # every response instead of enforced by refusal. The UI surfaces overruns
    # with an explanation rather than hiding them.
    budget_ms = _latency_target_ms()
    metadata = metadata or {}
    metadata.setdefault(
        "latency_budget",
        {
            "target_ms": budget_ms,
            "met": total_latency <= budget_ms,
            "over_by_ms": round(max(total_latency - budget_ms, 0.0), 1),
        },
    )
    return GuardrailResponse(
        route=route,
        answer=answer,
        retrieved_ids=retrieved_ids,
        citations=citations,
        reason_code=reason_code,
        stage_latencies_ms=stage_values,
        total_latency_ms=total_latency,
        transcript=transcript,
        metadata=metadata,
    )


def _latency_target_ms() -> float:
    raw = os.getenv("HH_RAG_LATENCY_TARGET_MS", "200").strip()
    try:
        value = float(raw)
    except ValueError:
        return 200.0
    return value if value > 0 else 200.0


def _empty_stages() -> dict[str, float]:
    return {name: 0.0 for name in STAGE_NAMES}


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _generate_answer(
    generator: Generator,
    question: str,
    contexts: list[GenerationContext],
    *,
    prompt_variant: str,
    language_code: str | None,
) -> Any:
    """Call generators with the requested language while keeping test doubles compatible."""

    generate = generator.generate
    kwargs: dict[str, Any] = {"prompt_variant": prompt_variant}
    try:
        parameters = inspect.signature(generate).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    if language_code is not None and (
        any(parameter.name == "language_code" for parameter in parameters)
        or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    ):
        kwargs["language_code"] = language_code
    return generate(question, contexts, **kwargs)


def _generate_with_timeout(
    generator: Generator,
    question: str,
    contexts: list[GenerationContext],
    *,
    prompt_variant: str,
    language_code: str | None,
    timeout_s: float,
) -> Any:
    """Run one generation under a hard wall-clock ceiling.

    A hung provider call (ZeroGPU queue stall, stuck CPU fallback) must never
    hold the request open indefinitely. On timeout the request fails closed
    with a structured provider error; the abandoned worker thread is left to
    finish or die in the background.
    """

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hh-generation")
    try:
        future = executor.submit(
            _generate_answer,
            generator,
            question,
            contexts,
            prompt_variant=prompt_variant,
            language_code=language_code,
        )
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            future.cancel()
            return {
                "provider": type(generator).__name__,
                "model": _field(generator, "model_name", None),
                "status": "error",
                "answer_status": None,
                "answer": "",
                "evidence_ids": (),
                "raw_output": "",
                "error_code": "generation_timeout",
                "error_message": f"generation exceeded {timeout_s:.0f}s wall clock",
                "latency_ms": timeout_s * 1000.0,
                "diagnostics": {"timeout_s": timeout_s},
            }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _retrieve_evidence(
    retriever: Retriever,
    query_embedding: Any,
    *,
    query_text: str,
    language_code: str | None,
) -> list[Any]:
    """Pass optional hybrid-retrieval inputs without breaking simple test doubles."""

    retrieve = retriever.retrieve
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(retrieve).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    names = {parameter.name for parameter in parameters}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if "query_text" in names or accepts_kwargs:
        kwargs["query_text"] = query_text
    if language_code is not None and ("language_code" in names or accepts_kwargs):
        kwargs["language_code"] = language_code
    return list(retrieve(query_embedding, **kwargs))


def _build_fast_tier(device: str) -> FastTier | None:
    """Load the resident extractive fast tier; disable it on any failure."""
    if os.getenv("HH_RAG_FAST_TIER", "1").strip().lower() in {"0", "false", "off"}:
        return None
    model = os.getenv("HH_RAG_FAST_TIER_MODEL", FROZEN_FAST_TIER_MODEL).strip() or (
        FROZEN_FAST_TIER_MODEL
    )
    threshold = os.getenv("HH_RAG_FAST_TIER_THRESHOLD", str(FROZEN_FAST_TIER_THRESHOLD))
    try:
        from hh_goa_rag.generation.local import ExtractiveQAEngine

        resolved = _resolve_fast_tier_path(model)
        try:
            engine = ExtractiveQAEngine(
                str(resolved) if resolved is not None else model,
                device=device,
                confidence_threshold=float(threshold),
            )
        except Exception:
            from hh_goa_rag.models import acquire_model

            root = Path(os.getenv("HH_RAG_MODEL_ROOT", "cache/models"))
            acquire_model(model, root)
            resolved = _resolve_fast_tier_path(model)
            if resolved is None:
                raise RuntimeError("fast-tier weights missing after acquisition") from None
            engine = ExtractiveQAEngine(
                str(resolved), device=device, confidence_threshold=float(threshold)
            )
    except Exception as error:
        print(f"Extractive fast tier disabled: {error!r}", flush=True)
        return None
    engine.warm_up(
        WARMUP_QUERY,
        [
            GenerationContext(
                parent_id="warmup-parent",
                chunk_id="warmup-chunk",
                text=WARMUP_PASSAGE,
                rank=1,
                score=1.0,
            )
        ],
    )
    return engine


def _resolve_fast_tier_path(model: str) -> Path | None:
    """Find project-local fast-tier weights without touching the network."""
    configured = os.getenv("HH_RAG_FAST_TIER_MODEL_PATH")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    root = Path(os.getenv("HH_RAG_MODEL_ROOT", "cache/models"))
    if root.is_dir():
        candidates.extend(sorted(root.glob(f"{safe_model_name(model)}--*")))
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    return None


def _run_fast_tier(
    fast_tier: FastTier,    question: str,
    contexts: list[GenerationContext],
    *,
    language_code: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Attempt one validated extractive answer; never raise into the pipeline."""
    outcome: dict[str, Any] = {
        "accepted": False,
        "status": "",
        "answer": "",
        "citation": None,
        "confidence": 0.0,
        "latency_ms": 0.0,
        "validation_error": None,
    }
    threshold = _fast_tier_threshold(language_code, thresholds)
    outcome["threshold_applied"] = threshold
    try:
        result = _call_fast_tier(fast_tier, question, contexts, threshold=threshold)
    except Exception:
        outcome["status"] = "exception"
        outcome["validation_error"] = "fast_tier_exception"
        return outcome
    status = str(_field(result, "status", ""))
    outcome["status"] = status
    outcome["confidence"] = float(_field(result, "confidence", 0.0))
    outcome["latency_ms"] = float(_field(result, "latency_ms", 0.0))
    if status != "ANSWER":
        outcome["validation_error"] = "fast_tier_abstained"
        return outcome
    answer = str(_field(result, "answer", "")).strip()
    evidence_ids = tuple(_field(result, "evidence_ids", ()) or ())
    citation = str(evidence_ids[0]) if evidence_ids else ""
    cited = next((item for item in contexts if item.parent_id == citation), None)
    if not answer or cited is None or answer not in cited.text:
        outcome["validation_error"] = "ungrounded_or_unknown_citation"
        return outcome
    if _is_low_information_span(answer):
        # Extractive QA is confidently wrong on passages that merely neighbour
        # the question: it returns a lone generic noun ("dealership",
        # "insomnia"). A usable span answers with a phrase, a figure, or a
        # named entity - so refuse bare single-token fragments.
        outcome["validation_error"] = "low_information_span"
        return outcome
    outcome.update(
        {
            "accepted": True,
            "answer": answer,
            "citation": citation,
            "validation_error": None,
        }
    )
    return outcome


def _answer_echoes_query(answer: str, query: str) -> bool:
    """True when the generated answer merely restates the question.

    Normalized containment either way (or near-total token overlap) means the
    model produced no information beyond what the user already said.
    """
    def normalize(value: str) -> str:
        return " ".join(str(value).split()).casefold()

    answer_norm = normalize(answer)
    query_norm = normalize(query)
    if not answer_norm or not query_norm:
        return False
    if answer_norm in query_norm or query_norm in answer_norm:
        return True
    answer_tokens = set(answer_norm.split())
    query_tokens = set(query_norm.split())
    if not answer_tokens or not query_tokens:
        return False
    overlap = len(answer_tokens & query_tokens)
    smaller = min(len(answer_tokens), len(query_tokens))
    return smaller > 0 and overlap / smaller >= 0.8


def _is_low_information_span(answer: str) -> bool:
    """True for bare single-token fragments that cannot constitute an answer.

    A span carrying a digit (``11 cups``, ``2020``), a Latin-script entity
    (``Bordeaux``, ``COVID-19``), or multiple words carries enough information
    to answer; a lone non-Latin noun is the classic confidently-wrong
    extraction from a passage that does not actually contain the answer.
    """
    stripped = answer.strip()
    if not stripped:
        return True
    if " " in stripped:
        return False
    if any(character.isdigit() for character in stripped):
        return False
    if any("a" <= character.lower() <= "z" for character in stripped):
        return False
    return True


def _fast_tier_threshold(
    language_code: str | None, thresholds: dict[str, float] | None
) -> float | None:
    """Resolve the effective cutoff: per-language override or engine default."""
    if not thresholds:
        return None
    key = language_key(language_code)
    if key and key in thresholds:
        return thresholds[key]
    return None


def _call_fast_tier(
    fast_tier: FastTier,
    question: str,
    contexts: list[GenerationContext],
    *,
    threshold: float | None,
) -> Any:
    """Pass an optional threshold override without breaking simple test doubles."""
    answer = fast_tier.answer
    if threshold is None:
        return answer(question, contexts)
    try:
        parameters = inspect.signature(answer).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_threshold = any(parameter.name == "threshold" for parameter in parameters) or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if not accepts_threshold:
        return answer(question, contexts)
    return answer(question, contexts, threshold=threshold)


def _cache_hit_response(
    cached: GuardrailResponse,
    base_metadata: dict[str, Any],
    *,
    lookup_ms: float,
) -> GuardrailResponse:
    """Return a labeled copy of a cached response with fresh transport metadata."""
    refresh = {
        key: value
        for key, value in base_metadata.items()
        if key in {"stt", "input_language", "requested_output_language"}
    }
    metadata = {
        **cached.metadata,
        **refresh,
        "cache_hit": True,
        "cache_lookup_ms": round(lookup_ms, 4),
    }
    return replace(cached, metadata=metadata)


def _parse_response_cache_env() -> ResponseCache | None:
    enabled = os.getenv("HH_RAG_RESPONSE_CACHE", "").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None
    ttl_raw = os.getenv("HH_RAG_RESPONSE_CACHE_TTL_S", "600").strip()
    try:
        ttl_s = float(ttl_raw)
    except ValueError as error:
        raise ValueError("HH_RAG_RESPONSE_CACHE_TTL_S must be numeric") from error
    return ResponseCache(ttl_s=ttl_s)


def _parse_deadline_env() -> tuple[float | None, float, float]:
    """Parse the strict-latency deadline configuration.

    ``HH_RAG_MAX_LATENCY_MS`` turns on deadline-guarded routing: generation is
    only started when the measured generator budget fits the remaining time,
    otherwise the request returns an honest ``DEADLINE_BUDGET_EXHAUSTED``
    refusal instead of a late answer.
    """
    raw = os.getenv("HH_RAG_MAX_LATENCY_MS", "").strip()
    if not raw:
        return None, DEFAULT_GENERATION_BUDGET_ESTIMATE_MS, DEFAULT_MIN_GENERATION_BUDGET_MS
    try:
        deadline_ms = float(raw)
        budget_estimate_ms = float(
            os.getenv(
                "HH_RAG_GENERATION_BUDGET_ESTIMATE_MS",
                str(DEFAULT_GENERATION_BUDGET_ESTIMATE_MS),
            )
        )
        min_budget_ms = float(
            os.getenv("HH_RAG_MIN_GENERATION_BUDGET_MS", str(DEFAULT_MIN_GENERATION_BUDGET_MS))
        )
    except ValueError as error:
        raise ValueError("deadline environment variables must be numeric") from error
    if deadline_ms <= 0 or budget_estimate_ms <= 0 or min_budget_ms <= 0:
        raise ValueError("deadline environment variables must be positive")
    return deadline_ms, budget_estimate_ms, min_budget_ms


def _parse_generation_timeout_env() -> float:
    """Hard wall-clock ceiling for one generation call; prevents hung requests."""
    raw = os.getenv("HH_RAG_GENERATION_TIMEOUT_S", str(DEFAULT_GENERATION_TIMEOUT_S)).strip()
    try:
        timeout_s = float(raw)
    except ValueError as error:
        raise ValueError("HH_RAG_GENERATION_TIMEOUT_S must be numeric") from error
    if timeout_s <= 0:
        raise ValueError("HH_RAG_GENERATION_TIMEOUT_S must be positive")
    return timeout_s


def _parse_fast_tier_min_budget_env() -> float:
    raw = os.getenv("HH_RAG_FAST_TIER_MIN_BUDGET_MS", str(DEFAULT_FAST_TIER_MIN_BUDGET_MS)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("HH_RAG_FAST_TIER_MIN_BUDGET_MS must be numeric") from error
    if value <= 0:
        raise ValueError("HH_RAG_FAST_TIER_MIN_BUDGET_MS must be positive")
    return value


def _parse_recovery_first_attempt_env() -> float:
    raw = os.getenv(
        "HH_RAG_RECOVERY_MAX_FIRST_ATTEMPT_MS", str(DEFAULT_RECOVERY_MAX_FIRST_ATTEMPT_MS)
    ).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("HH_RAG_RECOVERY_MAX_FIRST_ATTEMPT_MS must be numeric") from error
    if value <= 0:
        raise ValueError("HH_RAG_RECOVERY_MAX_FIRST_ATTEMPT_MS must be positive")
    return value


def _validate_fast_tier_thresholds(
    thresholds: dict[str, float] | None,
) -> dict[str, float]:
    if not thresholds:
        return {}
    validated: dict[str, float] = {}
    for code, value in thresholds.items():
        key = language_key(str(code))
        number = float(value)
        if not key or not 0.0 < number < 1.0:
            raise ValueError(
                f"fast-tier threshold for {code!r} must be a language key with a "
                "value in (0, 1)"
            )
        validated[key] = number
    return validated


def _parse_fast_tier_threshold_env() -> dict[str, float] | None:
    raw = os.getenv("HH_RAG_FAST_TIER_THRESHOLDS", "").strip()
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(value, (int, float))
        for key, value in parsed.items()
    ):
        raise ValueError(
            "HH_RAG_FAST_TIER_THRESHOLDS must be a JSON object mapping language "
            "codes to numeric confidence cutoffs"
        )
    return {str(key): float(value) for key, value in parsed.items()}


def _elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1e6


def _notify_stage(callback: Callable[[str], None] | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _portable_path(value: str | Path) -> Path:
    """Interpret persisted repository-relative paths on Windows and Linux."""
    return Path(str(value).replace("\\", "/"))
