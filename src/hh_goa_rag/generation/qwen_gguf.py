"""Local Qwen3.5 Q4 GGUF answer generation for the ZeroGPU Space."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import PromptVariant, build_messages
from .sarvam import GenerationContext, GenerationResult, _parse_answer

QWEN_GGUF_REPOSITORY = "ggml-org/Qwen3.5-0.8B-GGUF"
QWEN_GGUF_FILENAME = "Qwen3.5-0.8B-Q4_0.gguf"
QWEN_GGUF_MODEL = f"{QWEN_GGUF_REPOSITORY}/{QWEN_GGUF_FILENAME}"

# GBNF grammar forcing exactly the schema `_parse_answer` accepts: fixed key
# order, the two allowed statuses, a JSON string answer, and an array of
# evidence-ID strings. This removes malformed-JSON outputs that previously
# triggered the full-cost recovery retry.
ANSWER_JSON_GRAMMAR = r'''root ::= "{" ws status-kv "," ws answer-kv "," ws evidence-kv "}" ws
status-kv ::= "\"status\"" ws ":" ws status
status ::= "\"ANSWER\"" | "\"INSUFFICIENT_CONTEXT\""
answer-kv ::= "\"answer\"" ws ":" ws string
evidence-kv ::= "\"evidence_ids\"" ws ":" ws array
array ::= "[" ws (string ("," ws string)*)? "]"
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
ws ::= | " " | "\n" [ \t]{0,20}'''

_GRAMMAR_OBJ: Any | None = None
_MODEL: Any | None = None
_MODEL_PATH = ""
_MODEL_RUNTIME = "uninitialized"
_MODEL_LOCK = threading.Lock()
_GPU_ENTRY: Any | None = None
# Circuit breaker: once a ZeroGPU allocation fails (quota exhaustion, broken
# worker, "No CUDA GPUs are available"), stop paying the multi-second GPU
# acquisition cost on every request and serve from the resident CPU model.
_GPU_UNAVAILABLE = False


def _grammar_request_value() -> Any:
    """Compile the GBNF answer grammar once for this llama.cpp build.

    llama-cpp-python 0.3.35 requires a ``LlamaGrammar`` instance in the
    ``grammar`` field; older builds accepted the raw string. Compiling lazily
    keeps module import cheap and works on either build.
    """
    global _GRAMMAR_OBJ
    if _GRAMMAR_OBJ is not None:
        return _GRAMMAR_OBJ
    from llama_cpp import LlamaGrammar

    try:
        _GRAMMAR_OBJ = LlamaGrammar.from_string(ANSWER_JSON_GRAMMAR)
    except Exception as error:
        print(f"LlamaGrammar compilation failed; using raw GBNF: {error!r}", flush=True)
        _GRAMMAR_OBJ = ANSWER_JSON_GRAMMAR
    return _GRAMMAR_OBJ


@dataclass(frozen=True)
class QwenGGUFGenerationConfig:
    model: str = QWEN_GGUF_MODEL
    max_tokens: int = 128
    context_size: int = 2048
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.context_size <= 0 or self.timeout_s <= 0:
            raise ValueError("max_tokens, context_size, and timeout_s must be positive")


def set_model_path(path: str) -> None:
    """Set the Space-local GGUF path before the first GPU request."""
    global _MODEL_PATH
    _MODEL_PATH = path


def _cpu_threads() -> int:
    """Bound llama.cpp decode threads to the container's real CPU allowance.

    Hosts expose many cores while the container cgroup quota allows only a
    few; letting llama.cpp spawn host-core threads makes decoding slower, not
    faster.
    """
    try:
        quota_text = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        if len(quota_text) == 2 and quota_text[0] != "max":
            quota, period = int(quota_text[0]), int(quota_text[1])
            return max(1, min(quota // max(period, 1), 8))
    except Exception:
        pass
    try:
        quota = int(
            Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8")
        )
        period = int(
            Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8")
        )
        if quota > 0 and period > 0:
            return max(1, min(quota // period, 8))
    except Exception:
        pass
    return max(1, min(os.cpu_count() or 4, 8))


def resolve_gguf_model() -> str:
    """Resolve the served GGUF identity; defaults to the frozen Qwen3.5 0.8B.

    ``HH_RAG_GGUF_REPOSITORY`` / ``HH_RAG_GGUF_FILENAME`` allow a Space-side
    A/B (for example the Gemma-3-1B-it GGUF from the SLM bake-off) without
    code changes; prompts and the answer grammar are model-agnostic.
    """
    repository = os.getenv("HH_RAG_GGUF_REPOSITORY", "").strip() or QWEN_GGUF_REPOSITORY
    filename = os.getenv("HH_RAG_GGUF_FILENAME", "").strip() or QWEN_GGUF_FILENAME
    return f"{repository}/{filename}"


def _generate_on_gpu(
    messages: list[dict[str, str]], config: QwenGGUFGenerationConfig
) -> dict[str, Any]:
    """Run one deterministic Qwen Q4 generation inside a ZeroGPU allocation."""
    global _MODEL, _MODEL_RUNTIME

    if not _MODEL_PATH:
        raise RuntimeError("GENERATOR_GGUF_PATH is not configured")

    from llama_cpp import Llama

    with _MODEL_LOCK:
        load_started = time.perf_counter_ns()
        model_load_ms = 0.0
        if _MODEL is None:
            try:
                _MODEL = Llama(
                    model_path=_MODEL_PATH,
                    n_ctx=config.context_size,
                    n_batch=1024,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                _MODEL_RUNTIME = "llama.cpp/cuda"
            except Exception as error:
                print(f"Qwen CUDA load failed; using CPU fallback: {error!r}", flush=True)
                _MODEL = Llama(
                    model_path=_MODEL_PATH,
                    n_ctx=config.context_size,
                    n_batch=1024,
                    n_gpu_layers=0,
                    n_threads=_cpu_threads(),
                    n_threads_batch=_cpu_threads(),
                    verbose=False,
                )
                _MODEL_RUNTIME = "llama.cpp/cpu-fallback"
            model_load_ms = _elapsed_ms(load_started)

        prompt_started = time.perf_counter_ns()
        generation_started = time.perf_counter_ns()
        pieces: list[str] = []
        first_token_ms: float | None = None
        request = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": config.max_tokens,
            "stream": False,
            "grammar": _grammar_request_value(),
        }
        try:
            pieces, first_token_ms = _collect_stream(
                _MODEL, request, generation_started, disable_thinking=True
            )
        except Exception as error:
            if _MODEL_RUNTIME != "llama.cpp/cuda":
                raise
            print(f"Qwen CUDA generation failed; using CPU fallback: {error!r}", flush=True)
            _MODEL = Llama(
                model_path=_MODEL_PATH,
                n_ctx=config.context_size,
                n_batch=1024,
                n_gpu_layers=0,
                n_threads=_cpu_threads(),
                n_threads_batch=_cpu_threads(),
                verbose=False,
            )
            _MODEL_RUNTIME = "llama.cpp/cpu-fallback"
            pieces, first_token_ms = _collect_stream(
                _MODEL, request, generation_started, disable_thinking=False
            )

        raw_output = "".join(pieces).strip()
        return {
            "raw_output": raw_output,
            "model_load_ms": model_load_ms,
            "prompt_ms": _elapsed_ms(prompt_started),
            "generation_ms": _elapsed_ms(generation_started),
            "total_ms": _elapsed_ms(load_started),
            "time_to_first_token_ms": first_token_ms,
            "output_tokens": len(_MODEL.tokenize(raw_output.encode("utf-8"), add_bos=False)),
        }


def _generate_on_cpu(
    messages: list[dict[str, str]], config: QwenGGUFGenerationConfig
) -> dict[str, Any]:
    """Run the same GGUF request without relying on a ZeroGPU allocation.

    The CPU model is resident: it is loaded once per process and reused for
    every subsequent request instead of re-reading the GGUF from disk.
    """
    global _MODEL, _MODEL_RUNTIME

    if not _MODEL_PATH:
        raise RuntimeError("GENERATOR_GGUF_PATH is not configured")

    from llama_cpp import Llama

    with _MODEL_LOCK:
        load_started = time.perf_counter_ns()
        model_load_ms = 0.0
        if _MODEL is None:
            threads = _cpu_threads()
            print(
                f"Loading resident Qwen GGUF on CPU with {threads} threads",
                flush=True,
            )
            _MODEL = Llama(
                model_path=_MODEL_PATH,
                n_ctx=config.context_size,
                n_batch=1024,
                n_gpu_layers=0,
                n_threads=threads,
                n_threads_batch=threads,
                verbose=False,
            )
            _MODEL_RUNTIME = "llama.cpp/cpu-fallback"
            model_load_ms = _elapsed_ms(load_started)
        prompt_started = time.perf_counter_ns()
        generation_started = time.perf_counter_ns()
        request = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": config.max_tokens,
            "stream": False,
            "grammar": _grammar_request_value(),
        }
        pieces, first_token_ms = _collect_stream(
            _MODEL, request, generation_started, disable_thinking=False
        )
        raw_output = "".join(pieces).strip()
        return {
            "raw_output": raw_output,
            "model_load_ms": model_load_ms,
            "prompt_ms": _elapsed_ms(prompt_started),
            "generation_ms": _elapsed_ms(generation_started),
            "total_ms": _elapsed_ms(load_started),
            "time_to_first_token_ms": first_token_ms,
            "output_tokens": len(_MODEL.tokenize(raw_output.encode("utf-8"), add_bos=False)),
        }


def _collect_stream(
    model: Any,
    request: dict[str, Any],
    generation_started: int,
    *,
    disable_thinking: bool,
) -> tuple[list[str], float | None]:
    try:
        stream = (
            model.create_chat_completion(
                **request,
                chat_template_kwargs={"enable_thinking": False},
            )
            if disable_thinking
            else model.create_chat_completion(**request)
        )
    except TypeError:
        stream = model.create_chat_completion(**request)
    if not request.get("stream", True):
        choice = (stream.get("choices") or [{}])[0]
        content = str((choice.get("message") or {}).get("content") or "")
        return ([content] if content else []), None
    pieces: list[str] = []
    first_token_ms: float | None = None
    for event in stream:
        choice = (event.get("choices") or [{}])[0]
        content = str((choice.get("delta") or {}).get("content") or "")
        if content:
            if first_token_ms is None:
                first_token_ms = _elapsed_ms(generation_started)
            pieces.append(content)
    return pieces, first_token_ms


class QwenGGUFGeneration:
    """Generate grounded answers with the exact Qwen3.5 0.8B Q4_0 GGUF."""

    def __init__(self, *, config: QwenGGUFGenerationConfig | None = None) -> None:
        self.config = config or QwenGGUFGenerationConfig()
        set_model_path(os.environ.get("GENERATOR_GGUF_PATH", ""))
        _configure_gpu_entry()

    @classmethod
    def from_env(
        cls,
        *,
        config: QwenGGUFGenerationConfig | None = None,
    ) -> QwenGGUFGeneration:
        return cls(config=config)

    def warm_up(self) -> None:
        """Probe the CPU runtime once at boot.

        The probe (a) surfaces llama.cpp import/load problems in the Space
        logs at startup instead of on a user's first request and (b) pays the
        one-time GGUF load so later CPU fallback answers skip it. ZeroGPU is
        still acquired lazily for real answers when available.
        """
        if os.getenv("HH_RAG_QWEN_CPU_PROBE", "1").strip().lower() in {
            "0",
            "false",
            "off",
        }:
            return
        try:
            probe_messages = build_messages(
                "warmup",
                [
                    GenerationContext(
                        parent_id="warmup-parent",
                        chunk_id="warmup-chunk",
                        text="warmup passage",
                        rank=1,
                        score=1.0,
                    )
                ],
                variant="structured_evidence_ids",
            )
            started = time.perf_counter_ns()
            _generate_on_cpu(
                probe_messages,
                QwenGGUFGenerationConfig(max_tokens=8),
            )
            print(
                f"Qwen CPU warm-up ok in {_elapsed_ms(started):.0f} ms "
                f"(runtime={_MODEL_RUNTIME})",
                flush=True,
            )
        except Exception as error:
            print(f"Qwen CPU warm-up FAILED: {error!r}", flush=True)

    def close(self) -> None:
        """Release the worker-local model if the host asks the harness to close."""
        global _MODEL
        _MODEL = None

    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext | dict[str, Any]],
        *,
        prompt_variant: PromptVariant = "structured_evidence_ids",
        language_code: str | None = None,
    ) -> GenerationResult:
        prompt_contexts = list(contexts)[:5]
        messages = build_messages(
            question,
            prompt_contexts,
            variant=prompt_variant,
            language_code=language_code,
        )
        allowed_ids = {str(_field(context, "parent_id")) for context in prompt_contexts}
        started = time.perf_counter_ns()
        global _GPU_UNAVAILABLE
        _GPU_UNAVAILABLE = _GPU_UNAVAILABLE or (
            os.getenv("HH_RAG_DISABLE_ZEROGPU", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if not _GPU_UNAVAILABLE:
            try:
                observation = _gpu_entry()(messages, self.config)
            except Exception as error:
                # A ZeroGPU decorator failure can happen before _generate_on_gpu
                # runs, so the in-function CUDA→CPU fallback is not sufficient.
                # Trip the circuit breaker so later requests skip the doomed
                # GPU acquisition entirely, then retry on the resident CPU model.
                _GPU_UNAVAILABLE = True
                print(
                    "Qwen ZeroGPU request failed; disabling ZeroGPU for this "
                    f"process and retrying on CPU: {error!r}",
                    flush=True,
                )
                try:
                    observation = _generate_on_cpu(messages, self.config)
                except Exception as cpu_error:
                    print(
                        f"Qwen CPU fallback failed: {cpu_error!r}", flush=True
                    )
                    raise error from None
        else:
            observation = _generate_on_cpu(messages, self.config)
        try:
            raw_output = str(observation.get("raw_output") or "")
            cleaned_output = _clean_model_output(raw_output)
            parsed, diagnostics = _parse_answer(cleaned_output, allowed_ids)
            checkpoints = {
                "qwen_model_load_ms": observation.get("model_load_ms"),
                "qwen_prompt_ms": observation.get("prompt_ms"),
                "qwen_generation_ms": observation.get("generation_ms"),
                "qwen_total_ms": observation.get("total_ms"),
                "qwen_time_to_first_token_ms": observation.get("time_to_first_token_ms"),
                "qwen_output_tokens": observation.get("output_tokens"),
                "qwen_runtime": _MODEL_RUNTIME,
                "qwen_model_file": QWEN_GGUF_FILENAME,
            }
            if parsed is None:
                return self._result(
                    started,
                    status="error",
                    code="invalid_structured_output",
                    message=str(diagnostics.get("parse_error", "invalid JSON")),
                    raw_output=raw_output,
                    diagnostics={**diagnostics, "checkpoints": checkpoints},
                    ttft_ms=observation.get("time_to_first_token_ms"),
                    output_tokens=observation.get("output_tokens"),
                )
            return self._result(
                started,
                status="ok",
                answer_status=parsed["status"],
                answer=parsed["answer"],
                evidence_ids=tuple(parsed["evidence_ids"]),
                raw_output=raw_output,
                diagnostics={**diagnostics, "checkpoints": checkpoints},
                ttft_ms=observation.get("time_to_first_token_ms"),
                output_tokens=observation.get("output_tokens"),
            )
        except Exception as error:
            return self._result(
                started,
                status="error",
                code=type(error).__name__,
                message=str(error)[:500],
            )

    def _result(
        self,
        started: int,
        *,
        status: str,
        code: str | None = None,
        message: str | None = None,
        raw_output: str = "",
        answer_status: str | None = None,
        answer: str = "",
        evidence_ids: tuple[str, ...] = (),
        diagnostics: dict[str, Any] | None = None,
        ttft_ms: float | None = None,
        output_tokens: int | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            provider="qwen-gguf",
            model=self.config.model,
            status=status,  # type: ignore[arg-type]
            answer_status=answer_status,  # type: ignore[arg-type]
            answer=answer,
            evidence_ids=evidence_ids,
            raw_output=raw_output,
            latency_ms=_elapsed_ms(started),
            time_to_first_token_ms=ttft_ms,
            prompt_tokens=None,
            output_tokens=output_tokens,
            total_tokens=None,
            finish_reason=None,
            attempts=1,
            error_code=code,
            error_message=message,
            diagnostics=diagnostics or {},
        )


def _clean_model_output(raw_output: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name, "")
    return getattr(value, name, "")


def _gpu_entry() -> Any:
    if _GPU_ENTRY is not None:
        return _GPU_ENTRY
    return _generate_on_gpu


def _configure_gpu_entry() -> None:
    """Use ZeroGPU inside Spaces without requiring a non-standard env var."""

    global _GPU_ENTRY
    if _GPU_ENTRY is not None:
        return
    if not (os.environ.get("SPACE_ID") or os.environ.get("SPACES_ZERO_GPU") == "1"):
        return
    import spaces

    _GPU_ENTRY = spaces.GPU(duration=45)(_generate_on_gpu)


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000
