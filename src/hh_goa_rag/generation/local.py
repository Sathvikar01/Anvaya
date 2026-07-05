"""Experimental resident local answer engines for Phase 9 ablations.

Nothing in this module is wired into the frozen production harness.  The classes
load their tokenizer/model once, expose explicit warm-up methods, and preserve
the parent ID of every cited evidence item.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from .sarvam import GenerationContext

AnswerStatus = Literal["ANSWER", "INSUFFICIENT_CONTEXT", "INVALID"]
_NUMBER_RE = re.compile(r"(?<!\w)[£$€₹]?\s*\d+(?:[.,]\d+)*(?:\s*[-–]\s*[£$€₹]?\d+(?:[.,]\d+)*)?")
_CITATION_RE = re.compile(r"\[?E([1-9]\d*)\]?")


@dataclass(frozen=True)
class LocalAnswer:
    """A local answer plus provenance and deterministic validation diagnostics."""

    status: AnswerStatus
    answer: str
    evidence_ids: tuple[str, ...]
    confidence: float
    latency_ms: float
    raw_output: str = ""
    validation_error: str = ""
    grounding_overlap: float | None = None

    @property
    def answered(self) -> bool:
        return self.status == "ANSWER"


class ExtractiveQAEngine:
    """Multilingual extractive QA with span, citation, confidence, and abstention."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        max_length: int = 384,
        max_answer_tokens: int = 48,
        confidence_threshold: float = 0.5,
    ) -> None:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        started = time.perf_counter_ns()
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.max_question_tokens = max(1, max_length // 2)
        self.max_answer_tokens = max_answer_tokens
        self.confidence_threshold = confidence_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            local_files_only=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        _synchronize(self.device)
        self.load_ms = (time.perf_counter_ns() - started) / 1_000_000

    def warm_up(self, question: str, contexts: list[GenerationContext]) -> LocalAnswer:
        return self.answer(question, contexts)

    def answer(
        self,
        question: str,
        contexts: list[GenerationContext],
        *,
        threshold: float | None = None,
    ) -> LocalAnswer:
        import torch

        if not contexts:
            return LocalAnswer("INSUFFICIENT_CONTEXT", "", (), 0.0, 0.0)
        started = time.perf_counter_ns()
        # `truncation="only_second"` raises when the question alone fills the
        # window, which previously surfaced as a fast-tier exception and a
        # silent abstention. Cap the question so context truncation always
        # has room.
        question_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        if len(question_ids) > self.max_question_tokens:
            question = self.tokenizer.decode(
                question_ids[: self.max_question_tokens], skip_special_tokens=True
            ).strip()
        encoded = self.tokenizer(
            [question] * len(contexts),
            [context.text for context in contexts],
            max_length=self.max_length,
            padding=True,
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**model_inputs)
        _synchronize(self.device)

        best: tuple[float, str, str] | None = None
        best_margin = -math.inf
        for row_index, context in enumerate(contexts):
            sequence_ids = encoded.sequence_ids(row_index)
            context_indices = [
                index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
            ]
            if not context_indices:
                continue
            start_logits = output.start_logits[row_index].float().cpu()
            end_logits = output.end_logits[row_index].float().cpu()
            null_score = float(start_logits[0] + end_logits[0])
            top_starts = sorted(
                context_indices, key=lambda index: float(start_logits[index]), reverse=True
            )[:20]
            top_ends = sorted(
                context_indices, key=lambda index: float(end_logits[index]), reverse=True
            )[:20]
            for start_index in top_starts:
                for end_index in top_ends:
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > self.max_answer_tokens:
                        continue
                    char_start = int(offsets[row_index, start_index, 0])
                    char_end = int(offsets[row_index, end_index, 1])
                    if char_end <= char_start:
                        continue
                    span = context.text[char_start:char_end].strip()
                    if not span or span not in context.text:
                        continue
                    score = float(start_logits[start_index] + end_logits[end_index])
                    margin = score - null_score
                    if margin > best_margin:
                        best_margin = margin
                        best = (score, span, context.parent_id)

        confidence = _sigmoid(best_margin) if best else 0.0
        cutoff = self.confidence_threshold if threshold is None else threshold
        latency_ms = _elapsed_ms(started, self.device)
        if best is None or confidence < cutoff:
            return LocalAnswer(
                "INSUFFICIENT_CONTEXT",
                "",
                (),
                confidence,
                latency_ms,
                validation_error="below_confidence_threshold",
            )
        _, span, parent_id = best
        if not extractive_span_integrity(span):
            return LocalAnswer(
                "INSUFFICIENT_CONTEXT",
                "",
                (),
                confidence,
                latency_ms,
                raw_output=span,
                validation_error="span_integrity_check_failed",
            )
        return LocalAnswer("ANSWER", span, (parent_id,), confidence, latency_ms, raw_output=span)


class TinyGeneratorEngine:
    """Resident deterministic sub-1B generator with citation/grounding validation."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        max_new_tokens: int = 64,
        min_grounding_overlap: float = 0.35,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter_ns()
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.min_grounding_overlap = min_grounding_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        _synchronize(self.device)
        self.load_ms = (time.perf_counter_ns() - started) / 1_000_000

    def warm_up(self, question: str, contexts: list[GenerationContext]) -> LocalAnswer:
        return self.answer(question, contexts)

    def answer(self, question: str, contexts: list[GenerationContext]) -> LocalAnswer:
        import torch

        if not contexts:
            return LocalAnswer("INSUFFICIENT_CONTEXT", "", (), 0.0, 0.0)
        started = time.perf_counter_ns()
        messages = [
            {
                "role": "system",
                "content": (
                    "The question is answerable from the supplied evidence. "
                    "Answer briefly in Hindi "
                    "using only that evidence, then cite the evidence label in square brackets. "
                    "Never add outside facts and do not explain your work."
                ),
            },
            {"role": "user", "content": _generator_prompt(question, contexts)},
        ]
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "Qwen3" in self.model_name:
            template_kwargs["enable_thinking"] = False
        prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        _synchronize(self.device)
        output_ids = generated.sequences[0, encoded["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        confidence = _mean_token_probability(generated.scores, output_ids)
        latency_ms = _elapsed_ms(started, self.device)
        result = validate_generated_answer(
            raw,
            contexts,
            confidence=confidence,
            latency_ms=latency_ms,
            min_grounding_overlap=self.min_grounding_overlap,
        )
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        completed = bool(len(output_ids)) and int(output_ids[-1]) in set(eos_ids or [])
        if not completed and result.status == "ANSWER":
            return LocalAnswer(
                "INVALID",
                "",
                (),
                confidence,
                latency_ms,
                raw,
                "max_tokens_exhausted",
                result.grounding_overlap,
            )
        return result


def validate_generated_answer(
    raw_output: str,
    contexts: list[GenerationContext],
    *,
    confidence: float,
    latency_ms: float,
    min_grounding_overlap: float = 0.35,
) -> LocalAnswer:
    """Reject malformed, uncited, numerically novel, or weakly grounded generations."""
    raw = raw_output.strip()
    if "INSUFFICIENT_CONTEXT" in raw.upper():
        return LocalAnswer("INSUFFICIENT_CONTEXT", "", (), confidence, latency_ms, raw)
    citation_numbers = sorted({int(value) for value in _CITATION_RE.findall(raw)})
    if not citation_numbers:
        return LocalAnswer(
            "INVALID", "", (), confidence, latency_ms, raw, "missing_citation"
        )
    if any(value < 1 or value > len(contexts) for value in citation_numbers):
        return LocalAnswer(
            "INVALID", "", (), confidence, latency_ms, raw, "unknown_citation"
        )
    answer = _CITATION_RE.sub("", raw)
    answer = re.sub(
        r"(?im)^\s*(?:उत्तर|answer|स्रोत|source)\s*[:：-]?\s*",
        "",
        answer,
    )
    answer = re.sub(r"[\[\]]", "", answer).strip(" \n:;,-")
    if not answer:
        return LocalAnswer("INVALID", "", (), confidence, latency_ms, raw, "empty_answer")
    cited = [contexts[index - 1] for index in citation_numbers]
    evidence_text = " ".join(item.text for item in cited)
    novel_numbers = sorted(_numbers(answer) - _numbers(evidence_text))
    if novel_numbers:
        return LocalAnswer(
            "INVALID",
            "",
            (),
            confidence,
            latency_ms,
            raw,
            f"novel_numbers:{','.join(novel_numbers)}",
        )
    overlap = grounding_overlap(answer, evidence_text)
    if overlap < min_grounding_overlap:
        return LocalAnswer(
            "INVALID",
            "",
            (),
            confidence,
            latency_ms,
            raw,
            "low_grounding_overlap",
            overlap,
        )
    return LocalAnswer(
        "ANSWER",
        answer,
        tuple(item.parent_id for item in cited),
        confidence,
        latency_ms,
        raw,
        grounding_overlap=overlap,
    )


def grounding_overlap(answer: str, evidence: str) -> float:
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    evidence_tokens = set(_content_tokens(evidence))
    return sum(token in evidence_tokens for token in answer_tokens) / len(answer_tokens)


def extractive_span_integrity(span: str) -> bool:
    """Reject visibly truncated currency ranges while keeping the check deterministic."""
    currency_ranges = re.finditer(
        r"(?:डॉलर|पाउंड|रुपये|dollars?|pounds?)\s+(?:से|to)\s+\d",
        span,
        flags=re.IGNORECASE,
    )
    return all(
        re.search(r"\d[\d.,]*$", span[: match.start()].rstrip()) is not None
        for match in currency_ranges
    )


def select_route_tier(
    extractive_confidence: float,
    extractive_span: str,
    generator_status: AnswerStatus,
    *,
    threshold: float = 0.98,
    generator_enabled: bool = True,
) -> int:
    """Return a deterministic experimental routing tier without external state."""
    if extractive_confidence >= threshold and extractive_span_integrity(extractive_span):
        return 1
    if generator_enabled and generator_status == "ANSWER":
        return 2
    return 3


def _generator_prompt(question: str, contexts: list[GenerationContext]) -> str:
    evidence = "\n".join(
        f"[E{index}] {context.text}" for index, context in enumerate(contexts, start=1)
    )
    return f"प्रश्न: {question}\n\nसाक्ष्य:\n{evidence}\n\nउत्तर:"


def _content_tokens(text: str) -> list[str]:
    stopwords = {
        "और",
        "का",
        "की",
        "के",
        "को",
        "में",
        "से",
        "है",
        "हैं",
        "यह",
        "तो",
        "एक",
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "in",
        "to",
    }
    normalized = "".join(
        character
        if unicodedata.category(character)[0] in {"L", "M", "N"}
        else " "
        for character in text.casefold()
    )
    return [token for token in normalized.split() if token not in stopwords]


def _numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", value) for value in _NUMBER_RE.findall(text)}


def _mean_token_probability(scores: Any, output_ids: Any) -> float:
    import torch

    if not scores or len(output_ids) == 0:
        return 0.0
    log_probabilities = []
    for score, token_id in zip(scores, output_ids, strict=False):
        log_probability = torch.log_softmax(score[0].float(), dim=-1)[int(token_id)]
        log_probabilities.append(float(log_probability))
    return math.exp(sum(log_probabilities) / len(log_probabilities))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-min(value, 80)))
    exp_value = math.exp(max(value, -80))
    return exp_value / (1 + exp_value)


def _synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(started_ns: int, device: Any) -> float:
    _synchronize(device)
    return (time.perf_counter_ns() - started_ns) / 1_000_000
