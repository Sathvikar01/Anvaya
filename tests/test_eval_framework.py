import json
from pathlib import Path

import pytest

from eval.metrics import (
    EvaluationInputError,
    assert_frozen_retrieval_config,
    evaluate_e2e_records,
    evaluate_generation_records,
    evaluate_guardrail_records,
    evaluate_retrieval_records,
    evaluate_stt_records,
    load_dataset,
    word_error_counts,
    write_evaluation_run,
)


def _case(
    case_id: str,
    *,
    route: str = "answer",
    relevant: list[str] | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "category": "normal_answerable",
        "stt_reference": "one two three",
        "expected": {"route": route, "relevant_parent_ids": relevant or []},
    }


def _judgment() -> dict:
    return {
        "correctness": 1,
        "relevance": 0.9,
        "faithfulness": 0.8,
        "claims": [{"text": "fact", "supported": True}],
    }


def test_word_error_rate_uses_word_level_levenshtein() -> None:
    result = word_error_counts("One, two three", "one too three extra")
    assert result == {"errors": 2, "reference_words": 3, "wer": pytest.approx(2 / 3)}


def test_stage_metrics_have_fixed_denominators() -> None:
    answerable = _case("a", relevant=["p1", "p2"])
    failed = _case("b", relevant=["p3"])
    stt, _ = evaluate_stt_records(
        [
            (answerable, {"case_id": "a", "transcript": "one too three", "latency_ms": 10}),
            (failed, {"case_id": "b", "status": "timeout", "latency_ms": 100}),
        ]
    )
    assert stt["wer_micro"] == pytest.approx(1 / 3)
    assert stt["failure_rate"] == 0.5
    assert stt["latency"]["p100_ms"] == 100

    retrieval, _ = evaluate_retrieval_records(
        [
            (
                answerable,
                {"case_id": "a", "retrieved_parent_ids": ["x", "p1", "p2"], "latency_ms": 1},
            ),
            (failed, {"case_id": "b", "status": "error", "latency_ms": 2}),
        ]
    )
    assert retrieval["recall_at_1"] == 0
    assert retrieval["recall_at_3"] == 0.5
    assert retrieval["mrr_at_10"] == 0.25
    assert retrieval["failure_rate"] == 0.5


def test_generation_guardrail_and_e2e_metrics() -> None:
    answerable = _case("a", relevant=["p1"])
    generation, details = evaluate_generation_records(
        [
            (
                answerable,
                {"case_id": "a", "answer": "fact", "latency_ms": 20, "judgment": _judgment()},
            )
        ]
    )
    assert generation["correctness"] == 1
    assert generation["unsupported_claim_rate"] == 0
    assert details[0]["faithfulness"] == 0.8

    insufficient = _case("i", route="refuse_insufficient_context")
    guardrails, _ = evaluate_guardrail_records(
        [
            (
                answerable,
                {
                    "case_id": "a",
                    "decision": "refuse_insufficient_context",
                    "latency_ms": 1,
                },
            ),
            (
                insufficient,
                {"case_id": "i", "decision": "refuse_insufficient_context", "latency_ms": 1},
            ),
        ]
    )
    assert guardrails["false_refusal_rate"] == 1
    assert guardrails["insufficient_context_refusal_accuracy"] == 1

    prediction = {
        "case_id": "a",
        "decision": "answer",
        "retrieved_parent_ids": ["p1"],
        "answer": "fact",
        "citations": ["p1"],
        "judgment": _judgment(),
        "latency_ms": 199,
    }
    e2e, rows = evaluate_e2e_records([(answerable, prediction)])
    assert e2e["success_rate"] == 1
    assert e2e["latency"]["under_200ms_rate"] == 1
    assert rows[0]["success"] is True


def test_frozen_config_and_sealed_split_guards(tmp_path: Path) -> None:
    frozen = {
        "model": "BAAI/bge-m3",
        "chunking": {"strategy": "sentence", "max_words": 128},
        "index": {
            "engine": "faiss",
            "index_type": "hnsw",
            "m": 32,
            "ef_construction": 200,
            "ef_search": 128,
        },
        "top_k": 10,
        "search_oversample": 20,
        "normalization_method": "float32_l2_v1",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(frozen), encoding="utf-8")
    assert assert_frozen_retrieval_config(config_path)["m"] == 32
    frozen["index"]["ef_search"] = 64
    config_path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="frozen"):
        assert_frozen_retrieval_config(config_path)

    sealed = {
        "record_type": "case",
        "case_id": "secret",
        "split": "sealed_test",
        "category": "normal_answerable",
        "stt_reference": "secret",
        "expected": {"route": "answer"},
    }
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(sealed) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="explicit"):
        load_dataset(dataset_path, split="sealed_test")


def test_structured_run_writes_json_and_csv(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    predictions.write_text("{}\n", encoding="utf-8")
    summary, cases = write_evaluation_run(
        output_dir=tmp_path / "runs",
        run_id="experiment-1",
        stage="stt",
        dataset_path=dataset,
        predictions_path=predictions,
        split="development",
        metrics={"wer_micro": 0.1},
        details=[{"case_id": "a", "nested": {"x": 1}}],
        system_id="fixture",
    )
    envelope = json.loads(summary.read_text(encoding="utf-8"))
    assert envelope["evaluator_version"] == "voice-rag-eval-v1"
    assert envelope["dataset"]["sha256"]
    assert "case_id" in cases.read_text(encoding="utf-8-sig")
