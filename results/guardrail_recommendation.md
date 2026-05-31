# Guardrail and routing recommendation

## Frozen stack

Sarvam STT → BGE-M3 → sentence chunks (128 words) → FAISS HNSW → sarvam-105b →
Top-10 → `strict_context_only`.

No frozen STT, retrieval, chunking, index, generation-model, Top-K, or prompt setting was changed.
The sealed final Voice-RAG test was not read or executed. All threshold selection used the 24-case
development set (12 answerable, 4 insufficient-evidence, 4 off-topic, 4 unsafe).

## Final development metrics

- Overall routing accuracy: 100.0%
- Answerable acceptance rate: 100.0%
- Insufficient-context detection rate: 100.0%
- Off-topic detection rate: 100.0%
- Unsafe detection rate: 100.0%
- False refusal rate: 0.0%
- False answer rate: 0.0%
- Guardrail latency P50/P70/P95/P100: 0.0198 / 0.0217 / 0.0344 / 0.0694 ms

Guardrail latency is deterministic routing/validation overhead only; it excludes the already
measured frozen STT, embedding, retrieval, and generation stages.

## Threshold selection

Tested top-score thresholds: 0.60, 0.62, 0.64, 0.65, 0.67, 0.70, 0.72, 0.74. Every candidate retained the fixed consistency rescue:
top score ≥ 0.64 plus either Top-1−Top-2 gap ≤
0.005 or Top-1−Top-5 spread ≥ 0.12.

The selected threshold is **0.67**, the smallest tested value with zero false
answers and zero false refusals on development data. False answers carry weight 5 and benign false
refusals weight 1 in `weighted_error_cost`; this makes unsafe/grounding leakage more costly than a
conservative refusal. Qrel presence is reported for evaluation only and is not a runtime signal.

## Exact routing logic

1. Failed, empty, invalid, overlong, or extremely low-information transcripts → `STT_FAILURE`.
2. Deterministic credential-theft, weapon, physical-harm, or hate patterns → `UNSAFE`.
3. Deterministic creative-writing, live-score, transaction, or recipe patterns → `OFF_TOPIC`.
4. Otherwise run the frozen BGE-M3/FAISS Top-10 retriever.
5. Evidence is sufficient when Top-1 ≥ 0.67, or when the fixed consistency rescue above succeeds;
   otherwise → `INSUFFICIENT_CONTEXT` without generation.
6. Generate only after evidence passes, using frozen `sarvam-105b`/Top-10/`strict_context_only`.
7. A valid `INSUFFICIENT_CONTEXT` generation is respected. An answer must be schema-valid,
   non-empty, cite at least one supplied parent ID, and cite no unknown ID.
8. Malformed output, invalid refusal shape, missing/unknown citations, provider errors, or component
   exceptions fail closed to `SYSTEM_ERROR`.

Rules live in `src/hh_goa_rag/guardrails/input.py`, retrieval thresholds in
`src/hh_goa_rag/guardrails/retrieval.py`, grounding validation in
`src/hh_goa_rag/guardrails/grounding.py`, and orchestration in `src/hh_goa_rag/harness.py`.

## Confusion matrix

| Expected \ Predicted | ANSWER | INSUFFICIENT_CONTEXT | OFF_TOPIC | UNSAFE | STT_FAILURE | SYSTEM_ERROR |
|---|---:|---:|---:|---:|---:|---:|
| ANSWER | 12 | 0 | 0 | 0 | 0 | 0 |
| INSUFFICIENT_CONTEXT | 0 | 4 | 0 | 0 | 0 | 0 |
| OFF_TOPIC | 0 | 0 | 4 | 0 | 0 | 0 |
| UNSAFE | 0 | 0 | 0 | 4 | 0 | 0 |
| STT_FAILURE | 0 | 0 | 0 | 0 | 0 | 0 |
| SYSTEM_ERROR | 0 | 0 | 0 | 0 | 0 | 0 |

This is a small curated development evaluation. Perfect development routing is not evidence of
production-perfect safety; broader adversarial and multilingual policy evaluation remains needed.
