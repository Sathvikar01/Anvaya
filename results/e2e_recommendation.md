# Complete Voice-RAG integration and benchmark

Status: **PENDING_REAL_RECORDINGS**

## Final frozen architecture

Audio → Sarvam Saaras v3 STT → deterministic input guardrail → BGE-M3 query embedding →
FAISS HNSW (`M=32`, `efConstruction=200`, `efSearch=128`) over sentence chunks capped at 128 words
→ frozen evidence guardrail → `sarvam-105b` / Top-10 / `strict_context_only` → deterministic
grounding validation → structured response.

Every response contains route, answer, retrieved IDs, citations, reason code, detailed decision
provenance, STT/embedding/vector-search/guardrail/generation timings, and total E2E latency.

## Measured

Recording availability was measured at **0/24**. There are no formal real-voice E2E quality or latency numbers.

The previously measured selected generation configuration has generation-only latency
P50/P70/P95/P100 of **1580/1966/7009/11849 ms**. Therefore the current stack cannot satisfy the
**<200 ms complete-pipeline requirement**: a complete request cannot be faster than its generation
stage. This is a component lower bound, not a fabricated E2E measurement.

## Development-only

Guardrail threshold selection used only the 24-case development set. It selected the already-frozen
Top-1 threshold 0.67 with the fixed consistency rescue and obtained 24/24 route correctness there.
These development results are not mixed into the real-voice E2E tables.

## Smoke tests

Robustness protocol checks passed **10/10**: empty audio, corrupted audio, STT provider
error, irrelevant query, unsafe query, weak retrieval, generator timeout, malformed generation,
invalid citation, and internal exception all returned graceful structured routes without crashes.
Their timings are recorded only as `SMOKE_TEST` rows in `e2e_failure_analysis.csv` and are excluded
from formal latency summaries.

The text-path integration replay is separately classified as smoke testing:

```json
{
  "status": "PASS_CACHED_PROTOCOL_REPLAY",
  "classification": "SMOKE_TEST",
  "audio": "NOT_AVAILABLE",
  "transcript": "यूनाइटेड किंगडम में कौन से चार देश शामिल हैं",
  "route": "ANSWER",
  "retrieved": [
    {
      "rank": 1,
      "parent_id": "p-4e30291ba19848bb9c88e141",
      "score": 0.7378749847412109
    },
    {
      "rank": 2,
      "parent_id": "p-26e56367c88ddcdfac5a9de9",
      "score": 0.6912648677825928
    },
    {
      "rank": 3,
      "parent_id": "p-a4c6c58ee9ff663c159413e1",
      "score": 0.6478303670883179
    },
    {
      "rank": 4,
      "parent_id": "p-7316bd825dd2d5d19f448980",
      "score": 0.6465238332748413
    },
    {
      "rank": 5,
      "parent_id": "p-b03ff03fe83117ea176960a5",
      "score": 0.6380659341812134
    },
    {
      "rank": 6,
      "parent_id": "p-5aa4e4ebb5ca1fafd5c53ac3",
      "score": 0.6110961437225342
    },
    {
      "rank": 7,
      "parent_id": "p-d776d36cdc1e83c1c6c37ab1",
      "score": 0.6075615882873535
    },
    {
      "rank": 8,
      "parent_id": "p-b784459440b13b7ec365f678",
      "score": 0.5978888273239136
    },
    {
      "rank": 9,
      "parent_id": "p-41898b5e8969290abe1513a8",
      "score": 0.5936483144760132
    },
    {
      "rank": 10,
      "parent_id": "p-d17d485cf7b80f98a362e64f",
      "score": 0.5893312692642212
    }
  ],
  "evidence_decision": {
    "sufficient": true,
    "reason_code": null,
    "top_score": 0.7378749847412109,
    "top_two_gap": 0.046610116958618164,
    "top_to_fifth_spread": 0.09980905055999756,
    "top_three_mean": 0.6923234065373739,
    "decision_rule": "top_score"
  },
  "answer": "यूनाइटेड किंगडम में चार देश शामिल हैं: इंग्लैंड, स्कॉटलैंड, वेल्स और उत्तरी आयरलैंड।",
  "citations": [
    "p-4e30291ba19848bb9c88e141"
  ],
  "grounding": {
    "valid": true,
    "route": "ANSWER",
    "reason_code": "ANSWER_GROUNDED"
  },
  "final_response": {
    "route": "ANSWER",
    "answer": "यूनाइटेड किंगडम में चार देश शामिल हैं: इंग्लैंड, स्कॉटलैंड, वेल्स और उत्तरी आयरलैंड।",
    "retrieved_ids": [
      "p-4e30291ba19848bb9c88e141",
      "p-26e56367c88ddcdfac5a9de9",
      "p-a4c6c58ee9ff663c159413e1",
      "p-7316bd825dd2d5d19f448980",
      "p-b03ff03fe83117ea176960a5",
      "p-5aa4e4ebb5ca1fafd5c53ac3",
      "p-d776d36cdc1e83c1c6c37ab1",
      "p-b784459440b13b7ec365f678",
      "p-41898b5e8969290abe1513a8",
      "p-d17d485cf7b80f98a362e64f"
    ],
    "citations": [
      "p-4e30291ba19848bb9c88e141"
    ],
    "reason_code": "ANSWER_GROUNDED",
    "stage_latencies_ms": {
      "stt": 0.0,
      "input_validation": 0.0249,
      "route_check": 0.0064,
      "embedding": 0.0004,
      "retrieval": 0.0005,
      "evidence_guardrail": 0.0034,
      "generation": 0.0007,
      "grounding_validation": 0.0151,
      "query_embedding": 0.0004,
      "vector_search": 0.0005,
      "guardrails": 0.0498,
      "total_end_to_end": 0.0698
    },
    "total_latency_ms": 0.0698,
    "transcript": "यूनाइटेड किंगडम में कौन से चार देश शामिल हैं",
    "metadata": {
      "decision_trace": [
        {
          "stage": "input_validation",
          "allow": true,
          "route": null,
          "reason_code": null
        },
        {
          "stage": "route_check",
          "allow": true,
          "route": null,
          "reason_code": null
        },
        {
          "stage": "evidence_guardrail",
          "allow": true,
          "route": null,
          "reason_code": null,
          "decision_rule": "top_score"
        },
        {
          "stage": "grounding_validation",
          "allow": true,
          "route": "ANSWER",
          "reason_code": "ANSWER_GROUNDED"
        }
      ],
      "retrieved": [
        {
          "rank": 1,
          "parent_id": "p-4e30291ba19848bb9c88e141",
          "score": 0.7378749847412109
        },
        {
          "rank": 2,
          "parent_id": "p-26e56367c88ddcdfac5a9de9",
          "score": 0.6912648677825928
        },
        {
          "rank": 3,
          "parent_id": "p-a4c6c58ee9ff663c159413e1",
          "score": 0.6478303670883179
        },
        {
          "rank": 4,
          "parent_id": "p-7316bd825dd2d5d19f448980",
          "score": 0.6465238332748413
        },
        {
          "rank": 5,
          "parent_id": "p-b03ff03fe83117ea176960a5",
          "score": 0.6380659341812134
        },
        {
          "rank": 6,
          "parent_id": "p-5aa4e4ebb5ca1fafd5c53ac3",
          "score": 0.6110961437225342
        },
        {
          "rank": 7,
          "parent_id": "p-d776d36cdc1e83c1c6c37ab1",
          "score": 0.6075615882873535
        },
        {
          "rank": 8,
          "parent_id": "p-b784459440b13b7ec365f678",
          "score": 0.5978888273239136
        },
        {
          "rank": 9,
          "parent_id": "p-41898b5e8969290abe1513a8",
          "score": 0.5936483144760132
        },
        {
          "rank": 10,
          "parent_id": "p-d17d485cf7b80f98a362e64f",
          "score": 0.5893312692642212
        }
      ],
      "evidence_decision": {
        "sufficient": true,
        "reason_code": null,
        "top_score": 0.7378749847412109,
        "top_two_gap": 0.046610116958618164,
        "top_to_fifth_spread": 0.09980905055999756,
        "top_three_mean": 0.6923234065373739,
        "decision_rule": "top_score"
      },
      "generation": {
        "provider_status": "ok",
        "answer_status": "ANSWER",
        "error_code": null
      },
      "grounding": {
        "valid": true,
        "route": "ANSWER",
        "reason_code": "ANSWER_GROUNDED"
      }
    }
  },
  "timing_note": "Cached protocol replay timing is not a formal benchmark"
}
```

## Pending

All real-voice completion, route, answer-quality, citation, refusal, WER, retrieval degradation, per-stage latency, latency contribution, under-200-ms, and failure-rate metrics are pending 24 real recordings. The concrete audio trace is also pending.

Recordings must be real human speech, marked `ready` in `eval/stt_manifest.jsonl`, and present at
their declared paths before rerunning `python eval/evaluate_e2e.py`. Missing recordings are never
scored, and synthetic or cached timings are never promoted into formal E2E metrics.
