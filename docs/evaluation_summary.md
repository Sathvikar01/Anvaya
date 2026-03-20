# Evaluation summary

This document separates measured development/sealed-test results, formal results, and smoke tests.

## Measured retrieval results

### Embedding comparison — development, 1,000 queries

| Model | Recall@10 | MRR@10 | nDCG@10 | Query embedding P50 |
| --- | ---: | ---: | ---: | ---: |
| BAAI/bge-m3 | 0.8578 | 0.5039 | 0.5873 | 9.1646 ms |
| multilingual-e5-base | 0.8421 | 0.5062 | 0.5857 | 4.9594 ms |
| gte-multilingual-base | 0.8217 | 0.4726 | 0.5544 | 5.4121 ms |
| jina-embeddings-v3 | 0.2953 | 0.1331 | 0.1699 | 55.9051 ms |
| IndicBERT-v3-4B | 0.3438 | 0.1624 | 0.2040 | 34.6878 ms |

### Chunking comparison — development, 1,000 queries

| Strategy | Recall@10 | MRR@10 | nDCG@10 | Retrieval P50 |
| --- | ---: | ---: | ---: | ---: |
| fixed-size | 0.8578 | 0.5039 | 0.5873 | 1.5932 ms |
| overlapping | 0.8524 | 0.4956 | 0.5794 | 1.7624 ms |
| sentence-based | 0.8625 | 0.5073 | 0.5908 | 1.5964 ms |
| semantic | 0.8469 | 0.4978 | 0.5795 | 2.3307 ms |
| parent-child | 0.8296 | 0.4807 | 0.5624 | 4.5994 ms |

### Index comparison — development, 1,000 queries

| Backend | Recall@10 | MRR@10 | nDCG@10 | Retrieval P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| FAISS FlatIP | 0.8625 | 0.5073 | 0.5908 | 1.5890 / 1.8279 ms |
| FAISS HNSW | 0.8625 | 0.5073 | 0.5908 | 0.3425 / 0.4395 ms |
| FAISS IVF-Flat | 0.8387 | 0.5010 | 0.5803 | 0.1752 / 0.2445 ms |
| Qdrant local exact | 0.8625 | 0.5073 | 0.5908 | 40.5374 / 53.7085 ms |
| Chroma local | 0.8625 | 0.5073 | 0.5908 | 2.6992 / 3.3179 ms |

### Sealed final retrieval test

| Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | P50 / P95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3464 | 0.6495 | 0.7899 | 0.8908 | 0.5371 | 0.6212 | 0.3876 / 0.5309 ms |

## Measured generation results — development

The comparisons used 12 cached gold-query retrieval cases. Codex performed a blinded qualitative
review; the 1–5 correctness, relevance, and faithfulness scores are not human ratings.

| Selected stage | Configuration | C / R / F | Mean quality | P50 / P70 / P95 / P100 |
| --- | --- | ---: | ---: | ---: |
| Model | sarvam-105b | 4.667 / 4.583 / 5.000 | 4.750 | 1420 / 1591 / 32108 / 32232 ms |
| Top-K | 10 | 4.667 / 4.667 / 5.000 | 4.778 | 1384 / 1557 / 2327 / 3104 ms |
| Prompt | strict_context_only | 4.667 / 4.667 / 5.000 | 4.778 | 1580 / 1966 / 7009 / 11849 ms |

The selected prompt had 12/12 successful calls, 100% schema validity, 100% grounded citation
validity, and zero serious grounding failures in this development comparison.

## Guardrails — DEVELOPMENT evaluation

The 24 curated development cases achieved 24/24 route correctness, 0% false refusal, and 0% false
answer. Guardrail-only P50/P70/P95/P100 latency was 0.0198/0.0217/0.0344/0.0694 ms. These cases
were used in threshold selection and are not a formal holdout or real-voice result.

## Formal Voice E2E

**PENDING — 0/24 real recordings are available.**

No formal completion, route accuracy, answer quality, citation validity, refusal correctness, WER,
retrieval degradation, latency, under-200-ms rate, or failure rate is reported. Pending CSV cells
remain blank by design.

## Smoke tests

The integration protocol has 10/10 passing structured checks for empty audio, corrupted audio, STT
provider failure, off-topic input, unsafe input, weak retrieval, generator timeout, malformed
generation, invalid citation, and internal component error. A cached text-path replay also passes.
These timings and outcomes are labeled `SMOKE_TEST` and excluded from formal tables.

## Latency limitation

The measured Sarvam generation P50 is approximately 1580 ms. Generation alone exceeds the
challenge's complete-pipeline target of less than 200 ms, so the current frozen stack cannot meet
that target. This limitation is not obscured by excluding slow calls or substituting cached timing.
