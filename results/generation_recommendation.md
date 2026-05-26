# Generation recommendation

## Final selection

**sarvam-105b → Top-10 → strict_context_only → C/R/F 4.67/4.67/5.00 →
1580/1966/7009/11849 ms**

At the user's direction, Codex performed the blinded qualitative review; these are not human scores.
Model/configuration identity stayed sealed until every correctness, relevance, and faithfulness
score was persisted. Automated diagnostic heuristics were not substituted for these qualitative
scores. A faithfulness score of 1 was defined as a serious grounding failure and made a
configuration ineligible. Latency was used only as the model tie-breaker.

Top-K was evaluated only with `sarvam-105b` at K = 1, 3, 5, and 10. The selected value is the
smallest K achieving the highest observed mean quality without a serious grounding failure.
The prompt comparison then held that model and Top-K fixed.

## Ablation summary

### Model

| Model | Correctness | Relevance | Faithfulness | Mean quality | P50 / P95 latency | Selected |
|---|---:|---:|---:|---:|---:|---:|
| sarvam-105b | 4.667 | 4.583 | 5.000 | 4.750 | 1420 / 32108 ms | True |
| sarvam-105b-conversations | 4.583 | 4.667 | 5.000 | 4.750 | 5146 / 6662 ms | False |

### Top-K

| K | Correct. | Relevance | Faithful. | Mean quality | Citation valid. | P50 / P95 | Selected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.500 | 4.333 | 5.000 | 4.611 | 100.0% | 1092 / 32173 ms | False |
| 3 | 4.583 | 4.333 | 5.000 | 4.639 | 100.0% | 1440 / 22732 ms | False |
| 5 | 4.667 | 4.417 | 5.000 | 4.694 | 100.0% | 1502 / 9181 ms | False |
| 10 | 4.667 | 4.667 | 5.000 | 4.778 | 100.0% | 1384 / 2327 ms | True |

### Prompt

| Prompt | Correct. | Relevance | Faithful. | Mean quality | Tokens | P50 / P95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_only_refusal | 4.667 | 4.583 | 5.000 | 4.750 | 1900.2 | 1330 / 7373 ms | False |
| strict_context_only | 4.667 | 4.667 | 5.000 | 4.778 | 1890.2 | 1580 / 7009 ms | True |
| structured_evidence_ids | 4.667 | 4.583 | 5.000 | 4.750 | 1928.2 | 1629 / 4333 ms | False |

## Selected quality and performance

- Mean correctness: 4.667/5
- Mean relevance: 4.667/5
- Mean faithfulness: 5.000/5
- Serious grounding failures: 0
- Grounded citation validity: 100.0%
- Appropriate answer/refusal behavior: 100.0%
- Generation latency P50/P70/P95/P100: 1580/1966/7009/11849 ms
- Mean prompt/context tokens: 1890.2

All 12 Top-10 cases contained relevant evidence, so the prompt ablation had no missing-context case
on which to distinguish refusal behavior. All variants answered all 12 cases. In the Top-K ablation,
K=1 lacked a relevant parent for 4/12 cases and refused 0/4, yielding 66.7% appropriate context
behavior; this limitation is recorded rather than treated as a quality score.

## Latency target

The current Sarvam generation stack **does not satisfy the <200 ms complete-pipeline target**.
Its measured generation latency alone is P50 1579.9 ms and P100
11849.3 ms. Since the complete pipeline also includes STT and
retrieval, it cannot be faster than this measured generation stage under the evaluated stack.

## Scope

The frozen STT and retriever were not modified. The experiment reused the cached gold-query
retrieval snapshot. No guardrails, UI, or deployment work is included.
