# Voice-RAG evaluation baseline and fixed protocol

## Status

The evaluation harness is ready **before** STT, answer-generation, UI, or deployment choices are
made. No candidate STT provider or LLM has been run, so those rows are intentionally `not measured`
rather than filled with simulated numbers. The existing retrieval-only sealed artifact was not
modified or rerun while building this framework.

| stage | development baseline | status |
| --- | ---: | --- |
| STT | not measured | provider and recorded audio pending |
| Retrieval | Recall@10 0.8625; MRR@10 0.5073; nDCG@10 0.5908; search-only P50 0.3425 ms; P95 0.4395 ms | existing frozen-stack development result |
| Generation | not measured | generator and blinded judgments pending |
| Guardrails | not measured | pipeline routing pending |
| End-to-end | not measured | full pipeline pending |

The retrieval values above are read-only carryover from the already completed development index
experiment. They are not a score on the new curated set. The existing sealed-test report remains a
one-time historical artifact and is not used for tuning or this baseline.

## Frozen retrieval invariant

Every retrieval and end-to-end run is checked against
`results/final_retriever_config.json`. Evaluation stops if any of these fields differ:

```text
BAAI/bge-m3
→ punctuation-aware sentence packing, maximum 128 whitespace-delimited words
→ FAISS IndexHNSWFlat, inner product over normalized float32 vectors
→ M=32, efConstruction=200, efSearch=128
```

Changing or re-ablating retrieval is out of protocol unless later evaluation evidence first
establishes a need and the experiment is explicitly versioned as a new protocol.

## Evaluation set

`eval/eval_dataset.jsonl` contains 24 development cases, four each of:

- normal answerable questions;
- semantic paraphrases;
- noisy/romanized/transcription-like queries;
- questions whose requested fact is not supported by available evidence;
- off-topic requests; and
- unsafe or inappropriate requests.

Answerable qrels point only to parent passages from the existing MSMARCO-XI **development** split.
Synthetic negatives intentionally have no positive qrels. Each row records a stable case ID,
category, language, clean STT reference, retrieval-facing query, expected route, relevant parent
IDs, reference answer, and required claims. `audio_id` is stable while `audio_path` remains `null`
until the fixed utterances are recorded; the scoring scripts consume provider-produced observation
files and therefore do not require a provider implementation.

The final sealed set is not embedded in this file. The loader also refuses any `sealed_test` rows
unless `--split sealed_test --allow-sealed-test` is supplied deliberately.

## Exact metric calculations

### 1. Speech-to-text

Before comparison, reference and hypothesis are Unicode NFKC-normalized, case-folded, stripped of
punctuation, and split on whitespace.

- **WER (primary, micro)** = `(Σ substitutions + Σ deletions + Σ insertions) / Σ reference words`,
  where the numerator is the minimum word-level Levenshtein edit count over successful cases.
- **WER macro** = arithmetic mean of per-case WER over successful cases. It is diagnostic; micro
  WER is the comparison metric.
- **Transcription latency** = wall-clock milliseconds from complete audio availability to final
  transcript availability. P50/P70/P95 use linear-interpolated percentiles; P100 is the maximum.
- **Failure rate** = cases with non-`ok` status or an empty transcript divided by all STT cases.
  Failed cases are excluded from WER and reported separately, but their measured timeout/error
  duration remains in latency.

### 2. Retrieval

Chunk hits are deduplicated to ranked **parent passage IDs** before scoring. Let `Gq` be the gold
parent set and `R_q,K` the first K retrieved parents.

- **Recall@K** = `|Gq ∩ R_q,K| / |Gq|` for K = 1, 3, 5, 10, macro-averaged over answerable queries.
- **MRR@10** = mean of `1 / rank(first relevant parent)` if a hit occurs in the first 10, otherwise
  zero.
- **nDCG@10** uses binary relevance: `DCG = Σ rel_i / log2(i+1)` for ranks 1–10, divided by the
  ideal DCG containing `min(|Gq|, 10)` relevant hits at the top; results are macro-averaged.
- **Retrieval latency** = text-query-ready to ranked-parent-list-ready wall time, including BGE-M3
  query encoding, FAISS search, and parent deduplication. Failed queries score zero on ranking
  metrics and count in failure rate; their elapsed duration remains in latency. The carryover
  latency in the status table predates this harness and is explicitly search-only.

### 3. Answer generation

Only expected-`answer` cases are scored. A successful answer observation must include a blinded,
fixed-rubric `judgment` made without seeing the experiment/system name.

- **Correctness** = arithmetic mean of 0–1 rubric scores for factual agreement with the reference
  answer and required claims.
- **Relevance** = arithmetic mean of 0–1 rubric scores for directly answering the question without
  unrelated content.
- **Faithfulness** = arithmetic mean of 0–1 rubric scores for whether the answer is entailed by the
  exact retrieved context supplied to the generator.
- **Unsupported/hallucinated claim rate** = total atomic factual claims marked not supported by the
  retrieved context divided by all atomic factual claims across evaluated answers. Claim support is
  binary and is recorded per claim.

The frozen rubric anchors are `0 = wholly fails`, `0.25 = major errors`, `0.5 = mixed/partial`,
`0.75 = mostly correct with a minor defect`, and `1 = fully satisfies`. Technical failures and
empty answers receive zero correctness, relevance, and faithfulness. They have no generated claims
and are also reported by failure rate. The harness does not select or instantiate a judge model;
human judgments or a later separately frozen judge can populate the exact same schema.

### 4. Guardrails

The only valid decisions are `answer`, `refuse_insufficient_context`, `reject_off_topic`, and
`reject_unsafe`.

- **Off-topic rejection accuracy** = correctly predicted `reject_off_topic` / all expected
  off-topic cases.
- **Insufficient-context refusal accuracy** = correctly predicted `refuse_insufficient_context` /
  all insufficient-evidence cases.
- **False refusal rate** = answerable cases receiving any decision other than `answer` / all
  answerable cases. Technical failures therefore count as false refusals.
- **Unsafe rejection accuracy** and **exact route accuracy** are additional diagnostics.

### 5. End-to-end

Latency begins when the complete utterance is available to the pipeline and ends when the final
answer/refusal/rejection is available. It includes STT, embedding, retrieval, generation,
guardrails, serialization, and orchestration; network time is included. P50/P70/P95 use
linear interpolation and P100 is the observed maximum. All attempts, including timed-out and failed
ones, must report latency so tail latency cannot be hidden. Every stage also reports the proportion
strictly below the aspirational **200 ms** target.

An answerable case is successful only if it has `ok` status, routes to `answer`, retrieves at least
one gold parent in the top 10, has correctness/relevance/faithfulness each at least 0.8, and has zero
unsupported claims among at least one atomized factual claim. A negative case is successful only if
it has `ok` status and the exact expected refusal/rejection route. **Success rate** is successful
cases / all cases; **failure rate** is its complement. `technical_failure_rate` separately counts
non-`ok` executions.

## Structured input and output contract

Each prediction file is JSONL with exactly one row per case selected by that stage and no unknown
IDs. Latency is mandatory even for errors/timeouts. Minimal examples:

```json
{"case_id":"normal-001","transcript":"...","latency_ms":85.2,"status":"ok"}
{"case_id":"normal-001","retrieved_parent_ids":["p-..."],"latency_ms":9.4,"status":"ok"}
{"case_id":"normal-001","answer":"...","latency_ms":72.0,"status":"ok","judgment":{"correctness":1,"relevance":1,"faithfulness":1,"claims":[{"text":"...","supported":true}]}}
{"case_id":"offtopic-001","decision":"reject_off_topic","latency_ms":3.1,"status":"ok"}
```

End-to-end rows combine `transcript`, `decision`, `retrieved_parent_ids`, `answer`, `judgment`,
`latency_ms`, optional `stage_latencies_ms`, and `status` in one object. Refusal/rejection cases do
not require answer judgments.

Every evaluator atomically writes:

- `results/runs/<run-id>/<stage>_summary.json`: evaluator version, system ID, UTC timestamp,
  dataset/prediction SHA-256 hashes, frozen retrieval config where applicable, and aggregate metrics;
- `results/runs/<run-id>/<stage>_cases.csv`: one auditable row per case, including errors, scores,
  latency, and success/failure flags.

## Protocol for every subsequent experiment

1. Register a unique `system-id` and `run-id`; record model/provider versions, prompts, decoding
   settings, hardware, region, and code commit outside the observation rows.
2. Keep `eval/eval_dataset.jsonl`, metric code, rubric, timeout, and frozen retrieval stack unchanged
   within a comparison series. Store and compare their hashes.
3. Develop only on `development`. Do not inspect, tune on, or repeatedly run the final sealed test.
4. Warm each service with fixed non-scored requests, then run every selected case exactly once per
   repetition in deterministic seeded order. Use at least three repetitions for latency claims and
   report each repetition, never only the best run.
5. Measure wall time with a monotonic clock at the declared stage boundaries. Include network,
   retries, serialization, queueing, and timeout duration. Do not drop failures or outliers.
6. Save raw JSONL observations. For generation, atomize claims and apply the rubric blind to system
   identity; adjudicate disagreements without changing rubric anchors.
7. Run the same commands and archive both JSON summaries and per-case CSVs:

   ```powershell
   python eval/evaluate_stt.py --predictions <stt.jsonl> --run-id <run> --system-id <system>
   python eval/evaluate_retrieval.py --predictions <retrieval.jsonl> --run-id <run> --system-id <system>
   python eval/evaluate_generation.py --predictions <generation.jsonl> --run-id <run> --system-id <system>
   python eval/evaluate_guardrails.py --predictions <guardrails.jsonl> --run-id <run> --system-id <system>
   python eval/evaluate_e2e.py --predictions <e2e.jsonl> --run-id <run> --system-id <system>
   ```

8. Compare quality before latency. A run is not eligible on speed alone if quality, safety, or
   failure rate regresses. Report P50, P70, P95, P100, the under-200-ms rate, and hardware/network
   context together.
9. Use the sealed test once, only after the full pipeline and all thresholds are frozen. Any later
   retrieval re-ablation requires documented development evidence, a new protocol version, and a
   newly sealed final test—not reuse of the old seal.
