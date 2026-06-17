# Phase 9 — Local answer-generation ablation

## Decision

**YES, at P50 only.** A resident high-confidence extractive tier can reduce measured estimated post-EOS P50 below 200 ms without lowering accepted-answer quality, while the untouched Sarvam-105B path remains necessary for the 25% fallback tail. P95 and P100 remain far above 200 ms.

All local quality scores below are **Codex qualitative evaluation, not ground truth**. The questions, retrieved evidence, and 12 development case IDs were unchanged. The existing Sarvam-105B strict-context Top-10 run is the frozen quality baseline.

## Hardware and runtime audit

- Machine: Dell Pro Max Tower T2 FCT2250, Windows 11.
- CPU: Intel Core Ultra 9 285, 24 physical / 24 logical cores.
- RAM: 127.46 GiB.
- GPU: NVIDIA GeForce RTX 5090, 31.84 GiB VRAM, compute capability 12.0.
- CUDA: available through PyTorch 2.11.0+cu128 (CUDA build 12.8); NVIDIA driver reports CUDA 13.2 support. `nvcc` is not installed.
- Installed usable runtimes: Transformers 5.5.4 + PyTorch CUDA; bitsandbytes 0.49.2; Ollama is installed but has no resident models; ONNX Runtime 1.29.0 is installed with CPU/Azure providers only. llama.cpp and vLLM are not installed.
- Model discovery used accessible official model cards and then `local_files_only=True` for measurement: `deepset/xlm-roberta-base-squad2-distilled`, `Qwen/Qwen2.5-0.5B-Instruct`, and `Qwen/Qwen3-0.6B`.

Native FP16/BF16 was selected. The three models individually occupy roughly 1–2 GiB of weights, so 4-bit quantization was not useful for capacity on 32 GiB VRAM and would introduce another quality/runtime variable. Top-3 context, greedy decoding, and a hard 64-token ceiling were used for both generators.

## Protocol

- Exact frozen input: `cache/generation/gold_contexts_top10.jsonl`, 12 answerable development cases.
- Local context: unchanged Top-3 prefix of each case's frozen Top-10 evidence.
- Resident lifecycle: load tokenizer/model once, first inference recorded as cold, one additional warm-up, then five measured repetitions per case. Warm latency is each case's median across those five repetitions; percentiles are over the 12 per-case medians.
- Extractive output is a verbatim span. Parent citation ID is retained from the selected evidence item. Confidence is the sigmoid of best-span logit margin over the no-answer score.
- Generator output must finish before the token ceiling, contain only known evidence labels, introduce no novel numbers, and pass deterministic lexical-grounding validation. Invalid output abstains/falls back.
- Fixed measured pre-answer stages: STT 37.00 ms + embedding 8.50 ms + FAISS 0.40 ms + guardrails 0.02 ms = **45.92 ms**. Therefore the answer-stage P50 budget is **154.08 ms**.

## Final comparison

| Answer engine | Coverage | C | R | F | Answer P50 (ms) | Estimated E2E/Post-EOS P50 (ms) | <200ms |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Sarvam-105B baseline | 100.0% | 4.67 | 4.67 | 5.00 | 1579.88 | 1625.80 | NO |
| Extractive QA (selected) | 75.0% | 4.89 | 5.00 | 5.00 | 12.42 | 58.34 | YES |
| Qwen2.5-0.5B-Instruct | 0.0% | — | — | — | 121.02 | 166.94 | YES |
| Qwen3-0.6B | 58.3% | 3.43 | 4.00 | 5.00 | 1360.99 | 1406.91 | NO |
| Hybrid (selected) | 100.0% | 4.67 | 4.83 | 5.00 | 12.97 | 58.89 | YES |

Coverage means a validated local answer, except baseline/hybrid coverage which includes Sarvam fallback. C/R/F are means over returned answers. A latency YES does not make an engine quality-eligible: Qwen2.5 has zero validated coverage, and Qwen3 has serious correctness failures.

## Cold versus warm

| Engine | Load (ms) | First inference (ms) | Load + first (ms) | Warm-up inference (ms) |
|---|---:|---:|---:|---:|
| XLM-R distilled QA | 2349.17 | 221.34 | 2570.50 | 13.42 |
| Qwen2.5-0.5B | 1078.94 | 1390.41 | 2469.35 | 1110.45 |
| Qwen3-0.6B | 1201.25 | 1315.07 | 2516.32 | 1294.97 |

Only warmed measurements inform the production decision. The local components must be created at application startup, retained, and warmed before traffic.

## Extractive QA

The selected 0.98 threshold plus a deterministic truncated-currency-span integrity check accepts 9/12 cases (75%). Its accepted-answer C/R/F is 4.89/5.00/5.00; citation validity and verbatim-span rates are both 100%. The rejected cases are the incorrect `$716` distractor, the incomplete notary-fee span, and a malformed gutter-cost range. This is deliberately conservative.

## Tiny local generators

- Qwen2.5-0.5B-Instruct: estimated post-EOS P50 166.94 ms (**YES**), but 0/12 outputs survive citation/completion validation. It is rejected for zero answer coverage.
- Qwen3-0.6B: estimated post-EOS P50 1406.91 ms (**NO**), 7/12 validated coverage, and C/R/F 3.43/4.00/5.00. It is rejected for latency and three serious correctness failures; five other outputs hit the 64-token ceiling and were rejected.

Neither tested generator is a safe Sarvam replacement.

## Hybrid routing

The experimental three-tier router is deterministic: extractive confidence ≥0.98 and span-integrity pass → Tier 1; otherwise run the local generator and accept only completed/cited/numerically grounded output → Tier 2; otherwise → untouched Sarvam-105B Tier 3. Because neither generator passed model-level quality qualification, the selected safe row disables Tier 2 and routes extractive abstentions directly to Tier 3. The measured deterministic branch/span-check P50 overhead is 0.0010 ms and is included in hybrid latency.

| Hybrid | Tier 1 | Tier 2 | Tier 3 | Answer P50 | P70 | P95 | P100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| extractive_0.98_then_Qwen2.5-0.5B-Instruct_then_sarvam_105b | 75.0% | 0.0% | 25.0% | 12.97 | 13.52 | 6414.82 | 11979.47 |
| extractive_0.98_then_Qwen3-0.6B_then_sarvam_105b | 75.0% | 0.0% | 25.0% | 12.97 | 13.52 | 7821.06 | 13401.78 |
| extractive_0.98_then_sarvam_105b_tier2_disabled_after_ablation | 75.0% | 0.0% | 25.0% | 12.97 | 13.52 | 6289.94 | 11862.68 |

For both tested generators, the three cases reaching Tier 2 were rejected, so Tier 2 handles 0%. In the selected safe router, Tier-1 C/R/F is 4.89/5.00/5.00; Tier 2 is disabled; Tier-3 C/R/F is 4.00/4.33/5.00. Overall C/R/F is 4.67/4.83/5.00, with 100% valid citations. Estimated post-EOS P50/P70/P95/P100 is 58.89/59.44/6335.86/11908.60 ms.

## Recommendation

Deploy the resident XLM-R distilled extractive engine as a 0.98-confidence, span-integrity-checked fast tier and fall straight through to the unchanged Sarvam-105B baseline when it abstains; do not deploy either tiny generator. This measured architecture gives 75% local handling, 100% overall coverage, preserved qualitative quality, and sub-200-ms estimated post-EOS P50, while Sarvam generation remains the measured P95/P100 bottleneck.
