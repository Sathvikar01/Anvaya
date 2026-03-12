---
title: Anvaya — Grounded Voice Intelligence
emoji: 🪷
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
short_description: Multilingual grounded voice and text assistant
python_version: "3.12"
startup_duration_timeout: 1h
---

# Canonical deployment target

Hugging Face Space: https://sathvik0101-avyaya-voice-intelligence.hf.space/

# Anvaya

Anvaya keeps the original HTML/CSS/JS interface and runs the existing FastAPI
voice/text workflow inside a Gradio Space. The pinned BGE-M3 model downloads
on first startup; the serving FAISS index (HNSW, 22,413 chunks — 15 language
partitions including an English partition built from the original MS MARCO
passages) and passages are included in the repository.

Set `SARVAM_API_KEY` as a Space secret for voice input.

## Pipeline (voice/text → grounded answer)

```
input → guardrails (unsafe/off-topic/filler) → BGE-M3 query embedding (~35 ms)
      → language-partitioned FAISS HNSW search — every query hits its own
        language partition; English has a dedicated `en` partition, and rare
        UI languages (kok/ks/sd/mai) pivot to their closest available Indic
        partition instead of leaking across scripts (~15 ms)
      → evidence-sufficiency gate + unicode trigram hybrid rerank
      → resident extractive fast tier (xlm-roberta-base-squad2-distilled,
        verbatim span + citation, <150 ms) — single-token junk spans are
        rejected by a structural guard
      → if no confident span: show the closest retrieved passage with its
        match strength (honest, never silent) instead of forcing slow
        generative text
      → grounding validation (schema, citation whitelist, answer–evidence
        overlap) before anything reaches the user
```

Every response carries `route`, `reason_code`, per-stage latencies, retrieved
evidence with citations, and the decision trace of every guardrail. The system
answers only from retrieved context and refuses honestly otherwise:
`INSUFFICIENT_CONTEXT` / `OFF_TOPIC` / `UNSAFE` are first-class outcomes.

## Measured latency — extractive-only serving (30-query mixed battery)

Every response completes inside the 200 ms budget. Grounded answers are
resident verbatim spans + citations; everything else shows the closest
retrieved passage with its match strength (honest, never silent). The
generative chain (local Gemma-3-1B-it GGUF + Groq fallback) remains in the
codebase and re-enables via `HH_RAG_FALLBACK=generative`.

Server-side end-to-end (`total_latency_ms`, excludes client network RTT):

| Percentile | All 30 queries |
| --- | --- |
| P50 | 95.9 ms |
| P70 | 110.2 ms |
| **P100** | **135.3 ms** |

Zero requests exceeded 200 ms (min 31.7 ms, max 135.3 ms). Voice requests
scope timing to the RAG pipeline (post-STT); network speech-to-text is
reported separately in `stage_latencies_ms.stt`.
Raw measurements: `results/latency_report.json`.

Answer coverage is precision-first: when retrieval evidence cannot support a
confident verbatim span, the system refuses rather than guessing - rule 6's
"knows when not to answer" is implemented literally.

## Resilience notes

- In default `fast_tier_only` mode generative construction is skipped entirely
  (no 800 MB GGUF load, no API clients) — boot is faster and request path has
  no dead code. Set `HH_RAG_FALLBACK=generative` to re-enable the chain: local
  Gemma-3-1B-it GGUF (GBNF grammar) → Groq fallback, with ZeroGPU circuit
  breaker, cgroup-bounded llama.cpp threads (83 s → ~2 s warm-up), and
  bounded HTTP clients so one slow provider cannot starve later requests.
- Every harness exception is logged with full context; clients receive only a
  structured `reason_code`.

## Space variables and secrets

| Variable | Default | Purpose |
| --- | --- | --- |
| `SARVAM_API_KEY` | secret, required | Sarvam STT authentication (voice) |
| `GROQ_API_KEY` | secret | Optional generative fallback (disabled in default extractive-only mode) |
| `HH_RAG_FALLBACK` | `fast_tier_only` (set by wrapper) | `fast_tier_only` = extractive spans + closest-passage fallback, no generative calls; `generative` re-enables the full chain |
| `HH_RAG_GENERATOR` | `resilient` | Ordered generation chain when fallback is `generative` |
| `HH_RAG_GENERATION_CHAIN` | `local,groq` | Tier order when generative is enabled |
| `HH_RAG_RESPONSE_CACHE` | `1` (set by wrapper) | LRU+TTL cache for repeated identical queries; hits are labeled `cache_hit` in metadata |
| `HH_RAG_FAST_TIER_THRESHOLD` | `0.75` (wrapper) / `0.78` (Space var) | Global extractive confidence cutoff |
| `HH_RAG_FAST_TIER_THRESHOLDS` | `{"en-IN": 0.88}` (Space var) | Per-language overrides |
| `HH_RAG_GGUF_REPOSITORY` | `ggml-org/gemma-3-1b-it-GGUF` | Local SLM GGUF repository (used when generative is enabled) |
| `HH_RAG_GGUF_FILENAME` | `gemma-3-1b-it-Q4_K_M.gguf` | Local SLM GGUF file |
| `HH_RAG_LATENCY_TARGET_MS` | `200` | Target for `metadata.latency_budget` flagging (honest, never enforced by refusal) |
| `HH_RAG_MAX_LATENCY_MS` | off | Strict deadline guard — set to `200` to refuse instead of answering over budget |

The demo runs extractive-only by default for the strict <200 ms guarantee. Set `HH_RAG_FALLBACK=generative` to re-enable the full Gemma/Groq chain. Raw measurements: `results/latency_report.json`.

---

## Original GitHub Documentation (Preserved)

The following sections are the original GitHub documentation retained for research reproducibility. The Space deployment above is the canonical serving stack.

## Architecture

```text
Voice
→ Sarvam Saaras v3
→ Guardrails
→ BGE-M3
→ FAISS HNSW
→ Evidence gate
→ Groq openai/gpt-oss-20b
→ Grounding validation
```

The selected stack uses fixed 128-word chunks, normalized BGE-M3 embeddings,
FAISS HNSW (`M=32`, `efConstruction=200`, `efSearch=128`), and Top-10 retrieval with the
`strict_context_only` generation prompt. Answers require supporting retrieved evidence and
valid citations; otherwise the deterministic route is `INSUFFICIENT_CONTEXT`. The deterministic routes are `ANSWER`,
`INSUFFICIENT_CONTEXT`, `OFF_TOPIC`, `UNSAFE`, `STT_FAILURE`, and `SYSTEM_ERROR`.

The native harness is the default orchestrator. Install the optional `orchestration` extra and
set `HH_RAG_ORCHESTRATOR=langgraph` to run the same frozen pipeline through LangGraph. The graph
does not change retrieval, generation, or KB-only grounding behavior.

The FastAPI route is intentionally thin:

```text
POST /api/query/audio → VoiceRAGHarness.handle_audio() → GuardrailResponse.to_dict()
```

The browser shows only stages actually reached by the harness, measured request/stage latency,
retrieved passages and scores, and model-cited evidence IDs. See [architecture](docs/architecture.md)
for the complete runtime and trust boundaries.

## Run locally

Prerequisites: Python 3.11–3.14, the frozen local artifacts referenced by
`results/final_retriever_config.json`, Sarvam credentials for STT, and a Groq API key for answer generation.

```powershell
python -m pip install -e ".[app,web]"
Copy-Item .env.example .env
# Edit .env and set SARVAM_API_KEY and GROQ_API_KEY. Never commit this file.
python -m uvicorn hh_goa_rag.web:app --host 0.0.0.0 --port 8000 --workers 1
```

Open `http://localhost:8000`. Microphone capture works on localhost or an HTTPS deployment.
The health check is `GET /health`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SARVAM_API_KEY` | required | Sarvam STT authentication |
| `GROQ_API_KEY` | required | Groq answer-generation authentication (`openai/gpt-oss-20b`) |
| `HH_RAG_DEVICE` | `auto` | `auto`, `cpu`, or a CUDA device |
| `HH_RAG_ENV_FILE` | `.env` | Local dotenv path |
| `HH_RAG_RETRIEVER_CONFIG` | `results/final_retriever_config.json` | Frozen artifact manifest |

The service validates the key, config, BGE-M3 cache, FAISS HNSW index, and chunk mapping before marking
`/health` ready. It uses one worker because the model and progress registry must not be duplicated.

## Docker

The image intentionally includes the exact frozen model, index, and chunk artifacts (about 2.4 GB
before Python/runtime layers).

```powershell
docker build -t anvaya-voice-rag:phase7 .
docker run --rm --env-file .env -p 8000:8000 anvaya-voice-rag:phase7
```

Or use `docker compose up --build`. The container runs as a non-root user, has an application
health check, is offline with respect to Hugging Face model resolution, and never copies `.env`.

## Retrieval ablations — measured

The reproducible multilingual protocol is documented in
[`docs/retrieval_ablation_protocol.md`](docs/retrieval_ablation_protocol.md). The current
configuration first downloads and verifies the complete MSMARCO-XI parquet inventory (all
available Indic languages), then creates balanced development and sealed-test artifacts. It
reports both overall and per-language quality plus embedding, index-build, and retrieval
latency. Winners are selected only on development; the validation artifact is reserved for the
final locked run.

The table below preserves the original development ablation record. The current serving
configuration is the BGE-M3/HNSW stack described above.

All development ablations used 1,000 fixed queries and parent-level qrels. The sealed test was run
once on the selected stack.

| Stage / selected candidate | Recall@10 | MRR@10 | nDCG@10 | Retrieval P50 |
| --- | ---: | ---: | ---: | ---: |
| Embedding: BAAI/bge-m3 | 0.8578 | 0.5039 | 0.5873 | 1.6711 ms |
| Chunking: sentence-based | 0.8625 | 0.5073 | 0.5908 | 1.5964 ms |
| Index: FAISS HNSW | 0.8625 | 0.5073 | 0.5908 | 0.3425 ms |
| Sealed final test | 0.8908 | 0.5371 | 0.6212 | 0.3876 ms |

BGE-M3 won the quality-first embedding comparison. Sentence packing produced the highest measured
development nDCG@10. FAISS HNSW matched exact-search quality and won the latency tie-breaker; faster
IVF-Flat reduced Recall@10 to 0.8387. Full tables are summarized in
[evaluation summary](docs/evaluation_summary.md).

## Generation ablations — measured

The experiments reused one cached gold-query retrieval snapshot, so they could not change STT or
retrieval. The qualitative review was performed blinded by Codex, not human judges.

| Comparison | Selected | Correctness | Relevance | Faithfulness | P50 / P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Model | `sarvam-105b` | 4.667 | 4.583 | 5.000 | 1420 / 32108 ms |
| Top-K | `10` | 4.667 | 4.667 | 5.000 | 1384 / 2327 ms |
| Prompt | `strict_context_only` | 4.667 | 4.667 | 5.000 | 1580 / 7009 ms |

The selected prompt recorded 100% schema validity and grounded citation validity over its 12
measured cases, with no serious grounding failure. This small development comparison is not a
formal production-quality estimate.

## Guardrail evaluation — DEVELOPMENT

The 24-case development set contains 12 answerable, 4 insufficient-evidence, 4 off-topic, and 4
unsafe cases. It selected the already-frozen Top-1 threshold of 0.67 with the fixed consistency
rescue.

| Metric | DEVELOPMENT result |
| --- | ---: |
| Routing accuracy | 24/24 (100%) |
| False refusal rate | 0% |
| False answer rate | 0% |
| Guardrail P50 / P70 / P95 / P100 | 0.0198 / 0.0217 / 0.0344 / 0.0694 ms |

These are threshold-selection development results, not formal Voice E2E results.

## Voice E2E

**Formal Voice E2E evaluation: PENDING**

The manifest currently contains **0/24 real recordings**. Result CSVs contain pending rows with
blank metrics; no synthetic audio, cached latency, or smoke timing is promoted into formal tables.
The separately labeled robustness smoke suite currently passes 10/10 structured failure checks.

### Exact benchmark commands

Install evaluation and development dependencies first:

```powershell
python -m pip install -e ".[experiment,stt,generation,web,dev]"
```

1. List devices and record all 24 real-human samples. Replace device `5` and speaker metadata with
   the actual setup.

```powershell
python eval/record_audio.py --list-devices
python eval/record_audio.py --all-pending --speaker-id speaker-01 --device 5
```

2. Run Sarvam REST and streaming STT evaluation plus frozen-retriever degradation analysis.

```powershell
python eval/evaluate_stt.py --manifest eval/stt_manifest.jsonl --dataset eval/eval_dataset.jsonl --run-id sarvam-real-001 --transport both --device auto --env-file .env
```

3. Run the formal 24-sample Voice-RAG E2E evaluation.

```powershell
python eval/evaluate_e2e.py --manifest eval/stt_manifest.jsonl --dataset eval/eval_dataset.jsonl --output-dir results --env-file .env --device auto
```

4. Regenerate final result tables by rerunning steps 2 and 3 after all manifest rows are `ready`.
   They write `results/stt_evaluation.csv`, `results/stt_retrieval_impact.csv`, and all
   `results/e2e_*.csv`/`results/e2e_recommendation.md` artifacts. Missing recordings cause explicit
   pending output; they are never scored.

`eval/record_audio.py` and `eval/evaluate_e2e.py` remain the authoritative recording and formal
benchmark entry points.

## Latency limitation

**The measured Sarvam generation P50 is ~1580 ms, therefore the current stack cannot satisfy the
challenge's <200 ms complete-pipeline target.**

This is reported without adjustment: generation alone is slower than the end-to-end target, before
adding STT, embedding, search, or guardrails.

## Verification and submission material

```powershell
python -m pytest
ruff check .
```

- [Demo script](docs/demo_script.md)
- [Architecture](docs/architecture.md)
- [Evaluation summary](docs/evaluation_summary.md)
- [Submission checklist](docs/submission_checklist.md)

No API key or real evaluation recording is committed. Generated model/data artifacts are
repository-local and ignored by Git; the Docker build consumes the exact local frozen artifacts.