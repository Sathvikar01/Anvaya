# Architecture

## Frozen data path

```text
Browser microphone or upload
  → browser resampling and PCM16 WAV encoding (16 kHz, mono)
  → POST /api/query/audio
  → VoiceRAGHarness.handle_audio
      → Sarvam Saaras v3 STT
      → transcript validation and deterministic safety/topic routing
      → BGE-M3 query embedding
      → FAISS HNSW parent retrieval, Top-10
      → deterministic evidence-sufficiency gate
      → Groq `openai/gpt-oss-20b` with low reasoning and strict_context_only
      → citation/schema grounding validation
  → structured GuardrailResponse
  → transcript, route, answer/refusal, evidence, citations, measured timings
```

The API route owns transport concerns only: bounded temporary upload storage, exact WAV validation,
thread offload, progress observation, and serialization. All routing and inference decisions remain
inside `VoiceRAGHarness` and its frozen components.

## Frozen configuration

| Component | Selection |
| --- | --- |
| STT | Sarvam Saaras v3, `transcribe` |
| Audio contract | 16 kHz, mono, PCM16 WAV |
| Retriever | `BAAI/bge-m3` |
| Chunking | fixed words, maximum 128 words |
| Index | FAISS HNSW (`M=32`, `efConstruction=200`, `efSearch=128`) |
| Retrieval depth | Top-10 unique parents |
| Evidence threshold | Top-1 ≥ 0.67 or frozen consistency rescue |
| Generator | Groq `openai/gpt-oss-20b`, low reasoning, maximum 128 output tokens |
| Prompt | `strict_context_only` |
| Output validation | schema, non-empty answer, known citation, citation required |

## Browser audio boundary

Microphone samples are captured through Web Audio. Uploads are decoded through the browser's
native audio decoder. Multi-channel uploads are mixed to mono, samples are downsampled to 16 kHz,
clamped, and encoded into a PCM16 RIFF/WAVE payload. Audio is rejected before submission when it is
empty/silent, undecodable, or longer than 30 seconds. The backend independently validates sample
rate, channel count, sample width, frame count, duration, and total upload size.

Microphone capture requires a secure context: localhost during development or HTTPS in deployment.

## Deterministic routes

| Route | Runtime meaning |
| --- | --- |
| `ANSWER` | Evidence passed and generated answer passed grounding validation |
| `INSUFFICIENT_CONTEXT` | Retrieval or model indicates inadequate evidence |
| `OFF_TOPIC` | Deterministic policy says the request is outside the dataset task |
| `UNSAFE` | Deterministic unsafe-query routing declined the request |
| `STT_FAILURE` | Invalid/empty transcript or STT provider/audio failure |
| `SYSTEM_ERROR` | A component failed closed or generation validation failed |

The frontend maps these routes to friendly copy but never chooses or changes a route.

## Progress and latency

The harness emits observation-only stage notifications at real boundaries: Transcribing, Checking
query, Retrieving evidence, Generating answer, and Validating grounding. The API marks Complete only
after the structured response returns. Early refusals show only stages actually executed. No
percentages, synthetic durations, or timed animations are used as progress claims.

The response exposes STT, input validation, route check, embedding, retrieval, evidence gate,
generation, grounding validation, aggregated guardrail, and total request timings measured with a
monotonic clock.

## Deployment boundary

The Docker image includes the exact project-local BGE-M3 cache, the FAISS index, and the chunk
mapping. It runs one non-root Uvicorn worker and reports ready
only after the secret and all artifacts validate and the harness loads. Remote deployment therefore
requires a container host with enough image storage and memory; a static or edge-only host is not
compatible with this Python/model runtime.
