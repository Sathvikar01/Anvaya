# HH Goa submission checklist

Statuses reflect repository state, not aspirational claims.

## Frozen experimental system

- [x] Sarvam Saaras v3 STT configuration unchanged.
- [x] BGE-M3 retriever unchanged.
- [x] Sentence chunking and 128-word cap unchanged.
- [x] FAISS HNSW `M=32`, `efConstruction=200`, `efSearch=128` unchanged.
- [x] Evidence/guardrail thresholds unchanged.
- [x] `sarvam-105b`, Top-10, and `strict_context_only` unchanged.
- [x] API delegates to `VoiceRAGHarness`; no pipeline logic duplicated in routes.

## Live demo UX

- [x] Project name and short description.
- [x] Microphone Record/Stop controls and optional upload.
- [x] Browser conversion to 16 kHz mono PCM16 WAV.
- [x] Permission-denied, empty, recording, upload, and backend error messages.
- [x] Transcript, final answer/refusal, all six routes, and preserved reason codes.
- [x] Retrieved passage, source/chunk ID, score, and cited-ID highlighting.
- [x] Large evidence context collapsed by default.
- [x] Actual stage notifications only; no fake timing or percentages.
- [x] Collapsed measured timings/retrieval trace and measured total latency.

## API and operations

- [x] `POST /api/query/audio` returns existing structured harness output.
- [x] `GET /health` validates application readiness.
- [x] Upload size, duration, frame count, format, and WAV parameters validated.
- [x] Temporary audio deleted after each request.
- [x] Environment/frozen artifact validation; API keys excluded from Git/Docker context.
- [x] Non-root single-worker Docker image and health check.
- [x] Local and Docker run commands documented.
- [ ] Public HTTPS live URL — blocked until an authenticated container host with sufficient image
      storage/memory is available.

## Evaluation and reporting

- [x] `eval/record_audio.py` preserved and documented for 24 real samples.
- [x] Exact STT, formal E2E, and result-regeneration commands documented.
- [x] Missing formal metrics remain blank; no synthetic metrics.
- [x] Retrieval/generation tables include only measured results.
- [x] Guardrail 24-case results labeled DEVELOPMENT.
- [x] Smoke, development, and formal results separated.
- [x] Formal Voice E2E explicitly PENDING at 0/24 recordings.
- [x] Measured ~1580 ms Sarvam generation P50 and inability to meet <200 ms documented.

## Submission package

- [x] Submission-quality `README.md` and four requested `docs/` helpers.
- [ ] Record final video after a live HTTPS deployment is available.
- [ ] Add 24 real-human recordings and regenerate formal Voice E2E tables.
- [ ] Paste the final verified live URL into the submission form.

## Final verification gate

- [x] Full pytest suite (78 tests) and Ruff pass.
- [x] 6.0 GB Docker image builds; local and container health checks pass.
- [x] Browser page, upload conversion, result rendering, and console verified.
- [ ] Real microphone Record/Stop — the available in-app browser has no microphone device; the
      no-device error path was verified.
- [x] Live `ANSWER`, `INSUFFICIENT_CONTEXT`, `OFF_TOPIC`, `UNSAFE`, and `STT_FAILURE` routes; error
      routes/invalid uploads are also covered by automated tests and local API checks.
- [x] Secret scan confirms no committed credential.
- [x] Phase 7 committed separately.
