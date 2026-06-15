# Sarvam STT evaluation and recommendation

## Fixed configuration

- Provider: Sarvam AI only
- Model: `saaras:v3`
- Mode: `transcribe`
- Evaluation audio: real human speech, mono PCM16 WAV, 16 kHz
- REST: synchronous `/speech-to-text`, maximum 30-second files
- Streaming: 64 ms chunks, high VAD sensitivity, VAD signals and flush enabled

## Measurement status

- Real audio samples evaluated: **0**
- REST WER/failure/latency: **unmeasured**
- Streaming WER/failure/latency: **unmeasured**
- Gold-text versus transcript retrieval degradation: **unmeasured**
- REST versus Streaming decision: **pending real recordings**

The manifest contains a balanced 24-sample recording plan, but every row remains `pending` until a
human records it with `eval/record_audio.py`. No synthetic audio, fake transcript, or fabricated
metric is used. The existing retrieval stack and sealed test are unchanged.

Saaras v3's generally available WebSocket provides VAD events and finalized transcript messages;
it does not guarantee true interim partial hypotheses. Time to first partial will remain unavailable
unless the service actually returns a non-final transcript.

The <200 ms complete-pipeline target is not claimed. It can only be assessed after STT, retrieval,
generation, and orchestration are measured together.

## Predeclared integration-mode selection rule

Streaming will be selected only if its WER is no more than 0.02 worse than REST, its failure rate is
no more than 0.01 worse, and its P95 end-of-speech-to-final latency is lower than REST P95 request
latency. Otherwise REST remains the benchmark/debugging integration while the streaming result is
reported as not yet suitable.
