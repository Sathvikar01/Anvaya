# Submission demo script

Target length: 2–3 minutes. Keep Technical details collapsed until the final step.

1. **Introduce Anvaya.** Show the landing screen and say: “Anvaya is a multilingual Voice-RAG
   assistant that answers only from retrieved evidence.” Point out the ready health indicator.

2. **Speak an answerable query.** Use the verified live-demo prompt:
   “परमाणु चिकित्सा प्रौद्योगिकीविद् के लिए औसत वेतन क्या है?” Stop the recording.

3. **Show the real pipeline.** Let the stage list move through Transcribing, Checking query,
   Retrieving evidence, Generating answer, Validating grounding, and Complete. Do not cut in fake
   progress or overlay a target time.

4. **Show transcript and retrieval.** Read the transcript displayed by Sarvam. Open the cited source
   card and point out the parent ID, chunk ID, measured retrieval score, and passage text.

5. **Show grounded answer.** Read the answer and point out that the evidence ID cited by the model is
   highlighted. Briefly mention that unknown or missing citations fail closed.

6. **Speak an off-topic query.** Use “मेरे लिए बारिश पर एक प्रेम कविता लिखो।” Show that the
   deterministic route is `OFF_TOPIC`, the friendly response explains the dataset boundary, and
   retrieval/generation stages were not fabricated after the early route.

7. **Show latency and evaluation.** Expand Technical details on the answerable result and show the
   measured STT, embedding, search, guardrail, generation, and total latency. Then show
   `docs/evaluation_summary.md`: retrieval/generation measurements, guardrail results labeled
   DEVELOPMENT, and Formal Voice E2E marked PENDING at 0/24 recordings.

8. **Close honestly.** State: “The selected Sarvam generation P50 is about 1.58 seconds, so this
   frozen stack does not meet the challenge's under-200-millisecond complete-pipeline target.”

Before recording the submission video, run `docs/submission_checklist.md` and use an HTTPS URL (or
localhost) so browser microphone permission is available.
