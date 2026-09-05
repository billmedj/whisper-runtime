# Word-alignment diagnostic on T4

Date: 2026-09-05. Source commit: `05e83e1`.

## Result

Neither stream completed. Both preserved committed output, accounted for accepted
input, and released their resources. The new word-alignment profile is not yet
qualified for continuous use on this configuration.

| Profile | Committed audio | Input | Decodes | Loop time | Final reason |
| --- | ---: | ---: | ---: | ---: | --- |
| Segment agreement with context | 8.00 s | 33 s | 17 | 3.623 s | `eof_unresolved` |
| Word agreement with context | 5.70 s | 33 s | 17 | 5.291 s | `anchor_missing` |

The two cells used the same T4, `tiny.en` FP32 checkpoint, PCM, seed, decode
settings, and two-second preview interval, holdback, and left context. Coalescing
was disabled. One word-alignment warmup ran before cell timing. The input was
three copies of the pinned 11-second JFK fixture, supplied in one-second chunks
without wall-clock pacing.

These times are diagnostic observations, not comparable service latencies:
the streams published different amounts of text, neither finished, and the
caller did not supply audio in real time. Memory measurements also include
shared allocator history. Peak allocated memory was 344,988,160 bytes for the
segment cell and 565,033,472 bytes for the word cell. Peak reserved memory was
1,577,058,304 and 4,276,092,928 bytes respectively. No memory reduction is shown.

The record's text-distance values compare partial output with model-generated
controls, not human reference transcripts. They must not be presented as a
recognition error rate for a completed transcription.

## Cause of the word-profile stop

Trace 4 commits `Americans ask not what` through 5,700 ms. Its estimated words
include `ask` at 2,340–3,860 ms, `not` at 3,860–4,620 ms, and `what` at
4,620–5,700 ms. The controller then retains audio from 3,700 ms.

The old anchor filter removed words ending before that origin. It kept `ask`,
although most of that word's audio had already been removed. Traces 5 and 6
start with `not what`; they cannot match the required `ask not what` anchor.
The stream continues to report `anchor_missing` through EOF.

`not what` remains available, with timing differences of 140–160 ms within the
configured 200 ms tolerance. The repair is to select anchor words whose entire
estimated span remains in retained audio. It does not require more context,
larger buffers, altered timestamps, or a text-only search for a later repetition.
If no complete anchor remains, the decision must still be unresolved.

The failed GPU record is evidence for the diagnosis, not validation of that
repair. Local regression tests can exercise the recorded boundary; another
registered GPU run is required to test the corrected rolling trajectory.

The local repair is implemented. A regression uses the exact tokens and times
from traces 4–6. It removes only the cut anchor word, preserves `not what`, and
selects new text over processed coverage 5,700–9,920 ms. This is a counterfactual
decision on recorded hypotheses, not a resumed GPU run: later analyses will
change when the window advances. Missing complete anchors and misplaced repeated
phrases still fail closed. The source suite passes 384 tests, including these
two new cases; the repository-tool suite passes 253 tests.

## Evidence and scope

- [Raw result](../../evidence/modal-t4-tiny-en-word-alignment-v4-2026-09-05.json).
- [Attempt receipt](../../evidence/modal-t4-tiny-en-word-alignment-v4-2026-09-05.attempt.jsonl).
- [CPU transport receipt](../../evidence/modal-word-alignment-v4-transport-2026-09-05.attempt.jsonl).
- [Registration](../../experiments/modal-stream-boundary-diagnostic-v4.json).

The JSON record is 367,241 bytes. Its SHA-256 is
`91858153060af97a793c72af99e33af8092ebc5e82f2ed3003e8d247536d1a6d`.
It binds the 20-file source snapshot with digest
`007f0f6ecee7eabc7604755ccd1ed24c0e9bfa59e34034469cc2b6b0fa7ad5c6`.
The worker returned 12,230 compressed bytes through the verified synchronous
transport. The client saved them before decoding.

The earlier local CPU smoke used a different PCM conversion. Its success does
not imply that this T4 configuration succeeds. This comparison does not isolate
whether hardware or PCM changes altered the intermediate hypotheses.

## Execution and cost

One T4 function call ran, with a 120-second timeout and no automatic retry.
A preceding Windows console encoding error occurred before any function call;
the empty application was stopped. After enabling UTF-8 output, one CPU-only
transport check passed before the GPU call. All three applications were stopped
with zero tasks after the test.

The monthly metered total increased from USD 0.09021114 to USD 0.10021114.
The observed change was USD 0.01; billed cost remained zero under credits.
This is not a per-call invoice or a remaining-credit balance. The execution
estimate used [Modal's published rates](https://modal.com/pricing); startup,
storage, and billing lag are outside that estimate.

Next: validate the anchor repair on the recorded failure locally, then repeat a
short GPU case. Do not widen tolerances or start a long-session run to bypass
this failure. Defaults remain unchanged.
