# Paced PCM replay: T4 diagnostic

## Result

The 43.660-second registered speech/pause mixture completes at wall-clock audio
speed. An independent producer supplies 20 ms chunks while the model owner
decodes and publishes results. The first nonempty preview appears at 2.097
seconds. The first commit appears at 4.215 seconds. Five commits precede EOF;
the sixth completes the transcript. No admitted samples are lost, and no
committed span is repeated.

The final text matches both the full-mixture Whisper control and the
concatenated fixture controls exactly. Six normalized word edits remain against
the 88-word human reference, as in the controls. This is completion without an
observed recognition regression on this input, not error-free transcription.

All 32 registered checks pass in the
[record](../../evidence/modal-t4-tiny-en-paced-replay-2026-09-05.json).
The test runs inside one Modal worker. It does not measure microphone capture,
PC-to-server transport, display latency or long-session behavior.

## What changed

Source commit: `8b8a292afb789e500dc23c4eef218ab35702ad3d`.

The [paced replay driver](../PACED_REPLAY.md) separates source time from decoder
progress. Each chunk becomes available at its end sample divided by 16,000.
The producer does not wait for decoding or output callbacks. It records offer
and admission timestamps separately and stops on excessive lateness or a full
buffer. It never retries rejected input or reports a partial source as EOF.

The existing controller, native transaction, model weights and publication
policy are unchanged. This run uses `quiet_endpoint_stream/v1+input_evidence/v1`
with the distinct pacing identity `wall-clock-pcm/v1`. The quiet detector
proposes the same five endpoints as the unpaced control:

```text
65920, 181440, 343040, 431040, 546880
```

EOF closes the remaining input at sample 698,560. Fixture boundaries are used
only for evaluation; the driver does not supply them to the controller.

Three intermediate analyses propose `you` at pause-to-speech transitions.
Their trace indices are 8, 14 and 21; their source endpoints are samples
213,440, 375,040 and 578,880. The input is not digital silence, and the policy
records `uncertain` with `conflicting_speech_score`. Each analysis emits an
empty preview and waits for more input. None authorizes a commit or eviction.
The later complete units match the controls. This is evidence that these three
uncertain transitions retain their speech, not general hallucination detection.

## Measurements

All replay timestamps use one monotonic origin after model loading, warmup and
offline controls. An output time is the time the driver observes an event
batch, not the time a user sees subtitles.

| Measurement | Result |
| --- | --- |
| Input duration | 43.660 s |
| Offered / accepted / committed samples | 698,560 / 698,560 / 698,560 |
| Input chunks | 2,183 at 20 ms |
| First nonempty preview | 2.096529 s |
| First commit | 4.215293 s |
| Commits before EOF / total | 5 / 6 |
| Final events | 1 |
| Driver elapsed time | 43.852087 s |
| Driver completion after input-finish call | 191.233 ms |
| Maximum source-offer lateness | 2.020 ms |
| Maximum admission-completion lateness | 2.122 ms |
| Allowed offer or admission lateness | 250 ms |
| Peak controller PCM buffer | 165,120 samples, or 10.320 s |
| Configured controller PCM limit | 640,000 samples, or 40 s |
| Final buffered samples | 0 |
| Native decodes | 25 |
| Summed source submitted to decoding | 129.660 s |
| Provisional / commit / final events | 25 / 6 / 1 |
| Undelivered events | 0 |
| Human-reference word edits | 6 / 88 |

Across six commits, source-end-to-output lag ranges from 95.293 to 235.273 ms.
Its nearest-rank median is 158.674 ms; nearest-rank p95 equals the maximum in
this small sample. For 25 previews, median lag is 112.365 ms and p95 is
191.898 ms. Raw values and admission timestamps are retained in `pacing`.

These lags are **not word or subtitle latency**. A quiet-run endpoint already
includes the 600 ms quiet interval. The source endpoint is not necessarily the
last spoken word. Time spent waiting for a preview or the end of a spoken unit
must not be removed from a user-facing latency claim. The legacy
`source_progress` block still describes source coverage only; wall-clock
measurements are in `pacing`.

The unpaced comparison used 24 decodes and 119.660 seconds of summed source.
The paced schedule therefore performed more work by these counters. Summed
source duration is not a measure of encoder FLOPs or billed GPU time. There is
no demonstrated GPU saving or speed advantage over stock Whisper.

## Local verification

- 500 runtime tests pass, including 19 paced-driver tests.
- 296 repository-tool tests pass, including 31 corpus-harness tests.
- Strict typing and Ruff lint/format checks pass.
- The tests use the existing native adapter and transaction/fence fixtures.
- No backend, model weights or Lean proofs changed. Packages were not rebuilt.

Local tests block the model owner while the producer continues. They cover
buffer overload, input lag, slow consumers, cancellation, drain limits and
retained-transaction errors. A failed output callback returns the undelivered
batch remainder. It does not trigger automatic redelivery or model execution.
Closing the stream from a callback cannot make a partial replay pass.

The host rechecks admission timing, contiguous commit coverage, native output
correspondence and terminal events. Mutated records with a missing final event,
shortened commit or coverage gap fail validation.

A separate in-team read-only audit matched all 24 source-file hashes to disk
and the stated commit. It reconstructed all 25 observed PCM slices, checked
the six committed revisions against native results, and recomputed admission
times, coverage, endpoint times and human-reference edits. No discrepancy was
found. This is an internal evidence check, not external certification.

## Resources and record identity

One CPU transport probe and one paid T4 function ran. The GPU function has a
180-second execution timeout, zero automatic retries and at most one container.
It reused the cached `tiny.en` checkpoint in FP32 and the frozen LibriSpeech
PCM. No model or corpus download was needed. Both applications were checked
after completion: stopped, with zero tasks.

- CPU application: `ap-JKVWc8XyDFxv6qFhr8v92E`.
- GPU application: `ap-vPGWENVkRuRthJXsNvmlMW`.
- GPU function: `fc-01M1RVQ380241Y6JJ3P9SJFYD0`.
- Environment: Python 3.13.3, PyTorch 2.6.0+cu124, Modal 1.5.5, Tesla T4.
- Model state was unchanged before and after the run.

Execution limits are not a spending cap. This run did not query the account
balance or actual billed cost. The recorded 43.852 seconds covers the replay
driver, not model loading, controls, startup or total GPU allocation.

Record SHA-256:
`2a1ab358225420d7b5a9a308dbe2ac09e1566999247ec74384bdced82251fdea`.

Source snapshot digest:
`db921858f5ecb7c0001e40148ef15d086db7d1349380522cf98befd5b603a9c4`.

Registration SHA-256:
`0de0f0af9ec7a9f949d636721f036ed457b4417aa9e12da3faf6de42f1ebf925`.

The [GPU receipt](../../evidence/modal-t4-tiny-en-paced-replay-2026-09-05.attempt.jsonl)
and [transport receipt](../../evidence/modal-paced-replay-transport-2026-09-05.attempt.jsonl)
are archived without overwriting earlier evidence. Each copy was hash-checked.

## Reproduction and next gate

Use the pinned Modal environment and frozen assets from the preceding
[automatic-endpoint diagnostic](2026-09-05-automatic-endpoints.md). Choose a fresh
replay name; receipts prevent repeat paid calls in an existing namespace. Set
`WHISPER_MODAL_ENABLE_WORD_CORPUS=1` before running:

```sh
python -m modal run infra/modal_word_corpus.py --transport-preflight-only --replay-id paced-replay-20260905
python -m modal run infra/modal_word_corpus.py --confirm-paid-gpu --paced-replay --replay-id paced-replay-20260905
```

The next integration is a bounded PC-to-Modal PCM transport using this same
clock and controller. Test disconnection, backpressure and cancellation before
using a real microphone. Keep transport latency separate from model latency.

The acoustic and long-session gates remain open: weak speech, noisy pauses,
overlap, words near detected boundaries and at least 30 minutes of paced input.
Speech without a suitable endpoint still stops at the analysis limit. The
current quiet detector is an amplitude heuristic, not a general speech
detector. Cancellation is cooperative and cannot forcibly interrupt a blocked
native call or callback.
