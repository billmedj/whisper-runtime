# Automatic quiet-run endpoints: T4 diagnostic

## Result

The registered 43.660-second speech/pause mixture completes without supplied
boundaries. The detector observes admitted PCM in fixed frames and proposes five
endpoints. One continuous controller publishes six contiguous units, including
the final unit at EOF. All 698,560 samples are accounted for; the final buffer
is empty. The five constructed pauses remain inside the processed units.

The complete text matches both the full-mixture Whisper control and the
concatenated individual-fixture controls exactly, including punctuation.
Against the independent human reference, six normalized word edits remain over
88 words. The same recognition errors occur in the model controls. Completion
does not mean error-free recognition.

The [record](../../evidence/modal-t4-tiny-en-automatic-endpoints-2026-09-05.json)
contains all 22 checks passing. These cover source accounting, publication,
endpoint provenance and resource release. Recognition distances are recorded
separately; they are not substituted for lifecycle checks.

## Implementation

Source commit: `0063476d929f4a8808e76d073a37ad16992f9fe3`.

`QuietEndpointDetector` uses Python's standard library. The existing continuous
controller scans successfully admitted PCM under its input lock. It queues
endpoint observations, not audio copies. Model work cannot advance past the
earliest pending endpoint. The existing transaction and resource-release path
still controls text publication and audio eviction. Defaults are unchanged.

The registered profile is `quiet_endpoint_stream/v1+input_evidence/v1`:

- Mono 16 kHz signed 16-bit PCM, in fixed 20 ms frames.
- A quiet frame has an absolute peak no greater than 32 sample units.
- After nonquiet input, 600 ms of consecutive quiet can close a unit that is
  at least one second long.
- Sustained quiet closes periodic units at ten seconds, even at input start.
- A boundary is the detection endpoint, not the quiet onset. No samples are
  trimmed, and EOF does not pad a partial frame.

These are amplitude thresholds, not a learned voice activity detector. Quiet
speech may satisfy them. A quiet proposal cannot authorize empty publication:
the existing evidence policy still requires exact digital-zero input for that.
Nonzero input with uncertain output stops with its PCM retained.

The controller retains endpoint identity through cancellation and retry. New
input cannot replace an active endpoint. A successful commit removes only its
own proposal; later proposals and partial-frame state remain intact. Automatic
mode rejects manual `seal_unit()` calls.

## Diagnostic design

This run reuses the frozen LibriSpeech mixture, cached `tiny.en` checkpoint,
FP32 precision, seed 7 and T4 region from the
[caller-delimited control](2026-09-05-source-units.md). It changes the endpoint
source. The driver supplies ordinary one-second chunks and a final partial
chunk. It never calls `seal_unit()` or splits input at fixture boundaries.

Fixture boundaries remain evaluation metadata. They do not control admission,
decoding or publication. The five observed endpoint samples are:

```text
65920, 181440, 343040, 431040, 546880
```

EOF closes the remaining input through sample 698,560. Each of the six units
contains one of the repeated speech occurrences. An independent batch
calculation of the frame rule agrees with the streaming detector. Local scans
with single-sample, 321-sample, one-second and whole-input chunks produce the
same proposals. This establishes detector partition invariance for this input,
not recognition equivalence for every preview schedule.

The record binds each observed PCM hash to its exact source slice. Each commit
references a text revision that must equal its prepared native result, or its
typed empty publication. Mutation tests reject changed published text even
when source spans and lifecycle counters still look valid.

## Measurements and limits

| Measurement | Result |
| --- | --- |
| Accepted / committed samples | 698,560 / 698,560 |
| Input chunks | 44 |
| Automatic endpoints / total commits | 5 / 6 |
| Final events | 1 |
| Final buffered samples | 0 |
| Peak buffered samples | 170,560, or 10.660 seconds |
| Native decodes | 24 |
| Summed source duration submitted to decoding | 119.660 seconds |
| Word alignment in the measured stream | None |
| Measured stream loop | 2.417695238 seconds |
| Final unit committed at EOF | 9.480 seconds of source |
| Exact text match against both model controls | Yes |
| Human-reference word edits | 6 / 88 |

The replay is unpaced. Loop time excludes model loading, warmup and offline
controls. It does not measure live caption latency or qualify real-time use.
The oracle run used 23 decodes and 83.660 seconds of summed source; this run
uses more work by those counters. Summed source does not measure fixed-size
encoder FLOPs. No GPU saving or speed advantage is established.

This corpus contains clean speech and constructed two-second digital-zero
pauses. It does not test microphone noise, overlapping speakers, whispering or
arbitrary quiet speech. A fixed low threshold can miss noisy pauses; weak speech
can cause a false split. Retaining PCM does not prove recognition survived a
split. Speech without a suitable boundary still stops at the analysis limit
rather than being cut automatically.

## Verification and resources

- 481 runtime tests pass, including 23 endpoint tests.
- 291 repository-tool tests pass, including 26 corpus harness tests.
- Strict typing and Ruff lint/format checks pass.
- Existing transaction/fence fixtures exercise real commit and recovery paths.
- No model weights, native backend or formal proofs were changed.
- Wheel and source packages were not rebuilt in this iteration.

A separate in-team read-only audit reconstructed all 24 PCM observations,
checked 23 source-file hashes against the stated commit, recalculated the five
endpoints and human-reference edit distance, and joined the six raw committed
revisions to their native results. It found no discrepancy. This is a recorded
diagnostic audit, not external certification.

One CPU transport probe and one GPU function ran. The GPU function has a
180-second execution timeout, zero automatic retries and at most one container.
It used the cached model and local PCM; no corpus or checkpoint download was
needed. Execution limits are not a spending cap. Account balance and billed
cost were not measured. Both applications were verified stopped with zero tasks:

- CPU: `ap-S3cOfbwuuhAMbtS9yLoAyv`.
- GPU: `ap-EFqivIuDTDLdx2KwNrn1ib`.
- GPU function: `fc-01M1RRH8CN213Q8VPGHTVFV0W4`.

Record SHA-256:
`280fbbfe5d58047725015b5897b4069ed9b6394e292736b403b68a6202457de0`.

Source snapshot digest:
`6b8e6500e722fe91873b13ec32d786500e46b9256079e0250e76dd28ad994ad0`.

Registration SHA-256:
`feec9fa99ea854950dbde4bcadc25dbb0b2056aa6e02ee2ce855703d86bad61e`.

The [GPU receipt](../../evidence/modal-t4-tiny-en-automatic-endpoints-2026-09-05.attempt.jsonl)
and [transport receipt](../../evidence/modal-automatic-endpoints-transport-2026-09-05.attempt.jsonl)
are separate from previous attempts. No prior evidence was overwritten.

## Reproduction and next gate

Use the pinned Modal environment described in the preceding source-unit report.
Choose a fresh replay name; existing receipts block reuse. Set
`WHISPER_MODAL_ENABLE_WORD_CORPUS=1` before these commands.

```sh
python -m modal run infra/modal_word_corpus.py --transport-preflight-only --replay-id automatic-endpoints-20260905
python -m modal run infra/modal_word_corpus.py --confirm-paid-gpu --automatic-endpoints --replay-id automatic-endpoints-20260905
```

Next, measure endpoint and recognition failures on quiet speech, noisy pauses
and words near boundaries. Keep the frozen control and do not tune thresholds
on the acceptance set. Then replay at audio speed with decoder and consumer
delays to measure finalization latency, buffer growth and overload behavior.
The 30-minute continuous-use gate remains open.
