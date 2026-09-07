# Source-paced EOF context retry

## Registered comparison

The preceding [incremental controller test](2026-09-06-eof-context-retry.md)
recovered a blocked 33.66-second development input. This comparison supplies
20 ms PCM chunks at source speed through the existing independent producer
thread. Model work must not slow the source clock. No microphone is required.

One T4 call runs three cells in this order:

1. No-added-pauses input, 33.66 seconds, retry disabled.
2. The same input with `eof_context_retry=True`.
3. Noisy prefix, 10.89 seconds, retry enabled.

All cells use the existing `tiny.en` checkpoint, FP32, seed 7 and legacy word
alignment. No model or runtime acceptance threshold changes. No warmup model
calls are hidden from the record. The first arm is cold, so whole-arm times
cannot establish a speedup. Input arrival timing may change window scheduling;
cross-arm native observations are compared, not assumed identical.

The configured function timeout is 150 seconds. Before starting each cell,
reserve its source duration, up to 15 seconds to drain EOF and 20 seconds for
cleanup. Stop new native admissions after 130 seconds or after 80 windows in
total; each cell permits at most 30 windows. Use one container, no automatic
retries, a read-only cached model and blocked worker network access. These
execution limits do not represent a monetary spending cap.

## Predictions and acceptance

The no-added-pauses retry should complete if the streamed observations reach
the anchored refusal recovered by the preceding experiment. Completion must
come from the ordinary transaction and publication path, with the earlier
prefix unchanged. If the baseline already completes or its schedule differs,
report that result rather than claiming an exact matched recovery.

The archived noisy-prefix failure has a different cause. The native result
repeats retained words; the proposed new suffix contains only punctuation.
The retained boundary clamps the fixed context retry to the original window.
If this state repeats, expect `unavailable / no_distinct_anchor_window`, no
additional decode and no final event. The remaining PCM must stay retained.
A refusal is not successful transcription.

Record every native window and encoder forward, the full source-admission
clock, emitted events, word alignments, retained PCM hashes and retry receipt.
Measure first nonempty text, first commit and delay from input EOF to final
output. These are same-worker measurements, not microphone, network or
word-level caption latency. Score text against the fixed human reference only
after decoding. The reference must not select a crop, prompt or publication.

Report successful stream completion separately from preserved safety checks
and unresolved acoustic cases. This development comparison does not qualify
unseen speakers, arbitrary noise, long sessions or production use.

## Observed result

The single call completed the registered comparison. The no-added-pauses retry
completed at source speed. The noisy prefix still failed recognition; its
refusal and retained input matched the prediction.

| Cell | Outcome | Committed audio | Native windows | Encoder forwards | Word edits |
| --- | --- | ---: | ---: | ---: | ---: |
| No added pauses, baseline | EOF refusal | 21.26 / 33.66 s | 17 | 34 | 36 / 88 |
| Same input, retry enabled | Complete | 33.66 / 33.66 s | 18 | 36 | 6 / 88 |
| Noisy prefix, retry enabled | No distinct retry window | 3.68 / 10.89 s | 6 | 12 | 18 / 26 |

Both no-added-pauses cells admitted 1,683 chunks. Their native observations
match through the original refusal, including the frozen prefix. The new
window `[20.22, 33.66]` seconds then passes the unchanged strict selector and
commits the remaining 12.40 seconds. Its buffer is empty and it emits one final
event. All source, output-clock, PCM, publication and capacity checks pass for
that cell.

The retry cell's first nonempty text appears at 2.126 seconds, its first commit
at 6.208 seconds, and its final event 326.527 ms after input EOF. Driver return
occurs 327.473 ms after EOF. Maximum measured source lag is 1.183 ms and maximum
admission lag 1.271 ms, below the configured 250 ms limit.

The baseline's cold first text appears at 6.694 seconds. Its source lag reaches
193.160 ms, still within the same bound. The ordered cold/warm difference is
not a measured benefit of the retry policy. No word-level caption latency,
startup guarantee or speedup follows from these three observations.

### Noisy refusal

The noisy source admits all 545 chunks. Its EOF result covers the old anchor,
but the proposed new suffix contains only punctuation. The 500 ms context
guard clamps to the already analyzed `[1.68, 10.89]` second window. The runtime
records `unavailable / no_distinct_anchor_window` and does not spend another
decode on the same input.

No final event is emitted. All 147,360 retained samples remain available:
7.21 seconds of uncommitted audio and 2 seconds of committed context. Source
admission, output integrity and resource checks pass; completion checks fail.
This is a preserved refusal, not recovered transcription.

The next acoustic diagnostic should distinguish crop effects from decoder
history on this exact refusal. Existing retained and head-only observations
provide null-prompt controls. A bounded prompt comparison can use only the
model's already published prefix, never the human reference. Keep that test
nonpublishing: prompted agreement must not weaken anchor checks or grant a
head-only result join authority. The separate long-noise anchor-timing failure
needs its own test; it is not this missing-continuation failure.

That [fixed 2x2 comparison is now recorded](2026-09-06-noisy-context-prompt.md).
Published decoder history restores the next utterance on the same retained
audio, but the result lacks the old anchor. Publication stays refused. The
report identifies an earlier unprompted joint observation for local continuity
analysis; no additional prompt sweep is warranted by this result.

## Evidence and resource accounting

The [unchanged result JSON](../../evidence/modal-t4-tiny-en-paced-context-retry-2026-09-06.json)
has SHA-256:

`fa856cb4f0028109e7524b6e563fd008600d6fe84043d86c8f433784bdc2e3bf`

The executed 37-file source snapshot has digest:

`42e810e8c1a4dcd5f232b8048ff786f7bece7177d58c129284187bd52f9ddcdb`

All 41 native windows and 82 encoder forwards are accounted for. The model
fingerprint stays unchanged, all cells release declared capacity, and the
Modal app stops with zero tasks. The worker reports 88.713 seconds elapsed.
There is one GPU call, no automatic retry and no model download.

The shared experiment worker now accepts a registered producer and uses the
existing paced driver. It does not add another runtime, source loop or model
dependency. It also stops later cells after execution failures such as drain
timeouts, even when the paced driver returns no exception. An expected policy
refusal remains distinct from those failures.
