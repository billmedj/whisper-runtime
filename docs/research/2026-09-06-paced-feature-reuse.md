# Encoder reuse in a paced stream

## Registered question

Does same-window encoder reuse preserve the output of the continuous controller
when audio arrives at source speed? Does it reduce observed work or output delay?

This is a paired diagnostic on one T4, not a performance qualification. The
[fixed-window comparison](2026-09-06-alignment-feature-handoff.md) already records
exact word parity on eight inputs. It cannot establish the behavior of a live
controller whose next window depends on when audio becomes available.

## Inputs and order

Use the existing hashed LibriSpeech PCM assets and deterministic acoustic-case
builder. No new audio or model download is needed. All inputs are development
data. References are used only to score emitted text, never to route inference.

| Input | Duration | Order |
| --- | --- | --- |
| Full speech/pause mixture | 43.660 s | Legacy, reuse |
| Attenuated mixture prefix | 10.890 s | Reuse, legacy |
| Noisy mixture prefix | 10.890 s | Legacy, reuse |

The two short cases contain the first two complete source utterances and their
intervening pause: the first 174,240 samples of the existing transformed
mixtures. These fixture boundaries define the test input, not controller
endpoint decisions. Record exact PCM hashes and reference text. The combined
source duration is 130.880 seconds across six cells.

## Fixed configuration

One worker uses the cached FP32 `tiny.en` checkpoint, PyTorch 2.6, one model
binding and one CUDA lane. Apply the optional alignment patch in the disposable
backend checkout before importing Whisper. Record the original and patched
backend identities and check the registered model fingerprint.

Both arms use the same opt-in native execution profile. A diagnostic run facade
selects legacy alignment for the control; it does not replace model state or
change the native profile. The continuous policy uses the existing automatic
quiet endpoints plus `word_boundary_fallback=True` and `left_context_ms=2000`.
Retain all other settings. Do not enable context expansion or the resolution
probe, or change timestamp tolerances, between arms.

The independent producer supplies 20 ms chunks at their absolute source
deadlines. Use the existing paced replay driver and its failure limits. Run one
short legacy/reuse warmup pair outside the measured cells. Keep those calls in
the record and the total native-window bound.

## Observations and acceptance

Keep raw admissions, output events, native decision traces and lifecycle checks.
Compare normalized committed text, committed samples, ordered commit spans and
aligned outputs from identical analysis windows. Do not rewrite differing
window boundaries or ignore a failed arm to manufacture equality.

Record encoder forwards and input frames per native window. Reuse should remove
alignment encoder work, but total savings depend on the number of analyses.
Record decode/alignment host times separately from source-paced duration. Device
timing is reported only if measured explicitly with symmetric instrumentation;
otherwise leave it unavailable. Neither source duration nor a declared resource
lease measures GPU occupancy or energy.

Record first output, first commit, source-to-output delay distributions, EOF
drain time, memory before/after close and peak allocations. Output callback
times exclude display and PC-to-Modal transport latency. No general throughput
multiplier or production-readiness claim follows from one worker.

A useful result requires both arms to finish the case, preserve the committed
text and coverage, and pass input-accounting and resource-release checks. A
different segmentation remains a reported difference. Record errors against
the human reference separately; equal output can still be wrong.

## Execution limits

Use one Modal GPU call with no automatic retry, a 180-second worker timeout,
one container and the existing CPU transport preflight. At most 128 native
windows, including warmup, can start. Before each paced cell, reserve its input
duration plus the 15-second drain allowance and 15 seconds for result handling.
If that time is unavailable, return a partial budget-stop record. These bounds
limit the experiment; they are not an invoice cap.

Freeze source and this plan before launch. Preserve raw output and failed cells.
Test record validation locally before consuming GPU credits. The separate
handoff work remains local and cannot change this comparison's policy.

## Recorded result

All six cells ran at source commit
`b1901c7b33fd3be7385056ef962caf5775e68cf7`. The
[raw record](../../evidence/modal-t4-tiny-en-paced-features-2026-09-06.json)
has SHA-256
`ada0b2584f60edbb38bce9086d897a244694936d2704ee70c452ec1bd564a4b2`.
Its status is **failed** because both noisy cells stop before full publication.
There was no budget stop, transport error or automatic retry.

| Input | Result in both arms | Native windows per arm | Encoder forwards, legacy / reuse | Native host time, legacy / reuse |
| --- | --- | --- | --- | --- |
| Full mixture | Complete, 43.660 s | 25 | 47 / 25 | 3,630.9 / 2,620.5 ms |
| Attenuated prefix | Complete, 10.890 s | 6 | 11 / 6 | 726.1 / 626.8 ms |
| Noisy prefix | Unresolved at 3.680 of 10.890 s | 6 | 12 / 6 | 1,067.1 / 624.1 ms |

For every pair, committed text and ordered commit spans match exactly. The
32 common aligned analysis windows also have identical raw words, tokens and
times. No aligned windows are unpaired or ambiguous. The number of analyses
does not increase with reuse in this run. Whole decision traces are not
byte-identical: their `accepted_through_sample` observations reflect different
producer arrival times. Do not normalize those raw observations away.

The complete mixture has six word edits against its 88-word human reference
in both arms. The attenuated prefix has one edit against 26 words in both arms.
Equal output does not mean error-free transcription.

### Timing and memory

On the full mixture, encoder forwards fall by 46.8%. Summed native host calls
take 27.8% less time in the reuse arm. This sum includes decode, alignment,
publication/cleanup calls and measurement overhead; it excludes waiting for
source audio and is not CUDA-event timing or billed GPU duration. There is one
pair per input on one worker, with limited warmup. Shape-specific startup work
and scheduling variation remain possible confounders. No general speed or
energy claim follows.

First nonempty output arrives at 2.125 s in the control and 2.101 s with reuse.
First commit arrives at 4.482 s and 4.231 s. EOF drain takes 175.904 ms and
169.833 ms. These are same-worker callback times, not screen or network latency.
Raw source-to-output lag distributions remain in each cell; the source-time
endpoint of a commit is not the spoken onset time of each word.

All three pairs show the same peak allocated-memory change:
290,098,176 bytes with legacy alignment, 190,260,736 bytes with reuse, a 34.4%
reduction in the observed PyTorch allocation peak. Reserved allocator memory
remains 436,207,616 bytes (416 MiB). The smaller live allocation peak therefore
does not demonstrate that the process returned that memory to the GPU driver.
Post-close allocated memory is exactly 160,720,896 bytes in every cell.

All 76 native runs, including two warmups, close and release capacity. The model
fingerprint is unchanged. Worker execution takes 147.877 seconds, including
initialization, warmup, paced input and validation work. Modal application
`ap-h14ATgULbTjgVqEmtnd6kU` stopped at 2026-09-06 10:02:56 UTC+08 with zero tasks.
This uses one paid function; it is not a 30-minute or microphone qualification.

### The remaining noisy-input failure

Both arms accept all 174,240 samples and publish 58,880. The retained buffer
starts at sample 26,880 and contains 147,360 samples, including context before
the committed boundary. Neither arm emits a final event.

The last native text is `to its place amidst the tents.` in both arms. It repeats
already committed content and supplies no new lexical continuation. The
`no_lexical_text` refusal does not mean the native text is empty. Its meaning
here is that the controller cannot publish the remaining speech from this
hypothesis. The second source utterance is still untranscribed. Its omission
produces an 18-edit distance against the full reference; do not score the short
published prefix as a successful transcript.

The new [handoff assessment](2026-09-06-resolution-handoff.md) addresses what
evidence a different retained-audio window must provide before a continuation
can be considered. No such recovery is enabled by this comparison.

## Next step

Keep encoder reuse opt-in. It now has exact-output evidence on these paced
inputs, but does not resolve noisy boundaries. Use the retained failed state
to test a fixed head-only and anchor-overlap comparison, with model/options and
PCM identities attached. Establish the handoff rule before another full replay;
do not loosen the refusal to turn this record green. Extend to unseen inputs
and longer sessions after the short noisy case can complete under a measured
quality policy.
