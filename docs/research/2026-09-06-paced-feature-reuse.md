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
