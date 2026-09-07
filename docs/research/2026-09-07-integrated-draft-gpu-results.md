# Native verified drafts: T4 results

The integrated draft path passes the
[registered comparison](2026-09-07-integrated-draft-gpu-plan.md).
Decoder forwards fall by 29.0%, and pooled decode-phase native wall time falls
by 18.2%. Both comparison orders improve. All four streams produce the same
tokens and events. Defaults remain unchanged.

## Method

One cached `tiny.en` model runs on one T4, with FP32, English, greedy decoding
and seed 7. Both arms use the same 20/24-second retained context and alignment
encoder reuse. Control uses `max_draft_tokens=0`; candidate uses 32. The native
adapter verifies proposed tokens against current audio. No decoder method is
patched, and publication thresholds do not change.

The order is control, candidate, candidate, control (ABBA). Each fresh stream
receives 93.1 seconds at source pace: two copies of the existing 46.55-second
speech/silence/noise/silence fixture. There are 1,489,600 samples per stream.
Warmup and one cancellation probe precede measurement.

This repeats known audio. It is not held-out accuracy or sustained-live
qualification. One fixed ABBA order cannot exclude nonlinear timing drift or
establish a confidence interval.

## Measurements

| Measurement per stream | Control A1 | Candidate B1 | Candidate B2 | Control A2 |
|---|---:|---:|---:|---:|
| Native windows | 49 | 49 | 49 | 49 |
| Encoder forwards | 49 | 49 | 49 | 49 |
| Decoder forwards, including alignment | 2,382 | 1,692 | 1,692 | 2,382 |
| Decode-phase native wall, seconds | 11.681 | 9.634 | 9.527 | 11.745 |
| Alignment native wall, seconds | 2.613 | 0.371 | 0.366 | 0.352 |
| Stream events | 77 | 77 | 77 | 77 |
| Commits, including silence coverage | 27 | 27 | 27 | 27 |
| Word edits / reference words | 14 / 228 | 14 / 228 | 14 / 228 | 14 / 228 |

Pooling the two cells per arm gives these changes:

- Decode-phase native wall: 23.426 to 19.161 seconds, **18.2% lower**.
  The two adjacent comparisons improve by 17.5% and 18.9%.
- Decode-phase decoder CUDA interval sum: 17.183 to 12.570 seconds, 26.8% lower.
- All encoder and decoder CUDA interval sums: 19.139 to 14.731 seconds,
  23.0% lower.

Each candidate verifies 690 of 1,246 proposed tokens across 48 draft windows.
These matches remove 690 decoder forwards. Decode-only forwards fall from
2,346 to 1,656, while submitted decode token positions rise from 2,346 to 2,902.
Batch verification reduces sequential calls, not the number of token positions
submitted. These measurements do not establish FLOPs, energy or GPU cost savings.

Decode-phase wall includes native startup, encoder work and token stepping.
It is not decoder-only wall time or end-to-end transcript latency. CUDA events
measure stream intervals around forwards, not kernel-only busy time.

## Output and cleanup

The offline audit finds exact equality of all 77 stream events, current-audio
token sequences, native text, window spans, selected policy decisions and
commit source endpoints across the comparisons. All input has accounted-for
coverage, FINAL occurs once, and every native window releases its state and
capacity. Model hashes and hooks remain unchanged. The cancellation probe
also passes.

Scores are not bit-identical: maximum absolute differences are about
`9.54e-7` for average log probability and `1.42e-6` for no-speech probability.
Admission timings differ, although source-clock contracts pass. No score
tolerance or publication rule was relaxed. The unchanged 14 word edits show
that this optimization does not fix transcription errors.

## Timing and memory limits

The first control has 2.613 seconds of alignment wall time, versus
0.352–0.371 seconds in subsequent cells. Several host-side alignment operations
are longer despite similar forward CUDA intervals. Their exact cause was not
instrumented. Do not attribute this alignment difference, or the larger pooled
total-wall improvement, to drafts.

Allocated device memory returns to 160,720,896 bytes after each stream.
Peak allocation is 230,310,912 bytes for A1 and 231,343,616 for both candidates
and the final control. These allocator measurements show neither a retained
allocation increase nor a demonstrated candidate-specific memory saving. They
are not total process memory measurements.

## Records and next step

The single worker completed in 386.595 seconds, within its 480-second limit.
It used 201 native windows including warmup and cancellation. App
`ap-2Vne3EccdVZEVwEHj5qe9P` is confirmed stopped with zero tasks. There was no
retry. The registered $0.167227 planning estimate is not a measured bill.

The [record archive](../../evidence/modal-integrated-draft-2026-09-07.zip)
contains the JSON result, original compressed response, preflight and attempt
journal. Every archive entry matches its saved original. SHA-256:
`f4e37d41a63449188eede2bc838f7bba3f5a3170e34b4adcfc821d9271ff0d4f`.
Frozen source digest:
`c83c9202da80bf7bbf6a7eff2d387861a651a6538b287db3c7496f3ffa17713f`.

The [offline verifier](../../tools/audit_integrated_draft.py) recomputes the
[audit evidence](../../evidence/integrated-draft-gpu-audit-2026-09-07.json)
without importing the producer or contacting Modal. With the saved files under
`artifacts/modal/integrated-draft-t4-20260907-v1` and the frozen checkout, run
`python tools/audit_integrated_draft.py` from the repository root.

Next qualification should combine new, held-out audio with score-threshold
cases and fresh-process restore checks. Other models, decoding modes and
longer live sessions remain unqualified. No further GPU run is queued.
