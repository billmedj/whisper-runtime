# Word-alignment replay on T4

Date: 2026-09-05. Tested source: `212b293`.

## Result

The complete-anchor repair removes the first recorded blockage. The word stream
now commits through 12.78 seconds, up from 5.70 seconds. It then stops at a
different anchor mismatch. Neither profile completes the 33-second input.

| Profile | Committed coverage | Decodes | Loop time | EOF reason |
| --- | ---: | ---: | ---: | --- |
| Segment agreement | 8.00 s | 17 | 3.286 s | `eof_unresolved` |
| Word agreement | 12.78 s | 17 | 4.569 s | `anchor_missing` |

The [v5 registration](../../experiments/modal-stream-boundary-diagnostic-v5.json)
keeps the v4 input, model, FP32 precision, seed, decode options, chunk size,
preview cadence, context, holdback, tolerance, warmup, and resource limits.
The segment events and decision traces match v4 after replacing the run ID.
Word commits end at 1.76, 5.70, 9.92, and 12.78 seconds. Trace 6 confirms the
previously predicted 5.70–9.92 second publication on the real T4 run.

Both profiles preserve committed output, account for all 528,000 accepted
samples, and release runtime capacity. Neither emits a final event. The word
profile retains a buffer of 355,520 samples, including 323,520 uncommitted
samples. Retained left context accounts for the difference.

This is one unpaced replay of a repeated fixture, not a live qualification.
Loop times cover different amounts of published output. Text distances compare
partial transcripts with model-generated controls. They do not establish an
accuracy, latency, or efficiency improvement. Peak allocated device memory is
344,988,160 bytes for segments and 565,032,960 bytes for words. Reserved-memory
peaks include allocator history from the preceding cell.

## Remaining boundary mismatch

Trace 8 publishes `your country. And so my fellow` through 12,780 ms, then
retains audio from 10,780 ms. Its four-word anchor is `And so my fellow`.

| Word | Frozen estimate | Estimate after the window shift |
| --- | --- | --- |
| And | 11,660–11,720 ms | 10,780–11,700 ms |
| so | 11,720–12,060 ms | 11,700–12,060 ms |
| my | 12,060–12,460 ms | 12,060–12,440 ms |
| fellow | 12,460–12,780 ms | 12,440–12,760 ms |

The new estimates are from traces 9 and 10. Text and tokens match. Every end
estimate differs by at most 20 ms. However, the first word starts at the new
analysis origin: its start moves by 880 ms, beyond the 200 ms tolerance. That
single comparison rejects the full anchor. Later traces remain unresolved.

This is distinct from v4: all four frozen word spans remain in retained audio.
The local backend computes word times with full-window dynamic time warping.
The observed mismatch is consistent with a window-edge alignment effect, not
evidence that the word moved in the source audio. A repair must distinguish this
edge estimate from an internal timing mismatch without accepting a later
repetition or changing committed output.

## Local repair after the run

The policy now permits a window-edge start estimate under a narrow condition:
the match must begin at the first observed word, that word must start exactly at
the analysis origin and extend left of its frozen start, and the anchor must
contain at least two words. Every text and token value must match. The first
word's end and all remaining word bounds still use the original tolerance.

New-word comparisons, ambiguity checks, and publication boundaries are
unchanged. The policy does not change raw timestamps or previously committed
text. This is an explicit change to the anchor-matching rule, not evidence of
exact acoustic timing. The v5 record predates this repair and cannot validate
its later rolling trajectory.

Four new local tests cover the recorded boundary and rejection cases, bringing
the runtime suite to 388 passing tests. On the exact trace 8–10 word estimates,
the repaired policy selects coverage 12,780–18,000 ms and text
`Americans ask not what your country can do`. This is a counterfactual policy
decision, not an additional GPU publication. A separate case exercises explicit
EOF at 20,000 ms; the real replay continued to 33,000 ms.

The tests reject singleton anchors, internal or off-origin shifts, changed
text or tokens, displaced word ends, changed neighboring bounds, ambiguity,
and later repetitions. They also retain strict comparison of new words and
validation of both growing observations. Independent code review found no
blocker within this policy's stated scope. The next GPU replay must keep the
same settings and test the new rolling trajectory through EOF.

## Evidence

- [Raw result](../../evidence/modal-t4-tiny-en-word-alignment-v5-2026-09-05.json).
- [Attempt receipt](../../evidence/modal-t4-tiny-en-word-alignment-v5-2026-09-05.attempt.jsonl).
- [CPU transport receipt](../../evidence/modal-word-alignment-v5-transport-2026-09-05.attempt.jsonl).
- [Predecessor and first repair](2026-09-05-word-alignment-comparison.md).

The raw result is 344,184 bytes, SHA-256
`78a493bf14957c22088533b2d2ff2286cb0f28d5c352b1bde7eff84b171429e3`.
The 20-file source snapshot digest is
`fc0e7614ad1f76dc514cc950934475ab7f4b2ba30a069e3e19b3e80e7920fe18`.
The worker returned 12,595 compressed bytes. Independent review checked the
receipt, decoded result, source hashes, publication events, input accounting,
and privacy before the unchanged files were copied into evidence.

## Execution and cost

One CPU transport probe and one T4 function call ran. The GPU function had a
120-second timeout and no retries. Both applications stopped with zero tasks.
No second GPU call was made.

Immediately after the run, Modal reported USD 0.10456006 for ephemeral apps,
up from USD 0.09758927 before it: an observed increase of USD 0.00697079. The
top-level monthly metered total still showed USD 0.10021114, and billed cost was
zero. These fields can update at different times. This is a billing observation,
not a final per-run invoice or remaining-credit balance.
