# Word-aligned T4 replay reaches EOF

Date: 2026-09-05. Tested source: `8e64089`.

## Result

The word-aligned profile completes the registered 33-second replay after the
window-edge anchor repair. The segment profile remains unresolved at 8 seconds.
The top-level record therefore remains `unresolved`; only the word cell completes.

| Profile | Committed coverage | Commits | Decodes | Loop time | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Segment agreement | 8.00 s | 1 | 17 | 3.569 s | Unresolved at EOF |
| Word agreement | 33.00 s | 8 | 17 | 4.420 s | Completed |

The word profile commits through 1.76, 5.70, 9.92, 12.78, 18.00, 21.22, 21.74,
and 33.00 seconds. Seven commits occur before EOF. All 528,000 accepted samples
are accounted for, the PCM buffer is empty at completion, committed revisions
remain unchanged, and the runtime restores its declared capacity. Peak retained
PCM is 212,160 samples, or 13.26 seconds. This is not total process memory.

After case and punctuation normalization, the 66 words match all three
model-generated controls with zero word edits. The complete strings do not
match: punctuation, capitalization, and spacing differ, including a separated
period at a publication boundary. The controls are not independent human
annotations. This result does not establish a general recognition error rate.

## Interpretation

The [v6 registration](../../experiments/modal-stream-boundary-diagnostic-v6.json)
preserves v5 audio, model, FP32 precision, seed, options, cells, cadence,
holdback, context, tolerance, warmup, transport, and budget. The runtime uses
the existing repair from `159b593`. No buffer limit or runtime default changed.

The blocked 12.78-second boundary advances to 18.00 seconds as predicted locally.
Subsequent windows reach EOF. However, the last commit covers 21.74-33.00 seconds:
completion relies on the explicit EOF rule, which does not require a second
agreeing observation. The preceding analysis reports `unstable`. This needs
further testing before claiming low-latency publication throughout a stream.

The input repeats the same 11-second JFK fixture three times and is unpaced.
The 4.420-second loop is not live caption latency. Profiles publish different
amounts of text, so their timings are not a speed comparison. Word-profile peak
allocated GPU memory is 565,032,960 bytes; reserved-memory peaks include prior
allocator history. No GPU-efficiency claim follows.

Next: distinct short recordings with references, silence, repetitions, and
boundary cuts; then paced long-session qualification. Keep D1 open until its
30-minute acceptance cases pass. Preserve the unresolved segment result.

## Evidence and verification

- [Raw record](../../evidence/modal-t4-tiny-en-word-alignment-v6-2026-09-05.json).
- [Attempt receipt](../../evidence/modal-t4-tiny-en-word-alignment-v6-2026-09-05.attempt.jsonl).
- [CPU transport receipt](../../evidence/modal-word-alignment-v6-transport-2026-09-05.attempt.jsonl).
- [Predecessor](2026-09-05-word-alignment-replay.md).

Record: 360,593 bytes; SHA-256
`ed5565fdcc86fabaad7d66d4122842a0ee99b6533a6d60535c9684c9a3346769`.
The 20-file source snapshot digest is
`ea690cd0bec14cb18811aa1e22dad5d0bed95e7014c1f9bb269334b6e3d31241`.
The synchronous worker result is 13,123 compressed bytes, saved before decoding.

Preflight: 388 runtime tests and 255 repository-tool tests pass. The first tool
suite attempt used an interpreter without JSON Schema support and failed;
rerunning in the existing project validation environment passed. No dependency
installation or GPU call was used to resolve that local environment mismatch.
Lint and formatting checks pass.

## Execution and cost

One CPU transport call and one T4 function call ran, without automatic retries.
The T4 function timeout was 120 seconds. Both apps stopped with zero tasks.

Modal reported ephemeral-app cost of USD 0.10456006 before and USD 0.11223549
after, an observed increase of USD 0.00767543. The monthly metered total moved
from USD 0.10021114 to USD 0.11021114; billed cost remained zero. These counters
can update at different times. They do not provide an exact per-run invoice or
current remaining-credit balance.
