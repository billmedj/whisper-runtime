# Fixed context-guard observations

## Registered question

Does a fixed earlier start restore the anchor and complete continuation in the
three states that lack this observation? This is a boundary diagnostic, not a
live recovery or performance qualification. Two other states already have the
required input interval recorded; do not decode them again.

The rule was fixed in the [CPU disagreement report](2026-09-06-resolution-disagreements.md):
subtract 500 ms from the anchor onset on the 20 ms grid, clip to retained audio,
and preserve EOF. The guard is not fitted to reference transcripts. Earlier
controls already rule out this guard as a universal recovery method: the noisy
control still omits the continuation, and the speaker-8455 control still changes
the frozen name. The remaining observations test its behavior on other failures.

| State | Guarded interval, ms | Work in this run |
| --- | --- | --- |
| Paced noisy | 1680–10890 | Reuse the seven-window control |
| Context, no added pauses | 20220–33660 | One new native observation |
| Context, continuous noise | 31900–43660 | One new native observation |
| Speaker 2961, clean | 620–8220 | One new native observation |
| Speaker 8455, noise | 640–7740 | Reuse the earlier word-resolution control |

## Identity and execution limits

Use the same cached FP32 `tiny.en`, pinned backend with legacy alignment,
decode options, seed, and exact PCM transformations. No borrowed alignment
features, fresh-control repetition, warmup windows, retry, or model download.
Use one T4 function, one model owner and CUDA lane, at most three native
windows, and the existing 180-second function timeout. Before each admission,
preserve the worker's cleanup-time reserve. A failed release stops more work.
These bounds are not a dollar invoice cap.

Bind the saved input and every copied alignment to the byte-hashed seven-window
archive. Check PCM slices and effective execution identities. Preserve the
source-checked legacy bridge and its limitations for the older records: their
backend trees differ and historical package identities were not measured.
Two speaker-specific prefixes are synthetic bootstrap states, not live commits.
Matching an archived control does not constitute fresh control reproduction.

Freeze source before launch, run local protocol and assessment tests, and
verify CPU transport before paid execution. Retain raw output and an exclusive
attempt receipt, including partial failure. Do not overwrite an attempt.

## Output comparison

Keep each historical refusal and its raw observations unchanged. The new helper
requires the actual planned guarded interval and the existing head-only
candidate interval. It then applies the previous output criteria in order:

1. One exact raw text/token anchor in the complete observation.
2. Every anchor start and end within the existing 200 ms tolerance.
3. A lexical continuation, not only punctuation or an empty suffix.
4. Neither continuation starts before the frozen boundary or observed anchor end.
5. Complete suffixes match in raw unit text and tokens.
6. Every suffix start and end agrees within 200 ms.

Do not normalize capitalization, select a shorter convenient suffix, move the
committed boundary, or widen tolerances. Separate text/token/timing diagnostics
remain visible even when the first strict refusal stops the assessment. A
structurally eligible result still has `publication_authorized=False`: two
observations can share an omission. No text is published and no audio is evicted.

The head-only proposals have already been scored. Do not describe their old
scores as a recognition improvement caused by the new guarded observation.
References do not choose a window or determine a handoff.

## Interpretation

Record all three new observations, successful or failed, with encoder counts,
padded frames, decoder steps, host times, memory, and release status. Historical
inference is reused evidence, not free work or a fresh measurement. Do not infer
GPU device time, battery savings, or service latency from the host timings.

If a guard restores a word, the changed input boundary mattered in this case.
It does not isolate phonetic context from decoding context or position effects.
If it fails, reject this guard's sufficiency on that state. Do not start a search
over more windows. Choose the next change from the full set of retained results.

## Recorded result

One T4 attempt completed all three new observations. The two existing controls
were reused unchanged. Source was frozen at
`fe64dde23afe1d54ca638d7e0d47a208841f727a`. The
[raw record](../../evidence/modal-t4-tiny-en-context-guard-2026-09-06.json) has SHA-256
`538ff91508b1ea2074a25fb6600860577a917b37d4826f324c2cce2ec8ef71f1`.
The [attempt receipt](../../evidence/modal-context-guard-attempt-2026-09-06.jsonl)
records one start and one successful archive write. The source snapshot identifies
the registration before this result section was added.

All five comparisons remain rejected. All three new observations contain an
exact text/word-token anchor; two restore an anchor absent from the earlier
onset crop. This does not establish a valid continuation boundary.

| State | Observation | Remaining failure |
| --- | --- | --- |
| Paced noisy, reused | Exact anchor, no lexical continuation | A trailing period spans 7260 ms; it is not evidence of speech coverage. |
| No added pauses, new | All word text and word tokens match the earlier onset crop | `For` starts at the new crop origin, 500 ms earlier. The `She`/`she` suffix difference remains. |
| Continuous noise, new | Restores `and bird and tree` | `tree` ends 340 ms later than the frozen end. The suffix has an extra period; the head-only candidate starts before the observed anchor ends. |
| Speaker 2961, new | Restores `danger`; all 15 suffix units match the candidate's text and word tokens | An interior anchor boundary moves 300 ms. The continuation starts 20 ms before the frozen head. |
| Speaker 8455, reused | Name still differs | No exact anchor. |

For the no-pause case, all anchor ends match the frozen ends exactly, despite
the changed start of `For`. Native timestamp-bearing token sequences differ;
only the word tokens are unchanged. For speaker 2961, suffix timing differences
are at most 100 ms. That agreement does not resolve the interior anchor drift or
the boundary crossing. It also does not establish correct recognition: two
decodes can produce the same wrong words.

### Work and lifecycle

The new slices contain 32.80 seconds of audio in total. Each uses two encoder
forwards, one for decoding and one for legacy alignment, with 3000 padded frames
per forward. Feature reuse is disabled to preserve the comparison profile.

| New observation | Host window time | Decoder steps | Peak PyTorch allocated bytes |
| --- | ---: | ---: | ---: |
| No added pauses | 4516.81 ms | 47 | 290098176 |
| Continuous noise | 588.28 ms | 33 | 290098176 |
| Speaker 2961 | 352.60 ms | 35 | 290098176 |

The first call had no warmup. All three calls closed and restored declared
capacity. Allocation after result-handle release was 160720896 bytes in each
case; this includes the resident model and is not zero GPU use. Model
fingerprints were unchanged. The recorded elapsed interval was 19.59 seconds,
including the observation loop and post-run identity/input checks.
These host durations exclude initial setup and are not billed time, device
time, live latency, or an efficiency benchmark.

After the record was returned, Modal reported app
`ap-2mPUbmVKQc5mcvBKRmEfSg` as stopped with zero tasks. No retry was launched.
The validator, raw archive joins, PCM hashes, effective execution identity and
source-file hashes passed local replay. Legacy provenance limits remain as
registered above.

### Next change

The results distinguish lexical agreement from alignment estimates tied to an
input crop. They do not identify the true acoustic boundary. Adding context
alone, ignoring the first word, or raising one tolerance would not address all
the recorded failures.

The registered guard assessor deliberately checks every anchor boundary. The
existing live word policy already has a limited window-origin start exception;
this experiment neither changes that exception nor measures its live outcome.
The interior shift and suffix disagreement need separate treatment.

Before another GPU run, trace the pinned alignment code with these three saved
counterexamples. Locate where crop-relative estimates enter the frozen stream
boundary. Keep exact PCM, crop and alignment identity attached to timing
evidence; matching text must not permit timing or encoder features to move
between crops. Test crop-edge and interior drift separately. Do not change
publication, audio retention, or legacy defaults until a boundary rule has
specific counterexamples and a defined acceptance test.

The archive regression tests run without audio assets or a GPU:

```sh
PYTHONPATH=src python -B -m unittest tools.test_modal_context_guard.ContextGuardArchiveTests
```
