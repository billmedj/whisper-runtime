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
