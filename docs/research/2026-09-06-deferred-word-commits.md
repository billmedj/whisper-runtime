# Deferred word commits with a bounded fallback reserve

The opt-in `defer_word_commits=True` scheduling option prevents the known short
noisy case from irreversibly rebasing at an early estimated word boundary. It
does not recover an already failed/committed session, repair native timestamps,
change word comparison, borrow a reference transcript, or authorize a new join.
Existing configuration defaults and profile IDs remain unchanged when disabled.

## Rule and explicit tradeoff

The option requires the existing hybrid `word_boundary_fallback` profile and
adds `+deferred_word_commit/v1` to its identity. Quiet endpoints and EOF still
close through the ordinary input-evidence, alignment, transaction and release
checks. During an open source unit, earlier text remains provisional.

For window limit `W`, preview interval `I`, and retained left context `L`, reserve
`R = max(I, L)`. The fallback pair is admitted at offsets `W-R-I` and `W-R` from
the retained origin. Configuration requires `R+I < W`. Both coalesced and
non-divisible-interval schedules are clamped so neither observation is skipped.
The existing word policy alone decides whether that pair can commit. Failure
can still consume the reserved interval and reach an explicit unresolved limit;
there is no forced cut or confidence override.

Before the pair, alignment is skipped only when the analysis begins exactly at
the committed head: the whole native result can then serve as provisional text.
After a rebase, previews still use alignment to exclude already published words.
No early preview becomes an agreement witness for the fallback pair.

The tradeoff is deliberate: visible provisional text still updates, but stable
commits wait for a quiet endpoint, fallback, or EOF. A no-pause stream can therefore
wait about 28 seconds for its first stable commit with the tested configuration.
This is not a word-latency improvement or guaranteed noise/long-session solution.

## Exact CPU comparison

Both arms used the same existing cached tiny.en FP32 CPU backend/checkpoint,
seed 7, English decoding, timestamps enabled, and this configuration; the only
arm difference was `defer_word_commits`:

```python
ContinuousStreamConfig(
    preview_interval_ms=2000,
    max_window_ms=30000,
    max_buffer_ms=40000,
    holdback_ms=2000,
    timestamp_tolerance_ms=200,
    left_context_ms=2000,
    coalesce_previews=False,
    source_units=True,
    input_evidence=True,
    endpointing=QuietEndpointConfig(),
    word_boundary_fallback=True,
    word_context_limit_ms=6000,
    eof_context_retry=True,
    defer_word_commits=True,
)
```

Input was admitted in accelerated 2-second chunks, with owner work drained
between chunks; this is **not** source-paced latency. No human reference influenced
scheduling or publication. The reported word edits use it only after commits.

| Exact cached input | Baseline | Deferred with reserve |
| --- | --- | --- |
| Noisy prefix, 10.89 s | Stops at 3.68 s; 18/26 word edits; no FINAL | All 174,240 samples committed; 1/26 edits; FINAL; empty buffer |
| No added pauses, 33.66 s | All 538,560 samples committed; 6/88 edits | All 538,560 samples committed; same 6/88 edits |

No quiet endpoints were detected on either input. The new long-case pair is
`0..26000` and `0..28000 ms`; its first ordinary commit covers through `25580 ms`
(409,280 samples), followed by EOF coverage. Model resources were released in
every cell. Early alignment is absent from 5/6 short-case windows and 12/17
long-case windows. These are observed preparations, not an instrumented encoder
count or a speedup claim. The separate CPU runs have variable wall time.

Original JSON stdout for the two successful candidate cells is preserved in
`evidence/native-cpu-deferred-word-commits-candidate-2026-09-06.jsonl`; the baseline
long control is in
`evidence/native-cpu-deferred-word-commits-baseline-no-added-2026-09-06.jsonl`.
The first, rejected 28/30-second strategy is preserved in the original task stdout
and an explicitly labelled extracted summary
`evidence/native-cpu-deferred-word-commits-initial-failure-summary-2026-09-06.json`.
It hit a conflicting native speech score at 30 seconds and committed nothing.
Its reporter's post-`close()` `done=True` meant abandonment, not completion;
the corrected reproducer captures completion before closing and records FINAL.

Rerun without downloads or GPU:

```powershell
$env:PYTHONPATH = 'src;tests;.'
python -B tools/verify_deferred_word_commits.py --arm both --model "PATH/TO/EXISTING/tiny.en.pt"
python -B -m unittest test_deferred_word_commits
```

The parent process imposes a 240-second worker deadline and each cell an
80-second cooperative limit. Paths must already contain the verified setup,
checkpoint and corpus. The tool prints JSON only and does not create evidence
files. Full 43.66-second noise, paced T4 and broader acoustics remain separate
qualification gates; do not infer their result from the short noisy success.

## Counterexamples retained

Fourteen deterministic tests cover strict opt-in/profile identity, skipped early
alignment without eviction, ordinary fallback authority, coalescing, non-divisible
intervals, larger context reserve, quiet and EOF closure, nonlexical EOF,
conflicting speech scores, unstable fallback, missing anchor after rebasing,
33-second uninterrupted input with successful ordinary fallback, removal of
published context from previews, and cleanup-failure fencing. No old acceptance
gate or existing default profile is relaxed by this option.
