# Word confirmation with right context in both observations

## Result

The `low-latency-v2` CPU replay completes the 46.55-second diagnostic input that
failed with `low-latency-v1`. It commits all 744,800 samples, emits one FINAL,
empties the audio buffer and returns native capacity. The first nonempty commit
occurs after 6 seconds of admitted audio. The transcript has 7 word edits against
114 reference words, matching the standard control's count.

This is accelerated CPU replay, **not live latency**, GPU qualification or
endurance. It uses a known diagnostic input, not held-out accuracy coverage.
The [event audit](../../evidence/two-observation-cpu-2026-09-07.json) reconstructs
the committed transcript and TXT/SRT/VTT exports from the recorded events. It
also checks the runtime source hashes and terminal receipts.

## Failure and correction

The earlier T4 attempt confirmed its first caption at 7.746 seconds but stopped
with only 38.10 of 46.55 seconds committed. A later alignment moved the end of an
already confirmed word by 560 ms, beyond the unchanged 200 ms tolerance.

The two observations used to confirm that word were not equally mature. The
earlier one had only 320 ms of audio after the word. The policy applied its
two-second holdback only to the later observation. Two similar estimates near a
window edge were enough to freeze a boundary that later estimates contradicted.

The new profile sets `previous_holdback_ms=2000`. Each candidate word must now
have two seconds of right context in both observations. It still needs matching
words, bounded timestamp differences and supported audio. Actual EOF retains
the existing explicit completion and frozen-anchor checks. No model weights,
decoder thresholds or timestamp tolerances change.

The parameter defaults to zero. Existing profiles retain their settings, and
older checkpoints decode the missing field as zero. The new profile has a
distinct execution ID and publication-policy suffix.

## Boundary recovery

A rejected automatic quiet-endpoint hint no longer has to stop the stream when
there is room for more analysis. After native cleanup, the scheduler can discard
that hint while retaining all PCM, committed text and frozen anchors. This does
not apply to caller boundaries, actual EOF, unsupported audio or the hard window
limit. The rejected result publishes no text.

That recovery alone did not solve this recording. Three isolated CPU windows
either repeated the timing mismatch or produced only punctuation after the
anchor. Moving the retry start to an observed word boundary also produced only
punctuation. All four candidates were rejected and closed cleanly. These negative
results motivated preventing the premature confirmation rather than weakening
the later anchor check.

The successful full replay uses 25 native analyses and no EOF context retry.
Its peak retained buffer is 459,520 samples, below the 640,000-sample limit.

## Remaining qualification

The first v2 T4 attempt stopped on source lateness: 360 ms at frame 306,
above the unchanged 250 ms limit. It accepted 6.12 seconds of audio before
stopping. No completed transcription was qualified. Cleanup was verified and
the provider later reported a stopped app with zero tasks.

That delay occurred during the first CUDA DTW call, which also invokes a lazy
CPU backtrace compiler. A local probe measured 1.71 seconds for its first dense
int32 call and 0.45 seconds for its first strided call; repeated calls took tens
of microseconds. Heartbeat gaps were 104 and 47 ms. These measurements do not
establish the cause of the full 360 ms GPU-run delay. The next experiment moves
those two compilations before readiness without warming model inference or
changing any source-clock gate.

The [subsequent T4 check](2026-09-07-low-latency-v2-gpu-results.md) now passes
both source-paced sessions after explicit startup compilation. It meets the
registered eight-second first-commit target, complete coverage, word-edit gate
and cleanup requirements.
Native endurance, physical capture and clean-platform installation remain
separate release gates. The completed short replay does not close them.
