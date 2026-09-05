# Continuous speech and quiet boundaries

## Problem

The quiet-endpoint profile publishes a complete source unit after an observed
quiet interval or source EOF. An open unit produces previews only. Speech or
background noise can prevent a quiet endpoint, so the unit reaches the analysis
limit without a commit. The runtime then stops and retains uncommitted audio.

A forced cut at the window limit would permit progress without establishing a
safe word boundary. Raising the quiet threshold could classify weak speech as
a pause. Neither change addresses the publication problem.

## Implementation under test

Combine the existing word-agreement policy with observed quiet endpoints in a
separate opt-in profile. During an open unit, two growing analyses can publish
an agreed lexical prefix. Retain left context and match already published words
before selecting the next suffix. At a quiet endpoint, resolve the remaining
suffix under an explicit closed-input contract. Only source EOF ends the stream.

An endpoint proposes an input boundary; it does not prove silence or recognition
accuracy. Digital-zero input retains its separate empty-publication rule.
Uncertain input never grants permission to discard accepted audio. Unresolved
word anchors can still stop the stream rather than allow a guessed continuation.

Existing source-unit and word-aligned profiles keep their defaults. Native
transactions, ownership, completion fences and the transport remain unchanged.

## Diagnostic scope

Use the existing three licensed LibriSpeech clips and their human references:

- The registered 43.660-second mixture with inserted digital-zero pauses.
- The same six speech occurrences without the inserted pauses.
- The original mixture with deterministic low-amplitude noise added throughout.
- The original mixture attenuated to stress weak-signal handling.

These are controlled transformations, not a natural conversation corpus. Removing
inserted pauses does not remove silence inside the original clips. Noise can
prevent this detector from proposing an endpoint without establishing continuous
human speech. Record the exact transformations, hashes and sample counts.

Run one legacy quiet-profile control and four new-profile cells on a single
cached `tiny.en` T4 worker. Compare each cell with the full-input offline model
control and the independent reference. Record completion separately from text
quality: full sample coverage is not proof of complete speech recognition.

The qualification requires full completion and valid publications, with no
additional normalized reference edits compared with the matched offline control.
Preserve every failure. Source-time replay is unpaced and establishes no live
latency or compute advantage.

## Execution order

1. Reproduce the no-endpoint stop in a local regression test.
2. Test word publication, quiet closure, rebasing, cancellation and cleanup.
3. Verify deterministic audio construction and the diagnostic locally.
4. Freeze the source and registration. Check CPU result transport.
5. Run one T4 diagnostic, without an automatic retry.
6. Review the raw results before recording a qualification outcome.

The T4 function has a 180-second execution timeout, one container and no
configured retries. Resource limits are not a billing cap; startup and provider
rescheduling have separate behavior. No new model download is required.

## Result

Pending. No acoustic qualification is claimed by this plan.
