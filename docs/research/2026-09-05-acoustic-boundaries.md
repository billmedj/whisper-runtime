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

**The grouped diagnostic fails.** The control, normal hybrid and attenuated
hybrid cells complete. The two inputs without detected quiet endpoints do not.
No acoustic qualification or live-performance improvement is established.

The [T4 record](../../evidence/modal-t4-tiny-en-acoustic-boundaries-2026-09-05.json)
uses source commit `14e5218b4b7dd864c5e8ec2a3044ed7a678b15f6`, cached `tiny.en`
and FP32. It follows a successful
[CPU transport check](../../evidence/modal-acoustic-preflight-2026-09-05.json).
One GPU invocation runs all five cells and their matched offline controls.

| Cell | Input | Accepted | Committed | Decodes | Human-reference edits | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy quiet control | 43.66 s | 43.66 s | 43.66 s | 24 | 6 | Pass |
| Hybrid, same input | 43.66 s | 43.66 s | 43.66 s | 24 | 6 | Pass |
| Hybrid, no added pauses | 33.66 s | 33.66 s | 21.26 s | 17 | 36, partial | Fail |
| Hybrid, continuous noise | 43.66 s | 35.00 s | 6.10 s | 18 | 79, partial | Fail |
| Hybrid, attenuation by 32 | 43.66 s | 43.66 s | 43.66 s | 24 | 5 | Pass |

Each offline control has six edits against the same 88-token human reference.
The normal hybrid matches its control after normalization. Attenuation has one
fewer edit on this input; this does not establish an accuracy advantage.
The failed cells compare partial text against the full reference. Their edit
counts include uncommitted speech and must not be reported as completed-stream
recognition error rates. Regex normalization splits apostrophes, so these counts
are not counts of distinct misrecognized spoken words.

Both failures preserve contiguous committed coverage and retain all accepted,
uncommitted PCM. Neither emits a final event or claims success. All cells restore
native capacity. The model fingerprint is unchanged. Preserved sample accounting
does not establish that Whisper recognized every word in a committed interval.

## What the failures show

### A reanalyzed window can lose its lexical continuation

In the no-added-pauses case, the last commit ends at 21.26 seconds. Two seconds
of left context remain, so the next analysis starts at 19.26 seconds, inside
the previously recognized word `amidst`. The anchor is `. For a while`.

The analysis ending at 30 seconds contains that anchor and more speech. The
preceding and following analyses do not provide the required agreement. At EOF,
Whisper emits only `It's the tents.` from the retained window. The continuation
is absent from that native result. Relaxing timestamp comparison cannot recover
words the model did not emit.

The controller stops with 12.40 seconds of unpublished input and two seconds of
published left context retained. This is a recognition-context failure combined
with a conservative publication rule, not an input-queue loss.

### A fixed-duration overlap can leave one fragile anchor word

In the noisy case, no quiet endpoint is detected. The commit ending at 6.10
seconds publishes `. For`. Rebasing to 4.10 seconds removes the punctuation
anchor, whose estimated start was 3.56 seconds. Only `For` remains as a complete
retained anchor word.

Its frozen estimate is 5.88-6.10 seconds. The next growing analyses retain the
same token and end time but place its start at the window edge, 4.10 seconds.
The existing edge-matching exception requires at least two anchor words. It
therefore rejects this single-word relocation. Matching one frequent word
elsewhere would risk omitting or duplicating speech, so the guard stays intact.

The controller stops with 28.90 seconds of unpublished accepted input and two
seconds of left context. The remaining 8.66 seconds of the fixture were never
admitted. They are not silently discarded accepted input.

## Verification and resources

Before the run, 534 runtime tests and 352 repository-tool tests passed, with
strict typing and Ruff checks. The native backend, Lean model and model weights
were not changed. No package was rebuilt.

After the run, two tests were added from the observed word traces. They preserve
the single-retained-anchor guard and reject a continuation that appears in only
one intermediate analysis, then disappears at EOF. All 536 runtime tests pass.
These tests document the failures; they do not repair them. The runtime source
remains the version used by the T4 run.

A second coding agent reviewed the saved artifacts. It checked all 28 frozen
source files, reconstructed all four inputs byte-for-byte, verified every native
analysis PCM hash, joined selected words to publications, and recomputed the
reference edits and sample accounting. This is an independent artifact review
within the development session, not an external security or acoustic audit.

The CPU application `ap-9BkHuuvHC4FmDdBU2dRhEH` and GPU application
`ap-NN6fsNoC8RgX2tzlkJa3K9` were checked as stopped with zero tasks. No further GPU
run was launched. No model or corpus download was needed.

Observed account metering increased from $0.18042229 to $0.19042229 across the
CPU/GPU work. The ephemeral-app subtotal increased by $0.01046169. These counters
can update at different times; they are not a per-run invoice or a current credit
balance. The observed cost increase was approximately one cent.

GPU record SHA-256:
`37ccc47ba7854523e06c54b7bcf809afc76de627233168ce9a4adde9f335a2bc`.

CPU record SHA-256:
`a0d22ca6cf3af49c17d3160406e156241daabdc44de266d77d361dbcb8c0b029`.

Source snapshot digest:
`884136f2e1920432fc5ccc89951a16284228349fce545f083d217886320c6244`.

The JSON records and attempt journals were copied to `evidence` without overwrite.
Each copy matches the source hash. The compressed GPU response is retained in
local artifacts with SHA-256
`d3d2d0fc29691f73e47e813c086b1d23f1619dad853a04ea52874e17aa17030b`.

## Next change

Make retained context depend on complete published anchor words, within an
explicit audio bound, rather than cutting it at an arbitrary two-second offset.
Test the one-word anchor and mid-word rebase cases locally before another GPU
run. A window must not move unless the retained evidence supports the next
publication check; unresolved input must still produce explicit backpressure.

This is a candidate repair, not a demonstrated solution to the missing native
continuation. If the model still omits speech, evaluate a bounded alternative
analysis window and record its added work. Do not bypass anchor checks, raise
the quiet threshold, or mark the remaining input complete to force a pass.

Keep this five-cell diagnostic as the next acoustic gate. Only after it passes
should the same profile advance to paced network replay and longer sessions.
