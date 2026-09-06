# Separate recognition differences from handoff decisions

## Implemented change

`diagnose_word_sequence` is a small, pure helper in the existing word-policy
module. It reports complete-unit text relation, per-unit token equality, and
maximum absolute differences in estimated start and end times. It does not
select words, rewrite output, or change a publication rule.

Case-fold equality describes strings only. `US` and `us`, for example, can
case-fold equally without having the same meaning. Whitespace, punctuation,
unit boundaries, and raw tokens remain intact. Timing differences are reported
only for nonempty, equal-length sequences whose corresponding units case-fold
equally. Other cases return unknown timing, not zero. No subsequence search,
reference transcript, model call, new dependency, or runtime cache is involved.

The [CPU replay](../../evidence/resolution-disagreements-2026-09-06.json) reads
the byte-hashed [seven-window T4 archive](../../evidence/modal-t4-tiny-en-resolution-handoff-2026-09-06.json).
It preserves every original refusal. A complete-suffix comparison requires one
unique exact raw text/token anchor. Missing or repeated anchors leave that
comparison unavailable. This supplies diagnostics, not a replacement assessor.

## What the saved observations establish

The no-added-pauses pair has 35 suffix units on each side. Only capitalization
and the corresponding token differ; maximum timing differences are 60 ms. The
new helper exposes these facts separately. The original rejection remains.

The noisy overlap has no lexical unit ending after the frozen boundary, but
assigns 7,260 ms after its last lexical estimate to punctuation. The head-only
candidate has 18 lexical units after the boundary. These are descriptions of
the model's estimates, not measurements of speech coverage or silence.

An earlier-start control already exists for this noisy case. It restores the
four-word anchor with start/end differences of at most 40/80 ms, yet still
omits the next utterance. Its suffix has one punctuation unit versus the
head-only candidate's 19 units. **Restoring an anchor is not sufficient to
restore a continuation.** The tighter cut cannot be the sole explanation for
this omission; both the tighter and earlier-start observations omit it.

The speaker-8455 earlier-start control also exists. It says `Eve as` where the
frozen anchor says `Eva's`, and therefore still lacks the exact anchor. This
does not establish a general recognition limit or diagnose its acoustic cause.

## Bounded context comparison, not yet executed

Fix a 500 ms guard before the estimated anchor onset, rounded to the existing
20 ms grid and clipped to retained audio. Keep the same EOF. The 500 ms guard
is an experimental design constant, not a calibrated optimum. Do not choose
new starts after inspecting a result.

| State | Guarded interval, ms | Available observation |
| --- | --- | --- |
| Paced noisy | 1680–10890 | Existing fresh control from the seven-window run |
| Context, no added pauses | 20220–33660 | New window needed |
| Context, continuous noise | 31900–43660 | New window needed |
| Speaker 2961, clean | 620–8220 | New window needed |
| Speaker 8455, noise | 640–7740 | Existing archived control from the word-resolution run |

Only three intervals lack a recorded observation. Interval equality alone does
not authorize reuse: check exact PCM, model, options, seed, and backend identity.
The two reusable slices match their saved hashes. The speaker-8455 observation
retains the earlier source-checked legacy bridge: backend trees differ and old
package identities remain unmeasured. No new source identity is invented.

A future comparison must retain the old controls and failures, use legacy
alignment, prohibit retries, and record every observation's cost. Compare raw
anchor uniqueness, estimated times, and complete suffixes separately. Do not
case-fold acceptance or change the existing 200 ms tolerance. References can
score proposals afterward; they cannot select a window or a correspondence.

If a guard restores a missing word, it shows sensitivity to the changed input
boundary. It cannot by itself distinguish recovered phonetic context from
changed decoding context or position. If it fails, reject this guard's
sufficiency, not every possible boundary mechanism. The saved noisy control
already rules out this guard as a universal recovery solution on these cases.

No GPU work was launched for this CPU follow-up. No additional framework,
automatic retry, live publication, or recognition improvement is claimed.

## Reproduce locally

With `PYTHONPATH=src`, run:

```sh
python -B -m tools.analyze_resolution_disagreements evidence/modal-t4-tiny-en-resolution-handoff-2026-09-06.json
python -B -m unittest tools.test_analyze_resolution_disagreements
python -B -m unittest discover -s tests -p test_word_sequence_diagnostic.py
```

The replay needs no audio assets, torch, Modal client, or network. It writes
JSON to standard output and leaves the input archive unchanged. Its source
hashes identify the diagnostic code, not the code that ran the historical GPU
observations.
