# Group-boundary comparison

## Scope

This is a post-hoc CPU experiment on the five saved context-guard states. It
tests whether a complete word group can remain identifiable when its internal
timing estimates change. It does not run inference or change the live policy.
The speaker-2961 and speaker-8455 prefixes are synthetic bootstrap states, not
previously completed live publication.

The [input record](../../evidence/modal-t4-tiny-en-context-guard-2026-09-06.json)
is byte-hash bound. The [comparison report](../../evidence/group-correspondence-2026-09-06.json)
preserves each original refusal and identifies its analysis source files.
Report SHA-256:
`03e20afc4e03ceb70066748c9e143288aaf33d3e4a7cfe618d63c1a04007064d`.

## Source trace: why the first word follows the crop

The exact timing source used in the T4 record was reconstructed from local Git
objects and experimental patch 0008. It matches patched tree
`32163d5cdb87babc1cd415a86cc5a58116c86a16`, timing blob
`e517e20086585488c5e6668505ac03758c2c213e`, and timing-file SHA-256
`25c9bca776064721b5f8fbe1fff7a4d62cfc9e4f65fa5012d1c5076affc39b93`.

In that source:

1. The dynamic-time-warping path is constrained to start at `(0, 0)`.
2. The first word boundary selects the first path jump. For an ordinary finite
   path, the raw first word therefore starts at time zero within the crop.
3. Our native adapter calls `find_alignment` directly, then adds the crop origin
   to the returned times. It does not call `add_word_timestamps`.

This explains why `For` starts at 20220 ms after the crop begins at 20220 ms.
It is not a measured acoustic onset. The existing live policy already has a
limited exception for this first-word effect; the original guard experiment
deliberately used stricter per-boundary comparison.

Interior boundaries are different. The aligner normalizes and filters
crop-specific attention, then derives adjacent word ends and starts from shared
path jumps. Interior durations can change while the word sequence stays fixed.
The archived record has neither attention matrices nor DTW paths, so the exact
cause of the 300 ms interior shift remains unknown. Rounding milliseconds alone
cannot explain that nonuniform shift.

`add_word_timestamps` applies additional duration, punctuation and segment-boundary
heuristics. Switching to it would change the evidence contract. Those heuristics
are not independent measurements of speech boundaries.

## Rules compared

The two group variants require one exact, ordered occurrence of the complete
anchor, including word text and tokens. They compare the first start and last
end within 200 ms and report internal shifts without rejecting on them. Both
reuse the existing limited first-word origin exception. Neither normalizes
capitalization, removes punctuation, selects a shorter matching suffix, moves
the frozen head, or changes recorded estimates.

The complete continuation must contain lexical text and match the candidate's
raw word text and tokens, with every continuation time within 200 ms. The
variants differ only in backward crossing relative to the later of the frozen
head and observed anchor end:

| Comparison | Anchor timing | Backward continuation crossing |
| --- | --- | --- |
| Original guard assessment | Every word boundary; no origin exception | None |
| Group, strict boundary | Outer bounds, with origin exception | None |
| Group, live-sized tolerance | Same group rule | At most 200 ms |

The last variant borrows the live policy's numerical bound; it is not a replay
of that policy. Also, with ordered spans and the group-end bound, the 200 ms
crossing cap follows already. It is not an independent safeguard. The zero
variant adds a stricter constraint. Comparing the two exposes its effect.

## Results

| State | Original | Group, strict boundary | Group, 200 ms boundary |
| --- | --- | --- | --- |
| Paced noisy | No lexical continuation | Same refusal | Same refusal |
| No added pauses | Anchor timing | Complete suffix differs | Complete suffix differs |
| Continuous noise | Anchor timing | Group end differs by 340 ms | Same refusal |
| Speaker 2961 | Interior timing differs by 300 ms | Continuation crosses by 20 ms | Structurally eligible |
| Speaker 8455 | Anchor absent | Same refusal | Same refusal |

The group rule alone does not make speaker 2961 eligible under a zero-crossing
requirement. Its outer start is unchanged and its outer end differs by 20 ms;
eligibility also needs the separate allowance for that 20 ms crossing. All 15
continuation units match the existing candidate. No new text was generated,
so this is not a recognition-quality improvement or a live recovery result.

Output agreement cannot establish acoustic coverage or source authenticity.
Regression tests deliberately show that both old and new comparisons can agree
on zero-duration anchors or outputs with a substituted window ID. Such results
still have no publication authority. The fixed archive hash binds this replay;
a live producer would also need to validate session state, exact PCM, model and
alignment identity. Shared omissions can survive exact output agreement.

## Additional saved-history validation

The [fixed inventory](2026-09-06-group-validation-inventory.md) was recorded
before scoring. Its [CPU report](../../evidence/group-validation-2026-09-06.json)
reconstructs anchors only from confirmed, ordered historical commits. Group
results cannot change that history. Report SHA-256:
`356a9fcf14771676ce829043d6e637bf7fc99c841d0d1bd69662b869111973b5`.

| Recorded input | Enrolled observations | Strict matches | Group matches |
| --- | ---: | ---: | ---: |
| JFK fixture, repeated three times | 12 | 12 | 12 |
| Speaker 1995, standalone utterance | 1 | 1 | 1 |
| Speaker 672, standalone utterance | 1 | 1 | 1 |

The report includes all 24 traces. Seven have no committed anchor and three
have only one anchor word; these ten exclusions remain explicit. The 14 enrolled
observations represent eight anchor states. JFK supplies one source fixture
outside the five tuning states. The other utterances already occur in the
tuning mixtures, so this is not a statistically held-out evaluation.

Both rules match all 14 enrolled observations. There is no observed regression
or additional gain in this set. Two JFK observations use the existing crop-origin
exception for an 880 ms first-start shift. All interior shifts are at most
160 ms, so this set does not independently test the larger interior shift seen
in speaker 2961. The group rule does not cap interior shifts; it reports them
and constrains only the outer bounds. This tests anchor correspondence, not
complete continuation or live recovery.

Synthetic tests expose a tradeoff absent from these recorded results: the
strict matcher can identify one timed occurrence among repeated phrases, while
the group rule rejects multiple exact occurrences. Group matching must not
replace that working path. Other tests preserve refusals for changed tokens,
case, order, group endpoints and relocated phrases. Tests also show why a match
alone cannot exclude shared omissions or authenticate its source audio.

## Next gate

Keep the runtime unchanged. The next test should exercise the complete proposed
handoff, including the continuation, against a frozen audio-retention boundary.
Retain the strict path when it succeeds. Use group matching only as an explicit
fallback candidate; reject ambiguity rather than rewriting the committed prefix.
Before enabling recovery, establish what permits the audio-retention boundary
to advance. Exact output agreement alone cannot certify acoustic coverage.

When another native diagnostic becomes necessary, retain compact alignment-path
evidence around the disputed boundary. Repeating the same text-only result cannot
identify the internal timing cause. Reuse encoder features only for the exact
same input and execution profile already supported by the adapter.

## Reproduce without a GPU

With `PYTHONPATH=src`:

```sh
python -B -m tools.analyze_group_correspondence evidence/modal-t4-tiny-en-context-guard-2026-09-06.json
python -B -m unittest tools.test_analyze_group_correspondence
python -B -m tools.analyze_group_validation
python -B -m unittest tools.test_analyze_group_validation tools.test_group_anchor_diagnostic
```

The optional `--output` argument creates a new file and refuses to overwrite an
existing file. The tests require no audio assets, model, Torch or Modal client.
