# Bounded word-context comparison

Plan frozen on 2026-09-05. GPU execution completed on 2026-09-06 (Asia/Singapore).

Hypothesis: the no-added-pauses and noise64 failures in the frozen acoustic-boundary
diagnostic stem from context beginning inside estimated words or losing necessary
committed anchors. An opt-in word-aware retention bound may restore completion;
this is a testable hypothesis, not a recognition-improvement claim.

Keep the quiet-endpoint/hybrid behavior and `left_context_ms=2000` unchanged by
default. The candidate enables `word_context_limit_ms=6000` only with
`word_boundary_fallback=True`: 2000 ms remains desired overlap, while 6000 ms caps
backward snapping to whole estimated words and retention of overlap anchors.
The next anchor retains at least two lexical words and omits leading standalone
punctuation only from that anchor; published text, internal punctuation, native
tokens/times, and the matcher remain unchanged.
Alignment times remain estimates. Existing publication, coverage, and quality
checks are not relaxed. The distinct profile appends `+word_context/v1` before
`+input_evidence/v1`.

Use `experiments/modal-acoustic-diagnostic-v2.json` and the existing harness.
Preserve the v1 registration and evidence from commit `14e5218`. Freeze and commit
the candidate before exactly one CPU transport preflight and one explicitly
authorized grouped T4 call, with the same frozen source snapshot for both.
Use the cached `tiny.en` FP32 model, unchanged four input recipes and hashes,
1000 ms chunks, seed 7, four per-input offline controls, and these ordered cells:

1. `quiet:mixed-control` — legacy completed control, six human edits / 88 words,
   zero normalized edits against its offline control.
2. `hybrid:concatenated-no-added-pauses` — reproduce safe `policy_resolution` failure.
3. `hybrid:mixed-continuous-noise64` — reproduce safe `policy_resolution` failure.
4. `context:mixed-control`.
5. `context:concatenated-no-added-pauses`.
6. `context:mixed-continuous-noise64`.
7. `context:mixed-attenuated32`.

Each candidate must emit one real final event, commit full input, pass every
integrity check, restore native capacity, and have no more human-reference word
edits than its own offline control. Require all seven cells and an unchanged model.
Policy controls must preserve all non-completion checks and restored capacity;
they remain unqualified. Unexpected baseline outcomes invalidate the matched
comparison. Record `control_outcomes`, `controls_match_expected`,
`candidate_qualified`, and `all_cells_qualified` separately. Top-level
`qualified` / `status` refers explicitly to candidate qualification with matched
controls, never to all seven cells passing.

Retain exclusive attempt receipts, compressed raw results, source/registration
hashes and worker identity. No automatic retry or extra GPU run; a policy failure
continues to the next cell, while a lifecycle failure stops. Report actual results
without latency, GPU efficiency, production, or beyond-fixture generalization claims.

## Result

**The candidate does not qualify.** The noisy input progresses from 6.10 to
33.54 seconds of committed coverage. Both no-pause cases still stop before EOF.
Normal and attenuated inputs complete within their offline-relative quality
gate. Existing controls reproduce the preceding failures.

The [GPU record](../../evidence/modal-t4-tiny-en-word-context-2026-09-06.json)
binds source commit `b0e15ebcf510135278a7f9e6b59e31417b0a7edb`.
`candidate_qualified=false`, `all_cells_qualified=false`, and
`controls_match_expected=true`. All seven cells ran on one worker with unchanged
model state and reported restored native capacity. The two expected control
failures remain failures; they were not reclassified as passing cells.

| Cell | Input / accepted | Committed | Decodes | Human-reference edits | Qualified |
| --- | --- | ---: | ---: | ---: | --- |
| Quiet control | 43.66 / 43.66 s | 43.66 s | 24 | 6 | Yes |
| Fixed context, no added pauses | 33.66 / 33.66 s | 21.26 s | 17 | 36, partial | No |
| Fixed context, noise | 43.66 / 35.00 s | 6.10 s | 18 | 79, partial | No |
| Word context, normal | 43.66 / 43.66 s | 43.66 s | 24 | 6 | Yes |
| Word context, no added pauses | 33.66 / 33.66 s | 21.26 s | 17 | 36, partial | No |
| Word context, noise | 43.66 / 43.66 s | 33.54 s | 22 | 22, partial | No |
| Word context, attenuated | 43.66 / 43.66 s | 43.66 s | 24 | 6 | Yes |

The four offline controls each have six edits against the 88-token human
reference. Completed candidate transcripts match their controls after
normalization. The preceding v1 attenuated candidate had five human edits;
this candidate has six. There is no demonstrated recognition improvement.

Partial edit counts compare unfinished text with the full reference. The lower
noise count partly reflects further progress, not an independently measured
accuracy gain. Contiguous source coverage does not prove recognition without
omissions. Previously published text stays immutable; all accepted, uncommitted
audio remains retained in the failing cells. Neither emits a final event.

Post-run inspection finds that every committed transcript is an exact normalized
word prefix of its matched offline control. Noise advances from 9 to 69 of the
control's 86 words; the 17-word offline distance is the unpublished suffix.
The nine previously published words remain unchanged. No additional lexical
regression appears in that prefix relative to the offline model, which can
itself be wrong. This observation does not change the failed acceptance result.

## The remaining failures are different

### No added pauses: correct estimated cut, missing native words

The last cut moves from 19.26 to 19.18 seconds, the observed start of `amidst`.
The retained lexical anchor is `For a while`, with exact original tokens and
times. At EOF, the native result still contains only `It's the tents.`. It lacks
both that anchor and the remaining speech. The controller cannot publish a
continuation absent from the native result.

Thus avoiding an estimated mid-word cut is not sufficient recognition context.
The candidate retains 12.40 seconds of unpublished audio plus 2.08 seconds of
published context. Its committed text is identical to the matched failing
baseline. No policy threshold was relaxed to claim completion.

### Noise: native continuation exists, but an anchor end moves

The earlier one-word blockage is passed. The new final analysis starts at
31.16 seconds, with committed coverage through 33.54 seconds. Its frozen anchor
is `and bird and tree`. At EOF, those exact words and tokens are present, along
with a continuation.

The frozen `tree` estimate is 33.18-33.54 seconds. The final estimate is
33.16-34.08 seconds: the start differs by 20 ms, but the end differs by 540 ms.
This exceeds the unchanged 200 ms matching tolerance. Publication is rejected.
The record also contains earlier lexical/casing variation; do not describe the
whole sequence as a timing-only failure. The final rejection has a specific
timing mismatch rather than a missing native suffix.

All 43.66 seconds were admitted. The buffer retains 10.12 seconds of unpublished
audio plus 2.38 seconds of context. Increasing every timestamp tolerance would
also change repeated-word and boundary safeguards; this result does not justify
that change.

## Work and resource limits

This unpaced diagnostic is not a live or performance benchmark. The noisy
candidate makes 22 decodes versus 18 in the baseline, reaches further, and
submits 270.78 versus 306.90 summed source seconds to decoding. Its peak PCM
buffer is 26.36 versus 30.90 seconds. These counters cover different amounts of
completed work and exclude padded encoder work and process/device memory.
They do not establish a GPU saving.

The no-added-pauses candidate does more repeated source analysis (190.08 versus
177.34 summed seconds) without improving final coverage. Context retention is
not uniformly beneficial.

One CPU transport check and one seven-cell T4 invocation were run, with cached
model and input assets, no configured retry, and no further GPU call. The GPU
application `ap-gvHCnzOs7gTPlobI212fwT` stopped with zero tasks at 00:03:44 +08:00.
The CPU application `ap-A8kL34FBlHmSkqGAUsBco4` also stopped with zero tasks.

Observed account metering rose from $0.19042229 to $0.21042229. The ephemeral-app
subtotal rose from $0.19482851 to $0.20807714. These asynchronously updated
counters indicate roughly two cents of additional metered cost, not an exact
per-run invoice or remaining credit balance.

## Verification and evidence

The runtime suite passes 549 tests, including 13 new context tests. The tools
suite passes 358 tests, including 16 diagnostic tests and six new comparison
checks. Strict typing, Ruff
and repository checks pass. Neither the native backend nor the Lean model changed.
No package was rebuilt or GitHub push performed.

Separate coding agents reviewed the runtime change and the saved artifacts.
The artifact review checked raw/JSON identity, receipts, all 28 frozen source
files, reconstructed inputs, model identity, publication joins and reference
metrics. Reported worker capacity is remote evidence, not a local physical
remeasurement. These development reviews are not an external audit.

The review also reconstructs 146 native analysis PCM hashes and 36 word
publications. All 15 partial context-retention decisions preserve complete
estimated words and at least two lexical witnesses. Observed retained context
is 2.000-3.960 seconds, within the six-second limit. One decision keeps origin
zero; these are not 15 strictly advancing rebases.

- GPU JSON SHA-256: `f96e524e01574b6f2f8df3f9e09932b990313bba9354799a58703da043526b85`.
- CPU JSON SHA-256: `d7a954e672e9acd82ee5e204677f2f75122183d72794ff12bffe880a75b40bc3`.
- Source snapshot: `182fb9819a0bc7aabd95eaeee049e6bcf8753f27795c4a0559236e1a488df408`.
- Retained compressed response SHA-256: `456b40de1f42ec2cc0981af123c6899dde9e562dc0a96c64a47002a064fc1e09`.

The JSON records and attempt receipts were copied to `evidence` without overwrite
and checked against their local source hashes. The compressed response remains
in local artifacts. The candidate stays opt-in; the acoustic gate remains open.

## Next discriminating tests

1. Use the saved noisy alignments to distinguish exact lexical identity from
   timestamp uncertainty. Any proposed rule must still reject relocated repeated
   phrases, changed tokens, missing anchors and overlaps with new speech. Do not
   globally raise the existing tolerance.
2. For the no-added-pauses case, compare a bounded alternative context start with
   the current one. Changed-window recognition requires a fresh model run; cached
   traces cannot predict it. Test whether the omitted words reappear before
   adding a fallback to the runtime.
3. Repeat the full acoustic gate only after a candidate passes its local safety
   checks. Defer longer live sessions and efficiency claims until completion.
