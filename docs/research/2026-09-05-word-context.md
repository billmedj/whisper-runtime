# Bounded word-context diagnostic plan — 2026-09-05

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
