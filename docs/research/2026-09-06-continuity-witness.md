# Read-only continuity witness — 2026-09-06

The earlier joint observation contains useful continuity evidence, but does **not**
justify publishing the noisy head-only continuation under the current contract.
This change makes the evidence reusable and the remaining refusals explicit; it
does not add a recovery path, change a timestamp, or retire audio.

## Small composition, not another publication policy

`src/whisper_runtime/adapters/continuity_witness.py` combines the existing
`diagnose_word_anchor`, `diagnose_word_sequence`, `AudioObservation`,
`ContinuousAnalysisIdentity`, and immutable `SessionState` values. It derives
the anchor from the last actual `AlignedPublication`, never from an unpublished
continuation. Callers supply real frozen/current session snapshots and the PCM
covering both windows; exact slice hashes and sample counts are checked.

The diagnostic requires a unique published anchor in the earlier joint window.
It compares the entire following suffix with the candidate prefix starting before
the joint endpoint. It does not search for a convenient subsequence, clip a word
crossing that endpoint, repair punctuation, or fit a new tolerance. A separate
lexical view omits standalone non-alphanumeric units only to explain a raw
representation mismatch; it cannot remove that refusal.

Its immutable result has reasons, existing anchor/sequence diagnostics, and signed
candidate-minus-joint timing deltas. It has no publication, replacement anchor,
state mutation, or audio-retirement operation. Empty reasons means only observed
prefix correspondence: two recognitions can share the same omission.

## Fixed archived replay

Run with the existing cached corpus, without model loading or inference:

```powershell
$env:PYTHONPATH = 'src;tests;.'
python -B -m tools.verify_continuity_witness
python -B -m unittest test_continuity_witness tools.test_continuity_witness
```

The stdout-only reproducer pins the exact paced and prompt archive file hashes,
uses the existing source-case PCM verifier and archive parsers, checks actual
COMMIT events against the complete saved publication history, and replays that
history through `Session.from_snapshot`. The selected observations are fixed:

- Paced archive `noisy-prefix/retry`, zero-based `decision_traces[3]`
  (`decode_index=4`), joint window `0..8000 ms`.
- Prompt archive `head-only/null`, initial decode attempt, `3680..10890 ms`.
- Actual published state: version 2, head `3680 ms`. The old observation's words
  `[4:8]`, `place amidst the tents`, match this published anchor uniquely and
  exactly. The following period was **not** published.

The two receipts have equal model/checkpoint bytes, backend commit/tree, and
recorded effective decode/preprocessing/alignment fields; all 35 common dispatched files
have equal hashes. The later receipt additionally records `reuse_decode_features`
and tokenizer EOT; this replay uses its initial decode and legacy alignment, not
its prompted redecode. Derived identity hashes describe recorded metadata, not
a new execution attestation or transfer of session ownership. The hash-pinned
diagnostic worker declares backend label `pytorch-cuda-noisy-context-prompt`;
the published model declares `pytorch-cuda-eof-context-retry`. Both labels remain
intact: their full `ContinuousAnalysisIdentity` values are **not equal**, despite
matching numerical provenance. No human reference
or recognition score is passed to the witness. The original receipt's known
control-comparison flag is neither consulted nor edited.

Relevant byte identities, also emitted by the reproducer:

| Input | SHA-256 |
| --- | --- |
| Full admitted noisy PCM | `5276e641f14eaae484a8da6f916e44f9373239fbe70c70ba41a3f1a36f747108` |
| Joint PCM `0..8000` | `d2c40c23c42369a6bbc926b7a8e5667aa469a305728d396f590573bb0cb3545a` |
| Candidate PCM `3680..10890` | `cc9317e38baca23065fe325b32d4c095532fc29ca4b60763c7ed9b6e9f2e39c9` |
| Shared PCM `3680..8000` | `3a41a7a793d040f4e510407769afd7754cf38f46a4316fd4d3f8fb540d6188aa` |

## Exact remaining refusal

The full continuation has 12 joint units versus 10 candidate units. The ten
lexical units and their token sequences match exactly; the raw representations
do not. Important native times are unchanged:

| Unit (token) | Joint start/end ms | Candidate start/end ms | Delta start/end ms |
| --- | --- | --- | --- |
| `.` (13), joint word 8 | 3680 / 4100 | absent | — |
| `For` (1114), joint word 9 | 4100 / 6100 | 3680 / 6100 | −420 / 0 |
| `,` (11), joint word 12 | 6440 / 6660 | absent | — |
| `she` (673), joint word 13 | 6660 / 6740 | 6440 / 6700 | −220 / −40 |
| `happy` (3772), joint word 19 | 7860 / 7980 | 7880 / 8240 | +20 / +260 |

The full report first retains `analysis_identity_mismatch` and
`published_model_mismatch` for those distinct declarations; it does not invent a
transferable session identity. At unchanged 200 ms tolerance the content reasons
are exactly `representation_mismatch`, `lexical_timing_mismatch`, and
`overlap_cuts_candidate_word`. A crop-origin exception for an already published
anchor cannot be repurposed for the unpublished `For`. Excluding `happy` merely
because it crosses 8000 ms would hide the incomplete overlap.

## Integration recommendation and next gate

Keep this diagnostic out of publication and eviction decisions. The archive
already supplies a joint lexical observation; another matching transcript alone
does not answer the unresolved boundary question. Prompt history is conditioning,
not an independently observed acoustic join. The current live identity receipt
also lacks verified artifact/effective-alignment fields and therefore remains
incomplete for this witness.

Under the unchanged contract, a future source-bound, unprompted joint observation
must reproduce the actual published anchor and whole continuation units with
compatible native timing, without cutting the compared last word. It must still
pass the ordinary publication path; witness agreement is not sufficient.

If repeated measurements show that stable text cannot share one exact word-time
boundary, the minimum alternative is an **explicitly versioned output contract**
separating textual commitment, subtitle-time uncertainty, and processed source
coverage. Such a contract needs independently supported boundary/coverage
evidence and omission counterexamples before it can justify PCM eviction. Do not
silently drop punctuation, widen 200 ms, promote unpublished words to an anchor,
or treat a DTW onset as proof of the first audible phoneme. No new GPU run is
required or authorized by this note.

Validation: 20 pure synthetic tests and 4 cached-archive tests pass, including
omission, repetition, punctuation/token/case changes, timing shifts, overlap cuts,
PCM/identity changes, stale state, zero-duration words, shared-omission limits,
immutability, byte-preserved archives, and reference-text mutation invariance.
See also [group correspondence](2026-09-06-group-correspondence.md),
[resolution handoff](2026-09-06-resolution-handoff.md), and
[prompt diagnostic](2026-09-06-noisy-context-prompt.md).
