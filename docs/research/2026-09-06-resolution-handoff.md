# A bounded structural handoff assessment

`tools/word_anchor_reconciliation.py::assess_resolution_handoff` now makes the
missing handoff evidence testable. It reads immutable observations; it cannot
decode, publish, advance the frozen prefix, evict PCM, or complete a stream.
The original refusal reason and anchor diagnostic remain in every assessment.
No reference transcript is an input. Existing resolution evidence is unchanged.

## One overlap, one complete continuation

The current EOF `resolution_probe` observes only `[committed head, EOF]`.
That candidate cannot itself reproduce an anchor ending at its left boundary.
The assessment proposes one future comparison window: the first frozen anchor's
estimated start, rounded down to the absolute 20 ms grid, through the same EOF.
It must be distinct from both existing windows and wholly within retained PCM.
This is a fixed alternative to test, not a new automatic retry or an implicit
increase to an experiment's decode budget. An attempted but absent witness is
exhausted; the helper never searches other starts.

An observed overlap is structurally eligible only when:

- Its original source, frozen anchor, session version, and exact window agree;
  the live version, retained origin, and committed head have not changed.
- Both observations' SHA-256 hashes match their exact slices of supplied
  retained mono 16 kHz s16le PCM. The original refusal's full-slice hash and
  sample count are also checked when its audio observation is present; otherwise
  `original_source_pcm_correspondence` remains explicitly unproven and that
  source trace is trusted caller input.
- The frozen anchor contains at least two lexical words, ends at the head, and
  has exactly one contiguous raw-text/token occurrence in the overlap. A second
  occurrence anywhere rejects the witness before selecting by timestamps.
- Every anchor start and end stays within the existing 200 ms tolerance. No
  onset exception or terminal-end extension is stacked onto this rule.
- The entire remaining overlap suffix agrees with the entire head-only result:
  exact raw text and tokens, including punctuation and repetitions, with every
  start/end within the same tolerance. Both continuations begin after the frozen
  head and observed anchor end. No subsequence may be chosen to hide disagreement.

The result always has `publication_authorized=False`. Eligibility still lists
`acoustic_boundary_coverage` and `model_tokenizer_options_provenance` as missing:
the observation type carries neither an acoustic proof nor the exact model,
tokenizer, and options identity. Matching PCM hashes bind bytes, not the actual
computation that produced the words. The proposed estimated onset may itself cut
speech. Agreement by two recognitions is not independent acoustic ground truth.

## What the frozen cases actually establish

The tests replay all four cells in the unchanged
[terminal-window record](../../evidence/modal-t4-tiny-en-word-resolution-2026-09-06.json).
All return `needs_overlap`; none contains the required new overlap observation.
Their proposed windows follow the fixed rule above, without examining references.

| Recorded cell | Original anchor diagnostic | Future window, ms |
| --- | --- | --- |
| Context, no added pauses | Lexical missing | 20,720–33,660 |
| Context, continuous noise | Timing mismatch | 32,400–43,660 |
| 2961 clean | Timing mismatch | 1,120–8,220 |
| 8455 noise | Lexical missing | 1,100–7,740 |

The last two prefixes were constructed from past-only bootstrap observations,
not real stream commits. Their previous held-out designation is historical;
using them to develop this rule does not create a new held-out validation.
The replay uses a local version tag, not a claimed archived live-session version;
it explicitly lacks supplied PCM correspondence. Native raw words and original
failure diagnostics are preserved. No new overlap was decoded in this change.

Synthetic tests reject stale versions/spans, wrong PCM, repeated or relocated
anchors, token/punctuation differences, and a boundary word seen only by the
overlap. A negative test deliberately lets both candidates omit the same spoken
boundary word: the arrays can still agree structurally. This demonstrates why
structural eligibility cannot authorize recovery or omission-free completion.

## Next bounded measurement

Register the fixed overlap as an explicitly costed alternative in a future
terminal-state comparison. Keep every disagreement and failed decode; score
full prefix-plus-suffix outputs against human references only after assessment.
Record model/tokenizer/options identity, exact retained PCM/version, padded
encoder work, and verification costs. A successful local handoff check would
still not demonstrate recovered live continuation: publication and EOF coverage
remain separate, unresolved authority boundaries.

CPU reproduction: `PYTHONPATH=src python -B -m unittest tools.test_word_anchor_reconciliation`.
No GPU, network execution, runtime publication change, or historical evidence
rewrite is part of this assessment.
