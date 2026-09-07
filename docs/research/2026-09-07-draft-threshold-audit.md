# Native draft score thresholds: local audit, 2026-09-07

## Result

The saved integrated T4 screen has exact gate decisions, tokens, text and events
in both registered pairs. Replaying all 196 saved native results through the real
`assess_audio` policy produces no score-threshold crossing. The smallest relevant
distance to the active threshold is **0.35662387907505033**. Maximum paired score
changes are `1.4156103134155273e-6` for `no_speech_prob` and
`9.5367431640625e-7` for `avg_logprob`; compression ratio is unchanged.

This corpus was far from the publication boundary. It does not qualify boundary
parity. Synthetic, identical-token native results straddling that boundary change
real controller publication decisions. These are deterministic counterexamples,
not observed new-audio failures or new model/GPU runs.

## Actual consumers and comparisons

| Consumer | Comparison or role | Effect in the integrated native lane |
| --- | --- | --- |
| `adapters/audio_evidence.py:192-210` | `no_speech_prob >= 0.6` | Nonzero PCM plus lexical output becomes `uncertain/conflicting_speech_score`; strictly below is only a `speech_candidate`. No average-logprob override. |
| `AudioEvidenceDecision.__post_init__` in the same file | Enforces the same `< 0.6` / `>= 0.6` partition | Rejects inconsistent manually constructed classifications; does not compare draft and ordinary scores. |
| `continuous_stream.py:893-903` and `_defer_audio` | Consume the evidence state | Uncertain previews wait with empty text; EOF or closed source-unit refusals raise `StreamNeedsResolutionError`, retain PCM, make no commit and close/release the native owner. |
| `continuous_stream.py:969-980` | Lexical check on the selected publication | A lexical whole result cannot authorize a nonlexical suffix. This is not a numeric score cutoff. |
| `continuous_stream.py:1476` | Resolution-probe evidence | Observational only: the probe has no publication authority. |
| `native_result.py` | Snapshots/validates scores | Finite optional metadata; no-speech in `[0,1]`, ratio nonnegative. Neither `avg_logprob` nor compression ratio drives runtime acceptance. |
| `_draft_inference.py:144-164` | Accepted current token ID equals proposed draft ID | Only matching current-audio rows are reused; mismatch crops self-KV and resumes ordinary inference. No epsilon, threshold band, ordinary-score comparison or score fallback. |

Exact digital-zero PCM precedes every model score and authorizes empty coverage;
nonlexical text and a missing score remain uncertain. With `input_evidence=False`,
the continuous controller has no implicit score gate. Direct native-window
publication validates ownership, completeness and provenance, not model confidence.
Timing/word policies consume token/text/timing agreement, not these scalar scores.

The local patched backend inspected at
`.tmp-composed-native/backend/whisper/decoding.py` computes no-speech probability
from the SOT softmax, applies unchanged logit filters, then selects greedy argmax.
The draft wrapper preserves the initial SOT rows but their larger parallel
forward shape can produce different floating-point values. The timestamp filter
also compares timestamp log-probability mass **strictly greater than** the maximum
text-token log probability. Scalar result-score deltas do not bound those logit
margins. One greedy sequence is the only supported nonempty-draft mode, so the
backend's final sequence ranker has no competing candidate in this lane.

Do not import upstream `transcribe()` policy into this audit. Its defaults are
ratio `> 2.4`, average logprob `< -1.0`, and no-speech `> 0.6`, with additional
average-logprob conditions for fallback/skip. They exist in the local backend
`transcribe.py` and the separate `infra/modal_word_corpus.py` offline-control
configuration, but this integrated adapter uses `DecodingTask._start_run`, not
`transcribe()`. Tests around `-1` and `2.4` verify their *absence* as native runtime
publication cutoffs; they do not emulate or qualify upstream transcription.

## Deterministic tests

New tests exercise draft limits 0 and 32 through the actual controller with
reconstructed `NativeWindowResult` values and existing real transaction fixtures.
Both streams receive the same 32 raw token IDs; the second decode verifies that
the hint is empty for draft0 and length 32 for draft32. No decoder is simulated
as if it established numerical equivalence.

- Adjacent binary64 scores below, at, and above the unchanged `0.6` policy.
- Native binary32 neighbors `0.5999999642372131` and `0.6000000238418579`.
  Decimal `0.6` itself is not exactly representable in binary32.
- Same-side perturbations retain exact events; crossings in either direction
  change refusal/commit outcomes with identical reconstructed tokens.
- Both EOF and explicitly sealed source-unit boundaries preserve the inclusive
  gate. Refusals retain all 3,200 input samples, have no commit/final event and
  no state version advance, and restore capacity.
- Average logprob around `-1` (also `-100` and `0`) cannot override the gate;
  compression ratio around `2.4` (also `100`) cannot introduce a new cutoff.
- Digital silence, nonlexical output, missing score, and disabled gating retain
  their existing behavior.
- Audit tests reject stale trace scores/decisions, mismatched spans, counts and
  arm ordering, and explicitly report crossings despite identical tokens.

Run from the repository in PowerShell:

```powershell
$env:PYTHONPATH = 'src;.;tests'
python -m unittest test_draft_thresholds tools.test_audit_draft_thresholds test_audio_evidence test_continuous_evidence test_continuous_draft test_native_draft -q
python tools/audit_draft_thresholds.py
```

Result: **88 tests passed**, including 15 new tests. Ruff check and formatting
pass for all three new Python files. The offline record is
`evidence/draft-threshold-audit-2026-09-07.json`; it includes raw-input and policy
source SHA-256 values. No runtime file, frozen preflight file, or GPU job was
modified or launched for this audit.

## Guarantee and minimum remedy

For identical non-score inputs, scores on the same side of `0.6` produce the same
audio-evidence classification. A score crossing can produce different
publication and scheduling even if tokens are identical. The existing guard is
fail-closed for the score it actually receives; it does not establish ordinary
versus draft decision parity. Saved raw scores and differing trace metadata also
preclude claiming universal bitwise result identity from token/event agreement.

No threshold change or runtime patch is justified by this screen: it contains no
actual crossing. Keep draft32 opt-in and keep parity claims empirical and
input-scoped. If exact ordinary-path decisions are a requirement, the immediately
available conservative choice is `max_draft_tokens=0`. A future optimization could
obtain the canonical ordinary SOT score under the same input/options before the
gate; that needs a separate implementation and ownership/cost qualification.
An epsilon fallback is not a guarantee unless its error bound is proved for the
supported executions. In particular, the observed maximum delta is **not** such
a bound, and moving/rounding/relaxing the threshold is not a parity remedy.
