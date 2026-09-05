# Local word-resolution diagnostics

## Scope

The existing word policy reports `anchor_missing` for several different
observations. This change adds detail without changing acceptance. The runtime
records which observation failed, exact text/token match counts and signed
boundary differences. It distinguishes unavailable context, absent lexical text,
token mismatch, timing mismatch, relocation and ambiguity.

The diagnostic is bounded by the current analysis and its four-word anchor. It
retains counts and at most four pairs of boundary deltas, not a candidate history.
No model, network service or GPU is needed. Existing profile IDs, publication
rules, retained PCM, resource ownership and native results remain unchanged.
The new optional trace field is additive; it is not a calibrated confidence score.

## Experimental correspondence

`tools/word_anchor_reconciliation.py` tests one narrow proposal on saved words.
It is not imported by the streaming runtime and cannot publish, advance coverage
or request another decode.

Eligibility requires all of the following:

- Explicit EOF, with at least two lexical anchor words fully retained.
- One exact contiguous text/token occurrence in the entire current analysis.
- The terminal anchor word is lexical and ends at the frozen committed boundary.
- Every word start and every nonterminal end satisfies the existing tolerance.
- Only the terminal end extends rightward, by at most 1,000 ms.
- A lexical continuation exists after both the observed and frozen anchor ends.

The 1,000 ms cap is an experimental limit, not an acoustic error bound fitted or
validated on independent audio. The matcher does not stack the window-origin
onset exception with terminal-end relaxation. It rejects ambiguous or relocated
phrases, changed tokens, other boundary shifts and missing continuation. It never
rewrites native word times or a previously committed anchor.

An eligible proposal establishes only structural correspondence. A missing spoken
word can occupy the extended interval without appearing in the alignment. A
negative example in the tests preserves this limitation explicitly. Eligibility
must not be described as omission-free recognition or completion of a stream.

## Reproduce the local analysis

```sh
python tools/analyze_word_resolution.py evidence/modal-t4-tiny-en-word-context-2026-09-06.json
```

The command prints JSON. It does not modify its input or start model work.
It reconstructs anchors from event-confirmed word publications, keeps their
original times and processes every terminal policy-failure cell. It records input
and analysis-code hashes. It does not establish new PCM/model provenance beyond
the archived evidence or simulate a changed sequence of future analyses.

## Recorded result

The [local replay](../../evidence/word-resolution-replay-2026-09-06.json) reads the
unchanged seven-cell T4 record. It includes all four policy failures and lists
the three completed cells as excluded from this failure-only analysis. All four
legacy terminal reasons reproduce exactly. No qualification changes.

| Cell | Current anchor observation | Experimental proposal |
| --- | --- | --- |
| Fixed context, no added pauses | Lexical anchor absent | Rejected: absent text |
| Fixed context, noise | Timing mismatch | Rejected: not EOF |
| Word context, no added pauses | Lexical anchor absent | Rejected: absent text |
| Word context, noise | Timing mismatch | Eligible: terminal end extends 540 ms |

The last noisy anchor has a 420 ms terminal-end shift in the preceding
observation and a 540 ms shift at EOF, both measured against the frozen commit.
The two observations differ by only 120 ms, but their agreement is not an
independent acoustic witness. Neither observation replaces the frozen times.
The no-pause word-context failure has no previous observation after its final
rebase; the preceding commit must not be used as a second witness.

Tests cover exact identity, ambiguity, relocation, boundary shifts, counterfactual
missing speech and malformed trace/commit joins. Fourteen diagnostic tests include
a comparison with the legacy matching rule on 1,000 generated cases. A regression
test preserves the existing case where the committed boundary follows the previous
observation's end. The runtime suite passes 563 tests; the repository-tool suite
passes 392 tests. Ruff, strict runtime typing and repository checks pass.

No GPU, model download, package rebuild or remote publication was performed.
These results establish diagnosis and structural eligibility only. The two
acoustic failures remain open.

## Next experiment

1. Freeze the selector and divide audio by speaker into development and held-out
   groups before examining candidate outcomes.
2. Compare the unchanged policy, local correspondence and one bounded alternative
   acoustic window. Use only retained input; an earlier evicted window is not an
   available recovery action.
3. Compare a diagnostic-guided choice with uniform choices under explicit budgets.
   Count diagnostic, verification and padded-encoder cost, not only decode calls.
4. Check complete transcripts against human references, including omissions,
   duplicates and repeated phrases. Keep failed or unfinished outputs in the
   denominator. Offline-model agreement is a separate metric.

Missing words require new model observations. Saved traces cannot establish what
a different acoustic window would produce. Do not enable a recovery policy or
claim latency/GPU savings from this local structural analysis alone.

## Registered terminal-window experiment

`experiments/modal-word-resolution-v1.json` freezes four states: the two EOF
failures above and two new LibriSpeech speakers. The new inputs, references,
source hashes and conversion recipe are in
`experiments/modal-word-resolution-inputs-v1.json`. Selection uses fixed dataset
rows and duration bounds, not recognition outcomes. They are held out from this
project's prior tests, not necessarily from Whisper training.

For each state, compare the existing retained window, the local shadow proposal
on that same result, and one window starting at the frozen prefix end. The new
window uses only retained samples. It drops anchor context and can cut through
speech if a model timestamp is wrong. It has no authority to publish.

The development controls must reproduce text, aligned words and the strict
failure reason. Held-out states use one past-only bootstrap window and a
1,000 ms holdback to construct a synthetic prefix. This prefix is not an actual
stream commit. An unresolved bootstrap stays in the report as a failed case.

Routing uses the current anchor diagnostic: absent text or changed tokens select
the alternative window; an eligible terminal-end mismatch selects the local
proposal; other cases retain the strict result. Human references are scored
after selection and never determine the selected arm. Each arm's complete
proposed text includes the frozen prefix. Failed arms retain the prefix and
report their missing continuation.

One T4, one cached FP32 `tiny.en`, ten native windows maximum, no warmup, no
configured retry and a 180-second function timeout bound the experiment. Two
windows per state plus two bootstraps account for all ten calls. Arm order
alternates. Phase times, actual encoder/decoder forwards, padded encoder frames
and peak memory include instrumentation overhead. All native leases must close,
and model parameters must remain unchanged.

```sh
python tools/prepare_resolution_inputs.py --download
python -m infra.modal_word_resolution --replay-id word-resolution-20260906 --preflight
python -m infra.modal_word_resolution --replay-id word-resolution-20260906 --confirm-paid-gpu
```

Run in the registered Modal environment with the runtime installed or `src` on
`PYTHONPATH`. The source tree must be committed and clean. The CPU probe binds
the exact snapshot. Result paths are exclusive; a failed attempt is not retried
automatically. Cached model weights must already exist in the registered volume.

This experiment tests terminal counterfactuals, not a live scheduling policy.
Both native arms are executed to measure their outcomes. A routed-cost estimate
does not equal the cost of this experiment, and a shorter audio slice does not
remove Whisper's fixed encoder padding. Stream recovery, acoustic coverage and
general efficiency remain unqualified regardless of these four results.
