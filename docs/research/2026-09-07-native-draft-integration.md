# Verified drafts: native integration and CPU regression

The runtime now has an opt-in greedy draft path. It uses a request-local
inference wrapper, not the diagnostic's process-wide method patches.
`ContinuousStreamConfig(max_draft_tokens=32)` enables it; `0` remains the default.
No model weights, publication rules, CLI defaults or backend source were changed.

## Integration

The stream retains at most 32 raw token IDs from a completed analysis as an
untrusted proposal. The native adapter attaches the wrapper to the owned decode
run before prefill. The existing Whisper filters, token-selection loop,
alignment and runtime publication policy remain authoritative.

Current-audio features and decoder cache are fresh for each window. A rejected
proposal discards its suffix; ordinary decoding continues from the actual
selected prefix. The wrapper validates IDs, limits text-context use and rejects
unsupported cache or decoding modes. Cleanup drops speculative rows, audio
references, hints and owned cache. No new runtime dependency is required.

Hints are stream-local and absent from saved checkpoints. The configuration
survives restore; the first restored window decodes without a hint. Old v1
savepoints missing this option still load with it disabled. Auxiliary recovery
and prompt re-decode do not reuse hints.

## Actual model test

One locally cached `tiny.en` model, English FP32 greedy, seed 7, CPU only.
All three arms used the same 33.66-second speech input, 20/24-second retained
context and alignment-feature reuse. Input was admitted in two-second chunks
without real-time pacing. This is not a timing benchmark.

| Measurement | Ordinary | Draft | Draft with restore |
| --- | ---: | ---: | ---: |
| Decoder forwards | 1,009 | 810 | 820 |
| Encoder forwards | 17 | 17 | 17 |
| Decoder input token positions | 1,844 | 2,105 | 2,083 |
| Native window traces | 17 | 17 | 17 |
| Full stream events | 29 | 29 | 29 |
| Commits | 11 | 11 | 11 |
| Outstanding resource leases after close | 0 | 0 | 0 |

Both candidates match the control's full events, commit text and spans, raw
tokens, selected decisions and stream metrics. All 538,560 samples are accounted
for and FINAL occurs once. Drafts save 199 decoder forwards (19.7%). The restored
arm saves 189 (18.7%); its empty initial hint costs ten calls relative to the
uninterrupted draft arm.

The third arm saves a logical checkpoint after a nonfinal commit, closes the
stream and restores it from disk. It reuses the same loaded model and worker.
This demonstrates same-process boundary restore, not cross-process migration,
mid-token continuation or a durable GPU cache.

A separate real speculative-prefill cancellation leaves no published state,
speculative rows, audio reference, hint or owned cache. Native capacity is
released. The model fingerprint and registered instrumentation hooks are
unchanged after all arms and the probe.

Scores differ: maximum absolute changes are `3.3171280572341644e-7` in average
log probability and `1.3336539268493652e-6` in no-speech probability. Submitted
decoder token positions increase by 14.2% in the uninterrupted draft arm.
These results establish fewer calls with the listed output parity on this
fixture, not lower FLOPs, energy or universal numeric equivalence.

## Verification and evidence

- Runtime suite: 850 tests pass, with two optional/platform skips.
- The 15 wrapper tests also pass with the installed CPU Torch environment.
- Forty-three new focused tests cover the wrapper, native ownership and stream
  integration, including mismatches, filter rejection, context bounds, cleanup,
  isolation, cancellation, legacy savepoint loading and restore.
- Strict type checking passes for all 31 source files; Ruff passes.

The [archive](../../evidence/native-draft-integration-2026-09-07.zip) contains
the unmodified result and all seven source files listed in its `sources` map,
stored by basename. Entry hashes were verified against the result. Archive
SHA-256: `e9dd1cdac8e662dc70c3816c6363e56b24219a2293a0ca10ebcb0d6023b3e846`.
Result SHA-256: `63838d725079e86ceaa9eb9434e0792c7947d0f88edafbb73a896c196e4b1786`.

An independent offline review compared all saved events and cleanup flags.
After execution, the driver's future success gate was strengthened to include
full event equality and cleared hints, not just commits and decision traces.
The archived executed driver is preserved; inference was not repeated.

To repeat the bounded CPU test with an existing native setup, use
`tools/verify_native_draft.py --model PATH --manifest PATH --output NEW_PATH`.
The parent imposes a 240-second limit and will not replace an existing result.
It does not contact Modal or download a model.

## Remaining boundary

The [earlier T4 result](2026-09-07-draft-gpu-results.md) used the diagnostic
implementation. The integrated path still needs order-balanced GPU timing,
longer held-out streams, score-threshold cases and fresh-process restore
qualification. The current tests do not qualify other models or decoding
modes. No new GPU invocation was made for this integration.
