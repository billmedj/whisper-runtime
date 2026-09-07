# Matched T4 test of verified token drafts

Registered before execution on 2026-09-07. The local CPU result reduced
sequential decoder forwards but increased decoder input tokens. This test
measures whether the change reduces actual inference time on T4.

## Fixed comparison

One ephemeral worker loads the existing cached `tiny.en` model once. Both
arms use FP32, English, greedy decoding, seed 7, one owned CUDA lane, the same
patched backend and exact-window alignment encoder reuse. The stream keeps
20/24-second context and all existing publication and input-evidence checks.

1. `fast-reuse`: ordinary decoding.
2. `fast-draft32`: at most 32 raw tokens from the preceding completed window
   serve as untrusted proposals. Current-audio features and decoder caches are
   fresh for each new window. The original per-token filters, timestamp rules,
   alignment and publication policy still decide the result.

Clear draft state between arms and before each measured stream. Warm each arm
with two 8-second windows so the candidate exercises a real draft prefill.
Run one additional bounded cleanup probe after speculative prefill. Record
its actual terminal operation and verify cache, wrapper, hook and native lane
cleanup before admitting measured work. Warmup and cleanup are excluded from
the measured arm summaries. A cleanup failure stops the worker.

The registered input is 46.55 seconds of PCM16 mono at 16 kHz: 33.66 seconds of
continuous-fixture speech, one second of digital silence, the 10.89-second
noisy prefix, then one second of digital silence. Its 744,800 samples have SHA-256
`b8902a2cfd5c45b8a17d6b489f6ddc752cf76d7b620efa27ec6e173ea6e2f826`.
The independent reference contains 114 words. Replay uses the existing
source-paced driver, not an accelerated CPU admission loop.

## Measurements and gates

Retain raw per-window native results, draft acceptance counts, decoder input
tokens, decision traces, events and exact source identities. Resolve CUDA
events only after the existing completion fence; do not synchronize each token
for measurement.

Report separately:

- Encoder and decoder forward counts, input sizes and CUDA-event intervals.
- Inclusive native operation wall time by phase and in total. This includes
  draft verification, scalar transfers, filtering, result handling and cleanup;
  forward-only brackets omit some of this work.
- Peak PyTorch allocated and reserved memory, plus post-close memory.
- Exact commit text and source boundaries, the analysis endpoints supporting
  each commit, native token sequences and policy decisions. Record score differences without silently
  increasing numeric tolerances.
- First commit and drain timing, excluding deployment and model startup.

Both arms must terminate, account for the full input, emit FINAL once, restore
capacity and pass the existing publication checks. Compare reference errors
and exact committed output between arms. Compare the control with the preceding
T4 result as a reproduction check, not as a substitute for the matched control.

A forward-count reduction alone does not pass the efficiency gate. Require
lower total forward intervals and lower inclusive operation wall time, with
unchanged committed output and supporting analysis endpoints. The receiver's
accepted-input position at event emission depends on processing speed; report
it separately rather than requiring exact equality. This distinction is fixed
before either measured arm runs. Report a memory
increase as a tradeoff, not a free saving. Fixed-order, single-run timing remains
a screening result, not a general speed, energy or production claim.

## Resources and provenance

- Exactly one GPU invocation; one T4, two CPU cores, 4 GiB host memory.
- Worker timeout 240 seconds; reserve 20 seconds for cleanup. No automatic retry.
- Zero minimum containers, one maximum, single-use worker; no deployment.
- Cached model volume read-only; no remote network or model download.
- Explicit source-file and public-fixture allowlist. Freeze and verify hashes
  locally and remotely. Source mounts avoid rebuilding a layer for each file.
- Preserve the compressed worker response before validation, plus the local
  preflight and attempt journal. Partial or failed runs remain evidence.

At [Modal's published prices](https://modal.com/pricing), checked 2026-09-07,
T4 costs $0.000164/s, a physical CPU core $0.0000131/s, and host memory
$0.00000222/GiB/s. Applying the listed maximum region multiplier of 1.75 to
the full worker bound gives a planning compute estimate of **$0.0836136**.
This is not a billing cap or invoice. Startup, image work, storage, network,
taxes and provider rescheduling are excluded. Do not purchase credits or run
another GPU attempt automatically.

Production source and CLI defaults are unchanged. A passing short test would
not qualify sampling, beam search, other models, sustained live input,
checkpoint/restart or release readiness.
