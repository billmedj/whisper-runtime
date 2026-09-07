# Integrated T4 screen: completed

The user explicitly approved this 63-file/480-second payload. One invocation
started at `2026-09-07T00:51:00Z` and completed at `2026-09-07T00:57:48Z`.
Modal app `ap-2Vne3EccdVZEVwEHj5qe9P` is stopped with zero tasks. No second
invocation is queued. The registered comparison passes; see the
[results and limits](2026-09-07-integrated-draft-gpu-results.md).

The previous launch was rejected before process creation because the prior
59-file/240-second approval did not cover this payload. That rejected launch
started no app and created no attempt journal. The current invocation uses
the subsequent explicit approval, not the prior approval.

## Frozen payload

- Replay: `integrated-draft-t4-20260907-v1`.
- Preflight: `artifacts/modal/integrated-draft-t4-20260907-v1/integrated-draft-preflight.json`.
- Source: 63 files, 1,275,199 bytes. No credentials or research papers.
- Source digest: `c83c9202da80bf7bbf6a7eff2d387861a651a6538b287db3c7496f3ffa17713f`.
- Audio: existing public fixtures; registered PCM repeats two complete cycles.
- One T4 invocation, 480-second limit, no automatic retry or deployment.
- Planning compute: $0.167227, not an invoice or billing cap. Startup, image
  work, storage, network, taxes and provider rescheduling are excluded.

## Local checks completed

Twenty-seven focused tests and Ruff pass. Resource construction with the
installed Modal 1.5.5 SDK passes without starting an app. A real CPU smoke
with cached `tiny.en` validates the new observer on two eight-second windows:
the draft accepts 32 tokens, metadata capture matches native results, and
cleanup releases capacity without changing model state or hooks.

The [registered plan](2026-09-07-integrated-draft-gpu-plan.md) fixes an ABBA
comparison on four 93.1-second source-paced streams. It does not claim held-out
accuracy, sustained-live qualification or a general speedup.

The offline audit verifies all 63 frozen source hashes, the compressed response,
full output parity and terminal cleanup. The worker took 386.595 seconds and
used 201 native windows, including warmup and cancellation. The raw response,
preflight and attempt journal are preserved in the
[record archive](../../evidence/modal-integrated-draft-2026-09-07.zip).
No billing total was retrieved. No new GPU run is authorized by this report.
