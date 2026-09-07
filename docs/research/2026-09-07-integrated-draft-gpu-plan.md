# Integrated drafts: order-balanced T4 screen

Registered before execution. Test the native opt-in, not the earlier decoder
method patches. Defaults and publication thresholds remain unchanged.

## Fixed scope

One cached `tiny.en`, English FP32 greedy, seed 7, one T4 and owned native lane.
Both arms use 20/24-second retained context and alignment-feature reuse. Only
`max_draft_tokens` differs: 0 for control, 32 for the candidate.

Run control, candidate, candidate, control in that order. Each fresh stream
receives two complete copies of the registered 46.55-second mixed
speech/silence/noise/silence input: 93.1 seconds, 1,489,600 samples. Compute
and freeze its actual PCM hash and repeated reference before launch. This
tests a cycle boundary, not new speakers, held-out accuracy or sustained live.

Warm both paths with two unpaced eight-second windows. The second candidate
warmup must use a verified draft. Then cancel one real speculative prefill and
verify unchanged publication state, released cache and returned native capacity.
Stop on a failed warmup, cleanup probe or measured stream. Do not retry.

## Measurements

Keep raw forward events, token positions, per-phase native operation wall time,
memory peaks, raw results, selected policy decisions, full stream events and
source-admission clocks. Observe the native adapter; do not patch decoder
methods. Resolve CUDA events after native completion fences, never per token.

Require all input to be accounted for, FINAL once and terminal cleanup. Use the
same stream ID for each fresh arm and compare full events. Compare current-audio
token sequences, commit spans/text and supporting analysis endpoints separately
from small score differences. No numeric tolerance or publication gate changes.

Report both adjacent comparisons and pooled equal-count totals. A call-count
reduction alone is insufficient: the pooled decoder wall time and CUDA
forward-interval sum must fall with exact output parity. Report alignment time
separately. The comparison remains one fixed-order screening run, not a general
speed, energy, confidence-interval or production claim.

## Resource bound

Exactly one GPU invocation, 480-second worker limit, two CPU cores, 4 GiB host
memory, at most 256 native windows. Reserve 20 seconds for cleanup and admit
each stream only if its 93.1 seconds plus the existing 15-second drain bound
fit. Zero minimum containers, one maximum, single-use worker, no automatic
retry, no deployment. The existing model volume is read-only; worker network
access is blocked. Upload only the frozen source allowlist and public fixtures.

At [Modal's prices](https://modal.com/pricing), checked 2026-09-07, applying the
listed maximum regional multiplier to 480 seconds gives **$0.1672272** of
planning compute. This is not an invoice or billing cap. Startup, image work,
storage, network, taxes and provider rescheduling are excluded.

Store the compressed response before local validation. Preserve partial runs,
the attempt journal and frozen preflight. No GPU retry is automatic.
