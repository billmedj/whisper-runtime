# Registered native release endurance plan — not executed

This is a prepared experiment, not a passing result, a release certificate, or
permission to spend. The exact source/plan preflight must be frozen and reviewed
before the separate `--confirm-paid-gpu` command receives authorization.

## Scope and workload

One ephemeral Modal **plain generator Function**, one Tesla T4, one loaded pinned
English `tiny.en` FP32 model, one native Worker, and one inference-owner thread:

1. The existing registered 46.55-second smoke cycle, with its existing <=10% word
   edit-rate development gate. This is the memory warmup and first session.
2. Four sequential, separately finalized, source-paced 3,600-second sessions.

Every stage uses the same `conservative-v1` profile as the installed CLI and the unchanged
`CLI_STREAM_CONFIG`, with no draft tokens or alignment-feature reuse. Preflight
captures the entire profile record and exact source-file hashes. Drift fails
before dispatch; the worker and every terminal record check the same contract.
The remote worker imports the reviewed source overlay through `PYTHONPATH` and
reuses `NativeOwner` to construct streams. This is source/profile qualification,
not a test of installing or invoking the released wheel on that worker.
Test/plan hashes are frozen separately for review but are not uploaded or run on
the paid worker; its explicit file manifest contains only the runtime/harness
and the existing pinned source/registration helpers.

The source reuses the existing public/registered main and noisy corpus case
allowlist: full 33.66-second concatenated speech, one second of digital silence,
the 10.89-second registered noisy prefix, and another second of silence. Each
hour contains whole cycles followed by silence, never a clipped final word.
Frames are generated incrementally from the 1,489,600-byte seed, in 640-byte
(20 ms) frames; no one-hour PCM files or unbounded audio history are produced.

**This is not one four-hour session or four hours of varied new speech.**
`live-v2` remains capped at one hour. This harness deliberately calls the existing
native SDK driver instead of ASGI, so it does **not** extend the existing
WebSocket/WAN, browser, physical-microphone or capture-device qualification.
No automatic reconnect, retry, second model, or concurrent inference is requested.

## Preregistered acceptance and failure rules

Each phase must show a real FINAL after actual EOF; exact offered, accepted and
committed sample coverage; matching incremental source SHA-256; no buffered
samples; one final event, an ordered exact revision chain, and restored native
queue/lease/budget capacity after close. Model fingerprint, model-load count
(exactly one), source digest, instance identity, and session numbers 1 through 5
must match. Any failed smoke or phase prevents all later phases.

Source pacing uses an independent monotonic clock; more than 250 ms lateness,
full audio/event queues, missing data, invalid events, failed native cleanup,
identity drift or capacity retention aborts. The same driving loop as the installed CLI handles
input admission, owner-thread inference, real EOF, coverage and cooperative
cleanup. EOF drain is limited to 30 seconds. Native calls are not forcibly
interruptible; the provider timeout is the outer process-level boundary.

Memory gates are fixed **before running**, not fitted to observed results:

| Measure | Registered criterion |
|---|---|
| Warmup baseline | After smoke closes, Python GC and CUDA synchronization; never reset later |
| Terminal live CUDA allocation | Exact return to that baseline after each hourly cleanup |
| CUDA reserved allocation | No more than baseline +256 MiB and an absolute 2,048 MiB ceiling |
| Process current RSS | No more than baseline +512 MiB and an absolute 3,584 MiB ceiling |
| Checkpoints | At event boundaries at least approximately every 300 seconds, plus every terminal; live allocation may be higher during active work |

No `empty_cache` call disguises growth. CUDA allocated/reserved/peak allocated/
peak reserved and current Linux `/proc/self/statm` RSS are recorded. Reserved
and RSS allowances accommodate bounded allocator/runtime caching rather than
requiring all caching to disappear. These guardrails are engineering limits,
not a statistically calibrated leak test; growth below them can still matter.
Intervals between event callbacks are not a hard real-time sampling guarantee.

All event envelopes and memory checkpoints are streamed to per-phase JSONL,
flushed incrementally and SHA-256 hashed. Terminal receipts preserve full EOF,
native capacity/model identity, source/configuration and memory evidence.
Each phase caps events at 100,000 and its log at 32 MiB; each message at 512 KiB.
A 64-message mailbox prevents an unavailable receipt consumer growing worker
memory without bound. Missing/duplicate phases, identity changes, absent terminal
receipts or truncated delivery never become success. There is no persistent
remote volume for outputs; a disconnect/crash may leave only partial local logs.

## Resources, price, and authorization

The one planned worker is T4 / 2 CPU cores / 4,096 MiB RAM, AWS `us-west`, minimum
zero containers, maximum one, zero warm buffer, single-use container, two-second
scaledown, zero requested application retries, startup timeout 180 seconds and Function
execution timeout **14,800 seconds**. This includes 14,446.55 seconds of source
audio plus approximately 353 seconds of execution headroom. The per-phase
deadlines are 190 seconds for smoke and 3,690 seconds per hour; the overall
14,800-second deadline takes precedence.

Runtime egress is blocked and Modal access restricted. The existing model cache
is looked up with `create_if_missing=False` and mounted read-only. Weights are
loaded by verified local path; the harness never downloads models. Image builds
reuse the existing pinned backend/dependency recipe and therefore may access
package/source registries **during authorized build**, not during inference.
No persistent deployment, writable model cache, proxy token or web endpoint is
created. Only the small registered public PCM seed is sent with the function.

At the [Modal prices](https://modal.com/pricing) checked 2026-09-07, the existing
registration's conservative 1.75 regional multiplier yields:

`14800 × (0.000164 + 2×0.0000131 + 4×0.00000222) × 1.75 = $5.156172`

This is a compute planning estimate, **not a physical spending cap**. Build,
startup, storage, egress, tax, price changes and provider crash rescheduling are
excluded. [Modal timeouts](https://modal.com/docs/guide/timeouts) are per attempt,
exclude scheduling, and may overrun by seconds. [Modal failures and retries](https://modal.com/docs/guide/retries)
state that container crashes may be rescheduled even without configured
application retries. The local collector rejects duplicated phases or a changed
worker and exits the ephemeral app; the harness cannot guarantee that the
provider never allocated a replacement container before that observation.

Modal SDK 1.5.5 does not support a retry policy for generator Functions: even
`retries=0` is rejected while defining the function. The decorator therefore
omits `retries` (`sdk_retry_policy: null`), with no application retry requested.
The first authorized attempt stopped locally at that SDK validation, before
`app.run` or GPU work; its failed receipt and journal remain untouched. The
`release-soak-20260907-sdkfix` preflight freezes this compatibility correction,
with the same workload, resource limits and price estimate. A real pinned-SDK
local-definition regression now covers the complete resource factory while
forbidding App run/deploy, resource hydration, remote iteration and socket calls.

## Local checks and subsequent execution

```console
python -m unittest discover -s tools -p test_modal_release_soak.py -v
python -m infra.modal_release_soak --replay-id release-soak-20260907 --preflight
```

Preflight is CPU-local and requires the existing registered corpus artifacts. It
does not import Modal, create a resource, download a model/input, or write PCM
copies. It writes `artifacts/modal/<id>/release-soak-preflight.json`. Scripted
short tests exercise the actual CLI driving loop with fake native work; they are
not native/GPU/long-duration evidence. There is intentionally no paid short-mode
flag that could be confused with the full registration.

Only after explicit approval of the frozen payload:

```console
python -m infra.modal_release_soak --replay-id release-soak-20260907 --confirm-paid-gpu
```

Matching preflight, Modal SDK 1.5.5 and a new attempt journal are mandatory.
Existing attempts/outputs are never overwritten or retried. A changed working
tree requires a fresh preflight and review, not a relaxed identity check.
