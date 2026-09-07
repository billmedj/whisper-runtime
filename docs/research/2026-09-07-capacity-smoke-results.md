# Provider memory-capacity smoke result

Date: 2026-09-07. Status: **failed**, `source_late`.

The [registered test](2026-09-07-capacity-smoke-plan.md) submitted one T4 with
`memory=(4096, 4096)`, a 300-second execution timeout and no configured retry.
It stopped during the first source-paced session. The second session and the
four hourly endurance sessions did not start. The attempt was not retried.
A [separately authorized instrumented replay](2026-09-07-source-clock-replay-results.md)
later completed; it does not change this failure.

## Observed result

- Initialization reached `ready`; one model and one worker were used.
- Thirteen provisional/replacement text events arrived. No completed session
  or final transcript was recorded. No complete-transcript accuracy is claimed.
- The producer exceeded the existing 250 ms source-clock tolerance.
- Failure cleanup passed; terminal CUDA allocated memory was 160,720,896 bytes.
- The accounting sampler reported no errors. This is not a successful capacity
  test: the required workload did not finish.
- The Modal app `ap-WYU0hredvINaPUkzk4dG87` stopped at 05:03:24 UTC.
  A read-only query at 05:04:56 UTC confirmed `stopped` and zero tasks.

This is not an observed out-of-memory termination. It also does not establish
that the memory limit had no effect on scheduling. No host-memory or throttling
telemetry was captured. The earlier failed 3.5 GiB guest-RSS gate is unchanged.

## What the timing record establishes

The first CUDA alignment call took 1.244384580 seconds. The next recorded item
after its measurements was a text replacement, followed by cleanup and the
source-clock failure. The previous uncapped attribution test completed both
sessions despite a 1.187880052-second first alignment call. Total alignment
duration alone therefore does not explain the difference.

Audio production already runs on a separate thread from inference. The current
source-clock check combines wake-up delay with time spent offering the preceding
frame. It did not retain the failed frame index, exact lateness or preceding
offer gap. The record cannot distinguish scheduling, GIL contention or admission
delay after the fact.

The inspected backend's CUDA DTW path still calls CPU `backtrace`, compiled by
Numba on first use without `nogil=True`. This is a candidate to test, not a
diagnosis of this run. Numba documents lazy specialization and explicit GIL
release for compiled functions. [Numba compilation options](https://numba.readthedocs.io/en/stable/user/jit.html).

The immediate local work is to retain bounded source-timing measurements and
test cold versus prepared CPU alignment with a separate heartbeat thread.
Keep the source clock, tolerance, input and decoder behavior unchanged. Do not
add another input thread: the CLI already has one.

## Reproducible record

[Evidence ZIP](../../evidence/modal-capacity-smoke-2026-09-07.zip): 1,456,320 bytes;
65 payload files plus a manifest. SHA-256:

```text
4a99b2231f74072bf4768a950a59fe0d313cad0172d9c8ae3e4628e5e14e8355
```

It contains raw observations, terminal result, preflight, attempt journal,
public input PCM, provider-stop observation, the independent audit, 51 uploaded
source files, five review files and the verification helpers. Frozen source:

```text
cd72ddd79339666056b5a4d722c34ac796ae739b3e79b9d548a0c9c9e02c38bf
```

All frozen byte hashes, the 25-observation sequence and the singleton worker
identity pass the independent audit. The audit retains `capacity_smoke_passed:
false`; valid evidence is not a successful experiment. Archive-only readback
verification also passes.

Before remote execution, 871 runtime tests passed with seven skips, and 41
focused harness tests passed with pinned Modal 1.5.5. Ruff and strict mypy on
32 runtime modules passed. These checks do not substitute for native stability.

One local invocation stopped at Git ownership verification before an attempt
journal or Modal call. A process-scoped Git setting resolved it; global Git
configuration was not changed. There was one actual remote attempt.

The preflight estimate was USD 0.104517 for the maximum execution interval,
excluding image construction, startup and other charges. Actual billed cost
and remaining account balance were not retrieved.

The package remains `0.1.0.dev0`. There is no stable release or GitHub push.

## Subsequent local checks

A fresh CPU process exercised the exact cached backend's `backtrace` function
on a synthetic strided int32 trace, with a separate 20 ms heartbeat. No GPU,
model inference or network was used. The environment was Windows, Python
3.13.1, Torch 2.6.0+cpu, NumPy 2.5.2 and Numba 0.67.0.

| Operation | Call duration | Maximum heartbeat lateness in the phase |
|---|---:|---:|
| Sleep that releases the GIL | 600.326 ms | 14.827 ms |
| Cold backtrace specialization | 1,638.637 ms | 78.176 ms |
| Prepared backtrace | 0.664 ms | 16.757 ms |
| Deliberate GIL-held sleep control | 603.897 ms | 590.560 ms |

No heartbeat deadline fell inside the sub-millisecond prepared call; its phase
maximum is not an in-call delay. The positive control confirms that the probe
detects a long stall. Cold backtrace did not reproduce the 250 ms miss on this
machine. This negative result does not exclude different behavior on T4/gVisor
or interference from another part of alignment. No warmup or backend change was
made on the strength of this hypothesis.

The [CPU probe archive](../../evidence/cpu-source-clock-probe-2026-09-07.zip)
contains the probe and raw heartbeat record. Size: 6,179 bytes; SHA-256:

```text
f34f41a14525d6e6db9e583563bdcd8974c7dcce4fbbac05781a82f7f7fc48da
```

Both remote harnesses now share a bounded source-clock receipt. It records the
last offered frame, failed frame, wake-up lateness and yield-to-resume gap. The
latter includes buffer admission, caller work and scheduling; it is not a pure
measurement of `push`. Receipts are retained on success and failure. The source
origin and 250 ms threshold remain unchanged.

Forty-five focused harness tests pass with Modal 1.5.5. Tests cover the exact
threshold, absolute clock, delayed caller, finite measurements, failure cleanup
and receipt retention. The existing concurrent-admission regression also checks
that model work does not hold the input lock. The subsequent instrumented replay
tests these receipts in a new namespace. It passes the short gates but does not
establish the cause of this earlier failure or qualify endurance.
