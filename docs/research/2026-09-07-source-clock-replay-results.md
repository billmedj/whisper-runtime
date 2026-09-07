# Instrumented source-clock replay

Date: 2026-09-07. Result: **short capacity smoke passed**. Not release-qualified.

One separately authorized T4 attempt completed two source-paced 46.55-second
sessions under the submitted Modal 4 GiB memory limit. Input, model, execution
profile, timeout and acceptance thresholds match the
[previous failed capacity smoke](2026-09-07-capacity-smoke-results.md). The only
uploaded changes add bounded source-clock measurements to the two harnesses.
No engine, backend, weights, warmup or allocator settings changed.

## Measured result

| Measure | Session 1 | Session 2 |
|---|---:|---:|
| Source samples accounted for | 744,800 | 744,800 |
| Input frames | 2,328 | 2,328 |
| Decode calls | 25 | 25 |
| Transcript events | 30 | 30 |
| FINAL events | 1 | 1 |
| Reference word edits | 7 / 114 | 7 / 114 |
| Session elapsed time | 47.252 s | 47.103 s |
| Maximum source wake-up lateness | 222.770 ms | 2.134 ms |
| Maximum yield-to-resume gap | 0.890 ms | 1.201 ms |
| Terminal CUDA allocated bytes | 160,720,896 | 160,720,896 |
| Terminal CUDA reserved bytes | 358,612,992 | 358,612,992 |

Both sessions restored runtime capacity. TXT, SRT and VTT exports match byte for
byte. Session clocks start after the model/stream factory; image construction
and initial model setup are excluded. The repeated input is not an independent
accuracy dataset. Guest RSS remains an observation, not verified physical RAM.

## Interpretation

The first session passed the unchanged 250 ms source-clock bound by 27.230 ms.
The second had much more margin. The small yield-to-resume gaps do not support
a long buffer-admission stall in this run; these gaps also include caller work
and scheduling, so they are not isolated `push` timings.

The first CUDA DTW call lasted 1.228263094 seconds. Its duration is similar to
the earlier failed run and an earlier successful uncapped diagnostic. The
source-clock receipt retains maxima but not the frame at each maximum. It
cannot align the largest wake-up delay with an exact DTW interval.

This result shows that the registered short workload can complete under the
submitted provider limit. It does not establish that the earlier timing failure
is fixed, prove GIL causality, qualify cold-start reliability, or establish
four-hour memory stability. No threshold was relaxed and the previous failed
records remain failed.

Before release, address the narrow first-session timing margin and complete the
native endurance gate. Separate platform-installation and physical-capture
checks also remain open. No hourly stage or additional paid attempt ran here.

## Evidence and execution scope

[Evidence archive](../../evidence/modal-source-clock-replay-2026-09-07.zip):
1,465,323 bytes, 65 payload files plus a manifest. SHA-256:

```text
13d533f6e124abcf552bd34c57dcfaa3ac36d1a43a4de1e2f9a2d0f8f39250ad
```

The archive contains 51 exact uploaded files, five frozen review files, raw
input, observations, preflight, attempt journal, terminal result, provider-stop
observation and independent verification tools. Uploaded-source digest:

```text
1c26f68d7b0b666602500fbca9befeb2f976c5f82d407f39b2b54693d5a3d646
```

Independent checks pass for all frozen bytes, 76 ordered observations, one
observed worker/model, transcript coverage, word edits, exports, source-clock
arithmetic, CUDA plateaus and cleanup. Archive-only verification also passes.
Forty-five focused harness tests passed with pinned Modal 1.5.5 before execution.

The invocation used one T4, a 300-second function timeout, a 180-second startup
timeout, `memory=(4096, 4096)` and no configured retries. The planning compute
estimate was USD 0.104517, excluding build, startup and other charges. Actual
invoice and account balance were not retrieved.

App `ap-fctaqHYG0G4Ex4WfvmMxaj` stopped at 05:27:48 UTC. A read-only Modal query
at 05:28:56 UTC confirmed `stopped` and zero tasks. Local command exit code: 0.
The package remains `0.1.0.dev0`; no commit, push or release was made.
