# Verified-startup source-clock replay

Date: 2026-09-07. Result: **short capacity smoke passed**. Not release-qualified.

One separately authorized T4 attempt completed two source-paced 46.55-second
sessions with the conservative profile and submitted `memory=(4096, 4096)`.
The 250 ms source-lateness threshold, input, backend, model, memory gates and
300-second function timeout were unchanged. There was no JIT warmup or
configured retry.

The harness now validates the explicit backend checkout and dependencies before
using the CLI's guarded backend importer with eager Triton import. Ambient or
changed Whisper modules are rejected, and subsequent sessions reuse the verified
module, model and worker. Eager import is not kernel compilation or warmup.
SourceClock v2 records the maximum-delay frame and yield/resume interval on the
same worker monotonic clock as the first observed DTW operation.

Relative to the [previous replay](2026-09-07-source-clock-replay-results.md),
five uploaded files changed: the three live/diagnostic/soak harnesses, `cli.py`
and `audio_source.py`. The latter two include completion/cleanup guards and a
microphone final-callback race fix. This run uses the native stream driver,
not the installed console entrypoint or physical microphone; it does not
qualify those separate changes or clean-platform installation.

## Measured result

| Measure | Session 1 | Session 2 |
|---|---:|---:|
| Accounted source samples | 744,800 | 744,800 |
| Input frames / decode calls | 2,328 / 25 | 2,328 / 25 |
| Transcript events / FINAL events | 30 / 1 | 30 / 1 |
| Reference word edits | 7 / 114 | 7 / 114 |
| Session elapsed time | 47.268 s | 47.160 s |
| Maximum wake lateness | 206.387 ms | 1.235 ms |
| Maximum yield/resume gap | 0.546 ms | 0.593 ms |
| Terminal CUDA allocated bytes | 160,720,896 | 160,720,896 |
| Terminal CUDA reserved bytes | 358,612,992 | 358,612,992 |

Both sessions restored runtime capacity, used one unchanged loaded model, and
produced identical TXT/SRT/VTT hashes, also matching the preceding passed
replay. Peak CUDA allocated bytes were 290,099,200. Session elapsed time starts
after the model/stream factory and excludes image construction and initial
model setup. The repeated fixture is not independent accuracy evidence.

## Timing interpretation

The first session's worst wake was zero-based frame 1,429, spanning samples
457,280–457,600. It was due at source elapsed 28.600000 seconds and woke at
28.806387228 seconds, leaving 43.613 ms below the unchanged 250 ms bound.

The first CUDA DTW operation spanned elapsed **28.546384724–29.781312995 seconds**
on the same worker clock. Its measured duration was 1.234925196 seconds. The
entire 206.387228 ms due-to-wake interval lies inside this operation. This is
direct temporal overlap, not proof that DTW, Numba compilation, the GIL, or any
one component caused the delay.

The largest yield/resume gap was instead at frame 478, elapsed
9.580605925–9.581151791 seconds, outside that DTW interval. No measured
yield/resume gap approached the large wake delay. These gaps include caller
work, admission and scheduling; they are not isolated `push` timings.

The corrected startup path completed this short workload. It does not establish
cold-start reliability, explain the earlier failed wake deadline, demonstrate
four-hour memory stability, or independently measure physical host RAM/provider
limit enforcement. The historical 3.5 GiB RSS failure and all earlier failed
records remain unchanged. No endurance stage ran.

## Evidence and execution scope

[Exact-byte archive](../../evidence/modal-verified-startup-2026-09-07.zip):
1,475,269 bytes; 66 payload files plus a manifest. SHA-256:

```text
66c81aa36cb0b80bc943f41bc88a71e40b807201d0a5e35c5ffc98ef89d1c87a
```

The archive retains all 51 uploaded files (1,093,448 bytes), five frozen review
files, raw PCM, preflight, observations, result, journal, provider-stop receipt,
independent v2 audit and local verification helpers. Uploaded-source digest:

```text
1cca75d057dc843df5a75ff1039475604e2f70c7550ad5d5834f54a5cca59bb5
```

The independent audit verifies frozen bytes, 76 ordered observations, one
worker/model, complete source coverage, committed transcript projections,
word edits, export hashes, source-clock arithmetic, DTW interval/counter
consistency, terminal CUDA parity and capacity return. Fifteen deterministic
local auditor tests pass, including historical v1 pass/failure preservation and
v2 mutation refusals. Archive-only verification passes without reading the
current checkout. These checks do not reclassify old records.

App `ap-5kyZu1CYvSdHNRiaiiyFAS` completed its ephemeral context. A later read-only
Modal CLI receipt records `stopped`, zero tasks, and provider-reported stop time
**06:22:47 UTC / 14:22:47 +08:00**. The receipt does not record its query time.
The scope was one T4, one container, a 300-second function timeout and a
180-second startup timeout. The preregistered compute estimate was USD 0.104517,
excluding startup/build and other charges; no actual invoice was retrieved.
The package remains `0.1.0.dev0`; this evidence is not a stable release.
