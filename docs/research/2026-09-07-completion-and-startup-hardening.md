# Completion and native-startup fixes

Date: 2026-09-07. Local checks pass. **Not release-qualified.**

This work fixes completion races and makes the GPU test harness load the same
verified backend as the CLI. No paid job ran. Model weights, backend pins,
execution profiles and acceptance thresholds are unchanged.

## Fixed behavior

- The EOF deadline starts before `finish_input()` exposes the end of input.
  Previously, a concurrent owner could finish before the producer set the clock.
- The driver checks cancellation, source errors and the deadline after native
  work, before each returned event, and after joining the input thread. A final
  native step can no longer hide cancellation or a missed deadline.
- The CLI delays its FINAL announcement until driver validation and cleanup
  succeed. Cleanup also precedes exports. Cancellation is still cooperative:
  this does not interrupt a native call that is already running. Export-write
  failures can still leave newly created partial files; existing files are not
  overwritten.
- Microphone capture rechecks the queue when a timed-out read coincides with
  device shutdown. A final callback frame already in the queue is preserved.
  A stopped device with an empty queue still fails.
- The native harness validates the backend, dependencies and checkpoint before
  importing Whisper through the CLI's guarded loader. CUDA prerequisites load
  before the source clock starts. Ambient or changed Whisper modules fail
  validation. Successive sessions retain one verified import, model and worker.

The previous harness could import an ambient Whisper module before validating
a separate Git directory. It also omitted the CLI's eager Triton import. These
are verified startup differences, not proof of what caused `source_late` in the
[earlier failed run](2026-09-07-capacity-smoke-results.md). Importing Triton
declarations does not compile or warm all kernels.

## Timing observations

`source-clock-v2` records the frame at the maximum wake-up delay and the largest
yield-to-resume interval. A worker-local monotonic origin can be compared with
the recorded first-DTW interval. Storage remains bounded; no new input thread
or per-frame trace was added. The 250 ms source deadline is unchanged.

Failure receipts distinguish offered samples and their hash from the native
accepted count. An offered-prefix hash is not presented as an accepted-prefix
hash. Yield-to-resume time includes caller work and scheduling, not just push.

## Local validation

| Check | Result |
|---|---|
| Runtime suite | 877 tests, 7 skipped; no failures |
| Repository-tool suite | 853 tests, 5 skipped; no failures |
| Native harness, soak and diagnostic tests with Modal SDK 1.5.5 | 66 tests; no skips or failures; no remote work |
| Ruff, formatting, strict mypy | Pass; 32 typed runtime modules |
| Wheel contents | Pass; 32 installed Python modules match source and wheel bytes |
| Installed help | Pass without Torch or Whisper on the host Python |
| Installed CLI, cached `tiny.en`, CPU, 11-second JFK WAV | One FINAL; TXT/SRT/VTT match the previous fixture exports |

The CLI tests include deterministic cancellation, deadline, source-close and
cleanup failures. The microphone test controls the callback/timeout interleaving;
it does not use a physical device. The native run uses existing pinned ML
dependencies and a cached checkpoint. It does not qualify a clean OS install,
live CPU speed, remote GPU behavior or transcription accuracy on new audio.

Local artifacts are under `artifacts/release-hardening-20260907/` (not packaged):

| Artifact | SHA-256 |
|---|---|
| Wheel | `6cf66ae79d3691dd6d059cc667731c219d1df81d9eb7409d27a1bf56cc7e00a4` |
| Installed TXT | `b141cb01b4b691dac27c6ebc485a15914b5dbb427f5ab2f72b81b295779ff22b` |
| Installed SRT | `a204199c282638ddb0737629e8779f71126bd8fb47e143ad6359e22c9da6926c` |
| Installed VTT | `bdbeac0e9ae92f93723940dd1d7c710dc363560ad02978a5e2137edc7d7586cd` |

## Next validation

A local-only preflight froze 51 source files (1,093,448 bytes) for
`verified-startup-clock-20260907`. Source manifest digest:
`1cca75d057dc843df5a75ff1039475604e2f70c7550ad5d5834f54a5cca59bb5`.
The snapshot contains the reviewed working bytes, not a clean committed revision.

The prepared workload uses the same two 46.55-second sessions, one T4, a submitted
4 GiB memory limit, a 300-second execution timeout and no configured retries.
It has not run. Startup timing needs fresh qualification on this corrected
source. Earlier GPU records and archives remain unchanged. Four-hour native
endurance and release-platform checks remain open; there was no GitHub push.
