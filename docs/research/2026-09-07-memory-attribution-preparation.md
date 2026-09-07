# Memory attribution: local checks and prepared T4 diagnostic

Subsequent execution is recorded in the [T4 diagnostic results](2026-09-07-memory-attribution-results.md).
The preparation record below remains historical.

The previous T4 smoke exceeded its registered 3.5 GiB process-RSS ceiling.
That result remains failed. No further Modal execution took place in this pass.
The cause of the 4.93 GiB RSS reading is not yet established.

## Changes and local checks

- The endurance command now exits with status 1 when its persisted result is
  failed. Previously it printed a report path and exited successfully even
  when the report recorded failure. Sixteen focused tests pass with Modal 1.5.5.
- CI now installs the pinned Modal validation dependency. Its resource-definition
  tests forbid network connections, resource hydration and remote execution.
- CI now tests wheel console and module help from a separate, dependency-free
  environment outside the source checkout. This workflow change has not run on
  GitHub; no code was pushed.
- The runtime suite passes: 871 tests, 7 skips. The repository-tool sweep passes:
  832 tests, 4 skips. Two later diagnostic regressions are covered by the separate
  final 18-test diagnostic run under Modal 1.5.5, with no skips. These counts are
  overlapping suites, not independent tests to add together.
- Ruff passes on 224 Python files; strict mypy passes on 32 runtime modules.
- A new wheel installs offline without optional dependencies in a fresh Python
  environment. Console and module help work outside the checkout. All 32 installed
  runtime modules match both the wheel and current source. This is not a clean-OS
  or native dependency installation test.

Wheel SHA-256:
`a5bffd339adb9249a75c7923c0cde59dd1816b51f396aebac70d04b91a6783a1`.
Historical release-candidate archives and reports were not overwritten.

## Local CPU observation

A separate Windows CPU probe uses the conservative factory, existing pinned
dependencies and cached model. It transcribes the same 744,800-sample public input
without source-clock pacing. Coverage completes with FINAL; the reference score
is 7 edits in 114 words (6.14%).

Working set is 463,745,024 bytes after model fingerprinting, 489,803,776 bytes after
transcription and stream close, and 488,755,200 bytes after releasing the stream
and garbage collection. Windows working set is not directly comparable to Linux
RSS or CUDA allocator counters. One short run cannot establish absence of leaks.
The first local probe omitted a required cancellation argument and stopped before
inference; the corrected invocation supplies it. This was a probe defect.

The script, output and local check logs remain under
`artifacts/release-memory-20260907/`, excluded from distribution. This local
observation does not replace the portable raw records of the failed T4 attempt.

## Prepared diagnostic, not executed

[Producer](../../infra/modal_memory_diagnostic.py) and
[tests](../../tools/test_modal_memory_diagnostic.py) observe the existing native
factory. They do not alter model weights, decoder rules, allocator settings or
the release memory policy.

- Record memory before native imports, after Torch and Whisper imports, after
  model loading and fingerprinting, and after two closed sessions.
- Observe the first CPU/CUDA alignment call without substituting an implementation.
- Read Linux `smaps_rollup`, `statm`, thread counts, and existing CUDA counters.
  Do not import Torch or start a Numba pool to measure a baseline.
- Run two identical 46.55-second, source-paced sessions on one model and worker.
  Stop on source, text, coverage, cleanup or measurement failure.
- Use one T4, a 300-second function timeout, no application retries, no persistent
  deployment and the existing read-only model cache. The 4 GiB memory request is
  not a provider-enforced usage ceiling.

The local preflight freezes 51 upload files (1,070,724 bytes), review-file hashes
and a 1,489,600-byte public PCM seed. Its source digest is
`abebed8d9084175fac95577e0bb8770fe04421f92985020070afdfc7fe8c1135`.
Root independently matched the live files, frozen copies and seed hashes.

Registered compute estimate: USD 0.104517, excluding startup, image construction,
storage and other charges. It is not a billing cap. The prepared replay identifier
is `memory-attribution-prepared-20260907`. A paid invocation requires explicit
approval and the unchanged preflight; the attempt namespace cannot be reused.

These observations should identify whether host memory appears during startup,
alignment compilation, or repeated sessions. They do not qualify endurance.
The four-hour test, advertised-platform installation and physical capture gates
remain open. Version remains `0.1.0.dev0`; no stable release or push occurred.
