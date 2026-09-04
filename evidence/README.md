# Integration evidence

This directory contains four committed records from real backend runs:

- `native-cpu-tiny-en-jfk-2026-09-03.json` records one
  `NativeWhisperAdapter` transaction.
- `native-cpu-tiny-en-jfk-interleaving-2026-09-03.json` records two staged
  decode runs on one loaded model, early cleanup of one run, and completion of
  the other.
- `native-cpu-tiny-en-jfk-threaded-2026-09-04.json` records the same isolation
  case across two operating-system threads in the patched decoder backend.
- `native-cpu-tiny-en-jfk-runtime-concurrency-2026-09-04.json` records two
  transactions through the experimental two-lane runtime adapter.

Each record identifies the runtime revision, backend source tree, model
checkpoint, input, environment, and observed outcome.

The native CI workflow repeats the same-model interleaving check and publishes
its record as a 30-day artifact. The check covers state separation, early
cleanup, rejection of cancelled-run reuse, and a survivor that matches an
isolated baseline within the recorded scalar tolerance, which is zero in CI.
Its format is defined by `native-interleaving.schema.json`.

The workflow also runs the decoder isolation case in two operating-system
threads after preparing both encoder outputs sequentially. Each thread enters
its first outer decoder call. A barrier in the first decoder block holds both
calls before either continues. The record identifies each
owner thread, captures the start and end of each outer call, and requires the
two intervals to overlap. It also records the explicit decode options. The
committed record and each 30-day CI artifact are validated against
`native-threaded.schema.json` and by cross-field checks in
`tools/validate_threaded_record.py`.

The workflow then sends two real-model transactions through the experimental
two-lane `NativeWhisperAdapter` profile. It verifies admission and declared
budget state while both calls are live, cancels one request through the runtime
transaction, and requires the other request to commit the isolated-baseline
text. The cancelled session stays empty. The queue and declared budget must be
fully restored, and a later adapter call must succeed. The CI artifact is
validated against `native-runtime-concurrency.schema.json` and by
`tools/validate_runtime_concurrency_record.py`. The committed record captures
the same contract on one Windows CPU configuration.

Each record applies only to its stated configuration. The records are not
performance benchmarks. The committed two-thread check exercises the patched
Whisper backend below the runtime adapter. The adapter-level CI check uses
caller threads; it does not exercise a runtime-owned scheduler. Encoder
preparation remains serialized. The declared resource vectors are
admission-ledger values, not measured RAM or device memory. No check establishes
kernel overlap, throughput, CUDA behavior, production readiness, or behavior
on other models, devices, operating systems, or dependency versions. Each
two-thread check covers one controlled case; it is not a general thread-safety
guarantee.
