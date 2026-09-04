# Integration evidence

This directory contains two committed records from real backend runs:

- `native-cpu-tiny-en-jfk-2026-09-03.json` records one
  `NativeWhisperAdapter` transaction.
- `native-cpu-tiny-en-jfk-interleaving-2026-09-03.json` records two staged
  decode runs on one loaded model, early cleanup of one run, and completion of
  the other.

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
resulting 30-day artifact is validated against `native-threaded.schema.json`
and by cross-field checks in `tools/validate_threaded_record.py`.

Each record applies only to its stated configuration. The records are not
performance benchmarks. The two-thread check exercises the patched Whisper
backend below the runtime adapter. It does not exercise concurrent encoder
calls, the scheduler, or adapter concurrency. Neither check establishes kernel
overlap, throughput, CUDA behavior, production readiness, or behavior on other
models, devices, operating systems, or dependency versions. The two-thread
check covers one controlled case; it is not a general thread-safety guarantee.
