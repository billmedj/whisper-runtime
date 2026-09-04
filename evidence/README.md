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

Each record applies only to its stated configuration. The records are not
performance benchmarks. The interleaving check does not establish parallel
kernel execution, thread safety, or behavior on other models, devices,
operating systems, or dependency versions.
