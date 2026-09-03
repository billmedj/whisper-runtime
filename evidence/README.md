# Integration evidence

This directory contains one committed record from a real
`NativeWhisperAdapter` transaction. It identifies the runtime revision, backend
source tree, model checkpoint, input, environment, and observed outcome.

The native CI workflow also publishes a 30-day artifact for its same-model
interleaving check. The artifact records state separation, early cleanup,
rejection of cancelled-run reuse, and a survivor that matches an isolated
baseline within the recorded scalar tolerance, which is zero in CI. Its format
is defined by `native-interleaving.schema.json`.

Each record applies only to its stated configuration. The records are not
performance benchmarks. The interleaving check does not establish parallel
kernel execution, thread safety, or behavior on other models, devices,
operating systems, or dependency versions.
