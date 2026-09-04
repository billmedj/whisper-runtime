# Draft native CUDA qualification contract

`evidence/modal-native-cuda-qualification.schema.json` defines the version-one
record for one fixed native CUDA qualification cell. This schema is not a
performance-benchmark schema, and no committed evidence record currently
satisfies it.

The contract has two purposes:

1. check output compatibility between the pinned patched Whisper backend and
   `NativeWhisperAdapter`; and
2. exercise cancellation and four narrow, harness-injected failure points at
   the transaction boundary.

The validator checks schema conformance, source and artifact identity, event
order, resource-accounting relations, and summaries derived from raw samples.
It cannot prove that a producer observed physical execution honestly.
Provider logs and independent review remain necessary.

## Registered cell

The versioned registration is
[`experiments/native-cuda-qualification-v1.json`](../experiments/native-cuda-qualification-v1.json).
It fixes the source policy, backend revision, T4 worker profile, model,
checkpoint, input, decode options, resource contract, measurement boundaries,
sample counts, fault points, exclusion rule, and required invariants.

The record must bind the registration by repository-relative path, SHA-256
digest, and runtime commit. These fields prove which tracked registration was
used. They do not prove that the registration was public before the worker ran.
That ordering requires an external public timestamp, such as the time of a
Git commit hosted by an independent service.

The qualification uses one worker, with worker ordinal `0` and expected worker
count `1`. It runs:

- two warm-up control/runtime pairs;
- five measured control/runtime pairs;
- three cancellation runs; and
- two repetitions at each of four fault points.

Each pair contains:

- a control run through the pinned, patched Whisper backend without the
  transaction wrapper; and
- a runtime run through `NativeWhisperAdapter` with the strict single-lane CUDA
  profile.

Both paths use the registered checkpoint, decoded input, options, seed,
precision, device, process image, clock, and allocator probes. The seed is
reset before each run. Network access is disabled during measured work.

Model loading and artifact verification occur before timing. Runtime
measurements start before admission and end after the budget is restored.
Control measurements start before the backend call and end after backend
quiescence. The fault injector is installed only for fault runs.

## Raw event stream

The record contains one global event array. Sequence numbers start at zero.
Offsets are strictly increasing nanoseconds from one `time.monotonic_ns`
origin. Each event binds a sanitized worker and run identifier. Runtime events
also bind the session, request, transaction, and lease identifiers.

Runs do not overlap. Their canonical order is:

1. paired control and runtime warm-ups;
2. paired control and runtime measurements;
3. cancellation runs; and
4. each fault run followed by its post-recovery reuse run.

A successful runtime run must have this exact sequence:

```text
run start -> lease acquired -> completion fence -> result published
          -> transaction committed -> lease released -> budget restored
          -> run complete
```

A cancellation run must record incomplete decoder work before cancellation:

```text
run start -> lease acquired -> incomplete decoder step -> cancel requested
          -> backend quiescent -> transaction aborted -> lease released
          -> budget restored -> run complete
```

A fault run arms exactly two injections at one registered point. The first
injection fails normal close. The second fails automatic cleanup. Manual
recovery begins only after both failures leave the transaction retained:

```text
run start -> lease acquired -> fault armed -> fault triggered (1)
          -> fault triggered (2) -> transaction retained
          -> competing request rejected -> recovery started
          -> backend quiescent -> transaction aborted -> lease released
          -> budget restored -> run complete
```

The closed fault set is:

- decoder cleanup;
- completion-event creation;
- completion-event record; and
- completion-event synchronization.

These are harness-injected failures. They are not CUDA driver, hardware, or
process failures. A wider fault boundary requires a new schema version.

## Budget and allocator observations

Every runtime run records the full available resource vector before admission,
while the lease is held, at quiescence, and after release. The validator derives
the held and restored values from the registered capacity and reservation.

Every run records synchronized PyTorch CUDA allocator values:

- allocated and reserved baselines;
- final allocated and reserved values;
- peak allocated and reserved values; and
- peak deltas derived from the baselines.

The validator binds the logical CUDA device to the observed device index. It
checks resource capacity against physical device memory, allocator relations,
registered tolerances, and reservation limits. These readings cover the
PyTorch allocator. They do not measure every driver allocation or show that the
logical resource vector is physically enforced.

## Diagnostic summaries

Raw samples are mandatory and warm-ups are excluded. The validator recalculates
`min`, `p50`, `p95`, and `max` by nearest rank for:

- control and runtime wall time;
- cancellation request to backend quiescence;
- control and runtime peak allocated and reserved deltas; and
- recovery and trigger-to-quiescence latency for each fault point.

These values are diagnostic. Five measured pairs cannot support a performance
claim. `p99` is always `not_estimated` in this schema.

## Provenance validation

The record binds:

- clean runtime and backend repositories, commits, and trees;
- the qualification registration path and digest;
- the backend patch manifest path and digest;
- producer script, schema, validator, and dependency-inventory paths and
  digests;
- the container image digest and resolved environment versions;
- the worker, provider, region, GPU, and monotonic clock;
- checkpoint, input manifest, input bytes, decoded PCM, preprocessing options,
  decode options, and their digests; and
- the registered cell, exclusion rule, and outcome.

The command-line validator derives repository URL, `HEAD` commit, and `HEAD`
tree from the supplied checkouts. Both checkouts must be available and clean,
including no untracked files or dirty submodules. The runtime checkout must be
the checkout that contains the validator and every bound runtime artifact.
Caller-supplied Git identity strings are not accepted.

The validator also requires each artifact to be tracked at the runtime
checkout's `HEAD`. It compares the repository-relative path and the SHA-256
digest with the record. A matching digest at another path is not sufficient.

```powershell
python -B tools/validate_modal_native_cuda_qualification.py <record.json> `
  --runtime-checkout . `
  --backend-checkout <clean-patched-whisper-checkout> `
  --qualification-manifest experiments/native-cuda-qualification-v1.json `
  --patch-manifest patches/openai-whisper/SHA256SUMS `
  --producer-script infra/modal_native_cuda_qualification.py `
  --dependency-inventory infra/modal-native-cuda-requirements.lock
```

Repository URLs must use normalized credential-free HTTPS. Published paths
must be repository-relative.

## Failed cells and hygiene

The schema accepts `passed` and `failed` outcomes. A failed record must retain
its raw observations, name its failure class, and include a short summary. A
derived-invariant failure must mark the affected derived values as false.
Malformed provenance, inconsistent summaries, and broken raw-field derivations
are validation errors, not experimental failures.

The registration permits one attempt and no exclusions. Worker allocation or
startup failures that occur before a record can be produced therefore require
an external campaign log. The local record validator cannot establish that the
log is complete.

The validator rejects duplicate JSON keys, non-finite values, absolute home
paths, unsafe repository URLs, and several known credential formats. This is a
best-effort publication check, not a complete secret scanner. Human review and
the repository-wide secret scan are still required.

## Claim boundary

A valid record qualifies only its exact registered source, workload, worker,
hardware, and environment. It does not establish production readiness, general
CUDA correctness, physical kernel preemption, untested failure handling,
latency or throughput performance, or behavior on another configuration.

A future performance campaign must use a new evidence schema or version. That
format must require the full performance metrics, repeated fresh workers,
matched measurement probes on control and runtime paths, raw samples,
registered aggregation rules, and uncertainty estimates defined in
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md).
