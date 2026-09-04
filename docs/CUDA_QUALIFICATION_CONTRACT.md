# Draft native CUDA qualification contract

`evidence/modal-native-cuda-qualification.schema.json` defines the
`1-draft` record format for one fixed native CUDA qualification cell. Version
six is the current executed campaign under this format. Its committed evidence
record satisfies the schema and semantic validator. This schema is not a
performance-benchmark schema.

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
[`experiments/native-cuda-qualification-v6.json`](../experiments/native-cuda-qualification-v6.json).
It fixes the source policy, backend revision, T4 worker profile, model,
checkpoint, input, decode options, resource contract, measurement boundaries,
sample counts, fault points, exclusion rule, and required invariants.

The [evidence index](../evidence/README.md) records all attempts and their
limits. Version six has one passing
[qualification record](../evidence/modal-t4-tiny-en-jfk-native-cuda-qualification-v6-2026-09-04.json)
and one append-only
[attempt receipt](../evidence/modal-native-cuda-qualification-v6-attempt-2026-09-04.jsonl).
The same registration path and digest were public in commit
`bf46687d6c0f837426d85a1f97c60dd64128f9ed` before the worker ran.

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
run start -> fault armed -> lease acquired -> fault triggered (1)
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
process failures. Each injection stops immediately before its named delegate
call. A wider fault boundary requires a new schema version.

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
- producer script, trace layer, schema, validator, and direct image-input paths
  and digests;
- the Modal image object identifier, a sorted Python distribution metadata
  inventory and its canonical digest, and exact environment versions;
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
Before GPU work begins, the producer also binds the tracked input manifest to
that `HEAD`, verifies its digest, and checks its registered fixture fields
against the cell. It collects the dependency inventory before the campaign.
Modal supplies its function module outside the image's distribution metadata;
the worker records that module version independently. The Torch module and
distribution versions must match exactly because the imported Torch module
executes the registered workload.

## Run the registered cell

Run this command only from the clean public commit that contains the producer,
registration, schema, validator, trace layer, image-input file, and input
manifest:

```powershell
$env:WHISPER_RUNTIME_COMMIT = git rev-parse HEAD
$env:WHISPER_MODAL_ENABLE_REMOTE_RESOURCES = "1"
python -m modal run -m infra.modal_native_cuda_qualification `
  --preflight-only

python -m modal run -m infra.modal_native_cuda_qualification `
  --output artifacts/modal/native-cuda-qualification-v6.json `
  --confirm-paid-gpu
```

The first command runs the registered image definition on a CPU worker. It
binds the source, checks the imported modules and dependency inventory, and
runs the record and producer contract tests. It does not create an attempt
receipt or guarantee the image object ID of a later invocation. The second
command repeats that gate. Its CPU and GPU functions reference the same Modal
Image object. It then primes the pinned model cache and requests one T4 with
the Modal selectors `aws` and `us-west`. The GPU worker must report
`CLOUD_PROVIDER_AWS` and `us-west-2`. The command then writes one record.
After the CPU gate passes and before the registered campaign begins, it creates
the attempt receipt. The producer accepts only the registered output path shown
above and refuses to overwrite its record or receipt.

The module form is mandatory. A file-path invocation changes the local module
identity and can make Modal unable to deserialize the remote function. The
producer rejects that form before it creates an attempt receipt.

## Validate a record

```powershell
python -B tools/validate_modal_native_cuda_qualification.py <record.json> `
  --runtime-checkout . `
  --backend-checkout <clean-patched-whisper-checkout> `
  --qualification-manifest experiments/native-cuda-qualification-v6.json `
  --patch-manifest patches/openai-whisper/SHA256SUMS `
  --producer-script infra/modal_native_cuda_qualification.py `
  --image-inputs infra/modal-native-cuda-image-inputs.lock
```

Repository URLs must use normalized credential-free HTTPS. Published paths
must be repository-relative.

Modal exposes an opaque `im-...` image object identifier, not an OCI content
digest. The record therefore stores that identifier. The lock file binds the
direct Python requirement specifiers passed to pip and uv. It is not a complete
container package manifest and does not hash base-image layers, apt packages,
or transitive wheel files. The tracked client constraint pins the Modal SDK
used to define the function, and the worker records the injected runtime module
version. The observed, sorted Python distribution inventory stores one metadata
record returned by `importlib.metadata.distribution(name)` for each normalized
name discovered by `importlib.metadata.distributions()`. It does not prove
imported module bytes. Source and build-input hashes remain the integrity
anchors. These fields do not turn the Modal identifier into a content digest.

The resource contract is the logical capacity of this one-lane runtime
profile. It equals the per-run reservation, so the available logical vector is
zero while a run owns the lane. `gpu.total_memory_bytes` separately records
the physical device capacity. The logical budget is not a physical GPU memory
enforcement mechanism.

## Failed cells and hygiene

The schema accepts `passed` and `failed` outcomes. A failed record must retain
its raw observations, name its failure class, and include a short summary. A
derived-invariant failure must mark the affected derived values as false.
Malformed provenance, inconsistent summaries, and broken raw-field derivations
are validation errors, not experimental failures.

The registration permits one attempt and no exclusions. After the CPU preflight
passes and before cache-prime or GPU campaign dispatch, the producer creates
`<record>.attempt.jsonl` with exclusive-create semantics.
It appends either `record-published` or `attempt-failed` after an ordinary
Python exception. A process termination can leave only `attempt-started`; the
producer does not relabel that incomplete attempt. A second invocation refuses
the existing receipt and cannot dispatch another worker for the same output.

The receipt is not a qualification record. Startup, allocation, and incomplete
campaign failures cannot satisfy the full evidence schema because the required
GPU observations do not exist. `publish_all_attempts` means that every local
dispatch attempt has this append-only receipt; it does not mean that an invalid
campaign is padded into a qualification record. External logs remain necessary
to establish behavior outside the local process boundary.

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
