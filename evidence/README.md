# Integration evidence

The [bounded-preview CPU smoke](native-cpu-tiny-en-jfk-bounded-previews-2026-09-05.json)
records three local JFK replays, same-PCM control results, revision events, and
resource release. It is a diagnostic smoke record, not a registered CUDA
qualification or a live-performance benchmark. See the
[test scope](../docs/BOUNDED_STREAMING.md#local-cpu-smoke).

This directory contains nine committed records from real backend runs:

- `native-cpu-tiny-en-jfk-2026-09-03.json` records one
  `NativeWhisperAdapter` transaction.
- `native-cpu-tiny-en-jfk-interleaving-2026-09-03.json` records two staged
  decode runs on one loaded model, early cleanup of one run, and completion of
  the other.
- `native-cpu-tiny-en-jfk-threaded-2026-09-04.json` records the same isolation
  case across two operating-system threads in the patched decoder backend.
- `native-cpu-tiny-en-jfk-runtime-concurrency-2026-09-04.json` records two
  transactions through the experimental two-lane runtime adapter.
- `modal-t4-tiny-en-jfk-cuda-readiness-gcp-2026-09-04.json` and
  `modal-t4-tiny-en-jfk-cuda-readiness-aws-2026-09-04.json` record two separate
  direct-backend CUDA executions on Modal T4 workers. The workers ran in GCP
  `asia-southeast2` and AWS `us-west-2`.
- `modal-t4-tiny-en-jfk-native-cuda-transaction-aws-2026-09-04.json` and
  `modal-t4-tiny-en-jfk-native-cuda-transaction-gcp-2026-09-04.json` record two
  native-adapter transactions on Modal T4 workers. The workers ran in AWS
  `us-west-2` and GCP `europe-west2`.
- `modal-t4-tiny-en-jfk-native-cuda-qualification-v6-2026-09-04.json` records
  the registered single-worker qualification on an AWS `us-west-2` T4.

Each record identifies the runtime revision, backend source tree, model
checkpoint, input, environment, and observed outcome.

`modal-native-cuda-qualification.schema.json` defines the draft record for one
fixed native CUDA qualification cell. The cell uses one worker, two warm-up pairs, five
measured pairs, three cancellation runs, and two repetitions at each of four
fault points. Its p50 and p95 summaries are diagnostic; p99 is
`not_estimated`. The version-six record satisfies this contract.

The active registration is
[`experiments/native-cuda-qualification-v6.json`](../experiments/native-cuda-qualification-v6.json).
A record binds its path, digest, and runtime commit. The local validator cannot
prove that the registration was public before execution; that requires an
external public timestamp. The same version-six manifest and digest were public
in commit `bf46687d6c0f837426d85a1f97c60dd64128f9ed` before execution. A future
performance campaign requires a new schema or version, full performance
metrics, and matched control/runtime probes. See
[`docs/CUDA_QUALIFICATION_CONTRACT.md`](../docs/CUDA_QUALIFICATION_CONTRACT.md) and
[`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

During the version-one dispatch, the operator observed repeated Modal
deserialization errors and stopped the run before any inference. Its
append-only
[`attempt-started` receipt](modal-native-cuda-qualification-v1-attempt-2026-09-04.jsonl)
is retained without a synthetic terminal event. The receipt alone proves the
local dispatch boundary, not the remote worker state or the reported cause.
The version-two producer required the canonical module invocation and rejected
a file-path invocation before dispatch.

For version two, the operator observed that the worker reached the registered
T4 cell and stopped during its first control decode because the transcript
digest did not match the registration. Its
[`attempt-failed` receipt](modal-native-cuda-qualification-v2-attempt-2026-09-04.jsonl)
is retained. The receipt itself records the `gpu-campaign` stage, exception
type, and message digest; it does not independently prove the allocated GPU or
preserve the observed transcript. The earlier passing AWS record used
timestamp-free decoding. Version two registered timestamp-token decoding with
the timestamp-free transcript digest. Version three aligned
`without_timestamps` with that digest and was executed once.

For version three, Modal logs show that the worker completed the registered
warm-up, measured, and cancellation runs. The first `cleanup` fault scenario
then completed retention and recovery before an incorrect event-order
expectation stopped the harness. The
[`attempt-failed` receipt](modal-native-cuda-qualification-v3-attempt-2026-09-04.jsonl)
records the `gpu-campaign` stage, exception type, and message digest. It does not
contain the transient event trace, and no qualification record was published.
This was a harness assertion failure, not a Whisper, CUDA, driver, or hardware
failure. Version four records `fault-armed` only after the injection plan exists
and before the protected operation acquires its lease. It also records that
harness faults occur before the named delegate call.

For version four, Modal logs show that the single attempt completed the
registered GPU campaign on a T4 and reached the post-campaign dependency
inventory. Inventory then failed because the environment exposed more than one
visible metadata version for `idna`. The
[`attempt-failed` receipt](modal-native-cuda-qualification-v4-attempt-2026-09-04.jsonl)
records the `gpu-campaign` stage, exception type, and message digest. The
receipt does not contain the transient GPU observations, and no qualification
record was published. The attempt supports no passing qualification claim.

Version five changed the inventory to one metadata record returned by
`importlib.metadata.distribution(name)` for each normalized name discovered by
`importlib.metadata.distributions()`. Modal logs show that its single T4
attempt completed the registered GPU campaign and constructed a schema-valid
record. Internal semantic validation then rejected the record because the
harness required distribution metadata for Modal to equal the
platform-injected module version. A separate CPU diagnostic inspected the same
Modal image (`im-GLEsEGZRFsSkRtNGoxP69W`) in run
`ap-u7s5B07rhiT7g276cf2856`. It observed Modal 1.5.5 at
`/pkg/modal/__init__.py` with no Modal distribution metadata. It also observed
Torch 2.6.0+cu124 from matching module and distribution metadata, and selected
idna 3.19 while idna 3.10 remained visible under `/__modal/deps`. This
diagnostic was not a qualification attempt. The
[`attempt-failed` receipt](modal-native-cuda-qualification-v5-attempt-2026-09-04.jsonl)
records only the failure stage, exception type, and message digest. Modal logs
support the detailed execution path and cause. No qualification record was
published, so the attempt supports no passing qualification claim.

Version six treats the platform-injected Modal module version and the
distribution inventory as separate provenance observations. It does not
require a Modal distribution entry. It retains exact Torch module and
distribution equality because Torch executed the workload. It collects the
dependency inventory and runs a CPU contract rehearsal before creating the GPU
attempt receipt. Its single attempt produced a
[`record-published` receipt](modal-native-cuda-qualification-v6-attempt-2026-09-04.jsonl)
and a
[passing qualification record](modal-t4-tiny-en-jfk-native-cuda-qualification-v6-2026-09-04.json).
The record binds runtime commit
`9c2494234f08b24325d427ea422818b24f460c0c`, an AWS `us-west-2` Tesla T4,
the `tiny.en` checkpoint, and the pinned JFK input. It passed the schema and
semantic validator, including exact output compatibility, cancellation,
retention, recovery, resource-ledger, completion-fence, and publication
relations. The record SHA-256 is
`e3374c6f39f0739336706ce161e3836396f440b01c6254210d90e806678476bb`.
The receipt SHA-256 is
`641c3cd248de75fc7c96096fee8aee6ff1000adb3ba06cbe916db9c096255d02`.
This is qualification evidence for one fixed cell, not a performance or
production-readiness result. The inventory describes distribution metadata; it
does not prove the bytes of imported modules. Source and build-input hashes
remain the integrity anchors.

`modal-cuda-readiness.schema.json` defines the version-one T4 record. Both
committed records passed its schema and semantic validator. They bind the same
runtime commit, backend tree, patch manifest, model checkpoint, decoded PCM,
and transcript. A passing record covers the direct patched backend only. The
CUDA rejection boundary was exercised; no runtime transaction was admitted or
executed. No record covers worker admission, the transaction lifecycle, a CUDA
completion fence, or a performance benchmark. See
[`docs/MODAL_GPU_VALIDATION.md`](../docs/MODAL_GPU_VALIDATION.md).

`modal-native-cuda-transaction.schema.json` defines the separate version-two
adapter transaction record. Both committed records bind runtime commit
`28415364d167f71d5b0cdf441b0738ae4689b683` and tree
`9b0c3f5788635bb4a8044307d3f13dfec5690131`. Each record covers one
instrumented successful transaction, cooperative cancellation, injected fence
failure and recovery, post-recovery reuse, and one unproxied native control
transaction. They also bind source, model, input, environment, trace order,
terminal state, and resource state. See
[`docs/MODAL_NATIVE_CUDA_VALIDATION.md`](../docs/MODAL_NATIVE_CUDA_VALIDATION.md).

The version-two records passed the closed schema and semantic validator. Their
trace event sequences, state snapshots, transcript, source identities, model
state, and resource outcomes match across the two providers. The injected
synchronization failure occurs in the harness before the delegate call. It does
not represent a physical CUDA driver failure. The records are not performance
or production readiness claims.

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
