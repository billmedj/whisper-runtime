# Native CUDA transaction validation

This optional check runs `NativeWhisperAdapter` on one Modal T4. It records a
bounded transaction case, not a general CUDA or production-readiness claim.

The check covers four outcomes on the pinned `tiny.en` model and JFK fixture:

1. one successful transaction;
2. one cooperative cancellation after a real token step;
3. one injected completion-event failure, quarantine, blocked reuse, and manual
   recovery;
4. one successful control transaction after recovery with the native backend
   and CUDA objects restored to their original, unproxied implementations.

The first three cases use delegate proxies to record the native stages, private
CUDA stream, completion event, session publication, and resource release. The
proxies call the real PyTorch and patched Whisper objects. The control case runs
outside that patch. It checks that the observation layer is not required for a
successful adapter transaction.

## Evidence boundary

The record establishes the following facts only for its exact source revision,
dependency image, T4, model, input, and decode options:

- worker admission occurs before CUDA stream creation;
- native work observed by the harness uses one private CUDA stream;
- each observed decode run owns a child random generator on `cuda:0`, distinct
  from both task-owned generator sources;
- a successful CUDA event fence returns while the request is still running,
  the session is still unpublished, and the queue slot and lease remain held;
- session publication and resource release follow that fence;
- cancellation is requested by the controller without a controller-side CUDA
  call, after the first real token step returns `False` and the run reports that
  it is still incomplete; it is then acknowledged at the next checkpoint;
- a failed event fence retains the transaction and its capacity;
- new work on the bound model is rejected while that transaction is retained;
- successful manual recovery aborts the retained request and releases capacity;
- the adapter can decode the exact fixture again after recovery;
- the strong model-state digest and hook inventory match before and after all
  cases.

Before the scenarios start, the harness also requires evaluation mode, every
model parameter and buffer on `cuda:0`, and every floating model tensor in
FP32. These checks bind the recorded execution profile to the loaded model.

The injected failure is a deterministic harness fault before the delegate
event synchronization call. It is evidence for the runtime recovery path. It
is not evidence about a real CUDA driver failure.

The execution profile uses FP32. CUDA memory is measured around the successful
case and compared with the declared ledger value. The ledger does not enforce
physical device memory. The record is not a latency or throughput benchmark.
Cancellation is cooperative between decode steps; it does not preempt a CUDA
kernel. One passing T4 record does not establish support for other devices,
models, decode modes, concurrent CUDA transactions, or production workloads.
Manual recovery runs synchronously on the originating Python thread after the
retained error. This case does not establish cross-thread or process recovery.

## Isolation and source identity

The image clones the full public commit supplied in
`WHISPER_RUNTIME_COMMIT`. The run records that commit, its Git tree, and hashes
of the adapter and transaction-critical runtime files. It also fixes the
patched Whisper tree, patch manifest, checkpoint, audio file, and decoded PCM.

The model-cache prime function can use the network. The single-use GPU function
mounts that cache read-only, blocks outbound network access, and restricts
access to other Modal resources. The function probes both controls and records
the results. Do not put a Modal token or another credential in a command,
record, issue, or artifact.

The adapter identity probe used inside a transaction checks the prebound Python
object and exact device only. It does not hash CUDA tensors or copy them to the
CPU. Strong state hashes are taken before the scenarios and after a final
global CUDA synchronization. This keeps the transaction event, rather than the
identity probe, as the observed completion fence.

## Local checks without a GPU

Use a clean checkout of the exact public commit that you intend to run:

```powershell
py -3.13 -m pip install -e ".[validation,quality,modal-validation]" "build>=1.2,<2"
$env:WHISPER_RUNTIME_COMMIT = (git rev-parse HEAD).Trim()
$env:WHISPER_MODAL_ENABLE_REMOTE_RESOURCES = "1"
py -3.13 -B -m unittest discover -s tools -p "test_modal_native_cuda_record.py" -v
py -3.13 -m ruff check infra tools
py -3.13 -m ruff format --check infra tools
py -3.13 -m compileall -q infra tools
py -3.13 -c "import infra.modal_native_cuda_validation"
```

Modal definitions are lazy. The last command constructs them locally; it does
not build the image or start a remote function.

## Run the paid check

Authenticate the pinned Modal client once:

```powershell
py -3.13 -m modal setup
py -3.13 -m modal token info
```

Then run from the same clean, public commit:

```powershell
$env:WHISPER_RUNTIME_COMMIT = (git rev-parse HEAD).Trim()
$env:WHISPER_MODAL_ENABLE_REMOTE_RESOURCES = "1"
$output = "artifacts/modal/$env:WHISPER_RUNTIME_COMMIT/native-cuda-transaction.json"
py -3.13 -m modal run --env=main -m infra.modal_native_cuda_validation `
  --output $output `
  --confirm-paid-gpu
```

There are two independent opt-ins. The environment variable permits Modal
resource definitions. The CLI flag permits the cache and T4 calls. The output
must be a new `.json` path below `artifacts/modal/`; the harness refuses to
overwrite it.

Check current [Modal pricing](https://modal.com/pricing) before a run. The
repository does not dispatch a GPU during ordinary tests or CI.

## Validate a record

The schema closes every object. The semantic validator also checks source
bindings, exact claims, trace ordering, terminal states, fence state, resource
restoration, failure containment, transcript equality, and secret and path
sanitization.

```powershell
$commit = (git rev-parse HEAD).Trim()
$tree = (git show -s --format=%T $commit).Trim()
py -3.13 tools/validate_modal_native_cuda_record.py <record.json> `
  --expected-runtime-commit $commit `
  --expected-runtime-tree $tree
```

Review the JSON and workflow log before committing a record. A future change to
the adapter, trace contract, scenario, or claim boundary requires a new schema
version. Do not broaden version-two evidence after publication.
