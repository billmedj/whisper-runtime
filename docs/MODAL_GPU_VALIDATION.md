# Modal GPU validation

This check answers one question: can the pinned patched Whisper backend decode
the pinned `tiny.en` case on a Modal T4 with CUDA?

It does not run a transaction through `NativeWhisperAdapter`. The current
adapter is CPU-only. The check calls it once with a CUDA tensor and requires a
rejection before worker admission. A passing record therefore contains
`runtime_adapter_exercised=false`, `worker_admission_exercised=false`, and
`cuda_completion_fence_exercised=false`.

The caller must provide the full public commit that contains the harness. The
remote image clones that commit and records its commit and tree. The check also
fixes the CPU-only adapter digest, OpenAI Whisper base and patched trees, patch
manifest, `tiny.en` checkpoint, and JFK input. A future CUDA adapter requires a
new check and evidence schema. Do not expand the meaning of a version-one
record.

## Remote boundary

The run has three remote phases:

1. Modal builds an image from public, content-addressed Git sources and pinned
   Python packages.
2. A CPU function downloads `tiny.en`, verifies its SHA-256 digest, and commits
   it to the `whisper-runtime-model-cache-v1` Volume.
3. A single-use T4 function mounts that Volume read-only, blocks network access,
   and restricts access to other Modal resources. It runs a staged decode, an
   identical reuse decode, and the adapter boundary check.

The GPU function records synchronized CUDA time, measured PyTorch allocation
peaks, GPU and driver identity, dependency versions, source identities, model
state, decoded PCM identity, output, and both isolation probes. These timings
are diagnostics. They are not a performance benchmark. Device memory is
measured but not enforced as a runtime budget.

The image build and cache-prime phase need network access. The T4 phase does
not. No Modal token, model-provider key, GitHub credential, or user path enters
the image or evidence record.

## Local checks without credentials or GPU

Use CPython 3.13 in a local environment:

```powershell
py -3.13 -m pip install -e ".[validation,quality,modal-validation]" "build>=1.2,<2"
$env:WHISPER_RUNTIME_COMMIT = (git rev-parse HEAD).Trim()
$env:WHISPER_MODAL_ENABLE_REMOTE_RESOURCES = "1"
py -3.13 -B -m unittest discover -s tools -p "test_modal_cuda_record.py" -v
py -3.13 -m ruff check infra tools
py -3.13 -m ruff format --check infra tools
py -3.13 -m compileall -q infra tools
py -3.13 -c "import infra.modal_gpu_validation"
```

The final command constructs the Modal definitions locally. Modal handles are
lazy, so it does not build an image or start a remote function.

## First paid run

Install the pinned client and authenticate once:

```powershell
py -3.13 -m pip install "modal==1.5.5"
py -3.13 -m modal setup
py -3.13 -m modal token info
```

Set the fixed source revision, then run the check:

```powershell
$env:WHISPER_RUNTIME_COMMIT = (git rev-parse HEAD).Trim()
$env:WHISPER_MODAL_ENABLE_REMOTE_RESOURCES = "1"
$output = "artifacts/modal/$env:WHISPER_RUNTIME_COMMIT/cuda-backend-readiness.json"
py -3.13 -m modal run --env=main -m infra.modal_gpu_validation `
  --output $output `
  --confirm-paid-gpu
```

The environment opt-in is checked before the image and functions are defined.
It permits Modal to initialize or build declared resources. The explicit CLI
flag is a second guard before cache or GPU function dispatch. The output path
is required and must be a new `.json` path below `artifacts/modal/`. On success,
the validated record is written there. That directory is ignored by Git.

The published Modal T4 rate was $0.000164 per second ($0.5904 per hour) on
September 4, 2026. The GPU function has separate 900-second startup and
execution timeouts. A conservative 1,800-second T4 exposure is $0.2952 at that
rate. Image build, CPU, memory, storage, data transfer, credits, and future
pricing affect the actual charge. Check
[Modal pricing](https://modal.com/pricing) before each run.

## Manual GitHub run

The `Modal GPU readiness` workflow runs only through `workflow_dispatch` and
requires its paid-run checkbox. Add these two repository Actions secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Use a Modal token created for CI. Do not paste either value into a workflow,
issue, log, artifact, or evidence file. Select an existing Modal environment in
the workflow form. A separate `validation` environment is preferable; `main`
is the default because every new Modal workspace has it.

The workflow stores the record as a 30-day GitHub artifact. The repository does
not commit a GPU result until the artifact has passed review.

## Published records

Two reviewed records are committed under `evidence/`. They use the same pinned
runtime, patched backend, model checkpoint, decoded PCM, and expected output.
Modal placed one T4 worker in GCP `asia-southeast2` and the other in AWS
`us-west-2`. Both workers produced the exact transcript, preserved the recorded
model-state fingerprint, denied the outbound network probe, and denied a write
to the read-only model cache.

These records establish only the direct patched-backend cases that they state.
The CUDA rejection boundary was exercised; no runtime transaction was admitted
or executed. The records do not establish adapter-level CUDA execution, memory
enforcement, latency, throughput, or production readiness.

## Record validation

The JSON Schema closes every object and fixes the scope-defining values. The
semantic validator checks cross-field timing, memory, admission, claim, and
secret-sanitization rules:

```powershell
$commit = (git rev-parse HEAD).Trim()
$tree = (git show -s --format=%T $commit).Trim()
py -3.13 tools/validate_modal_cuda_record.py <record.json> `
  --expected-runtime-commit $commit `
  --expected-runtime-tree $tree
```

The validator checks a historical record against the record schema. It does
not require a future checkout to retain the CPU-only adapter source.
