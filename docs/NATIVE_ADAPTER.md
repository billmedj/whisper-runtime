# Native CPU adapter

`NativeWhisperAdapter` connects one suspendable Whisper decode run to one
runtime transaction. It is an experimental, CPU-only boundary for a single
unbatched 30-second mel window. The package requires Python 3.10 or later.

The adapter requires a Whisper build that provides:

- `DecodingOptions(generator=...)`;
- `DecodingTask._start_run(mel)`;
- a run with `prefill()`, `step()`, `complete`, `finalize()`, and `cleanup()`.

The complete backend patch series, its pinned OpenAI Whisper base commit, file
digests, and application instructions are in
[`patches/openai-whisper`](../patches/openai-whisper/README.md). The adapter is
not compatible with an unmodified OpenAI Whisper release.

The source distribution contains the patch series. The wheel contains the
runtime package only. Wheel users must obtain the backend patches from the
source archive or repository.

PyTorch and Whisper are loaded when `decode_window()` starts. Importing
`whisper_runtime` does not load either package.

## Transaction boundary

The adapter reserves the worker's full resource capacity. The worker must have
`queue_capacity=1`. One model object is bound to one adapter kind for that
object's lifetime. Load a new model object when migrating between the native
and legacy adapters.

One call performs these operations:

1. Draw a seed from the transaction-local random stream.
2. Create a new CPU `torch.Generator` from that seed.
3. Submit run creation and encoder work.
4. Submit prefill.
5. Submit each token step separately, with a checkpoint after every step.
6. Submit finalization and require exactly one decode result.
7. Commit one `WindowResult` containing its text and declared time span.

The CPU execution scope owns the decode handle. Its completion fence calls
`cleanup()` before the runtime releases the lease. Cancellation is cooperative:
it takes effect at the checkpoint after a submitted stage or token step. It
does not interrupt a CPU kernel that is already running.

An aborted attempt does not publish its transaction-local random state. A new
request with the same `rng_seed` recreates the same generator seed and restarts
the window from its beginning.

## Use

```python
from whisper_runtime import Budget, ModelSnapshot, RequestState, ResourceVector
from whisper_runtime import Session, Worker
from whisper_runtime.adapters import (
    NativeDecodeOptions,
    NativeExecutionProfile,
    NativeWhisperAdapter,
)

snapshot = ModelSnapshot(
    model_id="tiny.en",
    revision="pinned-revision",
    backend="pytorch-cpu",
    fingerprint="sha256:checkpoint-digest",
)
capacity = ResourceVector(
    memory_bytes=1_000_000_000,
    compute_units=1,
    stream_slots=1,
)
worker = Worker(
    "tiny-en-cpu",
    snapshot,
    Budget(capacity),
    queue_capacity=1,
)
adapter = NativeWhisperAdapter(
    worker,
    model,
    identity_probe,
    NativeExecutionProfile("tiny.en/cpu", capacity),
)

state = adapter.decode_window(
    session=Session("session-1"),
    request=RequestState(
        "request-1",
        "session-1",
        snapshot,
        rng_seed=7,
    ),
    window_id="window-1",
    mel=mel,
    start_ms=0,
    end_ms=30_000,
    options=NativeDecodeOptions(language="en", beam_size=5),
)
```

`mel` must have the exact shape `(model.dims.n_mels, whisper.audio.N_FRAMES)`.
The adapter adds the batch dimension. The identity probe must bind the loaded
weights to the declared `ModelSnapshot`; metadata alone is not a strong
checkpoint identity.

This adapter does not yet support audio batches, CUDA fences, stage-specific
resource costs, durable mid-window checkpoints, alignment, or streaming.

## Local smoke test

`tools/smoke_native_whisper.py` runs one real window and reports the committed
session version, transcript, elapsed time, queue depth, and resource release.
It verifies that the imported Whisper worktree is clean and that its full Git
revision equals `--revision`. It also rejects an ancestor repository, an
untracked or ignored package file, and an imported module that Git does not
track. `--expected-model-fingerprint` binds the loaded state to a recorded
SHA-256 fingerprint. The command fails unless the request commits, the first
session version is published, the queue returns to zero, and the declared
budget is restored. It does not download a checkpoint when `--download-root`
already contains the selected model.

```powershell
$env:PYTHONPATH = "src;C:\path\to\suspendable-whisper"
python tools\smoke_native_whisper.py C:\path\to\audio.flac `
  --model tiny.en `
  --download-root C:\path\to\model-cache `
  --revision <full-source-commit> `
  --expected-model-fingerprint "sha256:<loaded-state-digest>" `
  --expected-text "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
```

The unit tests use controlled backend doubles to force lifecycle failures and
race boundaries deterministically. The smoke command is the integration check
against the patched decoder and a real model checkpoint.

## Same-model interleaving check

`tools/verify_native_interleaving.py` tests the staged decoder below the
serialized runtime adapter. It loads one model, records an isolated baseline,
and creates two request-owned decode runs from the same model.

The tool follows a recorded token-step schedule. It cleans one run after one
decoder step and continues the other to completion. The check requires:

- distinct run, inference, decoder, token-storage, and KV-cache objects;
- no change to the survivor cache when the other run is cleaned;
- rejection of further work by the cleaned run;
- a survivor result that matches the isolated baseline within the recorded
  scalar tolerance, which is zero in CI;
- no net change to model state or registered hooks across the full check;
- a successful control decode after both overlapping lifetimes end.

The tool emits a JSON record and validates every required assertion. The record
format is defined in `evidence/native-interleaving.schema.json`.

This check does not send two transactions through `NativeWhisperAdapter`. The
adapter still requires `queue_capacity=1`. The check does not claim parallel
kernel execution, thread-safe execution, or higher throughput.

## Same-model OS-thread isolation check

`tools/verify_native_threaded.py` repeats the decoder isolation case in two
worker threads. It prepares both encoder outputs sequentially and creates a
separate `DecodingTask` for each worker. Each worker creates and operates its
own run. Barriers and events fix the test order. Each thread enters its first
outer `decoder.forward` call. A barrier in the first decoder block holds both
outer calls before either continues. Temporary instrumentation records the
owning Python and native thread identifiers and the start and end of each outer
call. The check requires those two intervals to overlap.

The cancelled worker cleans its run after one token step. The other worker
continues to completion. The check requires:

- two distinct Python and native worker thread identifiers;
- both instrumented decoder calls to run on their recorded owner threads;
- overlapping intervals for the two outer decoder calls;
- separate request state and disjoint KV-cache storage;
- unchanged survivor cache across cancellation of the other run;
- a survivor result equal to the isolated baseline within the recorded scalar
  tolerance, which is zero in CI;
- restoration of the instrumented decoder and first-block methods and no net
  change to model state or execution-hook registries;
- successful model reuse after both worker runs end.

The tool emits JSON. `tools/validate_threaded_record.py` validates the record
against `evidence/native-threaded.schema.json` and checks relations that JSON
Schema cannot express, including thread ownership, interval overlap, fixture
identity, result equality, and patch-manifest identity. The record also binds
the language, temperature, timestamp mode, and numeric precision options used
by the decoder.

The check exercises `whisper.decoding.DecodingTask._start_run` in the patched
Whisper backend. It runs below `NativeWhisperAdapter`; it does not exercise
concurrent encoder calls, the runtime scheduler, or adapter concurrency.
Overlapping decoder-call bodies do not show that PyTorch kernels execute
simultaneously. The check makes no claim about kernel overlap, throughput,
CUDA, production readiness, or general thread safety across other models,
devices, operating systems, or dependency versions. The adapter remains
serialized.
