# Native adapter

`NativeWhisperAdapter` connects one suspendable Whisper decode run to one
runtime transaction. It is an experimental boundary for a single unbatched
30-second mel window. CPU is the default and validated path. A strict
single-lane CUDA path is implemented for testing. The package requires Python
3.10 or later.

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

For a repository checkout, the
[real backend quick start](REAL_BACKEND_QUICKSTART.md) creates the pinned
backend and an isolated Python environment with one command. It also provides a
minimal native transaction example. The bootstrap does not download a model.

PyTorch and Whisper are loaded when `decode_window()` or `start_window()` starts. Importing
`whisper_runtime` does not load either package.

## Transaction boundary

`NativeExecutionProfile.device` fixes the execution device. It defaults to
`cpu`. CUDA profiles require a canonical `cuda:N` value; bare `cuda` is
rejected. `NativeExecutionProfile.resources` is the fixed declared cost of one
transaction. The default profile admits one transaction and requires
`queue_capacity=1`. An experimental CPU profile can set
`max_concurrent_decodes=2`. That profile requires `queue_capacity=2` and a
worker capacity equal to twice the per-transaction resource vector. Admission
fails before inference when either bound is wrong.

One model object is bound to one adapter kind, worker, profile, and concurrency
limit for that object's lifetime. Load a new model object when migrating
between profiles or adapters.

The blocking API performs these operations:

1. Admit the request and start its execution scope.
2. Draw a seed from the transaction-local random stream.
3. Submit model validation, task creation, and input preparation.
4. Check for cancellation before encoder work.
5. Submit run creation and encoder preparation.
6. Submit prefill.
7. Submit each token step separately, with a checkpoint after every step.
8. Submit finalization and require exactly one decode result.
9. Commit one `WindowResult` containing its text and declared time span.

`start_window()` uses the same path but returns after step 6. Its
`NativeWindowRun` exposes one-token `step()`, cooperative `cancel()`, supervisor
`stop()`, explicit `finish()`, and idempotent `close()`. `finish()` refuses an
incomplete run; it never executes hidden token steps. The thread that starts
the run owns its driver methods. Another thread may request cancellation,
which the owner observes before the next submission. It may also call `stop()`:
the runtime signals a live owner, but it fences and reclaims the transaction
when that owner has exited.

Use the handle as a context manager. Leaving the context without a commit
aborts the transaction, fences the backend, and releases or quarantines the
lease under the same rules as `decode_window()`.

`True` from `cancel()` or `stop()` means that state changed, a signal was
delivered, or recovery progressed. It does not by itself mean that the lease
was released. Read `capacity_released` for that fact. `closed` only reports
that driver methods can no longer advance the run. `stop()` can be retried
after a cleanup failure. A failed completion fence raises an error and keeps
the transaction quarantined, with its lease held, until recovery succeeds.

The execution scope owns the decode handle. Its completion fence calls
`cleanup()` before the runtime releases the lease. Cancellation is cooperative:
it takes effect at the checkpoint after a submitted stage or token step. It
does not interrupt work that is already running.

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

Use the managed handle when an external scheduler owns progress:

```python
with adapter.start_window(
    session=session,
    request=request,
    window_id="window-1",
    mel=mel,
    start_ms=0,
    end_ms=30_000,
) as run:
    while not run.complete:
        run.step()  # one token step, then a cancellation checkpoint
    state = run.finish(committed_through_ms=30_000)
```

`committed_through_ms` is an explicit caller assertion that the audio prefix
`[0, value)` is immutable. A later commit can advance this boundary or leave it
unchanged. It cannot move the boundary backwards, publish a result that starts
inside the committed prefix, or advance beyond that result's end. The session
keeps the boundary when old window records are evicted. The runtime does not
infer finality from a window end time because committed windows may contain
gaps.

`mel` must have the exact shape `(model.dims.n_mels, whisper.audio.N_FRAMES)`.
The adapter adds the batch dimension. The identity probe must bind the loaded
weights to the declared `ModelSnapshot`; metadata alone is not a strong
checkpoint identity.

This adapter does not yet support audio batches, stage-specific resource costs,
durable mid-window checkpoints, alignment, or streaming. The committed CUDA
records cover only the pinned single-lane case described below. They do not
establish general CUDA compatibility, memory bounds, latency, or throughput.

## Strict CUDA profile

The CUDA profile is deliberately narrow:

- `device` must be an exact `cuda:N` value;
- `max_concurrent_decodes` must be `1`;
- one transaction must reserve positive memory and compute values and exactly
  one stream slot;
- the model must already be on that exact device;
- the input mel must remain a CPU `float32` tensor.

The adapter creates the CUDA stream only after the worker admits the
transaction. It copies the batched mel to the selected device on that stream.
Task construction, run creation, prefill, token steps, finalization, the final
model identity check, and cleanup use the same stream. The task and run must use
the patched built-in request-local cache path. Extensions and legacy cache
fallbacks are rejected.

After the submission gate drains, the scope runs cleanup, records one CUDA
event on the stream, and waits for the event. The session cannot commit and the
worker cannot release its lease before that wait succeeds. A cleanup, event
creation, event record, or event synchronization failure quarantines the
transaction. Recovery retries idempotent cleanup and records a new event.
`request_stop()` only sets a host-side latch; it does not call CUDA from the
cancelling thread.

The first stream use performs a conservative device synchronization after
admission. This establishes a boundary with model initialization before the
private stream starts. This path does not support CUDA concurrency, word
alignment, or external mutation of the bound model. The adapter does not derive
its resource vector from measured GPU memory.

Two validated integration records exercise this boundary with the same pinned
`tiny.en` FP32 case on separate AWS and GCP T4 workers. They observe admission,
one private stream, cleanup, the completion event, publication ordering,
cooperative cancellation, retained failure, manual recovery, and native reuse.
See [the native CUDA validation guide](MODAL_NATIVE_CUDA_VALIDATION.md). The
records do not convert the declared memory vector into an enforced limit or a
general CUDA compatibility claim.

```python
cuda_cost = ResourceVector(
    memory_bytes=2_000_000_000,
    compute_units=1,
    stream_slots=1,
)
cuda_worker = Worker(
    "tiny-en-cuda-0",
    cuda_snapshot,
    Budget(cuda_cost),
    queue_capacity=1,
)
cuda_adapter = NativeWhisperAdapter(
    cuda_worker,
    cuda_model,
    identity_probe,
    NativeExecutionProfile(
        "tiny.en/cuda-0-float32",
        cuda_cost,
        device="cuda:0",
    ),
)
```

The memory value above is illustrative. It is not a published profile.

## Experimental two-lane CPU profile

The two-lane profile admits at most two transactions. It serializes task
construction, language detection, and encoder preparation on the model
binding. Once `_start_run()` returns, each admitted transaction can operate its
own decoder run outside that preparation lock.

The adapter enables this path only when the patched backend reports both the
built-in decoder path and request-local KV-cache support. A legacy extension or
hook-based cache fallback is rejected during admitted task construction, before
any encoder or decoder operation. The created run is checked again before
decode starts. The historical adapter remains strictly single-lane.

```python
transaction_cost = ResourceVector(
    memory_bytes=500_000_000,
    compute_units=1,
    stream_slots=1,
)
capacity = ResourceVector(
    memory_bytes=1_000_000_000,
    compute_units=2,
    stream_slots=2,
)
worker = Worker(
    "tiny-en-cpu-dual",
    snapshot,
    Budget(capacity),
    queue_capacity=2,
)
adapter = NativeWhisperAdapter(
    worker,
    model,
    identity_probe,
    NativeExecutionProfile(
        "tiny.en/cpu-dual",
        transaction_cost,
        max_concurrent_decodes=2,
    ),
)
```

Cancellation and cleanup remain transaction-local. One cancelled run does not
stop its peer. Each completion fence cleans its exact run before the worker
releases that lease. A cleanup failure blocks new run creation immediately.
Multiple failed fences retain separate recovery handles; recovering one does
not clear another.

The resource vector is trusted configuration, not a measurement. Two admitted
transactions do not prove that PyTorch kernels overlap or improve throughput.
The repository includes a real-model verifier and one committed record for this
adapter boundary. CI also publishes each result as a temporary artifact. Use
the profile for controlled CPU experiments only.

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
`NativeWhisperAdapter` boundary. It loads one model, records an isolated
baseline, and creates two request-owned decode runs from the same model.

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
experimental two-lane adapter profile has separate controlled unit coverage;
it is not part of this recorded check. The check does not claim parallel kernel
execution, thread-safe execution, or higher throughput.

## Same-model OS-thread isolation check

`tools/verify_native_threaded.py` repeats the decoder isolation case in two
worker threads. It prepares both encoder outputs sequentially and creates a
separate `DecodingTask` for each worker. Each worker creates and operates its
own run. Barriers and events fix the test order. Each thread enters its first
outer `decoder.forward` call. A barrier in the first decoder block holds both
outer calls before either continues. Temporary instrumentation records the
owning Python and native thread identifiers and the start and end of each outer
call. The check requires those two intervals to overlap.

The repository includes one validated record at
`evidence/native-cpu-tiny-en-jfk-threaded-2026-09-04.json`.

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
Overlapping recorded outer decoder-call intervals do not show that PyTorch
kernels execute simultaneously. The check makes no claim about kernel overlap,
throughput, CUDA, production readiness, or general thread safety across other
models, devices, operating systems, or dependency versions. The default
adapter profile remains serialized.

## Runtime adapter concurrency check

`tools/verify_native_runtime_concurrency.py` exercises the experimental
two-lane profile through `NativeWhisperAdapter.decode_window`. It uses two
caller threads, two independent sessions, and explicit deterministic
`tiny.en` options. The check requires:

- two admitted transactions and full reservation of the declared two-lane
  budget;
- non-overlapping `_start_run` intervals, which include encoder preparation;
- overlapping recorded lifetimes for the first outer decoder calls;
- distinct request state and disjoint KV-cache storage;
- cancellation through `RequestState.cancel()` after both first token steps;
- no publication by the cancelled request and lease release after its cleanup;
- no change to the survivor cache while the cancelled run is cleaned;
- one survivor commit equal to the isolated-baseline text;
- an empty queue, restored budget, unchanged model state, restored
  instrumentation, and a successful adapter reuse call.

The cancellation occurs while both first `step()` submissions remain admitted.
The cancelled owner then reaches the adapter's next transaction checkpoint,
where cleanup and lease release occur. The verifier does not call native
cleanup as a substitute for runtime cancellation.

The tool emits JSON. `tools/validate_runtime_concurrency_record.py` validates
it against `evidence/native-runtime-concurrency.schema.json` and enforces event,
ownership, cancellation, queue, lease, budget, session, and result relations.
Native CI publishes the validated record as a 30-day artifact.
The repository also includes one validated Windows CPU record at
`evidence/native-cpu-tiny-en-jfk-runtime-concurrency-2026-09-04.json`.

This is an integration check, not a benchmark. The caller threads are created
by the verifier; the current worker does not schedule them. Encoder preparation
remains serialized. Declared resource accounting is verified, but RAM and
device memory are not measured or enforced. The recorded decoder-call
lifetimes do not prove kernel overlap, throughput improvement, CUDA behavior,
or production readiness.
