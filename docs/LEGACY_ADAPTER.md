# Historical Whisper adapter

`LegacyWhisperAdapter` places an existing synchronous
`model.transcribe(audio, **options)` call behind the runtime transaction
boundary. It is a migration path. It is not the staged backend defined in RFC
0001.

## Contract

The adapter requires:

- one loaded model object;
- one `Worker` with `queue_capacity=1`;
- an execution profile that reserves the worker's complete capacity;
- a probe that derives the declared `ModelSnapshot` from the loaded weights;
- an explicit input identity;
- an `ExecutionScope` whose fence covers all backend work.

One model object cannot be bound to a second worker. Separate adapters for the
same object share one binding. A retained transaction closes that binding until
exact recovery succeeds.

## Construction

```python
from whisper_runtime import Budget, ModelSnapshot, ResourceVector, Worker
from whisper_runtime.adapters import LegacyExecutionProfile, LegacyWhisperAdapter

capacity = ResourceVector(
    memory_bytes=2_000_000_000,
    compute_units=1,
    stream_slots=1,
)
snapshot = ModelSnapshot(
    model_id="tiny.en",
    revision="checkpoint revision",
    backend="pytorch-cpu",
    fingerprint="sha256:<loaded-weight digest>",
)
worker = Worker(
    "cpu-0",
    snapshot,
    Budget(capacity),
    queue_capacity=1,
)
profile = LegacyExecutionProfile("tiny.en/cpu", capacity)

# `probe_loaded_weights` belongs to the application. It must derive the same
# ModelSnapshot from the live model object; a model name alone is not enough.
adapter = LegacyWhisperAdapter(
    worker,
    model,
    probe_loaded_weights,
    profile,
)
```

Each call supplies a `Session`, a `RequestState`, an input identity, and an
execution scope. Use `ImmediateFence` only when the backend call is fully
synchronous. The returned envelope contains a detached legacy result mapping,
the committed session version, canonical option and payload digests, the model
identity, the execution profile, input provenance, and host-side timings.

If a call raises `LegacyTranscriptionRetainedError`, do not repeat inference.
Inspect `error.envelope` to determine whether publication occurred, retain the
error object, and call `error.recover()`. The model binding reopens only after
the exact transaction reaches terminal cleanup.

## Limits

The adapter treats one complete historical transcription as one transaction.
It cannot pause between decoder tokens, reuse encoded windows across fallback
attempts, form safe decoder batches, or account for individual stages. It also
cannot protect a model or audio object from unrelated code that mutates it in
the same process. The native staged backend must provide those properties.
