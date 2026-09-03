"""Commit one synchronous result through the runtime boundary."""

from whisper_runtime import (
    Budget,
    ImmediateFence,
    ModelSnapshot,
    RequestState,
    ResourceVector,
    Session,
    WindowResult,
    Worker,
)

model = ModelSnapshot(
    model_id="example",
    revision="example-revision",
    backend="synchronous-example",
    fingerprint="sha256:example",
)
capacity = ResourceVector(
    memory_bytes=1_000,
    compute_units=1,
    stream_slots=1,
)
budget = Budget(capacity)
worker = Worker("worker-1", model, budget, queue_capacity=1)
session = Session("session-1")
request = RequestState(
    request_id="request-1",
    session_id=session.session_id,
    model=model,
    rng_seed=7,
)

transaction = worker.prepare(
    session=session,
    request=request,
    window_id="window-1",
    resources=capacity,
)

# ImmediateFence is valid only when backend work is complete before commit.
transaction.start(ImmediateFence())
state = transaction.commit(
    WindowResult(
        window_id="window-1",
        text="Example transcript",
        start_ms=0,
        end_ms=1_000,
    )
)

assert state.version == 1
assert worker.queue_depth == 0
assert budget.available == capacity
print(state.windows[-1].result.text)
