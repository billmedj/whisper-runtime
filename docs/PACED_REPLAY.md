# Paced PCM replay

`drive_paced` supplies recorded PCM at its original sample rate while the existing
continuous controller performs model work independently. It needs no microphone,
sound driver, network service or additional package. This is a replay driver,
not a new recognition or publication policy.

## Use

Construct `ContinuousTranscriptStream` with the chosen native adapter and
publication profile. Call the driver on the thread that created the stream.
Input must be nonempty mono 16 kHz signed 16-bit little-endian PCM.

```python
from whisper_runtime.adapters import PacedReplayConfig, drive_paced

# stream and pcm are prepared by the caller. Model setup precedes the replay.
with stream:
    report = drive_paced(
        stream,
        pcm,
        config=PacedReplayConfig(chunk_ms=20),
        on_event=lambda event, elapsed_ns: consume(event, elapsed_ns),
    )
    if report.error is not None:
        raise report.error
    if report.status != "completed":
        raise RuntimeError(f"Replay stopped: {report.status}")
```

This example leaves retained-transaction exceptions with the existing stream
context manager and caller. The driver itself never closes the stream. Inspect
the report, retained PCM and recovery authority before deciding how to recover
or abandon failed work. Do not repeat inference to recover a missing delivery.

## Source clock

A producer thread offers each chunk at its exact end sample divided by 16,000.
For example, samples 0 through 319 become available after 20 ms, not at time
zero. The last partial chunk uses its actual sample count. Deadlines are
absolute offsets from one monotonic origin; decoding time cannot move them.

The source does not wait for decoding or output callbacks. A full stream buffer
rejects the complete offered chunk, without consuming its sequence number. The
driver reports `source_overload`; it does not retry the chunk later and label
the replay live. Scheduling jitter within the declared tolerance is recorded.
Excessive offer lateness produces `source_lag`. Excessive admission-completion
lateness produces `admission_lag`, including on the last chunk.

Only successful admission of the full input permits a call to `finish_input`.
No failure or cancellation manufactures EOF. The creating thread drives
`step()` and invokes output/trace callbacks. The producer is stopped and joined
before the driver returns or propagates an owner-thread interruption.

## Bounds and failure handling

Defaults are 20 ms chunks, at most 120 seconds of input, 250 ms of offer or
admission lateness, and 15 seconds for post-input draining. These are replay
limits, not guarantees for real microphones or end-user subtitle latency.
There is also a fixed step limit. The controller retains its own PCM bound.

Admission records are bounded by the declared input and chunk lengths. The
driver does not retain a transcript or trace history; optional callbacks can
persist those records. Callback storage and the input bytes supplied by the
caller are outside the controller's PCM-buffer measurement.

Slow consumers can increase the backlog or cause an explicit failure. A callback
exception returns the original exception and a bounded `undelivered_events`
remainder. It begins with the event whose callback raised. That event may have
reached the consumer before the exception. Reconcile delivery by sequence number;
callbacks and model work are not retried automatically.

Native recovery exceptions are preserved as the original objects. A failed trace
callback cannot replace a native execution error. A successful result requires
full admission and commitment, a completed producer, an input-finish timestamp,
and exactly one final event. Closing the stream from a callback cannot turn
partial processing into success.

Cancellation and drain deadlines are cooperative. They do not forcibly stop a
blocking native step or callback. A remote execution timeout is a separate outer
limit. After a failed replay, the caller still owns cleanup and any live recovery
handle; returning a failure does not imply that GPU resources were released.

## Measurements

Each `PacedAdmission` records source bounds and three relative timestamps:

- `scheduled_ns`: the time at which the chunk becomes available at source speed.
- `offered_ns`: the time immediately before `push`.
- `accepted_ns`: the time immediately after successful `push` returns.

The report separates maximum offer lateness from maximum admission lateness.
Its buffer snapshots are observations, not an atomic association with the
producer's push: the model owner may commit before the snapshot is read.
Use the controller's high-water mark for peak buffered PCM.

Output callbacks receive the monotonic time at which their event batch was
observed. Events in one batch share that timestamp. It does not measure actual
display time or a remote consumer's acknowledgement. Trace callback timestamps
can follow output handling. `input_finished_ns` marks the producer's call to
`finish_input`; the producer makes no such call on a partial-source failure.

Subtracting a publication's source endpoint from its observation time measures
source-end-to-output lag. For quiet-run units, the endpoint already includes the
quiet detection interval. This lag is not word latency and does not include the
entire wait from the end of speech. Model loading, warmup, network transport and
audio hardware need separate measurements.

## Current scope

The local suite uses the existing scripted native adapter with real transaction
and fence fixtures. It tests timing, concurrency, overload, cancellation,
callback failure and recovery. These tests do not establish ASR quality.

The Modal corpus harness has a separate `--paced-replay` variant. It runs the
same input-evidence and automatic-endpoint policy inside one T4 worker. It does
not send microphone data from the PC. The separate
[WebSocket replay](NETWORK_REPLAY.md) tests transport, disconnection and
authentication without changing the recognition controller. Its measurements
must remain distinct from this same-worker replay.
