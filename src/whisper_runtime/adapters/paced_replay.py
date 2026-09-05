"""Paced file input for the existing continuous controller; no audio device.

One producer admits PCM on absolute monotonic deadlines while the creating
thread drives decoding. Output callbacks run on that owner. The caller retains
stream cleanup and recovery authority, including after a failed replay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Callable, Literal

from .continuous_stream import ContinuousDecodeTrace, ContinuousTranscriptStream
from .native_stream import AudioBufferFullError, StreamEventKind, TranscriptEvent

ReplayStatus = Literal[
    "completed",
    "source_overload",
    "source_lag",
    "admission_lag",
    "cancelled",
    "drain_timeout",
    "runtime_error",
    "step_limit",
]


@dataclass(frozen=True, slots=True)
class PacedReplayConfig:
    chunk_ms: int = 20
    max_input_ms: int = 120_000
    max_source_lag_ms: int = 250
    max_drain_ms: int = 15_000
    idle_wait_ms: int = 2
    max_steps: int = 100_000

    def __post_init__(self) -> None:
        bounds = {
            "chunk_ms": (1, 1_000),
            "max_input_ms": (1, 3_600_000),
            "max_source_lag_ms": (0, 60_000),
            "max_drain_ms": (1, 120_000),
            "idle_wait_ms": (1, 1_000),
            "max_steps": (1, 10_000_000),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")


@dataclass(frozen=True, slots=True)
class PacedAdmission:
    """Relative monotonic times; acceptance completes when push returns."""

    sequence_number: int
    start_sample: int
    end_sample: int
    scheduled_ns: int
    offered_ns: int
    accepted_ns: int
    buffered_samples: int


@dataclass(frozen=True, slots=True)
class PacedReplayResult:
    status: ReplayStatus
    elapsed_ns: int
    input_finished_ns: int | None
    offered_samples: int
    accepted_samples: int
    driver_steps: int
    max_source_lag_ns: int
    max_admission_lag_ns: int
    admissions: tuple[PacedAdmission, ...]
    error: BaseException | None = field(default=None, repr=False)
    undelivered_events: tuple[TranscriptEvent, ...] = ()
    undelivered_event_ns: int | None = None


def drive_paced(
    stream: ContinuousTranscriptStream,
    pcm: bytes,
    *,
    config: PacedReplayConfig | None = None,
    on_event: Callable[[TranscriptEvent, int], None] | None = None,
    on_trace: Callable[[ContinuousDecodeTrace, int], None] | None = None,
    cancel: Event | None = None,
) -> PacedReplayResult:
    """Replay bounded PCM without slowing its clock to match the decoder.

    Requires a pristine stream on its owner thread. Every chunk becomes available
    at its end sample / 16000, including a partial last chunk. Admission overflow
    or offer/admission lateness beyond the configured tolerance stops the source; there
    is no delayed retry or synthetic EOF. Small scheduling jitter is measured.

    Timing records are bounded by input length / chunk length. Events and traces
    are sent to caller callbacks, not retained here. Slow callbacks do not stop
    the producer clock; their effects on backlog are part of the replay.

    The producer is stopped and joined before return or exception propagation.
    This function never closes the stream. On failure, its accepted PCM and any
    retained transaction remain available to the owner. ``error`` preserves the
    original exception object. Deadlines and cancellation are cooperative: a
    blocking native step or callback cannot be forcibly interrupted here.
    ``input_finished_ns`` marks the call to finish_input, not a capture-device
    timestamp. All recorded times use the same monotonic origin after setup.
    On callback failure, ``undelivered_events`` starts with the event whose
    callback failed. That event may already have reached its consumer. Use event
    sequence numbers to reconcile delivery; never repeat inference to resend it.
    """
    if not isinstance(stream, ContinuousTranscriptStream):
        raise TypeError("stream must be ContinuousTranscriptStream")
    if config is not None and not isinstance(config, PacedReplayConfig):
        raise TypeError("config must be PacedReplayConfig or None")
    config = config or PacedReplayConfig()
    if not isinstance(pcm, bytes):
        raise TypeError("pcm must be bytes")
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM must contain complete, nonempty 16-bit samples")
    if len(pcm) // 2 > config.max_input_ms * 16:
        raise ValueError("PCM exceeds the replay input limit")
    if cancel is not None and not isinstance(cancel, Event):
        raise TypeError("cancel must be a threading.Event or None")
    for callback in (on_event, on_trace):
        if callback is not None and not callable(callback):
            raise TypeError("replay callbacks must be callable or None")
    # This public accessor also verifies the creating-thread requirement.
    if (
        stream.last_trace is not None
        or stream.accepted_samples
        or stream.expected_chunk
        or stream.active
        or stream.input_finished
        or stream.done
    ):
        raise ValueError("paced replay requires a pristine stream")

    stop, changed, producer_done = Event(), Event(), Event()
    admissions: list[PacedAdmission] = []
    producer_status: ReplayStatus | None = None
    producer_error: BaseException | None = None
    input_finished_ns: int | None = None
    offered_samples = max_lag = max_admission_lag = 0
    started = time.monotonic_ns()

    def elapsed() -> int:
        return time.monotonic_ns() - started

    def produce() -> None:
        nonlocal producer_status, producer_error, input_finished_ns
        nonlocal offered_samples, max_lag, max_admission_lag
        try:
            chunk_bytes = config.chunk_ms * 32
            for sequence, offset in enumerate(range(0, len(pcm), chunk_bytes)):
                end_byte = min(offset + chunk_bytes, len(pcm))
                end_sample = end_byte // 2
                scheduled = end_sample * 62_500
                while not stop.is_set():
                    if cancel is not None and cancel.is_set():
                        producer_status = "cancelled"
                        return
                    remaining = scheduled - elapsed()
                    if remaining <= 0:
                        break
                    stop.wait(min(remaining / 1_000_000_000, 0.05))
                if stop.is_set():
                    return
                offered = elapsed()
                offered_samples = end_sample
                max_lag = max(max_lag, offered - scheduled)
                if offered - scheduled > config.max_source_lag_ms * 1_000_000:
                    producer_status = "source_lag"
                    return
                try:
                    stream.push(sequence, pcm[offset:end_byte])
                except AudioBufferFullError as error:
                    producer_status, producer_error = "source_overload", error
                    return
                accepted = elapsed()
                max_admission_lag = max(max_admission_lag, accepted - scheduled)
                admissions.append(
                    PacedAdmission(
                        sequence,
                        offset // 2,
                        end_sample,
                        scheduled,
                        offered,
                        accepted,
                        stream.metrics.buffered_samples,
                    )
                )
                changed.set()
                if accepted - scheduled > config.max_source_lag_ms * 1_000_000:
                    producer_status = "admission_lag"
                    return
            input_finished_ns = elapsed()
            stream.finish_input()
        except BaseException as error:
            producer_status, producer_error = "runtime_error", error
        finally:
            producer_done.set()
            changed.set()

    status: ReplayStatus = "completed"
    error: BaseException | None = None
    undelivered: tuple[TranscriptEvent, ...] = ()
    undelivered_ns: int | None = None
    steps = trace_index = final_events = 0

    def capture_trace() -> None:
        nonlocal trace_index
        trace = stream.last_trace
        if trace is not None and trace.decode_index != trace_index:
            trace_index = trace.decode_index
            if on_trace is not None:
                on_trace(trace, elapsed())

    producer = Thread(target=produce, name="whisper-paced-input")
    producer.start()
    try:
        while True:
            if cancel is not None and cancel.is_set():
                status = "cancelled"
                break
            if producer_done.is_set():
                if producer_status is not None:
                    status, error = producer_status, producer_error
                    break
                assert input_finished_ns is not None
                if elapsed() - input_finished_ns > config.max_drain_ms * 1_000_000:
                    status = "drain_timeout"
                    break
            if stream.done:
                break
            if not stream.ready:
                changed.wait(config.idle_wait_ms / 1_000)
                changed.clear()
                continue
            if steps >= config.max_steps:
                status = "step_limit"
                break
            steps += 1
            try:
                batch = stream.step()
            except Exception:
                # A trace callback must not replace native recovery authority.
                try:
                    capture_trace()
                except Exception:
                    pass
                raise
            observed = elapsed()
            final_events += sum(event.kind is StreamEventKind.FINAL for event in batch)
            if on_event is not None:
                for index, event in enumerate(batch):
                    try:
                        on_event(event, observed)
                    except Exception:
                        undelivered, undelivered_ns = batch[index:], observed
                        raise
            capture_trace()
    except Exception as failure:
        status, error = "runtime_error", failure
    finally:
        stop.set()
        producer.join()
    if status == "completed":
        if producer_status is not None:
            status, error = producer_status, producer_error
        elif not (
            input_finished_ns is not None
            and stream.accepted_samples == len(pcm) // 2
            and stream.metrics.committed_samples == len(pcm) // 2
            and stream.done
            and final_events == 1
        ):
            status = "runtime_error"
            error = RuntimeError(
                "paced replay ended before complete source publication"
            )
    return PacedReplayResult(
        status=status,
        elapsed_ns=elapsed(),
        input_finished_ns=input_finished_ns,
        offered_samples=offered_samples,
        accepted_samples=stream.accepted_samples,
        driver_steps=steps,
        max_source_lag_ns=max_lag,
        max_admission_lag_ns=max_admission_lag,
        admissions=tuple(admissions),
        error=error,
        undelivered_events=undelivered,
        undelivered_event_ns=undelivered_ns,
    )
