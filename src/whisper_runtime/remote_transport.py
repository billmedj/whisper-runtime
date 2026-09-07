"""Bounded duplex PCM clients for replay-v1 and live-v2 WebSockets.

This is paced file input, not microphone capture. The source clock starts only
after READY. Each 20 ms frame becomes eligible at its *end* sample deadline;
network backpressure never moves that clock. A late/blocked send stops input
without EOF or retry. Server and client monotonic clocks are never subtracted.

``stream_live`` instead consumes an unknown-length asynchronous PCM source after
READY; its producer supplies capture/file pacing. Both paths use the same
bounded duplex transport, exact event validation and terminal coverage checks.

``replay`` needs aiohttp only when called. ``_replay_socket`` is also usable with
a small fake socket (async send_json/send_bytes/receive_json) in offline tests.
Every failure returns a JSON-safe partial record with a fixed error code. The
optional callback must be an async function accepting (event, client_elapsed_ns)
and must not perform blocking work; it has an independent bounded deadline.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import struct
import time
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, cast
from urllib.parse import urlsplit

from .pcm_live import (
    LIVE_PROTOCOL,
    MAX_LIVE_EVENTS,
    MAX_LIVE_SAMPLES,
    MAX_LIVE_SECONDS,
)

PROTOCOL = "pcm-websocket/v1"
SAMPLE_RATE_HZ = 16_000
CHUNK_SAMPLES = 320
MAX_SAMPLES = 1_920_000
MAX_MESSAGE_BYTES = 1_048_576
FRAME_HEADER = struct.Struct("!IQ")
_PRERECORDED = object()
_EVENT_FIELDS = frozenset(
    "sequence_number kind segment_id revision start_sample end_sample "
    "sample_rate_hz text supersedes_revision committed_through_sample "
    "committed_through_ms session_version".split()
)
_ERROR_CODES = frozenset(
    "invalid_pcm input_limit callback_must_be_async receive_failed invalid_message "
    "message_limit invalid_event event_sequence invalid_final "
    "incomplete_commit_coverage invalid_commit commit_coverage invalid_done "
    "server_failed incomplete_input accepted_hash_mismatch invalid_admissions "
    "source_late premature_done missing_final unexpected_message event_after_final "
    "event_limit invalid_event_envelope premature_final callback_failed invalid_ready "
    "incomplete_sender invalid_url invalid_headers missing_aiohttp unexpected_frame "
    "redirect_forbidden total_timeout send_timeout eof_send_timeout "
    "callback_timeout start_send_timeout ready_timeout drain_timeout "
    "event_revision commit_revision revision_after_commit event_echo_mismatch "
    "unsupported_transport invalid_source source_timeout source_failed invalid_live_frame".split()
)


@dataclass(frozen=True)
class ReplayConfig:
    """Timeouts may be lowered for tests, never raised beyond safety caps."""

    ready_timeout_s: float = 90.0
    send_timeout_s: float = 0.25
    drain_timeout_s: float = 30.0
    total_timeout_s: float = 160.0
    max_source_lateness_s: float = 0.25
    callback_timeout_s: float = 0.25
    max_server_events: int = 1_000
    MAX_TOTAL_TIMEOUT = 160
    MAX_SERVER_EVENTS = 1_000

    def __post_init__(self) -> None:
        for name, cap in (
            ("ready_timeout_s", 90),
            ("send_timeout_s", 0.25),
            ("drain_timeout_s", 30),
            ("total_timeout_s", self.MAX_TOTAL_TIMEOUT),
            ("max_source_lateness_s", 0.25),
            ("callback_timeout_s", 0.25),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= cap
            ):
                raise ValueError(f"invalid replay limit: {name}")
        if type(self.max_server_events) is not int or not (
            1 <= self.max_server_events <= self.MAX_SERVER_EVENTS
        ):
            raise ValueError("invalid replay limit: max_server_events")


@dataclass(frozen=True)
class LiveConfig(ReplayConfig):
    """Unknown-length source, with finite session/sample/event/idle limits."""

    total_timeout_s: float = MAX_LIVE_SECONDS
    max_server_events: int = MAX_LIVE_EVENTS
    max_samples: int = MAX_LIVE_SAMPLES
    source_timeout_s: float = 10.0
    MAX_TOTAL_TIMEOUT = MAX_LIVE_SECONDS
    MAX_SERVER_EVENTS = MAX_LIVE_EVENTS

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.max_samples) is not int
            or not 1 <= self.max_samples <= MAX_LIVE_SAMPLES
        ):
            raise ValueError("invalid live limit: max_samples")
        if (
            isinstance(self.source_timeout_s, bool)
            or not isinstance(self.source_timeout_s, (float, int))
            or not math.isfinite(self.source_timeout_s)
            or not 0 < self.source_timeout_s <= 10
        ):
            raise ValueError("invalid live limit: source_timeout_s")


class ReplayError(Exception):
    """An internal fixed-code failure; never includes transport exception text."""

    def __init__(self, code: str) -> None:
        super().__init__(code if code in _ERROR_CODES else "operation_failed")


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _new_record(pcm: bytes) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "failed",
        "sample_count": len(pcm) // 2,
        "sha256": (
            hashlib.sha256(pcm).hexdigest()
            if 0 < len(pcm) <= MAX_SAMPLES * 2 and len(pcm) % 2 == 0
            else None
        ),
        "send_records": [],
        "events": [],
        "server_done": None,
        "client_metrics": {
            "sent_samples": 0,
            "sent_chunks": 0,
            "sender_finished": False,
            "eof_sent": False,
            "input_finished_ns": None,
            "eof_offered_ns": None,
            "done_received_ns": None,
            "time_to_first_text_ns": None,
            "time_to_first_commit_ns": None,
            "eof_to_done_ns": None,
            "max_source_lateness_ns": 0,
            "max_send_wait_ns": 0,
            "elapsed_ns": None,
        },
    }


def _check_input(pcm: bytes, on_event: Any) -> None:
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise ReplayError("invalid_pcm")
    if len(pcm) // 2 > MAX_SAMPLES:
        raise ReplayError("input_limit")
    if on_event is not None and not inspect.iscoroutinefunction(on_event):
        raise ReplayError("callback_must_be_async")


async def _bounded(awaitable: Awaitable[Any], seconds: float, code: str) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError:
        raise ReplayError(code) from None


async def _receive(socket: Any) -> dict[str, Any]:
    try:
        # The real transport rejects non-text messages and oversize payloads
        # before this call returns. Check fake sockets and JSON complexity too.
        message = await socket.receive_json()
    except asyncio.CancelledError:
        raise
    except Exception:
        raise ReplayError("receive_failed") from None
    try:
        size = len(
            json.dumps(message, ensure_ascii=True, allow_nan=False).encode("utf-8")
        )
    except (TypeError, ValueError, RecursionError):
        raise ReplayError("invalid_message") from None
    if size > MAX_MESSAGE_BYTES:
        raise ReplayError("message_limit")
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ReplayError("invalid_message")
    return message


def _check_event(
    event: Any,
    *,
    previous_sequence: int,
    committed: int,
    offered_samples: int,
    input_samples: int,
) -> int:
    if not isinstance(event, dict) or set(event) != _EVENT_FIELDS:
        raise ReplayError("invalid_event")
    sequence = event["sequence_number"]
    if not _integer(sequence, 1) or sequence != previous_sequence + 1:
        raise ReplayError("event_sequence")
    kind = event["kind"]
    if kind not in ("provisional", "replace", "commit", "final"):
        raise ReplayError("invalid_event")
    version = event["session_version"]
    if kind == "final":
        if not _integer(version) or any(
            event[field] is not None
            for field in _EVENT_FIELDS - {"sequence_number", "kind", "session_version"}
        ):
            raise ReplayError("invalid_final")
        if committed != input_samples:
            raise ReplayError("incomplete_commit_coverage")
        return committed
    if (
        not isinstance(event["segment_id"], str)
        or not event["segment_id"].strip()
        or not _integer(event["revision"], 1)
        or not _integer(event["start_sample"])
        or not _integer(event["end_sample"])
        or not event["start_sample"] <= event["end_sample"] <= offered_samples
        or type(event["sample_rate_hz"]) is not int
        or event["sample_rate_hz"] != SAMPLE_RATE_HZ
        or (version is not None and not _integer(version))
    ):
        raise ReplayError("invalid_event")
    if kind == "commit":
        if (
            event["text"] is not None
            or event["supersedes_revision"] is not None
            or not _integer(version, 1)
            or not _integer(event["committed_through_sample"])
            or not _integer(event["committed_through_ms"])
        ):
            raise ReplayError("invalid_commit")
        end = event["end_sample"]
        if (
            event["start_sample"] != committed
            or end <= committed
            or event["committed_through_sample"] != end
            or event["committed_through_ms"] != end // 16
        ):
            raise ReplayError("commit_coverage")
        return cast(int, end)
    if (
        not isinstance(event["text"], str)
        or event["committed_through_sample"] is not None
        or event["committed_through_ms"] is not None
    ):
        raise ReplayError("invalid_event")
    if kind == "replace":
        if (
            not _integer(event["supersedes_revision"], 1)
            or event["supersedes_revision"] != event["revision"] - 1
        ):
            raise ReplayError("invalid_event")
    elif event["supersedes_revision"] is not None:
        raise ReplayError("invalid_event")
    return committed


def _check_done(message: dict[str, Any], record: dict[str, Any]) -> None:
    if message.get("status") not in ("completed", "failed", "cancelled"):
        raise ReplayError("invalid_done")
    # Do not preserve server exception strings, including error_code, on errors.
    if message["status"] != "completed":
        raise ReplayError("server_failed")
    metrics = message.get("metrics")
    if not isinstance(metrics, dict):
        raise ReplayError("invalid_done")
    for key in (
        "accepted_samples",
        "committed_samples",
        "buffered_samples",
        "trace_count",
    ):
        if not _integer(metrics.get(key)):
            raise ReplayError("invalid_done")
    if (
        metrics["accepted_samples"] != record["sample_count"]
        or metrics["committed_samples"] != record["sample_count"]
        or metrics["buffered_samples"] != 0
    ):
        raise ReplayError("incomplete_input")
    if metrics.get("accepted_sha256") != record["sha256"]:
        raise ReplayError("accepted_hash_mismatch")
    if record["protocol"] == LIVE_PROTOCOL:
        if (
            message.get("protocol") != LIVE_PROTOCOL
            or message.get("source_eof_received") is not True
            or not _integer(metrics.get("accepted_chunks"))
            or metrics["accepted_chunks"] != record["client_metrics"]["sent_chunks"]
            or not _integer(metrics.get("event_count"))
            or metrics["event_count"] != record["client_metrics"]["received_events"]
        ):
            raise ReplayError("invalid_done")
        return
    admissions = message.get("admissions")
    if not isinstance(admissions, list) or len(admissions) != len(
        record["send_records"]
    ):
        raise ReplayError("invalid_admissions")
    previous_accepted_ns = 0
    for admission, sent in zip(admissions, record["send_records"]):
        if (
            not isinstance(admission, dict)
            or any(
                not _integer(admission.get(key))
                for key in (
                    "sequence_number",
                    "start_sample",
                    "end_sample",
                    "received_ns",
                    "accepted_ns",
                )
            )
            or any(
                admission[key] != sent[key]
                for key in ("sequence_number", "start_sample", "end_sample")
            )
            or not previous_accepted_ns
            <= admission["received_ns"]
            <= admission["accepted_ns"]
        ):
            raise ReplayError("invalid_admissions")
        previous_accepted_ns = admission["accepted_ns"]
    if "events" in message and message["events"] != [
        {
            "type": "event",
            "event": entry["event"],
            "server_elapsed_ns": entry["server_elapsed_ns"],
        }
        for entry in record["events"]
    ]:
        raise ReplayError("event_echo_mismatch")


async def _replay_socket(
    socket: Any,
    pcm: bytes,
    *,
    config: ReplayConfig | None = None,
    on_event: Callable[[dict[str, Any], int], Awaitable[None]] | None = None,
    _live_source: Any = _PRERECORDED,
) -> dict[str, Any]:
    """Replay on one socket; always join sender/receiver before returning.

    Socket ownership remains with the caller. Cancellation is reported in the
    partial result after a best-effort bounded CANCEL; no reconnect is attempted.
    Timing values other than server_elapsed_ns share the post-READY client origin.
    """
    config = config or ReplayConfig()
    live = _live_source is not _PRERECORDED
    protocol = LIVE_PROTOCOL if live else PROTOCOL
    record = _new_record(pcm if isinstance(pcm, bytes) else b"")
    if live:
        record.update(protocol=protocol, diagnostic_history="not_retained")
        record["client_metrics"]["received_events"] = 0
        record["client_metrics"]["timing_scope"] = (
            "post-ready transport; capture timestamps unavailable"
        )
    metrics = record["client_metrics"]
    origin: int | None = None
    producer: asyncio.Task[None] | None = None
    consumer: asyncio.Task[None] | None = None
    source_finished = asyncio.Event()
    offered_samples = 0
    eof_offered_ns: int | None = None
    live_sample_limit = (
        getattr(config, "max_samples", MAX_LIVE_SAMPLES) if live else MAX_SAMPLES
    )

    def elapsed() -> int:
        assert origin is not None
        return time.monotonic_ns() - origin

    async def frames() -> AsyncGenerator[bytes, None]:
        if live:
            iterator = aiter(_live_source)
            try:
                while True:
                    try:
                        chunk = await _bounded(
                            anext(iterator),
                            cast(LiveConfig, config).source_timeout_s,
                            "source_timeout",
                        )
                    except StopAsyncIteration:
                        return
                    except (ReplayError, asyncio.CancelledError):
                        raise
                    except Exception:
                        raise ReplayError("source_failed") from None
                    yield chunk
            finally:
                # The iterator is owned for this session; async generators can
                # release capture resources on EOF, cancellation and failure.
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await _bounded(close(), config.send_timeout_s, "source_timeout")
        else:
            for offset in range(0, len(pcm), CHUNK_SAMPLES * 2):
                yield pcm[offset : offset + CHUNK_SAMPLES * 2]

    async def send_source() -> None:
        nonlocal offered_samples, eof_offered_ns
        digest, offset, sequence, partial = hashlib.sha256(), 0, 0, False
        source_frames = frames()
        try:
            async for chunk in source_frames:
                if (
                    not isinstance(chunk, bytes)
                    or not chunk
                    or len(chunk) % 2
                    or len(chunk) > CHUNK_SAMPLES * 2
                    or partial
                ):
                    raise ReplayError("invalid_live_frame" if live else "invalid_pcm")
                end_sample = offset // 2 + len(chunk) // 2
                if live and end_sample > live_sample_limit:
                    raise ReplayError("input_limit")
                partial = len(chunk) < CHUNK_SAMPLES * 2
                await send_frame(chunk, sequence, offset, end_sample, digest)
                sequence += 1
                offset += len(chunk)
        finally:
            await source_frames.aclose()
        if live:
            if not offset:
                raise ReplayError("invalid_pcm")
            record["sha256"] = digest.hexdigest()
        await finish_source()

    async def send_frame(
        chunk: bytes, sequence: int, offset: int, end_sample: int, digest: Any
    ) -> None:
        nonlocal offered_samples
        scheduled_ns = end_sample * 62_500
        # Only prerecorded input owns a synthetic source clock. Live capture
        # supplies its own pacing and is not delayed or silently dropped here.
        while not live and (remaining := scheduled_ns - elapsed()) > 0:
            await asyncio.sleep(remaining / 1_000_000_000)
        if consumer is not None and consumer.done():
            consumer.result()
            raise ReplayError("premature_done")
        offered_ns = elapsed()
        lateness = offered_ns - scheduled_ns
        if not live:
            metrics["max_source_lateness_ns"] = max(
                metrics["max_source_lateness_ns"], lateness
            )
        if not live and lateness > config.max_source_lateness_s * 1_000_000_000:
            raise ReplayError("source_late")
        entry = {
            "sequence_number": sequence,
            "start_sample": offset // 2,
            "end_sample": end_sample,
            "scheduled_ns": scheduled_ns,
            "offered_ns": offered_ns,
            "completed_ns": None,
        }
        if not live:
            record["send_records"].append(entry)
        offered_samples = end_sample
        frame = FRAME_HEADER.pack(sequence, offset // 2) + chunk
        await _bounded(socket.send_bytes(frame), config.send_timeout_s, "send_timeout")
        completed_ns = elapsed()
        entry["completed_ns"] = completed_ns
        metrics["sent_samples"] = end_sample
        metrics["sent_chunks"] += 1
        if live:
            digest.update(chunk)
            record["sample_count"] = end_sample
            record["sha256"] = digest.hexdigest()
        metrics["max_send_wait_ns"] = max(
            metrics["max_send_wait_ns"], completed_ns - offered_ns
        )
        if (
            not live
            and completed_ns - scheduled_ns
            > config.max_source_lateness_s * 1_000_000_000
        ):
            raise ReplayError("source_late")

    async def finish_source() -> None:
        nonlocal eof_offered_ns
        if consumer is not None and consumer.done():
            consumer.result()
            raise ReplayError("premature_done")
        eof_offered_ns = elapsed()
        metrics["eof_offered_ns"] = eof_offered_ns
        await _bounded(
            socket.send_json(
                {
                    "type": "eof",
                    "chunks": metrics["sent_chunks"],
                    "samples": metrics["sent_samples"],
                    "sha256": record["sha256"],
                }
            ),
            config.send_timeout_s,
            "eof_send_timeout",
        )
        metrics["input_finished_ns"] = elapsed()
        metrics["eof_sent"] = True
        metrics["sender_finished"] = True
        source_finished.set()

    async def receive_events() -> None:
        sequence, committed, final_count = 0, 0, 0
        revisions: dict[str, tuple[int, int, int]] = {}
        committed_segments: set[str] = set()
        while True:
            message = await _receive(socket)
            received_ns = elapsed()
            if message["type"] == "done":
                if message.get("status") in ("failed", "cancelled"):
                    # Keep useful partial counters without trusting arbitrary
                    # server error strings or reflecting its exception details.
                    raw_metrics = message.get("metrics", {})
                    safe_metrics = (
                        {
                            key: raw_metrics[key]
                            for key in (
                                "accepted_samples",
                                "committed_samples",
                                "buffered_samples",
                                "trace_count",
                            )
                            if _integer(raw_metrics.get(key))
                        }
                        if isinstance(raw_metrics, dict)
                        else {}
                    )
                    record["server_done"] = {
                        "type": "done",
                        "status": message["status"],
                        "metrics": safe_metrics,
                    }
                    raise ReplayError("server_failed")
                if eof_offered_ns is None:
                    raise ReplayError("premature_done")
                # A response may reach receive() while the final send await is
                # finishing. Require successful EOF completion, not task order.
                await _bounded(
                    source_finished.wait(), config.send_timeout_s, "incomplete_sender"
                )
                _check_done(message, record)
                if final_count != 1:
                    raise ReplayError("missing_final")
                metrics["done_received_ns"] = received_ns
                # Use EOF offer for this local interval. It precedes server done
                # even if the EOF send completion resumes after receive().
                metrics["eof_to_done_ns"] = received_ns - eof_offered_ns
                record["server_done"] = copy.deepcopy(message)
                return
            if message["type"] != "event":
                raise ReplayError("unexpected_message")
            if final_count:
                raise ReplayError("event_after_final")
            if sequence >= config.max_server_events:
                raise ReplayError("event_limit")
            if set(message) != {"type", "event", "server_elapsed_ns"} or not _integer(
                message.get("server_elapsed_ns")
            ):
                raise ReplayError("invalid_event_envelope")
            event = message.get("event")
            committed = _check_event(
                event,
                previous_sequence=sequence,
                committed=committed,
                offered_samples=offered_samples,
                input_samples=offered_samples if live else record["sample_count"],
            )
            event = cast(dict[str, Any], event)
            sequence = event["sequence_number"]
            kind = event["kind"]
            if kind != "final":
                segment = event["segment_id"]
                revision = (
                    event["revision"],
                    event["start_sample"],
                    event["end_sample"],
                )
                if segment in committed_segments:
                    raise ReplayError("revision_after_commit")
                previous = revisions.get(segment)
                if kind == "commit":
                    if previous != revision:
                        raise ReplayError("commit_revision")
                    committed_segments.add(segment)
                else:
                    if kind == "provisional":
                        if previous is not None or revision[0] != 1:
                            raise ReplayError("event_revision")
                    elif (
                        previous is None
                        or revision[0] != previous[0] + 1
                        or revision[1] != previous[1]
                    ):
                        raise ReplayError("event_revision")
                    revisions[segment] = revision
            if kind == "final":
                if eof_offered_ns is None:
                    raise ReplayError("premature_final")
                final_count += 1
            if (
                kind in ("provisional", "replace")
                and event["text"].strip()
                and metrics["time_to_first_text_ns"] is None
            ):
                metrics["time_to_first_text_ns"] = received_ns
            if kind == "commit" and metrics["time_to_first_commit_ns"] is None:
                metrics["time_to_first_commit_ns"] = received_ns
            if live:
                metrics["received_events"] += 1
            else:
                record["events"].append(
                    {
                        "event": copy.deepcopy(event),
                        "server_elapsed_ns": message["server_elapsed_ns"],
                        "client_elapsed_ns": received_ns,
                    }
                )
            if on_event is not None:
                try:
                    await _bounded(
                        on_event(copy.deepcopy(event), received_ns),
                        config.callback_timeout_s,
                        "callback_timeout",
                    )
                except (ReplayError, asyncio.CancelledError):
                    raise
                except Exception:
                    raise ReplayError("callback_failed") from None

    async def run() -> None:
        nonlocal origin, producer, consumer, live_sample_limit
        await _bounded(
            socket.send_json(
                {
                    "type": "start",
                    "protocol": protocol,
                    **(
                        {}
                        if live
                        else {
                            "sample_count": record["sample_count"],
                            "sha256": record["sha256"],
                        }
                    ),
                }
            ),
            config.send_timeout_s,
            "start_send_timeout",
        )
        ready = await _bounded(
            _receive(socket), config.ready_timeout_s, "ready_timeout"
        )
        if (
            ready.get("type") != "ready"
            or ready.get("protocol") != protocol
            or type(ready.get("chunk_samples")) is not int
            or ready["chunk_samples"] != CHUNK_SAMPLES
            or type(ready.get("sample_rate_hz")) is not int
            or ready["sample_rate_hz"] != SAMPLE_RATE_HZ
        ):
            raise ReplayError("invalid_ready")
        if live:
            limits = ready.get("limits")
            if (
                not isinstance(limits, dict)
                or not _integer(limits.get("max_samples"), 1)
                or limits["max_samples"] > MAX_LIVE_SAMPLES
            ):
                raise ReplayError("invalid_ready")
            live_sample_limit = min(live_sample_limit, limits["max_samples"])
        origin = time.monotonic_ns()
        producer = asyncio.create_task(send_source(), name="pcm-replay-sender")
        consumer = asyncio.create_task(receive_events(), name="pcm-replay-receiver")
        finished, _ = await asyncio.wait(
            (producer, consumer), return_when=asyncio.FIRST_COMPLETED
        )
        for task in finished:
            task.result()
        if not producer.done():
            # A valid done must already have observed source_finished.
            raise ReplayError("incomplete_sender")
        await _bounded(consumer, config.drain_timeout_s, "drain_timeout")
        record["status"] = "completed"

    try:
        if live:
            _check_live_input(_live_source, config, on_event)
        else:
            _check_input(pcm, on_event)
        await _bounded(run(), config.total_timeout_s, "total_timeout")
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        record["error_code"] = "cancelled"
    except ReplayError as exc:
        record["error_code"] = str(exc)
    except Exception:
        # aiohttp can put complete URLs/headers into exceptions. Never retain
        # that exception object, repr, traceback, or message in the result.
        record["error_code"] = "transport_failed"
    finally:
        tasks = [task for task in (producer, consumer) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if record["status"] != "completed":
            try:
                await asyncio.wait_for(
                    socket.send_json({"type": "cancel"}), config.send_timeout_s
                )
            except (Exception, asyncio.CancelledError):
                pass
        if origin is not None:
            metrics["elapsed_ns"] = elapsed()
    return record


def _check_live_input(source: Any, config: ReplayConfig, on_event: Any) -> None:
    if not isinstance(source, AsyncIterable) or not isinstance(config, LiveConfig):
        raise ReplayError("invalid_source")
    if on_event is not None and not inspect.iscoroutinefunction(on_event):
        raise ReplayError("callback_must_be_async")


async def _stream_live_socket(
    socket: Any,
    source: AsyncIterable[bytes],
    *,
    config: LiveConfig | None = None,
    on_event: Callable[[dict[str, Any], int], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    return await _replay_socket(
        socket,
        b"",
        config=config or LiveConfig(),
        on_event=on_event,
        _live_source=source,
    )


def _redact(value: Any, sensitive: tuple[str, ...]) -> Any:
    """Defense in depth if a peer reflects a URL/header in metadata or text."""
    if isinstance(value, str):
        for secret in sensitive:
            if secret:
                value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {_redact(k, sensitive): _redact(v, sensitive) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item, sensitive) for item in value]
    return value


async def replay(
    url: str,
    pcm: bytes,
    *,
    headers: Mapping[str, str] | None = None,
    config: ReplayConfig | None = None,
    on_event: Callable[[dict[str, Any], int], Awaitable[None]] | None = None,
    _live_source: Any = _PRERECORDED,
) -> dict[str, Any]:
    """Connect once; return bounded JSON evidence without URLs or credentials.

    Credentials belong exclusively in headers. Redirects are not followed by
    this client. The session/socket close on all terminal paths. Do not pass a
    synchronous callback: it could stall the independent source clock.
    """
    config = config or ReplayConfig()
    record = _new_record(pcm if isinstance(pcm, bytes) else b"")
    if _live_source is not _PRERECORDED:
        record.update(protocol=LIVE_PROTOCOL, diagnostic_history="not_retained")
    sensitive: tuple[str, ...] = ()
    try:
        if _live_source is not _PRERECORDED:
            _check_live_input(_live_source, config, on_event)
        else:
            _check_input(pcm, on_event)
        if not isinstance(url, str):
            raise ReplayError("invalid_url")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ("ws", "wss")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ReplayError("invalid_url")
        if headers is not None and (
            not isinstance(headers, Mapping)
            or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in headers.items()
            )
        ):
            raise ReplayError("invalid_headers")
        request_headers = dict(headers or {})
        sensitive = (url, *request_headers.values())
        try:
            import aiohttp
        except ImportError:
            raise ReplayError("missing_aiohttp") from None

        class JsonSocket:
            def __init__(self, socket: Any) -> None:
                self.socket = socket

            async def send_json(self, payload: dict[str, Any]) -> None:
                await self.socket.send_json(payload)

            async def send_bytes(self, payload: bytes) -> None:
                await self.socket.send_bytes(payload)

            async def receive_json(self) -> Any:
                message = await self.socket.receive()
                if message.type != aiohttp.WSMsgType.TEXT:
                    raise ReplayError("unexpected_frame")
                if len(message.data.encode("utf-8")) > MAX_MESSAGE_BYTES:
                    raise ReplayError("message_limit")
                return json.loads(message.data)

        async def connected() -> dict[str, Any]:
            nonlocal record
            started = time.monotonic()
            trace = aiohttp.TraceConfig()

            async def reject_redirect(*args: Any) -> None:
                raise ReplayError("redirect_forbidden")

            trace.on_request_redirect.append(reject_redirect)
            timeout = aiohttp.ClientTimeout(total=config.ready_timeout_s)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=False, trace_configs=[trace]
            ) as session:
                # aiohttp 3.14.3 retries idempotent HTTP handshakes once by
                # default. Its public ws_connect has no retry switch. Verify
                # the inspected private switch and fail closed if unavailable.
                if not hasattr(session, "_retry_connection"):
                    raise ReplayError("unsupported_transport")
                session._retry_connection = False
                # ws_connect follows redirects in aiohttp. Guard every HTTP
                # redirect before it can forward proxy credentials elsewhere.
                async with session.ws_connect(
                    url,
                    headers=request_headers,
                    max_msg_size=MAX_MESSAGE_BYTES,
                    autoping=True,
                    heartbeat=None,
                    timeout=aiohttp.ClientWSTimeout(ws_close=config.send_timeout_s),
                ) as socket:
                    remaining = config.total_timeout_s - (time.monotonic() - started)
                    if remaining <= 0:
                        raise ReplayError("total_timeout")
                    record = await _replay_socket(
                        JsonSocket(socket),
                        pcm,
                        config=replace(config, total_timeout_s=remaining),
                        on_event=on_event,
                        _live_source=_live_source,
                    )
                    return record

        overall_started = time.monotonic()
        record = await _bounded(connected(), config.total_timeout_s, "total_timeout")
        # The low-level helper preserves partial evidence when cancelled. If
        # the outer connection-inclusive deadline caused that cancellation,
        # wait_for can return its result; retain it but classify the timeout.
        if record["status"] == "cancelled" and (
            time.monotonic() - overall_started >= config.total_timeout_s
        ):
            record["status"] = "failed"
            record["error_code"] = "total_timeout"
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        record["error_code"] = "cancelled"
    except ReplayError as exc:
        record["status"] = "failed"
        record["error_code"] = str(exc)
    except Exception:
        record["status"] = "failed"
        record["error_code"] = "connection_failed"
    return cast(dict[str, Any], _redact(record, sensitive))


async def stream_live(
    url: str,
    source: AsyncIterable[bytes],
    *,
    headers: Mapping[str, str] | None = None,
    config: LiveConfig | None = None,
    on_event: Callable[[dict[str, Any], int], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Send an unknown-length live source after READY; verify its hash at EOF.

    Yield full 640-byte PCM frames, optionally one short terminal frame. The
    source supplies capture pacing; blocked sends/idle sources fail, never drop.
    Events go to the bounded async callback, not an ever-growing result history.
    No reconnect, resend, durable acknowledgement or capture device is provided.
    """
    return await replay(
        url,
        b"",
        headers=headers,
        config=config or LiveConfig(),
        on_event=on_event,
        _live_source=source,
    )
