"""Bounded reference WebSocket transport for the continuous controller.

Deploy only behind authentication (Modal's proxy authentication in the example).
The transport does not grant publication authority or retry failed inference.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import re
import struct
import threading
import time
from dataclasses import asdict
from typing import Any, Callable

from examples.pcm_live import LIVE_PROTOCOL, LiveLimits
from whisper_runtime.adapters import ContinuousTranscriptStream, StreamEventKind

PROTOCOL = "pcm-websocket/v1"
HEADER = struct.Struct("!IQ")
CHUNK_SAMPLES = 320
MAX_SAMPLES = 120 * 16_000
MAX_MESSAGE_BYTES = 1_048_576
MAX_EVENTS = 1_000
Factory = Callable[[], tuple[ContinuousTranscriptStream, Callable[[], dict[str, Any]]]]


class ProtocolError(ValueError):
    """An invalid message; details must not include untrusted input."""


def parse_control(text: object) -> dict[str, Any]:
    if not isinstance(text, str) or len(text.encode("utf-8")) > 1_024:
        raise ProtocolError("invalid_control")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProtocolError("duplicate_field")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (ValueError, RecursionError) as error:
        raise ProtocolError("invalid_control") from error
    if not isinstance(value, dict):
        raise ProtocolError("invalid_control")
    return value


def validate_start(value: dict[str, Any]) -> None:
    if value == {"type": "start", "protocol": LIVE_PROTOCOL}:
        return
    if (
        set(value) != {"type", "protocol", "sample_count", "sha256"}
        or value["type"] != "start"
        or value["protocol"] != PROTOCOL
        or type(value["sample_count"]) is not int
        or not 1 <= value["sample_count"] <= MAX_SAMPLES
        or not isinstance(value["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise ProtocolError("invalid_start")


class StreamConnection:
    """One native owner, thread-safe admission, and a bounded output mailbox."""

    def __init__(
        self,
        factory: Factory,
        start: dict[str, Any],
        *,
        live_limits: LiveLimits | None = None,
    ) -> None:
        validate_start(start)
        self.factory, self.start = factory, start
        self.live = start["protocol"] == LIVE_PROTOCOL
        self.live_limits = live_limits or LiveLimits()
        self.chunks = self.trace_count = self.event_count = self.final_count = 0
        self.last_event_kind = None
        self.partial_frame = False
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.guard = threading.RLock()
        self.output: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        self.stream: ContinuousTranscriptStream | None = None
        self.error_code: str | None = None
        self.original_error: BaseException | None = None
        self.eof = False
        self.eof_ns: int | None = None
        self.origin_ns = 0
        self.digest = hashlib.sha256()
        self.samples = 0
        self.admissions: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.traces: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.thread = threading.Thread(
            target=self._own, name="whisper-websocket-owner", daemon=True
        )

    def elapsed(self) -> int:
        return time.monotonic_ns() - self.origin_ns

    def abort(self, code: str) -> None:
        with self.guard:
            self.error_code = self.error_code or code
            self.stop.set()

    def _emit(self, message: dict[str, Any]) -> None:
        # Bound serialized size as well as queue length.
        if len(json.dumps(message).encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ProtocolError("output_too_large")
        try:
            self.output.put_nowait(message)
        except queue.Full as error:
            raise ProtocolError("output_overload") from error

    def admit(self, packet: object) -> None:
        if not isinstance(packet, bytes) or not HEADER.size < len(packet) <= 652:
            raise ProtocolError("invalid_audio")
        sequence, start_sample = HEADER.unpack_from(packet)
        pcm = packet[HEADER.size :]
        with self.guard:
            if self.stream is None or self.stop.is_set() or self.eof:
                raise ProtocolError("input_closed")
            expected = (
                len(pcm) // 2
                if self.live
                else min(CHUNK_SAMPLES, self.start["sample_count"] - self.samples)
            )
            if self.live and self.samples + expected > self.live_limits.max_samples:
                raise ProtocolError("input_limit")
            if (
                expected <= 0
                or len(pcm) != expected * 2
                or sequence != self.chunks
                or start_sample != self.samples
                or (self.live and self.partial_frame)
            ):
                raise ProtocolError("invalid_audio_sequence")
            received = self.elapsed()
            self.stream.push(sequence, pcm)
            accepted = self.elapsed()
            self.digest.update(pcm)
            self.samples += expected
            self.chunks += 1
            self.partial_frame = expected < CHUNK_SAMPLES
            if not self.live:
                self.admissions.append(
                    {
                        "sequence_number": sequence,
                        "start_sample": start_sample,
                        "end_sample": self.samples,
                        "received_ns": received,
                        "accepted_ns": accepted,
                    }
                )

    def finish(self, value: dict[str, Any]) -> None:
        with self.guard:
            if (
                set(value) != {"type", "chunks", "samples", "sha256"}
                or value["type"] != "eof"
                or type(value["chunks"]) is not int
                or type(value["samples"]) is not int
                or value["chunks"] != self.chunks
                or value["samples"] != self.samples
                or self.samples <= 0
                or (not self.live and self.samples != self.start["sample_count"])
                or value["sha256"] != self.digest.hexdigest()
                or (not self.live and value["sha256"] != self.start["sha256"])
                or self.stream is None
                or self.stop.is_set()
                or self.eof
            ):
                raise ProtocolError("invalid_eof")
            self.stream.finish_input()
            self.eof = True
            self.eof_ns = self.elapsed()

    def _own(self) -> None:
        finalize = None
        metadata: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        complete = False
        try:
            stream, finalize = self.factory()
            if not isinstance(stream, ContinuousTranscriptStream):
                raise TypeError("factory must return a continuous stream")
            with self.guard:
                self.stream = stream
                self.origin_ns = time.monotonic_ns()
            if self.stop.is_set():
                return
            self._emit(
                {
                    "type": "ready",
                    "protocol": self.start["protocol"],
                    "sample_rate_hz": 16_000,
                    "chunk_samples": CHUNK_SAMPLES,
                    **({"limits": asdict(self.live_limits)} if self.live else {}),
                }
            )
            last_trace = -1
            driver_steps = 0
            while driver_steps < (
                self.live_limits.max_driver_steps if self.live else 100_000
            ):
                if self.stop.is_set() or stream.done:
                    break
                if self.eof_ns is not None and self.elapsed() - self.eof_ns > 15e9:
                    raise ProtocolError("drain_timeout")
                if self.live:
                    if self.elapsed() > self.live_limits.max_duration_s * 1e9:
                        raise ProtocolError("session_timeout")
                    if not stream.ready:
                        self.stop.wait(0.002)
                        continue
                driver_steps += 1
                batch = stream.step()
                observed = self.elapsed()
                messages = [
                    {
                        "type": "event",
                        "event": asdict(event),
                        "server_elapsed_ns": observed,
                    }
                    for event in batch
                ]
                # v1 preserves its committed diagnostic batch even if delivery
                # saturates. Live counts it, then fails explicitly on overflow.
                self.event_count += len(messages)
                self.final_count += sum(
                    event.kind == StreamEventKind.FINAL for event in batch
                )
                if batch:
                    self.last_event_kind = batch[-1].kind
                if not self.live:
                    self.events.extend(messages)
                if self.event_count > (
                    self.live_limits.max_events if self.live else MAX_EVENTS
                ):
                    raise ProtocolError("event_limit")
                for message in messages:
                    self._emit(message)
                trace = stream.last_trace
                if trace is not None and trace.decode_index != last_trace:
                    self.trace_count += 1
                    if self.trace_count > (
                        self.live_limits.max_driver_steps if self.live else MAX_EVENTS
                    ):
                        raise ProtocolError("trace_limit")
                    if not self.live:
                        self.traces.append(asdict(trace))
                    last_trace = trace.decode_index
                if not batch and not stream.active:
                    self.stop.wait(0.002)
            # EOF and admission counters are one receiver-side transaction.
            with self.guard:
                complete = (
                    not self.stop.is_set()
                    and self.eof
                    and (self.live or self.samples == self.start["sample_count"])
                    and stream.done
                    and stream.metrics.committed_samples == self.samples
                    and self.final_count == 1
                    and self.last_event_kind == StreamEventKind.FINAL
                )
            if not complete and self.error_code is None:
                self.abort("incomplete_stream")
        except BaseException as error:
            self.original_error = error
            if self.stream is not None and not self.live:
                try:
                    trace = self.stream.last_trace
                    if (
                        trace is not None
                        and len(self.traces) < MAX_EVENTS
                        and (
                            not self.traces
                            or self.traces[-1]["decode_index"] != trace.decode_index
                        )
                    ):
                        self.traces.append(asdict(trace))
                except BaseException:
                    pass  # Preserve the original native recovery exception.
            self.abort(
                str(error) if isinstance(error, ProtocolError) else "runtime_error"
            )
        finally:
            # Close admission before owner-thread cleanup. Never manufacture EOF.
            with self.guard:
                self.stop.set()
            # Native close can synchronize a device. Never hold the ingress
            # guard during that work: cancellation must not block the ASGI loop.
            if self.stream is not None:
                metrics = asdict(self.stream.metrics)
                try:
                    self.stream.close()
                except BaseException as error:
                    self.original_error = self.original_error or error
                    self.error_code = self.error_code or "cleanup_failed"
            if finalize is not None:
                try:
                    metadata = finalize()
                except BaseException as error:
                    self.original_error = self.original_error or error
                    self.error_code = self.error_code or "finalization_failed"
            self.result = {
                "type": "done",
                "status": "completed"
                if complete and self.error_code is None
                else "failed",
                "error_code": self.error_code,
                "metrics": {
                    **metrics,
                    "accepted_sha256": self.digest.hexdigest(),
                    "trace_count": self.trace_count if self.live else len(self.traces),
                    **(
                        {
                            "accepted_chunks": self.chunks,
                            "event_count": self.event_count,
                        }
                        if self.live
                        else {}
                    ),
                },
                "source_eof_received": self.eof,
                "eof_received_ns": self.eof_ns,
                "elapsed_ns": self.elapsed() if self.origin_ns else None,
                "admissions": self.admissions,
                "events": self.events,
                "decision_traces": self.traces,
                "metadata": metadata,
            }
            if self.live:
                for key in ("admissions", "events", "decision_traces"):
                    self.result.pop(key)
                self.result.update(
                    protocol=LIVE_PROTOCOL, diagnostic_history="not_retained"
                )
            self.finished.set()


def make_app(
    factory: Factory,
    *,
    expected_samples: int | None = None,
    expected_sha256: str | None = None,
    live_limits: LiveLimits | None = None,
) -> Callable[..., Any]:
    """Create one-shot ASGI application; authentication belongs to the ingress.

    No authentication check here can prevent an upstream GPU cold start.
    Native interruption is cooperative; the host must enforce its outer timeout.
    A disconnected client cannot recover this process-local stream by reconnecting.
    """
    used = False
    claim = threading.Lock()
    live_limits = live_limits or LiveLimits()

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        nonlocal used
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "websocket":
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        with claim:
            rejected, used = used, True
        connection: StreamConnection | None = None
        tasks: list[asyncio.Task[Any]] = []

        async def write(value: dict[str, Any]) -> None:
            payload = json.dumps(value, allow_nan=False, separators=(",", ":"))
            if len(payload.encode("utf-8")) > MAX_MESSAGE_BYTES:
                raise ProtocolError("output_too_large")
            await asyncio.wait_for(send({"type": "websocket.send", "text": payload}), 5)

        try:
            opening = await asyncio.wait_for(receive(), 10)
            if rejected or opening["type"] != "websocket.connect":
                await send({"type": "websocket.close", "code": 1008})
                return
            await send({"type": "websocket.accept"})
            message = await asyncio.wait_for(receive(), 10)
            if (
                message["type"] != "websocket.receive"
                or message.get("bytes") is not None
            ):
                raise ProtocolError("invalid_start")
            start = parse_control(message.get("text"))
            validate_start(start)
            if start["protocol"] == LIVE_PROTOCOL and (
                expected_samples is not None or expected_sha256 is not None
            ):
                raise ProtocolError("unregistered_source")
            if (
                expected_samples is not None
                and start["sample_count"] != expected_samples
            ) or (expected_sha256 is not None and start["sha256"] != expected_sha256):
                raise ProtocolError("unregistered_source")
            connection = StreamConnection(factory, start, live_limits=live_limits)
            ready = asyncio.Event()
            connection.thread.start()

            async def input_loop() -> None:
                assert connection is not None
                await ready.wait()
                while not connection.finished.is_set():
                    # After EOF there is no next audio message to time out.
                    # Keep observing disconnects while the owner drains.
                    item = (
                        await receive()
                        if connection.eof
                        else await asyncio.wait_for(
                            receive(),
                            live_limits.input_timeout_s if connection.live else 10,
                        )
                    )
                    if item["type"] == "websocket.disconnect":
                        connection.abort("disconnected")
                        return
                    if item["type"] != "websocket.receive":
                        raise ProtocolError("invalid_message")
                    if item.get("bytes") is not None:
                        if item.get("text") is not None:
                            raise ProtocolError("invalid_message")
                        connection.admit(item["bytes"])
                    else:
                        value = parse_control(item.get("text"))
                        if value == {"type": "cancel"}:
                            connection.abort("cancelled")
                            return
                        connection.finish(value)

            async def output_loop() -> None:
                assert connection is not None
                while True:
                    try:
                        item = connection.output.get_nowait()
                    except queue.Empty:
                        if connection.finished.is_set():
                            break
                        await asyncio.sleep(0.002)
                        continue
                    await write(item)
                    if item["type"] == "ready":
                        ready.set()
                assert connection.result is not None
                await write(connection.result)

            tasks = [
                asyncio.create_task(input_loop()),
                asyncio.create_task(output_loop()),
            ]
            done, _ = await asyncio.wait(
                tasks,
                timeout=live_limits.max_duration_s if connection.live else 170,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise ProtocolError("connection_timeout")
            for task in done:
                task.result()
            # Input returns only on disconnection/cancellation; owner stops first.
            if tasks[1] not in done:
                await asyncio.wait_for(asyncio.shield(tasks[1]), 5)
        except (Exception, asyncio.CancelledError) as error:
            if connection is not None:
                connection.abort(
                    str(error)
                    if isinstance(error, ProtocolError)
                    else "transport_failed"
                )
            # Fixed code only; native/provider exceptions can contain secrets or paths.
            try:
                await write(
                    {
                        "type": "done",
                        "status": "failed",
                        "error_code": "transport_failed",
                    }
                )
            except Exception:
                pass
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if connection is not None:
                if not connection.finished.is_set():
                    connection.abort("disconnected")
                # Do not block the ASGI loop or claim a timed-out native thread stopped.
                deadline = time.monotonic() + 5
                while connection.thread.is_alive() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
            try:
                await asyncio.wait_for(
                    send({"type": "websocket.close", "code": 1000}), 1
                )
            except Exception:
                pass

    return app
