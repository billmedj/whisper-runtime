"""Isolated duplex client checks: no credentials, network, backend, or GPU."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from examples import replay_websocket as client


def _event(sequence=1, kind="provisional", start=0, end=320):
    event = dict.fromkeys(client._EVENT_FIELDS)
    event.update(sequence_number=sequence, kind=kind, session_version=0)
    if kind != "final":
        event.update(
            segment_id="segment:0",
            revision=1,
            start_sample=start,
            end_sample=end,
            sample_rate_hz=16_000,
            text="first text",
        )
    if kind == "commit":
        event.update(
            revision=2,
            text=None,
            committed_through_sample=end,
            committed_through_ms=end // 16,
            session_version=1,
        )
    if kind == "replace":
        event.update(revision=2, supersedes_revision=1)
    return {"type": "event", "event": event, "server_elapsed_ns": 10**15}


class SourceClock:
    """Advance only the client's source clock; keep asyncio deadlines real."""

    def __init__(self, *, sleep_gate=None):
        self.now_ns = 0
        self.sleep_gate = sleep_gate

    async def sleep(self, seconds):
        if self.sleep_gate is not None:
            await self.sleep_gate.wait()
        self.now_ns += round(seconds * 1_000_000_000)
        await asyncio.sleep(0)

    def patch(self):
        # Replace module references, not process-wide asyncio.sleep or time.
        return mock.patch.multiple(
            client,
            time=SimpleNamespace(monotonic_ns=lambda: self.now_ns),
            asyncio=SimpleNamespace(**{**vars(asyncio), "sleep": self.sleep}),
        )


class DuplexSocket:
    """Minimal socket whose server output does not depend on client receives."""

    def __init__(self, pcm, *, ready=True, response_transform=None):
        self.pcm = pcm
        self.acknowledge = ready
        self.response_transform = response_transform
        self.messages = asyncio.Queue()
        self.controls = []
        self.frames = []
        self.frame_offered_ns = []
        self.first_frame = asyncio.Event()
        self.receiving = 0
        self.sending = 0
        self.receive_cancelled = False
        self.send_cancelled = False
        self.send_delay = 0
        self.send_gate = None
        self.ready_delay = 0
        self.on_frame_message = None
        self.finish_response = True

    async def send_json(self, payload):
        self.controls.append(copy.deepcopy(payload))
        if payload["type"] == "start" and self.acknowledge:
            if self.ready_delay:
                await asyncio.sleep(self.ready_delay)
            await self.messages.put(
                {
                    "type": "ready",
                    "protocol": client.PROTOCOL,
                    "sample_rate_hz": 16_000,
                    "chunk_samples": 320,
                    "optional_server_metadata": "accepted",
                }
            )
        if payload["type"] == "eof" and self.finish_response:
            n = len(self.pcm) // 2
            response = [
                _event(2, "replace", end=n),
                _event(3, "commit", end=n),
                _event(4, "final"),
                {
                    "type": "done",
                    "status": "completed",
                    "metrics": {
                        "accepted_samples": n,
                        "committed_samples": n,
                        "buffered_samples": 0,
                        "accepted_sha256": hashlib.sha256(self.pcm).hexdigest(),
                        "trace_count": 1,
                    },
                    "admissions": [
                        {
                            "sequence_number": sequence,
                            "start_sample": sequence * 320,
                            "end_sample": sequence * 320 + (len(frame) - 12) // 2,
                            "received_ns": sequence * 2,
                            "accepted_ns": sequence * 2 + 1,
                        }
                        for sequence, frame in enumerate(self.frames)
                    ],
                },
            ]
            if self.response_transform:
                response = self.response_transform(response)
            for message in response:
                await self.messages.put(message)
            # Exercise the EOF send-completion / server-done reception race.
            await asyncio.sleep(0)

    async def send_bytes(self, frame):
        self.sending += 1
        try:
            self.frames.append(frame)
            self.frame_offered_ns.append(time.monotonic_ns())
            self.first_frame.set()
            if len(self.frames) == 1:
                message = self.on_frame_message or _event()
                await self.messages.put(message)
            if self.send_delay:
                await asyncio.sleep(self.send_delay)
            if self.send_gate is not None:
                await self.send_gate.wait()
        except asyncio.CancelledError:
            self.send_cancelled = True
            raise
        finally:
            self.sending -= 1

    async def receive_json(self):
        self.receiving += 1
        try:
            result = await self.messages.get()
            if isinstance(result, Exception):
                raise result
            return result
        except asyncio.CancelledError:
            self.receive_cancelled = True
            raise
        finally:
            self.receiving -= 1


def hold_first_send_until_response(socket):
    """Order receipt of a terminal/error response before the next source frame."""
    delivered = asyncio.Event()
    original_send, original_receive = socket.send_bytes, socket.receive_json

    async def receive():
        message = await original_receive()
        if socket.frames:
            delivered.set()
        return message

    async def send(frame):
        await original_send(frame)
        if len(socket.frames) == 1:
            # The receiver processes the response without another await before
            # raising. Delivery, not a hoped-for 20 ms scheduler gap, is required.
            await delivered.wait()

    socket.receive_json, socket.send_bytes = receive, send
    return delivered


class ReplayWebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pcm = b"\x01\x80" * 777  # 320 + 320 + a partial final 137 samples.
        self.config = client.ReplayConfig(
            ready_timeout_s=0.5,
            drain_timeout_s=0.5,
            total_timeout_s=1,
        )

    async def _run(self, socket, **kwargs):
        before = asyncio.all_tasks()
        result = await client._replay_socket(
            socket, self.pcm, config=kwargs.pop("config", self.config), **kwargs
        )
        await asyncio.sleep(0)
        self.assertEqual(socket.receiving, 0)
        self.assertEqual(socket.sending, 0)
        self.assertFalse(asyncio.all_tasks() - before)
        json.dumps(result, allow_nan=False)
        return result

    async def test_success_exact_start_frames_partial_eof_and_local_clocks(self):
        socket = DuplexSocket(self.pcm)
        result = await self._run(socket)
        self.assertEqual(result["status"], "completed", result)
        digest = hashlib.sha256(self.pcm).hexdigest()
        self.assertEqual(
            socket.controls,
            [
                {
                    "type": "start",
                    "protocol": client.PROTOCOL,
                    "sample_count": 777,
                    "sha256": digest,
                },
                {"type": "eof", "chunks": 3, "samples": 777, "sha256": digest},
            ],
        )
        reconstructed = b""
        for sequence, frame in enumerate(socket.frames):
            self.assertEqual(
                client.FRAME_HEADER.unpack(frame[:12]), (sequence, sequence * 320)
            )
            reconstructed += frame[12:]
        self.assertEqual(reconstructed, self.pcm)
        self.assertEqual([len(frame) for frame in socket.frames], [652, 652, 286])
        self.assertEqual(
            [entry["scheduled_ns"] for entry in result["send_records"]],
            [20_000_000, 40_000_000, 48_562_500],
        )
        for entry in result["send_records"]:
            self.assertGreaterEqual(entry["offered_ns"], entry["scheduled_ns"])
            self.assertGreaterEqual(entry["completed_ns"], entry["offered_ns"])
        metrics = result["client_metrics"]
        self.assertTrue(metrics["sender_finished"])
        self.assertTrue(metrics["eof_sent"])
        self.assertEqual(metrics["sent_samples"], 777)
        self.assertLess(metrics["time_to_first_text_ns"], 200_000_000)
        self.assertLess(metrics["time_to_first_commit_ns"], 200_000_000)
        self.assertGreaterEqual(metrics["eof_to_done_ns"], 0)
        self.assertEqual(result["events"][0]["server_elapsed_ns"], 10**15)

    async def test_ready_warmup_is_outside_source_clock(self):
        socket = DuplexSocket(self.pcm)
        socket.ready_delay = 0.05
        started = time.monotonic_ns()
        result = await self._run(socket)
        self.assertEqual(result["status"], "completed", result)
        self.assertGreater(socket.frame_offered_ns[0] - started, 60_000_000)
        self.assertLess(result["send_records"][0]["offered_ns"], 100_000_000)

    async def test_receiver_and_async_callback_do_not_pause_source(self):
        self.pcm = b"\x01\x00" * 3_200
        socket = DuplexSocket(self.pcm)
        observations = []

        async def on_event(event, elapsed_ns):
            if event["kind"] == "provisional":
                observations.append(len(socket.frames))
                event["text"] = "mutated callback copy"
                await asyncio.sleep(0.1)
                observations.append(len(socket.frames))

        result = await self._run(socket, on_event=on_event)
        self.assertEqual(result["status"], "completed", result)
        self.assertLess(observations[0], observations[1])
        self.assertLess(observations[0], len(socket.frames))
        self.assertEqual(result["events"][0]["event"]["text"], "first text")

    async def test_slow_send_is_cancelled_and_never_sends_eof(self):
        socket = DuplexSocket(self.pcm)
        socket.send_delay = 10
        result = await self._run(
            socket, config=replace(self.config, send_timeout_s=0.015)
        )
        self.assertEqual(result["error_code"], "send_timeout")
        self.assertTrue(socket.send_cancelled)
        self.assertTrue(socket.receive_cancelled)
        self.assertNotIn("eof", [m["type"] for m in socket.controls])
        self.assertEqual(socket.controls[-1], {"type": "cancel"})
        self.assertIsNone(result["send_records"][0]["completed_ns"])

    async def test_send_completion_source_lateness_prevents_eof(self):
        socket = DuplexSocket(self.pcm)
        clock = SourceClock()
        original_send = socket.send_bytes

        async def delayed_send(frame):
            await original_send(frame)
            clock.now_ns += 40_000_000

        socket.send_bytes = delayed_send
        with clock.patch():
            result = await self._run(
                socket, config=replace(self.config, max_source_lateness_s=0.02)
            )
        self.assertEqual(result["error_code"], "source_late")
        self.assertEqual(result["client_metrics"]["sent_chunks"], 1)
        self.assertEqual(result["send_records"][0]["offered_ns"], 20_000_000)
        self.assertEqual(result["send_records"][0]["completed_ns"], 60_000_000)
        self.assertNotIn("eof", [m["type"] for m in socket.controls])

    async def test_malformed_control_disconnected_and_oversize_messages(self):
        cases = [
            (None, "invalid_message"),
            ({"type": "ready"}, "unexpected_message"),
            ({"type": "ping"}, "unexpected_message"),
            (RuntimeError("secret wss://private.invalid/token"), "receive_failed"),
            (
                {"type": "event", "body": "x" * client.MAX_MESSAGE_BYTES},
                "message_limit",
            ),
            (
                {"type": "event", "event": {}, "server_elapsed_ns": -1},
                "invalid_event_envelope",
            ),
        ]
        for message, expected in cases:
            with self.subTest(expected=expected, message_type=type(message).__name__):
                socket = DuplexSocket(self.pcm)
                # None needs an explicit queue override because it is a sentinel.
                original_send = socket.send_bytes

                async def send(frame):
                    await original_send(frame)
                    if len(socket.frames) == 1:
                        socket.messages.get_nowait()
                        await socket.messages.put(message)

                socket.send_bytes = send
                hold_first_send_until_response(socket)
                with SourceClock().patch():
                    result = await self._run(socket)
                self.assertEqual(result["error_code"], expected, result)
                self.assertNotIn("secret", json.dumps(result))
                self.assertNotIn("private.invalid", json.dumps(result))
                self.assertNotIn("eof", [m["type"] for m in socket.controls])

    async def test_terminal_validation_failure_cases(self):
        def mutate_commit(response):
            response[1]["event"]["start_sample"] = 1
            return response

        def mutate_hash(response):
            response[-1]["metrics"]["accepted_sha256"] = "0" * 64
            return response

        def mutate_samples(response):
            response[-1]["metrics"]["accepted_samples"] -= 1
            return response

        def mutate_sequence(response):
            response[0]["event"]["sequence_number"] = 1
            return response

        def mutate_status(response):
            response[-1]["status"] = "failed"
            response[-1]["error_code"] = "wss://secret.invalid/credential"
            return response

        cases = [
            (lambda r: [r[0], r[1], r[-1]], "missing_final"),
            (lambda r: [r[0], r[1], r[2], r[2], r[-1]], "event_after_final"),
            (lambda r: [_event(2, "final"), r[-1]], "incomplete_commit_coverage"),
            (mutate_commit, "commit_coverage"),
            (mutate_hash, "accepted_hash_mismatch"),
            (mutate_samples, "incomplete_input"),
            (mutate_sequence, "event_sequence"),
            (mutate_status, "server_failed"),
        ]
        for transform, expected in cases:
            with self.subTest(expected=expected):
                socket = DuplexSocket(self.pcm, response_transform=transform)
                result = await self._run(socket)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error_code"], expected, result)
                self.assertNotIn("secret.invalid", json.dumps(result))

    async def test_premature_done_stops_producer(self):
        socket = DuplexSocket(self.pcm)
        socket.on_frame_message = {"type": "done", "status": "completed"}
        delivered = hold_first_send_until_response(socket)
        with SourceClock().patch():
            result = await self._run(socket)
        self.assertEqual(result["error_code"], "premature_done")
        self.assertTrue(delivered.is_set())
        self.assertEqual(len(socket.frames), 1)

    async def test_timeout_paths(self):
        socket = DuplexSocket(self.pcm, ready=False)
        result = await self._run(
            socket, config=replace(self.config, ready_timeout_s=0.01)
        )
        self.assertEqual(result["error_code"], "ready_timeout")
        self.assertEqual(socket.frames, [])
        socket = DuplexSocket(self.pcm)
        # Hold source pacing until cancellation instead of racing the 10 ms
        # total timeout against the first 20 ms source deadline.
        with SourceClock(sleep_gate=asyncio.Event()).patch():
            result = await self._run(
                socket, config=replace(self.config, total_timeout_s=0.01)
            )
        self.assertEqual(result["error_code"], "total_timeout")
        self.assertEqual(socket.frames, [])
        socket = DuplexSocket(self.pcm)
        socket.finish_response = False
        result = await self._run(
            socket, config=replace(self.config, drain_timeout_s=0.01)
        )
        self.assertEqual(result["error_code"], "drain_timeout")
        self.assertTrue(result["client_metrics"]["eof_sent"])

    async def test_invalid_ready(self):
        for value in (True, 321, None):
            socket = DuplexSocket(self.pcm, ready=False)
            await socket.messages.put(
                {
                    "type": "ready",
                    "protocol": client.PROTOCOL,
                    "sample_rate_hz": 16_000,
                    "chunk_samples": value,
                }
            )
            result = await self._run(socket)
            self.assertEqual(result["error_code"], "invalid_ready")
            self.assertEqual(socket.frames, [])

    async def test_callback_timeout_and_failure_keep_partial_events(self):
        async def stalled(event, elapsed_ns):
            await asyncio.sleep(10)

        async def failed(event, elapsed_ns):
            raise ValueError("secret credential")

        for callback, expected in (
            (stalled, "callback_timeout"),
            (failed, "callback_failed"),
        ):
            socket = DuplexSocket(self.pcm)
            result = await self._run(
                socket,
                on_event=callback,
                config=replace(self.config, callback_timeout_s=0.015),
            )
            self.assertEqual(result["error_code"], expected)
            self.assertEqual(len(result["events"]), 1)
            self.assertNotIn("credential", json.dumps(result))

    async def test_external_cancellation_joins_both_tasks_and_sends_cancel(self):
        socket = DuplexSocket(self.pcm)
        # Block inside the first send: observing its event does not otherwise
        # prevent the producer from reaching EOF before cancellation is delivered.
        socket.send_gate = asyncio.Event()
        with SourceClock().patch():
            task = asyncio.create_task(
                client._replay_socket(socket, self.pcm, config=self.config)
            )
            await asyncio.wait_for(socket.first_frame.wait(), timeout=1)
            task.cancel()
            result = await task
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(socket.receiving, 0)
        self.assertEqual(socket.sending, 0)
        self.assertTrue(socket.send_cancelled)
        self.assertEqual(len(socket.frames), 1)
        self.assertEqual(socket.controls[-1], {"type": "cancel"})
        self.assertNotIn("eof", [m["type"] for m in socket.controls])

    async def test_event_limit_and_invalid_event_fields(self):
        socket = DuplexSocket(self.pcm)
        result = await self._run(
            socket, config=replace(self.config, max_server_events=1)
        )
        self.assertEqual(result["error_code"], "event_limit")
        for key, value in (
            ("sequence_number", True),
            ("end_sample", 999),
            ("kind", "unknown"),
        ):
            socket = DuplexSocket(self.pcm)
            message = _event()
            message["event"][key] = value
            socket.on_frame_message = message
            result = await self._run(socket)
            self.assertEqual(result["status"], "failed")

    async def test_pcm_validation_and_async_only_callback(self):
        for pcm in (b"", b"x", "not bytes", b"\0\0" * (client.MAX_SAMPLES + 1)):
            socket = DuplexSocket(b"\0\0")
            result = await client._replay_socket(socket, pcm, config=self.config)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(socket.frames, [])
        socket = DuplexSocket(self.pcm)
        callback = mock.Mock()
        result = await self._run(socket, on_event=callback)
        self.assertEqual(result["error_code"], "callback_must_be_async")
        callback.assert_not_called()

    async def test_public_api_invalid_url_and_defensive_redaction(self):
        record = await client.replay("https://secret.invalid", self.pcm)
        self.assertEqual(record["error_code"], "invalid_url")
        self.assertNotIn("secret.invalid", json.dumps(record))
        self.assertEqual(
            client._redact(
                {"message": ["server echoed secret https://private"]},
                ("secret", "https://private"),
            ),
            {"message": ["server echoed [redacted] [redacted]"]},
        )

    async def test_revision_reference_sequence_gap_and_event_echo(self):
        def wrong_revision(response):
            response[1]["event"]["revision"] = 9
            return response

        def post_commit(response):
            after = _event(4, "replace", end=777)
            after["event"].update(revision=3, supersedes_revision=2)
            response[2]["event"]["sequence_number"] = 5
            return [*response[:2], after, *response[2:]]

        def sequence_gap(response):
            response[0]["event"]["sequence_number"] = 3
            return response

        def wrong_echo(response):
            response[-1]["events"] = []
            return response

        for transform, expected in (
            (wrong_revision, "commit_revision"),
            (post_commit, "revision_after_commit"),
            (sequence_gap, "event_sequence"),
            (wrong_echo, "event_echo_mismatch"),
        ):
            with self.subTest(expected=expected):
                result = await self._run(
                    DuplexSocket(self.pcm, response_transform=transform)
                )
                self.assertEqual(result["error_code"], expected, result)

    async def test_missing_and_malformed_admission_evidence_fails(self):
        def missing(response):
            response[-1]["admissions"].pop()
            return response

        def malformed(response):
            response[-1]["admissions"][1]["received_ns"] = 0
            return response

        for transform in (missing, malformed):
            with self.subTest(transform=transform.__name__):
                result = await self._run(
                    DuplexSocket(self.pcm, response_transform=transform)
                )
                self.assertEqual(result["error_code"], "invalid_admissions", result)

    async def test_early_server_failure_preserves_only_safe_partial_counters(self):
        socket = DuplexSocket(self.pcm)
        socket.on_frame_message = {
            "type": "done",
            "status": "failed",
            "error_code": "secret credential",
            "metrics": {"accepted_samples": 320, "trace_count": "secret"},
        }
        result = await self._run(socket)
        self.assertEqual(result["error_code"], "server_failed")
        self.assertEqual(result["server_done"]["metrics"], {"accepted_samples": 320})
        self.assertNotIn("secret", json.dumps(result))

    async def test_untrusted_custom_error_cannot_expose_exception_text(self):
        async def failed(event, elapsed_ns):
            raise client.ReplayError("secret credential")

        result = await self._run(DuplexSocket(self.pcm), on_event=failed)
        self.assertEqual(result["error_code"], "operation_failed")
        self.assertNotIn("secret", json.dumps(result))

    async def test_public_transport_options_cleanup_and_redirect_guard(self):
        socket = DuplexSocket(self.pcm)
        socket_context = SimpleNamespace(closed=False)
        session_context = SimpleNamespace(closed=False)
        captured = {}

        class Context:
            def __init__(self, value, state):
                self.value, self.state = value, state

            async def __aenter__(self):
                return self.value

            async def __aexit__(self, *exc):
                self.state.closed = True

        class RawSocket:
            send_json = socket.send_json
            send_bytes = socket.send_bytes

            async def receive(self):
                value = await socket.receive_json()
                return SimpleNamespace(type=1, data=json.dumps(value))

        class Session:
            _retry_connection = True

            def ws_connect(self, url, **kwargs):
                captured.update(url=url, socket_options=kwargs)
                captured["retry_enabled"] = self._retry_connection
                return Context(RawSocket(), socket_context)

        def session(**kwargs):
            captured["session_options"] = kwargs
            return Context(Session(), session_context)

        fake_aiohttp = SimpleNamespace(
            ClientSession=session,
            TraceConfig=lambda: SimpleNamespace(on_request_redirect=[]),
            ClientTimeout=lambda **kw: kw,
            ClientWSTimeout=lambda **kw: kw,
            WSMsgType=SimpleNamespace(TEXT=1),
        )
        with mock.patch.dict("sys.modules", {"aiohttp": fake_aiohttp}):
            result = await client.replay(
                "wss://private.invalid/websocket",
                self.pcm,
                headers={"Modal-Key": "private-key", "Modal-Secret": "private-secret"},
                config=self.config,
            )
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(socket_context.closed)
        self.assertTrue(session_context.closed)
        self.assertFalse(captured["retry_enabled"])
        self.assertFalse(captured["session_options"]["trust_env"])
        self.assertTrue(captured["socket_options"]["autoping"])
        self.assertEqual(captured["socket_options"]["max_msg_size"], 1_048_576)
        self.assertEqual(
            captured["socket_options"]["headers"]["Modal-Key"], "private-key"
        )
        self.assertNotIn("private", json.dumps(result))
        trace = captured["session_options"]["trace_configs"][0]
        with self.assertRaisesRegex(client.ReplayError, "redirect_forbidden"):
            await trace.on_request_redirect[0](None, None, None)


class LiveDuplexSocket(DuplexSocket):
    def __init__(self, pcm, *, max_samples=client.MAX_LIVE_SAMPLES, transform=None):
        self.max_samples = max_samples

        def live_done(response):
            done = response[-1]
            done.update(protocol=client.LIVE_PROTOCOL, source_eof_received=True)
            done["metrics"].update(accepted_chunks=len(self.frames), event_count=4)
            done.pop("admissions")
            return transform(response) if transform else response

        super().__init__(pcm, response_transform=live_done)

    async def send_json(self, payload):
        if payload["type"] == "start":
            self.controls.append(copy.deepcopy(payload))
            await self.messages.put(
                dict(
                    type="ready",
                    protocol=client.LIVE_PROTOCOL,
                    sample_rate_hz=16000,
                    chunk_samples=320,
                    limits={"max_samples": self.max_samples},
                )
            )
        else:
            await super().send_json(payload)


class LiveReplayTests(unittest.IsolatedAsyncioTestCase):
    def source(self, chunks, *, fail=False):
        self.source_closed = False

        async def generate():
            try:
                for chunk in chunks:
                    yield chunk
                if fail:
                    raise RuntimeError("secret source failure")
            finally:
                self.source_closed = True

        return generate()

    async def test_invalid_live_source_or_replay_config_fails_before_start(self):
        for source, config in (
            (None, client.LiveConfig()),
            ([bytes(640)], client.LiveConfig()),
            (self.source([bytes(640)]), client.ReplayConfig()),
        ):
            sock = LiveDuplexSocket(bytes(640))
            result = await client._stream_live_socket(sock, source, config=config)
            self.assertEqual(result["protocol"], client.LIVE_PROTOCOL)
            self.assertEqual(result["error_code"], "invalid_source")
            self.assertNotIn("start", [item["type"] for item in sock.controls])

    async def test_start_needs_no_future_length_or_hash_and_eof_binds_observed_source(
        self,
    ):
        pcm = b"\x01\x00" * 777
        sock, received = LiveDuplexSocket(pcm), []

        async def event(value, elapsed):
            received.append(value)

        result = await client._stream_live_socket(
            sock, self.source([pcm[:640], pcm[640:1280], pcm[1280:]]), on_event=event
        )
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(
            sock.controls[0], {"type": "start", "protocol": client.LIVE_PROTOCOL}
        )
        self.assertEqual(
            sock.controls[-1],
            dict(
                type="eof",
                samples=777,
                chunks=3,
                sha256=hashlib.sha256(pcm).hexdigest(),
            ),
        )
        self.assertEqual(result["events"], [])
        self.assertEqual(result["send_records"], [])
        self.assertEqual(result["client_metrics"]["received_events"], 4)
        self.assertEqual(received[-1]["kind"], "final")
        self.assertTrue(self.source_closed)

    async def test_invalid_source_frames_never_manufacture_eof(self):
        for chunks in ([], [b""], [bytes(3)], [bytes(642)], [bytes(2), bytes(640)]):
            with self.subTest(lengths=list(map(len, chunks))):
                sock = LiveDuplexSocket(bytes(640))
                result = await client._stream_live_socket(sock, self.source(chunks))
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("eof", [item["type"] for item in sock.controls])
                self.assertTrue(self.source_closed)

    async def test_negotiated_input_cap_stops_before_over_limit_frame(self):
        sock = LiveDuplexSocket(bytes(1280), max_samples=320)
        result = await client._stream_live_socket(
            sock, self.source([bytes(640), bytes(640)])
        )
        self.assertEqual(result["error_code"], "input_limit")
        self.assertEqual(len(sock.frames), 1)
        self.assertNotIn("eof", [item["type"] for item in sock.controls])

    async def test_live_source_crosses_v1_duration_without_replay_sleep_or_history_growth(
        self,
    ):
        pcm = bytes((client.MAX_SAMPLES + 320) * 2)
        sock = LiveDuplexSocket(pcm)

        async def source():
            for offset in range(0, len(pcm), 640):
                yield pcm[offset : offset + 640]

        result = await client._stream_live_socket(sock, source())
        self.assertEqual(result["status"], "completed", result)
        self.assertGreater(result["sample_count"], client.MAX_SAMPLES)
        self.assertEqual(result["client_metrics"]["sent_chunks"], 6001)
        self.assertEqual(result["send_records"], [])
        self.assertEqual(result["events"], [])

    async def test_source_exception_is_redacted_and_iterator_closed(self):
        sock = LiveDuplexSocket(bytes(640))
        result = await client._stream_live_socket(
            sock, self.source([bytes(640)], fail=True)
        )
        self.assertEqual(result["error_code"], "source_failed")
        self.assertNotIn("secret", json.dumps(result))
        self.assertNotIn("eof", [item["type"] for item in sock.controls])
        self.assertTrue(self.source_closed)

    async def test_idle_source_and_blocked_send_are_bounded_and_cancelled(self):
        closed = asyncio.Event()

        async def idle():
            try:
                await asyncio.Event().wait()
                yield bytes(640)
            finally:
                closed.set()

        sock = LiveDuplexSocket(bytes(640))
        result = await client._stream_live_socket(
            sock, idle(), config=replace(client.LiveConfig(), source_timeout_s=0.01)
        )
        self.assertEqual(result["error_code"], "source_timeout")
        self.assertTrue(closed.is_set())
        sock = LiveDuplexSocket(bytes(640))
        sock.send_delay = 0.05
        result = await client._stream_live_socket(
            sock,
            self.source([bytes(640)]),
            config=replace(client.LiveConfig(), send_timeout_s=0.01),
        )
        self.assertEqual(result["error_code"], "send_timeout")
        self.assertTrue(sock.send_cancelled)
        self.assertTrue(self.source_closed)
        self.assertNotIn("eof", [item["type"] for item in sock.controls])

    async def test_cancellation_joins_receiver_and_closes_live_source(self):
        entered, closed = asyncio.Event(), asyncio.Event()

        async def source():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield bytes(640)
            finally:
                closed.set()

        sock = LiveDuplexSocket(bytes(640))
        task = asyncio.create_task(client._stream_live_socket(sock, source()))
        await entered.wait()
        task.cancel()
        result = await task
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(closed.is_set())
        self.assertEqual(sock.receiving, 0)
        self.assertEqual(sock.controls[-1], {"type": "cancel"})

    async def test_done_digest_and_chunk_counters_are_not_trusted(self):
        for key, value, expected in (
            ("accepted_sha256", "0" * 64, "accepted_hash_mismatch"),
            ("accepted_chunks", 5, "invalid_done"),
        ):

            def change(response):
                response[-1]["metrics"][key] = value
                return response

            sock = LiveDuplexSocket(bytes(640), transform=change)
            result = await client._stream_live_socket(sock, self.source([bytes(640)]))
            self.assertEqual(result["error_code"], expected, result)


class ReplayConfigTests(unittest.TestCase):
    def test_limits_cannot_disable_bounds(self):
        for kwargs in (
            {"send_timeout_s": 0},
            {"send_timeout_s": 1},
            {"total_timeout_s": 161},
            {"ready_timeout_s": float("nan")},
            {"max_source_lateness_s": True},
            {"max_server_events": True},
            {"max_server_events": 1001},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                client.ReplayConfig(**kwargs)

    def test_live_limits_are_separate_finite_and_longer_than_prerecorded_v1(self):
        self.assertEqual(client.LiveConfig().max_samples, 3600 * 16000)
        self.assertEqual(client.ReplayConfig().total_timeout_s, 160)
        for kwargs in (
            {"total_timeout_s": 3701},
            {"max_samples": True},
            {"max_samples": 3600 * 16000 + 1},
            {"source_timeout_s": 0},
            {"source_timeout_s": float("inf")},
            {"max_server_events": 100001},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                client.LiveConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
