"""Live-v2 protocol, bounded native-owner integration and loopback WebSocket QA.

CPU fixtures are imported under a temporary path, as in the other tools tests.
"""

import asyncio
import hashlib
import importlib.util
import json
import socket
import sys
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from examples import pcm_websocket as server
from examples import replay_websocket as client
from examples.pcm_live import LIVE_PROTOCOL, LiveLimits
from whisper_runtime.adapters import ContinuousStreamConfig, ContinuousTranscriptStream


def live_start():
    return {"type": "start", "protocol": LIVE_PROTOCOL}


def eof(pcm, chunks):
    return {
        "type": "eof",
        "chunks": chunks,
        "samples": len(pcm) // 2,
        "sha256": hashlib.sha256(pcm).hexdigest(),
    }


class LiveProtocolTests(unittest.TestCase):
    def connection(self, **limits):
        connection = server.StreamConnection(
            Mock(), live_start(), live_limits=LiveLimits(**limits)
        )
        connection.stream = Mock()
        return connection

    def test_live_start_omits_future_metadata_and_rejects_mixed_versions(self):
        server.validate_start(live_start())
        for change in (
            {"sha256": "0" * 64},
            {"sample_count": 320},
            {"protocol": server.PROTOCOL},
        ):
            with self.subTest(change=change), self.assertRaises(server.ProtocolError):
                server.validate_start({**live_start(), **change})

    def test_final_length_hash_and_chunks_are_verified_only_at_eof(self):
        connection = self.connection()
        pcm = b"\x01\x00" * 337
        connection.admit(server.HEADER.pack(0, 0) + pcm[:640])
        connection.admit(server.HEADER.pack(1, 320) + pcm[640:])
        for change in (
            {"chunks": 1},
            {"samples": 336},
            {"sha256": "0" * 64},
            {"samples": True},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(server.ProtocolError, "invalid_eof"),
            ):
                connection.finish({**eof(pcm, 2), **change})
        connection.stream.finish_input.assert_not_called()
        connection.finish(eof(pcm, 2))
        connection.stream.finish_input.assert_called_once_with()
        self.assertEqual(connection.admissions, [])
        with self.assertRaises(server.ProtocolError):
            connection.finish(eof(pcm, 2))

    def test_sequence_gap_duplicate_odd_and_post_partial_frames_rejected(self):
        connection = self.connection()
        frame = server.HEADER.pack(0, 0) + bytes(640)
        connection.admit(frame)
        for bad in (
            frame,
            server.HEADER.pack(2, 320) + bytes(640),
            server.HEADER.pack(1, 321) + bytes(640),
            server.HEADER.pack(1, 320) + bytes(3),
        ):
            with self.subTest(size=len(bad)), self.assertRaises(server.ProtocolError):
                connection.admit(bad)
        connection.admit(server.HEADER.pack(1, 320) + bytes(2))
        with self.assertRaises(server.ProtocolError):
            connection.admit(server.HEADER.pack(2, 321) + bytes(2))
        self.assertEqual(connection.stream.push.call_count, 2)

    def test_live_cap_is_explicit_and_not_the_legacy_120_second_cap(self):
        connection = self.connection(max_samples=server.MAX_SAMPLES + 320)
        connection.samples = server.MAX_SAMPLES
        connection.chunks = server.MAX_SAMPLES // 320
        connection.admit(
            server.HEADER.pack(connection.chunks, connection.samples) + bytes(640)
        )
        with self.assertRaisesRegex(server.ProtocolError, "input_limit"):
            connection.admit(
                server.HEADER.pack(connection.chunks, connection.samples) + bytes(640)
            )
        self.assertEqual(connection.stream.push.call_count, 1)
        self.assertEqual(LiveLimits().max_samples, 3600 * 16000)

    def test_live_limits_cannot_disable_bounds(self):
        for options in (
            {"max_samples": 0},
            {"max_samples": True},
            {"max_duration_s": float("inf")},
            {"max_duration_s": 3701},
            {"input_timeout_s": 0},
            {"max_events": 100001},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LiveLimits(**options)


class LiveTransportTests(unittest.IsolatedAsyncioTestCase):
    def factory(self):
        # Keep tools discovery independent of PYTHONPATH=tests. Restore the
        # search path before any server thread or asynchronous work starts.
        with patch.object(
            sys, "path", [str(Path(__file__).resolve().parents[1] / "tests"), *sys.path]
        ):
            fixture = importlib.import_module("test_continuous_evidence")
        self.closed = threading.Event()
        self.created = 0

        def create():
            self.created += 1
            self.adapter = fixture.EvidenceNativeAdapter(fixture.speech_result)
            stream = ContinuousTranscriptStream(
                self.adapter,
                stream_id="live-test",
                mel_builder=lambda value: value,
                config=ContinuousStreamConfig(
                    preview_interval_ms=10,
                    max_window_ms=100,
                    max_buffer_ms=200,
                    holdback_ms=0,
                    input_evidence=True,
                    source_units=True,
                ),
            )
            self.stream = stream
            stream.finish_input = Mock(wraps=stream.finish_input)

            def finalize():
                self.closed.set()
                return {
                    "capacity_restored": self.adapter.budget.available
                    == self.adapter.capacity
                }

            return stream, finalize

        return create

    async def connect_asgi(self, app):
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()

        class Socket:
            async def send_json(self, value):
                await incoming.put(
                    {"type": "websocket.receive", "text": json.dumps(value)}
                )

            async def send_bytes(self, value):
                await incoming.put({"type": "websocket.receive", "bytes": value})

            async def receive_json(self):
                message = await outgoing.get()
                if message["type"] != "websocket.send":
                    raise ConnectionError("closed")
                return json.loads(message["text"])

        await incoming.put({"type": "websocket.connect"})
        task = asyncio.create_task(
            app({"type": "websocket"}, incoming.get, outgoing.put)
        )
        self.assertEqual((await outgoing.get())["type"], "websocket.accept")

        async def cleanup():
            await incoming.put({"type": "websocket.disconnect"})
            await asyncio.wait_for(task, 3)

        self.addAsyncCleanup(cleanup)
        return Socket(), incoming

    async def test_unknown_source_to_real_controller_has_compact_verified_completion(
        self,
    ):
        sock, _ = await self.connect_asgi(server.make_app(self.factory()))
        pcm, received = b"\x01\x00" * 337, []

        async def source():
            self.assertEqual(self.created, 1)  # Source consumption starts after READY.
            yield pcm[:640]
            yield pcm[640:]

        async def event(value, elapsed_ns):
            received.append(value)

        result = await client._stream_live_socket(sock, source(), on_event=event)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["sample_count"], 337)
        self.assertEqual(result["sha256"], hashlib.sha256(pcm).hexdigest())
        self.assertEqual(result["events"], [])
        self.assertEqual(result["send_records"], [])
        self.assertEqual(received[-1]["kind"], "final")
        self.assertNotIn("admissions", result["server_done"])
        self.assertNotIn("decision_traces", result["server_done"])
        self.assertTrue(result["server_done"]["metadata"]["capacity_restored"])
        self.assertTrue(self.closed.is_set())

    async def test_pinned_prerecorded_server_rejects_live_before_native_setup(self):
        sock, _ = await self.connect_asgi(
            server.make_app(self.factory(), expected_samples=320)
        )
        await sock.send_json(live_start())
        self.assertEqual((await sock.receive_json())["status"], "failed")
        self.assertEqual(self.created, 0)

    async def test_disconnect_does_not_manufacture_eof_and_releases_capacity(self):
        sock, incoming = await self.connect_asgi(server.make_app(self.factory()))
        await sock.send_json(live_start())
        self.assertEqual((await sock.receive_json())["protocol"], LIVE_PROTOCOL)
        await sock.send_bytes(server.HEADER.pack(0, 0) + bytes(640))
        await incoming.put({"type": "websocket.disconnect"})
        for _ in range(100):
            if self.closed.is_set():
                break
            await asyncio.sleep(0.005)
        self.assertTrue(self.closed.is_set())
        self.stream.finish_input.assert_not_called()
        self.assertEqual(self.adapter.budget.available, self.adapter.capacity)

    async def test_server_event_limit_is_failure_not_silent_output_drop(self):
        sock, _ = await self.connect_asgi(
            server.make_app(self.factory(), live_limits=LiveLimits(max_events=1))
        )

        async def source():
            yield bytes(640)

        result = await client._stream_live_socket(sock, source())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(
            result["client_metrics"]["sender_finished"]
            and result["status"] == "completed"
        )
        self.assertTrue(self.closed.is_set())

    async def test_server_session_deadline_closes_an_idle_live_connection(self):
        sock, _ = await self.connect_asgi(
            server.make_app(self.factory(), live_limits=LiveLimits(max_duration_s=0.03))
        )
        await sock.send_json(live_start())
        self.assertEqual((await sock.receive_json())["type"], "ready")
        result = await asyncio.wait_for(sock.receive_json(), 1)
        self.assertEqual(result["status"], "failed")
        for _ in range(100):
            if self.closed.is_set():
                break
            await asyncio.sleep(0.005)
        self.assertTrue(self.closed.is_set())
        self.stream.finish_input.assert_not_called()

    @unittest.skipUnless(
        importlib.util.find_spec("aiohttp") and importlib.util.find_spec("uvicorn"),
        "optional loopback WebSocket dependencies unavailable",
    )
    async def test_real_loopback_websocket_uses_public_live_client(self):
        import uvicorn

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        address = listener.getsockname()
        listener.listen(8)
        app = server.make_app(self.factory())
        host = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="critical"))
        task = asyncio.create_task(host.serve(sockets=[listener]))
        try:
            for _ in range(100):
                if host.started:
                    break
                await asyncio.sleep(0.005)
            self.assertTrue(host.started)

            async def source():
                yield bytes(640)
                yield bytes(34)

            result = await client.stream_live(
                f"ws://127.0.0.1:{address[1]}/",
                source(),
                config=replace(
                    client.LiveConfig(), ready_timeout_s=2, total_timeout_s=5
                ),
            )
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["sample_count"], 337)
        finally:
            host.should_exit = True
            await asyncio.wait_for(task, 3)
            listener.close()


if __name__ == "__main__":
    unittest.main()
