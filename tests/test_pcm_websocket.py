"""Network framing and native-owner lifecycle without sockets or a GPU."""

import asyncio
import hashlib
import json
import threading
import time
import unittest
from unittest.mock import patch

from test_continuous_evidence import EvidenceNativeAdapter, speech_result

from examples.pcm_websocket import (
    HEADER,
    PROTOCOL,
    ProtocolError,
    StreamConnection,
    make_app,
    parse_control,
    validate_start,
)
from whisper_runtime.adapters import ContinuousStreamConfig, ContinuousTranscriptStream


def start_message(pcm):
    return {
        "type": "start",
        "protocol": PROTOCOL,
        "sample_count": len(pcm) // 2,
        "sha256": hashlib.sha256(pcm).hexdigest(),
    }


class SocketTests(unittest.IsolatedAsyncioTestCase):
    def factory(self, *, fail_once=None, on_step=None):
        self.adapter = None
        self.closed_thread = None
        self.factory_calls = 0

        def create():
            self.factory_calls += 1
            self.adapter = EvidenceNativeAdapter(
                speech_result, fail_once=fail_once, on_step=on_step
            )
            adapter = self.adapter
            stream = ContinuousTranscriptStream(
                adapter,
                stream_id="socket-test",
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

            def finalize():
                self.closed_thread = threading.current_thread().name
                self.assertTrue(stream.done)
                return {
                    "capacity_restored": adapter.budget.available == adapter.capacity
                }

            return stream, finalize

        return create

    async def exchange(self, app, pcm, *, after_ready=None):
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        await incoming.put({"type": "websocket.connect"})
        task = asyncio.create_task(
            app({"type": "websocket"}, incoming.get, outgoing.put)
        )
        self.assertEqual((await outgoing.get())["type"], "websocket.accept")
        await incoming.put(
            {"type": "websocket.receive", "text": json.dumps(start_message(pcm))}
        )
        ready = json.loads((await asyncio.wait_for(outgoing.get(), 2))["text"])
        if ready["type"] != "ready":
            await task
            return [ready]
        if after_ready is not None:
            await after_ready(incoming)
        else:
            for sequence, offset in enumerate(range(0, len(pcm), 640)):
                await incoming.put(
                    {
                        "type": "websocket.receive",
                        "bytes": HEADER.pack(sequence, offset // 2)
                        + pcm[offset : offset + 640],
                    }
                )
            await incoming.put(
                {
                    "type": "websocket.receive",
                    "text": json.dumps(
                        {
                            "type": "eof",
                            "chunks": (len(pcm) + 639) // 640,
                            "samples": len(pcm) // 2,
                            "sha256": hashlib.sha256(pcm).hexdigest(),
                        }
                    ),
                }
            )
        messages = []
        while True:
            item = await asyncio.wait_for(outgoing.get(), 3)
            if item["type"] == "websocket.close":
                break
            messages.append(json.loads(item["text"]))
        await asyncio.wait_for(task, 3)
        return messages

    async def test_complete_partial_chunk_through_real_transaction(self):
        pcm = b"\x01\x00" * 337
        app = make_app(self.factory())
        messages = await self.exchange(app, pcm)
        done = messages[-1]
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["metrics"]["accepted_samples"], 337)
        self.assertEqual(done["metrics"]["committed_samples"], 337)
        self.assertEqual(done["metrics"]["buffered_samples"], 0)
        self.assertEqual(
            done["metrics"]["accepted_sha256"], hashlib.sha256(pcm).hexdigest()
        )
        self.assertEqual(len(done["admissions"]), 2)
        self.assertEqual(done["events"], messages[:-1])
        self.assertEqual(done["events"][-1]["event"]["kind"], "final")
        self.assertTrue(done["metadata"]["capacity_restored"])
        self.assertEqual(self.closed_thread, "whisper-websocket-owner")
        self.assertFalse(
            any(t.name == self.closed_thread for t in threading.enumerate())
        )

    async def test_disconnect_does_not_create_source_eof(self):
        async def disconnect(incoming):
            await incoming.put(
                {"type": "websocket.receive", "bytes": HEADER.pack(0, 0) + bytes(640)}
            )
            await incoming.put({"type": "websocket.disconnect"})

        messages = await self.exchange(
            make_app(self.factory()), bytes(1280), after_ready=disconnect
        )
        done = messages[-1]
        self.assertEqual(done["status"], "failed")
        self.assertFalse(done["source_eof_received"])
        self.assertIsNone(done["eof_received_ns"])
        self.assertEqual(done["metrics"]["accepted_samples"], 320)
        self.assertTrue(done["metadata"]["capacity_restored"])

    async def test_cancel_is_not_success(self):
        async def cancel(incoming):
            await incoming.put(
                {"type": "websocket.receive", "text": '{"type":"cancel"}'}
            )

        done = (
            await self.exchange(
                make_app(self.factory()), bytes(640), after_ready=cancel
            )
        )[-1]
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["error_code"], "cancelled")
        self.assertFalse(done["source_eof_received"])

    async def test_native_failure_does_not_emit_success_or_leak_error(self):
        messages = await self.exchange(
            make_app(self.factory(fail_once="step")), bytes(640)
        )
        self.assertEqual(messages[-1]["status"], "failed")
        self.assertNotIn("injected", json.dumps(messages))
        self.assertEqual(self.closed_thread, "whisper-websocket-owner")

    async def test_invalid_start_does_not_load_model(self):
        app = make_app(self.factory(), expected_sha256="0" * 64)
        messages = await self.exchange(app, bytes(640))
        self.assertEqual(messages[-1]["status"], "failed")
        self.assertEqual(self.factory_calls, 0)

    async def test_duplicate_packet_stops_without_false_eof(self):
        async def duplicate(incoming):
            for _ in range(2):
                await incoming.put(
                    {
                        "type": "websocket.receive",
                        "bytes": HEADER.pack(0, 0) + bytes(640),
                    }
                )

        messages = await self.exchange(
            make_app(self.factory()), bytes(1280), after_ready=duplicate
        )
        self.assertEqual(messages[-1]["status"], "failed")
        self.assertTrue(self.adapter.budget.available == self.adapter.capacity)

    async def test_input_continues_while_owner_is_blocked(self):
        entered, release = threading.Event(), threading.Event()

        def slow_step():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test owner did not resume")

        connection = StreamConnection(
            self.factory(on_step=slow_step), start_message(bytes(1280))
        )
        connection.thread.start()
        try:
            for _ in range(100):
                if connection.stream is not None:
                    break
                await asyncio.sleep(0.005)
            connection.admit(HEADER.pack(0, 0) + bytes(640))
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.005)
            self.assertTrue(entered.is_set())
            connection.admit(HEADER.pack(1, 320) + bytes(640))
            self.assertEqual(connection.samples, 640)
        finally:
            connection.abort("cancelled")
            release.set()
            await asyncio.to_thread(connection.thread.join, 2)
        self.assertFalse(connection.thread.is_alive())

    def test_strict_protocol_inputs(self):
        for text in ("[]", '{"type":"start","type":"eof"}', "{", '"x"', "x" * 1025):
            with self.subTest(text=text[:32]), self.assertRaises(ProtocolError):
                parse_control(text)
        for value in (True, 0, -1, 1920001, 1.0, "320"):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                validate_start({**start_message(bytes(640)), "sample_count": value})
        connection = StreamConnection(self.factory(), start_message(bytes(640)))
        for packet in (None, b"", bytes(12), bytes(653), bytearray(652)):
            with self.subTest(packet=type(packet)), self.assertRaises(ProtocolError):
                connection.admit(packet)

    async def test_eof_bookkeeping_is_atomic_with_owner_completion(self):
        pcm = bytes(640)
        connection = StreamConnection(self.factory(), start_message(pcm))
        connection.thread.start()
        while connection.stream is None:
            await asyncio.sleep(0.002)
        connection.admit(HEADER.pack(0, 0) + pcm)
        finish = connection.stream.finish_input

        def delayed_finish():
            result = finish()
            # Expose runtime EOF well before receiver-side bookkeeping returns.
            time.sleep(0.05)
            return result

        connection.stream.finish_input = delayed_finish
        try:
            await asyncio.to_thread(
                connection.finish,
                {
                    "type": "eof",
                    "chunks": 1,
                    "samples": 320,
                    "sha256": hashlib.sha256(pcm).hexdigest(),
                },
            )
            await asyncio.to_thread(connection.thread.join, 2)
            self.assertEqual(connection.result["status"], "completed")
        finally:
            connection.abort("test_cleanup")
            await asyncio.to_thread(connection.thread.join, 2)

    async def test_native_close_does_not_hold_ingress_guard(self):
        closing, release, aborted = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        create = self.factory()

        def factory():
            stream, finalize = create()
            close = stream.close

            def slow_close():
                closing.set()
                release.wait(2)
                return close()

            stream.close = slow_close
            return stream, finalize

        connection = StreamConnection(factory, start_message(bytes(640)))
        connection.thread.start()
        while connection.stream is None:
            await asyncio.sleep(0.002)
        connection.abort("cancelled")
        while not closing.is_set():
            await asyncio.sleep(0.002)

        def cancel_again():
            connection.abort("disconnected")
            aborted.set()

        other = threading.Thread(target=cancel_again)
        other.start()
        try:
            await asyncio.sleep(0.03)
            self.assertTrue(aborted.is_set())
        finally:
            release.set()
            await asyncio.to_thread(other.join, 2)
            await asyncio.to_thread(connection.thread.join, 2)

    async def test_event_limit_retains_last_committed_batch(self):
        with patch("examples.pcm_websocket.MAX_EVENTS", 1):
            messages = await self.exchange(make_app(self.factory()), bytes(640))
        done = messages[-1]
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["error_code"], "event_limit")
        self.assertTrue(
            any(item["event"]["kind"] == "commit" for item in done["events"])
        )

    async def test_finalizer_exception_is_redacted(self):
        create = self.factory()

        def factory():
            stream, finalize = create()

            def fail():
                finalize()
                raise RuntimeError("secret-sentinel-do-not-transmit")

            return stream, fail

        messages = await self.exchange(make_app(factory), bytes(640))
        self.assertEqual(messages[-1]["status"], "failed")
        self.assertEqual(messages[-1]["error_code"], "finalization_failed")
        self.assertNotIn("secret-sentinel", json.dumps(messages))

    async def test_paced_client_and_native_server_exchange(self):
        from examples.replay_websocket import _replay_socket

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
                    raise ConnectionError("socket closed")
                return json.loads(message["text"])

        app = make_app(self.factory())
        await incoming.put({"type": "websocket.connect"})
        task = asyncio.create_task(
            app({"type": "websocket"}, incoming.get, outgoing.put)
        )
        self.assertEqual((await outgoing.get())["type"], "websocket.accept")
        try:
            record = await _replay_socket(Socket(), b"\x01\x00" * 337)
            self.assertEqual(record["status"], "completed", record.get("error_code"))
            self.assertEqual(record["client_metrics"]["sent_samples"], 337)
            self.assertTrue(record["client_metrics"]["eof_sent"])
            self.assertEqual(record["events"][-1]["event"]["kind"], "final")
        finally:
            await incoming.put({"type": "websocket.disconnect"})
            await asyncio.wait_for(task, 3)

    async def test_audio_idle_timeout_does_not_apply_after_eof(self):
        wait_for = asyncio.wait_for

        async def short_input_wait(awaitable, timeout):
            return await wait_for(awaitable, 0.02 if timeout == 10 else timeout)

        # Final native work exceeds the scaled input-idle timeout, but stays
        # inside the separate drain deadline. No audio is expected after EOF.
        app = make_app(self.factory(on_step=lambda: time.sleep(0.04)))
        with patch("examples.pcm_websocket.asyncio.wait_for", short_input_wait):
            messages = await self.exchange(app, bytes(640))
        self.assertEqual(messages[-1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
