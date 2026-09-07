"""Installed client bridge and CLI over real loopback, with scripted SDK work."""

import asyncio
import importlib.util
import io
import json
import socket
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from test_continuous_evidence import EvidenceNativeAdapter, speech_result

from whisper_runtime.adapters import (
    AudioBufferFullError,
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamEventKind,
    TranscriptEvent,
)
from whisper_runtime.cli import main
from whisper_runtime.remote import (
    RemoteError,
    environment_headers,
    transcribe_remote,
    validate_server,
)


class RemoteSettingsTests(unittest.TestCase):
    def test_headers_are_environment_only_and_errors_never_include_values(self):
        self.assertEqual(
            environment_headers(
                ["Authorization=AUTH", "Modal-Key=KEY"],
                environment={"AUTH": "Bearer secret", "KEY": "another-secret"},
            ),
            {"Authorization": "Bearer secret", "Modal-Key": "another-secret"},
        )
        for mappings, environment in (
            (["Authorization=Bearer secret"], {}),
            (["Authorization=AUTH"], {}),
            (["Authorization=AUTH"], {"AUTH": "secret\r\nInjected: value"}),
            (["Host=AUTH"], {"AUTH": "secret"}),
            (["Sec-WebSocket-Key=AUTH"], {"AUTH": "secret"}),
            (["X-Auth=AUTH", "x-auth=AUTH"], {"AUTH": "secret"}),
        ):
            with self.subTest(mappings=mappings), self.assertRaises(RemoteError) as ctx:
                environment_headers(mappings, environment=environment)
            self.assertNotIn("secret", str(ctx.exception))

    def test_remote_tls_and_no_url_credentials(self):
        for url in ("wss://example.com/pcm", "ws://127.0.0.1:123/", "ws://[::1]/"):
            validate_server(url)
        for url in (
            "https://example.com/secret",
            "ws://example.com/secret",
            "wss://user:secret@example.com/",
            "wss://example.com/?token=secret",
            "wss://example.com/#secret",
            "ws://127.0.0.1:secret/",
        ):
            with self.subTest(url=url), self.assertRaises(RemoteError) as ctx:
                validate_server(url)
            self.assertNotIn("secret", str(ctx.exception))

    def test_examples_keep_transport_module_identity(self):
        from examples import pcm_live, replay_websocket
        from whisper_runtime import pcm_live as installed_live
        from whisper_runtime import remote_transport

        self.assertIs(replay_websocket, remote_transport)
        self.assertIs(pcm_live, installed_live)


class RemoteBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_only_when_transport_requests_and_closes_on_completion(self):
        seen = []
        stopped = threading.Event()

        def chunks():
            try:
                seen.append("read")
                yield bytes(640)
            finally:
                stopped.set()

        async def transport(url, source, **kwargs):
            self.assertEqual(seen, [])
            self.assertEqual(await anext(source), bytes(640))
            event = TranscriptEvent(1, StreamEventKind.FINAL, session_version=0)
            await kwargs["on_event"](asdict(event), 1)
            return {"status": "completed", "sample_count": 320}

        events = []
        with patch("whisper_runtime.remote.stream_live", transport):
            samples = await transcribe_remote(
                "ws://127.0.0.1/",
                chunks(),
                cancel=threading.Event(),
                on_event=events.append,
            )
        self.assertEqual(samples, 320)
        self.assertEqual(events[0].kind, StreamEventKind.FINAL)
        self.assertTrue(stopped.is_set())

    async def test_cancellation_is_not_completion_and_closes_even_before_ready(self):
        source = Mock()
        with (
            patch(
                "whisper_runtime.remote.stream_live",
                AsyncMock(
                    return_value={"status": "cancelled", "error_code": "cancelled"}
                ),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            await transcribe_remote(
                "ws://127.0.0.1/",
                source,
                cancel=threading.Event(),
                on_event=lambda e: None,
            )
        source.close.assert_called_once_with()

    async def test_cleanup_and_secret_reflection_errors_are_fixed_and_not_delivered(
        self,
    ):
        for failure in ("cleanup", "reflection", "bare_token"):
            with self.subTest(failure=failure):
                source, callback = Mock(), Mock()
                if failure == "cleanup":
                    source.close.side_effect = RuntimeError("Bearer secret")

                async def transport(url, chunks, **kwargs):
                    if failure != "cleanup":
                        text = "Bearer secret" if failure == "reflection" else "secret"
                        await kwargs["on_event"]({"text": text}, 0)
                    return {"status": "completed", "sample_count": 320}

                with patch("whisper_runtime.remote.stream_live", transport):
                    with self.assertRaisesRegex(RemoteError, "^remote_client_failed$"):
                        await transcribe_remote(
                            "ws://127.0.0.1/",
                            source,
                            cancel=threading.Event(),
                            on_event=callback,
                            headers={"Authorization": "Bearer secret"},
                        )
                callback.assert_not_called()


@unittest.skipUnless(
    importlib.util.find_spec("aiohttp") and importlib.util.find_spec("uvicorn"),
    "optional real-loopback test dependencies unavailable",
)
class RemoteCliLoopbackTests(unittest.IsolatedAsyncioTestCase):
    async def start_server(self, *, failure=None):
        import uvicorn

        from examples import pcm_websocket

        self.closed = threading.Event()
        self.request_headers = []
        self.sent_final = False

        def factory():
            self.adapter = EvidenceNativeAdapter(
                speech_result, fail_once="prepare" if failure == "native" else None
            )
            self.stream = ContinuousTranscriptStream(
                self.adapter,
                stream_id="remote-cli-test",
                mel_builder=lambda pcm: pcm,
                config=ContinuousStreamConfig(
                    preview_interval_ms=10,
                    max_window_ms=100,
                    max_buffer_ms=200,
                    holdback_ms=0,
                    input_evidence=True,
                    source_units=True,
                ),
            )
            self.stream.finish_input = Mock(wraps=self.stream.finish_input)
            if failure == "overload":
                self.stream.push = Mock(side_effect=AudioBufferFullError("secret"))

            def finalize():
                self.closed.set()
                return {
                    "capacity_restored": self.adapter.budget.available
                    == self.adapter.capacity
                }

            return self.stream, finalize

        app = pcm_websocket.make_app(factory)

        async def capture(scope, receive, send):
            self.request_headers.extend(scope.get("headers", []))
            disconnected = False

            async def checked_send(message):
                nonlocal disconnected
                if disconnected:
                    return
                if message["type"] == "websocket.send":
                    value = json.loads(message["text"])
                    if value.get("event", {}).get("kind") == "final":
                        self.sent_final = True
                    if value.get("type") == "done":
                        if failure == "invalid_done":
                            value["metrics"]["accepted_sha256"] = "0" * 64
                            message = {**message, "text": json.dumps(value)}
                        elif failure == "disconnect_after_final":
                            disconnected = True
                            await send({"type": "websocket.close", "code": 1000})
                            return
                await send(message)

            await app(scope, receive, checked_send)

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        host = uvicorn.Server(
            uvicorn.Config(capture, lifespan="off", log_level="critical")
        )
        task = asyncio.create_task(host.serve(sockets=[listener]))

        async def cleanup():
            host.should_exit = True
            await asyncio.wait_for(task, 3)
            listener.close()

        self.addAsyncCleanup(cleanup)
        for _ in range(100):
            if host.started:
                break
            await asyncio.sleep(0.005)
        self.assertTrue(host.started)
        return f"ws://127.0.0.1:{listener.getsockname()[1]}/"

    async def invoke_cli(self, server, *, microphone=False):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "source.pcm"
            audio.write_bytes(b"\x01\x00" * 337)
            paths = {
                kind: Path(directory) / f"out.{kind}" for kind in ("txt", "srt", "vtt")
            }
            args = [
                "--server",
                server,
                "--header-env",
                "Authorization=REMOTE_TEST_AUTH",
            ]
            args += ["--microphone", "--duration", "1"] if microphone else [str(audio)]
            for kind, path in paths.items():
                args.extend(["--" + kind, str(path)])

            def run():
                with (
                    redirect_stdout(io.StringIO()) as stdout,
                    redirect_stderr(io.StringIO()) as stderr,
                ):
                    code = main(args)
                return code, stdout.getvalue(), stderr.getvalue()

            with (
                patch.dict("os.environ", {"REMOTE_TEST_AUTH": "Bearer secret"}),
                patch("whisper_runtime.native_setup.create_stream") as native,
            ):
                code, stdout, stderr = await asyncio.to_thread(run)
            native.assert_not_called()
            exports = {
                kind: path.read_text() for kind, path in paths.items() if path.exists()
            }
            self.assertNotIn(
                "Bearer secret", stdout + stderr + "".join(exports.values())
            )
            return code, stdout, stderr, exports

    async def test_full_cli_uses_actual_sdk_commits_and_three_exports(self):
        server = await self.start_server()
        code, stdout, stderr, exports = await self.invoke_cli(server)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(set(exports), {"txt", "srt", "vtt"})
        self.assertIn("[commit", stdout)
        self.assertIn("[final] complete", stdout)
        self.assertNotIn("[provisional", exports["txt"])
        self.assertTrue(exports["vtt"].startswith("WEBVTT"))
        self.assertEqual(self.stream.accepted_samples, 337)
        self.assertEqual(self.stream.metrics.committed_samples, 337)
        self.assertIn((b"authorization", b"Bearer secret"), self.request_headers)
        self.assertTrue(self.closed.is_set())

    async def assert_final_without_verified_done_fails(self, failure):
        server = await self.start_server(failure=failure)
        code, stdout, stderr, exports = await self.invoke_cli(server)
        self.assertTrue(self.sent_final)
        self.assertEqual(code, 1, stderr)
        self.assertIn("[commit", stdout)
        self.assertNotIn("[final] complete", stdout)
        self.assertEqual(exports, {})
        self.assertTrue(self.closed.is_set())

    async def test_final_then_invalid_done_has_no_complete_message_or_exports(self):
        await self.assert_final_without_verified_done_fails("invalid_done")

    async def test_final_then_disconnect_has_no_complete_message_or_exports(self):
        await self.assert_final_without_verified_done_fails("disconnect_after_final")

    async def test_native_failure_never_creates_success_exports(self):
        server = await self.start_server(failure="native")
        code, _, stderr, exports = await self.invoke_cli(server)
        self.assertEqual(code, 1, stderr)
        self.assertEqual(exports, {})
        self.assertTrue(self.closed.is_set())
        self.assertEqual(self.adapter.budget.available, self.adapter.capacity)

    async def test_admission_overload_is_an_error_without_exports_or_retry(self):
        server = await self.start_server(failure="overload")
        code, _, stderr, exports = await self.invoke_cli(server)
        self.assertEqual(code, 1, stderr)
        self.assertEqual(exports, {})
        self.assertEqual(self.stream.push.call_count, 1)
        self.stream.finish_input.assert_not_called()
        self.assertTrue(self.closed.is_set())

    async def test_source_cancellation_has_no_eof_or_success_exports(self):
        server = await self.start_server()

        def capture(*, cancel, **kwargs):
            yield bytes(640)
            # The real bounded source can return StopIteration after noticing
            # cancellation during next(). That must not become a normal EOF.
            cancel.set()

        with patch("whisper_runtime.cli.microphone_chunks", capture):
            code, _, stderr, exports = await self.invoke_cli(server, microphone=True)
        self.assertEqual(code, 130, stderr)
        self.assertEqual(exports, {})
        self.stream.finish_input.assert_not_called()
        for _ in range(100):
            if self.closed.is_set():
                break
            await asyncio.sleep(0.005)
        self.assertTrue(self.closed.is_set())


if __name__ == "__main__":
    unittest.main()
