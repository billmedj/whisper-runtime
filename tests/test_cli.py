"""CLI entrypoint and actual SDK driving with scripted native work only."""

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from test_continuous_stream import ScriptedNativeAdapter

from whisper_runtime.adapters import StreamEventKind
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
)
from whisper_runtime.audio_source import AudioSourceError
from whisper_runtime.captions import CommittedTranscript
from whisper_runtime.cli import drive_stream, main


class CliTests(unittest.TestCase):
    def stream(self, **kwargs):
        adapter = ScriptedNativeAdapter(**kwargs)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="cli-test",
            mel_builder=lambda pcm: pcm,
            config=ContinuousStreamConfig(
                preview_interval_ms=100,
                max_window_ms=500,
                max_buffer_ms=600,
                holdback_ms=100,
            ),
        )
        return stream, adapter

    def test_help_works_without_site_packages_or_ml_imports(self):
        source = Path(__file__).resolve().parents[1] / "src"
        script = f"import sys; sys.path.insert(0, {str(source)!r}); from whisper_runtime.cli import main; main(['--help'])"
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--microphone", result.stdout)
        self.assertIn("--setup-manifest", result.stdout)
        self.assertIn("--server", result.stdout)
        self.assertIn("--profile", result.stdout)
        self.assertIn("experimental-optimized-v1", result.stdout)

    def test_unknown_profile_is_usage_error_before_native_or_remote_work(self):
        with (
            patch("whisper_runtime.native_setup.create_stream") as native,
            patch("whisper_runtime.remote.transcribe_remote") as remote,
            redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as error,
        ):
            main(["audio.pcm", "--profile", "latest"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())
        native.assert_not_called()
        remote.assert_not_called()

    def test_local_profile_defaults_and_explicit_choice_reach_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.pcm"
            path.write_bytes(bytes(6400))
            for arguments, profile in (
                ([], "conservative-v1"),
                (["--profile", "conservative-v1"], "conservative-v1"),
                (
                    ["--profile", "experimental-optimized-v1"],
                    "experimental-optimized-v1",
                ),
            ):
                stream, _ = self.stream()
                with (
                    self.subTest(profile=profile),
                    patch(
                        "whisper_runtime.native_setup.create_stream",
                        return_value=stream,
                    ) as factory,
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()) as stderr,
                ):
                    self.assertEqual(
                        main(
                            [
                                str(path),
                                "--setup-manifest",
                                "m",
                                "--model",
                                "m",
                                *arguments,
                            ]
                        ),
                        0,
                    )
                self.assertEqual(factory.call_args.kwargs["profile"], profile)
                self.assertIn(f"Local execution profile: {profile}", stderr.getvalue())

    def test_remote_arguments_are_exclusive_and_validated_before_connecting(self):
        for args in (
            ["--server", "ws://127.0.0.1/", "--model", "m"],
            ["--server", "ws://127.0.0.1/", "--setup-manifest", "m"],
            ["--server", "ws://127.0.0.1/", "--device", "cpu"],
            ["--server", "ws://127.0.0.1/", "--profile", "conservative-v1"],
            ["--server", "ws://127.0.0.1/", "--profile", "experimental-optimized-v1"],
            ["--server", "ws://127.0.0.1/", "--drain-timeout", "31"],
            ["--setup-manifest", "m", "--model", "m", "--header-env", "X-A=AUTH"],
        ):
            with (
                self.subTest(args=args),
                patch("whisper_runtime.remote.transcribe_remote") as remote,
                patch("whisper_runtime.native_setup.create_stream") as native,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(["audio.pcm", *args]), 1)
            remote.assert_not_called()
            native.assert_not_called()

    def test_remote_failure_after_final_and_completion_without_final_never_export(self):
        from whisper_runtime.adapters import StreamEventKind, TranscriptEvent
        from whisper_runtime.remote import RemoteError

        with tempfile.TemporaryDirectory() as directory:
            path, output = (Path(directory) / name for name in ("audio.pcm", "out.txt"))
            path.write_bytes(bytes(640))
            for emit_final in (False, True):

                async def remote(server, source, *, on_event, **kwargs):
                    source.close()
                    if emit_final:
                        on_event(
                            TranscriptEvent(1, StreamEventKind.FINAL, session_version=0)
                        )
                        raise RemoteError("server_failed")
                    return 320

                with (
                    self.subTest(emit_final=emit_final),
                    patch("whisper_runtime.remote.transcribe_remote", remote),
                    redirect_stdout(io.StringIO()) as stdout,
                    redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(
                        main(
                            [
                                str(path),
                                "--server",
                                "ws://127.0.0.1/",
                                "--txt",
                                str(output),
                            ]
                        ),
                        1,
                    )
                self.assertFalse(output.exists())
                self.assertNotIn("[final] complete", stdout.getvalue())

    def test_sdk_backpressure_preserves_complete_pcm_and_actual_commit_revisions(self):
        stream, adapter = self.stream()
        transcript = CommittedTranscript()
        chunks = [b"\x00\x01" * 1600] * 8
        drive_stream(stream, iter(chunks), on_event=transcript.accept, cancel=Event())
        self.assertTrue(transcript.final)
        self.assertEqual(transcript.head, 12800)
        self.assertEqual(stream.accepted_samples, 12800)
        self.assertEqual(adapter.budget.lease_count, 0)
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertIn("word-", transcript.render("txt"))

    def test_live_overflow_and_native_failure_close_without_fake_final(self):
        stream, adapter = self.stream()
        transcript = CommittedTranscript()
        with self.assertRaisesRegex(AudioSourceError, "runtime buffer"):
            drive_stream(
                stream,
                iter([bytes(32000)]),
                on_event=transcript.accept,
                cancel=Event(),
                live=True,
            )
        self.assertFalse(transcript.final)
        stream, adapter = self.stream(fail_once="prepare")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            drive_stream(
                stream, iter([bytes(3200)]), on_event=transcript.accept, cancel=Event()
            )
        self.assertEqual(adapter.budget.lease_count, 0)

    def test_cancel_and_output_callback_failure_close_the_native_owner(self):
        stream, adapter = self.stream()
        stop = Event()
        stop.set()
        with self.assertRaises(KeyboardInterrupt):
            drive_stream(
                stream, iter([bytes(640)]), on_event=lambda event: None, cancel=stop
            )
        stream, adapter = self.stream()

        def fail(event):
            raise OSError("consumer failed")

        with self.assertRaisesRegex(OSError, "consumer failed"):
            drive_stream(stream, iter([bytes(3200)]), on_event=fail, cancel=Event())
        self.assertEqual(adapter.budget.lease_count, 0)

    def test_main_exports_real_sdk_commits_and_refuses_overwrite_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = (Path(directory) / name for name in ("audio.pcm", "out.txt"))
            path.write_bytes(bytes(6400))
            args = [
                str(path),
                "--setup-manifest",
                "manifest.json",
                "--model",
                "tiny.en.pt",
                "--txt",
                str(output),
            ]
            stream, _ = self.stream()
            with (
                patch(
                    "whisper_runtime.native_setup.create_stream", return_value=stream
                ),
                redirect_stdout(io.StringIO()) as stdout,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(args), 0)
            self.assertIn("[commit", stdout.getvalue())
            self.assertIn("[final]", stdout.getvalue())
            self.assertIn("word-", output.read_text())
            with (
                patch("whisper_runtime.native_setup.create_stream") as factory,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(args), 1)
            factory.assert_not_called()

    def test_failed_main_never_exports_preview_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = (Path(directory) / name for name in ("audio.pcm", "out.txt"))
            path.write_bytes(bytes(6400))
            stream, _ = self.stream(fail_once="prepare")
            with (
                patch(
                    "whisper_runtime.native_setup.create_stream", return_value=stream
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            str(path),
                            "--setup-manifest",
                            "m",
                            "--model",
                            "m",
                            "--txt",
                            str(output),
                        ]
                    ),
                    1,
                )
            self.assertFalse(output.exists())

    def test_late_cleanup_failure_cannot_return_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.pcm"
            output = Path(directory) / "out.txt"
            path.write_bytes(bytes(6400))
            stream, _ = self.stream()
            with (
                patch(
                    "whisper_runtime.native_setup.create_stream", return_value=stream
                ),
                patch.object(
                    stream,
                    "close",
                    side_effect=[None, RuntimeError("late close failure"), None],
                ),
                redirect_stdout(io.StringIO()) as stdout,
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(
                    main(
                        [
                            str(path),
                            "--setup-manifest",
                            "m",
                            "--model",
                            "m",
                            "--txt",
                            str(output),
                        ]
                    ),
                    1,
                )
            self.assertIn("late close failure", stderr.getvalue())
            self.assertNotIn("[final]", stdout.getvalue())
            self.assertFalse(output.exists())

    def test_unpaced_oversized_single_chunk_is_rejected_without_busy_retry(self):
        stream, _ = self.stream()
        with self.assertRaisesRegex(AudioSourceError, "one source chunk"):
            drive_stream(
                stream,
                iter([bytes(32000)]),
                on_event=lambda event: None,
                cancel=Event(),
            )

    def test_final_native_step_rechecks_cancellation_and_drain_deadline(self):
        for reason, error in (
            ("cancel", KeyboardInterrupt),
            ("deadline", TimeoutError),
        ):
            with self.subTest(reason=reason):
                stream, adapter = self.stream()
                cancel, clock, events = Event(), SimpleNamespace(now=0.0), []
                original_step = stream.step

                def final_step():
                    returned = original_step()
                    if any(event.kind is StreamEventKind.FINAL for event in returned):
                        if reason == "cancel":
                            cancel.set()
                        else:
                            clock.now = 2.0
                    return returned

                with (
                    patch.object(stream, "step", final_step),
                    patch("whisper_runtime.cli.time.monotonic", lambda: clock.now),
                    self.assertRaises(error),
                ):
                    drive_stream(
                        stream,
                        iter([bytes(6400)]),
                        on_event=events.append,
                        cancel=cancel,
                        drain_timeout=1,
                    )
                self.assertFalse(
                    any(event.kind is StreamEventKind.FINAL for event in events)
                )
                self.assertEqual(adapter.budget.lease_count, 0)
                self.assertEqual(adapter.worker.queue_depth, 0)

    def test_eof_deadline_starts_before_finish_input_exposes_eof(self):
        stream, adapter = self.stream()
        clock, events = SimpleNamespace(now=0.0), []
        finish = stream.finish_input

        def delayed_finish():
            result = finish()
            clock.now = 2.0
            return result

        with (
            patch.object(stream, "finish_input", delayed_finish),
            patch("whisper_runtime.cli.time.monotonic", lambda: clock.now),
            self.assertRaises(TimeoutError),
        ):
            drive_stream(
                stream,
                iter([bytes(6400)]),
                on_event=events.append,
                cancel=Event(),
                drain_timeout=1,
            )
        self.assertFalse(any(event.kind is StreamEventKind.FINAL for event in events))
        self.assertEqual(adapter.budget.lease_count, 0)

    def test_callback_cancellation_stops_the_rest_of_a_final_event_batch(self):
        stream, adapter = self.stream()
        cancel, events = Event(), []

        def receive(event):
            events.append(event)
            if event.kind is StreamEventKind.COMMIT:
                cancel.set()

        with self.assertRaises(KeyboardInterrupt):
            drive_stream(stream, iter([bytes(6400)]), on_event=receive, cancel=cancel)
        self.assertFalse(any(event.kind is StreamEventKind.FINAL for event in events))
        self.assertEqual(adapter.budget.lease_count, 0)

    def test_final_callback_cannot_bypass_the_last_success_checkpoint(self):
        for reason, error in (
            ("cancel", KeyboardInterrupt),
            ("deadline", TimeoutError),
        ):
            with self.subTest(reason=reason):
                stream, adapter = self.stream()
                cancel, clock = Event(), SimpleNamespace(now=0.0)

                def receive(event):
                    if event.kind is StreamEventKind.FINAL:
                        if reason == "cancel":
                            cancel.set()
                        else:
                            clock.now = 2.0

                with (
                    patch("whisper_runtime.cli.time.monotonic", lambda: clock.now),
                    self.assertRaises(error),
                ):
                    drive_stream(
                        stream,
                        iter([bytes(6400)]),
                        on_event=receive,
                        cancel=cancel,
                        drain_timeout=1,
                    )
                self.assertEqual(adapter.budget.lease_count, 0)

    def test_local_final_is_not_announced_or_exported_before_driver_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = (Path(directory) / name for name in ("audio.pcm", "out.txt"))
            path.write_bytes(bytes(6400))
            stream, adapter = self.stream()
            close, calls = stream.close, 0

            def failed_cleanup():
                nonlocal calls
                calls += 1
                close()
                if calls == 1:
                    raise RuntimeError("driver cleanup failed")

            with (
                patch(
                    "whisper_runtime.native_setup.create_stream", return_value=stream
                ),
                patch.object(stream, "close", failed_cleanup),
                redirect_stdout(io.StringIO()) as stdout,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            str(path),
                            "--setup-manifest",
                            "m",
                            "--model",
                            "m",
                            "--txt",
                            str(output),
                        ]
                    ),
                    1,
                )
            self.assertNotIn("[final] complete", stdout.getvalue())
            self.assertFalse(output.exists())
            self.assertEqual(adapter.budget.lease_count, 0)


if __name__ == "__main__":
    unittest.main()
