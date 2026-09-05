"""Dependency-free checks for the optional timestamp publication smoke mode."""

from __future__ import annotations

import io
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import run_native_example as runner

from examples import native_transaction as example
from whisper_runtime import (
    AudioSpan,
    Budget,
    ImmediateFence,
    ModelSnapshot,
    ResourceVector,
    Worker,
)
from whisper_runtime.adapters import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWhisperAdapter,
    NativeWindowResult,
    NativeWindowRun,
)


class SegmentExampleArgumentsTests(unittest.TestCase):
    def test_runner_preserves_default_and_stream_arguments(self) -> None:
        for arguments, expected_text in (
            ([], True),
            (["--stream-preview-ms", "500", "--stream-chunk-ms", "100"], True),
            (["--segment-publication-check"], False),
        ):
            with self.subTest(arguments=arguments):
                paths = runner.SetupPaths.from_root(Path("fixture-setup"))
                setup = SimpleNamespace(paths=paths, python=paths.root / "python")
                with (
                    patch.object(
                        sys, "argv", ["runner", "--root", str(paths.root), *arguments]
                    ),
                    patch.object(runner, "load_validated_setup", return_value=setup),
                    patch.object(Path, "is_file", return_value=True),
                    patch.object(
                        runner.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0),
                    ) as run,
                ):
                    self.assertEqual(runner.main(), 0)
                command = run.call_args.args[0]
                self.assertEqual("--expected-text" in command, expected_text)
                self.assertNotIn("--allow-model-download", command)
                for argument in arguments:
                    self.assertIn(argument, command)
                if not arguments:
                    self.assertNotIn("--stream-preview-ms", command)
                    self.assertNotIn("--segment-publication-check", command)

    def test_modes_are_mutually_exclusive_in_runner_and_example(self) -> None:
        for module, required in (
            (runner, []),
            (
                example,
                [
                    "--manifest",
                    "manifest.json",
                    "--audio",
                    "jfk.flac",
                    "--model-cache",
                    "models",
                ],
            ),
        ):
            with self.subTest(module=module.__name__):
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "example",
                            *required,
                            "--stream-preview-ms",
                            "500",
                            "--segment-publication-check",
                        ],
                    ),
                    patch("sys.stderr", new_callable=io.StringIO),
                    self.assertRaises(SystemExit) as raised,
                ):
                    module.parse_args()
                self.assertEqual(raised.exception.code, 2)

    def test_example_defaults_remain_untimed_non_streaming_and_no_download(
        self,
    ) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "example",
                "--manifest",
                "manifest.json",
                "--audio",
                "jfk.flac",
                "--model-cache",
                "models",
            ],
        ):
            args = example.parse_args()
        self.assertFalse(args.segment_publication_check)
        self.assertIsNone(args.stream_preview_ms)
        self.assertFalse(args.allow_model_download)


class SegmentPublicationExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = ModelSnapshot("test", "1", "reference", "sha256:test")
        capacity = ResourceVector(memory_bytes=100, compute_units=1, stream_slots=1)
        self.worker = Worker("test", self.identity, Budget(capacity), queue_capacity=1)
        self.segments = (
            NativeTimestampSegment(AudioSpan(1_000, 2_000), " Hello", (1,)),
            NativeTimestampSegment(AudioSpan(2_000, 4_000), " world.", (2,)),
        )
        self.complete = True
        self.change_suffix_tokens = False
        self.adapter = Mock(spec=NativeWhisperAdapter)
        self.adapter.model_identity = self.identity
        self.adapter.decode_window.side_effect = self.decode
        self.adapter.start_window.side_effect = self.start
        self.handles: list[MagicMock] = []

    def result(
        self, arguments: dict[str, object], span: AudioSpan | None = None
    ) -> NativeWindowResult:
        selected = (
            self.segments
            if span is None
            else tuple(
                segment
                for segment in self.segments
                if segment.span.start_ms >= span.start_ms
                and segment.span.end_ms <= span.end_ms
            )
        )
        output = span or AudioSpan(arguments["start_ms"], arguments["end_ms"])
        tokens = (
            (9,)
            if self.change_suffix_tokens and arguments["window_id"] == "segment-suffix"
            else (10, 1, 20, 20, 2, 30)
        )
        metadata = NativeDecodeMetadata(
            language="en",
            tokens=tokens,
            segments=self.segments,
            timestamps_complete=self.complete,
        )
        return NativeWindowResult(
            window_id=arguments["window_id"],
            text="".join(segment.text for segment in selected).strip(),
            start_ms=output.start_ms,
            end_ms=output.end_ms,
            analysis_span=AudioSpan(arguments["start_ms"], arguments["end_ms"]),
            metadata=metadata,
        )

    def start(self, **arguments: object) -> MagicMock:
        transaction = self.worker.prepare(
            session=arguments["session"],
            request=arguments["request"],
            window_id=arguments["window_id"],
            resources=self.worker.budget.capacity,
        )
        transaction.start(ImmediateFence())

        def close(*_: object) -> bool:
            if not transaction.capacity_released:
                transaction.abort()
            return False

        handle = MagicMock(spec=NativeWindowRun)
        handle.__enter__.return_value = handle
        handle.__exit__.side_effect = close
        handle.complete = False
        handle.step.side_effect = lambda: setattr(handle, "complete", True)
        handle.prepare_result.return_value = self.result(arguments)
        handle.finish.side_effect = lambda **finish: transaction.commit(
            self.result(arguments, finish.get("publication_span")),
            committed_through_ms=finish.get("committed_through_ms"),
        )
        self.handles.append(handle)
        return handle

    def decode(self, **arguments: object) -> object:
        with self.start(**arguments) as handle:
            return handle.finish()

    def test_same_full_analysis_publishes_two_nonoverlapping_segments(self) -> None:
        mel = object()
        state, report = example.run_segment_publication_check(
            adapter=self.adapter, mel=mel, duration_ms=5_000
        )
        self.assertEqual(state.version, 2)
        self.assertEqual(state.committed_through_ms, 4_000)
        self.assertEqual(report["reassembled_text"], "Hello world.")
        self.assertEqual(
            report["publication_spans"],
            [
                {"start_ms": 1_000, "end_ms": 2_000},
                {"start_ms": 2_000, "end_ms": 4_000},
            ],
        )
        self.assertTrue(report["previous_record_unchanged"])
        self.assertTrue(report["prepared_result_reused"])
        self.assertEqual(self.adapter.decode_window.call_count, 1)
        self.assertEqual(self.adapter.start_window.call_count, 2)
        for handle in self.handles[1:]:
            self.assertEqual(handle.prepare_result.call_count, 2)
            handle.step.assert_called_once()
            handle.finish.assert_called_once()
        calls = (
            self.adapter.decode_window.call_args_list
            + self.adapter.start_window.call_args_list
        )
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertIs(call.kwargs["mel"], mel)
            self.assertIs(call.kwargs["options"], calls[0].kwargs["options"])
            self.assertFalse(call.kwargs["options"].without_timestamps)
            self.assertEqual(
                (call.kwargs["start_ms"], call.kwargs["end_ms"]), (0, 5_000)
            )
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.worker.budget.lease_count, 0)

    def test_incomplete_control_is_rejected_before_publication(self) -> None:
        self.complete = False
        with self.assertRaisesRegex(RuntimeError, "at least two complete timestamp"):
            example.run_segment_publication_check(
                adapter=self.adapter, mel=object(), duration_ms=5_000
            )
        self.assertEqual(self.adapter.decode_window.call_count, 1)

    def test_single_segment_control_is_rejected_before_publication(self) -> None:
        self.segments = self.segments[:1]
        with self.assertRaisesRegex(RuntimeError, "at least two complete timestamp"):
            example.run_segment_publication_check(
                adapter=self.adapter, mel=object(), duration_ms=5_000
            )
        self.assertEqual(self.adapter.decode_window.call_count, 1)

    def test_token_drift_fails_same_options_control_check(self) -> None:
        self.change_suffix_tokens = True
        with self.assertRaisesRegex(RuntimeError, "same-options control"):
            example.run_segment_publication_check(
                adapter=self.adapter, mel=object(), duration_ms=5_000
            )
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.worker.budget.lease_count, 0)


if __name__ == "__main__":
    unittest.main()
