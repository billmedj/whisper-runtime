"""Local guards for the short Modal continuous-transcript diagnostic."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE": "0"}):
    smoke = importlib.import_module("infra.modal_continuous_smoke")


class _Remote:
    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.result = {"status": "passed"} if result is None else result
        self.error = error

    def remote(self, *arguments: object) -> object:
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return self.result


class _Kind(str, Enum):
    PROVISIONAL = "provisional"


@dataclass
class _Event:
    sequence_number: int
    kind: _Kind


class StreamNeedsResolutionError(Exception):
    pass


class _PartialThenUnresolvedStream:
    def __init__(self) -> None:
        self.chunks = 0
        self.processed = 0

    @property
    def ready(self) -> bool:
        return self.chunks > self.processed

    def push(self, sequence: int, content: bytes) -> None:
        del sequence, content
        self.chunks += 1

    def step(self) -> tuple[_Event, ...]:
        if self.processed == 0:
            self.processed += 1
            return (_Event(1, _Kind.PROVISIONAL),)
        raise StreamNeedsResolutionError("untrusted path or backend details")

    def finish_input(self) -> None:
        raise AssertionError("the failing stream must stop before EOF")

    def __enter__(self) -> _PartialThenUnresolvedStream:
        return self

    def __exit__(self, *exc: object) -> None:
        del exc


class ModalContinuousSmokeTests(unittest.TestCase):
    def make_root(self, destination: Path) -> Path:
        (destination / "src" / "whisper_runtime").mkdir(parents=True)
        (destination / "src" / "whisper_runtime" / "sample.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (destination / "infra").mkdir()
        (destination / "infra" / "modal_continuous_smoke.py").write_text(
            "# diagnostic producer\n", encoding="utf-8"
        )
        (destination / "experiments").mkdir()
        manifest = smoke._read_registration()
        (destination / smoke.MANIFEST_PATH).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return destination

    def test_import_defines_no_remote_resources(self) -> None:
        self.assertIsNone(smoke.app)
        self.assertIsNone(smoke.run_continuous_smoke)
        self.assertIsNone(smoke.main)

    def test_registered_cell_closes_claims_and_bounds_paid_calls(self) -> None:
        manifest = smoke._read_registration()
        self.assertEqual(set(manifest["claim_boundary"].values()), {False})
        budget = manifest["paid_budget"]
        self.assertEqual(budget["maximum_gpu_function_calls"], 2)
        self.assertEqual(budget["gpu_seconds_per_call"], 600)
        self.assertEqual(budget["automatic_retries"], 0)
        self.assertEqual(budget["maximum_containers"], 1)
        self.assertEqual(budget["scaledown_window_seconds"], 2)
        self.assertEqual(budget["startup_timeout_seconds"], 300)
        cell = manifest["cell"]
        self.assertNotIn("expected_text_sha256", cell)
        self.assertEqual(len(cell["decoded_float32_fingerprint"]), 64)
        self.assertEqual(len(cell["converted_pcm_s16le_sha256"]), 64)

    def test_second_registration_is_the_final_campaign_call(self) -> None:
        manifest = json.loads(
            (smoke.ROOT / "experiments" / "modal-continuous-smoke-v2.json").read_text(
                encoding="utf-8"
            )
        )
        smoke._validate_registration(manifest)
        self.assertEqual(manifest["paid_budget"]["maximum_gpu_function_calls"], 1)
        self.assertEqual(manifest["campaign"]["this_manifest_call_ordinal"], 2)
        self.assertEqual(manifest["cell"]["duration_ms"], 33_000)
        self.assertEqual(manifest["cell"]["fixture_repetitions"], 3)
        self.assertFalse(manifest["cell"]["control_reference"]["full_33_second_decode"])

    def test_stream_resolution_failure_returns_partial_diagnostic(self) -> None:
        events, commits, steps, error = smoke._drive_stream(
            _PartialThenUnresolvedStream(), b"\x00\x00\x01\x00", chunk_bytes=2
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(commits, 0)
        self.assertEqual(steps, 1)
        self.assertEqual(
            error,
            {
                "error_class": "StreamNeedsResolutionError",
                "reason_code": "stable_prefix_not_found",
                "reason": (
                    "The bounded window reached its limit without a stable "
                    "contiguous prefix."
                ),
            },
        )
        self.assertNotIn("untrusted", json.dumps(error))

    def test_second_call_refuses_any_prior_second_receipt(self) -> None:
        manifest = json.loads(
            (smoke.ROOT / "experiments" / "modal-continuous-smoke-v2.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "artifacts" / "modal"
            receipts.mkdir(parents=True)
            (receipts / "continuous-smoke-v1-attempt-1.attempt.jsonl").write_text(
                "started\n", encoding="utf-8"
            )
            smoke._require_campaign_call_available(root, manifest)
            (receipts / "continuous-smoke-v1-attempt-2.attempt.jsonl").write_text(
                "failed or unknown\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "budget is exhausted"):
                smoke._require_campaign_call_available(root, manifest)

    def test_attempt_number_is_strictly_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(Path(temporary))
            self.assertIn("attempt-1", str(smoke._attempt_paths(root, 1)[0]))
            self.assertIn("attempt-2", str(smoke._attempt_paths(root, 2)[0]))
            for value in (0, 3):
                with self.assertRaises(ValueError):
                    smoke._attempt_paths(root, value)
            with self.assertRaises(TypeError):
                smoke._attempt_paths(root, True)

    def test_snapshot_is_deterministic_and_detects_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(Path(temporary))
            first = smoke._source_snapshot(root)
            self.assertEqual(first, smoke._source_snapshot(root))
            (root / "src" / "whisper_runtime" / "sample.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
            self.assertNotEqual(first["digest"], smoke._source_snapshot(root)["digest"])

    def test_event_checks_use_dynamic_full_window_control(self) -> None:
        events = [
            {
                "sequence_number": 1,
                "kind": "provisional",
                "segment_id": "s0",
                "revision": 1,
                "text": "hello world",
            },
            {
                "sequence_number": 2,
                "kind": "commit",
                "segment_id": "s0",
                "revision": 1,
                "committed_through_sample": 32_000,
            },
            {"sequence_number": 3, "kind": "final"},
        ]
        capacity = object()
        checks, text = smoke._evaluate_events(
            events,
            pre_eof_commit_count=1,
            input_samples=32_000,
            full_window_control_text=" hello   world ",
            reference_untimed_transcript_sha256=smoke._sha256_text("other"),
            state=SimpleNamespace(committed_through_ms=2_000),
            metrics=SimpleNamespace(
                committed_samples=32_000,
                buffered_samples=0,
                peak_buffered_samples=32_000,
            ),
            capacity=capacity,
            budget=SimpleNamespace(lease_count=0, available=capacity),
            worker=SimpleNamespace(queue_depth=0),
            max_buffer_samples=640_000,
        )
        self.assertEqual(text, "hello world")
        self.assertTrue(checks["continuous_matches_full_window_control"])
        self.assertTrue(checks["nonempty_final_transcript"])
        self.assertFalse(checks["reference_untimed_text_observed"])

    def test_paid_dispatch_is_single_use_and_writes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(Path(temporary))
            remote = _Remote()
            with self.assertRaises(RuntimeError):
                smoke._execute_local_attempt(
                    root=root,
                    attempt=1,
                    confirm_paid_gpu=False,
                    remote_function=remote,
                )
            self.assertEqual(remote.calls, [])
            output = smoke._execute_local_attempt(
                root=root,
                attempt=1,
                confirm_paid_gpu=True,
                remote_function=remote,
            )
            self.assertEqual(len(remote.calls), 1)
            self.assertEqual(json.loads(output.read_text())["status"], "passed")
            receipt = output.with_suffix(".attempt.jsonl")
            rows = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(
                [row["event"] for row in rows], ["attempt-started", "record-written"]
            )
            with self.assertRaises(FileExistsError):
                smoke._execute_local_attempt(
                    root=root,
                    attempt=1,
                    confirm_paid_gpu=True,
                    remote_function=remote,
                )
            self.assertEqual(len(remote.calls), 1)

    def test_remote_failure_consumes_attempt_without_a_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(Path(temporary))
            remote = _Remote(error=RuntimeError("remote failure"))
            with self.assertRaises(RuntimeError):
                smoke._execute_local_attempt(
                    root=root,
                    attempt=2,
                    confirm_paid_gpu=True,
                    remote_function=remote,
                )
            output, receipt = smoke._attempt_paths(root, 2)
            self.assertFalse(output.exists())
            rows = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(rows[-1]["event"], "attempt-failed")
            self.assertNotIn("error_message", rows[-1])


if __name__ == "__main__":
    unittest.main()
