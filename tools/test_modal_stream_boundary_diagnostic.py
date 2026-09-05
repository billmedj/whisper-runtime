"""Local guards for the bounded Modal stream-boundary diagnostic."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_runtime.adapters.native_stream import StreamEventKind, TranscriptEvent

with patch.dict(
    os.environ,
    {
        "WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE": "0",
        "WHISPER_MODAL_ENABLE_STREAM_BOUNDARY_DIAGNOSTIC": "0",
    },
):
    diagnostic = importlib.import_module("infra.modal_stream_boundary_diagnostic")


@dataclass(frozen=True)
class _Trace:
    decode_index: int


@dataclass(frozen=True)
class _RuntimeTrace:
    decode_index: int
    action: StreamEventKind


class _TraceStream:
    def __init__(self) -> None:
        self.last_trace = _Trace(1)


class _Call:
    object_id = "fc-test123"

    def __init__(self, receipt: Path, manifest: dict[str, object]) -> None:
        self.receipt = receipt
        self.manifest = manifest
        self.arguments: tuple[object, ...] | None = None

    def get(self) -> str:
        entries = [json.loads(line) for line in self.receipt.read_text().splitlines()]
        if [entry["event"] for entry in entries] != [
            "attempt-started",
            "call-dispatched",
        ]:
            raise AssertionError("the call ID was not persisted before retrieval")
        assert self.arguments is not None
        snapshot, registration_sha256 = self.arguments
        return json.dumps(
            {
                "schema_version": "1-diagnostic",
                "status": "completed",
                "claim_boundary": self.manifest["claim_boundary"],
                "source": {
                    "snapshot": snapshot,
                    "registration_sha256": registration_sha256,
                },
            }
        )


class _FailingCall(_Call):
    def get(self) -> str:
        super().get()
        raise RuntimeError("local retrieval failed")


class _Remote:
    def __init__(self, call: _Call) -> None:
        self.call = call
        self.calls: list[tuple[object, ...]] = []

    def spawn(self, *arguments: object) -> _Call:
        self.calls.append(arguments)
        self.call.arguments = arguments
        return self.call


class ModalStreamBoundaryDiagnosticTests(unittest.TestCase):
    def make_root(self, root: Path) -> dict[str, object]:
        (root / "src" / "whisper_runtime").mkdir(parents=True)
        (root / "src" / "whisper_runtime" / "sample.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        for relative in (
            diagnostic.PRODUCER_PATH,
            diagnostic.COMMON_PRODUCER_PATH,
            diagnostic.MANIFEST_PATH,
        ):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((diagnostic.ROOT / relative).read_bytes())
        return diagnostic._read_registration(root)

    def test_registration_fixes_one_unpaced_four_cell_call(self) -> None:
        manifest = diagnostic._read_registration()
        self.assertEqual(manifest["replay_pacing"], "unpaced-source-time")
        self.assertEqual(manifest["rng_seed"], 7)
        self.assertEqual(manifest["paid_budget"]["maximum_gpu_function_calls"], 1)
        self.assertEqual(manifest["paid_budget"]["automatic_retries"], 0)
        self.assertEqual(len(manifest["cells"]), 4)
        self.assertIn(
            diagnostic.COMMON_PRODUCER_PATH,
            manifest["source_policy"]["snapshot_paths"],
        )
        self.assertEqual(set(manifest["claim_boundary"].values()), {False})

    def test_attempt_guard_allows_only_one_paid_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, receipt = diagnostic._attempt_paths(root, 1)
            self.assertIn("attempt-1", output.name)
            self.assertIn("attempt-1", receipt.name)
            for value in (0, 2):
                with self.assertRaises(ValueError):
                    diagnostic._attempt_paths(root, value)
            with self.assertRaises(TypeError):
                diagnostic._attempt_paths(root, True)

    def test_snapshot_includes_the_reused_image_helper(self) -> None:
        snapshot = diagnostic._source_snapshot()
        paths = {entry["path"] for entry in snapshot["files"]}
        self.assertIn(diagnostic.COMMON_PRODUCER_PATH, paths)
        self.assertIn(diagnostic.PRODUCER_PATH, paths)
        self.assertIn(diagnostic.MANIFEST_PATH, paths)

    def test_trace_capture_is_ordered_and_immutable(self) -> None:
        traces: list[dict[str, object]] = []
        stream = _TraceStream()
        diagnostic._capture_trace(stream, traces)
        diagnostic._capture_trace(stream, traces)
        self.assertEqual(traces, [{"decode_index": 1}])
        stream.last_trace = _Trace(2)
        diagnostic._capture_trace(stream, traces)
        self.assertEqual([trace["decode_index"] for trace in traces], [1, 2])

    def test_str_enum_trace_has_a_strict_json_round_trip(self) -> None:
        payload = diagnostic._plain(
            {
                "trace": _RuntimeTrace(1, StreamEventKind.COMMIT),
                "event": TranscriptEvent(1, StreamEventKind.FINAL, session_version=0),
            }
        )
        self.assertIs(type(payload["trace"]["action"]), str)
        self.assertIs(type(payload["event"]["kind"]), str)
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        encoded = diagnostic._encode_worker_record(payload)
        self.assertIs(type(encoded), str)
        self.assertEqual(json.loads(encoded), payload)
        with self.assertRaisesRegex(ValueError, "finite"):
            diagnostic._encode_worker_record({"value": float("nan")})

    def test_worker_json_rejects_invalid_or_wrong_attempt_records(self) -> None:
        manifest = diagnostic._read_registration()
        snapshot = diagnostic._source_snapshot()
        registration_sha256 = diagnostic._sha256_file(
            diagnostic.ROOT / diagnostic.MANIFEST_PATH
        )
        valid = {
            "schema_version": "1-diagnostic",
            "claim_boundary": manifest["claim_boundary"],
            "source": {
                "snapshot": snapshot,
                "registration_sha256": registration_sha256,
            },
        }
        decoded = diagnostic._decode_worker_record(
            diagnostic._encode_worker_record(valid),
            expected_snapshot=snapshot,
            registration_sha256=registration_sha256,
            manifest=manifest,
        )
        self.assertEqual(decoded, valid)
        invalid_payloads = (
            "[]",
            '{"value":NaN}',
            diagnostic._encode_worker_record(
                {
                    **valid,
                    "source": {**valid["source"], "snapshot": {"digest": "0" * 64}},
                }
            ),
            diagnostic._encode_worker_record(
                {
                    **valid,
                    "source": {
                        **valid["source"],
                        "registration_sha256": "0" * 64,
                    },
                }
            ),
        )
        for payload in invalid_payloads:
            with (
                self.subTest(payload=payload[:30]),
                self.assertRaises((TypeError, ValueError)),
            ):
                diagnostic._decode_worker_record(
                    payload,
                    expected_snapshot=snapshot,
                    registration_sha256=registration_sha256,
                    manifest=manifest,
                )

    def test_spawn_persists_call_id_before_validated_result_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_root(root)
            _, receipt = diagnostic._attempt_paths(root, 1)
            call = _Call(receipt, manifest)
            remote = _Remote(call)
            output = diagnostic._execute_local_attempt(
                root=root,
                attempt=1,
                confirm_paid_gpu=True,
                remote_function=remote,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "completed")
            entries = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(
                [entry["event"] for entry in entries],
                ["attempt-started", "call-dispatched", "record-written"],
            )
            self.assertEqual(entries[1]["function_call_id"], call.object_id)
            self.assertEqual(len(remote.calls), 1)

    def test_failed_get_preserves_call_id_and_blocks_another_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_root(root)
            _, receipt = diagnostic._attempt_paths(root, 1)
            call = _FailingCall(receipt, manifest)
            remote = _Remote(call)
            with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
                diagnostic._execute_local_attempt(
                    root=root,
                    attempt=1,
                    confirm_paid_gpu=True,
                    remote_function=remote,
                )
            entries = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(
                [entry["event"] for entry in entries],
                ["attempt-started", "call-dispatched", "attempt-failed"],
            )
            self.assertEqual(entries[1]["function_call_id"], call.object_id)
            second = _Remote(_Call(receipt, manifest))
            with self.assertRaises(FileExistsError):
                diagnostic._execute_local_attempt(
                    root=root,
                    attempt=1,
                    confirm_paid_gpu=True,
                    remote_function=second,
                )
            self.assertEqual(second.calls, [])

    def _checks(
        self,
        events: list[dict[str, object]],
        *,
        committed: int,
        buffered: int,
        retained: int,
        state_ms: int | None,
        error: dict[str, object] | None = None,
    ) -> dict[str, bool]:
        accepted = (
            committed + buffered if retained == committed else retained + buffered
        )
        metrics = SimpleNamespace(
            accepted_samples=accepted,
            committed_samples=committed,
            buffered_samples=buffered,
            peak_buffered_samples=accepted,
            decode_count=1,
        )
        capacity = object()
        return diagnostic._event_checks(
            events,
            [{"decode_index": 1}],
            accepted_samples=accepted,
            total_samples=accepted,
            state=SimpleNamespace(committed_through_ms=state_ms),
            metrics=metrics,
            budget=SimpleNamespace(lease_count=0, available=capacity),
            worker=SimpleNamespace(queue_depth=0),
            capacity=capacity,
            retained_from_sample=retained,
            max_buffer_samples=max(accepted, 1),
            error_record=error,
        )

    def test_final_only_cannot_claim_a_committed_watermark(self) -> None:
        checks = self._checks(
            [{"sequence_number": 1, "kind": "final"}],
            committed=160,
            buffered=0,
            retained=160,
            state_ms=10,
        )
        self.assertFalse(checks["commit_watermark_matches_runtime"])

    def test_zero_commit_policy_resolution_is_accounted(self) -> None:
        events = [
            {
                "sequence_number": 1,
                "kind": "provisional",
                "segment_id": "s",
                "revision": 1,
                "start_sample": 0,
                "end_sample": 160,
            }
        ]
        checks = self._checks(
            events,
            committed=0,
            buffered=160,
            retained=0,
            state_ms=None,
            error={"category": "policy_resolution"},
        )
        required = diagnostic._read_registration()["outcomes"]["lifecycle_required"]
        self.assertTrue(all(checks[name] for name in required))
        self.assertFalse(checks["completion_required"])

    def test_commit_must_reference_the_published_revision_and_span(self) -> None:
        events = [
            {
                "sequence_number": 1,
                "kind": "provisional",
                "segment_id": "s",
                "revision": 1,
                "start_sample": 0,
                "end_sample": 160,
            },
            {
                "sequence_number": 2,
                "kind": "commit",
                "segment_id": "s",
                "revision": 1,
                "start_sample": 0,
                "end_sample": 160,
                "committed_through_sample": 160,
            },
            {"sequence_number": 3, "kind": "final"},
        ]
        checks = self._checks(
            events,
            committed=160,
            buffered=0,
            retained=160,
            state_ms=10,
        )
        required = diagnostic._read_registration()["outcomes"]["lifecycle_required"]
        self.assertTrue(all(checks[name] for name in required))
        events[1]["start_sample"] = 1
        checks = self._checks(
            events,
            committed=160,
            buffered=0,
            retained=160,
            state_ms=10,
        )
        self.assertFalse(checks["immutable_committed_revisions"])


if __name__ == "__main__":
    unittest.main()
