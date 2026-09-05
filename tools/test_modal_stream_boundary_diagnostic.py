"""Local guards for the bounded Modal stream-boundary diagnostic."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class _TraceStream:
    def __init__(self) -> None:
        self.last_trace = _Trace(1)


class _Remote:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def remote(self, *arguments: object) -> dict[str, object]:
        self.calls.append(arguments)
        return {"status": "completed"}


class ModalStreamBoundaryDiagnosticTests(unittest.TestCase):
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
