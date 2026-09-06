"""Validate the lane experiment without loading CUDA or Modal."""

import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infra import modal_cuda_lane as experiment


class CudaLaneExperimentTests(unittest.TestCase):
    def test_archived_blocked_record_has_exact_post_release_plateaus(self):
        record = json.loads(
            (
                experiment.ROOT
                / "evidence/modal-t4-tiny-en-cuda-lane-blocked-2026-09-06.json"
            ).read_bytes()
        )
        summary = experiment.summarize(record["cells"], order="blocked-v2")
        self.assertEqual(summary, record["summary"])
        self.assertTrue(summary["exact_alignment_equal"])
        self.assertTrue(summary["consecutive_reused_allocation_flat"])
        self.assertEqual(summary["consecutive_reused_allocated_deltas_bytes"], [0] * 6)
        self.assertTrue(record["model"]["unchanged"])
        self.assertEqual(summary["reused_stream_count"], 1)
        self.assertEqual(summary["fresh_stream_count"], 4)
        for cell in record["cells"][4:8]:
            self.assertEqual(
                cell["after_handle_release"]["allocated_bytes"]
                - cell["before"]["allocated_bytes"],
                8519680,
            )

    def test_archived_alternating_record_replays_exactly(self):
        record = json.loads(
            (
                experiment.ROOT / "evidence/modal-t4-tiny-en-cuda-lane-2026-09-06.json"
            ).read_bytes()
        )
        self.assertEqual(experiment.summarize(record["cells"]), record["summary"])
        self.assertTrue(record["summary"]["exact_alignment_equal"])
        self.assertFalse(record["summary"]["warm_reused_allocation_flat"])
        distinct = len(
            {stream for cell in record["cells"] for stream in cell["streams"]}
        )
        self.assertEqual(distinct, 7)
        self.assertEqual(
            record["after_handle_release"]["allocated_bytes"]
            - record["cells"][0]["before"]["allocated_bytes"],
            distinct * 8519680,
        )

    def test_blocked_schedule_separates_consecutive_calls_from_switches(self):
        cells = self.cells()
        released = 100
        for index, (cell, arm) in enumerate(
            zip(cells, experiment.ORDERS["blocked-v2"])
        ):
            cell["arm"] = arm
            switch = index == 0 or cells[index - 1]["arm"] != arm
            cell["after_close"]["allocated_bytes"] = cell["before"][
                "allocated_bytes"
            ] + (8 if arm == "fresh" else 1 if switch else 0)
            released += 8 if arm == "fresh" or index == 0 else 0
            cell["after_handle_release"] = dict(allocated_bytes=released)
        summary = experiment.summarize(cells, order="blocked-v2")
        self.assertFalse(summary["warm_reused_allocation_flat"])
        self.assertTrue(summary["consecutive_reused_allocation_flat"])
        cells[2]["after_handle_release"]["allocated_bytes"] += 1
        self.assertFalse(
            experiment.summarize(cells, order="blocked-v2")[
                "consecutive_reused_allocation_flat"
            ]
        )

    def cells(self):
        cells = []
        allocated = 100
        for index, arm in enumerate(["reused", "fresh"] * experiment.ROUNDS, 1):
            delta = 8 if arm == "fresh" or index == 1 else 0
            cells.append(
                dict(
                    call_index=index,
                    arm=arm,
                    before=dict(allocated_bytes=allocated),
                    after_close=dict(allocated_bytes=allocated + delta),
                    streams=[1 if arm == "reused" else index],
                    alignment=dict(
                        native=dict(window_id=f"lane:{index}", text="hello"),
                        words=[dict(start=0, end=500)],
                    ),
                    encoder_frames=[3000, 3000],
                    closed=True,
                    capacity_restored=True,
                    stale_cancel_changed_state=False,
                )
            )
            allocated += delta
        return cells

    def test_memory_deltas_do_not_confuse_reserve_with_live_allocation(self):
        summary = experiment.summarize(self.cells())
        self.assertTrue(summary["exact_alignment_equal"])
        self.assertTrue(summary["warm_reused_allocation_flat"])
        self.assertEqual(summary["reused_stream_count"], 1)
        self.assertEqual(summary["fresh_stream_count"], 6)
        self.assertEqual(
            summary["post_close_allocated_delta_bytes"]["reused"], [8, 0, 0, 0, 0, 0]
        )
        self.assertEqual(summary["post_close_allocated_delta_bytes"]["fresh"], [8] * 6)

    def test_changed_timing_and_memory_are_reported_not_hidden(self):
        cells = self.cells()
        cells[-1]["alignment"]["words"][0]["end"] = 501
        cells[2]["after_close"]["allocated_bytes"] += 1
        summary = experiment.summarize(cells)
        self.assertFalse(summary["exact_alignment_equal"])
        self.assertFalse(summary["warm_reused_allocation_flat"])

    def test_incomplete_lifecycle_or_order_cannot_pass(self):
        for field, value in (
            ("closed", False),
            ("capacity_restored", False),
            ("stale_cancel_changed_state", True),
            ("streams", []),
            ("call_index", 99),
            ("arm", "fresh"),
        ):
            cells = copy.deepcopy(self.cells())
            cells[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                experiment.summarize(cells)
        with self.assertRaises(ValueError):
            experiment.summarize(self.cells()[:-1])

    def test_no_gpu_call_without_explicit_flag(self):
        with patch.object(experiment.shared, "resources") as resources:
            with self.assertRaises(ValueError):
                experiment.run(replay_id="not-started")
            resources.assert_not_called()

    def test_default_validator_preserves_archived_blocked_record(self):
        record = json.loads(
            (
                experiment.ROOT
                / "evidence/modal-t4-tiny-en-cuda-lane-blocked-2026-09-06.json"
            ).read_bytes()
        )
        expected = record["source"]["snapshot"]
        experiment.validate_record(record, expected)
        for key, value in (("summary", {}), ("source", {"snapshot": {}})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.validate_record({**record, key: value}, expected)

    def test_launcher_uses_selected_producer_and_preserves_failed_validation_payload(
        self,
    ):
        for validation_fails in (False, True):
            with (
                self.subTest(validation_fails=validation_fails),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "infra/example_producer.py"
                source.parent.mkdir()
                source.write_bytes(b"# bound producer\n")
                expected = {"digest": "test-snapshot"}
                record = {"status": "completed", "source": {"snapshot": expected}}
                backend = SimpleNamespace(
                    _transport_probe_payload=Mock(return_value=b"probe"),
                    _write_bytes_exclusive=lambda path, data: path.write_bytes(data),
                    _decode_worker_record=Mock(return_value=record),
                )
                corpus = SimpleNamespace(
                    b=backend,
                    c=SimpleNamespace(
                        _write_json_exclusive=lambda path, value: path.write_text(
                            json.dumps(value), encoding="utf-8"
                        )
                    ),
                )
                producer = SimpleNamespace(
                    __name__="infra.example_producer",
                    PRODUCER="infra/example_producer.py",
                    CLAIMS={"benchmark": False},
                    snapshot=Mock(return_value=expected),
                    _corpus=Mock(return_value=corpus),
                    validate_record=Mock(
                        side_effect=ValueError("invalid evidence")
                        if validation_fails
                        else None
                    ),
                )
                app = SimpleNamespace(
                    app_id="ap-local-test", run=Mock(return_value=nullcontext())
                )
                echo = SimpleNamespace(remote=Mock(return_value=b"probe"))
                execute = SimpleNamespace(remote=Mock(return_value=b"raw evidence"))
                with patch.object(
                    experiment.shared, "resources", return_value=(app, echo, execute)
                ) as resources:
                    arguments = dict(
                        replay_id="selected-producer",
                        confirm_paid_gpu=True,
                        root=root,
                        producer=producer,
                    )
                    if validation_fails:
                        with self.assertRaises(ValueError):
                            experiment.run(**arguments)
                    else:
                        output = experiment.run(**arguments)
                        self.assertEqual(
                            json.loads(output.read_bytes())["app_id"], "ap-local-test"
                        )
                    resources.assert_called_once_with(
                        expected, root, worker_module="infra.example_producer"
                    )
                    execute.remote.assert_called_once_with()
                    producer.snapshot.assert_called_once_with(root)
                    producer._corpus.assert_called_once_with()
                    producer.validate_record.assert_called_once_with(record, expected)
                    backend._decode_worker_record.assert_called_once_with(
                        b"raw evidence",
                        expected_snapshot=expected,
                        registration_sha256=experiment.shared._sha(source.read_bytes()),
                        manifest={"claim_boundary": producer.CLAIMS},
                    )
                    output, receipt, raw = experiment.shared._paths(
                        root, "selected-producer", False
                    )
                    self.assertEqual(raw.read_bytes(), b"raw evidence")
                    self.assertEqual(output.exists(), not validation_fails)
                    entries = [
                        json.loads(line) for line in receipt.read_text().splitlines()
                    ]
                    self.assertEqual(
                        entries[-1]["event"],
                        "attempt-failed" if validation_fails else "record-written",
                    )
                    with self.assertRaises(FileExistsError):
                        experiment.run(**arguments)
                    execute.remote.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
