"""Validate the lane experiment without loading CUDA or Modal."""

import copy
import unittest
from unittest.mock import patch

from infra import modal_cuda_lane as experiment


class CudaLaneExperimentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
