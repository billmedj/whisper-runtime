"""Replay the parity criteria without a GPU or native backend."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from infra import modal_alignment_features as experiment


class AlignmentFeaturesExperimentTests(unittest.TestCase):
    def cells(self):
        result = []
        for index in range(8):
            arms = {}
            for name in ("baseline", "reuse"):
                arms[name] = dict(
                    closed=True,
                    capacity_restored=True,
                    alignment=dict(
                        native=dict(
                            window_id=f"{index}:{name}",
                            text="hello",
                            metadata=dict(tokens=[1]),
                        ),
                        words=[
                            dict(
                                text="hello",
                                span=dict(start_ms=0, end_ms=500),
                                tokens=[1],
                            )
                        ],
                    ),
                    encoder_calls=[
                        dict(phase="decode", input_frames=3000, use_sdpa=None)
                    ]
                    + (
                        [dict(phase="alignment", input_frames=3000, use_sdpa=False)]
                        if name == "baseline"
                        else []
                    ),
                )
            result.append(dict(id=str(index), arms=arms))
        return result

    def test_exact_outputs_and_encoder_work_are_separate(self):
        summary = experiment.summarize(self.cells())
        self.assertTrue(summary["all_native_equal"])
        self.assertTrue(summary["all_words_equal"])
        self.assertTrue(summary["all_expected_encoder_counts"])
        self.assertTrue(summary["all_expected_encoder_paths"])
        self.assertEqual(summary["baseline_encoder_calls"], 16)
        self.assertEqual(summary["reuse_encoder_calls"], 8)
        cells = self.cells()
        cells[0]["arms"]["reuse"]["alignment"]["words"][0]["span"]["end_ms"] += 20
        self.assertFalse(experiment.summarize(cells)["all_words_equal"])
        self.assertTrue(experiment.summarize(cells)["all_native_equal"])
        cells[0]["arms"]["reuse"]["encoder_calls"][0]["input_frames"] = 1500
        self.assertFalse(experiment.summarize(cells)["all_expected_encoder_paths"])

    def test_backend_and_model_identity_are_validated(self):
        cells = self.cells()
        for index, arm in enumerate(
            (arm for cell in cells for arm in cell["arms"].values()), 1
        ):
            arm["call_index"] = index
        expected = dict(digest="snapshot")
        inputs = [dict(id=cell["id"]) for cell in cells]
        record = dict(
            source=dict(snapshot=expected),
            inputs=inputs,
            backend=dict(
                patched_tree=experiment.PATCHED_TREE,
                patch_sha256=experiment.PATCH_SHA,
                base_tree=experiment.BASE_TREE,
                base_commit=experiment.BASE_COMMIT,
            ),
            model=dict(
                initial_sha256="registered", final_sha256="registered", unchanged=True
            ),
            status="completed",
            cells=cells,
            native_window_count=16,
            summary=experiment.summarize(cells),
            qualified=False,
        )
        corpus = SimpleNamespace(
            read_registration=lambda: dict(model=dict(model_state_sha256="registered"))
        )
        with (
            patch.object(experiment, "input_records", return_value=inputs),
            patch.object(experiment, "_corpus", return_value=corpus),
        ):
            experiment.validate_record(record, expected)
            for section, field, value in (
                ("backend", "base_tree", "wrong"),
                ("backend", "base_commit", "wrong"),
                ("model", "initial_sha256", "wrong"),
                ("model", "final_sha256", "changed"),
                ("model", "unchanged", False),
            ):
                changed = copy.deepcopy(record)
                changed[section][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    experiment.validate_record(changed, expected)

    def test_empty_text_can_skip_alignment_on_both_arms(self):
        cells = self.cells()
        for arm in cells[0]["arms"].values():
            arm["alignment"]["native"]["text"] = ""
            arm["alignment"]["words"] = []
            arm["encoder_calls"] = [dict(phase="decode")]
        self.assertTrue(experiment.summarize(cells)["all_expected_encoder_counts"])

    def test_incomplete_or_unclosed_pairs_fail(self):
        cells = self.cells()
        for broken in (cells[:-1], copy.deepcopy(cells)):
            if len(broken) == 8:
                broken[0]["arms"]["reuse"]["capacity_restored"] = False
            with self.assertRaises(ValueError):
                experiment.summarize(broken)

    def test_input_windows_are_bounded_without_new_downloads(self):
        if not (experiment.ROOT / "artifacts/resolution-inputs-v1").exists():
            self.skipTest("optional local PCM inputs absent")
        items = experiment.input_records()
        self.assertEqual(len(items), 8)
        self.assertEqual(len({item["id"] for item in items}), 8)
        self.assertEqual(items[-2]["sample_count"], 32000)
        self.assertTrue(
            all(
                item["sample_count"] == (item["end_ms"] - item["start_ms"]) * 16
                for item in items
            )
        )
        self.assertNotEqual(items[-1]["pcm_sha256"], items[4]["pcm_sha256"])

    def test_patch_identity_is_checked_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / experiment.PATCH
            target.parent.mkdir(parents=True)
            target.write_bytes(b"different patch")
            with patch.object(
                experiment.subprocess, "check_output", return_value=experiment.BASE_TREE
            ) as git:
                with self.assertRaises(ValueError):
                    experiment.apply_backend_patch(root, root)
                self.assertEqual(git.call_count, 1)


if __name__ == "__main__":
    unittest.main()
