"""Small offline checks; archived inputs and existing scripted tests only."""

import json
import subprocess
import sys
import textwrap
import unittest

from tools import verify_evidence_reuse as replay


class FixtureImportTests(unittest.TestCase):
    def test_import_and_fixture_replay_preserve_tools_module_resolution(self):
        script = textwrap.dedent("""
            import importlib.util
            import sys
            from pathlib import Path
            from types import ModuleType

            root = Path.cwd()
            sys.path.insert(0, str(root / "tools"))
            expected = (root / "tools/test_pcm_websocket.py").resolve()
            def resolved():
                return Path(importlib.util.find_spec("test_pcm_websocket").origin).resolve()
            assert resolved() == expected
            from tools import verify_evidence_reuse as replay
            assert resolved() == expected
            sentinel = ModuleType("test_continuity_witness")
            sys.modules["test_continuity_witness"] = sentinel
            before = sys.path[:]
            replay.clean_fixture_controls()
            replay.fixture_checks()
            assert sys.path == before
            assert sys.modules["test_continuity_witness"] is sentinel
            assert resolved() == expected
        """)
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=replay.ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(
    (replay.ROOT / "artifacts/speech-corpus-v1").is_dir(), "cached corpus unavailable"
)
class EvidenceReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = replay.verify()

    def test_duplicate_is_idempotent_not_another_observation(self):
        duplicate = self.report["duplicate"]
        self.assertTrue(duplicate["identical_result"])
        self.assertEqual(duplicate["additional_acoustic_observations"], 0)
        self.assertEqual(duplicate["evaluations"], 2)
        self.assertIn("overlap_cuts_candidate_word", duplicate["reasons"])
        self.assertEqual(
            duplicate["interpretation"], "pure_function_diagnostic_idempotence_only"
        )
        self.assertFalse(duplicate["production_deduplication_demonstrated"])

    def test_mutations_have_distinct_inputs_and_never_clear_refusal(self):
        self.assertEqual(
            self.report["archive_mutation_claim"],
            "preserved_refusals_not_individual_gate_causality",
        )
        rows = self.report["counterfactual_mutations"]
        hashes = {self.report["duplicate"]["input_sha256"]}
        for name, row in rows.items():
            with self.subTest(name=name):
                self.assertTrue(row["reasons"])
                self.assertNotIn(row["input_sha256"], hashes)
                hashes.add(row["input_sha256"])
        self.assertIn("candidate_pcm_mismatch", rows["changed_pcm"]["reasons"])
        self.assertIn(
            "candidate_pcm_mismatch",
            rows["extended_window_without_fresh_observation"]["reasons"],
        )
        self.assertIn(
            "overlap_cuts_candidate_word",
            rows["partial_word_still_crosses_overlap"]["reasons"],
        )

    def test_scope_and_existing_positive_reuse_are_explicit(self):
        self.assertEqual(
            set(self.report["existing_scripted_fixture_checks"].values()), {"passed"}
        )
        for key in (
            "shadow_planner_implemented",
            "next_work_selection_validated",
            "cache_implemented",
            "native_inference_run",
            "publication_authorized",
            "audio_retirement_authorized",
            "recognition_quality_claim",
        ):
            self.assertIs(self.report[key], False)
        self.assertNotIn(json.dumps(str(replay.ROOT))[1:-1], json.dumps(self.report))

    def test_clean_controls_isolate_specific_new_refusals(self):
        controls = self.report["clean_synthetic_controls"]
        self.assertEqual(controls["baseline"]["reasons"], ())
        self.assertTrue(controls["correspondence_only"])
        expected = {
            "changed_pcm": ("joint_pcm_mismatch", "candidate_pcm_mismatch"),
            "changed_model_identity": (
                "analysis_identity_mismatch",
                "published_model_mismatch",
            ),
            "changed_decode_options": ("analysis_identity_mismatch",),
            "changed_window_origin": ("incompatible_observation_windows",),
        }
        self.assertEqual(controls["mutations"].keys(), expected.keys())
        for label, reasons in expected.items():
            with self.subTest(label=label):
                row = controls["mutations"][label]
                self.assertEqual(row["newly_present_reasons"], reasons)
                self.assertNotEqual(
                    row["input_sha256"], controls["baseline"]["input_sha256"]
                )


if __name__ == "__main__":
    unittest.main()
