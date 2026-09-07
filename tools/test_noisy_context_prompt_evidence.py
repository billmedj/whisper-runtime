"""Replay the T4 observations and the explicitly bounded report correction."""

import copy
import json
import unittest
from unittest.mock import patch

from infra import modal_noisy_context_prompt as p
from tools.verify_noisy_context_prompt import KNOWN_CONTROL_ERROR_SHA, verify


class NoisyContextPromptEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(
            (
                p.ROOT
                / "evidence/modal-t4-tiny-en-noisy-context-prompt-2026-09-06.json"
            ).read_text(encoding="utf-8")
        )

    def test_receipt_is_preserved_and_all_other_checks_replay(self):
        before = copy.deepcopy(self.record)
        result = verify(self.record)
        self.assertEqual(self.record, before)
        self.assertEqual(p.shared._hash(self.record), KNOWN_CONTROL_ERROR_SHA)
        self.assertTrue(result["record_checks_pass"])
        self.assertTrue(result["report_only_correction"])
        self.assertFalse(result["original_control"]["retained_null_alignment_equal"])
        self.assertTrue(result["replayed_control"]["retained_null_alignment_equal"])
        self.assertFalse(result["inference_run"])

    def test_unknown_receipt_cannot_use_the_report_correction(self):
        for key, value in (("elapsed_ns", 1), ("qualified", True)):
            changed = copy.deepcopy(self.record)
            changed[key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "unrecognized"),
            ):
                verify(changed)

    def test_control_correction_does_not_bypass_record_validation(self):
        with patch.object(
            p, "validate_record", side_effect=ValueError("bad native result")
        ):
            with self.assertRaisesRegex(ValueError, "bad native result"):
                verify(self.record)

    def test_four_observations_share_two_encoder_admissions(self):
        self.assertEqual(self.record["native_window_count"], 2)
        self.assertEqual(self.record["decode_attempt_count"], 4)
        self.assertEqual(
            sum(len(c["window"]["encoder_calls"]) for c in self.record["cells"]), 6
        )
        for cell in self.record["cells"]:
            self.assertEqual(cell["session_version"], 0)
            self.assertTrue(cell["capacity_restored"])
            self.assertTrue(cell["first_snapshot_unchanged"])
            self.assertTrue(cell["same_lease"])
            self.assertTrue(all(not a["strict_candidate"] for a in cell["attempts"]))

    def test_prompt_changes_the_retained_text_not_its_join_authority(self):
        retained, head = self.record["cells"]
        original, prompted = retained["attempts"]
        self.assertEqual(original["audio_evidence"]["reason"], "no_lexical_text")
        self.assertEqual(prompted["word_decision"]["reason"], "anchor_missing")
        self.assertIsNone(prompted["recognition"]["hypothetical_full_text"])
        self.assertEqual(
            prompted["result"]["text"], head["attempts"][0]["result"]["text"]
        )
        self.assertEqual(
            head["attempts"][0]["alignment"]["words"],
            head["attempts"][1]["alignment"]["words"],
        )
        self.assertEqual(
            head["attempts"][0]["recognition"]["against_human_reference"][
                "word_edit_distance"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
