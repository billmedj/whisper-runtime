"""Offline archived-bridge regressions; require only the existing cached PCM."""

import unittest
from unittest.mock import patch

from tools import verify_continuity_witness as replay


@unittest.skipUnless(
    (replay.p.ROOT / "artifacts/speech-corpus-v1").is_dir(), "cached corpus unavailable"
)
class ArchivedContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = replay.verify()

    def test_exact_archive_refusals_are_preserved(self):
        witness = self.report["witness"]
        self.assertEqual(
            witness["reasons"],
            (
                "analysis_identity_mismatch",
                "published_model_mismatch",
                "representation_mismatch",
                "lexical_timing_mismatch",
                "overlap_cuts_candidate_word",
            ),
        )
        self.assertEqual(witness["anchor"]["word_start"], 4)
        self.assertEqual(witness["anchor"]["word_end"], 8)
        self.assertEqual(witness["anchor"]["lexical_match_count"], 1)
        self.assertEqual(witness["sequence"]["left_unit_count"], 12)
        self.assertEqual(witness["sequence"]["right_unit_count"], 10)
        self.assertEqual(witness["lexical"]["text_relation"], "exact_units")
        self.assertTrue(witness["lexical"]["unit_tokens_equal"])
        self.assertEqual(witness["lexical_deltas_ms"][0], (" For", -420, 0))
        self.assertEqual(witness["lexical_deltas_ms"][3], (" she", -220, -40))
        self.assertEqual(witness["lexical_deltas_ms"][-1], (" happy", 20, 260))

    def test_raw_word_evidence_is_not_repaired(self):
        joint, candidate = self.report["joint_suffix"], self.report["candidate_prefix"]
        self.assertEqual((joint[0]["text"], joint[4]["text"]), (".", ","))
        self.assertEqual(candidate[0]["span"], {"start_ms": 3680, "end_ms": 6100})
        self.assertEqual(candidate[-1]["span"], {"start_ms": 7880, "end_ms": 8240})
        self.assertEqual(self.report["published_head_ms"], 3680)
        self.assertEqual(self.report["published_version"], 2)
        self.assertEqual(
            self.report["provenance"]["identical_shared_dispatched_files"], 35
        )
        for field in (
            "human_reference_used",
            "inference_run",
            "publication_authorized",
            "audio_retirement_authorized",
        ):
            self.assertIs(self.report[field], False)

    def test_reference_is_not_a_routing_input(self):
        source = replay.p.source_case()
        source["reference_text"] = "arbitrarily replaced; never consulted by witness"
        with patch.object(replay.p, "source_case", return_value=source):
            self.assertEqual(replay.verify(), self.report)

    def test_archives_remain_byte_identical(self):
        for path, expected in (
            (replay.p.ARCHIVE, replay.p.ARCHIVE_SHA),
            (replay.PROMPT_ARCHIVE, replay.PROMPT_SHA),
        ):
            self.assertEqual(
                replay.p.shared._sha((replay.p.ROOT / path).read_bytes()), expected
            )


if __name__ == "__main__":
    unittest.main()
