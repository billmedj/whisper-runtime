"""The stream diagnostic must distinguish output parity from score identity."""

import unittest
from copy import deepcopy

from tools.verify_stream_draft import PARITY_FIELDS, stream_comparison


class StreamDraftComparisonTests(unittest.TestCase):
    def reference(self):
        return {
            **dict.fromkeys(PARITY_FIELDS),
            "text": "observed text",
            "commit_source_ends": [100, 200],
            "commit_input_samples": [120, 240],
            "metrics": {"decode_count": 2},
            "traces": [{"reason": "candidate"}],
            "forward_counters": {"decoder_forwards": 40, "encoder_forwards": 2},
            "final_trace": {"no_speech_prob": 0.1},
        }

    def test_score_difference_is_explicit_even_with_observable_parity(self):
        reference = self.reference()
        candidate = deepcopy(reference)
        candidate["forward_counters"]["decoder_forwards"] = 25
        candidate["final_trace"]["no_speech_prob"] += 0.000001
        result = stream_comparison(reference, candidate)
        self.assertTrue(result["all_observable_stream_fields_exact"])
        self.assertEqual(result["decoder_forwards_saved"], 15)
        self.assertNotEqual(result["final_no_speech_delta"], 0)
        self.assertFalse(result["numerical_score_identity_qualified"])
        self.assertFalse(result["full_event_payload_identity_qualified"])

    def test_changed_cadence_or_trace_cannot_pass_as_exact(self):
        for field, value in (
            ("text", "different"),
            ("commit_source_ends", [100, 190]),
            ("commit_input_samples", [120, 260]),
            ("metrics", {"decode_count": 3}),
            ("traces", [{"reason": "anchor_missing"}]),
        ):
            reference = self.reference()
            candidate = deepcopy(reference)
            candidate[field] = value
            with self.subTest(field=field):
                result = stream_comparison(reference, candidate)
                self.assertFalse(result["all_observable_stream_fields_exact"])
                self.assertFalse(result["exact_fields"][field])

    def test_missing_required_fields_are_rejected_in_either_or_both_records(self):
        for field in (*PARITY_FIELDS, "forward_counters", "final_trace"):
            for remove_reference, remove_candidate in (
                (True, False),
                (False, True),
                (True, True),
            ):
                reference = self.reference()
                candidate = deepcopy(reference)
                if remove_reference:
                    del reference[field]
                if remove_candidate:
                    del candidate[field]
                with self.subTest(
                    field=field, sides=(remove_reference, remove_candidate)
                ):
                    with self.assertRaisesRegex(ValueError, "missing required fields"):
                        stream_comparison(reference, candidate)


if __name__ == "__main__":
    unittest.main()
