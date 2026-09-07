"""The audit cannot silently accept wrong pairings or a stale gate decision."""

import unittest
from copy import deepcopy

from tools.audit_draft_thresholds import audit_record, classify, raw_result


def record():
    cells = []
    for position, arm, score in (
        ("a1", "fast-reuse", 0.1),
        ("b1", "fast-draft32", 0.100001),
        ("b2", "fast-draft32", 0.100001),
        ("a2", "fast-reuse", 0.1),
    ):
        raw = raw_result(score)
        cells.append(
            dict(
                position=position,
                arm=arm,
                events=[],
                windows=[
                    dict(start_ms=0, end_ms=1, decode_observation={"result": raw})
                ],
                decision_traces=[
                    dict(
                        analysis_start_sample=0,
                        analysis_end_sample=16,
                        audio_evidence=dict(
                            **classify(raw),
                            no_speech_prob=score,
                            observation=dict(
                                sample_count=16,
                                digital_silence=False,
                                pcm_sha256="0" * 64,
                            ),
                        ),
                    )
                ],
            )
        )
    return {"cells": cells}


class DraftThresholdAuditTests(unittest.TestCase):
    def test_reconstruction_reports_drift_separately_from_exact_gate_parity(self):
        result = audit_record(record())
        self.assertFalse(result["exact_decision_parity_guaranteed_for_all_inputs"])
        for pair in result["pairs"]:
            self.assertTrue(pair["gate_decisions_exact"])
            self.assertEqual(pair["reconstructed_gate_replays"], 2)
            self.assertGreater(
                pair["maximum_absolute_score_deltas"]["no_speech_prob"], 0
            )
            self.assertEqual(pair["gate_crossings"], [])
        self.assertFalse(result["synthetic_crossing"]["decision_parity"])

    def test_actual_crossing_is_reported_not_hidden_by_exact_tokens(self):
        value = record()
        cell = value["cells"][1]
        raw = cell["windows"][0]["decode_observation"]["result"]
        raw["no_speech_prob"] = 0.6
        cell["decision_traces"][0]["audio_evidence"].update(
            classify(raw), no_speech_prob=0.6
        )
        pair = audit_record(value)["pairs"][0]
        self.assertTrue(pair["tokens_exact"])
        self.assertFalse(pair["gate_decisions_exact"])
        self.assertEqual(len(pair["gate_crossings"]), 1)

    def test_stale_score_or_decision_is_rejected(self):
        for field, changed in (("no_speech_prob", 0.6), ("state", "uncertain")):
            value = record()
            value["cells"][1]["decision_traces"][0]["audio_evidence"][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_record(value)

    def test_wrong_schedule_span_or_count_is_rejected(self):
        original = record()
        for mutation in ("schedule", "span", "window_count", "trace_count"):
            value = deepcopy(original)
            cell = value["cells"][1]
            if mutation == "schedule":
                cell["arm"] = "fast-reuse"
            elif mutation == "span":
                cell["windows"][0]["end_ms"] = 2
            elif mutation == "window_count":
                cell["windows"].clear()
            else:
                cell["decision_traces"].clear()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit_record(value)


if __name__ == "__main__":
    unittest.main()
