"""Tests for receipt reconstruction. They do not load a model or contact Modal."""

import unittest
from copy import deepcopy

from tools.verify_draft_gpu import commits, policy


def events():
    base = dict(
        segment_id="one",
        revision=1,
        start_sample=0,
        end_sample=16000,
        session_version=1,
    )
    return [
        dict(base, kind="provisional", sequence_number=1, text="hello"),
        dict(base, kind="commit", sequence_number=2, text=None),
        dict(kind="final", sequence_number=3),
    ]


class AuditTests(unittest.TestCase):
    def test_committed_text_comes_from_exact_revision(self):
        self.assertEqual(
            commits(events()), [dict(start_sample=0, end_sample=16000, text="hello")]
        )

    def test_corrupt_commit_is_rejected(self):
        for mutation in ("revision", "start_sample", "sequence_number"):
            with self.subTest(mutation=mutation):
                value = events()
                value[1][mutation] += 1
                with self.assertRaises(ValueError):
                    commits(value)

    def test_post_final_event_is_rejected(self):
        value = events()
        value.append(dict(value[0], sequence_number=4))
        with self.assertRaises(ValueError):
            commits(value)

    def test_policy_separates_score_change_from_decision_change(self):
        first = dict(
            decision_traces=[
                dict(action="commit", reason="agreed", result=dict(avg_logprob=-0.1))
            ]
        )
        second = deepcopy(first)
        second["decision_traces"][0]["result"]["avg_logprob"] = -0.10000001
        self.assertEqual(policy(first), policy(second))
        second["decision_traces"][0]["reason"] = "other"
        self.assertNotEqual(policy(first), policy(second))


if __name__ == "__main__":
    unittest.main()
