"""Fail-closed report/orchestration helpers; no native model execution."""

import copy
import unittest

import verify_draft_restore_process as diagnostic


def receipts():
    events = [
        dict(kind="commit", start_sample=0, end_sample=10, committed_through_sample=10),
        dict(
            kind="commit", start_sample=10, end_sample=20, committed_through_sample=20
        ),
        dict(kind="final"),
    ]
    common = dict(
        sources={},
        model={},
        model_file_sha256="model",
        audio={"samples": 20},
        config={},
        options={},
        pipeline_identity="pipeline",
        manifest_sha256="manifest",
        state={},
        metrics={},
        cleanup={"cleared": True},
    )
    control = dict(
        common,
        pid=1,
        completed=True,
        events=events,
        traces=[{"tokens": [1]}, {"tokens": [2]}],
        input_admissions=[0, 1],
        native_admissions=[
            {"draft_tokens": [1], "parallel_prefills_at_start": 1},
            {"draft_tokens": [1], "parallel_prefills_at_start": 1},
        ],
    )
    producer = dict(
        common,
        pid=2,
        completed=False,
        ended_ns=2,
        events=events[:1],
        traces=control["traces"][:1],
        input_admissions=[0],
        native_admissions=control["native_admissions"][:1],
        boundary={"old_draft_tokens": [1]},
        checkpoint_fields=["config"],
    )
    consumer = dict(
        common,
        pid=3,
        completed=True,
        started_ns=3,
        events=events[1:],
        traces=control["traces"][1:],
        input_admissions=[1],
        native_admissions=[{"draft_tokens": [], "parallel_prefills_at_start": 0}],
        initial_restore={
            key: True
            for key in (
                "state_exact",
                "metrics_exact",
                "old_draft_absent",
                "inactive",
                "no_native_admissions",
                "no_lease",
            )
        },
    )
    orchestration = [
        dict(returncode=0, returned_ns=i, launched_ns=i) for i in (1, 2, 3)
    ]
    return tuple(
        copy.deepcopy(value) for value in (control, producer, consumer, orchestration)
    )


class DraftRestoreHelpersTests(unittest.TestCase):
    def test_score_projection_preserves_tokens_and_decisions(self):
        before = {
            "tokens": [1, 2],
            "action": "commit",
            "metadata": {"avg_logprob": -0.2},
        }
        after = {
            "tokens": [1, 2],
            "action": "commit",
            "metadata": {"avg_logprob": -0.3},
        }
        self.assertEqual(
            diagnostic.without_scores(before), diagnostic.without_scores(after)
        )
        after["tokens"][0] = 3
        self.assertNotEqual(
            diagnostic.without_scores(before), diagnostic.without_scores(after)
        )

    def test_difference_reports_exact_float_changes_and_paths(self):
        differences = diagnostic.differences(
            [{"probability": 0.25}], [{"probability": 0.5}]
        )
        self.assertEqual(
            differences,
            [
                {
                    "path": "/0/probability",
                    "control": 0.25,
                    "restored": 0.5,
                    "absolute_delta": 0.25,
                }
            ],
        )

    def test_difference_does_not_zip_away_missing_rows(self):
        self.assertTrue(diagnostic.differences([1, 2], [1]))
        self.assertTrue(diagnostic.differences({"token": 1}, {}))

    def test_phase_bound_is_short_and_fixed(self):
        self.assertEqual(diagnostic.PHASE_TIMEOUT_SECONDS, 120)
        self.assertEqual(diagnostic.CHUNK_BYTES, 64000)

    def test_combined_minimal_receipts_pass(self):
        self.assertEqual(diagnostic.combine(*receipts())["status"], "passed")

    def test_fail_closed_on_identity_tokens_events_resources_and_old_hint(self):
        mutations = (
            lambda c, p, r, o: r.update(pid=p["pid"]),
            lambda c, p, r, o: r.update(model_file_sha256="changed"),
            lambda c, p, r, o: r["traces"][0].update(tokens=[99]),
            lambda c, p, r, o: r["events"].pop(),
            lambda c, p, r, o: r["cleanup"].update(cleared=False),
            lambda c, p, r, o: r["initial_restore"].update(no_lease=False),
            lambda c, p, r, o: r["native_admissions"][0].update(draft_tokens=[1]),
            lambda c, p, r, o: r["native_admissions"][0].update(options="different"),
            lambda c, p, r, o: r["input_admissions"].append(2),
            lambda c, p, r, o: r["events"][0].update(start_sample=11),
            lambda c, p, r, o: p.update(checkpoint_fields=["draft_tokens"]),
            lambda c, p, r, o: o[1].update(returned_ns=4),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                values = receipts()
                mutate(*values)
                self.assertEqual(diagnostic.combine(*values)["status"], "failed")

    def test_diagnostic_float_differences_are_reported_not_hidden(self):
        values = receipts()
        values[0]["traces"][1]["avg_logprob"] = -0.1
        values[2]["traces"][0]["avg_logprob"] = -0.2
        result = diagnostic.combine(*values)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["exact_trace_scores"])
        self.assertEqual(len(result["trace_differences"]), 1)


if __name__ == "__main__":
    unittest.main()
