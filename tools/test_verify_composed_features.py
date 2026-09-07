"""Cross-arm checks reject outcomes that individually valid cells can conceal."""

import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from tools import verify_composed_features as verifier


def record():
    cells = []
    for arm, clock in zip(verifier.producer.ARMS, (4, 2, 2)):
        reuse = arm == "fast-reuse"
        events = [
            dict(kind="provisional", text="word"),
            dict(kind="commit", start_sample=0, end_sample=16000),
            dict(kind="final"),
        ]
        forwards = [
            dict(kind="encoder", phase="decode", interval_ms=1.0),
            dict(kind="decoder", phase="decode", interval_ms=2.0),
        ]
        if not reuse:
            forwards.append(dict(kind="encoder", phase="alignment", interval_ms=1.0))
        cells.append(
            dict(
                arm=arm,
                events=events,
                windows=[dict(forwards=forwards, closed=True, capacity_restored=True)],
                config=dict(
                    left_context_ms=2000 if arm == "cli-legacy" else 20000,
                    word_context_limit_ms=6000 if arm == "cli-legacy" else 24000,
                ),
                pacing=dict(event_elapsed_ns=[clock * 10**9] * 3),
                metrics=dict(accepted_samples=16000, committed_samples=16000),
                capacity_restored=True,
                stream_status="completed",
                done=True,
                summary=dict(
                    commits=[dict(start_sample=0, end_sample=16000, text="word")],
                    text="word",
                    forwards=dict(encoder=1 if reuse else 2, decoder=1),
                    forward_interval_ms=dict(
                        encoder=1.0 if reuse else 2.0, decoder=2.0
                    ),
                    first_commit_ns=clock * 10**9,
                ),
            )
        )
    return dict(status="completed", input=dict(sample_count=16000), cells=cells)


class ComposedAuditTests(unittest.TestCase):
    def test_cli_preserves_failed_report_and_returns_nonzero(self):
        for passed in (True, False):
            with (
                patch.object(
                    verifier,
                    "verify",
                    return_value={"preregistered_cross_arm_expectations_met": passed},
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(
                    verifier.main(["result.json", "preflight.json"]), 0 if passed else 1
                )
                self.assertIn(
                    "preregistered_cross_arm_expectations_met", output.getvalue()
                )

    def test_clean_cross_arm_controls_pass(self):
        self.assertTrue(
            verifier.audit(record())["preregistered_cross_arm_expectations_met"]
        )

    def test_valid_individual_summaries_do_not_hide_changed_fast_text(self):
        changed = record()
        reuse = changed["cells"][-1]
        reuse["events"][0]["text"] = reuse["summary"]["text"] = "changed"
        reuse["summary"]["commits"][0]["text"] = "changed"
        self.assertFalse(
            verifier.audit(changed)["preregistered_cross_arm_expectations_met"]
        )

    def test_raw_timing_or_count_tampering_is_rejected(self):
        for key in ("forwards", "forward_interval_ms"):
            changed = copy.deepcopy(record())
            changed["cells"][-1]["summary"][key]["decoder"] += 1
            with self.assertRaises(ValueError):
                verifier.audit(changed)

    def test_missing_final_or_partial_schedule_cannot_pass(self):
        changed = record()
        changed["cells"][-1]["events"][-1]["kind"] = "provisional"
        self.assertFalse(
            verifier.audit(changed)["preregistered_cross_arm_expectations_met"]
        )

    def test_added_decoder_or_lost_cleanup_cannot_pass(self):
        changed = record()
        reuse = changed["cells"][-1]
        reuse["windows"][0]["forwards"].append(
            dict(kind="decoder", phase="alignment", interval_ms=1.0)
        )
        reuse["summary"]["forwards"]["decoder"] += 1
        reuse["summary"]["forward_interval_ms"]["decoder"] += 1.0
        self.assertFalse(verifier.audit(changed)["checks"]["no_added_decoder_forwards"])
        changed = record()
        changed["cells"][-1]["windows"][0]["closed"] = False
        self.assertFalse(
            verifier.audit(changed)["preregistered_cross_arm_expectations_met"]
        )
        changed["cells"].pop()
        self.assertFalse(
            verifier.audit(changed)["preregistered_cross_arm_expectations_met"]
        )
