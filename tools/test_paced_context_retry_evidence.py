"""CPU regression of the immutable source-paced T4 retry and noisy refusal.

This replays recorded evidence, not another GPU run or a release qualification.
The later test file is intentionally absent from the executed source snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

from infra import modal_paced_context_retry as p
from tools.analyze_word_resolution import _alignment
from whisper_runtime.adapters.native_result import NativeTimestampSegment
from whisper_runtime.adapters.word_policy import compare_word_hypotheses
from whisper_runtime.state import AudioSpan

ARCHIVE = "evidence/modal-t4-tiny-en-paced-context-retry-2026-09-06.json"
SHA256 = "fa856cb4f0028109e7524b6e563fd008600d6fe84043d86c8f433784bdc2e3bf"
SOURCE_DIGEST = "42e810e8c1a4dcd5f232b8048ff786f7bece7177d58c129284187bd52f9ddcdb"


class PacedContextRetryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (p.ROOT / ARCHIVE).read_bytes()
        cls.digest, cls.record = hashlib.sha256(raw).hexdigest(), json.loads(raw)
        cls.baseline, cls.retry, cls.noisy = cls.record["cells"]
        cls.inputs = None
        corpus = p._corpus()
        if all(
            (p.ROOT / corpus.ASSET_PATH / fixture["filename"]).exists()
            for fixture in corpus.read_registration()["fixtures"]
        ):
            cls.inputs = p.cases()

    def require_pcm(self):
        if self.inputs is None:
            self.skipTest("registered local PCM assets are not installed")
        return {case["id"]: case["pcm"] for case in self.inputs}

    def test_archive_identity_and_scoped_recovery_not_noisy_qualification(self):
        self.assertEqual(self.digest, SHA256)
        snapshot = self.record["source"]["snapshot"]
        self.assertEqual(snapshot["digest"], SOURCE_DIGEST)
        self.assertEqual(snapshot["digest"], p.shared._hash(snapshot["files"]))
        self.assertNotIn(
            "tools/test_paced_context_retry_evidence.py",
            {item["path"] for item in snapshot["files"]},
        )
        self.assertEqual(self.record["status"], "completed")
        self.assertFalse(self.record["qualified"])
        self.assertFalse(any(self.record["claim_boundary"].values()))
        summary = self.record["summary"]
        self.assertTrue(summary["development_recovery_demonstrated"])
        self.assertTrue(summary["full_stream_completed"])
        self.assertTrue(summary["noisy_refusal_expected"])
        self.assertFalse(summary["noisy_stream_completed"])
        self.assertFalse(summary["noisy_recovery_demonstrated"])
        self.assertTrue(all(summary["cell_integrity"].values()))
        self.assertEqual(self.record["scope"]["pacing"], p._corpus().PACED_REPLAY)
        self.assertEqual(self.record["scope"]["gpu_calls"], 1)
        self.assertEqual(self.record["scope"]["warmup_windows"], 0)

    def test_frozen_producer_validator_replays_complete_actual_record(self):
        self.require_pcm()
        with patch.object(p, "cases", return_value=self.inputs):
            p.validate_record(self.record, self.record["source"]["snapshot"])

    def test_same_call_native_prefix_and_frozen_commits_match_before_one_retry(self):
        self.assertEqual(self.record["summary"]["cross_arm_comparison"], "matched")
        self.assertEqual(self.baseline["metrics"]["committed_samples"], 340160)
        self.assertEqual(self.retry["metrics"]["committed_samples"], 538560)
        receipt = self.retry["context_retry_observation"]
        self.assertEqual(receipt["source"], self.retry["decision_traces"][-2])
        before = {
            **self.retry,
            "decision_traces": self.retry["decision_traces"][:17],
            "events": self.baseline["events"],
            "metrics": self.baseline["metrics"],
        }
        self.assertEqual(
            p.prior.control_signature(before), p.prior.control_signature(self.baseline)
        )
        old = p.prior.commit_signature(self.baseline["events"])
        new = p.prior.commit_signature(self.retry["events"])
        self.assertEqual(new[:-1], old)
        self.assertEqual((new[-1]["start"], new[-1]["end"]), (340160, 538560))
        self.assertEqual(len(self.retry["windows"]), len(self.baseline["windows"]) + 1)
        self.assertEqual(receipt["status"], "recovered")
        self.assertEqual(receipt["analysis_start_sample"], 323520)
        self.assertEqual(receipt["committed_version"], receipt["session_version"] + 1)

    def test_unchanged_strict_word_policy_rejects_source_and_accepts_actual_retry(self):
        receipt = self.retry["context_retry_observation"]
        anchor = tuple(
            NativeTimestampSegment(
                AudioSpan(**word["span"]), word["text"], tuple(word["tokens"])
            )
            for word in receipt["anchor"]
        )
        options = dict(
            committed_through_ms=21260,
            anchor=anchor,
            holdback_ms=2000,
            timestamp_tolerance_ms=200,
            final=True,
        )
        refused = compare_word_hypotheses(
            None, _alignment(receipt["source"]["word_alignment"]), **options
        )
        recovered = compare_word_hypotheses(
            None, _alignment(receipt["candidate"]), **options
        )
        self.assertEqual(refused.reason, "anchor_missing")
        self.assertIsNone(refused.publication)
        self.assertIsNotNone(recovered.publication)
        self.assertEqual(
            p._canonical(asdict(recovered.publication)),
            self.retry["decision_traces"][-1]["word_publication"],
        )
        self.assertTrue(all(self.retry["checks"].values()))
        self.assertEqual(self.retry["events"][-1]["kind"], "final")

    def test_noisy_refusal_admits_full_input_but_keeps_uncommitted_audio_without_retry(
        self,
    ):
        noisy = self.noisy
        receipt = noisy["context_retry_observation"]
        self.assertEqual(noisy["error"]["category"], "policy_resolution")
        self.assertEqual(receipt["status"], "unavailable")
        self.assertEqual(receipt["reason"], "no_distinct_anchor_window")
        self.assertIsNone(receipt["candidate"])
        self.assertEqual(len(noisy["windows"]), receipt["source"]["decode_index"])
        self.assertFalse(
            any(":context-retry:" in w["window_id"] for w in noisy["windows"])
        )
        self.assertEqual(noisy["metrics"]["accepted_samples"], 174240)
        self.assertEqual(noisy["metrics"]["committed_samples"], 58880)
        self.assertEqual(
            noisy["retained_from_sample"] + noisy["metrics"]["buffered_samples"],
            noisy["metrics"]["accepted_samples"],
        )
        self.assertFalse(any(event["kind"] == "final" for event in noisy["events"]))
        self.assertTrue(p._integrity(noisy))
        self.assertFalse(noisy["checks"]["paced_terminal_coverage"])

    def test_full_native_cost_and_paced_latencies_remain_in_the_record(self):
        cells = self.record["cells"]
        self.assertEqual([len(cell["windows"]) for cell in cells], [17, 18, 6])
        self.assertEqual(self.record["native_window_count"], 41)
        self.assertEqual(
            [sum(len(w["encoder_calls"]) for w in cell["windows"]) for cell in cells],
            [34, 36, 12],
        )
        self.assertTrue(
            all(
                w["closed"] and w["capacity_restored"]
                for c in cells
                for w in c["windows"]
            )
        )
        self.assertEqual(
            [
                c["recognition"]["against_human_reference"]["word_edit_distance"]
                for c in cells
            ],
            [36, 6, 18],
        )
        self.assertTrue(self.record["model"]["unchanged"])
        self.assertTrue(self.record["capacity_restored"])
        pacing = self.retry["pacing"]
        self.assertEqual(pacing["first_nonempty_text_ns"], 2125613514)
        self.assertEqual(pacing["drain_after_input_finished_ns"], 327473056)
        self.assertEqual(pacing["event_elapsed_ns"][-1], 33986726293)
        self.assertEqual(pacing["elapsed_ns"], 33987672068)
        self.assertEqual(self.record["elapsed_ns"], 88713040566)

    def test_clock_tamper_fails_full_record_validation(self):
        self.require_pcm()
        for target in ("admission", "output"):
            record = copy.deepcopy(self.record)
            pacing = record["cells"][1]["pacing"]
            if target == "admission":
                pacing["admissions"][0]["offered_ns"] = 19999999
            else:
                pacing["event_elapsed_ns"][0] = 0
            with (
                self.subTest(target=target),
                patch.object(p, "cases", return_value=self.inputs),
            ):
                with self.assertRaisesRegex(ValueError, "checks do not replay"):
                    p.validate_record(record, record["source"]["snapshot"])

    def test_timeout_and_false_noise_completion_do_not_become_recovery(self):
        cells = copy.deepcopy(self.record["cells"])
        cells[2]["pacing"]["status"] = "drain_timeout"
        cells[2]["error"] = None
        self.assertFalse(p._integrity(cells[2]))
        summary = p.summarize(cells, model_unchanged=True, capacity_restored=True)
        self.assertFalse(summary["experiment_observation_complete"])
        self.assertFalse(summary["development_recovery_demonstrated"])
        self.require_pcm()
        record = copy.deepcopy(self.record)
        record["cells"][2]["stream_status"] = "completed"
        with patch.object(p, "cases", return_value=self.inputs):
            with self.assertRaisesRegex(ValueError, "completion claim differs"):
                p.validate_record(record, record["source"]["snapshot"])


if __name__ == "__main__":
    unittest.main()
