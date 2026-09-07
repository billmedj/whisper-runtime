"""CPU replay of the immutable, completed T4 EOF-context-retry experiment.

The recorded stream recovered this development fixture. This is not a new GPU
run, a held-out validation, an acoustic proof, or release qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

from infra import modal_eof_context_retry as p
from tools.analyze_word_resolution import _alignment
from whisper_runtime.adapters.native_result import NativeTimestampSegment
from whisper_runtime.adapters.word_policy import compare_word_hypotheses
from whisper_runtime.state import AudioSpan

ARCHIVE = "evidence/modal-t4-tiny-en-eof-context-retry-2026-09-06.json"
SHA256 = "4ab9914df3932a12909e21993f579c7b49d7e818fa7e94cc98ade7f4eae5a72e"
SOURCE_DIGEST = "b63ec4fe32e783400c683fea24183235f6133149d9132e50fc6566d80aa3ac25"


class EofContextRetryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (p.ROOT / ARCHIVE).read_bytes()
        cls.digest, cls.record = hashlib.sha256(raw).hexdigest(), json.loads(raw)
        cls.baseline, cls.retry, cls.control = cls.record["cells"]
        cls.inputs = None
        corpus = p._corpus()
        if all(
            (p.ROOT / corpus.ASSET_PATH / f["filename"]).exists()
            for f in corpus.read_registration()["fixtures"]
        ):
            cls.inputs = p.cases()

    def require_pcm(self):
        if self.inputs is None:
            self.skipTest("registered local PCM assets are not installed")
        return {case["id"]: case["pcm"] for case in self.inputs}

    def test_archive_identity_and_development_claim_boundary(self):
        self.assertEqual(self.digest, SHA256)
        snapshot = self.record["source"]["snapshot"]
        self.assertEqual(snapshot["digest"], SOURCE_DIGEST)
        self.assertEqual(snapshot["digest"], p.shared._hash(snapshot["files"]))
        self.assertEqual(self.record["status"], "completed")
        self.assertFalse(self.record["qualified"])
        self.assertEqual(self.record["claim_boundary"], p.CLAIMS)
        self.assertTrue(all(self.record["summary"].values()))
        self.assertEqual(self.record["scope"]["gpu_timeout_seconds"], 120)
        self.assertEqual(self.record["scope"]["configured_retries"], 0)
        self.assertFalse(self.record["scope"]["model_download"])

    def test_frozen_producer_validator_replays_entire_record(self):
        self.require_pcm()
        # The source snapshot is the one that actually ran. Do not replace it
        # with current working-tree bytes or add these later tests to its list.
        with patch.object(p, "cases", return_value=self.inputs):
            p.validate_record(self.record, self.record["source"]["snapshot"])

    def test_baseline_and_all_pre_retry_observations_reproduce_exactly(self):
        self.assertEqual(
            p.control_signature(self.baseline), p.control_signature(p.archive_cell())
        )
        self.assertEqual(
            self.baseline["decision_traces"][-1]["reason"], "anchor_missing"
        )
        self.assertEqual(self.baseline["metrics"]["committed_samples"], 340160)
        receipt = self.retry["context_retry_observation"]
        before = {
            **self.retry,
            "decision_traces": self.retry["decision_traces"][:17],
            "metrics": self.baseline["metrics"],
            "events": self.baseline["events"],
        }
        self.assertEqual(
            p.control_signature(before), p.control_signature(self.baseline)
        )
        old_commits = p.commit_signature(self.baseline["events"])
        new_commits = p.commit_signature(self.retry["events"])
        self.assertEqual(new_commits[: len(old_commits)], old_commits)
        self.assertEqual(len(new_commits), len(old_commits) + 1)
        self.assertEqual(receipt["source"], self.retry["decision_traces"][-2])
        self.assertEqual(receipt["session_version"], 3)
        self.assertEqual(receipt["committed_version"], 4)
        self.assertEqual(self.retry["state"]["version"], 4)
        self.assertEqual(new_commits[-1]["start"], 340160)
        self.assertEqual(new_commits[-1]["end"], 538560)

    def test_ordinary_word_policy_rejects_source_and_publishes_shifted_observation(
        self,
    ):
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
        self.assertEqual(self.retry["decision_traces"][-1]["action"], "commit")
        self.assertEqual(self.retry["decision_traces"][-1]["reason"], "source_unit")
        self.assertEqual(self.retry["events"][-1]["kind"], "final")
        self.assertTrue(all(self.retry["checks"].values()))
        self.assertEqual(self.retry["metrics"]["accepted_samples"], 538560)
        self.assertEqual(self.retry["metrics"]["committed_samples"], 538560)
        self.assertEqual(self.retry["metrics"]["buffered_samples"], 0)

    def test_actual_pcm_and_one_fixed_guard_remain_bound_to_the_source(self):
        pcm = self.require_pcm()[self.retry["case_id"]]
        receipt = self.retry["context_retry_observation"]
        source = receipt["source"]
        start = max(
            source["retained_from_sample"],
            (receipt["anchor"][0]["span"]["start_ms"] // 20 * 20 - 500) * 16,
        )
        self.assertEqual((start, receipt["analysis_end_sample"]), (323520, 538560))
        self.assertEqual(receipt["analysis_start_sample"], start)
        self.assertGreaterEqual(start, source["retained_from_sample"])
        self.assertEqual(
            hashlib.sha256(pcm[start * 2 : 538560 * 2]).hexdigest(),
            receipt["pcm_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(
                pcm[
                    source["analysis_start_sample"] * 2 : source["analysis_end_sample"]
                    * 2
                ]
            ).hexdigest(),
            source["audio_evidence"]["observation"]["pcm_sha256"],
        )
        self.assertEqual(len(self.retry["windows"]), len(self.baseline["windows"]) + 1)
        final_window = self.retry["windows"][-1]
        self.assertEqual(
            (final_window["start_ms"], final_window["end_ms"]), (20220, 33660)
        )
        self.assertEqual(final_window["pcm_sha256"], receipt["pcm_sha256"])
        p.validate_retry_observation(self.retry, pcm)

    def test_all_native_work_release_and_human_scores_stay_in_the_denominator(self):
        windows = [
            window for cell in self.record["cells"] for window in cell["windows"]
        ]
        self.assertEqual(
            [len(cell["windows"]) for cell in self.record["cells"]], [17, 18, 6]
        )
        self.assertEqual(self.record["native_window_count"], len(windows))
        self.assertEqual(len(windows), 41)
        encoders = [call for window in windows for call in window["encoder_calls"]]
        self.assertEqual(len(encoders), 81)
        self.assertEqual(sum(call["input_frames"] for call in encoders), 243000)
        self.assertTrue(
            all(window["closed"] and window["capacity_restored"] for window in windows)
        )
        self.assertTrue(self.record["model"]["unchanged"])
        self.assertTrue(self.record["capacity_restored"])
        self.assertEqual(
            [
                c["recognition"]["against_human_reference"]["word_edit_distance"]
                for c in self.record["cells"]
            ],
            [36, 6, 1],
        )
        self.assertEqual(
            [
                c["recognition"]["against_human_reference"]["reference_word_count"]
                for c in self.record["cells"]
            ],
            [88, 88, 26],
        )
        self.assertEqual(self.control["stream_status"], "completed")
        self.assertIsNone(self.control["context_retry_observation"])
        self.assertTrue(all(self.control["checks"].values()))

    def test_tampered_retry_pcm_crop_and_release_are_rejected(self):
        pcm = self.require_pcm()[self.retry["case_id"]]
        for field, value in (
            ("pcm_sha256", "0" * 64),
            ("analysis_start_sample", 323840),
            ("committed_version", 3),
        ):
            cell = copy.deepcopy(self.retry)
            cell["context_retry_observation"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_retry_observation(cell, pcm)
        cell = copy.deepcopy(self.retry)
        cell["windows"][-1]["capacity_restored"] = False
        with self.assertRaisesRegex(ValueError, "released native transaction"):
            p.validate_retry_observation(cell, pcm)

    def test_tampered_published_score_and_window_accounting_are_rejected(self):
        self.require_pcm()
        bad_score = copy.deepcopy(self.record)
        bad_score["cells"][1]["recognition"]["against_human_reference"][
            "word_edit_distance"
        ] = 0
        bad_count = copy.deepcopy(self.record)
        bad_count["native_window_count"] -= 1
        with patch.object(p, "cases", return_value=self.inputs):
            for record in (bad_score, bad_count):
                with (
                    self.subTest(
                        score=record["cells"][1]["recognition"][
                            "against_human_reference"
                        ]["word_edit_distance"]
                    ),
                    self.assertRaises(ValueError),
                ):
                    p.validate_record(record, record["source"]["snapshot"])


if __name__ == "__main__":
    unittest.main()
