"""CPU-only registration, source-clock, same-call and refusal claim checks."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from infra import modal_eof_context_retry as prior
from infra import modal_paced_context_retry as p
from tools import test_modal_eof_context_retry as retry_tests
from tools import test_modal_word_corpus as corpus_tests


class RegistrationTests(unittest.TestCase):
    def test_bounded_single_call_and_actual_source_speed(self):
        scope = p.scope()
        self.assertEqual((p.GPU_TIMEOUT_SECONDS, p.CLEANUP_RESERVE_SECONDS), (150, 20))
        self.assertEqual(
            (p.MAX_NATIVE_WINDOWS, p.MAX_NATIVE_WINDOWS_PER_CELL), (80, 30)
        )
        self.assertEqual(scope["pacing"], p._corpus().PACED_REPLAY)
        self.assertEqual(scope["chunk_bytes"], 640)
        self.assertEqual(scope["gpu_calls"], 1)
        self.assertEqual(scope["configured_retries"], 0)
        self.assertEqual(scope["warmup_windows"], 0)
        self.assertFalse(scope["model_download"])
        self.assertFalse(scope["reference_used_for_selection"])
        self.assertFalse(scope["historical_unpaced_baseline_required"])
        self.assertFalse(any(p.CLAIMS.values()))
        self.assertEqual(
            p.SCHEDULE,
            (
                ("concatenated-no-added-pauses", "baseline"),
                ("concatenated-no-added-pauses", "retry"),
                ("noisy-prefix", "retry"),
            ),
        )

    def test_only_retry_opt_in_changes_between_main_arms(self):
        baseline, retry = p.stream_config("baseline"), p.stream_config("retry")
        self.assertTrue(retry.pop("eof_context_retry"))
        self.assertFalse(baseline.pop("eof_context_retry"))
        self.assertEqual(baseline, retry)
        self.assertEqual(baseline["timestamp_tolerance_ms"], 200)
        self.assertEqual(baseline["word_context_limit_ms"], 6000)
        self.assertEqual(
            p.stream_config("retry", "noisy-prefix")["word_context_limit_ms"], 0
        )
        with self.assertRaisesRegex(ValueError, "unregistered case"):
            p.stream_config("retry", "best-noise")
        with self.assertRaisesRegex(ValueError, "unregistered arm"):
            p.stream_config("best-of-reference")

    def test_actual_pcm_recipe_identity_is_not_an_archive_substitution(self):
        if not (p.ROOT / "artifacts/speech-corpus-v1/6930-75918-0000.pcm").exists():
            self.skipTest("registered local PCM unavailable")
        inputs = p.input_records()
        self.assertEqual([item["sample_count"] for item in inputs], [538560, 174240])
        self.assertEqual([item["duration_seconds"] for item in inputs], [33.66, 10.89])
        self.assertEqual(inputs[1]["source_case_id"], "mixed-continuous-noise64")
        self.assertEqual(
            inputs[1]["pcm_sha256"],
            "5276e641f14eaae484a8da6f916e44f9373239fbe70c70ba41a3f1a36f747108",
        )
        self.assertEqual(inputs[1]["recipe"]["amplitude_s16"], 64)
        self.assertEqual(len(inputs[1]["reference_text"].split()), 26)

    def test_snapshot_tracks_current_helpers_without_historical_gate(self):
        def git(root, *args):
            return (
                p.PRODUCER
                if args[0] == "diff"
                else ""
                if args[0] == "ls-files"
                else "git-id"
            )

        with (
            patch.object(p, "_git", side_effect=git),
            patch.object(p, "input_records"),
            patch.object(
                prior, "archive_cell", side_effect=AssertionError("historical gate")
            ),
        ):
            snapshot = p.snapshot()
        paths = {item["path"] for item in snapshot["files"]}
        self.assertTrue(
            {
                p.PRODUCER,
                p.WORKER,
                p.TEST,
                prior.PRODUCER,
                "src/whisper_runtime/adapters/paced_replay.py",
            }
            <= paths
        )
        self.assertNotIn(prior.ARCHIVE, paths)
        self.assertEqual(snapshot["dirty_source_paths"], [p.PRODUCER])
        self.assertEqual(snapshot["digest"], p.shared._hash(snapshot["files"]))

    def test_unreviewed_runtime_source_still_blocks_snapshot(self):
        with (
            patch.object(p, "input_records"),
            patch.object(
                p, "_git", return_value="src/whisper_runtime/adapters/word_policy.py"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unreviewed dirty executed source"):
                p.snapshot()

    def test_worker_delegation_does_not_mutate_prior_module(self):
        original = prior.GPU_TIMEOUT_SECONDS
        with patch(
            "infra.eof_context_retry_worker.run_worker", return_value="receipt"
        ) as run:
            self.assertEqual(p.run_worker({"digest": "x"}), "receipt")
        run.assert_called_once_with({"digest": "x"}, producer=p, paced=True)
        self.assertEqual(prior.GPU_TIMEOUT_SECONDS, original)


class ClaimTests(unittest.TestCase):
    def cell(self, *, complete=False, receipt=None, error="policy_resolution"):
        return dict(
            metrics={
                "committed_samples": 640 if complete else 320,
                "accepted_samples": 640,
            },
            events=[],
            decision_traces=[],
            stream_status="completed" if complete else "failed",
            context_retry_observation=receipt,
            error=None if complete else {"category": error},
            capacity_restored=True,
            pacing={"status": "completed" if complete else "runtime_error"},
            checks={
                "native_pcm_binding": True,
                "paced_source_not_early": True,
                "full_input_committed_at_eof": complete,
                "paced_completed": complete,
            },
            recognition={
                "against_human_reference": {"word_edit_distance": 6 if complete else 36}
            },
        )

    def test_already_complete_is_not_fabricated_recovery(self):
        baseline, retry = self.cell(complete=True), self.cell(complete=True)
        noisy = self.cell()
        result = p.summarize(
            [baseline, retry, noisy], model_unchanged=True, capacity_restored=True
        )
        self.assertTrue(result["full_stream_completed"])
        self.assertEqual(result["retry_outcome"], "already_complete_without_retry")
        self.assertFalse(result["development_recovery_demonstrated"])
        self.assertTrue(result["noisy_refusal_expected"])
        self.assertFalse(result["noisy_stream_completed"])
        self.assertEqual(result["cross_arm_comparison"], "not_available")

    def test_refusal_is_not_recovery_and_integrity_failure_is_not_expected_noise(self):
        retry = self.cell(receipt={"status": "unavailable"})
        self.assertEqual(p.outcome(retry), "retry_refused")
        self.assertTrue(p._integrity(retry))
        retry["checks"]["native_pcm_binding"] = False
        self.assertFalse(p._integrity(retry))
        retry = self.cell(error="runtime")
        self.assertEqual(p.outcome(retry), "execution_failed")

    def test_within_retry_recovery_does_not_require_identical_paced_windows(self):
        baseline = self.cell()
        retry = self.cell(
            complete=True,
            receipt={"status": "recovered", "source": {"decode_index": 1}},
        )
        noisy = self.cell()
        with patch.object(
            prior,
            "control_signature",
            side_effect=["new-paced-prefix", "different-paced-baseline"],
        ):
            result = p.summarize(
                [baseline, retry, noisy], model_unchanged=True, capacity_restored=True
            )
        self.assertTrue(result["development_recovery_demonstrated"])
        self.assertEqual(result["cross_arm_comparison"], "unmatched")
        self.assertTrue(result["noisy_refusal_expected"])
        self.assertFalse(result["noisy_recovery_demonstrated"])
        self.assertFalse(p.CLAIMS["noisy_stream_recovery"])

    def test_missing_noisy_cell_and_changed_model_cannot_pass(self):
        baseline, retry = self.cell(), self.cell(complete=True)
        for cells, unchanged, capacity in (
            ([baseline, retry], True, True),
            ([baseline, retry, self.cell()], False, True),
            ([baseline, retry, self.cell()], True, False),
        ):
            result = p.summarize(
                cells, model_unchanged=unchanged, capacity_restored=capacity
            )
            self.assertFalse(result["experiment_observation_complete"])
            self.assertFalse(result["development_recovery_demonstrated"])

    def test_pacing_timeout_without_exception_is_not_an_integral_observation(self):
        for status in ("drain_timeout", "step_limit"):
            with self.subTest(status=status):
                retry = self.cell()
                retry["pacing"]["status"] = status
                retry["error"] = None
                self.assertFalse(p._integrity(retry))
                result = p.summarize(
                    [self.cell(), retry, self.cell()],
                    model_unchanged=True,
                    capacity_restored=True,
                )
                self.assertFalse(result["experiment_observation_complete"])
                self.assertFalse(result["development_recovery_demonstrated"])


class SourceClockTests(unittest.TestCase):
    def test_paced_clock_checks_are_added_to_native_checks(self):
        events, traces, pacing = corpus_tests.CorpusGuards().paced_observations()
        cell = dict(events=events, decision_traces=traces, pacing=pacing)
        with patch.object(
            prior, "cell_checks", return_value={"native_pcm_binding": True}
        ):
            checks = p.cell_checks(cell, bytes(1280))
            self.assertTrue(all(checks.values()))
            self.assertIn("paced_source_not_early", checks)
            cell["pacing"]["admissions"][0]["offered_ns"] = 19_999_999
            checks = p.cell_checks(cell, bytes(1280))
            self.assertFalse(checks["paced_source_not_early"])

    def test_missing_recovery_receipt_is_not_hidden_by_natural_completion(self):
        cell, pcm = retry_tests.RetryReceiptTests().fixture()
        cell["windows"][-1]["window_id"] = "paced:context-retry:3200:32000"
        cell["context_retry_observation"] = None
        with self.assertRaisesRegex(ValueError, "native window lacks its observation"):
            p.validate_retry_observation(cell, pcm)
        cell["windows"].pop()
        p.validate_retry_observation(cell, pcm)

    def test_recovery_receipt_retains_prior_strict_pcm_and_capacity_guards(self):
        cell, pcm = retry_tests.RetryReceiptTests().fixture()
        p.validate_retry_observation(cell, pcm)
        broken = copy.deepcopy(cell)
        broken["context_retry_observation"]["pcm_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "PCM identity differs"):
            p.validate_retry_observation(broken, pcm)


if __name__ == "__main__":
    unittest.main()
