"""CPU-only registration, strict evidence joins and bounded-admission tests."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infra import eof_context_retry_worker as worker
from infra import modal_eof_context_retry as p


class EofRetryRegistrationTests(unittest.TestCase):
    def test_fixed_scope_and_native_schedule(self):
        self.assertEqual(p.GPU_TIMEOUT_SECONDS, 120)
        self.assertEqual(p.MAX_NATIVE_WINDOWS, 60)
        self.assertEqual(p.CLEANUP_RESERVE_SECONDS, 20)
        self.assertEqual(p.scope()["warmup_windows"], 0)
        self.assertEqual(p.scope()["chunk_bytes"], 32000)
        self.assertEqual(p.scope()["configured_retries"], 0)
        self.assertEqual(len(p.SCHEDULE), 3)

    def test_config_only_changes_opt_in_between_main_arms(self):
        baseline = p.stream_config("baseline")
        retry = p.stream_config("retry")
        self.assertFalse(baseline["eof_context_retry"])
        self.assertTrue(retry["eof_context_retry"])
        retry["eof_context_retry"] = False
        self.assertEqual(baseline, retry)
        self.assertEqual(baseline["word_context_limit_ms"], 6000)
        self.assertEqual(baseline["timestamp_tolerance_ms"], 200)
        self.assertEqual(
            p.stream_config("retry", "attenuated-prefix")["word_context_limit_ms"], 0
        )
        with self.assertRaisesRegex(ValueError, "unregistered arm"):
            p.stream_config("best-of-reference")

    def test_archive_exact_failure_is_not_recovery(self):
        baseline = p.archive_cell()
        self.assertEqual(baseline["decision_traces"][-1]["reason"], "anchor_missing")
        self.assertEqual(baseline["metrics"]["committed_samples"], 340160)
        self.assertEqual(baseline["metrics"]["decode_count"], 17)
        result = p.summarize([baseline], model_unchanged=True, capacity_restored=True)
        self.assertTrue(result["baseline_failure_reproduced"])
        self.assertFalse(result["development_recovery_demonstrated"])

    def test_complete_claim_without_retry_receipt_is_not_success(self):
        baseline = p.archive_cell()
        retry = copy.deepcopy(baseline)
        retry["stream_status"] = "completed"
        retry["recognition"]["against_human_reference"]["word_edit_distance"] = 0
        control = copy.deepcopy(retry)
        result = p.summarize(
            [baseline, retry, control], model_unchanged=True, capacity_restored=True
        )
        self.assertFalse(result["retry_stream_completed"])
        self.assertFalse(result["development_recovery_demonstrated"])

    def test_changed_native_token_breaks_baseline_reproduction(self):
        baseline = p.archive_cell()
        baseline["decision_traces"][-1]["word_alignment"]["words"][0]["tokens"][0] += 1
        result = p.summarize([baseline], model_unchanged=True, capacity_restored=True)
        self.assertFalse(result["baseline_failure_reproduced"])

    def test_actual_cached_input_recipes_and_human_references(self):
        if not (p.ROOT / "artifacts/speech-corpus-v1/6930-75918-0000.pcm").exists():
            self.skipTest("registered local PCM not present")
        inputs = p.input_records()
        self.assertEqual([i["sample_count"] for i in inputs], [538560, 174240])
        self.assertEqual(
            inputs[1]["pcm_sha256"],
            "aac84b9d01d59b30830c4a17a750facd6c384b44685343e731ba9dae0b7a8236",
        )
        self.assertEqual(
            p._corpus().b._word_difference("", inputs[0]["reference_text"])[
                "reference_word_count"
            ],
            88,
        )
        self.assertEqual(len(inputs[1]["reference_text"].split()), 26)

    def test_snapshot_rejects_undeclared_executed_dirty_source(self):
        def git(root, *args):
            if args[0] == "diff":
                return "src/whisper_runtime/adapters/word_policy.py"
            return ""

        with (
            patch.object(p, "_git", side_effect=git),
            patch.object(p, "input_records"),
            patch.object(p, "archive_cell"),
        ):
            with self.assertRaisesRegex(ValueError, "unreviewed dirty executed source"):
                p.snapshot()

    def test_snapshot_declares_working_bytes_without_changing_shared_guard(self):
        def git(root, *args):
            if args[0] == "diff":
                return p.PRODUCER
            if args[0] == "ls-files":
                return p.WORKER
            return "frozen-git-id"

        with (
            patch.object(p, "_git", side_effect=git),
            patch.object(p, "input_records"),
            patch.object(p, "archive_cell"),
        ):
            snapshot = p.snapshot()
        self.assertEqual(snapshot["digest"], p.shared._hash(snapshot["files"]))
        self.assertEqual(snapshot["dirty_source_paths"], sorted([p.PRODUCER, p.WORKER]))
        self.assertIn(p.WORKER, {f["path"] for f in snapshot["files"]})
        self.assertIn(p.ARCHIVE, {f["path"] for f in snapshot["files"]})


class AdmissionTests(unittest.TestCase):
    def adapter(self, now=1):
        native = SimpleNamespace(model_identity="model", start_window=Mock())
        measured = worker._MeasuredAdapter(native, 0, clock=lambda: now)
        measured.pcm = bytes(32000)
        return measured

    @staticmethod
    def args():
        return dict(
            start_ms=0,
            end_ms=1000,
            window_id="w",
            mel=SimpleNamespace(shape=(80, 3000)),
        )

    def test_counts_decode_and_alignment_encoders_separately(self):
        measured = self.adapter()
        measured.start_window(**self.args())
        record = measured.records[0]
        tensor = SimpleNamespace(shape=(80, 3000))
        measured.active = (record, "decode")
        measured.observe_encoder(None, [tensor], {})
        measured.active = (record, "alignment")
        measured.observe_encoder(None, [tensor], {"use_sdpa": False})
        measured.observe_decoder(None, [SimpleNamespace(shape=(1, 17))], {})
        self.assertEqual(
            [e["phase"] for e in record["encoder_calls"]], ["decode", "alignment"]
        )
        self.assertEqual(sum(e["input_frames"] for e in record["encoder_calls"]), 6000)
        self.assertEqual(
            record["decoder_forwards"], [dict(phase="alignment", token_count=17)]
        )
        self.assertEqual(record["pcm_sha256"], p.shared._sha(bytes(32000)))

    def test_reserve_stops_before_native_execution(self):
        measured = self.adapter(100_000_000_000)
        with self.assertRaisesRegex(TimeoutError, "cleanup reserve"):
            measured.start_window(**self.args())
        measured.native.start_window.assert_not_called()
        self.assertEqual(measured.count, 0)

    def test_aggregate_and_cell_bounds_stop_before_native(self):
        measured = self.adapter()
        measured.count = 60
        with self.assertRaisesRegex(RuntimeError, "native-window bound"):
            measured.start_window(**self.args())
        measured.count, measured.records = 0, [None] * 20
        with self.assertRaisesRegex(RuntimeError, "native-window bound"):
            measured.start_window(**self.args())
        measured.native.start_window.assert_not_called()

    def test_escaped_source_or_unmeasured_encoder_is_rejected(self):
        measured = self.adapter()
        args = {**self.args(), "end_ms": 1001}
        with self.assertRaisesRegex(ValueError, "outside registered source"):
            measured.start_window(**args)
        with self.assertRaisesRegex(RuntimeError, "unmeasured encoder"):
            measured.observe_encoder(None, [], {})
        measured.native.start_window.assert_not_called()


class RetryReceiptTests(unittest.TestCase):
    def fixture(self):
        pcm = bytes(64000)
        source = dict(
            eof=True,
            word_publication=None,
            retained_from_sample=0,
            analysis_end_sample=32000,
            decode_index=1,
        )
        alignment = dict(native=dict(window_id="retry"), words=[])
        candidate_trace = dict(
            word_alignment=alignment,
            result=alignment["native"],
            action="commit",
            word_publication={"text": "next"},
        )
        receipt = dict(
            source=source,
            anchor=[dict(span=dict(start_ms=700, end_ms=1000))],
            analysis_start_sample=3200,
            analysis_end_sample=32000,
            pcm_sha256=p.shared._sha(pcm[6400:]),
            session_version=1,
            committed_version=2,
            candidate=alignment,
            status="recovered",
        )
        cell = dict(
            arm="retry",
            case_id="concatenated-no-added-pauses",
            stream_status="completed",
            context_retry_observation=receipt,
            decision_traces=[source, candidate_trace],
            state=dict(version=2),
            metrics=dict(committed_samples=32000),
            windows=[
                dict(start_ms=0, end_ms=2000, window_id="source"),
                dict(
                    start_ms=200,
                    end_ms=2000,
                    window_id="retry",
                    closed=True,
                    capacity_restored=True,
                ),
            ],
        )
        return cell, pcm

    def test_recovered_receipt_requires_fixed_crop_and_real_release(self):
        cell, pcm = self.fixture()
        p.validate_retry_observation(cell, pcm)
        for field, value in (
            ("pcm_sha256", "wrong"),
            ("analysis_start_sample", 3520),
            ("committed_version", 1),
        ):
            broken = copy.deepcopy(cell)
            broken["context_retry_observation"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_retry_observation(broken, pcm)
        cell["windows"][-1]["capacity_restored"] = False
        with self.assertRaisesRegex(ValueError, "released native transaction"):
            p.validate_retry_observation(cell, pcm)

    def test_changed_source_and_extra_window_are_rejected(self):
        cell, pcm = self.fixture()
        cell["context_retry_observation"]["source"] = {
            **cell["context_retry_observation"]["source"],
            "eof": False,
        }
        with self.assertRaisesRegex(ValueError, "frozen refusal"):
            p.validate_retry_observation(cell, pcm)
        cell, pcm = self.fixture()
        cell["windows"].insert(0, dict(start_ms=0, end_ms=1000, window_id="extra"))
        with self.assertRaisesRegex(ValueError, "more than one additional"):
            p.validate_retry_observation(cell, pcm)

    def test_baseline_retry_and_missing_receipt_are_rejected(self):
        cell, pcm = self.fixture()
        cell["arm"] = "baseline"
        with self.assertRaisesRegex(ValueError, "baseline performed a retry"):
            p.validate_retry_observation(cell, pcm)
        cell["arm"], cell["context_retry_observation"] = "retry", None
        with self.assertRaisesRegex(ValueError, "lacks its observation"):
            p.validate_retry_observation(cell, pcm)


if __name__ == "__main__":
    unittest.main()
