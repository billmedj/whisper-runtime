"""Offline checks for the paced diagnostic and its per-run alignment control."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from infra import modal_paced_features as experiment


class MeasuredFacadeTests(unittest.TestCase):
    def test_modes_cleanup_and_failure_restore_active_scope(self):
        calls = []

        class Raw:
            closed = capacity_released = False
            complete = True

            def prepare_word_alignment(self, *, reuse_alignment_features):
                calls.append(reuse_alignment_features)
                return "words"

            def close(self):
                self.closed = self.capacity_released = True
                return True

            def step(self):
                raise RuntimeError("native error")

        native = SimpleNamespace(
            model_identity="model", start_window=lambda **kw: Raw()
        )
        measured = experiment._MeasuredAdapter(native)
        for reuse in (False, True):
            measured.reuse = reuse
            run = measured.start_window(window_id=str(reuse), start_ms=0, end_ms=1000)
            self.assertEqual(run.prepare_word_alignment(), "words")
            self.assertTrue(run.complete)
            with self.assertRaises(RuntimeError):
                run.step()
            self.assertIsNone(measured.active)
            run.close()
            self.assertTrue(run.record["closed"])
            self.assertTrue(run.record["capacity_restored"])
        self.assertEqual(calls, [False, True])
        measured.count = experiment.MAX_NATIVE_WINDOWS
        with self.assertRaises(RuntimeError):
            measured.start_window(window_id="too-many", start_ms=0, end_ms=1000)
        with self.assertRaises(RuntimeError):
            measured.observe_encoder(None, [], {})


class PacedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (experiment.ROOT / "artifacts/speech-corpus-v1").exists():
            raise unittest.SkipTest("optional local PCM assets absent")
        cls.inputs = experiment.cases()

    def test_registered_schedule_and_profile(self):
        self.assertEqual(
            [item["sample_count"] for item in self.inputs], [698560, 174240, 174240]
        )
        seconds = sum(
            next(x["duration_seconds"] for x in self.inputs if x["id"] == case_id)
            for case_id, _ in experiment.SCHEDULE
        )
        self.assertAlmostEqual(seconds, 130.88)
        self.assertNotEqual(self.inputs[1]["pcm_sha256"], self.inputs[2]["pcm_sha256"])
        config = experiment.stream_config()
        self.assertTrue(config["word_boundary_fallback"])
        self.assertTrue(config["input_evidence"])
        self.assertEqual(config["left_context_ms"], 2000)
        self.assertFalse(config.get("resolution_probe", False))
        self.assertFalse(config.get("word_context_limit_ms", 0))
        reference = experiment._corpus().read_registration()["fixtures"]
        self.assertEqual(
            self.inputs[1]["reference_text"],
            " ".join(x["reference_text"] for x in reference[:2]),
        )

    def fixture(self):
        # Reuse historical clocks/events as a validator fixture, not new GPU evidence.
        corpus = experiment._corpus()
        b, c = corpus.b, corpus.c
        path = (
            experiment.ROOT / "evidence/modal-t4-tiny-en-paced-replay-2026-09-05.json"
        )
        old = json.loads(path.read_text())["cases"][0]
        item = self.inputs[0]
        checks = b._event_checks(
            old["events"],
            old["decision_traces"],
            accepted_samples=old["metrics"]["accepted_samples"],
            total_samples=item["sample_count"],
            state=SimpleNamespace(**old["state"]),
            metrics=SimpleNamespace(**old["metrics"]),
            budget=SimpleNamespace(available=1, lease_count=0),
            worker=SimpleNamespace(queue_depth=0),
            capacity=1,
            retained_from_sample=item["sample_count"],
            max_buffer_samples=40000 * 16,
            error_record=None,
        )
        checks.update(
            experiment.shared.publication_checks(
                old["events"], old["decision_traces"], item["pcm"]
            )
        )
        checks.update(
            corpus._paced_checks(
                old["events"],
                old["decision_traces"],
                old["pacing"],
                item["sample_count"],
            )
        )
        checks["profile_identity"] = True
        self.assertTrue(all(checks.values()))

        def window(index, reuse, start=0, end=2000):
            return dict(
                call_index=index,
                window_id=str(index),
                start_ms=start,
                end_ms=end,
                reuse_alignment_features=reuse,
                gpu_device_time_ms=None,
                operation_wall_ns=dict(decode=1),
                closed=True,
                capacity_restored=True,
                encoder_calls=[dict(phase="decode", input_frames=3000, use_sdpa=None)],
            )

        warmup = [
            dict(
                arm=arm,
                sample_count=32000,
                pcm_sha256=experiment.shared._sha(item["pcm"][:64000]),
                windows=[window(i + 1, arm == "reuse")],
                capacity_restored=True,
            )
            for i, arm in enumerate(("baseline", "reuse"))
        ]
        windows = [
            window(
                i + 3,
                False,
                t["analysis_start_sample"] // 16,
                t["analysis_end_sample"] // 16,
            )
            for i, t in enumerate(old["decision_traces"])
        ]
        text = c._normalized_committed_text(old["events"])
        cell = dict(
            case_id="mixed-control",
            arm="baseline",
            windows=windows,
            events=old["events"],
            decision_traces=old["decision_traces"],
            pacing=old["pacing"],
            checks=checks,
            metrics=old["metrics"],
            state=old["state"],
            retained_from_sample=item["sample_count"],
            profile_id="word_boundary_quiet_endpoint_stream/v1+input_evidence/v1",
            stream_status="completed",
            error=None,
            capacity_restored=True,
            recognition=dict(
                text=text,
                against_human_reference=b._word_difference(
                    text, item["reference_text"]
                ),
            ),
        )
        registered = corpus.read_registration()["model"]
        return dict(
            schema_version="1-diagnostic",
            experiment_id="modal-paced-features-v1",
            source=dict(snapshot={"test": "local"}),
            status="failed",
            qualified=False,
            claim_boundary=experiment.CLAIMS,
            scope=experiment.scope(),
            inputs=[{k: v for k, v in x.items() if k != "pcm"} for x in self.inputs],
            schedule=[
                dict(case_id=case_id, arm=arm) for case_id, arm in experiment.SCHEDULE
            ],
            cells=[cell],
            warmup=warmup,
            stop=dict(reason="budget_stop"),
            native_window_count=len(windows) + 2,
            backend=dict(
                base_commit=experiment.features.BASE_COMMIT,
                base_tree=experiment.features.BASE_TREE,
                patched_tree=experiment.features.PATCHED_TREE,
                patch_sha256=experiment.features.PATCH_SHA,
            ),
            model=dict(
                initial_sha256=registered["model_state_sha256"],
                final_sha256=registered["model_state_sha256"],
                checkpoint_sha256=registered["checkpoint_sha256"],
                backend_revision=experiment.features.PATCHED_TREE,
                unchanged=True,
            ),
            capacity_restored=True,
        )

    def validate(self, record):
        with patch.object(experiment, "cases", return_value=self.inputs):
            experiment.validate_record(record, {"test": "local"})

    def test_partial_budget_record_and_recomputed_clocks(self):
        record = self.fixture()
        self.validate(record)
        changed = copy.deepcopy(record)
        changed["cells"][0]["pacing"]["admissions"][0]["end_sample"] += 1
        with self.assertRaises(ValueError):
            self.validate(changed)

    def test_identity_counts_and_claims_fail_closed(self):
        record = self.fixture()
        mutations = [
            lambda r: r.update(qualified=True),
            lambda r: r.update(status="completed"),
            lambda r: r.update(native_window_count=999),
            lambda r: r["backend"].update(patched_tree="wrong"),
            lambda r: r["model"].update(final_sha256="changed"),
            lambda r: r["scope"].update(same_model_and_lane=False),
            lambda r: r["cells"][0]["windows"][0].update(reuse_alignment_features=True),
            lambda r: r["cells"][0]["recognition"].update(text="invented"),
            lambda r: r["cells"][0]["checks"].update(ordered_events=False),
        ]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(changed)


if __name__ == "__main__":
    unittest.main()
