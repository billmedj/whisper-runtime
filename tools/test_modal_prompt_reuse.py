"""Local prompt-reuse evidence checks; no CUDA, model, or Modal launch."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from infra import modal_acoustic_diagnostic as shared
from infra import modal_cuda_lane as launcher
from infra import modal_prompt_reuse as experiment


def evidence():
    """Small invented observations, never passed to a worker."""

    def result(text):
        return dict(
            window_id="matched-jfk-window",
            text=text,
            start_ms=0,
            end_ms=11000,
            metadata=dict(
                tokens=[1 if text == "first" else 2],
                avg_logprob=-0.2,
                no_speech_prob=0.01,
                segments=[dict(start_ms=0, end_ms=1000)],
            ),
        )

    def arm(first, final, attempts):
        return dict(
            first_result=result(first),
            committed_result=result(final),
            encoder_forwards=1,
            decode_attempts=attempts,
            decoder_steps=[4] * attempts,
            stream_ids=[7],
            lane_stream=7,
            lane_owner=99,
            same_lease=True,
            held_before_commit=True,
            immutable_provenance=True,
            committed_final=True,
            resources_released=True,
            lane_released=True,
            closed=True,
            session_version=1,
            request_status="committed",
        )

    cells = [
        dict(
            id=profile,
            decode_options=experiment.profile_options(profile),
            prompts=list(experiment.prompt_pair(profile)),
            baseline_a=arm("first", "first", 1),
            baseline_b=arm("second", "second", 1),
            reuse_a_b=arm("first", "second", 2),
        )
        for profile in experiment.PROFILES
    ]
    cancellation = dict(
        decode_attempts=2,
        encoder_forwards=1,
        decoder_steps=[4, 0],
        stream_ids=[7],
        lane_stream=7,
        lane_owner=99,
        same_lease=True,
        held_before_cancel=True,
        cancel_observed=True,
        immutable_provenance=True,
        resources_released=True,
        lane_released=True,
        closed=True,
        session_version=0,
        request_status="cancelled",
    )
    return cells, cancellation


class PromptReuseEvidenceTests(unittest.TestCase):
    def test_archived_t4_record_preserves_exact_parity_and_cancellation(self):
        path = (
            experiment.ROOT / "evidence/modal-t4-tiny-en-prompt-reuse-2026-09-06.json"
        )
        raw = path.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "1dcdd5a4620cb10606c87082e9344bbe04c0067caaf4a495d7d0dc56f2afc195",
        )
        record = json.loads(raw)
        snapshot = record["source"]["snapshot"]
        self.assertEqual(snapshot["digest"], shared._hash(snapshot["files"]))
        experiment.validate_record(record, snapshot)
        self.assertEqual(record["summary"]["baseline_encoder_forwards"], 6)
        self.assertEqual(record["summary"]["reuse_encoder_forwards"], 3)
        self.assertEqual(record["cancellation"]["session_version"], 0)

    def test_fixed_budget_and_decode_profile_options(self):
        self.assertEqual(
            experiment.WORK_BOUNDS,
            dict(
                gpu="T4",
                gpu_calls=1,
                max_containers=1,
                retries=0,
                gpu_timeout_seconds=120,
                startup_timeout_seconds=300,
                decode_attempts=14,
                max_driver_steps_per_attempt=96,
                max_hypotheses_per_step=2,
                sample_len=96,
                numeric_precision="float32",
                model_download=False,
            ),
        )
        self.assertEqual(experiment.PROFILES, ("greedy", "beam", "sample"))
        self.assertEqual(experiment.SEED, 7)
        self.assertEqual(
            experiment.prompt_pair("greedy"), (experiment.PROMPTS[0], None)
        )
        for profile in ("beam", "sample"):
            self.assertEqual(experiment.prompt_pair(profile), experiment.PROMPTS)
        for profile in experiment.PROFILES:
            options = experiment.profile_options(profile)
            self.assertEqual(options["sample_len"], 96)
            self.assertEqual(
                options["temperature"], 0.4 if profile == "sample" else 0.0
            )
            self.assertEqual(options["beam_size"], 2 if profile == "beam" else None)
            self.assertEqual(options["best_of"], 2 if profile == "sample" else None)
        with self.assertRaises(ValueError):
            experiment.profile_options("unbounded")

    def test_summary_counts_both_candidates_and_cancellation_without_claiming_speed(
        self,
    ):
        cells, cancellation = evidence()
        summary = experiment.summarize(cells, cancellation)
        self.assertTrue(summary["all_first_results_exact"])
        self.assertTrue(summary["all_second_results_exact"])
        self.assertTrue(summary["all_lifecycle_checks"])
        self.assertEqual(summary["baseline_encoder_forwards"], 6)
        self.assertEqual(summary["reuse_encoder_forwards"], 3)
        self.assertEqual(summary["decode_attempts"], 14)
        self.assertEqual(summary["cuda_stream_count"], 1)
        self.assertTrue(all(value is False for value in experiment.CLAIMS.values()))
        cells[0]["reuse_a_b"]["wall_ns"] = 10**12
        cells[0]["baseline_b"]["wall_ns"] = 1
        self.assertEqual(experiment.summarize(cells, cancellation), summary)

    def test_changed_text_tokens_scores_timestamps_and_provenance_are_not_normalized(
        self,
    ):
        for target in ("first_result", "committed_result"):
            for field in ("text", "window_id", "tokens", "avg_logprob", "segments"):
                cells, cancellation = evidence()
                result = cells[0]["reuse_a_b"][target]
                if field in ("text", "window_id"):
                    result[field] = "changed"
                elif field == "tokens":
                    result["metadata"][field] = [99]
                elif field == "segments":
                    result["metadata"][field][0]["end_ms"] += 20
                else:
                    result["metadata"][field] = -0.3
                with self.subTest(target=target, field=field):
                    summary = experiment.summarize(cells, cancellation)
                    self.assertEqual(
                        summary["all_first_results_exact"], target != "first_result"
                    )
                    self.assertEqual(
                        summary["all_second_results_exact"],
                        target != "committed_result",
                    )

    def test_completed_lifecycle_and_lease_failures_cannot_pass(self):
        for field, value in (
            *(
                (key, False)
                for key in (
                    "same_lease",
                    "held_before_commit",
                    "immutable_provenance",
                    "committed_final",
                    "resources_released",
                    "lane_released",
                    "closed",
                )
            ),
            ("session_version", 0),
            ("request_status", "cancelled"),
            ("lane_owner", 0),
            ("stream_ids", [8]),
        ):
            cells, cancellation = evidence()
            cells[0]["reuse_a_b"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                experiment.summarize(cells, cancellation)

    def test_cancellation_requires_no_publication_and_full_cleanup(self):
        for field, value in (
            *(
                (key, False)
                for key in (
                    "same_lease",
                    "held_before_cancel",
                    "cancel_observed",
                    "immutable_provenance",
                    "resources_released",
                    "lane_released",
                    "closed",
                )
            ),
            ("session_version", 1),
            ("request_status", "committed"),
            ("lane_owner", 0),
            ("stream_ids", [8]),
            ("decoder_steps", [4, 1]),
        ):
            cells, cancellation = evidence()
            cancellation[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                experiment.summarize(cells, cancellation)

    def test_counts_and_driver_steps_enforce_attempt_budget(self):
        for field, value in (
            ("decode_attempts", 3),
            ("encoder_forwards", 2),
            ("decoder_steps", [4]),
            ("decoder_steps", [0, 4]),
            ("decoder_steps", [4, experiment.SAMPLE_LEN + 1]),
            ("decoder_steps", [True, 4]),
        ):
            cells, cancellation = evidence()
            cells[0]["reuse_a_b"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                experiment.summarize(cells, cancellation)
        for field, value in (
            ("decode_attempts", 1),
            ("encoder_forwards", 2),
            ("decoder_steps", [4, False]),
        ):
            cells, cancellation = evidence()
            cancellation[field] = value
            with self.subTest(cancel_field=field), self.assertRaises(ValueError):
                experiment.summarize(cells, cancellation)

    def test_missing_or_reordered_profiles_options_prompts_and_multiple_lanes_fail(
        self,
    ):
        for mutation in ("missing", "order", "options", "prompts", "lane"):
            cells, cancellation = evidence()
            if mutation == "missing":
                cells.pop()
            elif mutation == "order":
                cells.reverse()
            elif mutation == "options":
                cells[0]["decode_options"]["sample_len"] += 1
            elif mutation == "prompts":
                cells[0]["prompts"].reverse()
            else:
                cells[0]["reuse_a_b"].update(lane_stream=8, stream_ids=[8])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                experiment.summarize(cells, cancellation)

    def test_control_mutation_and_missing_finite_native_scores_fail(self):
        for mutation in ("control", "tokens", "nan", "infinity"):
            cells, cancellation = evidence()
            result = cells[0]["baseline_a"]["committed_result"]
            if mutation == "control":
                result["text"] = "changed"
            elif mutation == "tokens":
                result["metadata"]["tokens"] = [True]
            else:
                result["metadata"]["avg_logprob"] = float(
                    "nan" if mutation == "nan" else "inf"
                )
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                experiment.summarize(cells, cancellation)


class PromptReuseBindingTests(unittest.TestCase):
    def setUp(self):
        cells, cancellation = evidence()
        self.expected = {"digest": "expected-source"}
        self.fixture = {"id": "invented-fixture", "sha256": "audio"}
        model = {"checkpoint_sha256": "checkpoint", "model_state_sha256": "state"}
        self.record = dict(
            schema_version="1-diagnostic",
            status="completed",
            qualified=False,
            error=None,
            claim_boundary=copy.deepcopy(experiment.CLAIMS),
            work_bounds=copy.deepcopy(experiment.WORK_BOUNDS),
            input=self.fixture,
            configuration=dict(
                rng_seed=7,
                prompts={
                    profile: list(experiment.prompt_pair(profile))
                    for profile in experiment.PROFILES
                },
            ),
            backend=dict(
                commit=experiment.BACKEND_COMMIT,
                tree=experiment.BACKEND_TREE,
                clean=True,
            ),
            model=dict(
                checkpoint_sha256="checkpoint",
                initial_sha256="state",
                final_sha256="state",
                unchanged=True,
            ),
            environment=dict(
                torch="2.6.0",
                cuda_devices=1,
                gpu="Tesla T4",
                device="cuda:0",
                precision="float32",
            ),
            cells=cells,
            cancellation=cancellation,
            summary=experiment.summarize(cells, cancellation),
            source={"snapshot": copy.deepcopy(self.expected)},
        )
        corpus = SimpleNamespace(read_registration=Mock(return_value={"model": model}))
        for mocked in (
            patch.object(experiment, "input_record", return_value=self.fixture),
            patch.object(experiment, "_corpus", return_value=corpus),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_complete_bound_record_replays_without_qualifying_general_claims(self):
        experiment.validate_record(self.record, self.expected)

    def test_false_summary_and_changed_source_model_input_or_budget_fail(self):
        for path, value in (
            (("source", "snapshot", "digest"), "different"),
            (("claim_boundary", "general_speedup"), True),
            (("work_bounds", "decode_attempts"), 15),
            (("work_bounds", "gpu_timeout_seconds"), 180),
            (("input", "sha256"), "different"),
            (("configuration", "rng_seed"), 8),
            (("configuration", "prompts", "greedy"), [None, experiment.PROMPTS[0]]),
            (("backend", "tree"), "different"),
            (("backend", "clean"), False),
            (("model", "checkpoint_sha256"), "different"),
            (("model", "initial_sha256"), "different"),
            (("model", "final_sha256"), "different"),
            (("model", "unchanged"), False),
            (("environment", "gpu"), "A100"),
            (("environment", "precision"), "float16"),
            (("summary", "decode_attempts"), 12),
            (("qualified",), True),
            (("status",), "failed"),
            (("error",), "failure"),
        ):
            record = copy.deepcopy(self.record)
            parent = record
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                experiment.validate_record(record, self.expected)

    def test_recomputed_summary_cannot_turn_real_parity_failure_into_success(self):
        record = self.record
        record["cells"][0]["reuse_a_b"]["committed_result"]["metadata"]["tokens"] = [99]
        record["summary"] = experiment.summarize(
            record["cells"], record["cancellation"]
        )
        with self.assertRaisesRegex(ValueError, "parity"):
            experiment.validate_record(record, self.expected)


class PromptReuseSnapshotTests(unittest.TestCase):
    def test_snapshot_hashes_working_bytes_and_declares_allowed_dirty_source(self):
        corpus = SimpleNamespace(
            MANIFEST_PATH="manifest.json",
            PRODUCER_PATH="corpus.py",
            HELPER_PATHS=("helper.py",),
        )
        names = {
            experiment.PRODUCER,
            shared.PRODUCER,
            shared.MANIFEST,
            *shared.BUILDERS,
            corpus.MANIFEST_PATH,
            corpus.PRODUCER_PATH,
            *corpus.HELPER_PATHS,
            "infra/modal_cuda_lane.py",
            "infra/modal_native_cuda_qualification.py",
            "infra/native_cuda_trace.py",
            "conformance/audio-manifest.json",
            "src/whisper_runtime/adapters/native_whisper.py",
        }
        native_path = "src/whisper_runtime/adapters/native_whisper.py"

        def git(_root, *args):
            if args[0] == "diff":
                return native_path
            if args[0] == "ls-files":
                return experiment.PRODUCER
            return "tracked-tree" if args[-1] == "HEAD^{tree}" else "commit"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            with (
                patch.object(experiment, "_corpus", return_value=corpus),
                patch.object(experiment, "_git", side_effect=git),
            ):
                first = experiment.snapshot(root)
                (root / native_path).write_bytes(b"changed allowed implementation")
                second = experiment.snapshot(root)
            self.assertEqual(
                first["dirty_source_paths"], sorted((experiment.PRODUCER, native_path))
            )
            self.assertEqual(first["tracked_tree"], "tracked-tree")
            self.assertEqual({item["path"] for item in first["files"]}, names)
            self.assertEqual(second["digest"], shared._hash(second["files"]))
            self.assertNotEqual(first["digest"], second["digest"])
            shared.verify_snapshot(second, root)
            with self.assertRaises(ValueError):
                shared.verify_snapshot(first, root)
            with (
                patch.object(experiment, "_corpus", return_value=corpus),
                patch.object(
                    experiment, "_git", return_value="src/whisper_runtime/unreviewed.py"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unreviewed dirty"):
                    experiment.snapshot(root)


class PromptReuseBudgetTests(unittest.TestCase):
    def test_invalid_timeout_is_rejected_before_import_or_resource_lookup(self):
        for timeout in (None, True, False, 0, -1, 181, 120.0, "120"):
            with (
                self.subTest(timeout=timeout),
                patch.object(shared, "_corpus") as corpus,
                patch.object(shared.importlib, "import_module") as importing,
                self.assertRaises(ValueError),
            ):
                shared.resources({}, timeout_seconds=timeout)
            corpus.assert_not_called()
            importing.assert_not_called()

    def test_resource_definition_bounds_gpu_and_reuses_read_only_model_cache(self):
        image = Mock()
        for method in (
            "apt_install",
            "pip_install",
            "uv_pip_install",
            "run_commands",
            "add_local_dir",
            "add_local_file",
            "env",
        ):
            getattr(image, method).return_value = image
        definitions = []

        def define(**options):
            definitions.append(options)
            return lambda function: function

        app = SimpleNamespace(function=define)
        volume = Mock()
        modal = SimpleNamespace(
            __version__="1.5.5",
            Image=SimpleNamespace(debian_slim=Mock(return_value=image)),
            App=Mock(return_value=app),
            Volume=SimpleNamespace(from_name=Mock(return_value=volume)),
        )
        corpus = SimpleNamespace(
            _helper=Mock(
                return_value=SimpleNamespace(
                    DIRECT_IMAGE_PACKAGES=(),
                    _build_command=Mock(return_value="build"),
                )
            ),
            BASE_COMMIT="registered-base",
            read_registration=Mock(return_value={"fixtures": []}),
            b=SimpleNamespace(
                MODEL_CACHE_NAME="existing-cache", MODEL_CACHE_MOUNT="/models"
            ),
        )
        with (
            patch.object(shared, "_corpus", return_value=corpus),
            patch.object(shared.importlib, "import_module", return_value=modal),
        ):
            returned_app, _, _ = shared.resources(
                {"files": []},
                timeout_seconds=120,
                worker_module="infra.modal_prompt_reuse",
            )
        self.assertIs(returned_app, app)
        self.assertEqual(len(definitions), 2)
        cpu, gpu = definitions
        self.assertNotIn("gpu", cpu)
        self.assertEqual(cpu["timeout"], 30)
        self.assertEqual(gpu["gpu"], "T4")
        self.assertEqual(gpu["timeout"], 120)
        for options in definitions:
            self.assertEqual(options["retries"], 0)
            self.assertEqual(options["max_containers"], 1)
            self.assertEqual(options["min_containers"], 0)
            self.assertIs(options["block_network"], True)
            self.assertIs(options["restrict_modal_access"], True)
            self.assertIs(options["single_use_containers"], True)
        modal.Volume.from_name.assert_called_once_with(
            "existing-cache", create_if_missing=False
        )
        volume.with_mount_options.assert_called_once_with(read_only=True)
        self.assertEqual(
            gpu["volumes"], {"/models": volume.with_mount_options.return_value}
        )

    def test_paid_permission_requires_literal_true_before_source_or_resources(self):
        producer = SimpleNamespace(snapshot=Mock())
        for permission in (False, None, 1, "yes"):
            with (
                self.subTest(permission=permission),
                patch.object(shared, "resources") as resources,
                self.assertRaises(ValueError),
            ):
                launcher.run(
                    replay_id="not-authorized",
                    confirm_paid_gpu=permission,
                    producer=producer,
                )
            producer.snapshot.assert_not_called()
            resources.assert_not_called()


class PromptReuseLaunchTests(unittest.TestCase):
    def assert_mock_launch(self, failure=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / experiment.PRODUCER
            source.parent.mkdir()
            source.write_bytes(b"# exact dispatched producer\n")
            expected = {"digest": "bound-snapshot"}
            record = {"status": "completed", "source": {"snapshot": expected}}
            backend = SimpleNamespace(
                _transport_probe_payload=Mock(return_value=b"probe"),
                _write_bytes_exclusive=lambda path, data: path.write_bytes(data),
                _decode_worker_record=Mock(return_value=record),
            )
            corpus = SimpleNamespace(
                b=backend,
                c=SimpleNamespace(
                    _write_json_exclusive=lambda path, value: path.write_text(
                        json.dumps(value), encoding="utf-8"
                    )
                ),
            )
            context = MagicMock()
            context.__exit__.return_value = False
            app = SimpleNamespace(
                app_id="ap-local-only", run=Mock(return_value=context)
            )
            echo = SimpleNamespace(
                remote=Mock(return_value=b"bad" if failure == "preflight" else b"probe")
            )
            execute = SimpleNamespace(remote=Mock(return_value=b"raw evidence"))
            if failure == "cancel":
                execute.remote.side_effect = KeyboardInterrupt("local cancellation")
            elif failure == "execute":
                execute.remote.side_effect = RuntimeError("local execution failure")
            with (
                patch.object(experiment, "snapshot", return_value=expected) as snapshot,
                patch.object(experiment, "_corpus", return_value=corpus),
                patch.object(
                    experiment,
                    "validate_record",
                    side_effect=(
                        ValueError("invalid evidence")
                        if failure == "validate"
                        else None
                    ),
                ) as validate,
                patch.object(
                    shared, "resources", return_value=(app, echo, execute)
                ) as resources,
            ):
                arguments = dict(
                    replay_id="prompt-reuse-local", confirm_paid_gpu=True, root=root
                )
                if failure:
                    error = (
                        KeyboardInterrupt
                        if failure == "cancel"
                        else ValueError
                        if failure == "validate"
                        else RuntimeError
                    )
                    with self.assertRaises(error):
                        experiment.run(**arguments)
                else:
                    output = experiment.run(**arguments)
                    self.assertEqual(
                        json.loads(output.read_bytes())["app_id"], app.app_id
                    )
                snapshot.assert_called_once_with(root)
                resources.assert_called_once_with(
                    expected,
                    root,
                    worker_module="infra.modal_prompt_reuse",
                    timeout_seconds=120,
                )
                app.run.assert_called_once_with(detach=False)
                context.__enter__.assert_called_once_with()
                context.__exit__.assert_called_once()
                echo.remote.assert_called_once_with(b"probe")
                self.assertEqual(
                    execute.remote.call_count, 0 if failure == "preflight" else 1
                )
                output, receipt, raw = shared._paths(
                    root, arguments["replay_id"], False
                )
                returned_payload = failure in (None, "validate")
                self.assertEqual(raw.exists(), returned_payload)
                if returned_payload:
                    self.assertEqual(raw.read_bytes(), b"raw evidence")
                    backend._decode_worker_record.assert_called_once_with(
                        b"raw evidence",
                        expected_snapshot=expected,
                        registration_sha256=shared._sha(source.read_bytes()),
                        manifest={"claim_boundary": experiment.CLAIMS},
                    )
                    validate.assert_called_once_with(record, expected)
                else:
                    validate.assert_not_called()
                self.assertEqual(output.exists(), failure is None)
                entries = [
                    json.loads(line) for line in receipt.read_text().splitlines()
                ]
                self.assertEqual(
                    entries[0],
                    {"event": "attempt-started", "source_digest": expected["digest"]},
                )
                self.assertEqual(
                    entries[-1]["event"],
                    "attempt-failed" if failure else "record-written",
                )
                before_retry = receipt.read_bytes()
                with self.assertRaises(FileExistsError):
                    experiment.run(**arguments)
                self.assertEqual(receipt.read_bytes(), before_retry)
                self.assertEqual(resources.call_count, 1)

    def test_selected_producer_timeout_source_binding_and_success_receipt(self):
        self.assert_mock_launch()

    def test_failed_preflight_never_calls_gpu_and_blocks_retry(self):
        self.assert_mock_launch("preflight")

    def test_validation_failure_preserves_raw_evidence_without_success_record(self):
        self.assert_mock_launch("validate")

    def test_execution_failure_and_cancellation_exit_app_and_block_retry(self):
        for failure in ("execute", "cancel"):
            with self.subTest(failure=failure):
                self.assert_mock_launch(failure)


if __name__ == "__main__":
    unittest.main()
