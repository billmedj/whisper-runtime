"""Local-only bounds, identity and comparison tests for the paired T4 probe."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from infra import modal_draft_features as x
from tools.test_modal_composed_features import Event


class DraftGpuTests(unittest.TestCase):
    def test_scope_keeps_matched_options_and_bounded_cost(self):
        spec = x.scope()
        self.assertEqual(
            spec["configurations"][x.ARMS[0]], spec["configurations"][x.ARMS[1]]
        )
        self.assertEqual(spec["draft_tokens"], dict(zip(x.ARMS, (0, 32))))
        self.assertEqual(spec["alignment_reuse"], dict.fromkeys(x.ARMS, True))
        self.assertEqual(
            (spec["gpu_calls"], spec["timeout_seconds"], spec["reserve_seconds"]),
            (1, 240, 20),
        )
        self.assertEqual(
            (spec["retries"], spec["min_containers"], spec["max_containers"]), (0, 0, 1)
        )
        self.assertLess(spec["planning_compute_usd"], 0.09)
        self.assertEqual(
            (spec["warmup_samples"], spec["warmup_windows_per_arm"]), (128000, 2)
        )
        with patch.object(x, "GPU_TIMEOUT_SECONDS", 241):
            with self.assertRaisesRegex(ValueError, "budget"):
                x.scope()
        with patch.object(x.time, "monotonic", return_value=158.45):
            self.assertFalse(x.admit(0, 61.55))
        with patch.object(x.time, "monotonic", return_value=158):
            self.assertTrue(x.admit(0, 61.55))

    def test_token_positions_and_forward_intervals_are_separate(self):
        stream = SimpleNamespace(cuda_stream=17)
        lane = SimpleNamespace(owner=object(), stream=stream)
        native = SimpleNamespace(
            model_identity="model", _model_binding=SimpleNamespace(_cuda_lane=lane)
        )
        cuda = SimpleNamespace(Event=Event, current_stream=lambda _: stream)
        measured = x.Measured(native, cuda, float("inf"))
        record = dict(
            encoder_calls=[],
            operation_wall_ns={},
            closed=False,
            capacity_restored=False,
        )
        measured.records, measured.active = [record], (record, "decode")
        for size in (35, 1):
            args = (SimpleNamespace(shape=(1, size)),)
            measured.before("decoder", None, args, {})
            measured.after("decoder", None, args, {}, None)
        self.assertEqual([c["input_tokens"] for c in record["forwards"]], [35, 1])
        self.assertTrue(all(c["interval_ms"] is None for c in record["forwards"]))
        record.update(closed=True, capacity_restored=True)
        lane.owner = None
        with patch.object(Event, "ready", True):
            measured.resolve()
        self.assertEqual([c["interval_ms"] for c in record["forwards"]], [1.25, 1.25])

    def test_hook_reset_and_start_failure_clean_the_unbound_run(self):
        run = SimpleNamespace(inference=object(), cleanup_calls=0)

        def cleanup():
            run.cleanup_calls += 1

        run.cleanup = cleanup

        class Task:
            options = SimpleNamespace(
                temperature=0,
                beam_size=None,
                best_of=None,
                language="en",
                fp16=False,
                without_timestamps=False,
            )

            def _start_run(self, mel):
                return run

        class Run:
            def finalize(self):
                return []

        module = SimpleNamespace(DecodingTask=Task, _DecodingRun=Run)
        original = Task._start_run
        with patch.object(x.importlib, "import_module", return_value=module):
            with x.DraftHook(SimpleNamespace(active=None)) as hook:
                hook.reset(True)
                hook.draft = (1, 2)
                with patch.object(
                    x,
                    "StreamDraftInference",
                    side_effect=RuntimeError("construction failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "construction"):
                        Task()._start_run(None)
                self.assertEqual(run.cleanup_calls, 1)
                hook.reset(False)
                self.assertFalse(hook.draft)
                self.assertFalse(hook.observations)
                self.assertIsNone(hook.last_inference)
        self.assertIs(Task._start_run, original)

    def paired(self):
        observation = dict(
            start_ms=0,
            end_ms=2000,
            result=dict(
                tokens=[1, 2],
                avg_logprob=-0.2,
                no_speech_prob=0.01,
                compression_ratio=1.0,
            ),
        )
        trace = dict(
            action="commit",
            reason="candidate",
            analysis_start_sample=0,
            analysis_end_sample=32000,
            audio_evidence=dict(
                state="speech_candidate", reason="model_speech_candidate"
            ),
        )
        control = dict(
            arm=x.ARMS[0],
            draft_observations=[observation],
            windows=[dict(start_ms=0, end_ms=2000)],
            decision_traces=[trace],
            checks=dict(done=True),
            pacing=dict(publication_inputs=[dict(accepted_samples=32320)]),
            summary=dict(
                commits=[dict(start_sample=0, end_sample=16000, text="hello")],
                word_errors=dict(edits=0),
                forwards=dict(decoder=10),
                decoder_input_tokens=12,
                forward_interval_ms=dict(encoder=1.0, decoder=5.0),
                operation_wall_ns=dict(decode=100),
                total_operation_wall_ns=100,
            ),
        )
        candidate = copy.deepcopy(control)
        candidate["arm"] = x.ARMS[1]
        candidate["summary"].update(
            forwards=dict(decoder=8),
            decoder_input_tokens=15,
            forward_interval_ms=dict(encoder=1.0, decoder=4.0),
            operation_wall_ns=dict(decode=90),
            total_operation_wall_ns=90,
        )
        return [control, candidate]

    def test_comparison_separates_numeric_change_and_real_efficiency(self):
        cells = self.paired()
        cells[1]["draft_observations"][0]["result"]["avg_logprob"] += 0.001
        cells[1]["pacing"]["publication_inputs"][0]["accepted_samples"] += 320
        result = x.comparison(cells)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["measured_efficiency_gate"])
        self.assertFalse(result["publication_input_positions_exact"])
        self.assertNotEqual(result["numeric_deltas"][0]["deltas"]["avg_logprob"], 0)
        self.assertEqual(result["decoder_input_tokens_delta"], 3)
        cells[1]["summary"]["total_operation_wall_ns"] = 101
        self.assertFalse(x.comparison(cells)["measured_efficiency_gate"])
        cells[1]["summary"]["forward_interval_ms"]["decoder"] = 6
        self.assertFalse(
            x.comparison(cells)["measured_efficiency"]["forward_intervals"]
        )

    def test_comparison_refuses_changed_commit_tokens_policy_or_checks(self):
        for field in (
            "commit",
            "tokens",
            "reason",
            "source_wait",
            "checks",
            "native_window",
        ):
            cells = self.paired()
            candidate = cells[1]
            if field == "commit":
                candidate["summary"]["commits"][0]["end_sample"] += 1
            elif field == "tokens":
                candidate["draft_observations"][0]["result"]["tokens"] = [1, 3]
            elif field == "reason":
                candidate["decision_traces"][0]["reason"] = "different"
            elif field == "source_wait":
                candidate["decision_traces"][0]["analysis_end_sample"] += 320
            elif field == "native_window":
                candidate["windows"][0]["start_ms"] = 1
            else:
                candidate["checks"]["done"] = False
            with self.subTest(field=field):
                self.assertFalse(x.comparison(cells)["accepted"])
                self.assertFalse(x.comparison(cells)["measured_efficiency_gate"])
        self.assertEqual(x.comparison([]), dict(paired=False, accepted=False))

    def test_silence_parity_compares_provenance_not_arm_ids_or_native_scores(self):
        cells = self.paired()
        for index, cell in enumerate(cells):
            cell["decision_traces"][0]["silence_publication"] = dict(
                window_id=f"arm-{index}",
                native=dict(window_id=f"arm-{index}", score=index),
                analysis_span=dict(start_ms=0, end_ms=2000),
                analysis_start_sample=0,
                start_ms=0,
                end_ms=2000,
                text="",
                observation=dict(pcm_sha256="same", digital_silence=True),
            )
        self.assertTrue(x.comparison(cells)["accepted"])
        cells[1]["decision_traces"][0]["silence_publication"]["observation"][
            "pcm_sha256"
        ] = "changed"
        self.assertFalse(x.comparison(cells)["accepted"])

    def test_preflight_freezes_bytes_and_does_not_launch(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(x, "resources") as paid,
        ):
            root = Path(temporary)
            frozen = dict(commit="a" * 40, digest="b" * 64, files=[])
            with (
                patch.object(x, "snapshot", return_value=frozen),
                patch.object(x, "input_plan", return_value=(b"00", {})),
            ):
                with self.assertRaisesRegex(ValueError, "confirm-paid"):
                    x.run(replay_id="draft-test", root=root)
                path = x.run(replay_id="draft-test", preflight=True, root=root)
                self.assertEqual(json.loads(path.read_text())["source"], frozen)
                with self.assertRaises(FileExistsError):
                    x.run(replay_id="draft-test", preflight=True, root=root)
                with patch.object(
                    x, "snapshot", return_value={**frozen, "digest": "changed"}
                ):
                    with self.assertRaisesRegex(ValueError, "frozen"):
                        x.run(replay_id="draft-test", confirm_paid_gpu=True, root=root)
            paid.assert_not_called()

    def test_snapshot_is_explicit_and_content_checked(self):
        frozen = x.snapshot()
        names = {item["path"] for item in frozen["files"]}
        self.assertTrue(
            {
                x.PRODUCER,
                x.TEST,
                x.PREREGISTRATION,
                "tools/verify_decoder_draft.py",
                "tools/verify_stream_draft.py",
                x.features.PATCH,
            }
            <= names
        )
        self.assertFalse(
            any(n.startswith((".git/", "artifacts/", "evidence/")) for n in names)
        )
        changed = copy.deepcopy(frozen)
        changed["files"][0]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "digest"):
            x.shared.verify_snapshot(changed, x.ROOT)

    def test_resources_mount_only_allowlisted_files_after_all_build_steps(self):
        registrations, uploads, volume_calls = [], [], []

        class Image:
            mounted = False

            def __getattr__(self, name):
                def fluent(*args, **kwargs):
                    if name == "add_local_file":
                        self.mounted = True
                        uploads.append((args, kwargs))
                    else:
                        self.assert_no_late_build()
                    return self

                return fluent

            def assert_no_late_build(self):
                if self.mounted:
                    raise AssertionError("build layer after runtime mount")

        class App:
            def __init__(self, name):
                pass

            def function(self, **kwargs):
                registrations.append(kwargs)
                return lambda fn: fn

        volume = SimpleNamespace(
            with_mount_options=lambda **kw: volume_calls.append(kw) or "readonly"
        )
        modal = SimpleNamespace(
            __version__="1.5.5",
            App=App,
            Image=SimpleNamespace(debian_slim=lambda **kw: Image()),
            Volume=SimpleNamespace(
                from_name=lambda *args, **kw: volume_calls.append(kw) or volume
            ),
        )
        original = x.importlib.import_module
        with patch.object(
            x.importlib,
            "import_module",
            side_effect=lambda n: modal if n == "modal" else original(n),
        ):
            frozen = x.snapshot()
            x.resources(frozen)
        self.assertEqual(len(registrations), 2)
        cpu, gpu = registrations
        self.assertNotIn("gpu", cpu)
        self.assertEqual((gpu["gpu"], gpu["timeout"], gpu["retries"]), ("T4", 240, 0))
        self.assertEqual((gpu["min_containers"], gpu["max_containers"]), (0, 1))
        self.assertTrue(
            gpu["block_network"]
            and gpu["restrict_modal_access"]
            and gpu["single_use_containers"]
        )
        self.assertFalse(gpu["include_source"])
        self.assertEqual(
            volume_calls, [dict(create_if_missing=False), dict(read_only=True)]
        )
        expected = {str(x.ROOT / item["path"]) for item in frozen["files"]}
        expected.update(
            str(x.ROOT / x._corpus().ASSET_PATH / f["filename"])
            for f in x._corpus().read_registration()["fixtures"]
        )
        self.assertEqual({str(args[0]) for args, kwargs in uploads}, expected)
        self.assertTrue(all(kwargs == dict(copy=False) for args, kwargs in uploads))


if __name__ == "__main__":
    unittest.main()
