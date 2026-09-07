"""CPU-only guards for the integrated, order-balanced draft diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from infra import modal_integrated_draft as x
from tools.test_modal_composed_features import Event


class IntegratedDraftTests(unittest.TestCase):
    def test_import_is_local_without_modal_torch_or_backend(self):
        code = (
            "import sys; import infra.modal_integrated_draft; "
            "assert not {'modal', 'torch', 'whisper'} & set(sys.modules)"
        )
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=os.pathsep.join((str(x.ROOT / "src"), str(x.ROOT))),
            WHISPER_MODAL_ENABLE_WORD_CORPUS="0",
            WHISPER_MODAL_ENABLE_REMOTE_RESOURCES="0",
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=x.ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_scope_keeps_one_gpu_and_only_draft_config_changes(self):
        spec = x.scope()
        self.assertEqual(
            x.SCHEDULE,
            (
                ("a1", "fast-reuse"),
                ("b1", "fast-draft32"),
                ("b2", "fast-draft32"),
                ("a2", "fast-reuse"),
            ),
        )
        self.assertEqual(spec["schedule"], [list(item) for item in x.SCHEDULE])
        self.assertEqual(
            tuple(
                spec[k]
                for k in (
                    "gpu_calls",
                    "timeout_seconds",
                    "reserve_seconds",
                    "maximum_native_windows",
                    "retries",
                    "min_containers",
                    "max_containers",
                )
            ),
            (1, 480, 20, 256, 0, 0, 1),
        )
        self.assertLess(spec["planning_compute_usd"], 0.18)
        self.assertFalse(spec["model_download"])
        self.assertFalse(spec["backend_method_patching"])
        self.assertEqual(spec["inference_owners"], 1)
        configs = [asdict(c) for c in x.configurations().values()]
        self.assertEqual(
            {key for key in configs[0] if configs[0][key] != configs[1][key]},
            {"max_draft_tokens"},
        )
        self.assertEqual([c["max_draft_tokens"] for c in configs], [0, 32])
        self.assertTrue(all(c["left_context_ms"] == 20000 for c in configs))
        self.assertTrue(all(c["word_context_limit_ms"] == 24000 for c in configs))
        self.assertFalse(any(x.CLAIMS.values()))

    def test_admission_preserves_cleanup_reserve_at_exact_boundary(self):
        for change in (479, 481):
            with patch.object(x, "GPU_TIMEOUT_SECONDS", change):
                with self.assertRaisesRegex(ValueError, "bound"):
                    x.scope()
        for change in (19, 21):
            with patch.object(x, "RESERVE_SECONDS", change):
                with self.assertRaisesRegex(ValueError, "bound"):
                    x.scope()
        # 93.1 s PCM + registered 15 s drain + 20 s cleanup reserve.
        with patch.object(x.time, "monotonic", return_value=451.9):
            self.assertFalse(x.admit(100, 108.1))
        with patch.object(x.time, "monotonic", return_value=451.8):
            self.assertTrue(x.admit(100, 108.1))

    def test_input_is_exactly_two_registered_cycles_with_shifted_spans(self):
        if not (x.ROOT / "artifacts/speech-corpus-v1").exists():
            self.skipTest("optional cached public PCM absent")
        pcm, plan = x.input_plan()
        seed, original = x.composed.input_plan()
        self.assertEqual(pcm, seed * 2)
        self.assertEqual(plan["sample_count"], 1489600)
        self.assertEqual(plan["duration_seconds"], 93.1)
        self.assertEqual(plan["cycles"], 2)
        self.assertEqual(plan["cycle_samples"], 744800)
        self.assertEqual(
            plan["pcm_sha256"],
            "ccdf47836c9027ad4be305f3484a45c1a8c49c4a432eed6cfa4ad484ed6c8883",
        )
        self.assertEqual(plan["pcm_sha256"], hashlib.sha256(pcm).hexdigest())
        self.assertEqual(plan["cycle_sha256"], original["pcm_sha256"])
        self.assertEqual(
            plan["reference_text"], " ".join([original["reference_text"]] * 2)
        )
        for span in plan["spans"]:
            source = original["spans"][span["kind"]]
            for key in ("start_sample", "end_sample"):
                self.assertEqual(span[key], source[key] + span["cycle"] * 744800)

    def measured(self, raw=None):
        stream = SimpleNamespace(cuda_stream=17)
        lane = SimpleNamespace(owner=object(), stream=stream)
        native = SimpleNamespace(
            model_identity="model",
            _model_binding=SimpleNamespace(_cuda_lane=lane),
            start_window=lambda **kwargs: raw,
        )
        cuda = SimpleNamespace(Event=Event, current_stream=lambda _: stream)
        measured = x.Measured(native, cuda, float("inf"))
        measured.reuse = True
        return measured, lane

    def test_native_window_bound_is_256_not_inherited_128(self):
        raw = SimpleNamespace(closed=True, capacity_released=True)
        measured, _ = self.measured(raw)
        for before in (128, 255):
            measured.count = before
            run = measured.start_window(
                window_id=f"w-{before}",
                start_ms=0,
                end_ms=2000,
                draft_tokens=(1, 2),
            )
            self.assertIs(run.raw, raw)
            self.assertEqual(run.record["call_index"], before + 1)
            self.assertEqual(run.record["draft_tokens"], [1, 2])
            self.assertTrue(run.record["reuse_alignment_features"])
        count = len(measured.records)
        with self.assertRaisesRegex(ValueError, "window bound"):
            measured.start_window(window_id="over", start_ms=0, end_ms=2000)
        self.assertEqual(len(measured.records), count)
        self.assertIsNone(measured.active)

    def test_result_observes_native_metadata_and_copies_wrapper_stats(self):
        metadata = SimpleNamespace(
            tokens=(1, 2),
            language="en",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
            temperature=0.0,
            compression_ratio=1.0,
        )
        result = SimpleNamespace(text="hello", metadata=metadata)
        inference = SimpleNamespace(
            stats={"matched_tokens": 1},
            rows=object(),
            audio_features=object(),
            draft=(1, 2),
            original=SimpleNamespace(kv_cache={1: 2}, hooks=[1]),
        )

        class Raw:
            closed = capacity_released = False
            _backend_run = SimpleNamespace(inference=inference)

            def prepare_result(self):
                return result

            def close(self):
                inference.original.kv_cache.clear()
                inference.original.hooks.clear()
                inference.rows = inference.audio_features = None
                inference.draft = ()
                self._backend_run = None
                self.closed = self.capacity_released = True
                return True

        measured, _ = self.measured(Raw())
        with patch.object(
            x.prior, "DraftHook", side_effect=AssertionError("legacy hook")
        ):
            run = measured.start_window(window_id="native", start_ms=0, end_ms=2000)
            self.assertIs(run.prepare_result(), result)
            observed = run.record["decode_observation"]
            self.assertEqual(observed["result"]["tokens"], (1, 2))
            self.assertEqual(observed["result"]["text"], "hello")
            self.assertEqual(observed["stats"], {"matched_tokens": 1})
            inference.stats["matched_tokens"] = 2
            self.assertEqual(observed["stats"], {"matched_tokens": 1})
            self.assertTrue(run.close())
        self.assertTrue(all(run.record["native_cleanup"].values()))
        self.assertTrue(run.record["closed"] and run.record["capacity_restored"])
        self.assertIsNone(measured.active)

    def test_native_observation_error_restores_scope_and_allows_close(self):
        class Raw:
            closed = capacity_released = False

            def prepare_result(self):
                return SimpleNamespace(metadata=None)

            def close(self):
                self.closed = self.capacity_released = True

        measured, _ = self.measured(Raw())
        run = measured.start_window(window_id="bad", start_ms=0, end_ms=2000)
        with self.assertRaisesRegex(ValueError, "metadata"):
            run.prepare_result()
        self.assertIsNone(measured.active)
        self.assertNotIn("decode_observation", run.record)
        measured.deadline = 0
        with self.assertRaises(TimeoutError):
            run.prepare_result()
        run.close()
        self.assertTrue(run.record["closed"])

    def test_forward_tokens_wait_for_native_fence_without_synchronization(self):
        measured, lane = self.measured()
        record = dict(
            encoder_calls=[],
            operation_wall_ns={},
            closed=False,
            capacity_restored=False,
        )
        measured.records, measured.active = [record], (record, "decode")
        for count in (35, 1):
            args = (SimpleNamespace(shape=(1, count)),)
            measured.before("decoder", None, args, {})
            measured.after("decoder", None, args, {}, None)
        self.assertEqual([f["input_tokens"] for f in record["forwards"]], [35, 1])
        self.assertTrue(all(f["interval_ms"] is None for f in record["forwards"]))
        with self.assertRaisesRegex(ValueError, "finished native"):
            measured.resolve()
        record.update(closed=True, capacity_restored=True)
        lane.owner = None
        with patch.object(Event, "ready", True):
            measured.resolve()
        self.assertEqual([f["interval_ms"] for f in record["forwards"]], [1.25, 1.25])
        self.assertFalse(measured.pending)

    def cells(self):
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
        template = dict(
            stream_status="completed",
            draft_observations=[observation],
            windows=[dict(start_ms=0, end_ms=2000)],
            decision_traces=[trace],
            checks=dict(done=True),
            events=[dict(kind="final", sequence_number=1)],
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
        cells = []
        for position, arm in x.SCHEDULE:
            cell = copy.deepcopy(template)
            cell.update(position=position, arm=arm)
            if arm == "fast-draft32":
                cell["summary"].update(
                    forwards=dict(decoder=8),
                    decoder_input_tokens=15,
                    forward_interval_ms=dict(encoder=1.0, decoder=4.0),
                    operation_wall_ns=dict(decode=90),
                    total_operation_wall_ns=90,
                )
            cells.append(cell)
        return cells

    def test_abba_analysis_keeps_two_pairs_and_separates_efficiency(self):
        cells = self.cells()
        result = x.analyze(cells)
        self.assertTrue(result["accepted"] and result["efficiency_gate"])
        self.assertEqual(len(result["pairs"]), 2)
        self.assertEqual(result["pooled_totals"]["fast-reuse"]["decode_wall_ns"], 200)
        self.assertEqual(result["pooled_totals"]["fast-draft32"]["decode_wall_ns"], 180)
        self.assertTrue(
            all(p["decoder_input_tokens_delta"] == 3 for p in result["pairs"])
        )
        cells[2]["summary"]["operation_wall_ns"]["decode"] = 111
        result = x.analyze(cells)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["pooled_faster"]["decode_wall_ns"])
        self.assertFalse(result["efficiency_gate"])

    def test_numeric_drift_and_admission_positions_are_reported_not_hidden(self):
        cells = self.cells()
        cells[2]["draft_observations"][0]["result"]["avg_logprob"] += 0.001
        cells[2]["pacing"]["publication_inputs"][0]["accepted_samples"] += 320
        result = x.analyze(cells)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["pairs"][1]["publication_input_positions_exact"])
        self.assertNotEqual(
            result["pairs"][1]["numeric_deltas"][0]["deltas"]["avg_logprob"],
            0,
        )

    def test_either_order_or_full_event_drift_blocks_parity_and_efficiency(self):
        for field in (
            "commit",
            "tokens",
            "reason",
            "source",
            "window",
            "checks",
            "event",
        ):
            cells = self.cells()
            candidate = cells[2]
            if field == "commit":
                candidate["summary"]["commits"][0]["end_sample"] += 1
            elif field == "tokens":
                candidate["draft_observations"][0]["result"]["tokens"] = [1, 3]
            elif field == "reason":
                candidate["decision_traces"][0]["reason"] = "changed"
            elif field == "source":
                candidate["decision_traces"][0]["analysis_end_sample"] += 320
            elif field == "window":
                candidate["windows"][0]["start_ms"] += 1
            elif field == "checks":
                candidate["checks"]["done"] = False
            else:
                candidate["events"][0]["sequence_number"] += 1
            with self.subTest(field=field):
                result = x.analyze(cells)
                self.assertFalse(result["accepted"])
                self.assertFalse(result["efficiency_gate"])
        cells = self.cells()
        for length in range(4):
            self.assertEqual(
                x.analyze(cells[:length]), dict(complete=False, accepted=False)
            )
        cells[1]["stream_status"] = "failed"
        self.assertEqual(x.analyze(cells), dict(complete=False, accepted=False))

    def test_preflight_freezes_bytes_without_launch_or_retry(self):
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
                for permission in (False, None, 1, "yes"):
                    with self.assertRaisesRegex(ValueError, "confirm-paid"):
                        x.run(
                            replay_id="integrated-test",
                            root=root,
                            confirm_paid_gpu=permission,
                        )
                path = x.run(replay_id="integrated-test", preflight=True, root=root)
                self.assertEqual(json.loads(path.read_text())["source"], frozen)
                with self.assertRaises(FileExistsError):
                    x.run(replay_id="integrated-test", preflight=True, root=root)
                with patch.object(
                    x, "snapshot", return_value={**frozen, "digest": "changed"}
                ):
                    with self.assertRaisesRegex(ValueError, "frozen"):
                        x.run(
                            replay_id="integrated-test",
                            confirm_paid_gpu=True,
                            root=root,
                        )
                receipt = x.shared._paths(root, "integrated-test", False)[1]
                x.live.prior._journal(receipt, "attempt-started", first=True)
                before = receipt.read_bytes()
                with self.assertRaisesRegex(ValueError, "no retry"):
                    x.run(replay_id="integrated-test", confirm_paid_gpu=True, root=root)
                self.assertEqual(receipt.read_bytes(), before)
            paid.assert_not_called()

    def test_resources_are_allowlisted_one_gpu_network_blocked_and_read_only(self):
        registrations, uploads, volume_calls = [], [], []

        class Image:
            mounted = False

            def __getattr__(self, name):
                def fluent(*args, **kwargs):
                    if name == "add_local_file":
                        self.mounted = True
                        uploads.append((args, kwargs))
                    elif self.mounted:
                        raise AssertionError("build layer after runtime mount")
                    return self

                return fluent

        class App:
            def __init__(self, name):
                pass

            def function(self, **kwargs):
                registrations.append(kwargs)
                return lambda function: function

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
        importing = x.importlib.import_module
        with patch.object(
            x.importlib,
            "import_module",
            side_effect=lambda name: modal if name == "modal" else importing(name),
        ):
            frozen = x.snapshot()
            x.resources(frozen)
        self.assertEqual(len(registrations), 2)
        cpu, gpu = registrations
        self.assertNotIn("gpu", cpu)
        self.assertEqual((gpu["gpu"], gpu["timeout"], gpu["retries"]), ("T4", 480, 0))
        for options in registrations:
            self.assertEqual(
                (options["min_containers"], options["max_containers"]), (0, 1)
            )
            self.assertTrue(
                options["block_network"]
                and options["restrict_modal_access"]
                and options["single_use_containers"]
            )
            self.assertFalse(options["include_source"])
        self.assertEqual(
            volume_calls, [dict(create_if_missing=False), dict(read_only=True)]
        )
        expected = {str(x.ROOT / item["path"]) for item in frozen["files"]}
        expected.update(
            str(x.ROOT / x._corpus().ASSET_PATH / fixture["filename"])
            for fixture in x._corpus().read_registration()["fixtures"]
        )
        self.assertEqual({str(args[0]) for args, _ in uploads}, expected)
        self.assertTrue(all(kwargs == dict(copy=False) for _, kwargs in uploads))
        self.assertTrue(
            all(
                str(args[1]).startswith("/") and "\\" not in str(args[1])
                for args, _ in uploads
            )
        )

    def test_snapshot_includes_integrated_source_and_rejects_changed_digest(self):
        frozen = x.snapshot()
        names = {item["path"] for item in frozen["files"]}
        self.assertTrue(
            {
                x.PRODUCER,
                x.TEST,
                x.PREREGISTRATION,
                "src/whisper_runtime/adapters/_draft_inference.py",
            }
            <= names
        )
        self.assertFalse(
            any(name.startswith((".git/", "artifacts/", "evidence/")) for name in names)
        )
        changed = copy.deepcopy(frozen)
        changed["files"][0]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "digest"):
            x.shared.verify_snapshot(changed, x.ROOT)


if __name__ == "__main__":
    unittest.main()
