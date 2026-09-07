"""CPU-only gates for the composed experiment; no Modal or native model calls."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from infra import modal_composed_features as x


class Event:
    ready = False

    def __init__(self, **kwargs):
        self.stream = None

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        return 1.25

    def synchronize(self):
        raise AssertionError("per-forward synchronization forbidden")


class ComposedTests(unittest.TestCase):
    def measured(self):
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
        measured.records = [record]
        measured.active = (record, "decode")
        return measured, record, lane

    def test_real_hook_counts_stream_and_finished_fence(self):
        measured, record, lane = self.measured()
        args = [SimpleNamespace(shape=(80, 3000))]
        for kind in ("encoder", "decoder", "decoder"):
            measured.before(kind, None, args, {})
            measured.after(kind, None, args, {}, object())
        self.assertEqual(len(record["encoder_calls"]), 1)
        self.assertEqual(
            [c["kind"] for c in record["forwards"]], ["encoder", "decoder", "decoder"]
        )
        with self.assertRaisesRegex(ValueError, "finished native"):
            measured.resolve()
        record.update(closed=True, capacity_restored=True)
        lane.owner = None
        with self.assertRaisesRegex(ValueError, "did not finish"):
            measured.resolve()
        with patch.object(Event, "ready", True):
            measured.resolve()
        self.assertEqual([f["interval_ms"] for f in record["forwards"]], [1.25] * 3)
        self.assertFalse(measured.pending)

    def test_unowned_forward_and_budget_refuse_before_work(self):
        measured, record, lane = self.measured()
        lane.stream = SimpleNamespace(cuda_stream=99)
        with self.assertRaisesRegex(ValueError, "owned stream"):
            measured.before("decoder", None, [], {})
        self.assertNotIn("forwards", record)
        measured.active, measured.deadline = None, 0
        called = []
        with self.assertRaises(TimeoutError):
            measured.invoke(record, "decode", lambda: called.append(1))
        measured.invoke(record, "close", lambda: called.append(2))
        self.assertEqual(called, [2])
        with patch.object(x.time, "monotonic", return_value=220):
            self.assertFalse(x.admit(0, 61.55))
        with patch.object(x.time, "monotonic", return_value=200):
            self.assertTrue(x.admit(0, 61.55))

    def test_symmetric_configuration_and_budget(self):
        spec = x.scope()
        self.assertEqual(spec["alignment"], dict(zip(x.ARMS, (False, False, True))))
        configs = spec["configurations"]
        self.assertEqual(configs["fast-legacy"], configs["fast-reuse"])
        changed = {
            k
            for k in configs["cli-legacy"]
            if configs["cli-legacy"][k] != configs["fast-legacy"][k]
        }
        self.assertEqual(changed, {"left_context_ms", "word_context_limit_ms"})
        self.assertEqual(
            (spec["gpu_calls"], spec["retries"], spec["min_containers"]), (1, 0, 0)
        )
        self.assertLess(spec["planning_compute_usd"], 1)
        with patch.object(x, "GPU_TIMEOUT_SECONDS", 301):
            with self.assertRaisesRegex(ValueError, "budget"):
                x.scope()

    def test_preflight_freezes_bytes_and_refuses_paid_mismatch_or_retry(self):
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
                    x.run(replay_id="test", root=root)
                path = x.run(replay_id="test", preflight=True, root=root)
                self.assertEqual(json.loads(path.read_text())["source"], frozen)
                with self.assertRaises(FileExistsError):
                    x.run(replay_id="test", preflight=True, root=root)
                with patch.object(
                    x, "snapshot", return_value={**frozen, "digest": "changed"}
                ):
                    with self.assertRaisesRegex(ValueError, "frozen"):
                        x.run(replay_id="test", confirm_paid_gpu=True, root=root)
                receipt = x.shared._paths(root, "test", False)[1]
                x.live.prior._journal(receipt, "attempt-started", first=True)
                with self.assertRaisesRegex(ValueError, "no retry"):
                    x.run(replay_id="test", confirm_paid_gpu=True, root=root)
            paid.assert_not_called()

    def test_registered_pcm_and_snapshot_are_explicit(self):
        if not (x.ROOT / "artifacts/speech-corpus-v1").exists():
            self.skipTest("optional cached public PCM absent")
        pcm, plan = x.input_plan()
        self.assertEqual(len(pcm), 1489600)
        self.assertEqual(
            plan["pcm_sha256"],
            "b8902a2cfd5c45b8a17d6b489f6ddc752cf76d7b620efa27ec6e173ea6e2f826",
        )
        frozen = x.snapshot()
        names = {f["path"] for f in frozen["files"]}
        self.assertTrue({x.PRODUCER, x.TEST, x.features.PATCH} <= names)
        self.assertFalse(
            any(n.startswith(("artifacts/", ".git/", "evidence/")) for n in names)
        )
        changed = copy.deepcopy(frozen)
        changed["files"][0]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "digest"):
            x.shared.verify_snapshot(changed, x.ROOT)

    def test_resource_allowlist_limits_and_read_only_cached_model(self):
        registrations, uploads, calls = [], [], []

        class Image:
            def __getattr__(self, name):
                def fluent(*args, **kwargs):
                    if name == "add_local_file":
                        uploads.append((args, kwargs))
                    return self

                return fluent

        class App:
            def __init__(self, name):
                pass

            def function(self, **kwargs):
                registrations.append(kwargs)
                return lambda fn: fn

        volume = SimpleNamespace(
            with_mount_options=lambda **kw: calls.append(kw) or "readonly"
        )
        modal = SimpleNamespace(
            __version__="1.5.5",
            App=App,
            Image=SimpleNamespace(debian_slim=lambda **kw: Image()),
            Volume=SimpleNamespace(
                from_name=lambda *a, **kw: calls.append(kw) or volume
            ),
        )
        original = x.importlib.import_module
        with patch.object(
            x.importlib,
            "import_module",
            side_effect=lambda n: modal if n == "modal" else original(n),
        ):
            frozen = x.snapshot()
            app, echo, execute = x.resources(frozen)
        self.assertEqual(len(registrations), 2)
        cpu, gpu = registrations
        self.assertNotIn("gpu", cpu)
        self.assertEqual(
            (
                gpu["gpu"],
                gpu["timeout"],
                gpu["min_containers"],
                gpu["max_containers"],
                gpu["retries"],
            ),
            ("T4", 300, 0, 1, 0),
        )
        self.assertTrue(
            gpu["block_network"]
            and gpu["restrict_modal_access"]
            and gpu["single_use_containers"]
        )
        self.assertFalse(gpu["include_source"])
        self.assertEqual(calls, [dict(create_if_missing=False), dict(read_only=True)])
        fixture_names = {
            f["filename"] for f in x._corpus().read_registration()["fixtures"]
        }
        expected = {str(x.ROOT / f["path"]) for f in frozen["files"]}
        expected.update(str(x.ROOT / x._corpus().ASSET_PATH / n) for n in fixture_names)
        self.assertEqual({str(args[0]) for args, kw in uploads}, expected)
        self.assertTrue(all(kw == dict(copy=True) for args, kw in uploads))

    def test_partial_receipt_identity_count_and_completion_tampering(self):
        model = x._corpus().read_registration()["model"]
        from dataclasses import asdict

        from whisper_runtime import ResourceVector
        from whisper_runtime.adapters import NativeDecodeOptions, NativeExecutionProfile

        plan = dict(
            sample_count=2, reference_text="hello", pcm_sha256=x.shared._sha(b"0000")
        )
        record = dict(
            experiment_id="modal-composed-features-v1",
            source=dict(snapshot={}),
            scope=x.scope(),
            input=plan,
            claim_boundary=x.CLAIMS,
            qualified=False,
            cells=[],
            warmup=[],
            native_window_count=0,
            stop=dict(reason="budget_stop"),
            status="partial",
            capacity_restored=True,
            model=dict(
                initial_sha256=model["model_state_sha256"],
                final_sha256=model["model_state_sha256"],
                checkpoint_sha256=model["checkpoint_sha256"],
                backend_revision=x.features.PATCHED_TREE,
                loads=1,
                unchanged=True,
            ),
            backend=dict(
                base_commit=x.features.BASE_COMMIT,
                base_tree=x.features.BASE_TREE,
                patched_tree=x.features.PATCHED_TREE,
                patch_sha256=x.features.PATCH_SHA,
            ),
        )
        record["effective_identity"] = dict(
            rng_seed=7,
            decode_options=asdict(
                NativeDecodeOptions(**x._corpus().read_registration()["decode_options"])
            ),
            profile=asdict(
                NativeExecutionProfile(
                    "tiny.en/composed-v1",
                    ResourceVector(
                        memory_bytes=2147483648, compute_units=1, stream_slots=1
                    ),
                    device="cuda:0",
                    reuse_alignment_features=True,
                )
            ),
        )
        with patch.object(x, "input_plan", return_value=(b"0000", plan)):
            x.validate_record(record, {})
            for key, value in (
                ("status", "completed"),
                ("native_window_count", 1),
                ("qualified", True),
            ):
                with self.assertRaises(ValueError):
                    x.validate_record({**record, key: value}, {})
            changed = copy.deepcopy(record)
            changed["model"]["initial_sha256"] = "other"
            with self.assertRaisesRegex(ValueError, "model"):
                x.validate_record(changed, {})


if __name__ == "__main__":
    unittest.main()
