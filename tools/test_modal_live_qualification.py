"""CPU-only live qualification registration, gating and resource-lifetime tests."""

import copy
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from infra import modal_live_qualification as p


def stage():
    return dict(sample_count=320, sha256="source-hash", timeout_seconds=120)


def done(phase="smoke"):
    return dict(
        type="done",
        status="completed",
        source_eof_received=True,
        metrics=dict(
            accepted_samples=320,
            committed_samples=320,
            buffered_samples=0,
            accepted_sha256="source-hash",
        ),
        metadata=dict(
            capacity_restored=True,
            model_unchanged=True,
            model_loads=1,
            source_digest="frozen-source",
            instance_id="same-worker",
            session_number=1 if phase == "smoke" else 2,
            peak_buffered_samples=320,
            max_buffer_samples=640,
        ),
    )


class PlanTests(unittest.TestCase):
    def test_short_plan_pins_overrides_and_rejects_preflight_drift(self):
        corpus = SimpleNamespace(
            ASSET_PATH="unused",
            read_registration=lambda root: {
                "fixtures": [{"reference_text": "one"}, {"reference_text": "two"}],
            },
        )
        shared = SimpleNamespace(
            _cases=lambda root, path: (
                {
                    "cases": [
                        {"id": "concatenated-no-added-pauses", "reference_text": "main"}
                    ]
                },
                {
                    "concatenated-no-added-pauses": bytes(538560 * 2),
                    "mixed-continuous-noise64": bytes(174240 * 2),
                },
            )
        )
        with self.assertRaisesRegex(ValueError, "require --short-only"):
            p.input_plan(left_context_ms=20000)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(p.prior, "_corpus", return_value=corpus),
            patch.dict(sys.modules, {"infra.modal_acoustic_diagnostic": shared}),
            patch.object(p, "snapshot", return_value={"digest": "frozen"}),
            patch.object(p, "attempt") as launch,
        ):
            _, default = p.input_plan()
            self.assertEqual(set(default["stages"]), {"smoke", "long"})
            self.assertEqual(default["stages"]["long"]["sample_count"], 1800 * 16000)
            self.assertNotIn("config_overrides", default)
            options = dict(
                replay_id="short",
                root=Path(directory),
                short_only=True,
                left_context_ms=20000,
                word_context_limit_ms=24000,
            )
            path = p.run(preflight=True, **options)
            plan = json.loads(path.read_text())["input"]
            self.assertEqual(set(plan["stages"]), {"smoke"})
            self.assertFalse(path.with_name("live-v01-long.pcm").exists())
            self.assertEqual(
                plan["config_overrides"],
                {
                    "left_context_ms": 20000,
                    "word_context_limit_ms": 24000,
                },
            )
            self.assertEqual(
                plan["stages"]["smoke"]["stream_config"]["left_context_ms"], 20000
            )
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                p.run(confirm_paid_gpu=True, **{**options, "left_context_ms": 18000})
            launch.assert_not_called()
        short_budget = p.budget_plan(short_only=True)
        self.assertEqual(short_budget["outer_seconds"], 210)
        self.assertEqual(short_budget["request_timeout_seconds"], 190)
        self.assertLess(short_budget["planning_compute_usd"], 0.08)

    def test_streaming_recipe_repeats_whole_cycles_and_pads_final_silence(self):
        seed = b"\x01\x00" * 337
        frames = list(p.frame_source(seed, 337 * 2 + 7))
        self.assertEqual(b"".join(frames), seed * 2 + bytes(14))
        self.assertTrue(all(len(frame) == 640 for frame in frames[:-1]))
        self.assertTrue(all(len(frame) % 2 == 0 for frame in frames))
        self.assertEqual(b"".join(p.frame_source(b"ab", 7)), b"ab" * 7)

    def test_invalid_or_unbounded_sources_are_rejected(self):
        for seed, count in (
            (b"", 1),
            (b"a", 1),
            (b"ab", 0),
            (b"ab", True),
            (b"ab", p.LONG_SECONDS * 16000 + 1),
        ):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                list(p.frame_source(seed, count))

    def test_budget_is_a_subdollar_estimate_not_a_physical_cap(self):
        budget = p.budget_plan()
        self.assertLess(budget["planning_compute_usd"], 1)
        self.assertFalse(budget["physical_spending_cap"])
        self.assertEqual(budget["outer_seconds"], 2200)
        self.assertEqual(budget["maximum_gpu_containers"], 1)
        self.assertEqual(budget["minimum_gpu_containers"], 0)
        self.assertEqual(budget["automatic_retries"], 0)
        self.assertTrue(budget["long_requires_smoke"])

    def test_stage_gate_rejects_partial_hash_capacity_and_model_drift(self):
        self.assertTrue(p.stage_valid(done(), stage(), expected_digest="frozen-source"))
        for section, key, value in (
            (None, "status", "failed"),
            ("metrics", "committed_samples", 319),
            ("metrics", "accepted_sha256", "wrong"),
            ("metrics", "buffered_samples", 1),
            ("metadata", "capacity_restored", False),
            ("metadata", "model_unchanged", False),
            ("metadata", "model_loads", 2),
            ("metadata", "source_digest", "changed"),
        ):
            record = copy.deepcopy(done())
            (record if section is None else record[section])[key] = value
            with self.subTest(key=key):
                self.assertFalse(
                    p.stage_valid(record, stage(), expected_digest="frozen-source")
                )

    def test_preflight_is_local_and_paid_dispatch_needs_unchanged_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(stages={"smoke": stage(), "long": stage()})
            with (
                patch.object(p, "input_plan", return_value=(bytes(32000), plan)),
                patch.object(p, "snapshot", return_value={"digest": "frozen"}),
                patch.object(p, "attempt") as launch,
            ):
                path = p.run(replay_id="synthetic", preflight=True, root=root)
                self.assertEqual(
                    json.loads(path.read_text())["status"], "local-preflight-passed"
                )
                launch.assert_not_called()
                with self.assertRaisesRegex(ValueError, "confirm-paid-gpu"):
                    p.run(replay_id="synthetic", root=root)
                with (
                    patch.object(p, "snapshot", return_value={"digest": "changed"}),
                    self.assertRaisesRegex(ValueError, "no longer matches"),
                ):
                    p.run(replay_id="synthetic", confirm_paid_gpu=True, root=root)
                launch.assert_not_called()


class NativeOwnerTests(unittest.TestCase):
    @contextmanager
    def native_fixture(self, *, reuse=False):
        from whisper_runtime import native_setup as setup

        class Model:
            training = False
            device = SimpleNamespace(type="cuda", index=0)

            def parameters(self):
                return ()

            buffers = parameters

        model = Model()
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda index: "Tesla T4",
                synchronize=lambda index: None,
                memory_allocated=lambda index: 0,
            ),
            set_num_threads=Mock(),
        )

        with tempfile.TemporaryDirectory() as directory:
            backend = Path(directory).resolve() / "backend"
            (backend / "whisper").mkdir(parents=True)
            modules = {}
            for name, filename in (
                ("whisper", "__init__.py"),
                ("whisper.triton_ops", "triton_ops.py"),
            ):
                path = backend / "whisper" / filename
                path.write_text("# scripted source only\n")
                module = ModuleType(name)
                module.__file__ = str(path)
                modules[name] = module
            events = []

            def git(path, *args):
                self.assertEqual(path, backend)
                events.append("git:" + " ".join(args))
                if args == ("rev-parse", "--show-toplevel"):
                    return str(backend)
                if args == ("rev-parse", "HEAD"):
                    return "a" * 40
                if args == ("rev-parse", "HEAD^{tree}"):
                    return setup.BACKEND_REUSE_TREE if reuse else setup.BACKEND_TREE
                return ""  # Empty status is valid; no helper rejects it.

            def imported(selected, *, eager_cuda):
                self.assertEqual(selected, setup.BackendSetup(backend, "a" * 40))
                self.assertTrue(eager_cuda)
                self.assertIn("dependencies", events)
                self.assertIn("checkpoint", events)
                self.assertIn("git:rev-parse HEAD^{tree}", events)
                self.assertIn("git:status --porcelain --untracked-files=all", events)
                events.append("guarded-import")
                sys.modules.update(modules)
                return modules["whisper"]

            def loaded(whisper, checkpoint, device):
                self.assertIs(whisper, modules["whisper"])
                self.assertIs(
                    sys.modules["whisper.triton_ops"], modules["whisper.triton_ops"]
                )
                events.append("model-load")
                return model

            corpus = SimpleNamespace(
                b=SimpleNamespace(
                    BACKEND_ROOT=backend, MODEL_CHECKPOINT_PATH=Path("tiny.en.pt")
                ),
                c=SimpleNamespace(
                    _command_output=Mock(
                        side_effect=AssertionError("use guarded Git helper")
                    )
                ),
            )
            with (
                patch.dict(sys.modules, {"torch": torch, "numpy": Mock()}),
                patch.object(p.prior, "_corpus", return_value=corpus),
                patch.object(
                    setup,
                    "validate_dependencies",
                    side_effect=lambda: events.append("dependencies"),
                ),
                patch.object(
                    setup,
                    "validate_checkpoint",
                    side_effect=lambda path: events.append("checkpoint"),
                ) as checkpoint,
                patch.object(
                    setup, "_import_backend", side_effect=imported
                ) as importer,
                patch.object(setup, "_load_model", side_effect=loaded) as load,
                patch.object(
                    setup,
                    "_prepare_alignment_backtrace",
                    return_value={"id": "cuda-backtrace-init-v1"},
                ) as prepared,
                patch.object(
                    setup, "_fingerprint", return_value=setup.MODEL_FINGERPRINT
                ),
                patch.object(setup, "_git", side_effect=git) as git_call,
            ):
                yield SimpleNamespace(
                    setup=setup,
                    backend=backend,
                    modules=modules,
                    events=events,
                    checkpoint=checkpoint,
                    importer=importer,
                    load=load,
                    prepared=prepared,
                    git=git_call,
                )

    def test_clean_git_status_and_two_sessions_reuse_one_guarded_import_model_worker(
        self,
    ):
        with self.native_fixture() as fixture:
            self.assertEqual(p.NativeOwner({}).config_overrides, {})
            owner = p.NativeOwner(
                {"digest": "frozen-source"},
                config_overrides={
                    "left_context_ms": 20000,
                    "word_context_limit_ms": 24000,
                },
            )
            first, finalize_first = owner.factory("smoke")
            self.assertEqual(first.config.left_context_ms, 20000)
            self.assertEqual(first.config.word_context_limit_ms, 24000)
            self.assertEqual(fixture.setup.CLI_STREAM_CONFIG.left_context_ms, 2000)
            worker = owner.worker
            first.close()
            self.assertEqual(finalize_first()["session_number"], 1)
            self.assertEqual(
                finalize_first()["stream_config"]["left_context_ms"], 20000
            )
            second, finalize_second = owner.factory("long")
            second.close()
            self.assertIs(owner.worker, worker)
            self.assertEqual(finalize_second()["session_number"], 2)
            self.assertEqual(owner.initialization_stage, "ready")
            fixture.load.assert_called_once()
            fixture.prepared.assert_called_once_with()
            self.assertEqual(
                finalize_second()["alignment_initialization"],
                {"id": "cuda-backtrace-init-v1"},
            )
            fixture.importer.assert_called_once_with(
                fixture.setup.BackendSetup(fixture.backend, "a" * 40), eager_cuda=True
            )
            self.assertLess(
                fixture.events.index("checkpoint"),
                fixture.events.index("guarded-import"),
            )
            self.assertLess(
                fixture.events.index("guarded-import"),
                fixture.events.index("model-load"),
            )
            self.assertEqual(fixture.git.call_count, 10)

    def test_named_profiles_select_exact_native_config_tree_and_reuse(self):
        from whisper_runtime.profiles import get_profile

        for name in ("conservative-v1", "experimental-optimized-v1"):
            selected = get_profile(name)
            with (
                self.subTest(name=name),
                self.native_fixture(reuse=selected.reuse_alignment_features) as fixture,
            ):
                owner = p.NativeOwner({"digest": "frozen-source"}, profile=name)
                for phase in ("smoke", "long"):
                    stream, finalize = owner.factory(phase)
                    self.assertEqual(stream.config, selected.stream_config)
                    execution = stream._adapter.execution_profile
                    self.assertEqual(execution.profile_id, selected.native_profile_id)
                    self.assertEqual(
                        execution.reuse_alignment_features,
                        selected.reuse_alignment_features,
                    )
                    stream.close()
                    metadata = finalize()
                    self.assertEqual(metadata["execution_profile"], asdict(selected))
                    self.assertEqual(metadata["config_overrides"], {})
                    self.assertEqual(
                        metadata["backend_tree"],
                        fixture.setup.BACKEND_REUSE_TREE
                        if selected.reuse_alignment_features
                        else fixture.setup.BACKEND_TREE,
                    )
                    self.assertEqual(metadata["backend_revision"], "a" * 40)
                fixture.importer.assert_called_once()
                fixture.load.assert_called_once()

    def test_named_profile_rejects_wrong_tree_before_import_or_model(self):
        for name, reuse in (
            ("conservative-v1", True),
            ("experimental-optimized-v1", False),
        ):
            with self.subTest(name=name), self.native_fixture(reuse=reuse) as fixture:
                with self.assertRaisesRegex(ValueError, "backend tree differs"):
                    p.NativeOwner({}, profile=name).factory("smoke")
                fixture.checkpoint.assert_not_called()
                fixture.importer.assert_not_called()
                fixture.load.assert_not_called()

    def test_profile_and_historical_override_contract_fail_before_native_work(self):
        with patch.object(p, "importlib") as imports:
            with self.assertRaisesRegex(ValueError, "unknown execution profile"):
                p.NativeOwner({}, profile="unknown")
            for profile, overrides in (
                ("conservative-v1", {"left_context_ms": 20000}),
                ("experimental-optimized-v1", {"max_draft_tokens": 0}),
                (None, {"holdback_ms": 0}),
            ):
                with self.assertRaises(ValueError):
                    p.NativeOwner({}, profile=profile, config_overrides=overrides)
            imports.import_module.assert_not_called()
        with self.native_fixture() as fixture:
            owner = p.NativeOwner(
                {"digest": "frozen-source"}, config_overrides={"left_context_ms": 4000}
            )
            stream, finalize = owner.factory("smoke")
            stream.close()
            metadata = finalize()
            self.assertIsNone(metadata["execution_profile"])
            self.assertEqual(metadata["native_profile_id"], "tiny.en/cli-fp32-v1")
            self.assertFalse(metadata["reuse_alignment_features"])
            self.assertEqual(metadata["stream_config"]["left_context_ms"], 4000)
            self.assertEqual(metadata["backend_tree"], fixture.setup.BACKEND_TREE)

    def test_ambient_whisper_or_submodule_fails_before_checkpoint_and_model(self):
        for name in ("whisper", "whisper.triton_ops"):
            with self.subTest(name=name), self.native_fixture() as fixture:
                ambient = ModuleType(name)
                ambient.__file__ = "unverified.py"
                with patch.dict(sys.modules, {name: ambient}):
                    with self.assertRaisesRegex(ValueError, "ambient Whisper"):
                        p.NativeOwner({}).factory("smoke")
                    self.assertIs(sys.modules[name], ambient)
                fixture.checkpoint.assert_not_called()
                fixture.importer.assert_not_called()
                fixture.load.assert_not_called()

    def test_source_or_ignored_import_drift_fails_before_backend_import(self):
        for wrong in ("tree", "ignored"):
            with self.subTest(wrong=wrong), self.native_fixture() as fixture:
                original = fixture.git.side_effect

                def altered(path, *args):
                    if wrong == "tree" and args == ("rev-parse", "HEAD^{tree}"):
                        return "wrong"
                    if wrong == "ignored" and args[0] == "ls-files":
                        return "whisper/untracked.py"
                    return original(path, *args)

                fixture.git.side_effect = altered
                with self.assertRaises(ValueError):
                    p.NativeOwner({}).factory("smoke")
                fixture.checkpoint.assert_not_called()
                fixture.importer.assert_not_called()
                fixture.load.assert_not_called()

    def test_guarded_import_failure_does_not_load_model_or_create_session(self):
        with self.native_fixture() as fixture:
            fixture.importer.side_effect = RuntimeError("CUDA prerequisite absent")
            owner = p.NativeOwner({})
            with self.assertRaisesRegex(RuntimeError, "CUDA prerequisite absent"):
                owner.factory("smoke")
            fixture.load.assert_not_called()
            self.assertEqual((owner.loads, owner.sessions), (0, 0))
            self.assertIsNone(owner.worker)
            self.assertIsNone(owner.backend_modules)
            self.assertEqual(owner.initialization_stage, "backend_import")

    def test_backtrace_compile_failure_does_not_open_a_stream(self):
        with self.native_fixture() as fixture:
            failure = RuntimeError("backtrace compilation failed")
            fixture.prepared.side_effect = failure
            owner = p.NativeOwner({})
            with self.assertRaises(RuntimeError) as observed:
                owner.factory("smoke")
            self.assertIs(observed.exception, failure)
            self.assertEqual((owner.loads, owner.sessions), (1, 0))
            self.assertIsNone(owner.worker)
            self.assertIsNone(owner.alignment_initialization)
            self.assertEqual(owner.initialization_stage, "alignment_backtrace")

    def test_reused_module_identity_or_path_drift_fails_before_second_checkpoint(self):
        for changed in ("identity", "path"):
            with self.subTest(changed=changed), self.native_fixture() as fixture:
                owner = p.NativeOwner({"digest": "frozen-source"})
                stream, finalize = owner.factory("smoke")
                stream.close()
                finalize()
                fixture.checkpoint.reset_mock()
                if changed == "identity":
                    replacement = ModuleType("whisper")
                    replacement.__file__ = fixture.modules["whisper"].__file__
                    sys.modules["whisper"] = replacement
                else:
                    fixture.modules["whisper.triton_ops"].__file__ = "unverified.py"
                with self.assertRaisesRegex(ValueError, "backend module"):
                    owner.factory("long")
                fixture.checkpoint.assert_not_called()
                fixture.importer.assert_called_once()
                fixture.load.assert_called_once()
                self.assertEqual(owner.sessions, 1)


class StageTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_success_never_requests_long_or_claims_full_completion(self):
        async def runner(url, headers, seed, plan, phase, digest):
            self.assertEqual(phase, "smoke")
            return {"passed": True}

        result = await p.staged_run(
            "unused", {}, b"ab", {"short_only": True}, "digest", runner=runner
        )
        self.assertEqual(result["status"], "completed-short")
        self.assertFalse(result["long_gate_passed"])
        self.assertIsNone(result["long"])
        registered = {**stage(), "stream_config": {"left_context_ms": 20000}}
        self.assertFalse(
            p.stage_valid(done(), registered, expected_digest="frozen-source")
        )

    async def test_initialization_failure_is_bounded_and_keeps_original_error(self):
        owner = SimpleNamespace(
            factory=Mock(side_effect=RuntimeError("secret-token-do-not-report")),
            initialization_stage="backend_clean",
            instance_id="failed-worker",
        )
        observed = []

        def make_handler(factory, **kwargs):
            async def handler(scope, receive, send):
                try:
                    factory()
                except RuntimeError:
                    record = dict(
                        type="done", status="failed", error_code="runtime_error"
                    )
                    await send({"type": "websocket.send", "text": json.dumps(record)})

            return handler

        async def send(value):
            observed.append(value)

        with (
            patch.object(p.prior, "_verify_source"),
            patch("examples.pcm_websocket.make_app", side_effect=make_handler),
            patch("builtins.print") as log,
        ):
            app = p.make_gpu_app(
                {"digest": "frozen-source"},
                {"stages": {"smoke": stage(), "long": stage()}},
                owner=owner,
            )
            await app({"type": "websocket", "path": "/smoke"}, None, send)
            value = json.loads(observed[-1]["text"])
            self.assertEqual(value["error_code"], "runtime_error")
            self.assertEqual(
                value["metadata"]["initialization_failure"],
                {
                    "stage": "backend_clean",
                    "exception_type": "RuntimeError",
                },
            )
            self.assertNotIn("secret-token", json.dumps(observed))
            self.assertNotIn("secret-token", log.call_args.args[0])
            await app(
                {"type": "http", "method": "GET", "path": "/receipt/smoke"},
                None,
                send,
            )
            receipt = json.loads(observed[-1]["body"])
            self.assertEqual(receipt["instance_id"], "failed-worker")
            self.assertEqual(receipt["done"], value)
            await app({"type": "websocket", "path": "/long"}, None, send)
            self.assertEqual(observed[-1]["code"], 1008)
            owner.factory.assert_called_once_with("smoke")

    async def test_preregistered_quality_gate_and_raw_event_receipt(self):
        async def client_run(url, source, *, headers, config, on_event):
            await on_event(
                dict(kind="provisional", segment_id="one", text="observed words"), 1
            )
            await on_event(dict(kind="commit", segment_id="one"), 2)
            return dict(status="completed", server_done=done())

        for rate in (0.05, 0.05001):
            with tempfile.TemporaryDirectory() as directory:
                score = dict(word_edit_rate=rate, word_edit_distance=1)
                with (
                    patch(
                        "examples.replay_websocket.stream_live", side_effect=client_run
                    ),
                    patch.object(
                        p,
                        "read_receipt",
                        return_value=dict(status="available", record=dict(done=done())),
                    ),
                    patch.object(
                        p.prior,
                        "_corpus",
                        return_value=SimpleNamespace(
                            b=SimpleNamespace(
                                _word_difference=lambda text, reference: score
                            )
                        ),
                    ),
                ):
                    result = await p.run_stage(
                        "unused",
                        {},
                        b"ab",
                        dict(
                            stages={"smoke": stage()},
                            smoke_reference="reference",
                            max_smoke_word_edit_rate=0.05,
                        ),
                        "smoke",
                        "frozen-source",
                        artifact_directory=Path(directory),
                    )
                self.assertEqual(result["passed"], rate == 0.05)
                raw = (Path(directory) / result["event_receipt"]["path"]).read_bytes()
                self.assertEqual(p.prior._sha(raw), result["event_receipt"]["sha256"])
                self.assertEqual(len(raw.splitlines()), 2)

    async def test_failed_smoke_prevents_long_request(self):
        phases = []

        async def runner(url, headers, seed, plan, phase, digest):
            phases.append(phase)
            return {"passed": False}

        result = await p.staged_run("unused", {}, b"ab", {}, "digest", runner=runner)
        self.assertEqual(phases, ["smoke"])
        self.assertFalse(result["long_gate_passed"])
        self.assertIsNone(result["long"])

    async def test_same_worker_model_and_second_session_are_required(self):
        for changed in (False, True):
            phases = []

            async def runner(url, headers, seed, plan, phase, digest):
                phases.append(phase)
                record = done(phase)
                if changed and phase == "long":
                    record["metadata"]["instance_id"] = "replacement-container"
                return {"passed": True, "result": {"server_done": record}}

            result = await p.staged_run(
                "unused", {}, b"ab", {}, "digest", runner=runner
            )
            self.assertEqual(phases, ["smoke", "long"])
            self.assertEqual(result["status"], "failed" if changed else "completed")

    async def test_server_rejects_out_of_order_repeat_and_failed_stage(self):
        for fail, short_only in ((False, False), (True, False), (False, True)):
            observed, owner = [], SimpleNamespace(factory=Mock())

            def make_handler(factory, **kwargs):
                async def handler(scope, receive, send):
                    factory()
                    record = done(scope["path"].strip("/"))
                    if fail:
                        record["metrics"]["accepted_sha256"] = "wrong"
                    await send({"type": "websocket.send", "text": json.dumps(record)})

                return handler

            with (
                patch.object(p.prior, "_verify_source"),
                patch("examples.pcm_websocket.make_app", side_effect=make_handler),
            ):
                app = p.make_gpu_app(
                    {"digest": "frozen-source"},
                    {
                        "stages": {"smoke": stage(), "long": stage()},
                        "short_only": short_only,
                    },
                    owner=owner,
                )

                async def send(value):
                    observed.append(value)

                await app({"type": "websocket", "path": "/long"}, None, send)
                self.assertEqual(observed[-1]["code"], 1008)
                owner.factory.assert_not_called()
                await app({"type": "websocket", "path": "/smoke"}, None, send)
                before = owner.factory.call_count
                await app(
                    {"type": "http", "method": "GET", "path": "/receipt/smoke"},
                    None,
                    send,
                )
                self.assertEqual(observed[-2]["status"], 200)
                report = json.loads(observed[-1]["body"])
                self.assertEqual(report["phase"], "smoke")
                self.assertEqual(report["source_digest"], "frozen-source")
                await app(
                    {"type": "http", "method": "GET", "path": "/receipt/smoke"},
                    None,
                    send,
                )
                self.assertEqual(observed[-2]["status"], 404)
                self.assertEqual(owner.factory.call_count, before)
                await app({"type": "websocket", "path": "/smoke"}, None, send)
                self.assertEqual(observed[-1]["code"], 1008)
                await app({"type": "websocket", "path": "/long"}, None, send)
                self.assertEqual(
                    owner.factory.call_count, 1 if fail or short_only else 2
                )


class ResourceTests(unittest.TestCase):
    def test_reuse_image_is_explicit_frozen_build_only_and_clean(self):
        from whisper_runtime.native_setup import BACKEND_REUSE_TREE, BACKEND_TREE

        image = Mock()
        image.run_commands.return_value = image
        self.assertIs(p.prepare_profile_image(image, "conservative-v1", {}), image)
        image.run_commands.assert_not_called()
        with self.assertRaisesRegex(ValueError, "unknown execution profile"):
            p.prepare_profile_image(image, "unknown", {})
        names = (
            p.REUSE_PATCH,
            (Path(p.REUSE_PATCH).parent / "SHA256SUMS").as_posix(),
        )
        files = [
            {"path": name, "size_bytes": len(data), "sha256": p.prior._sha(data)}
            for name in names
            for data in [(p.ROOT / name).read_bytes()]
        ]
        for entries in ([], files[:1], [{**files[0], "sha256": "wrong"}, files[1]]):
            with self.assertRaises(ValueError):
                p.prepare_profile_image(
                    image, "experimental-optimized-v1", {"files": entries}
                )
            image.run_commands.assert_not_called()
        self.assertIs(
            p.prepare_profile_image(
                image, "experimental-optimized-v1", {"files": files}
            ),
            image,
        )
        command = image.run_commands.call_args.args[0]
        self.assertIn("sha256sum --check SHA256SUMS", command)
        self.assertIn('test "$(git rev-parse HEAD^{tree})" = ' + BACKEND_TREE, command)
        self.assertIn('test "$(git write-tree)" = ' + BACKEND_REUSE_TREE, command)
        self.assertIn("GIT_AUTHOR_DATE=2000-01-01T00:00:00Z", command)
        self.assertIn("GIT_COMMITTER_DATE=2000-01-01T00:00:00Z", command)
        self.assertIn("commit --no-gpg-sign --no-verify", command)
        self.assertIn(
            'test "$(git rev-parse HEAD^{tree})" = ' + BACKEND_REUSE_TREE, command
        )
        self.assertTrue(
            command.endswith(
                'test -z "$(git status --porcelain --untracked-files=all)"'
            )
        )
        self.assertNotIn("git fetch", command)
        self.assertNotIn("load_model", command)

    def test_resource_configuration_is_authenticated_ephemeral_one_t4_readonly(self):
        image = Mock()
        for name in (
            "apt_install",
            "pip_install",
            "uv_pip_install",
            "run_commands",
            "add_local_file",
            "env",
        ):
            getattr(image, name).return_value = image
        options, asgi_options = [], []

        def register(**kwargs):
            options.append(kwargs)
            return lambda function: function

        def asgi(**kwargs):
            asgi_options.append(kwargs)
            return lambda function: function

        volume = Mock()
        modal = SimpleNamespace(
            Image=SimpleNamespace(debian_slim=lambda **kwargs: image),
            App=lambda label: SimpleNamespace(function=register),
            asgi_app=asgi,
            Volume=SimpleNamespace(from_name=Mock(return_value=volume)),
        )
        corpus = SimpleNamespace(
            BASE_COMMIT="base",
            _helper=lambda name: SimpleNamespace(
                DIRECT_IMAGE_PACKAGES=("pinned==1",),
                _build_command=lambda commit: "cached-build",
            ),
            b=SimpleNamespace(MODEL_CACHE_NAME="existing", MODEL_CACHE_MOUNT="/models"),
        )
        for short_only, timeout in ((False, 1890), (True, 190)):
            with self.subTest(short_only=short_only):
                options.clear()
                asgi_options.clear()
                modal.Volume.from_name.reset_mock()
                volume.reset_mock()
                with patch.object(p.prior, "_corpus", return_value=corpus):
                    p.resources(
                        modal,
                        {"files": []},
                        {"echo_sha256": "hash", "short_only": short_only},
                    )
                self.assertEqual(len(options), 2)
                cpu, gpu = options
                self.assertNotIn("gpu", cpu)
                self.assertEqual(gpu["gpu"], "T4")
                self.assertEqual(gpu["timeout"], timeout)
                self.assertFalse(gpu["single_use_containers"])
                self.assertEqual(
                    p.budget_plan(short_only=short_only)["automatic_retries"], 0
                )
                for item in options:
                    self.assertEqual(item["min_containers"], 0)
                    self.assertEqual(item["max_containers"], 1)
                    self.assertEqual(item["buffer_containers"], 0)
                    self.assertEqual(item["scaledown_window"], 2)
                    self.assertTrue(item["block_network"])
                    self.assertTrue(item["restrict_modal_access"])
                    self.assertNotIn(
                        "retries", item
                    )  # Modal rejects retries on ASGI functions.
                self.assertTrue(
                    all(item["requires_proxy_auth"] for item in asgi_options)
                )
                modal.Volume.from_name.assert_called_once_with(
                    "existing", create_if_missing=False
                )
                volume.with_mount_options.assert_called_once_with(read_only=True)


if __name__ == "__main__":
    unittest.main()
