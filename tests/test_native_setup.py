from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call, patch

from whisper_runtime import native_setup as setup
from whisper_runtime.profiles import DEFAULT_PROFILE, get_profile, list_profiles


class NativeSetupTests(unittest.TestCase):
    def test_real_hybrid_config_is_valid_without_model_dependencies(self):
        config = setup.CLI_STREAM_CONFIG
        self.assertTrue(config.source_units)
        self.assertTrue(config.word_boundary_fallback)
        self.assertTrue(config.eof_context_retry)
        self.assertTrue(config.defer_word_commits)
        self.assertEqual(config.preview_interval_ms, 2000)
        self.assertEqual(config.holdback_ms, 2000)
        self.assertEqual(config.word_context_limit_ms, 6000)
        self.assertFalse(config.word_alignment)
        self.assertEqual(config.left_context_ms, 2000)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.backend = self.root / "backend"
        package = self.backend / "whisper"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        self.document = {
            "schema_version": "2",
            "backend": {
                "path": str(self.backend),
                "url": "https://github.com/openai/whisper.git",
                "base_commit": setup.BACKEND_BASE,
                "tree": setup.BACKEND_TREE,
                "applied_commit": "a" * 40,
                "clean": True,
            },
        }
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.document), encoding="utf-8")

    def git_answers(self, *, status="", ignored="", tree=None):
        return [
            str(self.backend),
            "a" * 40,
            tree or setup.BACKEND_TREE,
            status,
            ignored,
        ]

    def test_matching_source_and_ignored_bytecode(self):
        with patch.object(
            setup,
            "_git",
            side_effect=self.git_answers(
                ignored="whisper/__pycache__/model.cpython-313.pyc"
            ),
        ):
            observed = setup.validate_backend(self.manifest)
        self.assertEqual(observed.path, self.backend)
        self.assertEqual(observed.revision, "a" * 40)

    def test_path_aliases_resolve_to_one_backend_without_accepting_another(self):
        hop = self.root / "path-alias"
        hop.mkdir()
        alias = hop / ".."
        aliased_backend = alias / "backend"
        self.assertNotEqual(aliased_backend, self.backend)
        self.assertEqual(aliased_backend.resolve(), self.backend)
        for reuse, tree in (
            (False, setup.BACKEND_TREE),
            (True, setup.BACKEND_REUSE_TREE),
        ):
            with self.subTest(reuse=reuse):
                self.document["backend"]["tree"] = tree
                self.document["backend"]["path"] = str(aliased_backend)
                self.write_manifest()
                answers = self.git_answers(tree=tree)
                answers[0] = str(aliased_backend)
                with patch.object(setup, "_git", side_effect=answers) as git:
                    observed = setup.validate_backend(
                        alias / "manifest.json", reuse_alignment_features=reuse
                    )
                self.assertEqual(observed, setup.BackendSetup(self.backend, "a" * 40))
                self.assertTrue(git.call_args_list)
                for invocation in git.call_args_list:
                    self.assertEqual(invocation.args[0], self.backend)

                self.document["backend"]["path"] = str(alias / "other" / "backend")
                self.write_manifest()
                with patch.object(setup, "_git") as git:
                    with self.assertRaisesRegex(
                        setup.NativeSetupError, "belong to this bootstrap directory"
                    ):
                        setup.validate_backend(
                            alias / "manifest.json", reuse_alignment_features=reuse
                        )
                    git.assert_not_called()

    def test_reuse_requires_its_exact_explicit_source_pin(self):
        for reuse, tree in (
            (False, setup.BACKEND_TREE),
            (True, setup.BACKEND_REUSE_TREE),
        ):
            with self.subTest(reuse=reuse):
                self.document["backend"]["tree"] = tree
                self.write_manifest()
                with patch.object(
                    setup, "_git", side_effect=self.git_answers(tree=tree)
                ) as git:
                    observed = setup.validate_backend(
                        self.manifest, reuse_alignment_features=reuse
                    )
                self.assertEqual(observed, setup.BackendSetup(self.backend, "a" * 40))
                self.assertEqual(
                    git.call_args_list,
                    [
                        call(self.backend, "rev-parse", "--show-toplevel"),
                        call(self.backend, "rev-parse", "HEAD"),
                        call(self.backend, "rev-parse", "HEAD^{tree}"),
                        call(
                            self.backend,
                            "status",
                            "--porcelain",
                            "--untracked-files=all",
                        ),
                        call(
                            self.backend,
                            "ls-files",
                            "--others",
                            "--ignored",
                            "--exclude-standard",
                            "--",
                            "whisper",
                        ),
                    ],
                )

    def test_selected_mode_rejects_other_and_unknown_manifest_trees(self):
        for reuse, tree in (
            (False, setup.BACKEND_REUSE_TREE),
            (True, setup.BACKEND_TREE),
            (False, "b" * 40),
            (True, "b" * 40),
        ):
            with self.subTest(reuse=reuse, tree=tree):
                self.document["backend"]["tree"] = tree
                self.write_manifest()
                with patch.object(setup, "_git") as git:
                    with self.assertRaises(setup.NativeSetupError):
                        setup.validate_backend(
                            self.manifest, reuse_alignment_features=reuse
                        )
                    git.assert_not_called()
        self.document["backend"]["tree"] = setup.BACKEND_REUSE_TREE
        self.write_manifest()
        with patch.object(setup, "_git") as git:
            with self.assertRaises(setup.NativeSetupError):
                setup.validate_backend(self.manifest)
            git.assert_not_called()

    def test_invalid_reuse_flag_refused_before_reading_manifest(self):
        for invalid in (None, 0, 1, "true", {}):
            with (
                self.subTest(invalid=invalid),
                patch.object(Path, "open") as source,
                self.assertRaisesRegex(TypeError, "reuse_alignment_features"),
            ):
                try:
                    setup.validate_backend(
                        self.manifest, reuse_alignment_features=invalid
                    )
                finally:
                    source.assert_not_called()

    def test_reuse_keeps_revision_worktree_cleanliness_and_actual_tree_checks(self):
        self.document["backend"]["tree"] = setup.BACKEND_REUSE_TREE
        self.write_manifest()
        answers = self.git_answers(tree=setup.BACKEND_REUSE_TREE)
        for index, value in (
            (0, str(self.root)),
            (1, "b" * 40),
            (2, setup.BACKEND_TREE),
            (2, "b" * 40),
            (3, " M whisper/model.py"),
            (4, "whisper/extra.py"),
        ):
            altered = list(answers)
            altered[index] = value
            with (
                self.subTest(index=index, value=value),
                patch.object(setup, "_git", side_effect=altered),
                self.assertRaises(setup.NativeSetupError),
            ):
                setup.validate_backend(self.manifest, reuse_alignment_features=True)

    def test_dirty_wrong_tree_and_extra_source_refused(self):
        for kwargs in (
            {"status": " M whisper/model.py"},
            {"tree": "b" * 40},
            {"ignored": "whisper/extra.py"},
            {"ignored": "whisper/model.pyc"},
        ):
            with (
                self.subTest(kwargs=kwargs),
                patch.object(setup, "_git", side_effect=self.git_answers(**kwargs)),
                self.assertRaises(setup.NativeSetupError),
            ):
                setup.validate_backend(self.manifest)

    def test_altered_manifest_rejected_before_git(self):
        for reuse in (False, True):
            self.document["backend"]["tree"] = (
                setup.BACKEND_REUSE_TREE if reuse else setup.BACKEND_TREE
            )
            for key, value in (
                ("clean", False),
                ("clean", 1),
                ("tree", "b" * 40),
                ("base_commit", "b" * 40),
                ("url", "https://example.com/whisper.git"),
                ("path", "relative"),
                ("path", str(self.root)),
            ):
                with self.subTest(reuse=reuse, key=key, value=value):
                    original = self.document["backend"][key]
                    self.document["backend"][key] = value
                    self.write_manifest()
                    with patch.object(setup, "_git") as git:
                        with self.assertRaises(setup.NativeSetupError):
                            setup.validate_backend(
                                self.manifest, reuse_alignment_features=reuse
                            )
                        git.assert_not_called()
                    self.document["backend"][key] = original

    def test_manifest_version_remains_exact_in_both_modes(self):
        for reuse in (False, True):
            self.document["backend"]["tree"] = (
                setup.BACKEND_REUSE_TREE if reuse else setup.BACKEND_TREE
            )
            for version in (None, 1, 2, "1", "3"):
                with self.subTest(reuse=reuse, version=version):
                    self.document["schema_version"] = version
                    self.write_manifest()
                    with patch.object(setup, "_git") as git:
                        with self.assertRaises(setup.NativeSetupError):
                            setup.validate_backend(
                                self.manifest, reuse_alignment_features=reuse
                            )
                        git.assert_not_called()

    def test_duplicate_and_oversized_manifest_refused(self):
        for data in ('{"schema_version":"2","schema_version":"2"}', " " * 1_048_577):
            self.manifest.write_text(data, encoding="utf-8")
            with self.assertRaises(setup.NativeSetupError):
                setup.validate_backend(self.manifest)

    def test_missing_and_wrong_checkpoint_refused(self):
        checkpoint = self.root / "tiny.en.pt"
        with self.assertRaises(setup.NativeSetupError):
            setup.validate_checkpoint(checkpoint)
        checkpoint.write_bytes(b"not weights")
        with self.assertRaises(setup.NativeSetupError):
            setup.validate_checkpoint(checkpoint)
        with patch.object(
            setup, "CHECKPOINT_SHA256", hashlib.sha256(b"not weights").hexdigest()
        ):
            setup.validate_checkpoint(checkpoint)

    def test_dependencies_and_device_fail_before_import(self):
        with patch.object(setup.importlib.metadata, "version", return_value="0"):
            with self.assertRaises(setup.NativeSetupError):
                setup.validate_dependencies()
        with patch.object(setup, "validate_backend") as validator:
            with self.assertRaises(setup.NativeSetupError):
                setup.create_stream(
                    manifest=self.manifest, model=self.root, device="mps"
                )
            validator.assert_not_called()

    def test_invalid_stream_options_fail_before_setup(self):
        for arguments, message in (
            *(
                ({"reuse_alignment_features": value}, "reuse_alignment_features")
                for value in (None, 0, 1, "true", {})
            ),
            *(({"config": value}, "config") for value in (False, {}, "default")),
        ):
            with self.subTest(arguments=arguments):
                with (
                    patch.object(setup, "validate_backend") as validator,
                    patch.object(setup, "validate_checkpoint") as checkpoint,
                    patch.object(setup.importlib, "import_module") as imported,
                    self.assertRaisesRegex(TypeError, message),
                ):
                    try:
                        setup.create_stream(
                            manifest=self.manifest, model=self.root, **arguments
                        )
                    finally:
                        validator.assert_not_called()
                        checkpoint.assert_not_called()
                        imported.assert_not_called()

    def test_stream_wires_only_explicit_reuse_and_preserves_supplied_config(self):
        custom = setup.ContinuousStreamConfig(preview_interval_ms=3000)
        for arguments, reuse, config in (
            ({}, False, setup.CLI_STREAM_CONFIG),
            ({"config": None}, False, setup.CLI_STREAM_CONFIG),
            ({"config": custom}, False, custom),
            ({"reuse_alignment_features": True}, True, setup.CLI_STREAM_CONFIG),
            ({"reuse_alignment_features": True, "config": custom}, True, custom),
        ):
            with self.subTest(arguments=arguments):
                backend_setup = setup.BackendSetup(self.backend, "a" * 40)
                loaded, torch, numpy, whisper = Mock(), Mock(), Mock(), Mock()
                with (
                    patch.object(
                        setup, "validate_backend", return_value=backend_setup
                    ) as validator,
                    patch.object(setup, "validate_checkpoint") as checkpoint,
                    patch.object(setup, "validate_dependencies") as dependencies,
                    patch.object(
                        setup.importlib, "import_module", side_effect=[torch, numpy]
                    ),
                    patch.object(setup, "_import_backend", return_value=whisper),
                    patch.object(setup, "_load_model", return_value=loaded),
                    patch.object(
                        setup, "_fingerprint", return_value=setup.MODEL_FINGERPRINT
                    ),
                    patch.object(setup, "NativeWhisperAdapter") as adapter,
                    patch.object(setup, "ContinuousTranscriptStream") as stream,
                ):
                    observed = setup.create_stream(
                        manifest=self.manifest, model=self.root, **arguments
                    )
                validator.assert_called_once_with(
                    self.manifest, reuse_alignment_features=reuse
                )
                checkpoint.assert_called_once_with(self.root)
                dependencies.assert_called_once_with()
                profile = adapter.call_args.args[3]
                self.assertEqual(profile.profile_id, "tiny.en/cli-fp32-v1")
                self.assertEqual(profile.device, "cpu")
                self.assertIs(profile.reuse_alignment_features, reuse)
                self.assertIs(stream.call_args.kwargs["config"], config)
                self.assertIs(stream.call_args.args[0], adapter.return_value)
                self.assertEqual(stream.call_args.kwargs["stream_id"], "caption-cli")
                self.assertEqual(stream.call_args.kwargs["rng_seed"], 7)
                self.assertEqual(
                    stream.call_args.kwargs["options"],
                    setup.NativeDecodeOptions(language="en", without_timestamps=False),
                )
                self.assertIs(observed, stream.return_value)

    def test_named_profiles_wire_exact_config_and_native_identity_on_both_devices(self):
        for selected in list_profiles():
            for device in ("cpu", "cuda:0"):
                with self.subTest(profile=selected.name, device=device):
                    backend_setup = setup.BackendSetup(self.backend, "a" * 40)
                    with (
                        patch.object(
                            setup, "validate_backend", return_value=backend_setup
                        ) as validator,
                        patch.object(setup, "validate_checkpoint"),
                        patch.object(setup, "validate_dependencies"),
                        patch.object(
                            setup.importlib,
                            "import_module",
                            side_effect=[Mock(), Mock()],
                        ),
                        patch.object(setup, "_import_backend") as imported,
                        patch.object(setup, "_load_model"),
                        patch.object(setup, "_prepare_alignment_backtrace") as prepared,
                        patch.object(
                            setup, "_fingerprint", return_value=setup.MODEL_FINGERPRINT
                        ),
                        patch.object(setup, "NativeWhisperAdapter") as adapter,
                        patch.object(setup, "ContinuousTranscriptStream") as stream,
                    ):
                        observed = setup.create_stream(
                            manifest=self.manifest,
                            model=self.root,
                            device=device,
                            profile=selected.name,
                        )
                    validator.assert_called_once_with(
                        self.manifest,
                        reuse_alignment_features=selected.reuse_alignment_features,
                    )
                    imported.assert_called_once_with(
                        backend_setup, eager_cuda=device == "cuda:0"
                    )
                    if device == "cuda:0":
                        prepared.assert_called_once_with()
                    else:
                        prepared.assert_not_called()
                    native_profile = adapter.call_args.args[3]
                    self.assertEqual(
                        native_profile.profile_id, selected.native_profile_id
                    )
                    self.assertEqual(native_profile.device, device)
                    self.assertEqual(native_profile.max_concurrent_decodes, 1)
                    self.assertIs(
                        native_profile.reuse_alignment_features,
                        selected.reuse_alignment_features,
                    )
                    self.assertFalse(native_profile.reuse_decode_features)
                    self.assertIs(
                        stream.call_args.kwargs["config"], selected.stream_config
                    )
                    self.assertIs(observed, stream.return_value)

    def test_named_profile_rejects_wrong_backend_tree_before_checkpoint_or_import(self):
        for selected in list_profiles():
            self.document["backend"]["tree"] = (
                setup.BACKEND_TREE
                if selected.reuse_alignment_features
                else setup.BACKEND_REUSE_TREE
            )
            self.write_manifest()
            with (
                self.subTest(profile=selected.name),
                patch.object(setup, "_git") as git,
                patch.object(setup, "validate_checkpoint") as checkpoint,
                patch.object(setup.importlib, "import_module") as imported,
                self.assertRaises(setup.NativeSetupError),
            ):
                try:
                    setup.create_stream(
                        manifest=self.manifest, model=self.root, profile=selected.name
                    )
                finally:
                    git.assert_not_called()
                    checkpoint.assert_not_called()
                    imported.assert_not_called()

    def test_invalid_profile_or_mixed_overrides_fail_before_setup(self):
        for arguments, error, message in (
            ({"profile": "latest"}, ValueError, "unknown execution profile"),
            ({"profile": ""}, ValueError, "unknown execution profile"),
            ({"profile": 1}, TypeError, "profile name"),
            ({"profile": get_profile(DEFAULT_PROFILE)}, TypeError, "profile name"),
            (
                {"profile": DEFAULT_PROFILE, "config": setup.CLI_STREAM_CONFIG},
                ValueError,
                "profile excludes",
            ),
            (
                {
                    "profile": "experimental-optimized-v1",
                    "reuse_alignment_features": True,
                },
                ValueError,
                "profile excludes",
            ),
        ):
            with (
                self.subTest(arguments=arguments),
                patch.object(setup, "validate_backend") as validator,
                patch.object(setup.importlib, "import_module") as imported,
                self.assertRaisesRegex(error, message),
            ):
                try:
                    setup.create_stream(
                        manifest=self.manifest, model=self.root, **arguments
                    )
                finally:
                    validator.assert_not_called()
                    imported.assert_not_called()

    def test_import_uses_verified_path_and_restores_python_settings(self):
        prefix, no_write = sys.pycache_prefix, sys.dont_write_bytecode
        with patch.dict(sys.modules):
            for name in tuple(sys.modules):
                if name == "whisper" or name.startswith("whisper."):
                    del sys.modules[name]
            module = setup._import_backend(setup.BackendSetup(self.backend, "a" * 40))
            self.assertEqual(module.VALUE, 7)
            self.assertEqual(
                Path(module.__file__).resolve(),
                self.backend / "whisper" / "__init__.py",
            )
            self.assertEqual(sys.pycache_prefix, prefix)
            self.assertEqual(sys.dont_write_bytecode, no_write)
            with self.assertRaises(setup.NativeSetupError):
                setup._import_backend(setup.BackendSetup(self.backend, "a" * 40))

    def test_path_loading_restores_named_mask_before_device_transfer(self):
        for device in ("cpu", "cuda:0"):
            with self.subTest(device=device):
                loaded = Mock()
                loaded.to.return_value = loaded.float.return_value = (
                    loaded.eval.return_value
                ) = loaded
                backend = ModuleType("verified-whisper")
                backend._ALIGNMENT_HEADS = {"tiny.en": b"verified-tiny-en-mask"}
                backend.load_model = Mock(return_value=loaded)
                checkpoint = self.root / "my-cached-checkpoint.pt"
                result = setup._load_model(backend, checkpoint, device)
                backend.load_model.assert_called_once_with(
                    str(checkpoint.resolve()), device="cpu"
                )
                self.assertIs(result, loaded)
                self.assertEqual(
                    loaded.mock_calls,
                    [
                        call.set_alignment_heads(b"verified-tiny-en-mask"),
                        call.to(device),
                        call.float(),
                        call.eval(),
                    ],
                )

    def test_missing_named_mask_refuses_before_loading_checkpoint(self):
        backend = ModuleType("verified-whisper")
        backend.load_model = Mock()
        for masks in (None, {}, {"tiny.en": "not encoded bytes"}):
            with self.subTest(masks=masks):
                backend._ALIGNMENT_HEADS = masks
                with self.assertRaisesRegex(setup.NativeSetupError, "alignment mask"):
                    setup._load_model(backend, self.root / "weights.pt", "cpu")
        backend.load_model.assert_not_called()

    def test_cuda_optional_module_import_runs_inside_fresh_cache_guard(self):
        (self.backend / "whisper" / "triton_ops.py").write_text(
            "import sys\nGUARD = (sys.pycache_prefix, sys.dont_write_bytecode)\n",
            encoding="utf-8",
        )
        previous = sys.pycache_prefix, sys.dont_write_bytecode
        with patch.dict(sys.modules):
            for name in tuple(sys.modules):
                if name == "whisper" or name.startswith("whisper."):
                    del sys.modules[name]
            setup._import_backend(
                setup.BackendSetup(self.backend, "a" * 40), eager_cuda=True
            )
            observed = sys.modules["whisper.triton_ops"].GUARD
            self.assertTrue(observed[1])
            self.assertIsInstance(observed[0], str)
            self.assertNotEqual(observed[0], previous[0])
            self.assertFalse(Path(observed[0]).exists())
        self.assertEqual((sys.pycache_prefix, sys.dont_write_bytecode), previous)

    def test_missing_cuda_prerequisite_is_not_imported_on_cpu_and_fails_cleanly(self):
        (self.backend / "whisper" / "triton_ops.py").write_text(
            "raise RuntimeError('missing test CUDA prerequisite')\n", encoding="utf-8"
        )
        previous = sys.pycache_prefix, sys.dont_write_bytecode
        with patch.dict(sys.modules):
            for name in tuple(sys.modules):
                if name == "whisper" or name.startswith("whisper."):
                    del sys.modules[name]
            setup._import_backend(setup.BackendSetup(self.backend, "a" * 40))
            self.assertNotIn("whisper.triton_ops", sys.modules)
            del sys.modules["whisper"]
            with self.assertRaisesRegex(RuntimeError, "missing test CUDA prerequisite"):
                setup._import_backend(
                    setup.BackendSetup(self.backend, "a" * 40), eager_cuda=True
                )
            self.assertFalse(
                any(
                    name == "whisper" or name.startswith("whisper.")
                    for name in sys.modules
                )
            )
        self.assertEqual((sys.pycache_prefix, sys.dont_write_bytecode), previous)


if __name__ == "__main__":
    unittest.main()
