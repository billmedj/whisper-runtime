from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bootstrap_native_backend
import native_backend_setup as setup
from bootstrap_native_backend import next_example_command
from native_backend_setup import (
    BACKEND_BASE_COMMIT,
    BACKEND_BASE_TREE,
    BACKEND_PATCHED_TREE,
    BACKEND_URL,
    PATCH_DIRECTORY,
    PYPI_INDEX,
    RUNTIME_ROOT,
    TORCH_CPU_INDEX,
    EnvironmentIdentity,
    GitIdentity,
    NativeSetupError,
    SetupPaths,
    build_manifest,
    checkpoint_digest_from_url,
    checkpoint_path,
    dependency_install_commands,
    environment_python,
    inspect_environment,
    isolated_command_environment,
    parse_checksum_manifest,
    require_cached_checkpoint,
    require_safe_setup_root,
    require_supported_python,
    verify_dependency_versions,
    verify_patch_manifest,
    verify_recorded_environment,
    write_manifest,
)
from run_native_example import native_example_environment


class NativeBackendSetupTests(unittest.TestCase):
    def expected_dependencies(self) -> dict[str, str]:
        return {
            "jsonschema": "4.25.1",
            "more-itertools": "11.1.0",
            "numba": "0.67.0",
            "numpy": "2.5.2",
            "setuptools": "82.0.1",
            "tiktoken": "0.14.0",
            "torch": "2.6.0+cpu",
            "tqdm": "4.70.0",
        }

    def environment_identity(self) -> EnvironmentIdentity:
        dependencies = self.expected_dependencies()
        return EnvironmentIdentity(
            python_version="Python 3.13.1",
            dependencies=dependencies,
            distributions={**dependencies, "pip": "25.2"},
        )

    def test_pinned_native_python_range_is_exact(self) -> None:
        for version in ((3, 12), (3, 13)):
            with self.subTest(version=version):
                require_supported_python(version)

        for version in ((3, 11), (3, 14)):
            with self.subTest(version=version):
                with self.assertRaisesRegex(NativeSetupError, "3.12 or 3.13"):
                    require_supported_python(version)

    def test_pinned_source_identities_are_full_git_objects(self) -> None:
        self.assertEqual(BACKEND_URL, "https://github.com/openai/whisper.git")
        for value in (BACKEND_BASE_COMMIT, BACKEND_BASE_TREE, BACKEND_PATCHED_TREE):
            self.assertEqual(len(value), 40)
            int(value, 16)

    def test_repository_patch_manifest_verifies_all_seven_patches(self) -> None:
        entries = verify_patch_manifest(PATCH_DIRECTORY)
        self.assertEqual(len(entries), 7)
        self.assertEqual(sorted(entries)[0][:5], "0001-")
        self.assertEqual(sorted(entries)[-1][:5], "0007-")

    def test_optional_patch_pin_matches_the_installed_source_contract(self) -> None:
        from whisper_runtime import native_setup

        self.assertEqual(setup.BACKEND_REUSE_TREE, native_setup.BACKEND_REUSE_TREE)
        record = setup.verify_alignment_reuse_patch()
        self.assertEqual(record["path"], setup.ALIGNMENT_REUSE_PATCH)
        self.assertEqual(record["sha256"], setup.ALIGNMENT_REUSE_SHA256)
        self.assertEqual(len(verify_patch_manifest()), 7)

    def test_reuse_flag_selects_a_separate_default_root(self) -> None:
        for flags, expected in (
            ([], setup.DEFAULT_SETUP_ROOT),
            (["--alignment-feature-reuse"], setup.DEFAULT_REUSE_SETUP_ROOT),
            (["--alignment-feature-reuse", "--root", "custom"], Path("custom")),
        ):
            with (
                self.subTest(flags=flags),
                patch.object(sys, "argv", ["bootstrap", *flags]),
            ):
                args = bootstrap_native_backend.parse_args()
                self.assertEqual(args.root, expected)
                self.assertEqual(args.alignment_feature_reuse, bool(flags))
        self.assertEqual(
            require_safe_setup_root(setup.DEFAULT_REUSE_SETUP_ROOT),
            setup.DEFAULT_REUSE_SETUP_ROOT.resolve(),
        )

    def test_verify_only_reuse_selection_rejects_a_standard_setup(self) -> None:
        paths = SetupPaths.from_root(Path("existing-standard"))
        validated = SimpleNamespace(
            backend=GitIdentity("c" * 40, BACKEND_PATCHED_TREE, True)
        )
        with (
            patch.object(
                sys,
                "argv",
                [
                    "bootstrap",
                    "--verify-only",
                    "--alignment-feature-reuse",
                    "--root",
                    str(paths.root),
                ],
            ),
            patch.object(
                bootstrap_native_backend, "load_validated_setup", return_value=validated
            ),
            patch.object(bootstrap_native_backend, "run_setup") as run_setup,
            self.assertRaisesRegex(SystemExit, "does not enable alignment"),
        ):
            bootstrap_native_backend.main()
        run_setup.assert_not_called()

    @contextmanager
    def local_backend_fixture(self, *, checkout_base=False):
        """Exercise actual Git application/validation without cloning or network."""
        from whisper_runtime import native_setup

        with tempfile.TemporaryDirectory(prefix="whisper-bootstrap-test-") as temporary:
            root = Path(temporary)
            paths = SetupPaths.from_root(root / "install")
            paths.backend.mkdir(parents=True)
            runtime = root / "runtime"
            patch_directory = runtime / "patches/openai-whisper"
            patch_directory.mkdir(parents=True)

            def git(*args):
                self.assertNotIn("clone", args)
                self.assertNotIn("fetch", args)
                return setup._git_output(paths.backend, *args)

            git("init", "--quiet")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.invalid")
            git("config", "commit.gpgSign", "false")
            git("config", "core.hooksPath", "")
            git("config", "core.autocrlf", "false")
            git("remote", "add", "origin", BACKEND_URL)
            package = paths.backend / "whisper"
            package.mkdir()
            (package / "__init__.py").write_bytes(b"")
            content = package / "fixture.txt"
            content.write_bytes(b"0\n")
            git("add", ".")
            git("commit", "--quiet", "--no-gpg-sign", "-m", "base")
            base = setup.git_identity(paths.backend)
            for number in range(1, 8):
                content.write_bytes(f"{number}\n".encode())
                git("add", ".")
                git("commit", "--quiet", "--no-gpg-sign", "-m", f"patch {number}")
            standard = setup.git_identity(paths.backend)
            git(
                "format-patch",
                "--output-directory",
                str(patch_directory),
                f"{base.commit}..HEAD",
            )
            content.write_bytes(b"8\n")
            git("add", ".")
            git("commit", "--quiet", "--no-gpg-sign", "-m", "optional reuse")
            reused = setup.git_identity(paths.backend)
            optional = runtime / setup.ALIGNMENT_REUSE_PATCH
            optional.parent.mkdir(parents=True)
            optional.write_text(
                git("diff", standard.commit, reused.commit) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            patch_files = {
                path.name: setup.sha256_file(path)
                for path in sorted(patch_directory.glob("*.patch"))
            }
            (patch_directory / "SHA256SUMS").write_text(
                "".join(f"{digest}  {name}\n" for name, digest in patch_files.items()),
                encoding="utf-8",
            )
            git(
                "checkout",
                "--detach",
                base.commit if checkout_base else standard.commit,
            )
            with ExitStack() as stack:
                for target, values in (
                    (
                        setup,
                        {
                            "RUNTIME_ROOT": runtime,
                            "PATCH_DIRECTORY": patch_directory,
                            "BACKEND_BASE_COMMIT": base.commit,
                            "BACKEND_BASE_TREE": base.tree,
                            "BACKEND_PATCHED_TREE": standard.tree,
                            "BACKEND_REUSE_TREE": reused.tree,
                            "ALIGNMENT_REUSE_SHA256": setup.sha256_file(optional),
                        },
                    ),
                    (
                        native_setup,
                        {
                            "BACKEND_BASE": base.commit,
                            "BACKEND_TREE": standard.tree,
                            "BACKEND_REUSE_TREE": reused.tree,
                        },
                    ),
                ):
                    for name, value in values.items():
                        stack.enter_context(patch.object(target, name, value))
                yield SimpleNamespace(
                    paths=paths,
                    patches=patch_files,
                    optional=optional,
                    standard=standard,
                    reuse=reused,
                    content=content,
                    git=git,
                )

    def test_local_git_keeps_seven_patch_default_then_applies_reuse_once(self) -> None:
        with self.local_backend_fixture(checkout_base=True) as fixture:
            standard = setup.ensure_backend(fixture.paths, fixture.patches)
            self.assertTrue(standard.clean)
            self.assertEqual(standard.tree, fixture.standard.tree)
            self.assertEqual(
                fixture.git(
                    "rev-list", "--count", f"{setup.BACKEND_BASE_COMMIT}..HEAD"
                ),
                "7",
            )
            reused = setup.ensure_backend(
                fixture.paths, fixture.patches, alignment_feature_reuse=True
            )
            self.assertTrue(reused.clean)
            self.assertEqual(reused.tree, fixture.reuse.tree)
            self.assertEqual(
                fixture.git(
                    "rev-list", "--count", f"{setup.BACKEND_BASE_COMMIT}..HEAD"
                ),
                "8",
            )
            with patch.object(setup, "_run", wraps=setup._run) as runner:
                self.assertEqual(
                    setup.ensure_backend(
                        fixture.paths, fixture.patches, alignment_feature_reuse=True
                    ),
                    reused,
                )
            for call in runner.call_args_list:
                self.assertFalse(
                    {"clone", "am", "apply", "commit"}.intersection(call.args[0])
                )
            with self.assertRaisesRegex(
                NativeSetupError, "pinned base or patched tree"
            ):
                setup.ensure_backend(fixture.paths, fixture.patches)
            self.assertEqual(fixture.git("rev-parse", "HEAD"), reused.commit)
            with patch.object(setup, "BACKEND_REUSE_TREE", "0" * 40):
                with self.assertRaisesRegex(
                    NativeSetupError, "pinned base or patched tree"
                ):
                    setup.ensure_backend(
                        fixture.paths, fixture.patches, alignment_feature_reuse=True
                    )
            fixture.content.write_bytes(b"dirty\n")
            with self.assertRaisesRegex(NativeSetupError, "source changes"):
                setup.ensure_backend(
                    fixture.paths, fixture.patches, alignment_feature_reuse=True
                )
            fixture.optional.write_bytes(b"tampered")
            with patch.object(
                setup, "_run", side_effect=AssertionError("must validate before Git")
            ):
                with self.assertRaisesRegex(NativeSetupError, "checksum mismatch"):
                    setup.ensure_backend(
                        fixture.paths, fixture.patches, alignment_feature_reuse=True
                    )

    def test_local_reuse_manifest_is_accepted_by_tool_and_sdk_validators(self) -> None:
        from whisper_runtime import native_setup

        with self.local_backend_fixture() as fixture:
            backend = setup.ensure_backend(
                fixture.paths, fixture.patches, alignment_feature_reuse=True
            )
            python = environment_python(fixture.paths.environment)
            python.parent.mkdir(parents=True)
            python.write_bytes(b"placeholder; environment probe mocked")
            runtime = GitIdentity("a" * 40, "b" * 40, True)
            environment = self.environment_identity()
            tools = {"git": "fixture Git", "ffmpeg": "fixture ffmpeg"}
            manifest = build_manifest(
                paths=fixture.paths,
                runtime=runtime,
                backend=backend,
                python=python,
                patches=fixture.patches,
                environment=environment,
                tools=tools,
                alignment_feature_reuse=True,
            )
            write_manifest(fixture.paths.manifest, manifest)
            with (
                patch.object(setup, "require_runtime_identity", return_value=runtime),
                patch.object(setup, "current_tool_versions", return_value=tools),
                patch.object(setup, "inspect_environment", return_value=environment),
                patch.object(
                    setup, "verify_patch_manifest", return_value=fixture.patches
                ),
            ):
                validated = setup.load_validated_setup(fixture.paths.manifest)
            self.assertEqual(validated.backend, backend)
            self.assertIs(manifest["bootstrap_downloaded_models"], False)
            actual = native_setup.validate_backend(
                fixture.paths.manifest, reuse_alignment_features=True
            )
            self.assertEqual(actual.path, fixture.paths.backend)
            self.assertEqual(actual.revision, backend.commit)
            with self.assertRaises(native_setup.NativeSetupError):
                native_setup.validate_backend(fixture.paths.manifest)
            manifest["optional_patch"]["sha256"] = "0" * 64
            write_manifest(fixture.paths.manifest, manifest)
            with self.assertRaisesRegex(NativeSetupError, "reuse patch does not match"):
                setup.load_validated_setup(fixture.paths.manifest)
            del manifest["optional_patch"]
            write_manifest(fixture.paths.manifest, manifest)
            with self.assertRaisesRegex(
                NativeSetupError, "patched tree does not match"
            ):
                setup.load_validated_setup(fixture.paths.manifest)

    def test_checksum_manifest_is_strict_and_rejects_duplicates(self) -> None:
        digest = "a" * 64
        parsed = parse_checksum_manifest(f"{digest}  0001-change.patch\n")
        self.assertEqual(parsed, {"0001-change.patch": digest})
        for invalid in (
            f"{digest} *0001-change.patch",
            f"{digest}  ../change.patch",
            f"{digest}  nested/change.patch",
            "not-a-digest  0001-change.patch",
            "",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(NativeSetupError):
                    parse_checksum_manifest(invalid)
        with self.assertRaisesRegex(NativeSetupError, "duplicate"):
            parse_checksum_manifest(
                f"{digest}  0001-change.patch\n{digest}  0001-change.patch\n"
            )

    def test_patch_verifier_detects_changed_and_unlisted_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            patch = directory / "0001-change.patch"
            patch.write_bytes(b"content\n")
            digest = hashlib.sha256(patch.read_bytes()).hexdigest()
            (directory / "SHA256SUMS").write_text(
                f"{digest}  {patch.name}\n", encoding="utf-8"
            )
            self.assertEqual(verify_patch_manifest(directory), {patch.name: digest})

            patch.write_bytes(b"changed\n")
            with self.assertRaisesRegex(NativeSetupError, "checksum mismatch"):
                verify_patch_manifest(directory)

            patch.write_bytes(b"content\n")
            (directory / "0002-unlisted.patch").write_bytes(b"extra\n")
            with self.assertRaisesRegex(NativeSetupError, "unlisted"):
                verify_patch_manifest(directory)

    def test_environment_paths_are_platform_specific(self) -> None:
        root = Path("setup") / "venv"
        self.assertEqual(
            environment_python(root, os_name="nt"),
            root / "Scripts" / "python.exe",
        )
        self.assertEqual(
            environment_python(root, os_name="posix"), root / "bin" / "python"
        )

    def test_install_plan_targets_only_the_selected_environment(self) -> None:
        python = Path("setup") / "venv" / "bin" / "python"
        linux = dependency_install_commands(python, platform="linux")
        self.assertEqual(len(linux), 4)
        self.assertTrue(all(command[0] == str(python) for command in linux))
        self.assertIn(TORCH_CPU_INDEX, linux[0])
        self.assertIn(PYPI_INDEX, linux[1])
        self.assertEqual(linux[2][-1], str(RUNTIME_ROOT))
        self.assertIn("--no-deps", linux[2])
        self.assertIn("--no-build-isolation", linux[2])
        self.assertTrue(all("--isolated" in command for command in linux))
        self.assertTrue(all("--no-cache-dir" in command for command in linux[:3]))
        self.assertFalse(any("--user" in command for command in linux))

        macos = dependency_install_commands(python, platform="darwin")
        self.assertNotIn(TORCH_CPU_INDEX, macos[0])
        self.assertIn(PYPI_INDEX, macos[0])
        self.assertIn("torch==2.6.0", macos[0])

    def test_child_environment_removes_python_and_pip_injection(self) -> None:
        source = {
            "PATH": "kept",
            "PIP_INDEX_URL": "https://example.invalid/simple",
            "pip_target": "elsewhere",
            "PYTHONHOME": "elsewhere",
            "PythonPath": "elsewhere",
            "PYTHONSTARTUP": "startup.py",
        }
        environment = isolated_command_environment(source)
        self.assertEqual(environment["PATH"], "kept")
        self.assertFalse(any(name.upper().startswith("PIP_") for name in environment))
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("PythonPath", environment)
        self.assertNotIn("PYTHONSTARTUP", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

        runtime_environment = native_example_environment(Path("backend"), source=source)
        self.assertNotIn("PYTHONHOME", runtime_environment)
        self.assertEqual(
            runtime_environment["PYTHONPATH"].split(os.pathsep)[0],
            "backend",
        )

    def test_environment_probe_reads_actual_distributions_without_pip(self) -> None:
        identity = self.environment_identity()
        document = {
            "implementation": "CPython",
            "python_version": "3.13.1",
            "dependencies": identity.dependencies,
            "distributions": identity.distributions,
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(document), stderr=""
        )
        with patch.object(setup, "_run", return_value=completed) as runner:
            observed = inspect_environment(Path("venv-python"))
        self.assertEqual(observed, identity)
        command = runner.call_args.args[0]
        self.assertEqual(command[:2], ("venv-python", "-I"))
        self.assertNotIn("pip", command)

        missing = dict(document)
        missing["dependencies"] = {
            name: version
            for name, version in identity.dependencies.items()
            if name != "numpy"
        }
        failed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(missing), stderr=""
        )
        with patch.object(setup, "_run", return_value=failed):
            with self.assertRaisesRegex(NativeSetupError, "dependency set"):
                inspect_environment(Path("venv-python"))

    def test_environment_verifier_rejects_live_drift(self) -> None:
        expected = self.environment_identity()
        recorded = {
            "python_version": expected.python_version,
            "dependencies": expected.dependencies,
            "resolved_distributions": expected.distributions,
        }
        with patch.object(setup, "inspect_environment", return_value=expected):
            self.assertEqual(
                verify_recorded_environment(recorded, Path("venv-python")), expected
            )

        changed = EnvironmentIdentity(
            python_version=expected.python_version,
            dependencies=expected.dependencies,
            distributions={**expected.distributions, "unexpected": "1.0"},
        )
        with patch.object(setup, "inspect_environment", return_value=changed):
            with self.assertRaisesRegex(NativeSetupError, "inventory"):
                verify_recorded_environment(recorded, Path("venv-python"))

        changed_version = EnvironmentIdentity(
            python_version="Python 3.12.9",
            dependencies=expected.dependencies,
            distributions=expected.distributions,
        )
        with patch.object(setup, "inspect_environment", return_value=changed_version):
            with self.assertRaisesRegex(NativeSetupError, "Python version"):
                verify_recorded_environment(recorded, Path("venv-python"))

        changed_dependencies = dict(expected.dependencies)
        changed_dependencies["numpy"] = "2.5.1"
        changed_packages = EnvironmentIdentity(
            python_version=expected.python_version,
            dependencies=changed_dependencies,
            distributions={**expected.distributions, "numpy": "2.5.1"},
        )
        with patch.object(setup, "inspect_environment", return_value=changed_packages):
            with self.assertRaisesRegex(NativeSetupError, "dependency versions"):
                verify_recorded_environment(recorded, Path("venv-python"))

    def test_verify_only_does_not_enter_setup_or_install_path(self) -> None:
        paths = SetupPaths.from_root(Path("existing-setup"))
        validated = SimpleNamespace(
            paths=paths,
            runtime=GitIdentity("a" * 40, "b" * 40, True),
            backend=GitIdentity("c" * 40, BACKEND_PATCHED_TREE, True),
            python=environment_python(paths.environment),
        )
        arguments = [
            "bootstrap_native_backend.py",
            "--root",
            str(paths.root),
            "--verify-only",
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(
                bootstrap_native_backend,
                "load_validated_setup",
                return_value=validated,
            ),
            patch.object(bootstrap_native_backend, "run_setup") as run_setup,
            patch("builtins.print"),
        ):
            self.assertEqual(bootstrap_native_backend.main(), 0)
        run_setup.assert_not_called()

    def test_dependency_check_accepts_cpu_wheel_local_version(self) -> None:
        observed = {
            "jsonschema": "4.25.1",
            "more-itertools": "11.1.0",
            "numba": "0.67.0",
            "numpy": "2.5.2",
            "setuptools": "82.0.1",
            "tiktoken": "0.14.0",
            "torch": "2.6.0+cpu",
            "tqdm": "4.70.0",
        }
        self.assertEqual(verify_dependency_versions(observed), observed)
        observed["numpy"] = "2.5.1"
        with self.assertRaisesRegex(NativeSetupError, "numpy"):
            verify_dependency_versions(observed)

    def test_checkpoint_requires_explicit_download_or_verified_bytes(self) -> None:
        content = b"checkpoint"
        digest = hashlib.sha256(content).hexdigest()
        url = f"https://example.invalid/models/{digest}/tiny.en.pt"
        self.assertEqual(checkpoint_digest_from_url(url), digest)

        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            expected_path = cache / "tiny.en.pt"
            self.assertEqual(checkpoint_path("tiny.en", url, cache), expected_path)
            with self.assertRaisesRegex(NativeSetupError, "No download|not found"):
                require_cached_checkpoint("tiny.en", url, cache, allow_download=False)
            self.assertEqual(
                require_cached_checkpoint("tiny.en", url, cache, allow_download=True),
                expected_path,
            )

            expected_path.write_bytes(content)
            self.assertEqual(
                require_cached_checkpoint("tiny.en", url, cache, allow_download=False),
                expected_path,
            )
            expected_path.write_bytes(b"changed")
            with self.assertRaisesRegex(NativeSetupError, "No download was attempted"):
                require_cached_checkpoint("tiny.en", url, cache, allow_download=False)

    def test_setup_layout_is_fixed_under_the_selected_root(self) -> None:
        paths = SetupPaths.from_root(Path("setup-root"))
        self.assertEqual(paths.backend, paths.root / "backend")
        self.assertEqual(paths.environment, paths.root / "venv")
        self.assertEqual(paths.manifest, paths.root / "manifest.json")

    def test_setup_root_cannot_write_generated_files_into_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repository = parent / "runtime"
            repository.mkdir()
            self.assertEqual(
                require_safe_setup_root(
                    repository / ".tmp-native", runtime_root=repository
                ),
                (repository / ".tmp-native").resolve(),
            )
            self.assertEqual(
                require_safe_setup_root(parent / "external", runtime_root=repository),
                (parent / "external").resolve(),
            )
            for unsafe in (repository, parent, repository / "src"):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(NativeSetupError):
                        require_safe_setup_root(unsafe, runtime_root=repository)

    def test_next_command_keeps_the_selected_setup_root(self) -> None:
        paths = SetupPaths.from_root(Path("custom setup"))
        self.assertEqual(
            next_example_command(paths),
            [
                "python",
                "tools/run_native_example.py",
                "--root",
                str(paths.root),
            ],
        )

    def test_manifest_validation_probes_the_live_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = SetupPaths.from_root(Path(temporary))
            python = environment_python(paths.environment)
            python.parent.mkdir(parents=True)
            python.write_bytes(b"placeholder")
            runtime = GitIdentity("a" * 40, "b" * 40, True)
            backend = GitIdentity("c" * 40, BACKEND_PATCHED_TREE, True)
            environment = self.environment_identity()
            tools = {"git": "git version test", "ffmpeg": "ffmpeg version test"}
            manifest = build_manifest(
                paths=paths,
                runtime=runtime,
                backend=backend,
                python=python,
                patches=verify_patch_manifest(),
                environment=environment,
                tools=tools,
            )
            write_manifest(paths.manifest, manifest)

            with (
                patch.object(setup, "require_runtime_identity", return_value=runtime),
                patch.object(
                    setup, "_backend_state", return_value=("patched", backend)
                ),
                patch.object(setup, "current_tool_versions", return_value=tools),
                patch.object(
                    setup, "inspect_environment", return_value=environment
                ) as probe,
            ):
                validated = setup.load_validated_setup(paths.manifest)
            self.assertEqual(validated.python, python)
            probe.assert_called_once_with(python)

            manifest["patches"]["manifest"] = "patches/other/SHA256SUMS"
            write_manifest(paths.manifest, manifest)
            with self.assertRaisesRegex(NativeSetupError, "checksum path"):
                setup.load_validated_setup(paths.manifest)

            manifest["patches"]["manifest"] = setup.PATCH_MANIFEST_PATH
            manifest["environment"]["constraints"] = "constraints/other.txt"
            write_manifest(paths.manifest, manifest)
            with self.assertRaisesRegex(NativeSetupError, "constraints path"):
                setup.load_validated_setup(paths.manifest)

            manifest["environment"]["constraints"] = setup.CONSTRAINTS_PATH
            expected_tools = dict(tools)
            manifest["tools"]["git"] = "changed"
            write_manifest(paths.manifest, manifest)
            with patch.object(
                setup, "current_tool_versions", return_value=expected_tools
            ):
                with self.assertRaisesRegex(NativeSetupError, "tool versions"):
                    setup.load_validated_setup(paths.manifest)

    def test_manifest_rejects_unknown_fields_paths_and_nonfinite_json(self) -> None:
        base = {
            "schema_version": setup.SCHEMA_VERSION,
            "created_at": "2026-09-04T00:00:00Z",
            "runtime": {},
            "backend": {},
            "patches": {},
            "environment": {},
            "tools": {},
            "bootstrap_downloaded_models": False,
        }
        invalid_documents = (
            {**base, "unknown": True},
            {**base, "created_at": float("nan")},
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            for document in invalid_documents:
                with self.subTest(document=document):
                    manifest.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(NativeSetupError):
                        setup.load_validated_setup(manifest)
            manifest.write_text(
                '{"schema_version":"2","schema_version":"2"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(NativeSetupError, "duplicate JSON field"):
                setup.load_validated_setup(manifest)


if __name__ == "__main__":
    unittest.main()
