from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from native_backend_setup import (
    BACKEND_BASE_COMMIT,
    BACKEND_BASE_TREE,
    BACKEND_PATCHED_TREE,
    BACKEND_URL,
    PATCH_DIRECTORY,
    TORCH_CPU_INDEX,
    NativeSetupError,
    SetupPaths,
    checkpoint_digest_from_url,
    checkpoint_path,
    dependency_install_commands,
    environment_python,
    parse_checksum_manifest,
    require_cached_checkpoint,
    require_supported_python,
    verify_dependency_versions,
    verify_patch_manifest,
)


class NativeBackendSetupTests(unittest.TestCase):
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
        self.assertEqual(len(linux), 3)
        self.assertTrue(all(command[0] == str(python) for command in linux))
        self.assertIn(TORCH_CPU_INDEX, linux[0])
        self.assertFalse(any("--user" in command for command in linux))

        macos = dependency_install_commands(python, platform="darwin")
        self.assertNotIn(TORCH_CPU_INDEX, macos[0])
        self.assertIn("torch==2.6.0", macos[0])

    def test_dependency_check_accepts_cpu_wheel_local_version(self) -> None:
        observed = {
            "jsonschema": "4.25.1",
            "more-itertools": "11.1.0",
            "numba": "0.67.0",
            "numpy": "2.5.2",
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


if __name__ == "__main__":
    unittest.main()
