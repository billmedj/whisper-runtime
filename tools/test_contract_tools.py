from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from capture_whisper_reference import git_metadata
from check_repository import contains_absolute_user_path, validate_audio_binding
from compare_whisper_fixtures import compare_fixtures
from smoke_native_whisper import verify_source_revision

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT / "conformance" / "fixtures" / "reference" / "jfk-tiny-en-greedy.json"
)


class ComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    def test_rejects_documents_that_do_not_match_the_schema(self) -> None:
        self.assertTrue(compare_fixtures({}, {}))

    def test_profile_is_part_of_comparison_identity(self) -> None:
        candidate = copy.deepcopy(self.reference)
        candidate["profile"] = "optimized"
        failures = compare_fixtures(self.reference, candidate)
        self.assertTrue(
            any("profile differs" in failure for failure in failures), failures
        )


class PortabilityTests(unittest.TestCase):
    def test_detects_json_escaped_windows_user_path(self) -> None:
        separator = chr(92) * 2
        encoded = '{"path":"' + "C:" + separator + "Users" + separator + "Alice" + '"}'
        self.assertTrue(contains_absolute_user_path(encoded))

    def test_accepts_portable_relative_path(self) -> None:
        self.assertFalse(contains_absolute_user_path("conformance/cache/jfk.flac"))


class AudioManifestTests(unittest.TestCase):
    def test_detects_manifest_mismatch(self) -> None:
        audio = {
            "sha256": "0" * 64,
            "size_bytes": 10,
            "sample_rate_hz": 16000,
            "sample_end": 8,
        }
        manifest = {
            "sha256": "1" * 64,
            "size_bytes": 11,
            "decoded_sample_rate_hz": 8000,
            "decoded_sample_count": 7,
        }
        failures = validate_audio_binding(audio, manifest, "fixture.audio")
        self.assertEqual(len(failures), 4)


class ProvenanceTests(unittest.TestCase):
    def _create_whisper_worktree(self, root: Path) -> tuple[Path, str]:
        module = root / "whisper" / "__init__.py"
        module.parent.mkdir()
        module.write_text("__version__ = 'test'\n", encoding="utf-8")
        (root / ".gitignore").write_text("__pycache__/\n*.py[cod]\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "config",
                "user.email",
                "test@example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Provenance Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "add", ".gitignore", "whisper/__init__.py"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        return module, revision

    def test_native_smoke_binds_the_reported_revision_to_executed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            module, revision = self._create_whisper_worktree(Path(temporary_directory))
            self.assertEqual(verify_source_revision(str(module), revision), revision)

    def test_native_smoke_rejects_a_false_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            module, _ = self._create_whisper_worktree(Path(temporary_directory))
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                verify_source_revision(str(module), "0" * 40)

    def test_native_smoke_rejects_tracked_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            module, revision = self._create_whisper_worktree(Path(temporary_directory))
            module.write_text("__version__ = 'changed'\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source changes"):
                verify_source_revision(str(module), revision)

    def test_native_smoke_rejects_untracked_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            module, revision = self._create_whisper_worktree(root)
            (module.parent / "override.py").write_text("VALUE = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source changes"):
                verify_source_revision(str(module), revision)

    def test_native_smoke_rejects_an_ancestor_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, revision = self._create_whisper_worktree(root)
            nested_module = root / "vendor" / "whisper" / "__init__.py"
            nested_module.parent.mkdir(parents=True)
            nested_module.write_text("__version__ = 'nested'\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not rooted"):
                verify_source_revision(str(nested_module), revision)

    def test_native_smoke_rejects_ignored_package_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            module, revision = self._create_whisper_worktree(root)
            bytecode = module.parent / "__pycache__" / "override.pyc"
            bytecode.parent.mkdir()
            bytecode.write_bytes(b"untracked executable payload")
            with self.assertRaisesRegex(RuntimeError, "ignored files"):
                verify_source_revision(str(module), revision)

    def test_audio_is_the_only_untracked_file_that_can_be_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Conformance Test"],
                check=True,
            )
            (root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "fixture"],
                check=True,
            )

            audio = root / "audio.flac"
            audio.write_bytes(b"first audio payload")
            commit, dirty, digest = git_metadata(
                root,
                excluded_paths={"audio.flac"},
            )
            self.assertRegex(commit or "", r"^[0-9a-f]{40}$")
            self.assertFalse(dirty)

            audio.write_bytes(b"different audio payload")
            _, dirty_after_audio_change, digest_after_audio_change = git_metadata(
                root,
                excluded_paths={"audio.flac"},
            )
            self.assertFalse(dirty_after_audio_change)
            self.assertEqual(digest, digest_after_audio_change)

            (root / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")
            _, dirty_with_source, digest_with_source = git_metadata(
                root,
                excluded_paths={"audio.flac"},
            )
            self.assertTrue(dirty_with_source)
            self.assertNotEqual(digest, digest_with_source)


if __name__ == "__main__":
    unittest.main()
