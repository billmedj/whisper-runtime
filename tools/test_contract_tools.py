from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from capture_whisper_reference import git_metadata
from check_repository import (
    contains_absolute_user_path,
    validate_audio_binding,
)
from compare_whisper_fixtures import compare_fixtures
from smoke_native_whisper import (
    verify_loaded_model_fingerprint,
    verify_source_revision,
    verify_terminal_invariants,
)
from validate_interleaving_record import validate_record
from verify_native_interleaving import (
    EXPECTED_ASSERTIONS,
    PINNED_WHISPER_BASE,
    PINNED_WHISPER_TREE,
    decode_results_match,
    verify_audio_manifest_binding,
    verify_passed_assertions,
)

from whisper_runtime import ResourceVector

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


class NativeSmokeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacity = ResourceVector(
            memory_bytes=16,
            compute_units=2,
            stream_slots=1,
        )

    def test_loaded_model_fingerprint_must_match_when_declared(self) -> None:
        verify_loaded_model_fingerprint("sha256:abc", None)
        verify_loaded_model_fingerprint("sha256:abc", "sha256:abc")
        with self.assertRaisesRegex(RuntimeError, "fingerprint does not match"):
            verify_loaded_model_fingerprint("sha256:def", "sha256:abc")

    def test_terminal_invariants_accept_one_clean_commit(self) -> None:
        verify_terminal_invariants(
            request_status="committed",
            session_version=1,
            queue_depth=0,
            available=self.capacity,
            capacity=self.capacity,
        )

    def test_terminal_invariants_reject_each_failed_postcondition(self) -> None:
        smaller = ResourceVector(
            memory_bytes=8,
            compute_units=1,
            stream_slots=0,
        )
        cases = (
            ({"request_status": "aborted"}, "committed status"),
            ({"session_version": 2}, "session version 1"),
            ({"queue_depth": 1}, "admission slot"),
            ({"available": smaller}, "budget was not restored"),
        )
        defaults = {
            "request_status": "committed",
            "session_version": 1,
            "queue_depth": 0,
            "available": self.capacity,
            "capacity": self.capacity,
        }
        for override, message in cases:
            with self.subTest(override=override):
                arguments = {**defaults, **override}
                with self.assertRaisesRegex(RuntimeError, message):
                    verify_terminal_invariants(**arguments)


class _ComparableFeatures:
    def __init__(self, value: str) -> None:
        self.value = value

    def equal(self, other: object) -> bool:
        return isinstance(other, _ComparableFeatures) and self.value == other.value


class NativeInterleavingContractTests(unittest.TestCase):
    def result(self, *, text: str = "same") -> SimpleNamespace:
        return SimpleNamespace(
            text=text,
            tokens=[1, 2, 3],
            language="en",
            temperature=0.0,
            compression_ratio=1.0,
            avg_logprob=-0.2,
            no_speech_prob=float("nan"),
            audio_features=_ComparableFeatures("features"),
        )

    def evidence(self) -> dict[str, object]:
        result = {
            "text": "same",
            "language": "en",
            "token_count": 3,
            "tokens_sha256": "1" * 64,
            "audio_features_sha256": "2" * 64,
            "temperature": 0.0,
            "compression_ratio": 1.0,
            "avg_logprob": -0.2,
            "no_speech_prob": None,
        }
        manifest = ROOT / "patches" / "openai-whisper" / "SHA256SUMS"
        return {
            "schema_version": "1",
            "recorded_at": "2026-09-04T00:00:00Z",
            "status": "passed",
            "scope": "patched_backend",
            "runtime": {
                "version": "0.1.0.dev0",
                "git_commit": "0" * 40,
                "git_tree": "1" * 40,
                "clean": True,
            },
            "backend": {
                "name": "openai-whisper-suspendable",
                "base_commit": PINNED_WHISPER_BASE,
                "applied_commit": "3" * 40,
                "git_tree": PINNED_WHISPER_TREE,
                "clean": True,
                "patch_manifest": "patches/openai-whisper/SHA256SUMS",
                "patch_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
            },
            "environment": {
                "platform": "test",
                "python": "3.13.1",
                "torch": "2.6.0+cpu",
                "numpy": "2.5.2",
                "tiktoken": "0.14.0",
                "numba": "0.67.0",
                "tqdm": "4.70.0",
                "more_itertools": "11.1.0",
                "jsonschema": "4.25.1",
                "ffmpeg": "ffmpeg version test",
                "cpu_threads": 1,
            },
            "model": {
                "name": "tiny.en",
                "device": "cpu",
                "checkpoint_sha256": "7" * 64,
                "loaded_state_before": "sha256:" + "8" * 64,
                "loaded_state_after": "sha256:" + "8" * 64,
                "execution_state_before": "sha256:" + "9" * 64,
                "execution_state_after": "sha256:" + "9" * 64,
            },
            "input": {
                "fixture_id": "openai-whisper-jfk-flac",
                "path": "tests/jfk.flac",
                "file_sha256": (
                    "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
                ),
                "size_bytes": 1152693,
                "sample_rate_hz": 16000,
                "source_sample_count": 176000,
                "cancelled": {
                    "sample_start": 0,
                    "sample_end": 88000,
                    "pcm_sha256": "3" * 64,
                    "mel_sha256": "4" * 64,
                },
                "survivor": {
                    "sample_start": 0,
                    "sample_end": 176000,
                    "pcm_sha256": "5" * 64,
                    "mel_sha256": "6" * 64,
                },
            },
            "execution": {
                "mode": "deterministic_sequential_interleaving",
                "two_overlapping_run_lifetimes": True,
                "parallel_kernels": False,
                "cancellation": "explicit_cleanup_after_token_step",
                "survivor_steps": 2,
                "cancelled_steps": 1,
                "survivor_cache_entries_at_cancellation": 16,
                "numeric_absolute_tolerance": 0.0,
                "elapsed_seconds": {
                    "baseline": 1.0,
                    "interleaving": 2.0,
                    "reuse_control": 1.0,
                    "total": 4.1,
                },
                "timing_is_benchmark": False,
                "schedule": [
                    "start:cancelled",
                    "start:survivor",
                    "prefill:cancelled",
                    "prefill:survivor",
                    "step:cancelled:1",
                    "step:survivor:1",
                    "cancel:cancelled",
                    "cleanup:cancelled:idempotent",
                    "step:survivor:2",
                    "finalize:survivor",
                    "cleanup:survivor:idempotent",
                ],
            },
            "assertions": {name: True for name in EXPECTED_ASSERTIONS},
            "results": {
                "isolated_baseline": copy.deepcopy(result),
                "survivor": copy.deepcopy(result),
                "reuse_control": copy.deepcopy(result),
            },
        }

    def test_result_comparison_handles_nan_and_detects_content_changes(self) -> None:
        self.assertTrue(
            decode_results_match(self.result(), self.result(), absolute_tolerance=0.0)
        )
        self.assertFalse(
            decode_results_match(
                self.result(),
                self.result(text="different"),
                absolute_tolerance=0.0,
            )
        )

    def test_all_named_assertions_must_pass(self) -> None:
        passed = {name: True for name in EXPECTED_ASSERTIONS}
        verify_passed_assertions(passed)
        failed = dict(passed)
        failed["model_state_unchanged"] = False
        with self.assertRaisesRegex(RuntimeError, "model_state_unchanged"):
            verify_passed_assertions(failed)
        missing = dict(passed)
        del missing["two_live_runs"]
        with self.assertRaisesRegex(RuntimeError, "two_live_runs"):
            verify_passed_assertions(missing)

    def test_audio_bytes_must_match_the_declared_fixture(self) -> None:
        verify_audio_manifest_binding(
            fixture_id="openai-whisper-jfk-flac",
            file_sha256=(
                "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
            ),
            size_bytes=1152693,
            sample_rate_hz=16000,
            sample_count=176000,
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            verify_audio_manifest_binding(
                fixture_id="openai-whisper-jfk-flac",
                file_sha256="0" * 64,
                size_bytes=1152693,
                sample_rate_hz=16000,
                sample_count=176000,
            )

    def test_semantic_validator_rejects_schedule_and_result_changes(self) -> None:
        evidence = self.evidence()
        self.assertEqual(
            validate_record(evidence, "evidence/test.json"),
            [],
        )

        evidence["execution"]["schedule"][5] = "step:survivor:2"
        evidence["results"]["survivor"]["text"] = "different"
        failures = validate_record(evidence, "evidence/test.json")
        self.assertTrue(any("schedule" in failure for failure in failures))
        self.assertTrue(any("survivor differs" in failure for failure in failures))

    def test_record_validator_rejects_non_finite_json_numbers(self) -> None:
        evidence = self.evidence()
        evidence["execution"]["elapsed_seconds"]["total"] = float("nan")
        failures = validate_record(evidence, "evidence/test.json")
        self.assertTrue(any("non-finite" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
