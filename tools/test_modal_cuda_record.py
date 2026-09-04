from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any

from validate_modal_cuda_record import (
    DEFAULT_SCHEMA,
    EXPECTED_ASSERTIONS,
    read_json,
    validate_record,
)

RUNTIME_COMMIT = "78588504aa5d02109334c99da1ccb3be84021ae7"
RUNTIME_TREE = "ec665a647154f7a86c63bd0bfd30fe8b4993bc58"
EXPECTED_TEXT = (
    "And so my fellow Americans ask not what your country can do for you, "
    "ask what you can do for your country."
)


def valid_record() -> dict[str, Any]:
    resource_vector = {
        "memory_bytes": 1_000_000_000,
        "compute_units": 1,
        "stream_slots": 1,
    }
    denied_probe = {
        "denied": True,
        "exception_type": "PermissionError",
        "errno": 13,
    }
    return {
        "schema_version": "1",
        "recorded_at": "2026-09-04T10:00:00Z",
        "status": "passed",
        "scope": {
            "evidence_kind": "patched-whisper-cuda-readiness",
            "statement": (
                "One pinned patched Whisper backend decoded one pinned fixture "
                "on one Modal T4. No runtime transaction was admitted or executed."
            ),
            "runtime_adapter_exercised": False,
        },
        "claims": {
            "patched_backend_cuda_decode": True,
            "runtime_adapter_exercised": False,
            "worker_admission_exercised": False,
            "transaction_lifecycle_exercised": False,
            "cuda_completion_fence_exercised": False,
            "performance_benchmark": False,
        },
        "runtime": {
            "repository": "https://github.com/billmedj/whisper-runtime.git",
            "native_adapter_path": "src/whisper_runtime/adapters/native_whisper.py",
            "native_adapter_sha256": (
                "3388992843384d2a4259588e9bc0e22dd971b7fd2fe4162e515330ef8b480d4c"
            ),
            "git_commit": RUNTIME_COMMIT,
            "git_tree": RUNTIME_TREE,
            "clean": True,
        },
        "backend": {
            "repository": "https://github.com/openai/whisper.git",
            "base_commit": "86098128c0b4f24f0e2aa2994de830614b474227",
            "base_tree": "f7b3cb8e12a2e84dccacc4c858c33d5a9c114688",
            "applied_commit": "1" * 40,
            "git_tree": "c011d2563c26763b5f147026e6b18ef85bccd4fb",
            "clean": True,
            "patch_manifest": "patches/openai-whisper/SHA256SUMS",
            "patch_manifest_sha256": (
                "0fa1a833b0c489056d77da21188519c7e16fde7825d06bc5b902ba23a01abeb5"
            ),
        },
        "modal": {
            "sdk_version": "1.5.5",
            "function_call_id": "fc-test",
            "image_id": "im-test",
            "task_id": "ta-test",
            "environment": "validation",
            "cloud_provider": "aws",
            "region": "us-east",
            "network_blocked": True,
            "modal_access_restricted": True,
            "model_cache": {
                "name": "whisper-runtime-model-cache-v1",
                "generation": 1,
                "mount_path": "/models",
                "read_only": True,
                "write_probe": denied_probe,
            },
            "network_probe": {
                **denied_probe,
                "target": "1.1.1.1:443",
            },
        },
        "environment": {
            "python": "3.13.7",
            "platform": "Linux-6.8-x86_64-with-glibc2.36",
            "torch": "2.6.0+cu124",
            "torch_git_version": "2" * 40,
            "cuda_runtime": "12.4",
            "cudnn": "90100",
            "nvidia_driver": "570.133.20",
            "ffmpeg": "ffmpeg version 5.1.6-0+deb12u1 Copyright FFmpeg",
            "numpy": "2.5.2",
            "tiktoken": "0.14.0",
            "numba": "0.67.0",
            "tqdm": "4.70.0",
            "more_itertools": "11.1.0",
        },
        "gpu": {
            "requested": "T4",
            "visible_device_count": 1,
            "device_index": 0,
            "name": "Tesla T4",
            "capability_major": 7,
            "capability_minor": 5,
            "total_memory_bytes": 15_000_000_000,
        },
        "model": {
            "name": "tiny.en",
            "device": "cuda:0",
            "dtype": "torch.float32",
            "checkpoint_path": "model-cache-v1/tiny.en.pt",
            "checkpoint_sha256": (
                "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
            ),
            "loaded_state_sha256_before": (
                "sha256:8041a80119a588f542472da35e97d0372fce1d9709ed9874475e9c03deac5de6"
            ),
            "loaded_state_sha256_after": (
                "sha256:8041a80119a588f542472da35e97d0372fce1d9709ed9874475e9c03deac5de6"
            ),
        },
        "input": {
            "fixture_id": "openai-whisper-jfk-flac",
            "path": "openai-whisper/tests/jfk.flac",
            "sha256": (
                "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
            ),
            "size_bytes": 1_152_693,
            "sample_rate_hz": 16_000,
            "sample_count": 176_000,
            "decoded_pcm_sha256": "3" * 64,
        },
        "decode": {
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,
            "without_timestamps": True,
            "fp16": True,
            "rng_seed": 7,
            "staged_result_count": 1,
            "text": EXPECTED_TEXT,
            "expected_text": EXPECTED_TEXT,
            "expected_text_matched": True,
            "reuse_text": EXPECTED_TEXT,
            "reuse_matched": True,
            "cleanup_calls": 2,
            "reuse_cleanup_calls": 2,
        },
        "timing": {
            "staged_decode_seconds": 1.0,
            "reuse_decode_seconds": 1.0,
            "total_seconds": 3.0,
            "synchronized": True,
        },
        "memory": {
            "allocated_before_bytes": 100,
            "peak_allocated_bytes": 300,
            "peak_reserved_bytes": 400,
            "allocated_after_bytes": 200,
            "measured": True,
            "enforced": False,
        },
        "adapter_boundary": {
            "adapter": "NativeWhisperAdapter",
            "device": "cuda:0",
            "attempted": True,
            "rejected_before_admission": True,
            "exception_type": "ValueError",
            "message": (
                "the native adapter requires an explicit CPU device for the mel tensor"
            ),
            "queue_depth_before": 0,
            "queue_depth_after": 0,
            "budget_available_before": resource_vector,
            "budget_available_after": copy.deepcopy(resource_vector),
        },
        "assertions": dict.fromkeys(EXPECTED_ASSERTIONS, True),
    }


class ModalCudaRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = read_json(DEFAULT_SCHEMA)

    def failures(self, record: dict[str, Any]) -> list[str]:
        return validate_record(
            record,
            self.schema,
            expected_runtime_commit=RUNTIME_COMMIT,
            expected_runtime_tree=RUNTIME_TREE,
        )

    def test_valid_record_passes(self) -> None:
        self.assertEqual(self.failures(valid_record()), [])

    def test_unknown_nested_field_is_rejected(self) -> None:
        record = valid_record()
        record["gpu"]["marketing_name"] = "cheap"  # type: ignore[index]
        self.assertTrue(any("marketing_name" in item for item in self.failures(record)))

    def test_runtime_adapter_claim_cannot_be_expanded(self) -> None:
        record = valid_record()
        record["claims"]["runtime_adapter_exercised"] = True  # type: ignore[index]
        self.assertTrue(
            any("runtime_adapter_exercised" in item for item in self.failures(record))
        )

    def test_runtime_tree_is_bound_by_caller(self) -> None:
        record = valid_record()
        record["runtime"]["git_tree"] = "4" * 40  # type: ignore[index]
        self.assertIn(
            "runtime.git_tree does not match the requested tree",
            self.failures(record),
        )

    def test_runtime_commit_is_bound_by_caller(self) -> None:
        record = valid_record()
        record["runtime"]["git_commit"] = "4" * 40  # type: ignore[index]
        self.assertIn(
            "runtime.git_commit does not match the requested commit",
            self.failures(record),
        )

    def test_sensitive_value_is_rejected(self) -> None:
        record = valid_record()
        record["environment"]["platform"] = "MODAL_TOKEN_SECRET"  # type: ignore[index]
        self.assertTrue(
            any("appears to contain a secret" in item for item in self.failures(record))
        )

    def test_absolute_user_path_is_rejected(self) -> None:
        record = valid_record()
        separator = chr(92)
        record["environment"]["platform"] = (  # type: ignore[index]
            f"C:{separator}Users{separator}operator{separator}host"
        )
        self.assertTrue(
            any("absolute user path" in item for item in self.failures(record))
        )

    def test_changed_budget_is_rejected(self) -> None:
        record = valid_record()
        record["adapter_boundary"]["budget_available_after"][  # type: ignore[index]
            "stream_slots"
        ] = 0
        self.assertTrue(any("stream_slots" in item for item in self.failures(record)))

    def test_false_and_unknown_assertions_are_rejected(self) -> None:
        record = valid_record()
        record["assertions"]["cleanup_idempotent"] = False  # type: ignore[index]
        record["assertions"]["unbounded_claim"] = True  # type: ignore[index]
        failures = self.failures(record)
        self.assertTrue(any("cleanup_idempotent" in item for item in failures))
        self.assertTrue(any("unbounded_claim" in item for item in failures))

    def test_incoherent_timing_is_rejected(self) -> None:
        record = valid_record()
        record["timing"]["total_seconds"] = 1.5  # type: ignore[index]
        self.assertIn(
            "timing.total_seconds is shorter than measured decode phases",
            self.failures(record),
        )

    def test_incoherent_cuda_memory_is_rejected(self) -> None:
        record = valid_record()
        record["memory"]["peak_reserved_bytes"] = 250  # type: ignore[index]
        self.assertIn(
            "reserved CUDA memory is below allocated CUDA memory",
            self.failures(record),
        )

    def test_non_finite_json_is_rejected_before_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                read_json(path)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(
                '{"status": "passed", "status": "failed"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                read_json(path)


if __name__ == "__main__":
    unittest.main()
