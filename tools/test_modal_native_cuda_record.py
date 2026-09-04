from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any

from validate_modal_native_cuda_record import (
    DEFAULT_SCHEMA,
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_ASSERTIONS,
    EXPECTED_CLAIMS,
    EXPECTED_SOURCE_PATHS,
    FULL_RESOURCES,
    ZERO_RESOURCES,
    read_json,
    validate_record,
)

RUNTIME_COMMIT = "0" * 40
RUNTIME_TREE = "1" * 40
EXPECTED_TEXT = (
    "And so my fellow Americans ask not what your country can do for you, "
    "ask what you can do for your country."
)


def _kind(name: str) -> str:
    if name.startswith("cuda:"):
        return "cuda"
    if name.startswith("controller:"):
        return "controller"
    if name.startswith(("task:", "run:", "model:")):
        return "backend"
    return "runtime"


def _trace(
    events: list[tuple[str, str | None, dict[str, Any] | None]],
    *,
    controller_names: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, (name, stream, state) in enumerate(events, start=1):
        event: dict[str, Any] = {
            "sequence": index,
            "offset_ns": index * 100,
            "name": name,
            "kind": _kind(name),
            "thread": "controller" if name in controller_names else "decode",
            "stream": stream,
        }
        if state is not None:
            event["state"] = state
        result.append(event)
    return result


def _fence_state(request_status: str) -> dict[str, Any]:
    return {
        "request_status": request_status,
        "session_version": 0,
        "queue_depth": 1,
        "lease_count": 1,
        "budget_available": copy.deepcopy(ZERO_RESOURCES),
    }


def _duplicate_event(trace: list[dict[str, Any]], name: str) -> None:
    index = next(index for index, event in enumerate(trace) if event["name"] == name)
    trace.insert(index + 1, copy.deepcopy(trace[index]))
    for sequence, event in enumerate(trace, start=1):
        event["sequence"] = sequence
        event["offset_ns"] = sequence * 100


def _success_trace() -> list[dict[str, Any]]:
    return _trace(
        [
            ("budget:lease:acquired", None, None),
            ("worker:admitted", None, None),
            ("cuda:device-synchronize:begin", None, None),
            ("cuda:device-synchronize:return", None, None),
            ("cuda:stream:create", "stream-1", None),
            ("task:construct:begin", "stream-1", None),
            ("task:construct:return", "stream-1", None),
            ("run:start:begin", "stream-1", None),
            ("run:start:submitted", "stream-1", None),
            ("run:child-generator:verified", "stream-1", None),
            ("run:prefill:begin", "stream-1", None),
            ("run:prefill:submitted", "stream-1", None),
            ("run:step:begin", "stream-1", None),
            ("run:step:submitted", "stream-1", None),
            ("run:finalize:begin", "stream-1", None),
            ("run:finalize:submitted", "stream-1", None),
            ("model:identity:verified", "stream-1", None),
            ("run:cleanup:begin", "stream-1", None),
            ("run:cleanup:submitted", "stream-1", None),
            ("cuda:event-1:create", None, None),
            ("cuda:event-1:record", "stream-1", None),
            ("cuda:event-1:synchronize:begin", "stream-1", None),
            ("cuda:event-1:query:return", "stream-1", None),
            (
                "cuda:event-1:synchronize:return",
                "stream-1",
                _fence_state("running"),
            ),
            ("session:commit:begin", None, None),
            ("session:commit:return", None, None),
            ("budget:lease:release:begin", None, None),
            ("budget:lease:release:return", None, None),
        ]
    )


def _cancellation_trace() -> list[dict[str, Any]]:
    controller = frozenset({"controller:cancel:begin", "controller:cancel:return"})
    return _trace(
        [
            ("budget:lease:acquired", None, None),
            ("worker:admitted", None, None),
            ("cuda:stream:create", "stream-1", None),
            ("run:start:submitted", "stream-1", None),
            ("run:child-generator:verified", "stream-1", None),
            ("run:step:begin", "stream-1", None),
            ("run:step:submitted", "stream-1", None),
            ("run:cancellation-rendezvous:incomplete", "stream-1", None),
            ("controller:cancel:begin", None, None),
            ("controller:cancel:return", None, None),
            ("run:cleanup:begin", "stream-1", None),
            ("run:cleanup:submitted", "stream-1", None),
            ("cuda:event-1:create", None, None),
            ("cuda:event-1:record", "stream-1", None),
            ("cuda:event-1:synchronize:begin", "stream-1", None),
            ("cuda:event-1:query:return", "stream-1", None),
            (
                "cuda:event-1:synchronize:return",
                "stream-1",
                _fence_state("cancelled"),
            ),
            ("budget:lease:release:begin", None, None),
            ("budget:lease:release:return", None, None),
        ],
        controller_names=controller,
    )


def _recovery_trace() -> list[dict[str, Any]]:
    return _trace(
        [
            ("budget:lease:acquired", None, None),
            ("worker:admitted", None, None),
            ("cuda:stream:create", "stream-1", None),
            ("run:start:submitted", "stream-1", None),
            ("run:child-generator:verified", "stream-1", None),
            ("run:finalize:submitted", "stream-1", None),
            ("run:cleanup:begin", "stream-1", None),
            ("run:cleanup:submitted", "stream-1", None),
            ("cuda:event-1:create", None, None),
            ("cuda:event-1:record", "stream-1", None),
            ("cuda:event-1:synchronize:begin", "stream-1", None),
            ("cuda:event-1:synchronize:injected-failure", "stream-1", None),
            ("run:cleanup:begin", "stream-1", None),
            ("run:cleanup:submitted", "stream-1", None),
            ("cuda:event-2:create", None, None),
            ("cuda:event-2:record", "stream-1", None),
            ("cuda:event-2:synchronize:begin", "stream-1", None),
            ("cuda:event-2:synchronize:injected-failure", "stream-1", None),
            ("runtime:manual-recovery:begin", None, None),
            ("run:cleanup:begin", "stream-1", None),
            ("run:cleanup:submitted", "stream-1", None),
            ("cuda:event-3:create", None, None),
            ("cuda:event-3:record", "stream-1", None),
            ("cuda:event-3:synchronize:begin", "stream-1", None),
            ("cuda:event-3:query:return", "stream-1", None),
            (
                "cuda:event-3:synchronize:return",
                "stream-1",
                _fence_state("running"),
            ),
            ("budget:lease:release:begin", None, None),
            ("budget:lease:release:return", None, None),
            ("runtime:manual-recovery:return", None, None),
        ]
    )


def _memory_snapshot() -> dict[str, int]:
    return {
        "allocated_bytes": 100,
        "reserved_bytes": 200,
        "peak_allocated_bytes": 300,
        "peak_reserved_bytes": 400,
    }


def _success() -> dict[str, Any]:
    return {
        "outcome": "committed",
        "request_status": "committed",
        "transaction_status": "committed",
        "session_version_before": 0,
        "session_version_after": 1,
        "window_count_after": 1,
        "text": EXPECTED_TEXT,
        "expected_text_matched": True,
        "queue_depth_after": 0,
        "budget_available_before": copy.deepcopy(FULL_RESOURCES),
        "budget_available_after": copy.deepcopy(FULL_RESOURCES),
        "backend_instrumented": True,
        "child_generator_checks": 1,
        "child_generators_on_profile_device": True,
        "child_generators_distinct": True,
        "traced_cleanup_calls": 1,
        "traced_stream_count": 1,
        "traced_event_count": 1,
        "wall_seconds": 1.0,
        "memory_before": _memory_snapshot(),
        "memory_after": _memory_snapshot(),
        "trace": _success_trace(),
    }


def valid_record() -> dict[str, Any]:
    denied_probe = {"denied": True, "exception_type": "PermissionError", "errno": 13}
    success = _success()
    return {
        "schema_version": "2",
        "recorded_at": "2026-09-04T10:00:00Z",
        "status": "passed",
        "scope": {
            "evidence_kind": "native-whisper-cuda-transaction",
            "statement": (
                "One pinned runtime completed, cancelled, quarantined, recovered, "
                "and reused NativeWhisperAdapter transactions on one Modal T4."
            ),
            "fault_injection_used": True,
        },
        "claims": copy.deepcopy(EXPECTED_CLAIMS),
        "runtime": {
            "repository": "https://github.com/billmedj/whisper-runtime.git",
            "git_commit": RUNTIME_COMMIT,
            "git_tree": RUNTIME_TREE,
            "clean": True,
            "source_files": [
                {
                    "path": path,
                    "sha256": EXPECTED_ADAPTER_SHA256
                    if index == 0
                    else f"{index + 2}" * 64,
                }
                for index, path in enumerate(EXPECTED_SOURCE_PATHS)
            ],
        },
        "backend": {
            "repository": "https://github.com/openai/whisper.git",
            "base_commit": "86098128c0b4f24f0e2aa2994de830614b474227",
            "base_tree": "f7b3cb8e12a2e84dccacc4c858c33d5a9c114688",
            "applied_commit": "8" * 40,
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
            "region": "us-west-2",
            "network_blocked": True,
            "modal_access_restricted": True,
            "model_cache": {
                "name": "whisper-runtime-model-cache-v1",
                "generation": 1,
                "mount_path": "/models",
                "read_only": True,
                "write_probe": denied_probe,
            },
            "network_probe": {**denied_probe, "target": "1.1.1.1:443"},
        },
        "environment": {
            "python": "3.13.7",
            "platform": "Linux-6.8-x86_64-with-glibc2.36",
            "torch": "2.6.0+cu124",
            "torch_git_version": "9" * 40,
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
            "evaluation_mode": True,
            "parameter_tensor_count": 42,
            "buffer_tensor_count": 17,
            "all_parameters_and_buffers_on_profile_device": True,
            "all_floating_tensors_fp32": True,
        },
        "input": {
            "fixture_id": "openai-whisper-jfk-flac",
            "path": "openai-whisper/tests/jfk.flac",
            "sha256": "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715",
            "size_bytes": 1_152_693,
            "sample_rate_hz": 16_000,
            "sample_count": 176_000,
            "decoded_pcm_sha256": "a" * 64,
            "mel_device_at_boundary": "cpu",
            "mel_dtype_at_boundary": "torch.float32",
        },
        "profile": {
            "profile_id": "tiny.en/cuda-0-float32-v1",
            "device": "cuda:0",
            "max_concurrent_decodes": 1,
            "resources": copy.deepcopy(FULL_RESOURCES),
            "fp16": False,
            "gpu_memory_measured": True,
            "gpu_memory_enforced": False,
        },
        "success": success,
        "cancellation": {
            "outcome": "cancelled",
            "exception_type": "RequestCancelledError",
            "cancel_returned": True,
            "request_status": "cancelled",
            "transaction_status": "aborted",
            "session_version_at_cancel": 0,
            "session_version_after": 0,
            "window_count_after": 0,
            "queue_depth_at_cancel": 1,
            "lease_count_at_cancel": 1,
            "queue_depth_after": 0,
            "budget_available_after": copy.deepcopy(FULL_RESOURCES),
            "backend_instrumented": True,
            "first_step_returned": False,
            "run_complete_after_first_step": False,
            "child_generator_checks": 1,
            "child_generators_on_profile_device": True,
            "child_generators_distinct": True,
            "traced_cleanup_calls": 1,
            "traced_stream_count": 1,
            "traced_event_count": 1,
            "controller_cuda_calls": 0,
            "trace": _cancellation_trace(),
        },
        "recovery": {
            "outcome": "recovered",
            "injected_failure": "cuda-event-synchronize-before-delegate",
            "injected_failure_count": 2,
            "retained_exception_type": "TransactionRetainedError",
            "retained_transaction_status": "quarantined",
            "retained_queue_depth": 1,
            "retained_lease_count": 1,
            "retained_budget_available": copy.deepcopy(ZERO_RESOURCES),
            "retained_session_version": 0,
            "retained_window_count": 0,
            "blocked_error_same_instance": True,
            "blocked_request_status": "created",
            "blocked_session_version": 0,
            "blocked_queue_depth": 1,
            "blocked_lease_count": 1,
            "recovery_returned": True,
            "recovered_transaction_status": "aborted",
            "queue_depth_after_recovery": 0,
            "budget_available_after_recovery": copy.deepcopy(FULL_RESOURCES),
            "session_version_after_recovery": 0,
            "request_status_after_recovery": "aborted",
            "child_generator_checks": 1,
            "child_generators_on_profile_device": True,
            "child_generators_distinct": True,
            "traced_cleanup_calls_before_recovery_complete": 3,
            "traced_event_count_before_recovery_complete": 3,
            "trace": _recovery_trace(),
            "post_recovery_reuse": _success(),
        },
        "unproxied_reuse": {
            **_success(),
            "backend_instrumented": False,
            "child_generator_checks": 0,
            "child_generators_on_profile_device": True,
            "child_generators_distinct": True,
            "traced_cleanup_calls": 0,
            "traced_stream_count": 0,
            "traced_event_count": 0,
            "trace": _trace(
                [
                    ("budget:lease:acquired", None, None),
                    ("worker:admitted", None, None),
                    ("session:commit:begin", None, None),
                    ("session:commit:return", None, None),
                    ("budget:lease:release:begin", None, None),
                    ("budget:lease:release:return", None, None),
                ]
            ),
        },
        "memory": {
            "baseline": _memory_snapshot(),
            "final": _memory_snapshot(),
            "observed_success_peak_delta_bytes": 200,
            "declared_memory_bytes": 1_000_000_000,
            "observed_peak_within_declaration": True,
        },
        "timing": {"total_seconds": 4.0, "performance_benchmark": False},
        "assertions": dict.fromkeys(EXPECTED_ASSERTIONS, True),
    }


class ModalNativeCudaRecordTests(unittest.TestCase):
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

    def assert_rejected(self, record: dict[str, Any], fragment: str) -> None:
        failures = self.failures(record)
        self.assertTrue(any(fragment in failure for failure in failures), failures)

    def test_valid_record_passes(self) -> None:
        self.assertEqual(self.failures(valid_record()), [])

    def test_unknown_field_is_rejected(self) -> None:
        record = valid_record()
        record["success"]["marketing_claim"] = True
        self.assert_rejected(record, "marketing_claim")

    def test_claim_boundary_cannot_expand(self) -> None:
        record = valid_record()
        record["claims"]["production_readiness"] = True
        self.assert_rejected(record, "production_readiness")

    def test_source_set_and_adapter_digest_are_exact(self) -> None:
        record = valid_record()
        record["runtime"]["source_files"][0]["sha256"] = "f" * 64
        self.assert_rejected(record, "adapter digest")
        record = valid_record()
        record["runtime"]["source_files"].reverse()
        self.assert_rejected(record, "exact pinned source set")

    def test_runtime_commit_and_tree_are_bound_by_caller(self) -> None:
        record = valid_record()
        record["runtime"]["git_commit"] = "e" * 40
        self.assert_rejected(record, "requested commit")
        record = valid_record()
        record["runtime"]["git_tree"] = "e" * 40
        self.assert_rejected(record, "requested tree")

    def test_fence_must_precede_publication_and_release(self) -> None:
        record = valid_record()
        trace = record["success"]["trace"]
        fence_index = next(
            index
            for index, item in enumerate(trace)
            if item["name"].endswith("synchronize:return")
        )
        commit_index = next(
            index
            for index, item in enumerate(trace)
            if item["name"] == "session:commit:begin"
        )
        trace[fence_index], trace[commit_index] = (
            trace[commit_index],
            trace[fence_index],
        )
        for index, item in enumerate(trace, start=1):
            item["sequence"] = index
            item["offset_ns"] = index * 100
        self.assert_rejected(record, "ordered event")

    def test_fence_state_binds_unpublished_admitted_resources(self) -> None:
        record = valid_record()
        state = next(
            item["state"]
            for item in record["success"]["trace"]
            if item["name"] == "cuda:event-1:synchronize:return"
        )
        state["session_version"] = 1
        self.assert_rejected(record, "session_version")
        record = valid_record()
        state = next(
            item["state"] for item in record["cancellation"]["trace"] if "state" in item
        )
        state["budget_available"] = copy.deepcopy(FULL_RESOURCES)
        self.assert_rejected(record, "budget_available")

    def test_state_is_rejected_on_non_fence_event(self) -> None:
        record = valid_record()
        record["success"]["trace"][0]["state"] = _fence_state("running")
        self.assert_rejected(record, "only valid on a fence return")

    def test_controller_cancellation_cannot_call_cuda(self) -> None:
        record = valid_record()
        cancel = next(
            item
            for item in record["cancellation"]["trace"]
            if item["name"] == "controller:cancel:return"
        )
        cancel["name"] = "cuda:device-synchronize:return"
        cancel["kind"] = "cuda"
        self.assert_rejected(record, "controller thread")

    def test_backend_event_context_is_bound(self) -> None:
        record = valid_record()
        event = next(
            item
            for item in record["success"]["trace"]
            if item["name"] == "run:prefill:submitted"
        )
        event.update(kind="runtime", thread="controller", stream=None)
        self.assert_rejected(record, "backend/decode/stream-1")

    def test_cancellation_controller_event_context_is_bound(self) -> None:
        for event_name in ("controller:cancel:begin", "controller:cancel:return"):
            with self.subTest(event_name=event_name):
                record = valid_record()
                event = next(
                    item
                    for item in record["cancellation"]["trace"]
                    if item["name"] == event_name
                )
                event.update(kind="runtime", thread="decode")
                self.assert_rejected(record, "controller/controller/no-stream")

    def test_cancellation_cannot_publish(self) -> None:
        record = valid_record()
        trace = record["cancellation"]["trace"]
        release = next(
            index
            for index, item in enumerate(trace)
            if item["name"] == "budget:lease:release:begin"
        )
        trace.insert(
            release,
            {
                "sequence": 0,
                "offset_ns": 0,
                "name": "session:commit:begin",
                "kind": "runtime",
                "thread": "decode",
                "stream": None,
            },
        )
        for index, item in enumerate(trace, start=1):
            item["sequence"] = index
            item["offset_ns"] = index * 100
        self.assert_rejected(record, "published a session commit")

    def test_cancellation_requires_an_incomplete_first_step_rendezvous(self) -> None:
        record = valid_record()
        record["cancellation"]["first_step_returned"] = True
        self.assert_rejected(record, "first_step_returned")
        record = valid_record()
        record["cancellation"]["run_complete_after_first_step"] = True
        self.assert_rejected(record, "run_complete_after_first_step")
        record = valid_record()
        event = next(
            item
            for item in record["cancellation"]["trace"]
            if item["name"] == "run:cancellation-rendezvous:incomplete"
        )
        event["name"] = "run:cancellation-rendezvous:missing"
        self.assert_rejected(record, "run:cancellation-rendezvous:incomplete")

    def test_child_generator_evidence_is_exact(self) -> None:
        record = valid_record()
        record["success"]["child_generator_checks"] = 0
        self.assert_rejected(record, "child_generator_checks")
        record = valid_record()
        record["cancellation"]["child_generators_on_profile_device"] = False
        self.assert_rejected(record, "child_generators_on_profile_device")
        record = valid_record()
        record["recovery"]["child_generators_distinct"] = False
        self.assert_rejected(record, "child_generators_distinct")
        record = valid_record()
        child_event = next(
            item
            for item in record["success"]["trace"]
            if item["name"] == "run:child-generator:verified"
        )
        child_event["name"] = "run:child-generator:unchecked"
        self.assert_rejected(record, "run:child-generator:verified")

    def test_unproxied_control_has_no_child_generator_proxy_trace(self) -> None:
        record = valid_record()
        record["unproxied_reuse"]["trace"].insert(
            2,
            {
                "sequence": 0,
                "offset_ns": 0,
                "name": "run:child-generator:verified",
                "kind": "backend",
                "thread": "decode",
                "stream": "stream-1",
            },
        )
        for index, item in enumerate(record["unproxied_reuse"]["trace"], start=1):
            item["sequence"] = index
            item["offset_ns"] = index * 100
        self.assert_rejected(record, "backend or CUDA proxy events")

    def test_model_profile_evidence_is_exact(self) -> None:
        record = valid_record()
        record["model"]["evaluation_mode"] = False
        self.assert_rejected(record, "evaluation_mode")
        record = valid_record()
        record["model"]["parameter_tensor_count"] = 0
        self.assert_rejected(record, "parameter_tensor_count")
        record = valid_record()
        record["model"]["buffer_tensor_count"] = 0
        self.assert_rejected(record, "buffer_tensor_count")
        record = valid_record()
        record["model"]["all_parameters_and_buffers_on_profile_device"] = False
        self.assert_rejected(record, "all_parameters_and_buffers_on_profile_device")
        record = valid_record()
        record["model"]["all_floating_tensors_fp32"] = False
        self.assert_rejected(record, "all_floating_tensors_fp32")

    def test_recovery_requires_two_failures_and_three_cleanup_attempts(self) -> None:
        record = valid_record()
        item = next(
            item
            for item in record["recovery"]["trace"]
            if item["name"] == "cuda:event-2:synchronize:injected-failure"
        )
        item["name"] = "cuda:event-2:synchronize:return"
        item["state"] = _fence_state("running")
        self.assert_rejected(record, "two ordered injected failures")
        record = valid_record()
        cleanup = [
            item
            for item in record["recovery"]["trace"]
            if item["name"] == "run:cleanup:submitted"
        ][-1]
        cleanup["name"] = "run:cleanup:skipped"
        self.assert_rejected(record, "exactly three cleanup attempts")

    def test_manual_recovery_event_name_kind_and_thread_are_exact(self) -> None:
        record = valid_record()
        event = next(
            item
            for item in record["recovery"]["trace"]
            if item["name"] == "runtime:manual-recovery:begin"
        )
        event["name"] = "controller:recover:begin"
        event["kind"] = "controller"
        self.assert_rejected(record, "runtime:manual-recovery:begin")
        record = valid_record()
        event = next(
            item
            for item in record["recovery"]["trace"]
            if item["name"] == "runtime:manual-recovery:return"
        )
        event["thread"] = "controller"
        self.assert_rejected(record, "runtime event on the decode thread")
        record = valid_record()
        event = next(
            item
            for item in record["recovery"]["trace"]
            if item["name"] == "runtime:manual-recovery:begin"
        )
        event["kind"] = "controller"
        self.assert_rejected(record, "runtime event on the decode thread")

    def test_retained_failure_holds_capacity_until_manual_recovery(self) -> None:
        record = valid_record()
        record["recovery"]["retained_budget_available"] = copy.deepcopy(FULL_RESOURCES)
        self.assert_rejected(record, "retained_budget_available")
        record = valid_record()
        record["recovery"]["blocked_queue_depth"] = 0
        self.assert_rejected(record, "blocked_queue_depth")

    def test_post_recovery_reuse_must_use_one_private_stream(self) -> None:
        record = valid_record()
        event = next(
            item
            for item in record["recovery"]["post_recovery_reuse"]["trace"]
            if item["name"] == "run:step:submitted"
        )
        event["stream"] = "stream-2"
        self.assert_rejected(record, "backend work did not use stream-1")

    def test_trace_sequence_and_time_are_monotonic(self) -> None:
        record = valid_record()
        record["success"]["trace"][2]["sequence"] = 99
        self.assert_rejected(record, "not contiguous")
        record = valid_record()
        record["success"]["trace"][2]["offset_ns"] = 0
        self.assert_rejected(record, "moves backwards")

    def test_summary_counts_cannot_hide_duplicate_success_events(self) -> None:
        record = valid_record()
        _duplicate_event(record["success"]["trace"], "cuda:stream:create")
        self.assert_rejected(record, "exactly 1 'cuda:stream:create'")

    def test_summary_counts_cannot_hide_duplicate_cancellation_events(self) -> None:
        record = valid_record()
        _duplicate_event(record["cancellation"]["trace"], "cuda:event-1:create")
        self.assert_rejected(record, "exactly 1 'cuda:event-1:create'")

    def test_recovery_event_and_cleanup_counts_are_derived_from_trace(self) -> None:
        record = valid_record()
        query = next(
            item
            for item in record["recovery"]["trace"]
            if item["name"] == "cuda:event-3:query:return"
        )
        query["name"] = "cuda:event-1:query:return"
        self.assert_rejected(record, "exactly 0 'cuda:event-1:query:return'")
        record = valid_record()
        _duplicate_event(record["recovery"]["trace"], "run:cleanup:begin")
        self.assert_rejected(record, "exactly 3 'run:cleanup:begin'")

    def test_memory_and_timing_are_coherent(self) -> None:
        record = valid_record()
        record["memory"]["observed_success_peak_delta_bytes"] = 1_000_000_001
        self.assert_rejected(record, "exceeds the declared memory vector")
        record = valid_record()
        record["memory"]["final"]["peak_reserved_bytes"] = 250
        self.assert_rejected(record, "peak_reserved_bytes is incoherent")
        record = valid_record()
        record["timing"]["total_seconds"] = 2.5
        self.assert_rejected(record, "shorter than its measured success phases")

    def test_secret_and_absolute_user_path_are_rejected(self) -> None:
        record = valid_record()
        record["environment"]["platform"] = "MODAL_TOKEN_SECRET"
        self.assert_rejected(record, "appears to contain a secret")
        record = valid_record()
        separator = chr(92)
        record["environment"]["platform"] = separator.join(
            ("C:", "Users", "operator", "checkout")
        )
        self.assert_rejected(record, "absolute user path")

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
