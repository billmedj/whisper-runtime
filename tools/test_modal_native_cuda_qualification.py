from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from validate_modal_native_cuda_qualification import (
    CANCELLATION_EVENTS,
    CONTROL_EVENTS,
    DEFAULT_QUALIFICATION_MANIFEST,
    DEFAULT_SCHEMA,
    FAULT_EVENTS,
    FAULT_POINTS,
    ROOT,
    SUCCESS_EVENTS,
    CheckoutIdentity,
    _repository_relative_path,
    bind_tracked_artifact,
    canonical_sha256,
    derive_checkout_identity,
    parse_args,
    read_json,
    sha256_file,
    summarize,
    validate_record,
)

RUNTIME_REPOSITORY = "https://github.com/billmedj/whisper-runtime"
BACKEND_REPOSITORY = "https://github.com/openai/whisper"
RUNTIME_COMMIT = "0" * 40
RUNTIME_TREE = "1" * 40
BACKEND_COMMIT = "a0b9695ae1cc52bad4b8626fe9fb6ea4ac0ee650"
BACKEND_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
PATCH_MANIFEST_SHA256 = "4" * 64
PRODUCER_SCRIPT_SHA256 = "5" * 64
SCHEMA_SHA256 = "6" * 64
VALIDATOR_SHA256 = "7" * 64
IMAGE_INPUTS_SHA256 = "8" * 64
PATCH_MANIFEST_PATH = "patches/openai-whisper/SHA256SUMS"
PRODUCER_SCRIPT_PATH = "infra/modal_native_cuda_qualification.py"
SCHEMA_PATH = "evidence/modal-native-cuda-qualification.schema.json"
VALIDATOR_PATH = "tools/validate_modal_native_cuda_qualification.py"
IMAGE_INPUTS_PATH = "infra/modal-native-cuda-image-inputs.lock"
QUALIFICATION_MANIFEST_PATH = "experiments/native-cuda-qualification-v1.json"
QUALIFICATION_MANIFEST_SHA256 = sha256_file(DEFAULT_QUALIFICATION_MANIFEST)
QUALIFICATION_MANIFEST = read_json(DEFAULT_QUALIFICATION_MANIFEST)
WORKER_ID = "9" * 64
RESULT_SHA256 = "dfe1af694958a82c9d89d31bf0378075a10d40f715e670b6e5169042c8680cec"
RUNTIME_IDENTITY = CheckoutIdentity(
    checkout=ROOT,
    repository=RUNTIME_REPOSITORY,
    git_commit=RUNTIME_COMMIT,
    git_tree=RUNTIME_TREE,
)
BACKEND_IDENTITY = CheckoutIdentity(
    checkout=ROOT,
    repository=BACKEND_REPOSITORY,
    git_commit=BACKEND_COMMIT,
    git_tree=BACKEND_TREE,
)


def _resource(memory: int, compute: int, streams: int) -> dict[str, int]:
    return {
        "memory_bytes": memory,
        "compute_units": compute,
        "stream_slots": streams,
    }


def _run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _make_git_checkout(
    parent: Path,
    name: str,
    *,
    origin: str = "https://github.com/example/fixture",
) -> Path:
    repository = parent / name
    repository.mkdir()
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.name", "Qualification Test")
    _run_git(repository, "config", "user.email", "qualification@example.invalid")
    tracked = repository / "tracked.txt"
    tracked.write_text("tracked at HEAD\n", encoding="utf-8")
    _run_git(repository, "add", "tracked.txt")
    _run_git(repository, "commit", "--quiet", "-m", "fixture")
    _run_git(repository, "remote", "add", "origin", origin)
    return repository


def _budget() -> dict[str, Any]:
    capacity = _resource(2_147_483_648, 1, 1)
    held = _resource(0, 0, 0)
    return {
        "available_before": capacity,
        "available_while_held": held,
        "available_at_quiescence": held,
        "available_after_release": capacity,
    }


def _memory(delta: int) -> dict[str, int]:
    baseline_allocated = 1_000
    baseline_reserved = 1_500
    return {
        "baseline_allocated_bytes": baseline_allocated,
        "final_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": baseline_allocated + delta,
        "baseline_reserved_bytes": baseline_reserved,
        "final_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": baseline_reserved + delta,
        "peak_allocated_delta_bytes": delta,
        "peak_reserved_delta_bytes": delta,
    }


def _identity(prefix: str) -> dict[str, str]:
    return {
        "run_id": f"{prefix}-run",
        "session_id": f"{prefix}-session",
        "request_id": f"{prefix}-request",
        "transaction_id": f"{prefix}-transaction",
        "lease_id": f"{prefix}-lease",
    }


def _success(prefix: str, iteration: int, delta: int) -> dict[str, Any]:
    return {
        **_identity(prefix),
        "iteration": iteration,
        "wall_ns": 1,
        "session_version_before": 0,
        "session_version_after": 1,
        "result_sha256": RESULT_SHA256,
        "memory": _memory(delta),
        "budget": _budget(),
    }


def _control(prefix: str, iteration: int, delta: int) -> dict[str, Any]:
    return {
        "run_id": f"{prefix}-run",
        "iteration": iteration,
        "wall_ns": 1,
        "result_sha256": RESULT_SHA256,
        "memory": _memory(delta),
    }


def _cancellation(iteration: int) -> dict[str, Any]:
    return {
        **_identity(f"cancel-{iteration}"),
        "iteration": iteration,
        "wall_ns": 1,
        "cancel_to_quiescence_ns": 1,
        "session_version_before": 0,
        "session_version_after": 0,
        "memory": _memory(200 + iteration),
        "budget": _budget(),
    }


def _fault(point: str, repetition: int, index: int) -> dict[str, Any]:
    run = {
        **_identity(f"fault-{point}-{repetition}"),
        "fault_point": point,
        "repetition": repetition,
        "fault_origin": "harness-injected",
        "planned_injection_count": 2,
        "backend_call_relation": "after-backend-call",
        "blocked_request_id": f"blocked-{point}-{repetition}",
        "wall_ns": 1,
        "injection_to_quiescence_ns": 1,
        "recovery_ns": 1,
        "session_version_before": 0,
        "session_version_after": 0,
        "memory": _memory(300 + index),
        "budget": _budget(),
    }
    reuse = _success(f"reuse-{point}-{repetition}", index, 400 + index)
    reuse["session_id"] = run["session_id"]
    run["post_recovery_reuse"] = reuse
    return run


def _append_control_events(events: list[dict[str, Any]], run: dict[str, Any]) -> None:
    for event_name in CONTROL_EVENTS:
        events.append(
            {
                "sequence": len(events),
                "offset_ns": len(events) * 10,
                "worker_id": WORKER_ID,
                "run_id": run["run_id"],
                "run_kind": "control",
                "event": event_name,
            }
        )
    run_events = [event for event in events if event["run_id"] == run["run_id"]]
    by_name = {event["event"]: event for event in run_events}
    run["wall_ns"] = (
        by_name["backend-quiescent"]["offset_ns"] - by_name["run-start"]["offset_ns"]
    )


def _append_transaction_events(
    events: list[dict[str, Any]],
    run: dict[str, Any],
    run_kind: str,
    event_names: tuple[str, ...],
) -> None:
    fault_trigger_ordinal = 0
    for event_name in event_names:
        event = {
            "sequence": len(events),
            "offset_ns": len(events) * 10,
            "worker_id": WORKER_ID,
            "run_id": run["run_id"],
            "run_kind": run_kind,
            "session_id": run["session_id"],
            "request_id": run["request_id"],
            "transaction_id": run["transaction_id"],
            "lease_id": run["lease_id"],
            "event": event_name,
        }
        if event_name == "decoder-step-incomplete":
            event["decoder_step"] = 1
        elif event_name == "fault-armed":
            event["fault_point"] = run["fault_point"]
            event["operation_ordinal"] = 1
            event["planned_injection_count"] = run["planned_injection_count"]
        elif event_name == "fault-triggered":
            fault_trigger_ordinal += 1
            event["fault_point"] = run["fault_point"]
            event["operation_ordinal"] = fault_trigger_ordinal
            event["error_type"] = "RuntimeError"
            event["error_sha256"] = "b" * 64
            event["backend_call_relation"] = run["backend_call_relation"]
        elif event_name == "new-work-rejected":
            event["blocked_request_id"] = run["blocked_request_id"]
        events.append(event)
    by_name = {
        event["event"]: event for event in events if event["run_id"] == run["run_id"]
    }
    run["wall_ns"] = (
        by_name["budget-restored"]["offset_ns"] - by_name["run-start"]["offset_ns"]
    )
    if run_kind == "cancellation":
        run["cancel_to_quiescence_ns"] = (
            by_name["backend-quiescent"]["offset_ns"]
            - by_name["cancel-requested"]["offset_ns"]
        )
    elif run_kind == "fault":
        first_trigger = next(
            event
            for event in events
            if event["run_id"] == run["run_id"] and event["event"] == "fault-triggered"
        )
        run["injection_to_quiescence_ns"] = (
            by_name["backend-quiescent"]["offset_ns"] - first_trigger["offset_ns"]
        )
        run["recovery_ns"] = (
            by_name["backend-quiescent"]["offset_ns"]
            - by_name["recovery-started"]["offset_ns"]
        )


def valid_record() -> dict[str, Any]:
    decode_options = {
        "temperature": 0,
        "beam_size": None,
        "best_of": None,
        "patience": None,
        "length_penalty": None,
        "sample_len": None,
        "without_timestamps": False,
        "fp16": False,
        "condition_on_previous_text": True,
    }
    preprocessing = {
        "loader": "whisper.load_audio",
        "sample_rate_hz": 16_000,
        "pad_or_trim_samples": 480_000,
        "mel_bins": 80,
    }
    warmups = [_success(f"warmup-{index}", index, 100 + index) for index in range(2)]
    control_warmups = [
        _control(f"control-warmup-{index}", index, 100 + index) for index in range(2)
    ]
    measured = [_success(f"measured-{index}", index, 200 + index) for index in range(5)]
    controls = [_control(f"control-{index}", index, 200 + index) for index in range(5)]
    cancellations = [_cancellation(index) for index in range(3)]
    faults = [
        _fault(point, repetition, index * 2 + repetition)
        for index, point in enumerate(FAULT_POINTS)
        for repetition in range(2)
    ]
    events: list[dict[str, Any]] = []
    for control, runtime in zip(control_warmups, warmups, strict=True):
        _append_control_events(events, control)
        _append_transaction_events(events, runtime, "warmup", SUCCESS_EVENTS)
    for control, runtime in zip(controls, measured, strict=True):
        _append_control_events(events, control)
        _append_transaction_events(events, runtime, "measured", SUCCESS_EVENTS)
    for run in cancellations:
        _append_transaction_events(events, run, "cancellation", CANCELLATION_EVENTS)
    for run in faults:
        _append_transaction_events(events, run, "fault", FAULT_EVENTS)
        _append_transaction_events(
            events, run["post_recovery_reuse"], "reuse", SUCCESS_EVENTS
        )
    resolved_dependencies = [
        {"name": "modal", "version": "1.5.5"},
        {"name": "torch", "version": "2.8.0"},
    ]
    record = {
        "schema_version": "1-draft",
        "recorded_at": "2026-09-04T12:00:00Z",
        "status": "passed",
        "outcome": {
            "registered_cell_id": "t4-tiny-en-jfk-qualification-v1",
            "exclusion_rule_id": "no-exclusions-v1",
            "result": "passed",
            "failure_class": "none",
            "failure_summary": None,
        },
        "qualification_registration": {
            "manifest_id": "native-cuda-qualification-v1",
            "manifest_path": QUALIFICATION_MANIFEST_PATH,
            "manifest_sha256": QUALIFICATION_MANIFEST_SHA256,
            "runtime_commit": RUNTIME_COMMIT,
        },
        "scope": {
            "evidence_kind": "native-whisper-cuda-qualification",
            "evidence_tier": "qualification",
            "fault_matrix_scope": "completion-boundary-qualification-subset",
            "statement": "A fixed CUDA qualification cell with raw diagnostic observations.",
            "hardware_execution": True,
            "fault_injection_used": True,
            "performance_benchmark": False,
            "production_readiness": False,
        },
        "runtime": {
            "repository": RUNTIME_REPOSITORY,
            "git_commit": RUNTIME_COMMIT,
            "git_tree": RUNTIME_TREE,
            "clean": True,
        },
        "backend": {
            "repository": BACKEND_REPOSITORY,
            "git_commit": BACKEND_COMMIT,
            "git_tree": BACKEND_TREE,
            "clean": True,
            "patch_manifest_path": PATCH_MANIFEST_PATH,
            "patch_manifest_sha256": PATCH_MANIFEST_SHA256,
        },
        "producer": {
            "repository": RUNTIME_REPOSITORY,
            "git_commit": RUNTIME_COMMIT,
            "git_tree": RUNTIME_TREE,
            "clean": True,
            "script_path": PRODUCER_SCRIPT_PATH,
            "script_sha256": PRODUCER_SCRIPT_SHA256,
            "schema_path": SCHEMA_PATH,
            "schema_sha256": SCHEMA_SHA256,
            "validator_path": VALIDATOR_PATH,
            "validator_sha256": VALIDATOR_SHA256,
            "image_inputs_path": IMAGE_INPUTS_PATH,
            "image_inputs_sha256": IMAGE_INPUTS_SHA256,
            "resolved_dependencies": resolved_dependencies,
            "resolved_dependencies_sha256": canonical_sha256(resolved_dependencies),
            "container_image_id": "im-qualificationfixture",
        },
        "worker": {
            "campaign_id": "t4-tiny-en-jfk-v1",
            "worker_id": WORKER_ID,
            "worker_ordinal": 0,
            "expected_worker_count": 1,
            "single_use": True,
        },
        "environment": {
            "python": "3.13.7",
            "platform": "Linux-6.8-x86_64",
            "torch": "2.8.0",
            "cuda_runtime": "12.8",
            "cudnn": "91002",
            "driver": "570.00",
            "modal_sdk": "1.5.5",
        },
        "gpu": {
            "cloud_provider": "CLOUD_PROVIDER_AWS",
            "region": "us-west-2",
            "visible_device_count": 1,
            "device_index": 0,
            "name": "Tesla T4",
            "compute_capability": "7.5",
            "total_memory_bytes": 15_637_086_208,
        },
        "clock": {
            "source": "time.monotonic_ns",
            "unit": "nanosecond",
            "origin": "worker-qualification-start",
            "resolution_ns": 1,
        },
        "workload": {
            "profile_id": "tiny.en-cuda0-float32-v1",
            "model": "tiny.en",
            "checkpoint_source": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
            "checkpoint_sha256": "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
            "fixture_id": "openai-whisper-jfk-flac",
            "input_manifest_source": "https://github.com/billmedj/whisper-runtime/blob/3ea09422106615b12a01ffe118fea57c10ab1050/conformance/audio-manifest.json",
            "input_manifest_sha256": "4be9e19aa78bf09159183b136896569c6603ecb9decd7537ded14c911d8bfd06",
            "input_source": "https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/tests/jfk.flac",
            "input_sha256": "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715",
            "decoded_pcm_sha256": "59be237a0814faee9f1279bc54c8a482fa7edc09a640ff0fdd18d14d41519065",
            "preprocessing_options": preprocessing,
            "preprocessing_options_sha256": canonical_sha256(preprocessing),
            "input_bytes": 1_152_693,
            "input_duration_ns": 11_000_000_000,
            "sample_rate_hz": 16_000,
            "channel_count": 1,
            "language": "en",
            "task": "transcribe",
            "numeric_precision": "float32",
            "compatibility_rule": "exact-result-sha256",
            "result_digest_encoding": "utf8-transcript-v1",
            "expected_result_sha256": RESULT_SHA256,
            "decode_options": decode_options,
            "decode_options_sha256": canonical_sha256(decode_options),
            "seed": 7,
            "seed_reset_each_run": True,
            "device": "cuda:0",
            "execution_mode": "single-lane",
            "pair_order": "control-first",
            "comparison_backend_mode": "native-unproxied",
            "fault_backend_mode": "scoped-harness-injector",
            "network_access_during_measured_work": False,
            "model_loaded_before_timing": True,
            "peak_stats_reset_each_run": True,
            "runtime_measurement_start": "before-admission",
            "runtime_measurement_end": "after-budget-restored",
            "control_measurement_start": "before-backend-call",
            "control_measurement_end": "after-backend-quiescent",
            "warmup_iterations": len(warmups),
            "control_iterations": len(controls),
            "measured_iterations": len(measured),
            "cancellation_iterations": len(cancellations),
            "fault_repetitions_per_point": 2,
            "resource_capacity": _resource(2_147_483_648, 1, 1),
            "resource_reservation": _resource(2_147_483_648, 1, 1),
            "allocation_tolerance_bytes": 67_108_864,
            "reserved_tolerance_bytes": 67_108_864,
        },
        "warmup_runs": warmups,
        "control_warmup_runs": control_warmups,
        "control_runs": controls,
        "measured_runs": measured,
        "cancellation_runs": cancellations,
        "fault_runs": faults,
        "events": events,
        "summaries": {
            "quantile_method": "nearest-rank",
            "p99_minimum_sample_count": 1000,
            "warmups_excluded": True,
            "control_wall_ns": summarize([run["wall_ns"] for run in controls]),
            "success_wall_ns": summarize([run["wall_ns"] for run in measured]),
            "cancellation_to_quiescence_ns": summarize(
                [run["cancel_to_quiescence_ns"] for run in cancellations]
            ),
            "success_peak_allocated_delta_bytes": summarize(
                [run["memory"]["peak_allocated_delta_bytes"] for run in measured]
            ),
            "success_peak_reserved_delta_bytes": summarize(
                [run["memory"]["peak_reserved_delta_bytes"] for run in measured]
            ),
            "control_peak_allocated_delta_bytes": summarize(
                [run["memory"]["peak_allocated_delta_bytes"] for run in controls]
            ),
            "control_peak_reserved_delta_bytes": summarize(
                [run["memory"]["peak_reserved_delta_bytes"] for run in controls]
            ),
            "fault_recovery_ns": {
                point: summarize(
                    [
                        run["recovery_ns"]
                        for run in faults
                        if run["fault_point"] == point
                    ]
                )
                for point in FAULT_POINTS
            },
            "fault_injection_to_quiescence_ns": {
                point: summarize(
                    [
                        run["injection_to_quiescence_ns"]
                        for run in faults
                        if run["fault_point"] == point
                    ]
                )
                for point in FAULT_POINTS
            },
        },
        "derived_invariants": {
            "event_stream_globally_ordered": True,
            "control_and_runtime_results_stable": True,
            "success_published_once_after_fence": True,
            "cancellation_after_incomplete_step_never_published": True,
            "fault_exactly_injected_never_published": True,
            "fault_blocked_new_work": True,
            "post_recovery_reuse_committed": True,
            "leases_held_until_quiescence": True,
            "budgets_retained_until_quiescence": True,
            "budgets_restored_after_release": True,
            "device_and_allocator_samples_within_bounds": True,
        },
    }
    return record


def validation_kwargs() -> dict[str, Any]:
    return {
        "runtime_identity": RUNTIME_IDENTITY,
        "backend_identity": BACKEND_IDENTITY,
        "qualification_manifest": QUALIFICATION_MANIFEST,
        "expected_qualification_manifest_path": QUALIFICATION_MANIFEST_PATH,
        "expected_qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA256,
        "expected_patch_manifest_path": PATCH_MANIFEST_PATH,
        "expected_patch_manifest_sha256": PATCH_MANIFEST_SHA256,
        "expected_producer_script_path": PRODUCER_SCRIPT_PATH,
        "expected_producer_script_sha256": PRODUCER_SCRIPT_SHA256,
        "expected_schema_path": SCHEMA_PATH,
        "expected_schema_sha256": SCHEMA_SHA256,
        "expected_validator_path": VALIDATOR_PATH,
        "expected_validator_sha256": VALIDATOR_SHA256,
        "expected_image_inputs_path": IMAGE_INPUTS_PATH,
        "expected_image_inputs_sha256": IMAGE_INPUTS_SHA256,
    }


def _mark_failed(record: dict[str, Any], failure_class: str) -> None:
    record["status"] = "failed"
    record["outcome"]["result"] = "failed"
    record["outcome"]["failure_class"] = failure_class
    record["outcome"]["failure_summary"] = "The registered cell did not pass."


class NativeCudaQualificationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = read_json(DEFAULT_SCHEMA)

    def failures(self, record: dict[str, Any], **overrides: Any) -> list[str]:
        kwargs = validation_kwargs()
        kwargs.update(overrides)
        return validate_record(record, self.schema, **kwargs)

    def assert_rejected(self, record: dict[str, Any], fragment: str) -> None:
        failures = self.failures(record)
        self.assertTrue(any(fragment in failure for failure in failures), failures)

    def test_valid_qualification_record_passes(self) -> None:
        self.assertEqual(self.failures(valid_record()), [])

    def test_schema_is_closed_and_draft_boundary_is_fixed(self) -> None:
        record = valid_record()
        record["marketing_claim"] = "production ready"
        self.assert_rejected(record, "marketing_claim")
        record = valid_record()
        record["scope"]["production_readiness"] = True
        self.assert_rejected(record, "production_readiness")

    def test_all_source_identities_and_artifacts_are_caller_bound(self) -> None:
        mutations = (
            (
                "runtime",
                "repository",
                "https://example.invalid/runtime",
                "runtime.repository",
            ),
            ("runtime", "git_commit", "f" * 40, "runtime.git_commit"),
            ("runtime", "git_tree", "f" * 40, "runtime.git_tree"),
            (
                "backend",
                "repository",
                "https://example.invalid/backend",
                "backend.repository",
            ),
            ("backend", "git_commit", "f" * 40, "backend.git_commit"),
            ("backend", "git_tree", "f" * 40, "backend.git_tree"),
            ("backend", "patch_manifest_sha256", "f" * 64, "patch_manifest_sha256"),
            ("producer", "script_sha256", "f" * 64, "script_sha256"),
            ("producer", "schema_sha256", "f" * 64, "schema_sha256"),
            ("producer", "validator_sha256", "f" * 64, "validator_sha256"),
            (
                "producer",
                "image_inputs_sha256",
                "f" * 64,
                "image_inputs_sha256",
            ),
        )
        for section, field, value, fragment in mutations:
            with self.subTest(field=f"{section}.{field}"):
                record = valid_record()
                record[section][field] = value
                self.assert_rejected(record, fragment)

    def test_resolved_environment_inventory_is_canonical_and_bound(self) -> None:
        record = valid_record()
        record["producer"]["resolved_dependencies"].reverse()
        record["producer"]["resolved_dependencies_sha256"] = canonical_sha256(
            record["producer"]["resolved_dependencies"]
        )
        self.assert_rejected(record, "resolved_dependencies must be sorted")

        record = valid_record()
        record["producer"]["resolved_dependencies"][1]["version"] = "2.8.1"
        self.assert_rejected(record, "resolved_dependencies_sha256 is not canonical")

        record = valid_record()
        record["producer"]["resolved_dependencies"][1]["version"] = "2.8.1"
        record["producer"]["resolved_dependencies_sha256"] = canonical_sha256(
            record["producer"]["resolved_dependencies"]
        )
        self.assert_rejected(record, "torch version does not match environment")

    def test_arbitrary_git_hashes_cannot_replace_checkout_identity(self) -> None:
        record = valid_record()
        record["runtime"]["git_commit"] = "f" * 40
        record["qualification_registration"]["runtime_commit"] = "f" * 40
        failures = self.failures(record)
        self.assertTrue(
            any("runtime.git_commit" in failure for failure in failures), failures
        )
        self.assertTrue(
            any(
                "qualification_registration.runtime_commit" in failure
                for failure in failures
            ),
            failures,
        )

    def test_cli_does_not_accept_expected_git_identity_strings(self) -> None:
        arguments = [
            "validate_modal_native_cuda_qualification.py",
            "record.json",
            "--runtime-checkout",
            str(ROOT),
            "--backend-checkout",
            str(ROOT),
            "--qualification-manifest",
            str(DEFAULT_QUALIFICATION_MANIFEST),
            "--patch-manifest",
            str(DEFAULT_QUALIFICATION_MANIFEST),
            "--producer-script",
            str(DEFAULT_QUALIFICATION_MANIFEST),
            "--image-inputs",
            str(DEFAULT_QUALIFICATION_MANIFEST),
            "--expected-runtime-commit",
            "f" * 40,
        ]
        stderr = StringIO()
        with patch("sys.argv", arguments), patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit):
                parse_args()
        self.assertIn(
            "unrecognized arguments: --expected-runtime-commit", stderr.getvalue()
        )

    def test_dirty_runtime_and_backend_checkouts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            runtime = _make_git_checkout(parent, "runtime")
            backend = _make_git_checkout(parent, "backend")
            self.assertEqual(
                derive_checkout_identity(runtime).repository,
                "https://github.com/example/fixture",
            )
            self.assertEqual(
                derive_checkout_identity(backend).repository,
                "https://github.com/example/fixture",
            )
            (runtime / "tracked.txt").write_text("dirty\n", encoding="utf-8")
            (backend / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dirty or contains untracked"):
                derive_checkout_identity(runtime)
            with self.assertRaisesRegex(ValueError, "dirty or contains untracked"):
                derive_checkout_identity(backend)

    def test_registration_manifest_must_be_tracked_and_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            tracked_repo = _make_git_checkout(parent, "tracked")
            manifest = tracked_repo / "experiments" / "qualification.json"
            manifest.parent.mkdir()
            manifest.write_text('{"manifest_version":"1"}\n', encoding="utf-8")
            _run_git(tracked_repo, "add", "experiments/qualification.json")
            _run_git(tracked_repo, "commit", "--quiet", "-m", "registration")
            tracked_identity = derive_checkout_identity(tracked_repo)
            self.assertEqual(
                bind_tracked_artifact(manifest, tracked_identity)[0],
                "experiments/qualification.json",
            )
            manifest.write_text('{"manifest_version":"2"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from checkout HEAD"):
                bind_tracked_artifact(manifest, tracked_identity)

            untracked_repo = _make_git_checkout(parent, "untracked")
            untracked_identity = derive_checkout_identity(untracked_repo)
            untracked_manifest = untracked_repo / "qualification.json"
            untracked_manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git command failed"):
                bind_tracked_artifact(untracked_manifest, untracked_identity)

    def test_registration_fields_and_manifest_cell_are_bound(self) -> None:
        mutations = (
            ("manifest_id", "other-manifest", "manifest_id"),
            ("manifest_path", "other/manifest.json", "manifest_path"),
            ("manifest_sha256", "f" * 64, "manifest_sha256"),
            ("runtime_commit", "f" * 40, "runtime_commit"),
        )
        for field, value, fragment in mutations:
            with self.subTest(field=field):
                record = valid_record()
                record["qualification_registration"][field] = value
                self.assert_rejected(record, fragment)

        altered_manifest = copy.deepcopy(QUALIFICATION_MANIFEST)
        altered_manifest["cell"]["model"] = "base.en"
        failures = self.failures(
            valid_record(), qualification_manifest=altered_manifest
        )
        self.assertTrue(
            any(
                "record model does not match manifest" in failure
                for failure in failures
            ),
            failures,
        )

    def test_all_artifact_paths_are_caller_bound(self) -> None:
        mutations = (
            ("backend", "patch_manifest_path"),
            ("producer", "script_path"),
            ("producer", "schema_path"),
            ("producer", "validator_path"),
            ("producer", "image_inputs_path"),
        )
        for section, field in mutations:
            with self.subTest(field=f"{section}.{field}"):
                record = valid_record()
                record[section][field] = f"alternate/{field}"
                self.assert_rejected(record, "does not match the caller-bound path")

    def test_cli_artifact_paths_must_resolve_inside_runtime_repository(self) -> None:
        self.assertEqual(
            _repository_relative_path(Path(__file__)),
            "tools/test_modal_native_cuda_qualification.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "artifact.txt"
            outside.write_text("same digest is not sufficient", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the runtime repository"):
                _repository_relative_path(outside, ROOT)

    def test_same_digest_at_another_cli_path_is_rejected(self) -> None:
        cases = (
            (
                "backend",
                "patch_manifest_path",
                "patch_manifest_sha256",
                "expected_patch_manifest_path",
                "expected_patch_manifest_sha256",
            ),
            (
                "producer",
                "script_path",
                "script_sha256",
                "expected_producer_script_path",
                "expected_producer_script_sha256",
            ),
            (
                "producer",
                "schema_path",
                "schema_sha256",
                "expected_schema_path",
                "expected_schema_sha256",
            ),
            (
                "producer",
                "image_inputs_path",
                "image_inputs_sha256",
                "expected_image_inputs_path",
                "expected_image_inputs_sha256",
            ),
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            original = Path(directory) / "registered-artifact"
            alternate = Path(directory) / "caller-artifact"
            original.write_bytes(b"identical artifact bytes")
            alternate.write_bytes(original.read_bytes())
            digest = sha256_file(original)
            self.assertEqual(digest, sha256_file(alternate))
            original_path = _repository_relative_path(original)
            alternate_path = _repository_relative_path(alternate)
            for section, path_field, hash_field, path_kwarg, hash_kwarg in cases:
                with self.subTest(field=f"{section}.{path_field}"):
                    record = valid_record()
                    record[section][path_field] = original_path
                    record[section][hash_field] = digest
                    kwargs = validation_kwargs()
                    kwargs[path_kwarg] = alternate_path
                    kwargs[hash_kwarg] = digest
                    failures = validate_record(record, self.schema, **kwargs)
                    self.assertTrue(
                        any(
                            f"{section}.{path_field} does not match the caller-bound path"
                            in failure
                            for failure in failures
                        ),
                        failures,
                    )

    def test_producer_must_belong_to_runtime_tree(self) -> None:
        record = valid_record()
        record["producer"]["git_tree"] = "f" * 40
        self.assert_rejected(record, "part of the bound runtime tree")

    def test_repository_uris_and_paths_are_publishable(self) -> None:
        record = valid_record()
        record["runtime"]["repository"] = "https://user:pass@example.com/runtime"
        self.assert_rejected(record, "HTTPS URI")
        record = valid_record()
        record["producer"]["script_path"] = "../private/producer.py"
        self.assert_rejected(record, "script_path")

    def test_global_event_sequence_order_and_worker_are_derived(self) -> None:
        record = valid_record()
        record["events"][1]["sequence"] = 0
        self.assert_rejected(record, "globally ordered")
        record = valid_record()
        record["events"][1]["offset_ns"] = record["events"][0]["offset_ns"]
        self.assert_rejected(record, "strictly increasing")
        record = valid_record()
        record["events"][0]["worker_id"] = "f" * 64
        self.assert_rejected(record, "single worker")

    def test_run_identity_and_wall_time_are_derived(self) -> None:
        record = valid_record()
        measured_id = record["measured_runs"][0]["run_id"]
        event = next(item for item in record["events"] if item["run_id"] == measured_id)
        event["request_id"] = "wrong-request"
        self.assert_rejected(record, "request_id does not match")
        record = valid_record()
        record["measured_runs"][0]["wall_ns"] += 1
        self.assert_rejected(record, "wall_ns is not derived")

    def test_control_and_runtime_order_alternates_by_worker(self) -> None:
        record = valid_record()
        record["workload"]["pair_order"] = "runtime-first"
        self.assert_rejected(record, "alternate by worker ordinal")

    def test_control_and_runtime_measurement_boundaries_are_fixed(self) -> None:
        fields = (
            "runtime_measurement_start",
            "runtime_measurement_end",
            "control_measurement_start",
            "control_measurement_end",
        )
        for field in fields:
            with self.subTest(field=field):
                record = valid_record()
                record["workload"][field] = "unspecified-boundary"
                self.assert_rejected(record, field)

    def test_cancellation_binds_incomplete_decoder_work(self) -> None:
        record = valid_record()
        run = record["cancellation_runs"][0]
        run["session_version_after"] = 1
        record["derived_invariants"][
            "cancellation_after_incomplete_step_never_published"
        ] = False
        _mark_failed(record, "derived-invariant")
        self.assertEqual(self.failures(record), [])

    def test_fault_trigger_and_blocked_attempt_are_derived(self) -> None:
        record = valid_record()
        run = record["fault_runs"][0]
        event = next(
            item
            for item in record["events"]
            if item["run_id"] == run["run_id"] and item["event"] == "fault-triggered"
        )
        event["backend_call_relation"] = "before-backend-call"
        self.assert_rejected(record, "fault_exactly_injected_never_published")

        record = valid_record()
        run = record["fault_runs"][0]
        triggers = [
            item
            for item in record["events"]
            if item["run_id"] == run["run_id"] and item["event"] == "fault-triggered"
        ]
        triggers[1]["operation_ordinal"] = 3
        self.assert_rejected(record, "fault_exactly_injected_never_published")

        record = valid_record()
        run = record["fault_runs"][0]
        triggers = [
            item
            for item in record["events"]
            if item["run_id"] == run["run_id"] and item["event"] == "fault-triggered"
        ]
        triggers[1]["error_sha256"] = "c" * 64
        self.assert_rejected(record, "fault_exactly_injected_never_published")

        record = valid_record()
        record["fault_runs"][0]["planned_injection_count"] = 1
        self.assert_rejected(record, "planned_injection_count")

        record = valid_record()
        run = record["fault_runs"][0]
        armed = next(
            item
            for item in record["events"]
            if item["run_id"] == run["run_id"] and item["event"] == "fault-armed"
        )
        armed["planned_injection_count"] = 1
        self.assert_rejected(record, "planned_injection_count")

        record = valid_record()
        run = record["fault_runs"][0]
        triggers = [
            item
            for item in record["events"]
            if item["run_id"] == run["run_id"] and item["event"] == "fault-triggered"
        ]
        triggers[1]["event"] = "fault-armed"
        self.assert_rejected(record, "exactly 2 fault-triggered")

        record = valid_record()
        run = record["fault_runs"][0]
        event = next(
            item
            for item in record["events"]
            if item["run_id"] == run["run_id"] and item["event"] == "new-work-rejected"
        )
        event["blocked_request_id"] = "different-blocked-request"
        self.assert_rejected(record, "fault_blocked_new_work")

    def test_post_recovery_reuse_is_linked(self) -> None:
        record = valid_record()
        record["fault_runs"][0]["post_recovery_reuse"]["session_id"] = "other-session"
        self.assert_rejected(record, "not linked to recovery")

    def test_budget_states_are_derived_from_capacity_and_reservation(self) -> None:
        record = valid_record()
        record["fault_runs"][0]["budget"]["available_at_quiescence"][
            "stream_slots"
        ] += 1
        self.assert_rejected(record, "budgets_retained_until_quiescence")

    def test_device_capacity_and_allocator_relations_are_derived(self) -> None:
        record = valid_record()
        record["workload"]["device"] = "cuda:1"
        self.assert_rejected(record, "device_index")
        record = valid_record()
        record["gpu"]["total_memory_bytes"] = 9_999
        self.assert_rejected(record, "capacity exceeds physical")
        record = valid_record()
        record["measured_runs"][0]["memory"]["peak_allocated_delta_bytes"] += 1
        self.assert_rejected(record, "delta is not derived")

    def test_canonical_workload_hashes_are_derived(self) -> None:
        record = valid_record()
        record["workload"]["decode_options"]["temperature"] = 0.5
        self.assert_rejected(record, "decode_options_sha256")
        record = valid_record()
        record["workload"]["preprocessing_options"]["mel_bins"] = 128
        self.assert_rejected(record, "preprocessing_options_sha256")

    def test_each_summary_family_is_recomputed(self) -> None:
        fields = (
            "control_wall_ns",
            "success_wall_ns",
            "cancellation_to_quiescence_ns",
            "success_peak_allocated_delta_bytes",
            "success_peak_reserved_delta_bytes",
            "control_peak_allocated_delta_bytes",
            "control_peak_reserved_delta_bytes",
        )
        for field in fields:
            with self.subTest(field=field):
                record = valid_record()
                record["summaries"][field]["p50"] += 1
                self.assert_rejected(record, f"summaries.{field}")
        for family in ("fault_recovery_ns", "fault_injection_to_quiescence_ns"):
            record = valid_record()
            record["summaries"][family]["event-record"]["max"] += 1
            self.assert_rejected(record, f"summaries.{family}")

    def test_percentile_policy_matches_experiment_protocol(self) -> None:
        self.assertEqual(summarize(list(range(5)))["p99"], "not_estimated")
        summary = summarize(list(range(1000)))
        self.assertEqual(summary["p99"], 989)
        record = valid_record()
        record["summaries"]["success_wall_ns"]["p99"] = 70
        self.assert_rejected(record, "success_wall_ns")

    def test_schema_rejects_performance_records(self) -> None:
        record = valid_record()
        record["scope"]["evidence_tier"] = "benchmark"
        self.assert_rejected(record, "evidence_tier")
        record = valid_record()
        record["scope"]["performance_benchmark"] = True
        self.assert_rejected(record, "performance_benchmark")

    def test_failed_qualification_gate_is_publishable(self) -> None:
        record = valid_record()
        _mark_failed(record, "qualification-gate")
        self.assertEqual(self.failures(record), [])

    def test_failed_output_compatibility_cell_is_publishable(self) -> None:
        record = valid_record()
        record["control_runs"][0]["result_sha256"] = "f" * 64
        record["derived_invariants"]["control_and_runtime_results_stable"] = False
        _mark_failed(record, "derived-invariant")
        self.assertEqual(self.failures(record), [])

    def test_status_and_failure_metadata_are_consistent(self) -> None:
        record = valid_record()
        record["outcome"]["result"] = "failed"
        self.assert_rejected(record, "outcome.result")
        record = valid_record()
        record["outcome"]["failure_summary"] = "unexpected"
        self.assert_rejected(record, "passing record")

    def test_known_secrets_and_user_paths_are_rejected(self) -> None:
        slash = "/"
        for value in (
            "hf_" + "a" * 40,
            "sk-proj-" + "a" * 40,
            slash.join(("", "root", ".cache", "private-model")),
            "file://" + slash.join(("", "home", "operator", "private")),
            "%2Fhome%2Foperator%2Fprivate",
        ):
            with self.subTest(value=value[:12]):
                record = valid_record()
                record["environment"]["platform"] = value
                failures = self.failures(record)
                self.assertTrue(
                    any(
                        fragment in failure
                        for fragment in ("secret pattern", "absolute user path")
                        for failure in failures
                    ),
                    failures,
                )

    def test_non_finite_numbers_and_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text('{"wall_ns": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                read_json(path)
            path.write_text(
                '{"status": "passed", "status": "failed"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                read_json(path)


if __name__ == "__main__":
    unittest.main()
