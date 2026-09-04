"""Validate a native CUDA transaction evidence record and its claim limits."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "evidence/modal-native-cuda-transaction.schema.json"

EXPECTED_ADAPTER_SHA256 = (
    "1e8aef1728d9f8d16af9ac54810696faece47726de84b29a476acf951c493e8d"
)
EXPECTED_SOURCE_PATHS = (
    "src/whisper_runtime/adapters/native_whisper.py",
    "src/whisper_runtime/adapters/_model_binding.py",
    "src/whisper_runtime/execution.py",
    "src/whisper_runtime/resources.py",
    "src/whisper_runtime/state.py",
    "src/whisper_runtime/transaction.py",
    "src/whisper_runtime/worker.py",
)
EXPECTED_CLAIMS = {
    "native_cuda_adapter_exercised": True,
    "worker_admission_exercised": True,
    "transaction_lifecycle_exercised": True,
    "cuda_completion_fence_exercised": True,
    "cooperative_cancellation_exercised": True,
    "quarantine_recovery_exercised": True,
    "unproxied_native_reuse_exercised": True,
    "physical_gpu_memory_enforced": False,
    "performance_benchmark": False,
    "production_readiness": False,
}
EXPECTED_ASSERTIONS = frozenset(
    {
        "runtime_source_pinned",
        "backend_source_pinned",
        "checkpoint_verified_before_load",
        "input_fixture_verified",
        "network_probe_denied",
        "model_cache_read_only",
        "native_adapter_committed",
        "success_exact_result_and_terminal_states",
        "publication_followed_cuda_fence",
        "success_fence_retained_transaction",
        "success_used_one_private_stream",
        "success_fence_and_cleanup_counts",
        "cancellation_prevented_publication",
        "cancellation_exact_terminal_state",
        "cancellation_held_capacity_at_request",
        "cancellation_fence_retained_transaction",
        "cancellation_fence_preceded_release",
        "cancelling_thread_made_no_cuda_call",
        "failed_fence_retained_capacity",
        "recovery_retained_exact_state",
        "recovery_blocked_new_work",
        "recovery_released_capacity",
        "recovery_fence_retained_transaction",
        "recovery_fence_preceded_release",
        "post_recovery_reuse_committed",
        "unproxied_native_reuse_committed",
        "native_components_restored_for_unproxied_reuse",
        "run_child_generators_verified",
        "cancellation_rendezvous_proved",
        "model_profile_verified",
        "persistent_model_state_unchanged",
        "model_hooks_unchanged",
        "observed_peak_within_declared_memory",
    }
)
FULL_RESOURCES = {
    "memory_bytes": 1_000_000_000,
    "compute_units": 1,
    "stream_slots": 1,
}
ZERO_RESOURCES = {"memory_bytes": 0, "compute_units": 0, "stream_slots": 0}
SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api_key|credential|identity_token|password|secret|token)(?:$|_)",
    re.IGNORECASE,
)
SENSITIVE_VALUE = re.compile(
    r"(?:github_pat_|gh[opsu]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"MODAL_TOKEN_(?:ID|SECRET))"
)
ABSOLUTE_USER_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/(?:home|Users)/[^/\s]+/)",
    re.IGNORECASE,
)
EVENT_RETURN = re.compile(r"^cuda:event-([1-9][0-9]*):synchronize:return$")
EVENT_NAME = re.compile(r"^cuda:event-([1-9][0-9]*):")


def _reject_nonstandard_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key is not permitted: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _walk(value: Any, location: str = "record") -> list[tuple[str, Any]]:
    result = [(location, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(_walk(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_walk(child, f"{location}[{index}]"))
    return result


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _event_names(trace: list[Any]) -> list[str]:
    return [
        str(event.get("name"))
        for event in trace
        if isinstance(event, dict) and isinstance(event.get("name"), str)
    ]


def _find_index(names: list[str], name: str, start: int = 0) -> int | None:
    try:
        return names.index(name, start)
    except ValueError:
        return None


def _require_ordered(
    failures: list[str], trace: list[Any], label: str, expected: tuple[str, ...]
) -> None:
    names = _event_names(trace)
    cursor = 0
    for name in expected:
        index = _find_index(names, name, cursor)
        if index is None:
            failures.append(f"{label} lacks ordered event {name!r}")
            return
        cursor = index + 1


def _require_exact_event_counts(
    failures: list[str],
    trace: list[Any],
    label: str,
    expected: dict[str, int],
) -> None:
    names = _event_names(trace)
    for name, count in expected.items():
        if names.count(name) != count:
            failures.append(f"{label} must contain exactly {count} {name!r}")


def _require_event_context(
    failures: list[str],
    trace: list[Any],
    label: str,
    name: str,
    *,
    kind: str,
    thread: str,
    stream: str | None,
) -> None:
    matches = [
        _mapping(event) for event in trace if _mapping(event).get("name") == name
    ]
    if len(matches) != 1:
        failures.append(f"{label} must contain exactly one {name!r}")
    elif (
        matches[0].get("kind") != kind
        or matches[0].get("thread") != thread
        or matches[0].get("stream") != stream
    ):
        failures.append(
            f"{label} event {name!r} must be {kind}/{thread}/{stream or 'no-stream'}"
        )


def _validate_trace(
    failures: list[str],
    trace: list[Any],
    label: str,
    *,
    expected_event_returns: dict[str, str],
    allow_session_commit: bool,
) -> None:
    previous_offset = -1
    observed_streams: set[str] = set()
    for index, raw_event in enumerate(trace):
        if not isinstance(raw_event, dict):
            continue
        sequence = raw_event.get("sequence")
        offset = raw_event.get("offset_ns")
        name = raw_event.get("name")
        kind = raw_event.get("kind")
        thread = raw_event.get("thread")
        stream = raw_event.get("stream")
        state = raw_event.get("state")
        location = f"{label}[{index}]"
        if sequence != index + 1:
            failures.append(f"{location}.sequence is not contiguous")
        if isinstance(offset, int) and not isinstance(offset, bool):
            if offset < previous_offset:
                failures.append(f"{location}.offset_ns moves backwards")
            previous_offset = offset
        if isinstance(stream, str):
            observed_streams.add(stream)
        if isinstance(name, str) and name.startswith(("task:", "run:", "model:")):
            if kind != "backend" or thread != "decode" or stream != "stream-1":
                failures.append(
                    f"{location} backend event must be backend/decode/stream-1"
                )
        if kind == "backend" and stream != "stream-1":
            failures.append(f"{location} backend work did not use stream-1")
        if (
            isinstance(name, str)
            and (
                name == "cuda:stream:create"
                or name.startswith("cuda:stream:")
                or (EVENT_NAME.match(name) is not None and not name.endswith(":create"))
            )
            and stream != "stream-1"
        ):
            failures.append(f"{location} did not identify the private stream")
        if isinstance(name, str) and name.startswith("cuda:") and kind != "cuda":
            failures.append(f"{location} CUDA event has the wrong kind")
        if kind == "cuda" and not (isinstance(name, str) and name.startswith("cuda:")):
            failures.append(f"{location} cuda kind has a non-CUDA event name")
        if thread == "controller" and kind == "cuda":
            failures.append(f"{location} records a CUDA call on the controller thread")
        is_return = isinstance(name, str) and EVENT_RETURN.fullmatch(name) is not None
        if state is not None and not is_return:
            failures.append(f"{location}.state is only valid on a fence return")
        if is_return:
            expected_status = expected_event_returns.get(str(name))
            if expected_status is None:
                failures.append(f"{location} has an unexpected fence return")
            elif state != {
                "request_status": expected_status,
                "session_version": 0,
                "queue_depth": 1,
                "lease_count": 1,
                "budget_available": ZERO_RESOURCES,
            }:
                failures.append(
                    f"{location}.state does not bind the pre-publication fence"
                )
    names = _event_names(trace)
    if observed_streams != {"stream-1"}:
        failures.append(f"{label} did not use exactly one private stream")
    for name in expected_event_returns:
        if names.count(name) != 1:
            failures.append(f"{label} must contain exactly one {name!r}")
    commits = [name for name in names if name.startswith("session:commit:")]
    if allow_session_commit:
        if commits != ["session:commit:begin", "session:commit:return"]:
            failures.append(f"{label} must publish exactly one session commit")
    elif commits:
        failures.append(f"{label} published a session commit")


def _validate_success(failures: list[str], value: Any, label: str = "success") -> None:
    success = _mapping(value)
    trace = _list(success.get("trace"))
    _validate_trace(
        failures,
        trace,
        f"{label}.trace",
        expected_event_returns={"cuda:event-1:synchronize:return": "running"},
        allow_session_commit=True,
    )
    names = _event_names(trace)
    steps = [name for name in names if name == "run:step:submitted"]
    if not steps:
        failures.append(f"{label}.trace did not submit a decode step")
    _require_ordered(
        failures,
        trace,
        f"{label}.trace",
        (
            "budget:lease:acquired",
            "worker:admitted",
            "cuda:device-synchronize:begin",
            "cuda:device-synchronize:return",
            "cuda:stream:create",
            "task:construct:begin",
            "task:construct:return",
            "run:start:begin",
            "run:start:submitted",
            "run:child-generator:verified",
            "run:prefill:begin",
            "run:prefill:submitted",
            "run:step:submitted",
            "run:finalize:begin",
            "run:finalize:submitted",
            "model:identity:verified",
            "run:cleanup:begin",
            "run:cleanup:submitted",
            "cuda:event-1:create",
            "cuda:event-1:record",
            "cuda:event-1:synchronize:begin",
            "cuda:event-1:query:return",
            "cuda:event-1:synchronize:return",
            "session:commit:begin",
            "session:commit:return",
            "budget:lease:release:begin",
            "budget:lease:release:return",
        ),
    )
    _validate_child_generators(failures, success, trace, label, instrumented=True)
    _require_exact_event_counts(
        failures,
        trace,
        f"{label}.trace",
        {
            "cuda:stream:create": 1,
            "run:cleanup:begin": 1,
            "run:cleanup:submitted": 1,
            "cuda:event-1:create": 1,
            "cuda:event-1:record": 1,
            "cuda:event-1:synchronize:begin": 1,
            "cuda:event-1:query:return": 1,
            "cuda:event-1:synchronize:return": 1,
        },
    )


def _validate_child_generators(
    failures: list[str],
    value: dict[str, Any],
    trace: list[Any],
    label: str,
    *,
    instrumented: bool,
) -> None:
    expected_checks = 1 if instrumented else 0
    if value.get("child_generator_checks") != expected_checks:
        failures.append(f"{label}.child_generator_checks must be {expected_checks}")
    if value.get("child_generators_on_profile_device") is not True:
        failures.append(f"{label} did not keep child generators on the profile device")
    if value.get("child_generators_distinct") is not True:
        failures.append(f"{label} did not use distinct child generators")
    trace_checks = _event_names(trace).count("run:child-generator:verified")
    if trace_checks != expected_checks:
        failures.append(
            f"{label}.trace must contain exactly {expected_checks} "
            "run:child-generator:verified events"
        )
    if instrumented:
        _require_event_context(
            failures,
            trace,
            f"{label}.trace",
            "run:child-generator:verified",
            kind="backend",
            thread="decode",
            stream="stream-1",
        )


def _validate_cancellation(failures: list[str], value: Any) -> None:
    cancellation = _mapping(value)
    trace = _list(cancellation.get("trace"))
    _validate_trace(
        failures,
        trace,
        "cancellation.trace",
        expected_event_returns={"cuda:event-1:synchronize:return": "cancelled"},
        allow_session_commit=False,
    )
    _require_ordered(
        failures,
        trace,
        "cancellation.trace",
        (
            "budget:lease:acquired",
            "worker:admitted",
            "run:start:submitted",
            "run:child-generator:verified",
            "run:step:submitted",
            "run:cancellation-rendezvous:incomplete",
            "controller:cancel:begin",
            "controller:cancel:return",
            "run:cleanup:begin",
            "run:cleanup:submitted",
            "cuda:event-1:record",
            "cuda:event-1:synchronize:begin",
            "cuda:event-1:query:return",
            "cuda:event-1:synchronize:return",
            "budget:lease:release:begin",
            "budget:lease:release:return",
        ),
    )
    _validate_child_generators(
        failures, cancellation, trace, "cancellation", instrumented=True
    )
    if cancellation.get("first_step_returned") is not False:
        failures.append("cancellation.first_step_returned must be false")
    if cancellation.get("run_complete_after_first_step") is not False:
        failures.append("cancellation.run_complete_after_first_step must be false")
    _require_event_context(
        failures,
        trace,
        "cancellation.trace",
        "run:cancellation-rendezvous:incomplete",
        kind="backend",
        thread="decode",
        stream="stream-1",
    )
    for event_name in ("controller:cancel:begin", "controller:cancel:return"):
        _require_event_context(
            failures,
            trace,
            "cancellation.trace",
            event_name,
            kind="controller",
            thread="controller",
            stream=None,
        )
    _require_exact_event_counts(
        failures,
        trace,
        "cancellation.trace",
        {
            "cuda:stream:create": 1,
            "run:cleanup:begin": 1,
            "run:cleanup:submitted": 1,
            "cuda:event-1:create": 1,
            "cuda:event-1:record": 1,
            "cuda:event-1:synchronize:begin": 1,
            "cuda:event-1:query:return": 1,
            "cuda:event-1:synchronize:return": 1,
        },
    )


def _validate_recovery(failures: list[str], value: Any) -> None:
    recovery = _mapping(value)
    trace = _list(recovery.get("trace"))
    _validate_trace(
        failures,
        trace,
        "recovery.trace",
        expected_event_returns={"cuda:event-3:synchronize:return": "running"},
        allow_session_commit=False,
    )
    names = _event_names(trace)
    injected = [
        name
        for name in names
        if re.fullmatch(r"cuda:event-[12]:synchronize:injected-failure", name)
    ]
    if injected != [
        "cuda:event-1:synchronize:injected-failure",
        "cuda:event-2:synchronize:injected-failure",
    ]:
        failures.append("recovery.trace must contain the two ordered injected failures")
    if names.count("run:cleanup:submitted") != 3:
        failures.append("recovery.trace must contain exactly three cleanup attempts")
    _require_ordered(
        failures,
        trace,
        "recovery.trace",
        (
            "budget:lease:acquired",
            "worker:admitted",
            "run:start:submitted",
            "run:child-generator:verified",
            "run:finalize:submitted",
            "run:cleanup:submitted",
            "cuda:event-1:record",
            "cuda:event-1:synchronize:begin",
            "cuda:event-1:synchronize:injected-failure",
            "run:cleanup:submitted",
            "cuda:event-2:record",
            "cuda:event-2:synchronize:begin",
            "cuda:event-2:synchronize:injected-failure",
            "runtime:manual-recovery:begin",
            "run:cleanup:submitted",
            "cuda:event-3:record",
            "cuda:event-3:synchronize:begin",
            "cuda:event-3:query:return",
            "cuda:event-3:synchronize:return",
            "budget:lease:release:begin",
            "budget:lease:release:return",
            "runtime:manual-recovery:return",
        ),
    )
    _validate_child_generators(failures, recovery, trace, "recovery", instrumented=True)
    for event_name in (
        "runtime:manual-recovery:begin",
        "runtime:manual-recovery:return",
    ):
        matches = [
            _mapping(event)
            for event in trace
            if _mapping(event).get("name") == event_name
        ]
        if len(matches) != 1:
            failures.append(f"recovery.trace must contain exactly one {event_name!r}")
        elif (
            matches[0].get("kind") != "runtime"
            or matches[0].get("thread") != "decode"
            or matches[0].get("stream") is not None
        ):
            failures.append(
                f"{event_name} must be a runtime event on the decode thread"
            )
    recovery_counts = {
        "cuda:stream:create": 1,
        "run:cleanup:begin": 3,
        "run:cleanup:submitted": 3,
    }
    for event_index in (1, 2, 3):
        prefix = f"cuda:event-{event_index}"
        recovery_counts[f"{prefix}:create"] = 1
        recovery_counts[f"{prefix}:record"] = 1
        recovery_counts[f"{prefix}:synchronize:begin"] = 1
        recovery_counts[f"{prefix}:query:return"] = int(event_index == 3)
        recovery_counts[f"{prefix}:synchronize:return"] = int(event_index == 3)
    _require_exact_event_counts(failures, trace, "recovery.trace", recovery_counts)
    _validate_success(
        failures, recovery.get("post_recovery_reuse"), "recovery.post_recovery_reuse"
    )


def _validate_unproxied_success(failures: list[str], value: Any) -> None:
    reuse = _mapping(value)
    trace = _list(reuse.get("trace"))
    names = _event_names(trace)
    previous_offset = -1
    for index, raw_event in enumerate(trace):
        event = _mapping(raw_event)
        location = f"unproxied_reuse.trace[{index}]"
        if event.get("sequence") != index + 1:
            failures.append(f"{location}.sequence is not contiguous")
        offset = event.get("offset_ns")
        if isinstance(offset, int) and not isinstance(offset, bool):
            if offset < previous_offset:
                failures.append(f"{location}.offset_ns moves backwards")
            previous_offset = offset
        if event.get("kind") != "runtime":
            failures.append(f"{location} is not a runtime-only observer event")
        if event.get("stream") is not None:
            failures.append(f"{location} exposes a proxy stream in the unproxied run")
        if "state" in event:
            failures.append(
                f"{location} exposes proxy fence state in the unproxied run"
            )
    if any(name.startswith(("cuda:", "task:", "run:", "model:")) for name in names):
        failures.append("unproxied_reuse.trace contains backend or CUDA proxy events")
    _validate_child_generators(
        failures, reuse, trace, "unproxied_reuse", instrumented=False
    )
    if names != [
        "budget:lease:acquired",
        "worker:admitted",
        "session:commit:begin",
        "session:commit:return",
        "budget:lease:release:begin",
        "budget:lease:release:return",
    ]:
        failures.append(
            "unproxied_reuse.trace does not match the runtime-only control run"
        )


def validate_semantics(
    record: Any,
    *,
    expected_runtime_commit: str | None = None,
    expected_runtime_tree: str | None = None,
) -> list[str]:
    """Return failures that need cross-field or sequence checks."""

    failures: list[str] = []
    if not isinstance(record, dict):
        return ["record must be an object"]

    runtime = _mapping(record.get("runtime"))
    if expected_runtime_commit is not None:
        if re.fullmatch(r"[0-9a-f]{40}", expected_runtime_commit) is None:
            failures.append("expected runtime commit must be a full lowercase Git hash")
        elif runtime.get("git_commit") != expected_runtime_commit:
            failures.append("runtime.git_commit does not match the requested commit")
    if expected_runtime_tree is not None:
        if re.fullmatch(r"[0-9a-f]{40}", expected_runtime_tree) is None:
            failures.append("expected runtime tree must be a full lowercase Git hash")
        elif runtime.get("git_tree") != expected_runtime_tree:
            failures.append("runtime.git_tree does not match the requested tree")

    source_files = _list(runtime.get("source_files"))
    paths = [item.get("path") for item in source_files if isinstance(item, dict)]
    if paths != list(EXPECTED_SOURCE_PATHS):
        failures.append(
            "runtime.source_files does not match the exact pinned source set"
        )
    if (
        source_files
        and _mapping(source_files[0]).get("sha256") != EXPECTED_ADAPTER_SHA256
    ):
        failures.append("the native CUDA adapter digest is not the expected source")

    if _mapping(record.get("claims")) != EXPECTED_CLAIMS:
        failures.append("claims differ from the closed v2 claim boundary")
    assertions = _mapping(record.get("assertions"))
    if set(assertions) != EXPECTED_ASSERTIONS:
        failures.append("assertions differ from the closed v2 assertion set")
    failed_assertions = sorted(
        key for key, value in assertions.items() if value is not True
    )
    if failed_assertions:
        failures.append(f"assertions failed: {', '.join(failed_assertions)}")

    _validate_success(failures, record.get("success"))
    _validate_cancellation(failures, record.get("cancellation"))
    _validate_recovery(failures, record.get("recovery"))
    _validate_unproxied_success(failures, record.get("unproxied_reuse"))

    success = _mapping(record.get("success"))
    cancellation = _mapping(record.get("cancellation"))
    recovery = _mapping(record.get("recovery"))
    unproxied = _mapping(record.get("unproxied_reuse"))
    model = _mapping(record.get("model"))
    if success.get("budget_available_before") != success.get("budget_available_after"):
        failures.append("the successful transaction did not restore its budget")
    if cancellation.get("controller_cuda_calls") != 0:
        failures.append("the cancelling thread made a CUDA call")
    if recovery.get("retained_budget_available") != ZERO_RESOURCES:
        failures.append(
            "the retained transaction did not retain all declared resources"
        )
    if recovery.get("budget_available_after_recovery") != FULL_RESOURCES:
        failures.append("manual recovery did not restore all declared resources")
    if unproxied.get("budget_available_before") != unproxied.get(
        "budget_available_after"
    ):
        failures.append("the unproxied control run did not restore its budget")
    if model.get("evaluation_mode") is not True:
        failures.append("model.evaluation_mode must be true")
    for field in ("parameter_tensor_count", "buffer_tensor_count"):
        value = model.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            failures.append(f"model.{field} must be a positive integer")
    if model.get("all_parameters_and_buffers_on_profile_device") is not True:
        failures.append("model tensors did not remain on the profile device")
    if model.get("all_floating_tensors_fp32") is not True:
        failures.append("model floating tensors did not remain float32")

    memory = _mapping(record.get("memory"))
    delta = memory.get("observed_success_peak_delta_bytes")
    declared = memory.get("declared_memory_bytes")
    if (
        all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (delta, declared)
        )
        and delta > declared
    ):
        failures.append("observed CUDA allocation exceeds the declared memory vector")
    for location in (
        "success.memory_before",
        "success.memory_after",
        "memory.baseline",
        "memory.final",
    ):
        cursor: Any = record
        for part in location.split("."):
            cursor = _mapping(cursor).get(part)
        snapshot = _mapping(cursor)
        allocated = snapshot.get("allocated_bytes")
        reserved = snapshot.get("reserved_bytes")
        peak_allocated = snapshot.get("peak_allocated_bytes")
        peak_reserved = snapshot.get("peak_reserved_bytes")
        if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (allocated, reserved, peak_allocated, peak_reserved)
        ):
            if reserved < allocated:
                failures.append(f"{location}.reserved_bytes is below allocated_bytes")
            if peak_allocated < allocated:
                failures.append(
                    f"{location}.peak_allocated_bytes is below allocated_bytes"
                )
            if peak_reserved < max(reserved, peak_allocated):
                failures.append(f"{location}.peak_reserved_bytes is incoherent")

    total = _mapping(record.get("timing")).get("total_seconds")
    phases = [
        success.get("wall_seconds"),
        _mapping(recovery.get("post_recovery_reuse")).get("wall_seconds"),
        unproxied.get("wall_seconds"),
    ]
    if all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in [total, *phases]
    ) and total < sum(phases):
        failures.append(
            "timing.total_seconds is shorter than its measured success phases"
        )

    modal_record = _mapping(record.get("modal"))
    if not modal_record.get("function_call_id"):
        failures.append("modal.function_call_id must identify the remote call")

    for location, value in _walk(record):
        key = location.rsplit(".", 1)[-1]
        if SENSITIVE_KEY.search(key):
            failures.append(f"{location} uses a sensitive field name")
        if isinstance(value, str):
            if SENSITIVE_VALUE.search(value):
                failures.append(f"{location} appears to contain a secret")
            if ABSOLUTE_USER_PATH.search(value):
                failures.append(f"{location} contains an absolute user path")
            if any(ord(character) < 32 for character in value):
                failures.append(f"{location} contains a control character")
    return failures


def validate_record(
    record: Any,
    schema: Any,
    *,
    expected_runtime_commit: str | None = None,
    expected_runtime_tree: str | None = None,
) -> list[str]:
    failures: list[str] = []
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import SchemaError
    except ImportError:
        return ["JSON Schema validation is unavailable; install the validation extra"]
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        return [f"the native CUDA transaction schema is invalid: {error}"]
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for error in sorted(
        validator.iter_errors(record), key=lambda item: list(item.path)
    ):
        field = ".".join(str(part) for part in error.absolute_path)
        location = f"record.{field}" if field else "record"
        failures.append(f"{location}: {error.message}")
    if not failures:
        failures.extend(
            validate_semantics(
                record,
                expected_runtime_commit=expected_runtime_commit,
                expected_runtime_tree=expected_runtime_tree,
            )
        )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--expected-runtime-commit")
    parser.add_argument("--expected-runtime-tree")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record = read_json(args.record)
        schema = read_json(args.schema)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"FAIL: cannot read validation input: {error}")
        return 1
    failures = validate_record(
        record,
        schema,
        expected_runtime_commit=args.expected_runtime_commit,
        expected_runtime_tree=args.expected_runtime_tree,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Modal native CUDA transaction record passed schema and semantic validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
