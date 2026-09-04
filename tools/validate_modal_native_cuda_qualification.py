"""Validate the draft native CUDA qualification evidence contract.

The validator checks a closed JSON Schema, derives source identities from clean
Git checkouts, and derives transaction, budget, and allocator observations from
raw events and samples. It checks record consistency. It cannot prove that the
producer observed physical hardware honestly or that a manifest was published
before execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "evidence/modal-native-cuda-qualification.schema.json"
DEFAULT_QUALIFICATION_MANIFEST = ROOT / "experiments/native-cuda-qualification-v2.json"
FAULT_POINTS = (
    "cleanup",
    "event-create",
    "event-record",
    "event-synchronize",
)
RESOURCE_FIELDS = ("memory_bytes", "compute_units", "stream_slots")
REQUIRED_INVARIANTS = (
    "event_stream_globally_ordered",
    "control_and_runtime_results_stable",
    "success_published_once_after_fence",
    "cancellation_after_incomplete_step_never_published",
    "fault_exactly_injected_never_published",
    "fault_blocked_new_work",
    "post_recovery_reuse_committed",
    "leases_held_until_quiescence",
    "budgets_retained_until_quiescence",
    "budgets_restored_after_release",
    "device_and_allocator_samples_within_bounds",
)
CONTROL_EVENTS = ("run-start", "backend-quiescent", "run-complete")
SUCCESS_EVENTS = (
    "run-start",
    "lease-acquired",
    "completion-fence",
    "result-published",
    "transaction-committed",
    "lease-released",
    "budget-restored",
    "run-complete",
)
CANCELLATION_EVENTS = (
    "run-start",
    "lease-acquired",
    "decoder-step-incomplete",
    "cancel-requested",
    "backend-quiescent",
    "transaction-aborted",
    "lease-released",
    "budget-restored",
    "run-complete",
)
FAULT_EVENTS = (
    "run-start",
    "lease-acquired",
    "fault-armed",
    "fault-triggered",
    "fault-triggered",
    "transaction-retained",
    "new-work-rejected",
    "recovery-started",
    "backend-quiescent",
    "transaction-aborted",
    "lease-released",
    "budget-restored",
    "run-complete",
)
EVENT_DETAIL_FIELDS = {
    "decoder_step",
    "fault_point",
    "operation_ordinal",
    "error_type",
    "error_sha256",
    "blocked_request_id",
    "backend_call_relation",
    "planned_injection_count",
}
DETAILS_BY_EVENT = {
    "decoder-step-incomplete": {"decoder_step"},
    "fault-armed": {
        "fault_point",
        "operation_ordinal",
        "planned_injection_count",
    },
    "fault-triggered": {
        "fault_point",
        "operation_ordinal",
        "error_type",
        "error_sha256",
        "backend_call_relation",
    },
    "new-work-rejected": {"blocked_request_id"},
}
SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api_key|authorization|credential|identity_token|password|"
    r"private_key|secret|token)(?:$|_)",
    re.IGNORECASE,
)
SENSITIVE_VALUE = re.compile(
    r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"sk-(?:(?:proj|ant)-)?[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|"
    r"glpat-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"AIza[A-Za-z0-9_-]{20,}|(?:AKIA|ASIA)[0-9A-Z]{16}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|MODAL_TOKEN_(?:ID|SECRET))",
    re.IGNORECASE,
)
ABSOLUTE_USER_PATH = re.compile(
    r"(?:(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/]|"
    r"/(?:home|Users|root)/)",
    re.IGNORECASE,
)
GIT_HASH = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CheckoutIdentity:
    checkout: Path
    repository: str
    git_commit: str
    git_tree: str


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(
            f"Git command failed ({' '.join(arguments)}){suffix}"
        ) from error
    return completed.stdout.strip()


def normalize_git_repository(value: str) -> str:
    """Normalize an HTTPS or common Git SSH origin to a public HTTPS URL."""

    candidate = value.strip()
    scp = re.fullmatch(r"git@([^:/\s]+):([^\s]+)", candidate)
    if scp is not None:
        host = scp.group(1).lower()
        raw_path = scp.group(2)
    else:
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"https", "ssh"} or not parsed.hostname:
            raise ValueError("Git origin must use HTTPS or SSH")
        if parsed.query or parsed.fragment or parsed.password is not None:
            raise ValueError("Git origin must not contain credentials or URL metadata")
        if parsed.scheme == "https" and parsed.username is not None:
            raise ValueError("HTTPS Git origin must not contain credentials")
        if parsed.scheme == "ssh" and parsed.username not in {None, "git"}:
            raise ValueError("SSH Git origin must use the git account")
        host = parsed.hostname.lower()
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        raw_path = parsed.path
    if unquote(raw_path) != raw_path:
        raise ValueError("Git origin path must not contain percent encoding")
    parts = raw_path.strip("/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Git origin path is not normalized")
    if parts[-1].endswith(".git"):
        parts[-1] = parts[-1][:-4]
    if not parts[-1]:
        raise ValueError("Git origin has no repository name")
    return f"https://{host}/{'/'.join(parts)}"


def derive_checkout_identity(checkout: Path) -> CheckoutIdentity:
    """Derive a clean checkout identity from Git instead of caller strings."""

    resolved = checkout.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Git checkout is not a directory: {resolved}")
    top_level = Path(_git_output(resolved, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != resolved:
        raise ValueError("Git checkout must name its top-level directory")
    status = _git_output(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ValueError("Git checkout is dirty or contains untracked files")
    repository = normalize_git_repository(
        _git_output(resolved, "remote", "get-url", "origin")
    )
    commit = _git_output(resolved, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git_output(resolved, "show", "-s", "--format=%T", "HEAD")
    if GIT_HASH.fullmatch(commit) is None or GIT_HASH.fullmatch(tree) is None:
        raise ValueError("Git checkout did not produce full commit and tree identities")
    return CheckoutIdentity(
        checkout=resolved,
        repository=repository,
        git_commit=commit,
        git_tree=tree,
    )


def bind_tracked_artifact(path: Path, identity: CheckoutIdentity) -> tuple[str, str]:
    """Bind an artifact to its tracked path and bytes at checkout HEAD."""

    if not path.is_file():
        raise ValueError(f"artifact is not a file: {path}")
    relative = _repository_relative_path(path, identity.checkout)
    _git_output(identity.checkout, "ls-files", "--error-unmatch", "--", relative)
    committed_blob = _git_output(identity.checkout, "rev-parse", f"HEAD:{relative}")
    worktree_blob = _git_output(identity.checkout, "hash-object", "--", relative)
    if committed_blob != worktree_blob:
        raise ValueError(f"artifact differs from checkout HEAD: {relative}")
    return relative, sha256_file(path)


def _walk(value: Any, location: str = "record") -> Iterable[tuple[str, Any]]:
    yield location, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{location}[{index}]")


def _nearest_rank(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered) / 100))
    return ordered[rank - 1]


def summarize(values: list[int]) -> dict[str, int | str]:
    """Return the exact nearest-rank summary required by the contract."""

    if not values:
        raise ValueError("a qualification distribution requires at least one sample")
    p99: int | str = (
        _nearest_rank(values, 99) if len(values) >= 1000 else "not_estimated"
    )
    return {
        "sample_count": len(values),
        "min": min(values),
        "p50": _nearest_rank(values, 50),
        "p95": _nearest_rank(values, 95),
        "p99": p99,
        "max": max(values),
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_iterations(
    failures: list[str], runs: list[dict[str, Any]], expected: int, label: str
) -> bool:
    valid = True
    if len(runs) != expected:
        failures.append(f"{label} count does not match workload")
        valid = False
    iterations = [run["iteration"] for run in runs]
    if iterations != list(range(len(runs))):
        failures.append(f"{label} iterations must be unique, ordered, and zero-based")
        valid = False
    return valid


def _subtract_resources(
    capacity: dict[str, int], reservation: dict[str, int]
) -> dict[str, int]:
    return {name: capacity[name] - reservation[name] for name in RESOURCE_FIELDS}


def _validate_budget(
    _failures: list[str],
    budget: dict[str, Any],
    workload: dict[str, Any],
    label: str,
) -> bool:
    capacity = workload["resource_capacity"]
    reservation = workload["resource_reservation"]
    held = _subtract_resources(capacity, reservation)
    expected = {
        "available_before": capacity,
        "available_while_held": held,
        "available_at_quiescence": held,
        "available_after_release": capacity,
    }
    valid = True
    for name, expected_vector in expected.items():
        if budget[name] != expected_vector:
            valid = False
    return valid


def _validate_memory(
    failures: list[str],
    memory: dict[str, Any],
    workload: dict[str, Any],
    gpu: dict[str, Any],
    label: str,
) -> bool:
    baseline_allocated = memory["baseline_allocated_bytes"]
    final_allocated = memory["final_allocated_bytes"]
    peak_allocated = memory["peak_allocated_bytes"]
    baseline_reserved = memory["baseline_reserved_bytes"]
    final_reserved = memory["final_reserved_bytes"]
    peak_reserved = memory["peak_reserved_bytes"]
    allocated_delta = memory["peak_allocated_delta_bytes"]
    reserved_delta = memory["peak_reserved_delta_bytes"]
    reservation = workload["resource_reservation"]["memory_bytes"]
    total = gpu["total_memory_bytes"]
    valid = True
    integrity_relations = (
        (
            baseline_allocated <= peak_allocated and final_allocated <= peak_allocated,
            "peak allocation is below an observed allocation",
        ),
        (
            baseline_reserved <= peak_reserved and final_reserved <= peak_reserved,
            "peak reserved memory is below an observed reservation",
        ),
        (
            baseline_allocated <= baseline_reserved
            and final_allocated <= final_reserved
            and peak_allocated <= peak_reserved,
            "allocated memory exceeds reserved memory",
        ),
        (
            allocated_delta == max(0, peak_allocated - baseline_allocated),
            "peak allocation delta is not derived from the sample",
        ),
        (
            reserved_delta == max(0, peak_reserved - baseline_reserved),
            "peak reserved delta is not derived from the sample",
        ),
    )
    for relation, message in integrity_relations:
        if not relation:
            failures.append(f"{label} {message}")
            valid = False
    outcome_relations = (
        final_allocated <= baseline_allocated + workload["allocation_tolerance_bytes"],
        final_reserved <= baseline_reserved + workload["reserved_tolerance_bytes"],
        allocated_delta <= reservation,
        reserved_delta <= reservation,
        max(
            baseline_allocated,
            final_allocated,
            peak_allocated,
            baseline_reserved,
            final_reserved,
            peak_reserved,
        )
        <= total,
    )
    valid &= all(outcome_relations)
    return valid


def _events_for_run(
    events_by_run: dict[str, list[dict[str, Any]]], run: dict[str, Any]
) -> list[dict[str, Any]]:
    return events_by_run.get(run["run_id"], [])


def _event_kinds(events: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(event["event"] for event in events)


def _require_raw_events(
    failures: list[str],
    events: list[dict[str, Any]],
    names: tuple[str, ...],
    label: str,
) -> bool:
    valid = True
    expected_counts = Counter(names)
    observed_counts = Counter(_event_kinds(events))
    for name, expected_count in expected_counts.items():
        if observed_counts[name] != expected_count:
            failures.append(
                f"{label} must record exactly {expected_count} {name} event(s)"
            )
            valid = False
    return valid


def _validate_event_details(
    failures: list[str], event: dict[str, Any], location: str
) -> bool:
    required = DETAILS_BY_EVENT.get(event["event"], set())
    present = EVENT_DETAIL_FIELDS.intersection(event)
    valid = present == required
    if not valid:
        failures.append(
            f"{location} detail fields do not match event type {event['event']}"
        )
    return valid


def _validate_run_event_identity(
    failures: list[str],
    run: dict[str, Any],
    events: list[dict[str, Any]],
    worker_id: str,
    expected_kind: str,
    label: str,
) -> bool:
    valid = True
    expected_identity = {
        "worker_id": worker_id,
        "run_id": run["run_id"],
        "run_kind": expected_kind,
        "session_id": run["session_id"],
        "request_id": run["request_id"],
        "transaction_id": run["transaction_id"],
        "lease_id": run["lease_id"],
    }
    for index, event in enumerate(events):
        for name, expected in expected_identity.items():
            if event[name] != expected:
                failures.append(
                    f"{label}.events[{index}].{name} does not match the run"
                )
                valid = False
        valid &= _validate_event_details(failures, event, f"{label}.events[{index}]")
    return valid


def _validate_wall_time(
    failures: list[str],
    run: dict[str, Any],
    events: list[dict[str, Any]],
    label: str,
    *,
    end_event: str,
) -> bool:
    if not events:
        failures.append(f"{label} has no raw events")
        return False
    starts = [event for event in events if event["event"] == "run-start"]
    ends = [event for event in events if event["event"] == end_event]
    if len(starts) != 1 or len(ends) != 1:
        failures.append(
            f"{label} must have one run-start and one {end_event} for wall_ns"
        )
        return False
    observed = ends[0]["offset_ns"] - starts[0]["offset_ns"]
    if run["wall_ns"] != observed:
        failures.append(f"{label}.wall_ns is not derived from run events")
        return False
    return True


def _validate_success_run(
    failures: list[str],
    run: dict[str, Any],
    events: list[dict[str, Any]],
    workload: dict[str, Any],
    gpu: dict[str, Any],
    worker_id: str,
    run_kind: str,
    label: str,
) -> dict[str, bool]:
    identity = _validate_run_event_identity(
        failures, run, events, worker_id, run_kind, label
    )
    complete = _require_raw_events(failures, events, SUCCESS_EVENTS, label)
    pattern = _event_kinds(events) == SUCCESS_EVENTS
    version = run["session_version_after"] == run["session_version_before"] + 1
    wall = _validate_wall_time(
        failures, run, events, label, end_event="budget-restored"
    )
    budget = _validate_budget(failures, run["budget"], workload, f"{label}.budget")
    memory = _validate_memory(failures, run["memory"], workload, gpu, f"{label}.memory")
    return {
        "publication": identity and complete and pattern and version and wall,
        "lease": pattern,
        "budget": budget and pattern,
        "memory": memory,
    }


def _validate_control_run(
    failures: list[str],
    run: dict[str, Any],
    events: list[dict[str, Any]],
    workload: dict[str, Any],
    gpu: dict[str, Any],
    worker_id: str,
    label: str,
) -> dict[str, bool]:
    identity = True
    for index, event in enumerate(events):
        expected = {
            "worker_id": worker_id,
            "run_id": run["run_id"],
            "run_kind": "control",
        }
        for name, value in expected.items():
            if event[name] != value:
                failures.append(
                    f"{label}.events[{index}].{name} does not match the run"
                )
                identity = False
        identity &= _validate_event_details(failures, event, f"{label}.events[{index}]")
    complete = _require_raw_events(failures, events, CONTROL_EVENTS, label)
    pattern = _event_kinds(events) == CONTROL_EVENTS
    wall = _validate_wall_time(
        failures, run, events, label, end_event="backend-quiescent"
    )
    memory = _validate_memory(failures, run["memory"], workload, gpu, f"{label}.memory")
    return {"control": identity and complete and pattern and wall, "memory": memory}


def _validate_cancellation_run(
    failures: list[str],
    run: dict[str, Any],
    events: list[dict[str, Any]],
    workload: dict[str, Any],
    gpu: dict[str, Any],
    worker_id: str,
    label: str,
) -> dict[str, bool]:
    identity = _validate_run_event_identity(
        failures, run, events, worker_id, "cancellation", label
    )
    complete = _require_raw_events(failures, events, CANCELLATION_EVENTS, label)
    pattern = _event_kinds(events) == CANCELLATION_EVENTS
    version = run["session_version_after"] == run["session_version_before"]
    wall = _validate_wall_time(
        failures, run, events, label, end_event="budget-restored"
    )
    latency = False
    step = False
    if complete:
        by_name = {event["event"]: event for event in events}
        observed = (
            by_name["backend-quiescent"]["offset_ns"]
            - by_name["cancel-requested"]["offset_ns"]
        )
        if run["cancel_to_quiescence_ns"] != observed:
            failures.append(f"{label}.cancel_to_quiescence_ns is not derived")
        latency = run["cancel_to_quiescence_ns"] == observed and observed > 0
        step = by_name["decoder-step-incomplete"]["decoder_step"] >= 1
    budget = _validate_budget(failures, run["budget"], workload, f"{label}.budget")
    memory = _validate_memory(failures, run["memory"], workload, gpu, f"{label}.memory")
    return {
        "cancellation": identity
        and complete
        and pattern
        and version
        and wall
        and latency
        and step,
        "lease": pattern,
        "budget": budget and pattern,
        "memory": memory,
    }


def _validate_fault_run(
    failures: list[str],
    run: dict[str, Any],
    events: list[dict[str, Any]],
    workload: dict[str, Any],
    gpu: dict[str, Any],
    worker_id: str,
    label: str,
) -> dict[str, bool]:
    identity = _validate_run_event_identity(
        failures, run, events, worker_id, "fault", label
    )
    complete = _require_raw_events(failures, events, FAULT_EVENTS, label)
    pattern = _event_kinds(events) == FAULT_EVENTS
    version = run["session_version_after"] == run["session_version_before"]
    wall = _validate_wall_time(
        failures, run, events, label, end_event="budget-restored"
    )
    exact_fault = False
    blocked = False
    latency = False
    if complete:
        by_name = {event["event"]: event for event in events}
        armed = by_name["fault-armed"]
        triggered = [event for event in events if event["event"] == "fault-triggered"]
        first_trigger = triggered[0]
        exact_fault = (
            armed["fault_point"] == run["fault_point"]
            and armed["operation_ordinal"] == 1
            and armed["planned_injection_count"] == run["planned_injection_count"] == 2
            and [event["operation_ordinal"] for event in triggered] == [1, 2]
            and all(
                event["fault_point"] == run["fault_point"]
                and event["error_type"] == first_trigger["error_type"] != ""
                and event["error_sha256"] == first_trigger["error_sha256"]
                and event["backend_call_relation"] == run["backend_call_relation"]
                for event in triggered
            )
        )
        blocked = (
            by_name["new-work-rejected"]["blocked_request_id"]
            == run["blocked_request_id"]
            and run["blocked_request_id"] != run["request_id"]
        )
        injection_to_quiescence = (
            by_name["backend-quiescent"]["offset_ns"] - first_trigger["offset_ns"]
        )
        recovery = (
            by_name["backend-quiescent"]["offset_ns"]
            - by_name["recovery-started"]["offset_ns"]
        )
        latency = (
            injection_to_quiescence > 0
            and recovery > 0
            and run["injection_to_quiescence_ns"] == injection_to_quiescence
            and run["recovery_ns"] == recovery
        )
        if run["injection_to_quiescence_ns"] != injection_to_quiescence:
            failures.append(f"{label}.injection_to_quiescence_ns is not derived")
        if run["recovery_ns"] != recovery:
            failures.append(f"{label}.recovery_ns is not derived")
    budget = _validate_budget(failures, run["budget"], workload, f"{label}.budget")
    memory = _validate_memory(failures, run["memory"], workload, gpu, f"{label}.memory")
    return {
        "fault": identity
        and complete
        and pattern
        and version
        and wall
        and exact_fault
        and latency,
        "blocked": pattern and blocked,
        "lease": pattern,
        "budget": budget and pattern,
        "memory": memory,
    }


def _check_bound_value(
    failures: list[str],
    actual: str,
    expected: str,
    label: str,
    pattern: re.Pattern[str],
) -> None:
    if pattern.fullmatch(expected) is None:
        failures.append(f"expected {label} has an invalid format")
    elif actual != expected:
        failures.append(f"{label} does not match the caller-bound value")


def _safe_repository_uri(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and parsed.query == ""
        and parsed.fragment == ""
    )


def _safe_relative_path(value: str) -> bool:
    candidate = value.replace("\\", "/")
    return (
        candidate != ""
        and not candidate.startswith("/")
        and not re.match(r"^[A-Za-z]:", candidate)
        and all(part not in ("", ".", "..") for part in candidate.split("/"))
    )


def _closed_manifest_object(
    value: Any,
    fields: set[str],
    location: str,
    failures: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        failures.append(f"{location} must be an object")
        return None
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        failures.append(f"{location} is missing fields: {', '.join(missing)}")
    if unknown:
        failures.append(f"{location} has unknown fields: {', '.join(unknown)}")
    return value


def validate_qualification_manifest(
    manifest: Any,
    location: str = "qualification manifest",
) -> list[str]:
    """Validate a closed qualification registration."""

    failures: list[str] = []
    top = _closed_manifest_object(
        manifest,
        {
            "manifest_version",
            "manifest_id",
            "state",
            "purpose",
            "claim_boundary",
            "source_policy",
            "cell",
            "resource_contract",
            "measurement_boundaries",
            "sampling",
            "faults",
            "exclusion_rule",
            "required_invariants",
            "timestamp_evidence",
        },
        location,
        failures,
    )
    if top is None:
        return failures
    manifest_version = top.get("manifest_version")
    manifest_ids = {
        "1": "native-cuda-qualification-v1",
        "2": "native-cuda-qualification-v2",
    }
    if manifest_version not in manifest_ids:
        failures.append(f"{location}.manifest_version must be '1' or '2'")
    elif top.get("manifest_id") != manifest_ids[manifest_version]:
        failures.append(
            f"{location}.manifest_id must be {manifest_ids[manifest_version]!r}"
        )
    if top.get("state") != "preregistered":
        failures.append(f"{location}.state must be 'preregistered'")
    if not isinstance(top.get("purpose"), str) or not top["purpose"].strip():
        failures.append(f"{location}.purpose must be a non-empty string")

    claim = _closed_manifest_object(
        top.get("claim_boundary"),
        {"evidence_tier", "performance_benchmark", "production_readiness"},
        f"{location}.claim_boundary",
        failures,
    )
    if claim is not None and claim != {
        "evidence_tier": "qualification",
        "performance_benchmark": False,
        "production_readiness": False,
    }:
        failures.append(f"{location}.claim_boundary is not qualification-only")

    source_policy = _closed_manifest_object(
        top.get("source_policy"),
        {
            "runtime",
            "runtime_repository",
            "backend",
            "backend_repository",
            "backend_commit",
            "backend_tree",
            "artifacts",
            "producer_script_path",
            "container_image",
            "dependencies",
        },
        f"{location}.source_policy",
        failures,
    )
    if source_policy is not None:
        expected_policy = {
            "runtime": "clean-head-containing-validator-and-manifest",
            "runtime_repository": "https://github.com/billmedj/whisper-runtime",
            "backend": "clean-head",
            "backend_repository": "https://github.com/openai/whisper",
            "backend_commit": "a0b9695ae1cc52bad4b8626fe9fb6ea4ac0ee650",
            "backend_tree": "c011d2563c26763b5f147026e6b18ef85bccd4fb",
            "artifacts": "tracked-at-runtime-head",
            "producer_script_path": "infra/modal_native_cuda_qualification.py",
            "container_image": "modal-object-id-required",
            "dependencies": (
                "tracked-image-inputs-and-observed-resolved-inventory-required"
            ),
        }
        if source_policy != expected_policy:
            failures.append(
                f"{location}.source_policy does not match the qualification contract"
            )

    cell_fields = {
        "registered_cell_id",
        "profile_id",
        "cloud_selector",
        "region_selector",
        "cloud_provider",
        "region",
        "gpu_name",
        "model",
        "checkpoint_source",
        "checkpoint_sha256",
        "fixture_id",
        "input_manifest_source",
        "input_manifest_sha256",
        "input_source",
        "input_sha256",
        "decoded_pcm_sha256",
        "input_bytes",
        "input_duration_ns",
        "sample_rate_hz",
        "channel_count",
        "language",
        "task",
        "numeric_precision",
        "device",
        "execution_mode",
        "pair_order",
        "seed",
        "seed_reset_each_run",
        "compatibility_rule",
        "result_digest_encoding",
        "expected_result_sha256",
        "preprocessing_options",
        "preprocessing_options_sha256",
        "decode_options",
        "decode_options_sha256",
    }
    cell = _closed_manifest_object(
        top.get("cell"), cell_fields, f"{location}.cell", failures
    )
    if cell is not None:
        for field in cell_fields - {
            "checkpoint_sha256",
            "input_sha256",
            "decoded_pcm_sha256",
            "input_manifest_sha256",
            "expected_result_sha256",
            "preprocessing_options_sha256",
            "decode_options_sha256",
            "seed",
            "seed_reset_each_run",
            "preprocessing_options",
            "decode_options",
            "input_bytes",
            "input_duration_ns",
            "sample_rate_hz",
            "channel_count",
        }:
            if not isinstance(cell.get(field), str) or not cell[field]:
                failures.append(f"{location}.cell.{field} must be a non-empty string")
        if cell.get("cloud_selector") != "aws":
            failures.append(f"{location}.cell.cloud_selector must be aws")
        if cell.get("region_selector") != "us-west":
            failures.append(f"{location}.cell.region_selector must be us-west")
        if cell.get("cloud_provider") != "CLOUD_PROVIDER_AWS":
            failures.append(
                f"{location}.cell.cloud_provider must be CLOUD_PROVIDER_AWS"
            )
        if cell.get("region") != "us-west-2":
            failures.append(f"{location}.cell.region must be us-west-2")
        if manifest_version in manifest_ids:
            expected_cell_id = f"t4-tiny-en-jfk-qualification-v{manifest_version}"
            if cell.get("registered_cell_id") != expected_cell_id:
                failures.append(
                    f"{location}.cell.registered_cell_id must be "
                    f"{expected_cell_id!r}"
                )
        for field in (
            "checkpoint_sha256",
            "input_sha256",
            "decoded_pcm_sha256",
            "input_manifest_sha256",
            "expected_result_sha256",
            "preprocessing_options_sha256",
            "decode_options_sha256",
        ):
            if SHA256.fullmatch(str(cell.get(field, ""))) is None:
                failures.append(f"{location}.cell.{field} must be a SHA-256 digest")
        if not isinstance(cell.get("seed"), int) or isinstance(cell.get("seed"), bool):
            failures.append(f"{location}.cell.seed must be an integer")
        for field in (
            "input_bytes",
            "input_duration_ns",
            "sample_rate_hz",
            "channel_count",
        ):
            if (
                not isinstance(cell.get(field), int)
                or isinstance(cell.get(field), bool)
                or cell[field] <= 0
            ):
                failures.append(f"{location}.cell.{field} must be a positive integer")
        if cell.get("seed_reset_each_run") is not True:
            failures.append(f"{location}.cell.seed_reset_each_run must be true")
        for field in ("preprocessing_options", "decode_options"):
            if not isinstance(cell.get(field), dict):
                failures.append(f"{location}.cell.{field} must be an object")
            elif cell.get(f"{field}_sha256") != canonical_sha256(cell[field]):
                failures.append(f"{location}.cell.{field}_sha256 is not canonical")

    resource = _closed_manifest_object(
        top.get("resource_contract"),
        {
            "capacity",
            "reservation",
            "allocation_tolerance_bytes",
            "reserved_tolerance_bytes",
        },
        f"{location}.resource_contract",
        failures,
    )
    if resource is not None:
        expected_resource = {
            "capacity": {
                "memory_bytes": 2_147_483_648,
                "compute_units": 1,
                "stream_slots": 1,
            },
            "reservation": {
                "memory_bytes": 2_147_483_648,
                "compute_units": 1,
                "stream_slots": 1,
            },
            "allocation_tolerance_bytes": 67_108_864,
            "reserved_tolerance_bytes": 67_108_864,
        }
        if resource != expected_resource:
            failures.append(
                f"{location}.resource_contract does not match the qualification contract"
            )

    boundaries = _closed_manifest_object(
        top.get("measurement_boundaries"),
        {"runtime_start", "runtime_end", "control_start", "control_end"},
        f"{location}.measurement_boundaries",
        failures,
    )
    expected_boundaries = {
        "runtime_start": "before-admission",
        "runtime_end": "after-budget-restored",
        "control_start": "before-backend-call",
        "control_end": "after-backend-quiescent",
    }
    if boundaries is not None and boundaries != expected_boundaries:
        failures.append(
            f"{location}.measurement_boundaries do not match the qualification contract"
        )

    sampling = _closed_manifest_object(
        top.get("sampling"),
        {
            "warmup_pairs",
            "measured_pairs",
            "cancellation_runs",
            "fault_repetitions_per_point",
        },
        f"{location}.sampling",
        failures,
    )
    expected_sampling = {
        "warmup_pairs": 2,
        "measured_pairs": 5,
        "cancellation_runs": 3,
        "fault_repetitions_per_point": 2,
    }
    if sampling is not None and sampling != expected_sampling:
        failures.append(
            f"{location}.sampling does not match the qualification contract"
        )

    faults = _closed_manifest_object(
        top.get("faults"),
        {"points", "planned_injections_per_run"},
        f"{location}.faults",
        failures,
    )
    if faults is not None and faults != {
        "points": list(FAULT_POINTS),
        "planned_injections_per_run": 2,
    }:
        failures.append(f"{location}.faults does not match the qualification contract")

    exclusion = _closed_manifest_object(
        top.get("exclusion_rule"),
        {"id", "allowed_classes", "max_attempts", "publish_all_attempts"},
        f"{location}.exclusion_rule",
        failures,
    )
    if exclusion is not None and exclusion != {
        "id": "no-exclusions-v1",
        "allowed_classes": [],
        "max_attempts": 1,
        "publish_all_attempts": True,
    }:
        failures.append(
            f"{location}.exclusion_rule does not match the qualification contract"
        )

    if top.get("required_invariants") != list(REQUIRED_INVARIANTS):
        failures.append(
            f"{location}.required_invariants does not match the qualification contract"
        )
    timestamp = _closed_manifest_object(
        top.get("timestamp_evidence"),
        {
            "local_validator_proves_prior_publication",
            "external_public_timestamp_required",
        },
        f"{location}.timestamp_evidence",
        failures,
    )
    if timestamp is not None and timestamp != {
        "local_validator_proves_prior_publication": False,
        "external_public_timestamp_required": True,
    }:
        failures.append(f"{location}.timestamp_evidence overstates local proof")
    return failures


def _validate_registration_binding(
    failures: list[str],
    record: dict[str, Any],
    manifest: Any,
    *,
    runtime_identity: CheckoutIdentity,
    backend_identity: CheckoutIdentity,
    expected_manifest_path: str,
    expected_manifest_sha256: str,
) -> None:
    registration = record["qualification_registration"]
    if registration["manifest_path"] != expected_manifest_path:
        failures.append(
            "qualification_registration.manifest_path does not match the "
            "caller-bound path"
        )
    _check_bound_value(
        failures,
        registration["manifest_sha256"],
        expected_manifest_sha256,
        "qualification_registration.manifest_sha256",
        SHA256,
    )
    _check_bound_value(
        failures,
        registration["runtime_commit"],
        runtime_identity.git_commit,
        "qualification_registration.runtime_commit",
        GIT_HASH,
    )
    manifest_failures = validate_qualification_manifest(manifest)
    failures.extend(manifest_failures)
    if manifest_failures or not isinstance(manifest, dict):
        return
    if registration["manifest_id"] != manifest["manifest_id"]:
        failures.append(
            "qualification_registration.manifest_id does not match manifest"
        )
    workload = record["workload"]
    outcome = record["outcome"]
    worker = record["worker"]
    gpu = record["gpu"]
    cell = manifest["cell"]
    if worker["campaign_id"] != manifest["manifest_id"]:
        failures.append("worker.campaign_id does not match manifest")
    for record_value, manifest_value, label in (
        (outcome["registered_cell_id"], cell["registered_cell_id"], "cell id"),
        (workload["profile_id"], cell["profile_id"], "profile"),
        (gpu["cloud_provider"], cell["cloud_provider"], "cloud provider"),
        (gpu["region"], cell["region"], "region"),
        (gpu["name"], cell["gpu_name"], "GPU"),
        (workload["model"], cell["model"], "model"),
        (workload["checkpoint_source"], cell["checkpoint_source"], "checkpoint source"),
        (workload["checkpoint_sha256"], cell["checkpoint_sha256"], "checkpoint"),
        (workload["fixture_id"], cell["fixture_id"], "fixture"),
        (
            workload["input_manifest_source"],
            cell["input_manifest_source"],
            "input manifest source",
        ),
        (
            workload["input_manifest_sha256"],
            cell["input_manifest_sha256"],
            "input manifest",
        ),
        (workload["input_source"], cell["input_source"], "input source"),
        (workload["input_sha256"], cell["input_sha256"], "input"),
        (workload["decoded_pcm_sha256"], cell["decoded_pcm_sha256"], "decoded PCM"),
        (workload["input_bytes"], cell["input_bytes"], "input bytes"),
        (workload["input_duration_ns"], cell["input_duration_ns"], "input duration"),
        (workload["sample_rate_hz"], cell["sample_rate_hz"], "sample rate"),
        (workload["channel_count"], cell["channel_count"], "channel count"),
        (workload["language"], cell["language"], "language"),
        (workload["task"], cell["task"], "task"),
        (workload["numeric_precision"], cell["numeric_precision"], "precision"),
        (workload["device"], cell["device"], "device"),
        (workload["execution_mode"], cell["execution_mode"], "execution mode"),
        (workload["pair_order"], cell["pair_order"], "pair order"),
        (workload["seed"], cell["seed"], "seed"),
        (
            workload["seed_reset_each_run"],
            cell["seed_reset_each_run"],
            "seed reset rule",
        ),
        (
            workload["compatibility_rule"],
            cell["compatibility_rule"],
            "compatibility rule",
        ),
        (
            workload["result_digest_encoding"],
            cell["result_digest_encoding"],
            "result digest encoding",
        ),
        (
            workload["expected_result_sha256"],
            cell["expected_result_sha256"],
            "expected result",
        ),
        (
            workload["preprocessing_options"],
            cell["preprocessing_options"],
            "preprocessing options",
        ),
        (
            workload["preprocessing_options_sha256"],
            cell["preprocessing_options_sha256"],
            "preprocessing options digest",
        ),
        (workload["decode_options"], cell["decode_options"], "decode options"),
        (
            workload["decode_options_sha256"],
            cell["decode_options_sha256"],
            "decode options digest",
        ),
    ):
        if record_value != manifest_value:
            failures.append(f"qualification record {label} does not match manifest")
    if outcome["exclusion_rule_id"] != manifest["exclusion_rule"]["id"]:
        failures.append("qualification record exclusion rule does not match manifest")
    source_policy = manifest["source_policy"]
    for actual, expected, label in (
        (runtime_identity.repository, source_policy["runtime_repository"], "runtime"),
        (backend_identity.repository, source_policy["backend_repository"], "backend"),
        (
            backend_identity.git_commit,
            source_policy["backend_commit"],
            "backend commit",
        ),
        (backend_identity.git_tree, source_policy["backend_tree"], "backend tree"),
    ):
        if actual != expected:
            failures.append(f"qualification {label} does not match manifest")
    if record["producer"]["script_path"] != source_policy["producer_script_path"]:
        failures.append("qualification producer script path does not match manifest")
    resource = manifest["resource_contract"]
    for actual, expected, label in (
        (workload["resource_capacity"], resource["capacity"], "resource capacity"),
        (
            workload["resource_reservation"],
            resource["reservation"],
            "resource reservation",
        ),
        (
            workload["allocation_tolerance_bytes"],
            resource["allocation_tolerance_bytes"],
            "allocation tolerance",
        ),
        (
            workload["reserved_tolerance_bytes"],
            resource["reserved_tolerance_bytes"],
            "reserved tolerance",
        ),
    ):
        if actual != expected:
            failures.append(f"qualification record {label} does not match manifest")
    sampling = manifest["sampling"]
    for field, manifest_field in (
        ("warmup_iterations", "warmup_pairs"),
        ("measured_iterations", "measured_pairs"),
        ("control_iterations", "measured_pairs"),
        ("cancellation_iterations", "cancellation_runs"),
        ("fault_repetitions_per_point", "fault_repetitions_per_point"),
    ):
        if workload[field] != sampling[manifest_field]:
            failures.append(f"qualification record {field} does not match manifest")
    boundaries = manifest["measurement_boundaries"]
    for field, manifest_field in (
        ("runtime_measurement_start", "runtime_start"),
        ("runtime_measurement_end", "runtime_end"),
        ("control_measurement_start", "control_start"),
        ("control_measurement_end", "control_end"),
    ):
        if workload[field] != boundaries[manifest_field]:
            failures.append(f"qualification record {field} does not match manifest")


def _validate_source_bindings(
    failures: list[str],
    record: dict[str, Any],
    *,
    runtime_identity: CheckoutIdentity,
    backend_identity: CheckoutIdentity,
    expected_patch_manifest_path: str,
    expected_patch_manifest_sha256: str,
    expected_producer_script_path: str,
    expected_producer_script_sha256: str,
    expected_schema_path: str,
    expected_schema_sha256: str,
    expected_validator_path: str,
    expected_validator_sha256: str,
    expected_image_inputs_path: str,
    expected_image_inputs_sha256: str,
) -> None:
    runtime = record["runtime"]
    backend = record["backend"]
    producer = record["producer"]
    for actual, expected, label in (
        (runtime["repository"], runtime_identity.repository, "runtime.repository"),
        (backend["repository"], backend_identity.repository, "backend.repository"),
    ):
        if not _safe_repository_uri(expected):
            failures.append(f"expected {label} must be an HTTPS repository URI")
        elif actual != expected:
            failures.append(f"{label} does not match the caller-bound value")
    _check_bound_value(
        failures,
        runtime["git_commit"],
        runtime_identity.git_commit,
        "runtime.git_commit",
        GIT_HASH,
    )
    _check_bound_value(
        failures,
        runtime["git_tree"],
        runtime_identity.git_tree,
        "runtime.git_tree",
        GIT_HASH,
    )
    _check_bound_value(
        failures,
        backend["git_commit"],
        backend_identity.git_commit,
        "backend.git_commit",
        GIT_HASH,
    )
    _check_bound_value(
        failures,
        backend["git_tree"],
        backend_identity.git_tree,
        "backend.git_tree",
        GIT_HASH,
    )
    _check_bound_value(
        failures,
        backend["patch_manifest_sha256"],
        expected_patch_manifest_sha256,
        "backend.patch_manifest_sha256",
        SHA256,
    )
    for actual, expected, label in (
        (
            backend["patch_manifest_path"],
            expected_patch_manifest_path,
            "backend.patch_manifest_path",
        ),
        (
            producer["script_path"],
            expected_producer_script_path,
            "producer.script_path",
        ),
        (
            producer["schema_path"],
            expected_schema_path,
            "producer.schema_path",
        ),
        (
            producer["validator_path"],
            expected_validator_path,
            "producer.validator_path",
        ),
        (
            producer["image_inputs_path"],
            expected_image_inputs_path,
            "producer.image_inputs_path",
        ),
    ):
        if not _safe_relative_path(expected):
            failures.append(f"expected {label} is not repository-relative")
        elif actual != expected:
            failures.append(f"{label} does not match the caller-bound path")
    _check_bound_value(
        failures,
        producer["script_sha256"],
        expected_producer_script_sha256,
        "producer.script_sha256",
        SHA256,
    )
    for field, expected in (
        ("schema_sha256", expected_schema_sha256),
        ("validator_sha256", expected_validator_sha256),
        ("image_inputs_sha256", expected_image_inputs_sha256),
    ):
        _check_bound_value(
            failures,
            producer[field],
            expected,
            f"producer.{field}",
            SHA256,
        )
    if (
        producer["repository"] != runtime["repository"]
        or producer["git_commit"] != runtime["git_commit"]
        or producer["git_tree"] != runtime["git_tree"]
    ):
        failures.append("producer identity must be part of the bound runtime tree")
    for label, value in (
        ("runtime.repository", runtime["repository"]),
        ("backend.repository", backend["repository"]),
        ("producer.repository", producer["repository"]),
    ):
        if not _safe_repository_uri(value):
            failures.append(
                f"{label} must be an HTTPS URI without credentials or query"
            )
    for label, value in (
        ("backend.patch_manifest_path", backend["patch_manifest_path"]),
        ("producer.script_path", producer["script_path"]),
        ("producer.schema_path", producer["schema_path"]),
        ("producer.validator_path", producer["validator_path"]),
        ("producer.image_inputs_path", producer["image_inputs_path"]),
    ):
        if not _safe_relative_path(value):
            failures.append(f"{label} must be a normalized repository-relative path")
    resolved = producer["resolved_dependencies"]
    names_and_versions = [
        (dependency["name"], dependency["version"]) for dependency in resolved
    ]
    if names_and_versions != sorted(names_and_versions):
        failures.append("producer.resolved_dependencies must be sorted")
    names = [name for name, _ in names_and_versions]
    if len(names) != len(set(names)):
        failures.append("producer.resolved_dependencies must have unique names")
    if producer["resolved_dependencies_sha256"] != canonical_sha256(resolved):
        failures.append("producer.resolved_dependencies_sha256 is not canonical")
    observed_versions = dict(names_and_versions)
    for name, environment_field in (("modal", "modal_sdk"), ("torch", "torch")):
        if observed_versions.get(name) != record["environment"][environment_field]:
            failures.append(
                f"producer resolved {name} version does not match environment"
            )


def _validate_global_events(
    failures: list[str],
    events: list[dict[str, Any]],
    expected_runs: list[dict[str, Any]],
    worker_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    valid = True
    sequences = [event["sequence"] for event in events]
    if sequences != list(range(len(events))):
        failures.append("events must be unique, globally ordered, and zero-based")
        valid = False
    offsets = [event["offset_ns"] for event in events]
    if any(left >= right for left, right in zip(offsets, offsets[1:], strict=False)):
        failures.append("events must have strictly increasing monotonic offsets")
        valid = False
    if any(event["worker_id"] != worker_id for event in events):
        failures.append("all events must bind the declared single worker")
        valid = False
    events_by_run: dict[str, list[dict[str, Any]]] = {}
    observed_order: list[str] = []
    previous_run: str | None = None
    for event in events:
        run_id = event["run_id"]
        events_by_run.setdefault(run_id, []).append(event)
        if run_id != previous_run:
            observed_order.append(run_id)
            previous_run = run_id
    expected_order = [run["run_id"] for run in expected_runs]
    if observed_order != expected_order:
        failures.append("events do not follow the canonical non-overlapping run order")
        valid = False
    if set(events_by_run) != set(expected_order):
        failures.append("events contain a missing or unknown run identifier")
        valid = False
    return events_by_run, valid


def _validate_identity_uniqueness(
    failures: list[str], runs: list[dict[str, Any]], fault_runs: list[dict[str, Any]]
) -> bool:
    valid = True
    for field in ("run_id", "request_id", "transaction_id", "lease_id"):
        values = [run[field] for run in runs]
        if len(values) != len(set(values)):
            failures.append(f"all {field} values must be unique")
            valid = False
    blocked = [run["blocked_request_id"] for run in fault_runs]
    primary_requests = {run["request_id"] for run in runs}
    if len(blocked) != len(set(blocked)) or any(
        item in primary_requests for item in blocked
    ):
        failures.append("blocked request identifiers must be unique competing requests")
        valid = False
    return valid


def _validate_device_and_workload(failures: list[str], record: dict[str, Any]) -> bool:
    workload = record["workload"]
    gpu = record["gpu"]
    scope = record["scope"]
    worker = record["worker"]
    valid = True
    if scope["evidence_tier"] != "qualification" or scope["performance_benchmark"]:
        failures.append("scope must remain qualification-only")
        valid = False
    if workload["device"] != f"cuda:{gpu['device_index']}":
        failures.append("workload.device does not match gpu.device_index")
        valid = False
    if gpu["device_index"] >= gpu["visible_device_count"]:
        failures.append("gpu.device_index is outside the visible device set")
        valid = False
    capacity = workload["resource_capacity"]
    reservation = workload["resource_reservation"]
    for field in RESOURCE_FIELDS:
        if reservation[field] > capacity[field]:
            failures.append(f"resource reservation exceeds capacity for {field}")
            valid = False
    if capacity["memory_bytes"] > gpu["total_memory_bytes"]:
        failures.append("resource memory capacity exceeds physical device memory")
        valid = False
    if workload["decode_options_sha256"] != canonical_sha256(
        workload["decode_options"]
    ):
        failures.append("workload.decode_options_sha256 is not canonical")
        valid = False
    if workload["preprocessing_options_sha256"] != canonical_sha256(
        workload["preprocessing_options"]
    ):
        failures.append("workload.preprocessing_options_sha256 is not canonical")
        valid = False
    if (
        workload["preprocessing_options"]["sample_rate_hz"]
        != workload["sample_rate_hz"]
    ):
        failures.append("preprocessing sample rate does not match the input profile")
        valid = False
    if workload["decode_options"]["fp16"] != (
        workload["numeric_precision"] == "float16"
    ):
        failures.append("decode fp16 setting does not match numeric precision")
        valid = False
    for field in ("checkpoint_source", "input_manifest_source", "input_source"):
        if not _safe_repository_uri(workload[field]):
            failures.append(f"workload.{field} must be a stable public HTTPS URI")
            valid = False
    if workload["warmup_iterations"] != 2:
        failures.append("qualification requires exactly two warm-up runs")
        valid = False
    if workload["measured_iterations"] != 5:
        failures.append("qualification requires exactly five measured runs")
        valid = False
    if worker["worker_ordinal"] != 0 or worker["expected_worker_count"] != 1:
        failures.append("qualification requires one registered worker")
        valid = False
    if worker["worker_ordinal"] >= worker["expected_worker_count"]:
        failures.append("worker.worker_ordinal is outside the registered campaign")
        valid = False
    if workload["control_iterations"] != workload["measured_iterations"]:
        failures.append("control and runtime measured iteration counts must match")
        valid = False
    expected_pair_order = (
        "control-first" if worker["worker_ordinal"] % 2 == 0 else "runtime-first"
    )
    if workload["pair_order"] != expected_pair_order:
        failures.append("workload.pair_order does not alternate by worker ordinal")
        valid = False
    return valid


def validate_semantics(
    record: Any,
    *,
    runtime_identity: CheckoutIdentity,
    backend_identity: CheckoutIdentity,
    qualification_manifest: Any,
    expected_qualification_manifest_path: str,
    expected_qualification_manifest_sha256: str,
    expected_patch_manifest_path: str,
    expected_patch_manifest_sha256: str,
    expected_producer_script_path: str,
    expected_producer_script_sha256: str,
    expected_schema_path: str,
    expected_schema_sha256: str,
    expected_validator_path: str,
    expected_validator_sha256: str,
    expected_image_inputs_path: str,
    expected_image_inputs_sha256: str,
) -> list[str]:
    """Return failures for relations across raw qualification observations."""

    if not isinstance(record, dict):
        return ["record must be an object"]
    failures: list[str] = []
    _validate_source_bindings(
        failures,
        record,
        runtime_identity=runtime_identity,
        backend_identity=backend_identity,
        expected_patch_manifest_path=expected_patch_manifest_path,
        expected_patch_manifest_sha256=expected_patch_manifest_sha256,
        expected_producer_script_path=expected_producer_script_path,
        expected_producer_script_sha256=expected_producer_script_sha256,
        expected_schema_path=expected_schema_path,
        expected_schema_sha256=expected_schema_sha256,
        expected_validator_path=expected_validator_path,
        expected_validator_sha256=expected_validator_sha256,
        expected_image_inputs_path=expected_image_inputs_path,
        expected_image_inputs_sha256=expected_image_inputs_sha256,
    )
    _validate_registration_binding(
        failures,
        record,
        qualification_manifest,
        runtime_identity=runtime_identity,
        backend_identity=backend_identity,
        expected_manifest_path=expected_qualification_manifest_path,
        expected_manifest_sha256=expected_qualification_manifest_sha256,
    )
    workload = record["workload"]
    gpu = record["gpu"]
    warmups = record["warmup_runs"]
    control_warmups = record["control_warmup_runs"]
    controls = record["control_runs"]
    measured = record["measured_runs"]
    cancellations = record["cancellation_runs"]
    faults = record["fault_runs"]
    _validate_iterations(
        failures, warmups, workload["warmup_iterations"], "warmup_runs"
    )
    _validate_iterations(
        failures,
        control_warmups,
        workload["warmup_iterations"],
        "control_warmup_runs",
    )
    _validate_iterations(
        failures, measured, workload["measured_iterations"], "measured_runs"
    )
    _validate_iterations(
        failures, controls, workload["control_iterations"], "control_runs"
    )
    _validate_iterations(
        failures,
        cancellations,
        workload["cancellation_iterations"],
        "cancellation_runs",
    )
    expected_fault_keys = [
        (point, repetition)
        for point in FAULT_POINTS
        for repetition in range(workload["fault_repetitions_per_point"])
    ]
    observed_fault_keys = [(run["fault_point"], run["repetition"]) for run in faults]
    fault_matrix = observed_fault_keys == expected_fault_keys
    if not fault_matrix:
        failures.append(
            "fault_runs must cover each fault point and repetition exactly once "
            "in canonical order"
        )
    ordered_runs: list[tuple[dict[str, Any], str]] = []
    control_first = workload["pair_order"] == "control-first"
    for control, runtime in zip(control_warmups, warmups, strict=False):
        pair = ((control, "control"), (runtime, "warmup"))
        ordered_runs.extend(pair if control_first else reversed(pair))
    for control, runtime in zip(controls, measured, strict=False):
        pair = ((control, "control"), (runtime, "measured"))
        ordered_runs.extend(pair if control_first else reversed(pair))
    ordered_runs.extend((run, "cancellation") for run in cancellations)
    for fault in faults:
        ordered_runs.append((fault, "fault"))
        ordered_runs.append((fault["post_recovery_reuse"], "reuse"))
    all_runs = [run for run, _ in ordered_runs]
    events_by_run, global_events = _validate_global_events(
        failures, record["events"], all_runs, record["worker"]["worker_id"]
    )
    transaction_runs = [run for run, kind in ordered_runs if kind != "control"]
    identities = _validate_identity_uniqueness(failures, transaction_runs, faults)
    device = _validate_device_and_workload(failures, record)

    control_warmup_results = [
        _validate_control_run(
            failures,
            run,
            _events_for_run(events_by_run, run),
            workload,
            gpu,
            record["worker"]["worker_id"],
            f"control_warmup_runs[{index}]",
        )
        for index, run in enumerate(control_warmups)
    ]
    control_results = [
        _validate_control_run(
            failures,
            run,
            _events_for_run(events_by_run, run),
            workload,
            gpu,
            record["worker"]["worker_id"],
            f"control_runs[{index}]",
        )
        for index, run in enumerate(controls)
    ]

    success_results: list[dict[str, bool]] = []
    for group, kind in ((warmups, "warmup"), (measured, "measured")):
        for index, run in enumerate(group):
            success_results.append(
                _validate_success_run(
                    failures,
                    run,
                    _events_for_run(events_by_run, run),
                    workload,
                    gpu,
                    record["worker"]["worker_id"],
                    kind,
                    f"{kind}_runs[{index}]",
                )
            )
    cancellation_results = [
        _validate_cancellation_run(
            failures,
            run,
            _events_for_run(events_by_run, run),
            workload,
            gpu,
            record["worker"]["worker_id"],
            f"cancellation_runs[{index}]",
        )
        for index, run in enumerate(cancellations)
    ]
    fault_results: list[dict[str, bool]] = []
    reuse_results: list[dict[str, bool]] = []
    reuse_linkage = True
    for index, run in enumerate(faults):
        fault_results.append(
            _validate_fault_run(
                failures,
                run,
                _events_for_run(events_by_run, run),
                workload,
                gpu,
                record["worker"]["worker_id"],
                f"fault_runs[{index}]",
            )
        )
        reuse = run["post_recovery_reuse"]
        linked = (
            reuse["session_id"] == run["session_id"]
            and reuse["session_version_before"] == run["session_version_after"]
        )
        if not linked:
            failures.append(
                f"fault_runs[{index}].post_recovery_reuse is not linked to recovery"
            )
            reuse_linkage = False
        reuse_results.append(
            _validate_success_run(
                failures,
                reuse,
                _events_for_run(events_by_run, reuse),
                workload,
                gpu,
                record["worker"]["worker_id"],
                "reuse",
                f"fault_runs[{index}].post_recovery_reuse",
            )
        )
    runtime_successful_runs = [
        *warmups,
        *measured,
        *(run["post_recovery_reuse"] for run in faults),
    ]
    runtime_result_hashes = {run["result_sha256"] for run in runtime_successful_runs}
    expected_result_sha256 = workload["expected_result_sha256"]
    runtime_stable = runtime_result_hashes == {expected_result_sha256}
    control_runtime_stable = {
        run["result_sha256"]
        for run in [*control_warmups, *controls, *runtime_successful_runs]
    } == {expected_result_sha256}
    expected_summaries = {
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
                [run["recovery_ns"] for run in faults if run["fault_point"] == point]
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
    }
    for name, expected in expected_summaries.items():
        if record["summaries"][name] != expected:
            failures.append(
                f"summaries.{name} does not match the raw nearest-rank samples"
            )
    transaction_results = [
        *success_results,
        *cancellation_results,
        *fault_results,
        *reuse_results,
    ]
    all_memory_results = [
        *transaction_results,
        *control_warmup_results,
        *control_results,
    ]
    derived = {
        "event_stream_globally_ordered": global_events and identities,
        "control_and_runtime_results_stable": control_runtime_stable
        and all(
            item["control"] for item in [*control_warmup_results, *control_results]
        ),
        "success_published_once_after_fence": all(
            item["publication"] for item in [*success_results, *reuse_results]
        ),
        "cancellation_after_incomplete_step_never_published": all(
            item["cancellation"] for item in cancellation_results
        ),
        "fault_exactly_injected_never_published": fault_matrix
        and all(item["fault"] for item in fault_results),
        "fault_blocked_new_work": all(item["blocked"] for item in fault_results),
        "post_recovery_reuse_committed": reuse_linkage
        and runtime_stable
        and all(item["publication"] for item in reuse_results),
        "leases_held_until_quiescence": all(
            item["lease"] for item in transaction_results
        ),
        "budgets_retained_until_quiescence": all(
            item["budget"] for item in transaction_results
        ),
        "budgets_restored_after_release": all(
            item["budget"] for item in transaction_results
        ),
        "device_and_allocator_samples_within_bounds": device
        and all(item["memory"] for item in all_memory_results),
    }
    for name, value in derived.items():
        if record["derived_invariants"][name] is not value:
            failures.append(
                f"derived_invariants.{name} does not match raw observations"
            )
    observed_pass = all(derived.values())
    declared_pass = record["status"] == "passed"
    if record["outcome"]["result"] != record["status"]:
        failures.append("outcome.result does not match status")
    if declared_pass and record["outcome"]["failure_class"] != "none":
        failures.append("a passing record must use failure_class=none")
    if not declared_pass and record["outcome"]["failure_class"] == "none":
        failures.append("a failed record must declare a failure class")
    if declared_pass and not observed_pass:
        failures.append("a passing status conflicts with a derived invariant")
    if (
        not declared_pass
        and record["outcome"]["failure_class"] == "derived-invariant"
        and observed_pass
    ):
        failures.append("derived-invariant failure has no failed derived invariant")
    if declared_pass and record["outcome"]["failure_summary"] is not None:
        failures.append("a passing record cannot contain a failure summary")
    if not declared_pass and record["outcome"]["failure_summary"] is None:
        failures.append("a failed record requires a failure summary")
    for location, value in _walk(record):
        key = location.rsplit(".", 1)[-1]
        if SENSITIVE_KEY.search(key):
            failures.append(f"{location} uses a sensitive field name")
        if isinstance(value, str):
            decoded = unquote(value)
            if SENSITIVE_VALUE.search(decoded):
                failures.append(f"{location} matches a known secret pattern")
            if ABSOLUTE_USER_PATH.search(decoded):
                failures.append(f"{location} contains an absolute user path")
            if any(
                ord(character) < 32 and character not in "\t\n\r" for character in value
            ):
                failures.append(f"{location} contains a control character")
    return failures


def validate_record(
    record: Any,
    schema: Any,
    *,
    runtime_identity: CheckoutIdentity,
    backend_identity: CheckoutIdentity,
    qualification_manifest: Any,
    expected_qualification_manifest_path: str,
    expected_qualification_manifest_sha256: str,
    expected_patch_manifest_path: str,
    expected_patch_manifest_sha256: str,
    expected_producer_script_path: str,
    expected_producer_script_sha256: str,
    expected_schema_path: str,
    expected_schema_sha256: str,
    expected_validator_path: str,
    expected_validator_sha256: str,
    expected_image_inputs_path: str,
    expected_image_inputs_sha256: str,
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
        return [f"the native CUDA qualification schema is invalid: {error}"]
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for error in sorted(
        validator.iter_errors(record),
        key=lambda item: tuple(str(part) for part in item.path),
    ):
        field = ".".join(str(part) for part in error.absolute_path)
        location = f"record.{field}" if field else "record"
        failures.append(f"{location}: {error.message}")
    if not failures:
        failures.extend(
            validate_semantics(
                record,
                runtime_identity=runtime_identity,
                backend_identity=backend_identity,
                qualification_manifest=qualification_manifest,
                expected_qualification_manifest_path=expected_qualification_manifest_path,
                expected_qualification_manifest_sha256=expected_qualification_manifest_sha256,
                expected_patch_manifest_path=expected_patch_manifest_path,
                expected_patch_manifest_sha256=expected_patch_manifest_sha256,
                expected_producer_script_path=expected_producer_script_path,
                expected_producer_script_sha256=expected_producer_script_sha256,
                expected_schema_path=expected_schema_path,
                expected_schema_sha256=expected_schema_sha256,
                expected_validator_path=expected_validator_path,
                expected_validator_sha256=expected_validator_sha256,
                expected_image_inputs_path=expected_image_inputs_path,
                expected_image_inputs_sha256=expected_image_inputs_sha256,
            )
        )
    return failures


def _required_path(parser: argparse.ArgumentParser, value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        parser.error(f"file does not exist: {path}")
    return path


def _repository_relative_path(path: Path, repository_root: Path = ROOT) -> str:
    """Return the resolved artifact path relative to the runtime repository."""

    resolved_root = repository_root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"artifact is outside the runtime repository: {resolved_path}"
        ) from error
    if not relative.parts:
        raise ValueError(f"artifact does not identify a file: {resolved_path}")
    return relative.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--runtime-checkout", required=True, type=Path)
    parser.add_argument("--backend-checkout", required=True, type=Path)
    parser.add_argument(
        "--qualification-manifest",
        required=True,
        type=lambda value: _required_path(parser, value),
    )
    parser.add_argument(
        "--patch-manifest",
        required=True,
        type=lambda value: _required_path(parser, value),
    )
    parser.add_argument(
        "--producer-script",
        required=True,
        type=lambda value: _required_path(parser, value),
    )
    parser.add_argument(
        "--image-inputs",
        required=True,
        type=lambda value: _required_path(parser, value),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record = read_json(args.record)
        schema = read_json(args.schema)
        qualification_manifest = read_json(args.qualification_manifest)
        runtime_identity = derive_checkout_identity(args.runtime_checkout)
        backend_identity = derive_checkout_identity(args.backend_checkout)
        if runtime_identity.checkout != ROOT.resolve(strict=True):
            raise ValueError(
                "validator must run from the supplied clean runtime checkout"
            )
        qualification_manifest_path, qualification_manifest_sha256 = (
            bind_tracked_artifact(args.qualification_manifest, runtime_identity)
        )
        manifest_path, manifest_sha256 = bind_tracked_artifact(
            args.patch_manifest, runtime_identity
        )
        producer_path, producer_sha256 = bind_tracked_artifact(
            args.producer_script, runtime_identity
        )
        schema_path, schema_sha256 = bind_tracked_artifact(
            args.schema, runtime_identity
        )
        validator_path, validator_sha256 = bind_tracked_artifact(
            Path(__file__), runtime_identity
        )
        image_inputs_path, image_inputs_sha256 = bind_tracked_artifact(
            args.image_inputs, runtime_identity
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"FAIL: cannot bind qualification input: {error}")
        return 1
    failures = validate_record(
        record,
        schema,
        runtime_identity=runtime_identity,
        backend_identity=backend_identity,
        qualification_manifest=qualification_manifest,
        expected_qualification_manifest_path=qualification_manifest_path,
        expected_qualification_manifest_sha256=qualification_manifest_sha256,
        expected_patch_manifest_path=manifest_path,
        expected_patch_manifest_sha256=manifest_sha256,
        expected_producer_script_path=producer_path,
        expected_producer_script_sha256=producer_sha256,
        expected_schema_path=schema_path,
        expected_schema_sha256=schema_sha256,
        expected_validator_path=validator_path,
        expected_validator_sha256=validator_sha256,
        expected_image_inputs_path=image_inputs_path,
        expected_image_inputs_sha256=image_inputs_sha256,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Draft native CUDA qualification passed schema and semantic validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
