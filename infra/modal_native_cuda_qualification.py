"""Run the registered native Whisper CUDA qualification campaign on Modal.

Importing this module does not import Modal, PyTorch, or Whisper. Remote
resources are defined only when ``WHISPER_MODAL_ENABLE_REMOTE_RESOURCES=1`` is
set. A paid function is dispatched only after the local entry point receives
``--confirm-paid-gpu``.

The output is qualification evidence for one preregistered T4 cell. It is not
a performance result or a production-readiness claim.
"""

from __future__ import annotations

import datetime as dt
import errno
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import re
import socket
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import patch

try:
    from .native_cuda_trace import (
        FaultPlan,
        FaultPoint,
        RuntimeBindings,
        ScenarioTrace,
        TaskProxy,
        TorchProxy,
        TraceRouter,
        TracingCuda,
    )
except ImportError:  # pragma: no cover - direct ``modal run`` script loading
    from native_cuda_trace import (  # type: ignore[no-redef]
        FaultPlan,
        FaultPoint,
        RuntimeBindings,
        ScenarioTrace,
        TaskProxy,
        TorchProxy,
        TraceRouter,
        TracingCuda,
    )

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "whisper-runtime-native-cuda-qualification"
SCHEMA_VERSION = "1-draft"
ATTEMPT_RECEIPT_VERSION = "1"
MODAL_SDK_VERSION = "1.5.5"
RUNTIME_REPOSITORY = "https://github.com/billmedj/whisper-runtime.git"
BACKEND_REPOSITORY = "https://github.com/openai/whisper.git"
BACKEND_BASE_COMMIT = "86098128c0b4f24f0e2aa2994de830614b474227"
BACKEND_BASE_TREE = "f7b3cb8e12a2e84dccacc4c858c33d5a9c114688"
BACKEND_COMMIT = "a0b9695ae1cc52bad4b8626fe9fb6ea4ac0ee650"
BACKEND_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
PATCH_MANIFEST_PATH = "patches/openai-whisper/SHA256SUMS"
QUALIFICATION_MANIFEST_PATH = "experiments/native-cuda-qualification-v1.json"
PRODUCER_PATH = "infra/modal_native_cuda_qualification.py"
TRACE_PATH = "infra/native_cuda_trace.py"
IMAGE_INPUTS_PATH = "infra/modal-native-cuda-image-inputs.lock"
REGISTERED_OUTPUT_PATH = "artifacts/modal/native-cuda-qualification-v1.json"
SCHEMA_PATH = "evidence/modal-native-cuda-qualification.schema.json"
VALIDATOR_PATH = "tools/validate_modal_native_cuda_qualification.py"
MODEL_CACHE_NAME = "whisper-runtime-model-cache-v1"
MODEL_CACHE_MOUNT = "/models"
MODEL_CHECKPOINT_PATH = Path(MODEL_CACHE_MOUNT) / "tiny.en.pt"
MODEL_STATE_SHA256 = (
    "sha256:8041a80119a588f542472da35e97d0372fce1d9709ed9874475e9c03deac5de6"
)
AUDIO_PATH = Path("/opt/openai-whisper/tests/jfk.flac")
GPU_REQUEST = "T4"
DEVICE = "cuda:0"
EXPECTED_COMPUTE_CAPABILITY = (7, 5)
RUNTIME_ROOT = Path("/opt/whisper-runtime")
BACKEND_ROOT = Path("/opt/openai-whisper")
PATCH_FILES = tuple(f"{index:04d}-" for index in range(1, 8))
GIT_HASH_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
MODAL_IMAGE_ID_PATTERN = re.compile(r"im-[A-Za-z0-9_-]{8,128}\Z")

FAULT_POINT_BY_NAME = {
    "cleanup": FaultPoint.CLEANUP,
    "event-create": FaultPoint.EVENT_CREATE,
    "event-record": FaultPoint.EVENT_RECORD,
    "event-synchronize": FaultPoint.EVENT_SYNCHRONIZE,
}
FAULT_MESSAGE_BY_NAME = {
    "cleanup": "injected native cleanup failure",
    "event-create": "injected CUDA event creation failure",
    "event-record": "injected CUDA event record failure",
    "event-synchronize": "injected CUDA event synchronization failure",
}
FAULT_TRACE_COUNTS = {
    "cleanup": (3, 1),
    "event-create": (3, 3),
    "event-record": (3, 3),
    "event-synchronize": (3, 3),
}
DIRECT_IMAGE_PACKAGES = (
    "jsonschema==4.25.1",
    "more-itertools==11.1.0",
    "numba==0.67.0",
    "numpy==2.5.2",
    "setuptools==82.0.1",
    "tiktoken==0.14.0",
    "tqdm==4.70.0",
)


def _validator() -> Any:
    """Load the local validator without making it an import-time dependency."""

    return importlib.import_module("tools.validate_modal_native_cuda_qualification")


def _read_registration(
    path: Path = ROOT / QUALIFICATION_MANIFEST_PATH,
) -> dict[str, Any]:
    validator = _validator()
    manifest = validator.read_json(path)
    failures = validator.validate_qualification_manifest(manifest)
    if failures:
        raise ValueError("invalid qualification registration: " + "; ".join(failures))
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resource_vector(value: object) -> dict[str, int]:
    return {
        "memory_bytes": int(getattr(value, "memory_bytes")),
        "compute_units": int(getattr(value, "compute_units")),
        "stream_slots": int(getattr(value, "stream_slots")),
    }


def _status(value: object) -> str:
    return str(getattr(value, "value", value))


def _required_runtime_commit() -> str:
    value = os.environ.get("WHISPER_RUNTIME_COMMIT", "")
    if GIT_HASH_PATTERN.fullmatch(value) is None:
        raise RuntimeError(
            "WHISPER_RUNTIME_COMMIT must name the full public qualification commit"
        )
    return value


def _required_modal_image_id() -> str:
    candidate = os.environ.get("MODAL_IMAGE_ID", "")
    if MODAL_IMAGE_ID_PATTERN.fullmatch(candidate) is None:
        raise RuntimeError("Modal did not expose a valid image object identifier")
    return candidate


def _required_modal_location(manifest: Mapping[str, Any]) -> tuple[str, str]:
    cloud = os.environ.get("MODAL_CLOUD_PROVIDER", "")
    region = os.environ.get("MODAL_REGION", "")
    cell = manifest["cell"]
    if cloud != cell["cloud_provider"] or region != cell["region"]:
        raise RuntimeError("the observed Modal location differs from the registration")
    return cloud, region


def _require_module_origin(
    module: object,
    root: Path,
    relative_path: str,
    label: str,
) -> None:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise RuntimeError(f"{label} has no source-file origin")
    actual = Path(origin).resolve(strict=True)
    expected = (root / relative_path).resolve(strict=True)
    if actual != expected:
        raise RuntimeError(f"{label} was imported from outside the bound checkout")


def _verify_module_origins(
    whisper_module: object,
    decoding_module: object,
    runtime_module: object,
    adapters_module: object,
    native_module: object,
) -> None:
    _require_module_origin(
        importlib.import_module(__name__),
        RUNTIME_ROOT,
        PRODUCER_PATH,
        "qualification producer",
    )
    _require_module_origin(
        importlib.import_module(TraceRouter.__module__),
        RUNTIME_ROOT,
        TRACE_PATH,
        "CUDA trace module",
    )
    _require_module_origin(
        whisper_module,
        BACKEND_ROOT,
        "whisper/__init__.py",
        "Whisper package",
    )
    _require_module_origin(
        decoding_module,
        BACKEND_ROOT,
        "whisper/decoding.py",
        "Whisper decoder",
    )
    _require_module_origin(
        runtime_module,
        RUNTIME_ROOT,
        "src/whisper_runtime/__init__.py",
        "runtime package",
    )
    _require_module_origin(
        adapters_module,
        RUNTIME_ROOT,
        "src/whisper_runtime/adapters/__init__.py",
        "runtime adapters package",
    )
    _require_module_origin(
        native_module,
        RUNTIME_ROOT,
        "src/whisper_runtime/adapters/native_whisper.py",
        "native Whisper adapter",
    )


def _resolved_dependencies() -> list[dict[str, str]]:
    """Return the canonical installed Python distribution inventory."""

    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name") or distribution.name
        name = re.sub(r"[-_.]+", "-", str(raw_name)).lower()
        version = str(distribution.version)
        if not name or not version:
            raise RuntimeError("an installed distribution has no name or version")
        previous = versions.setdefault(name, version)
        if previous != version:
            raise RuntimeError(f"multiple installed versions found for {name!r}")
    return [{"name": name, "version": versions[name]} for name in sorted(versions)]


def _git_invocation(root: Path, *arguments: str) -> list[str]:
    checkout = root.resolve()
    return [
        "git",
        "-c",
        f"safe.directory={checkout.as_posix()}",
        "-C",
        str(checkout),
        *arguments,
    ]


def _require_definition_checkout(runtime_commit: str) -> None:
    validator = _validator()
    identity = validator.derive_checkout_identity(ROOT)
    if identity.git_commit != runtime_commit:
        raise RuntimeError("the local checkout does not match WHISPER_RUNTIME_COMMIT")
    for relative in (
        QUALIFICATION_MANIFEST_PATH,
        PATCH_MANIFEST_PATH,
        PRODUCER_PATH,
        TRACE_PATH,
        SCHEMA_PATH,
        VALIDATOR_PATH,
        IMAGE_INPUTS_PATH,
    ):
        validator.bind_tracked_artifact(ROOT / relative, identity)


def _campaign_order(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, int, str | None], ...]:
    """Return the closed run order as ``(kind, iteration, fault)`` tuples."""

    sampling = manifest["sampling"]
    entries: list[tuple[str, int, str | None]] = []
    for iteration in range(int(sampling["warmup_pairs"])):
        entries.extend(
            (("control-warmup", iteration, None), ("warmup", iteration, None))
        )
    for iteration in range(int(sampling["measured_pairs"])):
        entries.extend((("control", iteration, None), ("measured", iteration, None)))
    entries.extend(
        ("cancellation", iteration, None)
        for iteration in range(int(sampling["cancellation_runs"]))
    )
    for fault_name in manifest["faults"]["points"]:
        for repetition in range(int(sampling["fault_repetitions_per_point"])):
            entries.extend(
                (
                    ("fault", repetition, str(fault_name)),
                    ("reuse", repetition, str(fault_name)),
                )
            )
    return tuple(entries)


def _output_path(value: str) -> Path:
    relative = PurePosixPath(value)
    if (
        value != REGISTERED_OUTPUT_PATH
        or "\\" in value
        or ":" in value
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(
            f"--output must be the registered path {REGISTERED_OUTPUT_PATH!r}"
        )
    return Path(*relative.parts)


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _attempt_receipt_path(record_path: Path) -> Path:
    return record_path.with_suffix(".attempt.jsonl")


def _append_receipt_event(
    path: Path,
    event: Mapping[str, Any],
    *,
    create: bool,
) -> None:
    payload = json.dumps(
        event, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    if "\n" in payload or "\r" in payload:
        raise ValueError("attempt receipt events must fit on one JSON line")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if create else "a"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(slots=True)
class AttemptReceipt:
    """Append-only local evidence for one registered campaign attempt."""

    path: Path
    common: dict[str, Any]
    _next_sequence: int = 1
    _terminal: bool = False

    @classmethod
    def begin(
        cls,
        record_path: Path,
        *,
        runtime_commit: str,
        manifest: Mapping[str, Any],
        manifest_sha256: str,
    ) -> AttemptReceipt:
        policy = manifest["exclusion_rule"]
        if policy != {
            "id": "no-exclusions-v1",
            "allowed_classes": [],
            "max_attempts": 1,
            "publish_all_attempts": True,
        }:
            raise ValueError("the qualification attempt policy is not closed")
        common = {
            "receipt_version": ATTEMPT_RECEIPT_VERSION,
            "campaign_id": str(manifest["manifest_id"]),
            "attempt": 1,
            "max_attempts": 1,
            "runtime_commit": runtime_commit,
            "manifest_sha256": manifest_sha256,
            "record_path": record_path.as_posix(),
        }
        receipt = cls(_attempt_receipt_path(record_path), common)
        receipt._append("attempt-started", stage="before-dispatch", create=True)
        return receipt

    def complete(self, record_sha256: str) -> None:
        self._append(
            "record-published",
            stage="record-write",
            record_sha256=record_sha256,
        )

    def fail(self, stage: str, error: Exception) -> None:
        self._append(
            "attempt-failed",
            stage=stage,
            error_type=type(error).__name__,
            error_sha256=_sha256_text(str(error)),
        )

    def _append(
        self, event: str, *, stage: str, create: bool = False, **details: Any
    ) -> None:
        if self._terminal:
            raise RuntimeError("the qualification attempt already has a terminal event")
        sequence = 0 if create else self._next_sequence
        item = {
            **self.common,
            "sequence": sequence,
            "recorded_at": _utc_now(),
            "event": event,
            "stage": stage,
            **details,
        }
        _append_receipt_event(self.path, item, create=create)
        if create:
            self._next_sequence = 1
        else:
            self._next_sequence += 1
        if event in {"record-published", "attempt-failed"}:
            self._terminal = True


def _require_paid_confirmation(confirm_paid_gpu: bool) -> None:
    if not confirm_paid_gpu:
        raise SystemExit(
            "No cache or GPU function was dispatched. Pass --confirm-paid-gpu "
            "to allocate the registered T4 cell."
        )


def _execute_registered_attempt(
    destination: Path,
    *,
    runtime_commit: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    prime_cache: Callable[[], object] | None,
    run_campaign: Callable[[], Mapping[str, Any]],
) -> None:
    """Dispatch one campaign and retain an append-only attempt receipt."""

    if (
        destination.exists()
        or destination.with_suffix(destination.suffix + ".tmp").exists()
    ):
        raise FileExistsError(f"refusing to overwrite an existing path: {destination}")
    receipt_path = _attempt_receipt_path(destination)
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing path: {receipt_path}")
    receipt = AttemptReceipt.begin(
        destination,
        runtime_commit=runtime_commit,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    stage = "cache-prime"
    try:
        if prime_cache is not None:
            prime_cache()
        stage = "gpu-campaign"
        record = run_campaign()
        if not isinstance(record, Mapping):
            raise TypeError("the qualification worker did not return an object")
        stage = "record-write"
        _write_record(destination, record)
    except Exception as error:
        receipt.fail(stage, error)
        raise
    receipt.complete(_sha256_file(destination))


@dataclass(slots=True)
class RunContext:
    """Mutable observations for one non-overlapping transaction run."""

    run_id: str
    run_kind: str
    iteration: int
    session_id: str
    request_id: str
    transaction_id: str
    lease_id: str
    fault_point: str | None = None
    transaction: object | None = None
    lease: object | None = None
    session: object | None = None
    request: object | None = None
    budget: dict[str, dict[str, int]] = field(default_factory=dict)
    high_events: list[str] = field(default_factory=list)
    fault_ordinal: int = 0


class QualificationEventLog:
    """Record one global, monotonic, non-overlapping qualification event stream."""

    def __init__(self, worker_id: str, *, clock: Callable[[], int] = time.monotonic_ns):
        self.worker_id = worker_id
        self._clock = clock
        self.origin_ns = clock()
        self.events: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def record(
        self,
        context: RunContext,
        event: str,
        **details: object,
    ) -> dict[str, object]:
        item: dict[str, object] = {
            "sequence": 0,
            "offset_ns": 0,
            "worker_id": self.worker_id,
            "run_id": context.run_id,
            "run_kind": context.run_kind,
            "event": event,
        }
        if context.run_kind != "control":
            item.update(
                {
                    "session_id": context.session_id,
                    "request_id": context.request_id,
                    "transaction_id": context.transaction_id,
                    "lease_id": context.lease_id,
                }
            )
        item.update(details)
        with self._lock:
            offset = self._clock() - self.origin_ns
            previous = int(self.events[-1]["offset_ns"]) if self.events else -1
            while offset <= previous:
                offset = self._clock() - self.origin_ns
            item["sequence"] = len(self.events)
            item["offset_ns"] = offset
            self.events.append(item)
            context.high_events.append(event)
        return item

    def events_for(self, context: RunContext) -> list[dict[str, object]]:
        return [item for item in self.events if item["run_id"] == context.run_id]


class QualificationTrace(ScenarioTrace):
    """Translate exact low-level trace boundaries to the public event contract."""

    def __init__(self, context: RunContext, events: QualificationEventLog) -> None:
        fault_plan = (
            FaultPlan({FAULT_POINT_BY_NAME[context.fault_point]: 2})
            if context.fault_point is not None
            else FaultPlan()
        )
        super().__init__(context.run_id, fault_plan=fault_plan)
        self.context = context
        self.qualification_events = events

    def record(
        self,
        name: str,
        *,
        kind: str,
        stream: str | None = None,
        state: dict[str, object] | None = None,
    ) -> None:
        super().record(name, kind=kind, stream=stream, state=state)
        if name == "run:cancellation-rendezvous:incomplete":
            self.qualification_events.record(
                self.context, "decoder-step-incomplete", decoder_step=1
            )
        elif name.endswith(":synchronize:return") and name.startswith("cuda:event-"):
            event = (
                "completion-fence"
                if self.context.run_kind in {"warmup", "measured", "reuse"}
                else "backend-quiescent"
            )
            if event not in self.context.high_events:
                if self.context.budget.get("available_at_quiescence") is not None:
                    raise RuntimeError("backend quiescence was recorded twice")
                bindings_state = state or {}
                budget = bindings_state.get("budget_available")
                if not isinstance(budget, dict):
                    raise RuntimeError("the CUDA fence omitted the held budget")
                expected_request = (
                    "cancelled"
                    if self.context.run_kind == "cancellation"
                    else "running"
                )
                if (
                    bindings_state.get("request_status") != expected_request
                    or bindings_state.get("session_version") != 0
                    or bindings_state.get("queue_depth") != 1
                    or bindings_state.get("lease_count") != 1
                ):
                    raise RuntimeError(
                        "the completion fence lost transaction ownership"
                    )
                self.context.budget["available_at_quiescence"] = {
                    key: int(value) for key, value in budget.items()
                }
                self.qualification_events.record(self.context, event)
        elif name.endswith(":injected-failure"):
            self._record_fault(name)

    def _record_fault(self, low_level_name: str) -> None:
        fault_name = self.context.fault_point
        if fault_name is None:
            raise RuntimeError(f"unexpected injected fault: {low_level_name}")
        marker = {
            "cleanup": "run:cleanup:injected-failure",
            "event-create": ":create:injected-failure",
            "event-record": ":record:injected-failure",
            "event-synchronize": ":synchronize:injected-failure",
        }[fault_name]
        if marker not in low_level_name:
            raise RuntimeError(
                f"fault trace {low_level_name!r} does not match {fault_name!r}"
            )
        self.context.fault_ordinal += 1
        self.qualification_events.record(
            self.context,
            "fault-triggered",
            fault_point=fault_name,
            operation_ordinal=self.context.fault_ordinal,
            error_type="RuntimeError",
            error_sha256=_sha256_text(FAULT_MESSAGE_BY_NAME[fault_name]),
            backend_call_relation="after-backend-call",
        )


def _run_wall_ns(
    events: QualificationEventLog,
    context: RunContext,
    *,
    end_event: str,
) -> int:
    observed = events.events_for(context)
    starts = [item for item in observed if item["event"] == "run-start"]
    ends = [item for item in observed if item["event"] == end_event]
    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError(
            f"run {context.run_id!r} must have one run-start and one {end_event}"
        )
    return int(ends[0]["offset_ns"]) - int(starts[0]["offset_ns"])


def _event_offset(
    events: QualificationEventLog, context: RunContext, event_name: str
) -> int:
    matches = [
        int(item["offset_ns"])
        for item in events.events_for(context)
        if item["event"] == event_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"run {context.run_id!r} must record one {event_name!r} event"
        )
    return matches[0]


def _memory_begin(torch_module: object) -> dict[str, int]:
    cuda = getattr(torch_module, "cuda")
    cuda.synchronize(0)
    gc.collect()
    cuda.reset_peak_memory_stats(0)
    return {
        "baseline_allocated_bytes": int(cuda.memory_allocated(0)),
        "baseline_reserved_bytes": int(cuda.memory_reserved(0)),
    }


def _memory_end(torch_module: object, baseline: Mapping[str, int]) -> dict[str, int]:
    cuda = getattr(torch_module, "cuda")
    cuda.synchronize(0)
    gc.collect()
    allocated = int(cuda.memory_allocated(0))
    reserved = int(cuda.memory_reserved(0))
    peak_allocated = int(cuda.max_memory_allocated(0))
    peak_reserved = int(cuda.max_memory_reserved(0))
    baseline_allocated = int(baseline["baseline_allocated_bytes"])
    baseline_reserved = int(baseline["baseline_reserved_bytes"])
    return {
        "baseline_allocated_bytes": baseline_allocated,
        "final_allocated_bytes": allocated,
        "peak_allocated_bytes": peak_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "final_reserved_bytes": reserved,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_delta_bytes": max(0, peak_reserved - baseline_reserved),
    }


def _summaries(
    control_runs: list[dict[str, Any]],
    measured_runs: list[dict[str, Any]],
    cancellation_runs: list[dict[str, Any]],
    fault_runs: list[dict[str, Any]],
) -> dict[str, object]:
    summarize = _validator().summarize
    fault_names = tuple(FAULT_POINT_BY_NAME)
    return {
        "quantile_method": "nearest-rank",
        "p99_minimum_sample_count": 1000,
        "warmups_excluded": True,
        "control_wall_ns": summarize([run["wall_ns"] for run in control_runs]),
        "success_wall_ns": summarize([run["wall_ns"] for run in measured_runs]),
        "cancellation_to_quiescence_ns": summarize(
            [run["cancel_to_quiescence_ns"] for run in cancellation_runs]
        ),
        "success_peak_allocated_delta_bytes": summarize(
            [run["memory"]["peak_allocated_delta_bytes"] for run in measured_runs]
        ),
        "success_peak_reserved_delta_bytes": summarize(
            [run["memory"]["peak_reserved_delta_bytes"] for run in measured_runs]
        ),
        "control_peak_allocated_delta_bytes": summarize(
            [run["memory"]["peak_allocated_delta_bytes"] for run in control_runs]
        ),
        "control_peak_reserved_delta_bytes": summarize(
            [run["memory"]["peak_reserved_delta_bytes"] for run in control_runs]
        ),
        "fault_recovery_ns": {
            name: summarize(
                [run["recovery_ns"] for run in fault_runs if run["fault_point"] == name]
            )
            for name in fault_names
        },
        "fault_injection_to_quiescence_ns": {
            name: summarize(
                [
                    run["injection_to_quiescence_ns"]
                    for run in fault_runs
                    if run["fault_point"] == name
                ]
            )
            for name in fault_names
        },
    }


def _model_fingerprint(model: object) -> str:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("the model does not expose state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("model.state_dict() did not return a mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        for field_value in (str(name), str(value.dtype), repr(tuple(value.shape))):
            encoded = field_value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(value.numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _hook_fingerprint(model: object) -> tuple[tuple[str, str, str, int], ...]:
    records: list[tuple[str, str, str, int]] = []
    for module_name, module in model.named_modules():
        for category, attribute in (
            ("forward", "_forward_hooks"),
            ("forward_pre", "_forward_pre_hooks"),
            ("backward", "_backward_hooks"),
        ):
            records.extend(
                (module_name, category, repr(key), id(callback))
                for key, callback in getattr(module, attribute, {}).items()
            )
    return tuple(sorted(records))


def _decoded_pcm_fingerprint(audio: object) -> str:
    value = audio.copy(order="C")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _probe_blocked_network() -> None:
    try:
        connection = socket.create_connection(("1.1.1.1", 443), timeout=1.0)
    except OSError:
        return
    connection.close()
    raise RuntimeError("the qualification worker could open an outbound connection")


def _probe_read_only_model_cache() -> None:
    probe = Path(MODEL_CACHE_MOUNT) / ".qualification-write-probe"
    try:
        probe.write_bytes(b"probe")
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
            raise RuntimeError(
                "the model-cache write probe failed for an unexpected reason"
            ) from error
        return
    probe.unlink(missing_ok=True)
    raise RuntimeError("the qualification worker could write to the model cache")


def _command_first_line(*arguments: str) -> str:
    result = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or not lines or not lines[0].strip():
        raise RuntimeError(f"version command failed: {arguments[0]}")
    return lines[0].strip()


def _build_command(runtime_commit: str) -> str:
    patch_directory = "/opt/whisper-runtime/patches/openai-whisper"
    apply_commands = " && ".join(
        f"git am --committer-date-is-author-date {patch_directory}/{prefix}*.patch"
        for prefix in PATCH_FILES
    )
    return " && ".join(
        (
            "git init --quiet /opt/whisper-runtime",
            "git -C /opt/whisper-runtime remote add origin " + RUNTIME_REPOSITORY,
            "git -C /opt/whisper-runtime fetch --depth=1 origin " + runtime_commit,
            "git -C /opt/whisper-runtime checkout --detach FETCH_HEAD",
            'test "$(git -C /opt/whisper-runtime rev-parse HEAD)" = ' + runtime_commit,
            f"cd {patch_directory} && sha256sum --check SHA256SUMS",
            "git init --quiet /opt/openai-whisper",
            "git -C /opt/openai-whisper remote add origin " + BACKEND_REPOSITORY,
            "git -C /opt/openai-whisper fetch --depth=1 origin " + BACKEND_BASE_COMMIT,
            "git -C /opt/openai-whisper checkout --detach FETCH_HEAD",
            "test \"$(git -C /opt/openai-whisper rev-parse 'HEAD^{tree}')\" = "
            + BACKEND_BASE_TREE,
            "git -C /opt/openai-whisper config user.name 'Whisper Runtime Check'",
            "git -C /opt/openai-whisper config user.email "
            "'whisper-runtime-check@users.noreply.github.com'",
            f"cd /opt/openai-whisper && {apply_commands}",
            'test "$(git -C /opt/openai-whisper rev-parse HEAD)" = ' + BACKEND_COMMIT,
            "test \"$(git -C /opt/openai-whisper rev-parse 'HEAD^{tree}')\" = "
            + BACKEND_TREE,
            'test -z "$(git -C /opt/openai-whisper status --porcelain '
            '--untracked-files=all)"',
            "python -m pip install --no-deps --no-build-isolation /opt/whisper-runtime",
            "python -m pip check",
        )
    )


def _prime_model_cache() -> dict[str, object]:
    manifest = _read_registration(RUNTIME_ROOT / QUALIFICATION_MANIFEST_PATH)
    checkpoint = manifest["cell"]
    destination = MODEL_CHECKPOINT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = str(checkpoint["checkpoint_sha256"])
    if destination.is_file() and _sha256_file(destination) == expected:
        return {"checkpoint_sha256": expected, "downloaded": False}
    temporary = destination.with_suffix(".pt.partial")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(
            str(checkpoint["checkpoint_source"]), timeout=120
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"checkpoint download returned HTTP {response.status}"
                )
            with temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if _sha256_file(temporary) != expected:
            raise RuntimeError("the downloaded checkpoint digest does not match")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"checkpoint_sha256": expected, "downloaded": True}


def _bind_provenance(
    runtime_root: Path,
    backend_root: Path,
    runtime_commit: str,
) -> tuple[Any, Any, dict[str, tuple[str, str]], dict[str, Any]]:
    validator = _validator()
    runtime = validator.derive_checkout_identity(runtime_root)
    backend = validator.derive_checkout_identity(backend_root)
    if runtime.git_commit != runtime_commit:
        raise RuntimeError(
            "the worker runtime checkout differs from the requested HEAD"
        )
    if runtime.repository != "https://github.com/billmedj/whisper-runtime":
        raise RuntimeError("the worker runtime origin is not the registered repository")
    if backend.repository != "https://github.com/openai/whisper":
        raise RuntimeError("the worker backend origin is not the registered repository")
    if backend.git_commit != BACKEND_COMMIT or backend.git_tree != BACKEND_TREE:
        raise RuntimeError("the worker backend differs from the registered source")
    artifact_paths = (
        QUALIFICATION_MANIFEST_PATH,
        PATCH_MANIFEST_PATH,
        PRODUCER_PATH,
        TRACE_PATH,
        SCHEMA_PATH,
        VALIDATOR_PATH,
        IMAGE_INPUTS_PATH,
    )
    bindings = {
        relative: validator.bind_tracked_artifact(runtime_root / relative, runtime)
        for relative in artifact_paths
    }
    manifest = _read_registration(runtime_root / QUALIFICATION_MANIFEST_PATH)
    return runtime, backend, bindings, manifest


def _validate_record_in_worker(
    record: dict[str, Any],
    *,
    runtime: Any,
    backend: Any,
    bindings: Mapping[str, tuple[str, str]],
    manifest: dict[str, Any],
) -> None:
    validator = _validator()
    schema = validator.read_json(RUNTIME_ROOT / SCHEMA_PATH)
    failures = validator.validate_record(
        record,
        schema,
        runtime_identity=runtime,
        backend_identity=backend,
        qualification_manifest=manifest,
        expected_qualification_manifest_path=bindings[QUALIFICATION_MANIFEST_PATH][0],
        expected_qualification_manifest_sha256=bindings[QUALIFICATION_MANIFEST_PATH][1],
        expected_patch_manifest_path=bindings[PATCH_MANIFEST_PATH][0],
        expected_patch_manifest_sha256=bindings[PATCH_MANIFEST_PATH][1],
        expected_producer_script_path=bindings[PRODUCER_PATH][0],
        expected_producer_script_sha256=bindings[PRODUCER_PATH][1],
        expected_schema_path=bindings[SCHEMA_PATH][0],
        expected_schema_sha256=bindings[SCHEMA_PATH][1],
        expected_validator_path=bindings[VALIDATOR_PATH][0],
        expected_validator_sha256=bindings[VALIDATOR_PATH][1],
        expected_image_inputs_path=bindings[IMAGE_INPUTS_PATH][0],
        expected_image_inputs_sha256=bindings[IMAGE_INPUTS_PATH][1],
    )
    if failures:
        raise RuntimeError(
            "qualification record validation failed: " + "; ".join(failures)
        )


class _QualificationRunner:
    """Execute the closed campaign with real runtime and backend objects."""

    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        worker_id: str,
        torch_module: Any,
        whisper_module: Any,
        native_module: Any,
        runtime_module: Any,
        adapters_module: Any,
    ) -> None:
        self.manifest = manifest
        self.cell = manifest["cell"]
        self.resource_contract = manifest["resource_contract"]
        self.torch = torch_module
        self.whisper = whisper_module
        self.native_module = native_module
        self.runtime = runtime_module
        self.adapters = adapters_module
        self.events = QualificationEventLog(worker_id)
        self.router = TraceRouter()
        self._active_context: RunContext | None = None
        self._active_session: object | None = None
        self._active_request: object | None = None
        self._active_lock = threading.Lock()

        capacity = runtime_module.ResourceVector(**self.resource_contract["capacity"])
        reservation = runtime_module.ResourceVector(
            **self.resource_contract["reservation"]
        )
        if capacity != reservation:
            raise RuntimeError(
                "the single-lane adapter requires logical capacity to equal "
                "the registered per-run reservation"
            )

        runner = self

        class QualificationBudget(runtime_module.Budget):
            def acquire(self, resources: object) -> object:
                context = runner._context()
                if resources != reservation:
                    raise RuntimeError(
                        "the runtime requested an unregistered resource vector"
                    )
                lease = super().acquire(resources)
                if context.lease is not None:
                    raise RuntimeError("one run acquired more than one lease")
                context.lease = lease
                context.budget["available_while_held"] = _resource_vector(
                    self.available
                )
                runner.events.record(context, "lease-acquired")
                return lease

            def _release(self, lease: object) -> None:
                context = runner._context()
                if context.lease is not lease or context.transaction is None:
                    raise RuntimeError("the run released an unknown lease")
                if "available_at_quiescence" not in context.budget:
                    raise RuntimeError("the run released its lease before quiescence")
                transaction_status = _status(context.transaction.status)
                if context.run_kind in {"warmup", "measured", "reuse"}:
                    if transaction_status != "committed":
                        raise RuntimeError("a successful run released before commit")
                    runner.events.record(context, "transaction-committed")
                else:
                    if transaction_status != "aborted":
                        raise RuntimeError("a stopped run released before abort")
                    runner.events.record(context, "transaction-aborted")
                super()._release(lease)
                runner.events.record(context, "lease-released")
                context.budget["available_after_release"] = _resource_vector(
                    self.available
                )
                runner.events.record(context, "budget-restored")

        class QualificationWorker(runtime_module.Worker):
            def prepare(self, **kwargs: object) -> object:
                context = runner._context()
                transaction = super().prepare(**kwargs)
                if context.transaction is not None:
                    raise RuntimeError("one run created more than one transaction")
                context.transaction = transaction
                return transaction

        class QualificationSession(runtime_module.Session):
            def _commit(self, expected_version: int, record: object) -> object:
                context = runner._context()
                if context.session is not self:
                    raise RuntimeError("a different session attempted publication")
                if context.high_events.count("completion-fence") != 1:
                    raise RuntimeError(
                        "publication did not follow one completion fence"
                    )
                if runner.worker.queue_depth != 1 or runner.budget.lease_count != 1:
                    raise RuntimeError(
                        "publication occurred outside the held transaction"
                    )
                state = super()._commit(expected_version, record)
                runner.events.record(context, "result-published")
                return state

        self.Session = QualificationSession
        self.budget = QualificationBudget(capacity)
        snapshot = runtime_module.ModelSnapshot(
            model_id=str(self.cell["model"]),
            revision=BACKEND_COMMIT,
            backend="pytorch-cuda-transaction",
            fingerprint=MODEL_STATE_SHA256,
        )
        self.snapshot = snapshot
        self.worker = QualificationWorker(
            "modal-native-cuda-qualification",
            snapshot,
            self.budget,
            queue_capacity=1,
            transaction_ttl_seconds=300,
        )
        bindings = RuntimeBindings(
            worker=self.worker,
            budget=self.budget,
            session_getter=self._session,
            request_getter=self._request,
            resource_snapshot=_resource_vector,
        )
        tracing_cuda = TracingCuda(
            torch_module.cuda,
            self.router,
            bindings,
            device=str(self.cell["device"]),
        )
        self.torch_proxy = TorchProxy(torch_module, tracing_cuda)
        self.real_components = native_module._load_native_components()
        decoding = importlib.import_module("whisper.decoding")
        self.DecodingOptions = decoding.DecodingOptions
        self.DecodingTask = decoding.DecodingTask

        def task_type(model: object, options: object) -> object:
            task = self.DecodingTask(model, options)
            return TaskProxy(
                task,
                self.router,
                self.torch,
                device=str(self.cell["device"]),
            )

        self.traced_components = native_module._NativeComponents(
            generator_type=self.real_components.generator_type,
            options_type=self.real_components.options_type,
            task_type=task_type,
            n_frames=self.real_components.n_frames,
            torch_module=self.torch_proxy,
        )
        self.model: object | None = None
        self.adapter: object | None = None

    def bind_model(self, model: object) -> None:
        self.model = model

        def identity_probe(observed: object) -> object:
            if observed is not self.model:
                raise RuntimeError("the adapter received a different model object")
            if str(getattr(observed, "device", "")) != self.cell["device"]:
                raise RuntimeError("the bound model moved to another device")
            return self.snapshot

        profile = self.adapters.NativeExecutionProfile(
            str(self.cell["profile_id"]),
            self.runtime.ResourceVector(**self.resource_contract["reservation"]),
            max_concurrent_decodes=1,
            device=str(self.cell["device"]),
        )
        self.adapter = self.adapters.NativeWhisperAdapter(
            self.worker,
            model,
            identity_probe,
            profile,
        )

    def _context(self) -> RunContext:
        with self._active_lock:
            context = self._active_context
        if context is None:
            raise RuntimeError("no qualification run is active")
        return context

    def _session(self) -> object:
        with self._active_lock:
            session = self._active_session
        if session is None:
            raise RuntimeError("no qualification session is active")
        return session

    def _request(self) -> object:
        with self._active_lock:
            request = self._active_request
        if request is None:
            raise RuntimeError("no qualification request is active")
        return request

    def _bind_active(
        self,
        context: RunContext,
        session: object,
        request: object,
    ) -> None:
        with self._active_lock:
            if self._active_context is not None:
                raise RuntimeError("qualification runs must not overlap")
            self._active_context = context
            self._active_session = session
            self._active_request = request

    def _replace_active_request(self, session: object, request: object) -> None:
        with self._active_lock:
            if self._active_context is None:
                raise RuntimeError("no qualification run is active")
            self._active_session = session
            self._active_request = request

    def _clear_active(self, context: RunContext) -> None:
        with self._active_lock:
            if self._active_context is not context:
                raise RuntimeError("qualification run ownership changed")
            self._active_context = None
            self._active_session = None
            self._active_request = None

    def _context_for(
        self,
        *,
        run_id: str,
        run_kind: str,
        iteration: int,
        session_id: str | None = None,
        fault_point: str | None = None,
    ) -> RunContext:
        selected_session = session_id or f"{run_id}-session"
        return RunContext(
            run_id=run_id,
            run_kind=run_kind,
            iteration=iteration,
            session_id=selected_session,
            request_id=f"{run_id}-request",
            transaction_id=f"{run_id}-transaction",
            lease_id=f"{run_id}-lease",
            fault_point=fault_point,
        )

    def _native_options(self) -> object:
        options = self.cell["decode_options"]
        return self.adapters.NativeDecodeOptions(
            language=str(self.cell["language"]),
            task=str(self.cell["task"]),
            temperature=float(options["temperature"]),
            beam_size=options["beam_size"],
            best_of=options["best_of"],
            patience=options["patience"],
            length_penalty=options["length_penalty"],
            sample_len=options["sample_len"],
            without_timestamps=bool(options["without_timestamps"]),
        )

    def _control_options(self) -> object:
        options = self.cell["decode_options"]
        source = random.Random(int(self.cell["seed"]))
        native_seed = source.randrange(1 << 63)
        generator = self.torch.Generator(device=self.cell["device"])
        generator.manual_seed(native_seed)
        return self.DecodingOptions(
            language=str(self.cell["language"]),
            task=str(self.cell["task"]),
            temperature=float(options["temperature"]),
            beam_size=options["beam_size"],
            best_of=options["best_of"],
            patience=options["patience"],
            length_penalty=options["length_penalty"],
            sample_len=options["sample_len"],
            without_timestamps=bool(options["without_timestamps"]),
            fp16=bool(options["fp16"]),
            generator=generator,
        )

    def _assert_events(self, context: RunContext, expected: tuple[str, ...]) -> None:
        observed = tuple(str(item["event"]) for item in self.events.events_for(context))
        if observed != expected:
            raise RuntimeError(
                f"run {context.run_id!r} emitted {observed!r}, expected {expected!r}"
            )

    def run_control(
        self, mel: object, *, iteration: int, warmup: bool
    ) -> dict[str, Any]:
        prefix = "control-warmup" if warmup else "control"
        context = self._context_for(
            run_id=f"{prefix}-{iteration}",
            run_kind="control",
            iteration=iteration,
        )
        baseline = _memory_begin(self.torch)
        self.events.record(context, "run-start")
        task = self.DecodingTask(self.model, self._control_options())
        batched_mel = mel.unsqueeze(0).to(
            device=self.cell["device"], non_blocking=False
        )
        results = task.run(batched_mel)
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError("the control decoder did not return one result")
        self.torch.cuda.synchronize(0)
        self.events.record(context, "backend-quiescent")
        memory = _memory_end(self.torch, baseline)
        self.events.record(context, "run-complete")
        result_sha256 = _sha256_text(str(getattr(results[0], "text", "")))
        if result_sha256 != self.cell["expected_result_sha256"]:
            raise RuntimeError("the control transcript differs from the registration")
        self._assert_events(context, ("run-start", "backend-quiescent", "run-complete"))
        return {
            "run_id": context.run_id,
            "iteration": iteration,
            "wall_ns": _run_wall_ns(
                self.events, context, end_event="backend-quiescent"
            ),
            "result_sha256": result_sha256,
            "memory": memory,
        }

    def run_success(
        self,
        mel: object,
        *,
        run_kind: str,
        iteration: int,
        run_id: str,
        session: object | None = None,
    ) -> tuple[dict[str, Any], object]:
        if self.adapter is None:
            raise RuntimeError("the native adapter is not bound")
        context = self._context_for(
            run_id=run_id,
            run_kind=run_kind,
            iteration=iteration,
            session_id=(str(session.session_id) if session is not None else None),
        )
        active_session = session or self.Session(context.session_id)
        request = self.runtime.RequestState(
            context.request_id,
            context.session_id,
            self.snapshot,
            rng_seed=int(self.cell["seed"]),
        )
        context.session = active_session
        context.request = request
        context.budget["available_before"] = _resource_vector(self.budget.available)
        baseline = _memory_begin(self.torch)
        self._bind_active(context, active_session, request)
        self.events.record(context, "run-start")
        trace = QualificationTrace(context, self.events)
        trace.set_decode_thread()
        try:
            with (
                self.router.activate(trace),
                patch.object(
                    self.native_module,
                    "_load_native_components",
                    return_value=self.traced_components,
                ),
            ):
                state = self.adapter.decode_window(
                    session=active_session,
                    request=request,
                    window_id=f"{run_id}-window",
                    mel=mel,
                    start_ms=0,
                    end_ms=int(self.cell["input_duration_ns"]) // 1_000_000,
                    options=self._native_options(),
                )
        finally:
            self._clear_active(context)
        memory = _memory_end(self.torch, baseline)
        self.events.record(context, "run-complete")
        expected_events = (
            "run-start",
            "lease-acquired",
            "completion-fence",
            "result-published",
            "transaction-committed",
            "lease-released",
            "budget-restored",
            "run-complete",
        )
        self._assert_events(context, expected_events)
        if (
            _status(request.status) != "committed"
            or context.transaction is None
            or _status(context.transaction.status) != "committed"
            or self.worker.queue_depth != 0
            or self.budget.lease_count != 0
            or trace.cleanup_calls != 1
            or trace.event_count != 1
            or len(trace.stream_labels) != 1
            or trace.child_generator_checks != 1
            or not trace.child_generators_on_device
            or not trace.child_generators_distinct
        ):
            raise RuntimeError(
                "the successful runtime run failed its transaction checks"
            )
        result = state.windows[-1].result
        result_sha256 = _sha256_text(result.text)
        if result_sha256 != self.cell["expected_result_sha256"]:
            raise RuntimeError("the runtime transcript differs from the registration")
        return (
            {
                "run_id": context.run_id,
                "session_id": context.session_id,
                "request_id": context.request_id,
                "transaction_id": context.transaction_id,
                "lease_id": context.lease_id,
                "iteration": iteration,
                "wall_ns": _run_wall_ns(
                    self.events, context, end_event="budget-restored"
                ),
                "session_version_before": state.version - 1,
                "session_version_after": state.version,
                "result_sha256": result_sha256,
                "memory": memory,
                "budget": context.budget,
            },
            active_session,
        )

    def run_cancellation(self, mel: object, *, iteration: int) -> dict[str, Any]:
        if self.adapter is None:
            raise RuntimeError("the native adapter is not bound")
        run_id = f"cancellation-{iteration}"
        context = self._context_for(
            run_id=run_id,
            run_kind="cancellation",
            iteration=iteration,
        )
        session = self.Session(context.session_id)
        request = self.runtime.RequestState(
            context.request_id,
            context.session_id,
            self.snapshot,
            rng_seed=int(self.cell["seed"]),
        )
        context.session = session
        context.request = request
        context.budget["available_before"] = _resource_vector(self.budget.available)
        baseline = _memory_begin(self.torch)
        self._bind_active(context, session, request)
        self.events.record(context, "run-start")
        trace = QualificationTrace(context, self.events)
        trace.cancel_after_step = True
        errors: list[BaseException] = []

        def decode() -> None:
            trace.set_decode_thread()
            try:
                self.adapter.decode_window(
                    session=session,
                    request=request,
                    window_id=f"{run_id}-window",
                    mel=mel,
                    start_ms=0,
                    end_ms=int(self.cell["input_duration_ns"]) // 1_000_000,
                    options=self._native_options(),
                )
            except BaseException as error:
                errors.append(error)

        thread: threading.Thread | None = None
        try:
            with (
                self.router.activate(trace) as activation,
                patch.object(
                    self.native_module,
                    "_load_native_components",
                    return_value=self.traced_components,
                ),
            ):
                thread = threading.Thread(
                    target=activation.wrap(decode),
                    name=f"qualification-decode-{iteration}",
                )
                thread.start()
                if not trace.step_submitted.wait(timeout=120):
                    raise RuntimeError(
                        "the decode did not reach one incomplete token step"
                    )
                if (
                    trace.cancel_step_returned is not False
                    or trace.cancel_run_complete_after_step is not False
                ):
                    raise RuntimeError("the cancellation rendezvous was not incomplete")
                if self.worker.queue_depth != 1 or self.budget.lease_count != 1:
                    raise RuntimeError("cancellation occurred without the held lease")
                self.events.record(context, "cancel-requested")
                if request.cancel() is not True:
                    raise RuntimeError(
                        "the admitted request did not accept cancellation"
                    )
                trace.release_step.set()
                thread.join(timeout=120)
                if thread.is_alive():
                    raise RuntimeError("the cancelled decode did not terminate")
        finally:
            trace.release_step.set()
            if thread is not None:
                thread.join(timeout=5)
            self._clear_active(context)
        if len(errors) != 1 or not isinstance(
            errors[0], self.runtime.RequestCancelledError
        ):
            observed = type(errors[0]).__name__ if errors else "no exception"
            raise RuntimeError(f"unexpected cancellation outcome: {observed}")
        memory = _memory_end(self.torch, baseline)
        self.events.record(context, "run-complete")
        expected_events = (
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
        self._assert_events(context, expected_events)
        controller_cuda_calls = sum(
            event["thread"] == "controller" and event["kind"] == "cuda"
            for event in trace.events
        )
        if (
            context.transaction is None
            or _status(context.transaction.status) != "aborted"
            or _status(request.status) != "cancelled"
            or session.snapshot().version != 0
            or self.worker.queue_depth != 0
            or self.budget.lease_count != 0
            or trace.cleanup_calls != 1
            or trace.event_count != 1
            or len(trace.stream_labels) != 1
            or trace.child_generator_checks != 1
            or not trace.child_generators_on_device
            or not trace.child_generators_distinct
            or controller_cuda_calls != 0
        ):
            raise RuntimeError("the cancellation run failed its transaction checks")
        cancel_offset = _event_offset(self.events, context, "cancel-requested")
        quiescent_offset = _event_offset(self.events, context, "backend-quiescent")
        return {
            "run_id": context.run_id,
            "session_id": context.session_id,
            "request_id": context.request_id,
            "transaction_id": context.transaction_id,
            "lease_id": context.lease_id,
            "iteration": iteration,
            "wall_ns": _run_wall_ns(self.events, context, end_event="budget-restored"),
            "cancel_to_quiescence_ns": quiescent_offset - cancel_offset,
            "session_version_before": 0,
            "session_version_after": session.snapshot().version,
            "memory": memory,
            "budget": context.budget,
        }

    def run_fault(
        self,
        mel: object,
        *,
        fault_name: str,
        repetition: int,
        fault_index: int,
    ) -> dict[str, Any]:
        if self.adapter is None:
            raise RuntimeError("the native adapter is not bound")
        run_id = f"fault-{fault_name}-{repetition}"
        context = self._context_for(
            run_id=run_id,
            run_kind="fault",
            iteration=repetition,
            fault_point=fault_name,
        )
        session = self.Session(context.session_id)
        request = self.runtime.RequestState(
            context.request_id,
            context.session_id,
            self.snapshot,
            rng_seed=int(self.cell["seed"]),
        )
        context.session = session
        context.request = request
        context.budget["available_before"] = _resource_vector(self.budget.available)
        blocked_id = f"blocked-{fault_name}-{repetition}"
        baseline = _memory_begin(self.torch)
        self._bind_active(context, session, request)
        self.events.record(context, "run-start")
        self.events.record(
            context,
            "fault-armed",
            fault_point=fault_name,
            operation_ordinal=1,
            planned_injection_count=2,
        )
        trace = QualificationTrace(context, self.events)
        trace.set_decode_thread()
        retained: BaseException | None = None
        try:
            with (
                self.router.activate(trace),
                patch.object(
                    self.native_module,
                    "_load_native_components",
                    return_value=self.traced_components,
                ),
            ):
                try:
                    self.adapter.decode_window(
                        session=session,
                        request=request,
                        window_id=f"{run_id}-window",
                        mel=mel,
                        start_ms=0,
                        end_ms=int(self.cell["input_duration_ns"]) // 1_000_000,
                        options=self._native_options(),
                    )
                except self.runtime.TransactionRetainedError as error:
                    retained = error
                if retained is None:
                    raise RuntimeError(
                        "the injected faults did not retain the transaction"
                    )
                transaction = retained.transaction
                if (
                    context.transaction is not transaction
                    or _status(transaction.status) != "quarantined"
                    or context.fault_ordinal != 2
                    or trace.fault_plan.observed(FAULT_POINT_BY_NAME[fault_name]) != 2
                    or trace.fault_plan.remaining(FAULT_POINT_BY_NAME[fault_name]) != 0
                    or self.worker.queue_depth != 1
                    or self.budget.lease_count != 1
                    or session.snapshot().version != 0
                ):
                    raise RuntimeError(
                        "the injected fault did not retain exact ownership"
                    )
                self.events.record(context, "transaction-retained")

                blocked_session = self.Session(f"{blocked_id}-session")
                blocked_request = self.runtime.RequestState(
                    blocked_id,
                    blocked_session.session_id,
                    self.snapshot,
                    rng_seed=int(self.cell["seed"]),
                )
                self._replace_active_request(blocked_session, blocked_request)
                try:
                    self.adapter.decode_window(
                        session=blocked_session,
                        request=blocked_request,
                        window_id=f"{blocked_id}-window",
                        mel=mel,
                        start_ms=0,
                        end_ms=int(self.cell["input_duration_ns"]) // 1_000_000,
                        options=self._native_options(),
                    )
                except self.runtime.TransactionRetainedError as blocked_error:
                    if blocked_error is not retained:
                        raise RuntimeError(
                            "the blocked request received a different retained error"
                        ) from blocked_error
                else:
                    raise RuntimeError("new work entered a retained model binding")
                if (
                    _status(blocked_request.status) != "created"
                    or blocked_session.snapshot().version != 0
                    or self.worker.queue_depth != 1
                    or self.budget.lease_count != 1
                ):
                    raise RuntimeError("the blocked request changed retained state")
                self.events.record(
                    context,
                    "new-work-rejected",
                    blocked_request_id=blocked_id,
                )
                self._replace_active_request(session, request)
                self.events.record(context, "recovery-started")
                if self.worker.recover(transaction) is not True:
                    raise RuntimeError("manual recovery did not close the transaction")
        finally:
            self._clear_active(context)
        memory = _memory_end(self.torch, baseline)
        self.events.record(context, "run-complete")
        expected_events = (
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
        self._assert_events(context, expected_events)
        expected_cleanup_count, expected_event_count = FAULT_TRACE_COUNTS[fault_name]
        if (
            context.transaction is None
            or _status(context.transaction.status) != "aborted"
            or _status(request.status) != "aborted"
            or session.snapshot().version != 0
            or self.worker.queue_depth != 0
            or self.budget.lease_count != 0
            or trace.cleanup_calls != expected_cleanup_count
            or trace.event_count != expected_event_count
            or len(trace.stream_labels) != 1
            or trace.child_generator_checks != 1
            or not trace.child_generators_on_device
            or not trace.child_generators_distinct
        ):
            raise RuntimeError("the fault run failed its recovery checks")
        first_trigger = next(
            int(item["offset_ns"])
            for item in self.events.events_for(context)
            if item["event"] == "fault-triggered"
        )
        quiescent = _event_offset(self.events, context, "backend-quiescent")
        recovery_started = _event_offset(self.events, context, "recovery-started")
        session_version_after_fault = session.snapshot().version
        reuse, _ = self.run_success(
            mel,
            run_kind="reuse",
            iteration=fault_index,
            run_id=f"reuse-{fault_name}-{repetition}",
            session=session,
        )
        return {
            "run_id": context.run_id,
            "session_id": context.session_id,
            "request_id": context.request_id,
            "transaction_id": context.transaction_id,
            "lease_id": context.lease_id,
            "fault_point": fault_name,
            "repetition": repetition,
            "fault_origin": "harness-injected",
            "planned_injection_count": 2,
            "backend_call_relation": "after-backend-call",
            "blocked_request_id": blocked_id,
            "wall_ns": _run_wall_ns(self.events, context, end_event="budget-restored"),
            "injection_to_quiescence_ns": quiescent - first_trigger,
            "recovery_ns": quiescent - recovery_started,
            "session_version_before": 0,
            "session_version_after": session_version_after_fault,
            "memory": memory,
            "budget": context.budget,
            "post_recovery_reuse": reuse,
        }


def _run_qualification_worker(
    runtime_commit: str,
    *,
    modal_module: Any,
) -> dict[str, Any]:
    """Execute the registered campaign in one isolated GPU worker."""

    torch = importlib.import_module("torch")
    whisper = importlib.import_module("whisper")
    runtime = importlib.import_module("whisper_runtime")
    adapters = importlib.import_module("whisper_runtime.adapters")
    native_module = importlib.import_module("whisper_runtime.adapters.native_whisper")
    decoding_module = importlib.import_module("whisper.decoding")
    runtime_identity, backend_identity, bindings, manifest = _bind_provenance(
        RUNTIME_ROOT,
        BACKEND_ROOT,
        runtime_commit,
    )
    _verify_module_origins(
        whisper,
        decoding_module,
        runtime,
        adapters,
        native_module,
    )
    observed_cloud, observed_region = _required_modal_location(manifest)
    _probe_blocked_network()
    _probe_read_only_model_cache()
    cell = manifest["cell"]
    if _campaign_order(manifest) != tuple(
        [
            *(
                entry
                for index in range(2)
                for entry in (("control-warmup", index, None), ("warmup", index, None))
            ),
            *(
                entry
                for index in range(5)
                for entry in (("control", index, None), ("measured", index, None))
            ),
            *(("cancellation", index, None) for index in range(3)),
            *(
                entry
                for fault_name in FAULT_POINT_BY_NAME
                for repetition in range(2)
                for entry in (
                    ("fault", repetition, fault_name),
                    ("reuse", repetition, fault_name),
                )
            ),
        ]
    ):
        raise RuntimeError(
            "the registration does not describe the version-one campaign"
        )
    if (
        _sha256_file(RUNTIME_ROOT / PATCH_MANIFEST_PATH)
        != bindings[PATCH_MANIFEST_PATH][1]
    ):
        raise RuntimeError("the patch manifest changed after provenance binding")
    if _sha256_file(MODEL_CHECKPOINT_PATH) != cell["checkpoint_sha256"]:
        raise RuntimeError("the cached checkpoint differs from the registration")
    if _sha256_file(AUDIO_PATH) != cell["input_sha256"]:
        raise RuntimeError("the audio fixture differs from the registration")
    if AUDIO_PATH.stat().st_size != cell["input_bytes"]:
        raise RuntimeError("the audio fixture size differs from the registration")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the registered cell requires one visible CUDA device")
    device_index = int(torch.cuda.current_device())
    if device_index != 0:
        raise RuntimeError("the registered CUDA device is not index zero")
    properties = torch.cuda.get_device_properties(device_index)
    device_name = str(torch.cuda.get_device_name(device_index))
    capability = tuple(
        int(value) for value in torch.cuda.get_device_capability(device_index)
    )
    if device_name != cell["gpu_name"] or capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError("the allocated GPU differs from the registered T4 cell")

    model = whisper.load_model(
        str(cell["model"]),
        device=str(cell["device"]),
        download_root=MODEL_CACHE_MOUNT,
    ).eval()
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    model_tensors = parameters + buffers
    floating_tensors = tuple(
        tensor for tensor in model_tensors if tensor.is_floating_point()
    )
    if (
        model.training is not False
        or not model_tensors
        or any(str(tensor.device) != cell["device"] for tensor in model_tensors)
        or not floating_tensors
        or any(tensor.dtype != torch.float32 for tensor in floating_tensors)
    ):
        raise RuntimeError("the loaded model differs from the registered FP32 profile")
    torch.cuda.synchronize(0)
    state_before = _model_fingerprint(model)
    hooks_before = _hook_fingerprint(model)
    if state_before != MODEL_STATE_SHA256:
        raise RuntimeError("the loaded model state differs from the pinned checkpoint")

    audio_module = importlib.import_module("whisper.audio")
    audio = whisper.load_audio(str(AUDIO_PATH))
    if (
        int(audio_module.SAMPLE_RATE) != cell["sample_rate_hz"]
        or len(audio)
        != cell["sample_rate_hz"] * cell["input_duration_ns"] // 1_000_000_000
        or _decoded_pcm_fingerprint(audio) != cell["decoded_pcm_sha256"]
    ):
        raise RuntimeError("the decoded audio differs from the registration")
    padded = whisper.pad_or_trim(
        audio,
        length=int(cell["preprocessing_options"]["pad_or_trim_samples"]),
    )
    mel = whisper.log_mel_spectrogram(
        padded,
        n_mels=int(cell["preprocessing_options"]["mel_bins"]),
    ).contiguous()
    if str(mel.device) != "cpu" or mel.dtype != torch.float32:
        raise RuntimeError("the registered adapter boundary requires CPU float32 mel")

    function_call_id = modal_module.current_function_call_id()
    if not isinstance(function_call_id, str) or not function_call_id:
        raise RuntimeError("Modal did not expose a function-call identifier")
    worker_id = _sha256_text(function_call_id)
    runner = _QualificationRunner(
        manifest=manifest,
        worker_id=worker_id,
        torch_module=torch,
        whisper_module=whisper,
        native_module=native_module,
        runtime_module=runtime,
        adapters_module=adapters,
    )
    runner.bind_model(model)

    warmup_runs: list[dict[str, Any]] = []
    control_warmup_runs: list[dict[str, Any]] = []
    for iteration in range(int(manifest["sampling"]["warmup_pairs"])):
        control_warmup_runs.append(
            runner.run_control(mel, iteration=iteration, warmup=True)
        )
        warmup, _ = runner.run_success(
            mel,
            run_kind="warmup",
            iteration=iteration,
            run_id=f"warmup-{iteration}",
        )
        warmup_runs.append(warmup)

    measured_runs: list[dict[str, Any]] = []
    control_runs: list[dict[str, Any]] = []
    for iteration in range(int(manifest["sampling"]["measured_pairs"])):
        control_runs.append(runner.run_control(mel, iteration=iteration, warmup=False))
        measured, _ = runner.run_success(
            mel,
            run_kind="measured",
            iteration=iteration,
            run_id=f"measured-{iteration}",
        )
        measured_runs.append(measured)

    cancellation_runs = [
        runner.run_cancellation(mel, iteration=iteration)
        for iteration in range(int(manifest["sampling"]["cancellation_runs"]))
    ]
    fault_runs: list[dict[str, Any]] = []
    fault_index = 0
    for fault_name in manifest["faults"]["points"]:
        for repetition in range(
            int(manifest["sampling"]["fault_repetitions_per_point"])
        ):
            fault_runs.append(
                runner.run_fault(
                    mel,
                    fault_name=str(fault_name),
                    repetition=repetition,
                    fault_index=fault_index,
                )
            )
            fault_index += 1

    torch.cuda.synchronize(0)
    if _model_fingerprint(model) != state_before:
        raise RuntimeError("the campaign changed persistent model state")
    if _hook_fingerprint(model) != hooks_before:
        raise RuntimeError("the campaign changed persistent model hooks")
    restored = native_module._load_native_components()
    if (
        restored.torch_module is not torch
        or restored.task_type is not runner.DecodingTask
        or restored.generator_type is not torch.Generator
    ):
        raise RuntimeError("the campaign did not restore native backend components")
    if runner.worker.queue_depth != 0 or runner.budget.lease_count != 0:
        raise RuntimeError("the campaign ended with retained runtime capacity")

    dependencies = _resolved_dependencies()
    canonical_sha256 = _validator().canonical_sha256
    resource = manifest["resource_contract"]
    workload = {
        "profile_id": cell["profile_id"],
        "model": cell["model"],
        "checkpoint_source": cell["checkpoint_source"],
        "checkpoint_sha256": cell["checkpoint_sha256"],
        "fixture_id": cell["fixture_id"],
        "input_manifest_source": cell["input_manifest_source"],
        "input_manifest_sha256": cell["input_manifest_sha256"],
        "input_source": cell["input_source"],
        "input_sha256": cell["input_sha256"],
        "decoded_pcm_sha256": cell["decoded_pcm_sha256"],
        "preprocessing_options": cell["preprocessing_options"],
        "preprocessing_options_sha256": cell["preprocessing_options_sha256"],
        "input_bytes": cell["input_bytes"],
        "input_duration_ns": cell["input_duration_ns"],
        "sample_rate_hz": cell["sample_rate_hz"],
        "channel_count": cell["channel_count"],
        "language": cell["language"],
        "task": cell["task"],
        "numeric_precision": cell["numeric_precision"],
        "compatibility_rule": cell["compatibility_rule"],
        "result_digest_encoding": cell["result_digest_encoding"],
        "expected_result_sha256": cell["expected_result_sha256"],
        "decode_options": cell["decode_options"],
        "decode_options_sha256": cell["decode_options_sha256"],
        "seed": cell["seed"],
        "seed_reset_each_run": cell["seed_reset_each_run"],
        "device": cell["device"],
        "execution_mode": cell["execution_mode"],
        "pair_order": cell["pair_order"],
        "comparison_backend_mode": "native-unproxied",
        "fault_backend_mode": "scoped-harness-injector",
        "network_access_during_measured_work": False,
        "model_loaded_before_timing": True,
        "peak_stats_reset_each_run": True,
        "runtime_measurement_start": manifest["measurement_boundaries"][
            "runtime_start"
        ],
        "runtime_measurement_end": manifest["measurement_boundaries"]["runtime_end"],
        "control_measurement_start": manifest["measurement_boundaries"][
            "control_start"
        ],
        "control_measurement_end": manifest["measurement_boundaries"]["control_end"],
        "warmup_iterations": manifest["sampling"]["warmup_pairs"],
        "control_iterations": manifest["sampling"]["measured_pairs"],
        "measured_iterations": manifest["sampling"]["measured_pairs"],
        "cancellation_iterations": manifest["sampling"]["cancellation_runs"],
        "fault_repetitions_per_point": manifest["sampling"][
            "fault_repetitions_per_point"
        ],
        "resource_capacity": resource["capacity"],
        "resource_reservation": resource["reservation"],
        "allocation_tolerance_bytes": resource["allocation_tolerance_bytes"],
        "reserved_tolerance_bytes": resource["reserved_tolerance_bytes"],
    }
    clock_resolution_ns = max(
        1,
        round(time.get_clock_info("monotonic").resolution * 1_000_000_000),
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": _utc_now(),
        "status": "passed",
        "outcome": {
            "registered_cell_id": cell["registered_cell_id"],
            "exclusion_rule_id": manifest["exclusion_rule"]["id"],
            "result": "passed",
            "failure_class": "none",
            "failure_summary": None,
        },
        "qualification_registration": {
            "manifest_id": manifest["manifest_id"],
            "manifest_path": bindings[QUALIFICATION_MANIFEST_PATH][0],
            "manifest_sha256": bindings[QUALIFICATION_MANIFEST_PATH][1],
            "runtime_commit": runtime_identity.git_commit,
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
            "repository": runtime_identity.repository,
            "git_commit": runtime_identity.git_commit,
            "git_tree": runtime_identity.git_tree,
            "clean": True,
        },
        "backend": {
            "repository": backend_identity.repository,
            "git_commit": backend_identity.git_commit,
            "git_tree": backend_identity.git_tree,
            "clean": True,
            "patch_manifest_path": bindings[PATCH_MANIFEST_PATH][0],
            "patch_manifest_sha256": bindings[PATCH_MANIFEST_PATH][1],
        },
        "producer": {
            "repository": runtime_identity.repository,
            "git_commit": runtime_identity.git_commit,
            "git_tree": runtime_identity.git_tree,
            "clean": True,
            "script_path": bindings[PRODUCER_PATH][0],
            "script_sha256": bindings[PRODUCER_PATH][1],
            "schema_path": bindings[SCHEMA_PATH][0],
            "schema_sha256": bindings[SCHEMA_PATH][1],
            "validator_path": bindings[VALIDATOR_PATH][0],
            "validator_sha256": bindings[VALIDATOR_PATH][1],
            "image_inputs_path": bindings[IMAGE_INPUTS_PATH][0],
            "image_inputs_sha256": bindings[IMAGE_INPUTS_PATH][1],
            "resolved_dependencies": dependencies,
            "resolved_dependencies_sha256": canonical_sha256(dependencies),
            "container_image_id": _required_modal_image_id(),
        },
        "worker": {
            "campaign_id": "native-cuda-qualification-v1",
            "worker_id": worker_id,
            "worker_ordinal": 0,
            "expected_worker_count": 1,
            "single_use": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
            "driver": _command_first_line(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ),
            "modal_sdk": str(modal_module.__version__),
        },
        "gpu": {
            "cloud_provider": observed_cloud,
            "region": observed_region,
            "visible_device_count": int(torch.cuda.device_count()),
            "device_index": device_index,
            "name": device_name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "total_memory_bytes": int(properties.total_memory),
        },
        "clock": {
            "source": "time.monotonic_ns",
            "unit": "nanosecond",
            "origin": "worker-qualification-start",
            "resolution_ns": clock_resolution_ns,
        },
        "workload": workload,
        "warmup_runs": warmup_runs,
        "control_warmup_runs": control_warmup_runs,
        "control_runs": control_runs,
        "measured_runs": measured_runs,
        "cancellation_runs": cancellation_runs,
        "fault_runs": fault_runs,
        "events": runner.events.events,
        "summaries": _summaries(
            control_runs,
            measured_runs,
            cancellation_runs,
            fault_runs,
        ),
        "derived_invariants": {name: True for name in manifest["required_invariants"]},
    }
    _validate_record_in_worker(
        record,
        runtime=runtime_identity,
        backend=backend_identity,
        bindings=bindings,
        manifest=manifest,
    )
    return record


def _locked_image_inputs() -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in (ROOT / IMAGE_INPUTS_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    expected = tuple(sorted((*DIRECT_IMAGE_PACKAGES, "modal==1.5.5", "torch==2.6.0")))
    if values != expected:
        raise RuntimeError("the direct image inputs do not match the producer")
    return values


def _definition_enabled() -> bool:
    return os.environ.get("WHISPER_MODAL_ENABLE_REMOTE_RESOURCES") == "1"


def _run_bound_worker(runtime_commit: str, *, modal_module: Any) -> dict[str, Any]:
    producer = importlib.import_module("infra.modal_native_cuda_qualification")
    return producer._run_qualification_worker(  # type: ignore[attr-defined]
        runtime_commit,
        modal_module=modal_module,
    )


def _define_modal_resources() -> tuple[Any, Any, Any, Any]:
    modal = importlib.import_module("modal")
    runtime_commit = _required_runtime_commit()
    if os.environ.get("MODAL_IS_REMOTE") != "1":
        _require_definition_checkout(runtime_commit)
    _locked_image_inputs()
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("ca-certificates", "ffmpeg", "git")
        .pip_install(
            "torch==2.6.0",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .uv_pip_install(*DIRECT_IMAGE_PACKAGES)
        .run_commands(_build_command(runtime_commit))
        .env(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": (
                    "/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime"
                ),
                "PYTHONUTF8": "1",
                "WHISPER_RUNTIME_COMMIT": runtime_commit,
                "WHISPER_MODAL_ENABLE_REMOTE_RESOURCES": "0",
            }
        )
    )
    model_cache = modal.Volume.from_name(
        MODEL_CACHE_NAME,
        create_if_missing=True,
    )
    app = modal.App(APP_NAME)

    @app.function(
        image=image,
        serialized=True,
        volumes={MODEL_CACHE_MOUNT: model_cache},
        cpu=2.0,
        memory=4096,
        timeout=600,
        startup_timeout=600,
        retries=0,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
    )
    def prime_model_cache() -> dict[str, object]:
        result = _prime_model_cache()
        if result["downloaded"] is True:
            model_cache.commit()
        return result

    @app.function(
        image=image,
        serialized=True,
        gpu=GPU_REQUEST,
        cloud="aws",
        region="us-west-2",
        volumes={MODEL_CACHE_MOUNT: model_cache.with_mount_options(read_only=True)},
        cpu=2.0,
        memory=4096,
        timeout=3600,
        startup_timeout=900,
        retries=0,
        max_containers=1,
        block_network=True,
        restrict_modal_access=True,
        single_use_containers=True,
        include_source=False,
    )
    def run_native_cuda_qualification() -> dict[str, Any]:
        observed_sdk = str(modal.__version__)
        if observed_sdk != MODAL_SDK_VERSION:
            raise RuntimeError(
                f"Modal SDK mismatch: expected {MODAL_SDK_VERSION}, "
                f"observed {observed_sdk}"
            )
        return _run_bound_worker(runtime_commit, modal_module=modal)

    @app.local_entrypoint()
    def main(
        output: str = "",
        skip_cache_prime: bool = False,
        confirm_paid_gpu: bool = False,
    ) -> None:
        _require_paid_confirmation(confirm_paid_gpu)
        destination = _output_path(output)
        manifest = _read_registration()
        _execute_registered_attempt(
            destination,
            runtime_commit=runtime_commit,
            manifest=manifest,
            manifest_sha256=_sha256_file(ROOT / QUALIFICATION_MANIFEST_PATH),
            prime_cache=(None if skip_cache_prime else prime_model_cache.remote),
            run_campaign=run_native_cuda_qualification.remote,
        )
        print(f"Wrote validated qualification evidence to {destination}")

    return app, prime_model_cache, run_native_cuda_qualification, main


if _definition_enabled():
    app, prime_model_cache, run_native_cuda_qualification, main = (
        _define_modal_resources()
    )
else:
    app = None
    prime_model_cache = None
    run_native_cuda_qualification = None
    main = None
