"""Run the bounded native CUDA transaction check on one Modal T4.

This check exercises ``NativeWhisperAdapter`` with the pinned patched Whisper
backend. It records one successful commit, one cooperative cancellation, and
one injected completion-fence failure followed by exact recovery. A final
unproxied control transaction checks adapter reuse after recovery. See
``docs/MODAL_NATIVE_CUDA_VALIDATION.md`` before starting a paid run.
"""

from __future__ import annotations

import datetime as dt
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import TypeVar
from unittest.mock import patch

import modal

from infra.modal_gpu_validation import (
    AUDIO_FIXTURE_ID,
    AUDIO_PATH,
    AUDIO_RELATIVE_PATH,
    AUDIO_SAMPLE_COUNT,
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_SHA256,
    AUDIO_SIZE_BYTES,
    BACKEND_BASE_COMMIT,
    BACKEND_BASE_TREE,
    BACKEND_PATCHED_TREE,
    BACKEND_REPOSITORY,
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_TEXT,
    GPU_REQUEST,
    MODAL_SDK_VERSION,
    MODEL_CACHE_GENERATION,
    MODEL_CACHE_MOUNT,
    MODEL_CACHE_NAME,
    MODEL_CHECKPOINT_PATH,
    MODEL_CHECKPOINT_SHA256,
    MODEL_NAME,
    MODEL_STATE_SHA256,
    MODEL_URL,
    PATCH_MANIFEST_SHA256,
    RNG_SEED,
    RUNTIME_COMMIT,
    RUNTIME_REPOSITORY,
    _command_first_line,
    _decoded_pcm_fingerprint,
    _git_invocation,
    _hook_fingerprint,
    _model_fingerprint,
    _probe_blocked_network,
    _probe_read_only_model_cache,
    _resource_vector,
    _sha256_file,
    _source_identity,
    image,
    model_cache,
)

APP_NAME = "whisper-runtime-native-cuda-transaction"
SCHEMA_VERSION = "2"
PROFILE_ID = "tiny.en/cuda-0-float32-v1"
DEVICE = "cuda:0"
DECLARED_MEMORY_BYTES = 1_000_000_000
INJECTED_SYNC_FAILURES = 2
NATIVE_ADAPTER_PATH = "src/whisper_runtime/adapters/native_whisper.py"
RUNTIME_SOURCE_PATHS = (
    NATIVE_ADAPTER_PATH,
    "src/whisper_runtime/adapters/_model_binding.py",
    "src/whisper_runtime/execution.py",
    "src/whisper_runtime/resources.py",
    "src/whisper_runtime/state.py",
    "src/whisper_runtime/transaction.py",
    "src/whisper_runtime/worker.py",
)
EXPECTED_NATIVE_ADAPTER_SHA256 = (
    "1e8aef1728d9f8d16af9ac54810696faece47726de84b29a476acf951c493e8d"
)
_T = TypeVar("_T")


app = modal.App(APP_NAME)


def _stable_thread_role(thread_id: int, decode_thread_id: int | None) -> str:
    if decode_thread_id is not None and thread_id == decode_thread_id:
        return "decode"
    return "controller"


class _ScenarioTrace:
    """Collect ordered observations without replacing runtime state changes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.started_ns = time.monotonic_ns()
        self.decode_thread_id: int | None = None
        self.events: list[dict[str, object]] = []
        self.stream_labels: dict[tuple[str, int, int], str] = {}
        self.event_count = 0
        self.cleanup_calls = 0
        self.injected_sync_failures_remaining = 0
        self.injected_sync_failures_observed = 0
        self.cancel_after_step = False
        self.cancel_step_returned: bool | None = None
        self.cancel_run_complete_after_step: bool | None = None
        self.child_generator_checks = 0
        self.child_generators_on_device = True
        self.child_generators_distinct = True
        self.step_submitted = threading.Event()
        self.release_step = threading.Event()
        self.transactions: list[object] = []
        self._lock = threading.Lock()

    def set_decode_thread(self) -> None:
        self.decode_thread_id = threading.get_ident()

    def stream_label(self, stream: object) -> str:
        handle = getattr(stream, "cuda_stream", None)
        device = getattr(stream, "device", None)
        device_type = getattr(device, "type", None)
        device_index = getattr(device, "index", None)
        if (
            isinstance(handle, bool)
            or not isinstance(handle, int)
            or device_type != "cuda"
            or isinstance(device_index, bool)
            or not isinstance(device_index, int)
        ):
            raise RuntimeError("CUDA stream identity is unavailable")
        key = (device_type, device_index, handle)
        with self._lock:
            label = self.stream_labels.get(key)
            if label is None:
                label = f"stream-{len(self.stream_labels) + 1}"
                self.stream_labels[key] = label
            return label

    def next_event_label(self) -> str:
        with self._lock:
            self.event_count += 1
            return f"event-{self.event_count}"

    def record(
        self,
        name: str,
        *,
        kind: str,
        stream: str | None = None,
        state: dict[str, object] | None = None,
    ) -> None:
        thread_id = threading.get_ident()
        event: dict[str, object] = {
            "sequence": 0,
            "offset_ns": 0,
            "name": name,
            "kind": kind,
            "thread": _stable_thread_role(thread_id, self.decode_thread_id),
            "stream": stream,
        }
        if state is not None:
            event["state"] = state
        with self._lock:
            event["sequence"] = len(self.events) + 1
            event["offset_ns"] = time.monotonic_ns() - self.started_ns
            self.events.append(event)

    def names(self) -> list[str]:
        return [str(event["name"]) for event in self.events]


class _TraceRouter:
    def __init__(self) -> None:
        self.current: _ScenarioTrace | None = None

    def require(self) -> _ScenarioTrace:
        if self.current is None:
            raise RuntimeError("no native CUDA validation scenario is active")
        return self.current


class _TracingCudaEvent:
    def __init__(
        self,
        delegate: object,
        router: _TraceRouter,
        label: str,
        worker: object,
        budget: object,
        session: object,
        request: object,
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._label = label
        self._worker = worker
        self._budget = budget
        self._session = session
        self._request = request
        self._stream_label: str | None = None

    def record(self, stream: object) -> None:
        trace = self._router.require()
        label = trace.stream_label(stream)
        if self._worker.queue_depth != 1 or self._budget.lease_count != 1:
            raise RuntimeError("the completion event was recorded without one lease")
        if self._session.snapshot().version != 0:
            raise RuntimeError("the session changed before the completion event")
        self._delegate.record(stream)
        self._stream_label = label
        trace.record(f"cuda:{self._label}:record", kind="cuda", stream=label)

    def synchronize(self) -> None:
        trace = self._router.require()
        if self._worker.queue_depth != 1 or self._budget.lease_count != 1:
            raise RuntimeError(
                "the completion event was synchronized without one lease"
            )
        if self._session.snapshot().version != 0:
            raise RuntimeError("the session changed before the completion fence")
        if self._stream_label is None:
            raise RuntimeError("the completion event was not recorded on a stream")
        trace.record(
            f"cuda:{self._label}:synchronize:begin",
            kind="cuda",
            stream=self._stream_label,
        )
        if trace.injected_sync_failures_remaining:
            trace.injected_sync_failures_remaining -= 1
            trace.injected_sync_failures_observed += 1
            trace.record(
                f"cuda:{self._label}:synchronize:injected-failure",
                kind="cuda",
                stream=self._stream_label,
            )
            raise RuntimeError("injected CUDA event synchronization failure")
        self._delegate.synchronize()
        completed = self._delegate.query()
        trace.record(
            f"cuda:{self._label}:query:return",
            kind="cuda",
            stream=self._stream_label,
        )
        if completed is not True:
            raise RuntimeError(
                "the CUDA event did not report completion after synchronize"
            )
        state = {
            "request_status": _status(self._request.status),
            "session_version": self._session.snapshot().version,
            "queue_depth": self._worker.queue_depth,
            "lease_count": self._budget.lease_count,
            "budget_available": _resource_vector(self._budget.available),
        }
        if (
            state["session_version"] != 0
            or state["queue_depth"] != 1
            or state["lease_count"] != 1
        ):
            raise RuntimeError("transaction ownership changed before fence completion")
        trace.record(
            f"cuda:{self._label}:synchronize:return",
            kind="cuda",
            stream=self._stream_label,
            state=state,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _TracingCuda:
    def __init__(
        self,
        delegate: object,
        router: _TraceRouter,
        worker: object,
        budget: object,
        session_getter: Callable[[], object],
        request_getter: Callable[[], object],
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._worker = worker
        self._budget = budget
        self._session_getter = session_getter
        self._request_getter = request_getter

    def is_available(self) -> bool:
        return bool(self._delegate.is_available())

    def device_count(self) -> int:
        return int(self._delegate.device_count())

    def synchronize(self, device: object | None = None) -> None:
        trace = self._router.require()
        if self._worker.queue_depth != 1 or self._budget.lease_count != 1:
            raise RuntimeError("CUDA initialization ran before worker admission")
        trace.record("cuda:device-synchronize:begin", kind="cuda")
        self._delegate.synchronize(device)
        trace.record("cuda:device-synchronize:return", kind="cuda")

    def Stream(self, *, device: object | None = None) -> object:
        trace = self._router.require()
        if device != DEVICE:
            raise RuntimeError(f"unexpected CUDA stream device: {device!r}")
        if self._worker.queue_depth != 1 or self._budget.lease_count != 1:
            raise RuntimeError("the CUDA stream was created before worker admission")
        stream = self._delegate.Stream(device=device)
        label = trace.stream_label(stream)
        trace.record("cuda:stream:create", kind="cuda", stream=label)
        return stream

    def Event(self, *, enable_timing: bool = False) -> _TracingCudaEvent:
        trace = self._router.require()
        if enable_timing is not False:
            raise RuntimeError("the transaction fence must use a non-timing event")
        label = trace.next_event_label()
        event = self._delegate.Event(enable_timing=enable_timing)
        trace.record(f"cuda:{label}:create", kind="cuda")
        return _TracingCudaEvent(
            event,
            self._router,
            label,
            self._worker,
            self._budget,
            self._session_getter(),
            self._request_getter(),
        )

    def device(self, device: object) -> AbstractContextManager[object]:
        return _TracingCudaContext(
            self._delegate.device(device),
            self._router,
            "cuda:device",
        )

    def stream(self, stream: object) -> AbstractContextManager[object]:
        label = self._router.require().stream_label(stream)
        return _TracingCudaContext(
            self._delegate.stream(stream),
            self._router,
            "cuda:stream",
            stream=label,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _TracingCudaContext(AbstractContextManager[object]):
    def __init__(
        self,
        delegate: AbstractContextManager[object],
        router: _TraceRouter,
        name: str,
        *,
        stream: str | None = None,
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._name = name
        self._stream = stream

    def __enter__(self) -> object:
        result = self._delegate.__enter__()
        self._router.require().record(
            f"{self._name}:enter",
            kind="cuda",
            stream=self._stream,
        )
        return result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            return self._delegate.__exit__(exc_type, exc_value, traceback)
        finally:
            self._router.require().record(
                f"{self._name}:exit",
                kind="cuda",
                stream=self._stream,
            )


class _TorchProxy:
    def __init__(self, delegate: object, cuda: _TracingCuda) -> None:
        self._delegate = delegate
        self.cuda = cuda
        self.float32 = delegate.float32

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


def _require_current_stream(torch_module: object, trace: _ScenarioTrace) -> str:
    if not trace.stream_labels:
        raise RuntimeError("a native stage ran before stream creation")
    current = torch_module.cuda.current_stream(device=DEVICE)
    label = trace.stream_label(current)
    if label != "stream-1":
        raise RuntimeError(f"a native stage ran on {label}, expected stream-1")
    return label


class _RunProxy:
    def __init__(
        self, delegate: object, router: _TraceRouter, torch_module: object
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._torch = torch_module

    @property
    def complete(self) -> bool:
        return bool(self._delegate.complete)

    @property
    def inference(self) -> object:
        return self._delegate.inference

    @property
    def _legacy_cache_lock(self) -> object:
        return self._delegate._legacy_cache_lock

    def _invoke(self, name: str, operation: Callable[[], _T]) -> _T:
        trace = self._router.require()
        label = _require_current_stream(self._torch, trace)
        trace.record(f"run:{name}:begin", kind="backend", stream=label)
        result = operation()
        trace.record(f"run:{name}:submitted", kind="backend", stream=label)
        return result

    def prefill(self) -> None:
        self._invoke("prefill", self._delegate.prefill)

    def step(self) -> bool:
        trace = self._router.require()
        result = self._invoke("step", self._delegate.step)
        if trace.cancel_after_step and not trace.step_submitted.is_set():
            run_complete = bool(self._delegate.complete)
            trace.cancel_step_returned = result
            trace.cancel_run_complete_after_step = run_complete
            if result is not False or run_complete:
                raise RuntimeError(
                    "the cancellation rendezvous requires one incomplete token step"
                )
            label = _require_current_stream(self._torch, trace)
            trace.record(
                "run:cancellation-rendezvous:incomplete",
                kind="backend",
                stream=label,
            )
            trace.step_submitted.set()
            if not trace.release_step.wait(timeout=60):
                raise RuntimeError(
                    "the cancellation controller did not release the step"
                )
        return result

    def finalize(self) -> list[object]:
        return self._invoke("finalize", self._delegate.finalize)

    def cleanup(self) -> None:
        trace = self._router.require()
        trace.cleanup_calls += 1
        self._invoke("cleanup", self._delegate.cleanup)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _TaskProxy:
    def __init__(
        self, delegate: object, router: _TraceRouter, torch_module: object
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._torch = torch_module

    @property
    def inference(self) -> object:
        return self._delegate.inference

    def _uses_legacy_extension(self) -> bool:
        return bool(self._delegate._uses_legacy_extension())

    def _start_run(self, mel: object) -> _RunProxy:
        trace = self._router.require()
        label = _require_current_stream(self._torch, trace)
        if str(getattr(mel, "device", "")) != DEVICE:
            raise RuntimeError("the native task did not receive the CUDA mel tensor")
        generator_device = str(getattr(self._delegate.options.generator, "device", ""))
        if generator_device != DEVICE:
            raise RuntimeError(
                f"the decode generator used {generator_device!r}, expected {DEVICE!r}"
            )
        trace.record("run:start:begin", kind="backend", stream=label)
        run = self._delegate._start_run(mel)
        trace.record("run:start:submitted", kind="backend", stream=label)
        decoder = getattr(run, "decoder", None)
        child_generator = getattr(decoder, "generator", None)
        source_generator = getattr(self._delegate, "_generator_source", None)
        option_generator = getattr(self._delegate.options, "generator", None)
        child_device = str(getattr(child_generator, "device", ""))
        generators_distinct = (
            child_generator is not None
            and source_generator is not None
            and option_generator is not None
            and child_generator is not source_generator
            and child_generator is not option_generator
        )
        trace.child_generator_checks += 1
        trace.child_generators_on_device &= child_device == DEVICE
        trace.child_generators_distinct &= generators_distinct
        if child_device != DEVICE or not generators_distinct:
            raise RuntimeError(
                "the decode run did not receive a distinct generator on cuda:0"
            )
        trace.record("run:child-generator:verified", kind="backend", stream=label)
        return _RunProxy(run, self._router, self._torch)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


def _memory_snapshot(torch_module: object) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch_module.cuda.memory_reserved(0)),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(0)),
    }


def _status(value: object) -> str:
    return str(getattr(value, "value", value))


def _run_success(
    *,
    adapter: object,
    worker: object,
    budget: object,
    snapshot: object,
    mel: object,
    router: _TraceRouter,
    session_factory: Callable[[str], object],
    session_setter: Callable[[object], None],
    request_setter: Callable[[object], None],
    torch_module: object,
    request_id: str,
    session_id: str,
    backend_instrumented: bool,
) -> dict[str, object]:
    from whisper_runtime import RequestState

    trace = _ScenarioTrace(request_id)
    trace.set_decode_thread()
    router.current = trace
    session = session_factory(session_id)
    session_setter(session)
    request = RequestState(request_id, session_id, snapshot, rng_seed=RNG_SEED)
    request_setter(request)
    budget_before = budget.available
    memory_before = _memory_snapshot(torch_module)
    torch_module.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    state = adapter.decode_window(
        session=session,
        request=request,
        window_id=f"{request_id}-window",
        mel=mel,
        start_ms=0,
        end_ms=11_000,
        options=__import__(
            "whisper_runtime.adapters", fromlist=["NativeDecodeOptions"]
        ).NativeDecodeOptions(
            language="en",
            task="transcribe",
            temperature=0.0,
            without_timestamps=True,
        ),
    )
    elapsed = time.perf_counter() - started
    gc.collect()
    memory_after = _memory_snapshot(torch_module)
    transaction = trace.transactions[-1]
    result = state.windows[-1].result
    router.current = None
    return {
        "outcome": "committed",
        "request_status": _status(request.status),
        "transaction_status": _status(transaction.status),
        "session_version_before": 0,
        "session_version_after": state.version,
        "window_count_after": len(state.windows),
        "text": result.text,
        "expected_text_matched": result.text == EXPECTED_TEXT,
        "queue_depth_after": worker.queue_depth,
        "budget_available_before": _resource_vector(budget_before),
        "budget_available_after": _resource_vector(budget.available),
        "backend_instrumented": backend_instrumented,
        "child_generator_checks": trace.child_generator_checks,
        "child_generators_on_profile_device": trace.child_generators_on_device,
        "child_generators_distinct": trace.child_generators_distinct,
        "traced_cleanup_calls": trace.cleanup_calls,
        "traced_stream_count": len(trace.stream_labels),
        "traced_event_count": trace.event_count,
        "wall_seconds": float(elapsed),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "trace": trace.events,
    }


def _run_cancellation(
    *,
    adapter: object,
    worker: object,
    budget: object,
    snapshot: object,
    mel: object,
    router: _TraceRouter,
    session_factory: Callable[[str], object],
    session_setter: Callable[[object], None],
    request_setter: Callable[[object], None],
    request_cancelled_error: type[BaseException],
) -> dict[str, object]:
    from whisper_runtime import RequestState

    trace = _ScenarioTrace("cancellation")
    trace.cancel_after_step = True
    router.current = trace
    session = session_factory("modal-native-cuda-cancel")
    session_setter(session)
    request = RequestState(
        "modal-native-cuda-cancel",
        session.session_id,
        snapshot,
        rng_seed=RNG_SEED,
    )
    request_setter(request)
    errors: list[BaseException] = []

    def decode() -> None:
        trace.set_decode_thread()
        try:
            adapter.decode_window(
                session=session,
                request=request,
                window_id="cancel-window",
                mel=mel,
                start_ms=0,
                end_ms=11_000,
                options=__import__(
                    "whisper_runtime.adapters", fromlist=["NativeDecodeOptions"]
                ).NativeDecodeOptions(
                    language="en",
                    task="transcribe",
                    temperature=0.0,
                    without_timestamps=True,
                ),
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=decode, name="native-cuda-decode")
    thread.start()
    if not trace.step_submitted.wait(timeout=120):
        trace.release_step.set()
        thread.join(timeout=30)
        raise RuntimeError("the real CUDA decode did not reach its first token step")
    admitted_queue_depth = worker.queue_depth
    admitted_lease_count = budget.lease_count
    session_version_at_cancel = session.snapshot().version
    trace.record("controller:cancel:begin", kind="controller")
    cancel_returned = request.cancel()
    trace.record("controller:cancel:return", kind="controller")
    trace.release_step.set()
    thread.join(timeout=120)
    if thread.is_alive():
        raise RuntimeError("the cancelled CUDA decode did not terminate")
    if len(errors) != 1 or not isinstance(errors[0], request_cancelled_error):
        observed = type(errors[0]).__name__ if errors else "no exception"
        raise RuntimeError(f"unexpected cancellation outcome: {observed}")
    transaction = trace.transactions[-1]
    record = {
        "outcome": "cancelled",
        "exception_type": type(errors[0]).__name__,
        "cancel_returned": cancel_returned,
        "request_status": _status(request.status),
        "transaction_status": _status(transaction.status),
        "session_version_at_cancel": session_version_at_cancel,
        "session_version_after": session.snapshot().version,
        "window_count_after": len(session.snapshot().windows),
        "queue_depth_at_cancel": admitted_queue_depth,
        "lease_count_at_cancel": admitted_lease_count,
        "queue_depth_after": worker.queue_depth,
        "budget_available_after": _resource_vector(budget.available),
        "backend_instrumented": True,
        "first_step_returned": trace.cancel_step_returned,
        "run_complete_after_first_step": trace.cancel_run_complete_after_step,
        "child_generator_checks": trace.child_generator_checks,
        "child_generators_on_profile_device": trace.child_generators_on_device,
        "child_generators_distinct": trace.child_generators_distinct,
        "traced_cleanup_calls": trace.cleanup_calls,
        "traced_stream_count": len(trace.stream_labels),
        "traced_event_count": trace.event_count,
        "controller_cuda_calls": sum(
            event["thread"] == "controller" and event["kind"] == "cuda"
            for event in trace.events
        ),
        "trace": trace.events,
    }
    router.current = None
    return record


def _run_recovery(
    *,
    adapter: object,
    worker: object,
    budget: object,
    snapshot: object,
    mel: object,
    router: _TraceRouter,
    session_factory: Callable[[str], object],
    session_setter: Callable[[object], None],
    request_setter: Callable[[object], None],
    retained_error_type: type[BaseException],
    transaction_status: object,
) -> dict[str, object]:
    from whisper_runtime import RequestState

    trace = _ScenarioTrace("recovery")
    trace.set_decode_thread()
    trace.injected_sync_failures_remaining = INJECTED_SYNC_FAILURES
    router.current = trace
    session = session_factory("modal-native-cuda-recovery")
    session_setter(session)
    request = RequestState(
        "modal-native-cuda-recovery",
        session.session_id,
        snapshot,
        rng_seed=RNG_SEED,
    )
    request_setter(request)
    retained: BaseException | None = None
    try:
        adapter.decode_window(
            session=session,
            request=request,
            window_id="recovery-window",
            mel=mel,
            start_ms=0,
            end_ms=11_000,
            options=__import__(
                "whisper_runtime.adapters", fromlist=["NativeDecodeOptions"]
            ).NativeDecodeOptions(
                language="en",
                task="transcribe",
                temperature=0.0,
                without_timestamps=True,
            ),
        )
    except retained_error_type as error:
        retained = error
    if retained is None:
        raise RuntimeError("the injected fence failures did not retain the transaction")
    transaction = retained.transaction
    if transaction.status is not transaction_status:
        raise RuntimeError(
            "the failed completion fence did not quarantine the transaction"
        )
    retained_queue_depth = worker.queue_depth
    retained_lease_count = budget.lease_count
    retained_budget = budget.available
    retained_session_version = session.snapshot().version
    retained_window_count = len(session.snapshot().windows)

    blocked_session = session_factory("modal-native-cuda-blocked")
    session_setter(blocked_session)
    blocked_request = RequestState(
        "modal-native-cuda-blocked",
        blocked_session.session_id,
        snapshot,
        rng_seed=RNG_SEED,
    )
    request_setter(blocked_request)
    blocked_error_same = False
    try:
        adapter.decode_window(
            session=blocked_session,
            request=blocked_request,
            window_id="blocked-window",
            mel=mel,
            start_ms=0,
            end_ms=11_000,
        )
    except retained_error_type as error:
        blocked_error_same = error is retained
    else:
        raise RuntimeError("new model work entered while a transaction was retained")
    blocked_queue_depth = worker.queue_depth
    blocked_lease_count = budget.lease_count
    blocked_session_version = blocked_session.snapshot().version

    session_setter(session)
    request_setter(request)
    trace.record("runtime:manual-recovery:begin", kind="runtime")
    recovered = worker.recover(transaction)
    trace.record("runtime:manual-recovery:return", kind="runtime")
    recovered_status = _status(transaction.status)
    queue_depth_after_recovery = worker.queue_depth
    budget_after_recovery = budget.available
    session_version_after_recovery = session.snapshot().version
    request_status_after_recovery = _status(request.status)

    reuse = _run_success(
        adapter=adapter,
        worker=worker,
        budget=budget,
        snapshot=snapshot,
        mel=mel,
        router=router,
        session_factory=session_factory,
        session_setter=session_setter,
        request_setter=request_setter,
        torch_module=__import__("torch"),
        request_id="post-recovery-reuse",
        session_id="modal-native-cuda-post-recovery",
        backend_instrumented=True,
    )
    record = {
        "outcome": "recovered",
        "injected_failure": "cuda-event-synchronize-before-delegate",
        "injected_failure_count": trace.injected_sync_failures_observed,
        "retained_exception_type": type(retained).__name__,
        "retained_transaction_status": "quarantined",
        "retained_queue_depth": retained_queue_depth,
        "retained_lease_count": retained_lease_count,
        "retained_budget_available": _resource_vector(retained_budget),
        "retained_session_version": retained_session_version,
        "retained_window_count": retained_window_count,
        "blocked_error_same_instance": blocked_error_same,
        "blocked_request_status": _status(blocked_request.status),
        "blocked_session_version": blocked_session_version,
        "blocked_queue_depth": blocked_queue_depth,
        "blocked_lease_count": blocked_lease_count,
        "recovery_returned": recovered,
        "recovered_transaction_status": recovered_status,
        "queue_depth_after_recovery": queue_depth_after_recovery,
        "budget_available_after_recovery": _resource_vector(budget_after_recovery),
        "session_version_after_recovery": session_version_after_recovery,
        "request_status_after_recovery": request_status_after_recovery,
        "child_generator_checks": trace.child_generator_checks,
        "child_generators_on_profile_device": trace.child_generators_on_device,
        "child_generators_distinct": trace.child_generators_distinct,
        "traced_cleanup_calls_before_recovery_complete": trace.cleanup_calls,
        "traced_event_count_before_recovery_complete": trace.event_count,
        "trace": trace.events,
        "post_recovery_reuse": reuse,
    }
    router.current = None
    return record


@app.function(
    image=image,
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
def prime_model_cache_v2() -> dict[str, object]:
    """Populate the pinned checkpoint cache before the isolated GPU phase."""

    destination = MODEL_CHECKPOINT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256_file(destination) == MODEL_CHECKPOINT_SHA256:
        return {
            "cache_generation": MODEL_CACHE_GENERATION,
            "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
            "downloaded": False,
        }
    temporary = destination.with_suffix(".pt.partial")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"checkpoint download returned HTTP {response.status}"
                )
            with temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        observed = _sha256_file(temporary)
        if observed != MODEL_CHECKPOINT_SHA256:
            raise RuntimeError("the downloaded checkpoint digest does not match")
        os.replace(temporary, destination)
        model_cache.commit()
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "cache_generation": MODEL_CACHE_GENERATION,
        "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "downloaded": True,
    }


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    volumes={MODEL_CACHE_MOUNT: model_cache.with_mount_options(read_only=True)},
    cpu=2.0,
    memory=4096,
    timeout=1200,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    block_network=True,
    restrict_modal_access=True,
    single_use_containers=True,
    include_source=False,
)
def validate_native_cuda_transactions() -> dict[str, object]:
    """Return one closed record for four real adapter transaction cases."""

    import numpy
    import torch
    import whisper
    from whisper.audio import SAMPLE_RATE
    from whisper.decoding import DecodingTask

    import whisper_runtime.adapters.native_whisper as native_module
    from whisper_runtime import (
        Budget,
        ModelSnapshot,
        RequestCancelledError,
        ResourceVector,
        Session,
        TransactionRetainedError,
        Worker,
    )
    from whisper_runtime.adapters import NativeExecutionProfile, NativeWhisperAdapter
    from whisper_runtime.transaction import TransactionStatus

    started = time.perf_counter()
    network_probe = _probe_blocked_network()
    model_cache_probe = _probe_read_only_model_cache()
    runtime_root = Path("/opt/whisper-runtime")
    backend_root = Path("/opt/openai-whisper")
    runtime = _source_identity(runtime_root)
    backend = _source_identity(backend_root)
    if runtime["git_commit"] != RUNTIME_COMMIT:
        raise RuntimeError("the runtime commit differs from the requested commit")
    if backend["git_tree"] != BACKEND_PATCHED_TREE:
        raise RuntimeError("the patched backend tree differs from the pinned tree")
    if _sha256_file(runtime_root / "patches/openai-whisper/SHA256SUMS") != (
        PATCH_MANIFEST_SHA256
    ):
        raise RuntimeError("the patch manifest differs from its pinned digest")
    source_files = [
        {"path": path, "sha256": _sha256_file(runtime_root / path)}
        for path in RUNTIME_SOURCE_PATHS
    ]
    if source_files[0]["sha256"] != EXPECTED_NATIVE_ADAPTER_SHA256:
        raise RuntimeError("the native CUDA adapter differs from its pinned source")
    if _sha256_file(MODEL_CHECKPOINT_PATH) != MODEL_CHECKPOINT_SHA256:
        raise RuntimeError("the cached checkpoint differs from its pinned digest")
    if _sha256_file(AUDIO_PATH) != AUDIO_SHA256:
        raise RuntimeError("the audio fixture differs from its pinned digest")
    if AUDIO_PATH.stat().st_size != AUDIO_SIZE_BYTES:
        raise RuntimeError("the audio fixture size differs from its manifest")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the check requires exactly one visible CUDA device")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    if "T4" not in device_name or capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError("the allocated GPU is not the requested T4")

    model = whisper.load_model(
        MODEL_NAME,
        device=DEVICE,
        download_root=MODEL_CACHE_MOUNT,
    ).eval()
    if str(model.device) != DEVICE:
        raise RuntimeError("the model loaded on an unexpected device")
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    model_tensors = parameters + buffers
    model_in_evaluation_mode = model.training is False
    all_model_tensors_on_device = bool(model_tensors) and all(
        str(tensor.device) == DEVICE for tensor in model_tensors
    )
    floating_model_tensors = tuple(
        tensor for tensor in model_tensors if tensor.is_floating_point()
    )
    all_floating_model_tensors_fp32 = bool(floating_model_tensors) and all(
        tensor.dtype == torch.float32 for tensor in floating_model_tensors
    )
    if not (
        model_in_evaluation_mode
        and all_model_tensors_on_device
        and all_floating_model_tensors_fp32
    ):
        raise RuntimeError("the loaded model does not match the CUDA FP32 profile")
    state_before = _model_fingerprint(model)
    hooks_before = _hook_fingerprint(model)
    if state_before != MODEL_STATE_SHA256:
        raise RuntimeError("the loaded model state differs from its fingerprint")
    audio = whisper.load_audio(str(AUDIO_PATH))
    if SAMPLE_RATE != AUDIO_SAMPLE_RATE_HZ or len(audio) != AUDIO_SAMPLE_COUNT:
        raise RuntimeError("the decoded audio differs from its manifest")
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
    ).contiguous()
    if str(mel.device) != "cpu" or mel.dtype != torch.float32:
        raise RuntimeError("the adapter input must remain CPU float32")

    snapshot = ModelSnapshot(
        model_id=MODEL_NAME,
        revision=str(backend["git_commit"]),
        backend="pytorch-cuda-transaction",
        fingerprint=state_before,
    )
    capacity = ResourceVector(
        memory_bytes=DECLARED_MEMORY_BYTES,
        compute_units=1,
        stream_slots=1,
    )
    router = _TraceRouter()

    class TracingBudget(Budget):
        def acquire(self, resources: object) -> object:
            lease = super().acquire(resources)
            router.require().record("budget:lease:acquired", kind="runtime")
            return lease

        def _release(self, lease: object) -> None:
            router.require().record("budget:lease:release:begin", kind="runtime")
            super()._release(lease)
            router.require().record("budget:lease:release:return", kind="runtime")

    budget = TracingBudget(capacity)

    class TracingWorker(Worker):
        def prepare(self, **kwargs: object) -> object:
            transaction = super().prepare(**kwargs)
            trace = router.require()
            trace.transactions.append(transaction)
            trace.record("worker:admitted", kind="runtime")
            return transaction

    worker = TracingWorker(
        "modal-native-cuda",
        snapshot,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=300,
    )
    active_session: list[object] = []
    active_request: list[object] = []

    class TracingSession(Session):
        def _commit(self, expected_version: int, record: object) -> object:
            trace = router.require()
            request = get_request()
            if (
                _status(request.status) != "running"
                or self.snapshot().version != 0
                or worker.queue_depth != 1
                or budget.lease_count != 1
            ):
                raise RuntimeError(
                    "session publication started outside its transaction"
                )
            trace.record("session:commit:begin", kind="runtime")
            state = super()._commit(expected_version, record)
            if state.version != 1 or worker.queue_depth != 1 or budget.lease_count != 1:
                raise RuntimeError("transaction ownership changed during publication")
            trace.record("session:commit:return", kind="runtime")
            return state

    def make_session(session_id: str) -> TracingSession:
        return TracingSession(session_id)

    def set_session(session: object) -> None:
        active_session[:] = [session]

    def get_session() -> object:
        if not active_session:
            raise RuntimeError("no session is bound to the active scenario")
        return active_session[0]

    def set_request(request: object) -> None:
        active_request[:] = [request]

    def get_request() -> object:
        if not active_request:
            raise RuntimeError("no request is bound to the active scenario")
        return active_request[0]

    original_component_loader = native_module._load_native_components
    real_components = original_component_loader()
    tracing_cuda = _TracingCuda(
        torch.cuda,
        router,
        worker,
        budget,
        get_session,
        get_request,
    )
    torch_proxy = _TorchProxy(torch, tracing_cuda)

    def task_type(observed_model: object, options: object) -> _TaskProxy:
        trace = router.require()
        label = _require_current_stream(torch, trace)
        trace.record("task:construct:begin", kind="backend", stream=label)
        task = DecodingTask(observed_model, options)
        trace.record("task:construct:return", kind="backend", stream=label)
        return _TaskProxy(task, router, torch)

    traced_components = native_module._NativeComponents(
        generator_type=real_components.generator_type,
        options_type=real_components.options_type,
        task_type=task_type,
        n_frames=real_components.n_frames,
        torch_module=torch_proxy,
    )

    def identity_probe(observed: object) -> ModelSnapshot:
        # This probe runs inside the private CUDA stream. A state-dict hash would
        # copy every parameter to the CPU and synchronize before the transaction
        # event, making the event-fence observation circular. Bind object identity
        # here, then compare strong state hashes before and after a global fence.
        if observed is not model:
            raise RuntimeError("the adapter received a different model object")
        if str(observed.device) != DEVICE:
            raise RuntimeError("the bound model moved to a different device")
        if router.current is not None and router.current.stream_labels:
            label = _require_current_stream(torch, router.current)
            router.current.record(
                "model:identity:verified", kind="backend", stream=label
            )
        return snapshot

    adapter = NativeWhisperAdapter(
        worker,
        model,
        identity_probe,
        NativeExecutionProfile(
            PROFILE_ID,
            capacity,
            max_concurrent_decodes=1,
            device=DEVICE,
        ),
    )
    memory_baseline = _memory_snapshot(torch)
    with patch.object(
        native_module,
        "_load_native_components",
        return_value=traced_components,
    ):
        success = _run_success(
            adapter=adapter,
            worker=worker,
            budget=budget,
            snapshot=snapshot,
            mel=mel,
            router=router,
            session_factory=make_session,
            session_setter=set_session,
            request_setter=set_request,
            torch_module=torch,
            request_id="modal-native-cuda-success",
            session_id="modal-native-cuda-success",
            backend_instrumented=True,
        )
        cancellation = _run_cancellation(
            adapter=adapter,
            worker=worker,
            budget=budget,
            snapshot=snapshot,
            mel=mel,
            router=router,
            session_factory=make_session,
            session_setter=set_session,
            request_setter=set_request,
            request_cancelled_error=RequestCancelledError,
        )
        recovery = _run_recovery(
            adapter=adapter,
            worker=worker,
            budget=budget,
            snapshot=snapshot,
            mel=mel,
            router=router,
            session_factory=make_session,
            session_setter=set_session,
            request_setter=set_request,
            retained_error_type=TransactionRetainedError,
            transaction_status=TransactionStatus.QUARANTINED,
        )

    restored_components = native_module._load_native_components()
    native_components_restored = (
        native_module._load_native_components is original_component_loader
        and restored_components.torch_module is torch
        and restored_components.task_type is DecodingTask
        and restored_components.generator_type is torch.Generator
    )
    if not native_components_restored:
        raise RuntimeError("the native backend components were not restored")

    # Control run: use the same adapter and model after recovery without the
    # component, task, run, torch, or CUDA proxies used to collect stage traces.
    unproxied_reuse = _run_success(
        adapter=adapter,
        worker=worker,
        budget=budget,
        snapshot=snapshot,
        mel=mel,
        router=router,
        session_factory=make_session,
        session_setter=set_session,
        request_setter=set_request,
        torch_module=torch,
        request_id="unproxied-native-reuse",
        session_id="modal-native-cuda-unproxied-reuse",
        backend_instrumented=False,
    )

    torch.cuda.synchronize(0)
    gc.collect()
    memory_final = _memory_snapshot(torch)
    state_after = _model_fingerprint(model)
    hooks_after = _hook_fingerprint(model)
    peak_delta = max(
        int(success["memory_after"]["peak_allocated_bytes"])
        - int(success["memory_before"]["allocated_bytes"]),
        0,
    )
    zero_resources = {
        "memory_bytes": 0,
        "compute_units": 0,
        "stream_slots": 0,
    }
    success_fence_state = _trace_state(
        success["trace"], "cuda:event-1:synchronize:return"
    )
    cancellation_fence_state = _trace_state(
        cancellation["trace"], "cuda:event-1:synchronize:return"
    )
    recovery_fence_state = _trace_state(
        recovery["trace"], "cuda:event-3:synchronize:return"
    )
    assertions = {
        "runtime_source_pinned": True,
        "backend_source_pinned": True,
        "checkpoint_verified_before_load": True,
        "input_fixture_verified": True,
        "network_probe_denied": network_probe["denied"] is True,
        "model_cache_read_only": model_cache_probe["denied"] is True,
        "native_adapter_committed": success["outcome"] == "committed",
        "success_exact_result_and_terminal_states": (
            success["text"] == EXPECTED_TEXT
            and success["expected_text_matched"] is True
            and success["request_status"] == "committed"
            and success["transaction_status"] == "committed"
            and success["session_version_before"] == 0
            and success["session_version_after"] == 1
            and success["window_count_after"] == 1
            and success["queue_depth_after"] == 0
            and success["budget_available_before"] == _resource_vector(capacity)
            and success["budget_available_after"] == _resource_vector(capacity)
        ),
        "publication_followed_cuda_fence": _ordered(
            success["trace"],
            "cuda:event-1:synchronize:return",
            "session:commit:begin",
            "session:commit:return",
            "budget:lease:release:begin",
        ),
        "success_fence_retained_transaction": success_fence_state
        == {
            "request_status": "running",
            "session_version": 0,
            "queue_depth": 1,
            "lease_count": 1,
            "budget_available": zero_resources,
        },
        "success_used_one_private_stream": success["traced_stream_count"] == 1,
        "success_fence_and_cleanup_counts": (
            success["traced_cleanup_calls"] == 1 and success["traced_event_count"] == 1
        ),
        "cancellation_prevented_publication": (
            cancellation["outcome"] == "cancelled"
            and cancellation["session_version_after"] == 0
        ),
        "cancellation_exact_terminal_state": (
            cancellation["cancel_returned"] is True
            and cancellation["exception_type"] == "RequestCancelledError"
            and cancellation["request_status"] == "cancelled"
            and cancellation["transaction_status"] == "aborted"
            and cancellation["session_version_at_cancel"] == 0
            and cancellation["window_count_after"] == 0
            and cancellation["queue_depth_after"] == 0
            and cancellation["budget_available_after"] == _resource_vector(capacity)
            and cancellation["traced_cleanup_calls"] == 1
            and cancellation["traced_stream_count"] == 1
            and cancellation["traced_event_count"] == 1
        ),
        "cancellation_held_capacity_at_request": (
            cancellation["queue_depth_at_cancel"] == 1
            and cancellation["lease_count_at_cancel"] == 1
        ),
        "cancellation_fence_retained_transaction": cancellation_fence_state
        == {
            "request_status": "cancelled",
            "session_version": 0,
            "queue_depth": 1,
            "lease_count": 1,
            "budget_available": zero_resources,
        },
        "cancellation_fence_preceded_release": _ordered(
            cancellation["trace"],
            "controller:cancel:return",
            "cuda:event-1:synchronize:return",
            "budget:lease:release:begin",
        ),
        "cancelling_thread_made_no_cuda_call": (
            cancellation["controller_cuda_calls"] == 0
        ),
        "failed_fence_retained_capacity": (
            recovery["retained_queue_depth"] == 1
            and recovery["retained_lease_count"] == 1
        ),
        "recovery_retained_exact_state": (
            recovery["injected_failure_count"] == INJECTED_SYNC_FAILURES
            and recovery["retained_exception_type"] == "TransactionRetainedError"
            and recovery["retained_transaction_status"] == "quarantined"
            and recovery["retained_budget_available"] == zero_resources
            and recovery["retained_session_version"] == 0
            and recovery["retained_window_count"] == 0
        ),
        "recovery_blocked_new_work": (
            recovery["blocked_error_same_instance"] is True
            and recovery["blocked_request_status"] == "created"
            and recovery["blocked_session_version"] == 0
            and recovery["blocked_queue_depth"] == 1
            and recovery["blocked_lease_count"] == 1
        ),
        "recovery_released_capacity": (
            recovery["recovery_returned"] is True
            and recovery["recovered_transaction_status"] == "aborted"
            and recovery["request_status_after_recovery"] == "aborted"
            and recovery["queue_depth_after_recovery"] == 0
            and recovery["budget_available_after_recovery"]
            == _resource_vector(capacity)
            and recovery["session_version_after_recovery"] == 0
            and recovery["traced_cleanup_calls_before_recovery_complete"] == 3
            and recovery["traced_event_count_before_recovery_complete"] == 3
        ),
        "recovery_fence_retained_transaction": recovery_fence_state
        == {
            "request_status": "running",
            "session_version": 0,
            "queue_depth": 1,
            "lease_count": 1,
            "budget_available": zero_resources,
        },
        "recovery_fence_preceded_release": _ordered(
            recovery["trace"],
            "cuda:event-1:synchronize:injected-failure",
            "cuda:event-2:synchronize:injected-failure",
            "runtime:manual-recovery:begin",
            "cuda:event-3:synchronize:return",
            "budget:lease:release:begin",
            "runtime:manual-recovery:return",
        ),
        "post_recovery_reuse_committed": (
            recovery["post_recovery_reuse"]["outcome"] == "committed"
            and recovery["post_recovery_reuse"]["text"] == EXPECTED_TEXT
            and recovery["post_recovery_reuse"]["request_status"] == "committed"
            and recovery["post_recovery_reuse"]["transaction_status"] == "committed"
        ),
        "unproxied_native_reuse_committed": (
            unproxied_reuse["outcome"] == "committed"
            and unproxied_reuse["expected_text_matched"] is True
            and unproxied_reuse["request_status"] == "committed"
            and unproxied_reuse["transaction_status"] == "committed"
            and unproxied_reuse["backend_instrumented"] is False
            and unproxied_reuse["traced_stream_count"] == 0
            and unproxied_reuse["traced_event_count"] == 0
            and unproxied_reuse["traced_cleanup_calls"] == 0
        ),
        "native_components_restored_for_unproxied_reuse": (native_components_restored),
        "run_child_generators_verified": (
            success["child_generator_checks"] == 1
            and success["child_generators_on_profile_device"] is True
            and success["child_generators_distinct"] is True
            and cancellation["child_generator_checks"] == 1
            and cancellation["child_generators_on_profile_device"] is True
            and cancellation["child_generators_distinct"] is True
            and recovery["child_generator_checks"] == 1
            and recovery["child_generators_on_profile_device"] is True
            and recovery["child_generators_distinct"] is True
            and recovery["post_recovery_reuse"]["child_generator_checks"] == 1
            and recovery["post_recovery_reuse"]["child_generators_on_profile_device"]
            is True
            and recovery["post_recovery_reuse"]["child_generators_distinct"] is True
        ),
        "cancellation_rendezvous_proved": (
            cancellation["first_step_returned"] is False
            and cancellation["run_complete_after_first_step"] is False
        ),
        "model_profile_verified": (
            model_in_evaluation_mode
            and all_model_tensors_on_device
            and all_floating_model_tensors_fp32
        ),
        "persistent_model_state_unchanged": state_after == state_before,
        "model_hooks_unchanged": hooks_after == hooks_before,
        "observed_peak_within_declared_memory": peak_delta <= DECLARED_MEMORY_BYTES,
    }
    if any(value is not True for value in assertions.values()):
        failed = sorted(name for name, value in assertions.items() if value is not True)
        raise RuntimeError(
            f"native CUDA transaction assertions failed: {', '.join(failed)}"
        )
    function_call_id = modal.current_function_call_id()
    if not isinstance(function_call_id, str) or not function_call_id:
        raise RuntimeError("Modal did not expose a function-call identifier")
    observed_modal_sdk = str(modal.__version__)
    if observed_modal_sdk != MODAL_SDK_VERSION:
        raise RuntimeError("the Modal SDK differs from the pinned version")

    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "passed",
        "scope": {
            "evidence_kind": "native-whisper-cuda-transaction",
            "statement": (
                "One pinned runtime completed, cancelled, quarantined, recovered, "
                "and reused NativeWhisperAdapter transactions on one Modal T4."
            ),
            "fault_injection_used": True,
        },
        "claims": {
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
        },
        "runtime": {
            "repository": RUNTIME_REPOSITORY,
            **runtime,
            "source_files": source_files,
        },
        "backend": {
            "repository": BACKEND_REPOSITORY,
            "base_commit": BACKEND_BASE_COMMIT,
            "base_tree": BACKEND_BASE_TREE,
            "applied_commit": backend["git_commit"],
            "git_tree": backend["git_tree"],
            "clean": backend["clean"],
            "patch_manifest": "patches/openai-whisper/SHA256SUMS",
            "patch_manifest_sha256": PATCH_MANIFEST_SHA256,
        },
        "modal": {
            "sdk_version": observed_modal_sdk,
            "function_call_id": function_call_id,
            "image_id": os.environ.get("MODAL_IMAGE_ID"),
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "environment": os.environ.get("MODAL_ENVIRONMENT"),
            "cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
            "region": os.environ.get("MODAL_REGION"),
            "network_blocked": True,
            "modal_access_restricted": True,
            "model_cache": {
                "name": MODEL_CACHE_NAME,
                "generation": MODEL_CACHE_GENERATION,
                "mount_path": MODEL_CACHE_MOUNT,
                "read_only": True,
                "write_probe": model_cache_probe,
            },
            "network_probe": network_probe,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
            "nvidia_driver": _command_first_line(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ),
            "ffmpeg": _command_first_line("ffmpeg", "-version"),
            "numpy": str(numpy.__version__),
            "tiktoken": str(importlib.metadata.version("tiktoken")),
            "numba": str(importlib.metadata.version("numba")),
            "tqdm": str(importlib.metadata.version("tqdm")),
            "more_itertools": str(importlib.metadata.version("more-itertools")),
        },
        "gpu": {
            "requested": GPU_REQUEST,
            "visible_device_count": int(torch.cuda.device_count()),
            "device_index": int(device_index),
            "name": str(device_name),
            "capability_major": int(capability[0]),
            "capability_minor": int(capability[1]),
            "total_memory_bytes": int(properties.total_memory),
        },
        "model": {
            "name": MODEL_NAME,
            "device": str(model.device),
            "dtype": str(next(model.parameters()).dtype),
            "evaluation_mode": model_in_evaluation_mode,
            "parameter_tensor_count": len(parameters),
            "buffer_tensor_count": len(buffers),
            "all_parameters_and_buffers_on_profile_device": (
                all_model_tensors_on_device
            ),
            "all_floating_tensors_fp32": all_floating_model_tensors_fp32,
            "checkpoint_path": "model-cache-v1/tiny.en.pt",
            "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
            "loaded_state_sha256_before": state_before,
            "loaded_state_sha256_after": state_after,
        },
        "input": {
            "fixture_id": AUDIO_FIXTURE_ID,
            "path": AUDIO_RELATIVE_PATH,
            "sha256": AUDIO_SHA256,
            "size_bytes": AUDIO_SIZE_BYTES,
            "sample_rate_hz": SAMPLE_RATE,
            "sample_count": len(audio),
            "decoded_pcm_sha256": _decoded_pcm_fingerprint(audio),
            "mel_device_at_boundary": str(mel.device),
            "mel_dtype_at_boundary": str(mel.dtype),
        },
        "profile": {
            "profile_id": PROFILE_ID,
            "device": DEVICE,
            "max_concurrent_decodes": 1,
            "resources": _resource_vector(capacity),
            "fp16": False,
            "gpu_memory_measured": True,
            "gpu_memory_enforced": False,
        },
        "success": success,
        "cancellation": cancellation,
        "recovery": recovery,
        "unproxied_reuse": unproxied_reuse,
        "memory": {
            "baseline": memory_baseline,
            "final": memory_final,
            "observed_success_peak_delta_bytes": peak_delta,
            "declared_memory_bytes": DECLARED_MEMORY_BYTES,
            "observed_peak_within_declaration": peak_delta <= DECLARED_MEMORY_BYTES,
        },
        "timing": {
            "total_seconds": float(time.perf_counter() - started),
            "performance_benchmark": False,
        },
        "assertions": assertions,
    }


def _ordered(trace: object, *expected: str) -> bool:
    if not isinstance(trace, list):
        return False
    names = [event.get("name") for event in trace if isinstance(event, dict)]
    try:
        positions = [names.index(name) for name in expected]
    except ValueError:
        return False
    return all(
        earlier < later
        for earlier, later in zip(positions, positions[1:], strict=False)
    )


def _trace_state(trace: object, name: str) -> object:
    if not isinstance(trace, list):
        return None
    matches = [
        event.get("state")
        for event in trace
        if isinstance(event, dict) and event.get("name") == name
    ]
    return matches[0] if len(matches) == 1 else None


def _write_record(path: Path, record: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_path(value: str) -> Path:
    relative = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or ":" in value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("artifacts", "modal")
        or relative.suffix != ".json"
    ):
        raise ValueError("--output must be a safe .json path below artifacts/modal")
    return Path(*relative.parts)


@app.local_entrypoint()
def main(
    output: str = "",
    skip_cache_prime: bool = False,
    confirm_paid_gpu: bool = False,
) -> None:
    """Prime the cache, run the native CUDA check, and validate its record."""

    if not confirm_paid_gpu:
        raise SystemExit(
            "No cache or GPU function was dispatched. Pass --confirm-paid-gpu "
            "to allocate the T4."
        )
    destination = _output_path(output)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite an existing path: {destination}")
    if not skip_cache_prime:
        prime_model_cache_v2.remote()
    record = validate_native_cuda_transactions.remote()
    _write_record(destination, record)
    checkout = Path(__file__).resolve().parents[1]
    runtime_tree = subprocess.run(
        _git_invocation(
            checkout,
            "show",
            "-s",
            "--format=%T",
            RUNTIME_COMMIT,
        ),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    validator = checkout / "tools/validate_modal_native_cuda_record.py"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(validator),
            str(destination),
            "--expected-runtime-commit",
            RUNTIME_COMMIT,
            "--expected-runtime-tree",
            runtime_tree,
        ],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(destination.resolve())
