"""Import-safe tracing proxies for native CUDA qualification runs.

The module depends only on the Python standard library. Callers provide the
CUDA, Torch, backend, and runtime objects that the proxies observe. The
proxies record ordering and ownership facts; they do not replace runtime
state transitions.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from types import TracebackType
from typing import ParamSpec, TypeVar

_T = TypeVar("_T")
_P = ParamSpec("_P")
ResourceSnapshot = Callable[[object], dict[str, int]]


def _status(value: object) -> str:
    return str(getattr(value, "value", value))


def _stable_thread_role(thread_id: int, decode_thread_id: int | None) -> str:
    if decode_thread_id is not None and thread_id == decode_thread_id:
        return "decode"
    return "controller"


@dataclass(frozen=True, slots=True)
class StreamIdentity:
    """Stable identity for one CUDA stream within a trace."""

    device_type: str
    device_index: int
    native_stream: int

    @classmethod
    def from_stream(cls, stream: object) -> StreamIdentity:
        native_stream = getattr(stream, "cuda_stream", None)
        device = getattr(stream, "device", None)
        device_type = getattr(device, "type", None)
        device_index = getattr(device, "index", None)
        if (
            isinstance(native_stream, bool)
            or not isinstance(native_stream, int)
            or device_type != "cuda"
            or isinstance(device_index, bool)
            or not isinstance(device_index, int)
        ):
            raise RuntimeError("CUDA stream identity is unavailable")
        return cls(device_type, device_index, native_stream)


class FaultPoint(str, Enum):
    """Registered Python delegation boundaries in one CUDA transaction.

    An injected fault stops immediately before the named delegate call. It is
    not evidence of a native or partially completed CUDA operation.
    """

    CLEANUP = "cleanup"
    EVENT_CREATE = "event-create"
    EVENT_RECORD = "event-record"
    EVENT_SYNCHRONIZE = "event-synchronize"


@dataclass(slots=True)
class FaultPlan:
    """Deterministic boundary-fault counters used by tracing proxies."""

    _remaining: dict[FaultPoint, int] = field(default_factory=dict)
    _observed: dict[FaultPoint, int] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        injections: Mapping[FaultPoint | str, int] | None = None,
    ) -> None:
        self._remaining = {}
        self._observed = {}
        self._lock = threading.Lock()
        for point, count in (injections or {}).items():
            self.set_remaining(point, count)

    @staticmethod
    def _point(value: FaultPoint | str) -> FaultPoint:
        try:
            return FaultPoint(value)
        except ValueError as error:
            raise ValueError(f"unknown native CUDA fault point: {value!r}") from error

    @staticmethod
    def _count(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("fault count must be a non-negative integer")
        return value

    def set_remaining(self, point: FaultPoint | str, count: int) -> None:
        registered = self._point(point)
        validated = self._count(count)
        with self._lock:
            self._remaining[registered] = validated

    def remaining(self, point: FaultPoint | str) -> int:
        registered = self._point(point)
        with self._lock:
            return self._remaining.get(registered, 0)

    def observed(self, point: FaultPoint | str) -> int:
        registered = self._point(point)
        with self._lock:
            return self._observed.get(registered, 0)

    def consume(self, point: FaultPoint | str) -> bool:
        registered = self._point(point)
        with self._lock:
            remaining = self._remaining.get(registered, 0)
            if remaining == 0:
                return False
            self._remaining[registered] = remaining - 1
            self._observed[registered] = self._observed.get(registered, 0) + 1
            return True


class ScenarioTrace:
    """Collect ordered observations for one qualification scenario."""

    def __init__(self, name: str, *, fault_plan: FaultPlan | None = None) -> None:
        self.name = name
        self.started_ns = time.monotonic_ns()
        self.decode_thread_id: int | None = None
        self.events: list[dict[str, object]] = []
        self.stream_labels: dict[StreamIdentity, str] = {}
        self.event_count = 0
        self.cleanup_calls = 0
        self.fault_plan = fault_plan if fault_plan is not None else FaultPlan()
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

    @property
    def injected_sync_failures_remaining(self) -> int:
        """Compatibility view used by the version 2 evidence harness."""

        return self.fault_plan.remaining(FaultPoint.EVENT_SYNCHRONIZE)

    @injected_sync_failures_remaining.setter
    def injected_sync_failures_remaining(self, count: int) -> None:
        self.fault_plan.set_remaining(FaultPoint.EVENT_SYNCHRONIZE, count)

    @property
    def injected_sync_failures_observed(self) -> int:
        """Compatibility view used by the version 2 evidence harness."""

        return self.fault_plan.observed(FaultPoint.EVENT_SYNCHRONIZE)

    def set_decode_thread(self) -> None:
        self.decode_thread_id = threading.get_ident()

    def stream_label(self, stream: object) -> str:
        identity = StreamIdentity.from_stream(stream)
        with self._lock:
            label = self.stream_labels.get(identity)
            if label is None:
                label = f"stream-{len(self.stream_labels) + 1}"
                self.stream_labels[identity] = label
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


class TraceActivation:
    """Unforgeable-by-identity route captured when work is submitted."""

    __slots__ = ("_owner_thread_id", "_router", "_trace")

    def __init__(
        self,
        router: TraceRouter,
        trace: ScenarioTrace,
        owner_thread_id: int,
    ) -> None:
        self._router = router
        self._trace = trace
        self._owner_thread_id = owner_thread_id

    @property
    def trace(self) -> ScenarioTrace:
        return self._trace

    def wrap(self, operation: Callable[_P, _T]) -> Callable[_P, _T]:
        """Bind a submitted callable to this activation when it executes."""

        @wraps(operation)
        def bound(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            with self._router.bind(self):
                return operation(*args, **kwargs)

        return bound

    def bind(self) -> AbstractContextManager[ScenarioTrace]:
        """Bind this activation around work in the current thread."""

        return self._router.bind(self)


class TraceRouter:
    """Route proxy observations to one explicitly active scenario.

    A router can be shared with a worker thread. Nested or concurrent
    activations on the same router are rejected. Exiting an activation always
    clears the route, including when the scenario raises. Post-recovery reuse
    must start after the recovery activation closes.
    """

    def __init__(self) -> None:
        self._current: TraceActivation | None = None
        self._lock = threading.Lock()
        self._thread_binding = threading.local()

    @property
    def current(self) -> ScenarioTrace | None:
        with self._lock:
            activation = self._current
        if activation is None:
            return None
        return self.require()

    def require(self) -> ScenarioTrace:
        with self._lock:
            activation = self._current
        if activation is None:
            raise RuntimeError("no native CUDA validation scenario is active")
        if threading.get_ident() == activation._owner_thread_id:
            return activation.trace
        bound = getattr(self._thread_binding, "activation", None)
        if bound is not activation:
            raise RuntimeError(
                "the current thread is not bound to the active native CUDA scenario"
            )
        return activation.trace

    @contextmanager
    def activate(self, trace: ScenarioTrace) -> Iterator[TraceActivation]:
        activation = TraceActivation(self, trace, threading.get_ident())
        with self._lock:
            if self._current is not None:
                raise RuntimeError(
                    "a native CUDA validation scenario is already active"
                )
            self._current = activation
        try:
            yield activation
        finally:
            with self._lock:
                if self._current is activation:
                    self._current = None

    @contextmanager
    def bind(self, activation: TraceActivation) -> Iterator[ScenarioTrace]:
        """Bind cross-thread work to its captured activation identity."""

        if activation._router is not self:
            raise RuntimeError("the trace activation belongs to another router")
        with self._lock:
            current = self._current
        if current is not activation:
            raise RuntimeError("the native CUDA trace activation is stale")
        previous = getattr(self._thread_binding, "activation", None)
        if previous is not None and previous is not activation:
            raise RuntimeError(
                "the current thread is bound to another trace activation"
            )
        self._thread_binding.activation = activation
        try:
            yield activation.trace
        finally:
            if previous is None:
                del self._thread_binding.activation
            else:
                self._thread_binding.activation = previous


@dataclass(frozen=True, slots=True)
class RuntimeBindings:
    """Runtime state inspected at CUDA ownership boundaries."""

    worker: object
    budget: object
    session_getter: Callable[[], object]
    request_getter: Callable[[], object]
    resource_snapshot: ResourceSnapshot

    def session(self) -> object:
        return self.session_getter()

    def request(self) -> object:
        return self.request_getter()


class TracingCudaEvent:
    def __init__(
        self,
        delegate: object,
        router: TraceRouter,
        label: str,
        bindings: RuntimeBindings,
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._label = label
        self._bindings = bindings
        self._session = bindings.session()
        self._request = bindings.request()
        self._stream_label: str | None = None

    def record(self, stream: object) -> None:
        trace = self._router.require()
        label = trace.stream_label(stream)
        if (
            self._bindings.worker.queue_depth != 1
            or self._bindings.budget.lease_count != 1
        ):
            raise RuntimeError("the completion event was recorded without one lease")
        if self._session.snapshot().version != 0:
            raise RuntimeError("the session changed before the completion event")
        if trace.fault_plan.consume(FaultPoint.EVENT_RECORD):
            trace.record(
                f"cuda:{self._label}:record:injected-failure",
                kind="cuda",
                stream=label,
            )
            raise RuntimeError("injected CUDA event record failure")
        self._delegate.record(stream)
        self._stream_label = label
        trace.record(f"cuda:{self._label}:record", kind="cuda", stream=label)

    def synchronize(self) -> None:
        trace = self._router.require()
        if (
            self._bindings.worker.queue_depth != 1
            or self._bindings.budget.lease_count != 1
        ):
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
        if trace.fault_plan.consume(FaultPoint.EVENT_SYNCHRONIZE):
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
            "queue_depth": self._bindings.worker.queue_depth,
            "lease_count": self._bindings.budget.lease_count,
            "budget_available": self._bindings.resource_snapshot(
                self._bindings.budget.available
            ),
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


class TracingCuda:
    def __init__(
        self,
        delegate: object,
        router: TraceRouter,
        bindings: RuntimeBindings,
        *,
        device: str,
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._bindings = bindings
        self._device = device

    def is_available(self) -> bool:
        return bool(self._delegate.is_available())

    def device_count(self) -> int:
        return int(self._delegate.device_count())

    def synchronize(self, device: object | None = None) -> None:
        trace = self._router.require()
        if (
            self._bindings.worker.queue_depth != 1
            or self._bindings.budget.lease_count != 1
        ):
            raise RuntimeError("CUDA initialization ran before worker admission")
        trace.record("cuda:device-synchronize:begin", kind="cuda")
        self._delegate.synchronize(device)
        trace.record("cuda:device-synchronize:return", kind="cuda")

    def Stream(self, *, device: object | None = None) -> object:
        trace = self._router.require()
        if device != self._device:
            raise RuntimeError(f"unexpected CUDA stream device: {device!r}")
        if (
            self._bindings.worker.queue_depth != 1
            or self._bindings.budget.lease_count != 1
        ):
            raise RuntimeError("the CUDA stream was created before worker admission")
        stream = self._delegate.Stream(device=device)
        label = trace.stream_label(stream)
        trace.record("cuda:stream:create", kind="cuda", stream=label)
        return stream

    def Event(self, *, enable_timing: bool = False) -> TracingCudaEvent:
        trace = self._router.require()
        if enable_timing is not False:
            raise RuntimeError("the transaction fence must use a non-timing event")
        label = trace.next_event_label()
        if trace.fault_plan.consume(FaultPoint.EVENT_CREATE):
            trace.record(
                f"cuda:{label}:create:injected-failure",
                kind="cuda",
            )
            raise RuntimeError("injected CUDA event creation failure")
        event = self._delegate.Event(enable_timing=enable_timing)
        trace.record(f"cuda:{label}:create", kind="cuda")
        return TracingCudaEvent(event, self._router, label, self._bindings)

    def device(self, device: object) -> AbstractContextManager[object]:
        return TracingCudaContext(
            self._delegate.device(device),
            self._router,
            "cuda:device",
        )

    def stream(self, stream: object) -> AbstractContextManager[object]:
        label = self._router.require().stream_label(stream)
        return TracingCudaContext(
            self._delegate.stream(stream),
            self._router,
            "cuda:stream",
            stream=label,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class TracingCudaContext(AbstractContextManager[object]):
    def __init__(
        self,
        delegate: AbstractContextManager[object],
        router: TraceRouter,
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


class TorchProxy:
    def __init__(self, delegate: object, cuda: TracingCuda) -> None:
        self._delegate = delegate
        self.cuda = cuda
        self.float32 = delegate.float32

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


def require_current_stream(
    torch_module: object,
    trace: ScenarioTrace,
    *,
    device: str,
) -> str:
    if not trace.stream_labels:
        raise RuntimeError("a native stage ran before stream creation")
    current = torch_module.cuda.current_stream(device=device)
    label = trace.stream_label(current)
    if label != "stream-1":
        raise RuntimeError(f"a native stage ran on {label}, expected stream-1")
    return label


class RunProxy:
    def __init__(
        self,
        delegate: object,
        router: TraceRouter,
        torch_module: object,
        *,
        device: str,
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._torch = torch_module
        self._device = device

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
        label = require_current_stream(self._torch, trace, device=self._device)
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
            label = require_current_stream(self._torch, trace, device=self._device)
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
        label = require_current_stream(self._torch, trace, device=self._device)
        trace.record("run:cleanup:begin", kind="backend", stream=label)
        if trace.fault_plan.consume(FaultPoint.CLEANUP):
            trace.record(
                "run:cleanup:injected-failure",
                kind="backend",
                stream=label,
            )
            raise RuntimeError("injected native cleanup failure")
        self._delegate.cleanup()
        trace.record("run:cleanup:submitted", kind="backend", stream=label)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class TaskProxy:
    def __init__(
        self,
        delegate: object,
        router: TraceRouter,
        torch_module: object,
        *,
        device: str,
    ) -> None:
        self._delegate = delegate
        self._router = router
        self._torch = torch_module
        self._device = device

    @property
    def inference(self) -> object:
        return self._delegate.inference

    def _uses_legacy_extension(self) -> bool:
        return bool(self._delegate._uses_legacy_extension())

    def _start_run(self, mel: object) -> RunProxy:
        trace = self._router.require()
        label = require_current_stream(self._torch, trace, device=self._device)
        if str(getattr(mel, "device", "")) != self._device:
            raise RuntimeError("the native task did not receive the CUDA mel tensor")
        generator_device = str(getattr(self._delegate.options.generator, "device", ""))
        if generator_device != self._device:
            raise RuntimeError(
                f"the decode generator used {generator_device!r}, "
                f"expected {self._device!r}"
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
        trace.child_generators_on_device &= child_device == self._device
        trace.child_generators_distinct &= generators_distinct
        if child_device != self._device or not generators_distinct:
            raise RuntimeError(
                f"the decode run did not receive a distinct generator on {self._device}"
            )
        trace.record("run:child-generator:verified", kind="backend", stream=label)
        return RunProxy(run, self._router, self._torch, device=self._device)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)
