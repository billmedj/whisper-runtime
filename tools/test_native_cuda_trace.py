from __future__ import annotations

import os
import subprocess
import sys
import threading
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from infra.native_cuda_trace import (
    FaultPlan,
    FaultPoint,
    RunProxy,
    RuntimeBindings,
    ScenarioTrace,
    StreamIdentity,
    TaskProxy,
    TraceRouter,
    TracingCuda,
)


@dataclass(frozen=True)
class _Device:
    type: str = "cuda"
    index: int = 0


@dataclass(frozen=True)
class _Stream:
    cuda_stream: int
    device: _Device = _Device()


@dataclass(frozen=True)
class _Resources:
    memory_bytes: int = 7
    compute_units: int = 1
    stream_slots: int = 1


class _Session:
    def __init__(self, version: int = 0) -> None:
        self.version = version

    def snapshot(self) -> object:
        return SimpleNamespace(version=self.version)


class _DelegateEvent:
    def __init__(self, *, query_result: bool = True) -> None:
        self.query_result = query_result
        self.recorded_stream: object | None = None
        self.synchronize_calls = 0

    def record(self, stream: object) -> None:
        self.recorded_stream = stream

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def query(self) -> bool:
        return self.query_result


class _Context:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self) -> _Context:
        self.entered = True
        return self

    def __exit__(self, *args: object) -> None:
        self.exited = True


class _Cuda:
    def __init__(self, *, event_query_result: bool = True) -> None:
        self.current = _Stream(41)
        self.event_query_result = event_query_result
        self.events: list[_DelegateEvent] = []
        self.contexts: list[_Context] = []
        self.synchronize_calls: list[object | None] = []

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def synchronize(self, device: object | None = None) -> None:
        self.synchronize_calls.append(device)

    def Stream(self, *, device: object | None = None) -> _Stream:
        del device
        return self.current

    def Event(self, *, enable_timing: bool = False) -> _DelegateEvent:
        if enable_timing:
            raise AssertionError("timing events are not expected")
        event = _DelegateEvent(query_result=self.event_query_result)
        self.events.append(event)
        return event

    def current_stream(self, *, device: object | None = None) -> _Stream:
        del device
        return self.current

    def device(self, device: object) -> _Context:
        del device
        context = _Context()
        self.contexts.append(context)
        return context

    def stream(self, stream: object) -> _Context:
        del stream
        context = _Context()
        self.contexts.append(context)
        return context


class _Run:
    complete = False
    inference = object()
    _legacy_cache_lock = object()

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cleanup_calls = 0
        self.decoder = SimpleNamespace(generator=SimpleNamespace(device="cuda:0"))

    def prefill(self) -> None:
        self.calls.append("prefill")

    def step(self) -> bool:
        self.calls.append("step")
        return False

    def finalize(self) -> list[object]:
        self.calls.append("finalize")
        return ["segment"]

    def cleanup(self) -> None:
        self.calls.append("cleanup")
        self.cleanup_calls += 1


class _Task:
    inference = object()

    def __init__(
        self,
        *,
        option_device: str = "cuda:0",
        child_device: str = "cuda:0",
        child_alias: str | None = None,
    ) -> None:
        self.options = SimpleNamespace(generator=SimpleNamespace(device=option_device))
        self._generator_source = SimpleNamespace(device=option_device)
        self.run = _Run()
        if child_alias == "source":
            self.run.decoder.generator = self._generator_source
        elif child_alias == "option":
            self.run.decoder.generator = self.options.generator
        else:
            self.run.decoder.generator = SimpleNamespace(device=child_device)
        self.received_mel: object | None = None

    def _uses_legacy_extension(self) -> bool:
        return True

    def _start_run(self, mel: object) -> _Run:
        self.received_mel = mel
        return self.run


def _resource_snapshot(value: object) -> dict[str, int]:
    return {
        "memory_bytes": value.memory_bytes,
        "compute_units": value.compute_units,
        "stream_slots": value.stream_slots,
    }


def _bindings(
    *,
    queue_depth: int = 1,
    lease_count: int = 1,
    session_version: int = 0,
) -> RuntimeBindings:
    worker = SimpleNamespace(queue_depth=queue_depth)
    budget = SimpleNamespace(lease_count=lease_count, available=_Resources())
    session = _Session(session_version)
    request = SimpleNamespace(status=SimpleNamespace(value="running"))
    return RuntimeBindings(
        worker,
        budget,
        lambda: session,
        lambda: request,
        _resource_snapshot,
    )


class FaultPlanTests(unittest.TestCase):
    def test_fault_points_match_the_manifest_vocabulary(self) -> None:
        self.assertEqual(
            [point.value for point in FaultPoint],
            ["cleanup", "event-create", "event-record", "event-synchronize"],
        )
        plan = FaultPlan({"cleanup": 1, "event-synchronize": 2})
        self.assertEqual(plan.remaining("cleanup"), 1)
        self.assertEqual(plan.remaining("event-synchronize"), 2)

    def test_each_registered_point_consumes_exact_count(self) -> None:
        plan = FaultPlan({point: 2 for point in FaultPoint})

        for point in FaultPoint:
            with self.subTest(point=point):
                self.assertTrue(plan.consume(point))
                self.assertTrue(plan.consume(point.value))
                self.assertFalse(plan.consume(point))
                self.assertEqual(plan.remaining(point), 0)
                self.assertEqual(plan.observed(point), 2)

    def test_invalid_point_and_count_are_rejected(self) -> None:
        for count in (-1, True, 1.5):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    FaultPlan({FaultPoint.CLEANUP: count})  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValueError, "unknown native CUDA fault point"):
            FaultPlan({"not-registered": 1})

    def test_version_two_sync_compatibility_view(self) -> None:
        trace = ScenarioTrace("v2")
        trace.injected_sync_failures_remaining = 2

        self.assertTrue(trace.fault_plan.consume(FaultPoint.EVENT_SYNCHRONIZE))
        self.assertEqual(trace.injected_sync_failures_remaining, 1)
        self.assertEqual(trace.injected_sync_failures_observed, 1)

    def test_concurrent_consumers_cannot_exceed_the_plan(self) -> None:
        plan = FaultPlan({FaultPoint.EVENT_RECORD: 2})
        barrier = threading.Barrier(9)
        outcomes: list[bool] = []
        lock = threading.Lock()

        def consume() -> None:
            barrier.wait()
            result = plan.consume(FaultPoint.EVENT_RECORD)
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=consume) for _ in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(outcomes.count(True), 2)
        self.assertEqual(outcomes.count(False), 6)
        self.assertEqual(plan.observed(FaultPoint.EVENT_RECORD), 2)
        self.assertEqual(plan.remaining(FaultPoint.EVENT_RECORD), 0)


class ImportSafetyTests(unittest.TestCase):
    def test_fresh_import_does_not_load_gpu_or_backend_packages(self) -> None:
        command = """
import builtins
original_import = builtins.__import__
forbidden = {'modal', 'torch', 'whisper'}
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in forbidden:
        raise AssertionError(f'forbidden import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import infra.native_cuda_trace
"""
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class TraceRouterTests(unittest.TestCase):
    def test_route_is_visible_to_worker_thread(self) -> None:
        router = TraceRouter()
        trace = ScenarioTrace("threaded")
        finished = threading.Event()

        def record() -> None:
            router.require().record("worker:event", kind="test")
            finished.set()

        with router.activate(trace) as activation:
            thread = threading.Thread(target=activation.wrap(record))
            thread.start()
            thread.join(timeout=2)
            self.assertTrue(finished.is_set())
            self.assertIs(router.require(), trace)
        self.assertEqual(trace.names(), ["worker:event"])
        with self.assertRaisesRegex(RuntimeError, "no native CUDA"):
            router.require()

    def test_nested_activation_is_rejected_without_replacing_route(self) -> None:
        router = TraceRouter()
        outer = ScenarioTrace("outer")
        inner = ScenarioTrace("inner")

        with router.activate(outer):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with router.activate(inner):
                    self.fail("nested activation entered")
            self.assertIs(router.require(), outer)

    def test_exception_clears_route_and_next_activation_is_isolated(self) -> None:
        router = TraceRouter()
        first = ScenarioTrace("first")
        second = ScenarioTrace("second")

        with self.assertRaisesRegex(LookupError, "stop"):
            with router.activate(first):
                first.record("first:event", kind="test")
                raise LookupError("stop")
        self.assertIsNone(router.current)
        with router.activate(second):
            second.record("second:event", kind="test")
        self.assertEqual(first.names(), ["first:event"])
        self.assertEqual(second.names(), ["second:event"])
        self.assertIsNone(router.current)

    def test_unbound_delayed_worker_from_a_cannot_observe_b(self) -> None:
        router = TraceRouter()
        first = ScenarioTrace("a")
        second = ScenarioTrace("b")
        submitted = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def delayed_a() -> None:
            submitted.set()
            release.wait(timeout=2)
            try:
                router.require().record("late-a", kind="test")
            except BaseException as error:
                errors.append(error)

        with router.activate(first):
            thread = threading.Thread(target=delayed_a)
            thread.start()
            self.assertTrue(submitted.wait(timeout=2))
        with router.activate(second):
            release.set()
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(first.names(), [])
        self.assertEqual(second.names(), [])
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "not bound")

    def test_unbound_thread_cannot_bypass_binding_with_current(self) -> None:
        router = TraceRouter()
        trace = ScenarioTrace("active")
        errors: list[BaseException] = []

        def inspect_current() -> None:
            try:
                current = router.current
                if current is not None:
                    current.record("bypass", kind="test")
            except BaseException as error:
                errors.append(error)

        with router.activate(trace):
            thread = threading.Thread(target=inspect_current)
            thread.start()
            thread.join(timeout=2)

        self.assertEqual(trace.names(), [])
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "not bound")

    def test_captured_worker_from_a_is_stale_during_b(self) -> None:
        router = TraceRouter()
        first = ScenarioTrace("a")
        second = ScenarioTrace("b")
        errors: list[BaseException] = []

        def delayed_a() -> None:
            router.require().record("late-a", kind="test")

        with router.activate(first) as activation:
            bound_a = activation.wrap(delayed_a)

        def capture_error() -> None:
            try:
                bound_a()
            except BaseException as error:
                errors.append(error)

        with router.activate(second):
            thread = threading.Thread(target=capture_error)
            thread.start()
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(first.names(), [])
        self.assertEqual(second.names(), [])
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "stale")

    def test_recovery_activation_closes_before_reuse_activation(self) -> None:
        router = TraceRouter()
        recovery = ScenarioTrace("recovery")
        reuse = ScenarioTrace("reuse")

        with router.activate(recovery):
            recovery.record("runtime:manual-recovery:return", kind="runtime")
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with router.activate(reuse):
                    self.fail("reuse started inside recovery activation")
        with router.activate(reuse):
            reuse.record("worker:admitted", kind="runtime")

        self.assertEqual(recovery.names(), ["runtime:manual-recovery:return"])
        self.assertEqual(reuse.names(), ["worker:admitted"])


class StreamIdentityTests(unittest.TestCase):
    def test_identity_uses_device_type_index_and_native_handle(self) -> None:
        trace = ScenarioTrace("streams")
        first = _Stream(9, _Device("cuda", 0))
        same = _Stream(9, _Device("cuda", 0))
        other_device = _Stream(9, _Device("cuda", 1))
        other_handle = _Stream(10, _Device("cuda", 0))

        self.assertEqual(StreamIdentity.from_stream(first).native_stream, 9)
        self.assertEqual(trace.stream_label(first), "stream-1")
        self.assertEqual(trace.stream_label(same), "stream-1")
        self.assertEqual(trace.stream_label(other_device), "stream-2")
        self.assertEqual(trace.stream_label(other_handle), "stream-3")

    def test_invalid_identity_is_rejected(self) -> None:
        invalid = (
            SimpleNamespace(cuda_stream=True, device=_Device()),
            SimpleNamespace(cuda_stream=1, device=_Device("cpu", 0)),
            SimpleNamespace(cuda_stream=1, device=_Device("cuda", True)),
            object(),
        )
        for stream in invalid:
            with self.subTest(stream=stream):
                with self.assertRaisesRegex(RuntimeError, "identity is unavailable"):
                    StreamIdentity.from_stream(stream)


class BackendProxyTests(unittest.TestCase):
    def test_task_and_run_proxy_preserve_calls_results_and_event_order(self) -> None:
        router = TraceRouter()
        trace = ScenarioTrace("backend")
        cuda = _Cuda()
        torch = SimpleNamespace(cuda=cuda)
        task = _Task()
        proxy = TaskProxy(task, router, torch, device="cuda:0")
        mel = SimpleNamespace(device="cuda:0")

        trace.stream_label(cuda.current)
        with router.activate(trace):
            self.assertIs(proxy.inference, task.inference)
            self.assertTrue(proxy._uses_legacy_extension())
            run = proxy._start_run(mel)
            self.assertIs(run.inference, task.run.inference)
            self.assertIs(run._legacy_cache_lock, task.run._legacy_cache_lock)
            self.assertFalse(run.complete)
            self.assertIsNone(run.prefill())
            self.assertFalse(run.step())
            self.assertEqual(run.finalize(), ["segment"])
            self.assertIsNone(run.cleanup())

        self.assertIs(task.received_mel, mel)
        self.assertEqual(task.run.calls, ["prefill", "step", "finalize", "cleanup"])
        self.assertEqual(trace.child_generator_checks, 1)
        self.assertTrue(trace.child_generators_on_device)
        self.assertTrue(trace.child_generators_distinct)
        self.assertEqual(trace.cleanup_calls, 1)
        self.assertEqual(
            trace.names(),
            [
                "run:start:begin",
                "run:start:submitted",
                "run:child-generator:verified",
                "run:prefill:begin",
                "run:prefill:submitted",
                "run:step:begin",
                "run:step:submitted",
                "run:finalize:begin",
                "run:finalize:submitted",
                "run:cleanup:begin",
                "run:cleanup:submitted",
            ],
        )
        self.assertTrue(all(event["stream"] == "stream-1" for event in trace.events))

    def test_task_proxy_rejects_wrong_input_and_generator_devices(self) -> None:
        cases = (
            (_Task(), "cpu", "mel tensor"),
            (_Task(option_device="cpu"), "cuda:0", "decode generator"),
            (_Task(child_device="cpu"), "cuda:0", "distinct generator"),
            (_Task(child_alias="source"), "cuda:0", "distinct generator"),
            (_Task(child_alias="option"), "cuda:0", "distinct generator"),
        )
        for task, mel_device, message in cases:
            with self.subTest(message=message):
                router = TraceRouter()
                trace = ScenarioTrace("invalid-task")
                cuda = _Cuda()
                trace.stream_label(cuda.current)
                proxy = TaskProxy(
                    task,
                    router,
                    SimpleNamespace(cuda=cuda),
                    device="cuda:0",
                )
                with router.activate(trace):
                    with self.assertRaisesRegex(RuntimeError, message):
                        proxy._start_run(SimpleNamespace(device=mel_device))
                self.assertNotIn("run:child-generator:verified", trace.names())


class TracingCudaTests(unittest.TestCase):
    def _cuda(
        self,
        router: TraceRouter,
        *,
        bindings: RuntimeBindings | None = None,
        event_query_result: bool = True,
    ) -> tuple[TracingCuda, _Cuda]:
        delegate = _Cuda(event_query_result=event_query_result)
        proxy = TracingCuda(
            delegate,
            router,
            bindings or _bindings(),
            device="cuda:0",
        )
        return proxy, delegate

    def test_event_path_preserves_version_two_event_shape_and_state(self) -> None:
        router = TraceRouter()
        trace = ScenarioTrace("success")
        cuda, delegate = self._cuda(router)

        with router.activate(trace):
            stream = cuda.Stream(device="cuda:0")
            event = cuda.Event(enable_timing=False)
            event.record(stream)
            event.synchronize()

        self.assertEqual(
            trace.names(),
            [
                "cuda:stream:create",
                "cuda:event-1:create",
                "cuda:event-1:record",
                "cuda:event-1:synchronize:begin",
                "cuda:event-1:query:return",
                "cuda:event-1:synchronize:return",
            ],
        )
        self.assertEqual(delegate.events[0].synchronize_calls, 1)
        self.assertEqual(
            trace.events[-1]["state"],
            {
                "request_status": "running",
                "session_version": 0,
                "queue_depth": 1,
                "lease_count": 1,
                "budget_available": {
                    "memory_bytes": 7,
                    "compute_units": 1,
                    "stream_slots": 1,
                },
            },
        )

    def test_two_sync_faults_are_exact_and_do_not_reach_delegate(self) -> None:
        router = TraceRouter()
        plan = FaultPlan({FaultPoint.EVENT_SYNCHRONIZE: 2})
        trace = ScenarioTrace("fault", fault_plan=plan)
        cuda, delegate = self._cuda(router)

        with router.activate(trace):
            stream = cuda.Stream(device="cuda:0")
            for expected_event in (1, 2):
                event = cuda.Event()
                event.record(stream)
                with self.assertRaisesRegex(RuntimeError, "synchronization failure"):
                    event.synchronize()
                self.assertEqual(trace.event_count, expected_event)
            third = cuda.Event()
            third.record(stream)
            third.synchronize()

        self.assertEqual(plan.observed(FaultPoint.EVENT_SYNCHRONIZE), 2)
        self.assertEqual(plan.remaining(FaultPoint.EVENT_SYNCHRONIZE), 0)
        self.assertEqual(
            [event.synchronize_calls for event in delegate.events], [0, 0, 1]
        )
        self.assertEqual(
            trace.names().count("cuda:event-1:synchronize:injected-failure"),
            1,
        )
        self.assertEqual(
            trace.names().count("cuda:event-2:synchronize:injected-failure"),
            1,
        )

    def test_registered_create_and_record_faults_stop_before_delegate(self) -> None:
        for point, message in (
            (FaultPoint.EVENT_CREATE, "creation failure"),
            (FaultPoint.EVENT_RECORD, "record failure"),
        ):
            with self.subTest(point=point):
                router = TraceRouter()
                trace = ScenarioTrace("fault", fault_plan=FaultPlan({point: 1}))
                cuda, delegate = self._cuda(router)
                with router.activate(trace):
                    stream = cuda.Stream(device="cuda:0")
                    if point is FaultPoint.EVENT_CREATE:
                        with self.assertRaisesRegex(RuntimeError, message):
                            cuda.Event()
                        self.assertEqual(delegate.events, [])
                    else:
                        event = cuda.Event()
                        with self.assertRaisesRegex(RuntimeError, message):
                            event.record(stream)
                        self.assertIsNone(delegate.events[0].recorded_stream)

    def test_cleanup_fault_is_traced_and_does_not_reach_delegate(self) -> None:
        router = TraceRouter()
        trace = ScenarioTrace(
            "cleanup",
            fault_plan=FaultPlan({FaultPoint.CLEANUP: 1}),
        )
        cuda, delegate_cuda = self._cuda(router)
        run = _Run()
        torch = SimpleNamespace(cuda=delegate_cuda)
        proxy = RunProxy(run, router, torch, device="cuda:0")

        with router.activate(trace):
            cuda.Stream(device="cuda:0")
            with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
                proxy.cleanup()
        self.assertEqual(run.cleanup_calls, 0)
        self.assertEqual(trace.cleanup_calls, 1)
        self.assertEqual(
            trace.names()[-2:],
            ["run:cleanup:begin", "run:cleanup:injected-failure"],
        )

    def test_context_exit_is_recorded_when_body_raises(self) -> None:
        router = TraceRouter()
        trace = ScenarioTrace("context")
        cuda, delegate = self._cuda(router)

        with router.activate(trace):
            with self.assertRaisesRegex(ValueError, "body"):
                with cuda.device("cuda:0"):
                    raise ValueError("body")
        self.assertEqual(trace.names(), ["cuda:device:enter", "cuda:device:exit"])
        self.assertTrue(delegate.contexts[0].exited)

    def test_error_paths_fail_closed(self) -> None:
        router = TraceRouter()
        cuda, _ = self._cuda(router)
        with self.assertRaisesRegex(RuntimeError, "no native CUDA"):
            cuda.Stream(device="cuda:0")

        trace = ScenarioTrace("errors")
        with router.activate(trace):
            with self.assertRaisesRegex(RuntimeError, "unexpected CUDA stream"):
                cuda.Stream(device="cuda:1")
            with self.assertRaisesRegex(RuntimeError, "non-timing event"):
                cuda.Event(enable_timing=True)
            event = cuda.Event()
            with self.assertRaisesRegex(RuntimeError, "not recorded"):
                event.synchronize()

    def test_ownership_and_query_failures_are_not_published_as_fences(self) -> None:
        cases = (
            (_bindings(queue_depth=0), True, "before worker admission"),
            (_bindings(session_version=1), True, "session changed"),
            (_bindings(), False, "did not report completion"),
        )
        for bindings, query_result, message in cases:
            with self.subTest(message=message):
                router = TraceRouter()
                trace = ScenarioTrace("invalid")
                cuda, _ = self._cuda(
                    router,
                    bindings=bindings,
                    event_query_result=query_result,
                )
                with router.activate(trace):
                    if "before worker admission" in message:
                        with self.assertRaisesRegex(RuntimeError, message):
                            cuda.Stream(device="cuda:0")
                        continue
                    stream = cuda.Stream(device="cuda:0")
                    event = cuda.Event()
                    if "session changed" in message:
                        with self.assertRaisesRegex(RuntimeError, message):
                            event.record(stream)
                        continue
                    event.record(stream)
                    with self.assertRaisesRegex(RuntimeError, message):
                        event.synchronize()
                self.assertNotIn(
                    "cuda:event-1:synchronize:return",
                    trace.names(),
                )


if __name__ == "__main__":
    unittest.main()
