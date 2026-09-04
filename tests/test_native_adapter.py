import random
import unittest
from collections.abc import Callable, Mapping
from threading import Event, Lock, Thread
from typing import cast
from unittest.mock import patch

from whisper_runtime import (
    Budget,
    ModelMismatchError,
    ModelSnapshot,
    RequestCancelledError,
    RequestState,
    RequestStatus,
    ResourceVector,
    Session,
    SessionState,
    TransactionRetainedError,
    TransactionStatus,
    WindowTransaction,
    Worker,
)
from whisper_runtime.adapters import (
    LegacyExecutionProfile,
    LegacyWhisperAdapter,
    NativeDecodeContractError,
    NativeDecodeOptions,
    NativeDependencyError,
    NativeExecutionProfile,
    NativeWhisperAdapter,
    native_whisper,
)


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> "FakeGenerator":
        self.seed = seed
        return self


FAKE_FLOAT32 = object()


class FakeDevice:
    def __init__(self, device_type: str, index: int | None = None) -> None:
        self.type = device_type
        self.index = index

    def __str__(self) -> str:
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"


class FakeMel:
    ndim = 2

    def __init__(
        self,
        device_type: str = "cpu",
        shape: tuple[int, int] = (80, 3_000),
        *,
        device_index: int | None = None,
        dtype: object = FAKE_FLOAT32,
    ) -> None:
        self.device = FakeDevice(device_type, device_index)
        self.dtype = dtype
        self.shape = shape
        self.unsqueeze_calls: list[int] = []

    def unsqueeze(self, dimension: int) -> tuple[str, "FakeMel"]:
        self.unsqueeze_calls.append(dimension)
        return ("batched", self)


class FakeCudaBatchedMel:
    def __init__(self, source: "FakeCudaMel") -> None:
        self.source = source
        self.device = FakeDevice("cpu")
        self.dtype = source.dtype

    def to(self, *, device: str, non_blocking: bool) -> "FakeCudaBatchedMel":
        if self.source.runtime.active_stream is None:
            raise AssertionError("the mel copy did not run on the CUDA stream")
        self.source.events.append("copy")
        self.source.copy_arguments.append((device, non_blocking))
        _, raw_index = device.split(":", maxsplit=1)
        self.device = FakeDevice("cuda", int(raw_index))
        return self


class FakeCudaMel(FakeMel):
    def __init__(
        self,
        runtime: "FakeCudaRuntime",
        events: list[str],
        *,
        dtype: object = FAKE_FLOAT32,
    ) -> None:
        super().__init__(dtype=dtype)
        self.runtime = runtime
        self.events = events
        self.copy_arguments: list[tuple[str, bool]] = []

    def unsqueeze(self, dimension: int) -> FakeCudaBatchedMel:
        self.unsqueeze_calls.append(dimension)
        return FakeCudaBatchedMel(self)

    def to(self, *, device: str, non_blocking: bool) -> object:
        del device, non_blocking
        raise AssertionError("the adapter must batch the mel before copying it")


class FakeCudaContext:
    def __init__(
        self,
        runtime: "FakeCudaRuntime",
        kind: str,
        value: object,
    ) -> None:
        self.runtime = runtime
        self.kind = kind
        self.value = value
        self.previous: object | None = None

    def __enter__(self) -> object:
        attribute = f"active_{self.kind}"
        self.previous = getattr(self.runtime, attribute)
        setattr(self.runtime, attribute, self.value)
        return self.value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback
        setattr(self.runtime, f"active_{self.kind}", self.previous)


class FakeCudaEvent:
    def __init__(self, runtime: "FakeCudaRuntime") -> None:
        self.runtime = runtime
        self.recorded_stream: object | None = None

    def record(self, stream: object) -> None:
        self.runtime.events.append("event:record")
        if self.runtime.active_device != "cuda:1":
            raise AssertionError("the event was recorded on the wrong device")
        if self.runtime.active_stream is not stream:
            raise AssertionError("the event was recorded outside its stream")
        self.recorded_stream = stream
        if self.runtime.fail_record:
            raise RuntimeError("event record failed")

    def synchronize(self) -> None:
        self.runtime.events.append("event:synchronize")
        if self.runtime.fail_event_synchronize:
            raise RuntimeError("event synchronize failed")


class FakeCudaRuntime:
    def __init__(self, events: list[str], *, device_count: int = 2) -> None:
        self.events = events
        self.visible_device_count = device_count
        self.available = True
        self.active_device: object | None = None
        self.active_stream: object | None = None
        self.fail_record = False
        self.fail_event_synchronize = False
        self.fail_event_creation = False
        self.streams: list[object] = []
        self.cuda_events: list[FakeCudaEvent] = []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.visible_device_count

    def synchronize(self, device: object | None = None) -> None:
        self.events.append(f"device:synchronize:{device}")

    def Stream(self, *, device: object | None = None) -> object:
        stream = object()
        self.events.append(f"stream:create:{device}")
        self.streams.append(stream)
        return stream

    def Event(self, *, enable_timing: bool = False) -> FakeCudaEvent:
        self.events.append(f"event:create:{enable_timing}")
        if self.fail_event_creation:
            raise RuntimeError("event creation failed")
        event = FakeCudaEvent(self)
        self.cuda_events.append(event)
        return event

    def device(self, device: object) -> FakeCudaContext:
        return FakeCudaContext(self, "device", device)

    def stream(self, stream: object) -> FakeCudaContext:
        return FakeCudaContext(self, "stream", stream)


class FakeTorchModule:
    float32 = FAKE_FLOAT32

    def __init__(self, cuda: FakeCudaRuntime) -> None:
        self.cuda = cuda


class FakeResult:
    def __init__(self, text: object) -> None:
        self.text = text


class FakeInference:
    def __init__(self, *, use_legacy_cache: bool = False) -> None:
        self._use_legacy_cache = use_legacy_cache


class FakeRun:
    def __init__(
        self,
        *,
        complete_after: int = 2,
        fail_stage: str | None = None,
        result_count: int = 1,
        result_text: object = " decoded text",
        on_prefill: Callable[[], None] | None = None,
        on_step: Callable[[], None] | None = None,
        on_finalize: Callable[[], None] | None = None,
        on_cleanup: Callable[[], None] | None = None,
        fail_cleanup: bool = False,
    ) -> None:
        self.complete_after = complete_after
        self.fail_stage = fail_stage
        self.result_count = result_count
        self.result_text = result_text
        self.on_prefill = on_prefill
        self.on_step = on_step
        self.on_finalize = on_finalize
        self.on_cleanup = on_cleanup
        self.fail_cleanup = fail_cleanup
        self.prefill_calls = 0
        self.step_calls = 0
        self.finalize_calls = 0
        self.cleanup_calls = 0
        self._complete = False
        self.inference = FakeInference()
        self._legacy_cache_lock: object | None = None

    @property
    def complete(self) -> bool:
        return self._complete

    def prefill(self) -> None:
        self.prefill_calls += 1
        if self.on_prefill is not None:
            self.on_prefill()
        if self.fail_stage == "prefill":
            raise RuntimeError("prefill failed")

    def step(self) -> bool:
        self.step_calls += 1
        if self.on_step is not None:
            self.on_step()
        if self.fail_stage == "step":
            raise RuntimeError("step failed")
        self._complete = self.step_calls >= self.complete_after
        return self._complete

    def finalize(self) -> list[object]:
        self.finalize_calls += 1
        if self.on_finalize is not None:
            self.on_finalize()
        if self.fail_stage == "finalize":
            raise RuntimeError("finalize failed")
        return [FakeResult(self.result_text) for _ in range(self.result_count)]

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.on_cleanup is not None:
            self.on_cleanup()
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")


class InvalidCleanableRun:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class InvalidRecoverableRun:
    def __init__(self) -> None:
        self.cleanup: object = None
        self.cleanup_calls = 0

    def enable_cleanup(self) -> None:
        def cleanup() -> None:
            self.cleanup_calls += 1

        self.cleanup = cleanup


class FakeTask:
    def __init__(self, harness: "BackendHarness") -> None:
        self._harness = harness
        self.inference = FakeInference(use_legacy_cache=harness.use_legacy_cache)

    def _uses_legacy_extension(self) -> bool:
        return self._harness.use_legacy_extension

    def _start_run(self, mel: object) -> object:
        if self._harness.on_start is not None:
            self._harness.on_start()
        self._harness.batched_mels.append(mel)
        if not self._harness.runs:
            raise AssertionError("no fake decode run remains")
        run = self._harness.runs.pop(0)
        if isinstance(run, FakeRun) and run.fail_stage == "start":
            raise RuntimeError("start failed")
        return run


class BackendHarness:
    def __init__(
        self,
        runs: list[object],
        *,
        use_legacy_extension: bool = False,
        use_legacy_cache: bool = False,
        on_start: Callable[[], None] | None = None,
        torch_module: object | None = None,
    ) -> None:
        self._runs = runs
        self.use_legacy_extension = use_legacy_extension
        self.use_legacy_cache = use_legacy_cache
        self.on_start = on_start
        self.torch_module = torch_module
        self.generators: list[FakeGenerator] = []
        self.option_kwargs: list[dict[str, object]] = []
        self.models: list[object] = []
        self.batched_mels: list[object] = []

    def generator_type(self, *, device: str) -> FakeGenerator:
        generator = FakeGenerator(device)
        self.generators.append(generator)
        return generator

    def options_type(self, **kwargs: object) -> object:
        self.option_kwargs.append(dict(kwargs))
        return object()

    def task_type(self, model: object, options: object) -> FakeTask:
        del options
        self.models.append(model)
        return FakeTask(self)

    @property
    def runs(self) -> list[object]:
        return self._runs

    def components(self) -> native_whisper._NativeComponents:
        return native_whisper._NativeComponents(
            generator_type=self.generator_type,
            options_type=self.options_type,
            task_type=self.task_type,
            n_frames=3_000,
            torch_module=cast(native_whisper._TorchModule, self.torch_module),
        )


class FakeNativeModel:
    def __init__(
        self,
        identity: ModelSnapshot,
        *,
        device: FakeDevice | None = None,
    ) -> None:
        self.identity = identity
        self.device = device or FakeDevice("cpu")
        self.dims = type("FakeDimensions", (), {"n_mels": 80})()

    def transcribe(
        self,
        audio: object,
        **decode_options: object,
    ) -> Mapping[str, object]:
        del audio, decode_options
        return {"text": "legacy-compatible"}


def probe(model: object) -> ModelSnapshot:
    return cast(ModelSnapshot, getattr(model, "identity"))


class NativeWhisperAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = ModelSnapshot(
            model_id="tiny.en",
            revision="suspendable-test",
            backend="pytorch-cpu",
            fingerprint="sha256:test-native-checkpoint",
        )
        self.capacity = ResourceVector(
            memory_bytes=1_024,
            compute_units=1,
            stream_slots=1,
        )
        self.budget = Budget(self.capacity)
        self.worker = Worker(
            "native-worker",
            self.identity,
            self.budget,
            queue_capacity=1,
        )
        self.profile = NativeExecutionProfile("tiny.en/cpu", self.capacity)
        self.model = FakeNativeModel(self.identity)
        self.adapter = NativeWhisperAdapter(
            self.worker,
            self.model,
            probe,
            self.profile,
        )

    def request(self, request_id: str = "request-1") -> RequestState:
        return RequestState(
            request_id,
            "session-1",
            self.identity,
            rng_seed=7,
        )

    def dual_adapter(
        self,
    ) -> tuple[
        NativeWhisperAdapter,
        Worker,
        Budget,
        ResourceVector,
    ]:
        transaction_cost = ResourceVector(
            memory_bytes=512,
            compute_units=1,
            stream_slots=1,
        )
        capacity = ResourceVector(
            memory_bytes=1_024,
            compute_units=2,
            stream_slots=2,
        )
        budget = Budget(capacity)
        worker = Worker(
            "native-dual-worker",
            self.identity,
            budget,
            queue_capacity=2,
        )
        adapter = NativeWhisperAdapter(
            worker,
            FakeNativeModel(self.identity),
            probe,
            NativeExecutionProfile(
                "tiny.en/cpu-dual",
                transaction_cost,
                max_concurrent_decodes=2,
            ),
        )
        return adapter, worker, budget, transaction_cost

    def decode(
        self,
        harness: BackendHarness,
        *,
        request: RequestState | None = None,
        session: Session | None = None,
        mel: FakeMel | None = None,
        options: NativeDecodeOptions | None = None,
    ) -> object:
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=harness.components(),
        ):
            return self.adapter.decode_window(
                session=session or Session("session-1"),
                request=request or self.request(),
                window_id="window-1",
                mel=mel or FakeMel(),
                start_ms=0,
                end_ms=30_000,
                options=options,
            )

    def test_submits_each_stage_and_commits_one_window(self) -> None:
        run = FakeRun(complete_after=2)
        harness = BackendHarness([run])
        request = self.request()
        mel = FakeMel()
        events: list[str] = []
        original_submit = WindowTransaction.submit
        original_checkpoint = WindowTransaction.checkpoint

        def tracked_submit(
            transaction: WindowTransaction,
            operation: Callable[[], object],
        ) -> object:
            events.append(f"submit:{getattr(operation, '__name__', 'unknown')}")
            return original_submit(transaction, operation)

        def tracked_checkpoint(transaction: WindowTransaction) -> None:
            events.append("checkpoint")
            original_checkpoint(transaction)

        with (
            patch.object(WindowTransaction, "submit", new=tracked_submit),
            patch.object(
                WindowTransaction,
                "checkpoint",
                new=tracked_checkpoint,
            ),
        ):
            committed = self.decode(
                harness,
                request=request,
                mel=mel,
                options=NativeDecodeOptions(temperature=0.2, best_of=2),
            )

        self.assertEqual(
            events,
            [
                "checkpoint",
                "submit:start_run",
                "checkpoint",
                "submit:prefill",
                "checkpoint",
                "submit:step",
                "checkpoint",
                "submit:step",
                "checkpoint",
                "submit:finalize",
                "checkpoint",
            ],
        )
        state = cast(object, committed)
        self.assertEqual(getattr(state, "version"), 1)
        windows = getattr(state, "windows")
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].result.text, " decoded text")
        self.assertEqual(request.status, RequestStatus.COMMITTED)
        self.assertEqual(self.budget.available, self.capacity)
        self.assertEqual(run.cleanup_calls, 1)
        self.assertEqual(mel.unsqueeze_calls, [0])
        self.assertEqual(harness.batched_mels, [("batched", mel)])
        self.assertEqual(len(harness.generators), 2)
        expected_seed = random.Random(7).randrange(1 << 63)
        self.assertEqual(harness.generators[0].seed, 0)
        self.assertEqual(harness.generators[1].seed, expected_seed)
        self.assertEqual(harness.generators[1].device, "cpu")
        self.assertIs(harness.option_kwargs[1]["generator"], harness.generators[1])
        self.assertIs(harness.option_kwargs[1]["fp16"], False)

    def test_dual_lane_serializes_start_and_overlaps_request_local_decode(
        self,
    ) -> None:
        adapter, worker, budget, transaction_cost = self.dual_adapter()
        first_start_entered = Event()
        release_first_start = Event()
        second_start_entered = Event()
        both_steps_entered = Event()
        release_steps = Event()
        counter_lock = Lock()
        start_calls = 0
        active_starts = 0
        maximum_active_starts = 0
        active_steps = 0
        maximum_active_steps = 0

        def on_start() -> None:
            nonlocal start_calls, active_starts, maximum_active_starts
            with counter_lock:
                start_calls += 1
                call = start_calls
                active_starts += 1
                maximum_active_starts = max(maximum_active_starts, active_starts)
            try:
                if call == 1:
                    first_start_entered.set()
                    if not release_first_start.wait(timeout=2):
                        raise AssertionError(
                            "first encoder preparation was not released"
                        )
                else:
                    second_start_entered.set()
            finally:
                with counter_lock:
                    active_starts -= 1

        def on_step() -> None:
            nonlocal active_steps, maximum_active_steps
            with counter_lock:
                active_steps += 1
                maximum_active_steps = max(maximum_active_steps, active_steps)
                if active_steps == 2:
                    both_steps_entered.set()
            try:
                if not both_steps_entered.wait(timeout=2):
                    raise AssertionError("both decoder steps did not overlap")
                if not release_steps.wait(timeout=2):
                    raise AssertionError("decoder steps were not released")
            finally:
                with counter_lock:
                    active_steps -= 1

        runs = [
            FakeRun(complete_after=1, result_text=" first", on_step=on_step),
            FakeRun(complete_after=1, result_text=" second", on_step=on_step),
        ]
        harness = BackendHarness(runs, on_start=on_start)
        results: dict[str, object] = {}
        errors: list[BaseException] = []
        result_lock = Lock()

        def decode(request_id: str) -> None:
            try:
                state = adapter.decode_window(
                    session=Session(f"session-{request_id}"),
                    request=RequestState(
                        request_id,
                        f"session-{request_id}",
                        self.identity,
                        rng_seed=7,
                    ),
                    window_id=f"window-{request_id}",
                    mel=FakeMel(),
                    start_ms=0,
                    end_ms=30_000,
                )
                with result_lock:
                    results[request_id] = state
            except BaseException as exc:
                with result_lock:
                    errors.append(exc)

        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=harness.components(),
        ):
            first = Thread(target=decode, args=("first",))
            second = Thread(target=decode, args=("second",))
            first.start()
            self.assertTrue(first_start_entered.wait(timeout=2))
            second.start()
            self.assertFalse(second_start_entered.wait(timeout=0.1))
            release_first_start.set()
            self.assertTrue(second_start_entered.wait(timeout=2))
            self.assertTrue(both_steps_entered.wait(timeout=2))
            self.assertEqual(worker.queue_depth, 2)
            self.assertEqual(budget.available, ResourceVector())
            release_steps.set()
            first.join(timeout=3)
            second.join(timeout=3)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(set(results), {"first", "second"})
        self.assertEqual(maximum_active_starts, 1)
        self.assertEqual(maximum_active_steps, 2)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, transaction_cost + transaction_cost)

    def test_dual_lane_requires_exact_queue_and_budget_capacity(self) -> None:
        transaction_cost = ResourceVector(
            memory_bytes=512,
            compute_units=1,
            stream_slots=1,
        )
        capacity = transaction_cost + transaction_cost
        profile = NativeExecutionProfile(
            "tiny.en/cpu-dual",
            transaction_cost,
            max_concurrent_decodes=2,
        )

        with self.assertRaisesRegex(ValueError, "queue must match"):
            NativeWhisperAdapter(
                Worker(
                    "wrong-queue",
                    self.identity,
                    Budget(capacity),
                    queue_capacity=1,
                ),
                FakeNativeModel(self.identity),
                probe,
                profile,
            )
        with self.assertRaisesRegex(ValueError, "worker capacity"):
            NativeWhisperAdapter(
                Worker(
                    "wrong-capacity",
                    self.identity,
                    Budget(transaction_cost),
                    queue_capacity=2,
                ),
                FakeNativeModel(self.identity),
                probe,
                profile,
            )
        with self.assertRaisesRegex(ValueError, "must be 1 or 2"):
            NativeExecutionProfile(
                "unsupported-concurrency",
                transaction_cost,
                max_concurrent_decodes=3,
            )

    def test_dual_lane_rejects_shared_backend_state_before_admission(self) -> None:
        adapter, worker, budget, transaction_cost = self.dual_adapter()
        cases = (
            (
                "legacy-extension",
                BackendHarness([], use_legacy_extension=True),
                "built-in suspendable decoder path",
            ),
            (
                "legacy-cache",
                BackendHarness([], use_legacy_cache=True),
                "request-local decoder cache support",
            ),
        )

        for request_id, harness, message in cases:
            with self.subTest(request_id=request_id):
                request = self.request(request_id)
                with patch.object(
                    native_whisper,
                    "_load_native_components",
                    return_value=harness.components(),
                ):
                    with self.assertRaisesRegex(NativeDependencyError, message):
                        adapter.decode_window(
                            session=Session("session-1"),
                            request=request,
                            window_id="window-1",
                            mel=FakeMel(),
                            start_ms=0,
                            end_ms=30_000,
                        )
                self.assertEqual(request.status, RequestStatus.CREATED)

        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, transaction_cost + transaction_cost)

    def test_dual_lane_cancellation_does_not_abort_its_peer(self) -> None:
        adapter, worker, budget, transaction_cost = self.dual_adapter()
        cancelled_request = RequestState(
            "cancelled",
            "session-cancelled",
            self.identity,
            rng_seed=7,
        )
        survivor_request = RequestState(
            "survivor",
            "session-survivor",
            self.identity,
            rng_seed=8,
        )
        first_step_entered = Event()
        both_steps_entered = Event()
        release_steps = Event()

        def cancel_step() -> None:
            first_step_entered.set()
            if not both_steps_entered.wait(timeout=2):
                raise AssertionError("survivor did not enter its decoder step")
            if not release_steps.wait(timeout=2):
                raise AssertionError("decoder steps were not released")
            cancelled_request.cancel()

        def survivor_step() -> None:
            both_steps_entered.set()
            if not release_steps.wait(timeout=2):
                raise AssertionError("decoder steps were not released")

        cancelled_run = FakeRun(complete_after=2, on_step=cancel_step)
        survivor_run = FakeRun(
            complete_after=1,
            result_text=" survivor",
            on_step=survivor_step,
        )
        harness = BackendHarness([cancelled_run, survivor_run])
        cancelled_errors: list[BaseException] = []
        survivor_states: list[SessionState] = []

        def decode_cancelled() -> None:
            try:
                adapter.decode_window(
                    session=Session("session-cancelled"),
                    request=cancelled_request,
                    window_id="window-cancelled",
                    mel=FakeMel(),
                    start_ms=0,
                    end_ms=30_000,
                )
            except BaseException as exc:
                cancelled_errors.append(exc)

        def decode_survivor() -> None:
            survivor_states.append(
                adapter.decode_window(
                    session=Session("session-survivor"),
                    request=survivor_request,
                    window_id="window-survivor",
                    mel=FakeMel(),
                    start_ms=0,
                    end_ms=30_000,
                )
            )

        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=harness.components(),
        ):
            cancelled_thread = Thread(target=decode_cancelled)
            survivor_thread = Thread(target=decode_survivor)
            cancelled_thread.start()
            self.assertTrue(first_step_entered.wait(timeout=2))
            survivor_thread.start()
            self.assertTrue(both_steps_entered.wait(timeout=2))
            self.assertEqual(worker.queue_depth, 2)
            self.assertEqual(budget.available, ResourceVector())
            release_steps.set()
            cancelled_thread.join(timeout=3)
            survivor_thread.join(timeout=3)

        self.assertFalse(cancelled_thread.is_alive())
        self.assertFalse(survivor_thread.is_alive())
        self.assertEqual(len(cancelled_errors), 1)
        self.assertIsInstance(cancelled_errors[0], RequestCancelledError)
        self.assertEqual(cancelled_request.status, RequestStatus.CANCELLED)
        self.assertEqual(cancelled_run.cleanup_calls, 1)
        self.assertEqual(len(survivor_states), 1)
        self.assertEqual(survivor_states[0].windows[0].result.text, " survivor")
        self.assertEqual(survivor_request.status, RequestStatus.COMMITTED)
        self.assertEqual(survivor_run.cleanup_calls, 1)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, transaction_cost + transaction_cost)

    def test_two_quarantines_remain_independently_recoverable(self) -> None:
        adapter, worker, budget, transaction_cost = self.dual_adapter()
        both_steps_entered = Event()
        release_steps = Event()
        step_lock = Lock()
        active_steps = 0

        def on_step() -> None:
            nonlocal active_steps
            with step_lock:
                active_steps += 1
                if active_steps == 2:
                    both_steps_entered.set()
            if not both_steps_entered.wait(timeout=2):
                raise AssertionError("both decoder steps did not overlap")
            if not release_steps.wait(timeout=2):
                raise AssertionError("decoder steps were not released")

        failed_runs = [
            FakeRun(complete_after=1, on_step=on_step, fail_cleanup=True),
            FakeRun(complete_after=1, on_step=on_step, fail_cleanup=True),
        ]
        successful_run = FakeRun(complete_after=1)
        harness = BackendHarness([*failed_runs, successful_run])
        retained: list[TransactionRetainedError] = []
        unexpected: list[BaseException] = []
        result_lock = Lock()

        def fail_decode(request_id: str) -> None:
            try:
                adapter.decode_window(
                    session=Session(f"session-{request_id}"),
                    request=RequestState(
                        request_id,
                        f"session-{request_id}",
                        self.identity,
                        rng_seed=7,
                    ),
                    window_id=f"window-{request_id}",
                    mel=FakeMel(),
                    start_ms=0,
                    end_ms=30_000,
                )
            except TransactionRetainedError as exc:
                with result_lock:
                    retained.append(exc)
            except BaseException as exc:
                with result_lock:
                    unexpected.append(exc)

        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=harness.components(),
        ):
            threads = [
                Thread(target=fail_decode, args=("failed-a",)),
                Thread(target=fail_decode, args=("failed-b",)),
            ]
            for thread in threads:
                thread.start()
            self.assertTrue(both_steps_entered.wait(timeout=2))
            release_steps.set()
            for thread in threads:
                thread.join(timeout=3)

            self.assertEqual(unexpected, [])
            self.assertEqual(len(retained), 2)
            self.assertEqual(worker.queue_depth, 2)
            self.assertEqual(budget.available, ResourceVector())

            for run in failed_runs:
                run.fail_cleanup = False
            self.assertTrue(worker.recover(retained[0].transaction))
            blocked_request = self.request("blocked-by-second-quarantine")
            with self.assertRaises(TransactionRetainedError):
                adapter.decode_window(
                    session=Session("session-1"),
                    request=blocked_request,
                    window_id="window-blocked",
                    mel=FakeMel(),
                    start_ms=0,
                    end_ms=30_000,
                )
            self.assertEqual(blocked_request.status, RequestStatus.CREATED)
            self.assertEqual(worker.queue_depth, 1)
            self.assertEqual(budget.available, transaction_cost)

            self.assertTrue(worker.recover(retained[1].transaction))
            committed = adapter.decode_window(
                session=Session("session-recovered"),
                request=RequestState(
                    "recovered",
                    "session-recovered",
                    self.identity,
                    rng_seed=7,
                ),
                window_id="window-recovered",
                mel=FakeMel(),
                start_ms=0,
                end_ms=30_000,
            )

        self.assertEqual(committed.version, 1)
        self.assertEqual(successful_run.cleanup_calls, 1)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, transaction_cost + transaction_cost)

    def test_cancellation_between_steps_cleans_without_commit(self) -> None:
        request = self.request()
        run = FakeRun(complete_after=3, on_step=request.cancel)
        session = Session("session-1")

        with self.assertRaises(RequestCancelledError):
            self.decode(
                BackendHarness([run]),
                request=request,
                session=session,
            )

        self.assertEqual(run.step_calls, 1)
        self.assertEqual(run.finalize_calls, 0)
        self.assertEqual(run.cleanup_calls, 1)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(session.snapshot().windows, ())
        self.assertEqual(request.status, RequestStatus.CANCELLED)
        self.assertEqual(self.budget.available, self.capacity)

    def test_stage_exceptions_clean_without_commit(self) -> None:
        for stage in ("prefill", "step", "finalize"):
            with self.subTest(stage=stage):
                identity = self.identity
                budget = Budget(self.capacity)
                worker = Worker(
                    f"worker-{stage}",
                    identity,
                    budget,
                    queue_capacity=1,
                )
                model = FakeNativeModel(identity)
                adapter = NativeWhisperAdapter(worker, model, probe, self.profile)
                request = RequestState(
                    f"request-{stage}",
                    "session-1",
                    identity,
                    rng_seed=7,
                )
                session = Session("session-1")
                run = FakeRun(complete_after=1, fail_stage=stage)
                harness = BackendHarness([run])
                with patch.object(
                    native_whisper,
                    "_load_native_components",
                    return_value=harness.components(),
                ):
                    with self.assertRaisesRegex(RuntimeError, f"{stage} failed"):
                        adapter.decode_window(
                            session=session,
                            request=request,
                            window_id="window-1",
                            mel=FakeMel(),
                            start_ms=0,
                            end_ms=1_000,
                        )

                self.assertEqual(run.cleanup_calls, 1)
                self.assertEqual(session.snapshot().version, 0)
                self.assertEqual(request.status, RequestStatus.ABORTED)
                self.assertEqual(budget.available, self.capacity)

    def test_start_exception_does_not_commit(self) -> None:
        run = FakeRun(fail_stage="start")
        request = self.request()
        session = Session("session-1")

        with self.assertRaisesRegex(RuntimeError, "start failed"):
            self.decode(
                BackendHarness([run]),
                request=request,
                session=session,
            )

        self.assertEqual(run.cleanup_calls, 0)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(self.budget.available, self.capacity)

    def test_aborted_retry_recreates_the_same_seed_with_a_fresh_generator(
        self,
    ) -> None:
        first_run = FakeRun(fail_stage="step")
        second_run = FakeRun(complete_after=1)
        harness = BackendHarness([first_run, second_run])

        with self.assertRaisesRegex(RuntimeError, "step failed"):
            self.decode(harness, request=self.request("request-first"))
        self.decode(harness, request=self.request("request-second"))

        self.assertEqual(len(harness.generators), 4)
        self.assertIsNot(harness.generators[1], harness.generators[3])
        self.assertEqual(harness.generators[1].seed, harness.generators[3].seed)

    def test_model_identity_is_checked_after_finalize(self) -> None:
        changed = ModelSnapshot(
            model_id="tiny.en",
            revision="changed",
            backend="pytorch-cpu",
            fingerprint="sha256:changed",
        )
        run = FakeRun(
            complete_after=1,
            on_finalize=lambda: setattr(self.model, "identity", changed),
        )
        request = self.request()
        session = Session("session-1")

        with self.assertRaises(ModelMismatchError):
            self.decode(
                BackendHarness([run]),
                request=request,
                session=session,
            )

        self.assertEqual(run.cleanup_calls, 1)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(request.status, RequestStatus.ABORTED)

    def test_model_identity_is_checked_before_run_creation(self) -> None:
        self.model.identity = ModelSnapshot(
            model_id="tiny.en",
            revision="changed-before-start",
            backend="pytorch-cpu",
            fingerprint="sha256:changed-before-start",
        )
        run = FakeRun()
        harness = BackendHarness([run])
        request = self.request()
        session = Session("session-1")

        with self.assertRaises(ModelMismatchError):
            self.decode(
                harness,
                request=request,
                session=session,
            )

        self.assertEqual(harness.generators, [])
        self.assertEqual(run.cleanup_calls, 0)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(request.status, RequestStatus.CREATED)

    def test_rejects_multiple_results_without_commit(self) -> None:
        run = FakeRun(complete_after=1, result_count=2)
        request = self.request()
        session = Session("session-1")
        with self.assertRaises(NativeDecodeContractError):
            self.decode(
                BackendHarness([run]),
                request=request,
                session=session,
            )
        self.assertEqual(run.cleanup_calls, 1)
        self.assertEqual(session.snapshot().version, 0)

    def test_legacy_and_native_cannot_bind_the_same_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "change adapter kind"):
            LegacyWhisperAdapter(
                self.worker,
                self.model,
                probe,
                LegacyExecutionProfile("tiny.en/cpu", self.capacity),
            )

    def test_constructor_does_not_load_optional_dependencies(self) -> None:
        other_model = FakeNativeModel(self.identity)
        with patch.object(
            native_whisper,
            "_load_native_components",
            side_effect=AssertionError("dependencies loaded during construction"),
        ):
            NativeWhisperAdapter(
                self.worker,
                other_model,
                probe,
                self.profile,
            )

    def test_backend_loader_normalizes_binary_import_failures(self) -> None:
        with patch.object(
            native_whisper,
            "import_module",
            side_effect=OSError("incompatible native library"),
        ):
            with self.assertRaisesRegex(NativeDependencyError, "requires PyTorch"):
                native_whisper._load_native_components()

    def test_preflight_interrupt_releases_the_shared_model_lock(self) -> None:
        request = self.request("preflight-interrupt")
        with patch.object(
            NativeWhisperAdapter,
            "_preflight_native_backend",
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.decode(BackendHarness([]), request=request)

        acquired = Event()

        def acquire_from_another_thread() -> None:
            if self.adapter._model_binding.lock.acquire(timeout=1):
                acquired.set()
                self.adapter._model_binding.lock.release()

        contender = Thread(target=acquire_from_another_thread)
        contender.start()
        contender.join(timeout=2)
        self.assertTrue(acquired.is_set())
        self.assertFalse(contender.is_alive())
        self.assertEqual(request.status, RequestStatus.CREATED)

    def test_rejects_batched_or_long_windows_before_admission(self) -> None:
        class BatchedMel:
            ndim = 3

            def unsqueeze(self, dimension: int) -> object:
                del dimension
                return object()

        with self.assertRaisesRegex(TypeError, "unbatched"):
            self.adapter.decode_window(
                session=Session("session-1"),
                request=self.request("batched"),
                window_id="window-1",
                mel=BatchedMel(),
                start_ms=0,
                end_ms=30_000,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed 30 seconds"):
            self.adapter.decode_window(
                session=Session("session-1"),
                request=self.request("long"),
                window_id="window-1",
                mel=FakeMel(),
                start_ms=0,
                end_ms=30_001,
            )
        self.assertEqual(self.worker.queue_depth, 0)

    def test_rejects_incompatible_mel_shape_before_admission(self) -> None:
        request = self.request("wrong-mel-shape")
        harness = BackendHarness([])
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=harness.components(),
        ):
            with self.assertRaisesRegex(ValueError, "mel shape must be"):
                self.adapter.decode_window(
                    session=Session("session-1"),
                    request=request,
                    window_id="window-1",
                    mel=FakeMel(shape=(80, 2_999)),
                    start_ms=0,
                    end_ms=30_000,
                )

        self.assertEqual(request.status, RequestStatus.CREATED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_rejects_non_cpu_model_and_mel_before_admission(self) -> None:
        with self.assertRaisesRegex(ValueError, "CPU device for the mel tensor"):
            self.adapter.decode_window(
                session=Session("session-1"),
                request=self.request("cuda-mel"),
                window_id="window-1",
                mel=FakeMel("cuda"),
                start_ms=0,
                end_ms=30_000,
            )

        self.model.device = FakeDevice("cuda")
        with self.assertRaisesRegex(ValueError, "CPU device for the model"):
            self.adapter.decode_window(
                session=Session("session-1"),
                request=self.request("cuda-model"),
                window_id="window-1",
                mel=FakeMel(),
                start_ms=0,
                end_ms=30_000,
            )
        self.assertEqual(self.worker.queue_depth, 0)

    def test_rejects_model_without_an_explicit_device_before_admission(self) -> None:
        del self.model.device
        request = self.request("missing-model-device")
        with self.assertRaisesRegex(ValueError, "CPU device for the model"):
            self.adapter.decode_window(
                session=Session("session-1"),
                request=request,
                window_id="window-1",
                mel=FakeMel(),
                start_ms=0,
                end_ms=30_000,
            )
        self.assertEqual(request.status, RequestStatus.CREATED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_invalid_handle_is_bound_and_cleaned_before_release(self) -> None:
        run = InvalidCleanableRun()
        request = self.request("invalid-cleanable")
        session = Session("session-1")

        with self.assertRaises(NativeDecodeContractError):
            self.decode(
                BackendHarness([run]),
                request=request,
                session=session,
            )

        self.assertEqual(run.cleanup_calls, 1)
        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_missing_cleanup_quarantines_until_quiescence_can_be_proven(self) -> None:
        run = InvalidRecoverableRun()
        request = self.request("invalid-no-cleanup")
        session = Session("session-1")

        with self.assertRaises(TransactionRetainedError) as raised:
            self.decode(
                BackendHarness([run]),
                request=request,
                session=session,
            )

        error = raised.exception
        self.assertIsInstance(error.operation_error, NativeDecodeContractError)
        self.assertIsInstance(error.retention_error, NativeDecodeContractError)
        self.assertEqual(error.transaction.status, TransactionStatus.QUARANTINED)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(self.worker.queue_depth, 1)
        self.assertEqual(self.budget.lease_count, 1)

        run.enable_cleanup()
        self.assertTrue(self.worker.recover(error.transaction))
        self.assertEqual(run.cleanup_calls, 1)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_failing_cleanup_quarantines_and_does_not_commit(self) -> None:
        run = FakeRun(complete_after=1, fail_cleanup=True)
        request = self.request("failing-cleanup")
        session = Session("session-1")

        with self.assertRaises(TransactionRetainedError) as raised:
            self.decode(
                BackendHarness([run]),
                request=request,
                session=session,
            )

        error = raised.exception
        self.assertEqual(error.transaction.status, TransactionStatus.QUARANTINED)
        self.assertIsNone(error.committed_state)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(self.worker.queue_depth, 1)
        self.assertEqual(self.budget.lease_count, 1)
        self.assertGreaterEqual(run.cleanup_calls, 2)

        run.fail_cleanup = False
        self.assertTrue(self.worker.recover(error.transaction))
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_incompatible_stock_options_fail_before_admission(self) -> None:
        request = self.request("stock-options")

        def stock_options(*, fp16: bool) -> object:
            del fp16
            return object()

        components = native_whisper._NativeComponents(
            generator_type=BackendHarness([]).generator_type,
            options_type=stock_options,
            task_type=lambda model, options: object(),
            n_frames=3_000,
        )
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=components,
        ):
            with self.assertRaises(NativeDependencyError):
                self.adapter.decode_window(
                    session=Session("session-1"),
                    request=request,
                    window_id="window-1",
                    mel=FakeMel(),
                    start_ms=0,
                    end_ms=30_000,
                )

        self.assertEqual(request.status, RequestStatus.CREATED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_missing_suspendable_api_fails_before_admission(self) -> None:
        request = self.request("stock-task")
        harness = BackendHarness([])
        components = native_whisper._NativeComponents(
            generator_type=harness.generator_type,
            options_type=harness.options_type,
            task_type=lambda model, options: object(),
            n_frames=3_000,
        )
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=components,
        ):
            with self.assertRaises(NativeDependencyError):
                self.adapter.decode_window(
                    session=Session("session-1"),
                    request=request,
                    window_id="window-1",
                    mel=FakeMel(),
                    start_ms=0,
                    end_ms=30_000,
                )

        self.assertEqual(request.status, RequestStatus.CREATED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)


class NativeWhisperCudaAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = ModelSnapshot(
            model_id="tiny.en",
            revision="suspendable-cuda-test",
            backend="pytorch-cuda",
            fingerprint="sha256:test-native-cuda-checkpoint",
        )
        self.capacity = ResourceVector(
            memory_bytes=4_096,
            compute_units=1,
            stream_slots=1,
        )

    def make_adapter(
        self,
        *,
        model_device: FakeDevice | None = None,
    ) -> tuple[NativeWhisperAdapter, Worker, Budget, FakeNativeModel]:
        budget = Budget(self.capacity)
        worker = Worker(
            "native-cuda-worker",
            self.identity,
            budget,
            queue_capacity=1,
        )
        model = FakeNativeModel(
            self.identity,
            device=model_device or FakeDevice("cuda", 1),
        )
        adapter = NativeWhisperAdapter(
            worker,
            model,
            probe,
            NativeExecutionProfile(
                "tiny.en/cuda-float32",
                self.capacity,
                device="cuda:1",
            ),
        )
        return adapter, worker, budget, model

    def request(self, request_id: str = "cuda-request") -> RequestState:
        return RequestState(
            request_id,
            "cuda-session",
            self.identity,
            rng_seed=7,
        )

    def decode(
        self,
        adapter: NativeWhisperAdapter,
        harness: BackendHarness,
        mel: FakeMel,
        *,
        request: RequestState | None = None,
        session: Session | None = None,
    ) -> SessionState:
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=harness.components(),
        ):
            return adapter.decode_window(
                session=session or Session("cuda-session"),
                request=request or self.request(),
                window_id="cuda-window",
                mel=mel,
                start_ms=0,
                end_ms=30_000,
            )

    def test_cuda_profile_is_single_lane_and_uses_canonical_device(self) -> None:
        for device in ("cuda", "cuda:01", "CUDA:0", "cuda:-1", "cuda:0 "):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ValueError, "canonical 'cuda:N'"):
                    NativeExecutionProfile(
                        "bad-device",
                        self.capacity,
                        device=device,
                    )

        with self.assertRaisesRegex(ValueError, "one concurrent decode"):
            NativeExecutionProfile(
                "cuda-dual",
                self.capacity,
                max_concurrent_decodes=2,
                device="cuda:1",
            )

        invalid_resources = (
            (
                ResourceVector(memory_bytes=0, compute_units=1, stream_slots=1),
                "device memory",
            ),
            (
                ResourceVector(memory_bytes=1, compute_units=0, stream_slots=1),
                "compute capacity",
            ),
            (
                ResourceVector(memory_bytes=1, compute_units=1, stream_slots=2),
                "exactly one stream",
            ),
        )
        for resources, message in invalid_resources:
            with self.subTest(resources=resources):
                with self.assertRaisesRegex(ValueError, message):
                    NativeExecutionProfile(
                        "bad-resources",
                        resources,
                        device="cuda:1",
                    )

    def test_cuda_model_device_is_fixed_when_the_adapter_is_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires device 'cuda:1'"):
            self.make_adapter(model_device=FakeDevice("cuda", 0))

    def test_model_binding_rejects_cpu_cuda_rebinding(self) -> None:
        cases = (
            (FakeDevice("cpu"), "cpu", FakeDevice("cuda", 0), "cuda:0"),
            (FakeDevice("cuda", 0), "cuda:0", FakeDevice("cpu"), "cpu"),
        )
        for (
            initial_model_device,
            initial_profile_device,
            target_model_device,
            target_profile_device,
        ) in cases:
            with self.subTest(
                initial=initial_profile_device,
                target=target_profile_device,
            ):
                worker = Worker(
                    f"binding-{initial_profile_device}-worker",
                    self.identity,
                    Budget(self.capacity),
                    queue_capacity=1,
                )
                model = FakeNativeModel(
                    self.identity,
                    device=initial_model_device,
                )
                NativeWhisperAdapter(
                    worker,
                    model,
                    probe,
                    NativeExecutionProfile(
                        "fixed-profile",
                        self.capacity,
                        device=initial_profile_device,
                    ),
                )
                model.device = target_model_device

                with self.assertRaisesRegex(
                    ValueError,
                    "cannot use multiple execution profiles",
                ):
                    NativeWhisperAdapter(
                        worker,
                        model,
                        probe,
                        NativeExecutionProfile(
                            "fixed-profile",
                            self.capacity,
                            device=target_profile_device,
                        ),
                    )

    def test_model_binding_rejects_rebinding_to_another_cuda_device(self) -> None:
        worker = Worker(
            "binding-cuda-worker",
            self.identity,
            Budget(self.capacity),
            queue_capacity=1,
        )
        model = FakeNativeModel(self.identity, device=FakeDevice("cuda", 0))
        NativeWhisperAdapter(
            worker,
            model,
            probe,
            NativeExecutionProfile(
                "fixed-profile",
                self.capacity,
                device="cuda:0",
            ),
        )
        model.device = FakeDevice("cuda", 1)

        with self.assertRaisesRegex(
            ValueError,
            "cannot use multiple execution profiles",
        ):
            NativeWhisperAdapter(
                worker,
                model,
                probe,
                NativeExecutionProfile(
                    "fixed-profile",
                    self.capacity,
                    device="cuda:1",
                ),
            )

    def test_cuda_rejects_runtime_and_input_mismatches_before_admission(self) -> None:
        cases = (
            ("unavailable", "CUDA is not available", "runtime"),
            ("out-of-range", "not visible", "runtime"),
            ("cuda-input", "requires device 'cpu'", "input"),
            ("wrong-dtype", "CPU float32", "input"),
        )
        for name, message, kind in cases:
            with self.subTest(name=name):
                events: list[str] = []
                runtime = FakeCudaRuntime(
                    events,
                    device_count=1 if name == "out-of-range" else 2,
                )
                if name == "unavailable":
                    runtime.available = False
                torch_module = FakeTorchModule(runtime)
                adapter, worker, budget, _ = self.make_adapter()
                run = FakeRun(complete_after=1)
                harness = BackendHarness([run], torch_module=torch_module)
                if name == "cuda-input":
                    mel: FakeMel = FakeMel("cuda", device_index=1)
                elif name == "wrong-dtype":
                    mel = FakeCudaMel(runtime, events, dtype=object())
                else:
                    mel = FakeCudaMel(runtime, events)

                with patch.object(
                    native_whisper,
                    "_load_native_components",
                    return_value=harness.components(),
                ):
                    error = NativeDependencyError if kind == "runtime" else ValueError
                    with self.assertRaisesRegex(error, message):
                        adapter.decode_window(
                            session=Session("cuda-session"),
                            request=self.request(name),
                            window_id="cuda-window",
                            mel=mel,
                            start_ms=0,
                            end_ms=30_000,
                        )

                self.assertEqual(worker.queue_depth, 0)
                self.assertEqual(budget.available, self.capacity)
                self.assertEqual(events, [])

    def test_cpu_default_does_not_call_cuda(self) -> None:
        events: list[str] = []
        runtime = FakeCudaRuntime(events)
        identity = ModelSnapshot(
            model_id="tiny.en",
            revision="cpu-no-cuda",
            backend="pytorch-cpu",
            fingerprint="sha256:cpu-no-cuda",
        )
        capacity = ResourceVector(memory_bytes=1, compute_units=1, stream_slots=1)
        adapter = NativeWhisperAdapter(
            Worker("cpu-worker", identity, Budget(capacity), queue_capacity=1),
            FakeNativeModel(identity),
            probe,
            NativeExecutionProfile("cpu-default", capacity),
        )
        harness = BackendHarness(
            [FakeRun(complete_after=1)],
            torch_module=FakeTorchModule(runtime),
        )
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=harness.components(),
        ):
            adapter.decode_window(
                session=Session("cpu-session"),
                request=RequestState(
                    "cpu-request",
                    "cpu-session",
                    identity,
                    rng_seed=7,
                ),
                window_id="cpu-window",
                mel=FakeMel(),
                start_ms=0,
                end_ms=30_000,
            )
        self.assertEqual(events, [])

    def test_cuda_work_is_admitted_copied_and_fenced_before_commit(self) -> None:
        events: list[str] = []
        runtime = FakeCudaRuntime(events)
        torch_module = FakeTorchModule(runtime)
        adapter, worker, budget, _ = self.make_adapter()

        def require_stream(stage: str) -> None:
            self.assertEqual(runtime.active_device, "cuda:1")
            self.assertIs(runtime.active_stream, runtime.streams[0])
            events.append(stage)

        run = FakeRun(
            complete_after=1,
            on_prefill=lambda: require_stream("prefill"),
            on_step=lambda: require_stream("step"),
            on_finalize=lambda: require_stream("finalize"),
            on_cleanup=lambda: require_stream("cleanup"),
        )
        harness = BackendHarness(
            [run],
            on_start=lambda: require_stream("start"),
            torch_module=torch_module,
        )
        mel = FakeCudaMel(runtime, events)
        session = Session("cuda-session")
        scopes: list[native_whisper._CudaDecodeScope] = []
        original_scope = native_whisper._CudaDecodeScope
        original_acquire = Budget.acquire
        original_release = Budget._release
        original_commit = Session._commit

        class RecordingScope(original_scope):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                scopes.append(self)

        def tracked_acquire(
            target: Budget,
            resources: ResourceVector,
        ) -> object:
            events.append("lease:acquire")
            return original_acquire(target, resources)

        def tracked_commit(
            target: Session,
            expected_version: int,
            record: object,
        ) -> SessionState:
            events.append("session:commit")
            return original_commit(target, expected_version, record)

        def tracked_release(target: Budget, lease: object) -> None:
            events.append("lease:release")
            original_release(target, lease)

        with (
            patch.object(native_whisper, "_CudaDecodeScope", RecordingScope),
            patch.object(Budget, "acquire", new=tracked_acquire),
            patch.object(Budget, "_release", new=tracked_release),
            patch.object(Session, "_commit", new=tracked_commit),
        ):
            committed = self.decode(adapter, harness, mel, session=session)

        ordered = (
            "lease:acquire",
            "device:synchronize:cuda:1",
            "stream:create:cuda:1",
            "copy",
            "start",
            "prefill",
            "step",
            "finalize",
            "cleanup",
            "event:create:False",
            "event:record",
            "event:synchronize",
            "session:commit",
            "lease:release",
        )
        self.assertEqual(
            [events.index(item) for item in ordered],
            sorted(events.index(item) for item in ordered),
        )
        self.assertEqual(harness.generators[0].device, "cuda:1")
        self.assertEqual(mel.copy_arguments, [("cuda:1", False)])
        self.assertIs(runtime.cuda_events[0].recorded_stream, runtime.streams[0])
        self.assertEqual(committed.version, 1)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(len(scopes), 1)
        self.assertIsNone(scopes[0]._run)
        self.assertIsNone(scopes[0]._stream)
        self.assertIsNone(scopes[0]._event)

    def test_cuda_cancellation_still_waits_for_the_event(self) -> None:
        events: list[str] = []
        runtime = FakeCudaRuntime(events)
        adapter, worker, budget, _ = self.make_adapter()
        request = self.request("cancelled")
        run = FakeRun(
            complete_after=2,
            on_step=request.cancel,
            on_cleanup=lambda: events.append("cleanup"),
        )
        harness = BackendHarness(
            [run],
            torch_module=FakeTorchModule(runtime),
        )
        session = Session("cuda-session")

        with self.assertRaises(RequestCancelledError):
            self.decode(
                adapter,
                harness,
                FakeCudaMel(runtime, events),
                request=request,
                session=session,
            )

        self.assertLess(events.index("cleanup"), events.index("event:record"))
        self.assertLess(events.index("event:record"), events.index("event:synchronize"))
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(request.status, RequestStatus.CANCELLED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_request_stop_never_calls_cuda(self) -> None:
        events: list[str] = []
        runtime = FakeCudaRuntime(events)
        adapter, _, _, _ = self.make_adapter()
        scope = native_whisper._CudaDecodeScope(
            adapter._model_binding,
            FakeTorchModule(runtime),
            "cuda:1",
        )
        scope.request_stop()
        self.assertEqual(events, [])
        self.assertIsNone(scope._stream)

    def test_cuda_generator_must_use_the_profile_device(self) -> None:
        events: list[str] = []
        runtime = FakeCudaRuntime(events)
        adapter, worker, budget, _ = self.make_adapter()
        harness = BackendHarness(
            [FakeRun(complete_after=1)],
            torch_module=FakeTorchModule(runtime),
        )
        components = harness.components()
        mismatched = native_whisper._NativeComponents(
            generator_type=lambda **kwargs: FakeGenerator("cuda:0"),
            options_type=components.options_type,
            task_type=components.task_type,
            n_frames=components.n_frames,
            torch_module=components.torch_module,
        )
        request = self.request("wrong-generator")
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=mismatched,
        ):
            with self.assertRaisesRegex(NativeDependencyError, "profile device"):
                adapter.decode_window(
                    session=Session("cuda-session"),
                    request=request,
                    window_id="cuda-window",
                    mel=FakeCudaMel(runtime, events),
                    start_ms=0,
                    end_ms=30_000,
                )

        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)
        self.assertIn("event:synchronize", events)

    def test_cuda_fence_fault_matrix_blocks_reuse_until_exact_recovery(self) -> None:
        for failure in ("cleanup", "create", "record", "synchronize"):
            with self.subTest(failure=failure):
                events: list[str] = []
                runtime = FakeCudaRuntime(events)
                adapter, worker, budget, _ = self.make_adapter()
                run = FakeRun(complete_after=1, fail_cleanup=failure == "cleanup")
                runtime.fail_event_creation = failure == "create"
                runtime.fail_record = failure == "record"
                runtime.fail_event_synchronize = failure == "synchronize"
                harness = BackendHarness(
                    [run],
                    torch_module=FakeTorchModule(runtime),
                )
                session = Session("cuda-session")
                request = self.request(failure)

                with self.assertRaises(TransactionRetainedError) as raised:
                    self.decode(
                        adapter,
                        harness,
                        FakeCudaMel(runtime, events),
                        request=request,
                        session=session,
                    )

                retained = raised.exception
                scope = cast(
                    native_whisper._CudaDecodeScope,
                    retained.transaction._execution,
                )
                self.assertEqual(
                    retained.transaction.status,
                    TransactionStatus.QUARANTINED,
                )
                self.assertIs(scope._run, run)
                self.assertEqual(session.snapshot().version, 0)
                self.assertEqual(request.status, RequestStatus.RUNNING)
                self.assertEqual(worker.queue_depth, 1)
                self.assertEqual(budget.lease_count, 1)
                self.assertEqual(
                    budget.available,
                    ResourceVector(memory_bytes=0, compute_units=0, stream_slots=0),
                )
                self.assertEqual(worker.quarantined_count, 1)
                self.assertEqual(run.cleanup_calls, 2)

                blocked_request = self.request(f"blocked-{failure}")
                blocked_session = Session("cuda-session")
                events_before_blocked_attempt = tuple(events)
                with self.assertRaises(TransactionRetainedError) as blocked:
                    self.decode(
                        adapter,
                        harness,
                        FakeCudaMel(runtime, events),
                        request=blocked_request,
                        session=blocked_session,
                    )

                self.assertIs(blocked.exception, retained)
                self.assertEqual(blocked_request.status, RequestStatus.CREATED)
                self.assertEqual(blocked_session.snapshot().version, 0)
                self.assertEqual(tuple(events), events_before_blocked_attempt)
                self.assertEqual(worker.queue_depth, 1)
                self.assertEqual(budget.lease_count, 1)

                run.fail_cleanup = False
                runtime.fail_event_creation = False
                runtime.fail_record = False
                runtime.fail_event_synchronize = False
                self.assertTrue(worker.recover(retained.transaction))
                self.assertEqual(run.cleanup_calls, 3)
                self.assertIsNone(scope._run)
                self.assertIsNone(scope._stream)
                self.assertIsNone(scope._event)
                self.assertEqual(request.status, RequestStatus.ABORTED)
                self.assertEqual(session.snapshot().version, 0)
                self.assertEqual(worker.queue_depth, 0)
                self.assertEqual(worker.quarantined_count, 0)
                self.assertEqual(budget.lease_count, 0)
                self.assertEqual(budget.available, self.capacity)

                reuse_run = FakeRun(complete_after=1, result_text=f"reused-{failure}")
                harness.runs.append(reuse_run)
                reuse_request = self.request(f"reuse-{failure}")
                reuse_session = Session("cuda-session")
                state = self.decode(
                    adapter,
                    harness,
                    FakeCudaMel(runtime, events),
                    request=reuse_request,
                    session=reuse_session,
                )

                self.assertEqual(state.version, 1)
                self.assertEqual(state.windows[-1].result.text, f"reused-{failure}")
                self.assertEqual(reuse_request.status, RequestStatus.COMMITTED)
                self.assertEqual(worker.queue_depth, 0)
                self.assertEqual(worker.quarantined_count, 0)
                self.assertEqual(budget.lease_count, 0)
                self.assertEqual(budget.available, self.capacity)


if __name__ == "__main__":
    unittest.main()
