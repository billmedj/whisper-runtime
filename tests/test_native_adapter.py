import random
import unittest
from collections.abc import Callable, Mapping
from threading import Event, Thread
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


class FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class FakeMel:
    ndim = 2

    def __init__(
        self,
        device_type: str = "cpu",
        shape: tuple[int, int] = (80, 3_000),
    ) -> None:
        self.device = FakeDevice(device_type)
        self.shape = shape
        self.unsqueeze_calls: list[int] = []

    def unsqueeze(self, dimension: int) -> tuple[str, "FakeMel"]:
        self.unsqueeze_calls.append(dimension)
        return ("batched", self)


class FakeResult:
    def __init__(self, text: object) -> None:
        self.text = text


class FakeRun:
    def __init__(
        self,
        *,
        complete_after: int = 2,
        fail_stage: str | None = None,
        result_count: int = 1,
        result_text: object = " decoded text",
        on_step: Callable[[], None] | None = None,
        on_finalize: Callable[[], None] | None = None,
        fail_cleanup: bool = False,
    ) -> None:
        self.complete_after = complete_after
        self.fail_stage = fail_stage
        self.result_count = result_count
        self.result_text = result_text
        self.on_step = on_step
        self.on_finalize = on_finalize
        self.fail_cleanup = fail_cleanup
        self.prefill_calls = 0
        self.step_calls = 0
        self.finalize_calls = 0
        self.cleanup_calls = 0
        self._complete = False

    @property
    def complete(self) -> bool:
        return self._complete

    def prefill(self) -> None:
        self.prefill_calls += 1
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

    def _start_run(self, mel: object) -> object:
        self._harness.batched_mels.append(mel)
        if not self._harness.runs:
            raise AssertionError("no fake decode run remains")
        run = self._harness.runs.pop(0)
        if isinstance(run, FakeRun) and run.fail_stage == "start":
            raise RuntimeError("start failed")
        return run


class BackendHarness:
    def __init__(self, runs: list[object]) -> None:
        self._runs = runs
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
        )


class FakeNativeModel:
    def __init__(self, identity: ModelSnapshot) -> None:
        self.identity = identity
        self.device = FakeDevice("cpu")
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


if __name__ == "__main__":
    unittest.main()
