"""Exercise lazy word alignment through the native transaction boundary."""

import math
import unittest
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import test_native_adapter as fixtures

from whisper_runtime import (
    AudioSpan,
    RequestState,
    RequestStatus,
    Session,
    TransactionRetainedError,
)
from whisper_runtime.adapters import (
    NativeDecodeContractError,
    NativeDependencyError,
    native_whisper,
)
from whisper_runtime.adapters.word_policy import (
    NativeWordAlignment,
    select_word_publication,
)


class Tokenizer:
    eot = 100
    timestamp_begin = 200

    def decode(self, tokens: list[int]) -> str:
        return "".join({1: " hello", 2: " world"}[token] for token in tokens)


class AlignableBatch:
    def __init__(self, mel: "AlignableMel") -> None:
        self.mel = mel
        self.indices: list[int] = []

    def __getitem__(self, index: int) -> "AlignableMel":
        self.indices.append(index)
        if index != 0:
            raise IndexError(index)
        return self.mel


class AlignableMel(fixtures.FakeMel):
    def __init__(self) -> None:
        super().__init__()
        self.batch = AlignableBatch(self)

    def unsqueeze(self, dimension: int) -> AlignableBatch:
        self.unsqueeze_calls.append(dimension)
        return self.batch


class AlignableCudaBatch(fixtures.FakeCudaBatchedMel):
    def __init__(self, source: "AlignableCudaMel") -> None:
        super().__init__(source)
        self.indices: list[int] = []

    def __getitem__(self, index: int) -> "AlignableCudaBatch":
        self.indices.append(index)
        if index != 0:
            raise IndexError(index)
        return self


class AlignableCudaMel(fixtures.FakeCudaMel):
    def __init__(self, runtime: fixtures.FakeCudaRuntime, events: list[str]) -> None:
        super().__init__(runtime, events)
        self.batch = AlignableCudaBatch(self)

    def unsqueeze(self, dimension: int) -> AlignableCudaBatch:
        self.unsqueeze_calls.append(dimension)
        return self.batch


class ResultRun(fixtures.FakeRun):
    def __init__(self, result: object) -> None:
        super().__init__(complete_after=1)
        self.result = result

    def finalize(self) -> list[object]:
        super().finalize()
        return [self.result]


class Harness(fixtures.BackendHarness):
    def task_type(self, model: object, options: object) -> fixtures.FakeTask:
        task = super().task_type(model, options)
        task.tokenizer = Tokenizer()
        return task


def raw_result(
    *,
    text: str = "hello world",
    tokens: list[int] | None = None,
    audio_features: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        tokens=[1, 2] if tokens is None else tokens,
        language="en",
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        temperature=0.0,
        compression_ratio=1.0,
        audio_features=audio_features,
    )


def timing(
    word: str,
    tokens: list[int],
    start: float,
    end: float,
    probability: float = 0.9,
) -> SimpleNamespace:
    return SimpleNamespace(
        word=word,
        tokens=tokens,
        start=start,
        end=end,
        probability=probability,
    )


class ImportHarness:
    def __init__(self, finder: Callable[..., object], *, request_local: bool = True):
        self.finder = finder
        self.request_local = request_local
        self.names: list[str] = []

    def __call__(self, name: str) -> object:
        self.names.append(name)
        if name == "whisper.model":
            return SimpleNamespace(
                _uses_request_local_alignment=lambda model: self.request_local
            )
        if name == "whisper.timing":
            return SimpleNamespace(find_alignment=self.finder)
        raise AssertionError(f"unexpected import: {name}")


class NativeWordAlignmentTests(unittest.TestCase):
    setUp = fixtures.NativeWhisperAdapterTests.setUp

    def request(self, number: int = 0) -> RequestState:
        return RequestState(f"request-{number}", "session-1", self.identity, rng_seed=7)

    def start(
        self,
        result: object,
        *,
        start_ms: int = 2_000,
        end_ms: int = 3_000,
        request: RequestState | None = None,
        session: Session | None = None,
    ) -> tuple[
        native_whisper.NativeWindowRun,
        ResultRun,
        AlignableMel,
        Session,
        RequestState,
    ]:
        backend = ResultRun(result)
        mel = AlignableMel()
        state = session or Session("session-1")
        native_request = request or self.request()
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=Harness(
                [backend],
                torch_module=fixtures.FakeTorchModule(fixtures.FakeCudaRuntime([])),
            ).components(),
        ):
            run = self.adapter.start_window(
                session=state,
                request=native_request,
                window_id="window-1",
                mel=mel,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        return run, backend, mel, state, native_request

    def test_alignment_is_lazy_cached_and_commits_an_explicit_selection(self) -> None:
        run, backend, mel, session, _ = self.start(
            raw_result(tokens=[200, 1, 250, 250, 2])
        )
        calls: list[tuple[object, object, list[int], object, int]] = []

        def find_alignment(
            model: object,
            tokenizer: object,
            tokens: list[int],
            input_mel: object,
            frames: int,
        ) -> list[SimpleNamespace]:
            calls.append((model, tokenizer, tokens, input_mel, frames))
            return [
                timing(" hello", [1], 0.1, 0.4),
                timing(" world", [2], 0.4, 0.9),
            ]

        imports = ImportHarness(find_alignment)
        self.assertEqual(mel.batch.indices, [])
        self.assertEqual(backend.finalize_calls, 0)
        with run, patch.object(native_whisper, "import_module", side_effect=imports):
            run.step()
            aligned = run.prepare_word_alignment()
            self.assertIs(run.prepare_word_alignment(), aligned)
            publication = select_word_publication(
                aligned, 0, 2, AudioSpan(2_000, 2_900)
            )
            committed = run.finish(
                aligned_publication=publication,
                committed_through_ms=2_900,
            )

        self.assertEqual(backend.finalize_calls, 1)
        self.assertEqual(len(calls), 1)
        model, tokenizer, tokens, input_mel, frames = calls[0]
        self.assertIs(model, self.model)
        self.assertIsInstance(tokenizer, Tokenizer)
        self.assertEqual(tokens, [1, 2])
        self.assertIs(input_mel, mel)
        self.assertEqual(frames, 100)
        self.assertEqual(mel.batch.indices, [0])
        self.assertEqual(imports.names, ["whisper.model", "whisper.timing"])
        self.assertEqual(
            [(word.span.start_ms, word.span.end_ms) for word in aligned.words],
            [(2_100, 2_400), (2_400, 2_900)],
        )
        self.assertIs(committed.windows[-1].result, publication)
        self.assertEqual(committed.committed_through_ms, 2_900)
        self.assertIsNone(run._alignment_model)
        self.assertIsNone(run._alignment_batched_mel)

    def test_empty_result_needs_no_backend_import_or_tensor_slice(self) -> None:
        run, backend, mel, _, _ = self.start(
            raw_result(text="", tokens=[]), start_ms=0, end_ms=0
        )
        with run:
            run.step()
            aligned = run.prepare_word_alignment()
            self.assertEqual(aligned.words, ())
            run.finish()
        self.assertEqual(backend.finalize_calls, 1)
        self.assertEqual(mel.batch.indices, [])

    def test_default_finish_never_imports_or_slices_for_alignment(self) -> None:
        run, backend, mel, session, _ = self.start(raw_result())
        with (
            run,
            patch.object(
                native_whisper,
                "import_module",
                side_effect=AssertionError("alignment must remain opt in"),
            ),
        ):
            run.step()
            run.finish()
        self.assertEqual(backend.finalize_calls, 1)
        self.assertEqual(mel.batch.indices, [])
        self.assertEqual(session.snapshot().version, 1)
        self.assertIsNone(run._alignment_model)
        self.assertIsNone(run._alignment_batched_mel)

    def test_alignment_rejects_the_legacy_shared_hook_path(self) -> None:
        run, backend, mel, session, request = self.start(raw_result())
        finder_calls = 0

        def finder(*args: object) -> object:
            nonlocal finder_calls
            del args
            finder_calls += 1
            return []

        imports = ImportHarness(finder, request_local=False)
        with patch.object(native_whisper, "import_module", side_effect=imports):
            run.step()
            with self.assertRaisesRegex(
                NativeDependencyError, "request-local attention"
            ):
                run.prepare_word_alignment()
        self.assertEqual(finder_calls, 0)
        self.assertEqual(mel.batch.indices, [])
        self.assertTrue(run.closed)
        self.assertEqual(backend.cleanup_calls, 1)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(request.status, RequestStatus.ABORTED)

    def test_alignment_rejects_malformed_backend_words(self) -> None:
        cases = {
            "tokens": [timing(" hello world", [1], 0.1, 0.9)],
            "text": [timing(" wrong", [1, 2], 0.1, 0.9)],
            "overlap": [
                timing(" hello", [1], 0.1, 0.6),
                timing(" world", [2], 0.5, 0.9),
            ],
            "bounds": [
                timing(" hello", [1], 0.1, 0.4),
                timing(" world", [2], 0.4, 1.1),
            ],
            "finite": [
                timing(" hello", [1], 0.1, 0.4),
                timing(" world", [2], 0.4, math.inf),
            ],
        }
        for number, (name, values) in enumerate(cases.items()):
            with self.subTest(name=name):
                run, backend, _, session, request = self.start(
                    raw_result(), request=self.request(number)
                )
                imports = ImportHarness(lambda *args, values=values: values)
                with patch.object(native_whisper, "import_module", side_effect=imports):
                    run.step()
                    with self.assertRaises(NativeDecodeContractError):
                        run.prepare_word_alignment()
                self.assertTrue(run.closed)
                self.assertEqual(backend.finalize_calls, 1)
                self.assertEqual(backend.cleanup_calls, 1)
                self.assertEqual(session.snapshot().version, 0)
                self.assertEqual(request.status, RequestStatus.ABORTED)

    def test_aligned_publication_must_be_from_the_cached_run(self) -> None:
        run, backend, _, session, _ = self.start(raw_result())
        values = [
            timing(" hello", [1], 0.1, 0.4),
            timing(" world", [2], 0.4, 0.9),
        ]
        imports = ImportHarness(lambda *args: values)
        with run, patch.object(native_whisper, "import_module", side_effect=imports):
            run.step()
            aligned = run.prepare_word_alignment()
            copied = NativeWordAlignment(native=aligned.native, words=aligned.words)
            foreign = select_word_publication(copied, 0, 2, AudioSpan(2_000, 2_900))
            with self.assertRaisesRegex(ValueError, "cached word alignment"):
                run.finish(aligned_publication=foreign)
            self.assertFalse(run.closed)
            run.finish()
        self.assertEqual(backend.finalize_calls, 1)
        self.assertEqual(session.snapshot().version, 1)

    def test_aligned_publication_is_mutually_exclusive_and_bounds_finality(
        self,
    ) -> None:
        run, backend, _, _, _ = self.start(raw_result())
        values = [
            timing(" hello", [1], 0.1, 0.4),
            timing(" world", [2], 0.4, 0.9),
        ]
        imports = ImportHarness(lambda *args: values)
        with run, patch.object(native_whisper, "import_module", side_effect=imports):
            run.step()
            aligned = run.prepare_word_alignment()
            publication = select_word_publication(
                aligned, 0, 2, AudioSpan(2_000, 2_900)
            )
            with self.assertRaisesRegex(ValueError, "cannot be used together"):
                run.finish(
                    publication_span=AudioSpan(2_000, 2_900),
                    aligned_publication=publication,
                )
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                run.finish(
                    aligned_publication=publication,
                    committed_through_ms=2_901,
                )
            self.assertFalse(run.closed)
            run.finish()
        self.assertEqual(backend.finalize_calls, 1)

    def test_failed_fence_retains_alignment_inputs_until_recovery(self) -> None:
        run, backend, mel, session, _ = self.start(raw_result())
        values = [
            timing(" hello", [1], 0.1, 0.4),
            timing(" world", [2], 0.4, 0.9),
        ]
        imports = ImportHarness(lambda *args: values)
        with patch.object(native_whisper, "import_module", side_effect=imports):
            run.step()
            run.prepare_word_alignment()
            backend.fail_cleanup = True
            with self.assertRaises(TransactionRetainedError):
                run.finish()

        self.assertTrue(run.closed)
        self.assertFalse(run.capacity_released)
        self.assertIs(run._alignment_model, self.model)
        self.assertIs(run._alignment_batched_mel, mel.batch)
        self.assertEqual(session.snapshot().version, 0)
        backend.fail_cleanup = False
        self.assertTrue(run.stop())
        self.assertTrue(run.capacity_released)
        self.assertIsNone(run._alignment_model)
        self.assertIsNone(run._alignment_batched_mel)


class AudioFeatures:
    def __init__(self, *, device: fixtures.FakeDevice | None = None) -> None:
        self.shape = (1_500, 4)
        self.dtype = fixtures.FAKE_FLOAT32
        self.device = device or fixtures.FakeDevice("cpu")


class NativeAlignmentFeatureReuseTests(unittest.TestCase):
    request = NativeWordAlignmentTests.request
    start = NativeWordAlignmentTests.start

    def setUp(self) -> None:
        fixtures.NativeWhisperAdapterTests.setUp(self)
        self.model = fixtures.FakeNativeModel(self.identity)
        self.model.dims.n_audio_state = 4
        self.profile = native_whisper.NativeExecutionProfile(
            "feature-reuse", self.capacity, reuse_alignment_features=True
        )
        self.adapter = native_whisper.NativeWhisperAdapter(
            self.worker, self.model, fixtures.probe, self.profile
        )

    @staticmethod
    def words() -> list[SimpleNamespace]:
        return [timing(" hello", [1], 0.1, 0.4), timing(" world", [2], 0.4, 0.9)]

    def test_only_own_finalized_features_are_cached_and_fenced_before_release(
        self,
    ) -> None:
        features = AudioFeatures()
        result = raw_result(audio_features=features)
        run, backend, _, session, _ = self.start(result)
        backend.audio_features = AudioFeatures()  # never borrow mutable run state
        calls = []

        def finder(*args: object, audio_features: object = None) -> object:
            calls.append(audio_features)
            self.assertIs(args[0], self.model)
            return self.words()

        self.assertIsNone(run._alignment_audio_features)
        with patch.object(
            native_whisper, "import_module", side_effect=ImportHarness(finder)
        ):
            run.step()
            run.prepare_result()
            self.assertIs(run._alignment_audio_features, features)
            result.audio_features = (
                AudioFeatures()
            )  # cached owner input cannot be replaced
            aligned = run.prepare_word_alignment()
            self.assertIs(run.prepare_word_alignment(), aligned)
            backend.on_cleanup = lambda: self.assertIs(
                run._alignment_audio_features, features
            )
            run.finish()
        self.assertEqual(calls, [features])
        self.assertEqual(backend.finalize_calls, 1)
        self.assertIsNone(run._alignment_audio_features)
        self.assertIsNone(run._backend_run)
        self.assertEqual(session.snapshot().version, 1)

    def test_opted_in_control_uses_legacy_signature_and_cannot_switch_cached_mode(
        self,
    ) -> None:
        run, backend, _, _, _ = self.start(raw_result(audio_features=AudioFeatures()))
        calls = []

        def legacy_finder(*args: object) -> object:
            calls.append(args)
            return self.words()

        with (
            run,
            patch.object(
                native_whisper,
                "import_module",
                side_effect=ImportHarness(legacy_finder),
            ),
        ):
            run.step()
            aligned = run.prepare_word_alignment(reuse_alignment_features=False)
            self.assertIs(
                run.prepare_word_alignment(reuse_alignment_features=False), aligned
            )
            with self.assertRaisesRegex(ValueError, "cannot change"):
                run.prepare_word_alignment()
            self.assertFalse(run.closed)
            run.finish()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 5)
        self.assertEqual(backend.finalize_calls, 1)

    def test_disabled_profile_cannot_be_upgraded_and_overrides_are_boolean(
        self,
    ) -> None:
        self.model = fixtures.FakeNativeModel(self.identity)
        self.adapter = native_whisper.NativeWhisperAdapter(
            self.worker,
            self.model,
            fixtures.probe,
            native_whisper.NativeExecutionProfile("default", self.capacity),
        )
        run, backend, _, _, _ = self.start(raw_result())
        with run:
            run.step()
            with self.assertRaisesRegex(ValueError, "opted-in profile"):
                run.prepare_word_alignment(reuse_alignment_features=True)
            for invalid in (0, 1, "true"):
                with self.subTest(value=invalid):
                    with self.assertRaisesRegex(TypeError, "boolean or None"):
                        run.prepare_word_alignment(reuse_alignment_features=invalid)
            self.assertFalse(run.closed)
        self.assertEqual(backend.finalize_calls, 0)

    def test_unsupported_backend_rejects_without_fallback_or_publication(self) -> None:
        for number, finder in enumerate(
            (lambda *args: self.words(), lambda *args, **kwargs: self.words())
        ):
            run, backend, _, session, _ = self.start(
                raw_result(audio_features=AudioFeatures()), request=self.request(number)
            )
            with patch.object(
                native_whisper, "import_module", side_effect=ImportHarness(finder)
            ):
                run.step()
                with self.assertRaisesRegex(
                    NativeDependencyError, "optional audio_features backend"
                ):
                    run.prepare_word_alignment()
            self.assertTrue(run.closed)
            self.assertIsNone(run._alignment_audio_features)
            self.assertIsNone(run._backend_run)
            self.assertEqual(backend.finalize_calls, 1)
            self.assertEqual(session.snapshot().version, 0)

    def test_missing_wrong_shape_precision_and_device_features_fail_closed(
        self,
    ) -> None:
        wrong_shape, wrong_dtype, wrong_device = (
            AudioFeatures(),
            AudioFeatures(),
            AudioFeatures(),
        )
        wrong_shape.shape = (1, 1_500, 4)
        wrong_dtype.dtype = object()
        wrong_device.device = fixtures.FakeDevice("cuda", 0)

        def finder(*args: object, audio_features: object = None) -> object:
            raise AssertionError("invalid features must not reach the backend")

        for number, features in enumerate(
            (None, wrong_shape, wrong_dtype, wrong_device)
        ):
            with self.subTest(features=features):
                run, _, _, session, request = self.start(
                    raw_result(audio_features=features), request=self.request(number)
                )
                with patch.object(
                    native_whisper, "import_module", side_effect=ImportHarness(finder)
                ):
                    run.step()
                    with self.assertRaisesRegex(
                        NativeDecodeContractError, "audio_features"
                    ):
                        run.prepare_word_alignment()
                self.assertTrue(run.capacity_released)
                self.assertIsNone(run._alignment_audio_features)
                self.assertEqual(session.snapshot().version, 0)
                self.assertEqual(request.status, RequestStatus.ABORTED)

    def test_failed_cleanup_retains_features_until_successful_recovery(self) -> None:
        features = AudioFeatures()
        run, backend, mel, session, _ = self.start(raw_result(audio_features=features))
        run.step()
        run.prepare_result()
        backend.fail_cleanup = True
        backend.on_cleanup = lambda: self.assertIs(
            run._alignment_audio_features, features
        )
        with self.assertRaises(TransactionRetainedError):
            run.close()
        self.assertFalse(run.capacity_released)
        self.assertIs(run._backend_run, backend)
        self.assertIs(run._alignment_audio_features, features)
        self.assertIs(run._alignment_batched_mel, mel.batch)
        backend.fail_cleanup = False
        self.assertTrue(run.stop())
        self.assertTrue(run.capacity_released)
        self.assertIsNone(run._backend_run)
        self.assertIsNone(run._alignment_audio_features)
        self.assertIsNone(run._alignment_batched_mel)
        self.assertEqual(session.snapshot().version, 0)

    def test_cancellation_does_not_release_features_before_owner_fence(self) -> None:
        features = AudioFeatures()
        run, backend, _, session, _ = self.start(raw_result(audio_features=features))
        run.step()
        run.prepare_result()
        self.assertTrue(run.cancel())
        self.assertIs(run._alignment_audio_features, features)
        self.assertEqual(backend.cleanup_calls, 0)
        run.close()
        self.assertTrue(run.capacity_released)
        self.assertIsNone(run._alignment_audio_features)
        self.assertIsNone(run._backend_run)
        self.assertEqual(session.snapshot().version, 0)

    def test_closed_run_does_not_keep_features_alive_or_reuse_them_in_next_window(
        self,
    ) -> None:
        features = AudioFeatures()
        reference = weakref.ref(features)
        run, backend, _, _, _ = self.start(raw_result(audio_features=features))
        run.step()
        run.prepare_result()
        run.close()
        del features, backend
        self.assertIsNone(reference())
        next_features = AudioFeatures()
        next_run, _, _, _, _ = self.start(
            raw_result(audio_features=next_features), request=self.request(1)
        )
        with next_run:
            self.assertIsNone(next_run._alignment_audio_features)
            next_run.step()
            next_run.prepare_result()
            self.assertIs(next_run._alignment_audio_features, next_features)
            self.assertIsNone(run._alignment_audio_features)


class NativeCudaWordAlignmentTests(unittest.TestCase):
    setUp = fixtures.NativeWhisperCudaAdapterTests.setUp
    make_adapter = fixtures.NativeWhisperCudaAdapterTests.make_adapter
    request = fixtures.NativeWhisperCudaAdapterTests.request

    def test_reused_features_stay_owned_through_failed_cuda_fence_and_recovery(
        self,
    ) -> None:
        events: list[str] = []
        runtime = fixtures.FakeCudaRuntime(events)
        adapter, _, budget, model = self.make_adapter(reuse_alignment_features=True)
        model.dims.n_audio_state = 4
        features = AudioFeatures(device=model.device)
        backend = ResultRun(raw_result(audio_features=features))
        harness = Harness([backend], torch_module=fixtures.FakeTorchModule(runtime))
        mel = AlignableCudaMel(runtime, events)
        streams = []

        def finder(*args: object, audio_features: object = None) -> object:
            self.assertIs(args[0], model)
            self.assertIs(audio_features, features)
            self.assertIsNotNone(runtime.active_stream)
            streams.append(runtime.active_stream)
            return NativeAlignmentFeatureReuseTests.words()

        with patch.object(
            native_whisper, "_load_native_components", return_value=harness.components()
        ):
            run = adapter.start_window(
                session=Session("cuda-session"),
                request=self.request(),
                window_id="cuda-feature-window",
                mel=mel,
                start_ms=0,
                end_ms=1_000,
            )
        original_wait = fixtures.FakeCudaEvent.synchronize

        def wait(event: fixtures.FakeCudaEvent) -> None:
            self.assertIs(run._alignment_audio_features, features)
            self.assertIs(run._backend_run, backend)
            original_wait(event)

        with (
            patch.object(
                native_whisper, "import_module", side_effect=ImportHarness(finder)
            ),
            patch.object(fixtures.FakeCudaEvent, "synchronize", new=wait),
        ):
            run.step()
            run.prepare_word_alignment()
            runtime.fail_event_synchronize = True
            with self.assertRaises(TransactionRetainedError):
                run.finish()
            self.assertFalse(run.capacity_released)
            self.assertIs(run._alignment_audio_features, features)
            runtime.fail_event_synchronize = False
            self.assertTrue(run.stop())
        self.assertEqual(streams, runtime.streams)
        self.assertTrue(run.capacity_released)
        self.assertIsNone(run._alignment_audio_features)
        self.assertIsNone(run._backend_run)
        self.assertEqual(budget.available, self.capacity)

    def test_alignment_uses_the_transaction_cuda_stream_and_fence(self) -> None:
        events: list[str] = []
        runtime = fixtures.FakeCudaRuntime(events)
        torch_module = fixtures.FakeTorchModule(runtime)
        backend = ResultRun(raw_result())
        harness = Harness([backend], torch_module=torch_module)
        adapter, _, budget, model = self.make_adapter()
        mel = AlignableCudaMel(runtime, events)
        observed_streams: list[object] = []

        def find_alignment(
            received_model: object,
            tokenizer: object,
            tokens: list[int],
            input_mel: object,
            frames: int,
        ) -> list[SimpleNamespace]:
            del tokenizer, tokens, frames
            self.assertIs(received_model, model)
            self.assertIs(input_mel, mel.batch)
            self.assertEqual(str(mel.batch.device), "cuda:1")
            self.assertIsNotNone(runtime.active_stream)
            observed_streams.append(runtime.active_stream)
            return [
                timing(" hello", [1], 0.1, 0.4),
                timing(" world", [2], 0.4, 0.9),
            ]

        imports = ImportHarness(find_alignment)
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=harness.components(),
        ):
            run = adapter.start_window(
                session=Session("cuda-session"),
                request=self.request(),
                window_id="cuda-window",
                mel=mel,
                start_ms=0,
                end_ms=1_000,
            )
        with run, patch.object(native_whisper, "import_module", side_effect=imports):
            run.step()
            run.prepare_word_alignment()
            run.finish()

        self.assertEqual(len(observed_streams), 1)
        self.assertIs(observed_streams[0], runtime.streams[0])
        self.assertEqual(mel.batch.indices, [0])
        self.assertIn("event:record", events)
        self.assertIn("event:synchronize", events)
        self.assertEqual(backend.cleanup_calls, 1)
        self.assertEqual(budget.available, self.capacity)


if __name__ == "__main__":
    unittest.main()
