"""Exercise lazy word alignment through the native transaction boundary."""

import math
import unittest
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
    def __init__(
        self, runtime: fixtures.FakeCudaRuntime, events: list[str]
    ) -> None:
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
    *, text: str = "hello world", tokens: list[int] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        tokens=[1, 2] if tokens is None else tokens,
        language="en",
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        temperature=0.0,
        compression_ratio=1.0,
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
            return_value=Harness([backend]).components(),
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


class NativeCudaWordAlignmentTests(unittest.TestCase):
    setUp = fixtures.NativeWhisperCudaAdapterTests.setUp
    make_adapter = fixtures.NativeWhisperCudaAdapterTests.make_adapter
    request = fixtures.NativeWhisperCudaAdapterTests.request

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
