"""Exercise timed publication through the native transaction, not a text wrapper."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_native_adapter as fixtures

from whisper_runtime import (
    AudioSpan,
    RequestCancelledError,
    RequestState,
    RequestStatus,
    ResourceVector,
    Session,
    TransactionRetainedError,
)
from whisper_runtime.adapters import (
    NativeDecodeContractError,
    NativeDependencyError,
    NativeWindowResult,
    native_whisper,
)


class Tokenizer:
    eot = 100
    timestamp_begin = 200

    def decode(self, tokens: list[int]) -> str:
        return "".join({1: " old", 2: " new", 3: " tail"}[token] for token in tokens)


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


class NativePublicationTests(unittest.TestCase):
    setUp = fixtures.NativeWhisperAdapterTests.setUp

    def request(self, number: int) -> RequestState:
        return RequestState(f"request-{number}", "session-1", self.identity, rng_seed=7)

    def timed_result(self) -> SimpleNamespace:
        # Analysis is [17000, 26000]. The complete prefix ends at 24000.
        return SimpleNamespace(
            text="old new tail",
            tokens=[200, 1, 350, 350, 2, 550, 550, 3],
            language="en",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
            temperature=0.0,
            compression_ratio=1.0,
            audio_features=object(),
        )

    def initial_commit(self, session: Session) -> object:
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=Harness([fixtures.FakeRun(result_text="old")]).components(),
        ):
            return self.adapter.decode_window(
                session=session,
                request=self.request(0),
                window_id="initial",
                mel=fixtures.FakeMel(),
                start_ms=0,
                end_ms=20_000,
                committed_through_ms=20_000,
            )

    def test_overlap_publishes_only_new_complete_segment(self) -> None:
        session = Session("session-1")
        original = self.initial_commit(session)
        raw = self.timed_result()
        run = ResultRun(raw)
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=Harness([run]).components(),
        ):
            state = self.adapter.decode_window(
                session=session,
                request=self.request(1),
                window_id="overlap",
                mel=fixtures.FakeMel(),
                start_ms=17_000,
                end_ms=26_000,
                publication_span=AudioSpan(20_000, 24_000),
                committed_through_ms=24_000,
            )
        result = state.windows[-1].result
        self.assertIsInstance(result, NativeWindowResult)
        self.assertEqual(result.text, "new")
        self.assertEqual((result.start_ms, result.end_ms), (20_000, 24_000))
        self.assertEqual(result.analyzed_span, AudioSpan(17_000, 26_000))
        self.assertEqual(state.windows[0], original.windows[0])
        self.assertEqual(state.committed_through_ms, 24_000)
        self.assertEqual(result.metadata.segments[0].text, " old")
        self.assertFalse(result.metadata.timestamps_complete)
        self.assertEqual(result.publication_segment_indices, (1,))
        raw.tokens.clear()
        self.assertEqual(result.metadata.tokens, (200, 1, 350, 350, 2, 550, 550, 3))
        self.assertFalse(hasattr(result.metadata, "audio_features"))
        self.assertEqual(run.cleanup_calls, 1)
        self.assertEqual(self.budget.available, self.capacity)
        self.assertEqual(self.worker.queue_depth, 0)

    def test_overlap_cannot_republish_committed_text(self) -> None:
        session = Session("session-1")
        original = self.initial_commit(session)
        request = self.request(1)
        run = ResultRun(self.timed_result())
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=Harness([run]).components(),
        ):
            with self.assertRaisesRegex(ValueError, "overlap committed"):
                self.adapter.decode_window(
                    session=session,
                    request=request,
                    window_id="rewrite",
                    mel=fixtures.FakeMel(),
                    start_ms=17_000,
                    end_ms=26_000,
                    publication_span=AudioSpan(17_000, 20_000),
                )
        self.assertEqual(session.snapshot(), original)
        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(run.cleanup_calls, 1)
        self.assertEqual(self.budget.available, self.capacity)

    def test_bad_selection_or_metadata_aborts_without_publication(self) -> None:
        cases = [
            (self.timed_result(), AudioSpan(20_001, 24_000)),
            (fixtures.FakeResult("untimed"), AudioSpan(20_000, 24_000)),
            (SimpleNamespace(text="bad", language="en", tokens=[True]), None),
        ]
        for number, (raw, selection) in enumerate(cases):
            with self.subTest(number=number):
                session = Session("session-1")
                request = self.request(number)
                run = ResultRun(raw)
                with patch.object(
                    native_whisper,
                    "_load_native_components",
                    return_value=Harness([run]).components(),
                ):
                    with self.assertRaises(NativeDecodeContractError):
                        self.adapter.decode_window(
                            session=session,
                            request=request,
                            window_id="invalid",
                            mel=fixtures.FakeMel(),
                            start_ms=17_000,
                            end_ms=26_000,
                            publication_span=selection,
                        )
                self.assertEqual(session.snapshot().version, 0)
                self.assertEqual(request.status, RequestStatus.ABORTED)
                self.assertEqual(run.cleanup_calls, 1)
                self.assertEqual(self.budget.available, self.capacity)

    def test_invalid_outer_span_is_rejected_before_admission(self) -> None:
        harness = Harness([])
        with patch.object(
            native_whisper, "_load_native_components", return_value=harness.components()
        ):
            with self.assertRaises(ValueError):
                self.adapter.decode_window(
                    session=Session("session-1"),
                    request=self.request(0),
                    window_id="outside",
                    mel=fixtures.FakeMel(),
                    start_ms=17_000,
                    end_ms=26_000,
                    publication_span=AudioSpan(16_000, 24_000),
                )
        self.assertEqual(harness.generators, [])
        self.assertEqual(self.worker.queue_depth, 0)

    def test_default_retains_full_text_and_input_bounds(self) -> None:
        raw = self.timed_result()
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=Harness([ResultRun(raw)]).components(),
        ):
            state = self.adapter.decode_window(
                session=Session("session-1"),
                request=self.request(0),
                window_id="full",
                mel=fixtures.FakeMel(),
                start_ms=17_000,
                end_ms=26_000,
            )
        result = state.windows[0].result
        self.assertEqual(result.text, raw.text)
        self.assertEqual((result.start_ms, result.end_ms), (17_000, 26_000))
        self.assertIsNone(state.committed_through_ms)
        self.assertEqual(self.budget.available, self.capacity)

    def test_release_failure_preserves_text_and_metadata_in_recovery_handle(
        self,
    ) -> None:
        session = Session("session-1")
        raw = self.timed_result()
        backend_run = ResultRun(raw)
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=Harness([backend_run]).components(),
        ):
            run = self.adapter.start_window(
                session=session,
                request=self.request(0),
                window_id="retained",
                mel=fixtures.FakeMel(),
                start_ms=17_000,
                end_ms=26_000,
            )
            while not run.complete:
                run.step()
            with patch.object(
                ResourceVector, "__add__", side_effect=MemoryError("release failed")
            ):
                with self.assertRaises(TransactionRetainedError) as raised:
                    run.finish(publication_span=AudioSpan(20_000, 24_000))
        error = raised.exception
        self.assertIs(error.committed_state, session.snapshot())
        self.assertEqual(error.committed_state.windows[-1].result.text, "new")
        self.assertEqual(
            error.committed_state.windows[-1].result.metadata.tokens, tuple(raw.tokens)
        )
        self.assertFalse(run.capacity_released)
        self.assertTrue(self.worker.recover(error.transaction))
        self.assertTrue(run.capacity_released)
        self.assertEqual(session.snapshot().version, 1)
        self.assertEqual(backend_run.finalize_calls, 1)
        self.assertEqual(self.budget.available, self.capacity)

    def test_prepare_exposes_segments_without_publishing_or_finalizing_twice(
        self,
    ) -> None:
        session = Session("session-1")
        raw = self.timed_result()
        backend_run = ResultRun(raw)
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=Harness([backend_run]).components(),
        ):
            with self.adapter.start_window(
                session=session,
                request=self.request(0),
                window_id="prepared",
                mel=fixtures.FakeMel(),
                start_ms=17_000,
                end_ms=26_000,
            ) as run:
                with self.assertRaises(NativeDecodeContractError):
                    run.prepare_result()
                self.assertEqual(backend_run.finalize_calls, 0)
                while not run.complete:
                    run.step()
                prepared = run.prepare_result()
                self.assertIs(run.prepare_result(), prepared)
                self.assertEqual(session.snapshot().version, 0)
                self.assertFalse(run.capacity_released)
                selection = prepared.metadata.segments[1].span
                raw.tokens.clear()
                state = run.finish(publication_span=selection)
        published = state.windows[-1].result
        self.assertIs(published.metadata, prepared.metadata)
        self.assertEqual(published.publication_segment_indices, (1,))
        self.assertEqual(published.text, "new")
        self.assertEqual(backend_run.finalize_calls, 1)
        self.assertEqual(self.budget.available, self.capacity)

    def test_prepared_result_does_not_bypass_cancellation(self) -> None:
        session = Session("session-1")
        backend_run = ResultRun(self.timed_result())
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=Harness([backend_run]).components(),
        ):
            with self.adapter.start_window(
                session=session,
                request=self.request(0),
                window_id="cancelled",
                mel=fixtures.FakeMel(),
                start_ms=17_000,
                end_ms=26_000,
            ) as run:
                while not run.complete:
                    run.step()
                prepared = run.prepare_result()
                run.cancel()
                with self.assertRaises(RequestCancelledError):
                    run.finish(publication_span=prepared.metadata.segments[1].span)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(backend_run.finalize_calls, 1)
        self.assertEqual(self.budget.available, self.capacity)

    def test_nonstandard_audio_context_cannot_claim_standard_timestamps(self) -> None:
        self.model.dims.n_audio_ctx = 1_000
        harness = Harness([])
        request = self.request(0)
        with patch.object(
            native_whisper, "_load_native_components", return_value=harness.components()
        ):
            with self.assertRaisesRegex(NativeDependencyError, "standard Whisper"):
                self.adapter.start_window(
                    session=Session("session-1"),
                    request=request,
                    window_id="unsupported",
                    mel=fixtures.FakeMel(),
                    start_ms=0,
                    end_ms=5_000,
                )
        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)
