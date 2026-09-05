"""Typed digital-silence coverage reuses the native transaction/fence boundary."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import test_native_word_alignment as fixtures

from whisper_runtime import (
    AudioSpan,
    RequestStatus,
    RuntimeStateError,
    TransactionRetainedError,
)
from whisper_runtime.adapters import native_whisper
from whisper_runtime.adapters.audio_evidence import AudioObservation, SilencePublication
from whisper_runtime.adapters.native_result import NativeTimestampSegment
from whisper_runtime.adapters.word_policy import (
    NativeWordAlignment,
    select_word_publication,
)


class NativeSilencePublicationTests(unittest.TestCase):
    setUp = fixtures.NativeWordAlignmentTests.setUp
    request = fixtures.NativeWordAlignmentTests.request
    start = fixtures.NativeWordAlignmentTests.start

    def publication(self, run, *, samples=None):
        native = run.prepare_result()
        span = native.analyzed_span
        count = (span.end_ms - span.start_ms) * 16 if samples is None else samples
        return SilencePublication(
            window_id=native.window_id,
            text="",
            start_ms=span.start_ms,
            end_ms=span.end_ms,
            analysis_span=span,
            native=native,
            observation=AudioObservation.from_pcm(bytes(count * 2)),
            analysis_start_sample=span.start_ms * 16,
        )

    def test_silence_commits_empty_text_without_alignment_or_rewriting_native(self):
        run, backend, mel, session, request = self.start(fixtures.raw_result())
        with (
            run,
            patch.object(
                native_whisper,
                "import_module",
                side_effect=AssertionError("no alignment needed"),
            ),
        ):
            run.step()
            publication = self.publication(run)
            committed = run.finish(
                silence_publication=publication, committed_through_ms=3_000
            )
        self.assertIs(committed.windows[-1].result, publication)
        self.assertEqual(committed.windows[-1].result.text, "")
        self.assertEqual(publication.native.text, "hello world")
        self.assertEqual(publication.native.metadata.tokens, (1, 2))
        self.assertEqual(backend.finalize_calls, 1)
        self.assertEqual(backend.cleanup_calls, 1)
        self.assertEqual(mel.batch.indices, [])
        self.assertEqual(session.snapshot().version, 1)
        self.assertEqual(committed.committed_through_ms, 3_000)
        self.assertEqual(request.status, RequestStatus.COMMITTED)
        self.assertTrue(run.capacity_released)

    def test_seven_sample_tail_commits_without_erasing_exact_observation_count(self):
        run, _, _, _, _ = self.start(fixtures.raw_result(), start_ms=0, end_ms=0)
        with run:
            run.step()
            publication = self.publication(run, samples=7)
            committed = run.finish(
                silence_publication=publication, committed_through_ms=0
            )
        self.assertEqual(committed.committed_through_ms, 0)
        self.assertEqual(committed.windows[-1].result.observation.sample_count, 7)

    def test_equal_copy_of_native_result_is_not_this_runs_cached_provenance(self):
        run, backend, _, session, _ = self.start(fixtures.raw_result())
        with run:
            run.step()
            publication = self.publication(run)
            copied = replace(publication, native=replace(publication.native))
            self.assertEqual(copied, publication)
            with self.assertRaisesRegex(ValueError, "cached native result"):
                run.finish(silence_publication=copied)
            self.assertFalse(run.closed)
            self.assertEqual(session.snapshot().version, 0)
            run.finish(silence_publication=publication)
        self.assertEqual(backend.finalize_calls, 1)

    def test_silence_selector_is_typed_mutually_exclusive_and_bounds_watermark(self):
        run, _, _, _, _ = self.start(fixtures.raw_result())
        with run:
            run.step()
            publication = self.publication(run)
            native = publication.native
            words = (
                NativeTimestampSegment(AudioSpan(2_000, 2_500), " hello", (1,)),
                NativeTimestampSegment(AudioSpan(2_500, 3_000), " world", (2,)),
            )
            aligned = select_word_publication(
                NativeWordAlignment(native, words), 0, 2, native.analyzed_span
            )
            with self.assertRaises(TypeError):
                run.finish(silence_publication=native)
            for other in (
                {"publication_span": native.analyzed_span},
                {"aligned_publication": aligned},
            ):
                with (
                    self.subTest(other=other),
                    self.assertRaisesRegex(ValueError, "cannot be used together"),
                ):
                    run.finish(silence_publication=publication, **other)
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                run.finish(silence_publication=publication, committed_through_ms=3_001)
            self.assertFalse(run.closed)
            run.finish(silence_publication=publication)

    def test_prepared_silence_publication_cannot_bypass_cancellation(self):
        run, backend, _, session, request = self.start(fixtures.raw_result())
        run.step()
        publication = self.publication(run)
        self.assertTrue(run.cancel())
        with self.assertRaises(RuntimeStateError):
            run.finish(silence_publication=publication)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(request.status, RequestStatus.CANCELLED)
        self.assertEqual(backend.finalize_calls, 1)
        self.assertTrue(run.capacity_released)

    def test_failed_cleanup_withholds_silence_commit_and_retains_until_recovery(self):
        run, backend, _, session, _ = self.start(fixtures.raw_result())
        run.step()
        publication = self.publication(run)
        backend.fail_cleanup = True
        with self.assertRaises(TransactionRetainedError) as raised:
            run.finish(silence_publication=publication)
        self.assertIsNone(raised.exception.committed_state)
        self.assertEqual(session.snapshot().version, 0)
        self.assertFalse(run.capacity_released)
        backend.fail_cleanup = False
        self.assertTrue(run.stop())
        self.assertTrue(run.capacity_released)
        self.assertEqual(session.snapshot().version, 0)

    def test_postcommit_release_failure_preserves_the_typed_committed_result(self):
        run, backend, _, session, _ = self.start(fixtures.raw_result())
        run.step()
        publication = self.publication(run)
        with patch.object(
            type(run._transaction._lease),
            "release",
            side_effect=RuntimeError("release failed"),
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                run.finish(silence_publication=publication)
        self.assertIsNotNone(raised.exception.committed_state)
        self.assertIs(raised.exception.committed_state.windows[-1].result, publication)
        self.assertEqual(session.snapshot().version, 1)
        self.assertFalse(run.capacity_released)
        self.assertTrue(run.stop())
        self.assertTrue(run.capacity_released)
        self.assertEqual(session.snapshot().version, 1)
        self.assertEqual(backend.finalize_calls, 1)


if __name__ == "__main__":
    unittest.main()
