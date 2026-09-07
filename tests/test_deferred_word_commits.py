"""Late hybrid fallback changes scheduling, never word-publication authority."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from test_continuous_endpoint_words import word_result
from test_continuous_evidence import EvidenceNativeAdapter, EvidenceNativeRun
from test_continuous_stream import pcm_ms

from whisper_runtime import TransactionRetainedError
from whisper_runtime.adapters import StreamEventKind
from whisper_runtime.adapters.audio_endpoints import QuietEndpointConfig
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)


class DeferredWordCommitTests(unittest.TestCase):
    def setUp(self):
        self.config = ContinuousStreamConfig(
            preview_interval_ms=1000,
            max_window_ms=5000,
            max_buffer_ms=8000,
            holdback_ms=1000,
            left_context_ms=1000,
            source_units=True,
            input_evidence=True,
            endpointing=QuietEndpointConfig(max_quiet_unit_ms=5000),
            word_boundary_fallback=True,
            defer_word_commits=True,
        )

    def stream(self, factory=word_result, **changes):
        adapter = EvidenceNativeAdapter(factory)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="deferred",
            config=replace(self.config, **changes),
            mel_builder=lambda content: content,
        )
        self.addCleanup(stream.close)
        return stream, adapter

    def drain(self, stream):
        events = []
        for _ in range(2000):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("unbounded controller drive")

    def test_strict_opt_in_and_profile_identity(self):
        self.assertIs(ContinuousStreamConfig().defer_word_commits, False)
        with self.assertRaises(ValueError):
            ContinuousStreamConfig(defer_word_commits=True)
        for value in (None, 1, "true"):
            with self.assertRaises(TypeError):
                replace(self.config, defer_word_commits=value)
        with self.assertRaises(ValueError):
            replace(self.config, preview_interval_ms=2500)
        old, _ = self.stream(defer_word_commits=False)
        new, _ = self.stream()
        self.assertEqual(
            new.profile_id,
            old.profile_id.replace(
                "+input_evidence", "+deferred_word_commit/v1+input_evidence"
            ),
        )

    def test_early_text_is_provisional_without_alignment_or_pcm_eviction(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(2000, 900))
        with patch.object(
            EvidenceNativeRun,
            "prepare_word_alignment",
            side_effect=AssertionError("early alignment"),
        ):
            events = self.drain(stream)
        self.assertTrue(any(event.text for event in events))
        self.assertFalse(any(event.kind is StreamEventKind.COMMIT for event in events))
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(stream.metrics.buffered_samples, 32000)
        self.assertEqual(stream.last_trace.reason, "word_commit_deferred")
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_last_two_growing_observations_use_ordinary_exact_policy(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(3000, 900))
        self.drain(stream)
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.last_trace.analysis_end_sample, 48000)
        self.assertIsNotNone(stream.last_trace.word_alignment)
        stream.push(1, pcm_ms(1000, 900))
        events = self.drain(stream)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.metrics.committed_samples, 3000 * 16)
        self.assertEqual(stream.retained_from_sample, 2000 * 16)
        self.assertEqual(stream.last_trace.reason, "candidate")
        self.assertTrue(any(event.kind is StreamEventKind.COMMIT for event in events))
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_coalescing_reserves_two_windows(self):
        stream, adapter = self.stream(coalesce_previews=True)
        stream.push(0, pcm_ms(5000, 900))
        self.drain(stream)
        self.assertEqual([call[2] for call in adapter.calls[:2]], [3000, 4000])
        self.assertEqual(stream.state.version, 1)

    def test_nondivisible_interval_cannot_skip_reserved_observation(self):
        stream, adapter = self.stream(preview_interval_ms=1500)
        stream.push(0, pcm_ms(5000, 900))
        self.drain(stream)
        self.assertEqual([call[2] for call in adapter.calls[:3]], [1500, 2000, 3500])
        self.assertGreater(stream.state.version, 0)

    def test_left_context_can_require_a_larger_reserve(self):
        stream, adapter = self.stream(left_context_ms=2000, coalesce_previews=True)
        stream.push(0, pcm_ms(4000, 900))
        self.drain(stream)
        self.assertEqual([call[2] for call in adapter.calls[:2]], [2000, 3000])
        self.assertGreater(stream.metrics.committed_samples, 0)

    def test_conflicting_score_at_reserved_second_observation_is_not_overridden(self):
        def conflicting(window_id, start, end):
            result = word_result(window_id, start, end)
            if end >= 4000:
                result = replace(
                    result, metadata=replace(result.metadata, no_speech_prob=0.8)
                )
            return result

        stream, _ = self.stream(conflicting)
        stream.push(0, pcm_ms(6000, 900))
        with self.assertRaises(StreamNeedsResolutionError):
            self.drain(stream)
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.metrics.buffered_samples, 6000 * 16)

    def test_eof_after_rebase_cannot_bypass_a_missing_published_anchor(self):
        def missing(window_id, start, end):
            result = word_result(window_id, start, end)
            if start > 0:
                words = tuple(
                    w for w in result.metadata.segments if w.span.start_ms >= 3000
                )
                result = replace(
                    result,
                    text="".join(w.text for w in words).strip(),
                    metadata=replace(
                        result.metadata,
                        segments=words,
                        tokens=tuple(t for w in words for t in w.tokens),
                    ),
                )
            return result

        stream, _ = self.stream(missing)
        stream.push(0, pcm_ms(4000, 900))
        self.drain(stream)
        head = stream.metrics.committed_samples
        stream.finish_input()
        with self.assertRaises(StreamNeedsResolutionError):
            self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, head)
        self.assertEqual(stream.state.version, 1)

    def test_quiet_and_eof_close_early_under_existing_gates(self):
        for quiet in (False, True):
            with self.subTest(quiet=quiet):
                stream, _ = self.stream()
                stream.push(0, pcm_ms(1000, 900))
                self.drain(stream)
                stream.push(1, pcm_ms(600, 0 if quiet else 900))
                if not quiet:
                    stream.finish_input()
                events = self.drain(stream)
                self.assertEqual(stream.metrics.committed_samples, 1600 * 16)
                self.assertEqual(
                    stream.last_trace.source_unit.origin,
                    "quiet_run" if quiet else "end_of_input",
                )
                self.assertEqual(
                    any(e.kind is StreamEventKind.FINAL for e in events), not quiet
                )

    def test_nonlexical_eof_remains_unresolved_with_all_pcm(self):
        def nonlexical(window_id, start, end):
            result = word_result(window_id, start, end)
            return replace(
                result,
                text="",
                metadata=replace(result.metadata, segments=(), tokens=()),
            )

        stream, _ = self.stream(nonlexical)
        stream.push(0, pcm_ms(1600, 64))
        stream.finish_input()
        with self.assertRaises(StreamNeedsResolutionError):
            self.drain(stream)
        self.assertEqual(stream.metrics.buffered_samples, 1600 * 16)
        self.assertEqual(stream.state.version, 0)

    def test_unstable_fallback_pair_does_not_force_a_cut(self):
        def unstable(window_id, start, end):
            result = word_result(window_id, start, end)
            if end >= 4000:
                words = tuple(
                    replace(w, text=w.text + str(end)) for w in result.metadata.segments
                )
                return replace(
                    result,
                    text="".join(w.text for w in words).strip(),
                    metadata=replace(result.metadata, segments=words),
                )
            return result

        stream, adapter = self.stream(unstable)
        stream.push(0, pcm_ms(6000, 900))
        with self.assertRaises(StreamNeedsResolutionError):
            self.drain(stream)
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.metrics.buffered_samples, 6000 * 16)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_33_seconds_without_quiet_uses_fallback_not_a_forced_30_second_stop(self):
        stream, _ = self.stream(max_window_ms=30000, max_buffer_ms=40000)
        stream.push(0, pcm_ms(33000, 900))
        events = self.drain(stream)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.metrics.committed_samples, 28000 * 16)
        stream.finish_input()
        events += self.drain(stream)
        self.assertTrue(stream.done)
        self.assertEqual(stream.metrics.committed_samples, 33000 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assertEqual(sum(e.kind is StreamEventKind.COMMIT for e in events), 2)

    def test_deferred_preview_after_rebase_excludes_published_words(self):
        stream, _ = self.stream(max_window_ms=8000, max_buffer_ms=10000)
        stream.push(0, pcm_ms(7000, 900))
        self.drain(stream)
        stream.push(1, pcm_ms(1000, 900))
        events = self.drain(stream)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.last_trace.reason, "word_commit_deferred")
        self.assertTrue(events[-1].text.startswith("w60 "))
        self.assertNotIn("w59", events[-1].text)

    def test_cleanup_failure_does_not_expose_a_deferred_preview(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(1000, 900))
        stream.step()
        run = stream._run
        stream.step()
        run.fence.fail = True
        with self.assertRaises(TransactionRetainedError):
            stream.step()
        self.assertEqual(stream.metrics.events_emitted, 0)
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.retained_from_sample, 0)
        run.fence.fail = False
        self.assertTrue(adapter.worker.recover(run.transaction))
        self.drain(stream)
        self.assertEqual(adapter.budget.available, adapter.capacity)


if __name__ == "__main__":
    unittest.main()
