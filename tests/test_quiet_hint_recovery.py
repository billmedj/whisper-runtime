"""Discard rejected automatic hints, never PCM or frozen publication authority."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import test_continuous_endpoint_words as endpoint_fixtures
from test_audio_endpoints import short_config
from test_continuous_endpoint_words import replace_words, word_result
from test_continuous_evidence import EvidenceNativeAdapter
from test_continuous_stream import pcm_ms

from whisper_runtime import TransactionRetainedError
from whisper_runtime.adapters import NativeStreamError
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    SourceUnit,
    StreamNeedsResolutionError,
)
from whisper_runtime.adapters.word_policy import AnchorDiagnostic, WordAgreementDecision


class QuietHintRecoveryTests(unittest.TestCase):
    decode = endpoint_fixtures.ContinuousEndpointWordTests.decode
    drain = endpoint_fixtures.ContinuousEndpointWordTests.drain
    assert_coverage = endpoint_fixtures.ContinuousEndpointWordTests.assert_coverage
    assert_released = endpoint_fixtures.ContinuousEndpointWordTests.assert_released

    def pending_hint(self, *, max_window_ms=500, eof_context_retry=False):
        config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=max_window_ms,
            max_buffer_ms=800,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            left_context_ms=100,
            input_evidence=True,
            source_units=True,
            endpointing=short_config(),
            word_boundary_fallback=True,
            eof_context_retry=eof_context_retry,
        )
        adapter = EvidenceNativeAdapter(word_result)
        stream = ContinuousTranscriptStream(
            adapter, stream_id="quiet-hint", config=config, mel_builder=lambda pcm: pcm
        )
        self.addCleanup(stream.close)
        stream.push(0, pcm_ms(200, 900))
        events = self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 100 * 16)
        self.assertTrue(stream._word_anchor)
        adapter.result_factory = lambda window_id, start, end: replace_words(
            word_result(window_id, start, end),
            lambda word: replace(word, text=" rejected", tokens=(999,)),
        )
        stream.push(1, pcm_ms(40, 1))
        self.assertEqual(len(stream._endpoints), 1)
        return stream, adapter, events

    def frozen(self, stream):
        return (
            stream.state,
            bytes(stream._audio),
            stream._head,
            stream._retained,
            stream._word_anchor,
            stream.metrics.events_emitted,
            stream.expected_chunk,
            stream.accepted_samples,
        )

    def admission(self, stream):
        return (
            tuple(stream._endpoints),
            stream._unit,
            stream._run_unit,
            stream._retry_analysis,
            stream._retry_unit,
            stream._last_endpoint,
            stream._next_endpoint,
            stream._previous,
            stream._previous_eligible,
            stream._word_previous,
            stream.metrics.decode_count,
        )

    def test_skip_preserves_prefix_and_pcm_then_strict_growth_and_eof_complete(self):
        stream, adapter, events = self.pending_hint()
        frozen = self.frozen(stream)
        hint = stream._endpoints[0]
        self.assertEqual(self.decode(stream), ())
        self.assertEqual(self.frozen(stream), frozen)
        self.assertFalse(stream._endpoints)
        self.assertIsNone(stream._unit)
        self.assertIsNone(stream._run_unit)
        self.assertIsNone(stream._retry_analysis)
        self.assertIsNone(stream._retry_unit)
        self.assertIsNone(stream._previous)
        self.assertIsNone(stream._word_previous)
        self.assertFalse(stream._previous_eligible)
        self.assertFalse(stream._unresolved_eof)
        self.assertIsNone(stream.context_retry_observation)
        self.assertEqual(stream._last_endpoint, hint.end_sample)
        self.assertEqual(stream._next_endpoint, hint.end_sample + 100 * 16)
        trace = stream.last_trace
        self.assertEqual(trace.reason, "anchor_missing")
        self.assertEqual(trace.action, "wait_for_input")
        self.assertEqual(trace.source_unit.endpoint, hint)
        self.assertEqual(trace.anchor_diagnostic.status, "lexical_missing")
        self.assertIsNone(trace.word_publication)
        self.assertFalse(trace.eof)
        self.assert_released(adapter)
        calls = len(adapter.calls)
        for _ in range(3):
            self.assertFalse(stream.ready)
            self.assertEqual(stream.step(), ())
        self.assertEqual(len(adapter.calls), calls)

        adapter.result_factory = word_result
        state = stream.state
        stream.push(2, pcm_ms(100, 900))
        events.extend(self.drain(stream))
        self.assertEqual(adapter.calls[-1][2], 340)
        self.assertEqual(stream.state, state)
        self.assertEqual(stream.last_trace.reason, "incomplete")
        stream.push(3, pcm_ms(100, 900))
        events.extend(self.drain(stream))
        self.assertGreater(stream.state.version, state.version)
        self.assertEqual(stream.state.windows[0], state.windows[0])
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assertEqual(
            self.assert_coverage(events, 440 * 16), ["w0", "w1", "w2", "w3", "w4"]
        )
        self.assertFalse(any("rejected" in (event.text or "") for event in events))
        self.assert_released(adapter)

    def test_eof_arriving_after_hint_admission_is_processed_as_actual_eof_next(self):
        stream, adapter, events = self.pending_hint()
        stream.step()
        stream.step()
        self.assertFalse(stream._run_final)
        stream.finish_input()
        self.assertEqual(stream.step(), ())
        self.assertFalse(stream.done)
        self.assertEqual(stream.last_trace.action, "wait_for_input")
        adapter.result_factory = word_result
        events.extend(self.drain(stream))
        self.assertTrue(stream.done)
        self.assertEqual(stream.last_trace.source_unit.origin, "end_of_input")
        self.assertEqual(self.assert_coverage(events, 240 * 16), ["w0", "w1", "w2"])

    def test_only_current_hint_is_removed_and_rejected_drafts_are_cleared(self):
        stream, adapter, _ = self.pending_hint()
        stream.step()
        stream.step()
        stream.push(2, pcm_ms(100, 900) + pcm_ms(40, 1))
        remaining = stream._endpoints[1]
        stream._draft_tokens = (999,)
        frozen = self.frozen(stream)
        self.assertEqual(stream.step(), ())
        self.assertEqual(tuple(stream._endpoints), (remaining,))
        self.assertEqual(stream._draft_tokens, ())
        self.assertEqual(self.frozen(stream), frozen)
        self.assert_released(adapter)

    def test_ambiguous_anchor_keeps_its_original_diagnostic(self):
        stream, adapter, _ = self.pending_hint()
        diagnostic = AnchorDiagnostic("ambiguous", timed_match_count=2)
        decision = WordAgreementDecision(
            "anchor_ambiguous",
            next_anchor=stream._word_anchor,
            anchor_diagnostic=diagnostic,
        )
        with patch(
            "whisper_runtime.adapters.continuous_stream.compare_word_hypotheses",
            return_value=decision,
        ):
            self.assertEqual(self.decode(stream), ())
        self.assertEqual(stream.last_trace.reason, "anchor_ambiguous")
        self.assertEqual(stream.last_trace.action, "wait_for_input")
        self.assertIs(stream.last_trace.anchor_diagnostic, diagnostic)
        self.assert_released(adapter)

    def test_bound_caller_and_actual_eof_refusals_still_latch(self):
        for boundary in ("bound", "caller", "end_of_input", "eof"):
            with self.subTest(boundary=boundary):
                stream, adapter, _ = self.pending_hint(
                    max_window_ms=240 if boundary == "bound" else 500,
                    eof_context_retry=boundary == "eof",
                )
                if boundary in ("caller", "end_of_input"):
                    stream._unit = SourceUnit(stream._head, 240 * 16, boundary)
                if boundary == "eof":
                    stream.finish_input()
                frozen, hints = self.frozen(stream), tuple(stream._endpoints)
                with self.assertRaisesRegex(
                    StreamNeedsResolutionError, "anchor_missing"
                ):
                    self.decode(stream)
                self.assertEqual(self.frozen(stream), frozen)
                self.assertEqual(tuple(stream._endpoints), hints)
                self.assertEqual(stream.last_trace.action, "unresolved")
                calls = len(adapter.calls)
                with self.assertRaises(StreamNeedsResolutionError):
                    stream.step()
                self.assertEqual(len(adapter.calls), calls)
                self.assert_released(adapter)
                if boundary == "eof":
                    self.assertEqual(
                        stream.context_retry_observation.status, "unavailable"
                    )

    def test_audio_score_and_other_word_refusals_are_not_skipped(self):
        for reason in ("conflicting_speech_score", "unstable", "incomplete"):
            with self.subTest(reason=reason):
                stream, adapter, _ = self.pending_hint()
                if reason == "conflicting_speech_score":
                    adapter.result_factory = lambda window_id, start, end: replace(
                        word_result(window_id, start, end),
                        metadata=replace(
                            word_result(window_id, start, end).metadata,
                            no_speech_prob=0.8,
                        ),
                    )
                frozen, hints = self.frozen(stream), tuple(stream._endpoints)
                with patch(
                    "whisper_runtime.adapters.continuous_stream.compare_word_hypotheses",
                    return_value=WordAgreementDecision(
                        reason
                        if reason != "conflicting_speech_score"
                        else "anchor_missing"
                    ),
                ):
                    with self.assertRaises(StreamNeedsResolutionError):
                        self.decode(stream)
                self.assertEqual(self.frozen(stream), frozen)
                self.assertEqual(tuple(stream._endpoints), hints)
                self.assertEqual(stream.last_trace.reason, reason)
                self.assertEqual(stream.last_trace.action, "unresolved")
                self.assert_released(adapter)

    def test_fence_failure_does_not_pop_or_change_frozen_admission(self):
        stream, adapter, _ = self.pending_hint()
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        frozen, admission = self.frozen(stream), self.admission(stream)
        run.fence.fail = True
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        self.assertIsNone(raised.exception.committed_state)
        self.assertEqual(self.frozen(stream), frozen)
        self.assertEqual(self.admission(stream), admission)
        with self.assertRaisesRegex(NativeStreamError, "retained"):
            stream.step()
        self.assertEqual(self.admission(stream), admission)
        run.fence.fail = False
        self.assertTrue(adapter.worker.recover(run.transaction))
        self.assertEqual(stream.step(), ())
        self.assertEqual(self.decode(stream), ())
        self.assertEqual(adapter.calls[-1], adapter.calls[-2])
        self.assertEqual(adapter.inputs[-1], adapter.inputs[-2])
        self.assertEqual(self.frozen(stream), frozen)
        self.assertFalse(stream._endpoints)
        self.assert_released(adapter)

    def test_growth_after_skip_still_stops_at_the_original_hard_bound(self):
        stream, adapter, _ = self.pending_hint(max_window_ms=250)
        state, anchor = stream.state, stream._word_anchor
        self.assertEqual(self.decode(stream), ())
        self.assertEqual(stream._next_endpoint, 340 * 16)
        self.assertFalse(stream.ready)
        stream.push(2, pcm_ms(10, 900))
        with self.assertRaisesRegex(StreamNeedsResolutionError, "no stable"):
            self.drain(stream)
        self.assertEqual(adapter.calls[-1][2], 250)
        self.assertEqual(stream.state, state)
        self.assertEqual(stream._word_anchor, anchor)
        self.assertEqual(stream.metrics.buffered_samples, 250 * 16)
        calls = len(adapter.calls)
        with self.assertRaises(StreamNeedsResolutionError):
            stream.step()
        self.assertEqual(len(adapter.calls), calls)
        self.assert_released(adapter)

    def test_queue_mismatch_is_verified_after_close_without_popping(self):
        stream, adapter, _ = self.pending_hint()
        stream.step()
        stream.step()
        stream._endpoints[0] = replace(stream._endpoints[0], peak=2)
        frozen, admission = self.frozen(stream), self.admission(stream)
        with self.assertRaisesRegex(NativeStreamError, "queued endpoint"):
            stream.step()
        self.assertEqual(self.frozen(stream), frozen)
        self.assertEqual(self.admission(stream), admission)
        self.assertTrue(adapter.runs[-1].closed)
        self.assert_released(adapter)


if __name__ == "__main__":
    unittest.main()
