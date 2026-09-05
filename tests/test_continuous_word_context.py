"""Bounded whole-word context is opt-in; estimates never authorize forced cuts."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import test_continuous_endpoint_words as endpoint_fixtures
from test_audio_endpoints import short_config
from test_continuous_evidence import EvidenceNativeAdapter
from test_continuous_stream import pcm_ms
from test_word_policy import aligned, word

from whisper_runtime import AudioSpan, TransactionRetainedError
from whisper_runtime.adapters import NativeStreamError, StreamEventKind
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)
from whisper_runtime.adapters.word_policy import (
    WordAgreementDecision,
    select_word_publication,
)


class ContinuousWordContextTests(unittest.TestCase):
    decode = endpoint_fixtures.ContinuousEndpointWordTests.decode
    drain = endpoint_fixtures.ContinuousEndpointWordTests.drain
    assert_coverage = endpoint_fixtures.ContinuousEndpointWordTests.assert_coverage
    assert_released = endpoint_fixtures.ContinuousEndpointWordTests.assert_released

    def setUp(self):
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=1000,
            max_buffer_ms=1200,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            left_context_ms=100,
            source_units=True,
            input_evidence=True,
            endpointing=short_config(),
            word_boundary_fallback=True,
            word_context_limit_ms=400,
        )

    def stream(self, adapter=None, **changes):
        adapter = adapter or EvidenceNativeAdapter(endpoint_fixtures.word_result)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="whole-word-context",
            config=replace(self.config, **changes),
            mel_builder=lambda content: content,
        )
        self.addCleanup(stream.close)
        return stream, adapter

    def candidate(self, stream, words, *, start, end, watermark, selected_start=0):
        alignment = aligned(words, start=start, end=end)
        publication = select_word_publication(
            alignment,
            selected_start,
            len(words),
            AudioSpan(watermark, words[-1].span.end_ms),
        )
        stream._retained = stream._run_start = start * 16
        stream._head = watermark * 16
        stream._run_end = end * 16
        return WordAgreementDecision(
            "candidate", publication, words[selected_start:][-4:]
        )

    def test_config_is_strict_opt_in_and_reserves_two_growing_observations(self):
        self.assertEqual(ContinuousStreamConfig().word_context_limit_ms, 0)
        for value in (True, None, "400", 400.0):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(self.config, word_context_limit_ms=value)
        for value in (-20, 80, 101, 800):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.config, word_context_limit_ms=value)
        with self.assertRaises(ValueError):
            ContinuousStreamConfig(word_context_limit_ms=400)
        self.assertEqual(
            replace(self.config, word_context_limit_ms=100).left_context_ms, 100
        )

    def test_profile_suffix_and_disabled_behavior_do_not_change_legacy_defaults(self):
        for enabled in (False, True):
            for coalesce in (False, True):
                with self.subTest(enabled=enabled, coalesce=coalesce):
                    stream, _ = self.stream(
                        word_context_limit_ms=400 if enabled else 0,
                        coalesce_previews=coalesce,
                    )
                    self.assertEqual(
                        stream.profile_id,
                        ("coalesced_" if coalesce else "")
                        + "word_boundary_quiet_endpoint_stream/v1"
                        + ("+word_context/v1" if enabled else "")
                        + "+input_evidence/v1",
                    )
                    if not coalesce:
                        stream.push(0, pcm_ms(200, 900))
                        self.drain(stream)
                        self.assertEqual(
                            stream.metrics.committed_samples, 0 if enabled else 100 * 16
                        )

    def test_recorded_mid_word_cut_moves_back_to_amidst_not_forward(self):
        # Reduced exact trace 13, no-added-pauses diagnostic, 2026-09-05.
        stream, _ = self.stream(
            max_window_ms=30000,
            max_buffer_ms=40000,
            left_context_ms=2000,
            word_context_limit_ms=6000,
        )
        words = (
            word(1295, 18840, 19180, " place"),
            word(31095, 19180, 19500, " amidst"),
            word(262, 19500, 19840, " the"),
            word(29804, 19840, 20280, " tents"),
            word(13, 20280, 20700, "."),
            word(1114, 20700, 20920, " For"),
            word(257, 20920, 21000, " a"),
            word(981, 21000, 21260, " while"),
        )
        decision = self.candidate(
            stream, words, start=17840, end=26000, watermark=19840, selected_start=3
        )
        self.assertEqual(stream._word_context_start(decision), 19180 * 16)
        self.assertEqual(decision.next_anchor, words[-4:])
        self.assertEqual(stream.retained_from_sample, 17840 * 16)

    def test_recorded_punctuation_plus_for_needs_two_lexical_anchor_units(self):
        stream, _ = self.stream(
            max_window_ms=30000,
            max_buffer_ms=40000,
            left_context_ms=2000,
            word_context_limit_ms=6000,
        )
        words = (word(13, 3560, 5880, "."), word(1114, 5880, 6100, " For"))
        decision = self.candidate(stream, words, start=1560, end=28000, watermark=3560)
        self.assertIsNone(stream._word_context_start(decision))
        words += (word(257, 6100, 6360, " a"),)
        decision = self.candidate(stream, words, start=1560, end=28000, watermark=3560)
        self.assertEqual(stream._word_context_start(decision), 3560 * 16)

    def test_only_leading_anchor_punctuation_is_removed_without_rewriting_publication(
        self,
    ):
        words = (
            word(13, 0, 20, "."),
            word(1, 20, 100, " one"),
            word(11, 100, 120, ","),
            word(2, 120, 200, " two"),
            word(3, 200, 300, " three"),
            word(4, 300, 400, " four"),
            word(5, 400, 500, " five"),
        )

        def observed(window_id, start, end):
            result = endpoint_fixtures.word_result(window_id, start, end)
            selected = tuple(
                item
                for item in words
                if item.span.start_ms >= start and item.span.end_ms <= end
            )
            return replace(
                result,
                text="".join(item.text for item in selected).strip(),
                metadata=replace(
                    result.metadata,
                    segments=selected,
                    tokens=tuple(token for item in selected for token in item.tokens),
                ),
            )

        stream, adapter = self.stream(EvidenceNativeAdapter(observed))
        stream.push(0, pcm_ms(300, 900))
        events = self.drain(stream)
        publication = stream.last_trace.word_publication
        self.assertEqual(publication.text, ". one, two")
        self.assertEqual(publication.alignment.words[: publication.word_end], words[:4])
        self.assertEqual(stream._word_anchor, words[1:4])
        self.assertEqual(stream.retained_from_sample, 20 * 16)
        stream.push(1, pcm_ms(200, 900))
        events.extend(self.drain(stream))
        self.assertEqual(stream.last_trace.word_publication.text, "three four")
        self.assertTrue(
            all(run.result.text.startswith("one,") for run in adapter.runs[3:])
        )
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assertEqual(
            self.assert_coverage(events, 500 * 16),
            [".", "one,", "two", "three", "four", "five"],
        )
        self.assert_released(adapter)

    def test_anchor_count_ignores_lexical_words_crowded_out_by_punctuation(self):
        stream, _ = self.stream()
        words = tuple(
            word(index + 1, index * 20, (index + 1) * 20, text)
            for index, text in enumerate((" one", " two", ".", ",", ";", " three"))
        )
        decision = self.candidate(stream, words, start=0, end=300, watermark=0)
        self.assertIsNone(stream._word_context_start(decision))

    def test_cap_rejects_long_complete_anchor_without_cutting_forward(self):
        stream, _ = self.stream()
        words = (word(1, 0, 450), word(2, 450, 500))
        decision = self.candidate(stream, words, start=0, end=700, watermark=0)
        before = stream.metrics
        self.assertIsNone(stream._word_context_start(decision))
        self.assertEqual(stream.metrics, before)
        self.assertEqual(stream.retained_from_sample, 0)
        # An exactly bounded complete anchor is allowed.
        words = (word(1, 100, 450), word(2, 450, 500))
        decision = self.candidate(stream, words, start=0, end=700, watermark=0)
        self.assertEqual(stream._word_context_start(decision), 100 * 16)

    def test_frame_rounding_rechecks_earlier_word_and_never_resurrects_evicted_pcm(
        self,
    ):
        stream, _ = self.stream()
        words = (
            word(1, 107, 207),
            word(2, 207, 233),
            word(3, 233, 257),
            word(4, 257, 300),
            word(5, 300, 350),
        )
        decision = self.candidate(
            stream, words, start=107, end=600, watermark=257, selected_start=3
        )
        # Desired 250 becomes 247 on the origin-relative frame grid; that is
        # inside word 3, whose start 233 rounds to 227, inside word 2.
        self.assertEqual(stream._word_context_start(decision), 207 * 16)
        decision = self.candidate(stream, words[-2:], start=257, end=600, watermark=257)
        self.assertEqual(stream._word_context_start(decision), 257 * 16)

    def test_deferred_candidate_keeps_witness_then_rolls_with_bounded_exact_pcm(self):
        for coalesce in (False, True):
            with self.subTest(coalesce=coalesce):
                stream, adapter = self.stream(coalesce_previews=coalesce)
                source = pcm_ms(2400, 900)
                events, traces = [], []
                for index in range(24):
                    stream.push(index, source[index * 3200 : (index + 1) * 3200])
                    events.extend(self.drain(stream, traces))
                    self.assertLessEqual(
                        stream.metrics.committed_samples - stream.retained_from_sample,
                        400 * 16,
                    )
                    if index == 1:
                        self.assertEqual(stream.last_trace.reason, "context_unresolved")
                        self.assertIsNotNone(stream._word_previous)
                        self.assertEqual(stream.state.version, 0)
                        self.assertEqual(stream.metrics.buffered_samples, 200 * 16)
                stream.finish_input()
                events.extend(self.drain(stream, traces))
                self.assertEqual(
                    self.assert_coverage(events, len(source) // 2),
                    [f"w{index}" for index in range(24)],
                )
                self.assertTrue(any(call[1] > 0 for call in adapter.calls))
                for call, content in zip(adapter.calls, adapter.inputs):
                    _, start, end, _ = call
                    self.assertLessEqual(end - start, 1000)
                    self.assertEqual(content, source[start * 32 : end * 32])
                self.assert_released(adapter)

    def test_full_window_backlog_reserves_fresh_pair_after_rebase(self):
        for coalesce in (False, True):
            with self.subTest(coalesce=coalesce):
                stream, adapter = self.stream(coalesce_previews=coalesce)
                source = pcm_ms(1200, 900)
                stream.push(0, source)
                stream.finish_input()
                traces = []
                events = self.drain(stream, traces)
                self.assertEqual(
                    self.assert_coverage(events, len(source) // 2),
                    [f"w{index}" for index in range(12)],
                )
                self.assertTrue(any(t.reason == "candidate" for t in traces))
                self.assertTrue(any(call[1] > 0 for call in adapter.calls))
                self.assert_released(adapter)

    def test_closed_unit_eof_and_digital_silence_bypass_partial_context_guard(self):
        for boundary in ("quiet", "eof", "silence"):
            with self.subTest(boundary=boundary):
                stream, adapter = self.stream()
                source = pcm_ms(100, 0 if boundary == "silence" else 900)
                if boundary == "quiet":
                    source += pcm_ms(40, 1)
                stream.push(0, source)
                if boundary != "quiet":
                    stream.finish_input()
                with patch.object(
                    stream,
                    "_word_context_start",
                    side_effect=AssertionError("partial only"),
                ):
                    events = self.drain(stream)
                self.assertTrue(any(e.kind is StreamEventKind.COMMIT for e in events))
                if boundary == "quiet":
                    self.assertFalse(stream.done)
                    self.assertEqual(stream._word_anchor, ())
                else:
                    self.assertTrue(stream.done)
                self.assertEqual(stream.retained_from_sample, len(source) // 2)
                self.assert_released(adapter)

    def test_missing_native_continuation_still_fails_without_false_completion(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(300, 900))
        events = self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 200 * 16)
        before = stream.metrics
        adapter.result_factory = lambda window_id, start, end: (
            endpoint_fixtures.replace_words(
                endpoint_fixtures.word_result(window_id, start, end),
                lambda item: replace(item, text=" missing", tokens=(999,)),
            )
        )
        stream.finish_input()
        with self.assertRaises(StreamNeedsResolutionError):
            self.drain(stream)
        self.assertEqual(stream.last_trace.reason, "anchor_missing")
        self.assertEqual(stream.metrics.committed_samples, before.committed_samples)
        self.assertEqual(stream.metrics.buffered_samples, before.buffered_samples)
        self.assertFalse(stream.done)
        self.assertFalse(any(e.kind is StreamEventKind.FINAL for e in events))
        self.assert_released(adapter)

    def test_committed_cleanup_recovery_uses_frozen_origin_despite_new_pcm_and_eof(
        self,
    ):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(400, 900))
        events = self.drain(stream)
        stream.push(1, pcm_ms(100, 900))
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        before = stream.metrics
        old_anchor = stream._word_anchor
        with patch.object(
            type(run.transaction._lease),
            "release",
            side_effect=RuntimeError("release failed"),
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                stream.step()
            self.assertIsNotNone(raised.exception.committed_state)
            self.assertEqual(stream.metrics, before)
            self.assertEqual(stream.retained_from_sample, 0)
            self.assertEqual(stream._word_anchor, old_anchor)
            stream.push(2, pcm_ms(100, 900))
            stream.finish_input()
            with self.assertRaisesRegex(NativeStreamError, "retained"):
                stream.step()
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        with patch.object(
            stream, "_word_context_start", side_effect=AssertionError("already frozen")
        ):
            recovered = stream.step()
        self.assertEqual(sum(e.kind is StreamEventKind.COMMIT for e in recovered), 1)
        self.assertEqual(stream.metrics.committed_samples, 400 * 16)
        self.assertEqual(stream.retained_from_sample, 200 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 400 * 16)
        events.extend(recovered)
        events.extend(self.drain(stream))
        self.assertEqual(
            self.assert_coverage(events, 600 * 16), [f"w{index}" for index in range(6)]
        )
        self.assert_released(adapter)
