"""Scripted controller integration for opt-in source-positioned word publication."""

import unittest
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import patch

from test_continuous_stream import ScriptedNativeAdapter, ScriptedNativeRun, pcm, pcm_ms

from whisper_runtime import AudioSpan, RuntimeStateError, TransactionRetainedError
from whisper_runtime.adapters import (
    NativeDecodeMetadata,
    NativeStreamError,
    NativeTimestampSegment,
    NativeWindowResult,
    StreamEventKind,
)
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)
from whisper_runtime.adapters.word_policy import (
    AlignedPublication,
    NativeWordAlignment,
    compare_word_hypotheses,
)


def word(start: int, end: int, token: int | None = None) -> NativeTimestampSegment:
    token = start // 100 + 1 if token is None else token
    return NativeTimestampSegment(AudioSpan(start, end), f" word-{token}", (token,))


def regular_words(start: int, end: int) -> tuple[NativeTimestampSegment, ...]:
    return tuple(
        word(at, at + 100) for at in range((start + 99) // 100 * 100, end - 99, 100)
    )


class ScriptedWordRun(ScriptedNativeRun):
    """Keep the real transaction/fence/owner lifecycle, replace only model work."""

    def __init__(self, original: ScriptedNativeRun) -> None:
        super().__init__(
            original.adapter, original.transaction, original.result, original.fence
        )
        self.alignment: NativeWordAlignment | None = None

    def prepare_word_alignment(self) -> NativeWordAlignment:
        native = self.prepare_result()
        if self.alignment is not None:
            return self.alignment
        try:
            self.adapter.fault("align")
            self.alignment = NativeWordAlignment(
                native,
                self.adapter.word_factory(
                    native.analyzed_span.start_ms, native.analyzed_span.end_ms
                ),
            )
            self.adapter.alignments.append(self.alignment)
            return self.alignment
        except BaseException as error:
            self._close_owner(error)
            raise

    def finish(
        self,
        *,
        committed_through_ms=None,
        publication_span=None,
        aligned_publication=None,
    ):
        if aligned_publication is None:
            return super().finish(
                committed_through_ms=committed_through_ms,
                publication_span=publication_span,
            )
        state = None
        try:
            self.transaction.checkpoint()
            self.adapter.finish_attempts.append(aligned_publication)
            self.adapter.fault("finish")
            if not isinstance(aligned_publication, AlignedPublication):
                raise TypeError("expected an aligned publication")
            if aligned_publication.alignment is not self.alignment:
                raise ValueError(
                    "publication must retain this run's prepared alignment"
                )
            state = self.transaction.commit(
                aligned_publication, committed_through_ms=committed_through_ms
            )
        except BaseException as error:
            self._close_owner(error, state)
            raise
        self._close_owner(None, state)
        return state


class ScriptedWordAdapter(ScriptedNativeAdapter):
    def __init__(
        self,
        word_factory: Callable[
            [int, int], tuple[NativeTimestampSegment, ...]
        ] = regular_words,
        **kwargs,
    ) -> None:
        self.word_factory = word_factory
        self.alignments: list[NativeWordAlignment] = []
        self.finish_attempts: list[AlignedPublication] = []

        def native(window_id: str, start: int, end: int) -> NativeWindowResult:
            words = word_factory(start, end)
            tokens = tuple(token for item in words for token in item.tokens)
            text = "".join(item.text for item in words)
            # One whole native segment intentionally crosses later commit
            # boundaries; the separate words preserve their own estimated times.
            segments = (
                (NativeTimestampSegment(AudioSpan(start, end), text, tokens),)
                if words
                else ()
            )
            return NativeWindowResult(
                window_id,
                text.strip(),
                start,
                end,
                metadata=NativeDecodeMetadata(
                    "en", tokens, segments=segments, timestamps_complete=bool(words)
                ),
            )

        super().__init__(native, **kwargs)

    def start_window(self, **kwargs) -> ScriptedWordRun:
        run = ScriptedWordRun(super().start_window(**kwargs))
        self.runs[-1] = run
        return run


class AlignedStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=600,
            max_buffer_ms=700,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            left_context_ms=100,
            word_alignment=True,
        )

    def stream(self, adapter, **kwargs):
        return ContinuousTranscriptStream(
            adapter,
            stream_id="aligned-test",
            mel_builder=lambda value: value,
            config=kwargs.pop("config", self.config),
            **kwargs,
        )

    def decode(self, stream):
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.step(), ())
        return stream.step()

    def drain(self, stream):
        events = []
        for _ in range(200):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("scripted stream did not reach a scheduling boundary")

    def assert_released(self, adapter):
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def commit_first_word(self, stream):
        stream.push(0, pcm_ms(100, 1))
        self.decode(stream)
        stream.push(1, pcm_ms(100, 2))
        events = self.decode(stream)
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual(events[-1].committed_through_sample, 100 * 16)
        return events

    def test_opt_in_is_boolean_and_has_separate_profiles(self):
        self.assertIs(ContinuousStreamConfig().word_alignment, False)
        for value in (0, 1, None, "true"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(self.config, word_alignment=value)
        for coalesced, expected in (
            (False, "word_agreement_stream/v1"),
            (True, "coalesced_word_agreement_stream/v1"),
        ):
            stream = self.stream(
                ScriptedWordAdapter(),
                config=replace(self.config, coalesce_previews=coalesced),
            )
            self.assertEqual(stream.profile_id, expected)
            stream.close()

    def test_default_native_profile_never_requests_word_alignment(self):
        adapter = ScriptedWordAdapter()
        stream = self.stream(adapter, config=replace(self.config, word_alignment=False))
        stream.push(0, pcm_ms(200))
        stream.finish_input()
        events = self.drain(stream)
        self.assertEqual(adapter.alignments, [])
        self.assertEqual(adapter.finish_attempts, [])
        self.assertIsInstance(stream.state.windows[-1].result, NativeWindowResult)
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
        self.assert_released(adapter)

    def test_aligned_suffix_publishes_across_a_straddling_native_segment(self):
        adapter = ScriptedWordAdapter()
        stream = self.stream(adapter)
        first = self.commit_first_word(stream)
        immutable = first[-1]
        stream.push(2, pcm_ms(100, 3))
        self.decode(stream)
        stream.push(3, pcm_ms(100, 4))
        events = self.decode(stream)
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual((events[-1].start_sample, events[-1].end_sample), (1600, 4800))
        self.assertEqual(events[0].text, "word-2 word-3")
        result = stream.state.windows[-1].result
        self.assertIsInstance(result, AlignedPublication)
        self.assertEqual((result.word_start, result.word_end), (1, 3))
        self.assertEqual(
            result.alignment.native.metadata.segments[0].span, AudioSpan(0, 400)
        )
        self.assertEqual(result.alignment.native.text, "word-1 word-2 word-3 word-4")
        self.assertIs(stream.last_trace.word_alignment, result.alignment)
        self.assertEqual(stream.last_trace.word_publication, result)
        self.assertEqual(stream.last_trace.result, result.alignment.native)
        self.assertIsNone(stream.last_trace.publication_span)
        self.assertEqual(stream.retained_from_sample, 200 * 16)
        self.assertEqual(first[-1], immutable)
        self.assertNotEqual(events[-1].segment_id, immutable.segment_id)
        for call, content in zip(adapter.calls, adapter.inputs):
            self.assertEqual(len(content) // 2, (call[2] - call[1]) * 16)
        stream.close()
        self.assert_released(adapter)

    def test_unique_text_reappearing_later_cannot_move_the_source_frontier(self):
        def repeated(start, end):
            if end <= 200:
                return regular_words(start, end)
            return tuple(
                item
                for item in (word(200, 300, 1), word(300, 400, 8))
                if item.span.end_ms <= end
            )

        adapter = ScriptedWordAdapter(repeated)
        stream = self.stream(adapter)
        self.commit_first_word(stream)
        events = []
        for sequence in (2, 3, 4):
            stream.push(sequence, pcm_ms(100, sequence))
            events.extend(self.decode(stream))
        self.assertFalse(any(event.kind is StreamEventKind.COMMIT for event in events))
        self.assertEqual(stream.metrics.committed_samples, 1600)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.metrics.buffered_samples, 500 * 16)
        self.assertEqual(len(adapter.finish_attempts), 1)
        stream.finish_input()
        with self.assertRaises(StreamNeedsResolutionError):
            self.decode(stream)
        count = len(adapter.alignments)
        self.assertEqual(stream.last_trace.action, "unresolved")
        self.assertEqual(stream.last_trace.reason, "anchor_missing")
        self.assertEqual(stream.last_trace.anchor_diagnostic.status, "relocated")
        self.assertEqual(stream.last_trace.anchor_diagnostic.observation, "current")
        for _ in range(3):
            with self.assertRaises(StreamNeedsResolutionError):
                stream.step()
        self.assertEqual(len(adapter.alignments), count)
        self.assertEqual(stream.metrics.committed_samples, 1600)
        self.assertEqual(stream.metrics.buffered_samples, 500 * 16)
        stream.close()

    def test_a_word_itself_crossing_the_frontier_is_not_clipped(self):
        def straddling(start, end):
            if end <= 200:
                return regular_words(start, end)
            return (word(0, 60, 1), word(60, 200, 2)) + tuple(
                item for item in regular_words(200, end)
            )

        adapter = ScriptedWordAdapter(straddling)
        stream = self.stream(
            adapter, config=replace(self.config, timestamp_tolerance_ms=20)
        )
        self.commit_first_word(stream)
        for sequence in (2, 3):
            stream.push(sequence, pcm_ms(100))
            events = self.decode(stream)
            self.assertFalse(
                any(event.kind is StreamEventKind.COMMIT for event in events)
            )
        self.assertEqual(stream.metrics.committed_samples, 1600)
        self.assertEqual(adapter.alignments[-1].words[1].span, AudioSpan(60, 200))
        stream.finish_input()
        with self.assertRaises(StreamNeedsResolutionError):
            self.decode(stream)
        self.assertEqual(stream.last_trace.reason, "anchor_missing")
        self.assertEqual(stream.last_trace.anchor_diagnostic.status, "timing_mismatch")
        self.assertEqual(len(adapter.finish_attempts), 1)
        stream.close()

    def test_small_anchored_boundary_drift_keeps_estimates_and_next_anchor(self):
        def measured_words(start, end):
            if start == 0:
                estimates = (
                    NativeTimestampSegment(AudioSpan(100, 180), " do", (1,)),
                    NativeTimestampSegment(AudioSpan(180, 202), " for", (2,)),
                )
            else:
                estimates = (
                    NativeTimestampSegment(AudioSpan(100, 172), " do", (1,)),
                    NativeTimestampSegment(AudioSpan(172, 202), " for", (2,)),
                    NativeTimestampSegment(AudioSpan(202, 450), " you", (3,)),
                )
            return tuple(
                item
                for item in estimates
                if item.span.start_ms >= start and item.span.end_ms <= end
            )

        adapter = ScriptedWordAdapter(measured_words)
        stream = self.stream(
            adapter,
            config=replace(
                self.config, coalesce_previews=True, timestamp_tolerance_ms=10
            ),
        )
        stream.push(0, pcm_ms(200))
        self.decode(stream)
        stream.push(1, pcm_ms(100))
        first = self.decode(stream)
        self.assertEqual(first[0].text, "do")
        self.assertEqual(first[-1].committed_through_sample, 180 * 16)
        original = stream.state.windows[-1].result
        self.assertEqual(original.alignment.words[0].span, AudioSpan(100, 180))

        stream.push(2, pcm_ms(100))
        self.decode(stream)
        stream.push(3, pcm_ms(100))
        second = self.decode(stream)
        self.assertEqual(second[0].text, "for")
        self.assertEqual(
            (second[-1].start_sample, second[-1].end_sample), (180 * 16, 202 * 16)
        )
        shifted = stream.state.windows[-1].result
        self.assertEqual(shifted.boundary_tolerance_ms, 10)
        self.assertEqual(shifted.start_ms, 180)
        self.assertEqual(
            shifted.alignment.words[shifted.word_start].span, AudioSpan(172, 202)
        )
        self.assertEqual(original.alignment.words[0].span, AudioSpan(100, 180))

        # The next observation retains the newly published 'for' at its original
        # 172..202 estimate, not an overlapping concatenation of old/new anchors.
        stream.push(4, pcm_ms(100))
        self.decode(stream)
        stream.push(5, pcm_ms(100))
        following = self.decode(stream)
        self.assertEqual(following[0].text, "you")
        self.assertEqual(
            (following[-1].start_sample, following[-1].end_sample), (202 * 16, 450 * 16)
        )
        self.assertEqual(len(adapter.finish_attempts), 3)
        self.assertEqual(stream.metrics.committed_samples, 450 * 16)
        stream.close()
        self.assert_released(adapter)

    def test_rolling_stream_keeps_exact_pcm_bounded_anchor_and_unique_text(self):
        adapter = ScriptedWordAdapter()
        stream = self.stream(adapter)
        source = b"".join(pcm_ms(100, value) for value in range(30))
        events = []
        with patch(
            "whisper_runtime.adapters.continuous_stream.compare_word_hypotheses",
            wraps=compare_word_hypotheses,
        ) as compare:
            for sequence in range(30):
                stream.push(sequence, source[sequence * 3200 : (sequence + 1) * 3200])
                events.extend(self.drain(stream))
            stream.finish_input()
            events.extend(self.drain(stream))
        self.assertTrue(stream.done)
        self.assertEqual(stream.metrics.committed_samples, len(source) // 2)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assertLessEqual(stream.metrics.peak_buffered_samples, 700 * 16)
        self.assertLessEqual(len(stream.state.windows), 4)
        for call in compare.call_args_list:
            self.assertIsInstance(call.kwargs["anchor"], tuple)
            self.assertLessEqual(len(call.kwargs["anchor"]), 4)
        for call, content in zip(adapter.calls, adapter.inputs):
            self.assertEqual(content, source[call[1] * 32 : call[2] * 32])
            self.assertLessEqual(call[2] - call[1], 600)
        revisions = {}
        committed_ids = set()
        text = []
        watermark = 0
        for event in events:
            if event.kind in (StreamEventKind.PROVISIONAL, StreamEventKind.REPLACE):
                self.assertNotIn(event.segment_id, committed_ids)
                revisions[(event.segment_id, event.revision)] = event.text
            elif event.kind is StreamEventKind.COMMIT:
                self.assertNotIn(event.segment_id, committed_ids)
                committed_ids.add(event.segment_id)
                text.append(revisions[(event.segment_id, event.revision)])
                self.assertEqual(event.start_sample, watermark)
                watermark = event.end_sample
        self.assertEqual(
            " ".join(text), " ".join(f"word-{index}" for index in range(1, 31))
        )
        self.assertEqual(watermark, len(source) // 2)
        self.assertGreater(len(committed_ids), 4)
        self.assert_released(adapter)

    def test_coalesced_backlog_reserves_two_observations_before_publication(self):
        adapter = ScriptedWordAdapter()
        stream = self.stream(
            adapter, config=replace(self.config, coalesce_previews=True)
        )
        source = pcm_ms(700, 6)
        stream.push(0, source)
        first = self.decode(stream)
        self.assertEqual(adapter.calls[0][1:3], (0, 500))
        self.assertEqual([event.kind for event in first], [StreamEventKind.PROVISIONAL])
        self.assertEqual(stream.metrics.committed_samples, 0)
        second = self.decode(stream)
        self.assertEqual(adapter.calls[1][1:3], (0, 600))
        self.assertEqual(second[-1].committed_through_sample, 500 * 16)
        self.drain(stream)
        stream.finish_input()
        self.drain(stream)
        self.assertTrue(stream.done)
        self.assertEqual(stream.metrics.committed_samples, 700 * 16)
        self.assert_released(adapter)

    def test_alignment_from_another_native_result_is_rejected_before_publication(self):
        adapter = ScriptedWordAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        wrong = NativeWordAlignment(
            replace(run.result, window_id="wrong-window"), regular_words(0, 100)
        )
        with patch.object(run, "prepare_word_alignment", return_value=wrong):
            with self.assertRaisesRegex(NativeStreamError, "does not match"):
                stream.step()
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assertEqual(stream.metrics.decode_count, 0)
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(adapter.finish_attempts, [])
        self.assert_released(adapter)
        stream.close()

    def test_final_context_suffix_need_not_end_at_the_eof_timestamp(self):
        def suffix(start, end):
            if end <= 200:
                return regular_words(start, end)
            return (word(0, 100), word(100, 180, 2), word(180, 240, 3))

        adapter = ScriptedWordAdapter(suffix)
        stream = self.stream(adapter)
        self.commit_first_word(stream)
        stream.push(2, pcm_ms(60, 3) + pcm(7, 4))
        stream.finish_input()
        events = self.drain(stream)
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
        result = stream.state.windows[-1].result
        self.assertTrue(result.final)
        self.assertEqual(result.text, "word-2 word-3")
        self.assertEqual((result.start_ms, result.end_ms), (100, 260))
        self.assertEqual(result.alignment.words[-1].span.end_ms, 240)
        self.assertEqual(events[-2].end_sample, 260 * 16 + 7)
        self.assertEqual(stream.metrics.committed_samples, stream.accepted_samples)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assert_released(adapter)

    def test_empty_word_eof_accounts_for_processed_input_without_silence_claim(self):
        for samples in (0, 7, 1600):
            with self.subTest(samples=samples):
                adapter = ScriptedWordAdapter(lambda start, end: ())
                stream = self.stream(adapter)
                if samples:
                    stream.push(0, pcm(samples, 9))
                stream.finish_input()
                events = self.drain(stream)
                self.assertTrue(stream.done)
                self.assertEqual(stream.metrics.committed_samples, samples)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assertEqual(
                    sum(event.kind is StreamEventKind.FINAL for event in events), 1
                )
                if samples:
                    result = stream.state.windows[-1].result
                    self.assertIsInstance(result, AlignedPublication)
                    self.assertTrue(result.final)
                    self.assertEqual(result.text, "")
                    self.assertEqual((result.word_start, result.word_end), (0, 0))
                    self.assertEqual(result.alignment.words, ())
                    self.assertEqual(events[-2].end_sample, samples)
                    self.assertNotIn("silence", stream.last_trace.reason)
                else:
                    self.assertEqual(adapter.alignments, [])
                    self.assertEqual(len(events), 1)
                self.assert_released(adapter)

    def test_cancellation_with_new_pcm_and_eof_retries_the_same_admission(self):
        for coalesced in (False, True):
            with self.subTest(coalesced=coalesced):
                adapter = ScriptedWordAdapter()
                stream = self.stream(
                    adapter, config=replace(self.config, coalesce_previews=coalesced)
                )
                stream.push(0, pcm_ms(250, 7))
                stream.step()
                stream.push(1, pcm_ms(100, 8))
                stream.finish_input()
                self.assertTrue(stream.cancel_active())
                with self.assertRaises(RuntimeStateError):
                    stream.step()
                events = self.decode(stream)
                self.assertEqual(adapter.calls[0], adapter.calls[1])
                self.assertEqual(adapter.inputs[0], adapter.inputs[1])
                self.assertFalse(stream.last_trace.eof)
                self.assertEqual(
                    [event.kind for event in events], [StreamEventKind.PROVISIONAL]
                )
                self.assertEqual(stream.metrics.committed_samples, 0)
                self.assertEqual(stream.state.version, 0)
                self.drain(stream)
                self.assertTrue(stream.done)
                self.assertEqual(stream.metrics.committed_samples, 350 * 16)
                self.assert_released(adapter)

    def test_alignment_failure_does_not_publish_or_advance_the_cursor(self):
        adapter = ScriptedWordAdapter(fail_once="align")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))
        with self.assertRaises(RuntimeError) as raised:
            self.decode(stream)
        self.assertIs(raised.exception, adapter.error)
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.metrics.decode_count, 0)
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assert_released(adapter)
        self.decode(stream)
        self.assertEqual(adapter.calls[0], adapter.calls[1])
        stream.close()

    def test_failed_finish_does_not_treat_candidate_words_as_committed(self):
        adapter = ScriptedWordAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))
        self.decode(stream)
        stream.push(1, pcm_ms(100))
        before = stream.metrics
        adapter.fail_once = "finish"
        with self.assertRaises(RuntimeError):
            self.decode(stream)
        self.assertEqual(stream.metrics, before)
        self.assertEqual(stream.state.version, 0)
        events = self.decode(stream)
        self.assertEqual(events[-1].committed_through_sample, 1600)
        self.assertEqual(events[0].text, "word-1")
        self.assertEqual([item.word_start for item in adapter.finish_attempts], [0, 0])
        self.assertEqual(adapter.calls[-1], adapter.calls[-2])
        self.assert_released(adapter)
        stream.close()

    def test_precommit_fence_recovery_cannot_advance_anchor_or_publish(self):
        adapter = ScriptedWordAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))
        self.decode(stream)
        stream.push(1, pcm_ms(100))
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        before = stream.metrics
        run.fence.fail = True
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        self.assertIsNone(raised.exception.committed_state)
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.metrics, before)
        with self.assertRaises(NativeStreamError):
            stream.step()
        run.fence.fail = False
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        self.assertEqual(stream.step(), ())
        events = self.decode(stream)
        self.assertEqual(events[0].text, "word-1")
        self.assertEqual(events[-1].committed_through_sample, 1600)
        self.assertEqual([item.word_start for item in adapter.finish_attempts], [0, 0])
        self.assert_released(adapter)
        stream.close()

    def test_committed_release_recovery_emits_once_without_realignment(self):
        adapter = ScriptedWordAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))
        self.decode(stream)
        stream.push(1, pcm_ms(100))
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        before = stream.metrics
        with patch.object(
            type(run.transaction._lease),
            "release",
            side_effect=RuntimeError("release failed"),
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                stream.step()
            self.assertIsNotNone(raised.exception.committed_state)
            self.assertEqual(stream.state.version, 1)
            self.assertEqual(stream.metrics, before)
            self.assertEqual(stream.retained_from_sample, 0)
        alignment_count = len(adapter.alignments)
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        events = stream.step()
        self.assertEqual(
            [event.kind for event in events],
            [StreamEventKind.REPLACE, StreamEventKind.COMMIT],
        )
        self.assertEqual(events[0].text, "word-1")
        self.assertEqual(stream.metrics.committed_samples, 1600)
        self.assertEqual(len(adapter.alignments), alignment_count)
        self.assertEqual(len(adapter.finish_attempts), 1)
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.state.version, 1)
        self.assert_released(adapter)
        committed_id = events[-1].segment_id
        stream.push(2, pcm_ms(100))
        self.decode(stream)
        stream.push(3, pcm_ms(100))
        following = self.decode(stream)
        self.assertEqual(following[0].text, "word-2 word-3")
        self.assertEqual(following[-1].kind, StreamEventKind.COMMIT)
        self.assertNotEqual(following[-1].segment_id, committed_id)
        self.assertEqual(stream.metrics.committed_samples, 300 * 16)
        self.assertEqual(len(adapter.finish_attempts), 2)
        stream.close()


if __name__ == "__main__":
    unittest.main()
