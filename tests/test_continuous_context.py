"""Bounded left-context regressions using real, deterministic transactions."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from test_continuous_stream import ScriptedNativeAdapter, pcm, pcm_ms

from whisper_runtime import AudioSpan, RuntimeStateError, TransactionRetainedError
from whisper_runtime.adapters import (
    AudioBufferFullError,
    NativeDecodeMetadata,
    NativeStreamError,
    NativeTimestampSegment,
    NativeWindowResult,
    StreamEventKind,
    TranscriptEvent,
)
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)


def complete_result(window_id: str, start: int, end: int) -> NativeWindowResult:
    """Use globally aligned words, including a closed final fractional word."""
    segments = []
    position = start
    while position < end:
        boundary = min(end, (position // 100 + 1) * 100)
        segments.append(
            NativeTimestampSegment(
                AudioSpan(position, boundary), f" word-{position}", (position + 1,)
            )
        )
        position = boundary
    return NativeWindowResult(
        window_id=window_id,
        start_ms=start,
        end_ms=end,
        text="".join(segment.text for segment in segments).strip(),
        metadata=NativeDecodeMetadata(
            language="en",
            tokens=tuple(token for segment in segments for token in segment.tokens),
            segments=tuple(segments),
            timestamps_complete=bool(segments),
        ),
    )


class ContinuousLeftContextTests(unittest.TestCase):
    def config(self, **changes: int) -> ContinuousStreamConfig:
        arguments = {
            "preview_interval_ms": 100,
            "max_window_ms": 500,
            "max_buffer_ms": 600,
            "holdback_ms": 100,
            "timestamp_tolerance_ms": 0,
            "left_context_ms": 100,
        }
        arguments.update(changes)
        return ContinuousStreamConfig(**arguments)

    def stream(
        self,
        adapter: ScriptedNativeAdapter,
        *,
        config: ContinuousStreamConfig | None = None,
    ) -> ContinuousTranscriptStream:
        return ContinuousTranscriptStream(
            adapter,
            stream_id="context-test",
            mel_builder=lambda content: content,
            config=config or self.config(),
        )

    def drain(self, stream: ContinuousTranscriptStream) -> list[TranscriptEvent]:
        events = []
        for _ in range(1_000):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("context stream did not reach a scheduling boundary")

    def decode(self, stream: ContinuousTranscriptStream) -> tuple[TranscriptEvent, ...]:
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.step(), ())
        return stream.step()

    def assert_released(self, adapter: ScriptedNativeAdapter) -> None:
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def assert_audio_accounting(self, stream: ContinuousTranscriptStream) -> None:
        metrics = stream.metrics
        self.assertEqual(
            metrics.buffered_samples,
            metrics.accepted_samples - stream.retained_from_sample,
        )
        self.assertLessEqual(stream.retained_from_sample, metrics.committed_samples)
        self.assertLessEqual(
            metrics.committed_samples - stream.retained_from_sample,
            stream.config.left_context_ms * 16,
        )
        self.assertLessEqual(metrics.buffered_samples, stream.config.max_buffer_ms * 16)

    def assert_no_republished_context(self, events: list[TranscriptEvent]) -> None:
        watermark = 0
        committed_ids = set()
        for event in events:
            if event.kind is StreamEventKind.FINAL:
                continue
            self.assertNotIn(event.segment_id, committed_ids)
            self.assertEqual(event.start_sample, watermark)
            if event.kind is StreamEventKind.COMMIT:
                self.assertGreater(event.end_sample, watermark)
                self.assertEqual(event.committed_through_sample, event.end_sample)
                watermark = event.end_sample
                committed_ids.add(event.segment_id)
            else:
                for word in event.text.split():
                    self.assertTrue(word.startswith("word-"), word)
                    self.assertGreaterEqual(
                        int(word.removeprefix("word-")) * 16, watermark
                    )

    def test_context_is_opt_in_and_invalid_context_is_rejected(self) -> None:
        self.assertEqual(ContinuousStreamConfig().left_context_ms, 0)
        for value, error in ((True, TypeError), (1.5, TypeError), (-1, ValueError)):
            with self.subTest(value=value), self.assertRaises(error):
                self.config(left_context_ms=value)
        with self.assertRaises(ValueError):
            self.config(left_context_ms=500)
        with self.assertRaises(ValueError):
            self.config(left_context_ms=400)
        with self.assertRaises(ValueError):
            ContinuousStreamConfig(left_context_ms=1001)
        self.assertEqual(self.config(left_context_ms=380).left_context_ms, 380)

    def test_zero_context_preserves_existing_events_and_pcm_schedule(self) -> None:
        outcomes = []
        for config in (
            ContinuousStreamConfig(
                preview_interval_ms=100,
                max_window_ms=500,
                max_buffer_ms=600,
                holdback_ms=100,
                timestamp_tolerance_ms=0,
            ),
            self.config(left_context_ms=0),
        ):
            adapter = ScriptedNativeAdapter(complete_result)
            stream = self.stream(adapter, config=config)
            stream.push(0, pcm_ms(450, 8))
            events = self.drain(stream)
            self.assertEqual(
                stream.retained_from_sample, stream.metrics.committed_samples
            )
            stream.finish_input()
            events.extend(self.drain(stream))
            outcomes.append((events, adapter.calls, adapter.inputs, stream.metrics))
            self.assert_released(adapter)
        self.assertEqual(*outcomes)

    def test_committed_left_context_is_retained_with_exact_absolute_pcm_offsets(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        source = b"".join(pcm_ms(1, index) for index in range(600))
        stream.push(0, source[: 250 * 32])
        events = self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 100 * 16)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(stream.metrics.buffered_samples, 250 * 16)
        self.assert_audio_accounting(stream)
        stream.push(1, source[250 * 32 :])
        events.extend(self.drain(stream))
        self.assertGreater(stream.retained_from_sample, 0)
        self.assert_audio_accounting(stream)
        for call, content in zip(adapter.calls, adapter.inputs):
            self.assertEqual(content, source[call[1] * 32 : call[2] * 32])
            self.assertLessEqual(call[2] - call[1], 500)
        self.assertTrue(any(call[1] > 0 for call in adapter.calls))
        self.assert_no_republished_context(events)
        self.assertEqual(
            stream.metrics.decoded_source_samples,
            sum(len(content) // 2 for content in adapter.inputs),
        )
        for record in stream.state.windows:
            self.assertGreaterEqual(
                record.result.start_ms, record.result.analyzed_span.start_ms
            )
            self.assertLessEqual(
                record.result.end_ms, record.result.analyzed_span.end_ms
            )
        self.assertTrue(
            any(
                record.result.start_ms > record.result.analyzed_span.start_ms
                for record in stream.state.windows
            )
        )
        self.assert_released(adapter)

    def test_retained_context_counts_toward_atomic_input_buffer_limit(self) -> None:
        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(250))
        self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 100 * 16)
        # Only 150 ms is uncommitted, but another 100 ms is retained context.
        self.assertEqual(stream.metrics.buffered_samples, 250 * 16)
        stream.push(1, pcm_ms(350, 1))
        before = stream.metrics
        with self.assertRaises(AudioBufferFullError):
            stream.push(2, pcm(1, 2))
        self.assertEqual(stream.metrics, before)
        self.assertEqual(stream.expected_chunk, 2)
        self.drain(stream)
        self.assertEqual(stream.push(2, pcm(1, 2)), 600 * 16 + 1)
        self.assert_audio_accounting(stream)
        self.assertEqual(stream.metrics.peak_buffered_samples, 600 * 16)
        self.assert_released(adapter)

    def test_aligned_context_near_window_limit_still_has_growing_observations(
        self,
    ) -> None:
        def grid_result(window_id: str, start: int, end: int) -> NativeWindowResult:
            segments = tuple(
                NativeTimestampSegment(
                    AudioSpan(position, position + 20),
                    f" word-{position}",
                    (position + 1,),
                )
                for position in range(start, end, 20)
            )
            return NativeWindowResult(
                window_id=window_id,
                start_ms=start,
                end_ms=end,
                text="".join(segment.text for segment in segments).strip(),
                metadata=NativeDecodeMetadata(
                    language="en",
                    tokens=tuple(segment.tokens[0] for segment in segments),
                    segments=segments,
                    timestamps_complete=True,
                ),
            )

        adapter = ScriptedNativeAdapter(grid_result)
        stream = self.stream(adapter, config=self.config(left_context_ms=380))
        source = b"".join(pcm_ms(1, index) for index in range(600))
        stream.push(0, source)
        events = self.drain(stream)
        self.assertGreaterEqual(stream.metrics.committed_samples, 500 * 16)
        self.assertGreater(stream.retained_from_sample, 0)
        self.assert_audio_accounting(stream)
        self.assert_no_republished_context(events)
        for call, content in zip(adapter.calls, adapter.inputs):
            self.assertEqual(content, source[call[1] * 32 : call[2] * 32])
            self.assertLessEqual(call[2] - call[1], 500)
        self.assert_released(adapter)

    def test_trace_is_frozen_and_preserves_full_analysis_separately_from_publication(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        self.assertIsNone(stream.last_trace)
        stream.push(0, pcm_ms(500))
        self.assertEqual(stream.step(), ())
        self.assertIsNone(stream.last_trace)
        self.assertEqual(stream.step(), ())
        self.assertIsNone(stream.last_trace)
        first_events = stream.step()
        first = stream.last_trace
        self.assertEqual(first.decode_index, 1)
        self.assertEqual(first.action, "preview")
        self.assertEqual(first.reason, "incomplete")
        self.assertEqual(
            (first.analysis_start_sample, first.analysis_end_sample), (0, 100 * 16)
        )
        self.assertIs(first.result, adapter.runs[0].result)
        self.assertEqual(first_events[0].text, first.result.text)
        with self.assertRaises(FrozenInstanceError):
            first.action = "changed"
        with self.assertRaises(FrozenInstanceError):
            first.result.text = "changed"

        self.decode(stream)
        committed = stream.last_trace
        self.assertEqual(committed.decode_index, 2)
        self.assertEqual(committed.action, "commit")
        self.assertEqual(committed.reason, "candidate")
        self.assertEqual(committed.publication_span, AudioSpan(0, 100))
        self.assertEqual(committed.analysis_end_sample, 200 * 16)
        self.assertEqual(committed.committed_before_sample, 0)
        self.assertEqual(committed.retained_from_sample, 0)
        self.assertFalse(committed.eof)
        self.assertEqual(stream.metrics.committed_samples, 100 * 16)

        preview_events = self.decode(stream)
        preview = stream.last_trace
        self.assertEqual(preview.decode_index, 3)
        self.assertEqual(preview.action, "preview")
        self.assertEqual(preview.analysis_start_sample, 0)
        self.assertEqual(preview.committed_before_sample, 100 * 16)
        self.assertEqual(preview.result.text, "word-0 word-100 word-200")
        self.assertEqual(preview_events[0].text, "word-100 word-200")
        self.assertIsNone(preview.publication_span)
        self.decode(stream)
        self.decode(stream)
        latest = stream.last_trace
        self.assertEqual(latest.decode_index, 5)
        self.assertEqual(latest.analysis_start_sample, 200 * 16)
        self.assertEqual(latest.retained_from_sample, 200 * 16)
        self.assertEqual(latest.committed_before_sample, 300 * 16)
        self.assertEqual(latest.analysis_end_sample, 500 * 16)
        self.assertEqual(first.decode_index, 1)
        self.assertEqual(first.result.text, "word-0")
        self.assert_released(adapter)

    def test_trace_inspection_is_owner_only_and_retained_position_is_read_only(
        self,
    ) -> None:
        stream = self.stream(ScriptedNativeAdapter(complete_result))
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaisesRegex(NativeStreamError, "creating thread"):
                executor.submit(lambda: stream.last_trace).result(timeout=2)
            self.assertEqual(
                executor.submit(lambda: stream.retained_from_sample).result(timeout=2),
                0,
            )
        with self.assertRaises(AttributeError):
            stream.retained_from_sample = 100

    def test_changed_old_context_cannot_reappear_in_provisional_or_committed_text(
        self,
    ) -> None:
        def changing_context(
            window_id: str, start: int, end: int
        ) -> NativeWindowResult:
            result = complete_result(window_id, start, end)
            if start == 0 and end >= 300:
                segments = (
                    replace(
                        result.metadata.segments[0],
                        text=f" forbidden-context-{end}",
                        tokens=(10_000 + end,),
                    ),
                ) + result.metadata.segments[1:]
                return replace(
                    result,
                    text="".join(segment.text for segment in segments).strip(),
                    metadata=replace(result.metadata, segments=segments),
                )
            return result

        adapter = ScriptedNativeAdapter(changing_context)
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(500))
        events = self.drain(stream)
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assertTrue(
            any("forbidden-context" in run.result.text for run in adapter.runs)
        )
        self.assertFalse(
            any("forbidden-context" in (event.text or "") for event in events)
        )
        self.assert_no_republished_context(events)
        self.assertEqual(stream.metrics.committed_samples, 500 * 16)
        self.assert_released(adapter)

    def test_context_chunk_partition_equivalence_and_bounded_history(self) -> None:
        source = b"".join(pcm_ms(1, index) for index in range(550))
        outcomes = []
        for partitions in ((550,), (50,) * 11, (17, 83, 200, 107, 143)):
            adapter = ScriptedNativeAdapter(complete_result)
            stream = self.stream(adapter)
            events = []
            offset = 0
            for index, milliseconds in enumerate(partitions):
                stream.push(index, source[offset : offset + milliseconds * 32])
                offset += milliseconds * 32
                events.extend(self.drain(stream))
            stream.finish_input()
            events.extend(self.drain(stream))
            self.assert_no_republished_context(events)
            self.assert_audio_accounting(stream)
            outcomes.append((events, adapter.calls, adapter.inputs, stream.state))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], outcomes[2])

        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        for sequence in range(30):
            stream.push(sequence, pcm_ms(100, sequence))
            self.drain(stream)
            self.assertLessEqual(len(stream.state.windows), 4)
            self.assert_audio_accounting(stream)
        self.assertGreater(stream.state.version, 4)
        self.assertEqual(len(stream.state.windows), 4)
        self.assert_released(adapter)

    def test_cancelled_context_decode_retries_identical_source_without_eviction(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(200, 7))
        self.drain(stream)
        stream.push(1, pcm_ms(100, 8))
        before = stream.metrics
        retained = stream.retained_from_sample
        previous_trace = stream.last_trace
        stream.step()
        cancelled_input = adapter.inputs[-1]
        self.assertTrue(stream.cancel_active())
        with self.assertRaises(RuntimeStateError):
            stream.step()
        self.assertEqual(stream.metrics, before)
        self.assertEqual(stream.retained_from_sample, retained)
        self.assertIs(stream.last_trace, previous_trace)
        self.assert_released(adapter)
        self.drain(stream)
        self.assertEqual(adapter.inputs[-1], cancelled_input)
        self.assertEqual(adapter.calls[-1], adapter.calls[-2])
        self.assertEqual(stream.metrics.decode_count, before.decode_count + 1)
        self.assertEqual(
            stream.last_trace.decode_index, previous_trace.decode_index + 1
        )
        self.assert_audio_accounting(stream)

    def test_context_fence_failure_preserves_both_positions_until_recovery(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(300, 11))
        self.decode(stream)
        self.decode(stream)
        before = stream.metrics
        retained = stream.retained_from_sample
        previous_trace = stream.last_trace
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        run.fence.fail = True
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        self.assertIsNone(raised.exception.committed_state)
        self.assertEqual(stream.metrics, before)
        self.assertEqual(stream.retained_from_sample, retained)
        prepared_trace = stream.last_trace
        self.assertEqual(prepared_trace.decode_index, previous_trace.decode_index + 1)
        self.assertEqual(prepared_trace.action, "preview")
        with self.assertRaises(NativeStreamError):
            stream.step()
        run.fence.fail = False
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        self.assertEqual(stream.step(), ())
        self.drain(stream)
        self.assertEqual(adapter.inputs[-1], adapter.inputs[-2])
        self.assertEqual(
            stream.last_trace.decode_index, prepared_trace.decode_index + 1
        )
        self.assert_audio_accounting(stream)
        self.assert_released(adapter)

    def test_post_commit_retention_recovery_trims_context_and_emits_exactly_once(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(400, 12))
        events = []
        for _ in range(3):
            events.extend(self.decode(stream))
        self.assertEqual(stream.metrics.committed_samples, 100 * 16)
        self.assertEqual(stream.retained_from_sample, 0)
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        before = stream.metrics
        with patch.object(
            type(run.transaction._lease),
            "release",
            side_effect=RuntimeError("injected context lease failure"),
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                stream.step()
            self.assertIsNotNone(raised.exception.committed_state)
            self.assertEqual(stream.state.version, 2)
            self.assertEqual(stream.metrics, before)
            self.assertEqual(stream.retained_from_sample, 0)
            prepared_trace = stream.last_trace
            self.assertEqual(prepared_trace.action, "commit")
            self.assertEqual(prepared_trace.publication_span, AudioSpan(100, 300))
            self.assertEqual(prepared_trace.committed_before_sample, 100 * 16)
            self.assertEqual(prepared_trace.retained_from_sample, 0)
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        calls = len(adapter.calls)
        recovered = stream.step()
        events.extend(recovered)
        self.assertEqual(
            [event.kind for event in recovered],
            [StreamEventKind.REPLACE, StreamEventKind.COMMIT],
        )
        self.assertEqual(stream.metrics.committed_samples, 300 * 16)
        self.assertEqual(stream.retained_from_sample, 200 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 200 * 16)
        self.assertEqual(len(adapter.calls), calls)
        self.assertEqual(stream.metrics.decode_count, before.decode_count + 1)
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.metrics.events_emitted, before.events_emitted + 2)
        self.assertIs(stream.last_trace, prepared_trace)
        self.assert_no_republished_context(events)
        self.assert_released(adapter)

    def test_empty_eof_with_context_enabled_does_not_decode(self) -> None:
        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        stream.finish_input()
        self.assertEqual(
            [event.kind for event in self.drain(stream)], [StreamEventKind.FINAL]
        )
        self.assertEqual(adapter.calls, [])
        self.assertEqual(stream.retained_from_sample, 0)
        self.assert_audio_accounting(stream)
        self.assertEqual(stream.step(), ())

    def test_eof_with_only_already_committed_context_emits_final_without_redecode(
        self,
    ) -> None:
        def stretching_result(
            window_id: str, start: int, end: int
        ) -> NativeWindowResult:
            segment = NativeTimestampSegment(AudioSpan(start, end), "one word", (1,))
            return NativeWindowResult(
                window_id=window_id,
                start_ms=start,
                end_ms=end,
                text=segment.text,
                metadata=NativeDecodeMetadata(
                    language="en",
                    tokens=(1,),
                    segments=(segment,),
                    timestamps_complete=True,
                ),
            )

        adapter = ScriptedNativeAdapter(stretching_result)
        stream = self.stream(
            adapter, config=self.config(holdback_ms=0, timestamp_tolerance_ms=100)
        )
        stream.push(0, pcm_ms(200))
        self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 200 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 100 * 16)
        self.assertEqual(stream.retained_from_sample, 100 * 16)
        calls = len(adapter.calls)
        stream.finish_input()
        events = self.drain(stream)
        self.assertEqual([event.kind for event in events], [StreamEventKind.FINAL])
        self.assertEqual(len(adapter.calls), calls)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.retained_from_sample, 200 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assert_audio_accounting(stream)

    def test_eof_tail_selects_only_uncommitted_native_segments_and_exact_samples(
        self,
    ) -> None:
        for extra_samples in (0, 7):
            with self.subTest(extra_samples=extra_samples):
                adapter = ScriptedNativeAdapter(complete_result)
                stream = self.stream(adapter)
                source = pcm_ms(260, 21) + pcm(extra_samples, 22)
                stream.push(0, source)
                events = self.drain(stream)
                self.assertEqual(stream.metrics.committed_samples, 100 * 16)
                stream.finish_input()
                events.extend(self.drain(stream))
                self.assertEqual(adapter.inputs[-1], source)
                record = stream.state.windows[-1]
                self.assertEqual(record.result.analyzed_span, AudioSpan(0, 260))
                self.assertEqual(
                    (record.result.start_ms, record.result.end_ms), (100, 260)
                )
                self.assertEqual(record.result.text, "word-100 word-200")
                self.assertEqual(record.result.publication_segment_indices, (1, 2))
                self.assertEqual(events[-2].committed_through_sample, len(source) // 2)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assertEqual(stream.retained_from_sample, len(source) // 2)
                trace = stream.last_trace
                self.assertTrue(trace.eof)
                self.assertEqual(trace.action, "commit")
                self.assertEqual(trace.reason, "eof")
                self.assertEqual(trace.publication_span, AudioSpan(100, 260))
                self.assertEqual(trace.analysis_end_sample, len(source) // 2)
                self.assertEqual(trace.analysis_start_sample, 0)
                self.assert_no_republished_context(events)
                self.assert_audio_accounting(stream)
                self.assert_released(adapter)

    def test_context_eof_refuses_straddling_missing_or_gapped_native_tail(self) -> None:
        for kind in (
            "straddle",
            "leading-gap",
            "interior-gap",
            "missing-timestamps",
            "incomplete-timestamps",
        ):
            with self.subTest(kind=kind):
                adapter = ScriptedNativeAdapter(complete_result)
                stream = self.stream(adapter)
                stream.push(0, pcm_ms(260, 30))
                self.drain(stream)
                self.assertEqual(stream.metrics.committed_samples, 100 * 16)
                before = stream.metrics

                def malformed_tail(
                    window_id: str, start: int, end: int
                ) -> NativeWindowResult:
                    result = complete_result(window_id, start, end)
                    if kind == "missing-timestamps":
                        return replace(result, metadata=None)
                    if kind == "incomplete-timestamps":
                        return replace(
                            result,
                            metadata=replace(
                                result.metadata, timestamps_complete=False
                            ),
                        )
                    if kind == "straddle":
                        spans = ((0, 120), (120, end))
                    elif kind == "leading-gap":
                        spans = ((0, 100), (120, end))
                    else:
                        spans = ((0, 100), (100, 200), (220, end))
                    segments = tuple(
                        NativeTimestampSegment(
                            AudioSpan(left, right), f" word-{left}", (left + 1,)
                        )
                        for left, right in spans
                    )
                    return replace(
                        result,
                        text="".join(segment.text for segment in segments).strip(),
                        metadata=replace(result.metadata, segments=segments),
                    )

                adapter.result_factory = malformed_tail
                stream.finish_input()
                with self.assertRaises(StreamNeedsResolutionError):
                    self.drain(stream)
                self.assertEqual(
                    stream.metrics.committed_samples, before.committed_samples
                )
                self.assertEqual(
                    stream.metrics.buffered_samples, before.buffered_samples
                )
                self.assertEqual(stream.retained_from_sample, 0)
                self.assertEqual(stream.state.version, 1)
                self.assertFalse(stream.done)
                trace = stream.last_trace
                self.assertEqual(trace.action, "unresolved")
                self.assertEqual(trace.reason, "eof_unresolved")
                self.assertIsNone(trace.publication_span)
                self.assertEqual(trace.committed_before_sample, 100 * 16)
                self.assertEqual(trace.retained_from_sample, 0)
                self.assertTrue(trace.eof)
                self.assert_released(adapter)

                unresolved_metrics = stream.metrics
                calls = len(adapter.calls)
                for _ in range(3):
                    with self.assertRaises(StreamNeedsResolutionError):
                        stream.step()
                    self.assertEqual(len(adapter.calls), calls)
                    self.assertEqual(stream.metrics, unresolved_metrics)
                    self.assertIs(stream.last_trace, trace)
                self.assertTrue(stream.close())
                self.assertEqual(stream.step(), ())

    def test_unresolved_context_eof_cleanup_recovery_records_once_without_redecode(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter(complete_result)
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(260, 33))
        self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 100 * 16)
        adapter.result_factory = lambda window_id, start, end: replace(
            complete_result(window_id, start, end), metadata=None
        )
        stream.finish_input()
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        before = stream.metrics
        calls = len(adapter.calls)
        run.fence.fail = True
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        trace = stream.last_trace
        self.assertEqual(trace.action, "unresolved")
        self.assertEqual(trace.reason, "eof_unresolved")
        self.assertIs(raised.exception.transaction, run.transaction)
        self.assertIsNone(raised.exception.committed_state)
        self.assertEqual(stream.metrics, before)
        self.assertTrue(stream.active)
        self.assertFalse(run.capacity_released)
        with self.assertRaisesRegex(NativeStreamError, "retained"):
            stream.step()
        run.fence.fail = False
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        self.assertEqual(stream.step(), ())
        self.assertFalse(stream.active)
        self.assertEqual(stream.metrics.decode_count, before.decode_count + 1)
        self.assertEqual(
            stream.metrics.decoded_source_samples,
            before.decoded_source_samples + 260 * 16,
        )
        self.assertEqual(stream.metrics.events_emitted, before.events_emitted)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(stream.metrics.committed_samples, 100 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 260 * 16)
        unresolved_metrics = stream.metrics
        for _ in range(3):
            with self.assertRaises(StreamNeedsResolutionError):
                stream.step()
            self.assertEqual(stream.metrics, unresolved_metrics)
            self.assertEqual(len(adapter.calls), calls)
            self.assertIs(stream.last_trace, trace)
        self.assertTrue(stream.close())
        self.assert_released(adapter)

    def test_context_straddle_or_gap_cannot_evict_uncommitted_audio_when_window_fills(
        self,
    ) -> None:
        for kind in ("straddle", "gap"):
            with self.subTest(kind=kind):
                adapter = ScriptedNativeAdapter(complete_result)
                stream = self.stream(adapter)
                stream.push(0, pcm_ms(200, 31))
                self.drain(stream)

                def invalid_prefix(
                    window_id: str, start: int, end: int
                ) -> NativeWindowResult:
                    result = complete_result(window_id, start, end)
                    spans = (
                        ((0, 120), (120, end))
                        if kind == "straddle"
                        else ((0, 100), (120, end))
                    )
                    segments = tuple(
                        NativeTimestampSegment(
                            AudioSpan(left, right), f" word-{left}", (left + 1,)
                        )
                        for left, right in spans
                    )
                    return replace(
                        result,
                        text="".join(segment.text for segment in segments).strip(),
                        metadata=replace(result.metadata, segments=segments),
                    )

                adapter.result_factory = invalid_prefix
                stream.push(1, pcm_ms(400, 32))
                with self.assertRaises(StreamNeedsResolutionError):
                    self.drain(stream)
                self.assertEqual(stream.metrics.committed_samples, 100 * 16)
                self.assertEqual(stream.retained_from_sample, 0)
                self.assertEqual(stream.metrics.buffered_samples, 600 * 16)
                self.assertEqual(stream.state.version, 1)
                self.assertLessEqual(adapter.calls[-1][2] - adapter.calls[-1][1], 500)
                self.assert_released(adapter)


if __name__ == "__main__":
    unittest.main()
