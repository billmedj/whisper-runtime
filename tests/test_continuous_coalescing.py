"""Opt-in endpoint coalescing contracts, not ASR quality or speed measurements."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from test_continuous_stream import ScriptedNativeAdapter, pcm, pcm_ms

from whisper_runtime import RuntimeStateError, TransactionRetainedError
from whisper_runtime.adapters import NativeStreamError, StreamEventKind, TranscriptEvent
from whisper_runtime.adapters.continuous_stream import (
    COALESCED_CONTEXT_CONTINUOUS_PROFILE,
    COALESCED_CONTINUOUS_PROFILE,
    CONTEXT_CONTINUOUS_PROFILE,
    CONTINUOUS_PROFILE,
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)


class ContinuousCoalescingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=500,
            max_buffer_ms=600,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            coalesce_previews=True,
        )

    def stream(
        self,
        adapter: ScriptedNativeAdapter,
        *,
        config: ContinuousStreamConfig | None = None,
    ) -> ContinuousTranscriptStream:
        return ContinuousTranscriptStream(
            adapter,
            stream_id="coalescing-test",
            mel_builder=lambda content: content,
            config=config or self.config,
        )

    def decode(self, stream: ContinuousTranscriptStream) -> tuple[TranscriptEvent, ...]:
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.step(), ())
        return stream.step()

    def drain(self, stream: ContinuousTranscriptStream) -> list[TranscriptEvent]:
        events = []
        for _ in range(1_000):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("stream did not reach a scheduling boundary")

    def assert_released(self, adapter: ScriptedNativeAdapter) -> None:
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_flag_is_strictly_boolean_and_defaults_to_false(self) -> None:
        self.assertIs(ContinuousStreamConfig().coalesce_previews, False)
        for value in (0, 1, None, "true", 1.0):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(self.config, coalesce_previews=value)

    def test_profiles_distinguish_both_opt_in_combinations(self) -> None:
        for coalesce, context, expected in (
            (False, 0, CONTINUOUS_PROFILE),
            (False, 100, CONTEXT_CONTINUOUS_PROFILE),
            (True, 0, COALESCED_CONTINUOUS_PROFILE),
            (True, 100, COALESCED_CONTEXT_CONTINUOUS_PROFILE),
        ):
            with self.subTest(coalesce=coalesce, context=context):
                stream = self.stream(
                    ScriptedNativeAdapter(),
                    config=replace(
                        self.config,
                        coalesce_previews=coalesce,
                        left_context_ms=context,
                    ),
                )
                self.assertEqual(stream.profile_id, expected)
                stream.close()
        self.assertEqual(
            len(
                {
                    CONTINUOUS_PROFILE,
                    CONTEXT_CONTINUOUS_PROFILE,
                    COALESCED_CONTINUOUS_PROFILE,
                    COALESCED_CONTEXT_CONTINUOUS_PROFILE,
                }
            ),
            4,
        )

    def test_backlog_skips_obsolete_previews_but_preserves_exact_pcm(self) -> None:
        source = b"".join(pcm_ms(100, value) for value in range(6))
        counts = []
        transcripts = []
        for enabled, expected_endpoints in (
            (False, [100, 200, 300, 400, 500, 600]),
            (True, [400, 500, 600]),
        ):
            with self.subTest(enabled=enabled):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(
                    adapter, config=replace(self.config, coalesce_previews=enabled)
                )
                stream.push(0, source)
                events = self.drain(stream)
                self.assertEqual(
                    [call[2] for call in adapter.calls], expected_endpoints
                )
                counts.append(stream.metrics.decode_count)
                stream.finish_input()
                events.extend(self.drain(stream))
                revisions = {}
                committed_text = []
                watermark = 0
                for event in events:
                    if event.kind in (
                        StreamEventKind.PROVISIONAL,
                        StreamEventKind.REPLACE,
                    ):
                        revisions[event.segment_id] = event.text
                    elif event.kind is StreamEventKind.COMMIT:
                        self.assertEqual(event.start_sample, watermark)
                        watermark = event.committed_through_sample
                        committed_text.append(revisions[event.segment_id])
                transcripts.append(" ".join(committed_text))
                self.assertEqual(watermark, len(source) // 2)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                for call, content in zip(adapter.calls, adapter.inputs):
                    self.assertEqual(content, source[call[1] * 32 : call[2] * 32])
                    self.assertLessEqual(call[2] - call[1], 500)
                self.assert_released(adapter)
        self.assertLess(counts[1], counts[0])
        self.assertEqual(transcripts[0], transcripts[1])

    def test_first_coalesced_observation_cannot_commit_or_consume_audio(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(600))
        events = self.decode(stream)
        self.assertEqual(
            [event.kind for event in events], [StreamEventKind.PROVISIONAL]
        )
        self.assertEqual(adapter.calls[0][1:3], (0, 400))
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assertEqual(stream.metrics.buffered_samples, 600 * 16)
        self.assertIsNone(stream.last_trace.publication_span)
        stream.close()

    def test_an_existing_observation_still_limits_what_the_latest_can_commit(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))
        self.decode(stream)
        stream.push(1, pcm_ms(500))
        events = self.decode(stream)
        self.assertEqual([call[2] for call in adapter.calls], [100, 500])
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual(events[-1].committed_through_ms, 100)
        stream.close()

    def test_unchanged_audio_does_not_supply_a_second_observation(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(250))
        self.decode(stream)
        for _ in range(3):
            self.assertFalse(stream.ready)
            self.assertEqual(stream.step(), ())
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(stream.state.version, 0)
        stream.push(1, pcm_ms(100))
        self.decode(stream)
        self.assertEqual([call[2] for call in adapter.calls], [250, 350])
        stream.close()

    def test_first_endpoint_reserves_growth_for_large_intervals_and_context(
        self,
    ) -> None:
        for interval, context in ((1, 0), (100, 380), (250, 240), (400, 80), (499, 0)):
            with self.subTest(interval=interval, context=context):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(
                    adapter,
                    config=replace(
                        self.config,
                        preview_interval_ms=interval,
                        left_context_ms=context,
                    ),
                )
                stream.push(0, pcm_ms(600))
                self.decode(stream)
                self.assertEqual(stream.state.version, 0)
                first_end = adapter.calls[-1][2]
                self.assertGreaterEqual(first_end, interval)
                self.assertLess(first_end, 500)
                self.decode(stream)
                self.assertGreater(adapter.calls[-1][2], first_end)
                self.assertEqual(adapter.calls[-1][2], 500)
                stream.close()

    def test_thirty_second_backlog_does_not_fill_the_first_window(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(
            adapter, config=ContinuousStreamConfig(coalesce_previews=True)
        )
        stream.push(0, pcm_ms(30_000))
        self.decode(stream)
        self.assertEqual(adapter.calls[-1][1:3], (0, 29_000))
        self.assertEqual(stream.state.version, 0)
        self.decode(stream)
        self.assertEqual(adapter.calls[-1][1:3], (0, 30_000))
        self.assertGreater(stream.metrics.committed_samples, 0)
        stream.close()

    def test_rebased_near_full_context_reserves_a_second_growing_endpoint(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter, config=replace(self.config, left_context_ms=380))
        stream.push(0, pcm_ms(600))
        self.decode(stream)
        self.decode(stream)
        self.assertEqual(stream.metrics.committed_samples, 400 * 16)
        self.assertEqual(stream.retained_from_sample, 20 * 16)
        self.decode(stream)
        self.assertEqual(adapter.calls[-1][1:3], (20, 420))
        self.decode(stream)
        self.assertEqual(adapter.calls[-1][1:3], (20, 520))
        # The synthetic timestamps straddle the watermark at this origin.
        # Coalescing must not bypass that unrelated publication restriction.
        with self.assertRaises(StreamNeedsResolutionError):
            stream.step()
        self.assertEqual(stream.metrics.committed_samples, 400 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 580 * 16)
        stream.close()

    def test_eof_within_window_preserves_the_existing_full_result_contract(
        self,
    ) -> None:
        for source in (b"", pcm(7, 3), pcm_ms(260, 4) + pcm(7, 5)):
            with self.subTest(samples=len(source) // 2):
                outputs = []
                for enabled in (False, True):
                    adapter = ScriptedNativeAdapter()
                    stream = self.stream(
                        adapter,
                        config=replace(self.config, coalesce_previews=enabled),
                    )
                    if source:
                        stream.push(0, source)
                    stream.finish_input()
                    events = self.drain(stream)
                    self.assertTrue(stream.done)
                    self.assertEqual(stream.metrics.committed_samples, len(source) // 2)
                    self.assertEqual(stream.metrics.buffered_samples, 0)
                    self.assert_released(adapter)
                    outputs.append(
                        (adapter.calls, adapter.inputs, events, stream.state)
                    )
                self.assertEqual(outputs[0], outputs[1])

    def test_eof_during_a_preview_does_not_change_its_admitted_audio(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        source = pcm_ms(250, 8)
        stream.push(0, source)
        stream.step()
        stream.push(1, pcm_ms(50, 9))
        stream.finish_input()
        events = self.drain(stream)
        self.assertEqual(adapter.inputs, [source, source + pcm_ms(50, 9)])
        self.assertEqual([call[2] for call in adapter.calls], [250, 300])
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
        self.assertEqual(stream.metrics.committed_samples, 300 * 16)
        self.assert_released(adapter)

    def test_cancelled_first_observation_retries_without_advancing_cadence(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(600, 10))
        before = stream.metrics
        stream.step()
        self.assertTrue(stream.cancel_active())
        with self.assertRaises(RuntimeStateError):
            stream.step()
        self.assertEqual(stream.metrics, before)
        self.assertIsNone(stream.last_trace)
        self.assert_released(adapter)
        events = self.decode(stream)
        self.assertEqual(adapter.calls[0], adapter.calls[1])
        self.assertEqual(adapter.inputs[0], adapter.inputs[1])
        self.assertEqual(
            [event.kind for event in events], [StreamEventKind.PROVISIONAL]
        )
        self.assertEqual(stream.state.version, 0)
        stream.close()

    def test_initial_start_step_and_prepare_failures_preserve_cadence(self) -> None:
        for stage in ("start", "step", "prepare"):
            with self.subTest(stage=stage):
                adapter = ScriptedNativeAdapter(fail_once=stage)
                stream = self.stream(adapter)
                stream.push(0, pcm_ms(600, 11))
                before = stream.metrics
                with self.assertRaises(RuntimeError) as raised:
                    self.decode(stream)
                self.assertIs(raised.exception, adapter.error)
                self.assertEqual(stream.metrics, before)
                self.assertIsNone(stream.last_trace)
                self.assert_released(adapter)
                self.decode(stream)
                self.assertEqual(adapter.calls[0], adapter.calls[1])
                self.assertEqual(adapter.inputs[0], adapter.inputs[1])
                self.assertEqual(stream.state.version, 0)
                stream.close()

    def test_failed_finish_retries_the_same_second_observation(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(600, 12))
        self.decode(stream)
        before = stream.metrics
        adapter.fail_once = "finish"
        with self.assertRaises(RuntimeError) as raised:
            self.decode(stream)
        self.assertIs(raised.exception, adapter.error)
        self.assertEqual(stream.metrics, before)
        self.assertEqual(stream.state.version, 0)
        self.assert_released(adapter)
        events = self.decode(stream)
        self.assertEqual(adapter.calls[-1], adapter.calls[-2])
        self.assertEqual(adapter.inputs[-1], adapter.inputs[-2])
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual(stream.metrics.committed_samples, 400 * 16)
        stream.close()

    def test_precommit_fence_failure_cannot_advance_either_observation(self) -> None:
        for previous_observation in (False, True):
            with self.subTest(previous_observation=previous_observation):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(adapter)
                stream.push(0, pcm_ms(600, 13))
                if previous_observation:
                    self.decode(stream)
                before = stream.metrics
                stream.step()
                stream.step()
                run = adapter.runs[-1]
                run.fence.fail = True
                with self.assertRaises(TransactionRetainedError) as raised:
                    stream.step()
                self.assertIsNone(raised.exception.committed_state)
                self.assertEqual(stream.metrics, before)
                self.assertEqual(stream.state.version, 0)
                with self.assertRaises(NativeStreamError):
                    stream.step()
                run.fence.fail = False
                self.assertTrue(adapter.worker.recover(raised.exception.transaction))
                self.assertEqual(stream.step(), ())
                self.assertEqual(stream.metrics, before)
                events = self.decode(stream)
                self.assertEqual(adapter.calls[-1], adapter.calls[-2])
                self.assertEqual(adapter.inputs[-1], adapter.inputs[-2])
                self.assertEqual(
                    events[-1].kind,
                    StreamEventKind.COMMIT
                    if previous_observation
                    else StreamEventKind.PROVISIONAL,
                )
                self.assert_released(adapter)
                stream.close()

    def test_committed_release_recovery_publishes_once_without_another_decode(
        self,
    ) -> None:
        for context in (0, 100):
            with self.subTest(context=context):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(
                    adapter, config=replace(self.config, left_context_ms=context)
                )
                stream.push(0, pcm_ms(600, 14))
                self.decode(stream)
                before = stream.metrics
                stream.step()
                stream.step()
                run = adapter.runs[-1]
                with patch.object(
                    type(run.transaction._lease),
                    "release",
                    side_effect=RuntimeError("injected coalescing lease failure"),
                ):
                    with self.assertRaises(TransactionRetainedError) as raised:
                        stream.step()
                    self.assertIsNotNone(raised.exception.committed_state)
                    self.assertEqual(stream.state.version, 1)
                    self.assertEqual(stream.metrics, before)
                    self.assertEqual(stream.retained_from_sample, 0)
                self.assertTrue(adapter.worker.recover(raised.exception.transaction))
                events = stream.step()
                self.assertEqual(
                    [event.kind for event in events],
                    [StreamEventKind.REPLACE, StreamEventKind.COMMIT],
                )
                self.assertEqual(len(adapter.calls), 2)
                self.assertEqual(stream.metrics.committed_samples, 400 * 16)
                self.assertEqual(stream.retained_from_sample, (400 - context) * 16)
                self.assert_released(adapter)
                self.decode(stream)
                self.assertEqual(adapter.calls[-1][1:3], (400 - context, 600))
                self.assertEqual(stream.state.version, 1)
                stream.close()

    def test_cancel_retry_freezes_input_and_eof_despite_new_pcm(self) -> None:
        for eof in (False, True):
            with self.subTest(eof=eof):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(adapter)
                original = pcm_ms(250, 21)
                added = pcm_ms(100, 22)
                stream.push(0, original)
                stream.step()
                stream.push(1, added)
                if eof:
                    stream.finish_input()
                self.assertTrue(stream.cancel_active())
                with self.assertRaises(RuntimeStateError):
                    stream.step()
                events = self.decode(stream)
                self.assertEqual(adapter.calls[0], adapter.calls[1])
                self.assertEqual(adapter.inputs, [original, original])
                self.assertEqual(
                    [event.kind for event in events], [StreamEventKind.PROVISIONAL]
                )
                self.assertFalse(stream.last_trace.eof)
                self.assertEqual(stream.state.version, 0)
                self.assertEqual(stream.metrics.buffered_samples, 350 * 16)
                self.drain(stream)
                self.assertEqual(adapter.calls[-1][1:3], (0, 350))
                self.assertEqual(adapter.inputs[-1], original + added)
                if eof:
                    self.assertTrue(stream.done)
                self.assert_released(adapter)
                stream.close()

    def test_start_step_prepare_and_finish_retry_ignore_concurrent_arrival(
        self,
    ) -> None:
        for stage in ("start", "step", "prepare", "finish"):
            with self.subTest(stage=stage):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(adapter)
                source = pcm_ms(250, 23)
                stream.push(0, source)
                if stage == "finish":
                    self.decode(stream)
                    stream.push(1, pcm_ms(100, 24))
                    source += pcm_ms(100, 24)
                adapter.fail_once = stage
                if stage == "start":
                    original_start = adapter.start_window

                    def start_with_arrival(**kwargs):
                        stream.push(stream.expected_chunk, pcm_ms(100, 25))
                        return original_start(**kwargs)

                    with patch.object(adapter, "start_window", start_with_arrival):
                        with self.assertRaises(RuntimeError):
                            stream.step()
                else:
                    stream.step()
                    stream.push(stream.expected_chunk, pcm_ms(100, 25))
                    if stage != "step":
                        stream.step()
                    with self.assertRaises(RuntimeError):
                        stream.step()
                before_retry = stream.metrics
                failed_call = adapter.calls[-1]
                self.assert_released(adapter)
                self.decode(stream)
                self.assertEqual(adapter.calls[-1], failed_call)
                self.assertEqual(adapter.inputs[-1], source)
                self.assertEqual(adapter.inputs[-2], source)
                self.assertEqual(
                    stream.metrics.accepted_samples, before_retry.accepted_samples
                )
                self.assertEqual(
                    stream.metrics.decode_count, before_retry.decode_count + 1
                )
                stream.close()

    def test_precommit_recovery_keeps_original_input_when_more_pcm_arrives(
        self,
    ) -> None:
        for previous_observation in (False, True):
            with self.subTest(previous_observation=previous_observation):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(adapter)
                source = pcm_ms(250, 26)
                stream.push(0, source)
                if previous_observation:
                    self.decode(stream)
                    source += pcm_ms(100, 27)
                    stream.push(1, pcm_ms(100, 27))
                stream.step()
                stream.step()
                run = adapter.runs[-1]
                run.fence.fail = True
                with self.assertRaises(TransactionRetainedError) as raised:
                    stream.step()
                self.assertIsNone(raised.exception.committed_state)
                stream.push(stream.expected_chunk, pcm_ms(100, 28))
                before_retry = stream.metrics
                run.fence.fail = False
                self.assertTrue(adapter.worker.recover(raised.exception.transaction))
                self.assertEqual(stream.step(), ())
                self.assertEqual(stream.metrics, before_retry)
                self.decode(stream)
                self.assertEqual(adapter.calls[-1], adapter.calls[-2])
                self.assertEqual(adapter.inputs[-1], source)
                self.assertEqual(adapter.inputs[-2], source)
                self.assert_released(adapter)
                stream.close()

    def test_failed_preprocessing_freezes_its_admitted_input(self) -> None:
        adapter = ScriptedNativeAdapter()
        attempts = []

        def build(content: bytes) -> bytes:
            attempts.append(content)
            if len(attempts) == 1:
                stream.push(1, pcm_ms(100, 30))
                raise RuntimeError("injected preprocessing failure after new PCM")
            return content

        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="coalescing-preprocess-test",
            mel_builder=build,
            config=self.config,
        )
        source = pcm_ms(250, 29)
        stream.push(0, source)
        with self.assertRaises(RuntimeError):
            stream.step()
        self.decode(stream)
        self.assertEqual(attempts, [source, source])
        self.assertEqual(adapter.calls[-1][1:3], (0, 250))
        self.assertEqual(stream.state.version, 0)
        self.decode(stream)
        self.assertEqual(adapter.inputs[-1], source + pcm_ms(100, 30))
        stream.close()


if __name__ == "__main__":
    unittest.main()
