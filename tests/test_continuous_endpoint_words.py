"""Hybrid endpoint/word coverage and recovery contracts, without a model or GPU."""

import hashlib
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_audio_endpoints import short_config
from test_continuous_evidence import EvidenceNativeAdapter
from test_continuous_stream import pcm, pcm_ms

from whisper_runtime import AudioSpan, RuntimeStateError, TransactionRetainedError
from whisper_runtime.adapters import (
    NativeDecodeMetadata,
    NativeStreamError,
    NativeTimestampSegment,
    NativeWindowResult,
    StreamEventKind,
)
from whisper_runtime.adapters.audio_evidence import SilencePublication
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)
from whisper_runtime.adapters.word_policy import AlignedPublication


def word_result(window_id, start, end):
    """Globally named words; include the short last word of a closed analysis."""
    words = []
    position = start
    while position < end:
        boundary = min(end, (position // 100 + 1) * 100)
        words.append(
            NativeTimestampSegment(
                AudioSpan(position, boundary),
                f" w{position // 100}",
                (position // 100 + 1,),
            )
        )
        position = boundary
    return NativeWindowResult(
        window_id=window_id,
        start_ms=start,
        end_ms=end,
        text="".join(word.text for word in words).strip(),
        metadata=NativeDecodeMetadata(
            language="en",
            tokens=tuple(token for word in words for token in word.tokens),
            segments=tuple(words),
            timestamps_complete=True,
            avg_logprob=-0.001,
            no_speech_prob=0.01,
        ),
    )


def replace_words(result, transform):
    words = tuple(transform(word) for word in result.metadata.segments)
    return replace(
        result,
        text="".join(word.text for word in words).strip(),
        metadata=replace(
            result.metadata,
            segments=words,
            tokens=tuple(token for word in words for token in word.tokens),
        ),
    )


class ContinuousEndpointWordTests(unittest.TestCase):
    def setUp(self):
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=500,
            max_buffer_ms=600,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            left_context_ms=100,
            source_units=True,
            input_evidence=True,
            endpointing=short_config(),
            word_boundary_fallback=True,
        )

    def stream(self, adapter=None, **kwargs):
        adapter = adapter or EvidenceNativeAdapter(word_result)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="endpoint-words",
            config=kwargs.pop("config", self.config),
            mel_builder=kwargs.pop("mel_builder", lambda content: content),
            **kwargs,
        )
        self.addCleanup(stream.close)
        return stream, adapter

    def decode(self, stream):
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.step(), ())
        return stream.step()

    def drain(self, stream, traces=None):
        events = []
        previous = stream.last_trace
        for _ in range(300):
            if not stream.ready:
                return events
            events.extend(stream.step())
            trace = stream.last_trace
            if trace is not previous and traces is not None:
                traces.append(trace)
            previous = trace
        self.fail("stream did not reach a scheduling boundary")

    def assert_released(self, adapter):
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def assert_coverage(self, events, samples):
        cursor = 0
        revisions = {}
        texts = []
        for event in events:
            if event.kind in (StreamEventKind.PROVISIONAL, StreamEventKind.REPLACE):
                revisions[event.segment_id, event.revision] = event.text
            elif event.kind is StreamEventKind.COMMIT:
                self.assertEqual(event.start_sample, cursor)
                self.assertGreater(event.end_sample, cursor)
                self.assertEqual(event.committed_through_sample, event.end_sample)
                cursor = event.end_sample
                texts.extend(revisions[event.segment_id, event.revision].split())
        self.assertEqual(cursor, samples)
        self.assertEqual(
            sum(event.kind is StreamEventKind.FINAL for event in events), 1
        )
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
        return texts

    def assert_inputs(self, adapter, source, max_window_ms=500):
        for call, content in zip(adapter.calls, adapter.inputs):
            _, start, end, _ = call
            self.assertLessEqual(end - start, max_window_ms)
            self.assertEqual(content, source[start * 32 : start * 32 + len(content)])
        self.assert_released(adapter)

    def test_flag_is_opt_in_and_strictly_boolean(self):
        self.assertIs(ContinuousStreamConfig().word_boundary_fallback, False)
        for value in (0, 1, None, "true", 1.0):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(self.config, word_boundary_fallback=value)

    def test_flag_requires_the_explicit_hybrid_configuration(self):
        for changes in (
            {"source_units": False},
            {"input_evidence": False},
            {"endpointing": None},
            {"left_context_ms": 0},
            {"word_alignment": True},
            {"word_boundary_fallback": False},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_profiles_are_distinct_from_both_legacy_profiles(self):
        for coalesce in (False, True):
            with self.subTest(coalesce=coalesce):
                stream, _ = self.stream(
                    config=replace(self.config, coalesce_previews=coalesce)
                )
                self.assertEqual(
                    stream.profile_id,
                    ("coalesced_" if coalesce else "")
                    + "word_boundary_quiet_endpoint_stream/v1+input_evidence/v1",
                )

    def test_continuous_paced_speech_rolls_without_a_quiet_cut_or_repeated_words(self):
        for coalesce in (False, True):
            with self.subTest(coalesce=coalesce):
                stream, adapter = self.stream(
                    config=replace(self.config, coalesce_previews=coalesce)
                )
                source = pcm_ms(1400, 47)
                events, traces = [], []
                for index in range(14):
                    stream.push(index, source[index * 3200 : (index + 1) * 3200])
                    events.extend(self.drain(stream, traces))
                    self.assertLessEqual(
                        stream.metrics.committed_samples - stream.retained_from_sample,
                        100 * 16,
                    )
                stream.finish_input()
                events.extend(self.drain(stream, traces))
                self.assertEqual(
                    self.assert_coverage(events, 1400 * 16),
                    [f"w{index}" for index in range(14)],
                )
                self.assertTrue(all(t.source_unit is None for t in traces[:-1]))
                self.assertEqual(traces[-1].source_unit.origin, "end_of_input")
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assert_inputs(adapter, source)

    def test_backlogged_eof_rebases_only_after_a_stable_word_prefix(self):
        for coalesce in (False, True):
            with self.subTest(coalesce=coalesce):
                stream, adapter = self.stream(
                    config=replace(self.config, coalesce_previews=coalesce)
                )
                source = pcm_ms(600, 64)
                stream.push(0, source)
                stream.finish_input()
                traces = []
                events = self.drain(stream, traces)
                self.assertEqual(
                    self.assert_coverage(events, 600 * 16),
                    [f"w{index}" for index in range(6)],
                )
                self.assertTrue(any(t.reason == "candidate" for t in traces))
                self.assertFalse(traces[0].eof)
                self.assertTrue(traces[-1].eof)
                self.assert_inputs(adapter, source)

    def test_queued_quiet_endpoint_beyond_window_is_preserved_until_it_fits(self):
        for coalesce in (False, True):
            with self.subTest(coalesce=coalesce):
                stream, adapter = self.stream(
                    config=replace(self.config, coalesce_previews=coalesce)
                )
                source = pcm_ms(500, 900) + pcm_ms(40, 1)
                stream.push(0, source)
                proposal = stream._endpoints[0]
                traces = []
                events = self.drain(stream, traces)
                unit_trace = traces[-1]
                self.assertEqual(unit_trace.source_unit.endpoint, proposal)
                self.assertEqual(unit_trace.analysis_end_sample, 540 * 16)
                self.assertGreater(unit_trace.committed_before_sample, 0)
                self.assertEqual(unit_trace.reason, "source_unit")
                self.assertFalse(unit_trace.eof)
                self.assertTrue(unit_trace.word_publication.final)
                self.assertFalse(any(e.kind is StreamEventKind.FINAL for e in events))
                self.assertEqual(tuple(stream._endpoints), ())
                stream.finish_input()
                events.extend(self.drain(stream))
                self.assertEqual(
                    self.assert_coverage(events, 540 * 16),
                    [f"w{index}" for index in range(6)],
                )
                self.assert_inputs(adapter, source)

    def test_closed_unit_selects_only_new_words_then_clears_context_and_anchor(self):
        stream, adapter = self.stream()
        source = pcm_ms(400, 900) + pcm_ms(100, 1)
        stream.push(0, source[: 400 * 32])
        events = self.drain(stream)
        before = stream.metrics.committed_samples
        self.assertGreater(before, 0)
        self.assertTrue(stream._word_anchor)
        stream.push(1, source[400 * 32 :])
        events.extend(self.drain(stream))
        trace = stream.last_trace
        publication = trace.word_publication
        self.assertIsInstance(publication, AlignedPublication)
        self.assertEqual(publication.start_ms, before // 16)
        self.assertGreater(publication.word_start, 0)
        self.assertNotEqual(publication.text, trace.result.text)
        self.assertEqual(stream._word_anchor, ())
        self.assertEqual(stream.retained_from_sample, 440 * 16)
        self.assertEqual(trace.source_unit.end_sample, 440 * 16)
        self.assertFalse(stream.done)
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assert_coverage(events, 500 * 16)
        self.assert_inputs(adapter, source)

    def test_quiet_observation_before_partial_watermark_is_not_rewritten(self):
        endpoint = replace(
            short_config(), quiet_ms=400, min_unit_ms=400, max_quiet_unit_ms=600
        )
        config = replace(
            self.config, endpointing=endpoint, max_window_ms=600, max_buffer_ms=800
        )
        stream, adapter = self.stream(config=config)
        source = pcm_ms(200, 900) + pcm_ms(400, 1)
        stream.push(0, source[: 500 * 32])
        events = self.drain(stream)
        before = stream.metrics.committed_samples
        self.assertGreater(before, 200 * 16)
        stream.push(1, source[500 * 32 :])
        proposal = stream._endpoints[0]
        self.assertEqual(proposal.quiet_start_sample, 200 * 16)
        events.extend(self.drain(stream))
        trace = stream.last_trace
        self.assertEqual(trace.source_unit.endpoint, proposal)
        self.assertEqual(trace.source_unit.start_sample, 200 * 16)
        self.assertEqual(trace.committed_before_sample, before)
        self.assertEqual(trace.word_publication.start_ms, before // 16)
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assertEqual(
            self.assert_coverage(events, 600 * 16), [f"w{index}" for index in range(6)]
        )
        self.assert_inputs(adapter, source, max_window_ms=600)

    def test_late_endpoint_does_not_promote_or_cross_an_active_preview(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(100, 900))
        events = self.drain(stream)
        stream.push(1, pcm_ms(100, 900))
        stream.step()
        admitted = adapter.calls[-1]
        stream.push(2, pcm_ms(40, 1))
        endpoint = stream._endpoints[0]
        stream.step()
        events.extend(stream.step())
        trace = stream.last_trace
        self.assertEqual(adapter.calls[-1], admitted)
        self.assertIsNone(trace.source_unit)
        self.assertLess(stream.metrics.committed_samples, endpoint.end_sample)
        self.assertEqual(stream._endpoints[0], endpoint)
        events.extend(self.drain(stream))
        self.assertEqual(stream.last_trace.source_unit.endpoint, endpoint)
        self.assertEqual(stream.last_trace.word_publication.start_ms, 100)
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assertEqual(self.assert_coverage(events, 240 * 16), ["w0", "w1", "w2"])
        self.assert_released(adapter)

    def test_new_quiet_unit_starts_without_the_prior_units_anchor(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(200, 900))
        events = self.drain(stream)
        stream.push(1, pcm_ms(40, 1))
        events.extend(self.drain(stream))
        self.assertEqual(stream._word_anchor, ())
        stream.push(2, pcm_ms(100, 900) + pcm_ms(40, 1))
        events.extend(self.drain(stream))
        trace = stream.last_trace
        self.assertEqual(trace.analysis_start_sample, 240 * 16)
        self.assertEqual(trace.word_publication.word_start, 0)
        self.assertEqual(trace.source_unit.start_sample, 240 * 16)
        self.assertEqual(stream._word_anchor, ())
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assert_coverage(events, 380 * 16)
        self.assert_released(adapter)

    def test_cancellation_retries_frozen_preview_despite_new_endpoint_and_eof(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(100, 900))
        stream.step()
        admitted, content = adapter.calls[-1], adapter.inputs[-1]
        stream.push(1, pcm_ms(40, 1))
        stream.finish_input()
        self.assertTrue(stream.cancel_active())
        with self.assertRaises(RuntimeStateError):
            stream.step()
        self.assertEqual(stream.metrics.committed_samples, 0)
        events = self.decode(stream)
        self.assertEqual(adapter.calls[-1], admitted)
        self.assertEqual(adapter.inputs[-1], content)
        self.assertIsNone(stream.last_trace.source_unit)
        self.assertFalse(stream.last_trace.eof)
        events = list(events) + self.drain(stream)
        self.assertEqual(self.assert_coverage(events, 140 * 16), ["w0", "w1"])
        self.assert_released(adapter)

    def test_native_failures_keep_frozen_pcm_and_endpoint_on_retry(self):
        for stage in ("start", "step", "prepare", "finish"):
            with self.subTest(stage=stage):
                stream, adapter = self.stream(
                    EvidenceNativeAdapter(word_result, fail_once=stage)
                )
                source = pcm_ms(100, 900) + pcm_ms(40, 1)
                stream.push(0, source)
                proposal = stream._endpoints[0]
                with self.assertRaisesRegex(RuntimeError, "injected native failure"):
                    self.drain(stream)
                admitted, content = adapter.calls[-1], adapter.inputs[-1]
                self.assertEqual(stream.metrics.committed_samples, 0)
                self.assertEqual(stream.metrics.buffered_samples, len(source) // 2)
                stream.push(1, pcm_ms(100, 900))
                stream.finish_input()
                events = self.decode(stream)
                self.assertEqual(adapter.calls[-1], admitted)
                self.assertEqual(adapter.inputs[-1], content)
                self.assertEqual(stream.last_trace.source_unit.endpoint, proposal)
                self.assertFalse(stream.last_trace.eof)
                events = list(events) + self.drain(stream)
                self.assert_coverage(events, 240 * 16)
                self.assert_released(adapter)

    def test_preprocessing_failure_does_not_readmit_a_later_endpoint(self):
        attempts = []

        def build(content):
            attempts.append(content)
            if len(attempts) == 1:
                raise RuntimeError("preprocessing failed")
            return content

        stream, adapter = self.stream(mel_builder=build)
        stream.push(0, pcm_ms(100, 900))
        with self.assertRaisesRegex(RuntimeError, "preprocessing failed"):
            stream.step()
        stream.push(1, pcm_ms(40, 1))
        stream.finish_input()
        events = self.decode(stream)
        self.assertEqual(attempts[0], attempts[1])
        self.assertIsNone(stream.last_trace.source_unit)
        self.assertFalse(stream.last_trace.eof)
        self.assert_coverage(list(events) + self.drain(stream), 140 * 16)
        self.assert_released(adapter)

    def test_commit_release_recovery_publishes_once_before_evicting_or_clearing(self):
        for closed in (False, True):
            with self.subTest(closed=closed):
                stream, adapter = self.stream()
                stream.push(0, pcm_ms(200 if closed else 100, 900))
                events = self.drain(stream)
                stream.push(1, pcm_ms(40, 1) if closed else pcm_ms(100, 900))
                stream.step()
                stream.step()
                run = adapter.runs[-1]
                before, anchor = stream.metrics, stream._word_anchor
                with patch.object(
                    type(run.transaction._lease),
                    "release",
                    side_effect=RuntimeError("release failed"),
                ):
                    with self.assertRaises(TransactionRetainedError) as raised:
                        stream.step()
                    self.assertIsNotNone(raised.exception.committed_state)
                    self.assertEqual(stream.metrics, before)
                    self.assertEqual(stream._word_anchor, anchor)
                    with self.assertRaisesRegex(NativeStreamError, "retained"):
                        stream.step()
                self.assertTrue(adapter.worker.recover(raised.exception.transaction))
                recovered = stream.step()
                self.assertEqual(
                    sum(e.kind is StreamEventKind.COMMIT for e in recovered), 1
                )
                self.assertEqual(stream._word_anchor == (), closed)
                count = len(adapter.calls)
                self.assertEqual(stream.step(), ())
                self.assertEqual(len(adapter.calls), count)
                stream.finish_input()
                events.extend(recovered)
                events.extend(self.drain(stream))
                self.assert_coverage(events, (240 if closed else 200) * 16)
                self.assert_released(adapter)

    def test_fence_failure_keeps_unpublished_audio_and_anchor_until_recovered(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(200, 900))
        events = self.drain(stream)
        stream.push(1, pcm_ms(40, 1))
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        before, anchor = stream.metrics, stream._word_anchor
        run.fence.fail = True
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        self.assertIsNone(raised.exception.committed_state)
        self.assertEqual(stream.metrics, before)
        self.assertEqual(stream._word_anchor, anchor)
        run.fence.fail = False
        self.assertTrue(adapter.worker.recover(run.transaction))
        stream.step()
        events.extend(self.drain(stream))
        self.assertEqual(adapter.calls[-1], adapter.calls[-2])
        self.assertEqual(adapter.inputs[-1], adapter.inputs[-2])
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assertEqual(self.assert_coverage(events, 240 * 16), ["w0", "w1", "w2"])
        self.assert_released(adapter)

    def test_unstable_no_pause_speech_stops_without_forced_cut_or_eviction(self):
        def unstable(window_id, start, end):
            return replace_words(
                word_result(window_id, start, end),
                lambda word: replace(word, text=f" v{end}"),
            )

        stream, adapter = self.stream(EvidenceNativeAdapter(unstable))
        source = pcm_ms(500, 47)
        stream.push(0, source)
        with self.assertRaisesRegex(StreamNeedsResolutionError, "no stable"):
            self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assertEqual(stream.metrics.buffered_samples, 500 * 16)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(stream.state.version, 0)
        self.assertFalse(stream.done)
        self.assertEqual(stream.last_trace.action, "preview")
        self.assert_inputs(adapter, source)

    def test_missing_quiet_anchor_waits_but_actual_eof_still_refuses(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(200, 900))
        self.drain(stream)
        before = stream.metrics

        def changed(window_id, start, end):
            return replace_words(
                word_result(window_id, start, end),
                lambda word: replace(word, text=" changed"),
            )

        adapter.result_factory = changed
        stream.push(1, pcm_ms(40, 1))
        self.assertEqual(self.drain(stream), [])
        self.assertEqual(stream.last_trace.reason, "anchor_missing")
        self.assertEqual(stream.last_trace.action, "wait_for_input")
        self.assertFalse(stream._endpoints)
        self.assertEqual(stream.metrics.committed_samples, before.committed_samples)
        stream.finish_input()
        with self.assertRaisesRegex(StreamNeedsResolutionError, "anchor_missing"):
            self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, before.committed_samples)
        self.assertEqual(stream.metrics.events_emitted, before.events_emitted)
        self.assertEqual(stream.metrics.buffered_samples, 240 * 16)
        self.assertEqual(len(stream._endpoints), 0)
        calls = len(adapter.calls)
        with self.assertRaisesRegex(StreamNeedsResolutionError, "input boundary"):
            stream.step()
        self.assertEqual(len(adapter.calls), calls)
        self.assert_released(adapter)

    def test_weak_nonzero_quiet_is_not_silence_and_uncertainty_retains_it(self):
        for confident in (False, True):
            with self.subTest(confident=confident):

                def result(window_id, start, end):
                    native = word_result(window_id, start, end)
                    return (
                        native
                        if confident
                        else replace(
                            native,
                            metadata=replace(native.metadata, no_speech_prob=None),
                        )
                    )

                stream, adapter = self.stream(EvidenceNativeAdapter(result))
                source = pcm_ms(200, 1)
                stream.push(0, source)
                if confident:
                    events = self.drain(stream)
                    self.assertIsInstance(
                        stream.state.windows[-1].result, AlignedPublication
                    )
                    self.assertIsNone(stream.last_trace.silence_publication)
                    self.assertTrue(events[0].text)
                    stream.finish_input()
                    self.assert_coverage(events + self.drain(stream), 200 * 16)
                else:
                    with self.assertRaises(StreamNeedsResolutionError):
                        self.drain(stream)
                    self.assertEqual(stream.metrics.committed_samples, 0)
                    self.assertEqual(stream.metrics.buffered_samples, 200 * 16)
                    self.assertEqual(
                        stream.last_trace.audio_evidence.state, "uncertain"
                    )
                    self.assertFalse(stream.done)
                self.assert_inputs(adapter, source)

    def test_exact_zero_closed_unit_uses_silence_without_word_alignment(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(200))
        events = self.drain(stream)
        self.assertIsInstance(stream.state.windows[-1].result, SilencePublication)
        self.assertIsNone(stream.last_trace.word_alignment)
        self.assertEqual(stream._word_anchor, ())
        self.assertEqual(stream.retained_from_sample, 200 * 16)
        stream.finish_input()
        self.assertEqual(
            self.assert_coverage(events + self.drain(stream), 200 * 16), []
        )
        self.assert_released(adapter)

    def test_fractional_final_pcm_is_covered_and_hashed_without_padding(self):
        stream, adapter = self.stream()
        source = pcm_ms(600, 64) + pcm(7, 45)
        stream.push(0, source[: 400 * 32])
        events = self.drain(stream)
        stream.push(1, source[400 * 32 :])
        stream.finish_input()
        events.extend(self.drain(stream))
        self.assertEqual(
            self.assert_coverage(events, 600 * 16 + 7),
            [f"w{index}" for index in range(6)],
        )
        observation = stream.last_trace.audio_evidence.observation
        self.assertEqual(
            observation.pcm_sha256, hashlib.sha256(adapter.inputs[-1]).hexdigest()
        )
        self.assertEqual(observation.sample_count, len(adapter.inputs[-1]) // 2)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assert_inputs(adapter, source)


if __name__ == "__main__":
    unittest.main()
