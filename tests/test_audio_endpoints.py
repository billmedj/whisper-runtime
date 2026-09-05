"""Causal quiet-run proposals retain PCM and use the existing transaction path."""

import hashlib
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from test_continuous_evidence import EvidenceNativeAdapter, speech_result, you_result
from test_continuous_stream import pcm, pcm_ms

from whisper_runtime import RuntimeStateError, TransactionRetainedError
from whisper_runtime.adapters import (
    AudioBufferFullError,
    AudioSequenceError,
    NativeStreamError,
    StreamEventKind,
)
from whisper_runtime.adapters.audio_endpoints import (
    QuietEndpointConfig,
    QuietEndpointDetector,
    QuietEndpointProposal,
)
from whisper_runtime.adapters.audio_evidence import SilencePublication
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    SourceUnit,
    StreamNeedsResolutionError,
)


def short_config():
    return QuietEndpointConfig(
        frame_ms=20,
        quiet_ms=40,
        min_unit_ms=60,
        max_quiet_unit_ms=200,
        quiet_peak=32,
    )


class QuietEndpointDetectorTests(unittest.TestCase):
    def test_defaults_are_explicit_and_frozen(self):
        config = QuietEndpointConfig()
        self.assertEqual(
            (
                config.frame_ms,
                config.quiet_ms,
                config.min_unit_ms,
                config.max_quiet_unit_ms,
                config.quiet_peak,
            ),
            (20, 600, 1000, 10000, 32),
        )
        with self.assertRaises(FrozenInstanceError):
            config.quiet_peak = 0
        self.assertEqual(QuietEndpointDetector().accepted_samples, 0)

    def test_configuration_rejects_ambiguous_types_and_invalid_ranges(self):
        config = short_config()
        for name in config.__dataclass_fields__:
            for value in (True, False, None, 1.0, "20"):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(TypeError),
                ):
                    replace(config, **{name: value})
        for changes in (
            {"frame_ms": 0},
            {"frame_ms": 101},
            {"quiet_ms": 0},
            {"quiet_ms": 41},
            {"min_unit_ms": 61},
            {"max_quiet_unit_ms": 201},
            {"quiet_ms": 80},
            {"min_unit_ms": 220},
            {"max_quiet_unit_ms": 30020},
            {"quiet_peak": -1},
            {"quiet_peak": 32768},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)
        for value in (False, {}, 1):
            with self.subTest(value=value), self.assertRaises(TypeError):
                QuietEndpointDetector(value)

    def test_proposals_do_not_depend_on_pcm_chunk_boundaries(self):
        source = (
            pcm_ms(20, 400)
            + pcm_ms(40, -17)
            + pcm_ms(200)
            + pcm_ms(20, -32768)
            + pcm_ms(40, 32)
            + pcm(7, 1)
        )
        size = len(source) // 2
        whole = QuietEndpointDetector(short_config()).observe(0, source)
        self.assertEqual(
            [(item.end_sample, item.quiet_start_sample, item.peak) for item in whole],
            [(60 * 16, 20 * 16, 17), (260 * 16, 60 * 16, 0), (320 * 16, 280 * 16, 32)],
        )
        generator = random.Random(729)
        random_parts = []
        remaining = size
        while remaining:
            part = min(remaining, generator.randint(1, 799))
            random_parts.append(part)
            remaining -= part
        for parts in ((size,), (1,) * size, tuple(random_parts)):
            with self.subTest(parts=len(parts)):
                detector = QuietEndpointDetector(short_config())
                offset = 0
                proposals = []
                for part in parts:
                    proposals.extend(
                        detector.observe(
                            offset, source[offset * 2 : (offset + part) * 2]
                        )
                    )
                    offset += part
                    self.assertEqual(detector.accepted_samples, offset)
                self.assertEqual(tuple(proposals), whole)
        with self.assertRaises(FrozenInstanceError):
            whole[0].end_sample = 1

    def test_detection_is_causal_and_uses_only_complete_frames(self):
        detector = QuietEndpointDetector(short_config())
        source = pcm_ms(20, 33) + pcm_ms(40, 32)
        self.assertEqual(detector.observe(0, source[:-2]), ())
        self.assertEqual(detector.accepted_samples, 60 * 16 - 1)
        proposals = detector.observe(60 * 16 - 1, source[-2:])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].end_sample, 60 * 16)
        self.assertEqual(proposals[0].quiet_start_sample, 20 * 16)
        self.assertLess(proposals[0].quiet_start_sample, proposals[0].end_sample)

    def test_one_nonquiet_sample_resets_the_entire_frame_and_quiet_run(self):
        detector = QuietEndpointDetector(short_config())
        source = pcm_ms(20, 40) + pcm_ms(20) + pcm(319) + pcm(1, -33) + pcm_ms(40, 32)
        proposals = detector.observe(0, source)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].end_sample, 100 * 16)
        self.assertEqual(proposals[0].quiet_start_sample, 60 * 16)

    def test_pure_quiet_runs_have_a_separate_bounded_unit_interval(self):
        for value in (0, 1, -32):
            with self.subTest(value=value):
                detector = QuietEndpointDetector(short_config())
                self.assertEqual(detector.observe(0, pcm_ms(199, value)), ())
                proposals = detector.observe(199 * 16, pcm_ms(401, value))
                self.assertEqual(
                    [item.end_sample for item in proposals], [3200, 6400, 9600]
                )
                self.assertEqual(
                    [item.quiet_start_sample for item in proposals], [0, 3200, 6400]
                )
                self.assertTrue(all(item.peak == abs(value) for item in proposals))

    def test_rejected_observations_do_not_change_detector_history(self):
        detector = QuietEndpointDetector(short_config())
        first = pcm_ms(20, 40) + pcm(7)
        detector.observe(0, first)
        cursor = len(first) // 2
        for position, payload, error in (
            (True, pcm(1), TypeError),
            (float(cursor), pcm(1), TypeError),
            (-1, pcm(1), ValueError),
            (cursor - 1, pcm(1), ValueError),
            (cursor + 1, pcm(1), ValueError),
            (cursor, bytearray(2), TypeError),
            (cursor, "pcm", TypeError),
            (cursor, b"", ValueError),
            (cursor, b"\x00", ValueError),
        ):
            with self.subTest(position=position, payload=payload):
                with self.assertRaises(error):
                    detector.observe(position, payload)
                self.assertEqual(detector.accepted_samples, cursor)
        remainder = pcm(40 * 16 - 7)
        actual = detector.observe(cursor, remainder)
        control = QuietEndpointDetector(short_config()).observe(0, first + remainder)
        self.assertEqual(actual, control)

    def test_no_endpoint_is_invented_for_a_continuous_nonquiet_signal(self):
        detector = QuietEndpointDetector(short_config())
        self.assertEqual(detector.observe(0, pcm_ms(1500, 33)), ())
        self.assertEqual(detector.accepted_samples, 1500 * 16)

    def test_proposal_metadata_is_typed_and_cannot_escape_its_source_unit(self):
        proposal = QuietEndpointProposal(960, 320, 32)
        for name in proposal.__dataclass_fields__:
            for value in (True, 1.0, "0", None):
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    replace(proposal, **{name: value})
        for changes in (
            {"quiet_start_sample": -1},
            {"quiet_start_sample": 960},
            {"end_sample": 0},
            {"peak": -1},
            {"peak": 32769},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(proposal, **changes)
        unit = SourceUnit(0, 960, "quiet_run", proposal)
        for changes in (
            {"end_sample": 961},
            {"start_sample": 321},
            {"origin": "caller"},
            {"origin": "end_of_input"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(unit, **changes)
        for value in (None, {}, True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(unit, endpoint=value)


class QuietEndpointStreamTests(unittest.TestCase):
    def setUp(self):
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=500,
            max_buffer_ms=600,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            source_units=True,
            input_evidence=True,
            endpointing=short_config(),
        )

    def stream(self, adapter=None, **kwargs):
        adapter = adapter or EvidenceNativeAdapter(you_result)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="quiet-endpoint-test",
            config=kwargs.pop("config", self.config),
            mel_builder=kwargs.pop("mel_builder", lambda source: source),
            **kwargs,
        )
        self.addCleanup(stream.close)
        return stream, adapter

    def decode(self, stream):
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.step(), ())
        return stream.step()

    def drain(self, stream):
        events = []
        for _ in range(100):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("stream did not reach a scheduling boundary")

    def assert_released(self, adapter):
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_auto_profile_is_explicit_and_requires_source_units(self):
        self.assertIsNone(ContinuousStreamConfig().endpointing)
        for value in (True, 1, {}, "quiet"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(self.config, endpointing=value)
        with self.assertRaises(ValueError):
            replace(self.config, source_units=False)
        with self.assertRaises(ValueError):
            replace(
                self.config,
                endpointing=replace(short_config(), max_quiet_unit_ms=520),
            )
        stream, _ = self.stream()
        self.assertEqual(
            stream.profile_id, "quiet_endpoint_stream/v1+input_evidence/v1"
        )
        coalesced, _ = self.stream(config=replace(self.config, coalesce_previews=True))
        self.assertEqual(
            coalesced.profile_id,
            "coalesced_quiet_endpoint_stream/v1+input_evidence/v1",
        )
        stream.push(0, pcm_ms(100, 44))
        before = stream.metrics
        with self.assertRaises(NativeStreamError):
            stream.seal_unit(1600)
        self.assertEqual(stream.metrics, before)

    def test_early_endpoint_is_ready_before_preview_cadence(self):
        stream, adapter = self.stream()
        source = pcm_ms(20, 40) + pcm_ms(40)
        stream.push(0, source[:-2])
        self.assertFalse(stream.ready)
        stream.push(1, source[-2:])
        self.assertTrue(stream.ready)
        events = self.decode(stream)
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual(stream.metrics.committed_samples, 60 * 16)
        unit = stream.last_trace.source_unit
        self.assertEqual(unit.origin, "quiet_run")
        self.assertEqual(unit.end_sample, 60 * 16)
        self.assertEqual(unit.endpoint.end_sample, unit.end_sample)
        self.assertEqual(unit.endpoint.quiet_start_sample, 20 * 16)
        self.assertEqual(adapter.inputs, [source])
        self.assert_released(adapter)

    def test_multiple_queued_units_are_contiguous_and_eof_does_not_cut_the_queue(self):
        first = pcm_ms(100, 41) + pcm_ms(40)
        second = pcm_ms(100, 42) + pcm_ms(40)
        tail = pcm(7, 43)
        source = first + second + tail
        expected = [first, second, tail]
        stream, adapter = self.stream()
        stream.push(0, source)
        stream.finish_input()
        events = []
        start = 0
        for index, chunk in enumerate(expected):
            events.extend(self.decode(stream))
            end = start + len(chunk) // 2
            unit = stream.last_trace.source_unit
            self.assertEqual((unit.start_sample, unit.end_sample), (start, end))
            self.assertEqual(unit.origin, "quiet_run" if index < 2 else "end_of_input")
            self.assertEqual(stream.last_trace.eof, index == 2)
            self.assertEqual(stream.metrics.committed_samples, end)
            self.assertEqual(stream.metrics.buffered_samples, len(source) // 2 - end)
            self.assertEqual(
                stream.last_trace.audio_evidence.observation.pcm_sha256,
                hashlib.sha256(chunk).hexdigest(),
            )
            start = end
        self.assertEqual(adapter.inputs, expected)
        self.assertEqual(
            sum(event.kind is StreamEventKind.COMMIT for event in events), 3
        )
        self.assertEqual(
            sum(event.kind is StreamEventKind.FINAL for event in events), 1
        )
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assertTrue(stream.done)
        self.assert_released(adapter)

    def test_quiet_nonzero_samples_are_transcribed_not_certified_as_silence(self):
        for value in (1, -32):
            with self.subTest(value=value):
                stream, adapter = self.stream()
                source = pcm_ms(200, value)
                stream.push(0, source)
                events = self.decode(stream)
                self.assertEqual(events[0].text, "you you")
                self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
                self.assertEqual(stream.last_trace.source_unit.origin, "quiet_run")
                self.assertIsNone(stream.last_trace.silence_publication)
                self.assertFalse(
                    stream.last_trace.audio_evidence.observation.digital_silence
                )
                self.assertNotIsInstance(
                    stream.state.windows[-1].result, SilencePublication
                )
                self.assertEqual(adapter.inputs, [source])
                self.assert_released(adapter)

    def test_a_false_quiet_endpoint_cannot_override_uncertain_speech_evidence(self):
        stream, adapter = self.stream(EvidenceNativeAdapter())
        source = pcm_ms(200, 1)
        stream.push(0, source)
        with self.assertRaises(StreamNeedsResolutionError):
            self.decode(stream)
        self.assertEqual(stream.last_trace.source_unit.origin, "quiet_run")
        self.assertEqual(stream.last_trace.action, "unresolved")
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assertEqual(stream.metrics.buffered_samples, len(source) // 2)
        self.assertEqual(stream.state.version, 0)
        before = stream.metrics
        for _ in range(3):
            with self.assertRaises(StreamNeedsResolutionError):
                stream.step()
        self.assertEqual(stream.metrics, before)
        self.assertEqual(adapter.inputs, [source])
        self.assert_released(adapter)

    def test_exact_silence_still_uses_the_existing_typed_publication(self):
        stream, adapter = self.stream()
        source = pcm_ms(400)
        stream.push(0, source)
        stream.finish_input()
        events = []
        for end in (200 * 16, 400 * 16):
            events.extend(self.decode(stream))
            self.assertEqual(stream.last_trace.reason, "digital_silence")
            self.assertEqual(stream.last_trace.result.text, "you you")
            self.assertEqual(stream.last_trace.source_unit.origin, "quiet_run")
            self.assertEqual(stream.last_trace.source_unit.end_sample, end)
            self.assertIsInstance(stream.state.windows[-1].result, SilencePublication)
        self.assertTrue(all(event.text in (None, "") for event in events))
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
        self.assertEqual(adapter.inputs, [pcm_ms(200)] * 2)
        self.assert_released(adapter)

    def test_rejected_pushes_do_not_advance_endpoint_state(self):
        stream, adapter = self.stream()
        source = pcm_ms(100, 45)
        stream.push(0, source)
        before = stream.metrics
        for sequence, payload, error in (
            (2, pcm_ms(40), AudioSequenceError),
            (True, pcm_ms(40), TypeError),
            (1, b"\x00", ValueError),
            (1, pcm_ms(501), AudioBufferFullError),
        ):
            with self.subTest(sequence=sequence, error=error.__name__):
                with self.assertRaises(error):
                    stream.push(sequence, payload)
                self.assertEqual(stream.metrics, before)
        stream.push(1, pcm_ms(40))
        events = self.decode(stream)
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual(stream.last_trace.source_unit.end_sample, 140 * 16)
        self.assertEqual(adapter.inputs, [source + pcm_ms(40)])
        self.assert_released(adapter)

    def test_global_eof_does_not_pad_a_partial_quiet_frame(self):
        stream, adapter = self.stream()
        source = pcm_ms(20, 46) + pcm_ms(40)[:-2]
        stream.push(0, source)
        self.assertFalse(stream.ready)
        stream.finish_input()
        events = self.decode(stream)
        unit = stream.last_trace.source_unit
        self.assertEqual(unit.origin, "end_of_input")
        self.assertIsNone(unit.endpoint)
        self.assertEqual(unit.end_sample, len(source) // 2)
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
        self.assertEqual(adapter.inputs, [source])
        self.assert_released(adapter)

    def test_missing_quiet_boundary_stops_at_window_without_evicting_input(self):
        stream, adapter = self.stream()
        source = pcm_ms(600, 47)
        stream.push(0, source)
        with self.assertRaises(StreamNeedsResolutionError):
            self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assertEqual(stream.metrics.buffered_samples, len(source) // 2)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertIsNone(stream.last_trace.source_unit)
        self.assertFalse(stream.done)
        self.assertTrue(all(len(chunk) // 2 <= 500 * 16 for chunk in adapter.inputs))
        self.assert_released(adapter)

    def test_failed_endpoint_retry_keeps_exact_unit_after_more_pcm_and_eof(self):
        for stage in ("preprocess", "start", "step", "prepare", "finish"):
            with self.subTest(stage=stage):
                attempts = []

                def build(source):
                    attempts.append(source)
                    if stage == "preprocess" and len(attempts) == 1:
                        raise RuntimeError("preprocessing failed")
                    return source

                adapter = EvidenceNativeAdapter(speech_result, fail_once=stage)
                stream, adapter = self.stream(adapter, mel_builder=build)
                first = pcm_ms(100, 48) + pcm_ms(40)
                second = pcm_ms(100, 49) + pcm_ms(40)
                stream.push(0, first)
                with self.assertRaises(RuntimeError):
                    self.decode(stream)
                stream.push(1, second)
                stream.finish_input()
                events = self.decode(stream)
                self.assertEqual(stream.last_trace.source_unit.end_sample, 140 * 16)
                self.assertEqual(stream.last_trace.source_unit.origin, "quiet_run")
                self.assertFalse(stream.last_trace.eof)
                self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
                self.assertEqual(attempts[:2], [first, first])
                self.assertEqual(stream.metrics.buffered_samples, len(second) // 2)
                final = self.decode(stream)
                self.assertEqual(final[-1].kind, StreamEventKind.FINAL)
                self.assertEqual(adapter.inputs[-1], second)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assert_released(adapter)

    def test_endpoint_arriving_during_cancelled_preview_does_not_promote_retry(self):
        stream, adapter = self.stream()
        preview = pcm_ms(100, 50)
        stream.push(0, preview)
        stream.step()
        stream.push(1, pcm_ms(40))
        stream.finish_input()
        self.assertTrue(stream.cancel_active())
        with self.assertRaises(RuntimeStateError):
            stream.step()
        events = self.decode(stream)
        self.assertIsNone(stream.last_trace.source_unit)
        self.assertFalse(stream.last_trace.eof)
        self.assertFalse(any(event.kind is StreamEventKind.COMMIT for event in events))
        self.assertEqual(stream.metrics.committed_samples, 0)
        final = self.decode(stream)
        self.assertEqual(final[-1].kind, StreamEventKind.FINAL)
        self.assertEqual(stream.last_trace.source_unit.origin, "quiet_run")
        self.assertEqual(adapter.inputs, [preview, preview, preview + pcm_ms(40)])
        self.assertEqual(adapter.calls[0][0], adapter.calls[1][0])
        self.assertNotEqual(adapter.calls[1][0], adapter.calls[2][0])
        self.assert_released(adapter)

    def test_future_oversized_endpoint_does_not_block_a_frozen_preview_retry(self):
        for stage in ("preprocess", "cancel"):
            with self.subTest(stage=stage):
                attempts = []

                def build(source):
                    attempts.append(source)
                    if stage == "preprocess" and len(attempts) == 1:
                        raise RuntimeError("preprocessing failed")
                    return source

                stream, adapter = self.stream(mel_builder=build)
                preview = pcm_ms(100, 51)
                stream.push(0, preview)
                if stage == "preprocess":
                    with self.assertRaisesRegex(RuntimeError, "preprocessing"):
                        stream.step()
                else:
                    stream.step()
                    self.assertTrue(stream.cancel_active())
                    with self.assertRaises(RuntimeStateError):
                        stream.step()
                stream.push(1, pcm_ms(400, 52) + pcm_ms(40))
                events = self.decode(stream)
                self.assertIsNone(stream.last_trace.source_unit)
                self.assertFalse(
                    any(event.kind is StreamEventKind.COMMIT for event in events)
                )
                self.assertEqual(attempts[:2], [preview, preview])
                self.assertEqual(stream.metrics.committed_samples, 0)
                with self.assertRaisesRegex(StreamNeedsResolutionError, "window"):
                    stream.step()
                self.assertEqual(stream.metrics.buffered_samples, 540 * 16)
                self.assert_released(adapter)

    def test_producer_can_admit_an_endpoint_while_the_owner_has_an_active_preview(self):
        stream, adapter = self.stream()
        preview = pcm_ms(100, 53)
        stream.push(0, preview)
        stream.step()
        with ThreadPoolExecutor(max_workers=1) as executor:
            accepted = executor.submit(stream.push, 1, pcm_ms(40)).result(timeout=2)
            self.assertEqual(accepted, 140 * 16)
        self.assertEqual(stream.step(), ())
        events = stream.step()
        self.assertIsNone(stream.last_trace.source_unit)
        self.assertFalse(any(event.kind is StreamEventKind.COMMIT for event in events))
        final = self.decode(stream)
        self.assertEqual(final[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual(stream.last_trace.source_unit.origin, "quiet_run")
        self.assertEqual(adapter.inputs, [preview, preview + pcm_ms(40)])
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assert_released(adapter)

    def test_endpoint_queue_advances_once_after_fence_or_release_recovery(self):
        for value in (0, 54):
            for phase in ("fence", "release"):
                with self.subTest(value=value, phase=phase):
                    stream, adapter = self.stream()
                    unit_pcm = (
                        pcm_ms(200) if value == 0 else pcm_ms(100, value) + pcm_ms(40)
                    )
                    unit_samples = len(unit_pcm) // 2
                    stream.push(0, unit_pcm)
                    stream.step()
                    stream.step()
                    before = stream.metrics
                    run = adapter.runs[-1]
                    context = (
                        patch.object(
                            type(run.transaction._lease),
                            "release",
                            side_effect=RuntimeError("release failed"),
                        )
                        if phase == "release"
                        else patch.object(run.fence, "fail", True)
                    )
                    with context:
                        with self.assertRaises(TransactionRetainedError) as raised:
                            stream.step()
                        self.assertEqual(
                            raised.exception.committed_state is not None,
                            phase == "release",
                        )
                        self.assertEqual(stream.metrics, before)
                        self.assertEqual(stream.retained_from_sample, 0)
                        stream.push(1, unit_pcm)
                        stream.finish_input()
                        with self.assertRaisesRegex(NativeStreamError, "retained"):
                            stream.step()
                    self.assertTrue(adapter.worker.recover(run.transaction))
                    first = stream.step()
                    if phase == "fence":
                        self.assertEqual(first, ())
                        first = self.decode(stream)
                    self.assertEqual(first[-1].kind, StreamEventKind.COMMIT)
                    self.assertEqual(stream.metrics.committed_samples, unit_samples)
                    self.assertEqual(stream.metrics.buffered_samples, unit_samples)
                    self.assertFalse(stream.last_trace.eof)
                    self.assertEqual(
                        stream.last_trace.source_unit.end_sample, unit_samples
                    )
                    second = self.decode(stream)
                    self.assertEqual(second[-1].kind, StreamEventKind.FINAL)
                    self.assertTrue(stream.last_trace.eof)
                    self.assertEqual(
                        stream.last_trace.source_unit.start_sample, unit_samples
                    )
                    self.assertEqual(
                        stream.last_trace.source_unit.end_sample, unit_samples * 2
                    )
                    self.assertEqual(stream.state.version, 2)
                    self.assertEqual(stream.metrics.decode_count, 2)
                    self.assertEqual(stream.metrics.buffered_samples, 0)
                    self.assertEqual(len(adapter.calls), 3 if phase == "fence" else 2)
                    self.assert_released(adapter)


if __name__ == "__main__":
    unittest.main()
