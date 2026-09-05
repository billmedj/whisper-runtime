"""Caller-sealed source coverage uses native transactions, never text overlap."""

import hashlib
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from test_continuous_evidence import EvidenceNativeAdapter, speech_result, you_result
from test_continuous_stream import pcm, pcm_ms

from whisper_runtime import RuntimeStateError, TransactionRetainedError
from whisper_runtime.adapters import NativeStreamError, StreamEventKind
from whisper_runtime.adapters.audio_evidence import SilencePublication
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)


class SourceUnitTests(unittest.TestCase):
    def setUp(self):
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=500,
            max_buffer_ms=600,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            source_units=True,
            input_evidence=True,
        )

    def stream(self, adapter=None, **kwargs):
        adapter = adapter or EvidenceNativeAdapter(speech_result)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="source-unit-test",
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

    def assert_unit(self, stream, start, end, origin="caller", *, eof=False):
        trace = stream.last_trace
        self.assertEqual(trace.eof, eof)
        self.assertEqual(trace.source_unit.start_sample, start)
        self.assertEqual(trace.source_unit.end_sample, end)
        self.assertEqual(trace.source_unit.origin, origin)
        self.assertEqual(trace.analysis_start_sample, start)
        self.assertEqual(trace.analysis_end_sample, end)
        return trace.source_unit

    def test_opt_in_flag_and_incompatible_configuration_are_rejected(self):
        self.assertIs(ContinuousStreamConfig().source_units, False)
        for value in (0, 1, None, "true", 1.0):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(self.config, source_units=value)
        for changes in (
            {"input_evidence": False},
            {"left_context_ms": 100},
            {"word_alignment": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)
        for coalesce in (False, True):
            stream, _ = self.stream(
                config=replace(self.config, coalesce_previews=coalesce)
            )
            self.assertEqual(
                stream.profile_id,
                ("coalesced_" if coalesce else "")
                + "source_unit_stream/v1+input_evidence/v1",
            )

    def test_open_units_do_not_commit_even_confident_speech_or_exact_zeros(self):
        for value in (0, 900):
            for coalesce in (False, True):
                with self.subTest(value=value, coalesce=coalesce):
                    stream, adapter = self.stream(
                        EvidenceNativeAdapter(you_result),
                        config=replace(self.config, coalesce_previews=coalesce),
                    )
                    source = pcm_ms(250, value)
                    stream.push(0, source)
                    events = self.drain(stream)
                    self.assertTrue(events)
                    self.assertFalse(
                        any(event.kind is StreamEventKind.COMMIT for event in events)
                    )
                    if not value:
                        self.assertTrue(all(not event.text for event in events))
                    self.assertEqual(stream.metrics.committed_samples, 0)
                    self.assertEqual(stream.metrics.buffered_samples, len(source) // 2)
                    self.assertEqual(stream.state.version, 0)
                    self.assertIsNone(stream.last_trace.source_unit)
                    self.assertEqual(stream.last_trace.reason, "source_unit_open")
                    self.assert_released(adapter)

    def test_sealed_fractional_units_are_partition_invariant_and_keep_repeated_text(
        self,
    ):
        source = pcm_ms(250, 21) + pcm(7, 22)
        outputs = []
        for parts in ((4007,), (7, 1000, 3000)):
            with self.subTest(parts=parts):
                stream, adapter = self.stream(EvidenceNativeAdapter(you_result))
                offset = 0
                for sequence, samples in enumerate(parts):
                    stream.push(sequence, source[offset * 2 : (offset + samples) * 2])
                    offset += samples
                events = []
                for start, end in ((0, 1607), (1607, 4007)):
                    stream.seal_unit(end)
                    batch = self.decode(stream)
                    events.extend(batch)
                    unit = self.assert_unit(stream, start, end)
                    with self.assertRaises(FrozenInstanceError):
                        unit.end_sample = end + 1
                    self.assertEqual(stream.last_trace.reason, "source_unit")
                    self.assertEqual(batch[0].text, "you")
                    self.assertEqual(batch[-1].kind, StreamEventKind.COMMIT)
                    self.assertEqual(batch[-1].committed_through_sample, end)
                    self.assertEqual(stream.retained_from_sample, end)
                    self.assertFalse(stream.done)
                self.assertNotEqual(events[0].segment_id, events[2].segment_id)
                stream.finish_input()
                events.extend(self.drain(stream))
                self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assertEqual(stream.metrics.committed_samples, 4007)
                self.assertEqual(stream.state.version, 2)
                self.assertEqual(
                    [event.sequence_number for event in events], list(range(1, 6))
                )
                self.assertEqual(
                    adapter.inputs, [source[: 1607 * 2], source[1607 * 2 :]]
                )
                outputs.append((events, stream.state, adapter.inputs))
                self.assert_released(adapter)
        self.assertEqual(outputs[0], outputs[1])

    def test_eof_closes_remaining_unit_without_finalizing_a_prior_seal_early(self):
        stream, adapter = self.stream(EvidenceNativeAdapter(you_result))
        source = pcm_ms(250, 31) + pcm(7, 32)
        stream.push(0, source)
        stream.seal_unit(1607)
        stream.finish_input()
        first = self.decode(stream)
        self.assert_unit(stream, 0, 1607, eof=False)
        self.assertEqual(first[-1].kind, StreamEventKind.COMMIT)
        self.assertFalse(stream.done)
        second = self.decode(stream)
        self.assert_unit(stream, 1607, 4007, "end_of_input", eof=True)
        self.assertEqual(second[-1].kind, StreamEventKind.FINAL)
        self.assertEqual(stream.metrics.committed_samples, 4007)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assertTrue(stream.done)
        self.assertEqual(stream.step(), ())
        self.assertEqual(len(adapter.calls), 2)
        self.assert_released(adapter)

    def test_sealed_silence_uses_typed_empty_publication_and_exact_unit_coverage(self):
        stream, adapter = self.stream(EvidenceNativeAdapter(you_result))
        source = pcm_ms(250) + pcm(7)
        stream.push(0, source)
        stream.seal_unit(1607)
        events = self.decode(stream)
        self.assertEqual(events[0].text, "")
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assert_unit(stream, 0, 1607)
        self.assertEqual(stream.last_trace.reason, "digital_silence")
        self.assertIsInstance(stream.state.windows[-1].result, SilencePublication)
        self.assertEqual(stream.metrics.buffered_samples, 2400)
        self.assertFalse(stream.done)
        stream.finish_input()
        self.drain(stream)
        self.assert_unit(stream, 1607, 4007, "end_of_input", eof=True)
        self.assertEqual(stream.metrics.committed_samples, 4007)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assert_released(adapter)

    def test_seal_validation_is_atomic_owner_only_and_boundary_is_immutable(self):
        stream, _ = self.stream()
        stream.push(0, pcm_ms(600, 7))
        before = stream.metrics
        for end in (True, None, 1.5, "1600"):
            with self.subTest(end=end), self.assertRaises(TypeError):
                stream.seal_unit(end)
        for end in (-1, 0, 500 * 16 + 1, 600 * 16 + 1):
            with (
                self.subTest(end=end),
                self.assertRaises((ValueError, NativeStreamError)),
            ):
                stream.seal_unit(end)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaises(NativeStreamError):
                executor.submit(stream.seal_unit, 1600).result(timeout=2)
        self.assertEqual(stream.metrics, before)
        stream.seal_unit(1600)
        for end in (1600, 3200):
            with (
                self.subTest(end=end),
                self.assertRaises((ValueError, NativeStreamError)),
            ):
                stream.seal_unit(end)
        self.decode(stream)
        self.assertEqual(stream.metrics.committed_samples, 1600)
        self.assertEqual(stream.metrics.buffered_samples, 500 * 16)
        with self.assertRaises((ValueError, NativeStreamError)):
            stream.seal_unit(1600)
        legacy, _ = self.stream(config=replace(self.config, source_units=False))
        legacy.push(0, pcm_ms(100, 8))
        with self.assertRaises(NativeStreamError):
            legacy.seal_unit(1600)

    def test_seal_cannot_rewind_completed_active_or_retry_preview(self):
        for phase in ("completed", "active", "retry"):
            with self.subTest(phase=phase):
                attempts = []

                def build(content):
                    attempts.append(content)
                    if phase == "retry" and len(attempts) == 1:
                        raise RuntimeError("preprocessing failed")
                    return content

                stream, adapter = self.stream(mel_builder=build)
                stream.push(0, pcm_ms(250, 43))
                if phase == "completed":
                    self.decode(stream)
                elif phase == "active":
                    stream.step()
                else:
                    with self.assertRaisesRegex(RuntimeError, "preprocessing"):
                        stream.step()
                with self.assertRaises((ValueError, NativeStreamError)):
                    stream.seal_unit(1599)
                stream.seal_unit(1600)
                if phase == "active":
                    self.assertEqual(stream.step(), ())
                    preview = stream.step()
                elif phase == "retry":
                    preview = self.decode(stream)
                else:
                    preview = ()
                self.assertFalse(
                    any(event.kind is StreamEventKind.COMMIT for event in preview)
                )
                self.assertEqual(stream.metrics.committed_samples, 0)
                self.assertIsNone(stream.last_trace.source_unit)
                final = self.decode(stream)
                self.assertEqual(final[-1].kind, StreamEventKind.COMMIT)
                self.assert_unit(stream, 0, 1600)
                self.assertEqual(adapter.inputs, [pcm_ms(100, 43)] * 2)
                self.assertNotEqual(adapter.calls[0][0], adapter.calls[1][0])
                self.assert_released(adapter)

    def test_failed_sealed_analysis_retries_exact_pcm_unit_and_eof_snapshot(self):
        for stage in ("preprocess", "start", "step", "prepare", "finish"):
            with self.subTest(stage=stage):
                attempts = []

                def build(content):
                    attempts.append(content)
                    if stage == "preprocess" and len(attempts) == 1:
                        raise RuntimeError("preprocessing failed")
                    return content

                adapter = EvidenceNativeAdapter(speech_result, fail_once=stage)
                stream, adapter = self.stream(adapter, mel_builder=build)
                source = pcm_ms(250, 44) + pcm(7, 45)
                added = pcm_ms(100, 46)
                stream.push(0, source)
                stream.seal_unit(len(source) // 2)
                with self.assertRaises(RuntimeError):
                    self.decode(stream)
                stream.push(1, added)
                stream.finish_input()
                events = self.decode(stream)
                self.assert_unit(stream, 0, len(source) // 2, eof=False)
                evidence = stream.last_trace.audio_evidence.observation
                self.assertEqual(
                    evidence.pcm_sha256, hashlib.sha256(source).hexdigest()
                )
                self.assertEqual(evidence.sample_count, len(source) // 2)
                self.assertEqual(attempts[:2], [source, source])
                self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
                self.assertEqual(stream.metrics.buffered_samples, len(added) // 2)
                self.assertFalse(stream.done)
                self.assertTrue(stream.input_finished)
                tail = self.decode(stream)
                self.assert_unit(
                    stream,
                    len(source) // 2,
                    (len(source) + len(added)) // 2,
                    "end_of_input",
                    eof=True,
                )
                self.assertEqual(tail[-1].kind, StreamEventKind.FINAL)
                self.assertEqual(adapter.inputs[-1], added)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assert_released(adapter)

    def test_cancelled_preview_seal_and_new_pcm_do_not_rewrite_retry_identity(self):
        stream, adapter = self.stream(
            config=replace(self.config, coalesce_previews=True)
        )
        source = pcm_ms(250, 47)
        added = pcm_ms(100, 48)
        stream.push(0, source)
        stream.step()
        stream.seal_unit(len(source) // 2)
        stream.push(1, added)
        stream.finish_input()
        self.assertTrue(stream.cancel_active())
        with self.assertRaises(RuntimeStateError):
            stream.step()
        preview = self.decode(stream)
        self.assertIsNone(stream.last_trace.source_unit)
        self.assertFalse(stream.last_trace.eof)
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assertFalse(any(event.kind is StreamEventKind.COMMIT for event in preview))
        unit = self.decode(stream)
        self.assert_unit(stream, 0, len(source) // 2, eof=False)
        self.assertEqual(unit[-1].kind, StreamEventKind.COMMIT)
        self.assertFalse(stream.done)
        self.assertEqual(adapter.inputs[:3], [source] * 3)
        self.assertEqual(adapter.calls[0][0], adapter.calls[1][0])
        self.assertNotEqual(adapter.calls[1][0], adapter.calls[2][0])
        final = self.decode(stream)
        self.assertEqual(final[-1].kind, StreamEventKind.FINAL)
        self.assertEqual(adapter.inputs[-1], added)
        self.assertEqual(stream.metrics.committed_samples, 350 * 16)
        self.assert_released(adapter)

    def test_uncertain_unit_latches_without_repeated_decode_or_pcm_discard(self):
        for origin in ("caller", "end_of_input"):
            with self.subTest(origin=origin):
                adapter = EvidenceNativeAdapter()
                stream, adapter = self.stream(adapter)
                source = pcm_ms(250, 49)
                stream.push(0, source)
                if origin == "caller":
                    stream.seal_unit(len(source) // 2)
                else:
                    stream.finish_input()
                with self.assertRaises(StreamNeedsResolutionError):
                    self.decode(stream)
                self.assert_unit(
                    stream, 0, len(source) // 2, origin, eof=origin == "end_of_input"
                )
                self.assertEqual(stream.last_trace.action, "unresolved")
                before = stream.metrics
                for _ in range(3):
                    with self.assertRaises(StreamNeedsResolutionError):
                        stream.step()
                self.assertEqual(stream.metrics, before)
                self.assertEqual(stream.metrics.buffered_samples, len(source) // 2)
                self.assertEqual(stream.metrics.committed_samples, 0)
                self.assertEqual(stream.state.version, 0)
                self.assertEqual(stream.metrics.events_emitted, 0)
                self.assertEqual(len(adapter.calls), 1)
                self.assertFalse(stream.done)
                self.assert_released(adapter)

    def test_caller_can_partition_oversized_remaining_input_after_eof(self):
        stream, adapter = self.stream()
        source = pcm_ms(600, 51)
        stream.push(0, source)
        stream.seal_unit(50 * 16)
        stream.finish_input()
        first = self.decode(stream)
        self.assert_unit(stream, 0, 50 * 16)
        self.assertEqual(first[-1].kind, StreamEventKind.COMMIT)
        self.assertFalse(stream.done)
        stream.seal_unit(300 * 16)
        second = self.decode(stream)
        self.assert_unit(stream, 50 * 16, 300 * 16)
        self.assertEqual(second[-1].kind, StreamEventKind.COMMIT)
        self.assertFalse(stream.done)
        last = self.decode(stream)
        self.assert_unit(stream, 300 * 16, 600 * 16, "end_of_input", eof=True)
        self.assertEqual(last[-1].kind, StreamEventKind.FINAL)
        self.assertEqual(stream.metrics.committed_samples, 600 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assertEqual(
            adapter.inputs,
            [source[: 50 * 32], source[50 * 32 : 300 * 32], source[300 * 32 :]],
        )
        self.assert_released(adapter)

    def test_caller_cannot_replace_an_active_or_retry_frozen_implicit_eof_unit(self):
        for phase in ("active", "retry"):
            with self.subTest(phase=phase):
                attempts = []

                def build(content):
                    attempts.append(content)
                    if phase == "retry" and len(attempts) == 1:
                        raise RuntimeError("preprocessing failed")
                    return content

                stream, adapter = self.stream(mel_builder=build)
                source = pcm_ms(250, 52)
                stream.push(0, source)
                stream.finish_input()
                if phase == "retry":
                    with self.assertRaisesRegex(RuntimeError, "preprocessing"):
                        stream.step()
                else:
                    stream.step()
                with self.assertRaises((ValueError, NativeStreamError)):
                    stream.seal_unit(len(source) // 2)
                if phase == "retry":
                    events = self.decode(stream)
                else:
                    self.assertEqual(stream.step(), ())
                    events = stream.step()
                self.assert_unit(stream, 0, len(source) // 2, "end_of_input", eof=True)
                self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                self.assertEqual(adapter.inputs, [source])
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assert_released(adapter)

    def test_sealed_fence_and_release_recovery_order_publication_once(self):
        for value in (0, 50):
            for phase in ("fence", "release"):
                with self.subTest(value=value, phase=phase):
                    stream, adapter = self.stream(EvidenceNativeAdapter(you_result))
                    source = pcm_ms(250, value) + pcm(7, value)
                    stream.push(0, source)
                    stream.seal_unit(len(source) // 2)
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
                        self.assertEqual(stream.state.version, int(phase == "release"))
                        self.assertEqual(stream.metrics, before)
                        self.assertEqual(stream.retained_from_sample, 0)
                        with self.assertRaisesRegex(NativeStreamError, "retained"):
                            stream.step()
                    self.assertTrue(adapter.worker.recover(run.transaction))
                    events = stream.step()
                    if phase == "fence":
                        self.assertEqual(events, ())
                        self.assertEqual(stream.metrics, before)
                        events = self.decode(stream)
                    self.assertEqual(
                        [event.kind for event in events],
                        [StreamEventKind.PROVISIONAL, StreamEventKind.COMMIT],
                    )
                    self.assertEqual(stream.metrics.committed_samples, len(source) // 2)
                    self.assertEqual(stream.metrics.buffered_samples, 0)
                    self.assertEqual(stream.state.version, 1)
                    self.assertEqual(stream.metrics.decode_count, 1)
                    self.assertEqual(len(adapter.calls), 2 if phase == "fence" else 1)
                    self.assert_unit(stream, 0, len(source) // 2)
                    self.assertFalse(stream.done)
                    stream.finish_input()
                    self.assertEqual(
                        [event.kind for event in stream.step()], [StreamEventKind.FINAL]
                    )
                    self.assertEqual(stream.step(), ())
                    self.assert_released(adapter)


if __name__ == "__main__":
    unittest.main()
