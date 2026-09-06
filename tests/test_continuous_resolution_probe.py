"""One connected EOF observation is not publication or acoustic coverage proof."""

import hashlib
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import test_continuous_word_context as context_fixtures
from test_continuous_endpoint_words import replace_words, word_result
from test_continuous_stream import pcm_ms

from whisper_runtime import RequestCancelledError, TransactionRetainedError
from whisper_runtime.adapters import NativeStreamError, StreamEventKind
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    StreamNeedsResolutionError,
)


class ContinuousResolutionProbeTests(unittest.TestCase):
    stream = context_fixtures.ContinuousWordContextTests.stream
    decode = context_fixtures.ContinuousWordContextTests.decode
    drain = context_fixtures.ContinuousWordContextTests.drain
    assert_released = context_fixtures.ContinuousWordContextTests.assert_released

    def setUp(self):
        context_fixtures.ContinuousWordContextTests.setUp(self)
        self.config = replace(self.config, resolution_probe=True)

    def pending(self, *, enabled=True, candidate=word_result, finish=True):
        stream, adapter = self.stream(resolution_probe=enabled)
        source = pcm_ms(100, 900) + pcm_ms(100, 901) + pcm_ms(100, 902)
        stream.push(0, source)
        events = self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 200 * 16)
        self.assertEqual(stream.retained_from_sample, 0)

        def observed(window_id, start, end):
            if start == 200:
                return candidate(window_id, start, end)
            return replace_words(
                word_result(window_id, start, end),
                lambda item: replace(item, text=" absent", tokens=(999,)),
            )

        adapter.result_factory = observed
        if finish:
            stream.finish_input()
        return stream, adapter, source, events

    def assert_frozen(self, stream, state, anchor, source):
        self.assertEqual(stream.state, state)
        self.assertEqual(stream._word_anchor, anchor)
        self.assertEqual(stream.metrics.committed_samples, 200 * 16)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(bytes(stream._audio), source)
        self.assertFalse(stream.done)

    def test_configuration_is_boolean_and_requires_word_policy(self):
        self.assertFalse(ContinuousStreamConfig().resolution_probe)
        for value in (None, 0, 1, 0.0, "true"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(self.config, resolution_probe=value)
        with self.assertRaisesRegex(ValueError, "word-aligned"):
            ContinuousStreamConfig(resolution_probe=True)
        self.assertTrue(
            ContinuousStreamConfig(
                word_alignment=True, resolution_probe=True
            ).resolution_probe
        )
        stream, _ = self.stream()
        self.assertTrue(
            stream.profile_id.endswith("+resolution_probe/v1+input_evidence/v1")
        )
        disabled, _ = self.stream(resolution_probe=False)
        self.assertNotIn("resolution_probe", disabled.profile_id)

    def test_default_refusal_never_decodes_a_probe(self):
        stream, adapter, source, _ = self.pending(enabled=False)
        before = len(adapter.calls)
        with self.assertRaisesRegex(StreamNeedsResolutionError, "anchor_missing"):
            self.decode(stream)
        self.assertIsNone(stream.resolution_observation)
        self.assertEqual(stream.last_trace.reason, "anchor_missing")
        for _ in range(3):
            with self.assertRaises(StreamNeedsResolutionError):
                stream.step()
        self.assertEqual(len(adapter.calls), before + 1)
        self.assertEqual(bytes(stream._audio), source)
        self.assert_released(adapter)

    def test_distinct_probe_retains_original_refusal_and_never_publishes(self):
        stream, adapter, source, events = self.pending()
        before = stream.metrics
        state, anchor = stream.state, stream._word_anchor
        self.assertEqual(self.decode(stream), ())
        observation = stream.resolution_observation
        self.assertEqual(observation.status, "scheduled")
        self.assertIs(observation.source, stream.last_trace)
        self.assertEqual(observation.source.reason, "anchor_missing")
        self.assertEqual(observation.source.anchor_diagnostic.status, "lexical_missing")
        self.assertEqual(observation.session_version, state.version)
        self.assertEqual(observation.anchor, anchor)
        self.assertEqual(observation.analysis_start_sample, 200 * 16)
        self.assertEqual(observation.analysis_end_sample, 300 * 16)
        self.assertEqual(
            observation.pcm_sha256, hashlib.sha256(source[200 * 32 :]).hexdigest()
        )
        with self.assertRaises(FrozenInstanceError):
            observation.reason = "publish"
        with patch.object(
            type(adapter.runs[-1]),
            "finish",
            side_effect=AssertionError("never publish"),
        ):
            with self.assertRaisesRegex(
                StreamNeedsResolutionError, "anchor_missing.*observed.*no_publication"
            ):
                self.decode(stream)
        self.assertEqual(adapter.calls[-1][1:3], (200, 300))
        self.assertEqual(adapter.inputs[-1], source[200 * 32 :])
        self.assertIs(adapter.options[-1], adapter.options[-2])
        observed = stream.resolution_observation
        self.assertIs(observed.source, observation.source)
        self.assertEqual(observed.candidate.native.text, "w2")
        self.assertEqual(observed.status, "observed")
        self.assertEqual(stream.last_trace.action, "resolution_observation")
        self.assertIsNone(stream.last_trace.word_publication)
        self.assertIsNone(stream.last_trace.publication_span)
        self.assertEqual(stream.metrics.decode_count, before.decode_count + 2)
        self.assertEqual(
            stream.metrics.decoded_source_samples,
            before.decoded_source_samples + 400 * 16,
        )
        self.assertEqual(stream.metrics.events_emitted, before.events_emitted)
        self.assertFalse(any(event.kind is StreamEventKind.FINAL for event in events))
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)
        count = len(adapter.calls)
        for _ in range(4):
            with self.assertRaises(StreamNeedsResolutionError):
                stream.step()
        self.assertEqual(len(adapter.calls), count)

    def test_plausible_repeated_and_nonlexical_candidates_have_no_authority(self):
        for text in (" w2", " w0 w1 w0 w1", "."):
            with self.subTest(text=text):

                def candidate(window_id, start, end):
                    return replace_words(
                        word_result(window_id, start, end),
                        lambda item: replace(item, text=text),
                    )

                stream, adapter, source, _ = self.pending(candidate=candidate)
                state, anchor = stream.state, stream._word_anchor
                self.decode(stream)
                with self.assertRaises(StreamNeedsResolutionError):
                    self.decode(stream)
                # A matching-looking suffix cannot prove that the acoustic cut
                # did not exclude an unrecognized spoken word before its onset.
                self.assertEqual(
                    stream.resolution_observation.candidate.native.text, text.strip()
                )
                self.assert_frozen(stream, state, anchor, source)
                self.assert_released(adapter)

    def test_same_window_is_not_redecoded(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(100, 900))
        stream.finish_input()
        # Force a refusal without a committed prefix: head..end is identical.
        with patch(
            "whisper_runtime.adapters.continuous_stream.compare_word_hypotheses"
        ) as compare:
            from whisper_runtime.adapters.word_policy import WordAgreementDecision

            compare.return_value = WordAgreementDecision("incomplete")
            with self.assertRaises(StreamNeedsResolutionError):
                self.decode(stream)
        self.assertEqual(stream.resolution_observation.status, "unavailable")
        self.assertEqual(stream.resolution_observation.reason, "no_distinct_suffix")
        self.assertIsNone(stream._resolution_pcm)
        for _ in range(3):
            with self.assertRaises(StreamNeedsResolutionError):
                stream.step()
        self.assertEqual(len(adapter.calls), 1)
        self.assert_released(adapter)

    def test_successful_eof_remains_an_ordinary_commit(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(300, 900))
        self.drain(stream)
        stream.finish_input()
        events = self.drain(stream)
        self.assertTrue(stream.done)
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
        self.assertIsNone(stream.resolution_observation)
        self.assert_released(adapter)

    def test_non_eof_refusal_does_not_schedule_a_probe(self):
        stream, adapter, source, _ = self.pending(finish=False)
        extra = pcm_ms(100, 903)
        stream.push(1, extra)
        self.assertEqual(len(self.decode(stream)), 1)
        self.assertIsNone(stream.resolution_observation)
        self.assertEqual(bytes(stream._audio), source + extra)
        self.assert_released(adapter)

    def test_closed_source_unit_is_not_an_eof_probe(self):
        stream, adapter, source, _ = self.pending(finish=False)
        extra = pcm_ms(100, 903) + pcm_ms(40, 1)
        stream.push(1, extra)
        with self.assertRaisesRegex(StreamNeedsResolutionError, "source unit"):
            self.drain(stream)
        self.assertIsNotNone(stream.last_trace.source_unit)
        self.assertFalse(stream.last_trace.eof)
        self.assertIsNone(stream.resolution_observation)
        self.assertEqual(bytes(stream._audio), source + extra)
        self.assert_released(adapter)

    def test_preprocessing_or_start_failure_consumes_the_only_attempt(self):
        for stage in ("mel", "start"):
            with self.subTest(stage=stage):
                stream, adapter, source, _ = self.pending()
                self.decode(stream)
                state, anchor = stream.state, stream._word_anchor
                if stage == "mel":
                    stream._mel_builder = lambda _: (_ for _ in ()).throw(
                        RuntimeError("mel")
                    )
                else:
                    adapter.fail_once = "start"
                with self.assertRaises(RuntimeError):
                    stream.step()
                self.assertEqual(stream.resolution_observation.status, "failed")
                self.assertIsNone(stream._resolution_pcm)
                count = len(adapter.calls)
                for _ in range(3):
                    with self.assertRaises(StreamNeedsResolutionError):
                        stream.step()
                self.assertEqual(len(adapter.calls), count)
                self.assert_frozen(stream, state, anchor, source)
                self.assert_released(adapter)

    def test_cancelled_and_stopped_probe_cannot_retry_or_finalize(self):
        for action in ("cancel", "stop"):
            with self.subTest(action=action):
                stream, adapter, source, _ = self.pending()
                self.decode(stream)
                stream.step()
                state, anchor = stream.state, stream._word_anchor
                if action == "cancel":
                    self.assertTrue(stream.cancel_active())
                    with self.assertRaises(RequestCancelledError):
                        stream.step()
                else:
                    self.assertTrue(stream.stop_active())
                    self.assertEqual(stream.step(), ())
                self.assertEqual(stream.resolution_observation.status, "failed")
                count = len(adapter.calls)
                with self.assertRaises(StreamNeedsResolutionError):
                    stream.step()
                self.assertEqual(len(adapter.calls), count)
                self.assert_frozen(stream, state, anchor, source)
                self.assert_released(adapter)

    def test_original_fence_must_recover_before_candidate_admission(self):
        stream, adapter, source, _ = self.pending()
        stream.step()
        stream.step()
        original = adapter.runs[-1]
        original.fence.fail = True
        state, anchor = stream.state, stream._word_anchor
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        count = len(adapter.calls)
        self.assertEqual(stream.resolution_observation.status, "scheduled")
        with self.assertRaisesRegex(NativeStreamError, "retained"):
            stream.step()
        self.assertEqual(len(adapter.calls), count)
        self.assert_frozen(stream, state, anchor, source)
        original.fence.fail = False
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        self.assertEqual(stream.step(), ())
        with self.assertRaises(StreamNeedsResolutionError):
            self.decode(stream)
        self.assertEqual(len(adapter.calls), count + 1)
        self.assertEqual(stream.resolution_observation.status, "observed")
        self.assert_released(adapter)

    def test_candidate_fence_failure_retains_capacity_without_publication(self):
        stream, adapter, source, _ = self.pending()
        self.decode(stream)
        stream.step()
        stream.step()
        candidate = adapter.runs[-1]
        candidate.fence.fail = True
        state, anchor = stream.state, stream._word_anchor
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        self.assertEqual(stream.resolution_observation.status, "failed")
        self.assertIsNotNone(stream.resolution_observation.candidate)
        self.assertIsNone(raised.exception.committed_state)
        with self.assertRaisesRegex(NativeStreamError, "retained"):
            stream.step()
        self.assert_frozen(stream, state, anchor, source)
        candidate.fence.fail = False
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        self.assertEqual(stream.step(), ())
        count = len(adapter.calls)
        with self.assertRaises(StreamNeedsResolutionError):
            stream.step()
        self.assertEqual(len(adapter.calls), count)
        self.assert_released(adapter)

    def test_retained_startup_failure_requires_exact_recovery_without_retry(self):
        stream, adapter, source, _ = self.pending()
        self.decode(stream)
        state, anchor = stream.state, stream._word_anchor
        original_start = adapter.start_window

        def fail_start(**kwargs):
            candidate = original_start(**kwargs)
            candidate.fence.fail = True
            candidate.close()
            self.fail("the startup transaction should be retained")

        with patch.object(adapter, "start_window", side_effect=fail_start):
            with self.assertRaises(TransactionRetainedError) as raised:
                stream.step()
        retained = raised.exception
        count = len(adapter.calls)
        self.assertEqual(stream.resolution_observation.status, "failed")
        with self.assertRaises(TransactionRetainedError) as repeated:
            stream.step()
        self.assertIs(repeated.exception, retained)
        with self.assertRaisesRegex(NativeStreamError, "retained"):
            stream.close()
        self.assertEqual(len(adapter.calls), count)
        self.assert_frozen(stream, state, anchor, source)
        adapter.runs[-1].fence.fail = False
        self.assertTrue(adapter.worker.recover(retained.transaction))
        with self.assertRaises(StreamNeedsResolutionError):
            stream.step()
        self.assertEqual(len(adapter.calls), count)
        self.assert_released(adapter)

    def test_candidate_result_must_bind_the_exact_admitted_window(self):
        stream, adapter, source, _ = self.pending()
        self.decode(stream)
        stream.step()
        stream.step()
        state, anchor = stream.state, stream._word_anchor
        candidate = adapter.runs[-1]
        candidate.result = replace(candidate.result, window_id="another-window")
        with self.assertRaisesRegex(NativeStreamError, "admitted audio"):
            stream.step()
        self.assertEqual(stream.resolution_observation.status, "failed")
        self.assertIsNone(stream.resolution_observation.candidate)
        self.assertEqual(stream.last_trace.reason, "anchor_missing")
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)

    def test_frozen_prefix_is_checked_before_probe_preprocessing(self):
        stream, adapter, source, _ = self.pending()
        self.decode(stream)
        state, anchor = stream.state, stream._word_anchor
        count = len(adapter.calls)
        stream._word_anchor = ()  # Simulate invalid state replacement by a caller.
        with patch.object(
            stream, "_mel_builder", side_effect=AssertionError("not admitted")
        ):
            with self.assertRaisesRegex(NativeStreamError, "frozen resolution"):
                stream.step()
        stream._word_anchor = anchor
        self.assertEqual(stream.resolution_observation.status, "failed")
        self.assertEqual(len(adapter.calls), count)
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)


if __name__ == "__main__":
    unittest.main()
