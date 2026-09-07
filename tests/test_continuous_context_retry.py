"""One strict, anchor-bearing EOF retry uses real commits and native cleanup."""

import hashlib
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import test_continuous_endpoint_words as endpoint_fixtures
from test_continuous_endpoint_words import replace_words, word_result
from test_continuous_evidence import EvidenceNativeAdapter
from test_continuous_stream import pcm_ms

from whisper_runtime import RequestCancelledError, TransactionRetainedError
from whisper_runtime.adapters import NativeStreamError, StreamEventKind
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)


class ContinuousContextRetryTests(unittest.TestCase):
    decode = endpoint_fixtures.ContinuousEndpointWordTests.decode
    drain = endpoint_fixtures.ContinuousEndpointWordTests.drain
    assert_coverage = endpoint_fixtures.ContinuousEndpointWordTests.assert_coverage
    assert_released = endpoint_fixtures.ContinuousEndpointWordTests.assert_released

    def setUp(self):
        self.config = ContinuousStreamConfig(
            preview_interval_ms=500,
            max_window_ms=4000,
            max_buffer_ms=5000,
            holdback_ms=200,
            timestamp_tolerance_ms=0,
            left_context_ms=2000,
            word_alignment=True,
            input_evidence=True,
            eof_context_retry=True,
        )

    def stream(self, **changes):
        adapter = EvidenceNativeAdapter(word_result)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="context-retry",
            config=replace(self.config, **changes),
            mel_builder=lambda content: content,
            rng_seed=17,
        )
        self.addCleanup(stream.close)
        return stream, adapter

    def pending(self, *, enabled=True, candidate=word_result, finish=True):
        stream, adapter = self.stream(eof_context_retry=enabled)
        source = b"".join(pcm_ms(100, 900 + index) for index in range(25))
        stream.push(0, source)
        events = self.drain(stream)
        # These are actual mock-native transactions and emitted commits, not
        # an injected head or a bootstrap assembled from an uncommitted result.
        self.assertEqual(stream.metrics.committed_samples, 1500 * 16)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(stream._word_anchor[0].span.start_ms, 1100)

        def observed(window_id, start, end):
            if start == 600:
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
        self.assertEqual(stream.metrics.committed_samples, 1500 * 16)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(bytes(stream._audio), source)
        self.assertFalse(stream.done)

    def assert_exhausted(self, stream, adapter):
        count = len(adapter.calls)
        for _ in range(3):
            with self.assertRaises(StreamNeedsResolutionError):
                stream.step()
        self.assertEqual(len(adapter.calls), count)

    def test_configuration_is_opt_in_boolean_and_requires_words_and_evidence(self):
        self.assertFalse(ContinuousStreamConfig().eof_context_retry)
        for value in (None, 0, 1, 0.0, "true"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(self.config, eof_context_retry=value)
        for changes in (
            {"word_alignment": False},
            {"input_evidence": False},
            {"resolution_probe": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)
        enabled, _ = self.stream()
        disabled, _ = self.stream(eof_context_retry=False)
        self.assertIn("eof_context_retry", enabled.profile_id)
        self.assertNotIn("eof_context_retry", disabled.profile_id)

    def test_default_refusal_never_schedules_or_decodes_an_alternative(self):
        stream, adapter, source, _ = self.pending(enabled=False)
        state, anchor, count = stream.state, stream._word_anchor, len(adapter.calls)
        with self.assertRaisesRegex(StreamNeedsResolutionError, "anchor_missing"):
            self.decode(stream)
        self.assertIsNone(stream.context_retry_observation)
        self.assertEqual(len(adapter.calls), count + 1)
        self.assert_exhausted(stream, adapter)
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)

    def test_guard_clipped_to_original_window_is_not_redecoded(self):
        stream, adapter = self.stream()
        source = pcm_ms(1000, 900)
        stream.push(0, source)
        self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 500 * 16)
        self.assertEqual(stream._word_anchor[0].span.start_ms, 100)
        self.assertEqual(stream.retained_from_sample, 0)
        adapter.result_factory = lambda window_id, start, end: replace_words(
            word_result(window_id, start, end),
            lambda item: replace(item, text=" absent", tokens=(999,)),
        )
        stream.finish_input()
        state, anchor, count = stream.state, stream._word_anchor, len(adapter.calls)
        with self.assertRaises(StreamNeedsResolutionError):
            self.decode(stream)
        self.assertEqual(stream.context_retry_observation.status, "unavailable")
        self.assertEqual(stream.context_retry_observation.analysis_start_sample, 0)
        self.assertEqual(len(adapter.calls), count + 1)
        self.assert_exhausted(stream, adapter)
        self.assertEqual(stream.state, state)
        self.assertEqual(stream._word_anchor, anchor)
        self.assertEqual(bytes(stream._audio), source)
        self.assertFalse(stream.done)
        self.assert_released(adapter)

    def test_success_preserves_commits_and_uses_exact_guarded_pcm_once(self):
        stream, adapter, source, events = self.pending()
        state, anchor, before = stream.state, stream._word_anchor, stream.metrics
        self.assertEqual(self.decode(stream), ())
        scheduled = stream.context_retry_observation
        self.assertEqual(scheduled.status, "scheduled")
        self.assertIs(scheduled.source, stream.last_trace)
        self.assertEqual(scheduled.source.reason, "anchor_missing")
        self.assertEqual(scheduled.source.anchor_diagnostic.status, "lexical_missing")
        self.assertEqual(scheduled.anchor, anchor)
        self.assertEqual(scheduled.session_version, state.version)
        self.assertEqual(scheduled.analysis_start_sample, 600 * 16)
        self.assertEqual(scheduled.analysis_end_sample, 2500 * 16)
        self.assertEqual(
            scheduled.pcm_sha256, hashlib.sha256(source[600 * 32 :]).hexdigest()
        )
        self.assertIsNone(scheduled.committed_version)
        with self.assertRaises(FrozenInstanceError):
            scheduled.reason = "overwrite history"
        self.assert_frozen(stream, state, anchor, source)
        events.extend(self.decode(stream))
        recovered = stream.context_retry_observation
        self.assertEqual(recovered.status, "recovered")
        self.assertEqual(recovered.committed_version, state.version + 1)
        self.assertEqual(recovered.committed_version, stream.state.version)
        self.assertIs(recovered.source, scheduled.source)
        self.assertEqual(recovered.anchor, anchor)
        self.assertEqual(recovered.candidate.native.window_id, adapter.calls[-1][0])
        self.assertEqual(adapter.calls[-1][1:3], (600, 2500))
        self.assertEqual(adapter.inputs[-1], source[600 * 32 :])
        self.assertIs(adapter.options[-1], adapter.options[-2])
        self.assertEqual(stream.metrics.decode_count, before.decode_count + 2)
        self.assertEqual(
            stream.metrics.decoded_source_samples - before.decoded_source_samples,
            (2500 + 1900) * 16,
        )
        self.assertEqual(
            self.assert_coverage(events, 2500 * 16),
            [f"w{index}" for index in range(25)],
        )
        self.assertEqual(stream.state.windows[:-1], state.windows)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        count = len(adapter.calls)
        for _ in range(3):
            self.assertEqual(stream.step(), ())
        self.assertEqual(len(adapter.calls), count)
        self.assert_released(adapter)

    def test_changed_anchor_or_tokens_remain_unresolved_without_another_attempt(self):
        for field in ("text", "tokens"):
            with self.subTest(field=field):

                def candidate(window_id, start, end):
                    return replace_words(
                        word_result(window_id, start, end),
                        lambda item: (
                            replace(
                                item,
                                **{field: " changed" if field == "text" else (999,)},
                            )
                            if item.span.start_ms == 1100
                            else item
                        ),
                    )

                stream, adapter, source, _ = self.pending(candidate=candidate)
                state, anchor, count = (
                    stream.state,
                    stream._word_anchor,
                    len(adapter.calls),
                )
                self.decode(stream)
                with self.assertRaises(StreamNeedsResolutionError):
                    self.decode(stream)
                self.assertNotEqual(
                    stream.context_retry_observation.status, "recovered"
                )
                self.assertIsNone(stream.context_retry_observation.committed_version)
                self.assertEqual(len(adapter.calls), count + 2)
                self.assert_exhausted(stream, adapter)
                self.assert_frozen(stream, state, anchor, source)
                self.assert_released(adapter)

    def test_matching_anchor_with_punctuation_only_suffix_cannot_complete(self):
        def candidate(window_id, start, end):
            return replace_words(
                word_result(window_id, start, end),
                lambda item: (
                    replace(item, text=".", tokens=(13,))
                    if item.span.start_ms >= 1500
                    else item
                ),
            )

        stream, adapter, source, _ = self.pending(candidate=candidate)
        state, anchor = stream.state, stream._word_anchor
        self.decode(stream)
        with self.assertRaises(StreamNeedsResolutionError):
            self.decode(stream)
        self.assertEqual(stream.last_trace.reason, "no_lexical_text")
        self.assertEqual(stream.last_trace.anchor_diagnostic.status, "matched")
        self.assertIsNone(stream.last_trace.word_publication)
        self.assert_exhausted(stream, adapter)
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)

    def test_retry_does_not_relax_frozen_word_timestamp_tolerance(self):
        def candidate(window_id, start, end):
            return replace_words(
                word_result(window_id, start, end),
                lambda item: (
                    replace(item, span=replace(item.span, start_ms=1101))
                    if item.span.start_ms == 1100
                    else item
                ),
            )

        stream, adapter, source, _ = self.pending(candidate=candidate)
        state, anchor = stream.state, stream._word_anchor
        self.decode(stream)
        with self.assertRaises(StreamNeedsResolutionError):
            self.decode(stream)
        self.assertEqual(stream.last_trace.anchor_diagnostic.status, "timing_mismatch")
        self.assertIsNone(stream.last_trace.word_publication)
        self.assert_exhausted(stream, adapter)
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)

    def test_preprocessing_or_start_failure_consumes_the_only_attempt(self):
        for stage in ("mel", "start"):
            with self.subTest(stage=stage):
                stream, adapter, source, _ = self.pending()
                state, anchor = stream.state, stream._word_anchor
                self.decode(stream)
                if stage == "start":
                    adapter.fail_once = "start"
                    with self.assertRaises(RuntimeError):
                        stream.step()
                else:
                    with patch.object(
                        stream, "_mel_builder", side_effect=RuntimeError("mel")
                    ):
                        with self.assertRaises(RuntimeError):
                            stream.step()
                self.assertEqual(stream.context_retry_observation.status, "failed")
                self.assert_exhausted(stream, adapter)
                self.assert_frozen(stream, state, anchor, source)
                self.assert_released(adapter)

    def test_cancelled_or_stopped_retry_cannot_restart_or_finalize(self):
        for action in ("cancel", "stop"):
            with self.subTest(action=action):
                stream, adapter, source, _ = self.pending()
                state, anchor = stream.state, stream._word_anchor
                self.decode(stream)
                stream.step()
                if action == "cancel":
                    self.assertTrue(stream.cancel_active())
                    with self.assertRaises(RequestCancelledError):
                        stream.step()
                else:
                    self.assertTrue(stream.stop_active())
                    self.assertEqual(stream.step(), ())
                self.assertEqual(stream.context_retry_observation.status, "failed")
                self.assert_exhausted(stream, adapter)
                self.assert_frozen(stream, state, anchor, source)
                self.assert_released(adapter)

    def test_successful_ordinary_eof_does_not_schedule_a_retry(self):
        stream, adapter, _, events = self.pending()
        adapter.result_factory = word_result
        events.extend(self.drain(stream))
        self.assertTrue(stream.done)
        self.assertIsNone(stream.context_retry_observation)
        self.assertEqual(
            sum(event.kind == StreamEventKind.FINAL for event in events), 1
        )
        self.assert_released(adapter)

    def test_nonfinal_missing_anchor_only_publishes_a_preview(self):
        stream, adapter, source, _ = self.pending(finish=False)
        state, anchor = stream.state, stream._word_anchor
        extra = pcm_ms(500, 1000)
        stream.push(1, extra)
        self.assertEqual(len(self.decode(stream)), 1)
        self.assertIsNone(stream.context_retry_observation)
        self.assert_frozen(stream, state, anchor, source + extra)
        self.assert_released(adapter)

    def test_original_fence_must_release_before_retry_admission(self):
        stream, adapter, source, events = self.pending()
        state, anchor = stream.state, stream._word_anchor
        stream.step()
        stream.step()
        original = adapter.runs[-1]
        original.fence.fail = True
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        self.assertEqual(stream.context_retry_observation.status, "scheduled")
        count = len(adapter.calls)
        with self.assertRaisesRegex(NativeStreamError, "retained"):
            stream.step()
        self.assertEqual(len(adapter.calls), count)
        self.assert_frozen(stream, state, anchor, source)
        original.fence.fail = False
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        self.assertEqual(stream.step(), ())
        events.extend(self.decode(stream))
        self.assertEqual(stream.context_retry_observation.status, "recovered")
        self.assertEqual(len(adapter.calls), count + 1)
        self.assert_coverage(events, 2500 * 16)
        self.assert_released(adapter)

    def test_wrong_native_window_cannot_recover_the_frozen_prefix(self):
        stream, adapter, source, _ = self.pending()
        state, anchor = stream.state, stream._word_anchor
        self.decode(stream)
        stream.step()
        stream.step()
        adapter.runs[-1].result = replace(adapter.runs[-1].result, window_id="foreign")
        with self.assertRaisesRegex(NativeStreamError, "admitted audio"):
            stream.step()
        self.assertEqual(stream.context_retry_observation.status, "failed")
        self.assert_exhausted(stream, adapter)
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)

    def test_cancelled_retry_with_retained_capacity_cannot_complete_after_recovery(
        self,
    ):
        stream, adapter, source, _ = self.pending()
        state, anchor = stream.state, stream._word_anchor
        self.decode(stream)
        stream.step()
        run = adapter.runs[-1]
        run.fence.fail = True
        self.assertTrue(stream.cancel_active())
        with self.assertRaises(TransactionRetainedError) as raised:
            stream.step()
        self.assertIsNone(raised.exception.committed_state)
        self.assertEqual(stream.context_retry_observation.status, "failed")
        with self.assertRaisesRegex(NativeStreamError, "retained"):
            stream.step()
        self.assert_frozen(stream, state, anchor, source)
        run.fence.fail = False
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        self.assertEqual(stream.step(), ())
        self.assert_exhausted(stream, adapter)
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)

    def test_committed_retry_waits_for_release_then_emits_commit_and_final_exactly_once(
        self,
    ):
        stream, adapter, source, events = self.pending()
        before, count = stream.metrics, len(adapter.calls)
        self.decode(stream)
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        with patch.object(
            type(run.transaction._lease),
            "release",
            side_effect=RuntimeError("release failed"),
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                stream.step()
            self.assertIsNotNone(raised.exception.committed_state)
            self.assertEqual(stream.context_retry_observation.status, "running")
            self.assertIsNone(stream.context_retry_observation.committed_version)
            self.assertEqual(stream.metrics.events_emitted, before.events_emitted)
            self.assertEqual(stream.metrics.committed_samples, before.committed_samples)
            self.assertEqual(bytes(stream._audio), source)
            with self.assertRaisesRegex(NativeStreamError, "retained"):
                stream.step()
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        events.extend(stream.step())
        self.assertEqual(stream.context_retry_observation.status, "recovered")
        self.assertEqual(
            stream.context_retry_observation.committed_version, stream.state.version
        )
        self.assertEqual(len(adapter.calls), count + 2)
        self.assertEqual(stream.metrics.decode_count, before.decode_count + 2)
        self.assert_coverage(events, 2500 * 16)
        self.assertEqual(stream.step(), ())
        self.assert_released(adapter)

    def test_modified_pcm_cannot_be_restored_to_revive_a_failed_active_retry(self):
        for stage in ("before_admission", "before_resolution"):
            with self.subTest(stage=stage):
                stream, adapter, source, _ = self.pending()
                state, anchor = stream.state, stream._word_anchor
                self.decode(stream)
                if stage == "before_resolution":
                    stream.step()
                    stream.step()
                changed = 1000 * 32
                stream._audio[changed] ^= 1
                try:
                    with self.assertRaisesRegex(
                        NativeStreamError, "frozen context retry"
                    ):
                        stream.step()
                finally:
                    stream._audio[changed] ^= 1
                self.assertEqual(stream.context_retry_observation.status, "failed")
                # Restoring bytes after the failure must not let a still-open
                # native transaction proceed to a now-valid result and commit.
                self.assert_exhausted(stream, adapter)
                self.assert_frozen(stream, state, anchor, source)
                self.assert_released(adapter)

    def test_changed_anchor_is_rejected_before_preprocessing(self):
        stream, adapter, source, _ = self.pending()
        state, anchor = stream.state, stream._word_anchor
        self.decode(stream)
        count = len(adapter.calls)
        stream._word_anchor = ()
        try:
            with patch.object(
                stream, "_mel_builder", side_effect=AssertionError("not admitted")
            ):
                with self.assertRaises(NativeStreamError):
                    stream.step()
        finally:
            stream._word_anchor = anchor
        self.assertEqual(stream.context_retry_observation.status, "failed")
        self.assertEqual(len(adapter.calls), count)
        self.assert_exhausted(stream, adapter)
        self.assert_frozen(stream, state, anchor, source)
        self.assert_released(adapter)


if __name__ == "__main__":
    unittest.main()
