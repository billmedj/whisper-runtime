"""Input-evidence gating across scheduling and cleanup; no model or devices."""

import hashlib
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_continuous_stream import (
    ScriptedNativeAdapter,
    ScriptedNativeRun,
    pcm,
    pcm_ms,
    timed_result,
)

from whisper_runtime import TransactionRetainedError
from whisper_runtime.adapters import NativeStreamError, StreamEventKind, TranscriptEvent
from whisper_runtime.adapters.audio_evidence import SilencePublication, assess_audio
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)
from whisper_runtime.adapters.native_result import NativeWindowResult
from whisper_runtime.adapters.word_policy import AlignedPublication, NativeWordAlignment


def speech_result(window_id: str, start: int, end: int) -> NativeWindowResult:
    result = timed_result(window_id, start, end)
    return replace(
        result,
        metadata=replace(result.metadata, avg_logprob=-0.001, no_speech_prob=0.01),
    )


def you_result(window_id: str, start: int, end: int) -> NativeWindowResult:
    result = speech_result(window_id, start, end)
    segments = tuple(
        replace(segment, text=" you") for segment in result.metadata.segments
    )
    return replace(
        result,
        text="".join(segment.text for segment in segments).strip() or "you",
        metadata=replace(result.metadata, segments=segments),
    )


class EvidenceNativeRun(ScriptedNativeRun):
    """Reuse the fixture's real transaction and fence for an empty projection."""

    def prepare_word_alignment(self):
        return NativeWordAlignment(self.result, self.result.metadata.segments)

    def finish(self, *, silence_publication=None, aligned_publication=None, **kwargs):
        if silence_publication is None and aligned_publication is None:
            return super().finish(**kwargs)
        if silence_publication is not None and aligned_publication is not None:
            raise ValueError("cannot combine publication selectors")
        publication = (
            silence_publication
            if silence_publication is not None
            else aligned_publication
        )
        if isinstance(publication, SilencePublication):
            prepared = publication.native
        elif isinstance(publication, AlignedPublication):
            prepared = publication.alignment.native
        else:
            raise TypeError("expected a typed publication")
        if prepared is not self.result:
            raise ValueError("publication must bind the prepared native result")
        native = self.result
        self.result = publication
        try:
            return super().finish(**kwargs)
        finally:
            self.result = native


class EvidenceNativeAdapter(ScriptedNativeAdapter):
    def start_window(self, **kwargs):
        # Keep admission, ownership, commit and cleanup in the original fixtures.
        with patch("test_continuous_stream.ScriptedNativeRun", EvidenceNativeRun):
            return super().start_window(**kwargs)


class ContinuousEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=500,
            max_buffer_ms=600,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            input_evidence=True,
        )

    def stream(self, adapter: ScriptedNativeAdapter, **kwargs):
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="input-evidence-test",
            mel_builder=kwargs.pop("mel_builder", lambda content: content),
            config=kwargs.pop("config", self.config),
            **kwargs,
        )
        self.addCleanup(stream.close)
        return stream

    def decode(self, stream: ContinuousTranscriptStream) -> tuple[TranscriptEvent, ...]:
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.step(), ())
        return stream.step()

    def drain(self, stream: ContinuousTranscriptStream) -> list[TranscriptEvent]:
        events = []
        for _ in range(100):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("stream did not reach a scheduling boundary")

    def assert_retained(
        self, stream: ContinuousTranscriptStream, source: bytes
    ) -> None:
        self.assertEqual(stream.accepted_samples, len(source) // 2)
        self.assertEqual(stream.metrics.buffered_samples, len(source) // 2)
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(stream.state.version, 0)
        self.assertFalse(stream.done)

    def assert_observation(self, stream: ContinuousTranscriptStream, source: bytes):
        evidence = stream.last_trace.audio_evidence
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.observation.sample_count, len(source) // 2)
        self.assertEqual(
            evidence.observation.pcm_sha256, hashlib.sha256(source).hexdigest()
        )
        self.assertEqual(evidence.observation.digital_silence, not any(source))
        return evidence

    def assert_released(self, adapter: ScriptedNativeAdapter) -> None:
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_flag_is_opt_in_and_strictly_boolean(self) -> None:
        self.assertIs(ContinuousStreamConfig().input_evidence, False)
        for value in (0, 1, None, "true", 1.0):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(TypeError, "boolean"),
            ):
                replace(self.config, input_evidence=value)

    def test_profiles_distinguish_guard_without_changing_base_mode(self) -> None:
        for word in (False, True):
            for coalesce in (False, True):
                for context in (0, 100):
                    with self.subTest(word=word, coalesce=coalesce, context=context):
                        config = replace(
                            self.config,
                            word_alignment=word,
                            coalesce_previews=coalesce,
                            left_context_ms=context,
                        )
                        guarded = self.stream(ScriptedNativeAdapter(), config=config)
                        legacy = self.stream(
                            ScriptedNativeAdapter(),
                            config=replace(config, input_evidence=False),
                        )
                        self.assertEqual(
                            guarded.profile_id, legacy.profile_id + "+input_evidence/v1"
                        )

    def test_confident_you_on_zeros_is_suppressed_before_alignment_in_all_modes(self):
        source = pcm_ms(250) + pcm(7)
        for word in (False, True):
            for coalesce in (False, True):
                for context in (0, 100):
                    for eof in (False, True):
                        with self.subTest(
                            word=word, coalesce=coalesce, context=context, eof=eof
                        ):
                            adapter = EvidenceNativeAdapter(you_result)
                            stream = self.stream(
                                adapter,
                                config=replace(
                                    self.config,
                                    word_alignment=word,
                                    coalesce_previews=coalesce,
                                    left_context_ms=context,
                                ),
                            )
                            stream.push(0, source)
                            if eof:
                                stream.finish_input()
                            with patch.object(
                                EvidenceNativeRun,
                                "prepare_word_alignment",
                                create=True,
                                side_effect=AssertionError("alignment must not run"),
                            ) as alignment:
                                events = self.drain(stream)
                                alignment.assert_not_called()
                            self.assertTrue(events)
                            self.assertTrue(all(not event.text for event in events))
                            commits = [
                                event
                                for event in events
                                if event.kind is StreamEventKind.COMMIT
                            ]
                            self.assertTrue(commits)
                            expected_head = (
                                len(source) // 2 if eof or coalesce else 200 * 16
                            )
                            self.assertEqual(
                                commits[-1].committed_through_sample, expected_head
                            )
                            self.assertEqual(
                                stream.metrics.committed_samples, expected_head
                            )
                            retained = expected_head
                            self.assertEqual(stream.retained_from_sample, retained)
                            self.assertEqual(
                                stream.metrics.buffered_samples,
                                len(source) // 2 - retained,
                            )
                            self.assertEqual(stream.last_trace.action, "commit")
                            self.assertEqual(stream.done, eof)
                            if eof:
                                self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                                self.assertEqual(stream.step(), ())
                            evidence = self.assert_observation(
                                stream, adapter.inputs[-1]
                            )
                            self.assertEqual(evidence.state, "non_speech")
                            self.assertEqual(evidence.reason, "digital_silence")
                            self.assertIn("you", stream.last_trace.result.text)
                            publication = stream.state.windows[-1].result
                            self.assertIsInstance(publication, SilencePublication)
                            self.assertEqual(publication.text, "")
                            self.assertIs(publication.native, stream.last_trace.result)
                            self.assertEqual(
                                publication.observation, evidence.observation
                            )
                            self.assertEqual(
                                stream.last_trace.silence_publication, publication
                            )
                            self.assert_released(adapter)

    def test_near_zero_nonzero_pcm_is_not_silence_and_eof_keeps_exact_samples(self):
        for value in (-1, 1):
            with self.subTest(value=value):
                adapter = ScriptedNativeAdapter(speech_result)
                stream = self.stream(adapter)
                source = pcm_ms(100, value) + pcm(7, value)
                stream.push(0, source)
                stream.finish_input()
                events = self.drain(stream)
                evidence = self.assert_observation(stream, source)
                self.assertEqual(evidence.state, "speech_candidate")
                self.assertFalse(evidence.observation.digital_silence)
                self.assertEqual(adapter.inputs, [source])
                self.assertEqual(events[-2].kind, StreamEventKind.COMMIT)
                self.assertEqual(events[-2].committed_through_sample, len(source) // 2)
                self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assertTrue(stream.done)
                self.assert_released(adapter)

    def test_uncertain_output_never_grants_discard_authority_even_at_eof(self):
        for missing in ("metadata", "score", "lexical", "conflicting_score"):
            with self.subTest(missing=missing):

                def result_factory(window_id, start, end):
                    result = speech_result(window_id, start, end)
                    if missing == "metadata":
                        return replace(result, metadata=None)
                    if missing == "lexical":
                        return replace(
                            result,
                            text=" ",
                            metadata=replace(result.metadata, segments=()),
                        )
                    return replace(
                        result,
                        metadata=replace(
                            result.metadata,
                            no_speech_prob=None if missing == "score" else 0.6,
                        ),
                    )

                adapter = ScriptedNativeAdapter(result_factory)
                stream = self.stream(adapter)
                source = pcm_ms(250, 37)
                stream.push(0, source)
                events = self.drain(stream)
                self.assertTrue(events)
                self.assertTrue(all(event.text == "" for event in events))
                self.assert_retained(stream, source)
                self.assertEqual(stream.last_trace.audio_evidence.state, "uncertain")
                before_events = stream.metrics.events_emitted
                stream.finish_input()
                with self.assertRaises(StreamNeedsResolutionError):
                    self.drain(stream)
                self.assertEqual(stream.metrics.events_emitted, before_events)
                self.assertEqual(
                    self.assert_observation(stream, source).state, "uncertain"
                )
                self.assert_retained(stream, source)
                self.assert_released(adapter)

    def test_rejected_coalesced_witness_advances_schedule_but_cannot_agree(self):
        def result_factory(window_id, start, end):
            result = speech_result(window_id, start, end)
            return replace(
                result,
                metadata=replace(
                    result.metadata, no_speech_prob=0.8 if end == 400 else 0.01
                ),
            )

        adapter = ScriptedNativeAdapter(result_factory)
        stream = self.stream(
            adapter, config=replace(self.config, coalesce_previews=True)
        )
        source = pcm_ms(600, 5)
        stream.push(0, source)
        self.assertEqual(self.decode(stream)[0].text, "")
        self.assertEqual(stream.last_trace.audio_evidence.state, "uncertain")
        events = self.decode(stream)
        self.assertEqual([call[2] for call in adapter.calls], [400, 500])
        self.assertEqual([event.kind for event in events], [StreamEventKind.REPLACE])
        self.assertTrue(events[0].text)
        self.assertEqual(stream.last_trace.audio_evidence.state, "speech_candidate")
        self.assert_retained(stream, source)
        with self.assertRaises(StreamNeedsResolutionError):
            stream.step()
        self.assertEqual(len(adapter.calls), 2)
        self.assert_released(adapter)

    def test_clean_speech_still_commits_progressively(self):
        adapter = ScriptedNativeAdapter(speech_result)
        stream = self.stream(adapter)
        source = pcm_ms(250, 900)
        stream.push(0, source)
        events = self.drain(stream)
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual(events[-1].committed_through_sample, 100 * 16)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.last_trace.audio_evidence.state, "speech_candidate")
        self.assertEqual(stream.metrics.buffered_samples, 150 * 16)
        stream.finish_input()
        self.drain(stream)
        self.assertTrue(stream.done)
        self.assertEqual(stream.metrics.committed_samples, len(source) // 2)
        self.assert_released(adapter)

    def test_default_profile_does_not_add_gate_or_evidence(self):
        adapter = ScriptedNativeAdapter(you_result)
        stream = self.stream(adapter, config=replace(self.config, input_evidence=False))
        stream.push(0, pcm_ms(250))
        events = self.drain(stream)
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertIsNone(stream.last_trace.audio_evidence)
        self.assert_released(adapter)

    def test_failed_preprocessing_freezes_pcm_hash_sample_count_and_eof(self):
        for coalesce in (False, True):
            for eof in (False, True):
                with self.subTest(coalesce=coalesce, eof=eof):
                    attempts = []

                    def build(content):
                        attempts.append(content)
                        if len(attempts) == 1:
                            raise RuntimeError("injected preprocessing failure")
                        return content

                    adapter = ScriptedNativeAdapter()
                    stream = self.stream(
                        adapter,
                        mel_builder=build,
                        config=replace(self.config, coalesce_previews=coalesce),
                    )
                    source = pcm_ms(250, 41) + pcm(7, 42)
                    added = pcm_ms(100, 43)
                    frozen = source if coalesce else source[: 100 * 32]
                    stream.push(0, source)
                    with self.assertRaisesRegex(RuntimeError, "preprocessing"):
                        stream.step()
                    stream.push(1, added)
                    if eof:
                        stream.finish_input()
                    events = self.decode(stream)
                    self.assertEqual(attempts, [frozen, frozen])
                    self.assertEqual(adapter.inputs, [frozen])
                    self.assertEqual(events[0].text, "")
                    self.assertFalse(stream.last_trace.eof)
                    self.assert_observation(stream, frozen)
                    self.assert_retained(stream, source + added)
                    if eof:
                        with self.assertRaises(StreamNeedsResolutionError):
                            self.decode(stream)
                        self.assertTrue(stream.last_trace.eof)
                        self.assert_observation(stream, source + added)
                    else:
                        self.decode(stream)
                        extended = source + added if coalesce else source[: 200 * 32]
                        self.assert_observation(stream, extended)
                    self.assert_retained(stream, source + added)
                    self.assert_released(adapter)

    def test_rejected_output_fence_failure_retains_pcm_and_recovery_authority(self):
        for eof in (False, True):
            with self.subTest(eof=eof):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(
                    adapter, config=replace(self.config, coalesce_previews=True)
                )
                source = pcm_ms(250, 29) + pcm(7, 30)
                stream.push(0, source)
                if eof:
                    stream.finish_input()
                stream.step()
                stream.step()
                before = stream.metrics
                run = adapter.runs[-1]
                run.fence.fail = True
                try:
                    with self.assertRaises(TransactionRetainedError) as raised:
                        stream.step()
                    self.assertIsNone(raised.exception.committed_state)
                    self.assertEqual(stream.metrics, before)
                    self.assert_retained(stream, source)
                    evidence = self.assert_observation(stream, source)
                    self.assertEqual(evidence.state, "uncertain")
                    with self.assertRaisesRegex(NativeStreamError, "retained"):
                        stream.step()
                finally:
                    run.fence.fail = False
                    adapter.worker.recover(run.transaction)
                self.assertEqual(stream.step(), ())
                self.assert_retained(stream, source)
                if eof:
                    with self.assertRaises(StreamNeedsResolutionError):
                        stream.step()
                    self.assertEqual(len(adapter.calls), 1)
                    self.assertEqual(stream.metrics.events_emitted, 0)
                else:
                    self.assertEqual(stream.metrics, before)
                    added = pcm_ms(100, 7)
                    stream.push(1, added)
                    self.assertEqual(self.decode(stream)[0].text, "")
                    self.assertEqual(adapter.inputs, [source, source])
                    self.assert_observation(stream, source)
                    self.assert_retained(stream, source + added)
                self.assert_released(adapter)

    def test_silence_then_speech_requires_a_fresh_growing_pair(self):
        for word, context in ((False, 0), (False, 100), (True, 0), (True, 100)):
            with self.subTest(word_alignment=word, context=context):
                adapter = EvidenceNativeAdapter(speech_result)
                stream = self.stream(
                    adapter,
                    config=replace(
                        self.config, left_context_ms=context, word_alignment=word
                    ),
                )
                silence = pcm_ms(100)
                spoken = pcm_ms(300, 800)
                stream.push(0, silence)
                initial = self.decode(stream)
                self.assertEqual(initial[-1].kind, StreamEventKind.COMMIT)
                self.assertEqual(initial[0].text, "")
                self.assertEqual(stream.metrics.committed_samples, 100 * 16)
                self.assertEqual(stream.retained_from_sample, 100 * 16)
                stream.push(1, spoken)
                preview = self.decode(stream)
                self.assertEqual(
                    [event.kind for event in preview], [StreamEventKind.PROVISIONAL]
                )
                self.assertEqual(stream.metrics.committed_samples, 100 * 16)
                committed = self.decode(stream)
                self.assertEqual(committed[-1].kind, StreamEventKind.COMMIT)
                self.assertEqual(committed[-1].committed_through_sample, 200 * 16)
                self.assertEqual(committed[0].text, "word-100")
                self.assertEqual(committed[0].start_sample, 100 * 16)
                self.assertNotEqual(initial[0].segment_id, committed[0].segment_id)
                stream.finish_input()
                self.drain(stream)
                self.assertTrue(stream.done)
                self.assertEqual(stream.metrics.committed_samples, 400 * 16)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assert_released(adapter)

    def test_punctuation_context_suffix_cannot_discard_nonzero_pcm_at_eof(self):
        for enabled in (False, True):
            with self.subTest(input_evidence=enabled):
                adapter = ScriptedNativeAdapter(speech_result)
                stream = self.stream(
                    adapter,
                    config=replace(
                        self.config, left_context_ms=100, input_evidence=enabled
                    ),
                )
                source = pcm_ms(300, 900)
                stream.push(0, source[: 200 * 32])
                self.drain(stream)
                self.assertEqual(stream.metrics.committed_samples, 100 * 16)

                def punctuation_suffix(window_id, start, end):
                    result = speech_result(window_id, start, end)
                    segments = (result.metadata.segments[0],) + tuple(
                        replace(segment, text=" ...")
                        for segment in result.metadata.segments[1:]
                    )
                    return replace(
                        result,
                        text="".join(segment.text for segment in segments).strip(),
                        metadata=replace(result.metadata, segments=segments),
                    )

                adapter.result_factory = punctuation_suffix
                stream.push(1, source[200 * 32 :])
                stream.finish_input()
                before = stream.metrics
                if enabled:
                    with self.assertRaisesRegex(
                        StreamNeedsResolutionError, "no_lexical_text"
                    ):
                        self.drain(stream)
                    trace = stream.last_trace
                    self.assertEqual(trace.audio_evidence.state, "uncertain")
                    self.assertEqual(
                        assess_audio(
                            trace.audio_evidence.observation, trace.result
                        ).state,
                        "speech_candidate",
                    )
                    self.assertEqual(stream.metrics.committed_samples, 100 * 16)
                    self.assertEqual(stream.metrics.buffered_samples, len(source) // 2)
                    self.assertEqual(
                        stream.metrics.events_emitted, before.events_emitted
                    )
                    self.assertEqual(stream.state.version, 1)
                    self.assertEqual(stream.retained_from_sample, 0)
                    self.assertFalse(stream.done)
                else:
                    events = self.drain(stream)
                    self.assertEqual(events[0].text, "... ...")
                    self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                    self.assertEqual(stream.metrics.committed_samples, 300 * 16)
                    self.assertEqual(stream.metrics.buffered_samples, 0)
                    self.assertTrue(stream.done)
                self.assert_released(adapter)

    def test_empty_or_punctuation_word_suffix_cannot_borrow_retained_lexical_text(self):
        for suffix in ("", " ..."):
            for enabled in (False, True):
                with self.subTest(suffix=suffix, input_evidence=enabled):
                    adapter = EvidenceNativeAdapter(speech_result)
                    stream = self.stream(
                        adapter,
                        config=replace(
                            self.config,
                            word_alignment=True,
                            left_context_ms=100,
                            input_evidence=enabled,
                        ),
                    )
                    source = pcm_ms(300, 901)
                    stream.push(0, source[: 200 * 32])
                    self.drain(stream)
                    self.assertEqual(stream.metrics.committed_samples, 100 * 16)
                    self.assertEqual(len(stream._word_anchor), 1)

                    def word_suffix(window_id, start, end):
                        result = speech_result(window_id, start, end)
                        words = (result.metadata.segments[0],)
                        if suffix:
                            words += (
                                replace(result.metadata.segments[1], text=suffix),
                            )
                        return replace(
                            result,
                            text="".join(word.text for word in words).strip(),
                            metadata=replace(result.metadata, segments=words),
                        )

                    adapter.result_factory = word_suffix
                    stream.push(1, source[200 * 32 :])
                    stream.finish_input()
                    before = stream.metrics
                    if enabled:
                        with self.assertRaisesRegex(
                            StreamNeedsResolutionError, "no_lexical_text"
                        ):
                            self.drain(stream)
                        trace = stream.last_trace
                        self.assertEqual(trace.audio_evidence.state, "uncertain")
                        self.assertEqual(
                            assess_audio(
                                trace.audio_evidence.observation, trace.result
                            ).state,
                            "speech_candidate",
                        )
                        self.assertEqual(stream.metrics.committed_samples, 100 * 16)
                        self.assertEqual(
                            stream.metrics.buffered_samples, len(source) // 2
                        )
                        self.assertEqual(
                            stream.metrics.events_emitted, before.events_emitted
                        )
                        self.assertEqual(stream.state.version, 1)
                        self.assertEqual(stream.retained_from_sample, 0)
                        self.assertFalse(stream.done)
                    else:
                        events = self.drain(stream)
                        publication = stream.state.windows[-1].result
                        self.assertIsInstance(publication, AlignedPublication)
                        self.assertTrue(publication.final)
                        self.assertEqual(publication.text, suffix.strip())
                        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                        self.assertEqual(stream.metrics.committed_samples, 300 * 16)
                        self.assertEqual(stream.metrics.buffered_samples, 0)
                        self.assertTrue(stream.done)
                    self.assert_released(adapter)

    def test_silence_commit_clears_word_anchor_without_alignment(self):
        adapter = EvidenceNativeAdapter(you_result)
        stream = self.stream(adapter, config=replace(self.config, word_alignment=True))
        stream._word_anchor = you_result("prior", 0, 100).metadata.segments
        self.assertTrue(stream._word_anchor)
        stream.push(0, pcm_ms(100))
        with patch.object(
            EvidenceNativeRun, "prepare_word_alignment", create=True
        ) as alignment:
            events = self.decode(stream)
            alignment.assert_not_called()
        self.assertEqual(events[-1].kind, StreamEventKind.COMMIT)
        self.assertEqual(stream._word_anchor, ())
        self.assertIsNone(stream._word_previous)
        self.assertIsNone(stream._previous)
        self.assert_released(adapter)

    def test_silence_rolls_across_non_millisecond_sample_origins(self):
        for context in (0, 100):
            with self.subTest(context=context):
                adapter = EvidenceNativeAdapter(you_result)
                stream = self.stream(
                    adapter,
                    config=replace(
                        self.config, coalesce_previews=True, left_context_ms=context
                    ),
                )
                initial = pcm_ms(250) + pcm(7)
                added = pcm_ms(100) + pcm(10)
                stream.push(0, initial)
                self.decode(stream)
                self.assertEqual(stream.metrics.committed_samples, len(initial) // 2)
                stream.push(1, added)
                stream.finish_input()
                events = self.drain(stream)
                self.assertEqual(events[-2].start_sample, len(initial) // 2)
                self.assertEqual(
                    events[-2].committed_through_sample,
                    (len(initial) + len(added)) // 2,
                )
                self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                self.assert_observation(stream, added)
                self.assertEqual(
                    stream.last_trace.silence_publication.analysis_start_sample,
                    len(initial) // 2,
                )
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assertTrue(stream.done)
                self.assert_released(adapter)

    def test_short_silence_eof_commits_submillisecond_coverage_exactly(self):
        for samples in (1, 15, 17):
            with self.subTest(samples=samples):
                adapter = EvidenceNativeAdapter(you_result)
                stream = self.stream(adapter)
                source = pcm(samples)
                stream.push(0, source)
                stream.finish_input()
                events = self.decode(stream)
                self.assertEqual(events[0].text, "")
                self.assertEqual(events[-2].committed_through_sample, samples)
                self.assertEqual(events[-2].committed_through_ms, samples // 16)
                self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                self.assert_observation(stream, source)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assertTrue(stream.done)
                self.assert_released(adapter)

    def test_committed_silence_release_recovery_publishes_once_after_cleanup(self):
        for word in (False, True):
            for eof in (False, True):
                with self.subTest(word=word, eof=eof):
                    adapter = EvidenceNativeAdapter(you_result)
                    stream = self.stream(
                        adapter,
                        config=replace(
                            self.config, word_alignment=word, coalesce_previews=True
                        ),
                    )
                    source = pcm_ms(250) + pcm(7)
                    stream.push(0, source)
                    if eof:
                        stream.finish_input()
                    stream.step()
                    stream.step()
                    before = stream.metrics
                    run = adapter.runs[-1]
                    with patch.object(
                        type(run.transaction._lease),
                        "release",
                        side_effect=RuntimeError("injected silence release failure"),
                    ):
                        with self.assertRaises(TransactionRetainedError) as raised:
                            stream.step()
                        self.assertIsNotNone(raised.exception.committed_state)
                        self.assertIsInstance(
                            stream.state.windows[-1].result, SilencePublication
                        )
                        self.assertEqual(stream.state.version, 1)
                        self.assertEqual(stream.metrics, before)
                        self.assertEqual(stream.retained_from_sample, 0)
                        self.assertFalse(stream.done)
                        self.assertTrue(stream.active)
                        with self.assertRaisesRegex(NativeStreamError, "retained"):
                            stream.step()
                    self.assertTrue(
                        adapter.worker.recover(raised.exception.transaction)
                    )
                    events = stream.step()
                    expected = [StreamEventKind.PROVISIONAL, StreamEventKind.COMMIT]
                    if eof:
                        expected.append(StreamEventKind.FINAL)
                    self.assertEqual([event.kind for event in events], expected)
                    self.assertTrue(all(not event.text for event in events))
                    self.assertEqual(stream.metrics.committed_samples, len(source) // 2)
                    self.assertEqual(stream.metrics.buffered_samples, 0)
                    self.assertEqual(stream.done, eof)
                    self.assertEqual(len(adapter.calls), 1)
                    self.assertEqual(stream.state.version, 1)
                    self.assertEqual(stream.metrics.decode_count, 1)
                    self.assertEqual(stream.step(), ())
                    self.assert_released(adapter)


if __name__ == "__main__":
    unittest.main()
