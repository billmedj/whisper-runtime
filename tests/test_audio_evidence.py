"""Pure source-evidence classification; no model, devices, or VAD dependency."""

import hashlib
import unittest
from dataclasses import FrozenInstanceError, replace

from whisper_runtime.adapters.audio_evidence import (
    AudioEvidenceDecision,
    AudioObservation,
    SilencePublication,
    assess_audio,
)
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
    select_native_publication,
)
from whisper_runtime.state import AudioSpan, WindowResult


def native(text="you", score=0.1, *, metadata=True, end_ms=1):
    return NativeWindowResult(
        window_id="evidence-window",
        text=text,
        start_ms=0,
        end_ms=end_ms,
        metadata=(
            NativeDecodeMetadata(
                language="en",
                tokens=(1,) if text else (),
                avg_logprob=-0.01,
                no_speech_prob=score,
            )
            if metadata
            else None
        ),
    )


class AudioObservationTests(unittest.TestCase):
    def test_exact_zero_pcm_records_only_immutable_source_facts(self):
        pcm = b"\x00\x00" * 16
        observation = AudioObservation.from_pcm(pcm)
        self.assertEqual(observation.sample_count, 16)
        self.assertEqual(observation.pcm_sha256, hashlib.sha256(pcm).hexdigest())
        self.assertTrue(observation.digital_silence)
        self.assertFalse(hasattr(observation, "pcm"))
        with self.assertRaises(FrozenInstanceError):
            observation.sample_count = 17

    def test_quiet_or_single_nonzero_samples_are_never_digital_silence(self):
        for pcm in (
            b"\x01\x00" * 16,
            b"\xff\xff" * 16,
            b"\x00\x00" * 15 + b"\x01\x00",
            b"\x00\x01" + b"\x00\x00" * 15,
        ):
            with self.subTest(pcm=pcm):
                observation = AudioObservation.from_pcm(pcm)
                self.assertFalse(observation.digital_silence)
                for score in (0.0, 0.6, 1.0, None):
                    self.assertNotEqual(
                        assess_audio(observation, native(score=score)).state,
                        "non_speech",
                    )

    def test_hash_binds_exact_bytes_not_only_amplitude_or_count(self):
        before = AudioObservation.from_pcm(b"\x01\x00\x00\x00")
        after = AudioObservation.from_pcm(b"\x00\x00\x01\x00")
        self.assertEqual(before.sample_count, after.sample_count)
        self.assertNotEqual(before.pcm_sha256, after.pcm_sha256)

    def test_seven_sample_eof_tail_is_not_rounded_away(self):
        for value in (b"\x00\x00", b"\x01\x00"):
            with self.subTest(value=value):
                observation = AudioObservation.from_pcm(value * 7)
                decision = assess_audio(observation, native(end_ms=0))
                self.assertEqual(decision.observation.sample_count, 7)
                self.assertEqual(
                    decision.state,
                    "non_speech" if value == b"\x00\x00" else "speech_candidate",
                )

    def test_pcm_requires_nonempty_complete_immutable_sample_bytes(self):
        for value in (None, "00", bytearray(2), memoryview(bytes(2)), [0, 0]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                AudioObservation.from_pcm(value)
        for value in (b"", b"\x00", b"\x00\x00\x00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AudioObservation.from_pcm(value)

    def test_sample_count_is_a_nonboolean_positive_integer(self):
        observation = AudioObservation.from_pcm(bytes(2))
        for value in (True, False, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(observation, sample_count=value)
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(observation, sample_count=value)

    def test_digest_requires_exactly_64_hexadecimal_characters(self):
        observation = AudioObservation.from_pcm(bytes(2))
        for value in (None, bytes(64), 123):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(observation, pcm_sha256=value)
        for value in ("", "a" * 63, "a" * 65, "g" * 64, " " + "a" * 63):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(observation, pcm_sha256=value)
        self.assertEqual(
            replace(observation, pcm_sha256="ABCDEF01" * 8).pcm_sha256,
            "ABCDEF01" * 8,
        )

    def test_digital_silence_requires_an_actual_boolean(self):
        observation = AudioObservation.from_pcm(bytes(2))
        for value in (0, 1, None, "false"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(observation, digital_silence=value)


class AudioEvidenceDecisionTests(unittest.TestCase):
    def setUp(self):
        self.nonzero = AudioObservation.from_pcm(b"\x01\x00" * 16)
        self.zero = AudioObservation.from_pcm(bytes(32))

    def test_digital_zero_overrides_hallucinated_you_and_high_text_confidence(self):
        for score in (0.0, 0.1, 0.94770747423172, 1.0, None):
            with self.subTest(score=score):
                result = native("you", score)
                decision = assess_audio(self.zero, result)
                self.assertEqual(decision.state, "non_speech")
                self.assertEqual(decision.reason, "digital_silence")
                self.assertEqual(decision.no_speech_prob, score)
                self.assertEqual(result.text, "you")
                self.assertEqual(result.metadata.avg_logprob, -0.01)

    def test_digital_zero_takes_precedence_over_missing_or_nonlexical_output(self):
        for result in (native("", None), native("...", 1), native(metadata=False)):
            with self.subTest(result=result):
                decision = assess_audio(self.zero, result)
                self.assertEqual(
                    (decision.state, decision.reason), ("non_speech", "digital_silence")
                )

    def test_empty_and_punctuation_output_is_uncertain_before_score_rules(self):
        for text in ("", " \t\n", ".", "…!?", " -- ", "♪"):
            for score in (None, 0.0, 0.6, 1.0):
                with self.subTest(text=text, score=score):
                    decision = assess_audio(self.nonzero, native(text, score))
                    self.assertEqual(decision.state, "uncertain")
                    self.assertEqual(decision.reason, "no_lexical_text")
                    self.assertEqual(decision.no_speech_prob, score)

    def test_unicode_letters_and_numbers_are_lexical_candidates(self):
        for text in ("you", "é", "你好", "٣", "123", "word!"):
            with self.subTest(text=text):
                decision = assess_audio(self.nonzero, native(text))
                self.assertEqual(decision.state, "speech_candidate")
                self.assertEqual(decision.reason, "model_speech_candidate")

    def test_missing_metadata_or_missing_score_remains_uncertain(self):
        for result in (
            native(metadata=False),
            native(score=None),
            native(score=float("nan")),
        ):
            with self.subTest(result=result):
                decision = assess_audio(self.nonzero, result)
                self.assertEqual(decision.state, "uncertain")
                self.assertEqual(decision.reason, "missing_speech_score")
                self.assertIsNone(decision.no_speech_prob)

    def test_model_score_threshold_is_inclusive_and_not_nonspeech_proof(self):
        for score in (0.0, 0.599999999):
            with self.subTest(score=score):
                decision = assess_audio(self.nonzero, native(score=score))
                self.assertEqual(decision.state, "speech_candidate")
                self.assertEqual(decision.no_speech_prob, score)
        for score in (0.6, 0.94770747423172, 1.0):
            with self.subTest(score=score):
                decision = assess_audio(self.nonzero, native(score=score))
                self.assertEqual(decision.state, "uncertain")
                self.assertEqual(decision.reason, "conflicting_speech_score")
                self.assertEqual(decision.no_speech_prob, score)

    def test_decision_is_immutable_and_preserves_observation_identity(self):
        decision = assess_audio(self.nonzero, native())
        self.assertIs(decision.observation, self.nonzero)
        with self.assertRaises(FrozenInstanceError):
            decision.state = "non_speech"

    def test_assessment_rejects_wrong_input_types(self):
        with self.assertRaises(TypeError):
            assess_audio(None, native())
        with self.assertRaises(TypeError):
            assess_audio(self.nonzero, object())

    def test_decision_validates_observation_and_state_reason_pair(self):
        with self.assertRaises(TypeError):
            AudioEvidenceDecision(None, "uncertain", "no_lexical_text")
        decision = assess_audio(self.nonzero, native())
        for changes in (
            {"state": "speech"},
            {"state": None},
            {"reason": "digital_silence"},
            {"reason": None},
            {"state": "non_speech", "reason": "digital_silence"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(decision, **changes)
        with self.assertRaises(ValueError):
            AudioEvidenceDecision(self.zero, "uncertain", "no_lexical_text")

    def test_decision_rejects_invalid_scores_and_score_reason_conflicts(self):
        decision = AudioEvidenceDecision(self.nonzero, "uncertain", "no_lexical_text")
        for score in (True, False, "0.5", object()):
            with self.subTest(score=score), self.assertRaises(TypeError):
                replace(decision, no_speech_prob=score)
        for score in (-0.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                replace(decision, no_speech_prob=score)
        for state, reason, score in (
            ("uncertain", "missing_speech_score", 0.5),
            ("uncertain", "conflicting_speech_score", None),
            ("uncertain", "conflicting_speech_score", 0.5),
            ("speech_candidate", "model_speech_candidate", None),
            ("speech_candidate", "model_speech_candidate", 0.6),
        ):
            with (
                self.subTest(reason=reason, score=score),
                self.assertRaises(ValueError),
            ):
                AudioEvidenceDecision(self.nonzero, state, reason, score)
        self.assertIsInstance(replace(decision, no_speech_prob=1).no_speech_prob, float)


class SilencePublicationTests(unittest.TestCase):
    def publication(self, *, start_ms=0, end_ms=1, samples=16):
        result = native(score=0.94770747423172, end_ms=end_ms)
        return SilencePublication(
            window_id=result.window_id,
            text="",
            start_ms=start_ms,
            end_ms=end_ms,
            analysis_span=result.analyzed_span,
            native=result,
            observation=AudioObservation.from_pcm(bytes(samples * 2)),
            analysis_start_sample=0,
        )

    def test_nonspeech_coverage_is_empty_with_full_unmodified_native_provenance(self):
        publication = self.publication()
        decision = assess_audio(publication.observation, publication.native)
        self.assertEqual(decision.state, "non_speech")
        self.assertIsInstance(publication, WindowResult)
        self.assertEqual(publication.text, "")
        self.assertEqual(publication.native.text, "you")
        self.assertEqual(publication.native.metadata.tokens, (1,))
        self.assertEqual(publication.native.metadata.no_speech_prob, 0.94770747423172)
        self.assertIsNone(publication.native.publication_segment_indices)
        self.assertIs(replace(publication).native, publication.native)
        with self.assertRaises(FrozenInstanceError):
            publication.text = "you"

    def test_coverage_can_exclude_committed_context_not_source_observation(self):
        publication = self.publication(start_ms=1, end_ms=2, samples=32)
        self.assertEqual((publication.start_ms, publication.end_ms), (1, 2))
        self.assertEqual(publication.analyzed_span, AudioSpan(0, 2))
        self.assertEqual(publication.observation.sample_count, 32)

    def test_seven_sample_eof_tail_and_submillisecond_remainder_are_preserved(self):
        for duration, samples in ((0, 7), (1, 16), (1, 23), (1, 31)):
            with self.subTest(duration=duration, samples=samples):
                publication = self.publication(end_ms=duration, samples=samples)
                self.assertEqual(publication.end_ms, duration)
                self.assertEqual(publication.observation.sample_count, samples)

    def test_non_millisecond_origin_uses_both_exact_source_endpoints(self):
        publication = self.publication()
        shifted = replace(
            publication,
            analysis_start_sample=7,
            observation=AudioObservation.from_pcm(bytes(9 * 2)),
        )
        self.assertEqual(shifted.analysis_start_sample, 7)
        self.assertEqual(shifted.observation.sample_count, 9)
        self.assertEqual(shifted.analyzed_span, AudioSpan(0, 1))
        native = replace(publication.native, start_ms=250, end_ms=351)
        coalesced = SilencePublication(
            window_id=native.window_id,
            text="",
            start_ms=250,
            end_ms=351,
            analysis_span=native.analyzed_span,
            native=native,
            observation=AudioObservation.from_pcm(bytes(1610 * 2)),
            analysis_start_sample=4007,
        )
        self.assertEqual(
            coalesced.analysis_start_sample + coalesced.observation.sample_count, 5617
        )

    def test_sample_origin_requires_nonboolean_nonnegative_integer_and_exact_bounds(
        self,
    ):
        publication = self.publication()
        for origin in (True, False, None, 0.0, "0"):
            with self.subTest(origin=origin), self.assertRaises(TypeError):
                replace(publication, analysis_start_sample=origin)
        for origin in (-1, 16, 32):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                replace(publication, analysis_start_sample=origin)
        with self.assertRaisesRegex(ValueError, "sample boundaries"):
            replace(
                publication,
                analysis_start_sample=15,
                observation=AudioObservation.from_pcm(bytes(17 * 2)),
            )

    def test_nonzero_pcm_cannot_authorize_silence_publication(self):
        publication = self.publication()
        for pcm in (b"\x01\x00" * 16, b"\xff\xff" * 16, bytes(30) + b"\x01\x00"):
            with (
                self.subTest(pcm=pcm),
                self.assertRaisesRegex(ValueError, "digital-zero"),
            ):
                replace(publication, observation=AudioObservation.from_pcm(pcm))

    def test_wrong_types_and_nonempty_output_are_rejected(self):
        publication = self.publication()
        for changes in ({"native": object()}, {"observation": object()}):
            with self.subTest(changes=changes), self.assertRaises(TypeError):
                replace(publication, **changes)
        for text in ("you", " ", None, 0):
            with self.subTest(text=text), self.assertRaises(ValueError):
                replace(publication, text=text)

    def test_native_window_analysis_and_complete_suffix_are_required(self):
        publication = self.publication(end_ms=2, samples=32)
        for changes in (
            {"window_id": "other"},
            {"analysis_span": AudioSpan(0, 3)},
            {"end_ms": 1},
            {"start_ms": -1},
            {"start_ms": 3},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(publication, **changes)
        cropped = replace(publication.native, start_ms=1, analysis_span=AudioSpan(0, 2))
        with self.assertRaisesRegex(ValueError, "full native result"):
            replace(publication, native=cropped)

    def test_selected_native_segments_are_not_full_native_provenance(self):
        segment = NativeTimestampSegment(AudioSpan(0, 1), " you", (1,))
        full = NativeWindowResult(
            window_id="evidence-window",
            text="you",
            start_ms=0,
            end_ms=1,
            metadata=NativeDecodeMetadata("en", (1,), segments=(segment,)),
        )
        selected = select_native_publication(full, AudioSpan(0, 1))
        with self.assertRaisesRegex(ValueError, "full native result"):
            replace(self.publication(), native=selected)

    def test_observation_duration_cannot_be_shorter_or_a_whole_ms_longer(self):
        publication = self.publication()
        for samples in (1, 7, 15, 32, 33):
            with (
                self.subTest(samples=samples),
                self.assertRaisesRegex(ValueError, "sample boundaries"),
            ):
                replace(
                    publication,
                    observation=AudioObservation.from_pcm(bytes(samples * 2)),
                )


if __name__ == "__main__":
    unittest.main()
