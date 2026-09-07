"""Pure continuity observations, including counterexamples to publication."""

import copy
import unittest
from dataclasses import FrozenInstanceError, replace

from whisper_runtime import AudioSpan, ModelSnapshot, SessionState, WindowRecord
from whisper_runtime.adapters.audio_evidence import AudioObservation
from whisper_runtime.adapters.continuity_witness import assess_continuity_witness
from whisper_runtime.adapters.continuous_stream import ContinuousAnalysisIdentity
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.native_whisper import NativeDecodeOptions
from whisper_runtime.adapters.word_policy import (
    NativeWordAlignment,
    select_word_publication,
)


def word(text, token, start, end):
    return NativeTimestampSegment(AudioSpan(start, end), text, (token,))


def aligned(words, start, end):
    return NativeWordAlignment(
        NativeWindowResult(
            window_id=f"synthetic:{start}:{end}",
            text="".join(w.text for w in words).strip(),
            start_ms=start,
            end_ms=end,
            metadata=NativeDecodeMetadata(
                "en", tuple(t for w in words for t in w.tokens), no_speech_prob=0.01
            ),
        ),
        words,
    )


class ContinuityWitnessTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelSnapshot("tiny.en", "synthetic", "fake", "sha256:" + "f" * 64)
        self.identity = ContinuousAnalysisIdentity(
            self.model,
            NativeDecodeOptions(without_timestamps=False),
            7,
            tokenizer_artifact_identity="sha256:" + "a" * 64,
            preprocessing_identity="sha256:" + "b" * 64,
            backend_artifact_identity="sha256:" + "c" * 64,
            effective_alignment_mode="legacy_encoder",
        )
        self.anchor = (word(" we", 1, 100, 300), word(" said", 2, 300, 600))
        self.suffix = (word(" go", 3, 800, 1100), word(" home", 4, 1100, 1500))
        self.joint = aligned(self.anchor + self.suffix, 0, 2000)
        self.candidate = aligned(
            self.suffix + (word(" now", 5, 2200, 2500),), 600, 3000
        )
        publication = select_word_publication(self.joint, 0, 2, AudioSpan(0, 600))
        self.published = SessionState(
            "stream", 1, (WindowRecord("request", self.model, publication, 600),), 600
        )
        self.pcm = b"\x01\x00" * 48000
        self.arguments = dict(
            joint=self.joint,
            candidate=self.candidate,
            published=self.published,
            current=self.published,
            joint_audio=AudioObservation.from_pcm(self.pcm[:64000]),
            candidate_audio=AudioObservation.from_pcm(self.pcm[19200:]),
            joint_identity=self.identity,
            candidate_identity=self.identity,
            pcm=self.pcm,
        )

    def assess(self, **changes):
        return assess_continuity_witness(**{**self.arguments, **changes})

    def test_consistent_prefix_is_read_only_and_derives_only_published_anchor(self):
        before = copy.deepcopy(self.arguments)
        result = self.assess()
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.anchor.word_end, 2)
        self.assertEqual(result.sequence.left_unit_count, 2)
        self.assertEqual(result.lexical_deltas_ms, ((" go", 0, 0), (" home", 0, 0)))
        self.assertEqual(self.arguments, before)
        self.assertFalse(hasattr(result, "publication"))
        self.assertFalse(hasattr(result, "next_anchor"))
        with self.assertRaises(FrozenInstanceError):
            result.reasons = ("changed",)

    def test_stale_version_head_or_session_is_refused(self):
        for current in (
            replace(self.published, version=2),
            replace(self.published, committed_through_ms=700),
            replace(self.published, session_id="another"),
        ):
            with self.subTest(current=current):
                self.assertIn(
                    "stale_published_state", self.assess(current=current).reasons
                )

    def test_changed_audio_inside_either_observed_slice_is_refused(self):
        for sample, expected in (
            (100, "joint_pcm_mismatch"),
            (40000, "candidate_pcm_mismatch"),
        ):
            pcm = bytearray(self.pcm)
            pcm[sample * 2] = 2
            self.assertIn(expected, self.assess(pcm=bytes(pcm)).reasons)

    def test_source_sample_origin_or_digest_change_is_refused(self):
        self.assertIn("joint_pcm_mismatch", self.assess(pcm_start_sample=16).reasons)
        changed = replace(self.arguments["candidate_audio"], pcm_sha256="0" * 64)
        self.assertIn(
            "candidate_pcm_mismatch", self.assess(candidate_audio=changed).reasons
        )

    def test_changed_identity_or_missing_effective_provenance_is_refused(self):
        changed = replace(self.identity, preprocessing_identity="different")
        self.assertIn(
            "analysis_identity_mismatch",
            self.assess(candidate_identity=changed).reasons,
        )
        incomplete = replace(self.identity, effective_alignment_mode=None)
        self.assertIn(
            "incomplete_analysis_identity",
            self.assess(joint_identity=incomplete).reasons,
        )
        self.assertIn(
            "missing_analysis_identity", self.assess(candidate_identity=None).reasons
        )

    def test_model_must_match_actual_publication(self):
        changed = replace(
            self.identity, declared_model=replace(self.model, revision="other")
        )
        self.assertIn(
            "published_model_mismatch",
            self.assess(joint_identity=changed, candidate_identity=changed).reasons,
        )

    def test_prompt_history_and_decoder_prefix_are_not_independent_evidence(self):
        for options in (
            replace(self.identity.requested_decode_options, prompt="we said"),
            replace(self.identity.requested_decode_options, prefix=(1, 2)),
        ):
            changed = replace(self.identity, requested_decode_options=options)
            self.assertIn(
                "prompt_conditioned_observation",
                self.assess(joint_identity=changed, candidate_identity=changed).reasons,
            )

    def test_missing_word_is_not_searched_around(self):
        candidate = aligned(self.candidate.words[1:], 600, 3000)
        result = self.assess(candidate=candidate)
        self.assertIn("continuation_sequence_mismatch", result.reasons)
        self.assertEqual(result.lexical_deltas_ms, ())

    def test_repeated_anchor_anywhere_remains_ambiguous(self):
        repeated = tuple(
            replace(w, span=AudioSpan(w.span.start_ms + 1500, w.span.end_ms + 1500))
            for w in self.anchor
        )
        joint = aligned(self.anchor + self.suffix + repeated, 0, 2200)
        result = self.assess(
            joint=joint, joint_audio=AudioObservation.from_pcm(self.pcm[:70400])
        )
        self.assertEqual(result.anchor.lexical_match_count, 2)
        self.assertIn("joint_anchor_ambiguous", result.reasons)

    def test_repeated_continuation_is_not_subsequence_matched(self):
        repeated = (word(" go", 3, 1500, 1700), word(" home", 4, 1700, 1900))
        candidate = aligned(self.suffix + repeated, 600, 3000)
        self.assertIn(
            "continuation_sequence_mismatch", self.assess(candidate=candidate).reasons
        )

    def test_punctuation_is_explained_but_never_removed_for_acceptance(self):
        joint = aligned(self.anchor + (word(".", 13, 600, 800),) + self.suffix, 0, 2000)
        result = self.assess(joint=joint)
        self.assertIn("representation_mismatch", result.reasons)
        self.assertEqual(result.lexical.text_relation, "exact_units")
        self.assertEqual(result.sequence.text_relation, "different_units")

    def test_token_or_case_changes_remain_visible(self):
        for changed_word in (
            replace(self.suffix[0], tokens=(99,)),
            replace(self.suffix[0], text=" Go"),
        ):
            candidate = aligned((changed_word,) + self.candidate.words[1:], 600, 3000)
            self.assertTrue(self.assess(candidate=candidate).reasons)

    def test_punctuation_timing_is_not_hidden_by_lexical_view(self):
        joint = aligned(self.anchor + (word(".", 13, 600, 650),) + self.suffix, 0, 2000)
        candidate = aligned(
            (word(".", 13, 600, 900), word(" go", 3, 900, 1100))
            + self.candidate.words[1:],
            600,
            3000,
        )
        result = self.assess(joint=joint, candidate=candidate)
        self.assertIn("unit_timing_mismatch", result.reasons)
        self.assertNotIn("lexical_timing_mismatch", result.reasons)

    def test_420_ms_boundary_shift_has_no_new_origin_exception(self):
        joint_words = (
            replace(self.suffix[0], span=AudioSpan(1020, 1100)),
        ) + self.suffix[1:]
        candidate_words = (
            replace(self.suffix[0], span=AudioSpan(600, 1100)),
        ) + self.candidate.words[1:]
        result = self.assess(
            joint=aligned(self.anchor + joint_words, 0, 2000),
            candidate=aligned(candidate_words, 600, 3000),
        )
        self.assertIn("lexical_timing_mismatch", result.reasons)
        self.assertEqual(result.lexical_deltas_ms[0], (" go", -420, 0))

    def test_crossing_and_zero_duration_are_not_repaired(self):
        candidate = aligned(
            (word(" go", 3, 600, 600),) + self.candidate.words[1:], 600, 3000
        )
        self.assertIn(
            "zero_duration_continuation_unit", self.assess(candidate=candidate).reasons
        )
        moved = replace(self.anchor[-1], span=AudioSpan(300, 650))
        joint = aligned(self.anchor[:-1] + (moved,) + self.suffix, 0, 2000)
        candidate = aligned(
            (word(" go", 3, 600, 1100),) + self.candidate.words[1:], 600, 3000
        )
        self.assertIn(
            "continuation_crosses_published_boundary",
            self.assess(joint=joint, candidate=candidate).reasons,
        )

    def test_overlap_cut_does_not_hide_candidate_tail(self):
        candidate = aligned(self.suffix + (word(" more", 9, 1900, 2300),), 600, 3000)
        self.assertIn(
            "overlap_cuts_candidate_word", self.assess(candidate=candidate).reasons
        )

    def test_correspondence_cannot_exclude_a_shared_acoustic_omission(self):
        # Both recognitions omit the same possible spoken word. PCM is unchanged;
        # matching output arrays do not determine what the audio actually said.
        joint = aligned(self.anchor + self.suffix[1:], 0, 2000)
        candidate = aligned(self.candidate.words[1:], 600, 3000)
        result = self.assess(joint=joint, candidate=candidate)
        self.assertEqual(result.reasons, ())
        self.assertFalse(hasattr(result, "publication"))

    def test_missing_publication_or_unbound_head_is_refused(self):
        empty = SessionState("stream")
        self.assertIn(
            "missing_aligned_publication",
            self.assess(published=empty, current=empty).reasons,
        )
        changed = replace(self.published, committed_through_ms=601)
        self.assertIn(
            "unbound_published_boundary",
            self.assess(published=changed, current=changed).reasons,
        )

    def test_native_windows_must_straddle_the_actual_published_head(self):
        candidate = aligned(self.candidate.words, 590, 3000)
        self.assertIn(
            "incompatible_observation_windows", self.assess(candidate=candidate).reasons
        )

    def test_argument_contract_rejects_invalid_types(self):
        for name, value in (
            ("pcm", bytearray(self.pcm)),
            ("joint", None),
            ("candidate_identity", object()),
        ):
            with self.subTest(name=name), self.assertRaises(TypeError):
                self.assess(**{name: value})
        for name, value in (
            ("pcm", b"x"),
            ("pcm_start_sample", True),
            ("timestamp_tolerance_ms", -1),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.assess(**{name: value})


if __name__ == "__main__":
    unittest.main()
