"""Opt-in earlier-witness maturity without relaxing timestamps or final gates."""

import unittest
from dataclasses import replace

from test_continuous_endpoint_words import word_result
from test_continuous_evidence import EvidenceNativeAdapter
from test_continuous_stream import pcm_ms
from test_word_policy import aligned, word

from whisper_runtime.adapters import _checkpoint_io as checkpoint
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
)
from whisper_runtime.adapters.word_policy import compare_word_hypotheses


class PreviousHoldbackPolicyTests(unittest.TestCase):
    def test_immature_previous_end_cannot_freeze_an_agreeing_later_word(self):
        before = aligned(
            (word(1, 34260, 35720, " Concord"), word(2, 37660, 37940, " tents")),
            start=34260,
            end=38260,
        )
        current = aligned(
            (word(1, 34260, 35720, " Concord"), word(2, 37660, 38100, " tents")),
            start=34260,
            end=40260,
        )
        arguments = dict(committed_through_ms=34260, holdback_ms=2000)
        old = compare_word_hypotheses(before, current, **arguments)
        explicit_zero = compare_word_hypotheses(
            before, current, previous_holdback_ms=0, **arguments
        )
        proposed = compare_word_hypotheses(
            before, current, previous_holdback_ms=2000, **arguments
        )
        self.assertEqual(old, explicit_zero)
        self.assertEqual(old.publication.text, "Concord tents")
        self.assertEqual(old.publication.end_ms, 38100)
        self.assertEqual(proposed.publication.text, "Concord")
        self.assertEqual(proposed.publication.end_ms, 35720)
        self.assertEqual(current.words[-1].span.end_ms, 38100)

    def test_first_commit_can_remain_at_six_seconds_with_a_shorter_prefix(self):
        words = tuple(
            word(index + 1, index * 500, (index + 1) * 500) for index in range(7)
        )
        before, current = aligned(words, end=4000), aligned(words, end=6000)
        decision = compare_word_hypotheses(
            before,
            current,
            committed_through_ms=0,
            holdback_ms=2000,
            previous_holdback_ms=2000,
        )
        self.assertEqual(decision.publication.word_end, 4)
        self.assertEqual(decision.publication.end_ms, 2000)

    def test_same_two_sided_mature_word_can_commit_and_boundary_is_inclusive(self):
        words = (word(1, 100, 1000), word(2, 1100, 2000))
        before, current = aligned(words, end=4000), aligned(words, end=6000)
        decision = compare_word_hypotheses(
            before,
            current,
            committed_through_ms=0,
            holdback_ms=2000,
            previous_holdback_ms=2000,
        )
        self.assertEqual(decision.publication.word_end, 2)
        self.assertEqual(decision.publication.end_ms, 2000)
        blocked = compare_word_hypotheses(
            before,
            current,
            committed_through_ms=0,
            holdback_ms=2000,
            previous_holdback_ms=2001,
        )
        self.assertEqual(blocked.publication.word_end, 1)

    def test_current_holdback_and_timestamp_tolerance_still_apply(self):
        before = aligned((word(1, 100, 1000),), end=4000)
        changed = aligned((word(1, 100, 1201),), end=6000)
        self.assertIsNone(
            compare_word_hypotheses(
                before,
                changed,
                committed_through_ms=0,
                holdback_ms=2000,
                previous_holdback_ms=2000,
            ).publication
        )
        current = aligned(before.words, end=4500)
        self.assertIsNone(
            compare_word_hypotheses(
                before,
                current,
                committed_through_ms=0,
                holdback_ms=4000,
                previous_holdback_ms=2000,
            ).publication
        )

    def test_final_ignores_holdback_but_never_ignores_anchor_mismatch(self):
        current = aligned((word(1, 0, 500), word(2, 500, 900)), end=1000)
        arguments = dict(committed_through_ms=0, final=True, holdback_ms=2000)
        self.assertEqual(
            compare_word_hypotheses(None, current, **arguments),
            compare_word_hypotheses(
                None, current, previous_holdback_ms=2000, **arguments
            ),
        )
        invalid_anchor = (word(1, 0, 100),)
        refused = compare_word_hypotheses(
            None,
            current,
            committed_through_ms=500,
            anchor=invalid_anchor,
            final=True,
            previous_holdback_ms=2000,
        )
        self.assertIsNone(refused.publication)
        self.assertEqual(refused.reason, "anchor_missing")

    def test_previous_holdback_type_and_range_are_strict_even_at_final(self):
        current = aligned((word(1, 0, 100),))
        for value, error in (
            (True, TypeError),
            (1.5, TypeError),
            (None, TypeError),
            (-1, ValueError),
        ):
            with self.subTest(value=value), self.assertRaises(error):
                compare_word_hypotheses(
                    None,
                    current,
                    committed_through_ms=0,
                    previous_holdback_ms=value,
                    final=True,
                )


class PreviousHoldbackStreamTests(unittest.TestCase):
    def test_config_validation_and_zero_compatibility(self):
        self.assertEqual(ContinuousStreamConfig().previous_holdback_ms, 0)
        for value, error in (
            (True, TypeError),
            (1.5, TypeError),
            (-1, ValueError),
            (30000, ValueError),
        ):
            with self.subTest(value=value), self.assertRaises(error):
                ContinuousStreamConfig(word_alignment=True, previous_holdback_ms=value)
        with self.assertRaisesRegex(ValueError, "word-aligned"):
            ContinuousStreamConfig(previous_holdback_ms=100)

    def test_stream_forwards_holdback_and_versions_only_opted_in_identity(self):
        adapter = EvidenceNativeAdapter(word_result)
        config = ContinuousStreamConfig(
            preview_interval_ms=1000,
            max_window_ms=10000,
            holdback_ms=1000,
            word_alignment=True,
            input_evidence=True,
            previous_holdback_ms=1000,
        )
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="two-witnesses",
            config=config,
            mel_builder=lambda value: value,
        )
        self.addCleanup(stream.close)
        self.assertEqual(
            stream.profile_id,
            "word_agreement_stream/v1+previous_holdback1000/v1+input_evidence/v1",
        )
        for index in range(2):
            stream.push(index, pcm_ms(1000, 900))
            for _ in range(20):
                if not stream.ready:
                    break
                stream.step()
            self.assertEqual(stream.metrics.committed_samples, 0)
        stream.push(2, pcm_ms(1000, 900))
        for _ in range(20):
            if not stream.ready:
                break
            stream.step()
        self.assertEqual(stream.metrics.committed_samples, 16000)
        standard = ContinuousTranscriptStream(
            EvidenceNativeAdapter(word_result),
            stream_id="legacy-witness",
            config=replace(config, previous_holdback_ms=0),
            mel_builder=lambda value: value,
        )
        self.addCleanup(standard.close)
        self.assertEqual(
            standard.profile_id, "word_agreement_stream/v1+input_evidence/v1"
        )

    def test_new_checkpoint_field_roundtrips_exactly(self):
        config = ContinuousStreamConfig(word_alignment=True, previous_holdback_ms=2000)
        encoded = checkpoint.encode({"config": config}, (ContinuousStreamConfig,))
        restored = checkpoint.decode(encoded, (ContinuousStreamConfig,))["config"]
        self.assertEqual(restored, config)
        self.assertEqual(restored.previous_holdback_ms, 2000)


if __name__ == "__main__":
    unittest.main()
