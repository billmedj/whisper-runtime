"""Exact observed failure classes; diagnostics cannot authorize publication."""

import json
import random
import unittest
from dataclasses import FrozenInstanceError, asdict, replace

from test_word_policy import aligned, word

from whisper_runtime.adapters import diagnose_word_anchor
from whisper_runtime.adapters.word_policy import compare_word_hypotheses


class AnchorDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.anchor = (word(1, 100, 200), word(2, 200, 300))

    def diagnose(self, words, **kwargs):
        arguments = dict(
            committed_through_ms=300,
            anchor=self.anchor,
            timestamp_tolerance_ms=20,
        )
        arguments.update(kwargs)
        return diagnose_word_anchor(aligned(words, end=1000), **arguments)

    def test_missing_text_tokens_timing_and_relocation_are_distinct(self):
        cases = (
            ("lexical_missing", (word(8, 100, 200), word(9, 200, 300))),
            ("token_mismatch", (self.anchor[0], replace(self.anchor[1], tokens=(99,)))),
            ("timing_mismatch", (self.anchor[0], word(2, 200, 840))),
            ("relocated", (word(1, 400, 500), word(2, 500, 600))),
        )
        for status, words in cases:
            with self.subTest(status=status):
                current = aligned(words, end=1000)
                decision = compare_word_hypotheses(
                    None,
                    current,
                    committed_through_ms=300,
                    anchor=self.anchor,
                    timestamp_tolerance_ms=20,
                    final=True,
                )
                self.assertEqual(decision.reason, "anchor_missing")
                self.assertIsNone(decision.publication)
                self.assertEqual(decision.next_anchor, self.anchor)
                self.assertEqual(decision.anchor_diagnostic.status, status)
                self.assertEqual(current.words, words)
                self.assertEqual(decision.anchor_diagnostic, self.diagnose(words))

    def test_end_shift_is_reported_signed_without_rewriting_estimates(self):
        words = (word(1, 100, 200), word(2, 180, 840))
        # Ordered alignments reject an overlap before any diagnostic is evaluated.
        with self.assertRaises(ValueError):
            self.diagnose(words)
        words = (word(1, 100, 180), word(2, 180, 840))
        result = self.diagnose(words)
        self.assertEqual(result.start_deltas_ms, (0, -20))
        self.assertEqual(result.end_deltas_ms, (-20, 540))
        self.assertEqual((result.word_start, result.word_end), (0, 2))
        self.assertEqual(result.lexical_match_count, 1)
        self.assertEqual(result.timed_match_count, 0)

    def test_exact_repeats_remain_visible_when_old_timing_selects_one(self):
        words = self.anchor + (word(1, 400, 500), word(2, 500, 600))
        result = self.diagnose(words)
        self.assertEqual(result.status, "matched")
        self.assertEqual((result.lexical_match_count, result.timed_match_count), (2, 1))
        self.assertEqual((result.word_start, result.word_end), (0, 2))
        decision = compare_word_hypotheses(
            None,
            aligned(words),
            committed_through_ms=300,
            anchor=self.anchor,
            final=True,
        )
        self.assertEqual(decision.reason, "eof")
        self.assertEqual(decision.publication.word_start, 2)

    def test_multiple_untimed_occurrences_are_not_a_unique_timing_failure(self):
        anchor = (word(1, 400, 500), word(2, 500, 600))
        words = (
            word(1, 100, 200),
            word(2, 200, 300),
            word(1, 700, 800),
            word(2, 800, 900),
        )
        result = self.diagnose(words, anchor=anchor, committed_through_ms=600)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.lexical_match_count, 2)
        self.assertEqual(result.timed_match_count, 0)
        self.assertIsNone(result.word_start)
        self.assertEqual(result.start_deltas_ms, ())

    def test_multiple_timed_matches_preserve_ambiguous_rejection(self):
        anchor = (word(1, 400, 500), word(2, 500, 600))
        words = (
            word(1, 380, 430),
            word(2, 430, 480),
            word(1, 500, 550),
            word(2, 550, 600),
        )
        decision = compare_word_hypotheses(
            None,
            aligned(words),
            committed_through_ms=600,
            anchor=anchor,
            timestamp_tolerance_ms=120,
            final=True,
        )
        self.assertEqual(decision.reason, "anchor_ambiguous")
        self.assertEqual(decision.anchor_diagnostic.timed_match_count, 2)
        self.assertEqual(decision.anchor_diagnostic.status, "ambiguous")
        self.assertIsNone(decision.publication)

    def test_previous_observation_is_identified_not_blended_with_current(self):
        previous = aligned((self.anchor[0], word(2, 200, 600)), end=800)
        current = aligned(self.anchor + (word(3, 320, 400),), end=1000)
        decision = compare_word_hypotheses(
            previous,
            current,
            committed_through_ms=300,
            anchor=self.anchor,
            timestamp_tolerance_ms=20,
        )
        self.assertEqual(decision.reason, "anchor_missing")
        self.assertEqual(decision.anchor_diagnostic.status, "timing_mismatch")
        self.assertEqual(decision.anchor_diagnostic.observation, "previous")
        self.assertEqual(decision.anchor_diagnostic.end_deltas_ms, (0, 300))

    def test_matched_anchor_does_not_certify_unstable_continuation(self):
        previous = aligned(self.anchor + (word(3, 320, 400),), end=800)
        current = aligned(self.anchor + (word(4, 320, 400),), end=1000)
        decision = compare_word_hypotheses(
            previous,
            current,
            committed_through_ms=300,
            anchor=self.anchor,
            holdback_ms=100,
        )
        self.assertEqual(decision.reason, "unstable")
        self.assertEqual(decision.anchor_diagnostic.status, "matched")
        self.assertIsNone(decision.publication)

    def test_previous_analysis_can_end_before_current_committed_boundary(self):
        # Legacy comparison permits this previous witness. The diagnostic must
        # not add a new exception or infer agreement over unobserved input.
        anchor = (word(1, 50, 100), word(2, 100, 200))
        previous = aligned(anchor, end=250)
        current = aligned(anchor + (word(3, 350, 400),), end=1000)
        decision = compare_word_hypotheses(
            previous,
            current,
            committed_through_ms=300,
            anchor=anchor,
            timestamp_tolerance_ms=20,
        )
        self.assertEqual(decision.reason, "incomplete")
        self.assertIsNone(decision.publication)
        self.assertEqual(decision.anchor_diagnostic.status, "matched")
        detail = diagnose_word_anchor(
            previous,
            committed_through_ms=300,
            anchor=anchor,
            observation="previous",
        )
        self.assertEqual(detail.status, "matched")

    def test_unavailable_context_is_not_missing_native_text(self):
        for anchor in ((), self.anchor):
            with self.subTest(anchor=anchor):
                current = aligned((word(3, 320, 400),), start=250, end=1000)
                result = diagnose_word_anchor(
                    current,
                    committed_through_ms=300,
                    anchor=anchor,
                )
                self.assertEqual(result.status, "unavailable")
                decision = compare_word_hypotheses(
                    None,
                    current,
                    committed_through_ms=300,
                    anchor=anchor,
                    final=True,
                )
                self.assertEqual(decision.reason, "anchor_missing")
                self.assertEqual(decision.anchor_diagnostic.status, "unavailable")

    def test_origin_onset_exception_is_unchanged_and_drift_is_visible(self):
        anchor = (word(1, 400, 700), word(2, 700, 900))
        words = (word(1, 0, 690), word(2, 690, 900))
        result = self.diagnose(words, anchor=anchor, committed_through_ms=900)
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.start_deltas_ms, (-400, -10))

    def test_same_text_different_tokens_has_no_invented_word_slice(self):
        words = (self.anchor[0], replace(self.anchor[1], tokens=(7, 8)))
        result = self.diagnose(words)
        self.assertEqual((result.text_match_count, result.lexical_match_count), (1, 0))
        self.assertIsNone(result.word_start)
        self.assertEqual(result.end_deltas_ms, ())

    def test_diagnostics_are_immutable_plain_serializable_observations(self):
        result = self.diagnose(self.anchor)
        with self.assertRaises(FrozenInstanceError):
            result.status = "timing_mismatch"
        self.assertEqual(json.loads(json.dumps(asdict(result)))["status"], "matched")
        self.assertIsInstance(result.start_deltas_ms, tuple)

    def test_diagnostic_inputs_are_validated(self):
        cases = (
            ({"committed_through_ms": True}, TypeError),
            ({"committed_through_ms": 1001}, ValueError),
            ({"committed_through_ms": 100}, ValueError),
            ({"timestamp_tolerance_ms": -1}, ValueError),
            ({"timestamp_tolerance_ms": 1.5}, TypeError),
            ({"anchor": list(self.anchor)}, TypeError),
            ({"observation": "future"}, ValueError),
        )
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                self.diagnose(self.anchor, **kwargs)

    def test_timed_matches_preserve_legacy_rule_on_generated_ordered_words(self):
        # Independent expression of the pre-diagnostic matcher, including its
        # first-word onset exception. All inputs are deterministic and local.
        rng = random.Random(20260906)
        for _ in range(1000):
            words = tuple(
                word(rng.randrange(3), 100 + i * 100, 180 + i * 100)
                for i in range(rng.randrange(4, 16))
            )
            count = rng.randrange(1, 5)
            start = rng.randrange(len(words) - count + 1)
            offset = rng.choice((-80, -20, 0, 20, 100, 300))
            anchor = tuple(
                word(
                    item.tokens[0],
                    item.span.start_ms + offset,
                    item.span.end_ms + offset,
                )
                for item in words[start : start + count]
            )
            watermark = anchor[-1].span.end_ms
            tolerance = rng.choice((0, 20, 100, 200))
            expected = []
            for index in range(len(words) - len(anchor) + 1):
                found = words[index : index + len(anchor)]
                if found[-1].span.start_ms >= watermark:
                    continue
                matches = True
                for position, (old, new) in enumerate(zip(anchor, found)):
                    identity = old.text == new.text and old.tokens == new.tokens
                    end_matches = abs(old.span.end_ms - new.span.end_ms) <= tolerance
                    start_matches = (
                        abs(old.span.start_ms - new.span.start_ms) <= tolerance
                    )
                    onset = (
                        index == position == 0
                        and len(anchor) >= 2
                        and new.span.start_ms == 100
                        and new.span.start_ms < old.span.start_ms
                    )
                    matches &= identity and end_matches and (start_matches or onset)
                if matches:
                    expected.append(index + len(anchor))
            current = aligned(words, start=100, end=4000)
            # Partly cropped anchors are not supplied to the old matcher either.
            if anchor[0].span.start_ms < 100:
                continue
            result = diagnose_word_anchor(
                current,
                anchor=anchor,
                committed_through_ms=watermark,
                timestamp_tolerance_ms=tolerance,
            )
            self.assertEqual(result.timed_match_count, len(expected))
            if len(expected) == 1:
                self.assertEqual(result.status, "matched")
                self.assertEqual(result.word_end, expected[0])


if __name__ == "__main__":
    unittest.main()
