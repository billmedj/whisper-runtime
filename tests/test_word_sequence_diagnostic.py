"""Whole-sequence observations are not semantic or publication authority."""

import unittest
from dataclasses import FrozenInstanceError, fields

from test_word_policy import aligned, word

from whisper_runtime import AudioSpan
from whisper_runtime.adapters import (
    WordSequenceDiagnostic,
    diagnose_word_sequence,
)
from whisper_runtime.adapters.native_result import NativeTimestampSegment
from whisper_runtime.adapters.word_policy import compare_word_hypotheses


class WordSequenceDiagnosticTests(unittest.TestCase):
    def assert_no_timing(self, diagnostic: WordSequenceDiagnostic) -> None:
        self.assertIsNone(diagnostic.max_start_delta_ms)
        self.assertIsNone(diagnostic.max_end_delta_ms)

    def test_exact_units_and_tokens_have_zero_observed_deltas(self) -> None:
        words = (word(1, 100, 200, " She"), word(2, 250, 400, " reads."))
        self.assertEqual(
            diagnose_word_sequence(words, words),
            WordSequenceDiagnostic(2, 2, "exact_units", True, 0, 0),
        )

    def test_casefold_equal_units_report_tokens_and_sixty_ms_separately(self) -> None:
        left = (word(1, 100, 200, " She"), word(2, 300, 400, " reads"))
        right = (word(3, 160, 260, " she"), word(2, 300, 400, " reads"))
        self.assertEqual(
            diagnose_word_sequence(left, right),
            WordSequenceDiagnostic(2, 2, "casefold_equal_units", False, 60, 60),
        )

    def test_maxima_use_absolute_deltas_and_all_corresponding_units(self) -> None:
        left = (word(1, 100, 180, " a"), word(2, 300, 400, " b"))
        right = (word(1, 40, 170, " a"), word(2, 330, 480, " b"))
        expected = WordSequenceDiagnostic(2, 2, "exact_units", True, 60, 80)
        self.assertEqual(diagnose_word_sequence(left, right), expected)
        self.assertEqual(diagnose_word_sequence(right, left), expected)

    def test_exact_text_does_not_imply_equal_tokens(self) -> None:
        left = (word(1, 100, 200, " word"),)
        right = (word(2, 100, 200, " word"),)
        self.assertEqual(
            diagnose_word_sequence(left, right),
            WordSequenceDiagnostic(1, 1, "exact_units", False, 0, 0),
        )

    def test_token_equality_preserves_unit_boundaries(self) -> None:
        left = (
            NativeTimestampSegment(AudioSpan(100, 200), " a", (1, 2)),
            NativeTimestampSegment(AudioSpan(300, 400), " b", (3,)),
        )
        right = (
            NativeTimestampSegment(AudioSpan(100, 200), " a", (1,)),
            NativeTimestampSegment(AudioSpan(300, 400), " b", (2, 3)),
        )
        self.assertEqual(
            diagnose_word_sequence(left, right),
            WordSequenceDiagnostic(2, 2, "exact_units", False, 0, 0),
        )

    def test_equal_counts_and_tokens_do_not_align_different_lexical_units(self) -> None:
        left = (word(1, 100, 200, " cats"), word(2, 300, 400, " sit"))
        right = (word(1, 100, 200, " dogs"), word(2, 300, 400, " sit"))
        diagnostic = diagnose_word_sequence(left, right)
        self.assertEqual(diagnostic.text_relation, "different_units")
        self.assertTrue(diagnostic.unit_tokens_equal)
        self.assert_no_timing(diagnostic)

    def test_omission_does_not_report_timing_for_an_exact_prefix(self) -> None:
        complete = (word(1, 100, 200, " a"), word(2, 300, 400, " b"))
        for left, right in ((complete, complete[:1]), (complete[:1], complete)):
            with self.subTest(left_count=len(left), right_count=len(right)):
                diagnostic = diagnose_word_sequence(left, right)
                self.assertEqual(diagnostic.left_unit_count, len(left))
                self.assertEqual(diagnostic.right_unit_count, len(right))
                self.assertEqual(diagnostic.text_relation, "different_units")
                self.assertFalse(diagnostic.unit_tokens_equal)
                self.assert_no_timing(diagnostic)

    def test_empty_sequences_have_no_timing_witnesses(self) -> None:
        self.assertEqual(
            diagnose_word_sequence((), ()),
            WordSequenceDiagnostic(0, 0, "exact_units", True, None, None),
        )
        nonempty = (word(1, 100, 200),)
        for left, right in (((), nonempty), (nonempty, ())):
            with self.subTest(left_count=len(left)):
                diagnostic = diagnose_word_sequence(left, right)
                self.assertEqual(diagnostic.text_relation, "different_units")
                self.assertFalse(diagnostic.unit_tokens_equal)
                self.assert_no_timing(diagnostic)

    def test_punctuation_and_whitespace_are_not_removed_or_normalized(self) -> None:
        for left_text, right_text in (
            (" word", " word."),
            (" word", "word"),
            (" word ", " word"),
            (" two words", " two  words"),
            (" two\twords", " two words"),
            (" word,", " word;"),
        ):
            with self.subTest(left_text=left_text, right_text=right_text):
                diagnostic = diagnose_word_sequence(
                    (word(1, 100, 200, left_text),),
                    (word(1, 100, 200, right_text),),
                )
                self.assertEqual(diagnostic.text_relation, "different_units")
                self.assert_no_timing(diagnostic)

    def test_equal_joined_text_is_not_resegmented(self) -> None:
        left = (word(1, 0, 100, " ab"), word(2, 100, 200, "c"))
        right = (word(1, 0, 100, " a"), word(2, 100, 200, "bc"))
        self.assertEqual("".join(w.text for w in left), "".join(w.text for w in right))
        diagnostic = diagnose_word_sequence(left, right)
        self.assertEqual(diagnostic.text_relation, "different_units")
        self.assert_no_timing(diagnostic)

    def test_repeated_units_keep_positional_deltas_not_a_selected_occurrence(
        self,
    ) -> None:
        left = (word(1, 0, 100, " go"), word(1, 200, 300, " go"))
        right = (word(1, 200, 300, " go"), word(1, 400, 500, " go"))
        self.assertEqual(
            diagnose_word_sequence(left, right),
            WordSequenceDiagnostic(2, 2, "exact_units", True, 200, 200),
        )

    def test_repeated_shared_subsequence_does_not_hide_a_lexical_disagreement(
        self,
    ) -> None:
        left = (
            word(1, 0, 100, " go"),
            word(1, 100, 200, " go"),
            word(2, 200, 300, " stop"),
        )
        right = (
            word(1, 0, 100, " go"),
            word(2, 100, 200, " stop"),
            word(1, 200, 300, " go"),
        )
        diagnostic = diagnose_word_sequence(left, right)
        self.assertEqual(diagnostic.text_relation, "different_units")
        self.assert_no_timing(diagnostic)

    def test_casefold_collisions_do_not_authorize_strict_publication(self) -> None:
        for left_text, right_text in ((" US", " us"), (" ß", " ss")):
            with self.subTest(left_text=left_text, right_text=right_text):
                left = (word(1, 100, 200, left_text),)
                right = (word(1, 100, 200, right_text),)
                diagnostic = diagnose_word_sequence(left, right)
                self.assertEqual(
                    diagnostic,
                    WordSequenceDiagnostic(1, 1, "casefold_equal_units", True, 0, 0),
                )
                decision = compare_word_hypotheses(
                    aligned(left, end=500),
                    aligned(right, end=800),
                    committed_through_ms=0,
                    holdback_ms=100,
                )
                self.assertEqual(decision.reason, "unstable")
                self.assertIsNone(decision.publication)

    def test_diagnostic_and_inputs_are_immutable_without_publication_fields(
        self,
    ) -> None:
        left = (word(1, 100, 200, " She"),)
        right = (word(2, 160, 260, " she"),)
        snapshot = (left, right, left[0].span, right[0].span)
        diagnostic = diagnose_word_sequence(left, right)
        self.assertEqual((left, right, left[0].span, right[0].span), snapshot)
        self.assertEqual(left[0].text, " She")
        self.assertEqual(right[0].tokens, (2,))
        with self.assertRaises(FrozenInstanceError):
            diagnostic.text_relation = "exact_units"
        with self.assertRaises(FrozenInstanceError):
            left[0].text = " she"
        self.assertEqual(
            {field.name for field in fields(diagnostic)},
            {
                "left_unit_count",
                "right_unit_count",
                "text_relation",
                "unit_tokens_equal",
                "max_start_delta_ms",
                "max_end_delta_ms",
            },
        )

    def test_existing_validation_rejects_invalid_units_on_either_side(self) -> None:
        good = (word(1, 100, 200),)
        for invalid, error in (
            (None, TypeError),
            ("word", TypeError),
            ((object(),), TypeError),
            ((word(1, 200, 300), word(2, 100, 150)), ValueError),
            ((word(1, 100, 250), word(2, 200, 300)), ValueError),
            ((word(1, 100, 200, ""),), ValueError),
            ((word(1, 100, 200, " \t"),), ValueError),
            ((NativeTimestampSegment(AudioSpan(100, 200), " word", ()),), ValueError),
        ):
            for left, right in ((invalid, good), (good, invalid)):
                with self.subTest(left=left, right=right), self.assertRaises(error):
                    diagnose_word_sequence(left, right)

    def test_diagnosis_does_not_change_agreement_decisions(self) -> None:
        left = (word(1, 100, 200, " She"),)
        previous = aligned(left, end=500)
        for current_word, expected_reason in (
            (word(1, 100, 200, " She"), "candidate"),
            (word(1, 100, 200, " she"), "unstable"),
            (word(2, 100, 200, " She"), "unstable"),
            (word(1, 160, 260, " She"), "unstable"),
            (word(1, 100, 200, " He"), "unstable"),
        ):
            with self.subTest(current_word=current_word):
                current = aligned((current_word,), end=800)
                arguments = {
                    "committed_through_ms": 0,
                    "holdback_ms": 100,
                    "timestamp_tolerance_ms": 20,
                }
                before = compare_word_hypotheses(previous, current, **arguments)
                diagnose_word_sequence(previous.words, current.words)
                after = compare_word_hypotheses(previous, current, **arguments)
                self.assertEqual(before.reason, expected_reason)
                self.assertEqual(after, before)
                if before.publication is not None:
                    self.assertIs(before.publication.alignment, current)
                    self.assertIs(after.publication.alignment, current)


if __name__ == "__main__":
    unittest.main()
