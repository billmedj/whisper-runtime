"""Pure aligned-publication contracts; no acoustic accuracy or silence claims."""

import unittest
from dataclasses import FrozenInstanceError, replace

from whisper_runtime import AudioSpan, WindowResult
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.word_policy import (
    AlignedPublication,
    NativeWordAlignment,
    WordAgreementDecision,
    compare_word_hypotheses,
    select_word_publication,
)


def word(token: int, start: int, end: int, text: str | None = None):
    return NativeTimestampSegment(
        AudioSpan(start, end), f" word-{token}" if text is None else text, (token,)
    )


def aligned(words, *, start: int = 0, end: int = 1_000, language: str = "en"):
    text = "".join(item.text for item in words).strip()
    tokens = tuple(token for item in words for token in item.tokens)
    raw_segments = (
        (NativeTimestampSegment(AudioSpan(start, end), text, tokens),) if words else ()
    )
    native = NativeWindowResult(
        window_id=f"aligned-{start}-{end}",
        text=text,
        start_ms=start,
        end_ms=end,
        metadata=NativeDecodeMetadata(
            language=language,
            tokens=tokens,
            segments=raw_segments,
            timestamps_complete=bool(words),
        ),
    )
    return NativeWordAlignment(native, words)


class NativeWordAlignmentTests(unittest.TestCase):
    def test_immutable_copies_preserve_native_provenance(self) -> None:
        words = [word(1, 100, 200), word(2, 300, 400)]
        alignment = aligned(words)
        words.clear()
        self.assertEqual(len(alignment.words), 2)
        self.assertIs(type(alignment.words), tuple)
        self.assertEqual(len(alignment.native.metadata.segments), 1)
        self.assertEqual(
            alignment.native.metadata.segments[0].span, AudioSpan(0, 1_000)
        )
        with self.assertRaises(FrozenInstanceError):
            alignment.words = ()

    def test_rejects_selected_or_partial_native_result(self) -> None:
        alignment = aligned((word(1, 0, 100),))
        for native in (
            replace(alignment.native, publication_segment_indices=(0,)),
            replace(alignment.native, start_ms=100, analysis_span=AudioSpan(0, 1_000)),
        ):
            with self.subTest(native=native), self.assertRaises(ValueError):
                NativeWordAlignment(native, alignment.words)

    def test_rejects_overlapping_out_of_bounds_or_unaccounted_words(self) -> None:
        native = aligned((word(1, 0, 100),)).native
        for words in (
            (word(1, 0, 150), word(2, 100, 200)),
            (word(1, 900, 1_100),),
            (word(1, 0, 100, "different"),),
            (NativeTimestampSegment(AudioSpan(0, 100), " word-1", ()),),
            (word(1, 0, 100, " "),),
            (),
        ):
            with self.subTest(words=words), self.assertRaises(ValueError):
                NativeWordAlignment(native, words)
        with self.assertRaises(TypeError):
            NativeWordAlignment(native, (object(),))

    def test_empty_alignment_requires_empty_native_text(self) -> None:
        alignment = aligned(())
        self.assertEqual(alignment.native.text, "")
        self.assertEqual(alignment.words, ())
        with self.assertRaises(ValueError):
            NativeWordAlignment(replace(alignment.native, text=" "), ())


class AlignedPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.alignment = aligned((word(1, 100, 200), word(2, 300, 400)))

    def test_whole_word_selection_retains_exact_text_and_analysis(self) -> None:
        publication = select_word_publication(self.alignment, 1, 2, AudioSpan(200, 400))
        self.assertIsInstance(publication, WindowResult)
        self.assertIs(publication.alignment, self.alignment)
        self.assertIs(publication.alignment.native, self.alignment.native)
        self.assertEqual(publication.text, "word-2")
        self.assertEqual(publication.analyzed_span, AudioSpan(0, 1_000))
        self.assertEqual((publication.word_start, publication.word_end), (1, 2))
        with self.assertRaises(FrozenInstanceError):
            publication.word_end = 1

    def test_final_suffix_can_end_before_processed_eof(self) -> None:
        publication = select_word_publication(
            self.alignment, 1, 2, AudioSpan(200, 1_000), final=True
        )
        self.assertEqual(publication.text, "word-2")
        self.assertEqual(publication.end_ms, 1_000)
        self.assertEqual(publication.alignment.words[-1].span.end_ms, 400)
        self.assertTrue(publication.final)

    def test_empty_suffix_is_only_an_explicit_final_publication(self) -> None:
        publication = select_word_publication(
            self.alignment, 2, 2, AudioSpan(400, 1_000), final=True
        )
        self.assertEqual(publication.text, "")
        for start, end, coverage, final in (
            (2, 2, AudioSpan(400, 1_000), False),
            (1, 1, AudioSpan(400, 1_000), True),
            (0, 1, AudioSpan(0, 1_000), True),
        ):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                select_word_publication(self.alignment, start, end, coverage, final)

    def test_factory_and_exported_constructor_reject_inconsistent_values(self) -> None:
        for start, end, coverage, error in (
            (True, 1, AudioSpan(0, 200), TypeError),
            (-1, 1, AudioSpan(0, 200), ValueError),
            (0, 3, AudioSpan(0, 400), ValueError),
            (1, 0, AudioSpan(0, 200), ValueError),
            (0, 1, AudioSpan(150, 200), ValueError),
            (0, 1, AudioSpan(0, 250), ValueError),
            (0, 2, AudioSpan(0, 1_001), ValueError),
        ):
            with self.subTest(start=start, end=end, coverage=coverage):
                with self.assertRaises(error):
                    select_word_publication(self.alignment, start, end, coverage)
        publication = select_word_publication(self.alignment, 0, 1, AudioSpan(0, 200))
        for changes in (
            {"text": "invented"},
            {"window_id": "different"},
            {"analysis_span": AudioSpan(0, 2_000)},
            {"final": 1},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises((TypeError, ValueError)),
            ):
                replace(publication, **changes)

    def test_boundary_tolerance_is_explicit_bounded_and_start_only(self) -> None:
        with self.assertRaises(ValueError):
            select_word_publication(self.alignment, 0, 1, AudioSpan(150, 200))
        publication = select_word_publication(
            self.alignment,
            0,
            1,
            AudioSpan(150, 200),
            boundary_tolerance_ms=50,
        )
        self.assertEqual(publication.boundary_tolerance_ms, 50)
        self.assertEqual(publication.alignment.words[0].span, AudioSpan(100, 200))
        for tolerance, error in (
            (49, ValueError),
            (-1, ValueError),
            (True, TypeError),
            (1.5, TypeError),
        ):
            with self.subTest(tolerance=tolerance), self.assertRaises(error):
                replace(publication, boundary_tolerance_ms=tolerance)
        with self.assertRaises(ValueError):
            select_word_publication(
                self.alignment,
                0,
                1,
                AudioSpan(150, 190),
                boundary_tolerance_ms=100,
            )


class WordAgreementTests(unittest.TestCase):
    def decide(self, previous, current, **changes):
        arguments = {
            "committed_through_ms": 0,
            "holdback_ms": 100,
            "timestamp_tolerance_ms": 20,
        }
        arguments.update(changes)
        return compare_word_hypotheses(previous, current, **arguments)

    def test_native_segment_straddle_does_not_prevent_new_whole_words(self) -> None:
        anchor = (word(1, 0, 100),)
        before = aligned(anchor + (word(2, 150, 250),), end=500)
        current = aligned(before.words + (word(3, 300, 400),), end=700)
        decision = self.decide(before, current, committed_through_ms=100, anchor=anchor)
        self.assertEqual(decision.reason, "candidate")
        self.assertEqual(decision.publication.text, "word-2")
        self.assertEqual(
            (decision.publication.start_ms, decision.publication.end_ms), (100, 250)
        )
        self.assertIs(decision.publication.alignment, current)
        self.assertEqual(current.native.metadata.segments[0].span, AudioSpan(0, 700))

    def test_natural_pauses_are_explicitly_processed_not_inferred_silent(self) -> None:
        words = (word(1, 100, 200), word(2, 400, 500))
        decision = self.decide(aligned(words, end=600), aligned(words, end=900))
        self.assertEqual(decision.reason, "candidate")
        self.assertEqual(decision.publication.text, "word-1 word-2")
        self.assertEqual(
            (decision.publication.start_ms, decision.publication.end_ms), (0, 500)
        )
        self.assertFalse(hasattr(decision, "silence"))
        self.assertFalse(hasattr(decision.publication, "silence"))

    def test_nonfinal_requires_two_same_origin_strictly_growing_observations(
        self,
    ) -> None:
        words = (word(1, 100, 200),)
        current = aligned(words, end=900)
        for before, reason in (
            (None, "incomplete"),
            (current, "unstable"),
            (aligned(words, start=20, end=600), "unstable"),
            (aligned(words, end=600, language="fr"), "unstable"),
        ):
            with self.subTest(reason=reason):
                decision = self.decide(before, current)
                self.assertEqual(decision.reason, reason)
                self.assertIsNone(decision.publication)

    def test_exact_text_tokens_timing_and_holdback_limit_the_prefix(self) -> None:
        before = aligned((word(1, 0, 100), word(2, 100, 200)), end=400)
        for changed in (
            word(3, 100, 200),
            word(2, 100, 200, "word-2"),
            word(2, 140, 240),
        ):
            with self.subTest(changed=changed):
                current = aligned((before.words[0], changed), end=600)
                decision = self.decide(before, current)
                self.assertEqual(decision.publication.text, "word-1")
        current = aligned(before.words, end=600)
        decision = self.decide(before, current, holdback_ms=450)
        self.assertEqual(decision.publication.text, "word-1")

    def test_empty_hypotheses_do_not_authorize_nonfinal_progress(self) -> None:
        decision = self.decide(aligned((), end=400), aligned((), end=600))
        self.assertEqual(decision.reason, "incomplete")
        self.assertIsNone(decision.publication)
        self.assertEqual(decision.next_anchor, ())

    def test_later_unique_repetition_is_not_the_committed_occurrence(self) -> None:
        anchor = (word(1, 0, 100),)
        for repeat_start in (100, 200, 300):
            with self.subTest(repeat_start=repeat_start):
                words = (word(1, repeat_start, repeat_start + 100), word(2, 450, 500))
                decision = self.decide(
                    aligned(words, end=600),
                    aligned(words, end=900),
                    committed_through_ms=100,
                    anchor=anchor,
                    timestamp_tolerance_ms=1_000,
                )
                self.assertEqual(decision.reason, "anchor_missing")
                self.assertIsNone(decision.publication)
                self.assertIs(decision.next_anchor[0], anchor[0])

    def test_timed_original_anchor_can_coexist_with_a_later_repetition(self) -> None:
        anchor = (word(1, 0, 100),)
        words = anchor + (word(1, 200, 300), word(2, 350, 450))
        decision = self.decide(
            aligned(words, end=600),
            aligned(words, end=900),
            committed_through_ms=100,
            anchor=anchor,
            timestamp_tolerance_ms=1_000,
        )
        self.assertEqual(decision.reason, "candidate")
        self.assertEqual(decision.publication.text, "word-1 word-2")
        self.assertEqual(decision.publication.word_start, 1)

    def test_multiple_temporally_admissible_anchors_remain_ambiguous(self) -> None:
        anchor = (word(1, 0, 100),)
        words = (word(1, 0, 60), word(1, 80, 140), word(2, 250, 350))
        decision = self.decide(
            aligned(words, end=500),
            aligned(words, end=800),
            committed_through_ms=200,
            anchor=anchor,
            timestamp_tolerance_ms=100,
        )
        self.assertEqual(decision.reason, "anchor_ambiguous")
        self.assertIsNone(decision.publication)

    def test_old_anchor_is_unchanged_and_next_anchor_uses_only_new_words(self) -> None:
        anchor = (word(1, 0, 100),)
        before = aligned((word(1, 10, 110), word(2, 140, 240)), end=500)
        current = aligned((word(1, 20, 120), word(2, 150, 250)), end=800)
        decision = self.decide(before, current, committed_through_ms=100, anchor=anchor)
        self.assertEqual(decision.reason, "candidate")
        self.assertEqual(anchor[0].span, AudioSpan(0, 100))
        self.assertEqual(decision.next_anchor, (current.words[1],))
        self.assertIs(decision.next_anchor[0], current.words[1])

    def test_only_complete_anchor_spans_within_retained_input_are_used(self) -> None:
        anchor = tuple(
            word(index, index * 100, (index + 1) * 100) for index in range(4)
        )
        words = anchor[2:] + (word(4, 420, 500),)
        decision = self.decide(
            aligned(words, start=200, end=600),
            aligned(words, start=200, end=800),
            committed_through_ms=400,
            anchor=anchor,
        )
        self.assertEqual(decision.reason, "candidate")
        self.assertEqual(decision.next_anchor, (words[-1],))
        for supplied_anchor in ((), anchor[:2]):
            with self.subTest(anchor=supplied_anchor):
                missing = self.decide(
                    aligned(words, start=200, end=600),
                    aligned(words, start=200, end=800),
                    committed_through_ms=400,
                    anchor=supplied_anchor,
                )
                self.assertEqual(missing.reason, "anchor_missing")

    def test_t4_v4_traces_4_5_6_drop_cropped_ask_but_keep_timed_not_what(self) -> None:
        # Exact frozen words from trace 4's last publication. R=3700 cuts
        # through "ask", which is absent at the start of both later analyses.
        anchor = (
            word(3399, 1760, 2340, " Americans"),
            word(1265, 2340, 3860, " ask"),
            word(407, 3860, 4620, " not"),
            word(644, 4620, 5700, " what"),
        )
        trace5 = (
            (407, 3700, 4480, " not"),
            (644, 4480, 5700, " what"),
            (534, 5700, 6020, " your"),
            (1499, 6020, 6360, " country"),
            (460, 6360, 6780, " can"),
            (466, 6780, 7020, " do"),
            (329, 7020, 7240, " for"),
            (345, 7240, 8240, " you"),
            (1265, 8240, 8660, " ask"),
            (644, 8660, 8960, " what"),
            (345, 8960, 9280, " you"),
            (460, 9280, 9500, " can"),
            (466, 9500, 9740, " do"),
            (329, 9740, 9880, " for"),
            (534, 9880, 9980, " your"),
        )
        trace6 = (
            (407, 3700, 4460, " not"),
            (644, 4460, 5700, " what"),
            (534, 5700, 6120, " your"),
            (1499, 6120, 6360, " country"),
            (460, 6360, 6780, " can"),
            (466, 6780, 7020, " do"),
            (329, 7020, 7240, " for"),
            (345, 7240, 8240, " you"),
            (1265, 8240, 8660, " ask"),
            (644, 8660, 8980, " what"),
            (345, 8980, 9280, " you"),
            (460, 9280, 9500, " can"),
            (466, 9500, 9740, " do"),
            (329, 9740, 9920, " for"),
            (534, 9920, 10240, " your"),
            (1499, 10240, 10720, " country"),
            (13, 10720, 11600, "."),
        )
        before = aligned(tuple(word(*item) for item in trace5), start=3700, end=10000)
        current = aligned(tuple(word(*item) for item in trace6), start=3700, end=12000)
        decision = self.decide(
            before,
            current,
            committed_through_ms=5700,
            anchor=anchor,
            holdback_ms=2000,
            timestamp_tolerance_ms=200,
        )
        self.assertEqual(decision.reason, "candidate")
        self.assertEqual(decision.publication.word_start, 2)
        self.assertEqual(decision.publication.word_end, 14)
        self.assertEqual(decision.publication.start_ms, 5700)
        self.assertEqual(decision.publication.end_ms, 9920)
        self.assertEqual(
            decision.publication.text,
            "your country can do for you ask what you can do for",
        )
        self.assertEqual(decision.next_anchor, current.words[10:14])
        self.assertEqual(anchor[1].span, AudioSpan(2340, 3860))
        self.assertIs(decision.publication.alignment, current)

    def test_cropped_anchor_with_no_complete_word_cannot_match_later_repetition(
        self,
    ) -> None:
        anchor = (word(1265, 2340, 3860, " ask"),)
        repeated = (word(1265, 8240, 8660, " ask"), word(644, 8660, 8960, " what"))
        decision = self.decide(
            aligned(repeated, start=3700, end=10000),
            aligned(repeated, start=3700, end=12000),
            committed_through_ms=3860,
            anchor=anchor,
            holdback_ms=2000,
            timestamp_tolerance_ms=10000,
        )
        self.assertEqual(decision.reason, "anchor_missing")
        self.assertIsNone(decision.publication)
        self.assertEqual(decision.next_anchor, anchor)

    def test_start_drift_larger_than_tolerance_remains_unresolved(self) -> None:
        anchor = (word(1, 0, 80),)
        words = (word(1, 0, 70), word(2, 75, 200))
        current = aligned(words, end=800)
        for final in (False, True):
            decision = self.decide(
                aligned(words, end=500),
                current,
                committed_through_ms=100,
                anchor=anchor,
                final=final,
            )
            self.assertEqual(decision.reason, "unstable")
            self.assertIsNone(decision.publication)
            self.assertEqual(current.words[1].span.start_ms, 75)

    def test_bounded_start_drift_requires_a_timed_anchor_and_keeps_original_times(
        self,
    ) -> None:
        anchor = (word(1, 17_500, 18_000, " do"),)
        words = (word(1, 17_500, 17_920, " do"), word(2, 17_920, 18_200, " for"))
        before = aligned(words, start=17_000, end=19_000)
        current = aligned(words, start=17_000, end=20_000)
        for final in (False, True):
            with self.subTest(final=final):
                decision = self.decide(
                    before,
                    current,
                    committed_through_ms=18_000,
                    anchor=anchor,
                    timestamp_tolerance_ms=200,
                    final=final,
                )
                self.assertEqual(decision.reason, "eof" if final else "candidate")
                publication = decision.publication
                self.assertEqual(publication.start_ms, 18_000)
                self.assertEqual(publication.end_ms, 20_000 if final else 18_200)
                self.assertEqual(publication.boundary_tolerance_ms, 200)
                self.assertEqual(publication.text, "for")
                self.assertIs(publication.alignment.words[1], current.words[1])
                self.assertEqual(current.words[1].span, AudioSpan(17_920, 18_200))
                self.assertEqual(anchor[0].span, AudioSpan(17_500, 18_000))
                self.assertEqual(decision.next_anchor, (current.words[1],))
        missing = self.decide(
            before,
            current,
            committed_through_ms=18_000,
            timestamp_tolerance_ms=200,
        )
        self.assertEqual(missing.reason, "anchor_missing")
        self.assertIsNone(missing.publication)
        first = self.decide(
            before,
            current,
            committed_through_ms=18_000,
            anchor=anchor,
            timestamp_tolerance_ms=200,
        )
        later = aligned(
            words + (word(3, 18_250, 18_500, " you"),), start=17_000, end=21_000
        )
        newest = aligned(later.words, start=17_000, end=22_000)
        continued = self.decide(
            later,
            newest,
            committed_through_ms=18_200,
            anchor=first.next_anchor,
            timestamp_tolerance_ms=200,
        )
        self.assertEqual(continued.publication.text, "you")
        self.assertEqual(continued.next_anchor, (newest.words[-1],))

    def test_unanchored_new_block_retains_zero_boundary_tolerance(self) -> None:
        words = (word(1, 100, 200),)
        decision = self.decide(
            aligned(words, start=100, end=500),
            aligned(words, start=100, end=800),
            committed_through_ms=100,
            timestamp_tolerance_ms=200,
        )
        self.assertEqual(decision.publication.boundary_tolerance_ms, 0)

    def test_next_anchor_is_at_most_four_actual_published_words(self) -> None:
        words = tuple(word(index, index * 50, (index + 1) * 50) for index in range(10))
        current = aligned(words, end=900)
        decision = self.decide(aligned(words, end=600), current)
        self.assertEqual(decision.reason, "candidate")
        self.assertEqual(decision.publication.word_end, 10)
        self.assertEqual(decision.next_anchor, words[-4:])
        for left, right in zip(decision.next_anchor, words[-4:]):
            self.assertIs(left, right)

    def test_fresh_origin_does_not_search_for_prior_block_text(self) -> None:
        anchor = (word(1, 0, 100),)
        words = (word(1, 150, 250),)
        decision = self.decide(
            aligned(words, start=100, end=500),
            aligned(words, start=100, end=800),
            committed_through_ms=100,
            anchor=anchor,
        )
        self.assertEqual(decision.reason, "candidate")
        self.assertEqual(decision.publication.word_start, 0)
        self.assertEqual(decision.next_anchor, words)

    def test_eof_uses_explicit_completion_authority_not_an_agreement_claim(
        self,
    ) -> None:
        anchor = (word(1, 0, 100),)
        current = aligned(anchor + (word(2, 150, 250),), end=1_000)
        decision = self.decide(
            None,
            current,
            committed_through_ms=100,
            anchor=anchor,
            final=True,
            holdback_ms=2_000,
        )
        self.assertEqual(decision.reason, "eof")
        self.assertEqual(decision.publication.text, "word-2")
        self.assertEqual(
            (decision.publication.start_ms, decision.publication.end_ms), (100, 1_000)
        )
        self.assertTrue(decision.publication.final)

    def test_eof_empty_suffix_and_empty_fresh_window_are_explicit(self) -> None:
        anchor = (word(1, 0, 100),)
        decision = self.decide(
            None,
            aligned(anchor, end=1_000),
            committed_through_ms=100,
            anchor=anchor,
            final=True,
        )
        self.assertEqual(decision.reason, "eof")
        self.assertEqual(decision.publication.text, "")
        self.assertEqual(
            (decision.publication.word_start, decision.publication.word_end), (1, 1)
        )
        self.assertIs(decision.next_anchor[0], anchor[0])
        fresh = self.decide(
            None,
            aligned((), start=200, end=1_000),
            committed_through_ms=200,
            final=True,
        )
        self.assertEqual(fresh.reason, "eof")
        self.assertEqual(fresh.publication.text, "")
        self.assertEqual(fresh.publication.end_ms, 1_000)
        missing = self.decide(
            None,
            aligned((), end=1_000),
            committed_through_ms=100,
            anchor=anchor,
            final=True,
        )
        self.assertEqual(missing.reason, "anchor_missing")

    def test_input_and_decision_validation_remain_strict(self) -> None:
        words = (word(1, 0, 100),)
        before, current = aligned(words, end=500), aligned(words, end=800)
        for changes, error in (
            ({"committed_through_ms": True}, TypeError),
            ({"holdback_ms": -1}, ValueError),
            ({"timestamp_tolerance_ms": 1.5}, TypeError),
            ({"final": 1}, TypeError),
            ({"anchor": list(words)}, TypeError),
            ({"anchor": words * 5}, ValueError),
            ({"anchor": words}, ValueError),
        ):
            with self.subTest(changes=changes), self.assertRaises(error):
                self.decide(before, current, **changes)
        with self.assertRaises(TypeError):
            WordAgreementDecision("candidate")
        publication = select_word_publication(current, 0, 1, AudioSpan(0, 100))
        with self.assertRaises(ValueError):
            WordAgreementDecision("unstable", publication)
        with self.assertRaises(ValueError):
            WordAgreementDecision("eof", publication)
        decision = WordAgreementDecision("candidate", publication, words)
        with self.assertRaises(FrozenInstanceError):
            decision.next_anchor = ()
        self.assertIsInstance(publication, AlignedPublication)


if __name__ == "__main__":
    unittest.main()
