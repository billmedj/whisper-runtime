"""Token finality is separate from segment timing and audio coverage.

The caller owns unchanged PCM/model/options identity and may supply an empty
anchor only for a block in which it has not already published text.
"""

import unittest
from dataclasses import FrozenInstanceError, asdict, replace

from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.stream_policy import (
    TextAgreementReason,
    TextPrefixDecision,
    compare_hypotheses,
    resolve_text_prefix,
)
from whisper_runtime.state import AudioSpan, WindowResult


def hypothesis(
    groups: tuple[tuple[int, ...], ...],
    *,
    end: int,
    start: int = 0,
    spans: tuple[tuple[int, int], ...] | None = None,
    language: str = "en",
    complete: bool = True,
    raw_tokens: tuple[int, ...] | None = None,
) -> NativeWindowResult:
    if spans is None:
        spans = tuple(
            (start + 20 * index, start + 20 * (index + 1))
            for index in range(len(groups))
        )
    if len(spans) != len(groups):
        raise ValueError("each scripted token group needs a source span")
    segments = tuple(
        NativeTimestampSegment(
            AudioSpan(left, right),
            "".join(f" token-{token}" for token in group),
            group,
        )
        for group, (left, right) in zip(groups, spans)
    )
    return NativeWindowResult(
        window_id=f"text-window-{start}-{end}",
        start_ms=start,
        end_ms=end,
        text="".join(segment.text for segment in segments).strip(),
        metadata=NativeDecodeMetadata(
            language=language,
            tokens=raw_tokens
            if raw_tokens is not None
            else tuple(token for group in groups for token in group),
            segments=segments,
            timestamps_complete=complete,
        ),
    )


class StreamTextPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = hypothesis(((1, 2, 3),), end=1_000)
        self.current = hypothesis(((1, 2, 3, 4),), end=2_000)

    def assert_candidate(
        self,
        decision: TextPrefixDecision,
        tokens: tuple[int, ...],
        *,
        start: int,
        end: int,
        anchor: tuple[int, ...],
    ) -> None:
        self.assertIsInstance(decision, TextPrefixDecision)
        self.assertIs(decision.reason, TextAgreementReason.CANDIDATE)
        self.assertEqual(decision.tokens, tokens)
        self.assertEqual(decision.current_token_start, start)
        self.assertEqual(decision.current_token_end, end)
        self.assertEqual(decision.next_anchor, anchor)

    def test_same_tokens_agree_despite_changed_native_segmentation(self) -> None:
        previous = hypothesis(((1, 2), (3, 4)), end=1_000)
        current = hypothesis(((1,), (2, 3), (4, 5)), end=2_000)
        self.assert_candidate(
            resolve_text_prefix(previous, current),
            (1, 2, 3, 4),
            start=0,
            end=4,
            anchor=(1, 2, 3, 4),
        )

    def test_stable_prefix_can_end_inside_a_native_segment(self) -> None:
        previous = hypothesis(((1, 2, 3),), end=1_000)
        current = hypothesis(((1, 2, 9, 10),), end=2_000)
        self.assert_candidate(
            resolve_text_prefix(previous, current),
            (1, 2),
            start=0,
            end=2,
            anchor=(1, 2),
        )
        # The strict whole-segment resolver must retain its existing contract.
        strict = compare_hypotheses(
            previous,
            current,
            committed_through_ms=0,
            holdback_ms=0,
            timestamp_tolerance_ms=200,
        )
        self.assertIsNone(strict.publication_span)

    def test_unique_committed_anchor_can_cross_and_end_inside_segments(self) -> None:
        previous = hypothesis(((90, 91, 1), (2, 3), (4, 5)), end=1_000)
        current = hypothesis(((80, 1, 2, 3, 4), (6,)), end=2_000)
        self.assert_candidate(
            resolve_text_prefix(previous, current, committed_anchor=(1, 2)),
            (3, 4),
            start=3,
            end=5,
            anchor=(1, 2, 3, 4),
        )

    def test_committed_anchor_prevents_republication_on_next_decision(self) -> None:
        first = resolve_text_prefix(self.previous, self.current)
        self.assertEqual(first.tokens, (1, 2, 3))
        later = hypothesis(((1, 2), (3, 4, 5)), end=3_000)
        second = resolve_text_prefix(
            self.current, later, committed_anchor=first.next_anchor
        )
        self.assert_candidate(second, (4,), start=3, end=4, anchor=(1, 2, 3, 4))
        self.assertEqual(first.tokens + second.tokens, (1, 2, 3, 4))

    def test_repeated_anchor_in_either_hypothesis_is_ambiguous(self) -> None:
        unique = (1, 2, 3, 4)
        repeated = (1, 2, 3, 1, 2, 4)
        for previous_tokens, current_tokens in (
            (unique, repeated),
            (repeated, unique),
            (repeated, repeated),
        ):
            with self.subTest(previous=previous_tokens, current=current_tokens):
                decision = resolve_text_prefix(
                    hypothesis((previous_tokens,), end=1_000),
                    hypothesis((current_tokens,), end=2_000),
                    committed_anchor=(1, 2),
                )
                self.assertIs(decision.reason, TextAgreementReason.ANCHOR_AMBIGUOUS)
                self.assertEqual(decision.tokens, ())
                self.assertEqual(decision.next_anchor, (1, 2))

    def test_overlapping_anchor_occurrences_are_also_ambiguous(self) -> None:
        decision = resolve_text_prefix(
            hypothesis(((1, 1, 2),), end=1_000),
            hypothesis(((1, 1, 1, 2),), end=2_000),
            committed_anchor=(1, 1),
        )
        self.assertIs(decision.reason, TextAgreementReason.ANCHOR_AMBIGUOUS)
        self.assertEqual(decision.tokens, ())

    def test_absent_or_partial_anchor_never_uses_a_fuzzy_match(self) -> None:
        for previous_tokens, current_tokens in (
            ((1, 2, 3), (1, 9, 3)),
            ((1, 9, 3), (1, 2, 3)),
            ((1, 2, 3), (1,)),
            ((1,), (1, 2, 3)),
        ):
            with self.subTest(previous=previous_tokens, current=current_tokens):
                decision = resolve_text_prefix(
                    hypothesis((previous_tokens,), end=1_000),
                    hypothesis((current_tokens,), end=2_000),
                    committed_anchor=(1, 2),
                )
                self.assertIs(decision.reason, TextAgreementReason.ANCHOR_MISSING)
                self.assertEqual(decision.tokens, ())
                self.assertEqual(decision.next_anchor, (1, 2))

    def test_lcp_stops_at_first_token_difference_without_skipping_a_repeated_tail(
        self,
    ) -> None:
        decision = resolve_text_prefix(
            hypothesis(((1, 2, 3, 4),), end=1_000),
            hypothesis(((1, 9, 3, 4),), end=2_000),
        )
        self.assert_candidate(decision, (1,), start=0, end=1, anchor=(1,))

    def test_matching_display_text_cannot_override_different_token_ids(self) -> None:
        previous = hypothesis(((1, 2),), end=1_000)
        current = hypothesis(((9, 8),), end=2_000)
        current = replace(current, text=previous.text)
        decision = resolve_text_prefix(previous, current)
        self.assertIs(decision.reason, TextAgreementReason.UNSTABLE)
        self.assertEqual(decision.tokens, ())

    def test_closed_segment_tokens_are_used_instead_of_raw_timestamp_or_tail_ids(
        self,
    ) -> None:
        previous = hypothesis(
            ((1, 2),),
            end=1_000,
            raw_tokens=(50_000, 1, 2, 50_050, 9, 10),
            complete=False,
        )
        current = hypothesis(
            ((1,), (2, 3)),
            end=2_000,
            raw_tokens=(50_000, 1, 50_020, 50_021, 2, 3, 50_070, 9, 10),
            complete=False,
        )
        self.assert_candidate(
            resolve_text_prefix(previous, current),
            (1, 2),
            start=0,
            end=2,
            anchor=(1, 2),
        )

    def test_empty_hypotheses_never_authorize_silence_or_audio_coverage(self) -> None:
        for no_speech_prob in (None, 0.0, 1.0):
            with self.subTest(no_speech_prob=no_speech_prob):
                previous = hypothesis((), end=1_000)
                current = hypothesis((), end=2_000)
                current = replace(
                    current,
                    metadata=replace(current.metadata, no_speech_prob=no_speech_prob),
                )
                decision = resolve_text_prefix(previous, current)
                self.assertIs(decision.reason, TextAgreementReason.INCOMPLETE)
                self.assertEqual(decision.tokens, ())
                self.assertEqual(decision.next_anchor, ())
                for field in (
                    "publication_span",
                    "audio_coverage",
                    "committed_through_ms",
                ):
                    self.assertFalse(hasattr(decision, field), field)

    def test_metadata_is_required_even_if_full_backend_text_and_raw_ids_agree(
        self,
    ) -> None:
        for previous, current in (
            (replace(self.previous, metadata=None), self.current),
            (self.previous, replace(self.current, metadata=None)),
            (None, self.current),
        ):
            with self.subTest(previous=previous, current=current):
                decision = resolve_text_prefix(previous, current)
                self.assertIs(decision.reason, TextAgreementReason.INCOMPLETE)
                self.assertEqual(decision.tokens, ())

    def test_only_same_origin_strictly_growing_analyses_can_confirm(self) -> None:
        for current in (
            hypothesis(((1, 2, 3, 4),), end=1_000),
            hypothesis(((1, 2, 3, 4),), end=800),
            hypothesis(((1, 2, 3, 4),), start=20, end=2_000),
        ):
            with self.subTest(analysis=current.analyzed_span):
                decision = resolve_text_prefix(self.previous, current)
                self.assertIs(decision.reason, TextAgreementReason.UNSTABLE)
                self.assertEqual(decision.tokens, ())

    def test_language_changes_do_not_confirm_text(self) -> None:
        decision = resolve_text_prefix(
            self.previous, hypothesis(((1, 2, 3, 4),), end=2_000, language="fr")
        )
        self.assertIs(decision.reason, TextAgreementReason.UNSTABLE)
        self.assertEqual(decision.tokens, ())

    def test_shifted_or_gapped_segment_times_do_not_choose_or_rewrite_audio_bounds(
        self,
    ) -> None:
        previous = hypothesis(((1, 2),), end=1_000, spans=((0, 400),))
        current = hypothesis(
            ((1,), (2, 3)), end=2_000, spans=((120, 160), (700, 1_200))
        )
        snapshots = asdict(previous), asdict(current)
        previous_metadata = previous.metadata
        current_metadata = current.metadata
        decision = resolve_text_prefix(previous, current)
        self.assert_candidate(decision, (1, 2), start=0, end=2, anchor=(1, 2))
        self.assertEqual((asdict(previous), asdict(current)), snapshots)
        self.assertIs(previous.metadata, previous_metadata)
        self.assertIs(current.metadata, current_metadata)
        self.assertIsNone(
            compare_hypotheses(
                previous,
                current,
                committed_through_ms=0,
                holdback_ms=0,
                timestamp_tolerance_ms=0,
            ).publication_span
        )
        self.assertFalse(hasattr(decision, "publication_span"))

    def test_anchor_is_bounded_without_truncating_the_candidate_prefix(self) -> None:
        tokens = tuple(range(100))
        decision = resolve_text_prefix(
            hypothesis((tokens,), end=1_000),
            hypothesis((tokens + (100,),), end=2_000),
        )
        self.assert_candidate(decision, tokens, start=0, end=100, anchor=tokens[-32:])
        self.assertEqual(len(decision.next_anchor), 32)

    def test_next_anchor_uses_only_the_published_suffix_not_earlier_context(
        self,
    ) -> None:
        decision = resolve_text_prefix(
            hypothesis(((90, 91, 1, 2, 3, 4, 5),), end=1_000),
            hypothesis(((80, 1, 2, 3, 4, 5, 6),), end=2_000),
            committed_anchor=(1, 2),
            max_anchor_tokens=4,
        )
        self.assert_candidate(decision, (3, 4, 5), start=3, end=6, anchor=(2, 3, 4, 5))

    def test_maximum_supported_anchor_remains_bounded_after_progress(self) -> None:
        anchor = tuple(range(448))
        previous = hypothesis((anchor + (999,),), end=1_000)
        current = hypothesis((anchor + (999, 1000),), end=2_000)
        decision = resolve_text_prefix(
            previous, current, committed_anchor=anchor, max_anchor_tokens=448
        )
        self.assert_candidate(
            decision, (999,), start=448, end=449, anchor=anchor[1:] + (999,)
        )

    def test_input_anchor_must_be_an_immutable_bounded_tuple_of_token_ids(self) -> None:
        for anchor, error in (
            ([1, 2], TypeError),
            ("1 2", TypeError),
            ((True,), TypeError),
            ((1.0,), TypeError),
            ((-1,), ValueError),
            (tuple(range(33)), ValueError),
        ):
            with self.subTest(anchor=anchor), self.assertRaises(error):
                resolve_text_prefix(
                    self.previous, self.current, committed_anchor=anchor
                )

    def test_anchor_limit_must_be_an_integer_from_one_through_448(self) -> None:
        for limit, error in (
            (True, TypeError),
            (1.0, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (449, ValueError),
        ):
            with self.subTest(limit=limit), self.assertRaises(error):
                resolve_text_prefix(
                    self.previous, self.current, max_anchor_tokens=limit
                )

    def test_invalid_result_types_fail_before_resolving(self) -> None:
        for previous, current in (
            (42, self.current),
            (self.previous, None),
            (self.previous, WindowResult("plain", "text", 0, 2_000)),
        ):
            with (
                self.subTest(previous=previous, current=current),
                self.assertRaises(TypeError),
            ):
                resolve_text_prefix(previous, current)

    def test_decision_is_frozen_and_repeated_calls_are_pure(self) -> None:
        previous_snapshot = asdict(self.previous)
        current_snapshot = asdict(self.current)
        first = resolve_text_prefix(self.previous, self.current)
        second = resolve_text_prefix(self.previous, self.current)
        self.assertEqual(first, second)
        self.assertIs(type(first.tokens), tuple)
        self.assertIs(type(first.next_anchor), tuple)
        with self.assertRaises(FrozenInstanceError):
            first.tokens = (999,)
        self.assertEqual(asdict(self.previous), previous_snapshot)
        self.assertEqual(asdict(self.current), current_snapshot)


if __name__ == "__main__":
    unittest.main()
