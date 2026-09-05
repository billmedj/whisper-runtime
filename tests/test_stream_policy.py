import unittest
from dataclasses import FrozenInstanceError, replace

from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
    select_native_publication,
)
from whisper_runtime.adapters.stream_policy import (
    AgreementDecision,
    AgreementReason,
    compare_hypotheses,
)
from whisper_runtime.state import AudioSpan, WindowResult


def segment(
    start: int, end: int, text: str = " Hello", tokens: tuple[int, ...] = (1,)
) -> NativeTimestampSegment:
    return NativeTimestampSegment(AudioSpan(start, end), text, tokens)


def hypothesis(
    end: int,
    segments: tuple[NativeTimestampSegment, ...],
    *,
    start: int = 0,
    complete: bool = True,
) -> NativeWindowResult:
    return NativeWindowResult(
        window_id=f"window-{start}-{end}",
        text="".join(value.text for value in segments).strip(),
        start_ms=start,
        end_ms=end,
        metadata=NativeDecodeMetadata(
            language="en",
            tokens=tuple(token for value in segments for token in value.tokens),
            segments=segments,
            timestamps_complete=complete,
        ),
    )


class StreamPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = segment(0, 1_000)
        self.second = segment(1_000, 2_000, " world", (2,))
        self.previous = hypothesis(2_000, (self.first, self.second))
        self.current = hypothesis(3_000, (self.first, self.second))

    def compare(
        self,
        previous: NativeWindowResult | None = None,
        current: NativeWindowResult | None = None,
        **kwargs: int,
    ) -> AgreementDecision:
        return compare_hypotheses(
            self.previous if previous is None else previous,
            self.current if current is None else current,
            committed_through_ms=kwargs.pop("committed_through_ms", 0),
            **kwargs,
        )

    def test_two_growing_hypotheses_agree_on_exact_complete_prefix(self) -> None:
        decision = self.compare()
        self.assertEqual(decision.reason, AgreementReason.CANDIDATE)
        self.assertEqual(decision.publication_span, AudioSpan(0, 2_000))
        selected = select_native_publication(self.current, decision.publication_span)
        self.assertEqual(selected.text, "Hello world")

    def test_first_hypothesis_is_not_confirmation(self) -> None:
        decision = compare_hypotheses(None, self.current, committed_through_ms=0)
        self.assertEqual(decision, AgreementDecision(AgreementReason.INCOMPLETE))

    def test_repeated_same_audio_and_shrinking_audio_cannot_confirm(self) -> None:
        for current in (self.previous, replace(self.previous, end_ms=1_999)):
            with self.subTest(current=current):
                self.assertEqual(
                    self.compare(current=current).reason, AgreementReason.UNSTABLE
                )

    def test_changed_analysis_start_requires_a_new_pair(self) -> None:
        current = replace(self.current, start_ms=100)
        self.assertEqual(self.compare(current=current).reason, AgreementReason.UNSTABLE)

    def test_missing_metadata_or_timestamp_segments_is_incomplete(self) -> None:
        for previous, current in (
            (replace(self.previous, metadata=None), self.current),
            (self.previous, replace(self.current, metadata=None)),
            (hypothesis(2_000, ()), self.current),
            (self.previous, hypothesis(3_000, ())),
        ):
            with self.subTest(previous=previous, current=current):
                self.assertEqual(
                    self.compare(previous, current).reason, AgreementReason.INCOMPLETE
                )

    def test_closed_prefix_can_agree_despite_unfinished_global_tail(self) -> None:
        previous = hypothesis(2_000, (self.first,), complete=False)
        current = hypothesis(3_000, (self.first, self.second), complete=False)
        decision = self.compare(previous, current)
        self.assertEqual(decision.publication_span, AudioSpan(0, 1_000))

    def test_holdback_must_be_met_at_current_segment_end(self) -> None:
        self.assertEqual(
            self.compare(holdback_ms=2_000).publication_span, AudioSpan(0, 1_000)
        )
        self.assertEqual(
            self.compare(holdback_ms=2_001).reason, AgreementReason.INCOMPLETE
        )
        self.assertEqual(
            self.compare(holdback_ms=0).publication_span, AudioSpan(0, 2_000)
        )

    def test_text_is_compared_without_normalization(self) -> None:
        for text in ("Hello", " hello", " HELLO", " Hello ", " Hello."):
            current = hypothesis(3_000, (replace(self.first, text=text), self.second))
            with self.subTest(text=text):
                self.assertEqual(
                    self.compare(current=current).reason, AgreementReason.UNSTABLE
                )

    def test_identical_text_with_different_tokens_does_not_agree(self) -> None:
        current = hypothesis(3_000, (replace(self.first, tokens=(11,)), self.second))
        self.assertEqual(self.compare(current=current).reason, AgreementReason.UNSTABLE)

    def test_comparison_stops_before_unstable_tail_without_skipping_it(self) -> None:
        third = segment(2_000, 2_500, " again", (3,))
        previous = hypothesis(3_000, (self.first, self.second, third))
        current = hypothesis(
            4_000, (self.first, replace(self.second, text=" other"), third)
        )
        self.assertEqual(
            self.compare(previous, current).publication_span, AudioSpan(0, 1_000)
        )

    def test_later_match_is_not_used_after_initial_mismatch(self) -> None:
        current = hypothesis(3_000, (replace(self.first, text=" other"), self.second))
        decision = self.compare(current=current)
        self.assertEqual(decision.reason, AgreementReason.UNSTABLE)
        self.assertIsNone(decision.publication_span)

    def test_timestamp_tolerance_includes_limit_and_uses_current_bounds(self) -> None:
        current = hypothesis(
            4_000, (segment(0, 1_200), segment(1_200, 2_200, " world", (2,)))
        )
        self.assertEqual(
            self.compare(current=current).publication_span, AudioSpan(0, 2_200)
        )
        self.assertEqual(
            self.compare(current=current, timestamp_tolerance_ms=199).reason,
            AgreementReason.UNSTABLE,
        )

    def test_zero_timestamp_tolerance_requires_identical_times(self) -> None:
        self.assertEqual(
            self.compare(timestamp_tolerance_ms=0).reason, AgreementReason.CANDIDATE
        )
        current = hypothesis(3_000, (segment(0, 1_001),))
        self.assertEqual(
            self.compare(current=current, timestamp_tolerance_ms=0).reason,
            AgreementReason.UNSTABLE,
        )

    def test_only_segments_after_commit_watermark_are_compared(self) -> None:
        previous = hypothesis(
            2_000, (replace(self.first, text=" changed old text"), self.second)
        )
        decision = self.compare(previous, committed_through_ms=1_000)
        self.assertEqual(decision.publication_span, AudioSpan(1_000, 2_000))

    def test_segment_straddling_watermark_is_not_trimmed_or_skipped(self) -> None:
        for previous, current in (
            (self.previous, self.current),
            (hypothesis(2_000, (segment(0, 1_100),)), self.current),
        ):
            with self.subTest(previous=previous):
                self.assertEqual(
                    self.compare(previous, current, committed_through_ms=500).reason,
                    AgreementReason.UNSTABLE,
                )

    def test_gap_after_watermark_is_never_inferred_as_silence(self) -> None:
        for previous, current in (
            (hypothesis(2_000, (segment(100, 1_000),)), self.current),
            (self.previous, hypothesis(3_000, (segment(100, 1_000),))),
        ):
            with self.subTest(previous=previous, current=current):
                self.assertEqual(
                    self.compare(
                        previous, current, timestamp_tolerance_ms=10_000
                    ).reason,
                    AgreementReason.GAP,
                )

    def test_internal_gap_stops_before_gap_and_is_exposed_on_next_watermark(
        self,
    ) -> None:
        segments = (self.first, segment(1_200, 2_000, " world", (2,)))
        previous = hypothesis(2_000, segments)
        current = hypothesis(3_000, segments)
        self.assertEqual(
            self.compare(previous, current).publication_span, AudioSpan(0, 1_000)
        )
        self.assertEqual(
            self.compare(previous, current, committed_through_ms=1_000).reason,
            AgreementReason.GAP,
        )

    def test_overlap_in_a_hypothesis_cannot_expand_candidate(self) -> None:
        segments = (self.first, segment(900, 2_000, " world", (2,)))
        previous = hypothesis(2_000, segments)
        current = hypothesis(3_000, segments)
        self.assertEqual(
            self.compare(previous, current).publication_span, AudioSpan(0, 1_000)
        )
        self.assertEqual(
            self.compare(previous, current, committed_through_ms=1_000).reason,
            AgreementReason.UNSTABLE,
        )

    def test_analysis_start_after_watermark_is_an_uncovered_gap(self) -> None:
        segments = (segment(5_000, 6_000),)
        decision = self.compare(
            hypothesis(7_000, segments, start=5_000),
            hypothesis(8_000, segments, start=5_000),
        )
        self.assertEqual(decision.reason, AgreementReason.GAP)

    def test_nonzero_origin_is_preserved(self) -> None:
        segments = (segment(5_000, 6_000), segment(6_000, 7_000, " world", (2,)))
        decision = self.compare(
            hypothesis(7_000, segments, start=5_000),
            hypothesis(8_000, segments, start=5_000),
            committed_through_ms=5_000,
        )
        self.assertEqual(decision.publication_span, AudioSpan(5_000, 7_000))

    def test_watermark_at_or_past_all_available_segments_is_incomplete(self) -> None:
        for watermark in (2_000, 3_000, 4_000):
            with self.subTest(watermark=watermark):
                self.assertEqual(
                    self.compare(committed_through_ms=watermark).reason,
                    AgreementReason.INCOMPLETE,
                )

    def test_empty_or_whitespace_text_and_empty_tokens_do_not_prove_silence(
        self,
    ) -> None:
        for value in (
            segment(0, 1_000, ""),
            segment(0, 1_000, " \n\t"),
            segment(0, 1_000, tokens=()),
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    self.compare(
                        hypothesis(2_000, (value,)), hypothesis(3_000, (value,))
                    ).reason,
                    AgreementReason.INCOMPLETE,
                )

    def test_out_of_analysis_or_zero_duration_segment_is_not_a_candidate(self) -> None:
        for value in (segment(0, 0), segment(0, 2_001)):
            with self.subTest(value=value):
                decision = self.compare(
                    hypothesis(2_000, (value,)), hypothesis(3_000, (value,))
                )
                self.assertIsNone(decision.publication_span)

    def test_uses_full_analysis_provenance_of_selected_results(self) -> None:
        current = select_native_publication(self.current, AudioSpan(1_000, 2_000))
        self.assertEqual(
            self.compare(current=current).publication_span, AudioSpan(0, 2_000)
        )

    def test_policy_is_pure_and_does_not_retain_hypotheses_in_decision(self) -> None:
        previous_repr, current_repr = repr(self.previous), repr(self.current)
        first = self.compare()
        second = self.compare()
        self.assertEqual(first, second)
        self.assertEqual(repr(self.previous), previous_repr)
        self.assertEqual(repr(self.current), current_repr)
        self.assertFalse(hasattr(first, "__dict__"))
        self.assertFalse(hasattr(first, "previous"))
        self.assertFalse(hasattr(first, "current"))
        with self.assertRaises(FrozenInstanceError):
            first.reason = AgreementReason.GAP

    def test_invalid_options_are_rejected(self) -> None:
        for name in ("committed_through_ms", "holdback_ms", "timestamp_tolerance_ms"):
            for value in (True, False, 1.5, "100", -1):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    self.compare(**{name: value})

    def test_invalid_hypothesis_types_are_rejected(self) -> None:
        for previous, current in (
            (object(), self.current),
            (self.previous, object()),
            (None, WindowResult("w", "text", 0, 100)),
        ):
            with (
                self.subTest(previous=previous, current=current),
                self.assertRaises(TypeError),
            ):
                compare_hypotheses(previous, current, committed_through_ms=0)

    def test_decision_requires_consistent_reason_and_span(self) -> None:
        for reason, span in (
            ("candidate", AudioSpan(0, 1)),
            (AgreementReason.CANDIDATE, None),
            (AgreementReason.CANDIDATE, AudioSpan(0, 0)),
            (AgreementReason.GAP, AudioSpan(0, 1)),
        ):
            with (
                self.subTest(reason=reason, span=span),
                self.assertRaises((TypeError, ValueError)),
            ):
                AgreementDecision(reason, span)


if __name__ == "__main__":
    unittest.main()
