"""Synthetic structural tests, not acoustic recognition or completion evidence."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any

from tools.word_anchor_reconciliation import propose_terminal_end_reconciliation
from whisper_runtime import AudioSpan
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.word_policy import (
    NativeWordAlignment,
    compare_word_hypotheses,
)


def _word(
    token: int, start: int, end: int, text: str | None = None
) -> NativeTimestampSegment:
    return NativeTimestampSegment(
        AudioSpan(start, end), f" word-{token}" if text is None else text, (token,)
    )


def _aligned(
    words: tuple[NativeTimestampSegment, ...], *, start: int = 0, end: int = 10_000
) -> NativeWordAlignment:
    text = "".join(word.text for word in words).strip()
    tokens = tuple(token for word in words for token in word.tokens)
    segments = (
        (NativeTimestampSegment(AudioSpan(start, end), text, tokens),) if words else ()
    )
    return NativeWordAlignment(
        NativeWindowResult(
            window_id=f"synthetic-{start}-{end}",
            text=text,
            start_ms=start,
            end_ms=end,
            metadata=NativeDecodeMetadata(
                language="en",
                tokens=tokens,
                segments=segments,
                timestamps_complete=bool(words),
            ),
        ),
        words,
    )


class TerminalEndReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = (
            _word(1, 1_000, 1_400, " and"),
            _word(2, 1_500, 1_800, " bird"),
            _word(1, 1_900, 2_200, " and"),
            _word(3, 2_500, 3_000, " tree"),
        )
        self.observed = self.anchor[:-1] + (
            replace(self.anchor[-1], span=AudioSpan(2_480, 3_540)),
        )
        self.suffix = (_word(4, 3_600, 3_900, " continue"),)
        self.current = _aligned(self.observed + self.suffix)

    def propose(self, current: Any = None, **changes: Any):
        arguments = {
            "committed_through_ms": 3_000,
            "anchor": self.anchor,
            "final": True,
        }
        arguments.update(changes)
        return propose_terminal_end_reconciliation(
            self.current if current is None else current, **arguments
        )

    def assert_rejected(self, reason: str, current: Any = None, **changes: Any):
        proposal = self.propose(current, **changes)
        self.assertEqual((proposal.status, proposal.reason), ("rejected", reason))
        self.assertIsNone(proposal.word_start)
        self.assertIsNone(proposal.word_end)
        self.assertIsNone(proposal.end_extension_ms)

    def test_540ms_extension_is_only_a_frozen_proposal_and_baseline_still_rejects(self):
        before_current, before_anchor = asdict(self.current), self.anchor
        baseline = compare_word_hypotheses(
            None,
            self.current,
            committed_through_ms=3_000,
            anchor=self.anchor,
            final=True,
        )
        self.assertEqual(baseline.reason, "anchor_missing")
        proposal = self.propose()
        self.assertEqual(proposal.status, "eligible")
        self.assertEqual(proposal.reason, "terminal_end_only_extension")
        self.assertEqual((proposal.word_start, proposal.word_end), (0, 4))
        self.assertEqual(proposal.end_extension_ms, 540)
        self.assertFalse(hasattr(proposal, "publication"))
        self.assertFalse(hasattr(proposal, "committed_through_ms"))
        with self.assertRaises(FrozenInstanceError):
            proposal.word_end = 5
        self.assertEqual(asdict(self.current), before_current)
        self.assertIs(self.anchor, before_anchor)
        self.assertEqual(
            compare_word_hypotheses(
                None,
                self.current,
                committed_through_ms=3_000,
                anchor=self.anchor,
                final=True,
            ),
            baseline,
        )

    def test_strict_matches_need_no_reconciliation_including_tolerance_edges(self):
        for extension in (-200, 0, 200):
            with self.subTest(extension=extension):
                current = _aligned(
                    self.anchor[:-1]
                    + (
                        replace(
                            self.anchor[-1], span=AudioSpan(2_500, 3_000 + extension)
                        ),
                    )
                    + self.suffix
                )
                proposal = self.propose(current)
                self.assertEqual(proposal.status, "not_needed")
                self.assertEqual(proposal.reason, "within_strict_tolerance")
                self.assertEqual(proposal.end_extension_ms, extension)

    def test_not_needed_is_not_suffix_publication_authority(self):
        proposal = self.propose(_aligned(self.anchor))
        self.assertEqual(proposal.status, "not_needed")
        self.assertFalse(hasattr(proposal, "publication"))

    def test_cap_is_explicit_inclusive_and_not_a_global_tolerance(self):
        self.assertEqual(self.propose(max_end_extension_ms=540).status, "eligible")
        self.assert_rejected("extension_exceeds_cap", max_end_extension_ms=539)
        self.assert_rejected("extension_exceeds_cap", max_end_extension_ms=0)
        for extension, expected in ((1_000, "eligible"), (1_001, "rejected")):
            with self.subTest(extension=extension):
                current = _aligned(
                    self.observed[:-1]
                    + (_word(3, 2_480, 3_000 + extension, " tree"),)
                    + (_word(4, 4_100, 4_200, " continue"),)
                )
                self.assertEqual(self.propose(current).status, expected)

    def test_no_nonfinal_reconciliation(self):
        self.assert_rejected("not_final", final=False)

    def test_all_anchor_starts_remain_strict_including_terminal(self):
        for index in range(len(self.anchor)):
            with self.subTest(index=index):
                words = list(self.observed)
                old = words[index]
                words[index] = replace(
                    old, span=AudioSpan(old.span.start_ms - 201, old.span.end_ms)
                )
                if index:
                    previous = words[index - 1]
                    words[index - 1] = replace(
                        previous,
                        span=AudioSpan(
                            previous.span.start_ms, words[index].span.start_ms
                        ),
                    )
                self.assert_rejected(
                    "anchor_start_mismatch", _aligned(tuple(words) + self.suffix)
                )

    def test_no_stacking_with_existing_window_origin_onset_exception(self):
        words = (replace(self.observed[0], span=AudioSpan(0, 1_400)),) + self.observed[
            1:
        ]
        self.assert_rejected("anchor_start_mismatch", _aligned(words + self.suffix))

    def test_internal_end_drift_is_not_relaxed(self):
        changed = (
            replace(self.observed[0], span=AudioSpan(1_000, 1_199)),
        ) + self.observed[1:]
        self.assert_rejected(
            "nonterminal_end_mismatch", _aligned(changed + self.suffix)
        )

    def test_terminal_contraction_is_not_an_extension(self):
        changed = self.anchor[:-1] + (_word(3, 2_480, 2_799, " tree"),)
        self.assert_rejected(
            "terminal_end_not_rightward", _aligned(changed + self.suffix)
        )

    def test_tokens_and_text_are_exact_not_normalized(self):
        changed = (replace(self.observed[0], tokens=(99,)),) + self.observed[1:]
        self.assert_rejected("tokens_differ", _aligned(changed + self.suffix))
        for text in (" And", "and", " AND", " and."):
            with self.subTest(text=text):
                changed = (replace(self.observed[0], text=text),) + self.observed[1:]
                self.assert_rejected(
                    "anchor_text_absent", _aligned(changed + self.suffix)
                )

    def test_missing_anchor_and_partial_suffix_do_not_supply_evidence(self):
        for words in ((), self.suffix, self.observed[-1:] + self.suffix):
            with self.subTest(words=words):
                self.assert_rejected("anchor_text_absent", _aligned(words))

    def test_duplicate_exact_phrase_anywhere_is_ambiguous(self):
        repeated = tuple(
            replace(
                word,
                span=AudioSpan(word.span.start_ms + 4_000, word.span.end_ms + 4_000),
            )
            for word in self.anchor
        )
        self.assert_rejected(
            "anchor_ambiguous", _aligned(self.observed + self.suffix + repeated)
        )

    def test_a_unique_later_repeated_phrase_cannot_replace_original(self):
        relocated = tuple(
            replace(
                word,
                span=AudioSpan(word.span.start_ms + 4_000, word.span.end_ms + 4_000),
            )
            for word in self.observed
        )
        self.assert_rejected(
            "anchor_relocated", _aligned(relocated + (_word(4, 8_000, 8_200),))
        )

    def test_anchor_terminal_start_at_watermark_is_also_relocation(self):
        changed = self.observed[:-1] + (_word(3, 3_000, 3_540, " tree"),)
        self.assert_rejected("anchor_relocated", _aligned(changed + self.suffix))

    def test_single_lexical_anchor_is_insufficient_but_strict_match_is_not_needed(self):
        self.assert_rejected("insufficient_lexical_anchor", anchor=self.anchor[-1:])
        proposal = self.propose(
            _aligned(self.anchor + self.suffix), anchor=self.anchor[-1:]
        )
        self.assertEqual(proposal.status, "not_needed")

    def test_punctuation_is_not_a_second_lexical_witness(self):
        anchor = (_word(8, 1_000, 1_100, ","), self.anchor[-1])
        observed = anchor[:-1] + self.observed[-1:] + self.suffix
        self.assert_rejected(
            "insufficient_lexical_anchor", _aligned(observed), anchor=anchor
        )

    def test_two_lexical_words_then_terminal_punctuation_cannot_extend(self):
        anchor = self.anchor[:2] + (_word(8, 2_500, 3_000, "."),)
        observed = anchor[:-1] + (_word(8, 2_500, 3_540, "."),) + self.suffix
        self.assert_rejected(
            "nonlexical_terminal_anchor", _aligned(observed), anchor=anchor
        )

    def test_continuation_must_contain_lexical_text(self):
        for suffix in ((), (_word(8, 3_600, 3_700, "."),)):
            with self.subTest(suffix=suffix):
                self.assert_rejected(
                    "missing_lexical_continuation", _aligned(self.observed + suffix)
                )

    def test_continuation_may_touch_but_must_not_overlap_observed_end(self):
        touching = _aligned(self.observed + (_word(4, 3_540, 3_700),))
        self.assertEqual(self.propose(touching).status, "eligible")
        for start in (2_999, 3_539):
            with self.subTest(start=start), self.assertRaises(ValueError):
                _aligned(self.observed + (_word(4, start, 3_700),))

    def test_anchor_must_end_at_frozen_watermark_and_be_inside_analysis(self):
        self.assert_rejected("anchor_not_at_watermark", committed_through_ms=3_100)
        current = _aligned(self.observed[1:] + self.suffix, start=1_500)
        self.assert_rejected("anchor_outside_analysis", current)
        self.assert_rejected("watermark_outside_analysis", committed_through_ms=10_001)
        self.assert_rejected("empty_anchor", anchor=())

    def test_prior_context_is_not_included_in_anchor_indices(self):
        current = _aligned((_word(99, 100, 200),) + self.observed + self.suffix)
        proposal = self.propose(current)
        self.assertEqual(
            (proposal.status, proposal.word_start, proposal.word_end),
            ("eligible", 1, 5),
        )

    def test_missing_spoken_word_can_evade_all_structural_checks(self):
        # These same estimates are compatible with an unrecognized word during
        # [3_100, 3_400] that was swallowed into "tree". We have no PCM evidence
        # here to distinguish that world from a genuinely extended "tree".
        # Passing is intentionally a negative example of acoustic certification.
        proposal = self.propose()
        self.assertEqual(proposal.status, "eligible")
        self.assertGreater(self.observed[-1].span.end_ms, 3_400)
        self.assertEqual(len(self.current.words), 5)

    def test_invalid_argument_types_raise_including_booleans_as_integers(self):
        for name in (
            "committed_through_ms",
            "timestamp_tolerance_ms",
            "max_end_extension_ms",
        ):
            for value in (True, False, 1.0, "200", None):
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    self.propose(**{name: value})
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.propose(**{name: -1})
        for value in (0, 1, None, "true"):
            with self.subTest(final=value), self.assertRaises(TypeError):
                self.propose(final=value)
        for value in (list(self.anchor), None, (object(),)):
            with self.subTest(anchor=value), self.assertRaises(TypeError):
                self.propose(anchor=value)
        with self.assertRaises(TypeError):
            self.propose(object())

    def test_invalid_anchor_values_raise_without_modifying_current(self):
        before = asdict(self.current)
        invalid = (
            self.anchor + (_word(5, 3_000, 3_000),),
            (_word(1, 1_000, 1_400, " "),),
            (replace(self.anchor[0], tokens=()),),
            (self.anchor[0], _word(2, 1_399, 1_500)),
            (_word(1, 3_000, 3_001),),
        )
        for anchor in invalid:
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                self.propose(anchor=anchor)
        self.assertEqual(asdict(self.current), before)


if __name__ == "__main__":
    unittest.main()
