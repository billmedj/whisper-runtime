"""Synthetic structural tests, not acoustic recognition or completion evidence."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any

from tools.word_anchor_reconciliation import (
    assess_resolution_handoff,
    propose_terminal_end_reconciliation,
)
from whisper_runtime import AudioSpan
from whisper_runtime.adapters.audio_evidence import (
    AudioEvidenceDecision,
    AudioObservation,
)
from whisper_runtime.adapters.continuous_stream import (
    ContinuousDecodeTrace,
    ContinuousResolutionObservation,
)
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.word_policy import (
    NativeWordAlignment,
    compare_word_hypotheses,
    diagnose_word_anchor,
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


def _probe(current, candidate, anchor, head, *, version=2, pcm=None):
    span = current.native.analyzed_span
    source = ContinuousDecodeTrace(
        1,
        span.start_ms * 16,
        span.end_ms * 16,
        head * 16,
        span.start_ms * 16,
        True,
        "anchor_missing",
        None,
        current.native,
        "unresolved",
        word_alignment=current,
        audio_evidence=None
        if pcm is None
        else AudioEvidenceDecision(
            AudioObservation.from_pcm(pcm), "uncertain", "missing_speech_score"
        ),
        anchor_diagnostic=diagnose_word_anchor(
            current, committed_through_ms=head, anchor=anchor
        ),
    )
    return ContinuousResolutionObservation(
        source,
        anchor,
        version,
        head * 16,
        span.end_ms * 16,
        "observed",
        "no_publication_authority",
        None
        if pcm is None
        else hashlib.sha256(pcm[(head - span.start_ms) * 32 :]).hexdigest(),
        candidate,
    )


class ResolutionHandoffTests(unittest.TestCase):
    def setUp(self):
        self.anchor = (_word(1, 200, 600), _word(2, 600, 1000))
        self.suffix = (_word(3, 1000, 1600), _word(4, 1600, 2200))
        self.pcm = bytes(range(256)) * 625  # Exactly five seconds of s16le PCM.
        self.probe = _probe(
            _aligned((_word(99, 0, 1000),), end=5000),
            _aligned(self.suffix, start=1000, end=5000),
            self.anchor,
            1000,
            pcm=self.pcm,
        )
        self.overlap = replace(
            self.probe,
            analysis_start_sample=200 * 16,
            pcm_sha256=hashlib.sha256(self.pcm[200 * 32 :]).hexdigest(),
            candidate=_aligned(self.anchor + self.suffix, start=200, end=5000),
        )

    def assess(self, probe=None, **changes):
        arguments = dict(
            current_session_version=2,
            committed_through_sample=1000 * 16,
            retained_from_sample=0,
            retained_pcm=self.pcm,
        )
        arguments.update(changes)
        return assess_resolution_handoff(
            self.probe if probe is None else probe, **arguments
        )

    def reject(self, reason, **changes):
        result = self.assess(**changes)
        self.assertEqual((result.status, result.reason), ("rejected", reason))
        self.assertFalse(result.publication_authorized)
        self.assertIsNone(result.overlap_anchor_start)
        self.assertIsNone(result.overlap_anchor_end)

    def test_head_only_needs_one_predetermined_overlap_without_execution_authority(
        self,
    ):
        before = asdict(self.probe)
        result = self.assess()
        self.assertEqual(result.status, "needs_overlap")
        self.assertEqual(result.candidate_window, (200, 5000))
        self.assertIn("anchor_bearing_overlap_observation", result.missing_evidence)
        self.assertFalse(result.publication_authorized)
        self.assertEqual(result.original_reason, "anchor_missing")
        self.assertIs(
            result.original_anchor_diagnostic, self.probe.source.anchor_diagnostic
        )
        self.assertEqual(asdict(self.probe), before)
        self.reject("overlap_attempt_exhausted", overlap_attempted=True)

    def test_matching_overlap_is_structural_only_and_never_mutates_words(self):
        before = asdict(self.overlap)
        result = self.assess(overlap=self.overlap)
        self.assertEqual(result.status, "structurally_eligible")
        self.assertEqual(
            (result.overlap_anchor_start, result.overlap_anchor_end), (0, 2)
        )
        self.assertEqual(
            result.missing_evidence,
            ("acoustic_boundary_coverage", "model_tokenizer_options_provenance"),
        )
        self.assertFalse(result.publication_authorized)
        self.assertFalse(hasattr(result, "publication"))
        self.assertEqual(asdict(self.overlap), before)
        with self.assertRaises(FrozenInstanceError):
            result.publication_authorized = True

    def test_missing_and_mismatched_pcm_cannot_certify_correspondence(self):
        self.assertIn(
            "retained_pcm_correspondence",
            self.assess(retained_pcm=None).missing_evidence,
        )
        self.reject(
            "retained_pcm_correspondence", retained_pcm=None, overlap=self.overlap
        )
        self.reject("head_probe_pcm_mismatch", retained_pcm=self.pcm[:-2])
        self.reject(
            "head_probe_pcm_mismatch", probe=replace(self.probe, pcm_sha256="0" * 64)
        )
        self.reject(
            "overlap_pcm_mismatch", overlap=replace(self.overlap, pcm_sha256="0" * 64)
        )

    def test_stale_versions_spans_and_prefixes_are_not_reused(self):
        self.reject("stale_session_version", current_session_version=3)
        self.reject("stale_frozen_boundary", committed_through_sample=1001 * 16)
        self.reject("stale_frozen_boundary", retained_from_sample=16)
        self.reject(
            "overlap_stale_session_version",
            overlap=replace(self.overlap, session_version=3),
        )
        self.reject(
            "overlap_span_mismatch",
            overlap=replace(self.overlap, analysis_end_sample=4999 * 16),
        )
        self.reject(
            "head_probe_span_mismatch",
            probe=replace(self.probe, analysis_start_sample=999 * 16),
        )
        self.reject(
            "overlap_frozen_state_mismatch",
            overlap=replace(self.overlap, anchor=self.anchor[-1:]),
        )
        self.reject(
            "head_probe_not_observed", probe=replace(self.probe, status="failed")
        )
        self.reject(
            "overlap_not_observed", overlap=replace(self.overlap, status="running")
        )
        self.reject(
            "overlap_span_mismatch",
            overlap=replace(self.overlap, analysis_start_sample=3200.0),
        )

    def test_original_source_pcm_is_verified_or_explicitly_unproven(self):
        # This byte precedes BOTH candidate windows but belongs to the refusal.
        self.reject(
            "original_source_pcm_mismatch",
            retained_pcm=b"x" + self.pcm[1:],
            overlap=self.overlap,
        )
        source = replace(self.probe.source, audio_evidence=None)
        result = self.assess(
            replace(self.probe, source=source),
            overlap=replace(self.overlap, source=source),
        )
        self.assertEqual(result.status, "structurally_eligible")
        self.assertIn("original_source_pcm_correspondence", result.missing_evidence)
        self.assertFalse(result.publication_authorized)

    def test_repeated_anchor_is_rejected_before_choosing_a_timed_occurrence(self):
        repeated = tuple(
            replace(
                word, span=AudioSpan(word.span.start_ms + 3000, word.span.end_ms + 3000)
            )
            for word in self.anchor
        )
        changed = replace(
            self.overlap,
            candidate=_aligned(
                self.anchor + self.suffix + repeated, start=200, end=5000
            ),
        )
        self.reject("overlap_anchor_ambiguous", overlap=changed)
        relocated = replace(
            self.overlap, candidate=_aligned(repeated, start=200, end=5000)
        )
        self.reject("overlap_anchor_timing_mismatch", overlap=relocated)

    def test_omitted_boundary_word_is_detected_if_the_overlap_observes_it(self):
        suffix = (
            _word(9, 1000, 1200),
            replace(self.suffix[0], span=AudioSpan(1200, 1600)),
        ) + self.suffix[1:]
        changed = replace(
            self.overlap, candidate=_aligned(self.anchor + suffix, start=200, end=5000)
        )
        self.reject("complete_suffix_disagrees", overlap=changed)

    def test_common_omission_can_still_evade_structural_checks(self):
        # A spoken word in [1000,1200] omitted by BOTH observations cannot be
        # inferred from their matching arrays. A gap is not acoustic silence.
        suffix = (replace(self.suffix[0], span=AudioSpan(1200, 1600)),) + self.suffix[
            1:
        ]
        probe = replace(self.probe, candidate=_aligned(suffix, start=1000, end=5000))
        overlap = replace(
            self.overlap, candidate=_aligned(self.anchor + suffix, start=200, end=5000)
        )
        result = self.assess(probe, overlap=overlap)
        self.assertEqual(result.status, "structurally_eligible")
        self.assertIn("acoustic_boundary_coverage", result.missing_evidence)
        self.assertFalse(result.publication_authorized)

    def test_full_suffix_comparison_does_not_drop_punctuation_tokens_or_repetitions(
        self,
    ):
        for suffix in (
            (replace(self.suffix[0], tokens=(99,)),) + self.suffix[1:],
            (replace(self.suffix[0], text=" Word-3"),) + self.suffix[1:],
            self.suffix + (_word(13, 2300, 2400, "."),),
            self.suffix + (_word(4, 2300, 2400),),
        ):
            with self.subTest(suffix=suffix):
                changed = replace(
                    self.overlap,
                    candidate=_aligned(self.anchor + suffix, start=200, end=5000),
                )
                self.reject("complete_suffix_disagrees", overlap=changed)

    def test_timing_rules_do_not_stack_extensions_or_allow_boundary_crossing(self):
        changed = replace(
            self.overlap,
            candidate=_aligned(
                (replace(self.anchor[0], span=AudioSpan(200, 399)),)
                + self.anchor[1:]
                + self.suffix,
                start=200,
                end=5000,
            ),
        )
        self.reject("overlap_anchor_timing_mismatch", overlap=changed)
        suffix = (replace(self.suffix[0], span=AudioSpan(1201, 1600)),) + self.suffix[
            1:
        ]
        changed = replace(
            self.overlap, candidate=_aligned(self.anchor + suffix, start=200, end=5000)
        )
        self.reject("suffix_timing_mismatch", overlap=changed)
        anchor = self.anchor[:-1] + (
            replace(self.anchor[-1], span=AudioSpan(600, 1100)),
        )
        suffix = (replace(self.suffix[0], span=AudioSpan(1100, 1600)),) + self.suffix[
            1:
        ]
        changed = replace(
            self.overlap, candidate=_aligned(anchor + suffix, start=200, end=5000)
        )
        self.reject("head_continuation_crosses_observed_anchor", overlap=changed)

    def test_malformed_arguments_are_not_coerced(self):
        for name in (
            "current_session_version",
            "committed_through_sample",
            "retained_from_sample",
            "timestamp_tolerance_ms",
        ):
            for value in (True, 1.0, "2", None):
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    self.assess(**{name: value})
        with self.assertRaises(TypeError):
            self.assess(retained_pcm=bytearray(self.pcm))
        with self.assertRaises(ValueError):
            self.assess(retained_pcm=b"x")

    def test_archived_head_only_candidates_all_lack_anchor_bearing_observations(self):
        from tools.analyze_word_resolution import _alignment

        path = (
            Path(__file__).resolve().parents[1]
            / "evidence/modal-t4-tiny-en-word-resolution-2026-09-06.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = ((20720, 33660), (32400, 43660), (1120, 8220), (1100, 7740))
        self.assertEqual(len(record["cells"]), len(expected))
        for cell, window in zip(record["cells"], expected):
            with self.subTest(cell=cell["id"]):
                frozen = cell["frozen_state"]
                anchor = tuple(
                    NativeTimestampSegment(
                        AudioSpan(**word["span"]), word["text"], tuple(word["tokens"])
                    )
                    for word in frozen["anchor"]
                )
                probe = _probe(
                    _alignment(cell["raw_alignments"]["current"]),
                    _alignment(cell["raw_alignments"]["alternative"]),
                    anchor,
                    frozen["head_ms"],
                )
                # Version 2 here is a local replay tag, NOT an invented archived
                # live-session version. Missing PCM/provenance stays explicit.
                result = self.assess(
                    probe,
                    committed_through_sample=frozen["head_ms"] * 16,
                    retained_from_sample=frozen["retained_ms"] * 16,
                    retained_pcm=None,
                )
                self.assertEqual(result.status, "needs_overlap")
                self.assertEqual(result.candidate_window, window)
                self.assertEqual(
                    result.original_reason, cell["outcome"]["strict_reason"]
                )
                self.assertEqual(
                    result.original_anchor_diagnostic.status,
                    cell["current_diagnostic"]["status"],
                )
                self.assertFalse(result.publication_authorized)


if __name__ == "__main__":
    unittest.main()
