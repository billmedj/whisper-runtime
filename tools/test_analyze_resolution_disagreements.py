"""Archive replay and adversarial checks without PCM, torch, or Modal."""

import copy
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from tools import analyze_resolution_disagreements as replay
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeTimestampSegment, NativeWordAlignment
from whisper_runtime.adapters.native_result import NativeWindowResult


class ResolutionDisagreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = (
            Path(__file__).resolve().parents[1] / replay.ARCHIVE
        ).read_bytes()
        cls.record = json.loads(cls.payload)

    def test_archive_replay_preserves_every_refusal_without_running_inference(self):
        report = replay.analyze_bytes(self.payload)
        self.assertEqual(report, replay.analyze_bytes(self.payload))
        self.assertEqual(report["input_sha256"], replay.ARCHIVE_SHA)
        self.assertEqual(report["native_windows_executed"], 0)
        self.assertFalse(report["publication_authorized"])
        self.assertEqual(len(report["cells"]), 5)
        for diagnosed, saved in zip(report["cells"], self.record["cells"]):
            self.assertEqual(diagnosed["id"], saved["id"])
            self.assertEqual(
                diagnosed["original_assessment"], saved["summary"]["assessment"]
            )
            self.assertFalse(diagnosed["publication_authorized"])
        with self.assertRaisesRegex(ValueError, "fixed seven-window"):
            replay.analyze_bytes(self.payload + b" ")

    def test_casefold_disagreement_has_timing_but_absent_anchors_do_not(self):
        cells = replay.analyze_bytes(self.payload)["cells"]
        for index, cell in enumerate(cells):
            if index == 1:
                comparison = cell["complete_suffix_comparison"]
                self.assertEqual(cell["exact_anchor_occurrences"], [0])
                self.assertEqual(comparison["text_relation"], "casefold_equal_units")
                self.assertFalse(comparison["unit_tokens_equal"])
                self.assertEqual(comparison["max_start_delta_ms"], 60)
                self.assertEqual(comparison["max_end_delta_ms"], 60)
            else:
                self.assertEqual(cell["exact_anchor_occurrences"], [])
                self.assertIsNone(cell["complete_suffix_comparison"])
                self.assertIsNone(cell["boundary_checks"])

    def test_long_punctuation_estimate_is_not_reported_as_lexical_continuation(self):
        cell = replay.analyze_bytes(self.payload)["cells"][0]
        extents = cell["estimated_lexical_extents"]
        self.assertEqual(extents["overlap"]["lexical_units_ending_after_head"], 0)
        self.assertEqual(extents["overlap"]["trailing_nonlexical_estimate_ms"], 7260)
        self.assertEqual(extents["candidate"]["lexical_units_ending_after_head"], 18)
        # The existing earlier-start control restores the anchor, not the suffix.
        control = cell["prospective_context_guard"]["recorded_comparisons"]["current"]
        self.assertEqual(control["exact_anchor_occurrences"], [2])
        self.assertEqual(
            control["complete_suffix_comparison"]["text_relation"], "different_units"
        )
        self.assertIsNone(control["complete_suffix_comparison"]["max_start_delta_ms"])

    def test_fixed_guard_plan_reuses_intervals_conditionally_and_never_expands_pcm(
        self,
    ):
        report = replay.analyze_bytes(self.payload)
        self.assertEqual(report["unmatched_guard_window_count"], 3)
        self.assertEqual(
            [c["prospective_context_guard"]["start_ms"] for c in report["cells"]],
            [1680, 20220, 31900, 620, 640],
        )
        for cell, original in zip(report["cells"], self.record["inputs"]):
            plan, frozen = cell["prospective_context_guard"], original["frozen_state"]
            self.assertEqual(plan["requested_guard_ms"], 500)
            self.assertGreaterEqual(plan["start_ms"], frozen["retained_ms"])
            self.assertLess(plan["start_ms"], frozen["head_ms"])
            self.assertEqual(plan["end_ms"], frozen["end_ms"])
            self.assertLessEqual(plan["end_ms"] - plan["start_ms"], 30000)
            self.assertTrue(plan["reuse_requires_pcm_and_execution_identity_check"])
            self.assertEqual(
                plan["recorded_span_available"], bool(plan["matching_recorded_arms"])
            )

    def test_reference_text_and_prior_quality_scores_do_not_select_correspondence(self):
        frozen = copy.deepcopy(self.record["inputs"][1]["frozen_state"])
        cell = copy.deepcopy(self.record["cells"][1])
        before = copy.deepcopy((frozen, cell))
        expected = replay.diagnose_cell(frozen, cell)
        self.assertEqual((frozen, cell), before)
        frozen["committed_text"] = "unrelated text"
        cell["reference_text"] = "unrelated reference"
        cell["summary"]["against_human_reference"] = {"word_edit_distance": 0}
        cell["summary"]["proposed_text"] = "invented proposal"
        self.assertEqual(replay.diagnose_cell(frozen, cell), expected)

    def test_repeated_anchor_cannot_select_a_convenient_suffix(self):
        anchor = (NativeTimestampSegment(AudioSpan(100, 200), " word", (1,)),)
        words = anchor + (replace(anchor[0], span=AudioSpan(200, 300)),)
        observed = NativeWordAlignment(
            NativeWindowResult(
                window_id="synthetic", text="word word", start_ms=0, end_ms=500
            ),
            words,
        )
        result = replay._correspondence(anchor, observed, observed, 200)
        self.assertEqual(result["exact_anchor_occurrences"], [0, 1])
        self.assertIsNone(result["complete_suffix_comparison"])
        self.assertIsNone(result["boundary_checks"])

    def test_punctuation_only_and_empty_estimates_do_not_imply_speech_or_silence(self):
        for words, text in (
            ((), ""),
            ((NativeTimestampSegment(AudioSpan(0, 500), ".", (1,)),), "."),
        ):
            alignment = NativeWordAlignment(
                NativeWindowResult(
                    window_id="synthetic", text=text, start_ms=0, end_ms=500
                ),
                words,
            )
            result = replay._lexical_extent(alignment, 100)
            self.assertEqual(result["lexical_unit_count"], 0)
            self.assertIsNone(result["last_lexical_estimate_end_ms"])
            self.assertIsNone(result["trailing_nonlexical_estimate_ms"])


class GuardCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.anchor = (
            NativeTimestampSegment(AudioSpan(1000, 1500), " old", (1,)),
            NativeTimestampSegment(AudioSpan(1500, 2000), " words", (2,)),
        )
        self.suffix = (NativeTimestampSegment(AudioSpan(2100, 2400), " Next", (3,)),)
        self.frozen = dict(
            retained_ms=0,
            head_ms=2000,
            end_ms=6000,
            anchor=[asdict(w) for w in self.anchor],
        )

    def raw(self, words, start):
        return asdict(
            NativeWordAlignment(
                NativeWindowResult(
                    window_id=f"synthetic:{start}",
                    text="".join(w.text for w in words).strip(),
                    start_ms=start,
                    end_ms=6000,
                ),
                words,
            )
        )

    def assess(self, words=None, candidate=None):
        return replay.summarize_guard_correspondence(
            self.frozen,
            self.raw(self.anchor + self.suffix if words is None else words, 500),
            self.raw(self.suffix if candidate is None else candidate, 2000),
        )

    def test_strict_complete_agreement_never_authorizes_publication(self):
        result = self.assess()
        self.assertEqual(result["status"], "structurally_eligible")
        self.assertFalse(result["publication_authorized"])
        # Both observations could omit the same intervening spoken word.
        # There is no acoustic witness in these synthetic alignments.
        self.assertNotIn("acoustic_coverage", result)

    def test_guard_and_head_intervals_are_fixed(self):
        observed, candidate = (
            self.raw(self.anchor + self.suffix, 500),
            self.raw(self.suffix, 2000),
        )
        for wrong_observed, wrong_candidate in (
            (self.raw(self.anchor + self.suffix, 520), candidate),
            (observed, self.raw(self.suffix, 2020)),
        ):
            with self.assertRaisesRegex(ValueError, "interval differs"):
                replay.summarize_guard_correspondence(
                    self.frozen, wrong_observed, wrong_candidate
                )

    def test_casefold_and_token_mismatch_still_reject_the_complete_suffix(self):
        for changed in (
            replace(self.suffix[0], text=" next"),
            replace(self.suffix[0], tokens=(9,)),
        ):
            with self.subTest(changed=changed):
                result = self.assess(candidate=(changed,))
                self.assertEqual(result["reason"], "complete_suffix_disagrees")
                self.assertFalse(result["publication_authorized"])

    def test_repeated_raw_anchor_is_not_resolved_by_timing(self):
        repeated = tuple(
            replace(w, span=AudioSpan(w.span.start_ms + 2000, w.span.end_ms + 2000))
            for w in self.anchor
        )
        result = self.assess(self.anchor + self.suffix + repeated)
        self.assertEqual(result["reason"], "overlap_anchor_ambiguous")
        self.assertIsNone(result["diagnostics"]["complete_suffix_comparison"])

    def test_punctuation_and_empty_suffixes_cannot_pass_vacuously(self):
        for suffix in ((), (replace(self.suffix[0], text="."),)):
            with self.subTest(suffix=suffix):
                result = self.assess(self.anchor + suffix, suffix)
                self.assertEqual(
                    result["reason"], "overlap_has_no_lexical_continuation"
                )

    def test_anchor_timing_and_both_boundary_crossings_stay_separate(self):
        shifted = tuple(
            replace(w, span=AudioSpan(w.span.start_ms + 201, w.span.end_ms + 201))
            for w in self.anchor
        )
        late_suffix = (replace(self.suffix[0], span=AudioSpan(2400, 2600)),)
        cases = (
            (shifted + late_suffix, late_suffix, "overlap_anchor_timing_mismatch"),
            (
                self.anchor[:-1]
                + (
                    replace(self.anchor[-1], span=AudioSpan(1500, 1950)),
                    replace(self.suffix[0], span=AudioSpan(1950, 2400)),
                ),
                self.suffix,
                "overlap_continuation_crosses_boundary",
            ),
            (
                self.anchor[:-1]
                + (replace(self.anchor[-1], span=AudioSpan(1500, 2050)),)
                + self.suffix,
                (replace(self.suffix[0], span=AudioSpan(2000, 2400)),),
                "head_continuation_crosses_observed_anchor",
            ),
        )
        for observed, candidate, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(self.assess(observed, candidate)["reason"], reason)

    def test_suffix_timing_has_the_same_200_ms_limit(self):
        for offset, reason in (
            (200, "strict_overlap_and_suffix_agree"),
            (201, "suffix_timing_mismatch"),
        ):
            candidate = (
                replace(self.suffix[0], span=AudioSpan(2100 + offset, 2400 + offset)),
            )
            with self.subTest(offset=offset):
                self.assertEqual(self.assess(candidate=candidate)["reason"], reason)

    def test_frozen_bounds_and_anchor_cannot_be_rewritten(self):
        for key, value in (
            ("head_ms", True),
            ("retained_ms", -1),
            ("head_ms", 2001),
            ("anchor", []),
        ):
            changed = {**self.frozen, key: value}
            with (
                self.subTest(key=key, value=value),
                self.assertRaises((ValueError, TypeError)),
            ):
                replay.summarize_guard_correspondence(
                    changed,
                    self.raw(self.anchor + self.suffix, 500),
                    self.raw(self.suffix, 2000),
                )


if __name__ == "__main__":
    unittest.main()
