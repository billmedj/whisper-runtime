"""Pure synthetic routing/bootstrap checks; no native model or GPU execution."""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import asdict, replace

from infra.word_resolution_worker import _bootstrap, _evaluate
from tools.test_word_anchor_reconciliation import _aligned, _word
from whisper_runtime import AudioSpan

PARAMETERS = {
    "timestamp_tolerance_ms": 200,
    "max_end_extension_ms": 1000,
    "bootstrap_holdback_ms": 1000,
    "left_context_ms": 2000,
}


def _difference(text, reference):
    # Deliberately simple scoring stub: these tests cover routing, not WER.
    return {"word_edit_distance": 0 if text == reference else 1}


class WordResolutionRoutingTests(unittest.TestCase):
    def setUp(self):
        self.anchor = (
            _word(1, 1000, 1400, " and"),
            _word(2, 1500, 1800, " bird"),
            _word(1, 1900, 2200, " and"),
            _word(3, 2500, 3000, " tree"),
        )
        observed = self.anchor[:-1] + (
            replace(self.anchor[-1], span=AudioSpan(2480, 3540)),
        )
        self.suffix = (_word(4, 3600, 3900, " continue"),)
        self.current = _aligned(observed + self.suffix)
        self.alternative = _aligned((_word(5, 3600, 3900, " another"),), start=3000)
        self.case = {
            "id": "synthetic-terminal",
            "split": "development",
            "head_ms": 3000,
            "retained_ms": 0,
            "end_ms": 10000,
            "frozen_anchor": self.anchor,
            "committed_text": "and bird and tree",
            "reference_text": "and bird and tree continue",
        }

    def evaluate(self, current=None, **changes):
        return _evaluate(
            {**self.case, **changes},
            self.current if current is None else current,
            self.alternative,
            PARAMETERS,
            _difference,
        )

    def test_eligible_shadow_routes_from_current_timing_diagnostic(self):
        result = self.evaluate()
        self.assertEqual(result["current_diagnostic"]["status"], "timing_mismatch")
        self.assertEqual(result["shadow_proposal"]["status"], "eligible")
        self.assertEqual(result["shadow_proposal"]["end_extension_ms"], 540)
        self.assertEqual(result["routed"]["arm"], "shadow")
        self.assertEqual(result["outcome"]["strict_reason"], "anchor_missing")
        self.assertFalse(result["arms"]["baseline"]["available"])
        self.assertEqual(
            result["arms"]["baseline"]["text"], self.case["committed_text"]
        )
        self.assertEqual(result["arms"]["shadow"]["suffix_text"], "continue")
        self.assertEqual(result["arms"]["alternative"]["suffix_text"], "another")

    def test_human_best_arm_does_not_choose_the_route(self):
        first = self.evaluate()
        for reference in (
            self.case["committed_text"],
            "and bird and tree another",
            "an unrelated reference",
        ):
            with self.subTest(reference=reference):
                result = self.evaluate(reference_text=reference)
                self.assertEqual(result["routed"], first["routed"])
                self.assertEqual(
                    result["current_diagnostic"], first["current_diagnostic"]
                )
        alternative_is_best = self.evaluate(reference_text="and bird and tree another")
        self.assertEqual(
            alternative_is_best["arms"]["alternative"]["against_human_reference"][
                "word_edit_distance"
            ],
            0,
        )
        self.assertFalse(alternative_is_best["routed"]["uses_reference"])
        self.assertEqual(alternative_is_best["routed"]["arm"], "shadow")

    def test_missing_anchor_routes_alternative_without_inventing_shadow(self):
        result = self.evaluate(_aligned(self.suffix))
        self.assertEqual(result["current_diagnostic"]["status"], "lexical_missing")
        self.assertEqual(result["routed"]["arm"], "alternative")
        self.assertFalse(result["arms"]["shadow"]["available"])
        self.assertEqual(result["arms"]["shadow"]["text"], self.case["committed_text"])
        self.assertEqual(result["arms"]["shadow"]["suffix_text"], "")

    def test_changed_tokens_route_alternative_despite_matching_text(self):
        words = (replace(self.current.words[0], tokens=(99,)),) + self.current.words[1:]
        result = self.evaluate(_aligned(words))
        self.assertEqual(result["current_diagnostic"]["status"], "token_mismatch")
        self.assertEqual(result["routed"]["arm"], "alternative")

    def test_ambiguous_anchors_keep_baseline(self):
        repeated = tuple(
            replace(
                word,
                span=AudioSpan(word.span.start_ms + 4000, word.span.end_ms + 4000),
            )
            for word in self.anchor
        )
        result = self.evaluate(_aligned(self.current.words + repeated))
        self.assertEqual(result["current_diagnostic"]["status"], "ambiguous")
        self.assertEqual(result["shadow_proposal"]["reason"], "anchor_ambiguous")
        self.assertEqual(result["routed"]["arm"], "baseline")
        self.assertFalse(result["arms"]["baseline"]["available"])

    def test_timing_without_eligible_shadow_keeps_baseline(self):
        words = (
            replace(self.current.words[0], span=AudioSpan(0, 1400)),
        ) + self.current.words[1:]
        result = self.evaluate(_aligned(words))
        self.assertEqual(result["current_diagnostic"]["status"], "timing_mismatch")
        self.assertEqual(result["shadow_proposal"]["reason"], "anchor_start_mismatch")
        self.assertEqual(result["routed"]["arm"], "baseline")

    def test_strict_match_preserves_baseline_and_not_needed_shadow(self):
        result = self.evaluate(_aligned(self.anchor + self.suffix))
        self.assertEqual(result["routed"]["arm"], "baseline")
        self.assertEqual(result["outcome"]["strict_reason"], "eof")
        self.assertEqual(result["shadow_proposal"]["status"], "not_needed")
        self.assertTrue(result["arms"]["baseline"]["available"])
        self.assertEqual(
            result["arms"]["shadow"]["text"], result["arms"]["baseline"]["text"]
        )

    def test_reproduction_is_a_comparison_gate_not_a_routing_oracle(self):
        matched = self.evaluate(expected_native_text=self.current.native.text)
        mismatched = self.evaluate(expected_native_text="different saved result")
        self.assertTrue(matched["outcome"]["comparison_valid"])
        self.assertFalse(mismatched["outcome"]["comparison_valid"])
        self.assertEqual(matched["routed"], mismatched["routed"])
        self.assertIsNone(self.evaluate()["outcome"]["expected_native_text_reproduced"])

    def test_expected_words_normalize_tuples_to_json_and_require_full_identity(self):
        expected = json.loads(json.dumps([asdict(word) for word in self.current.words]))
        supplied = dict(
            expected_native_text=self.current.native.text,
            expected_words=expected,
            expected_strict_reason="anchor_missing",
        )
        matched = self.evaluate(**supplied)
        for key in (
            "expected_native_text_reproduced",
            "expected_alignment_reproduced",
            "strict_reason_reproduced",
            "comparison_valid",
        ):
            self.assertTrue(matched["outcome"][key])
        tuple_tokens = self.evaluate(
            expected_words=[asdict(word) for word in self.current.words]
        )
        self.assertTrue(tuple_tokens["outcome"]["expected_alignment_reproduced"])
        variants = [expected[:-1], expected + [expected[-1]], list(reversed(expected))]
        for field, value in (("text", " And"), ("tokens", [99]), ("tokens", [True])):
            changed = copy.deepcopy(expected)
            changed[0][field] = value
            variants.append(changed)
        for field, value in (
            ("start_ms", 1001),
            ("end_ms", 1401),
            ("start_ms", 1000.0),
        ):
            changed = copy.deepcopy(expected)
            changed[0]["span"][field] = value
            variants.append(changed)
        for changed in variants:
            with self.subTest(expected_words=changed):
                result = self.evaluate(**{**supplied, "expected_words": changed})
                self.assertTrue(result["outcome"]["expected_native_text_reproduced"])
                self.assertTrue(result["outcome"]["strict_reason_reproduced"])
                self.assertFalse(result["outcome"]["expected_alignment_reproduced"])
                self.assertFalse(result["outcome"]["comparison_valid"])
                self.assertEqual(result["routed"], matched["routed"])
        self.assertEqual(
            expected,
            json.loads(json.dumps([asdict(word) for word in self.current.words])),
        )

    def test_alignment_drift_invalidates_even_with_same_native_text_and_reason(self):
        words = (
            self.current.words[:-2]
            + (replace(self.current.words[-2], span=AudioSpan(2480, 3541)),)
            + self.current.words[-1:]
        )
        result = self.evaluate(
            _aligned(words),
            expected_native_text=self.current.native.text,
            expected_words=[asdict(word) for word in self.current.words],
            expected_strict_reason="anchor_missing",
        )
        self.assertTrue(result["outcome"]["expected_native_text_reproduced"])
        self.assertTrue(result["outcome"]["strict_reason_reproduced"])
        self.assertFalse(result["outcome"]["expected_alignment_reproduced"])
        self.assertFalse(result["outcome"]["comparison_valid"])
        self.assertEqual(result["routed"]["arm"], "shadow")

    def test_expected_reason_is_an_independent_post_routing_gate(self):
        supplied = dict(
            expected_native_text=self.current.native.text,
            expected_words=[asdict(word) for word in self.current.words],
            expected_strict_reason="anchor_missing",
        )
        matched = self.evaluate(**supplied)
        for key, value in (
            ("expected_native_text", "different native text"),
            ("expected_strict_reason", "eof"),
            ("expected_strict_reason", None),
            ("expected_words", None),
            ("expected_native_text", None),
        ):
            with self.subTest(field=key, value=value):
                result = self.evaluate(**{**supplied, key: value})
                self.assertFalse(result["outcome"]["comparison_valid"])
                self.assertEqual(result["routed"], matched["routed"])
        omitted = self.evaluate()["outcome"]
        self.assertIsNone(omitted["expected_alignment_reproduced"])
        self.assertIsNone(omitted["strict_reason_reproduced"])
        self.assertTrue(omitted["comparison_valid"])

    def test_outputs_are_proposals_and_plain_json_not_stream_completion(self):
        before = copy.deepcopy(self.case)
        result = self.evaluate()
        self.assertEqual(self.case, before)
        self.assertFalse(result["outcome"]["full_stream_completion"])
        self.assertTrue(
            all(not arm["publication_authority"] for arm in result["arms"].values())
        )
        self.assertNotIn("events", result)
        self.assertEqual(json.loads(json.dumps(result))["routed"]["arm"], "shadow")
        timing = result["local_timing"]
        self.assertGreaterEqual(
            timing["total_routing_wall_ns"],
            timing["strict_wall_ns"]
            + timing["diagnostic_wall_ns"]
            + timing["shadow_wall_ns"],
        )


class WordResolutionBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "synthetic-heldout",
            "split": "heldout",
            "end_ms": 10000,
            "bootstrap_ms": 4000,
            "reference_text": "not used to construct the prefix",
        }

    def test_prefix_uses_only_past_contiguous_whole_words_and_keeps_punctuation(self):
        words = (
            _word(1, 100, 200, " Alpha"),
            _word(2, 250, 260, ","),
            _word(3, 700, 1100, " beta"),
            _word(4, 1500, 1700, "."),
            _word(5, 2900, 3100, " straddles"),
            _word(6, 3200, 3300, " future"),
        )
        alignment = _aligned(words, end=4000)
        result = _bootstrap(self.case, alignment, PARAMETERS)
        self.assertEqual(result["committed_text"], "Alpha, beta")
        self.assertEqual(result["head_ms"], 1100)
        self.assertEqual(result["retained_ms"], 0)
        self.assertEqual(result["frozen_anchor"], words[:3])
        self.assertEqual(alignment.words, words)
        self.assertNotIn("head_ms", self.case)

    def test_last_four_anchor_units_trim_leading_punctuation_only(self):
        words = (
            _word(1, 100, 200, " Alpha"),
            _word(2, 2200, 2221, ","),
            _word(3, 2251, 2280, " beta"),
            _word(4, 2300, 2700, " gamma"),
            _word(5, 2800, 3000, " delta"),
        )
        result = _bootstrap(self.case, _aligned(words, end=4000), PARAMETERS)
        self.assertEqual(result["frozen_anchor"], words[2:])
        self.assertEqual(result["head_ms"], 3000)
        self.assertEqual(result["retained_ms"], 1000)
        self.assertEqual(result["committed_text"], "Alpha, beta gamma delta")

    def test_retained_start_floors_to_twenty_ms_and_preserves_anchor(self):
        words = (_word(1, 251, 301, " Alpha"), _word(2, 400, 2801, " beta"))
        result = _bootstrap(self.case, _aligned(words, end=4000), PARAMETERS)
        self.assertEqual(result["retained_ms"], 240)
        self.assertLessEqual(
            result["retained_ms"], result["frozen_anchor"][0].span.start_ms
        )

    def test_unresolved_bootstrap_cannot_make_a_positive_prefix(self):
        for words in (
            (),
            (_word(1, 100, 200, "."),),
            (_word(1, 0, 0, " zero"),),
            (_word(1, 3100, 3200, " after cutoff"),),
        ):
            with self.subTest(words=words):
                self.assertIsNone(
                    _bootstrap(self.case, _aligned(words, end=4000), PARAMETERS)
                )

    def test_reference_change_cannot_change_bootstrap_state(self):
        words = (_word(1, 100, 200, " Alpha"), _word(2, 400, 2801, " beta"))
        current = _aligned(words, end=4000)
        first = _bootstrap(self.case, current, PARAMETERS)
        second = _bootstrap(
            {**self.case, "reference_text": "unrelated"}, current, PARAMETERS
        )
        for key in ("head_ms", "retained_ms", "frozen_anchor", "committed_text"):
            self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
