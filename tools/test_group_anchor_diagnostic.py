"""Group-anchor shadows preserve strict evidence and have no continuation authority."""

import copy
import json
import subprocess
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

from tools.analyze_group_correspondence import assess_group_correspondence
from tools.analyze_word_resolution import _alignment
from tools.word_anchor_reconciliation import diagnose_group_anchor
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeTimestampSegment, NativeWordAlignment
from whisper_runtime.adapters.native_result import NativeWindowResult
from whisper_runtime.adapters.word_policy import (
    compare_word_hypotheses,
    diagnose_word_anchor,
)


def word(start, end, text, token):
    return NativeTimestampSegment(AudioSpan(start, end), text, (token,))


def aligned(words, start=500):
    return NativeWordAlignment(
        NativeWindowResult(
            window_id="synthetic",
            text="".join(w.text for w in words).strip(),
            start_ms=start,
            end_ms=6000,
        ),
        tuple(words),
    )


class GroupAnchorDiagnosticTests(unittest.TestCase):
    anchor = (word(1000, 1500, " old", 1), word(1500, 2000, " words", 2))

    def check(self, words=None, anchor=None, head=2000, expected="matched"):
        anchor = self.anchor if anchor is None else anchor
        observation = aligned(anchor if words is None else words)
        before = copy.deepcopy((asdict(observation), anchor))
        strict = diagnose_word_anchor(
            observation, anchor=anchor, committed_through_ms=head
        )
        result = diagnose_group_anchor(
            observation, anchor=anchor, committed_through_ms=head
        )
        self.assertEqual(result["status"], expected)
        self.assertEqual(result["strict"], asdict(strict))
        self.assertEqual(
            result["strict"],
            asdict(
                diagnose_word_anchor(
                    observation, anchor=anchor, committed_through_ms=head
                )
            ),
        )
        self.assertEqual((asdict(observation), anchor), before)
        self.assertFalse(result["publication_authorized"])
        self.assertNotIn("publication", result)
        self.assertNotIn("continuation", result)
        return result

    def test_interior_drift_does_not_change_strict_comparison(self):
        words = (word(1000, 1800, " old", 1), word(1800, 2000, " words", 2))
        observation = aligned(words + (word(2100, 2500, " next", 3),))
        arguments = dict(anchor=self.anchor, committed_through_ms=2000, final=True)
        before = compare_word_hypotheses(None, observation, **arguments)
        result = self.check(observation.words)
        self.assertEqual(result["strict"]["status"], "timing_mismatch")
        self.assertEqual(result["max_interior_delta_ms"], 300)
        self.assertEqual(
            (result["outer_start_delta_ms"], result["outer_end_delta_ms"]), (0, 0)
        )
        self.assertEqual(
            compare_word_hypotheses(None, observation, **arguments), before
        )
        self.assertIsNone(before.publication)

    def test_global_repetition_rejects_one_strict_timed_occurrence(self):
        later = (word(3000, 3500, " old", 1), word(3500, 4000, " words", 2))
        result = self.check(self.anchor + later, expected="ambiguous")
        self.assertEqual(result["strict"]["status"], "matched")
        self.assertEqual(result["strict"]["lexical_match_count"], 2)
        self.assertEqual(result["strict"]["timed_match_count"], 1)
        self.assertIsNone(result["word_start"])
        self.assertIsNone(result["word_end"])

    def test_raw_text_order_and_tokens_remain_exact(self):
        for first, expected in (
            (word(1000, 1500, " Old", 1), "lexical_missing"),
            (word(1000, 1500, " old", 99), "token_mismatch"),
        ):
            self.check((first, self.anchor[1]), expected=expected)
        self.check(
            (word(1000, 1500, " words", 2), word(1500, 2000, " old", 1)),
            expected="lexical_missing",
        )

    def test_origin_exception_requires_origin_and_first_end(self):
        result = self.check((word(500, 1500, " old", 1), self.anchor[1]))
        self.assertTrue(result["origin_start_exception"])
        self.assertEqual(result["outer_start_delta_ms"], -500)
        for words in (
            (word(600, 1500, " old", 1), self.anchor[1]),
            (word(500, 1701, " old", 1), word(1701, 2000, " words", 2)),
        ):
            self.assertFalse(
                self.check(words, expected="timing_mismatch")["origin_start_exception"]
            )

    def test_outer_200_is_inclusive_and_201_rejects(self):
        for offset in (-201, -200, 200, 201):
            expected = "matched" if abs(offset) <= 200 else "timing_mismatch"
            self.check(
                (word(1000 + offset, 1500, " old", 1), self.anchor[1]),
                expected=expected,
            )
            self.check(
                (self.anchor[0], word(1500, 2000 + offset, " words", 2)),
                expected=expected,
            )

    def test_a_unique_later_phrase_is_relocated(self):
        for start in (2000, 2001):
            self.check(
                (word(1900, start, " old", 1), word(start, 2100, " words", 2)),
                expected="relocated",
            )

    def test_empty_single_and_punctuation_edge_anchors(self):
        self.check(anchor=(), expected="unavailable")
        for anchor in (
            (word(1000, 2000, " old", 1),),
            (word(900, 1000, ".", 9),) + self.anchor,
            self.anchor + (word(2000, 2000, ".", 9),),
        ):
            self.check(anchor=anchor, expected="insufficient_anchor")
        self.check(head=2001, expected="insufficient_anchor")

    def test_zero_duration_and_foreign_id_do_not_gain_authority(self):
        anchor = (word(2000, 2000, " old", 1), word(2000, 2000, " words", 2))
        result = self.check(anchor=anchor)
        self.assertEqual(result["strict"]["status"], "matched")
        raw = asdict(aligned(anchor))
        raw["native"]["window_id"] = "another-session"
        self.assertEqual(
            diagnose_group_anchor(
                _alignment(raw), anchor=anchor, committed_through_ms=2000
            ),
            result,
        )

    def test_invalid_heads_and_unordered_anchors_fail_validation(self):
        for head in (True, -1, 1999, 6001, 2000.0):
            with self.subTest(head=head), self.assertRaises((TypeError, ValueError)):
                diagnose_group_anchor(
                    aligned(self.anchor), anchor=self.anchor, committed_through_ms=head
                )
        with self.assertRaises(ValueError):
            diagnose_group_anchor(
                aligned(self.anchor),
                anchor=tuple(reversed(self.anchor)),
                committed_through_ms=2000,
            )

    def test_import_requires_no_model_or_array_runtime(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'whisper', 'modal', 'numpy', 'numba'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.word_anchor_reconciliation import diagnose_group_anchor
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_calibration_arms_preserve_strict_and_guard_anchor_geometry(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "evidence/modal-t4-tiny-en-context-guard-2026-09-06.json"
        )
        record = json.loads(path.read_bytes())
        expected = (
            "matched",
            "matched",
            "timing_mismatch",
            "matched",
            "lexical_missing",
        )
        checked = 0
        for saved, cell, status in zip(record["inputs"], record["cells"], expected):
            frozen, raw = saved["frozen_state"], cell["raw_alignments"]
            anchor = tuple(
                NativeTimestampSegment(
                    AudioSpan(**w["span"]), w["text"], tuple(w["tokens"])
                )
                for w in frozen["anchor"]
            )
            for arm in ("current", "overlap", "guard"):
                observation = _alignment(raw[arm])
                arguments = dict(anchor=anchor, committed_through_ms=frozen["head_ms"])
                result = diagnose_group_anchor(observation, **arguments)
                self.assertEqual(
                    result["strict"],
                    asdict(diagnose_word_anchor(observation, **arguments)),
                )
                self.assertFalse(result["publication_authorized"])
                checked += 1
            self.assertEqual(result["status"], status)
            original = assess_group_correspondence(
                frozen, raw["guard"], raw["candidate"]
            )
            if original["diagnostics"] is not None:
                for new, old in (
                    ("outer_start_delta_ms", "anchor_start_delta_ms"),
                    ("outer_end_delta_ms", "anchor_end_delta_ms"),
                    ("max_interior_delta_ms", "max_interior_delta_ms"),
                    ("origin_start_exception", "origin_start_exception"),
                ):
                    self.assertEqual(result[new], original["diagnostics"][old])
        self.assertEqual(checked, 15)


if __name__ == "__main__":
    unittest.main()
