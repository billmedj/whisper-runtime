"""CPU-only group diagnostics: invented estimates and unchanged archive replay."""

import copy
import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.analyze_group_correspondence import (
    analyze_bytes,
    assess_group_correspondence,
)
from tools.analyze_resolution_disagreements import summarize_guard_correspondence
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeTimestampSegment, NativeWordAlignment
from whisper_runtime.adapters.native_result import NativeWindowResult


def word(start, end, text, token):
    return NativeTimestampSegment(AudioSpan(start, end), text, (token,))


def raw(words, start):
    return asdict(
        NativeWordAlignment(
            NativeWindowResult(
                window_id=f"synthetic:{start}",
                text="".join(w.text for w in words).strip(),
                start_ms=start,
                end_ms=6000,
            ),
            tuple(words),
        )
    )


class GroupCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.anchor = (word(1000, 1500, " old", 1), word(1500, 2000, " words", 2))
        self.suffix = (word(2100, 2400, " Next", 3), word(2450, 2700, " comes", 4))
        self.frozen = dict(
            retained_ms=0,
            head_ms=2000,
            end_ms=6000,
            anchor=[asdict(w) for w in self.anchor],
        )

    def assess(self, observed=None, candidate=None):
        inputs = (
            self.frozen,
            raw(self.anchor + self.suffix if observed is None else observed, 500),
            raw(self.suffix if candidate is None else candidate, 2000),
        )
        before = copy.deepcopy(inputs)
        result = assess_group_correspondence(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(result, assess_group_correspondence(*inputs))
        self.assertEqual(
            result["strict_assessment"], summarize_guard_correspondence(*inputs)
        )
        self.assertFalse(result["publication_authorized"])
        for name in (
            "strict_assessment",
            "group_strict_boundary",
            "group_live_boundary",
        ):
            self.assertFalse(result[name]["publication_authorized"])
        return result

    def reason(self, observed=None, candidate=None, expected="group_and_suffix_agree"):
        result = self.assess(observed, candidate)
        for name in ("group_strict_boundary", "group_live_boundary"):
            self.assertEqual(result[name]["reason"], expected)
            self.assertEqual(
                result[name]["status"],
                "structurally_eligible"
                if expected == "group_and_suffix_agree"
                else "rejected",
            )
        return result

    def test_interior_word_drift_is_only_a_group_diagnostic(self):
        drift = (word(1000, 1800, " old", 1), word(1800, 2000, " words", 2))
        result = self.reason(drift + self.suffix)
        self.assertEqual(
            result["strict_assessment"]["reason"], "overlap_anchor_timing_mismatch"
        )
        self.assertEqual(
            result["diagnostics"],
            dict(
                anchor_start_delta_ms=0,
                anchor_end_delta_ms=0,
                max_interior_delta_ms=300,
                origin_start_exception=False,
                observed_overlap_ms=0,
                candidate_overlap_ms=0,
            ),
        )

    def test_exact_anchor_order_text_and_tokens_are_indispensable(self):
        alternatives = (
            self.anchor[:1],
            (word(1000, 1500, " words", 2), word(1500, 2000, " old", 1)),
            (replace(self.anchor[0], text=" Old"), self.anchor[1]),
            (replace(self.anchor[0], tokens=(99,)), self.anchor[1]),
            (
                word(1000, 1200, " old", 1),
                word(1200, 1300, " inserted", 9),
                word(1500, 2000, " words", 2),
            ),
        )
        for anchor in alternatives:
            with self.subTest(anchor=anchor):
                result = self.reason(
                    anchor + self.suffix, expected="overlap_anchor_absent"
                )
                self.assertIsNone(result["diagnostics"])
        repeated = (word(3000, 3500, " old", 1), word(3500, 4000, " words", 2))
        result = self.reason(
            self.anchor + self.suffix + repeated, expected="overlap_anchor_ambiguous"
        )
        self.assertIsNone(result["diagnostics"])

    def test_relocated_group_is_not_old_material_even_when_unique(self):
        for start in (2000, 2001):
            moved = (word(1900, start, " old", 1), word(start, 2100, " words", 2))
            self.reason(moved + self.suffix, expected="group_anchor_relocated")

    def test_outer_edges_keep_signed_deltas_and_inclusive_200_ms_limit(self):
        for offset in (-201, -200, 200, 201):
            with self.subTest(edge="start", offset=offset):
                result = self.reason(
                    (word(1000 + offset, 1500, " old", 1), self.anchor[1])
                    + self.suffix,
                    expected="group_anchor_start_mismatch"
                    if abs(offset) > 200
                    else "group_and_suffix_agree",
                )
                self.assertEqual(result["diagnostics"]["anchor_start_delta_ms"], offset)
            with self.subTest(edge="end", offset=offset):
                suffix = (word(2400, 2700, " Next", 3),)
                result = self.reason(
                    (self.anchor[0], word(1500, 2000 + offset, " words", 2)) + suffix,
                    suffix,
                    "group_anchor_end_mismatch"
                    if abs(offset) > 200
                    else "group_and_suffix_agree",
                )
                self.assertEqual(result["diagnostics"]["anchor_end_delta_ms"], offset)

    def test_only_existing_window_origin_first_word_exception_is_permitted(self):
        result = self.reason((word(500, 1500, " old", 1), self.anchor[1]) + self.suffix)
        self.assertTrue(result["diagnostics"]["origin_start_exception"])
        self.assertEqual(result["diagnostics"]["anchor_start_delta_ms"], -500)
        for anchor in (
            (word(600, 1500, " old", 1), self.anchor[1]),
            (word(500, 700, " prefix", 9), word(700, 1500, " old", 1), self.anchor[1]),
            (word(500, 1701, " old", 1), word(1701, 2000, " words", 2)),
        ):
            result = self.reason(
                anchor + self.suffix, expected="group_anchor_start_mismatch"
            )
            self.assertFalse(result["diagnostics"]["origin_start_exception"])

    def test_complete_suffix_cannot_be_shortened_normalized_or_spliced(self):
        for candidate in (
            (),
            self.suffix[:1],
            self.suffix[1:],
            (replace(self.suffix[0], text=" next"), self.suffix[1]),
            (replace(self.suffix[0], tokens=(99,)), self.suffix[1]),
            (self.suffix[0], word(2400, 2450, " inserted", 8), self.suffix[1]),
            self.suffix + (word(2800, 2900, " extra", 9),),
        ):
            with self.subTest(candidate=candidate):
                self.reason(candidate=candidate, expected="complete_suffix_disagrees")
        for offset in (200, 201):
            candidate = self.suffix[:1] + (
                word(2450 + offset, 2700 + offset, " comes", 4),
            )
            self.reason(
                candidate=candidate,
                expected="suffix_timing_mismatch"
                if offset == 201
                else "group_and_suffix_agree",
            )

    def test_empty_punctuation_and_gaps_never_establish_acoustic_coverage(self):
        for suffix in ((), (word(2100, 5900, ".", 8),)):
            self.reason(
                self.anchor + suffix, suffix, "overlap_has_no_lexical_continuation"
            )
        suffix = (word(5000, 5500, " Next", 3),)
        result = self.reason(self.anchor + suffix, suffix)
        self.assertNotIn("acoustic_coverage", result)
        self.assertNotIn("silence", result)

    def test_structural_agreement_does_not_certify_duration_or_source_identity(self):
        anchor = (word(2000, 2000, " old", 1), word(2000, 2000, " words", 2))
        frozen = {**self.frozen, "anchor": [asdict(w) for w in anchor]}
        observed = raw(anchor + self.suffix, 1500)
        candidate = raw(self.suffix, 2000)
        for window_id in ("synthetic", "different-session-same-output"):
            observed["native"]["window_id"] = window_id
            result = assess_group_correspondence(frozen, observed, candidate)
            for key in (
                "strict_assessment",
                "group_strict_boundary",
                "group_live_boundary",
            ):
                self.assertEqual(result[key]["status"], "structurally_eligible")
                self.assertFalse(result[key]["publication_authorized"])
            # Neither the old nor new output-only rule authenticates a source or
            # proves a nonzero acoustic word interval. A producer must do that.
            self.assertFalse(result["publication_authorized"])

    def test_boundary_variants_are_separate_without_clipping_or_moving_head(self):
        for crossing in (20, 200, 201):
            observed = (
                self.anchor[0],
                word(1500, 2000 - crossing, " words", 2),
                word(2000 - crossing, 2400, " Next", 3),
            )
            result = self.assess(observed, (word(2000, 2400, " Next", 3),))
            self.assertEqual(result["diagnostics"]["observed_overlap_ms"], crossing)
            self.assertEqual(result["group_strict_boundary"]["status"], "rejected")
            # Ordered words cannot cross by 201 without violating the outer-end limit.
            self.assertEqual(
                result["group_live_boundary"]["reason"],
                "group_anchor_end_mismatch"
                if crossing == 201
                else "group_and_suffix_agree",
            )
            if crossing <= 200:
                self.assertEqual(
                    result["group_strict_boundary"]["reason"],
                    "overlap_continuation_crosses_boundary",
                )
        result = self.assess(
            (self.anchor[0], word(1500, 2050, " words", 2)) + self.suffix,
            (word(2000, 2400, " Next", 3), self.suffix[1]),
        )
        self.assertEqual(result["diagnostics"]["candidate_overlap_ms"], 50)
        self.assertEqual(
            result["group_strict_boundary"]["reason"],
            "head_continuation_crosses_observed_anchor",
        )
        self.assertEqual(
            result["group_live_boundary"]["status"], "structurally_eligible"
        )

    def test_fixed_spans_and_malformed_frozen_state_fail_validation(self):
        observed, candidate = (
            raw(self.anchor + self.suffix, 500),
            raw(self.suffix, 2000),
        )
        for left, right in (
            (raw(self.anchor + self.suffix, 520), candidate),
            (observed, raw(self.suffix, 2020)),
        ):
            with self.assertRaisesRegex(ValueError, "interval differs"):
                assess_group_correspondence(self.frozen, left, right)
        for key, value in (
            ("head_ms", True),
            ("head_ms", 2001),
            ("retained_ms", -1),
            ("anchor", []),
            ("anchor", self.frozen["anchor"][:1]),
            ("anchor", list(reversed(self.frozen["anchor"]))),
            ("anchor", [asdict(replace(w, tokens=())) for w in self.anchor]),
            ("anchor", [asdict(replace(w, text=".")) for w in self.anchor]),
            (
                "anchor",
                [asdict(word(i * 400, (i + 1) * 400, " w", i)) for i in range(5)],
            ),
        ):
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                assess_group_correspondence(
                    {**self.frozen, key: value}, observed, candidate
                )

    def test_references_do_not_affect_selection_and_outputs_do_not_alias_inputs(self):
        expected = self.assess()
        self.frozen.update(committed_text="unrelated", reference_text="different truth")
        result = self.assess()
        self.assertEqual(result, expected)
        result["strict_assessment"]["diagnostics"]["exact_anchor_occurrences"].append(
            99
        )
        result["diagnostics"]["anchor_end_delta_ms"] = 999
        self.assertEqual(self.assess(), expected)

    def test_module_import_and_assessment_need_no_inference_dependencies(self):
        script = """
import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'whisper', 'modal', 'numpy', 'soundfile'}:
        raise AssertionError('inference dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.analyze_group_correspondence import assess_group_correspondence
assert not assess_group_correspondence(*json.load(sys.stdin))['publication_authorized']
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            text=True,
            input=json.dumps(
                (
                    self.frozen,
                    raw(self.anchor + self.suffix, 500),
                    raw(self.suffix, 2000),
                )
            ),
            capture_output=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class GroupArchiveTests(unittest.TestCase):
    def test_cli_creates_output_once_and_preserves_existing_sentinel(self):
        root = Path(__file__).resolve().parents[1]
        archive = root / "evidence/modal-t4-tiny-en-context-guard-2026-09-06.json"
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            command = [
                sys.executable,
                "-B",
                "-m",
                "tools.analyze_group_correspondence",
                str(archive),
                "--output",
                str(output),
            ]
            first = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(
                json.loads(output.read_bytes()), analyze_bytes(archive.read_bytes())
            )
            output.write_bytes(b"existing report sentinel")
            second = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), b"existing report sentinel")

    def test_report_is_deterministic_hash_bound_and_never_an_inference_result(self):
        root = Path(__file__).resolve().parents[1]
        payload = (
            root / "evidence/modal-t4-tiny-en-context-guard-2026-09-06.json"
        ).read_bytes()
        report = analyze_bytes(payload)
        self.assertEqual(report, analyze_bytes(payload))
        self.assertEqual(report["input_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(report["native_windows_executed"], 0)
        self.assertFalse(report["publication_authorized"])
        self.assertFalse(report["full_stream_recovery"])
        for source, digest in report["source_sha256"].items():
            self.assertEqual(
                digest, hashlib.sha256((root / source).read_bytes()).hexdigest()
            )
        with self.assertRaisesRegex(ValueError, "fixed context-guard archive"):
            analyze_bytes(payload + b" ")

    def test_fixed_five_cells_compare_three_variants_without_changing_history(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "evidence/modal-t4-tiny-en-context-guard-2026-09-06.json"
        )
        record = json.loads(path.read_bytes())
        before = copy.deepcopy(record)
        self.assertEqual(len(record["cells"]), 5)
        for saved, cell in zip(record["inputs"], record["cells"]):
            with self.subTest(cell=cell["id"]):
                alignments = cell["raw_alignments"]
                result = assess_group_correspondence(
                    saved["frozen_state"], alignments["guard"], alignments["candidate"]
                )
                self.assertEqual(
                    result["strict_assessment"], cell["summary"]["guard_correspondence"]
                )
                for key in (
                    "strict_assessment",
                    "group_strict_boundary",
                    "group_live_boundary",
                ):
                    eligible = (
                        key == "group_live_boundary"
                        and cell["id"] == "heldout:2961-960-0004:clean"
                    )
                    self.assertEqual(
                        result[key]["status"],
                        "structurally_eligible" if eligible else "rejected",
                    )
                    self.assertFalse(result[key]["publication_authorized"])
                if cell["id"] == "heldout:2961-960-0004:clean":
                    self.assertEqual(result["diagnostics"]["observed_overlap_ms"], 20)
                    self.assertEqual(
                        result["diagnostics"]["max_interior_delta_ms"], 300
                    )
                    self.assertEqual(
                        result["group_strict_boundary"]["reason"],
                        "overlap_continuation_crosses_boundary",
                    )
                self.assertFalse(result["publication_authorized"])
        self.assertEqual(record, before)


if __name__ == "__main__":
    unittest.main()
