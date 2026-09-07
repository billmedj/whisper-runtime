"""Small synthetic guards plus an optional frozen-artifact count regression."""

import copy
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from tools import analyze_composed_decoder_cost as audit


def fixture():
    cells = []
    for arm in audit.ARMS:
        candidate = arm == "fast-reuse"
        tokens = [50363, 100, 50463] + ([101, 50464] if candidate else [])
        identity = f"composed:{arm}:window:0:32000:preview"
        window = {
            "start_ms": 0,
            "end_ms": 2000,
            "call_index": 10 if candidate else 1,
            "window_id": identity,
            "forwards": [{"kind": "encoder", "phase": "decode"}]
            + [{"kind": "decoder", "phase": "decode"} for _ in range(len(tokens) + 1)]
            + ([{"kind": "decoder", "phase": "alignment"}] if candidate else []),
        }
        trace = {
            "decode_index": 1,
            "analysis_start_sample": 0,
            "analysis_end_sample": 32000,
            "committed_before_sample": 0,
            "result": {
                "window_id": identity,
                "start_ms": 0,
                "end_ms": 2000,
                "analysis_span": {"start_ms": 0, "end_ms": 2000},
                "text": "alpha beta" if candidate else "alpha",
                "metadata": {"tokens": tokens},
            },
            "source_unit": None,
            "word_publication": None,
            "action": "preview",
            "reason": "incomplete",
            "eof": False,
        }
        cells.append(
            {
                "arm": arm,
                "input_sha256": "a" * 64,
                "windows": [window],
                "decision_traces": [trace],
            }
        )
    return {
        "cells": cells,
        "input": {"pcm_sha256": "a" * 64},
        "effective_identity": {"profile": {"profile_id": "tiny.en/composed-v1"}},
    }


class DecoderCostAuditTests(unittest.TestCase):
    def test_counts_use_raw_forwards_not_saved_summary(self):
        record = fixture()
        record["cells"][0]["summary"] = {"forwards": {"decoder": 99999}}
        report = audit.analyze(record)
        self.assertEqual(report["delta"]["decode_forwards"], 2)
        self.assertEqual(report["delta"]["alignment_forwards"], 1)
        self.assertEqual(report["delta"]["lexical_tokens"], 1)
        self.assertEqual(report["delta"]["timestamp_tokens"], 1)
        self.assertEqual(report["delta"]["special_tokens"], 0)
        self.assertEqual(report["changed_alignment_window_indices"], [0])
        self.assertFalse(report["windows"][0]["same_generated_text_and_tokens"])

    def test_arm_order_is_not_comparison_direction(self):
        record = fixture()
        record["cells"].reverse()
        report = audit.analyze(record)
        self.assertEqual(report["delta"]["decode_forwards"], 2)
        self.assertEqual(
            report["windows"][0]["control"]["trace_path"],
            "$.cells[1].decision_traces[0]",
        )

    def test_rejects_corrupt_accounting_and_join_identities(self):
        mutations = {
            "forward count": lambda c: c["windows"][0]["forwards"].pop(),
            "decode index": lambda c: c["decision_traces"][0].update(decode_index=2),
            "window identity": lambda c: c["windows"][0].update(window_id="other"),
            "trace count": lambda c: c["decision_traces"].clear(),
            "input identity": lambda c: c.update(input_sha256="b" * 64),
            "span identity": lambda c: c["decision_traces"][0].update(
                analysis_start_sample=16
            ),
            "result identity": lambda c: c["decision_traces"][0]["result"].update(
                window_id="other"
            ),
            "token ID": lambda c: c["decision_traces"][0]["result"]["metadata"][
                "tokens"
            ].__setitem__(0, -1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                record = fixture()
                mutate(record["cells"][0])
                with self.assertRaises(ValueError):
                    audit.analyze(record)

    def test_rejects_duplicate_windows_and_native_call_gaps(self):
        for mutation in ("duplicate_window", "native_call_gap"):
            record = fixture()
            for cell in record["cells"]:
                cell["windows"].append(copy.deepcopy(cell["windows"][0]))
                cell["decision_traces"].append(
                    copy.deepcopy(cell["decision_traces"][0])
                )
                cell["decision_traces"][1]["decode_index"] = 2
                cell["windows"][1]["call_index"] += (
                    2 if mutation == "native_call_gap" else 1
                )
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.analyze(record)

    def test_rejects_internally_consistent_but_unmatched_endpoint(self):
        record = fixture()
        cell = record["cells"][1]
        window, trace = cell["windows"][0], cell["decision_traces"][0]
        identity = "composed:fast-reuse:window:0:64000:preview"
        window.update(end_ms=4000, window_id=identity)
        trace["analysis_end_sample"] = 64000
        trace["result"].update(
            end_ms=4000,
            window_id=identity,
            analysis_span={"start_ms": 0, "end_ms": 4000},
        )
        with self.assertRaisesRegex(ValueError, "comparison endpoint identity"):
            audit.analyze(record)

    def test_rejects_duplicate_arms_and_unknown_tokenizer(self):
        for mutation in ("duplicate", "tokenizer"):
            record = fixture()
            if mutation == "duplicate":
                record["cells"].append(copy.deepcopy(record["cells"][0]))
            else:
                record["effective_identity"]["profile"]["profile_id"] = "tiny/other"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.analyze(record)

    def test_cli_json_and_fail_closed(self):
        with patch.object(audit.Path, "read_text", return_value=json.dumps(fixture())):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit.main([]), 0)
            self.assertEqual(
                json.loads(output.getvalue())["delta"]["decode_forwards"], 2
            )
        with patch.object(audit.Path, "read_text", return_value="{}"):
            with redirect_stderr(io.StringIO()) as error:
                self.assertEqual(audit.main([]), 1)
            self.assertIn("rejected", error.getvalue())

    @unittest.skipUnless(audit.DEFAULT_ARTIFACT.is_file(), "local artifact absent")
    def test_frozen_artifact_attribution(self):
        report = audit.analyze(json.loads(audit.DEFAULT_ARTIFACT.read_text("utf-8")))
        self.assertEqual(report["window_count_per_arm"], 25)
        self.assertEqual(report["totals"]["cli-legacy"]["decode_forwards"], 991)
        self.assertEqual(report["totals"]["fast-reuse"]["decode_forwards"], 1196)
        self.assertEqual(
            report["delta"], dict(zip(audit.COUNTS, (205, 11, 177, 28, 0)))
        )
        self.assertEqual(
            [row["delta"]["decode_forwards"] for row in report["windows"][12:18]],
            [-8, -6, 63, 60, 48, 48],
        )
        self.assertEqual(
            report["decode_delta_groups"]["windows_12_13"]["decode_forwards_delta"], -14
        )
        self.assertEqual(
            report["decode_delta_groups"]["windows_14_17"]["decode_forwards_delta"], 219
        )
        self.assertEqual(
            report["unchanged_decode_window_indices"], [*range(12), *range(18, 25)]
        )
        self.assertEqual(
            report["changed_alignment_window_indices"], [*range(3, 12), 21, 22]
        )
        self.assertEqual(report["eof_window_indices"], [24])


if __name__ == "__main__":
    unittest.main()
