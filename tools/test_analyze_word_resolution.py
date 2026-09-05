"""Local-only tests for frozen, event-confirmed terminal word rematching."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

from tools import analyze_word_resolution as analyzer
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.native_stream import StreamEventKind, TranscriptEvent
from whisper_runtime.adapters.word_policy import (
    NativeWordAlignment,
    select_word_publication,
)
from whisper_runtime.state import AudioSpan


def observation(index, end, *, start=0, head=0, extension=0, eof=False):
    words = tuple(
        NativeTimestampSegment(AudioSpan(a, b), text, (token,))
        for a, b, text, token in (
            (1000, 1200, ".", 13),
            (1200, 1600, " alpha", 1),
            (1600, 2000 + extension, " beta", 2),
            (2500, 2800, " gamma", 3),
        )
        if a >= start
    )
    native = NativeWindowResult(
        window_id=f"window-{index}",
        text="".join(word.text for word in words).strip(),
        start_ms=start,
        end_ms=end,
        metadata=NativeDecodeMetadata(
            language="en",
            tokens=tuple(token for word in words for token in word.tokens),
            segments=(),
            timestamps_complete=False,
        ),
    )
    alignment = NativeWordAlignment(native, words)
    return {
        "decode_index": index,
        "analysis_start_sample": start * 16,
        "analysis_end_sample": end * 16,
        "committed_before_sample": head * 16,
        "retained_from_sample": start * 16,
        "accepted_through_sample": end * 16,
        "eof": eof,
        "reason": "anchor_missing" if head else "incomplete",
        "action": "unresolved" if eof else "preview",
        "publication_span": None,
        "result": asdict(native),
        "word_alignment": asdict(alignment),
        "word_publication": None,
        "source_unit": None
        if not eof
        else {
            "origin": "end_of_input",
            "start_sample": head * 16,
            "end_sample": end * 16,
            "endpoint": None,
        },
    }


def fixture(*, profile="context", eof=True, pair=True, origin=1000):
    before = observation(1, 4000)
    commit = observation(2, 5000)
    alignment = analyzer._alignment(commit["word_alignment"])
    publication = select_word_publication(alignment, 0, 3, AudioSpan(0, 2000))
    commit.update(
        action="commit", reason="candidate", word_publication=asdict(publication)
    )
    current_end = 7000 if eof else origin + 30000
    traces = [before, commit]
    if pair:
        traces.append(
            observation(3, current_end - 1000, start=origin, head=2000, extension=300)
        )
    traces.append(
        observation(
            len(traces) + 1,
            current_end,
            start=origin,
            head=2000,
            extension=400,
            eof=eof,
        )
    )
    common = dict(
        segment_id="segment-0",
        revision=1,
        start_sample=0,
        end_sample=32000,
        sample_rate_hz=16000,
        session_version=1,
    )
    events = [
        asdict(
            TranscriptEvent(
                1, StreamEventKind.PROVISIONAL, text=publication.text, **common
            )
        ),
        asdict(
            TranscriptEvent(
                2,
                StreamEventKind.COMMIT,
                committed_through_sample=32000,
                committed_through_ms=2000,
                **common,
            )
        ),
    ]
    cell = {
        "id": f"{profile}:fixture",
        "profile": profile,
        "profile_id": "word_boundary_quiet_endpoint_stream/v1"
        + ("+word_context/v1" if profile == "context" else "")
        + "+input_evidence/v1",
        "stream_status": "policy_failure",
        "qualified": False,
        "error": {"category": "policy_resolution"},
        "decision_traces": traces,
        "events": events,
        "case": {"sample_count": (current_end if eof else current_end + 1000) * 16},
        "metrics": {
            "committed_samples": 32000,
            "accepted_samples": current_end * 16,
            "decode_count": len(traces),
        },
    }
    return {"schema_version": "1-diagnostic", "qualified": False, "cells": [cell]}


class WordResolutionReplayTests(unittest.TestCase):
    def analyze(self, record):
        return analyzer.analyze_bytes(json.dumps(record).encode())

    def test_valid_replay_is_bound_read_only_and_has_no_authority(self):
        record = fixture()
        payload = json.dumps(record).encode()
        original = copy.deepcopy(record)
        report = analyzer.analyze_bytes(payload)
        self.assertEqual(record, original)
        self.assertEqual(report["input_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(report["schema_version"], "word-resolution-replay/v1")
        self.assertFalse(any(report["claim_boundary"].values()))
        for name, digest in report["replay_source_sha256"].items():
            self.assertEqual(
                digest,
                hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / name).read_bytes()
                ).hexdigest(),
            )
        cell = report["cells"][0]
        self.assertEqual(cell["replayed_reason"], "anchor_missing")
        self.assertTrue(cell["legacy_reason_matches"])
        self.assertTrue(cell["same_continuation_pair"])
        self.assertEqual(cell["previous_decode_index"], 3)
        self.assertEqual(cell["current_diagnostic"]["status"], "timing_mismatch")
        self.assertEqual(cell["shadow_proposal"]["status"], "eligible")
        self.assertEqual(cell["shadow_proposal"]["end_extension_ms"], 400)
        self.assertFalse(cell["recorded_qualified"])

    def test_context_trims_only_leading_standalone_punctuation(self):
        contextual = self.analyze(fixture())["cells"][0]
        hybrid = self.analyze(fixture(profile="hybrid"))["cells"][0]
        self.assertEqual(
            [w["text"] for w in contextual["frozen_anchor"]], [" alpha", " beta"]
        )
        self.assertEqual(
            [w["text"] for w in hybrid["frozen_anchor"]], [".", " alpha", " beta"]
        )
        self.assertEqual(contextual["frozen_anchor_origin"]["word_start"], 0)
        self.assertEqual(contextual["frozen_anchor_origin"]["word_end"], 3)

    def test_retained_filter_drops_only_words_outside_analysis(self):
        cell = self.analyze(fixture(profile="hybrid", origin=1200))["cells"][0]
        self.assertEqual(len(cell["frozen_anchor"]), 3)
        self.assertEqual(
            [w["text"] for w in cell["retained_anchor"]], [" alpha", " beta"]
        )
        self.assertEqual(cell["shadow_proposal"]["status"], "eligible")

    def test_commit_is_not_an_observation_of_the_rebased_continuation(self):
        cell = self.analyze(fixture(pair=False))["cells"][0]
        self.assertIsNone(cell["previous_decode_index"])
        self.assertIsNone(cell["previous_diagnostic"])
        self.assertFalse(cell["same_continuation_pair"])

    def test_window_full_is_not_eof_or_shadow_completion_authority(self):
        cell = self.analyze(fixture(eof=False))["cells"][0]
        self.assertEqual(cell["terminal_kind"], "window_full")
        self.assertEqual(cell["shadow_proposal"]["status"], "rejected")
        self.assertEqual(cell["shadow_proposal"]["reason"], "not_final")

    def test_every_terminal_failure_is_included_and_successes_are_explicitly_excluded(
        self,
    ):
        record = fixture()
        record["cells"].append(fixture(profile="hybrid", eof=False)["cells"][0])
        record["cells"].append({"id": "quiet:success", "stream_status": "completed"})
        report = self.analyze(record)
        self.assertEqual(report["input_cell_count"], 3)
        self.assertEqual(report["terminal_failure_count"], 2)
        self.assertEqual(
            [c["id"] for c in report["cells"]], ["context:fixture", "hybrid:fixture"]
        )
        self.assertEqual(
            report["excluded_cells"],
            [{"id": "quiet:success", "stream_status": "completed"}],
        )

    def test_malformed_or_tampered_joins_are_rejected(self):
        for field in (
            "event_text",
            "event_span",
            "event_revision",
            "alignment_native",
            "publication_alignment",
            "publication_slice",
            "unconfirmed_commit",
            "action",
            "decode_order",
            "head",
            "metrics",
            "accepted",
            "false_eof",
            "unit_boundary",
            "terminal_alignment",
            "final_event",
            "qualified",
            "profile",
            "duplicate_join",
        ):
            record = fixture()
            cell = record["cells"][0]
            commit = cell["decision_traces"][1]
            terminal = cell["decision_traces"][-1]
            if field == "event_text":
                cell["events"][0]["text"] = "invented"
            elif field == "event_span":
                cell["events"][1]["end_sample"] += 16
            elif field == "event_revision":
                cell["events"][1]["revision"] = 2
            elif field == "alignment_native":
                commit["word_alignment"]["native"]["text"] = "invented"
            elif field == "publication_alignment":
                commit["word_publication"]["alignment"]["words"][0]["tokens"] = [900]
            elif field == "publication_slice":
                commit["word_publication"]["word_start"] = 1
            elif field == "unconfirmed_commit":
                cell["events"].pop()
            elif field == "action":
                commit["action"] = "preview"
            elif field == "decode_order":
                terminal["decode_index"] += 1
            elif field == "head":
                terminal["committed_before_sample"] += 16
            elif field == "metrics":
                cell["metrics"]["committed_samples"] += 16
            elif field == "accepted":
                terminal["accepted_through_sample"] -= 16
            elif field == "false_eof":
                cell["case"]["sample_count"] += 16
            elif field == "unit_boundary":
                terminal["source_unit"]["start_sample"] += 16
            elif field == "terminal_alignment":
                terminal["word_alignment"] = None
            elif field == "final_event":
                cell["events"].append({"kind": "final"})
            elif field == "profile":
                cell["profile"] = "hybrid"
            elif field == "duplicate_join":
                cell["decision_traces"][0] = copy.deepcopy(commit)
                cell["decision_traces"][0]["decode_index"] = 1
            else:
                cell["qualified"] = True
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                self.analyze(record)

    def test_invalid_schema_duplicate_ids_and_nonfinite_json_are_rejected(self):
        record = fixture()
        record["schema_version"] = "other"
        with self.assertRaises(ValueError):
            self.analyze(record)
        record = fixture()
        record["cells"].append(copy.deepcopy(record["cells"][0]))
        with self.assertRaises(ValueError):
            self.analyze(record)
        with self.assertRaises(ValueError):
            analyzer.analyze_bytes(b'{"schema_version": NaN}')

    def test_cli_writes_only_stdout_and_keeps_input_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evidence.json"
            payload = json.dumps(fixture()).encode()
            path.write_bytes(payload)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(analyzer.main([str(path)]), 0)
            self.assertEqual(json.loads(output.getvalue())["terminal_failure_count"], 1)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(list(Path(temp).iterdir()), [path])
            path.write_text("not json")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                analyzer.main([str(path)])

    def test_archived_all_four_failures_reconstruct_expected_observations(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "evidence/modal-t4-tiny-en-word-context-2026-09-06.json"
        )
        report = analyzer.analyze_bytes(path.read_bytes())
        self.assertEqual(report["terminal_failure_count"], 4)
        self.assertEqual(
            [c["terminal_decode_index"] for c in report["cells"]], [17, 18, 17, 22]
        )
        self.assertEqual(
            [c["previous_decode_index"] for c in report["cells"]], [16, 17, None, 21]
        )
        self.assertTrue(all(c["legacy_reason_matches"] for c in report["cells"]))
        self.assertEqual(
            [c["shadow_proposal"]["status"] for c in report["cells"]],
            ["rejected", "rejected", "rejected", "eligible"],
        )
        self.assertEqual(
            report["cells"][-1]["shadow_proposal"]["end_extension_ms"], 540
        )


if __name__ == "__main__":
    unittest.main()
