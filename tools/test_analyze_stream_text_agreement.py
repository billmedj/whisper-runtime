"""Offline tests for event-confirmed, text-only diagnostic replay."""

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

from tools import analyze_stream_text_agreement as analyzer
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.native_stream import StreamEventKind, TranscriptEvent
from whisper_runtime.state import AudioSpan


def _trace(
    index: int,
    end: int,
    groups: tuple[tuple[int, ...], ...],
    *,
    start: int = 0,
    head: int = 0,
    spans: tuple[tuple[int, int], ...] | None = None,
    publication: tuple[int, int] | None = None,
    action: str = "preview",
    eof: bool = False,
    language: str = "en",
    complete: bool = True,
) -> dict:
    if spans is None:
        spans = tuple(
            (start + i * 20, start + (i + 1) * 20) for i in range(len(groups))
        )
    segments = tuple(
        NativeTimestampSegment(
            AudioSpan(*span), "".join(f" token-{token}" for token in tokens), tokens
        )
        for span, tokens in zip(spans, groups)
    )
    result = NativeWindowResult(
        window_id=f"window:{start * 16}:{end * 16}",
        text="".join(segment.text for segment in segments).strip(),
        start_ms=start,
        end_ms=end,
        metadata=NativeDecodeMetadata(
            language=language,
            tokens=(9999,),  # Raw timestamp/tail tokens must never count.
            segments=segments,
            timestamps_complete=complete,
        ),
    )
    return {
        "decode_index": index,
        "analysis_start_sample": start * 16,
        "analysis_end_sample": end * 16,
        "committed_before_sample": head * 16,
        "retained_from_sample": start * 16,
        "eof": eof,
        "reason": "eof" if eof else "candidate" if publication else "incomplete",
        "publication_span": asdict(AudioSpan(*publication)) if publication else None,
        "result": asdict(result),
        "action": action,
    }


def _commit(
    trace: dict, *, sequence: int = 1, segment: str = "segment-0"
) -> list[dict]:
    parsed = analyzer._trace(trace)
    publication = analyzer._publication(parsed)
    assert publication is not None
    end, text, _tokens = publication
    common = {
        "segment_id": segment,
        "revision": 1,
        "start_sample": parsed.head,
        "end_sample": end,
        "sample_rate_hz": 16_000,
        "session_version": 1,
    }
    return [
        asdict(
            TranscriptEvent(sequence, StreamEventKind.PROVISIONAL, text=text, **common)
        ),
        asdict(
            TranscriptEvent(
                sequence + 1,
                StreamEventKind.COMMIT,
                committed_through_sample=end,
                committed_through_ms=end // 16,
                **common,
            )
        ),
    ]


def _cell(
    traces: list[dict], events: list[dict] | None = None, *, name: str = "baseline"
) -> dict:
    return {
        "cell_id": name,
        "status": "policy_resolution",
        "decision_traces": traces,
        "events": events or [],
    }


def _payload(*cells: dict) -> bytes:
    return json.dumps({"schema_version": "1-diagnostic", "cells": cells}).encode(
        "utf-8"
    )


class StreamTextAgreementAnalysisTests(unittest.TestCase):
    def analyze(
        self, traces: list[dict], events: list[dict] | None = None, **options: int
    ) -> dict:
        return analyzer.analyze_bytes(_payload(_cell(traces, events)), **options)[
            "cells"
        ][0]

    def test_segmentation_changes_yield_text_candidate_without_timed_publication(
        self,
    ) -> None:
        before = _trace(1, 100, ((1, 2), (3,)), spans=((0, 40), (40, 80)))
        after = _trace(2, 200, ((1,), (2, 3, 4)), spans=((20, 60), (80, 160)))
        report = self.analyze([before, after])
        self.assertEqual(report["counts"]["candidate_tokens"], 3)
        self.assertEqual(
            report["counts"]["text_candidate_with_no_timed_publication"], 1
        )
        self.assertEqual(report["confirmed_commits"], [])
        evidence = report["evidence"][0]
        self.assertEqual(
            (evidence["previous_decode_index"], evidence["decode_index"]), (1, 2)
        )
        self.assertEqual(
            (evidence["current_token_start"], evidence["current_token_end"]), (0, 3)
        )

    def test_action_commit_without_event_does_not_advance_anchor(self) -> None:
        before = _trace(1, 100, ((1,), (2,)))
        intended = _trace(2, 200, ((1,), (2,)), publication=(0, 20), action="commit")
        after = _trace(3, 300, ((1,), (2,), (3,)))
        report = self.analyze([before, intended, after])
        self.assertEqual(report["counts"]["candidate_tokens"], 4)
        self.assertEqual(report["confirmed_commits"], [])
        self.assertEqual([item["anchor_tokens"] for item in report["evidence"]], [0, 0])
        self.assertEqual(
            report["counts"]["text_candidate_with_no_timed_publication"], 2
        )

    def test_anchor_uses_actual_publication_not_entire_text_candidate(self) -> None:
        before = _trace(1, 100, ((1,), (2,), (3,)))
        commit = _trace(
            2, 200, ((1,), (2,), (3,)), publication=(0, 20), action="commit"
        )
        after = _trace(3, 300, ((1,), (2,), (3,), (4,)), head=20)
        report = self.analyze([before, commit, after], _commit(commit))
        self.assertEqual(report["counts"]["candidate_tokens"], 5)
        first, second = report["evidence"]
        self.assertEqual(first["publication_kind"], "whole_segments")
        self.assertFalse(first["text_candidate_with_no_timed_publication"])
        self.assertEqual(second["anchor_tokens"], 1)
        self.assertEqual(second["anchor_commit_sequences"], [2])
        self.assertEqual(second["candidate_tokens"], 2)
        self.assertEqual(second["current_token_start"], 1)
        self.assertEqual(report["confirmed_commits"][0]["closed_publication_tokens"], 1)

    def test_missing_and_ambiguous_confirmed_anchor_are_reported_with_indices(
        self,
    ) -> None:
        commit = _trace(1, 100, ((7,), (8,)), publication=(0, 20), action="commit")
        for groups, expected in (
            (((8, 9),), "anchor_missing"),
            (((7, 8, 7, 9),), "anchor_ambiguous"),
        ):
            with self.subTest(expected=expected):
                after = _trace(2, 200, groups, head=20)
                report = self.analyze([commit, after], _commit(commit))
                self.assertEqual(report["counts"][expected], 1)
                self.assertEqual(report["counts"]["candidate_tokens"], 0)
                self.assertEqual(report["evidence"][0]["decode_index"], 2)
                self.assertEqual(report["evidence"][0]["anchor_commit_sequences"], [2])

    def test_commit_can_confirm_a_replacement_revision_with_shorter_span(self) -> None:
        before = _trace(1, 100, ((1,), (2,)))
        commit = _trace(2, 200, ((1,), (2,)), publication=(0, 20), action="commit")
        events = _commit(commit, sequence=2)
        events[0].update(kind="replace", revision=2, supersedes_revision=1)
        events[1]["revision"] = 2
        preview = asdict(
            TranscriptEvent(
                1,
                StreamEventKind.PROVISIONAL,
                segment_id="segment-0",
                revision=1,
                start_sample=0,
                end_sample=1600,
                sample_rate_hz=16_000,
                text=before["result"]["text"],
                session_version=0,
            )
        )
        report = self.analyze([before, commit], [preview] + events)
        self.assertEqual(report["confirmed_commits"][0]["revision"], 2)
        self.assertEqual(report["confirmed_commits"][0]["commit_sequence"], 3)

    def test_multiple_commits_advance_only_the_confirmed_bounded_suffix(self) -> None:
        first = _trace(1, 100, ((1,), (2,), (3,)), publication=(0, 20), action="commit")
        second = _trace(
            2, 200, ((1,), (2,), (3,)), head=20, publication=(20, 40), action="commit"
        )
        after = _trace(3, 300, ((1,), (2,), (3,), (4,)), head=40)
        events = _commit(first) + _commit(second, sequence=3, segment="segment-1")
        report = self.analyze([first, second, after], events, max_anchor_tokens=2)
        self.assertEqual(report["evidence"][0]["candidate_tokens"], 2)
        self.assertEqual(report["evidence"][1]["candidate_tokens"], 1)
        self.assertEqual(report["evidence"][1]["anchor_tokens"], 2)
        self.assertEqual(report["evidence"][1]["anchor_commit_sequences"], [2, 4])

    def test_unrecoverable_committed_tokens_never_become_an_empty_context_anchor(
        self,
    ) -> None:
        first = _trace(1, 100, ((1,),), action="commit", eof=True)
        first["result"]["metadata"] = None
        after = _trace(2, 200, ((1,), (2,)), head=100)
        # Even an anomalous trace after such an EOF must not reclassify old text
        # as unpublished. This is a text diagnostic, not a full lifecycle audit.
        report = self.analyze([first, after], _commit(first))
        self.assertEqual(report["compared_pairs"], 0)
        self.assertEqual(
            report["skipped"][1]["reason"], "committed_text_anchor_unavailable"
        )

    def test_suffix_anchor_is_bounded_and_can_cross_segment_boundaries(self) -> None:
        tokens = tuple(range(1, 36))
        commit = _trace(
            1, 100, (tokens[:20], tokens[20:]), publication=(0, 40), action="commit"
        )
        after = _trace(2, 200, (tokens + (40,),), head=40)
        for limit in (2, 32):
            with self.subTest(limit=limit):
                report = self.analyze(
                    [commit, after], _commit(commit), max_anchor_tokens=limit
                )
                self.assertEqual(report["evidence"][0]["anchor_tokens"], limit)
                self.assertEqual(report["evidence"][0]["reason"], "incomplete")
        next_trace = _trace(3, 300, (tokens + (40, 41),), head=40)
        report = self.analyze(
            [commit, after, next_trace], _commit(commit), max_anchor_tokens=2
        )
        self.assertEqual(report["evidence"][1]["candidate_tokens"], 1)

    def test_origin_rebase_starts_fresh_only_when_no_committed_context_remains(
        self,
    ) -> None:
        commit = _trace(1, 100, ((1,),), publication=(0, 20), action="commit")
        after = _trace(2, 200, ((2,),), start=20, head=20)
        growing = _trace(3, 300, ((2, 3),), start=20, head=20)
        report = self.analyze([commit, after, growing], _commit(commit))
        self.assertEqual(report["skipped"][1]["reason"], "different_analysis_origin")
        self.assertEqual(report["evidence"][0]["anchor_tokens"], 0)
        self.assertEqual(report["evidence"][0]["candidate_tokens"], 1)

    def test_equal_or_shrinking_analysis_end_is_not_a_pair(self) -> None:
        traces = [
            _trace(index, end, ((1,),))
            for index, end in enumerate((100, 100, 80, 200), 1)
        ]
        report = self.analyze(traces)
        self.assertEqual(report["compared_pairs"], 1)
        self.assertEqual(
            [item["decode_index"] for item in report["skipped"]], [1, 2, 3]
        )
        self.assertEqual(report["evidence"][0]["previous_decode_index"], 3)

    def test_empty_text_is_not_silence_and_missing_metadata_is_incomplete(self) -> None:
        for after in (_trace(2, 200, ()), _trace(2, 200, ((1,),))):
            with self.subTest(after=after):
                after["result"]["metadata"] = (
                    None if after["result"]["text"] else after["result"]["metadata"]
                )
                report = self.analyze([_trace(1, 100, ((1,),)), after])
                self.assertEqual(report["counts"]["incomplete"], 1)
                self.assertEqual(report["counts"]["candidate_tokens"], 0)

    def test_language_mismatch_is_unstable(self) -> None:
        report = self.analyze(
            [_trace(1, 100, ((1,),)), _trace(2, 200, ((1,),), language="fr")]
        )
        self.assertEqual(report["counts"]["unstable"], 1)

    def test_full_result_eof_is_not_reported_as_timed_selection(self) -> None:
        before = _trace(1, 100, ((1,),))
        eof = _trace(2, 200, ((1,),), action="commit", eof=True)
        events = _commit(eof)
        events.append(
            asdict(TranscriptEvent(3, StreamEventKind.FINAL, session_version=1))
        )
        report = self.analyze([before, eof], events)
        self.assertEqual(
            report["confirmed_commits"][0]["publication_kind"], "full_result_eof"
        )
        self.assertEqual(
            report["counts"]["text_candidate_with_no_timed_publication"], 1
        )
        self.assertEqual(report["evidence"][0]["candidate_tokens"], 1)

    def test_fractional_sample_eof_span_is_preserved_in_confirmation(self) -> None:
        eof = _trace(1, 200, ((1,),), action="commit", eof=True)
        eof["analysis_end_sample"] += 7
        report = self.analyze([eof], _commit(eof))
        self.assertEqual(report["confirmed_commits"][0]["end_sample"], 3207)

    def test_final_only_empty_cell_has_no_candidate_or_silence_claim(self) -> None:
        report = self.analyze(
            [], [asdict(TranscriptEvent(1, StreamEventKind.FINAL, session_version=0))]
        )
        self.assertEqual(report["compared_pairs"], 0)
        self.assertEqual(report["confirmed_commits"], [])

    def test_commit_must_match_text_revision_span_and_full_trace_text(self) -> None:
        commit = _trace(1, 100, ((1,),), publication=(0, 20), action="commit")
        for mutation in ("revision", "span", "text", "sequence", "missing_text"):
            with self.subTest(mutation=mutation):
                events = _commit(commit)
                if mutation == "revision":
                    events[1]["revision"] = 2
                elif mutation == "span":
                    events[0]["end_sample"] += 16
                elif mutation == "text":
                    events[0]["text"] = "different"
                elif mutation == "sequence":
                    events[0]["sequence_number"] = 2
                else:
                    events.pop(0)
                    events[0]["sequence_number"] = 1
                with self.assertRaises(ValueError):
                    self.analyze([commit], events)

    def test_duplicate_matching_traces_are_rejected_not_arbitrarily_confirmed(
        self,
    ) -> None:
        commit = _trace(1, 100, ((1,),), publication=(0, 20), action="commit")
        duplicate = copy.deepcopy(commit)
        duplicate["decode_index"] = 2
        with self.assertRaisesRegex(ValueError, "unique"):
            self.analyze([commit, duplicate], _commit(commit))

    def test_head_advance_without_confirming_commit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "watermark advanced"):
            self.analyze([_trace(1, 100, ((1,),), head=20)])

    def test_committed_segment_cannot_be_reemitted(self) -> None:
        commit = _trace(1, 100, ((1,),), publication=(0, 20), action="commit")
        events = _commit(commit)
        repeated = copy.deepcopy(events[0])
        repeated.update(sequence_number=3, start_sample=320, end_sample=640)
        events.append(repeated)
        with self.assertRaisesRegex(ValueError, "committed segment"):
            self.analyze([commit], events)

    def test_four_cells_are_independent_and_exact_input_hash_is_reported(self) -> None:
        traces = [_trace(1, 100, ((1,),)), _trace(2, 200, ((1, 2),))]
        payload = _payload(*[_cell(traces, name=f"cell-{index}") for index in range(4)])
        report = analyzer.analyze_bytes(payload)
        self.assertEqual(report["input_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(
            [cell["counts"]["candidate_tokens"] for cell in report["cells"]], [1] * 4
        )
        self.assertEqual(set(report["claim_boundary"].values()), {False})
        self.assertNotEqual(
            report["input_sha256"],
            analyzer.analyze_bytes(payload + b"\n")["input_sha256"],
        )

    def test_invalid_anchor_limits_trace_tokens_and_schema_are_rejected(self) -> None:
        payload = _payload(_cell([]))
        for limit in (True, 0, 449, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                analyzer.analyze_bytes(payload, max_anchor_tokens=limit)
        for field, value in (("decode_index", True), ("retained_from_sample", 16)):
            trace = _trace(1, 100, ((1,),))
            trace[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.analyze([trace])
        trace = _trace(1, 100, ((1,),))
        trace["result"]["metadata"]["segments"][0]["tokens"] = [True]
        with self.assertRaises(TypeError):
            self.analyze([trace])
        for invalid in (
            b'{"schema_version":"other","cells":[]}',
            b'{"schema_version":"1-diagnostic","cells":[],"bad":NaN}',
            _payload(_cell([], name="same"), _cell([], name="same")),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                analyzer.analyze_bytes(invalid)

    def test_cli_writes_only_json_stdout_and_leaves_input_bytes_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            payload = _payload(
                _cell([_trace(1, 100, ((1,),)), _trace(2, 200, ((1, 2),))])
            )
            path.write_bytes(payload)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    analyzer.main([str(path), "--max-anchor-tokens", "2"]), 0
                )
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertEqual(json.loads(output.getvalue())["max_anchor_tokens"], 2)
            path.write_bytes(b"not-json")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                analyzer.main([str(path)])
            self.assertEqual(error.exception.code, 2)

    def test_v3_unique_anchor_in_later_repetition_has_no_publication_authority(
        self,
    ) -> None:
        # Exact token IDs and spans from left-context-2000 traces 24 -> 25 of
        # stream-boundary-diagnostic-v3-attempt-1.json (SHA256 220ef884ec52...).
        # The helper supplies synthetic text labels; no tokenizer or fixture
        # file is needed to reproduce this text-token alignment trap.
        anchor = (
            843,
            523,
            616,
            5891,
            3399,
            1265,
            407,
            644,
            534,
            1499,
            460,
            466,
            329,
            345,
        )
        lead = (
            1867,
            534,
            1499,
            460,
            466,
            329,
            345,
            11,
            1265,
            644,
            345,
            460,
            466,
            329,
            534,
            1499,
            13,
        )
        repeated = anchor + (11, 1265, 644, 345, 460)
        tail = (466, 329, 534, 1499, 13)
        commit = _trace(
            10,
            10000,
            (anchor,),
            spans=((0, 7440),),
            publication=(0, 7440),
            action="commit",
        )
        before = _trace(
            24,
            24000,
            (lead, repeated, tail, anchor[:5]),
            start=5440,
            head=7440,
            spans=((5440, 11520), (11520, 20360), (20360, 22560), (22560, 24000)),
        )
        after = _trace(
            25,
            25000,
            (lead, repeated, tail, anchor[:5] + (13,)),
            start=5440,
            head=7440,
            spans=((5440, 11520), (11520, 20360), (20360, 22520), (22520, 24200)),
        )
        for trace in (before, after):
            flat = tuple(
                token
                for segment in trace["result"]["metadata"]["segments"]
                for token in segment["tokens"]
            )
            self.assertEqual(
                [i for i in range(len(flat)) if flat[i : i + len(anchor)] == anchor],
                [17],
            )
        # The unique occurrence belongs to a segment strictly AFTER the real
        # committed boundary, not to the retained tail of the original phrase.
        self.assertGreater(
            after["result"]["metadata"]["segments"][1]["span"]["start_ms"], 7440
        )
        report = analyzer.analyze_bytes(
            _payload(_cell([commit, before, after], _commit(commit)))
        )
        evidence = report["cells"][0]["evidence"][0]
        self.assertEqual(
            (evidence["previous_decode_index"], evidence["decode_index"]), (24, 25)
        )
        self.assertEqual(evidence["anchor_tokens"], 14)
        self.assertEqual(evidence["anchor_commit_sequences"], [2])
        self.assertEqual(evidence["candidate_tokens"], 15)
        self.assertEqual(
            (evidence["current_token_start"], evidence["current_token_end"]), (31, 46)
        )
        self.assertEqual(evidence["publication_kind"], "none")
        self.assertTrue(evidence["text_candidate_with_no_timed_publication"])
        self.assertEqual(len(report["cells"][0]["confirmed_commits"]), 1)
        self.assertEqual(set(report["claim_boundary"].values()), {False})


if __name__ == "__main__":
    unittest.main()
