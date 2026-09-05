"""Read-only text-prefix replay of one stream-boundary diagnostic JSON record.

No model, tokenizer, Modal client, network, or audio processing is used. The
record's per-cell identity of PCM/model/options is assumed, not independently
established by its traces. Token counts are occurrences across comparisons,
not unique words, recognition accuracy, audio coverage, or runtime performance.

A timed publication here means an explicit whole-segment selection confirmed
by a COMMIT and its exact text revision. Full-result EOF commits are reported
separately: they do not establish model-timed segment agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
    select_native_publication,
)
from whisper_runtime.adapters.native_stream import StreamEventKind, TranscriptEvent
from whisper_runtime.adapters.stream_policy import (
    TextAgreementReason,
    resolve_text_prefix,
)
from whisper_runtime.state import AudioSpan


@dataclass(frozen=True)
class _Trace:
    index: int
    start: int
    end: int
    head: int
    eof: bool
    action: str
    span: AudioSpan | None
    result: NativeWindowResult


def _native_result(value: dict[str, Any]) -> NativeWindowResult:
    data = dict(value)
    if data.get("analysis_span") is not None:
        data["analysis_span"] = AudioSpan(**data["analysis_span"])
    if data.get("metadata") is not None:
        metadata = dict(data["metadata"])
        metadata["segments"] = tuple(
            NativeTimestampSegment(
                span=AudioSpan(**segment["span"]),
                text=segment["text"],
                tokens=tuple(segment["tokens"]),
            )
            for segment in metadata["segments"]
        )
        data["metadata"] = NativeDecodeMetadata(**metadata)
    return NativeWindowResult(**data)


def _trace(value: dict[str, Any]) -> _Trace:
    for name in (
        "decode_index",
        "analysis_start_sample",
        "analysis_end_sample",
        "committed_before_sample",
        "retained_from_sample",
    ):
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"trace {name} must be a nonnegative integer")
    trace = _Trace(
        value["decode_index"],
        value["analysis_start_sample"],
        value["analysis_end_sample"],
        value["committed_before_sample"],
        value["eof"],
        value["action"],
        AudioSpan(**value["publication_span"])
        if value["publication_span"] is not None
        else None,
        _native_result(value["result"]),
    )
    if (
        trace.index < 1
        or not trace.start <= trace.head < trace.end
        or trace.start != value["retained_from_sample"]
        or not isinstance(trace.eof, bool)
        or trace.action not in ("preview", "commit", "unresolved")
        or trace.result.analyzed_span != AudioSpan(trace.start // 16, trace.end // 16)
        or trace.result.publication_segment_indices is not None
    ):
        raise ValueError("trace does not describe a full admitted native analysis")
    return trace


def _commits(values: list[dict[str, Any]]) -> list[tuple[TranscriptEvent, str]]:
    """Require the exact preceding text revision; never infer success from intent."""
    latest: dict[str, TranscriptEvent] = {}
    committed_ids: set[str] = set()
    commits = []
    head = 0
    previous: TranscriptEvent | None = None
    for sequence, value in enumerate(values, 1):
        data = dict(value)
        data["kind"] = StreamEventKind(data["kind"])
        event = TranscriptEvent(**data)
        if event.sequence_number != sequence or (
            previous is not None and previous.kind is StreamEventKind.FINAL
        ):
            raise ValueError("events must be complete and ordered")
        if event.kind is StreamEventKind.FINAL:
            previous = event
            continue
        if event.sample_rate_hz != 16_000 or event.start_sample != head:
            raise ValueError("event source span does not begin at the verified head")
        assert event.segment_id is not None
        if event.segment_id in committed_ids:
            raise ValueError("a committed segment cannot be revised or committed twice")
        if event.kind in (StreamEventKind.PROVISIONAL, StreamEventKind.REPLACE):
            old = latest.get(event.segment_id)
            expected_revision = 1 if old is None else old.revision + 1
            expected_kind = (
                StreamEventKind.PROVISIONAL if old is None else StreamEventKind.REPLACE
            )
            if event.revision != expected_revision or event.kind is not expected_kind:
                raise ValueError("text revisions must be consecutive")
            latest[event.segment_id] = event
        else:
            text = latest.get(event.segment_id)
            if (
                text is None
                or previous != text
                or text.revision != event.revision
                or text.start_sample != event.start_sample
                or text.end_sample != event.end_sample
                or text.session_version != event.session_version
                or event.committed_through_sample != event.end_sample
                or event.end_sample <= head
            ):
                raise ValueError("COMMIT has no exact preceding text revision/span")
            assert text.text is not None
            commits.append((event, text.text))
            committed_ids.add(event.segment_id)
            head = event.end_sample
        previous = event
    return commits


def _publication(trace: _Trace) -> tuple[int, str, tuple[int, ...] | None] | None:
    if trace.action != "commit":
        return None
    if trace.span is None:
        if not trace.eof or trace.start != trace.head:
            return None
        selected = trace.result
    else:
        if trace.span.start_ms * 16 != trace.head:
            return None
        selected = select_native_publication(trace.result, trace.span)
    end = trace.end if trace.eof else selected.end_ms * 16
    if selected.start_ms != trace.head // 16 or selected.end_ms != end // 16:
        return None
    metadata = selected.metadata
    tokens = None
    if metadata is not None:
        indices = selected.publication_segment_indices
        if indices is not None:
            tokens = tuple(t for i in indices for t in metadata.segments[i].tokens)
        elif (
            metadata.timestamps_complete
            and selected.text
            == "".join(segment.text for segment in metadata.segments).strip()
        ):
            tokens = tuple(t for segment in metadata.segments for t in segment.tokens)
    return end, selected.text, tokens


def _analyze_cell(cell: dict[str, Any], max_anchor_tokens: int) -> dict[str, Any]:
    traces = [_trace(value) for value in cell["decision_traces"]]
    if any(left.index >= right.index for left, right in zip(traces, traces[1:])):
        raise ValueError("decode indices must strictly increase")
    publications = [_publication(trace) for trace in traces]
    confirmed: dict[int, tuple[TranscriptEvent, tuple[int, ...] | None]] = {}
    last_match = -1
    for event, text in _commits(cell["events"]):
        matches = [
            position
            for position, (trace, publication) in enumerate(zip(traces, publications))
            if publication is not None
            and trace.head == event.start_sample
            and publication[:2] == (event.end_sample, text)
        ]
        if len(matches) != 1 or matches[0] <= last_match:
            raise ValueError("COMMIT has no unique, ordered matching decision trace")
        last_match = matches[0]
        publication = publications[last_match]
        assert publication is not None
        confirmed[last_match] = (event, publication[2])

    counts = Counter({reason.value: 0 for reason in TextAgreementReason})
    counts.update(candidate_tokens=0, text_candidate_with_no_timed_publication=0)
    evidence: list[dict[str, Any]] = []
    skipped = []
    confirmations = []
    anchor: tuple[int, ...] | None = ()
    anchor_events: tuple[int, ...] = ()
    head = 0
    previous: _Trace | None = None
    for position, trace in enumerate(traces):
        if trace.head != head:
            raise ValueError("trace watermark advanced without a matched COMMIT")
        publication_kind = (
            "none"
            if position not in confirmed
            else "whole_segments"
            if trace.span is not None
            else "full_result_eof"
        )
        reason = None
        if previous is None:
            reason = "first_trace"
        elif previous.start != trace.start:
            reason = "different_analysis_origin"
        elif previous.end >= trace.end:
            reason = "analysis_end_not_growing"
        elif trace.start < head and not anchor:
            reason = "committed_text_anchor_unavailable"
        if reason is not None:
            skipped.append({"decode_index": trace.index, "reason": reason})
        else:
            assert previous is not None
            used_anchor = () if trace.start == head else anchor
            assert used_anchor is not None
            decision = resolve_text_prefix(
                previous.result,
                trace.result,
                committed_anchor=used_anchor,
                max_anchor_tokens=max_anchor_tokens,
            )
            counts[decision.reason.value] += 1
            counts["candidate_tokens"] += len(decision.tokens)
            no_timed = (
                decision.reason is TextAgreementReason.CANDIDATE
                and publication_kind != "whole_segments"
            )
            counts["text_candidate_with_no_timed_publication"] += int(no_timed)
            evidence.append(
                {
                    "previous_decode_index": previous.index,
                    "decode_index": trace.index,
                    "reason": decision.reason.value,
                    "candidate_tokens": len(decision.tokens),
                    "current_token_start": decision.current_token_start,
                    "current_token_end": decision.current_token_end,
                    "anchor_tokens": len(used_anchor),
                    "anchor_commit_sequences": list(anchor_events)
                    if used_anchor
                    else [],
                    "publication_kind": publication_kind,
                    "text_candidate_with_no_timed_publication": no_timed,
                }
            )
        if position in confirmed:
            event, tokens = confirmed[position]
            head = event.end_sample
            if tokens is None:
                anchor, anchor_events = None, ()
            elif tokens:
                old = anchor or ()
                anchor = (old + tokens)[-max_anchor_tokens:]
                anchor_events = (
                    (event.sequence_number,)
                    if len(tokens) >= max_anchor_tokens or not old
                    else anchor_events + (event.sequence_number,)
                )
                # Provenance is a conservative superset, bounded independently.
                anchor_events = anchor_events[-max_anchor_tokens:]
            confirmations.append(
                {
                    "decode_index": trace.index,
                    "commit_sequence": event.sequence_number,
                    "segment_id": event.segment_id,
                    "revision": event.revision,
                    "start_sample": event.start_sample,
                    "end_sample": event.end_sample,
                    "publication_kind": publication_kind,
                    "closed_publication_tokens": None
                    if tokens is None
                    else len(tokens),
                }
            )
        previous = trace
    return {
        "cell_id": cell["cell_id"],
        "input_status": cell.get("status"),
        "trace_count": len(traces),
        "compared_pairs": len(evidence),
        "counts": dict(counts),
        "confirmed_commits": confirmations,
        "evidence": evidence,
        "skipped": skipped,
    }


def analyze_bytes(payload: bytes, *, max_anchor_tokens: int = 32) -> dict[str, Any]:
    """Analyze exact input bytes without changing the record or its artifacts."""
    if (
        isinstance(max_anchor_tokens, bool)
        or not isinstance(max_anchor_tokens, int)
        or not 1 <= max_anchor_tokens <= 448
    ):
        raise ValueError("max_anchor_tokens must be an integer between 1 and 448")

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    record = json.loads(payload, parse_constant=invalid_constant)
    if record["schema_version"] != "1-diagnostic":
        raise ValueError("expected a stream-boundary 1-diagnostic record")
    cells = record["cells"]
    if not isinstance(cells, list) or not 1 <= len(cells) <= 4:
        raise ValueError("expected one to four attempted diagnostic cells")
    ids = [cell["cell_id"] for cell in cells]
    if any(not isinstance(value, str) or not value for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("cell IDs must be nonempty and unique")
    return {
        "schema_version": "text-agreement-replay/v1",
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "max_anchor_tokens": max_anchor_tokens,
        "scope": "same-cell, same-origin, growing analysis pairs; observed traces only",
        "count_unit": "candidate token occurrences and comparison pairs, not unique text",
        "timed_publication_definition": "event-confirmed explicit whole-segment selection",
        "source_identity": "unchanged PCM/model/tokenizer/options assumed within each cell",
        "claim_boundary": {
            "recognition_improvement": False,
            "audio_coverage": False,
            "performance_benchmark": False,
            "publication_authority": False,
        },
        "cells": [_analyze_cell(cell, max_anchor_tokens) for cell in cells],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="diagnostic JSON (read-only)")
    parser.add_argument("--max-anchor-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        report = analyze_bytes(
            args.input.read_bytes(), max_anchor_tokens=args.max_anchor_tokens
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
