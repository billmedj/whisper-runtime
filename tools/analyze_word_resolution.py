"""Read-only rematching of every terminal word-policy failure in saved evidence.

No model, tokenizer, audio processing, GPU, network or file writes are used.
Correspondence and shadow eligibility are not publication authority, acoustic
truth, recognition improvement, complete coverage, or a changed qualification.
Only observations actually recorded after the latest confirmed commit are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_stream_text_agreement import _commits, _native_result, _trace
from tools.word_anchor_reconciliation import propose_terminal_end_reconciliation
from whisper_runtime.adapters.native_result import NativeTimestampSegment
from whisper_runtime.adapters.word_policy import (
    AlignedPublication,
    NativeWordAlignment,
    compare_word_hypotheses,
    diagnose_word_anchor,
)
from whisper_runtime.state import AudioSpan

PARAMETERS = {
    "holdback_ms": 2000,
    "timestamp_tolerance_ms": 200,
    "max_window_ms": 30000,
    "max_end_extension_ms": 1000,
}
REPLAY_SOURCES = (
    "src/whisper_runtime/adapters/word_policy.py",
    "tools/word_anchor_reconciliation.py",
    "tools/analyze_word_resolution.py",
)


def _alignment(value: dict[str, Any]) -> NativeWordAlignment:
    if set(value) != {"native", "words"}:
        raise ValueError("alignment requires exactly native and words")
    return NativeWordAlignment(
        _native_result(value["native"]),
        tuple(
            NativeTimestampSegment(
                span=AudioSpan(**word["span"]),
                text=word["text"],
                tokens=tuple(word["tokens"]),
            )
            for word in value["words"]
        ),
    )


def _publication(
    value: dict[str, Any], alignment: NativeWordAlignment
) -> AlignedPublication:
    data = dict(value)
    if _alignment(data["alignment"]) != alignment:
        raise ValueError("publication alignment differs from its observation")
    data["alignment"] = alignment
    if data.get("analysis_span") is not None:
        data["analysis_span"] = AudioSpan(**data["analysis_span"])
    return AlignedPublication(**data)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _analyze_cell(cell: dict[str, Any]) -> dict[str, Any]:
    if cell["profile"] not in {"hybrid", "context"}:
        raise ValueError("word replay requires a registered hybrid/context profile")
    expected_profile = (
        "word_boundary_quiet_endpoint_stream/v1"
        + ("+word_context/v1" if cell["profile"] == "context" else "")
        + "+input_evidence/v1"
    )
    if (
        cell["id"].split(":", 1)[0] != cell["profile"]
        or cell["profile_id"] != expected_profile
    ):
        raise ValueError("cell profile does not bind its context-retention semantics")
    if (
        cell["qualified"] is not False
        or cell["error"]["category"] != "policy_resolution"
    ):
        raise ValueError("terminal policy failure cannot be qualified")
    if any(event["kind"] == "final" for event in cell["events"]):
        raise ValueError("terminal policy failure cannot contain a final event")
    values = cell["decision_traces"]
    if not isinstance(values, list) or not values:
        raise ValueError("terminal failure requires decision traces")
    traces = [_trace(value) for value in values]
    if [trace.index for trace in traces] != list(range(1, len(traces) + 1)):
        raise ValueError("complete consecutive decode indices are required")
    alignments = []
    publications = {}
    for index, (value, trace) in enumerate(zip(values, traces)):
        raw = value["word_alignment"]
        alignment = None if raw is None else _alignment(raw)
        if alignment is not None and alignment.native != trace.result:
            raise ValueError("alignment native differs from the recorded result")
        alignments.append(alignment)
        raw_publication = value["word_publication"]
        if raw_publication is not None:
            if trace.action != "commit" or alignment is None:
                raise ValueError("word publication requires a committed alignment")
            publication = _publication(raw_publication, alignment)
            if publication.start_ms * 16 != trace.head:
                raise ValueError("publication does not start at the recorded watermark")
            if publication.final is not (trace.eof or value["source_unit"] is not None):
                raise ValueError(
                    "publication finality differs from its boundary authority"
                )
            publications[index] = publication
        elif trace.action == "commit":
            raise ValueError(
                "word replay requires explicit word publications for commits"
            )

    confirmed = {}
    last = -1
    for event, text in _commits(cell["events"]):
        matches = [
            index
            for index, publication in publications.items()
            if (publication.start_ms * 16, publication.end_ms * 16, publication.text)
            == (event.start_sample, event.end_sample, text)
        ]
        if len(matches) != 1 or matches[0] <= last:
            raise ValueError("COMMIT has no unique ordered word-publication join")
        last = matches[0]
        confirmed[last] = event
    if set(confirmed) != set(publications):
        raise ValueError("a proposed word publication has no confirmed COMMIT")

    head = accepted = 0
    anchor: tuple[NativeTimestampSegment, ...] = ()
    previous = None
    previous_index = None
    anchor_origin = None
    for index, (value, trace, alignment) in enumerate(zip(values, traces, alignments)):
        if trace.head != head:
            raise ValueError("trace watermark differs from confirmed commits")
        now = _integer(value["accepted_through_sample"], "accepted_through_sample")
        if now < accepted or now < trace.end:
            raise ValueError("accepted input regresses or excludes the observation")
        accepted = now
        if index == len(traces) - 1:
            break
        if index in confirmed:
            publication = publications[index]
            selected = publication.alignment.words[
                publication.word_start : publication.word_end
            ]
            retained = tuple(
                word
                for word in anchor
                if word.span.start_ms >= trace.start // 16
                and word.span.end_ms > trace.start // 16
            )
            anchor = selected[-4:] if selected else retained
            if value["source_unit"] is not None:
                anchor = ()
            elif cell["profile"] == "context" and not trace.eof:
                while anchor and not any(char.isalnum() for char in anchor[0].text):
                    anchor = anchor[1:]
            anchor_origin = {
                "decode_index": trace.index,
                "commit_sequence_number": confirmed[index].sequence_number,
                "word_start": publication.word_start,
                "word_end": publication.word_end,
            }
            head = publication.end_ms * 16
            previous = previous_index = None
        else:
            previous = alignment
            previous_index = trace.index if alignment is not None else None

    terminal = traces[-1]
    current = alignments[-1]
    raw_terminal = values[-1]
    if current is None or terminal.action not in {"preview", "unresolved"}:
        raise ValueError("terminal failure requires an unresolved word observation")
    total = _integer(cell["case"]["sample_count"], "sample_count")
    metrics = cell["metrics"]
    for name in ("committed_samples", "accepted_samples", "decode_count"):
        _integer(metrics[name], name)
    if (
        accepted > total
        or metrics["committed_samples"] != head
        or metrics["accepted_samples"] != accepted
        or metrics["decode_count"] != len(traces)
    ):
        raise ValueError("terminal metrics do not match confirmed observations")
    if terminal.eof and (terminal.end != total or accepted != total):
        raise ValueError("EOF observation does not cover all recorded input")
    if terminal.eof and terminal.action != "unresolved":
        raise ValueError("failed EOF must be recorded as unresolved")
    unit = raw_terminal["source_unit"]
    if unit is not None and (
        unit["end_sample"] != terminal.end
        or unit["start_sample"] != head
        or unit["origin"] not in {"quiet_run", "end_of_input"}
        or (unit["origin"] == "end_of_input" and not terminal.eof)
    ):
        raise ValueError("terminal source unit differs from the admitted boundary")
    terminal_kind = (
        "eof"
        if terminal.eof
        else "source_unit"
        if unit is not None
        else "window_full"
        if terminal.end - terminal.start == PARAMETERS["max_window_ms"] * 16
        else "non_eof_policy_failure"
    )
    retained = tuple(
        word
        for word in anchor
        if word.span.start_ms >= terminal.start // 16
        and word.span.end_ms > terminal.start // 16
    )
    closed = terminal.eof or unit is not None
    decision = compare_word_hypotheses(
        previous,
        current,
        committed_through_ms=head // 16,
        anchor=anchor,
        holdback_ms=PARAMETERS["holdback_ms"],
        timestamp_tolerance_ms=PARAMETERS["timestamp_tolerance_ms"],
        final=closed,
    )
    same_continuation = previous is not None and (
        previous.native.analyzed_span.start_ms == current.native.analyzed_span.start_ms
        and previous.native.analyzed_span.end_ms < current.native.analyzed_span.end_ms
        and (previous.native.metadata.language if previous.native.metadata else None)
        == (current.native.metadata.language if current.native.metadata else None)
    )
    return {
        "id": cell["id"],
        "recorded_qualified": cell["qualified"],
        "terminal_kind": terminal_kind,
        "terminal_decode_index": terminal.index,
        "previous_decode_index": previous_index,
        "same_continuation_pair": same_continuation,
        "analysis_start_ms": terminal.start // 16,
        "analysis_end_ms": terminal.end // 16,
        "committed_through_ms": head // 16,
        "accepted_through_sample": accepted,
        "frozen_anchor_origin": anchor_origin,
        "frozen_anchor": [asdict(word) for word in anchor],
        "retained_anchor": [asdict(word) for word in retained],
        "recorded_reason": raw_terminal["reason"],
        "replayed_reason": decision.reason,
        "legacy_reason_matches": raw_terminal["reason"] == decision.reason,
        "current_diagnostic": asdict(
            diagnose_word_anchor(
                current,
                committed_through_ms=head // 16,
                anchor=retained,
                timestamp_tolerance_ms=PARAMETERS["timestamp_tolerance_ms"],
            )
        ),
        "previous_diagnostic": None
        if previous is None
        else asdict(
            diagnose_word_anchor(
                previous,
                committed_through_ms=head // 16,
                anchor=retained,
                timestamp_tolerance_ms=PARAMETERS["timestamp_tolerance_ms"],
                observation="previous",
            )
        ),
        "shadow_proposal": asdict(
            propose_terminal_end_reconciliation(
                current,
                committed_through_ms=head // 16,
                anchor=retained,
                final=terminal.eof,
                timestamp_tolerance_ms=PARAMETERS["timestamp_tolerance_ms"],
                max_end_extension_ms=PARAMETERS["max_end_extension_ms"],
            )
        ),
    }


def analyze_bytes(payload: bytes) -> dict[str, Any]:
    """Validate recorded joins and report every terminal policy failure unchanged."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    record = json.loads(payload, parse_constant=reject_constant)
    if record["schema_version"] != "1-diagnostic":
        raise ValueError("expected a 1-diagnostic evidence record")
    cells = record["cells"]
    if not isinstance(cells, list) or not 1 <= len(cells) <= 256:
        raise ValueError("expected one to 256 diagnostic cells")
    ids = [cell["id"] for cell in cells]
    if any(not isinstance(value, str) or not value for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("cell IDs must be nonempty and unique")
    failures = [cell for cell in cells if cell["stream_status"] == "policy_failure"]
    return {
        "schema_version": "word-resolution-replay/v1",
        "input_schema_version": record["schema_version"],
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "input_source": record.get("source"),
        "replay_source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
            ).hexdigest()
            for name in REPLAY_SOURCES
        },
        "input_qualified": record.get("qualified"),
        "parameters": dict(PARAMETERS),
        "scope": "all terminal policy failures; recorded observations only",
        "source_identity": "PCM/model/tokenizer/options provenance is recorded, not re-established by rematching",
        "claim_boundary": {
            "publication_authority": False,
            "qualification_changes": False,
            "recognition_improvement": False,
            "acoustic_ground_truth": False,
            "complete_coverage": False,
            "latency_or_performance": False,
        },
        "input_cell_count": len(cells),
        "terminal_failure_count": len(failures),
        "excluded_cells": [
            {"id": cell["id"], "stream_status": cell["stream_status"]}
            for cell in cells
            if cell["stream_status"] != "policy_failure"
        ],
        "cells": [_analyze_cell(cell) for cell in failures],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="saved diagnostic JSON (read-only)")
    args = parser.parse_args(argv)
    try:
        report = analyze_bytes(args.input.read_bytes())
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
