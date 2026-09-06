"""CPU-only, fixed-inventory validation of an opt-in group-anchor diagnostic.

All selected histories are validated before any group diagnostic is evaluated.
An experimental result never changes reconstructed state or the saved outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_stream_text_agreement import _commits, _trace
from tools.analyze_word_resolution import _alignment, _publication
from tools.word_anchor_reconciliation import diagnose_group_anchor

ROOT = Path(__file__).resolve().parents[1]
JFK_ARCHIVE = "evidence/modal-t4-tiny-en-word-alignment-v6-2026-09-05.json"
CORPUS_ARCHIVE = "evidence/modal-t4-tiny-en-word-corpus-v1-2026-09-05.json"
ARCHIVES = {
    JFK_ARCHIVE: "ed5565fdcc86fabaad7d66d4122842a0ee99b6533a6d60535c9684c9a3346769",
    CORPUS_ARCHIVE: "6c27715f06ec85e1c3591c6ef044adee783328d3e8f1be3451975a1a3b6b9fad",
}
INVENTORY_DOCUMENT = "docs/research/2026-09-06-group-validation-inventory.md"
INVENTORY_DOCUMENT_SHA = (
    "f87928348fa03d8f7f2d623e213ac4ad21908f569aea2b904ab0578f0d260a8a"
)
INVENTORY = (
    dict(
        path=JFK_ARCHIVE,
        collection="cells",
        id_key="cell_id",
        cell_id="word-left-context-2000",
        input_id="openai-whisper-jfk-flac",
        profile_id="word_agreement_stream/v1",
        session_id="modal-stream-boundary-diagnostic-v6:word-left-context-2000:session",
        expected_trace_count=17,
        sample_count=528000,
        pcm_sha256="a3fd62a5bb6caecc25585b9f81bc6f379726c723f5fec6f1557e01508a192372",
        speaker_id=None,
        independence="one_distinct_source_fixture_outside_tuning_states",
    ),
    dict(
        path=CORPUS_ARCHIVE,
        collection="cases",
        id_key="case_id",
        cell_id="1995-1837-0024",
        input_id="1995-1837-0024",
        profile_id="word_agreement_stream/v1",
        session_id="modal-word-corpus-v1:1995-1837-0024:session",
        expected_trace_count=3,
        sample_count=86160,
        pcm_sha256="c8161a085b6871c73e775254d8e578451cce924609e08d0778323630db8c7a2b",
        speaker_id=1995,
        independence="same_utterance_already_in_tuning_mixtures",
    ),
    dict(
        path=CORPUS_ARCHIVE,
        collection="cases",
        id_key="case_id",
        cell_id="672-122797-0072",
        input_id="672-122797-0072",
        profile_id="word_agreement_stream/v1",
        session_id="modal-word-corpus-v1:672-122797-0072:session",
        expected_trace_count=4,
        sample_count=127040,
        pcm_sha256="3bfeb4d05da42bcb00d1875e6dfe66e57ac1068afd957ab1851e74ad202becbc",
        speaker_id=672,
        independence="same_utterance_already_in_tuning_mixtures",
    ),
)
PARAMETERS = dict(timestamp_tolerance_ms=200, minimum_lexical_anchor_units=2)
REPLAY_SOURCES = (
    "tools/analyze_group_validation.py",
    "tools/analyze_word_resolution.py",
    "tools/analyze_stream_text_agreement.py",
    "tools/word_anchor_reconciliation.py",
    "src/whisper_runtime/adapters/word_policy.py",
    "src/whisper_runtime/adapters/native_result.py",
    INVENTORY_DOCUMENT,
)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _hash(value):
    return _sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _lexical(word):
    return any(character.isalnum() for character in word.text)


def _validate_history(record, selection):
    """Return an unscored replay; only confirmed archived commits advance it."""
    matches = [
        cell
        for cell in record[selection["collection"]]
        if cell.get(selection["id_key"]) == selection["cell_id"]
    ]
    if len(matches) != 1:
        raise ValueError("selected cell has no unique archive join")
    cell = matches[0]
    identity = record["input"] if selection["path"] == JFK_ARCHIVE else cell["input"]
    pcm_key = (
        "stream_pcm_s16le_sha256" if selection["path"] == JFK_ARCHIVE else "pcm_sha256"
    )
    input_key = "fixture" if selection["path"] == JFK_ARCHIVE else "id"
    if (
        record["schema_version"] != "1-diagnostic"
        or cell["profile_id"] != selection["profile_id"]
        or cell["state"]["session_id"] != selection["session_id"]
        or cell["status"] != "completed"
        or identity[pcm_key] != selection["pcm_sha256"]
        or identity[input_key] != selection["input_id"]
        or identity["sample_count"] != selection["sample_count"]
        or len(cell["decision_traces"]) != selection["expected_trace_count"]
    ):
        raise ValueError("selected history identity differs")
    traces, alignments, publications = [], [], {}
    for index, raw in enumerate(cell["decision_traces"]):
        trace = _trace(raw)
        if trace.index != index + 1 or trace.end > selection["sample_count"]:
            raise ValueError("complete consecutive bounded traces required")
        if raw.get("word_alignment") is None:
            raise ValueError("selected word history lacks raw alignment")
        alignment = _alignment(raw["word_alignment"])
        if alignment.native != trace.result:
            raise ValueError("raw alignment differs from native observation")
        if (
            raw.get("source_unit") is not None
            or raw.get("silence_publication") is not None
        ):
            raise ValueError(
                "selected profile does not have source-unit or silence commits"
            )
        publication = raw.get("word_publication")
        if publication is not None:
            if trace.action != "commit":
                raise ValueError("uncommitted word publication")
            parsed = _publication(publication, alignment)
            if parsed.start_ms * 16 != trace.head or parsed.final is not trace.eof:
                raise ValueError("publication watermark or finality differs")
            publications[index] = parsed
        elif trace.action == "commit":
            raise ValueError("commit lacks explicit word publication")
        traces.append(trace)
        alignments.append(alignment)

    confirmed, last = {}, -1
    for event, text in _commits(cell["events"]):
        matches = [
            index
            for index, pub in publications.items()
            if (pub.start_ms * 16, pub.end_ms * 16, pub.text)
            == (event.start_sample, event.end_sample, text)
        ]
        if len(matches) != 1 or matches[0] <= last:
            raise ValueError("COMMIT lacks a unique ordered publication join")
        last = matches[0]
        confirmed[last] = event
    if set(confirmed) != set(publications):
        raise ValueError("word publication lacks a confirmed COMMIT")

    head, anchor, anchor_origin, observations = 0, (), None, []
    for index, (raw, trace, aligned) in enumerate(
        zip(cell["decision_traces"], traces, alignments)
    ):
        if trace.head != head:
            raise ValueError("trace watermark differs from confirmed commits")
        retained = tuple(
            word
            for word in anchor
            if word.span.start_ms >= trace.start // 16
            and word.span.end_ms > trace.start // 16
        )
        exclusion = None
        if not anchor:
            exclusion = "no_committed_anchor"
        elif (
            len(retained) < 2
            or sum(_lexical(word) for word in retained) < 2
            or not _lexical(retained[0])
            or not _lexical(retained[-1])
            or retained[-1].span.end_ms != head // 16
        ):
            exclusion = "insufficient_anchor"
        anchor_state = (
            None
            if exclusion
            else _hash(
                dict(
                    session_id=selection["session_id"],
                    committed_through_ms=head // 16,
                    anchor_origin=anchor_origin,
                    anchor=[asdict(word) for word in retained],
                )
            )
        )
        observations.append(
            dict(
                decode_index=trace.index,
                enrollment="excluded" if exclusion else "enrolled",
                exclusion_reason=exclusion,
                anchor_state_id=anchor_state,
                anchor=[asdict(word) for word in retained],
                full_committed_anchor=[asdict(word) for word in anchor],
                anchor_origin=anchor_origin,
                analysis_start_ms=trace.start // 16,
                analysis_end_ms=trace.end // 16,
                head_ms=head // 16,
                eof=trace.eof,
                recorded_action=trace.action,
                recorded_reason=raw["reason"],
                raw_alignment_sha256=_hash(raw["word_alignment"]),
                raw_trace_sha256=_hash(raw),
                native_window_id=aligned.native.window_id,
                diagnostic=None,
                publication_authorized=False,
            )
        )
        if index in confirmed:
            pub = publications[index]
            selected = pub.alignment.words[pub.word_start : pub.word_end]
            # All three selected profiles use the existing four-unit word suffix.
            # They have neither source-unit/silence resets nor context trimming.
            anchor = selected[-4:] if selected else retained
            anchor_origin = dict(
                decode_index=trace.index,
                commit_sequence_number=confirmed[index].sequence_number,
                session_version=confirmed[index].session_version,
                word_start=pub.word_start,
                word_end=pub.word_end,
                publication_sha256=_hash(raw["word_publication"]),
            )
            head = pub.end_ms * 16
    metrics = cell["metrics"]
    if (
        head != selection["sample_count"]
        or metrics["accepted_samples"] != selection["sample_count"]
        or metrics["committed_samples"] != head
        or metrics["decode_count"] != len(traces)
        or cell["state"]["committed_through_ms"] * 16 != head
        or not traces[-1].eof
        or traces[-1].end != head
    ):
        raise ValueError("completed history metrics differ from confirmed state")
    result = dict(
        id=selection["cell_id"],
        input_id=selection["input_id"],
        session_id=selection["session_id"],
        source_archive=selection["path"],
        identity=dict(selection),
        recorded_input=identity,
        recorded_status=cell["status"],
        recorded_error=cell.get("error"),
        recorded_profile_id=cell["profile_id"],
        recorded_final_state_sha256=_hash(cell["state"]),
        recorded_events_sha256=_hash(cell["events"]),
        confirmed_commit_count=len(confirmed),
        traces=observations,
        publication_authorized=False,
    )
    return result, alignments


def analyze_records(archive_payloads: dict[str, bytes]) -> dict:
    """Validate the fixed byte-bound inventory, then score every enrolled trace."""
    if set(archive_payloads) != set(ARCHIVES):
        raise ValueError("exactly the two registered archives are required")
    if _sha((ROOT / INVENTORY_DOCUMENT).read_bytes()) != INVENTORY_DOCUMENT_SHA:
        raise ValueError("pre-scoring inventory document differs")
    records = {}
    for name, expected in ARCHIVES.items():
        payload = archive_payloads[name]
        if not isinstance(payload, bytes) or _sha(payload) != expected:
            raise ValueError("registered archive digest differs")
        records[name] = json.loads(payload)
    # Complete all source/state joins before examining any experimental score.
    histories = [
        _validate_history(records[selection["path"]], selection)
        for selection in INVENTORY
    ]
    for cell, alignments in histories:
        for observed, alignment in zip(cell["traces"], alignments):
            if observed["enrollment"] == "enrolled":
                # Reuse the already validated immutable native word instances.
                start = alignment.native.analyzed_span.start_ms
                origin = observed["anchor_origin"]["decode_index"] - 1
                raw_cell = next(
                    c
                    for c in records[cell["source_archive"]][
                        cell["identity"]["collection"]
                    ]
                    if c.get(cell["identity"]["id_key"]) == cell["id"]
                )
                pub = raw_cell["decision_traces"][origin]["word_publication"]
                committed = _alignment(pub["alignment"]).words[
                    pub["word_start"] : pub["word_end"]
                ][-4:]
                anchor = tuple(
                    w
                    for w in committed
                    if w.span.start_ms >= start and w.span.end_ms > start
                )
                if [asdict(word) for word in anchor] != observed["anchor"]:
                    raise ValueError("scoring anchor differs from validated history")
                observed["diagnostic"] = diagnose_group_anchor(
                    alignment,
                    anchor=anchor,
                    committed_through_ms=observed["head_ms"],
                    timestamp_tolerance_ms=PARAMETERS["timestamp_tolerance_ms"],
                )
    cells = [cell for cell, _ in histories]
    observations = [trace for cell in cells for trace in cell["traces"]]
    enrolled = [trace for trace in observations if trace["enrollment"] == "enrolled"]
    return dict(
        schema_version="group-validation/v1",
        parameters=dict(PARAMETERS),
        selection=[dict(item) for item in INVENTORY],
        inventory_document=dict(path=INVENTORY_DOCUMENT, sha256=INVENTORY_DOCUMENT_SHA),
        source_sha256={
            name: _sha((ROOT / name).read_bytes()) for name in REPLAY_SOURCES
        },
        archives=[
            dict(
                path=name,
                sha256=ARCHIVES[name],
                recorded_source=record["source"],
                recorded_model=record["model"],
            )
            for name, record in records.items()
        ],
        scope=dict(
            recorded_anchor_stage_only=True,
            group_rule_opt_in=True,
            histories_validated_before_scoring=True,
            state_changes_from_experimental_results=False,
            references_used_for_enrollment_or_correspondence=False,
            full_handoff_evaluated=False,
            new_native_windows=0,
            pcm_or_execution_identity_reestablished=False,
            historical_missing_measurements_preserved=True,
            statistically_held_out=False,
            publication_authorized=False,
        ),
        counts=dict(
            traces=len(observations),
            enrolled=len(enrolled),
            excluded=len(observations) - len(enrolled),
            distinct_anchor_states=len(
                {trace["anchor_state_id"] for trace in enrolled}
            ),
            distinct_outside_tuning_source_fixtures=1,
            exclusion_reasons=dict(
                Counter(
                    trace["exclusion_reason"]
                    for trace in observations
                    if trace["enrollment"] == "excluded"
                )
            ),
            group_statuses=dict(
                Counter(trace["diagnostic"]["status"] for trace in enrolled)
            ),
            strict_statuses=dict(
                Counter(trace["diagnostic"]["strict"]["status"] for trace in enrolled)
            ),
        ),
        cells=cells,
        publication_authorized=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="new JSON file; existing files are never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        report = analyze_records(
            {name: (ROOT / name).read_bytes() for name in ARCHIVES}
        )
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
