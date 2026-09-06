"""Inspect the fixed overlap archive without a model, PCM loading, or network.

The archived assessment remains authoritative for this experiment. Additional
comparisons describe outputs; they never accept a handoff or establish speech
coverage. The context-guard plan is prospective, not a measured improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from tools.analyze_word_resolution import _alignment
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeTimestampSegment, diagnose_word_sequence

ARCHIVE = "evidence/modal-t4-tiny-en-resolution-handoff-2026-09-06.json"
ARCHIVE_SHA = "545e6e8d0d420539b5e94333e5aeeba2095886577f9daca0382cc183344b9a6c"
GUARD_MS = 500


def _lexical_extent(alignment, head_ms):
    lexical = [w for w in alignment.words if any(c.isalnum() for c in w.text)]
    last_lexical = lexical[-1].span.end_ms if lexical else None
    last_unit = alignment.words[-1].span.end_ms if alignment.words else None
    return dict(
        lexical_unit_count=len(lexical),
        lexical_units_ending_after_head=sum(w.span.end_ms > head_ms for w in lexical),
        last_lexical_estimate_end_ms=last_lexical,
        last_unit_estimate_end_ms=last_unit,
        trailing_nonlexical_estimate_ms=None
        if last_lexical is None or last_unit is None
        else last_unit - last_lexical,
    )


def _correspondence(anchor, observed, candidate, head_ms):
    """Report only a unique exact anchor; do not resolve repeated occurrences."""
    words = observed.words
    occurrences = [
        index
        for index in range(len(words) - len(anchor) + 1)
        if all(
            a.text == b.text and a.tokens == b.tokens
            for a, b in zip(anchor, words[index : index + len(anchor)])
        )
    ]
    anchor_comparison = suffix_comparison = None
    boundary_checks = None
    if len(occurrences) == 1:
        start = occurrences[0]
        stop = start + len(anchor)
        suffix, candidate_words = words[stop:], candidate.words
        anchor_comparison = asdict(diagnose_word_sequence(anchor, words[start:stop]))
        suffix_comparison = asdict(diagnose_word_sequence(suffix, candidate_words))
        boundary = max(head_ms, words[stop - 1].span.end_ms)
        boundary_checks = dict(
            observed_suffix_starts_before_boundary=bool(suffix)
            and suffix[0].span.start_ms < boundary,
            head_candidate_starts_before_boundary=bool(candidate_words)
            and candidate_words[0].span.start_ms < boundary,
        )
    return dict(
        exact_anchor_occurrences=occurrences,
        anchor_comparison=anchor_comparison,
        complete_suffix_comparison=suffix_comparison,
        boundary_checks=boundary_checks,
    )


def diagnose_cell(frozen, cell):
    """Compare complete suffixes only after one exact raw text/token anchor.

    No approximate phrase, later convenient substring, reference transcript,
    or adjusted time tolerance selects a correspondence. An absent or repeated
    anchor leaves the suffix comparison unavailable, not silently successful.
    """
    aligned = {arm: _alignment(raw) for arm, raw in cell["raw_alignments"].items()}
    anchor = tuple(
        NativeTimestampSegment(AudioSpan(**w["span"]), w["text"], tuple(w["tokens"]))
        for w in frozen["anchor"]
    )

    guard_start = max(
        frozen["retained_ms"], anchor[0].span.start_ms // 20 * 20 - GUARD_MS
    )
    guard_span = AudioSpan(guard_start, frozen["end_ms"])
    matching = [
        arm
        for arm, value in aligned.items()
        if value.native.analyzed_span == guard_span
    ]
    return dict(
        id=cell["id"],
        original_assessment=cell["summary"]["assessment"],
        **_correspondence(
            anchor, aligned["overlap"], aligned["candidate"], frozen["head_ms"]
        ),
        estimated_lexical_extents={
            arm: _lexical_extent(value, frozen["head_ms"])
            for arm, value in aligned.items()
        },
        prospective_context_guard=dict(
            requested_guard_ms=GUARD_MS,
            start_ms=guard_start,
            end_ms=frozen["end_ms"],
            added_context_ms=anchor[0].span.start_ms // 20 * 20 - guard_start,
            matching_recorded_arms=matching,
            recorded_span_available=bool(matching),
            reuse_requires_pcm_and_execution_identity_check=True,
            recorded_comparisons={
                arm: _correspondence(
                    anchor, aligned[arm], aligned["candidate"], frozen["head_ms"]
                )
                for arm in matching
            },
        ),
        publication_authorized=False,
    )


def analyze_bytes(payload):
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ARCHIVE_SHA:
        raise ValueError("input differs from the fixed seven-window archive")
    record = json.loads(payload)
    cells = [
        diagnose_cell(item["frozen_state"], cell)
        for item, cell in zip(record["inputs"], record["cells"])
    ]
    root = Path(__file__).resolve().parents[1]
    return dict(
        schema_version="resolution-disagreements/v1",
        input_sha256=digest,
        source_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "tools/analyze_resolution_disagreements.py",
                "tools/analyze_word_resolution.py",
                "tools/analyze_stream_text_agreement.py",
                "src/whisper_runtime/adapters/word_policy.py",
                "src/whisper_runtime/adapters/native_result.py",
            )
        },
        scope="Recorded outputs only; no acoustic or execution-identity revalidation",
        native_windows_executed=0,
        unmatched_guard_window_count=sum(
            not c["prospective_context_guard"]["recorded_span_available"] for c in cells
        ),
        cells=cells,
        publication_authorized=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="fixed diagnostic archive (read-only)")
    args = parser.parse_args()
    try:
        report = analyze_bytes(args.input.read_bytes())
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
