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


def summarize_guard_correspondence(frozen, observed_raw, candidate_raw):
    """Assess the fixed earlier-start witness under unchanged output criteria.

    This is not the old onset-cut assessment with an altered input record. It
    requires the separately planned guarded span. PCM and effective execution
    identities must be checked by the producer before interpreting the result.
    Even structural agreement does not establish acoustic coverage or permit
    publication. No reference transcript is an input.
    """
    retained, head, end = (frozen[k] for k in ("retained_ms", "head_ms", "end_ms"))
    if any(type(value) is not int for value in (retained, head, end)):
        raise TypeError("frozen bounds must be integer milliseconds")
    if not 0 <= retained < head < end or end - retained > 30000:
        raise ValueError("invalid retained interval")
    anchor = tuple(
        NativeTimestampSegment(AudioSpan(**w["span"]), w["text"], tuple(w["tokens"]))
        for w in frozen["anchor"]
    )
    diagnose_word_sequence(anchor, anchor)  # Validate raw units and ordered estimates.

    def lexical(word):
        return any(c.isalnum() for c in word.text)

    if (
        not 2 <= len(anchor) <= 4
        or sum(lexical(w) for w in anchor) < 2
        or not lexical(anchor[0])
        or not lexical(anchor[-1])
        or anchor[0].span.start_ms < retained
        or anchor[-1].span.end_ms != head
    ):
        raise ValueError("invalid frozen lexical anchor")
    observed, candidate = _alignment(observed_raw), _alignment(candidate_raw)
    start = max(retained, anchor[0].span.start_ms // 20 * 20 - GUARD_MS)
    if observed.native.analyzed_span != AudioSpan(start, end):
        raise ValueError("observed interval differs from the fixed context guard")
    if candidate.native.analyzed_span != AudioSpan(head, end):
        raise ValueError("candidate interval differs from the frozen head and EOF")
    diagnostic = _correspondence(anchor, observed, candidate, head)
    occurrences = diagnostic["exact_anchor_occurrences"]
    reason = "strict_overlap_and_suffix_agree"
    if not occurrences:
        reason = "overlap_anchor_absent"
    elif len(occurrences) != 1:
        reason = "overlap_anchor_ambiguous"
    else:
        match, suffix = (
            diagnostic["anchor_comparison"],
            diagnostic["complete_suffix_comparison"],
        )
        boundaries = diagnostic["boundary_checks"]
        suffix_words = observed.words[occurrences[0] + len(anchor) :]
        if max(match["max_start_delta_ms"], match["max_end_delta_ms"]) > 200:
            reason = "overlap_anchor_timing_mismatch"
        elif not any(lexical(w) for w in suffix_words):
            reason = "overlap_has_no_lexical_continuation"
        elif boundaries["observed_suffix_starts_before_boundary"]:
            reason = "overlap_continuation_crosses_boundary"
        elif boundaries["head_candidate_starts_before_boundary"]:
            reason = "head_continuation_crosses_observed_anchor"
        elif (
            suffix["text_relation"] != "exact_units" or not suffix["unit_tokens_equal"]
        ):
            reason = "complete_suffix_disagrees"
        elif max(suffix["max_start_delta_ms"], suffix["max_end_delta_ms"]) > 200:
            reason = "suffix_timing_mismatch"
    return dict(
        status="structurally_eligible"
        if reason == "strict_overlap_and_suffix_agree"
        else "rejected",
        reason=reason,
        diagnostics=diagnostic,
        estimated_lexical_extents={
            "guard": _lexical_extent(observed, head),
            "candidate": _lexical_extent(candidate, head),
        },
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
