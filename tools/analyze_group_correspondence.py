"""Compare group-boundary rules on saved outputs; never authorize publication.

This post-hoc experiment keeps exact word text and tokens. It varies anchor
timing granularity and continuation tolerance separately. It does not run a
model, measure acoustic coverage, or change any streaming policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.analyze_resolution_disagreements import summarize_guard_correspondence
from tools.analyze_word_resolution import _alignment

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "evidence/modal-t4-tiny-en-context-guard-2026-09-06.json"
ARCHIVE_SHA = "538ff91508b1ea2074a25fb6600860577a917b37d4826f324c2cce2ec8ef71f1"
TOLERANCE_MS = 200


def _decision(reason):
    return dict(
        status="structurally_eligible"
        if reason == "group_and_suffix_agree"
        else "rejected",
        reason=reason,
        publication_authorized=False,
    )


def assess_group_correspondence(frozen, observed_raw, candidate_raw):
    """Match the whole unique anchor by its outer bounds, not internal splits.

    The original assessor validates the fixed guard/head intervals, raw units
    and frozen state. Its result remains unchanged. The two group variants
    differ only in backward continuation tolerance: zero or 200 ms. The latter
    borrows the live policy's bound but is not a replay of that policy. No raw
    time, prefix, or audio-retention boundary is moved.
    """
    strict = summarize_guard_correspondence(frozen, observed_raw, candidate_raw)
    correspondence = strict["diagnostics"]
    occurrences = correspondence["exact_anchor_occurrences"]
    diagnostics = None
    if not occurrences:
        reason = "overlap_anchor_absent"
    elif len(occurrences) != 1:
        reason = "overlap_anchor_ambiguous"
    else:
        observed, candidate = _alignment(observed_raw), _alignment(candidate_raw)
        anchor = frozen["anchor"]
        start = occurrences[0]
        stop = start + len(anchor)
        matched, suffix = observed.words[start:stop], observed.words[stop:]
        first, last = matched[0], matched[-1]
        head = frozen["head_ms"]
        boundary = max(head, last.span.end_ms)
        start_delta = first.span.start_ms - anchor[0]["span"]["start_ms"]
        end_delta = last.span.end_ms - anchor[-1]["span"]["end_ms"]
        origin_exception = (
            start == 0
            and first.span.start_ms == observed.native.analyzed_span.start_ms
            and start_delta < 0
            and abs(first.span.end_ms - anchor[0]["span"]["end_ms"]) <= TOLERANCE_MS
        )
        diagnostics = dict(
            anchor_start_delta_ms=start_delta,
            anchor_end_delta_ms=end_delta,
            max_interior_delta_ms=max(
                abs(getattr(new.span, edge) - old["span"][edge])
                for index, (old, new) in enumerate(zip(anchor, matched))
                for edge in ("start_ms", "end_ms")
                if (index, edge) not in ((0, "start_ms"), (len(anchor) - 1, "end_ms"))
            ),
            origin_start_exception=origin_exception,
            observed_overlap_ms=max(0, boundary - suffix[0].span.start_ms)
            if suffix
            else None,
            candidate_overlap_ms=max(0, boundary - candidate.words[0].span.start_ms)
            if candidate.words
            else None,
        )
        # Matching a later repetition is not recovery of the published occurrence.
        if last.span.start_ms > head or (
            last.span.start_ms == head and anchor[-1]["span"]["start_ms"] < head
        ):
            reason = "group_anchor_relocated"
        elif abs(start_delta) > TOLERANCE_MS and not origin_exception:
            reason = "group_anchor_start_mismatch"
        elif abs(end_delta) > TOLERANCE_MS:
            reason = "group_anchor_end_mismatch"
        elif not any(any(c.isalnum() for c in word.text) for word in suffix):
            reason = "overlap_has_no_lexical_continuation"
        else:
            reason = None

    def assess_boundary(tolerance):
        if reason is not None:
            return _decision(reason)
        comparison = correspondence["complete_suffix_comparison"]
        # With ordered spans and the group-end bound, a 200 ms crossing cap
        # follows already. Only the zero variant adds a stricter constraint.
        if diagnostics["observed_overlap_ms"] > tolerance:
            return _decision("overlap_continuation_crosses_boundary")
        # Empty candidates must reach the full-suffix mismatch, not compare None.
        if (
            diagnostics["candidate_overlap_ms"] is not None
            and diagnostics["candidate_overlap_ms"] > tolerance
        ):
            return _decision("head_continuation_crosses_observed_anchor")
        if (
            comparison["text_relation"] != "exact_units"
            or not comparison["unit_tokens_equal"]
        ):
            return _decision("complete_suffix_disagrees")
        if (
            max(comparison["max_start_delta_ms"], comparison["max_end_delta_ms"])
            > TOLERANCE_MS
        ):
            return _decision("suffix_timing_mismatch")
        return _decision("group_and_suffix_agree")

    return dict(
        strict_assessment=strict,
        group_strict_boundary=assess_boundary(0),
        group_live_boundary=assess_boundary(TOLERANCE_MS),
        diagnostics=diagnostics,
        publication_authorized=False,
    )


def analyze_bytes(payload):
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ARCHIVE_SHA:
        raise ValueError("input differs from the fixed context-guard archive")
    record = json.loads(payload)
    cells = []
    for saved, cell in zip(record["inputs"], record["cells"]):
        raw = cell["raw_alignments"]
        assessed = assess_group_correspondence(
            saved["frozen_state"], raw["guard"], raw["candidate"]
        )
        if assessed["strict_assessment"] != cell["summary"]["guard_correspondence"]:
            raise ValueError("original guard assessment changed")
        cells.append(dict(id=cell["id"], **assessed))
    sources = (
        "tools/analyze_group_correspondence.py",
        "tools/analyze_resolution_disagreements.py",
        "tools/analyze_word_resolution.py",
        "tools/analyze_stream_text_agreement.py",
        "src/whisper_runtime/adapters/word_policy.py",
        "src/whisper_runtime/adapters/native_result.py",
    )
    return dict(
        schema_version="group-correspondence/v1",
        input_sha256=digest,
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
        scope="Post-hoc saved-output comparison; no acoustic or execution-identity revalidation",
        parameters=dict(
            anchor_outer_tolerance_ms=TOLERANCE_MS,
            continuation_tolerances_ms=[0, TOLERANCE_MS],
            origin_start_rule="Existing live first-word origin exception in both group variants",
            interior_anchor_times="Reported, not matched in group variants",
        ),
        native_windows_executed=0,
        cells=cells,
        publication_authorized=False,
        full_stream_recovery=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="fixed context-guard archive")
    parser.add_argument(
        "--output", type=Path, help="create a new report; never overwrite"
    )
    args = parser.parse_args()
    try:
        report = analyze_bytes(args.input.read_bytes())
        encoded = (
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if args.output:
            with args.output.open("xb") as output:
                output.write(encoded.encode("utf-8"))
            print(args.output)
        else:
            print(encoded, end="")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
