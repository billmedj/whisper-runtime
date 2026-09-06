"""Pure, local experiments for terminal anchor reconciliation and handoff.

This module is not imported by the streaming runtime. A proposal is only a
structurally eligible correspondence in saved alignments: it has no publication,
coverage, commit, retry, or model-execution authority. The raw alignment and the
frozen anchor are never modified. In particular, a missing spoken word can evade
all these checks if the aligner assigns its interval to the preceding word.
Eligibility is therefore not acoustic proof or omission-free recognition.

The separate extension cap is an experimental bound, not a calibrated acoustic
error bound. It does not change the ordinary timestamp tolerance. No human or
offline reference transcript is an input to this rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from whisper_runtime.adapters.continuous_stream import ContinuousResolutionObservation
from whisper_runtime.adapters.native_result import NativeTimestampSegment
from whisper_runtime.adapters.word_policy import AnchorDiagnostic, NativeWordAlignment


@dataclass(frozen=True, slots=True)
class BoundedResolutionRoute:
    """One counterfactual choice, with no execution or publication authority."""

    arm: Literal["baseline", "shadow", "alternative", "unresolved"]
    reason: str
    candidate_window: tuple[int, int] | None = None


def plan_bounded_resolution(
    *,
    baseline_available: bool,
    reconciliation_eligible: bool,
    current_window: tuple[int, int],
    alternative_window: tuple[int, int],
    alternative_attempted: bool = False,
) -> BoundedResolutionRoute:
    """Prefer existing evidence, then one different window, without a retry loop.

    Window bounds identify intervals within the same retained PCM and fixed
    native configuration. The caller must preserve that provenance. A different
    interval is a different observation, not a guarantee of correct recognition.
    Call before decoding the alternative; record the attempt even if it fails.
    """
    for name, value in (
        ("baseline_available", baseline_available),
        ("reconciliation_eligible", reconciliation_eligible),
        ("alternative_attempted", alternative_attempted),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
    for window in (current_window, alternative_window):
        if (
            not isinstance(window, tuple)
            or len(window) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in window
            )
        ):
            raise TypeError("windows must be pairs of integer milliseconds")
        if not 0 <= window[0] < window[1]:
            raise ValueError("window bounds must be nonnegative and ordered")
    if (
        not current_window[0]
        <= alternative_window[0]
        < alternative_window[1]
        <= current_window[1]
    ):
        raise ValueError("alternative must stay inside the retained current window")
    if baseline_available:
        return BoundedResolutionRoute("baseline", "strict_publication_available")
    if reconciliation_eligible:
        return BoundedResolutionRoute("shadow", "local_reconciliation_eligible")
    if alternative_window == current_window:
        return BoundedResolutionRoute("unresolved", "identical_window")
    if alternative_attempted:
        return BoundedResolutionRoute("unresolved", "alternative_attempt_exhausted")
    return BoundedResolutionRoute(
        "alternative", "distinct_window_fallback", alternative_window
    )


@dataclass(frozen=True, slots=True)
class TerminalEndReconciliationProposal:
    """Diagnostic anchor indices, never a selected publication or new watermark."""

    status: Literal["eligible", "rejected", "not_needed"]
    reason: str
    word_start: int | None = None
    word_end: int | None = None
    end_extension_ms: int | None = None


def propose_terminal_end_reconciliation(
    current: NativeWordAlignment,
    *,
    committed_through_ms: int,
    anchor: tuple[NativeTimestampSegment, ...],
    final: bool,
    timestamp_tolerance_ms: int = 200,
    max_end_extension_ms: int = 1_000,
) -> TerminalEndReconciliationProposal:
    """Inspect a unique raw anchor with only its last end displaced rightward.

    Rejected proposals carry no usable slice. ``not_needed`` identifies an
    ordinary strictly matching anchor, not permission to publish its suffix.
    Only ``eligible`` requires the additional experimental constraints: EOF,
    at least two lexical witnesses, a lexical terminal word ending at the frozen
    watermark, and a lexical continuation after both old and observed ends.
    Uniqueness is deliberately checked across the whole current analysis, before
    timing: a second exact phrase is rejected even if it lies beyond the watermark.
    The existing window-origin onset exception is deliberately not reproduced.

    The caller remains responsible for binding the anchor and current result to
    the same admitted PCM, stream, model, tokenizer and decode/alignment options.
    This helper cannot infer that provenance from timestamps or word identity.
    """
    if not isinstance(current, NativeWordAlignment):
        raise TypeError("current must be NativeWordAlignment")
    if not isinstance(final, bool):
        raise TypeError("final must be a boolean")
    for name, value in (
        ("committed_through_ms", committed_through_ms),
        ("timestamp_tolerance_ms", timestamp_tolerance_ms),
        ("max_end_extension_ms", max_end_extension_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if not isinstance(anchor, tuple):
        raise TypeError("anchor must be an immutable tuple")
    if not all(isinstance(word, NativeTimestampSegment) for word in anchor):
        raise TypeError("anchor must contain NativeTimestampSegment values")
    if len(anchor) > 4:
        raise ValueError("anchor must contain at most four published words")
    if any(not word.tokens or not word.text.strip() for word in anchor):
        raise ValueError("anchor words require nonempty text and tokens")
    if any(
        left.span.end_ms > right.span.start_ms
        for left, right in zip(anchor, anchor[1:])
    ):
        raise ValueError("anchor estimates must be ordered without overlap")
    if any(word.span.end_ms > committed_through_ms for word in anchor):
        raise ValueError("anchor words must already be committed")

    def reject(reason: str) -> TerminalEndReconciliationProposal:
        return TerminalEndReconciliationProposal("rejected", reason)

    if not final:
        return reject("not_final")
    span = current.native.analyzed_span
    if not span.start_ms <= committed_through_ms <= span.end_ms:
        return reject("watermark_outside_analysis")
    if not anchor:
        return reject("empty_anchor")
    if any(word.span.start_ms < span.start_ms for word in anchor):
        return reject("anchor_outside_analysis")

    words = current.words
    candidates = []
    has_text_candidate = False
    for start in range(len(words) - len(anchor) + 1):
        observed = words[start : start + len(anchor)]
        if all(old.text == new.text for old, new in zip(anchor, observed)):
            has_text_candidate = True
            if all(old.tokens == new.tokens for old, new in zip(anchor, observed)):
                candidates.append(start)
    if not candidates:
        return reject("tokens_differ" if has_text_candidate else "anchor_text_absent")
    if len(candidates) != 1:
        return reject("anchor_ambiguous")

    start = candidates[0]
    end = start + len(anchor)
    observed = words[start:end]
    terminal = observed[-1]
    frozen_terminal = anchor[-1]
    if terminal.span.start_ms > committed_through_ms or (
        terminal.span.start_ms == committed_through_ms
        and frozen_terminal.span.start_ms < committed_through_ms
    ):
        return reject("anchor_relocated")
    if any(
        abs(old.span.start_ms - new.span.start_ms) > timestamp_tolerance_ms
        for old, new in zip(anchor, observed)
    ):
        return reject("anchor_start_mismatch")
    if any(
        abs(old.span.end_ms - new.span.end_ms) > timestamp_tolerance_ms
        for old, new in zip(anchor[:-1], observed[:-1])
    ):
        return reject("nonterminal_end_mismatch")

    extension = terminal.span.end_ms - frozen_terminal.span.end_ms
    if abs(extension) <= timestamp_tolerance_ms:
        return TerminalEndReconciliationProposal(
            "not_needed", "within_strict_tolerance", start, end, extension
        )
    if sum(_lexical(word) for word in anchor) < 2:
        return reject("insufficient_lexical_anchor")
    if not _lexical(frozen_terminal):
        return reject("nonlexical_terminal_anchor")
    if frozen_terminal.span.end_ms != committed_through_ms:
        return reject("anchor_not_at_watermark")
    if extension <= 0:
        return reject("terminal_end_not_rightward")
    if extension > max_end_extension_ms:
        return reject("extension_exceeds_cap")

    continuation = words[end:]
    if not continuation or not any(_lexical(word) for word in continuation):
        return reject("missing_lexical_continuation")
    if continuation[0].span.start_ms < max(committed_through_ms, terminal.span.end_ms):
        return reject("continuation_overlaps_boundary")
    return TerminalEndReconciliationProposal(
        "eligible", "terminal_end_only_extension", start, end, extension
    )


def _lexical(word: NativeTimestampSegment) -> bool:
    return any(character.isalnum() for character in word.text)


@dataclass(frozen=True, slots=True)
class ResolutionHandoffAssessment:
    """Conditional structural checks, never a publication or acoustic certificate.

    Even eligibility leaves common omissions and model/options provenance
    unproven. The observations do not contain those proofs. Missing evidence is
    explicit, and the original refusal is not overwritten by a candidate result.
    """

    status: Literal["needs_overlap", "rejected", "structurally_eligible"]
    reason: str
    original_reason: str
    original_anchor_diagnostic: AnchorDiagnostic | None
    missing_evidence: tuple[str, ...]
    candidate_window: tuple[int, int] | None = None
    overlap_anchor_start: int | None = None
    overlap_anchor_end: int | None = None
    publication_authorized: Literal[False] = False


def assess_resolution_handoff(
    probe: ContinuousResolutionObservation,
    *,
    current_session_version: int,
    committed_through_sample: int,
    retained_from_sample: int,
    retained_pcm: bytes | None = None,
    overlap: ContinuousResolutionObservation | None = None,
    overlap_attempted: bool = False,
    timestamp_tolerance_ms: int = 200,
) -> ResolutionHandoffAssessment:
    """Assess one head-only probe against at most one anchor-bearing witness.

    The future witness starts at the first frozen anchor's estimated onset,
    rounded backward to the absolute 20 ms grid, and ends at the same EOF.
    No search over starts, tolerance fitting, punctuation normalization or
    suffix-subsequence choice is allowed. Both complete suffixes must agree.
    This is stricter than the live policy, not a new publication policy.

    ``retained_pcm`` is mono 16 kHz s16le at ``retained_from_sample``. Its exact
    slices must match the hashes recorded by both observations. This verifies
    byte correspondence, not how/with which model the words were generated.
    A future producer may use the existing immutable observation shape for its
    overlapping window; the current runtime produces head-only probes only.
    """
    if not isinstance(probe, ContinuousResolutionObservation):
        raise TypeError("probe must be ContinuousResolutionObservation")
    if overlap is not None and not isinstance(overlap, ContinuousResolutionObservation):
        raise TypeError("overlap must be ContinuousResolutionObservation or None")
    if type(overlap_attempted) is not bool:
        raise TypeError("overlap_attempted must be a boolean")
    for value in (
        current_session_version,
        committed_through_sample,
        retained_from_sample,
        timestamp_tolerance_ms,
        probe.session_version,
        probe.analysis_start_sample,
        probe.analysis_end_sample,
    ):
        if type(value) is not int:
            raise TypeError("versions, sample bounds and tolerance must be integers")
        if value < 0:
            raise ValueError(
                "versions, sample bounds and tolerance must be nonnegative"
            )
    if retained_pcm is not None and not isinstance(retained_pcm, bytes):
        raise TypeError("retained_pcm must be bytes or None")
    if retained_pcm is not None and len(retained_pcm) % 2:
        raise ValueError("retained_pcm must contain complete s16le samples")

    source = probe.source
    unproven = ("acoustic_boundary_coverage", "model_tokenizer_options_provenance")

    def result(status, reason, *, missing=(), window=None, indices=None):
        return ResolutionHandoffAssessment(
            status,
            reason,
            source.reason,
            source.anchor_diagnostic,
            tuple(missing) + unproven,
            window,
            None if indices is None else indices[0],
            None if indices is None else indices[1],
        )

    def reject(reason):
        return result("rejected", reason, missing=(reason,))

    if probe.session_version != current_session_version:
        return reject("stale_session_version")
    if (
        source.committed_before_sample != committed_through_sample
        or source.retained_from_sample != retained_from_sample
    ):
        return reject("stale_frozen_boundary")
    if (
        source.eof is not True
        or source.action != "unresolved"
        or source.word_publication is not None
        or source.publication_span is not None
    ):
        return reject("source_is_not_an_eof_refusal")
    if (
        source.word_alignment is None
        or source.word_alignment.native != source.result
        or source.result.analyzed_span.start_ms * 16 != source.analysis_start_sample
        or source.result.analyzed_span.end_ms * 16 != source.analysis_end_sample
    ):
        return reject("source_alignment_span_mismatch")
    if probe.status != "observed" or probe.candidate is None:
        return reject("head_probe_not_observed")
    head, end = committed_through_sample, source.analysis_end_sample
    if (
        not retained_from_sample <= source.analysis_start_sample < head < end
        or (probe.analysis_start_sample, probe.analysis_end_sample) != (head, end)
        or probe.candidate.native.analyzed_span.start_ms * 16 != head
        or probe.candidate.native.analyzed_span.end_ms * 16 != end
    ):
        return reject("head_probe_span_mismatch")
    anchor = probe.anchor
    if (
        not isinstance(anchor, tuple)
        or not 2 <= len(anchor) <= 4
        or any(not isinstance(word, NativeTimestampSegment) for word in anchor)
        or any(not word.tokens or not word.text.strip() for word in anchor)
        or sum(_lexical(word) for word in anchor) < 2
        or not _lexical(anchor[0])
        or not _lexical(anchor[-1])
        or any(a.span.end_ms > b.span.start_ms for a, b in zip(anchor, anchor[1:]))
        or anchor[-1].span.end_ms * 16 != head
    ):
        return reject("insufficient_frozen_lexical_anchor")
    start_ms = anchor[0].span.start_ms // 20 * 20
    window = (start_ms, end // 16)
    if not source.analysis_start_sample < start_ms * 16 < head:
        return reject("no_distinct_retained_overlap")

    def pcm_matches(observation):
        if retained_pcm is None:
            return False
        start = observation.analysis_start_sample - retained_from_sample
        stop = observation.analysis_end_sample - retained_from_sample
        return (
            0 <= start < stop <= len(retained_pcm) // 2
            and sha256(retained_pcm[start * 2 : stop * 2]).hexdigest()
            == observation.pcm_sha256
        )

    if retained_pcm is not None and not pcm_matches(probe):
        return reject("head_probe_pcm_mismatch")
    if source.audio_evidence is None or retained_pcm is None:
        unproven += ("original_source_pcm_correspondence",)
    else:
        original_pcm = source.audio_evidence.observation
        start = source.analysis_start_sample - retained_from_sample
        stop = source.analysis_end_sample - retained_from_sample
        if (
            original_pcm.sample_count != stop - start
            or stop > len(retained_pcm) // 2
            or sha256(retained_pcm[start * 2 : stop * 2]).hexdigest()
            != original_pcm.pcm_sha256.lower()
        ):
            return reject("original_source_pcm_mismatch")
    if overlap is None:
        if overlap_attempted:
            return reject("overlap_attempt_exhausted")
        missing = ("anchor_bearing_overlap_observation",)
        if retained_pcm is None:
            missing += ("retained_pcm_correspondence",)
        return result(
            "needs_overlap",
            "head_only_has_no_anchor_witness",
            missing=missing,
            window=window,
        )
    if overlap.source != source or overlap.anchor != anchor:
        return reject("overlap_frozen_state_mismatch")
    if (
        type(overlap.session_version) is not int
        or overlap.session_version != probe.session_version
    ):
        return reject("overlap_stale_session_version")
    if overlap.status != "observed" or overlap.candidate is None:
        return reject("overlap_not_observed")
    alignment = overlap.candidate
    if (
        type(overlap.analysis_start_sample) is not int
        or type(overlap.analysis_end_sample) is not int
        or (overlap.analysis_start_sample, overlap.analysis_end_sample)
        != (start_ms * 16, end)
        or alignment.native.analyzed_span.start_ms != start_ms
        or alignment.native.analyzed_span.end_ms * 16 != end
    ):
        return reject("overlap_span_mismatch")
    if retained_pcm is None:
        return reject("retained_pcm_correspondence")
    if not pcm_matches(overlap):
        return reject("overlap_pcm_mismatch")
    words = alignment.words
    occurrences = [
        index
        for index in range(len(words) - len(anchor) + 1)
        if all(
            old.text == new.text and old.tokens == new.tokens
            for old, new in zip(anchor, words[index : index + len(anchor)])
        )
    ]
    if not occurrences:
        return reject("overlap_anchor_absent")
    if len(occurrences) != 1:
        return reject("overlap_anchor_ambiguous")
    first, stop = occurrences[0], occurrences[0] + len(anchor)
    matched = words[first:stop]
    if any(
        abs(old.span.start_ms - new.span.start_ms) > timestamp_tolerance_ms
        or abs(old.span.end_ms - new.span.end_ms) > timestamp_tolerance_ms
        for old, new in zip(anchor, matched)
    ):
        return reject("overlap_anchor_timing_mismatch")
    suffix = words[stop:]
    if not suffix or not any(_lexical(word) for word in suffix):
        return reject("overlap_has_no_lexical_continuation")
    if suffix[0].span.start_ms < max(head // 16, matched[-1].span.end_ms):
        return reject("overlap_continuation_crosses_boundary")
    candidate_words = probe.candidate.words
    if candidate_words and candidate_words[0].span.start_ms < max(
        head // 16, matched[-1].span.end_ms
    ):
        return reject("head_continuation_crosses_observed_anchor")
    if len(suffix) != len(candidate_words) or any(
        old.text != new.text or old.tokens != new.tokens
        for old, new in zip(suffix, candidate_words)
    ):
        return reject("complete_suffix_disagrees")
    if any(
        abs(old.span.start_ms - new.span.start_ms) > timestamp_tolerance_ms
        or abs(old.span.end_ms - new.span.end_ms) > timestamp_tolerance_ms
        for old, new in zip(suffix, candidate_words)
    ):
        return reject("suffix_timing_mismatch")
    return result(
        "structurally_eligible",
        "strict_overlap_and_suffix_agree",
        indices=(first, stop),
    )
