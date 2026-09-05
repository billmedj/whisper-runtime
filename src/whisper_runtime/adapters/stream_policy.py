"""Pure, conservative agreement between two growing timestamped hypotheses.

Inspired by LocalAgreement: Liu, Spanakis, and Niehues (Interspeech 2020),
https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.html,
and its LocalAgreement-2 application to Whisper by Machacek, Dabre, and Bojar
(IJCNLP 2023), https://aclanthology.org/2023.ijcnlp-demo.3/.

This is a stricter whole-segment policy, not their reference implementation:
text and tokens must match exactly, timestamp shifts are bounded, and neither
gaps nor whitespace-only output constitute evidence of silence. Agreement is
not a guarantee of transcription accuracy or offline-output equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..state import AudioSpan
from .native_result import NativeTimestampSegment, NativeWindowResult


class AgreementReason(str, Enum):
    """Whether a consecutive prefix is publishable or why comparison must wait."""

    CANDIDATE = "candidate"
    INCOMPLETE = "incomplete"
    UNSTABLE = "unstable"
    GAP = "gap"


@dataclass(frozen=True, slots=True)
class AgreementDecision:
    """A candidate span in the current hypothesis, or a non-publication reason."""

    reason: AgreementReason
    publication_span: AudioSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, AgreementReason):
            raise TypeError("reason must be an AgreementReason")
        if self.reason is AgreementReason.CANDIDATE:
            if not isinstance(self.publication_span, AudioSpan):
                raise TypeError("a candidate requires an AudioSpan")
            if self.publication_span.start_ms == self.publication_span.end_ms:
                raise ValueError("a candidate span must not be empty")
        elif self.publication_span is not None:
            raise ValueError("only a candidate may contain a publication span")


def compare_hypotheses(
    previous: NativeWindowResult | None,
    current: NativeWindowResult,
    *,
    committed_through_ms: int,
    holdback_ms: int = 1_000,
    timestamp_tolerance_ms: int = 200,
) -> AgreementDecision:
    """Find the agreed, mature prefix beginning exactly at the commit watermark.

    Only two same-start analyses with a strictly growing end count as successive
    evidence. The caller owns the previous result and should replace it only
    after successful transaction cleanup. It must also bind both hypotheses to
    the same stream, unchanged audio prefix, model, and decoding options; these
    identities are not carried by NativeWindowResult. This function retains no
    state.

    A closed prefix remains usable when full-analysis ``timestamps_complete``
    is false. Comparison stops at the first gap, straddling segment, mismatch,
    empty text, or insufficient holdback; any earlier accepted prefix is returned
    without skipping that boundary. Timestamp tolerance never bridges a gap.
    """

    if previous is not None and not isinstance(previous, NativeWindowResult):
        raise TypeError("previous must be a NativeWindowResult or None")
    if not isinstance(current, NativeWindowResult):
        raise TypeError("current must be a NativeWindowResult")
    for name, value in (
        ("committed_through_ms", committed_through_ms),
        ("holdback_ms", holdback_ms),
        ("timestamp_tolerance_ms", timestamp_tolerance_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must not be negative")

    if previous is None:
        return AgreementDecision(AgreementReason.INCOMPLETE)
    previous_span = previous.analyzed_span
    current_span = current.analyzed_span
    if (
        previous_span.start_ms != current_span.start_ms
        or current_span.end_ms <= previous_span.end_ms
    ):
        return AgreementDecision(AgreementReason.UNSTABLE)
    if current_span.start_ms > committed_through_ms:
        return AgreementDecision(AgreementReason.GAP)
    if committed_through_ms >= current_span.end_ms:
        return AgreementDecision(AgreementReason.INCOMPLETE)
    if previous.metadata is None or current.metadata is None:
        return AgreementDecision(AgreementReason.INCOMPLETE)

    previous_segments = previous.metadata.segments
    current_segments = current.metadata.segments
    previous_index = _first_uncommitted(previous_segments, committed_through_ms)
    current_index = _first_uncommitted(current_segments, committed_through_ms)
    previous_end = committed_through_ms
    current_end = committed_through_ms
    stop_reason = AgreementReason.INCOMPLETE
    while previous_index < len(previous_segments) and current_index < len(
        current_segments
    ):
        before = previous_segments[previous_index]
        after = current_segments[current_index]
        if before.span.start_ms > previous_end or after.span.start_ms > current_end:
            stop_reason = AgreementReason.GAP
            break
        if (
            before.span.start_ms < previous_end
            or after.span.start_ms < current_end
            or before.span.end_ms <= before.span.start_ms
            or after.span.end_ms <= after.span.start_ms
            or before.span.end_ms > previous_span.end_ms
            or after.span.end_ms > current_span.end_ms
        ):
            stop_reason = AgreementReason.UNSTABLE
            break
        if (
            not before.tokens
            or not after.tokens
            or not before.text.strip()
            or not after.text.strip()
        ):
            break
        if (
            before.text != after.text
            or before.tokens != after.tokens
            or abs(before.span.start_ms - after.span.start_ms) > timestamp_tolerance_ms
            or abs(before.span.end_ms - after.span.end_ms) > timestamp_tolerance_ms
        ):
            stop_reason = AgreementReason.UNSTABLE
            break
        if after.span.end_ms > current_span.end_ms - holdback_ms:
            break
        previous_end = before.span.end_ms
        current_end = after.span.end_ms
        previous_index += 1
        current_index += 1

    if current_end > committed_through_ms:
        return AgreementDecision(
            AgreementReason.CANDIDATE, AudioSpan(committed_through_ms, current_end)
        )
    return AgreementDecision(stop_reason)


def _first_uncommitted(
    segments: tuple[NativeTimestampSegment, ...], committed_through_ms: int
) -> int:
    index = 0
    while index < len(segments) and segments[index].span.end_ms <= committed_through_ms:
        index += 1
    return index
