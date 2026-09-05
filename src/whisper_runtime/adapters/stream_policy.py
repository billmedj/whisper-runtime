"""Pure text and timed-segment comparisons for growing hypotheses.

Inspired by LocalAgreement: Liu, Spanakis, and Niehues (Interspeech 2020),
https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.html,
and its LocalAgreement-2 application to Whisper by Machacek, Dabre, and Bojar
(IJCNLP 2023), https://aclanthology.org/2023.ijcnlp-demo.3/.

``compare_hypotheses`` is a stricter whole-segment policy, not their reference
implementation: text and tokens must match exactly and timestamp shifts are
bounded. ``resolve_text_prefix`` compares only text tokens and supplies no
audio-coverage decision. Neither gaps nor empty text establish silence.
Agreement does not guarantee accuracy or offline-output equivalence.
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


class TextAgreementReason(str, Enum):
    """Text-only agreement; none of these outcomes authorizes audio eviction."""

    CANDIDATE = "candidate"
    INCOMPLETE = "incomplete"
    UNSTABLE = "unstable"
    ANCHOR_MISSING = "anchor_missing"
    ANCHOR_AMBIGUOUS = "anchor_ambiguous"


@dataclass(frozen=True, slots=True)
class TextPrefixDecision:
    """An exact candidate slice of the current hypothesis's closed text tokens.

    Token indices refer to flattened ``metadata.segments[*].tokens``, not raw
    decoder tokens, words, or audio positions. A candidate may split a timed
    segment or a byte-encoded character. Decode and validate its text separately;
    this object supplies no publication, silence, or audio-coverage authority.
    """

    reason: TextAgreementReason
    tokens: tuple[int, ...] = ()
    current_token_start: int | None = None
    current_token_end: int | None = None
    next_anchor: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.reason, TextAgreementReason):
            raise TypeError("reason must be a TextAgreementReason")
        _require_text_tokens(self.tokens, "tokens")
        _require_text_tokens(self.next_anchor, "next_anchor")
        if self.reason is TextAgreementReason.CANDIDATE:
            start, end = self.current_token_start, self.current_token_end
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or not self.tokens
                or end - start != len(self.tokens)
            ):
                raise ValueError("a candidate requires an exact nonempty token slice")
        elif (
            self.tokens
            or self.current_token_start is not None
            or self.current_token_end is not None
        ):
            raise ValueError("only a candidate can contain a token slice")


def resolve_text_prefix(
    previous: NativeWindowResult | None,
    current: NativeWindowResult,
    *,
    committed_anchor: tuple[int, ...] = (),
    max_anchor_tokens: int = 32,
) -> TextPrefixDecision:
    """Compare text tokens independently of timestamp-segment boundaries.

    The caller binds both hypotheses to unchanged prefix audio, model, tokenizer
    and decode options. Analyses must have the same origin and a growing end.
    An empty anchor is valid only for a block with no published text. A supplied
    anchor must occur exactly once in each hypothesis; repeated or missing
    matches are unresolved, never guessed. Only closed parsed segments count.

    ``next_anchor`` is proposed state, to retain only after a separate successful
    publication decision. It is bounded; the candidate is not truncated to that
    bound. Agreement neither proves recognition accuracy nor resolves audio
    gaps, end of speech, character boundaries, holdback, or resource lifetime.
    The current continuous profiles still use ``compare_hypotheses``.
    """
    if previous is not None and not isinstance(previous, NativeWindowResult):
        raise TypeError("previous must be a NativeWindowResult or None")
    if not isinstance(current, NativeWindowResult):
        raise TypeError("current must be a NativeWindowResult")
    _require_text_tokens(committed_anchor, "committed_anchor")
    if isinstance(max_anchor_tokens, bool) or not isinstance(max_anchor_tokens, int):
        raise TypeError("max_anchor_tokens must be an integer")
    if not 0 < max_anchor_tokens <= 448:
        raise ValueError("max_anchor_tokens must be between 1 and 448")
    if len(committed_anchor) > max_anchor_tokens:
        raise ValueError("committed_anchor exceeds max_anchor_tokens")

    def wait(reason: TextAgreementReason) -> TextPrefixDecision:
        return TextPrefixDecision(reason, next_anchor=committed_anchor)

    if previous is None:
        return wait(TextAgreementReason.INCOMPLETE)
    if (
        previous.analyzed_span.start_ms != current.analyzed_span.start_ms
        or previous.analyzed_span.end_ms >= current.analyzed_span.end_ms
    ):
        return wait(TextAgreementReason.UNSTABLE)
    if previous.metadata is None or current.metadata is None:
        return wait(TextAgreementReason.INCOMPLETE)
    if previous.metadata.language != current.metadata.language:
        return wait(TextAgreementReason.UNSTABLE)
    before = tuple(t for s in previous.metadata.segments for t in s.tokens)
    after = tuple(t for s in current.metadata.segments for t in s.tokens)
    if not before or not after:
        return wait(TextAgreementReason.INCOMPLETE)

    before_start = after_start = 0
    if committed_anchor:
        before_matches = _anchor_ends(before, committed_anchor)
        after_matches = _anchor_ends(after, committed_anchor)
        if len(before_matches) > 1 or len(after_matches) > 1:
            return wait(TextAgreementReason.ANCHOR_AMBIGUOUS)
        if not before_matches or not after_matches:
            return wait(TextAgreementReason.ANCHOR_MISSING)
        before_start, after_start = before_matches[0], after_matches[0]

    length = 0
    for left, right in zip(before[before_start:], after[after_start:]):
        if left != right:
            break
        length += 1
    if not length:
        return wait(
            TextAgreementReason.INCOMPLETE
            if before_start == len(before) or after_start == len(after)
            else TextAgreementReason.UNSTABLE
        )
    tokens = after[after_start : after_start + length]
    return TextPrefixDecision(
        TextAgreementReason.CANDIDATE,
        tokens,
        after_start,
        after_start + length,
        (committed_anchor + tokens)[-max_anchor_tokens:],
    )


def _require_text_tokens(tokens: tuple[int, ...], name: str) -> None:
    if not isinstance(tokens, tuple) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in tokens
    ):
        raise TypeError(f"{name} must be an immutable tuple of integer tokens")
    if any(token < 0 for token in tokens):
        raise ValueError(f"{name} tokens must not be negative")


def _anchor_ends(tokens: tuple[int, ...], anchor: tuple[int, ...]) -> list[int]:
    # Two matches suffice to reject ambiguity, including overlapping matches.
    matches = []
    for start in range(len(tokens) - len(anchor) + 1):
        if tokens[start : start + len(anchor)] == anchor:
            matches.append(start + len(anchor))
            if len(matches) == 2:
                break
    return matches
