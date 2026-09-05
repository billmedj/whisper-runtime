"""Pure, local experiment for one terminal anchor-end extension at EOF.

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
from typing import Literal

from whisper_runtime.adapters.native_result import NativeTimestampSegment
from whisper_runtime.adapters.word_policy import NativeWordAlignment


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
