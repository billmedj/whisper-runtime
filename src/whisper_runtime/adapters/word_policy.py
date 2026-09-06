"""Pure word agreement with estimated timing and explicit processed coverage.

This opt-in ASR policy is inspired by LocalAgreement, as documented in
``stream_policy``. It does not change native whole-segment publication. Words
reuse NativeTimestampSegment as plain immutable values; their times are alignment
estimates, not acoustic ground truth. Raw native segments remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..state import AudioSpan, WindowResult
from .native_result import NativeTimestampSegment, NativeWindowResult


@dataclass(frozen=True, slots=True)
class NativeWordAlignment:
    """Whole-word estimates bound to one full native result, without tensors."""

    native: NativeWindowResult
    words: tuple[NativeTimestampSegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.native, NativeWindowResult):
            raise TypeError("native must be a NativeWindowResult")
        span = self.native.analyzed_span
        if (
            self.native.publication_segment_indices is not None
            or self.native.start_ms != span.start_ms
            or self.native.end_ms != span.end_ms
        ):
            raise ValueError("word alignment requires the full native result")
        words = _words(self.words)
        if any(
            word.span.start_ms < span.start_ms or word.span.end_ms > span.end_ms
            for word in words
        ):
            raise ValueError("word estimates must be inside the native analysis")
        if "".join(word.text for word in words).strip() != self.native.text:
            raise ValueError("aligned word text must match the full native text")
        object.__setattr__(self, "words", words)


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignedPublication(WindowResult):
    """Exact whole-word text plus explicitly processed source coverage.

    start_ms/end_ms describe processed coverage, including any gaps between
    selected words. They do not certify silence, sound omission-free recognition,
    or exact acoustic alignment. Native provenance and word estimates are intact.
    A final publication relies on the caller's explicit EOF completion authority.
    Explicit boundary_tolerance_ms permits bounded estimated-word overlap at the
    processed start only; it never clips or rewrites the original word times.
    Nonfinal coverage must end on a unit containing Unicode alphanumeric text,
    not standalone punctuation. This is a publication rule, not a speech detector.
    """

    alignment: NativeWordAlignment
    word_start: int
    word_end: int
    final: bool = False
    boundary_tolerance_ms: int = 0

    def __post_init__(self) -> None:
        WindowResult.__post_init__(self)
        if not isinstance(self.alignment, NativeWordAlignment):
            raise TypeError("alignment must be NativeWordAlignment")
        if not isinstance(self.final, bool):
            raise TypeError("final must be a boolean")
        _index(self.word_start, "word_start")
        _index(self.word_end, "word_end")
        _index(self.boundary_tolerance_ms, "boundary_tolerance_ms")
        if self.boundary_tolerance_ms < 0:
            raise ValueError("boundary_tolerance_ms must not be negative")
        words = self.alignment.words
        if not 0 <= self.word_start <= self.word_end <= len(words):
            raise ValueError("publication requires a contiguous whole-word slice")
        native = self.alignment.native
        if (
            self.window_id != native.window_id
            or self.analyzed_span != native.analyzed_span
        ):
            raise ValueError("publication must retain its native window and analysis")
        selected = words[self.word_start : self.word_end]
        if self.text != "".join(word.text for word in selected).strip():
            raise ValueError("publication text must match its selected words")
        if any(
            word.span.start_ms < self.start_ms - self.boundary_tolerance_ms
            or word.span.end_ms > self.end_ms
            for word in selected
        ):
            raise ValueError(
                "selected words exceed processed coverage and its start tolerance"
            )
        if self.final:
            if (
                self.word_end != len(words)
                or self.end_ms != native.analyzed_span.end_ms
            ):
                raise ValueError(
                    "final publication must cover the full remaining suffix"
                )
        elif (
            not selected
            or not _has_lexical_text(selected[-1])
            or self.end_ms != selected[-1].span.end_ms
        ):
            raise ValueError("nonfinal coverage must end at its last lexical word")


def select_word_publication(
    alignment: NativeWordAlignment,
    start: int,
    end: int,
    coverage: AudioSpan,
    final: bool = False,
    *,
    boundary_tolerance_ms: int = 0,
) -> AlignedPublication:
    """Select exact words; coverage is an explicit policy choice, not silence proof."""
    if not isinstance(alignment, NativeWordAlignment):
        raise TypeError("alignment must be NativeWordAlignment")
    if not isinstance(coverage, AudioSpan):
        raise TypeError("coverage must be AudioSpan")
    _index(start, "start")
    _index(end, "end")
    return AlignedPublication(
        window_id=alignment.native.window_id,
        text="".join(word.text for word in alignment.words[start:end]).strip(),
        start_ms=coverage.start_ms,
        end_ms=coverage.end_ms,
        analysis_span=alignment.native.analyzed_span,
        alignment=alignment,
        word_start=start,
        word_end=end,
        final=final,
        boundary_tolerance_ms=boundary_tolerance_ms,
    )


@dataclass(frozen=True, slots=True)
class WordSequenceDiagnostic:
    """Separate text, token, and estimated-time differences without accepting them.

    Case-fold equality is a string property, not semantic equivalence. Deltas
    compare corresponding positions only when every unit is case-fold equal;
    they cannot locate a missing word or establish acoustic correspondence.
    Empty sequences supply no timing witnesses. No publication rule uses this
    diagnostic, and it contains no replacement text or selected word slice.
    """

    left_unit_count: int
    right_unit_count: int
    text_relation: Literal["exact_units", "casefold_equal_units", "different_units"]
    unit_tokens_equal: bool
    max_start_delta_ms: int | None
    max_end_delta_ms: int | None


def diagnose_word_sequence(
    left: tuple[NativeTimestampSegment, ...],
    right: tuple[NativeTimestampSegment, ...],
) -> WordSequenceDiagnostic:
    """Compare complete supplied sequences; do not search or repair alignment.

    The caller supplies the correspondence. Raw units, punctuation, whitespace,
    token segmentation and timing estimates remain intact. The function requires
    no model, reference transcript, threshold, or execution state.
    """
    before, after = _words(left), _words(right)
    same_length = len(before) == len(after)
    exact = same_length and all(a.text == b.text for a, b in zip(before, after))
    folded = same_length and all(
        a.text.casefold() == b.text.casefold() for a, b in zip(before, after)
    )
    return WordSequenceDiagnostic(
        len(before),
        len(after),
        "exact_units"
        if exact
        else "casefold_equal_units"
        if folded
        else "different_units",
        same_length and all(a.tokens == b.tokens for a, b in zip(before, after)),
        max(abs(a.span.start_ms - b.span.start_ms) for a, b in zip(before, after))
        if folded and before
        else None,
        max(abs(a.span.end_ms - b.span.end_ms) for a, b in zip(before, after))
        if folded and before
        else None,
    )


@dataclass(frozen=True, slots=True)
class AnchorDiagnostic:
    """Observed correspondence, not acoustic truth or publication authority.

    Deltas are observed minus frozen milliseconds. Counts include exact repeated
    phrases anywhere in the analysis, including after the committed boundary.
    A matched occurrence retains the existing timing rules; other exact lexical
    occurrences remain visible in the counts. No probability of correctness is
    inferred from those counts.
    """

    status: Literal[
        "matched",
        "unavailable",
        "lexical_missing",
        "token_mismatch",
        "timing_mismatch",
        "relocated",
        "ambiguous",
    ]
    observation: Literal["previous", "current"] = "current"
    text_match_count: int = 0
    lexical_match_count: int = 0
    timed_match_count: int = 0
    word_start: int | None = None
    word_end: int | None = None
    start_deltas_ms: tuple[int, ...] = ()
    end_deltas_ms: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class WordAgreementDecision:
    """A proposed publication and bounded anchor; retain only after commit/release."""

    reason: str
    publication: AlignedPublication | None = None
    next_anchor: tuple[NativeTimestampSegment, ...] = ()
    anchor_diagnostic: AnchorDiagnostic | None = None

    def __post_init__(self) -> None:
        if self.reason not in {
            "candidate",
            "eof",
            "incomplete",
            "unstable",
            "anchor_missing",
            "anchor_ambiguous",
        }:
            raise ValueError("unknown word agreement reason")
        anchor = _anchor(self.next_anchor)
        if self.reason in {"candidate", "eof"}:
            if not isinstance(self.publication, AlignedPublication):
                raise TypeError("a candidate requires AlignedPublication")
            if self.publication.final != (self.reason == "eof"):
                raise ValueError("decision reason must match EOF authority")
        elif self.publication is not None:
            raise ValueError("an unresolved decision cannot publish")
        if self.anchor_diagnostic is not None and not isinstance(
            self.anchor_diagnostic, AnchorDiagnostic
        ):
            raise TypeError("anchor_diagnostic must be AnchorDiagnostic or None")
        object.__setattr__(self, "next_anchor", anchor)


def compare_word_hypotheses(
    previous: NativeWordAlignment | None,
    current: NativeWordAlignment,
    *,
    committed_through_ms: int,
    anchor: tuple[NativeTimestampSegment, ...] = (),
    holdback_ms: int = 1_000,
    timestamp_tolerance_ms: int = 200,
    final: bool = False,
) -> WordAgreementDecision:
    """Agree on exact words, binding committed anchors to frozen source times.

    The caller binds unchanged prefix PCM, stream, model, tokenizer, decode and
    alignment options. Nonfinal observations require the same origin and a
    strictly growing analysis end. EOF finality explicitly waives that test, not
    anchor validation. Gaps are processed under this opt-in policy, never inferred
    to be silence. Empty or punctuation-only text cannot advance a nonfinal
    publication. Trailing standalone nonlexical units wait for a stable lexical
    word or explicit EOF; internal punctuation and all raw estimates stay intact.

    An anchor contains at most four actually published words. Only whole source
    word spans still available after the retained origin are used. A missing or
    ambiguous remaining anchor is never relocated to later repeated text. Published
    anchor times stay frozen rather than drifting with each new alignment. Only
    the first observed word at the exact analysis origin may extend left beyond
    start tolerance: a multi-word anchor must still match its text, tokens, first
    end and every remaining boundary. This handles a window-edge onset estimate,
    not an internal timing shift or a new-word comparison. A validated
    nonempty anchor permits start-boundary drift up to timestamp_tolerance_ms.
    The next anchor uses only newly published words from this single observation,
    avoiding a mixture of overlapping old and new timing estimates.
    """
    if previous is not None and not isinstance(previous, NativeWordAlignment):
        raise TypeError("previous must be NativeWordAlignment or None")
    if not isinstance(current, NativeWordAlignment):
        raise TypeError("current must be NativeWordAlignment")
    for name, value in (
        ("committed_through_ms", committed_through_ms),
        ("holdback_ms", holdback_ms),
        ("timestamp_tolerance_ms", timestamp_tolerance_ms),
    ):
        _index(value, name)
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if not isinstance(final, bool):
        raise TypeError("final must be a boolean")
    anchor = _anchor(anchor)
    if any(word.span.end_ms > committed_through_ms for word in anchor):
        raise ValueError("anchor words must have been published before the watermark")

    diagnostic = None

    def wait(
        reason: str, detail: AnchorDiagnostic | None = None
    ) -> WordAgreementDecision:
        return WordAgreementDecision(
            reason, next_anchor=anchor, anchor_diagnostic=detail
        )

    span = current.native.analyzed_span
    if not span.start_ms <= committed_through_ms <= span.end_ms:
        return wait("unstable")
    if not final:
        if previous is None:
            return wait("incomplete")
        before_span = previous.native.analyzed_span
        before_metadata, after_metadata = (
            previous.native.metadata,
            current.native.metadata,
        )
        if (
            before_span.start_ms != span.start_ms
            or before_span.end_ms >= span.end_ms
            or (before_metadata.language if before_metadata else None)
            != (after_metadata.language if after_metadata else None)
        ):
            return wait("unstable")

    before_start = after_start = 0
    retained_anchor: tuple[NativeTimestampSegment, ...] = ()
    if span.start_ms < committed_through_ms:
        retained_anchor = tuple(
            word
            for word in anchor
            if word.span.start_ms >= span.start_ms and word.span.end_ms > span.start_ms
        )
        if not retained_anchor or (
            not final and not any(_has_lexical_text(word) for word in retained_anchor)
        ):
            return wait("anchor_missing", AnchorDiagnostic("unavailable"))
        diagnostic = diagnose_word_anchor(
            current,
            anchor=retained_anchor,
            committed_through_ms=committed_through_ms,
            timestamp_tolerance_ms=timestamp_tolerance_ms,
        )
        if diagnostic.timed_match_count != 1:
            return wait(
                "anchor_ambiguous"
                if diagnostic.timed_match_count > 1
                else "anchor_missing",
                diagnostic,
            )
        assert diagnostic.word_end is not None
        after_start = diagnostic.word_end
        if not final:
            assert previous is not None
            before_diagnostic = diagnose_word_anchor(
                previous,
                anchor=retained_anchor,
                committed_through_ms=committed_through_ms,
                timestamp_tolerance_ms=timestamp_tolerance_ms,
                observation="previous",
            )
            if before_diagnostic.timed_match_count != 1:
                return wait(
                    "anchor_ambiguous"
                    if before_diagnostic.timed_match_count > 1
                    else "anchor_missing",
                    before_diagnostic,
                )
            assert before_diagnostic.word_end is not None
            before_start = before_diagnostic.word_end

    boundary_tolerance = timestamp_tolerance_ms if retained_anchor else 0
    minimum_start = committed_through_ms - boundary_tolerance
    selected_end = after_start
    if final:
        selected_end = len(current.words)
        if (
            after_start < selected_end
            and current.words[after_start].span.start_ms < minimum_start
        ):
            return wait("unstable", diagnostic)
        coverage_end = span.end_ms
    else:
        assert previous is not None
        stop_reason = "incomplete"
        for before, after in zip(
            previous.words[before_start:], current.words[after_start:]
        ):
            if (
                before.span.start_ms < minimum_start
                or after.span.start_ms < minimum_start
                or not _same_word(before, after, timestamp_tolerance_ms)
            ):
                stop_reason = "unstable"
                break
            if after.span.end_ms > span.end_ms - holdback_ms:
                break
            selected_end += 1
        while selected_end > after_start and not _has_lexical_text(
            current.words[selected_end - 1]
        ):
            selected_end -= 1
        if selected_end == after_start:
            return wait(stop_reason, diagnostic)
        coverage_end = current.words[selected_end - 1].span.end_ms
        if coverage_end <= committed_through_ms:
            return wait("incomplete", diagnostic)
    publication = select_word_publication(
        current,
        after_start,
        selected_end,
        AudioSpan(committed_through_ms, coverage_end),
        final,
        boundary_tolerance_ms=boundary_tolerance,
    )
    selected = current.words[after_start:selected_end]
    next_anchor = selected[-4:] if selected else retained_anchor
    return WordAgreementDecision(
        "eof" if final else "candidate", publication, next_anchor, diagnostic
    )


def _has_lexical_text(word: NativeTimestampSegment) -> bool:
    return any(character.isalnum() for character in word.text)


def _same_word(
    left: NativeTimestampSegment, right: NativeTimestampSegment, tolerance: int
) -> bool:
    return (
        left.text == right.text
        and left.tokens == right.tokens
        and abs(left.span.start_ms - right.span.start_ms) <= tolerance
        and abs(left.span.end_ms - right.span.end_ms) <= tolerance
    )


def diagnose_word_anchor(
    alignment: NativeWordAlignment,
    *,
    committed_through_ms: int,
    anchor: tuple[NativeTimestampSegment, ...],
    timestamp_tolerance_ms: int = 200,
    observation: Literal["previous", "current"] = "current",
) -> AnchorDiagnostic:
    """Classify the existing exact matcher without changing its acceptance rule.

    No text normalization, timestamp repair, model invocation, or reference
    transcript is used. A timing mismatch only describes estimated boundaries.
    The caller supplies the retained, already published anchor. The committed
    boundary may follow a previous observation's end; matching that observation
    does not certify the intervening input. Missing retained context is separate.
    """
    if not isinstance(alignment, NativeWordAlignment):
        raise TypeError("alignment must be NativeWordAlignment")
    for name, value in (
        ("committed_through_ms", committed_through_ms),
        ("timestamp_tolerance_ms", timestamp_tolerance_ms),
    ):
        _index(value, name)
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if observation not in ("previous", "current"):
        raise ValueError("unknown anchor observation")
    anchor = _anchor(anchor)
    watermark, tolerance = committed_through_ms, timestamp_tolerance_ms
    if any(word.span.end_ms > watermark for word in anchor):
        raise ValueError("anchor words must have been published before the watermark")
    span = alignment.native.analyzed_span
    if watermark < span.start_ms or (
        observation == "current" and watermark > span.end_ms
    ):
        raise ValueError("committed boundary must be inside the current analysis")
    if not anchor or any(
        word.span.start_ms < span.start_ms or word.span.end_ms <= span.start_ms
        for word in anchor
    ):
        return AnchorDiagnostic("unavailable", observation)
    words = alignment.words
    text_count = lexical_count = timed_count = 0
    lexical_start = timed_start = None
    eligible = False
    for start in range(len(words) - len(anchor) + 1):
        end = start + len(anchor)
        candidate = words[start:end]
        if not all(old.text == new.text for old, new in zip(anchor, candidate)):
            continue
        text_count += 1
        if not all(old.tokens == new.tokens for old, new in zip(anchor, candidate)):
            continue
        lexical_count += 1
        lexical_start = start
        # Even a unique matching phrase wholly after the watermark is new
        # source material, never a substitute for the committed occurrence.
        if words[end - 1].span.start_ms > watermark or (
            words[end - 1].span.start_ms == watermark
            and anchor[-1].span.start_ms < watermark
        ):
            continue
        eligible = True
        first, frozen = words[start], anchor[0]
        first_matches = _same_word(frozen, first, tolerance) or (
            start == 0
            and len(anchor) >= 2
            and first.span.start_ms == span.start_ms
            and first.span.start_ms < frozen.span.start_ms
            and first.text == frozen.text
            and first.tokens == frozen.tokens
            and abs(first.span.end_ms - frozen.span.end_ms) <= tolerance
        )
        if first_matches and all(
            _same_word(old, new, tolerance)
            for old, new in zip(anchor[1:], words[start + 1 : end])
        ):
            timed_count += 1
            timed_start = start
    status: Literal[
        "matched",
        "unavailable",
        "lexical_missing",
        "token_mismatch",
        "timing_mismatch",
        "relocated",
        "ambiguous",
    ]
    if timed_count == 1:
        status = "matched"
    elif timed_count > 1 or lexical_count > 1:
        status = "ambiguous"
    elif lexical_count:
        status = "timing_mismatch" if eligible else "relocated"
    else:
        status = "token_mismatch" if text_count else "lexical_missing"
    selected = (
        timed_start
        if timed_count == 1
        else (lexical_start if lexical_count == 1 else None)
    )
    candidate = () if selected is None else words[selected : selected + len(anchor)]
    return AnchorDiagnostic(
        status,
        observation,
        text_count,
        lexical_count,
        timed_count,
        selected,
        None if selected is None else selected + len(anchor),
        tuple(
            new.span.start_ms - old.span.start_ms for old, new in zip(anchor, candidate)
        ),
        tuple(new.span.end_ms - old.span.end_ms for old, new in zip(anchor, candidate)),
    )


def _index(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _words(
    value: tuple[NativeTimestampSegment, ...],
) -> tuple[NativeTimestampSegment, ...]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(word, NativeTimestampSegment) for word in value
    ):
        raise TypeError("words must contain NativeTimestampSegment values")
    words = tuple(value)
    if any(not word.tokens or not word.text.strip() for word in words):
        raise ValueError("a word must contain nonempty text and tokens")
    if any(
        left.span.end_ms > right.span.start_ms for left, right in zip(words, words[1:])
    ):
        raise ValueError("word estimates must be ordered without overlap")
    return words


def _anchor(
    value: tuple[NativeTimestampSegment, ...],
) -> tuple[NativeTimestampSegment, ...]:
    if not isinstance(value, tuple):
        raise TypeError("anchor must be an immutable tuple")
    if len(value) > 4:
        raise ValueError("anchor must contain at most four published words")
    return _words(value)
