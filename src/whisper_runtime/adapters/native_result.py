"""Immutable native decode provenance and conservatively timed publication."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from ..state import AudioSpan, WindowResult

_MISSING = object()
_METADATA_FIELDS = (
    "language",
    "tokens",
    "avg_logprob",
    "no_speech_prob",
    "temperature",
    "compression_ratio",
)
_TIMESTAMP_STEP_MS = 20
_MAX_TIMESTAMP_OFFSET = 1_500


class NativeTokenizer(Protocol):
    """The dependency-free tokenizer surface needed to interpret native tokens."""

    timestamp_begin: int
    eot: int

    def decode(self, tokens: list[int]) -> str:
        """Decode text tokens without adding or removing whitespace."""


@dataclass(frozen=True, slots=True)
class NativeTimestampSegment:
    """A model-timed segment, retaining its exact decoded spacing.

    ``tokens`` contains text-token IDs only. Standard Whisper's original boundary
    IDs are ``timestamp_begin + (boundary_ms - analysis_start_ms) // 20``.
    These model predictions are not a guarantee of acoustic alignment accuracy.
    """

    span: AudioSpan
    text: str
    tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.span, AudioSpan):
            raise TypeError("span must be an AudioSpan")
        if not isinstance(self.text, str):
            raise TypeError("segment text must be a string")
        object.__setattr__(self, "tokens", _token_tuple(self.tokens))


@dataclass(frozen=True, slots=True)
class NativeDecodeMetadata:
    """Plain immutable values for the full analysis, never model-owned tensors.

    All scores and tokens describe the full analysis, not a selected publication.
    ``timestamps_complete`` describes the full token sequence; false does not
    invalidate a separately selected, closed prefix of ``segments``.
    """

    language: str
    tokens: tuple[int, ...]
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    temperature: float | None = None
    compression_ratio: float | None = None
    segments: tuple[NativeTimestampSegment, ...] = ()
    timestamps_complete: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty string")
        object.__setattr__(self, "tokens", _token_tuple(self.tokens))
        for name in _METADATA_FIELDS[2:]:
            object.__setattr__(self, name, _score(name, getattr(self, name)))
        if not isinstance(self.segments, (tuple, list)) or not all(
            isinstance(segment, NativeTimestampSegment) for segment in self.segments
        ):
            raise TypeError("segments must contain NativeTimestampSegment values")
        object.__setattr__(self, "segments", tuple(self.segments))
        if not isinstance(self.timestamps_complete, bool):
            raise TypeError("timestamps_complete must be a boolean")


@dataclass(frozen=True, slots=True)
class NativeWindowResult(WindowResult):
    """Published text with immutable provenance from its full native analysis.

    ``publication_segment_indices=None`` denotes the default full backend text;
    explicit indices identify the published subset of ``metadata.segments``.
    """

    metadata: NativeDecodeMetadata | None = None
    publication_segment_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        WindowResult.__post_init__(self)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.metadata is not None and not isinstance(
            self.metadata, NativeDecodeMetadata
        ):
            raise TypeError("metadata must be NativeDecodeMetadata or None")
        if self.publication_segment_indices is None:
            return
        indices = self.publication_segment_indices
        if not isinstance(indices, (tuple, list)) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise TypeError("publication_segment_indices must contain integer indices")
        indices = tuple(indices)
        if (
            not indices
            or indices[0] < 0
            or any(left >= right for left, right in zip(indices, indices[1:]))
        ):
            raise ValueError(
                "publication_segment_indices must be nonempty and increasing"
            )
        if self.metadata is None or indices[-1] >= len(self.metadata.segments):
            raise ValueError(
                "publication_segment_indices must identify metadata segments"
            )
        if any(right != left + 1 for left, right in zip(indices, indices[1:])):
            raise ValueError("publication_segment_indices must be contiguous")
        selected = tuple(self.metadata.segments[index] for index in indices)
        if (
            self.start_ms != selected[0].span.start_ms
            or self.end_ms != selected[-1].span.end_ms
            or self.text != "".join(segment.text for segment in selected).strip()
        ):
            raise ValueError(
                "publication text and span must match the selected segments"
            )
        object.__setattr__(self, "publication_segment_indices", indices)


def build_native_window_result(
    raw: object,
    *,
    window_id: str,
    analysis_span: AudioSpan,
    tokenizer: NativeTokenizer | None = None,
    publication_span: AudioSpan | None = None,
) -> NativeWindowResult:
    """Snapshot a native result and optionally publish whole, timed segments.

    Malformed or unfinished timestamp tails are not guessed. A safely parsed
    prefix may still be selected, while its metadata records incomplete timing.
    The default preserves the backend's full text and the caller's audio bounds.
    """

    text = getattr(raw, "text", None)
    if not isinstance(text, str):
        raise TypeError("the native decode result must contain string text")
    if not isinstance(analysis_span, AudioSpan):
        raise TypeError("analysis_span must be an AudioSpan")

    values = {name: getattr(raw, name, _MISSING) for name in _METADATA_FIELDS}
    metadata = None
    if any(value is not _MISSING for value in values.values()):
        language = values["language"]
        if not isinstance(language, str) or not language.strip():
            raise ValueError("native metadata requires a non-empty language")
        tokens = _token_tuple(values["tokens"])
        segments: tuple[NativeTimestampSegment, ...] = ()
        complete = False
        if tokenizer is not None:
            segments, complete = _timestamp_segments(tokens, tokenizer, analysis_span)
            if (
                complete
                and "".join(segment.text for segment in segments).strip() != text
            ):
                raise ValueError("timestamp segment text does not match native text")
        metadata = NativeDecodeMetadata(
            language=language,
            tokens=tokens,
            avg_logprob=_score("avg_logprob", values["avg_logprob"]),
            no_speech_prob=_score("no_speech_prob", values["no_speech_prob"]),
            temperature=_score("temperature", values["temperature"]),
            compression_ratio=_score("compression_ratio", values["compression_ratio"]),
            segments=segments,
            timestamps_complete=complete,
        )

    result = NativeWindowResult(
        window_id=window_id,
        text=text,
        start_ms=analysis_span.start_ms,
        end_ms=analysis_span.end_ms,
        analysis_span=analysis_span,
        metadata=metadata,
    )
    if publication_span is not None:
        return select_native_publication(result, publication_span)
    return result


def select_native_publication(
    result: NativeWindowResult, publication_span: AudioSpan
) -> NativeWindowResult:
    """Select whole model-timed segments without rebuilding analysis metadata."""

    if not isinstance(result, NativeWindowResult):
        raise TypeError("result must be a NativeWindowResult")
    if not isinstance(publication_span, AudioSpan):
        raise TypeError("publication_span must be an AudioSpan")
    analysis_span = result.analyzed_span
    if (
        publication_span.start_ms < analysis_span.start_ms
        or publication_span.end_ms > analysis_span.end_ms
    ):
        raise ValueError("publication_span must be within analysis_span")
    metadata = result.metadata
    if metadata is None:
        raise ValueError("publication_span requires complete timestamp segments")
    indices = tuple(
        index
        for index, segment in enumerate(metadata.segments)
        if segment.span.start_ms >= publication_span.start_ms
        and segment.span.end_ms <= publication_span.end_ms
    )
    selected = tuple(metadata.segments[index] for index in indices)
    if (
        not selected
        or selected[0].span.start_ms != publication_span.start_ms
        or selected[-1].span.end_ms != publication_span.end_ms
    ):
        raise ValueError(
            "publication_span must exactly bound complete timestamp segments"
        )
    return replace(
        result,
        text="".join(segment.text for segment in selected).strip(),
        start_ms=publication_span.start_ms,
        end_ms=publication_span.end_ms,
        analysis_span=analysis_span,
        publication_segment_indices=indices,
    )


def _token_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("tokens must be a list or tuple of nonnegative integers")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise TypeError("tokens must be nonnegative integers, not booleans")
    if any(token < 0 for token in value):
        raise ValueError("tokens must not be negative")
    return tuple(value)


def _score(name: str, value: object) -> float | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number or None")
    score = float(value)
    if math.isnan(score):
        return None
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite")
    if name == "no_speech_prob" and not 0 <= score <= 1:
        raise ValueError("no_speech_prob must be between zero and one")
    if name in ("temperature", "compression_ratio") and score < 0:
        raise ValueError(f"{name} must not be negative")
    return score


def _timestamp_segments(
    tokens: tuple[int, ...], tokenizer: NativeTokenizer, analysis_span: AudioSpan
) -> tuple[tuple[NativeTimestampSegment, ...], bool]:
    timestamp_begin = tokenizer.timestamp_begin
    eot = tokenizer.eot
    if (
        isinstance(timestamp_begin, bool)
        or not isinstance(timestamp_begin, int)
        or isinstance(eot, bool)
        or not isinstance(eot, int)
        or eot < 0
        or timestamp_begin <= eot
    ):
        raise ValueError("tokenizer must declare ordered eot and timestamp token IDs")

    def timestamp_ms(token: int) -> int | None:
        offset = token - timestamp_begin
        if not 0 <= offset <= _MAX_TIMESTAMP_OFFSET:
            return None
        milliseconds = analysis_span.start_ms + offset * _TIMESTAMP_STEP_MS
        return milliseconds if milliseconds <= analysis_span.end_ms else None

    segments: list[NativeTimestampSegment] = []
    position = 0
    previous_end = analysis_span.start_ms
    while position < len(tokens):
        start_ms = timestamp_ms(tokens[position])
        if start_ms is None or start_ms < previous_end:
            break
        position += 1
        text_start = position
        while position < len(tokens) and tokens[position] < eot:
            position += 1
        if position == len(tokens) or position == text_start:
            break
        end_ms = timestamp_ms(tokens[position])
        if end_ms is None or end_ms <= start_ms:
            break
        text_tokens = tokens[text_start:position]
        text = tokenizer.decode(list(text_tokens))
        if not isinstance(text, str):
            raise TypeError("tokenizer.decode must return a string")
        segments.append(
            NativeTimestampSegment(AudioSpan(start_ms, end_ms), text, text_tokens)
        )
        previous_end = end_ms
        position += 1
        if position == len(tokens):
            return tuple(segments), True
    return tuple(segments), False
