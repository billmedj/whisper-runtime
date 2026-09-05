"""Bounded PCM facts and a conservative model-supported publication gate.

This is not voice activity detection. Exact digital silence describes the
recorded samples, not whether somebody spoke near a muted microphone. Nonzero
quiet audio is never certified as non-speech. A ``speech_candidate`` combines
nonzero input with lexical model output and a model score; it is not independent
evidence that speech occurred or that every source interval was transcribed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

from ..state import WindowResult
from .native_result import NativeWindowResult

_NO_SPEECH_THRESHOLD = 0.6
_DECISION_REASONS = {
    "non_speech": {"digital_silence"},
    "uncertain": {
        "no_lexical_text",
        "missing_speech_score",
        "conflicting_speech_score",
    },
    "speech_candidate": {"model_speech_candidate"},
}


@dataclass(frozen=True, slots=True)
class AudioObservation:
    """Immutable source-PCM facts; contains no audio or model-owned objects.

    Use ``from_pcm`` on the exact admitted bytes, before preprocessing or model
    padding. The stream owns the binding to sample positions and native analysis.
    A sample count is deliberately not rounded to whole milliseconds.
    """

    sample_count: int
    pcm_sha256: str
    digital_silence: bool

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise TypeError("sample_count must be an integer")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not isinstance(self.pcm_sha256, str):
            raise TypeError("pcm_sha256 must be a string")
        if len(self.pcm_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.pcm_sha256
        ):
            raise ValueError(
                "pcm_sha256 must contain exactly 64 hexadecimal characters"
            )
        if not isinstance(self.digital_silence, bool):
            raise TypeError("digital_silence must be a boolean")

    @classmethod
    def from_pcm(cls, pcm_s16le: bytes) -> AudioObservation:
        """Observe nonempty, complete mono s16le samples without an energy cutoff."""
        if not isinstance(pcm_s16le, bytes):
            raise TypeError("pcm_s16le must be bytes")
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise ValueError("PCM must contain one or more complete 16-bit samples")
        return cls(
            sample_count=len(pcm_s16le) // 2,
            pcm_sha256=hashlib.sha256(pcm_s16le).hexdigest(),
            digital_silence=not any(pcm_s16le),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SilencePublication(WindowResult):
    """Empty coverage for exact digital-zero input, retaining native provenance.

    The observation covers the full analysis. Its exact sample origin and count
    jointly project to the native millisecond bounds; neither endpoint needs to
    be millisecond-aligned. The stream retains exact publication samples too.
    Publication may exclude previously committed context but must reach the
    analysis end. The caller binds the observation to the admitted source PCM;
    neither model scores nor empty model output can supply this authority.
    """

    native: NativeWindowResult
    observation: AudioObservation
    analysis_start_sample: int

    def __post_init__(self) -> None:
        WindowResult.__post_init__(self)
        if not isinstance(self.native, NativeWindowResult):
            raise TypeError("native must be NativeWindowResult")
        if not isinstance(self.observation, AudioObservation):
            raise TypeError("observation must be AudioObservation")
        if isinstance(self.analysis_start_sample, bool) or not isinstance(
            self.analysis_start_sample, int
        ):
            raise TypeError("analysis_start_sample must be an integer")
        if self.analysis_start_sample < 0:
            raise ValueError("analysis_start_sample must not be negative")
        analysis = self.native.analyzed_span
        if (
            self.native.publication_segment_indices is not None
            or self.native.start_ms != analysis.start_ms
            or self.native.end_ms != analysis.end_ms
        ):
            raise ValueError("silence publication requires the full native result")
        if self.window_id != self.native.window_id or self.analyzed_span != analysis:
            raise ValueError("publication must retain its native window and analysis")
        if self.text != "":
            raise ValueError("silence publication text must be empty")
        if self.end_ms != analysis.end_ms:
            raise ValueError("silence publication must reach the analysis end")
        if not self.observation.digital_silence:
            raise ValueError("silence publication requires exact digital-zero PCM")
        if (
            self.analysis_start_sample // 16 != analysis.start_ms
            or (self.analysis_start_sample + self.observation.sample_count) // 16
            != analysis.end_ms
        ):
            raise ValueError(
                "observation sample boundaries must match the full analysis"
            )


@dataclass(frozen=True, slots=True)
class AudioEvidenceDecision:
    """A policy classification, not acoustic truth or audio-discard authority.

    The score still describes the entire native analysis. Neither this decision
    nor its source observation validates word timing or any particular gap.
    """

    observation: AudioObservation
    state: Literal["speech_candidate", "non_speech", "uncertain"]
    reason: str
    no_speech_prob: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, AudioObservation):
            raise TypeError("observation must be AudioObservation")
        if not isinstance(self.state, str) or self.state not in _DECISION_REASONS:
            raise ValueError("unknown audio evidence state")
        if (
            not isinstance(self.reason, str)
            or self.reason not in _DECISION_REASONS[self.state]
        ):
            raise ValueError("reason must match the audio evidence state")
        if (self.state == "non_speech") != self.observation.digital_silence:
            raise ValueError("only exact digital silence is classified as non-speech")
        score = self.no_speech_prob
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("no_speech_prob must be a finite number or None")
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(
                    "no_speech_prob must be finite and between zero and one"
                )
            object.__setattr__(self, "no_speech_prob", float(score))
        if self.reason == "missing_speech_score" and score is not None:
            raise ValueError("missing_speech_score requires an absent score")
        if self.reason == "conflicting_speech_score" and (
            score is None or score < _NO_SPEECH_THRESHOLD
        ):
            raise ValueError(
                "conflicting_speech_score requires a score of at least 0.6"
            )
        if self.state == "speech_candidate" and (
            score is None or score >= _NO_SPEECH_THRESHOLD
        ):
            raise ValueError("speech_candidate requires a score below 0.6")


def assess_audio(
    observation: AudioObservation, result: NativeWindowResult
) -> AudioEvidenceDecision:
    """Classify one bound observation/result without changing either value.

    Exact digital zeros take precedence over hallucinated text and all model
    confidence scores. All other PCM remains non-speech-unproven: absent lexical
    text or absent/conflicting model scores yield uncertainty, never silence.
    No amplitude threshold, EOF exception, or confidence override is applied.
    """
    if not isinstance(observation, AudioObservation):
        raise TypeError("observation must be AudioObservation")
    if not isinstance(result, NativeWindowResult):
        raise TypeError("result must be NativeWindowResult")
    score = result.metadata.no_speech_prob if result.metadata is not None else None
    if observation.digital_silence:
        return AudioEvidenceDecision(
            observation, "non_speech", "digital_silence", score
        )
    if not any(character.isalnum() for character in result.text):
        return AudioEvidenceDecision(observation, "uncertain", "no_lexical_text", score)
    if score is None:
        return AudioEvidenceDecision(observation, "uncertain", "missing_speech_score")
    if score >= _NO_SPEECH_THRESHOLD:
        return AudioEvidenceDecision(
            observation, "uncertain", "conflicting_speech_score", score
        )
    return AudioEvidenceDecision(
        observation, "speech_candidate", "model_speech_candidate", score
    )
