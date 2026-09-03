"""Optional adapters for existing speech-inference APIs."""

from .legacy_whisper import (
    LEGACY_WHISPER_ENVELOPE_VERSION,
    LegacyAdapterError,
    LegacyExecutionMeasurements,
    LegacyExecutionProfile,
    LegacyInputProvenance,
    LegacyOptionsMutationError,
    LegacyPayloadError,
    LegacyTranscribeOptions,
    LegacyTranscriptionEnvelope,
    LegacyTranscriptionRetainedError,
    LegacyWhisperAdapter,
    LegacyWhisperModel,
    ModelIdentityProbe,
)

__all__ = [
    "LEGACY_WHISPER_ENVELOPE_VERSION",
    "LegacyAdapterError",
    "LegacyExecutionMeasurements",
    "LegacyExecutionProfile",
    "LegacyInputProvenance",
    "LegacyOptionsMutationError",
    "LegacyPayloadError",
    "LegacyTranscriptionEnvelope",
    "LegacyTranscriptionRetainedError",
    "LegacyTranscribeOptions",
    "LegacyWhisperAdapter",
    "LegacyWhisperModel",
    "ModelIdentityProbe",
]
