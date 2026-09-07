"""Versioned local execution choices; importable without native dependencies.

Names describe fixed settings, not quality certification. Changing a registered
configuration requires a new name/version. No implicit "latest" alias, device
autodetection, remote negotiation, or mutable registration API is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

from .adapters.audio_endpoints import QuietEndpointConfig
from .adapters.continuous_stream import ContinuousStreamConfig

DEFAULT_PROFILE = "conservative-v1"


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Read-only description returned by the installed profile catalog.

    The native factory accepts the registered name, not caller-created records.
    All choices currently use tiny.en, English, FP32, and one inference lane.
    """

    name: str
    label: str
    native_profile_id: str
    experimental: bool
    stream_config: ContinuousStreamConfig
    reuse_alignment_features: bool


# Spell out every setting: upstream dataclass defaults must not silently change
# the meaning of an already published profile identifier.
_CONSERVATIVE_CONFIG = ContinuousStreamConfig(
    preview_interval_ms=2000,
    max_window_ms=30000,
    max_buffer_ms=40000,
    holdback_ms=2000,
    timestamp_tolerance_ms=200,
    left_context_ms=2000,
    coalesce_previews=False,
    word_alignment=False,
    input_evidence=True,
    source_units=True,
    endpointing=QuietEndpointConfig(
        frame_ms=20,
        quiet_ms=600,
        min_unit_ms=1000,
        max_quiet_unit_ms=10000,
        quiet_peak=32,
    ),
    word_boundary_fallback=True,
    word_context_limit_ms=6000,
    resolution_probe=False,
    eof_context_retry=True,
    defer_word_commits=True,
    max_draft_tokens=0,
    previous_holdback_ms=0,
)
_CATALOG = (
    ExecutionProfile(
        name=DEFAULT_PROFILE,
        label="Standard",
        native_profile_id="tiny.en/cli-fp32-v1",
        experimental=False,
        stream_config=_CONSERVATIVE_CONFIG,
        reuse_alignment_features=False,
    ),
    ExecutionProfile(
        name="low-latency-v1",
        label="Low latency (experimental)",
        native_profile_id="tiny.en/cli-low-latency-fp32-v1",
        experimental=True,
        stream_config=replace(
            _CONSERVATIVE_CONFIG,
            left_context_ms=20000,
            word_context_limit_ms=24000,
            defer_word_commits=False,
        ),
        reuse_alignment_features=True,
    ),
    ExecutionProfile(
        name="low-latency-v2",
        label="Low latency v2 (experimental)",
        native_profile_id="tiny.en/cli-low-latency-fp32-v2",
        experimental=True,
        stream_config=replace(
            _CONSERVATIVE_CONFIG,
            left_context_ms=20000,
            word_context_limit_ms=24000,
            defer_word_commits=False,
            previous_holdback_ms=2000,
        ),
        reuse_alignment_features=True,
    ),
    ExecutionProfile(
        name="experimental-optimized-v1",
        label="Optimized (experimental)",
        native_profile_id="tiny.en/cli-experimental-optimized-fp32-v1",
        experimental=True,
        stream_config=replace(
            _CONSERVATIVE_CONFIG,
            left_context_ms=20000,
            word_context_limit_ms=24000,
            max_draft_tokens=32,
        ),
        reuse_alignment_features=True,
    ),
)
_PROFILES = MappingProxyType({profile.name: profile for profile in _CATALOG})


def list_profiles() -> tuple[ExecutionProfile, ...]:
    """Return the immutable local catalog in display order, default first."""
    return _CATALOG


def get_profile(name: str) -> ExecutionProfile:
    """Resolve an exact versioned name, failing explicitly on unknown names."""
    if not isinstance(name, str):
        raise TypeError("profile name must be a string")
    try:
        return _PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown execution profile {name!r}; choose from {', '.join(_PROFILES)}"
        ) from None
