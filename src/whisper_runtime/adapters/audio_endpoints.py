"""Causal quiet-run endpoints for mono 16 kHz s16le PCM.

This amplitude heuristic proposes where to close an audio unit. It does not
identify speech, certify silence, remove samples, or authorize publication.
Quiet speech can satisfy the threshold; endpoint quality needs separate tests.
The detector retains counters and one partial frame, not an audio history.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuietEndpointConfig:
    frame_ms: int = 20
    quiet_ms: int = 600
    min_unit_ms: int = 1_000
    max_quiet_unit_ms: int = 10_000
    quiet_peak: int = 32

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 1 <= self.frame_ms <= 100:
            raise ValueError("frame_ms must be between 1 and 100")
        if (
            not self.frame_ms
            <= self.quiet_ms
            <= self.min_unit_ms
            <= (self.max_quiet_unit_ms)
            <= 30_000
        ):
            raise ValueError("require frame <= quiet <= min unit <= max quiet <= 30000")
        if any(
            value % self.frame_ms
            for value in (self.quiet_ms, self.min_unit_ms, self.max_quiet_unit_ms)
        ):
            raise ValueError("endpoint durations must be multiples of frame_ms")
        if not 0 <= self.quiet_peak <= 32767:
            raise ValueError("quiet_peak must be between 0 and 32767")


@dataclass(frozen=True, slots=True)
class QuietEndpointProposal:
    """Observed quiet range ending at the proposed boundary, not at its onset."""

    end_sample: int
    quiet_start_sample: int
    peak: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.quiet_start_sample < self.end_sample:
            raise ValueError("quiet range must be nonempty and nonnegative")
        if not 0 <= self.peak <= 32768:
            raise ValueError("peak must be between 0 and 32768")


class QuietEndpointDetector:
    """Observe contiguous PCM before scheduling analysis beyond its endpoints.

    Calls must be serialized. The stream does this under its admission lock.
    A rejected stream push must not reach this detector. Frame alignment is
    independent of chunk boundaries. Incomplete frames are not padded at EOF.
    Long quiet runs produce periodic proposals; active audio is never cut just
    to satisfy a time limit. Output size is bounded by admitted chunk duration.
    """

    def __init__(self, config: QuietEndpointConfig | None = None) -> None:
        if config is not None and not isinstance(config, QuietEndpointConfig):
            raise TypeError("config must be QuietEndpointConfig or None")
        self._config = config or QuietEndpointConfig()
        self._accepted = self._frame_count = self._frame_peak = 0
        self._quiet_count = self._quiet_peak = self._last_endpoint = 0
        self._nonquiet = False

    @property
    def config(self) -> QuietEndpointConfig:
        return self._config

    @property
    def accepted_samples(self) -> int:
        return self._accepted

    def observe(
        self, start_sample: int, pcm_s16le: bytes
    ) -> tuple[QuietEndpointProposal, ...]:
        if isinstance(start_sample, bool) or not isinstance(start_sample, int):
            raise TypeError("start_sample must be an integer")
        if start_sample != self._accepted:
            raise ValueError("PCM must start at the next unobserved sample")
        if not isinstance(pcm_s16le, bytes):
            raise TypeError("pcm_s16le must be bytes")
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise ValueError("PCM must contain complete, nonempty 16-bit samples")
        proposals = []
        frame_samples = self.config.frame_ms * 16
        for (sample,) in struct.iter_unpack("<h", pcm_s16le):
            self._accepted += 1
            self._frame_count += 1
            self._frame_peak = max(self._frame_peak, abs(sample))
            if self._frame_count != frame_samples:
                continue
            if self._frame_peak <= self.config.quiet_peak:
                self._quiet_count += frame_samples
                self._quiet_peak = max(self._quiet_peak, self._frame_peak)
            else:
                self._quiet_count = self._quiet_peak = 0
                self._nonquiet = True
            self._frame_count = self._frame_peak = 0
            unit_samples = self._accepted - self._last_endpoint
            if (
                self._quiet_count >= self.config.quiet_ms * 16
                and unit_samples >= self.config.min_unit_ms * 16
                and (
                    self._nonquiet or unit_samples >= self.config.max_quiet_unit_ms * 16
                )
            ):
                proposals.append(
                    QuietEndpointProposal(
                        self._accepted,
                        self._accepted - self._quiet_count,
                        self._quiet_peak,
                    )
                )
                self._last_endpoint = self._accepted
                self._quiet_count = self._quiet_peak = 0
                self._nonquiet = False
        return tuple(proposals)
