"""Dependency-free limits for the opt-in open-length PCM protocol."""

import math
from dataclasses import dataclass

LIVE_PROTOCOL = "pcm-websocket/live-v2"
MAX_LIVE_SAMPLES = 3_600 * 16_000
MAX_LIVE_EVENTS = 100_000
MAX_LIVE_SECONDS = 3_700


@dataclass(frozen=True)
class LiveLimits:
    """Finite server session limits, independent of the controller PCM buffer."""

    max_samples: int = MAX_LIVE_SAMPLES
    max_duration_s: float = MAX_LIVE_SECONDS
    input_timeout_s: float = 10.0
    max_events: int = MAX_LIVE_EVENTS
    max_driver_steps: int = 100_000

    def __post_init__(self) -> None:
        for name, cap in (
            ("max_samples", MAX_LIVE_SAMPLES),
            ("max_events", MAX_LIVE_EVENTS),
            ("max_driver_steps", 100_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"invalid live limit: {name}")
        for name, cap in (
            ("max_duration_s", MAX_LIVE_SECONDS),
            ("input_timeout_s", 10),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 < value <= cap
            ):
                raise ValueError(f"invalid live limit: {name}")
