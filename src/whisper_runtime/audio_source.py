"""Bounded mono 16 kHz signed-16LE sources; no implicit audio conversion."""

from __future__ import annotations

import importlib
import math
import queue
import sys
import time
import wave
from collections.abc import Generator
from pathlib import Path
from threading import Event
from typing import Any

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 320


class AudioSourceError(ValueError):
    """Invalid source, missed source deadline, or explicitly detected audio loss."""


def file_chunks(
    path: Path, *, paced: bool = False, cancel: Event | None = None
) -> Generator[bytes, None, None]:
    """Read WAV/PCM incrementally; paced mode never slows its source clock."""
    stop = cancel if cancel is not None else Event()
    extension = path.suffix.lower()
    if extension not in {".wav", ".wave", ".pcm", ".s16le"}:
        raise AudioSourceError(
            "use a mono 16 kHz 16-bit WAV, .pcm or .s16le file; no resampling is performed"
        )
    with path.open("rb") as raw:
        wav = None
        try:
            if extension in {".wav", ".wave"}:
                try:
                    wav = wave.open(raw, "rb")
                except (wave.Error, EOFError) as error:
                    raise AudioSourceError(f"invalid PCM WAV: {error}") from error
                if (
                    wav.getnchannels(),
                    wav.getsampwidth(),
                    wav.getframerate(),
                    wav.getcomptype(),
                ) != (1, 2, SAMPLE_RATE, "NONE"):
                    raise AudioSourceError(
                        "WAV must be uncompressed mono 16 kHz signed 16-bit PCM"
                    )
                total = wav.getnframes()
            else:
                size = raw.seek(0, 2)
                raw.seek(0)
                if size % 2:
                    raise AudioSourceError(
                        "raw PCM ends with an incomplete 16-bit sample"
                    )
                total = size // 2
            if total <= 0:
                raise AudioSourceError("audio source is empty")
            observed, started = 0, time.monotonic()
            while not stop.is_set():
                content = (
                    wav.readframes(CHUNK_SAMPLES)
                    if wav is not None
                    else raw.read(CHUNK_SAMPLES * 2)
                )
                if not content:
                    if observed != total:
                        raise AudioSourceError(
                            "audio file was truncated or changed while being read"
                        )
                    return
                if len(content) % 2 or observed + len(content) // 2 > total:
                    raise AudioSourceError(
                        "audio file contains incomplete samples or changed while being read"
                    )
                observed += len(content) // 2
                if paced:
                    due = started + observed / SAMPLE_RATE
                    while not stop.is_set() and time.monotonic() < due:
                        stop.wait(max(0, min(due - time.monotonic(), 0.05)))
                    if time.monotonic() - due > 0.25:
                        raise AudioSourceError(
                            "paced file source missed its clock by more than 250 ms; no audio was silently dropped"
                        )
                if not stop.is_set():
                    yield content
        finally:
            if wav is not None:
                wav.close()


def microphone_chunks(
    *, duration: float, cancel: Event, device: str | None = None
) -> Generator[bytes, None, None]:
    """Capture a finite recording; callback/queue overflow aborts explicitly."""
    if not math.isfinite(duration) or not 0 < duration <= 3600:
        raise AudioSourceError(
            "microphone duration must be greater than 0 and at most 3600 seconds"
        )
    if sys.byteorder != "little":
        raise AudioSourceError(
            "microphone capture currently supports little-endian hosts only"
        )
    try:
        sd = importlib.import_module("sounddevice")
    except ImportError as error:
        raise AudioSourceError(
            "microphone capture requires the optional 'microphone' extra (sounddevice) and PortAudio"
        ) from error
    target = max(1, int(duration * SAMPLE_RATE))
    chunks: queue.Queue[bytes] = queue.Queue(maxsize=64)
    failure: list[Exception] = []
    captured = 0

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        nonlocal captured
        del time_info
        if cancel.is_set():
            raise sd.CallbackAbort
        if status:
            failure.append(
                AudioSourceError(
                    f"microphone input status: {status}; capture aborted because audio continuity is not guaranteed"
                )
            )
            raise sd.CallbackAbort
        data = bytes(indata)
        if len(data) != frames * 2:
            failure.append(
                AudioSourceError("microphone returned an incomplete PCM frame")
            )
            raise sd.CallbackAbort
        data = data[: (target - captured) * 2]
        try:
            chunks.put_nowait(data)
        except queue.Full:
            failure.append(
                AudioSourceError(
                    "microphone capture queue overflow; no silent drop or restart"
                )
            )
            raise sd.CallbackAbort
        captured += len(data) // 2
        if captured >= target:
            raise sd.CallbackStop

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SAMPLES,
            channels=1,
            dtype="int16",
            device=device,
            callback=callback,
        ) as capture:
            delivered = 0
            last_input = time.monotonic()
            while delivered < target and not cancel.is_set():
                if failure:
                    raise failure[0]
                try:
                    content = chunks.get(timeout=0.1)
                except queue.Empty:
                    if not capture.active or time.monotonic() - last_input > 5:
                        # The final callback can enqueue PCM and stop the device
                        # after get() timed out but before active was inspected.
                        try:
                            content = chunks.get_nowait()
                        except queue.Empty:
                            raise AudioSourceError(
                                "microphone stopped or supplied no input for five seconds"
                            ) from None
                    else:
                        continue
                last_input = time.monotonic()
                delivered += len(content) // 2
                yield content
            if failure:
                raise failure[0]
    except AudioSourceError:
        raise
    except Exception as error:
        raise AudioSourceError(f"microphone capture failed: {error}") from error
