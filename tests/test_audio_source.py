"""PCM file integrity and optional-capture failures without devices or ML."""

import queue
import tempfile
import unittest
import wave
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from whisper_runtime.audio_source import (
    AudioSourceError,
    file_chunks,
    microphone_chunks,
)


class AudioSourceTests(unittest.TestCase):
    def test_raw_and_wav_preserve_every_sample_including_partial_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = b"\x01\x80" * 321
            pcm, wav = (Path(directory) / name for name in ("audio.pcm", "audio.wav"))
            pcm.write_bytes(raw)
            with wave.open(str(wav), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(raw)
            for path in (pcm, wav):
                chunks = list(file_chunks(path))
                self.assertEqual([len(chunk) for chunk in chunks], [640, 2])
                self.assertEqual(b"".join(chunks), raw)
            wav.write_bytes(wav.read_bytes()[:-2])
            with self.assertRaisesRegex(AudioSourceError, "truncated"):
                list(file_chunks(wav))

    def test_wrong_rate_incomplete_raw_and_empty_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.pcm"
            for raw in (b"", b"x"):
                path.write_bytes(raw)
                with self.assertRaises(AudioSourceError):
                    list(file_chunks(path))
            path = path.with_suffix(".wav")
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(44100)
                writer.writeframes(bytes(640))
            with self.assertRaisesRegex(AudioSourceError, "16 kHz"):
                list(file_chunks(path))

    def test_cancellation_and_pacing_deadline_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.pcm"
            path.write_bytes(bytes(640))
            stop = Event()
            stop.set()
            self.assertEqual(list(file_chunks(path, cancel=stop)), [])
            with patch(
                "whisper_runtime.audio_source.time.monotonic", side_effect=[0, 1, 1]
            ):
                with self.assertRaisesRegex(AudioSourceError, "250 ms"):
                    list(file_chunks(path, paced=True))

    def device(self, *, status=False, frames=1, active=True):
        class Stop(Exception):
            pass

        class Abort(Exception):
            pass

        class Capture:
            def __init__(self, **kwargs):
                self.callback, self.active = kwargs["callback"], active
                self.closed = False

            def __enter__(self):
                for _ in range(frames):
                    try:
                        self.callback(bytes(640), 320, None, status)
                    except (Stop, Abort):
                        self.active = False
                        break
                return self

            def __exit__(self, *args):
                self.closed = True

        return SimpleNamespace(
            RawInputStream=Capture, CallbackAbort=Abort, CallbackStop=Stop
        )

    def test_microphone_missing_dependency_fails_only_when_requested(self):
        with patch(
            "whisper_runtime.audio_source.importlib.import_module",
            side_effect=ImportError,
        ):
            with self.assertRaisesRegex(AudioSourceError, "optional 'microphone'"):
                next(microphone_chunks(duration=1, cancel=Event()))

    def test_capture_status_queue_overflow_and_inactive_device_stop_explicitly(self):
        for device, phrase in (
            (self.device(status="overflow"), "continuity"),
            (self.device(frames=65), "queue overflow"),
            (self.device(frames=0, active=False), "stopped"),
        ):
            with (
                self.subTest(phrase=phrase),
                patch(
                    "whisper_runtime.audio_source.importlib.import_module",
                    return_value=device,
                ),
            ):
                with self.assertRaisesRegex(AudioSourceError, phrase):
                    list(microphone_chunks(duration=2, cancel=Event()))

    def test_finite_microphone_capture_trims_only_the_requested_final_frame(self):
        with patch(
            "whisper_runtime.audio_source.importlib.import_module",
            return_value=self.device(frames=2),
        ):
            self.assertEqual(
                [len(c) for c in microphone_chunks(duration=0.025, cancel=Event())],
                [640, 160],
            )

    def test_active_but_silent_capture_callback_has_a_watchdog(self):
        with (
            patch("whisper_runtime.audio_source.time.monotonic", side_effect=[0, 6]),
            patch(
                "whisper_runtime.audio_source.importlib.import_module",
                return_value=self.device(frames=0),
            ),
        ):
            with self.assertRaisesRegex(AudioSourceError, "five seconds"):
                list(microphone_chunks(duration=1, cancel=Event()))

    def test_final_callback_between_queue_timeout_and_inactive_check_is_delivered(self):
        device, captures = self.device(frames=0), []
        payload = b"\x01\x80" * 320

        class Capture(device.RawInputStream):
            def __enter__(self):
                captures.append(self)
                return super().__enter__()

        class QueueAfterTimeout(queue.Queue):
            timed_out = False

            def get(self, *args, **kwargs):
                if not self.timed_out:
                    self.timed_out = True
                    try:
                        super().get(block=False)
                    except queue.Empty:
                        # Model a callback completing immediately after timeout,
                        # before the consumer inspects the device's active flag.
                        capture = captures[0]
                        try:
                            capture.callback(payload, 320, None, False)
                        except device.CallbackStop:
                            capture.active = False
                        raise
                return super().get(*args, **kwargs)

        device.RawInputStream = Capture
        with (
            patch("whisper_runtime.audio_source.queue.Queue", QueueAfterTimeout),
            patch(
                "whisper_runtime.audio_source.importlib.import_module",
                return_value=device,
            ),
        ):
            chunks = list(microphone_chunks(duration=0.02, cancel=Event()))
        self.assertEqual(chunks, [payload])
        self.assertFalse(captures[0].active)
        self.assertTrue(captures[0].closed)


if __name__ == "__main__":
    unittest.main()
