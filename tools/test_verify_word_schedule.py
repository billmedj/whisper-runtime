"""The experimental scheduler can advance a check, never postpone it."""

import unittest
from types import SimpleNamespace

from tools.verify_word_schedule import early_schedule
from whisper_runtime.adapters.continuous_stream import ContinuousTranscriptStream


class WordScheduleTests(unittest.TestCase):
    def test_schedule_is_independent_of_retention_until_reserve(self):
        for left, expected in (
            (2000, 8000),
            (6000, 8000),
            (20000, 8000),
            (24000, 4000),
        ):
            with self.subTest(left=left):
                candidate = object.__new__(early_schedule(ContinuousTranscriptStream))
                candidate._config = SimpleNamespace(
                    left_context_ms=left,
                    preview_interval_ms=2000,
                    max_window_ms=30000,
                )
                origin = 38400
                self.assertEqual(
                    candidate._word_fallback_start(origin), origin + expected * 16
                )
                for name in ("_word_context_start", "_resolve_words"):
                    self.assertIs(
                        getattr(candidate, name).__func__,
                        getattr(ContinuousTranscriptStream, name),
                    )

    def test_invalid_schedule(self):
        for value in (True, False, 0, -20, 8010, 8000.0, "8000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                early_schedule(ContinuousTranscriptStream, value)
