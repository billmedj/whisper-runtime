"""Bounded logical checkpoint primitives, independent of Whisper and devices."""

import unittest
from dataclasses import FrozenInstanceError, replace

from whisper_runtime import (
    AudioSpan,
    ModelSnapshot,
    Session,
    SessionState,
    StaleSessionError,
    WindowRecord,
    WindowResult,
)
from whisper_runtime.adapters.audio_endpoints import (
    QuietEndpointConfig,
    QuietEndpointDetector,
    QuietEndpointState,
)
from whisper_runtime.adapters.audio_evidence import AudioObservation, SilencePublication
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.word_policy import AlignedPublication, NativeWordAlignment


def pcm(samples: int, value: int = 0) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


class SessionCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ModelSnapshot("checkpoint-model", "1", "scripted", "sha256:test")
        self.record = WindowRecord(
            "request-0", self.model, WindowResult("window-0", "one", 0, 100), 100
        )
        self.state = SessionState("checkpoint-session", 1, (self.record,), 100)

    def test_empty_snapshot_creates_an_independent_holder(self) -> None:
        source = Session("empty")
        restored = Session.from_snapshot(source.snapshot(), history_limit=2)
        self.assertEqual(restored.snapshot(), source.snapshot())
        self.assertEqual(restored.history_limit, 2)
        restored._commit(0, self.record)
        self.assertEqual(source.snapshot().version, 0)
        self.assertEqual(restored.snapshot().version, 1)

    def test_restore_preserves_full_nested_word_and_silence_provenance(self) -> None:
        words = (
            NativeTimestampSegment(AudioSpan(0, 100), " one", (1,)),
            NativeTimestampSegment(AudioSpan(100, 200), " two", (2,)),
        )
        native = NativeWindowResult(
            "aligned",
            "one two",
            0,
            200,
            metadata=NativeDecodeMetadata(
                "en", (1, 2), segments=words, timestamps_complete=True
            ),
        )
        publication = AlignedPublication(
            window_id="aligned",
            text="one",
            start_ms=0,
            end_ms=100,
            analysis_span=AudioSpan(0, 200),
            alignment=NativeWordAlignment(native, words),
            word_start=0,
            word_end=1,
        )
        silence_native = NativeWindowResult("silence", "", 100, 110)
        silence = SilencePublication(
            window_id="silence",
            text="",
            start_ms=100,
            end_ms=110,
            native=silence_native,
            observation=AudioObservation.from_pcm(pcm(160)),
            analysis_start_sample=1600,
        )
        source = Session("provenance")
        source._commit(0, WindowRecord("words", self.model, publication, 100))
        source._commit(1, WindowRecord("quiet", self.model, silence, 110))
        restored = Session.from_snapshot(source.snapshot(), history_limit=4)
        self.assertEqual(restored.snapshot(), source.snapshot())
        for original, recovered in zip(
            source.snapshot().windows, restored.snapshot().windows
        ):
            self.assertIs(original, recovered)
            self.assertIs(original.result, recovered.result)
        next_record = WindowRecord(
            "next", self.model, WindowResult("next", "next", 110, 120), 120
        )
        with self.assertRaises(StaleSessionError):
            restored._commit(1, next_record)
        restored._commit(2, next_record)
        self.assertEqual(source.snapshot().version, 2)
        self.assertEqual(restored.snapshot().version, 3)

    def test_truncated_or_over_capacity_history_is_refused(self) -> None:
        source = Session("truncated", history_limit=1)
        source._commit(0, self.record)
        source._commit(
            1,
            WindowRecord(
                "next", self.model, WindowResult("next", "two", 100, 200), 200
            ),
        )
        with self.assertRaisesRegex(ValueError, "complete publication history"):
            Session.from_snapshot(source.snapshot())
        complete = SessionState(
            "complete", 2, (self.record, source.snapshot().windows[0]), 200
        )
        with self.assertRaisesRegex(ValueError, "history_limit"):
            Session.from_snapshot(complete, history_limit=1)
        for value in (True, False, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                Session.from_snapshot(self.state, history_limit=value)
        with self.assertRaises(ValueError):
            Session.from_snapshot(self.state, history_limit=0)

    def test_publication_replay_rejects_overlap_and_wrong_watermark(self) -> None:
        overlap = WindowRecord(
            "overlap", self.model, WindowResult("overlap", "two", 99, 200), 200
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            Session.from_snapshot(
                SessionState("overlap", 2, (self.record, overlap), 200)
            )
        for watermark in (None, 0, 99, 101):
            with self.subTest(watermark=watermark), self.assertRaises(ValueError):
                Session.from_snapshot(
                    replace(self.state, committed_through_ms=watermark)
                )
        with self.assertRaises(ValueError):
            Session.from_snapshot(SessionState("empty-watermark", 0, (), 100))
        invalid = replace(self.record, committed_through_ms=101)
        with self.assertRaisesRegex(ValueError, "result end"):
            Session.from_snapshot(SessionState("past-end", 1, (invalid,), 101))

    def test_ambiguous_counter_and_record_types_are_refused(self) -> None:
        for value in (None, {}, self.record):
            with self.subTest(value=value), self.assertRaises(TypeError):
                Session.from_snapshot(value)
        for version in (True, False, 1.0):
            with self.subTest(version=version), self.assertRaises(TypeError):
                Session.from_snapshot(replace(self.state, version=version))
        with self.assertRaises(TypeError):
            Session.from_snapshot(replace(self.state, windows=[self.record]))
        with self.assertRaises(TypeError):
            Session.from_snapshot(replace(self.state, windows=(None,)))
        for record, error in (
            (replace(self.record, request_id=1), ValueError),
            (replace(self.record, request_id=" "), ValueError),
            (replace(self.record, model="model"), TypeError),
            (replace(self.record, result="text"), TypeError),
            (
                replace(self.record, result=replace(self.record.result, text=1)),
                TypeError,
            ),
        ):
            with self.subTest(record=record), self.assertRaises(error):
                Session.from_snapshot(replace(self.state, windows=(record,)))

    def test_complete_history_without_audio_watermarks_remains_supported(self) -> None:
        record = replace(self.record, committed_through_ms=None)
        state = SessionState("legacy", 1, (record,))
        self.assertEqual(Session.from_snapshot(state).snapshot(), state)


class QuietEndpointCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = QuietEndpointConfig(
            frame_ms=1, quiet_ms=2, min_unit_ms=3, max_quiet_unit_ms=10, quiet_peak=32
        )

    def test_snapshot_is_frozen_and_empty_state_restores(self) -> None:
        detector = QuietEndpointDetector(self.config)
        state = detector.snapshot()
        self.assertEqual(state, QuietEndpointState())
        self.assertEqual(
            QuietEndpointDetector.from_snapshot(self.config, state).snapshot(), state
        )
        with self.assertRaises(FrozenInstanceError):
            state.accepted_samples = 1

    def test_every_split_restores_partial_frames_quiet_runs_and_proposals(self) -> None:
        source = pcm(32, 100) + pcm(64, 10) + pcm(128) + pcm(80, -32768) + pcm(96)
        control = QuietEndpointDetector(self.config)
        expected = control.observe(0, source)
        for split in range(len(source) // 2 + 1):
            with self.subTest(split=split):
                detector = QuietEndpointDetector(self.config)
                before = detector.observe(0, source[: split * 2]) if split else ()
                snapshot = detector.snapshot()
                restored = QuietEndpointDetector.from_snapshot(self.config, snapshot)
                after = (
                    restored.observe(split, source[split * 2 :])
                    if split < len(source) // 2
                    else ()
                )
                self.assertEqual(before + after, expected)
                self.assertEqual(restored.snapshot(), control.snapshot())
                self.assertEqual(detector.snapshot(), snapshot)

    def test_state_rejects_ambiguous_scalar_types(self) -> None:
        state = QuietEndpointState()
        for name in state.__dataclass_fields__:
            values = (
                (0, 1, None, "false")
                if name == "nonquiet"
                else (True, False, None, 1.0, "1")
            )
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    replace(state, **{name: value})
        for name in state.__dataclass_fields__:
            if name != "nonquiet":
                with self.subTest(name=name), self.assertRaises(ValueError):
                    replace(state, **{name: -1})

    def test_state_rejects_impossible_scalar_ranges(self) -> None:
        for changes in (
            {"last_endpoint": 1},
            {"frame_count": 1},
            {"quiet_count": 1},
            {"frame_peak": 1},
            {"quiet_peak": 1},
            {"accepted_samples": 1, "frame_count": 1, "frame_peak": 32769},
            {"accepted_samples": 16, "quiet_count": 16, "quiet_peak": 32769},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                QuietEndpointState(**changes)

    def test_restore_rejects_frame_grid_threshold_and_causal_corruption(self) -> None:
        for state in (
            QuietEndpointState(accepted_samples=16, frame_count=16),
            QuietEndpointState(accepted_samples=17, quiet_count=16),
            QuietEndpointState(accepted_samples=17, frame_count=1, quiet_count=15),
            QuietEndpointState(accepted_samples=17, frame_count=1, last_endpoint=1),
            QuietEndpointState(accepted_samples=16, last_endpoint=16),
            QuietEndpointState(accepted_samples=16, quiet_count=16, quiet_peak=33),
            QuietEndpointState(accepted_samples=16, quiet_count=16, nonquiet=True),
            QuietEndpointState(accepted_samples=16),
            QuietEndpointState(accepted_samples=48, quiet_count=32, nonquiet=True),
            QuietEndpointState(accepted_samples=160, quiet_count=160),
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                QuietEndpointDetector.from_snapshot(self.config, state)

    def test_restore_requires_exact_snapshot_and_config_types(self) -> None:
        for state in (None, {}, False):
            with self.subTest(state=state), self.assertRaises(TypeError):
                QuietEndpointDetector.from_snapshot(self.config, state)
        for config in (None, {}, False):
            with self.subTest(config=config), self.assertRaises(TypeError):
                QuietEndpointDetector.from_snapshot(config, QuietEndpointState())


if __name__ == "__main__":
    unittest.main()
