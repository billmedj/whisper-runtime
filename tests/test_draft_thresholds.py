"""Score boundaries through the real gate/controller, with reconstructed native results.

Draft hints are recorded, not numerically executed; no backend weights or device.
"""

import math
import struct
import unittest
from dataclasses import asdict, replace

from test_continuous_evidence import EvidenceNativeAdapter
from test_continuous_stream import pcm_ms, timed_result

from tools.audit_draft_thresholds import boundary_cases, classify, raw_result
from whisper_runtime.adapters.audio_evidence import (
    _NO_SPEECH_THRESHOLD,
    AudioObservation,
)
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)

BELOW = math.nextafter(0.6, -math.inf)
ABOVE = math.nextafter(0.6, math.inf)


class RecordingAdapter(EvidenceNativeAdapter):
    def __init__(self, score, avg_logprob, compression_ratio):
        def result(window_id, start, end):
            native = timed_result(window_id, start, end)
            return replace(
                native,
                metadata=replace(
                    native.metadata,
                    tokens=tuple(range(32)),
                    no_speech_prob=score,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                ),
            )

        super().__init__(result)
        self.drafts = []

    def start_window(self, *, draft_tokens=(), **kwargs):
        self.drafts.append(draft_tokens)
        return super().start_window(**kwargs)


class DraftThresholdTests(unittest.TestCase):
    def exercise(
        self,
        limit,
        score,
        *,
        avg_logprob=-1,
        compression_ratio=2.4,
        evidence=True,
        digital_silence=False,
        source_unit=False,
    ):
        adapter = RecordingAdapter(score, avg_logprob, compression_ratio)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="threshold-test",
            mel_builder=lambda pcm: pcm,
            config=ContinuousStreamConfig(
                preview_interval_ms=100,
                max_window_ms=500,
                max_buffer_ms=600,
                holdback_ms=100,
                timestamp_tolerance_ms=0,
                input_evidence=evidence,
                source_units=source_unit,
                max_draft_tokens=limit,
            ),
        )
        self.addCleanup(stream.close)
        events = []

        def drain():
            for _ in range(20):
                if not stream.ready:
                    return
                events.extend(asdict(event) for event in stream.step())
            self.fail("bounded fixture failed to quiesce")

        source = pcm_ms(100, 0 if digital_silence else 1)
        stream.push(0, source)
        drain()  # Populate the hint before the next decode, including rejected previews.
        stream.push(1, source)
        if source_unit:
            stream.seal_unit(3200)
        else:
            stream.finish_input()
        unresolved = False
        try:
            drain()
        except StreamNeedsResolutionError:
            unresolved = True
        self.assertEqual(len(adapter.drafts), 2)
        self.assertEqual(adapter.drafts[0], ())
        # Even refused/silence results can propose IDs; they are not accepted text.
        expected_hint = tuple(range(32)) if limit else ()
        self.assertEqual(adapter.drafts[1], expected_hint)
        self.assertTrue(
            all(run.closed and run.capacity_released for run in adapter.runs)
        )
        self.assertEqual(adapter.budget.available, adapter.capacity)
        if unresolved:
            self.assertEqual(stream.metrics.committed_samples, 0)
            self.assertEqual(stream.metrics.buffered_samples, 3200)
            self.assertEqual(stream.retained_from_sample, 0)
            self.assertEqual(stream.state.version, 0)
            self.assertFalse(stream.done)
            self.assertFalse(any(e["kind"] in ("commit", "final") for e in events))
        else:
            self.assertEqual(stream.done, not source_unit)
            self.assertEqual(stream.metrics.committed_samples, 3200)
            self.assertEqual(stream.metrics.buffered_samples, 0)
        return dict(
            events=events,
            unresolved=unresolved,
            committed_samples=stream.metrics.committed_samples,
            state=stream.last_trace.audio_evidence.state if evidence else None,
        )

    def test_fixed_production_threshold_and_adjacent_float_decisions(self):
        self.assertEqual(_NO_SPEECH_THRESHOLD, 0.6)
        self.assertEqual(
            [c["decision"]["state"] for c in boundary_cases()],
            ["speech_candidate", "uncertain", "uncertain"],
        )

    def test_identical_scores_have_exact_draft0_32_events_below_at_above(self):
        for score in (BELOW, 0.6, ABOVE):
            with self.subTest(score=score):
                a, b = (self.exercise(limit, score) for limit in (0, 32))
                self.assertEqual(a, b)
                self.assertEqual(a["unresolved"], score >= 0.6)

    def test_crossing_in_either_direction_changes_publication_without_token_change(
        self,
    ):
        for control, candidate in ((BELOW, 0.6), (0.6, BELOW), (BELOW, ABOVE)):
            with self.subTest(control=control, candidate=candidate):
                a = self.exercise(0, control)
                b = self.exercise(32, candidate)
                self.assertNotEqual(a["unresolved"], b["unresolved"])
                self.assertNotEqual(a["events"], b["events"])

    def test_same_side_differences_preserve_gate_and_events(self):
        for control, candidate in ((0.599998, BELOW), (0.6, 0.600002)):
            self.assertEqual(self.exercise(0, control), self.exercise(32, candidate))

    def test_source_unit_boundary_has_the_same_inclusive_gate(self):
        for score in (BELOW, 0.6, ABOVE):
            a = self.exercise(0, score, source_unit=True)
            b = self.exercise(32, score, source_unit=True)
            self.assertEqual(a, b)
            self.assertEqual(a["unresolved"], score >= 0.6)

    def test_native_fp32_neighbors_also_cross_the_unchanged_threshold(self):
        # 0.6 itself is not representable in binary32; these bracket it.
        low, high = (
            struct.unpack("!f", struct.pack("!I", bits))[0]
            for bits in (0x3F199999, 0x3F19999A)
        )
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)
        self.assertFalse(self.exercise(0, low)["unresolved"])
        self.assertTrue(self.exercise(32, high)["unresolved"])

    def test_avg_logprob_cannot_override_the_real_speech_gate(self):
        for score in (BELOW, 0.6, ABOVE):
            expected = self.exercise(0, score)
            for avg in (
                math.nextafter(-1.0, -math.inf),
                -1.0,
                math.nextafter(-1.0, math.inf),
                -100.0,
                0.0,
            ):
                with self.subTest(score=score, avg_logprob=avg):
                    self.assertEqual(
                        expected, self.exercise(32, score, avg_logprob=avg)
                    )

    def test_compression_ratio_is_metadata_not_a_runtime_acceptance_gate(self):
        for score in (BELOW, 0.6):
            expected = self.exercise(0, score)
            for ratio in (
                math.nextafter(2.4, -math.inf),
                2.4,
                math.nextafter(2.4, math.inf),
                100.0,
            ):
                self.assertEqual(
                    expected, self.exercise(32, score, compression_ratio=ratio)
                )

    def test_digital_silence_keeps_exact_empty_coverage_across_scores(self):
        expected = self.exercise(0, BELOW, digital_silence=True)
        for score in (BELOW, 0.6, ABOVE):
            self.assertEqual(expected, self.exercise(32, score, digital_silence=True))

    def test_gate_disabled_does_not_invent_an_implicit_threshold(self):
        expected = self.exercise(0, BELOW, evidence=False)
        for score in (BELOW, 0.6, ABOVE):
            self.assertEqual(expected, self.exercise(32, score, evidence=False))

    def test_missing_score_and_nonlexical_text_remain_uncertain(self):
        self.assertEqual(classify(raw_result(None))["reason"], "missing_speech_score")
        for score in (BELOW, 0.6, ABOVE):
            self.assertEqual(
                classify(raw_result(score, text="..."))["reason"], "no_lexical_text"
            )
            silence = AudioObservation.from_pcm(b"\x00\x00" * 16)
            self.assertEqual(
                classify(raw_result(score), silence)["reason"], "digital_silence"
            )


if __name__ == "__main__":
    unittest.main()
