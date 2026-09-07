"""A released word commit preserves its growing witness only without rebasing."""

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_continuous_endpoint_words import replace_words, word_result
from test_continuous_evidence import EvidenceNativeAdapter
from test_continuous_stream import pcm_ms

from whisper_runtime import TransactionRetainedError
from whisper_runtime.adapters import NativeStreamError, StreamEventKind
from whisper_runtime.adapters import stream_checkpoint as checkpoint
from whisper_runtime.adapters.audio_endpoints import QuietEndpointConfig
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
)
from whisper_runtime.captions import CommittedTranscript


class SameOriginWordCommitTests(unittest.TestCase):
    def stream(self, factory=word_result, **changes):
        config = ContinuousStreamConfig(
            preview_interval_ms=1000,
            max_window_ms=10000,
            max_buffer_ms=12000,
            holdback_ms=1000,
            left_context_ms=6000,
            source_units=True,
            input_evidence=True,
            endpointing=QuietEndpointConfig(max_quiet_unit_ms=5000),
            word_boundary_fallback=True,
            word_context_limit_ms=7000,
            defer_word_commits=True,
        )
        adapter = EvidenceNativeAdapter(factory)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="same-origin",
            config=replace(config, **changes),
            mel_builder=lambda content: content,
        )
        self.addCleanup(stream.close)
        return stream, adapter

    def drain(self, stream):
        events = []
        for _ in range(300):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("unbounded scheduling")

    def test_same_origin_commit_keeps_exact_accepted_witness_without_rewind(self):
        stream, adapter = self.stream()
        source = pcm_ms(4000, 900)
        stream.push(0, source)
        events = self.drain(stream)
        accepted = stream.state.windows[-1].result
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertIs(stream._word_previous, accepted.alignment)
        self.assertIs(stream._previous, accepted.alignment.native)
        self.assertEqual(stream._last_endpoint, 4000 * 16)
        self.assertEqual([call[2] for call in adapter.calls], [1000, 2000, 3000, 4000])
        self.assertEqual(stream.metrics.buffered_samples, 4000 * 16)
        self.assertEqual(adapter.budget.available, adapter.capacity)

        stream.push(1, pcm_ms(1000, 901))
        events += self.drain(stream)
        self.assertEqual(stream.state.version, 2)
        self.assertEqual(stream.metrics.committed_samples, 4000 * 16)
        self.assertEqual(adapter.calls[-1][1:3], (0, 5000))
        self.assertEqual(len(adapter.calls), 5)
        self.assertEqual(adapter.inputs[-1], source + pcm_ms(1000, 901))
        stream.finish_input()
        events += self.drain(stream)
        projection = CommittedTranscript()
        for event in events:
            projection.accept(event)
        self.assertEqual(projection.head, 5000 * 16)
        self.assertEqual(projection.render("txt").split(), [f"w{i}" for i in range(50)])
        self.assertIsNone(stream._word_previous)
        self.assertIsNone(stream._previous)

    def test_nondeferred_profile_preserves_its_original_pair_reset(self):
        stream, _ = self.stream(defer_word_commits=False)
        stream.push(0, pcm_ms(2000, 900))
        self.drain(stream)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(stream._last_endpoint, 1000 * 16)
        self.assertIsNone(stream._word_previous)
        self.assertIsNone(stream._previous)

    def test_same_origin_witness_can_use_remaining_growth_at_hard_bound(self):
        stream, adapter = self.stream(
            max_window_ms=5000,
            left_context_ms=2000,
            word_context_limit_ms=2000,
            holdback_ms=2000,
        )
        stream.push(0, pcm_ms(4000, 900))
        self.drain(stream)
        self.assertEqual(stream.metrics.committed_samples, 2000 * 16)
        self.assertEqual(stream.retained_from_sample, 0)
        self.assertEqual(stream._last_endpoint, 4000 * 16)
        self.assertEqual(stream._next_endpoint, 5000 * 16)
        stream.push(1, pcm_ms(1000, 900))
        stream.step()
        stream.step()
        stream.step()  # Inspect the boundary before the fresh-origin pair starts.
        self.assertEqual(
            [call[2] for call in adapter.calls], [1000, 2000, 3000, 4000, 5000]
        )
        self.assertEqual(stream.metrics.committed_samples, 3000 * 16)
        self.assertEqual(stream.retained_from_sample, 1000 * 16)
        self.assertIsNone(stream._word_previous)

    def test_rebase_still_starts_a_fresh_growing_pair(self):
        stream, adapter = self.stream(
            max_window_ms=5000, left_context_ms=1000, word_context_limit_ms=2000
        )
        stream.push(0, pcm_ms(4000, 900))
        self.drain(stream)
        self.assertEqual(stream.retained_from_sample, 2000 * 16)
        self.assertIsNone(stream._word_previous)
        self.assertIsNone(stream._previous)
        self.assertEqual(stream._last_endpoint, 3000 * 16)
        stream.push(1, pcm_ms(1000, 900))
        events = self.drain(stream)
        self.assertEqual(adapter.calls[-1][1:3], (2000, 5000))
        self.assertFalse(any(e.kind is StreamEventKind.COMMIT for e in events))
        self.assertEqual(stream.state.version, 1)

    def test_quiet_and_silence_units_clear_the_previous_witness(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(4000, 900))
        self.drain(stream)
        stream.push(1, pcm_ms(600))
        self.drain(stream)
        self.assertEqual(stream.last_trace.source_unit.origin, "quiet_run")
        self.assertEqual(stream.retained_from_sample, 4600 * 16)
        self.assertIsNone(stream._word_previous)
        self.assertIsNone(stream._previous)
        self.assertEqual(stream._word_anchor, ())
        self.assertEqual(adapter.budget.available, adapter.capacity)

        silent, _ = self.stream(endpointing=QuietEndpointConfig(max_quiet_unit_ms=1000))
        silent.push(0, pcm_ms(1000))
        self.drain(silent)
        self.assertEqual(silent.last_trace.reason, "digital_silence")
        self.assertIsNone(silent._word_previous)
        self.assertIsNone(silent._previous)
        self.assertEqual(silent._word_anchor, ())

    def test_accepted_witness_does_not_bypass_anchor_word_or_audio_checks(self):
        for failure in ("anchor", "new_word", "speech_score"):
            with self.subTest(failure=failure):

                def factory(window_id, start, end):
                    result = word_result(window_id, start, end)
                    if end != 5000:
                        return result
                    if failure == "speech_score":
                        return replace(
                            result,
                            metadata=replace(result.metadata, no_speech_prob=0.8),
                        )
                    target = 2900 if failure == "anchor" else 3000
                    return replace_words(
                        result,
                        lambda word: (
                            replace(word, text=" changed")
                            if word.span.start_ms == target
                            else word
                        ),
                    )

                stream, adapter = self.stream(factory)
                stream.push(0, pcm_ms(4000, 900))
                self.drain(stream)
                stream.push(1, pcm_ms(1000, 900))
                events = self.drain(stream)
                self.assertFalse(any(e.kind is StreamEventKind.COMMIT for e in events))
                self.assertEqual(stream.state.version, 1)
                self.assertEqual(stream.metrics.committed_samples, 3000 * 16)
                self.assertEqual(stream.retained_from_sample, 0)
                self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_witness_and_cursor_change_only_after_committed_lease_is_released(self):
        stream, adapter = self.stream()
        stream.push(0, pcm_ms(3000, 900))
        self.drain(stream)
        old_witness, before = stream._word_previous, stream.metrics
        stream.push(1, pcm_ms(1000, 900))
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        with patch.object(
            type(run.transaction._lease),
            "release",
            side_effect=RuntimeError("release failed"),
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                stream.step()
            self.assertIsNotNone(raised.exception.committed_state)
            self.assertEqual(stream.state.version, 1)
            self.assertIs(stream._word_previous, old_witness)
            self.assertEqual(stream._last_endpoint, 3000 * 16)
            self.assertEqual(stream.metrics.events_emitted, before.events_emitted)
            self.assertEqual(stream.metrics.committed_samples, 0)
            with self.assertRaises(NativeStreamError):
                stream.step()
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        events = stream.step()
        accepted = stream.state.windows[-1].result
        self.assertIs(stream._word_previous, accepted.alignment)
        self.assertEqual(stream._last_endpoint, 4000 * 16)
        self.assertEqual(sum(e.kind is StreamEventKind.COMMIT for e in events), 1)
        self.assertEqual(stream.step(), ())
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_checkpoint_derives_same_witness_without_new_cache_fields_or_redecode(self):
        stream, original = self.stream()
        stream.push(0, pcm_ms(4000, 900))
        self.drain(stream)
        before = stream._word_previous
        pipeline = "sha256:" + "c" * 64
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            path = Path(directory) / "same-origin.json"
            digest = stream.save_checkpoint(path, pipeline_identity=pipeline)
            self.assertIs(stream._word_previous, before)
            self.assertNotIn(b'"_word_previous"', path.read_bytes())
            fresh = EvidenceNativeAdapter(word_result)
            restored = ContinuousTranscriptStream.from_checkpoint(
                fresh,
                path=path,
                expected_sha256=digest,
                pipeline_identity=pipeline,
                mel_builder=lambda content: content,
            )
            self.addCleanup(restored.close)
            self.assertEqual(fresh.calls, [])
            self.assertIs(
                restored._word_previous, restored.state.windows[-1].result.alignment
            )
            self.assertIs(restored._previous, restored._word_previous.native)
            self.assertEqual(restored.metrics, stream.metrics)
            self.assertEqual(restored._last_endpoint, stream._last_endpoint)
            self.assertEqual(restored._next_endpoint, stream._next_endpoint)
            before_calls = len(original.calls)
            continuations = []
            for current in (stream, restored):
                current.push(1, pcm_ms(1000, 901))
                events = self.drain(current)
                current.finish_input()
                continuations.append(events + self.drain(current))
            self.assertEqual(continuations[0], continuations[1])
            self.assertEqual(restored.state, stream.state)
            self.assertEqual(restored.metrics, stream.metrics)
            self.assertEqual(fresh.calls, original.calls[before_calls:])
            self.assertEqual(
                [call[1:3] for call in fresh.calls], [(0, 5000), (0, 5000)]
            )
            # The second same-span call is explicit EOF, not a nonfinal rewind.
            self.assertEqual(fresh.budget.available, fresh.capacity)

    def test_checkpoint_rejects_mismatched_same_origin_position_even_with_new_digest(
        self,
    ):
        stream, _ = self.stream()
        stream.push(0, pcm_ms(4000, 900))
        self.drain(stream)
        pipeline = "sha256:" + "c" * 64
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            path = Path(directory) / "same-origin.json"
            stream.save_checkpoint(path, pipeline_identity=pipeline)
            saved = checkpoint.storage.decode(path.read_bytes(), checkpoint._TYPES)[
                "checkpoint"
            ]
            changes = (
                replace(saved, position=replace(saved.position, last_endpoint=64016)),
                replace(saved, position=replace(saved.position, next_endpoint=80016)),
                replace(saved, config=replace(saved.config, defer_word_commits=False)),
                replace(
                    saved,
                    position=replace(saved.position, retained=16),
                    pcm=saved.pcm[32:],
                ),
            )
            for index, changed in enumerate(changes):
                with self.subTest(index=index):
                    raw = checkpoint.storage.encode(
                        {"checkpoint": changed}, checkpoint._TYPES
                    )
                    altered = Path(directory) / f"altered-{index}.json"
                    altered.write_bytes(raw)
                    adapter = EvidenceNativeAdapter(word_result)
                    with self.assertRaises(ValueError):
                        ContinuousTranscriptStream.from_checkpoint(
                            adapter,
                            path=altered,
                            expected_sha256=hashlib.sha256(raw).hexdigest(),
                            pipeline_identity=pipeline,
                            mel_builder=lambda content: content,
                        )
                    self.assertEqual(adapter.calls, [])

    def test_checkpoint_refuses_a_foreign_or_half_present_derived_witness(self):
        for field in ("_word_previous", "_previous"):
            with self.subTest(field=field):
                stream, _ = self.stream()
                stream.push(0, pcm_ms(4000, 900))
                self.drain(stream)
                accepted = getattr(stream, field)
                setattr(stream, field, replace(accepted))
                with tempfile.TemporaryDirectory() as directory:
                    directory = Path(directory).resolve()
                    with self.assertRaises(NativeStreamError):
                        stream.save_checkpoint(
                            Path(directory) / "invalid.json",
                            pipeline_identity="sha256:" + "c" * 64,
                        )
                setattr(stream, field, None)
                with tempfile.TemporaryDirectory() as directory:
                    directory = Path(directory).resolve()
                    with self.assertRaises(NativeStreamError):
                        stream.save_checkpoint(
                            Path(directory) / "half.json",
                            pipeline_identity="sha256:" + "c" * 64,
                        )


if __name__ == "__main__":
    unittest.main()
