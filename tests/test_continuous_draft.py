"""Bounded, stream-local draft hints; no native model or accelerator required."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_continuous_stream import ScriptedNativeAdapter, pcm_ms, timed_result
from test_stream_checkpoint import PIPELINE, first_commit

from whisper_runtime import RuntimeStateError, TransactionRetainedError
from whisper_runtime.adapters import NativeDecodeOptions, StreamEventKind
from whisper_runtime.adapters.continuous_stream import (
    CONTINUOUS_PROFILE,
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
)


class DraftRecordingAdapter(ScriptedNativeAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drafts = []

    def start_window(self, *, draft_tokens=(), **kwargs):
        self.drafts.append(draft_tokens)
        return super().start_window(**kwargs)


def raw_result(window_id, start, end):
    result = timed_result(window_id, start, end)
    # Raw IDs, including timestamp/special IDs, must not be re-tokenized from text.
    tokens = (50364, 42, 50257) + tuple(range(100, 150))
    return replace(result, metadata=replace(result.metadata, tokens=tokens))


class ContinuousDraftTests(unittest.TestCase):
    def setUp(self):
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=500,
            max_buffer_ms=600,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
            max_draft_tokens=32,
        )

    def stream(self, adapter, **kwargs):
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id=kwargs.pop("stream_id", "draft-stream"),
            mel_builder=kwargs.pop("mel_builder", lambda pcm: pcm),
            config=kwargs.pop("config", self.config),
            history_limit=16,
            **kwargs,
        )
        self.addCleanup(stream.close)
        return stream

    def drain(self, stream):
        events = []
        for _ in range(1000):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("stream did not reach an idle boundary")

    def preview(self, stream):
        stream.push(0, pcm_ms(100, 5))
        self.drain(stream)
        self.assertEqual(stream.state.version, 0)

    def test_config_bound_and_greedy_only_opt_in(self):
        self.assertEqual(ContinuousStreamConfig().max_draft_tokens, 0)
        for value in (True, False, 1.0, None, "1", -1, 33):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    replace(self.config, max_draft_tokens=value)
        for value in (0, 1, 32):
            self.assertEqual(
                replace(self.config, max_draft_tokens=value).max_draft_tokens, value
            )
        for options in (
            NativeDecodeOptions(temperature=0.1),
            NativeDecodeOptions(beam_size=1),
            NativeDecodeOptions(beam_size=2),
            NativeDecodeOptions(temperature=0.1, best_of=2),
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.stream(DraftRecordingAdapter(), options=options)
                disabled = self.stream(
                    DraftRecordingAdapter(),
                    options=options,
                    config=replace(self.config, max_draft_tokens=0),
                )
                self.assertEqual(disabled.config.max_draft_tokens, 0)

    def test_default_does_not_send_new_keyword_or_change_profile(self):
        # Original strict adapter signature catches accidental keyword additions.
        adapter = ScriptedNativeAdapter(raw_result)
        stream = self.stream(adapter, config=replace(self.config, max_draft_tokens=0))
        self.assertEqual(stream.profile_id, CONTINUOUS_PROFILE)
        self.preview(stream)
        stream.push(1, pcm_ms(100, 6))
        self.drain(stream)
        self.assertEqual(stream._draft_tokens, ())

    def test_raw_prefix_is_bounded_and_used_only_on_following_window(self):
        adapter = DraftRecordingAdapter(raw_result)
        options = NativeDecodeOptions(prompt=(11, 12), prefix=(13,))
        stream = self.stream(adapter, options=options)
        self.assertNotEqual(stream.profile_id, CONTINUOUS_PROFILE)
        self.preview(stream)
        expected = raw_result("irrelevant", 0, 100).metadata.tokens[:32]
        self.assertEqual(stream._draft_tokens, expected)
        self.assertEqual(adapter.drafts, [()])
        stream.push(1, pcm_ms(100, 6))
        stream.step()
        self.assertEqual(adapter.drafts, [(), expected])
        self.assertEqual(adapter.options, [options, options])
        self.assertEqual(stream._options, options)

    def test_two_streams_sharing_adapter_never_share_hints(self):
        adapter = DraftRecordingAdapter(raw_result)
        first = self.stream(adapter, stream_id="first")
        second = self.stream(adapter, stream_id="second")
        self.preview(first)
        self.assertTrue(first._draft_tokens)
        self.assertEqual(second._draft_tokens, ())
        self.preview(second)
        self.assertEqual(adapter.drafts, [(), ()])
        first.close()
        self.assertEqual(first._draft_tokens, ())
        self.assertTrue(second._draft_tokens)

    def test_configured_hint_limit_is_observed(self):
        adapter = DraftRecordingAdapter(raw_result)
        stream = self.stream(adapter, config=replace(self.config, max_draft_tokens=1))
        self.preview(stream)
        self.assertEqual(stream._draft_tokens, (50364,))
        stream.push(1, pcm_ms(100, 6))
        stream.step()
        self.assertEqual(adapter.drafts, [(), (50364,)])

    def test_result_without_metadata_discards_hint_without_changing_decode(self):
        adapter = DraftRecordingAdapter(raw_result)
        stream = self.stream(adapter)
        self.preview(stream)
        adapter.result_factory = lambda window_id, start, end: replace(
            timed_result(window_id, start, end), metadata=None
        )
        stream.push(1, pcm_ms(100, 6))
        self.drain(stream)
        self.assertEqual(stream._draft_tokens, ())
        stream.finish_input()
        self.assertEqual(self.drain(stream)[-1].kind, StreamEventKind.FINAL)

    def test_close_and_final_drop_hints(self):
        adapter = DraftRecordingAdapter(raw_result)
        stream = self.stream(adapter)
        self.preview(stream)
        self.assertTrue(stream.close())
        self.assertEqual(stream._draft_tokens, ())

        other = self.stream(DraftRecordingAdapter(raw_result), stream_id="final")
        self.preview(other)
        other.finish_input()
        events = self.drain(other)
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
        self.assertEqual(other._draft_tokens, ())
        self.assertFalse(other.close())
        self.assertEqual(other._draft_tokens, ())

    def test_all_native_failures_drop_prior_hint_and_retry_without_it(self):
        for stage in ("start", "step", "prepare", "finish"):
            with self.subTest(stage=stage):
                adapter = DraftRecordingAdapter(raw_result)
                stream = self.stream(adapter, stream_id=stage)
                self.preview(stream)
                adapter.fail_once = stage
                stream.push(1, pcm_ms(100, 7))
                stream.finish_input()
                with self.assertRaises(RuntimeError) as raised:
                    self.drain(stream)
                self.assertIs(raised.exception, adapter.error)
                self.assertEqual(stream._draft_tokens, ())
                self.drain(stream)
                self.assertEqual(adapter.drafts[-1], ())

    def test_preprocessing_failure_drops_prior_hint(self):
        calls = []

        def build(pcm):
            calls.append(pcm)
            if len(calls) == 2:
                raise RuntimeError("preprocessing failed")
            return pcm

        adapter = DraftRecordingAdapter(raw_result)
        stream = self.stream(adapter, mel_builder=build)
        self.preview(stream)
        stream.push(1, pcm_ms(100, 7))
        with self.assertRaisesRegex(RuntimeError, "preprocessing failed"):
            stream.step()
        self.assertEqual(stream._draft_tokens, ())
        stream.step()
        self.assertEqual(adapter.drafts[-1], ())

    def test_cancellation_drops_prior_hint(self):
        adapter = DraftRecordingAdapter(raw_result)
        stream = self.stream(adapter)
        self.preview(stream)
        stream.push(1, pcm_ms(100, 7))
        stream.step()
        self.assertTrue(stream.cancel_active())
        with self.assertRaises(RuntimeStateError):
            stream.step()
        self.assertEqual(stream._draft_tokens, ())
        stream.step()
        self.assertEqual(adapter.drafts[-1], ())

    def test_externally_closed_run_discards_prior_hint(self):
        adapter = DraftRecordingAdapter(raw_result)
        stream = self.stream(adapter)
        self.preview(stream)
        stream.push(1, pcm_ms(100, 7))
        stream.step()
        adapter.runs[-1].close()
        stream.step()
        self.assertEqual(stream._draft_tokens, ())
        stream.step()
        self.assertEqual(adapter.drafts[-1], ())

    def test_retained_cleanup_failure_drops_prior_hint_before_recovery(self):
        adapter = DraftRecordingAdapter(raw_result)
        stream = self.stream(adapter)
        self.preview(stream)
        stream.push(1, pcm_ms(100, 7))
        stream.step()
        stream.step()
        run = adapter.runs[-1]
        run.fence.fail = True
        try:
            with self.assertRaises(TransactionRetainedError):
                stream.step()
            self.assertEqual(stream._draft_tokens, ())
        finally:
            run.fence.fail = False
            adapter.worker.recover(run.transaction)
        stream.step()
        stream.step()
        self.assertEqual(adapter.drafts[-1], ())

    def test_checkpoint_restores_opt_in_but_never_prior_hint(self):
        adapter = DraftRecordingAdapter(raw_result)
        stream = self.stream(adapter)
        first_commit(stream, pcm_ms(350, 9))
        self.assertTrue(stream._draft_tokens)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            path = Path(directory) / "draft-savepoint.json"
            digest = stream.save_checkpoint(path, pipeline_identity=PIPELINE)
            fresh = DraftRecordingAdapter(raw_result)
            restored = ContinuousTranscriptStream.from_checkpoint(
                fresh,
                path=path,
                expected_sha256=digest,
                pipeline_identity=PIPELINE,
                mel_builder=lambda pcm: pcm,
            )
            self.addCleanup(restored.close)
            self.assertEqual(restored.config, stream.config)
            self.assertEqual(restored.profile_id, stream.profile_id)
            self.assertEqual(restored.state, stream.state)
            self.assertEqual(restored._draft_tokens, ())
            self.assertEqual(fresh.drafts, [])
            restored.step()
            self.assertEqual(fresh.drafts, [()])
            saved_fields = json.loads(path.read_bytes())["payload"]["items"][
                "checkpoint"
            ]["fields"]
            self.assertNotIn("draft_tokens", saved_fields)
            self.assertNotIn("draft_tokens", saved_fields["options"]["fields"])

    def test_legacy_v1_checkpoint_missing_opt_in_restores_disabled(self):
        stream = self.stream(
            DraftRecordingAdapter(), config=replace(self.config, max_draft_tokens=0)
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            path = Path(directory) / "savepoint.json"
            stream.save_checkpoint(path, pipeline_identity=PIPELINE)
            raw = json.loads(path.read_bytes())
            raw["payload"]["items"]["checkpoint"]["fields"]["config"]["fields"].pop(
                "max_draft_tokens"
            )
            canonical = json.dumps(
                raw["payload"], sort_keys=True, separators=(",", ":")
            )
            raw["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
            legacy = Path(directory) / "legacy-savepoint.json"
            legacy.write_bytes(json.dumps(raw).encode())
            restored = ContinuousTranscriptStream.from_checkpoint(
                DraftRecordingAdapter(),
                path=legacy,
                expected_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
                pipeline_identity=PIPELINE,
                mel_builder=lambda pcm: pcm,
            )
            self.addCleanup(restored.close)
            self.assertEqual(restored.config.max_draft_tokens, 0)
            self.assertEqual(restored.config, stream.config)
            self.assertEqual(restored.profile_id, CONTINUOUS_PROFILE)


if __name__ == "__main__":
    unittest.main()
