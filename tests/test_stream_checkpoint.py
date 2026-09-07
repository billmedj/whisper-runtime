"""Logical savepoint/fresh-worker replay contracts; no native model or GPU."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import test_continuous_context_retry as retry_fixtures
from test_audio_endpoints import short_config
from test_continuous_endpoint_words import replace_words, word_result
from test_continuous_evidence import EvidenceNativeAdapter
from test_continuous_stream import ScriptedNativeAdapter, pcm_ms

from whisper_runtime import TransactionRetainedError
from whisper_runtime.adapters import (
    AudioSequenceError,
    NativeExecutionProfile,
    NativeStreamError,
    StreamEventKind,
)
from whisper_runtime.adapters import stream_checkpoint as checkpoint
from whisper_runtime.adapters.audio_evidence import SilencePublication
from whisper_runtime.adapters.continuous_stream import (
    ContinuousContextRetry,
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)
from whisper_runtime.adapters.word_policy import AlignedPublication

PIPELINE = "sha256:" + "a" * 64
INITIAL_PCM = pcm_ms(340, 900) + pcm_ms(10, 1)
NEXT_PCM = pcm_ms(30, 1) + pcm_ms(220, 901)


def make_stream(*, hybrid=False, history_limit=16, eof_context_retry=False):
    adapter = EvidenceNativeAdapter(word_result) if hybrid else ScriptedNativeAdapter()
    config = ContinuousStreamConfig(
        preview_interval_ms=100,
        max_window_ms=1000,
        max_buffer_ms=1200,
        holdback_ms=100,
        timestamp_tolerance_ms=0,
        eof_context_retry=eof_context_retry,
        **(
            dict(
                left_context_ms=100,
                source_units=True,
                input_evidence=True,
                endpointing=short_config(),
                word_boundary_fallback=True,
                word_context_limit_ms=400,
            )
            if hybrid
            else {}
        ),
    )
    return (
        ContinuousTranscriptStream(
            adapter,
            stream_id="checkpoint-stream",
            config=config,
            mel_builder=lambda value: value,
            rng_seed=73,
            history_limit=history_limit,
        ),
        adapter,
    )


def first_commit(stream, pcm=INITIAL_PCM):
    stream.push(0, pcm)
    events = []
    for _ in range(100):
        batch = stream.step()
        events.extend(batch)
        if any(event.kind is StreamEventKind.COMMIT for event in batch):
            return events
    raise AssertionError("fixture did not reach its first publication boundary")


def drain(stream):
    events = []
    for _ in range(1000):
        if not stream.ready:
            return events
        events.extend(stream.step())
    raise AssertionError("fixture did not reach an idle scheduling boundary")


def complete(stream):
    stream.push(1, NEXT_PCM)
    events = drain(stream)
    stream.finish_input()
    events.extend(drain(stream))
    return events


def json_value(value):
    return json.loads(json.dumps(value))


def process_stage(stage, directory, hybrid, digest=None):
    """Executed by separately terminated producer and fresh consumer processes."""
    path = Path(directory) / "process-checkpoint.json"
    stream, adapter = make_stream(hybrid=hybrid)
    if stage == "producer":
        prefix = first_commit(stream)
        result = {
            "digest": stream.save_checkpoint(path, pipeline_identity=PIPELINE),
            "prefix": [asdict(event) for event in prefix],
            "pid": os.getpid(),
        }
        stream.close()
    else:
        stream.close()
        stream = ContinuousTranscriptStream.from_checkpoint(
            adapter,
            path=path,
            expected_sha256=digest,
            pipeline_identity=PIPELINE,
            mel_builder=lambda value: value,
        )
        events = complete(stream)
        result = {
            "events": [asdict(event) for event in events],
            "state": asdict(stream.state),
            "metrics": asdict(stream.metrics),
            "calls": adapter.calls,
            "released": adapter.budget.available == adapter.capacity,
            "pid": os.getpid(),
        }
        stream.close()
    print(json.dumps(result))


class StreamCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.counter = 0

    def path(self):
        self.counter += 1
        return self.root / f"save-{self.counter}.json"

    def stream(self, **kwargs):
        stream, adapter = make_stream(**kwargs)
        self.addCleanup(stream.close)
        return stream, adapter

    def save(self, stream):
        path = self.path()
        digest = stream.save_checkpoint(path, pipeline_identity=PIPELINE)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        return path, digest

    def restore(self, adapter, path, digest, **changes):
        restored = ContinuousTranscriptStream.from_checkpoint(
            adapter,
            **{
                "path": path,
                "expected_sha256": digest,
                "pipeline_identity": PIPELINE,
                "mel_builder": lambda value: value,
                **changes,
            },
        )
        self.addCleanup(restored.close)
        return restored

    def assert_released(self, adapter):
        self.assertEqual(adapter.budget.available, adapter.capacity)
        self.assertEqual(adapter.budget.lease_count, 0)
        self.assertEqual(adapter.worker.queue_depth, 0)

    def tamper(self, path, transform):
        payload = checkpoint.storage.decode(path.read_bytes(), checkpoint._TYPES)
        payload["checkpoint"] = transform(payload["checkpoint"])
        raw = checkpoint.storage.encode(payload, checkpoint._TYPES)
        changed = self.path()
        changed.write_bytes(raw)
        return changed, hashlib.sha256(raw).hexdigest()

    def test_fresh_stream_save_restore_does_not_start_inference(self):
        stream, original = self.stream()
        stream.push(0, INITIAL_PCM)
        before = stream.metrics
        path, digest = self.save(stream)
        _, fresh = self.stream()
        restored = self.restore(fresh, path, digest)
        self.assertEqual(restored.state, stream.state)
        self.assertEqual(restored.metrics, before)
        self.assertEqual(bytes(restored._audio), INITIAL_PCM)
        self.assertEqual(restored.expected_chunk, 1)
        self.assertIsNone(restored._run)
        self.assertIsNone(restored.last_trace)
        self.assertIsNone(restored.resolution_observation)
        self.assertEqual(original.calls, [])
        self.assertEqual(fresh.calls, [])
        self.assert_released(original)
        self.assert_released(fresh)

    def test_legacy_v1_config_restores_with_retry_disabled(self):
        stream, _ = self.stream(hybrid=True)
        first_commit(stream)
        path, current_digest = self.save(stream)
        raw = json.loads(path.read_bytes())
        raw["payload"]["items"]["checkpoint"]["fields"]["config"]["fields"].pop(
            "eof_context_retry"
        )
        canonical = json.dumps(raw["payload"], sort_keys=True, separators=(",", ":"))
        raw["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        legacy = self.path()
        legacy.write_bytes(json.dumps(raw).encode())
        _, fresh = self.stream(hybrid=True)
        with self.assertRaisesRegex(ValueError, "trusted file digest"):
            self.restore(fresh, legacy, current_digest)
        digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
        restored = self.restore(fresh, legacy, digest)
        self.assertIs(restored.config.eof_context_retry, False)
        self.assertEqual(restored.config, stream.config)
        self.assertEqual(restored.profile_id, stream.profile_id)
        self.assertEqual(complete(restored), complete(stream))
        self.assertEqual(restored.state, stream.state)
        self.assert_released(fresh)

    def test_retry_opt_in_roundtrips_without_restoring_diagnostics(self):
        stream, _ = self.stream(hybrid=True, eof_context_retry=True)
        first_commit(stream)
        path, digest = self.save(stream)
        _, fresh = self.stream(hybrid=True)
        restored = self.restore(fresh, path, digest)
        self.assertIs(restored.config.eof_context_retry, True)
        self.assertEqual(restored.profile_id, stream.profile_id)
        self.assertIsNone(restored._context_retry)
        self.assertFalse(restored._context_retry_active)
        self.assertEqual(fresh.calls, [])

    def test_fresh_worker_matches_uninterrupted_events_state_and_bounded_pcm(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                stream, original = self.stream(hybrid=hybrid)
                prefix = first_commit(stream)
                before, state = stream.metrics, stream.state
                retained = stream.retained_from_sample
                anchor = stream._word_anchor
                detector = stream._endpoint_detector.snapshot() if hybrid else None
                path, digest = self.save(stream)
                self.assertEqual(stream.state, state)
                self.assertEqual(stream.metrics, before)
                _, fresh = self.stream(hybrid=hybrid)
                restored = self.restore(fresh, path, digest)
                self.assertIsNot(restored._session, stream._session)
                self.assertIsNot(restored._lock, stream._lock)
                self.assertEqual(restored.state, state)
                self.assertEqual(restored.metrics, before)
                self.assertEqual(restored._word_anchor, anchor)
                self.assertEqual(bytes(restored._audio), INITIAL_PCM[retained * 2 :])
                self.assertEqual(restored._options, stream._options)
                self.assertEqual(restored._seed, 73)
                self.assertEqual(restored.profile_id, stream.profile_id)
                if hybrid:
                    self.assertTrue(anchor)
                    self.assertEqual(restored._endpoint_detector.snapshot(), detector)
                    self.assertEqual(detector.frame_count, 10 * 16)
                    self.assertIsInstance(state.windows[-1].result, AlignedPublication)
                    self.assertEqual(
                        restored.state.windows[-1].result.alignment,
                        state.windows[-1].result.alignment,
                    )
                with self.assertRaises(AudioSequenceError):
                    restored.push(0, NEXT_PCM)
                self.assertEqual(restored.metrics, before)
                original_tail = complete(stream)
                restored_tail = complete(restored)
                self.assertEqual(restored_tail, original_tail)
                self.assertEqual(restored.state, stream.state)
                self.assertEqual(restored.metrics, stream.metrics)
                self.assertTrue(restored.done)
                self.assertEqual(restored.metrics.buffered_samples, 0)
                self.assertGreater(
                    restored_tail[0].sequence_number, prefix[-1].sequence_number
                )
                self.assertEqual(
                    sum(event.kind is StreamEventKind.FINAL for event in restored_tail),
                    1,
                )
                for call, pcm in zip(fresh.calls, fresh.inputs):
                    _, start, end, _ = call
                    self.assertGreaterEqual(start * 16, retained)
                    self.assertLessEqual(end - start, restored.config.max_window_ms)
                    self.assertEqual(
                        pcm, (INITIAL_PCM + NEXT_PCM)[start * 32 : end * 32]
                    )
                self.assert_released(original)
                self.assert_released(fresh)

    def test_separate_terminated_producer_and_consumer_match_control(self):
        repo = Path(__file__).resolve().parents[1]
        test_environment = dict(os.environ)
        test_environment["PYTHONPATH"] = os.pathsep.join(
            (str(repo / "src"), str(repo / "tests"))
        )
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid), tempfile.TemporaryDirectory() as folder:
                control, _ = self.stream(hybrid=hybrid)
                prefix, tail = first_commit(control), complete(control)

                def run_stage(stage, digest=None):
                    source = (
                        "from test_stream_checkpoint import process_stage; "
                        f"process_stage({stage!r}, {folder!r}, {hybrid!r}, {digest!r})"
                    )
                    process = subprocess.run(
                        [sys.executable, "-B", "-c", source],
                        cwd=repo,
                        env=test_environment,
                        text=True,
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(process.returncode, 0, process.stderr)
                    return json.loads(process.stdout)

                # subprocess.run has reaped the source process before restoration.
                produced = run_stage("producer")
                consumed = run_stage("consumer", produced["digest"])
                self.assertNotEqual(produced["pid"], consumed["pid"])
                self.assertEqual(
                    produced["prefix"], json_value([asdict(e) for e in prefix])
                )
                self.assertEqual(
                    consumed["events"], json_value([asdict(e) for e in tail])
                )
                self.assertEqual(consumed["state"], json_value(asdict(control.state)))
                self.assertEqual(consumed["metrics"], asdict(control.metrics))
                self.assertTrue(consumed["released"])

    def test_final_checkpoint_never_redelivers_final_or_starts_inference(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                stream, _ = self.stream(hybrid=hybrid)
                first_commit(stream)
                events = complete(stream)
                self.assertIs(events[-1].kind, StreamEventKind.FINAL)
                path, digest = self.save(stream)
                _, fresh = self.stream(hybrid=hybrid)
                restored = self.restore(fresh, path, digest)
                self.assertTrue(restored.done)
                self.assertFalse(restored.ready)
                self.assertEqual(restored.step(), ())
                self.assertEqual(drain(restored), [])
                self.assertEqual(restored.metrics, stream.metrics)
                self.assertEqual(fresh.calls, [])

    def test_queued_endpoint_and_partial_frame_survive_first_unit_publication(self):
        stream, _ = self.stream(hybrid=True)
        source = (
            pcm_ms(100, 900)
            + pcm_ms(40, 1)
            + pcm_ms(100, 901)
            + pcm_ms(40, 2)
            + pcm_ms(110, 902)
        )
        first_commit(stream, source)
        self.assertEqual(stream.metrics.committed_samples, 140 * 16)
        self.assertEqual(len(stream._endpoints), 1)
        self.assertEqual(stream._endpoints[0].end_sample, 280 * 16)
        before = stream._endpoint_detector.snapshot()
        self.assertEqual(before.frame_count, 10 * 16)
        path, digest = self.save(stream)
        _, fresh = self.stream(hybrid=True)
        restored = self.restore(fresh, path, digest)
        self.assertEqual(restored._endpoints, stream._endpoints)
        self.assertEqual(restored._endpoint_detector.snapshot(), before)
        self.assertEqual(complete(restored), complete(stream))
        self.assertEqual(restored.state, stream.state)
        self.assertEqual(restored.metrics, stream.metrics)

    def test_silence_publication_retains_nested_native_and_pcm_evidence(self):
        stream, _ = self.stream(hybrid=True)
        stream.push(0, pcm_ms(120))
        stream.finish_input()
        drain(stream)
        self.assertTrue(stream.done)
        self.assertTrue(
            all(
                isinstance(record.result, SilencePublication)
                for record in stream.state.windows
            )
        )
        path, digest = self.save(stream)
        _, fresh = self.stream(hybrid=True)
        restored = self.restore(fresh, path, digest)
        self.assertEqual(restored.state, stream.state)
        for before, after in zip(stream.state.windows, restored.state.windows):
            self.assertEqual(after.result.native, before.result.native)
            self.assertEqual(after.result.observation, before.result.observation)
            self.assertTrue(after.result.observation.digital_silence)
            self.assertEqual(after.result.text, "")
        self.assertEqual(restored.step(), ())
        self.assertEqual(fresh.calls, [])

    def test_eof_admitted_at_savepoint_replays_only_unfinished_suffix(self):
        stream, _ = self.stream(hybrid=True)
        first_commit(stream)
        stream.finish_input()
        path, digest = self.save(stream)
        _, fresh = self.stream(hybrid=True)
        restored = self.restore(fresh, path, digest)
        self.assertTrue(restored.input_finished)
        self.assertFalse(restored.done)
        self.assertEqual(drain(restored), drain(stream))
        self.assertEqual(restored.state, stream.state)

    def test_integrity_pipeline_model_and_execution_profile_mismatch_refuse(self):
        stream, _ = self.stream()
        first_commit(stream)
        path, digest = self.save(stream)
        _, fresh = self.stream()
        for changes in (
            {"expected_sha256": "0" * 64},
            {"pipeline_identity": "sha256:" + "b" * 64},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.restore(fresh, path, digest, **changes)
        with (
            patch.object(
                fresh,
                "model_identity",
                replace(fresh.model_identity, revision="different"),
            ),
            self.assertRaises(ValueError),
        ):
            self.restore(fresh, path, digest)
        fresh.execution_profile = NativeExecutionProfile("unexpected", fresh.capacity)
        with self.assertRaises(ValueError):
            self.restore(fresh, path, digest)
        self.assertEqual(fresh.calls, [])
        self.assert_released(fresh)

    def test_declared_execution_profile_roundtrips_exactly(self):
        stream, original = self.stream()
        profile = NativeExecutionProfile("test-profile", original.capacity)
        original.execution_profile = profile
        first_commit(stream)
        path, digest = self.save(stream)
        _, fresh = self.stream()
        fresh.execution_profile = profile
        restored = self.restore(fresh, path, digest)
        self.assertEqual(restored.state, stream.state)
        fresh.execution_profile = replace(profile, reuse_decode_features=True)
        with self.assertRaises(ValueError):
            self.restore(fresh, path, digest)

    def test_stream_profile_mismatch_with_recomputed_digest_refuses(self):
        stream, _ = self.stream()
        first_commit(stream)
        path, _ = self.save(stream)
        path, digest = self.tamper(
            path, lambda saved: replace(saved, profile_id="other")
        )
        _, fresh = self.stream()
        with self.assertRaises(ValueError):
            self.restore(fresh, path, digest)
        self.assertEqual(fresh.calls, [])

    def test_cross_field_corruption_refuses_even_with_recomputed_outer_digest(self):
        stream, _ = self.stream(hybrid=True)
        first_commit(stream)
        path, _ = self.save(stream)
        mutations = {
            "head": lambda s: replace(s, position=replace(s.position, head=0)),
            "pcm": lambda s: replace(s, pcm=s.pcm[:-2]),
            "retained": lambda s: replace(
                s, position=replace(s.position, retained=s.position.head + 1)
            ),
            "sequence": lambda s: replace(s, position=replace(s.position, sequence=0)),
            "segment": lambda s: replace(s, position=replace(s.position, segment=0)),
            "cursor": lambda s: replace(s, position=replace(s.position, chunk=0)),
            "endpoint": lambda s: replace(
                s, position=replace(s.position, last_endpoint=0)
            ),
            "premature_done": lambda s: replace(
                s, position=replace(s.position, done=True)
            ),
            "detector": lambda s: replace(s, detector=None),
            "anchor": lambda s: replace(
                s, word_anchor=(replace(s.word_anchor[0], text=" invented"),)
            ),
            "session_id": lambda s: replace(s, stream_id="different"),
        }
        _, fresh = self.stream(hybrid=True)
        for reason, transform in mutations.items():
            with self.subTest(reason=reason):
                changed, digest = self.tamper(path, transform)
                with self.assertRaises((TypeError, ValueError)):
                    self.restore(fresh, changed, digest)
        self.assertEqual(fresh.calls, [])

    def test_truncated_history_cannot_be_saved_or_restored(self):
        stream, _ = self.stream(history_limit=1)
        first_commit(stream)
        complete(stream)
        self.assertGreater(stream.state.version, 1)
        self.assertEqual(len(stream.state.windows), 1)
        with self.assertRaisesRegex(ValueError, "complete publication history"):
            self.save(stream)
        complete_stream, _ = self.stream()
        first_commit(complete_stream)
        complete(complete_stream)
        path, _ = self.save(complete_stream)
        changed, digest = self.tamper(
            path,
            lambda s: replace(
                s, session=replace(s.session, windows=s.session.windows[1:])
            ),
        )
        _, fresh = self.stream()
        with self.assertRaisesRegex(ValueError, "complete publication history"):
            self.restore(fresh, changed, digest)

    def test_active_preview_and_pending_save_refusal_is_nonmutating(self):
        for stage in ("active", "preview", "pending"):
            with self.subTest(stage=stage):
                stream, adapter = self.stream()
                if stage == "pending":
                    first_commit(stream)
                    saved_state = stream.state
                    stream._pending_state = saved_state
                    self.addCleanup(setattr, stream, "_pending_state", None)
                else:
                    stream.push(0, INITIAL_PCM)
                    stream.step()
                    if stage == "preview":
                        stream.step()
                        stream.step()
                        self.assertIsNotNone(stream._previous)
                before = (
                    stream.state,
                    stream.metrics,
                    bytes(stream._audio),
                    len(adapter.calls),
                )
                path = self.path()
                with self.assertRaisesRegex(NativeStreamError, "publication boundary"):
                    stream.save_checkpoint(path, pipeline_identity=PIPELINE)
                self.assertFalse(path.exists())
                self.assertEqual(
                    (
                        stream.state,
                        stream.metrics,
                        bytes(stream._audio),
                        len(adapter.calls),
                    ),
                    before,
                )

    def test_retained_cleanup_failure_cannot_be_saved_before_recovery(self):
        stream, adapter = self.stream()
        stream.push(0, INITIAL_PCM)
        for _ in range(5):
            stream.step()
        run = adapter.runs[-1]
        with patch.object(
            type(run.transaction._lease), "release", side_effect=RuntimeError("release")
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                stream.step()
        self.assertIsNotNone(raised.exception.committed_state)
        with self.assertRaises(NativeStreamError):
            self.save(stream)
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        events = stream.step()
        self.assertTrue(any(event.kind is StreamEventKind.COMMIT for event in events))
        self.save(stream)
        self.assert_released(adapter)

    def test_pending_or_failed_context_retry_cannot_be_saved(self):
        for status in ("scheduled", "running", "refused", "failed", "unavailable"):
            with self.subTest(status=status):
                stream, adapter = self.stream(hybrid=True, eof_context_retry=True)
                first_commit(stream)
                stream._context_retry = ContinuousContextRetry(
                    stream.last_trace,
                    stream._word_anchor,
                    stream.state.version,
                    stream.retained_from_sample,
                    stream.metrics.accepted_samples,
                    status,
                    "boundary_guard_test",
                )
                before = (stream.state, stream.metrics, bytes(stream._audio))
                path = self.path()
                with self.assertRaisesRegex(NativeStreamError, "publication boundary"):
                    stream.save_checkpoint(path, pipeline_identity=PIPELINE)
                self.assertFalse(path.exists())
                self.assertEqual(
                    (stream.state, stream.metrics, bytes(stream._audio)), before
                )
                self.assert_released(adapter)

    def test_recovered_context_retry_still_requires_resolved_execution_state(self):
        stream, _ = self.stream(hybrid=True, eof_context_retry=True)
        first_commit(stream)
        stream._context_retry = ContinuousContextRetry(
            stream.last_trace,
            stream._word_anchor,
            stream.state.version,
            stream.retained_from_sample,
            stream.metrics.accepted_samples,
            "recovered",
            "boundary_guard_test",
        )
        for pending, value in (
            ("_context_retry_active", True),
            ("_unresolved_eof", True),
            ("_retry_analysis", (0, 1, True)),
            ("_pending_state", stream.state),
        ):
            with self.subTest(pending=pending), patch.object(stream, pending, value):
                path = self.path()
                with self.assertRaisesRegex(NativeStreamError, "publication boundary"):
                    stream.save_checkpoint(path, pipeline_identity=PIPELINE)
                self.assertFalse(path.exists())
        # The diagnostic itself is non-authoritative, like the omitted last trace.
        path, digest = self.save(stream)
        _, fresh = self.stream(hybrid=True)
        restored = self.restore(fresh, path, digest)
        self.assertEqual(restored.state, stream.state)
        self.assertIsNone(restored._context_retry)
        self.assertFalse(restored._context_retry_active)

    def test_actual_context_retry_saves_only_after_commit_and_release(self):
        fixture = retry_fixtures.ContinuousContextRetryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        stream, adapter, _, _ = fixture.pending()
        fixture.decode(stream)
        self.assertEqual(stream.context_retry_observation.status, "scheduled")
        with self.assertRaises(NativeStreamError):
            self.save(stream)
        stream.step()
        self.assertEqual(stream.context_retry_observation.status, "running")
        with self.assertRaises(NativeStreamError):
            self.save(stream)
        stream.step()
        events = stream.step()
        self.assertEqual(stream.context_retry_observation.status, "recovered")
        self.assertTrue(stream.done)
        self.assertIs(events[-1].kind, StreamEventKind.FINAL)
        self.assert_released(adapter)
        path, digest = self.save(stream)
        _, fresh = self.stream(hybrid=True)
        restored = self.restore(fresh, path, digest)
        self.assertEqual(restored.state, stream.state)
        self.assertEqual(restored.metrics, stream.metrics)
        self.assertTrue(restored.done)
        self.assertIsNone(restored.context_retry_observation)
        self.assertEqual(restored.step(), ())
        self.assertEqual(fresh.calls, [])

    def test_unresolved_and_abandoned_streams_cannot_be_saved(self):
        stream, adapter = self.stream(hybrid=True)
        first_commit(stream)
        adapter.result_factory = lambda window_id, start, end: replace_words(
            word_result(window_id, start, end),
            lambda word: replace(word, text=" missing", tokens=(999,)),
        )
        stream.finish_input()
        with self.assertRaises(StreamNeedsResolutionError):
            drain(stream)
        before = (stream.state, stream.metrics, bytes(stream._audio))
        with self.assertRaises(NativeStreamError):
            self.save(stream)
        self.assertEqual((stream.state, stream.metrics, bytes(stream._audio)), before)
        stream.close()
        with self.assertRaises(NativeStreamError):
            self.save(stream)

    def test_save_requires_owner_thread_before_any_filesystem_write(self):
        stream, _ = self.stream()
        first_commit(stream)
        path = self.path()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                stream.save_checkpoint, path, pipeline_identity=PIPELINE
            )
            with self.assertRaises(NativeStreamError):
                future.result()
        self.assertFalse(path.exists())
        self.save(stream)

    def test_existing_checkpoint_is_not_overwritten(self):
        stream, _ = self.stream()
        first_commit(stream)
        path, digest = self.save(stream)
        with self.assertRaises(FileExistsError):
            stream.save_checkpoint(path, pipeline_identity=PIPELINE)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_history_bound_and_digest_inputs_are_strict(self):
        for value in (True, None, 0, 4097, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_stream(history_limit=value)
        stream, adapter = self.stream()
        for identity in ("", "sha256:" + "A" * 64, "a" * 64, "sha256:" + "g" * 64):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                stream.save_checkpoint(self.path(), pipeline_identity=identity)
        path, digest = self.save(stream)
        for digest in ("", "A" * 64, "sha256:" + digest, "g" * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                self.restore(adapter, path, digest)


if __name__ == "__main__":
    unittest.main()
