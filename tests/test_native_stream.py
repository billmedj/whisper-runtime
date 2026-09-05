import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from whisper_runtime import (
    Budget,
    ImmediateFence,
    ModelSnapshot,
    RequestState,
    ResourceVector,
    RuntimeStateError,
    Session,
    SessionState,
    TransactionRetainedError,
    TransactionStatus,
    WindowResult,
    WindowTransaction,
    Worker,
)
from whisper_runtime.adapters import (
    BOUNDED_PREFIX_PROFILE,
    AudioBufferFullError,
    AudioSequenceError,
    NativeDecodeOptions,
    NativeStreamConfig,
    NativeStreamError,
    NativeTranscriptStream,
    StreamEventKind,
    StreamMetrics,
    TranscriptEvent,
)

SAMPLES_PER_MS = 16


def pcm(samples: int, value: int = 0) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * samples


def pcm_ms(milliseconds: int, value: int = 0) -> bytes:
    return pcm(milliseconds * SAMPLES_PER_MS, value)


class ScriptedRun:
    def __init__(
        self,
        *,
        worker: Worker,
        transaction: WindowTransaction,
        result: WindowResult,
    ) -> None:
        self._worker = worker
        self._transaction = transaction
        self._result = result
        self._complete = False

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def closed(self) -> bool:
        return self._transaction.status in (
            TransactionStatus.COMMITTED,
            TransactionStatus.ABORTED,
            TransactionStatus.EXPIRED,
        )

    @property
    def capacity_released(self) -> bool:
        return self._transaction.capacity_released

    def step(self) -> bool:
        self._transaction.checkpoint()
        self._complete = True
        self._transaction.checkpoint()
        return True

    def finish(self, *, committed_through_ms: int | None = None) -> SessionState:
        return self._transaction.commit(
            self._result,
            committed_through_ms=committed_through_ms,
        )

    def cancel(self) -> bool:
        return self._worker.cancel(self._transaction)

    def stop(self) -> bool:
        return self._worker.stop(self._transaction)

    def close(self) -> bool:
        if self.closed:
            return False
        return self._transaction.abort()


class ScriptedAdapter:
    def __init__(
        self,
        text_for_end_ms: Callable[[int], str],
        *,
        fail_once_at_ms: int | None = None,
    ) -> None:
        self.model_identity = ModelSnapshot(
            model_id="scripted",
            revision="1",
            backend="test",
            fingerprint="sha256:scripted",
        )
        self.capacity = ResourceVector(
            memory_bytes=1,
            compute_units=1,
            stream_slots=1,
        )
        self.budget = Budget(self.capacity)
        self.worker = Worker(
            "scripted-worker",
            self.model_identity,
            self.budget,
            queue_capacity=1,
        )
        self._text_for_end_ms = text_for_end_ms
        self._fail_once_at_ms = fail_once_at_ms
        self.calls: list[tuple[str, int, int]] = []
        self.inputs: list[bytes] = []
        self.runs: list[ScriptedRun] = []

    def start_window(
        self,
        *,
        session: Session,
        request: RequestState,
        window_id: str,
        mel: object,
        start_ms: int,
        end_ms: int,
        options: NativeDecodeOptions | None = None,
    ) -> ScriptedRun:
        del options
        if not isinstance(mel, bytes):
            raise TypeError("test mel must be bytes")
        self.calls.append((window_id, end_ms, len(mel)))
        self.inputs.append(mel)
        if self._fail_once_at_ms == end_ms:
            self._fail_once_at_ms = None
            raise RuntimeError("injected decode failure")

        transaction = self.worker.prepare(
            session=session,
            request=request,
            window_id=window_id,
            resources=self.capacity,
        )
        transaction.start(ImmediateFence())
        run = ScriptedRun(
            worker=self.worker,
            transaction=transaction,
            result=WindowResult(
                window_id=window_id,
                text=self._text_for_end_ms(end_ms),
                start_ms=start_ms,
                end_ms=end_ms,
            ),
        )
        self.runs.append(run)
        return run


class NativeTranscriptStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NativeStreamConfig(
            preview_interval_ms=100,
            max_audio_ms=500,
        )

    def stream(
        self,
        adapter: ScriptedAdapter,
        *,
        mel_builder: Callable[[bytes], object] | None = None,
    ) -> NativeTranscriptStream:
        return NativeTranscriptStream(
            adapter,
            stream_id="test-stream",
            mel_builder=mel_builder or (lambda value: value),
            config=self.config,
            rng_seed=17,
        )

    @staticmethod
    def drain(stream: NativeTranscriptStream) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        while stream.ready:
            events.extend(stream.step())
        return events

    def test_config_exposes_one_fixed_bounded_profile(self) -> None:
        stream = self.stream(ScriptedAdapter(lambda _: "text"))

        self.assertEqual(stream.profile_id, BOUNDED_PREFIX_PROFILE)
        self.assertEqual(stream.config.sample_rate_hz, 16_000)
        for arguments, exception in (
            ({"sample_rate_hz": True}, TypeError),
            ({"sample_rate_hz": 8_000}, ValueError),
            ({"preview_interval_ms": 0}, ValueError),
            ({"max_audio_ms": 30_001}, ValueError),
            (
                {"preview_interval_ms": 2_000, "max_audio_ms": 1_000},
                ValueError,
            ),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(exception):
                    NativeStreamConfig(**arguments)

    def test_push_rejects_bad_sequence_and_pcm_before_mutation(self) -> None:
        stream = self.stream(ScriptedAdapter(lambda _: "text"))
        self.assertEqual(stream.push(0, pcm_ms(25)), 25 * SAMPLES_PER_MS)

        for sequence, value, exception in (
            (0, pcm_ms(10), AudioSequenceError),
            (2, pcm_ms(10), AudioSequenceError),
            (True, pcm_ms(10), TypeError),
            (1, b"", ValueError),
            (1, b"\x00", ValueError),
            (1, bytearray(pcm_ms(10)), TypeError),
        ):
            with self.subTest(sequence=sequence, value_type=type(value)):
                with self.assertRaises(exception):
                    stream.push(sequence, value)  # type: ignore[arg-type]
                self.assertEqual(stream.expected_chunk, 1)
                self.assertEqual(stream.accepted_samples, 25 * SAMPLES_PER_MS)

        expected = 35 * SAMPLES_PER_MS
        self.assertEqual(stream.push(1, pcm_ms(10)), expected)
        self.assertEqual(stream.expected_chunk, 2)
        self.assertEqual(stream.accepted_samples, expected)

    def test_buffer_limit_rejects_the_whole_chunk(self) -> None:
        stream = self.stream(ScriptedAdapter(lambda _: "text"))
        stream.push(0, pcm_ms(450))

        with self.assertRaisesRegex(AudioBufferFullError, "500 ms"):
            stream.push(1, pcm_ms(51))

        self.assertEqual(stream.expected_chunk, 1)
        self.assertEqual(stream.accepted_samples, 450 * SAMPLES_PER_MS)
        self.assertEqual(stream.metrics.accepted_chunks, 1)

    def test_step_exposes_start_token_step_and_publication_boundaries(self) -> None:
        adapter = ScriptedAdapter(lambda end: f"text-{end}")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(250))

        self.assertEqual(stream.step(), ())
        self.assertTrue(stream.active)
        self.assertEqual([call[1] for call in adapter.calls], [100])
        self.assertFalse(adapter.runs[0].complete)

        self.assertEqual(stream.step(), ())
        self.assertTrue(adapter.runs[0].complete)
        first = stream.step()
        self.assertFalse(stream.active)
        self.assertEqual(first[0].kind, StreamEventKind.PROVISIONAL)

        self.assertEqual(stream.step(), ())
        self.assertEqual([call[1] for call in adapter.calls], [100, 200])
        self.assertEqual(stream.step(), ())
        second = stream.step()
        self.assertEqual(second[0].kind, StreamEventKind.REPLACE)
        self.assertFalse(stream.ready)

    def test_finish_skips_obsolete_preview_endpoints(self) -> None:
        adapter = ScriptedAdapter(lambda end: f"text-{end}")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(250))
        stream.finish_input()

        events = self.drain(stream)

        self.assertEqual([call[1] for call in adapter.calls], [250])
        self.assertEqual(stream.metrics.decode_count, 1)
        self.assertEqual(
            [event.kind for event in events],
            [
                StreamEventKind.PROVISIONAL,
                StreamEventKind.COMMIT,
                StreamEventKind.FINAL,
            ],
        )

    def test_chunk_partitions_produce_the_same_scheduled_results(self) -> None:
        source = b"".join(pcm(1, index % 127) for index in range(250 * 16))

        def run(parts_ms: tuple[int, ...]) -> tuple[list[int], list[TranscriptEvent]]:
            adapter = ScriptedAdapter(lambda end: f"text-{end}")
            stream = self.stream(adapter)
            events: list[TranscriptEvent] = []
            offset = 0
            for sequence, part_ms in enumerate(parts_ms):
                part_bytes = part_ms * SAMPLES_PER_MS * 2
                stream.push(sequence, source[offset : offset + part_bytes])
                offset += part_bytes
                events.extend(self.drain(stream))
            stream.finish_input()
            events.extend(self.drain(stream))
            return [call[1] for call in adapter.calls], events

        one_chunk = run((250,))
        regular = run((50, 50, 50, 50, 50))
        irregular = run((17, 83, 61, 89))

        self.assertEqual(one_chunk, regular)
        self.assertEqual(one_chunk, irregular)

    def test_each_prefix_contains_the_exact_admitted_pcm(self) -> None:
        source = b"".join(pcm(1, index % 30000) for index in range(4000))
        inputs: list[bytes] = []

        def build(content: bytes) -> bytes:
            inputs.append(content)
            return content

        stream = self.stream(ScriptedAdapter(lambda _: "text"), mel_builder=build)
        stream.push(0, source[:500])
        stream.push(1, source[500:])
        self.drain(stream)
        stream.finish_input()
        self.drain(stream)

        self.assertEqual(inputs, [source[:3200], source[:6400], source])

    def test_span_change_creates_a_new_revision_when_text_is_unchanged(self) -> None:
        adapter = ScriptedAdapter(lambda end: "hello" if end <= 200 else "hello world")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(250))

        first = self.drain(stream)
        stream.finish_input()
        final = self.drain(stream)

        self.assertEqual(
            [event.kind for event in first],
            [StreamEventKind.PROVISIONAL, StreamEventKind.REPLACE],
        )
        self.assertEqual(first[1].text, "hello")
        self.assertEqual(first[1].end_sample, 200 * SAMPLES_PER_MS)
        self.assertEqual(first[1].supersedes_revision, 1)
        self.assertEqual(
            [event.kind for event in final],
            [StreamEventKind.REPLACE, StreamEventKind.COMMIT, StreamEventKind.FINAL],
        )
        self.assertEqual(
            [event.sequence_number for event in first + final], [1, 2, 3, 4, 5]
        )
        self.assertEqual(final[1].revision, 3)
        self.assertIsNone(final[1].text)

    def test_empty_hypothesis_has_a_revision_before_commit(self) -> None:
        stream = self.stream(ScriptedAdapter(lambda _: ""))
        stream.push(0, pcm_ms(50))
        stream.finish_input()

        events = self.drain(stream)

        self.assertEqual(
            [event.kind for event in events],
            [
                StreamEventKind.PROVISIONAL,
                StreamEventKind.COMMIT,
                StreamEventKind.FINAL,
            ],
        )
        self.assertEqual(events[0].text, "")
        self.assertEqual(events[0].revision, 1)
        self.assertEqual(events[1].revision, 1)

    def test_finish_at_a_published_threshold_commits_that_revision(self) -> None:
        adapter = ScriptedAdapter(lambda _: "stable")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))
        self.assertEqual(len(self.drain(stream)), 1)

        self.assertTrue(stream.finish_input())
        events = self.drain(stream)

        self.assertEqual([call[1] for call in adapter.calls], [100, 100])
        self.assertEqual(
            [event.kind for event in events],
            [StreamEventKind.COMMIT, StreamEventKind.FINAL],
        )
        self.assertEqual(events[0].revision, 1)
        self.assertEqual(stream.state.committed_through_ms, 100)

    def test_decode_failure_does_not_advance_the_schedule_or_metrics(self) -> None:
        adapter = ScriptedAdapter(lambda _: "retry", fail_once_at_ms=100)
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))

        with self.assertRaisesRegex(RuntimeError, "injected decode failure"):
            stream.step()

        self.assertTrue(stream.ready)
        self.assertFalse(stream.active)
        self.assertEqual(stream.state.version, 0)
        self.assertEqual(stream.metrics.decode_count, 0)
        events = self.drain(stream)
        self.assertEqual(events[0].text, "retry")
        self.assertEqual([call[1] for call in adapter.calls], [100, 100])

    def test_mel_failure_is_not_masked_or_counted(self) -> None:
        error = RuntimeError("preprocessing failed")

        def fail(_: bytes) -> object:
            raise error

        stream = self.stream(ScriptedAdapter(lambda _: "unused"), mel_builder=fail)
        stream.push(0, pcm_ms(100))

        with self.assertRaises(RuntimeError) as raised:
            stream.step()

        self.assertIs(raised.exception, error)
        self.assertTrue(stream.ready)
        self.assertEqual(stream.metrics.decode_count, 0)

    def test_active_run_can_pause_and_resume_without_restarting(self) -> None:
        adapter = ScriptedAdapter(lambda _: "paused")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))

        stream.step()
        self.assertTrue(stream.active)
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(stream.state.version, 0)

        self.assertEqual(len(adapter.calls), 1)
        stream.step()
        events = stream.step()
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(events[0].text, "paused")

    def test_active_run_can_be_cancelled_and_retried(self) -> None:
        adapter = ScriptedAdapter(lambda _: "retry")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100))
        stream.step()

        self.assertTrue(stream.cancel_active())
        with self.assertRaises(RuntimeStateError):
            stream.step()

        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)
        self.assertTrue(stream.ready)
        events = self.drain(stream)
        self.assertEqual(events[-1].text, "retry")
        self.assertEqual(len(adapter.calls), 2)

    def test_context_exit_releases_an_active_run(self) -> None:
        adapter = ScriptedAdapter(lambda _: "abandoned")

        with self.stream(adapter) as stream:
            stream.push(0, pcm_ms(100))
            stream.step()
            self.assertTrue(stream.active)
            self.assertEqual(adapter.worker.queue_depth, 1)

        self.assertTrue(stream.done)
        self.assertFalse(stream.active)
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_committed_result_is_delivered_once_after_retention_error(self) -> None:
        adapter = ScriptedAdapter(lambda _: "already committed")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(50))
        stream.finish_input()
        stream.step()
        stream.step()
        run = adapter.runs[0]
        finish = run.finish

        def retain(*, committed_through_ms: int | None = None) -> SessionState:
            state = finish(committed_through_ms=committed_through_ms)
            raise TransactionRetainedError(
                run._transaction,
                operation_error=None,
                retention_error=RuntimeError("simulated post-commit reporting failure"),
                committed_state=state,
            )

        with patch.object(run, "finish", side_effect=retain):
            with self.assertRaises(TransactionRetainedError):
                stream.step()

        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.metrics.events_emitted, 0)
        events = self.drain(stream)
        self.assertEqual(
            [event.kind for event in events],
            [
                StreamEventKind.PROVISIONAL,
                StreamEventKind.COMMIT,
                StreamEventKind.FINAL,
            ],
        )
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.step(), ())

    def test_owner_operations_reject_another_thread_before_mutation(self) -> None:
        stream = self.stream(ScriptedAdapter(lambda _: "text"))
        operations: tuple[Callable[[], object], ...] = (
            lambda: stream.push(0, pcm_ms(1)),
            stream.step,
            stream.finish_input,
            stream.close,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            for operation in operations:
                with self.assertRaisesRegex(NativeStreamError, "creating thread"):
                    executor.submit(operation).result()
        self.assertEqual(stream.accepted_samples, 0)
        self.assertFalse(stream.input_finished)
        self.assertFalse(stream.done)

    def test_final_publication_follows_the_runtime_commit(self) -> None:
        adapter = ScriptedAdapter(lambda _: "final text")
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(250))
        stream.finish_input()
        events = self.drain(stream)

        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.state.committed_through_ms, 250)
        self.assertEqual(stream.state.windows[-1].result.text, "final text")
        commit = [event for event in events if event.kind is StreamEventKind.COMMIT]
        self.assertEqual(len(commit), 1)
        self.assertEqual(commit[0].session_version, stream.state.version)
        self.assertEqual(commit[0].committed_through_sample, 250 * SAMPLES_PER_MS)
        self.assertEqual(commit[0].committed_through_ms, 250)
        self.assertIsNone(commit[0].text)
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_metrics_measure_source_prefix_reprocessing(self) -> None:
        stream = self.stream(ScriptedAdapter(lambda _: "same"))
        stream.push(0, pcm_ms(250))
        preview_events = self.drain(stream)
        stream.finish_input()
        final_events = self.drain(stream)
        metrics = stream.metrics

        self.assertEqual(metrics.accepted_chunks, 1)
        self.assertEqual(metrics.accepted_samples, 250 * SAMPLES_PER_MS)
        self.assertEqual(metrics.accepted_audio_ms, 250.0)
        self.assertEqual(metrics.decode_count, 3)
        self.assertEqual(metrics.decoded_source_samples, 550 * SAMPLES_PER_MS)
        self.assertEqual(metrics.decoded_source_audio_ms, 550.0)
        self.assertEqual(metrics.source_reprocessing_factor, 2.2)
        self.assertEqual(metrics.events_emitted, 5)
        self.assertEqual(len(preview_events + final_events), 5)

    def test_empty_finished_stream_emits_final_session_zero(self) -> None:
        adapter = ScriptedAdapter(lambda _: "unused")
        stream = self.stream(adapter)

        self.assertTrue(stream.finish_input())
        events = stream.step()

        self.assertEqual([event.kind for event in events], [StreamEventKind.FINAL])
        self.assertEqual(events[0].session_version, 0)
        self.assertTrue(stream.done)
        self.assertEqual(adapter.calls, [])
        self.assertEqual(stream.metrics.source_reprocessing_factor, 0.0)

    def test_finished_stream_is_idempotent_and_rejects_more_audio(self) -> None:
        stream = self.stream(ScriptedAdapter(lambda _: "final"))
        stream.push(0, pcm_ms(50))
        self.assertTrue(stream.finish_input())
        self.assertFalse(stream.finish_input())
        self.drain(stream)

        self.assertFalse(stream.ready)
        self.assertEqual(stream.step(), ())
        self.assertFalse(stream.finish_input())
        with self.assertRaisesRegex(NativeStreamError, "already finished"):
            stream.push(1, pcm_ms(10))

    def test_public_event_and_metric_invariants_are_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "millisecond watermark"):
            TranscriptEvent(
                sequence_number=1,
                kind=StreamEventKind.COMMIT,
                segment_id="segment",
                revision=1,
                start_sample=0,
                end_sample=15,
                sample_rate_hz=16_000,
                committed_through_sample=15,
                committed_through_ms=1,
                session_version=1,
            )
        with self.assertRaisesRegex(ValueError, "final event cannot contain text"):
            TranscriptEvent(
                sequence_number=1,
                kind=StreamEventKind.FINAL,
                text="duplicate payload",
                session_version=1,
            )
        with self.assertRaisesRegex(ValueError, "sample_rate_hz must be positive"):
            StreamMetrics(
                sample_rate_hz=0,
                accepted_chunks=0,
                accepted_samples=0,
                decode_count=0,
                decoded_source_samples=0,
                events_emitted=0,
            )

    def test_event_time_properties_preserve_sub_millisecond_source_span(self) -> None:
        event = TranscriptEvent(
            sequence_number=1,
            kind=StreamEventKind.PROVISIONAL,
            segment_id="segment",
            revision=1,
            start_sample=1,
            end_sample=3,
            sample_rate_hz=16_000,
            text="a",
        )

        self.assertEqual(event.start_ms, 0.0625)
        self.assertEqual(event.end_ms, 0.1875)

    def test_final_watermark_never_exceeds_a_fractional_millisecond(self) -> None:
        stream = self.stream(ScriptedAdapter(lambda _: "short"))
        stream.push(0, pcm(15))
        stream.finish_input()
        events = self.drain(stream)
        commit = next(event for event in events if event.kind is StreamEventKind.COMMIT)

        self.assertEqual(commit.end_sample, 15)
        self.assertEqual(commit.committed_through_sample, 0)
        self.assertEqual(commit.committed_through_ms, 0)
        self.assertEqual(stream.state.committed_through_ms, 0)

    def test_event_reducer_freezes_the_existing_revision_without_duplicate_text(
        self,
    ) -> None:
        stream = self.stream(
            ScriptedAdapter(lambda end: "ask now" if end == 100 else "ask not")
        )
        stream.push(0, pcm_ms(250))
        events = self.drain(stream)
        stream.finish_input()
        events.extend(self.drain(stream))
        revision = 0
        text = ""
        endpoint = 0
        committed = False

        for sequence, event in enumerate(events, start=1):
            self.assertEqual(event.sequence_number, sequence)
            if event.kind in (StreamEventKind.PROVISIONAL, StreamEventKind.REPLACE):
                self.assertFalse(committed)
                self.assertEqual(event.revision, revision + 1)
                self.assertEqual(event.supersedes_revision, revision or None)
                assert event.revision is not None and event.text is not None
                assert event.end_sample is not None
                revision, text, endpoint = event.revision, event.text, event.end_sample
            elif event.kind is StreamEventKind.COMMIT:
                self.assertEqual(
                    (event.revision, event.end_sample), (revision, endpoint)
                )
                self.assertIsNone(event.text)
                committed = True
            else:
                self.assertTrue(committed)
                self.assertEqual(event.session_version, stream.state.version)
                self.assertIsNone(event.text)
        self.assertEqual(text, stream.state.windows[-1].result.text)


if __name__ == "__main__":
    unittest.main()
