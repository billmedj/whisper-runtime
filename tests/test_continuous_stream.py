"""Deterministic rolling-stream contracts; no Whisper weights or devices needed."""

import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from unittest.mock import patch

from whisper_runtime import (
    AudioSpan,
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
    WindowTransaction,
    Worker,
)
from whisper_runtime.adapters import (
    AudioBufferFullError,
    AudioSequenceError,
    NativeDecodeMetadata,
    NativeDecodeOptions,
    NativeStreamError,
    NativeTimestampSegment,
    NativeWindowResult,
    StreamEventKind,
    TranscriptEvent,
)
from whisper_runtime.adapters.continuous_stream import (
    CONTINUOUS_PROFILE,
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    StreamNeedsResolutionError,
)
from whisper_runtime.adapters.native_result import select_native_publication


def pcm(samples: int, value: int = 0) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


def pcm_ms(milliseconds: int, value: int = 0) -> bytes:
    return pcm(milliseconds * 16, value)


def timed_result(window_id: str, start: int, end: int) -> NativeWindowResult:
    segments = tuple(
        NativeTimestampSegment(
            AudioSpan(position, position + 100),
            f" word-{position}",
            (position + 1,),
        )
        for position in range(start, end - 99, 100)
    )
    return NativeWindowResult(
        window_id=window_id,
        start_ms=start,
        end_ms=end,
        text="".join(segment.text for segment in segments).strip() or "tail",
        metadata=NativeDecodeMetadata(
            language="en",
            tokens=tuple(token for segment in segments for token in segment.tokens),
            segments=segments,
            timestamps_complete=bool(segments) and segments[-1].span.end_ms == end,
        ),
    )


class SwitchableFence(ImmediateFence):
    def __init__(self) -> None:
        self.fail = False
        self.wait_count = 0

    def wait(self) -> None:
        self.wait_count += 1
        if self.fail:
            raise RuntimeError("injected completion-fence failure")


class ScriptedNativeRun:
    """NativeWindowRun-compatible handle, including its real cleanup contract."""

    def __init__(
        self,
        adapter: "ScriptedNativeAdapter",
        transaction: WindowTransaction,
        result: NativeWindowResult,
        fence: SwitchableFence,
    ) -> None:
        self.adapter = adapter
        self.transaction = transaction
        self.result = result
        self.fence = fence
        self.complete = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed or self.transaction.status in (
            TransactionStatus.COMMITTED,
            TransactionStatus.ABORTED,
            TransactionStatus.EXPIRED,
        )

    @property
    def capacity_released(self) -> bool:
        return self.transaction.capacity_released

    def _close_owner(
        self, error: BaseException | None, state: SessionState | None = None
    ) -> None:
        try:
            self.adapter.worker._finish_execution(
                self.transaction, operation_error=error, committed_state=state
            )
        finally:
            self._closed = True
            self.transaction._owner_departed()

    def step(self) -> bool:
        try:
            self.transaction.checkpoint()
            self.adapter.fault("step")
            if self.adapter.on_step is not None:
                self.transaction.submit(self.adapter.on_step)
            self.complete = True
            self.transaction.checkpoint()
            return True
        except BaseException as error:
            self._close_owner(error)
            raise

    def prepare_result(self) -> NativeWindowResult:
        try:
            self.transaction.checkpoint()
            self.adapter.fault("prepare")
            if not self.complete:
                raise RuntimeError("result requested before token completion")
            return self.result
        except BaseException as error:
            self._close_owner(error)
            raise

    def finish(
        self,
        *,
        committed_through_ms: int | None = None,
        publication_span: AudioSpan | None = None,
    ) -> SessionState:
        state = None
        try:
            self.transaction.checkpoint()
            self.adapter.fault("finish")
            result = self.result
            if publication_span is not None:
                result = select_native_publication(result, publication_span)
            state = self.transaction.commit(
                result, committed_through_ms=committed_through_ms
            )
        except BaseException as error:
            self._close_owner(error, state)
            raise
        self._close_owner(None, state)
        return state

    def cancel(self) -> bool:
        return False if self.closed else self.adapter.worker.cancel(self.transaction)

    def stop(self) -> bool:
        if self.capacity_released:
            return False
        changed = self.adapter.worker.stop(self.transaction)
        if not self.capacity_released:
            changed = self.adapter.worker.recover(self.transaction) or changed
        return changed

    def close(self) -> bool:
        if self.closed:
            return False
        try:
            changed = self.transaction.abort()
        except BaseException as error:
            self._close_owner(error)
            raise
        self._close_owner(None)
        return changed


class ScriptedNativeAdapter:
    def __init__(
        self,
        result_factory: Callable[[str, int, int], NativeWindowResult] = timed_result,
        *,
        fail_once: str | None = None,
        on_step: Callable[[], object] | None = None,
    ) -> None:
        self.model_identity = ModelSnapshot("scripted", "1", "test", "sha256:test")
        self.capacity = ResourceVector(memory_bytes=1, compute_units=1, stream_slots=1)
        self.budget = Budget(self.capacity)
        self.worker = Worker(
            "scripted-worker", self.model_identity, self.budget, queue_capacity=1
        )
        self.result_factory = result_factory
        self.fail_once = fail_once
        self.on_step = on_step
        self.error = RuntimeError("injected native failure")
        self.calls: list[tuple[str, int, int, int]] = []
        self.inputs: list[bytes] = []
        self.runs: list[ScriptedNativeRun] = []
        self.options: list[NativeDecodeOptions | None] = []

    def fault(self, stage: str) -> None:
        if self.fail_once == stage:
            self.fail_once = None
            raise self.error

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
    ) -> ScriptedNativeRun:
        if not isinstance(mel, bytes):
            raise TypeError("scripted mel must retain its source PCM bytes")
        self.calls.append((window_id, start_ms, end_ms, len(mel) // 2))
        self.inputs.append(mel)
        self.options.append(options)
        self.fault("start")
        result = self.result_factory(window_id, start_ms, end_ms)
        transaction = self.worker.prepare(
            session=session,
            request=request,
            window_id=window_id,
            resources=self.capacity,
        )
        fence = SwitchableFence()
        transaction.start(fence)
        run = ScriptedNativeRun(self, transaction, result, fence)
        self.runs.append(run)
        return run


class ContinuousTranscriptStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ContinuousStreamConfig(
            preview_interval_ms=100,
            max_window_ms=500,
            max_buffer_ms=600,
            holdback_ms=100,
            timestamp_tolerance_ms=0,
        )

    def stream(
        self,
        adapter: ScriptedNativeAdapter,
        **kwargs: object,
    ) -> ContinuousTranscriptStream:
        return ContinuousTranscriptStream(
            adapter,
            stream_id="continuous-test",
            mel_builder=kwargs.pop("mel_builder", lambda content: content),
            config=kwargs.pop("config", self.config),
            **kwargs,
        )

    def drain(self, stream: ContinuousTranscriptStream) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        for _ in range(10_000):
            if not stream.ready:
                return events
            events.extend(stream.step())
        self.fail("stream did not reach a scheduling boundary")

    def assert_released(self, adapter: ScriptedNativeAdapter) -> None:
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(adapter.budget.available, adapter.capacity)

    def test_configuration_rejects_invalid_limits(self) -> None:
        for arguments, error in (
            ({"preview_interval_ms": True}, TypeError),
            ({"preview_interval_ms": 0}, ValueError),
            ({"preview_interval_ms": 500, "max_window_ms": 500}, ValueError),
            ({"max_window_ms": 30_001}, ValueError),
            ({"max_buffer_ms": 29_999}, ValueError),
            ({"max_buffer_ms": 120_001}, ValueError),
            ({"holdback_ms": 30_000}, ValueError),
            ({"timestamp_tolerance_ms": -1}, ValueError),
            ({"holdback_ms": 1.5}, TypeError),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(error):
                ContinuousStreamConfig(**arguments)

    def test_timestamps_are_required_and_enabled_by_default(self) -> None:
        adapter = ScriptedNativeAdapter()
        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.stream(adapter, options=NativeDecodeOptions(without_timestamps=True))
        stream = self.stream(adapter)
        self.assertEqual(stream.profile_id, CONTINUOUS_PROFILE)
        stream.push(0, pcm_ms(100))
        self.drain(stream)
        self.assertFalse(adapter.options[0].without_timestamps)
        self.assert_released(adapter)

    def test_invalid_pcm_or_sequence_does_not_mutate_admission(self) -> None:
        stream = self.stream(ScriptedNativeAdapter())
        stream.push(0, pcm(1))
        for sequence, content, error in (
            (0, pcm(1), AudioSequenceError),
            (2, pcm(1), AudioSequenceError),
            (True, pcm(1), TypeError),
            (1, b"", ValueError),
            (1, b"\x00", ValueError),
            (1, bytearray(pcm(1)), TypeError),
        ):
            before = stream.metrics
            with self.subTest(sequence=sequence, content=content):
                with self.assertRaises(error):
                    stream.push(sequence, content)
                self.assertEqual(stream.metrics, before)
        self.assertEqual(stream.push(1, pcm(1)), 2)

    def test_progressive_prefix_commit_precedes_eof(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(250))
        events = self.drain(stream)
        self.assertEqual(
            [event.kind for event in events],
            [
                StreamEventKind.PROVISIONAL,
                StreamEventKind.REPLACE,
                StreamEventKind.COMMIT,
            ],
        )
        self.assertEqual(events[0].text, "word-0")
        self.assertEqual(events[1].text, "word-0")
        self.assertEqual(events[2].committed_through_sample, 100 * 16)
        self.assertEqual(stream.metrics.buffered_samples, 150 * 16)
        self.assertEqual(stream.metrics.decode_count, 2)
        self.assertEqual(stream.metrics.decoded_source_samples, 300 * 16)
        self.assertEqual(
            stream.state.windows[-1].result.analyzed_span, AudioSpan(0, 200)
        )
        self.assertFalse(stream.input_finished)
        self.assertFalse(stream.done)
        self.assert_released(adapter)

    def test_buffer_rejection_preserves_sequence_for_retry_after_commit(self) -> None:
        stream = self.stream(ScriptedNativeAdapter())
        stream.push(0, pcm_ms(600, 1))
        before = stream.metrics
        with self.assertRaises(AudioBufferFullError):
            stream.push(1, pcm_ms(100, 2))
        self.assertEqual(stream.metrics, before)
        self.drain(stream)
        self.assertEqual(stream.expected_chunk, 1)
        self.assertEqual(stream.push(1, pcm_ms(100, 2)), 700 * 16)
        self.assertEqual(stream.expected_chunk, 2)
        self.assertLessEqual(stream.metrics.peak_buffered_samples, 600 * 16)

    def test_partition_equivalence_at_fixed_preview_schedule(self) -> None:
        source = b"".join(pcm(1, index % 30_000) for index in range(550 * 16))

        def run(parts: tuple[int, ...]) -> tuple[object, ...]:
            adapter = ScriptedNativeAdapter()
            stream = self.stream(adapter)
            events = []
            offset = 0
            for sequence, size in enumerate(parts):
                stream.push(sequence, source[offset : offset + size * 32])
                offset += size * 32
                events.extend(self.drain(stream))
            stream.finish_input()
            events.extend(self.drain(stream))
            for call, content in zip(adapter.calls, adapter.inputs):
                self.assertEqual(content, source[call[1] * 32 : call[2] * 32])
            return adapter.calls, adapter.inputs, events, stream.state

        self.assertEqual(run((550,)), run((50,) * 11))
        self.assertEqual(run((550,)), run((17, 83, 61, 89, 143, 157)))

    def test_rolls_past_thirty_seconds_without_revising_committed_ids(self) -> None:
        adapter = ScriptedNativeAdapter()
        config = ContinuousStreamConfig(
            preview_interval_ms=1_000,
            max_window_ms=3_000,
            max_buffer_ms=4_000,
            holdback_ms=1_000,
        )
        stream = self.stream(adapter, config=config)
        events = []
        for sequence in range(35):
            stream.push(sequence, pcm_ms(1_000, sequence))
            events.extend(self.drain(stream))
            self.assertLessEqual(len(stream.state.windows), 4)
            self.assertLessEqual(stream.metrics.buffered_samples, 4_000 * 16)
        self.assertGreater(stream.metrics.committed_samples, 30_000 * 16)
        stream.finish_input()
        events.extend(self.drain(stream))
        committed_ids = set()
        watermark = 0
        for event in events:
            if event.kind is StreamEventKind.FINAL:
                continue
            self.assertNotIn(event.segment_id, committed_ids)
            self.assertGreaterEqual(event.start_sample, watermark)
            if event.kind is StreamEventKind.COMMIT:
                self.assertEqual(event.start_sample, watermark)
                watermark = event.committed_through_sample
                committed_ids.add(event.segment_id)
        self.assertEqual(watermark, 35_000 * 16)
        self.assertGreater(len(committed_ids), 4)
        self.assertEqual(len(stream.state.windows), 4)
        self.assertEqual(stream.state.version, len(committed_ids))
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assertTrue(all(end - start <= 3_000 for _, start, end, _ in adapter.calls))
        self.assertTrue(any(start > 30_000 for _, start, _, _ in adapter.calls))
        self.assertEqual(
            [event.sequence_number for event in events], list(range(1, len(events) + 1))
        )
        self.assert_released(adapter)

    def test_producer_can_push_and_finish_while_owner_is_in_model_step(self) -> None:
        entered, produced = Event(), Event()

        def model_step() -> None:
            entered.set()
            if not produced.wait(2):
                raise AssertionError("producer blocked behind model work")

        adapter = ScriptedNativeAdapter(on_step=model_step)
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(100, 1))
        stream.step()

        def producer() -> None:
            if not entered.wait(2):
                raise AssertionError("owner never entered the model")
            try:
                stream.push(1, pcm_ms(57, 2))
                stream.finish_input()
            finally:
                produced.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(producer)
            stream.step()
            future.result(timeout=3)
        events = self.drain(stream)
        self.assertEqual(adapter.inputs[0], pcm_ms(100, 1))
        self.assertEqual(adapter.inputs[-1], pcm_ms(100, 1) + pcm_ms(57, 2))
        self.assertEqual(events[-2].committed_through_sample, 157 * 16)
        self.assertTrue(stream.done)
        self.assert_released(adapter)

    def test_model_driving_and_close_require_the_creating_thread(self) -> None:
        stream = self.stream(ScriptedNativeAdapter())
        stream.push(0, pcm_ms(100))
        before = stream.metrics
        with ThreadPoolExecutor(max_workers=1) as executor:
            for operation in (stream.step, stream.close):
                with self.assertRaisesRegex(NativeStreamError, "creating thread"):
                    executor.submit(operation).result(timeout=2)
        self.assertEqual(stream.metrics, before)
        self.assertFalse(stream.done)
        self.assertFalse(stream.active)

    def test_empty_eof_emits_final_once_without_decode(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        self.assertTrue(stream.finish_input())
        self.assertFalse(stream.finish_input())
        self.assertEqual(
            [event.kind for event in self.drain(stream)], [StreamEventKind.FINAL]
        )
        self.assertEqual(adapter.calls, [])
        self.assertEqual(stream.step(), ())
        self.assertFalse(stream.close())

    def test_eof_preserves_short_and_non_millisecond_pcm_exactly(self) -> None:
        for samples in (1, 15, 17, 100 * 16 + 7):
            with self.subTest(samples=samples):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(adapter)
                source = pcm(samples, 31)
                stream.push(0, source)
                stream.finish_input()
                events = self.drain(stream)
                self.assertEqual(adapter.inputs, [source])
                self.assertEqual(events[-2].committed_through_sample, samples)
                self.assertEqual(events[-2].committed_through_ms, samples // 16)
                self.assertEqual(stream.metrics.committed_samples, samples)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assertEqual(stream.state.committed_through_ms, samples // 16)
                self.assertEqual(stream.step(), ())
                with self.assertRaises(NativeStreamError):
                    stream.push(1, pcm(1))
                self.assert_released(adapter)

    def test_eof_after_preview_flushes_uncommitted_tail(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        stream.push(0, pcm_ms(200) + pcm(7, 1))
        prior = self.drain(stream)
        stream.finish_input()
        events = prior + self.drain(stream)
        commits = [event for event in events if event.kind is StreamEventKind.COMMIT]
        self.assertEqual(
            [event.committed_through_sample for event in commits], [1_600, 3_207]
        )
        self.assertEqual(adapter.inputs[-1], pcm_ms(100) + pcm(7, 1))
        self.assertEqual(events[-1].kind, StreamEventKind.FINAL)

    def test_cancellation_retries_identical_input_without_counting_failed_decode(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        source = pcm_ms(100, 22)
        stream.push(0, source)
        self.assertFalse(stream.cancel_active())
        stream.step()
        self.assertTrue(stream.cancel_active())
        with self.assertRaises(RuntimeStateError):
            stream.step()
        self.assertEqual(stream.metrics.decode_count, 0)
        self.assertEqual(stream.metrics.events_emitted, 0)
        self.assertEqual(stream.metrics.buffered_samples, 1_600)
        self.assert_released(adapter)
        events = self.drain(stream)
        self.assertEqual(adapter.inputs, [source, source])
        self.assertEqual(adapter.calls[0], adapter.calls[1])
        self.assertEqual(events[0].revision, 1)
        self.assertEqual(stream.state.version, 0)

    def test_errors_at_each_native_boundary_retain_input_for_retry(self) -> None:
        for stage in ("start", "step", "prepare", "finish"):
            with self.subTest(stage=stage):
                adapter = ScriptedNativeAdapter(fail_once=stage)
                stream = self.stream(adapter)
                source = pcm_ms(37, 5)
                stream.push(0, source)
                stream.finish_input()
                with self.assertRaises(RuntimeError) as raised:
                    self.drain(stream)
                self.assertIs(raised.exception, adapter.error)
                self.assertEqual(stream.state.version, 0)
                self.assertEqual(stream.metrics.events_emitted, 0)
                self.assertEqual(stream.metrics.decode_count, 0)
                self.assertEqual(stream.metrics.buffered_samples, 37 * 16)
                self.assert_released(adapter)
                events = self.drain(stream)
                self.assertEqual(adapter.inputs, [source, source])
                self.assertEqual(events[-1].kind, StreamEventKind.FINAL)
                self.assertEqual(stream.state.version, 1)

    def test_mel_builder_failure_retains_schedule_and_exact_audio(self) -> None:
        error = RuntimeError("injected preprocessing failure")
        attempts = []

        def build(content: bytes) -> bytes:
            attempts.append(content)
            if len(attempts) == 1:
                raise error
            return content

        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter, mel_builder=build)
        stream.push(0, pcm_ms(100, 8))
        with self.assertRaises(RuntimeError) as raised:
            stream.step()
        self.assertIs(raised.exception, error)
        self.assertEqual(adapter.calls, [])
        self.assertEqual(stream.metrics.decode_count, 0)
        self.drain(stream)
        self.assertEqual(attempts[0], attempts[1])

    def test_unstable_or_gapped_audio_is_never_silently_evicted(self) -> None:
        def unstable(window_id: str, start: int, end: int) -> NativeWindowResult:
            result = timed_result(window_id, start, end)
            segments = tuple(
                replace(segment, text=f" revision-{end}", tokens=(end,))
                for segment in result.metadata.segments
            )
            return replace(
                result,
                text="".join(segment.text for segment in segments).strip(),
                metadata=replace(result.metadata, segments=segments),
            )

        def gap(window_id: str, start: int, end: int) -> NativeWindowResult:
            result = timed_result(window_id, start, end)
            segments = result.metadata.segments[1:]
            return replace(
                result,
                text="".join(segment.text for segment in segments).strip(),
                metadata=replace(
                    result.metadata,
                    segments=segments,
                    timestamps_complete=bool(segments),
                ),
            )

        for factory in (unstable, gap):
            with self.subTest(factory=factory.__name__):
                adapter = ScriptedNativeAdapter(factory)
                stream = self.stream(adapter)
                source = pcm_ms(600, 7)
                stream.push(0, source)
                with self.assertRaises(StreamNeedsResolutionError):
                    self.drain(stream)
                before = stream.metrics
                for _ in range(2):
                    with self.assertRaises(StreamNeedsResolutionError):
                        stream.step()
                    self.assertEqual(stream.metrics, before)
                self.assertEqual(before.buffered_samples, 600 * 16)
                self.assertEqual(before.committed_samples, 0)
                self.assertEqual(stream.state.version, 0)
                self.assertEqual(adapter.inputs[-1], source[: 500 * 32])
                self.assert_released(adapter)
                stream.close()

    def test_exhausted_window_reports_resolution_even_at_exact_buffer_limit(
        self,
    ) -> None:
        def untimed(window_id: str, start: int, end: int) -> NativeWindowResult:
            return NativeWindowResult(
                window_id=window_id, text="unresolved", start_ms=start, end_ms=end
            )

        for interval in (100, 200):
            with self.subTest(preview_interval_ms=interval):
                adapter = ScriptedNativeAdapter(untimed)
                stream = self.stream(
                    adapter,
                    config=ContinuousStreamConfig(
                        preview_interval_ms=interval,
                        max_window_ms=500,
                        max_buffer_ms=500,
                        holdback_ms=100,
                    ),
                )
                stream.push(0, pcm_ms(500))
                with self.assertRaises(StreamNeedsResolutionError):
                    self.drain(stream)
                self.assertEqual(adapter.calls[-1][2], 500)
                self.assertEqual(stream.metrics.buffered_samples, 8_000)
                self.assertEqual(stream.metrics.committed_samples, 0)
                self.assertEqual(stream.expected_chunk, 1)
                with self.assertRaises(AudioBufferFullError):
                    stream.push(1, pcm(1))
                self.assert_released(adapter)

    def test_wrong_native_source_closes_run_without_advancing_input(self) -> None:
        for wrong_field in ("span", "window_id"):
            with self.subTest(wrong_field=wrong_field):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(adapter)
                stream.push(0, pcm_ms(100))
                stream.step()
                run = adapter.runs[0]
                if wrong_field == "span":
                    run.result = replace(run.result, end_ms=101)
                else:
                    run.result = replace(run.result, window_id="another-window")
                with self.assertRaisesRegex(NativeStreamError, "admitted audio"):
                    self.drain(stream)
                self.assertEqual(stream.metrics.decode_count, 0)
                self.assertEqual(stream.metrics.buffered_samples, 100 * 16)
                self.assert_released(adapter)

    def test_small_commit_at_full_window_rebases_to_two_growing_observations(
        self,
    ) -> None:
        for prefix_ms in (50, 100):
            with self.subTest(prefix_ms=prefix_ms):

                def delayed_prefix(
                    window_id: str, start: int, end: int
                ) -> NativeWindowResult:
                    if start:
                        return timed_result(window_id, start, end)
                    # Only the 400/500 ms observations agree, and only on a
                    # prefix no longer than one preview interval.
                    token = end if end < 400 else 10_000
                    segments = (
                        NativeTimestampSegment(
                            AudioSpan(0, prefix_ms), f" prefix-{token}", (token,)
                        ),
                    )
                    if end > prefix_ms:
                        segments += (
                            NativeTimestampSegment(
                                AudioSpan(prefix_ms, end),
                                f" pending-{end}",
                                (20_000 + end,),
                            ),
                        )
                    return NativeWindowResult(
                        window_id=window_id,
                        start_ms=start,
                        end_ms=end,
                        text="".join(segment.text for segment in segments).strip(),
                        metadata=NativeDecodeMetadata(
                            language="en",
                            tokens=tuple(
                                token
                                for segment in segments
                                for token in segment.tokens
                            ),
                            segments=segments,
                            timestamps_complete=True,
                        ),
                    )

                adapter = ScriptedNativeAdapter(delayed_prefix)
                stream = self.stream(adapter)
                source = pcm_ms(600, 19)
                stream.push(0, source)
                events = self.drain(stream)
                self.assertEqual(
                    [(start, end) for _, start, end, _ in adapter.calls],
                    [(0, end) for end in (100, 200, 300, 400, 500)]
                    + [(prefix_ms, prefix_ms + 400), (prefix_ms, prefix_ms + 500)],
                )
                commits = [
                    event for event in events if event.kind is StreamEventKind.COMMIT
                ]
                self.assertEqual(
                    [event.committed_through_sample for event in commits],
                    [prefix_ms * 16, (prefix_ms + 400) * 16],
                )
                for call, content in zip(adapter.calls, adapter.inputs):
                    self.assertEqual(content, source[call[1] * 32 : call[2] * 32])
                self.assertEqual(stream.state.version, 2)
                self.assertFalse(stream.input_finished)
                self.assertFalse(stream.ready)
                self.assert_released(adapter)

    def test_fence_failure_blocks_publication_and_retries_after_exact_recovery(
        self,
    ) -> None:
        for final in (False, True):
            with self.subTest(final=final):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(adapter)
                source = pcm_ms(100, 42)
                stream.push(0, source)
                if final:
                    stream.finish_input()
                stream.step()
                stream.step()
                run = adapter.runs[0]
                run.fence.fail = True
                with self.assertRaises(TransactionRetainedError) as raised:
                    stream.step()
                self.assertIs(raised.exception.transaction, run.transaction)
                self.assertIsNone(raised.exception.committed_state)
                self.assertTrue(run.closed)
                self.assertFalse(run.capacity_released)
                self.assertEqual(stream.state.version, 0)
                self.assertEqual(stream.metrics.decode_count, 0)
                self.assertEqual(stream.metrics.events_emitted, 0)
                with self.assertRaisesRegex(NativeStreamError, "retained"):
                    stream.step()
                with self.assertRaisesRegex(NativeStreamError, "retained"):
                    stream.close()
                self.assertFalse(stream.done)
                run.fence.fail = False
                self.assertTrue(adapter.worker.recover(raised.exception.transaction))
                self.assertEqual(stream.step(), ())
                events = self.drain(stream)
                self.assertEqual(adapter.inputs, [source, source])
                self.assertEqual(events[0].revision, 1)
                self.assertEqual(stream.metrics.decode_count, 1)
                self.assert_released(adapter)

    def test_committed_state_is_reemitted_once_after_retained_lease_recovery(
        self,
    ) -> None:
        for final in (False, True):
            with self.subTest(final=final):
                adapter = ScriptedNativeAdapter()
                stream = self.stream(adapter)
                stream.push(0, pcm_ms(200))
                if final:
                    stream.finish_input()
                else:
                    for _ in range(3):
                        stream.step()
                stream.step()
                stream.step()
                run = adapter.runs[-1]
                before = stream.metrics
                with patch.object(
                    type(run.transaction._lease),
                    "release",
                    side_effect=RuntimeError("lease release failed"),
                ):
                    with self.assertRaises(TransactionRetainedError) as raised:
                        stream.step()
                    self.assertIsNotNone(raised.exception.committed_state)
                    self.assertEqual(stream.state.version, 1)
                    self.assertEqual(stream.metrics, before)
                    self.assertFalse(run.capacity_released)
                    with self.assertRaises(NativeStreamError):
                        stream.step()
                self.assertTrue(adapter.worker.recover(raised.exception.transaction))
                calls = len(adapter.calls)
                events = stream.step()
                self.assertEqual(
                    [event.kind for event in events[:2]],
                    [
                        StreamEventKind.REPLACE
                        if not final
                        else StreamEventKind.PROVISIONAL,
                        StreamEventKind.COMMIT,
                    ],
                )
                self.assertEqual(len(adapter.calls), calls)
                self.assertEqual(stream.state.version, 1)
                self.assertEqual(stream.metrics.decode_count, before.decode_count + 1)
                self.assertEqual(stream.step(), ())
                self.assertEqual(
                    stream.metrics.events_emitted, before.events_emitted + len(events)
                )
                self.assert_released(adapter)

    def test_stop_recovers_a_closed_retained_run_and_keeps_audio_retryable(
        self,
    ) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        self.assertFalse(stream.stop_active())
        stream.push(0, pcm_ms(100))
        stream.step()
        run = adapter.runs[0]
        run.fence.fail = True
        stream.cancel_active()
        with self.assertRaises(TransactionRetainedError):
            stream.step()
        run.fence.fail = False
        self.assertTrue(stream.stop_active())
        self.assertFalse(stream.stop_active())
        self.assertEqual(stream.step(), ())
        self.assertEqual(stream.metrics.buffered_samples, 1_600)
        self.drain(stream)
        self.assertEqual(len(adapter.calls), 2)
        self.assert_released(adapter)

    def test_context_preserves_retained_error_and_recovery_authority(self) -> None:
        adapter = ScriptedNativeAdapter()
        stream = self.stream(adapter)
        with self.assertRaises(TransactionRetainedError) as raised:
            with stream:
                stream.push(0, pcm_ms(100))
                stream.step()
                run = adapter.runs[0]
                run.fence.fail = True
                stream.cancel_active()
                stream.step()
        self.assertIs(raised.exception.transaction, run.transaction)
        self.assertTrue(stream.active)
        self.assertFalse(stream.done)
        run.fence.fail = False
        self.assertTrue(adapter.worker.recover(raised.exception.transaction))
        self.assertTrue(stream.close())
        self.assertTrue(stream.done)
        self.assert_released(adapter)


if __name__ == "__main__":
    unittest.main()
