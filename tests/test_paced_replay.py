"""A paced producer cannot borrow the decoder's time or discard accepted audio."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock, PropertyMock, patch

from test_continuous_evidence import EvidenceNativeAdapter, speech_result
from test_continuous_stream import pcm, pcm_ms

from whisper_runtime import TransactionRetainedError
from whisper_runtime.adapters import (
    AudioBufferFullError,
    NativeStreamError,
    StreamEventKind,
)
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
)
from whisper_runtime.adapters.paced_replay import PacedReplayConfig, drive_paced


class PacedReplayTests(unittest.TestCase):
    def setUp(self):
        self.config = PacedReplayConfig(chunk_ms=5)
        self.stream_config = ContinuousStreamConfig(
            preview_interval_ms=5,
            max_window_ms=100,
            max_buffer_ms=200,
            holdback_ms=0,
            timestamp_tolerance_ms=0,
            input_evidence=True,
            source_units=True,
        )

    def stream(self, adapter=None, **kwargs):
        adapter = adapter or EvidenceNativeAdapter(speech_result)
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="paced-test",
            config=kwargs.pop("config", self.stream_config),
            mel_builder=kwargs.pop("mel_builder", lambda content: content),
            **kwargs,
        )
        self.addCleanup(stream.close)
        return stream, adapter

    def test_configuration_has_strict_bounded_integers(self):
        limits = {
            "chunk_ms": (1, 1000),
            "max_input_ms": (1, 3_600_000),
            "max_source_lag_ms": (0, 60_000),
            "max_drain_ms": (1, 120_000),
            "idle_wait_ms": (1, 1000),
            "max_steps": (1, 10_000_000),
        }
        for field, (minimum, maximum) in limits.items():
            for value in (True, False, None, "1", 1.0):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(TypeError),
                ):
                    replace(self.config, **{field: value})
            for value in (minimum - 1, maximum + 1):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    replace(self.config, **{field: value})
            for value in (minimum, maximum):
                self.assertEqual(
                    getattr(replace(self.config, **{field: value}), field), value
                )
        with self.assertRaises(FrozenInstanceError):
            self.config.chunk_ms = 10

    def test_invalid_inputs_fail_before_admission_or_model_work(self):
        stream, adapter = self.stream()
        for source, error in (
            (None, TypeError),
            (bytearray(pcm(1)), TypeError),
            (memoryview(pcm(1)), TypeError),
            ("audio", TypeError),
            (b"", ValueError),
            (b"a", ValueError),
            (pcm(17), ValueError),
        ):
            with self.subTest(source=source), self.assertRaises(error):
                drive_paced(stream, source, config=replace(self.config, max_input_ms=1))
        for kwargs in (
            {"config": object()},
            {"on_event": 3},
            {"on_trace": 3},
            {"cancel": object()},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                drive_paced(stream, pcm(1), **kwargs)
        with self.assertRaises(TypeError):
            drive_paced(object(), pcm(1))
        self.assertEqual(stream.accepted_samples, 0)
        self.assertFalse(stream.input_finished)
        self.assertEqual(adapter.calls, [])

    def test_existing_input_closed_stream_and_wrong_owner_are_rejected(self):
        for state in ("input", "eof", "closed"):
            with self.subTest(state=state):
                stream, _ = self.stream()
                if state == "input":
                    stream.push(0, pcm(1))
                elif state == "eof":
                    stream.finish_input()
                else:
                    stream.close()
                before = stream.metrics
                with self.assertRaises((ValueError, NativeStreamError)):
                    drive_paced(stream, pcm(1))
                self.assertEqual(stream.metrics, before)
        stream, _ = self.stream()
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaises(NativeStreamError):
                executor.submit(drive_paced, stream, pcm(1)).result(timeout=2)
        self.assertEqual(stream.accepted_samples, 0)

    def test_partial_chunk_uses_absolute_sample_end_and_never_future_audio(self):
        stream, adapter = self.stream()
        source = pcm_ms(10, 27) + pcm(7, 28)
        owner = threading.current_thread()
        observed = []
        traces = []
        producer_threads = []
        original_push = stream.push

        def push(sequence, content):
            producer_threads.append(threading.current_thread())
            return original_push(sequence, content)

        def event_callback(event, elapsed_ns):
            self.assertIs(threading.current_thread(), owner)
            observed.append((event, elapsed_ns))

        def trace_callback(trace, elapsed_ns):
            self.assertIs(threading.current_thread(), owner)
            self.assertLessEqual(
                trace.analysis_end_sample * 1_000_000_000 // 16_000, elapsed_ns
            )
            traces.append((trace, elapsed_ns))

        with (
            patch.object(stream, "push", side_effect=push),
            patch.object(stream, "close") as close,
        ):
            result = drive_paced(
                stream,
                source,
                config=self.config,
                on_event=event_callback,
                on_trace=trace_callback,
            )
            close.assert_not_called()
        self.assertEqual(result.status, "completed")
        self.assertIsNone(result.error)
        self.assertEqual(result.offered_samples, 167)
        self.assertEqual(result.accepted_samples, 167)
        self.assertEqual(
            [item.sequence_number for item in result.admissions], [0, 1, 2]
        )
        self.assertEqual(
            [(item.start_sample, item.end_sample) for item in result.admissions],
            [(0, 80), (80, 160), (160, 167)],
        )
        self.assertEqual(
            [item.scheduled_ns for item in result.admissions],
            [5_000_000, 10_000_000, 10_437_500],
        )
        for item in result.admissions:
            self.assertLessEqual(item.scheduled_ns, item.offered_ns)
            self.assertLessEqual(item.offered_ns, item.accepted_ns)
            self.assertLessEqual(item.accepted_ns, result.elapsed_ns)
            self.assertLessEqual(
                item.buffered_samples, self.stream_config.max_buffer_ms * 16
            )
        self.assertLessEqual(
            result.admissions[-1].accepted_ns, result.input_finished_ns
        )
        self.assertLessEqual(result.input_finished_ns, result.elapsed_ns)
        self.assertEqual(
            result.max_source_lag_ns,
            max(item.offered_ns - item.scheduled_ns for item in result.admissions),
        )
        self.assertEqual(
            result.max_admission_lag_ns,
            max(item.accepted_ns - item.scheduled_ns for item in result.admissions),
        )
        self.assertEqual(result.undelivered_events, ())
        self.assertIsNone(result.undelivered_event_ns)
        self.assertTrue(producer_threads)
        self.assertTrue(
            all(
                thread is not owner and not thread.is_alive()
                for thread in producer_threads
            )
        )
        self.assertEqual(
            sum(event.kind is StreamEventKind.FINAL for event, _ in observed), 1
        )
        self.assertTrue(
            all(0 <= elapsed <= result.elapsed_ns for _, elapsed in observed + traces)
        )
        self.assertEqual(len(traces), len({trace.decode_index for trace, _ in traces}))
        self.assertTrue(stream.done)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assertEqual(adapter.worker.queue_depth, 0)
        self.assertEqual(stream.step(), ())
        with self.assertRaises(FrozenInstanceError):
            result.status = "cancelled"
        with self.assertRaises(FrozenInstanceError):
            result.admissions[0].end_sample = 81

    def test_producer_admits_later_chunk_while_owner_is_blocked(self):
        third_admitted = threading.Event()
        owner_entered = threading.Event()
        accepted_while_blocked = []

        def build(content):
            if not owner_entered.is_set():
                owner_entered.set()
                self.assertTrue(third_admitted.wait(2), "producer waited for decoder")
                accepted_while_blocked.append(stream.accepted_samples)
            return content

        stream, _ = self.stream(mel_builder=build)
        original_push = stream.push

        def push(sequence, content):
            if sequence == 1:
                self.assertTrue(owner_entered.wait(2), "owner never began decoding")
            accepted = original_push(sequence, content)
            if sequence == 2:
                third_admitted.set()
            return accepted

        with patch.object(stream, "push", side_effect=push):
            result = drive_paced(stream, pcm_ms(20, 30), config=self.config)
        self.assertEqual(result.status, "completed")
        self.assertTrue(owner_entered.is_set())
        self.assertGreaterEqual(accepted_while_blocked[0], 15 * 16)
        self.assertEqual(stream.metrics.accepted_chunks, 4)

    def test_slow_event_consumer_does_not_pause_the_source_clock(self):
        callback_entered = threading.Event()
        third_admitted = threading.Event()
        accepted_during_callback = []
        stream, _ = self.stream()
        original_push = stream.push

        def push(sequence, content):
            if sequence == 1:
                self.assertTrue(callback_entered.wait(2))
            accepted = original_push(sequence, content)
            if sequence == 2:
                third_admitted.set()
            return accepted

        def consume(event, elapsed_ns):
            if not callback_entered.is_set():
                callback_entered.set()
                self.assertTrue(third_admitted.wait(2), "source waited for consumer")
                accepted_during_callback.append(stream.accepted_samples)

        with patch.object(stream, "push", side_effect=push):
            result = drive_paced(
                stream, pcm_ms(20, 38), config=self.config, on_event=consume
            )
        self.assertEqual(result.status, "completed")
        self.assertGreaterEqual(accepted_during_callback[0], 15 * 16)
        self.assertEqual(stream.metrics.accepted_chunks, 4)

    def test_overload_rejects_whole_chunk_without_retry_eof_or_hidden_close(self):
        second_offered = threading.Event()
        entered = threading.Event()
        attempted = []

        def build(content):
            entered.set()
            self.assertTrue(second_offered.wait(2), "producer did not offer next chunk")
            return content

        stream, _ = self.stream(
            config=replace(
                self.stream_config,
                preview_interval_ms=1,
                max_window_ms=3,
                max_buffer_ms=3,
            ),
            mel_builder=build,
        )
        original_push = stream.push

        def push(sequence, content):
            attempted.append((sequence, len(content)))
            if sequence == 1:
                self.assertTrue(entered.wait(2))
            try:
                return original_push(sequence, content)
            finally:
                if sequence == 1:
                    second_offered.set()

        with (
            patch.object(stream, "push", side_effect=push),
            patch.object(stream, "close") as close,
        ):
            result = drive_paced(
                stream, pcm_ms(6, 31), config=replace(self.config, chunk_ms=2)
            )
            close.assert_not_called()
        self.assertEqual(result.status, "source_overload")
        self.assertIsInstance(result.error, AudioBufferFullError)
        self.assertEqual(attempted, [(0, 64), (1, 64)])
        self.assertEqual(result.offered_samples, 64)
        self.assertEqual(result.accepted_samples, 32)
        self.assertEqual(len(result.admissions), 1)
        self.assertEqual(stream.metrics.buffered_samples, 32)
        self.assertEqual(stream.expected_chunk, 1)
        self.assertFalse(stream.input_finished)
        self.assertIsNone(result.input_finished_ns)
        self.assertFalse(stream.done)

    def test_source_lag_stops_instead_of_catching_up_with_a_burst(self):
        stream, _ = self.stream()
        metrics = ContinuousTranscriptStream.metrics
        owner = threading.current_thread()

        def delayed_record():
            if threading.current_thread() is not owner:
                threading.Event().wait(0.15)
            return metrics.fget(stream)

        with (
            patch.object(stream, "push", wraps=stream.push) as push,
            patch.object(
                ContinuousTranscriptStream,
                "metrics",
                new_callable=PropertyMock,
                side_effect=delayed_record,
            ),
        ):
            result = drive_paced(
                stream,
                pcm_ms(15, 32),
                config=replace(self.config, max_source_lag_ms=50),
            )
        self.assertEqual(result.status, "source_lag")
        self.assertEqual(push.call_count, 1)
        self.assertEqual(result.accepted_samples, 80)
        self.assertGreater(result.max_source_lag_ns, 50_000_000)
        self.assertFalse(stream.input_finished)
        self.assertIsNone(result.input_finished_ns)

    def test_late_final_admission_is_reported_without_synthetic_eof(self):
        stream, _ = self.stream()
        original_push = stream.push

        def slow_push(sequence, content):
            threading.Event().wait(0.15)
            return original_push(sequence, content)

        with patch.object(stream, "push", side_effect=slow_push) as push:
            result = drive_paced(
                stream,
                pcm_ms(5, 41),
                config=replace(self.config, max_source_lag_ms=50),
            )
        self.assertEqual(result.status, "admission_lag")
        self.assertEqual(push.call_count, 1)
        self.assertEqual(result.offered_samples, 80)
        self.assertEqual(result.accepted_samples, 80)
        self.assertEqual(len(result.admissions), 1)
        self.assertGreater(result.max_admission_lag_ns, 50_000_000)
        self.assertEqual(stream.metrics.buffered_samples, 80)
        self.assertFalse(stream.input_finished)
        self.assertIsNone(result.input_finished_ns)

    def test_producer_error_is_returned_unchanged_without_eof_or_retry(self):
        stream, _ = self.stream()
        error = RuntimeError("input admission failed")
        with patch.object(stream, "push", side_effect=error) as push:
            result = drive_paced(stream, pcm_ms(10, 39), config=self.config)
        self.assertEqual(result.status, "runtime_error")
        self.assertIs(result.error, error)
        self.assertEqual(push.call_count, 1)
        self.assertEqual(result.offered_samples, 80)
        self.assertEqual(result.accepted_samples, 0)
        self.assertEqual(result.admissions, ())
        self.assertFalse(stream.input_finished)

    def test_cancellation_wakes_source_wait_and_joins_before_return(self):
        stream, _ = self.stream()
        cancel = threading.Event()
        timer = threading.Timer(0.02, cancel.set)
        timer.start()
        try:
            with (
                patch.object(stream, "push", wraps=stream.push) as push,
                patch.object(stream, "close") as close,
            ):
                result = drive_paced(
                    stream,
                    pcm_ms(1000),
                    config=replace(self.config, chunk_ms=1000),
                    cancel=cancel,
                )
                push.assert_not_called()
                close.assert_not_called()
            self.assertEqual(result.status, "cancelled")
            self.assertLess(result.elapsed_ns, 500_000_000)
            self.assertEqual(result.admissions, ())
            self.assertEqual(result.accepted_samples, 0)
            self.assertFalse(stream.input_finished)
        finally:
            timer.cancel()
            timer.join()

    def test_step_limit_leaves_active_work_for_caller(self):
        stream, adapter = self.stream()
        with patch.object(stream, "close") as close:
            result = drive_paced(
                stream, pcm_ms(1, 33), config=replace(self.config, max_steps=1)
            )
            close.assert_not_called()
        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.driver_steps, 1)
        self.assertTrue(stream.active)
        self.assertEqual(stream.metrics.buffered_samples, 16)
        self.assertEqual(adapter.worker.queue_depth, 1)

    def test_drain_timeout_is_cooperative_and_retains_unpublished_work(self):
        adapter = EvidenceNativeAdapter(
            speech_result, on_step=lambda: threading.Event().wait(0.03)
        )
        stream, _ = self.stream(adapter)
        with patch.object(stream, "close") as close:
            result = drive_paced(
                stream, pcm_ms(1, 34), config=replace(self.config, max_drain_ms=10)
            )
            close.assert_not_called()
        self.assertEqual(result.status, "drain_timeout")
        self.assertGreaterEqual(
            result.elapsed_ns - result.input_finished_ns, 10_000_000
        )
        self.assertEqual(stream.metrics.committed_samples, 0)
        self.assertEqual(stream.metrics.buffered_samples, 16)
        self.assertTrue(stream.active)
        self.assertEqual(len(adapter.calls), 1)

    def test_callback_failure_preserves_original_error_and_does_not_close(self):
        for callback in ("on_event", "on_trace"):
            with self.subTest(callback=callback):
                stream, _ = self.stream()
                error = RuntimeError("consumer failed")
                called = []

                def fail(value, elapsed_ns):
                    called.append((value, elapsed_ns))
                    raise error

                with patch.object(stream, "close") as close:
                    result = drive_paced(
                        stream, pcm_ms(1, 35), config=self.config, **{callback: fail}
                    )
                    close.assert_not_called()
                self.assertEqual(result.status, "runtime_error")
                self.assertIs(result.error, error)
                self.assertEqual(len(called), 1)
                self.assertEqual(result.accepted_samples, 16)

    def test_five_processing_failures_keep_error_identity_and_accepted_pcm(self):
        for stage in ("preprocess", "start", "step", "prepare", "finish"):
            with self.subTest(stage=stage):
                adapter = EvidenceNativeAdapter(speech_result, fail_once=stage)

                def build(content):
                    if stage == "preprocess":
                        raise adapter.error
                    return content

                stream, _ = self.stream(adapter, mel_builder=build)
                with patch.object(stream, "close") as close:
                    result = drive_paced(stream, pcm_ms(1, 36), config=self.config)
                    close.assert_not_called()
                self.assertEqual(result.status, "runtime_error")
                self.assertIs(result.error, adapter.error)
                self.assertEqual(result.accepted_samples, 16)
                self.assertEqual(stream.metrics.buffered_samples, 16)
                self.assertEqual(stream.metrics.committed_samples, 0)
                self.assertFalse(stream.done)
                self.assertLessEqual(len(adapter.calls), 1)

    def test_callback_failure_returns_undelivered_batch_without_auto_replay(self):
        stream, _ = self.stream()
        error = RuntimeError("commit consumer failed")
        attempted = []

        def consume(event, elapsed_ns):
            attempted.append((event, elapsed_ns))
            if event.kind is StreamEventKind.COMMIT:
                raise error

        result = drive_paced(
            stream, pcm_ms(1, 42), config=self.config, on_event=consume
        )
        self.assertEqual(result.status, "runtime_error")
        self.assertIs(result.error, error)
        self.assertEqual(
            [event.kind for event, _ in attempted],
            [StreamEventKind.PROVISIONAL, StreamEventKind.COMMIT],
        )
        self.assertEqual(
            [event.kind for event in result.undelivered_events],
            [StreamEventKind.COMMIT, StreamEventKind.FINAL],
        )
        self.assertIs(result.undelivered_events[0], attempted[-1][0])
        self.assertEqual(result.undelivered_event_ns, attempted[-1][1])
        self.assertTrue(stream.done)
        self.assertEqual(stream.state.version, 1)
        self.assertEqual(stream.metrics.committed_samples, 16)
        self.assertEqual(stream.step(), ())

    def test_callback_closing_stream_early_cannot_report_completed_replay(self):
        stream, _ = self.stream()
        callback_entered = threading.Event()
        original_push = stream.push

        def push(sequence, content):
            if sequence == 1:
                self.assertTrue(callback_entered.wait(2))
            return original_push(sequence, content)

        def consume(event, elapsed_ns):
            if not callback_entered.is_set():
                stream.close()
                callback_entered.set()

        with patch.object(stream, "push", side_effect=push):
            result = drive_paced(
                stream, pcm_ms(20, 43), config=self.config, on_event=consume
            )
        self.assertEqual(result.status, "runtime_error")
        self.assertLess(result.accepted_samples, 20 * 16)
        self.assertIsNone(result.input_finished_ns)
        self.assertEqual(stream.metrics.committed_samples, 0)

    def test_trace_callback_failure_does_not_replace_native_operation_error(self):
        adapter = EvidenceNativeAdapter(speech_result, fail_once="finish")
        stream, _ = self.stream(adapter)
        callback_error = RuntimeError("trace sink failed")
        callback = Mock(side_effect=callback_error)
        result = drive_paced(
            stream, pcm_ms(1, 40), config=self.config, on_trace=callback
        )
        self.assertEqual(result.status, "runtime_error")
        self.assertIs(result.error, adapter.error)
        callback.assert_called_once()
        self.assertEqual(stream.metrics.buffered_samples, 16)
        self.assertEqual(stream.metrics.committed_samples, 0)

    def test_retained_transaction_reaches_caller_unchanged_before_recovery(self):
        for phase in ("fence", "release"):
            with self.subTest(phase=phase), ExitStack() as stack:
                stream, adapter = self.stream()
                original_step = stream.step
                errors = []
                installed = False

                def step():
                    nonlocal installed
                    if adapter.runs and adapter.runs[-1].complete and not installed:
                        installed = True
                        run = adapter.runs[-1]
                        if phase == "fence":
                            stack.enter_context(patch.object(run.fence, "fail", True))
                        else:
                            stack.enter_context(
                                patch.object(
                                    type(run.transaction._lease),
                                    "release",
                                    side_effect=RuntimeError("release failed"),
                                )
                            )
                    try:
                        return original_step()
                    except BaseException as error:
                        errors.append(error)
                        raise

                with (
                    patch.object(stream, "step", side_effect=step),
                    patch.object(stream, "close") as close,
                ):
                    result = drive_paced(stream, pcm_ms(1, 37), config=self.config)
                    close.assert_not_called()
                self.assertEqual(result.status, "runtime_error")
                self.assertEqual(len(errors), 1)
                self.assertIs(result.error, errors[0])
                self.assertIsInstance(result.error, TransactionRetainedError)
                run = adapter.runs[-1]
                self.assertIs(result.error.transaction, run.transaction)
                self.assertEqual(
                    result.error.committed_state is not None, phase == "release"
                )
                self.assertEqual(stream.metrics.buffered_samples, 16)
                self.assertEqual(stream.metrics.committed_samples, 0)
                self.assertFalse(run.capacity_released)
                stack.close()
                self.assertTrue(adapter.worker.recover(run.transaction))
                for _ in range(10):
                    if stream.done:
                        break
                    stream.step()
                self.assertTrue(stream.done)
                self.assertEqual(stream.metrics.buffered_samples, 0)
                self.assertEqual(stream.state.version, 1)


if __name__ == "__main__":
    unittest.main()
