import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from unittest.mock import patch

from whisper_runtime import (
    Budget,
    ImmediateFence,
    ModelSnapshot,
    QueueFullError,
    ReapReport,
    RequestCancelledError,
    RequestState,
    RequestStatus,
    RequestTerminalError,
    ResourceVector,
    Session,
    SessionState,
    StaleSessionError,
    TransactionExpiredError,
    TransactionRetainedError,
    TransactionStateError,
    TransactionStatus,
    WindowResult,
    WindowTransaction,
    Worker,
)


class RecordingFence:
    def __init__(self) -> None:
        self.stop_requests = 0
        self.waits = 0

    def request_stop(self) -> None:
        self.stop_requests += 1

    def completion_fence(self) -> "RecordingFence":
        return self

    def wait(self) -> None:
        self.waits += 1


class AdvancingFence(RecordingFence):
    def __init__(self, advance: Callable[[], None]) -> None:
        super().__init__()
        self._advance = advance

    def wait(self) -> None:
        super().wait()
        self._advance()


class FailOnceFence(RecordingFence):
    def wait(self) -> None:
        super().wait()
        if self.waits == 1:
            raise RuntimeError("backend fence failed")


class SwitchableFailFence(RecordingFence):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def wait(self) -> None:
        super().wait()
        if self.fail:
            raise RuntimeError("backend fence failed")


class FailFirstStopFence(RecordingFence):
    def request_stop(self) -> None:
        super().request_stop()
        if self.stop_requests == 1:
            raise RuntimeError("stop signal failed")


class BlockingFailFirstStopFence(FailFirstStopFence):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def request_stop(self) -> None:
        if self.stop_requests == 0:
            self.first_started.set()
            self.release_first.wait()
        super().request_stop()


class BlockingStopFence(RecordingFence):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def request_stop(self) -> None:
        super().request_stop()
        self.started.set()
        self.release.wait()


class BlockingFenceAndStop(RecordingFence):
    def __init__(self) -> None:
        super().__init__()
        self.fence_started = threading.Event()
        self.release_fence = threading.Event()
        self.stop_started = threading.Event()
        self.release_stop = threading.Event()

    def wait(self) -> None:
        super().wait()
        self.fence_started.set()
        self.release_fence.wait()

    def request_stop(self) -> None:
        super().request_stop()
        self.stop_started.set()
        self.release_stop.wait()


class ReentrantStopFence(RecordingFence):
    def __init__(self) -> None:
        super().__init__()
        self.transaction: WindowTransaction | None = None

    def request_stop(self) -> None:
        super().request_stop()
        if self.transaction is None:
            raise AssertionError("transaction is not bound")
        self.transaction.cancel()


class RuntimeTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ModelSnapshot(
            model_id="whisper-large-v3-turbo",
            revision="2024-09",
            backend="pytorch",
            fingerprint="sha256:test-model",
        )
        self.capacity = ResourceVector(
            memory_bytes=1_000,
            compute_units=4,
            stream_slots=4,
        )
        self.cost = ResourceVector(
            memory_bytes=100,
            compute_units=1,
            stream_slots=1,
        )

    def request(self, request_id: str, session_id: str, seed: int) -> RequestState:
        return RequestState(
            request_id=request_id,
            session_id=session_id,
            model=self.model,
            rng_seed=seed,
        )

    @staticmethod
    def result(window_id: str, text: str) -> WindowResult:
        return WindowResult(
            window_id=window_id,
            text=text,
            start_ms=0,
            end_ms=1_000,
        )

    def test_window_timestamps_require_non_boolean_integers(self) -> None:
        for field, value in (("start_ms", True), ("end_ms", 1.5)):
            with self.subTest(field=field):
                arguments: dict[str, object] = {
                    "window_id": "window",
                    "text": "text",
                    "start_ms": 0,
                    "end_ms": 1_000,
                }
                arguments[field] = value
                with self.assertRaisesRegex(TypeError, f"{field} must be an integer"):
                    WindowResult(**arguments)  # type: ignore[arg-type]

    def test_stale_commit_is_rejected_and_releases_resources(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=2)
        session = Session("session")
        first = worker.prepare(
            session=session,
            request=self.request("request-a", "session", 1),
            window_id="window-a",
            resources=self.cost,
        )
        stale = worker.prepare(
            session=session,
            request=self.request("request-b", "session", 2),
            window_id="window-b",
            resources=self.cost,
        )

        first.start(ImmediateFence())
        stale.start(ImmediateFence())
        first.commit(self.result("window-a", "first"))
        with self.assertRaises(StaleSessionError):
            stale.commit(self.result("window-b", "stale"))

        self.assertEqual(session.snapshot().version, 1)
        self.assertEqual(len(session.snapshot().windows), 1)
        self.assertIs(stale.status, TransactionStatus.ABORTED)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(budget.lease_count, 0)
        self.assertEqual(worker.queue_depth, 0)

    def test_repeated_commit_is_idempotent_only_for_the_same_result(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        transaction = worker.prepare(
            session=session,
            request=self.request("request", "session", 3),
            window_id="window",
            resources=self.cost,
        )
        result = self.result("window", "committed")

        transaction.start(ImmediateFence())
        first_state = transaction.commit(result)
        second_state = transaction.commit(result)

        self.assertIs(first_state, second_state)
        self.assertEqual(first_state.version, 1)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(budget.lease_count, 0)
        with self.assertRaises(TransactionStateError):
            transaction.commit(self.result("window", "different"))
        with self.assertRaises(TransactionStateError):
            transaction.abort()

    def test_abort_restores_resources_once(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        transaction = worker.prepare(
            session=session,
            request=self.request("request", "session", 4),
            window_id="window",
            resources=self.cost,
        )

        self.assertEqual(budget.in_use, self.cost)
        self.assertTrue(transaction.abort())
        self.assertFalse(transaction.abort())
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(budget.lease_count, 0)
        self.assertEqual(worker.queue_depth, 0)
        with self.assertRaises(TransactionStateError):
            transaction.commit(self.result("window", "late"))

    def test_worker_rejects_admission_beyond_queue_bound(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        first_session = Session("session-a")
        first = worker.prepare(
            session=first_session,
            request=self.request("request-a", "session-a", 5),
            window_id="window-a",
            resources=self.cost,
        )

        with self.assertRaises(QueueFullError):
            worker.prepare(
                session=Session("session-b"),
                request=self.request("request-b", "session-b", 6),
                window_id="window-b",
                resources=self.cost,
            )

        self.assertEqual(worker.queue_depth, 1)
        self.assertEqual(budget.in_use, self.cost)
        first.abort()

        replacement = worker.prepare(
            session=Session("session-b"),
            request=self.request("request-b", "session-b", 6),
            window_id="window-b",
            resources=self.cost,
        )
        replacement.abort()
        self.assertEqual(budget.available, self.capacity)

    def test_independent_sessions_commute(self) -> None:
        def run(order: tuple[str, str]) -> tuple[SessionState, SessionState]:
            budget = Budget(self.capacity)
            worker = Worker("gpu-0", self.model, budget, queue_capacity=2)
            sessions = {"a": Session("session-a"), "b": Session("session-b")}
            requests = {
                "a": self.request("request-a", "session-a", 101),
                "b": self.request("request-b", "session-b", 202),
            }
            transactions = {
                key: worker.prepare(
                    session=sessions[key],
                    request=requests[key],
                    window_id=f"window-{key}",
                    resources=self.cost,
                )
                for key in ("a", "b")
            }
            for transaction in transactions.values():
                transaction.start(ImmediateFence())

            for key in order:
                draw = transactions[key].randrange(1_000_000)
                transactions[key].commit(self.result(f"window-{key}", str(draw)))

            self.assertEqual(budget.available, self.capacity)
            self.assertEqual(worker.queue_depth, 0)
            return sessions["a"].snapshot(), sessions["b"].snapshot()

        self.assertEqual(run(("a", "b")), run(("b", "a")))

    def test_cancellation_aborts_and_cleans_up(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 7)
        transaction = worker.prepare(
            session=session,
            request=request,
            window_id="window",
            resources=self.cost,
        )

        self.assertTrue(worker.cancel(transaction))
        self.assertTrue(request.cancelled)
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(budget.lease_count, 0)
        self.assertEqual(worker.queue_depth, 0)
        self.assertFalse(worker.cancel(transaction))
        with self.assertRaises(TransactionStateError):
            transaction.commit(self.result("window", "cancelled"))

    def test_commit_and_cancel_have_one_atomic_winner(self) -> None:
        for iteration in range(100):
            budget = Budget(self.capacity)
            worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
            session = Session(f"session-{iteration}")
            request = self.request(
                f"request-{iteration}", session.session_id, iteration
            )
            transaction = worker.prepare(
                session=session,
                request=request,
                window_id=f"window-{iteration}",
                resources=self.cost,
            )
            start = threading.Barrier(2)
            running = threading.Event()
            commit_rejected: list[bool] = []
            cancellation_result: list[bool] = []

            def commit() -> None:
                transaction.start(ImmediateFence())
                running.set()
                start.wait()
                try:
                    transaction.commit(self.result(f"window-{iteration}", "committed"))
                except (RequestCancelledError, TransactionStateError):
                    commit_rejected.append(True)

            def cancel() -> None:
                running.wait()
                start.wait()
                cancellation_result.append(worker.cancel(transaction))

            commit_thread = threading.Thread(target=commit)
            cancel_thread = threading.Thread(target=cancel)
            commit_thread.start()
            cancel_thread.start()
            commit_thread.join()
            cancel_thread.join()

            if session.snapshot().version == 1:
                self.assertIs(transaction.status, TransactionStatus.COMMITTED)
                self.assertFalse(request.cancelled)
                self.assertFalse(cancellation_result[0])
                self.assertFalse(commit_rejected)
            else:
                self.assertEqual(session.snapshot().version, 0)
                self.assertIs(transaction.status, TransactionStatus.ABORTED)
                self.assertTrue(request.cancelled)
                self.assertTrue(cancellation_result[0])
                self.assertEqual(commit_rejected, [True])

            self.assertEqual(budget.available, self.capacity)
            self.assertEqual(budget.lease_count, 0)
            self.assertEqual(worker.queue_depth, 0)

    def test_request_identity_is_read_only(self) -> None:
        request = self.request("request", "session", 8)

        with self.assertRaises(AttributeError):
            request.request_id = "replacement"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            request.session_id = "replacement"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            request.model = self.model  # type: ignore[misc]

        self.assertEqual(request.request_id, "request")
        self.assertEqual(request.session_id, "session")

    def test_direct_cancel_and_commit_have_one_atomic_winner(self) -> None:
        for iteration in range(100):
            budget = Budget(self.capacity)
            worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
            session = Session(f"session-{iteration}")
            request = self.request(
                f"request-{iteration}", session.session_id, iteration
            )
            transaction = worker.prepare(
                session=session,
                request=request,
                window_id=f"window-{iteration}",
                resources=self.cost,
            )
            start = threading.Barrier(2)
            running = threading.Event()
            commit_rejected: list[bool] = []
            cancellation_result: list[bool] = []

            def commit() -> None:
                transaction.start(ImmediateFence())
                running.set()
                start.wait()
                try:
                    transaction.commit(self.result(f"window-{iteration}", "committed"))
                except (RequestCancelledError, TransactionStateError):
                    commit_rejected.append(True)

            def cancel() -> None:
                running.wait()
                start.wait()
                cancellation_result.append(request.cancel())

            commit_thread = threading.Thread(target=commit)
            cancel_thread = threading.Thread(target=cancel)
            commit_thread.start()
            cancel_thread.start()
            commit_thread.join()
            cancel_thread.join()

            if session.snapshot().version == 1:
                self.assertIs(transaction.status, TransactionStatus.COMMITTED)
                self.assertIs(request.status, RequestStatus.COMMITTED)
                self.assertFalse(cancellation_result[0])
                self.assertFalse(commit_rejected)
            else:
                self.assertEqual(session.snapshot().version, 0)
                self.assertIs(transaction.status, TransactionStatus.ABORTED)
                self.assertIs(request.status, RequestStatus.CANCELLED)
                self.assertTrue(cancellation_result[0])
                self.assertEqual(commit_rejected, [True])

            self.assertEqual(budget.available, self.capacity)
            self.assertEqual(budget.lease_count, 0)
            self.assertEqual(worker.queue_depth, 0)

    def test_cancel_after_commit_cannot_change_the_request_outcome(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 9)
        transaction = worker.prepare(
            session=session,
            request=request,
            window_id="window",
            resources=self.cost,
        )

        transaction.start(ImmediateFence())
        transaction.commit(self.result("window", "committed"))

        self.assertFalse(request.cancel())
        self.assertFalse(worker.cancel(transaction))
        self.assertIs(request.status, RequestStatus.COMMITTED)
        self.assertIs(transaction.status, TransactionStatus.COMMITTED)

    def test_session_retains_only_its_configured_commit_horizon(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session", history_limit=2)

        for index in range(5):
            transaction = worker.prepare(
                session=session,
                request=self.request(f"request-{index}", "session", index),
                window_id=f"window-{index}",
                resources=self.cost,
            )
            transaction.start(ImmediateFence())
            transaction.commit(self.result(f"window-{index}", f"result-{index}"))

        state = session.snapshot()
        self.assertEqual(state.version, 5)
        self.assertEqual(len(state.windows), 2)
        self.assertEqual(
            [record.result.window_id for record in state.windows],
            ["window-3", "window-4"],
        )

    def test_committed_prefix_advances_atomically_and_never_implicitly(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")

        first = worker.prepare(
            session=session,
            request=self.request("request-0", "session", 0),
            window_id="window-0",
            resources=self.cost,
        )
        first.start(ImmediateFence())
        initial = first.commit(
            self.result("window-0", "first"),
            committed_through_ms=750,
        )

        self.assertEqual(initial.version, 1)
        self.assertEqual(initial.committed_through_ms, 750)
        self.assertEqual(initial.windows[0].committed_through_ms, 750)

        second = worker.prepare(
            session=session,
            request=self.request("request-1", "session", 1),
            window_id="window-1",
            resources=self.cost,
        )
        second.start(ImmediateFence())
        unchanged = second.commit(
            WindowResult(
                window_id="window-1",
                text="second",
                start_ms=750,
                end_ms=1_000,
            )
        )

        self.assertEqual(unchanged.version, 2)
        self.assertEqual(unchanged.committed_through_ms, 750)
        self.assertIsNone(unchanged.windows[-1].committed_through_ms)

    def test_committed_prefix_regression_aborts_without_publication(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")

        first = worker.prepare(
            session=session,
            request=self.request("request-0", "session", 0),
            window_id="window-0",
            resources=self.cost,
        )
        first.start(ImmediateFence())
        first.commit(
            self.result("window-0", "first"),
            committed_through_ms=1_000,
        )

        regressing = worker.prepare(
            session=session,
            request=self.request("request-1", "session", 1),
            window_id="window-1",
            resources=self.cost,
        )
        regressing.start(ImmediateFence())
        with self.assertRaisesRegex(ValueError, "must not regress"):
            regressing.commit(
                WindowResult(
                    window_id="window-1",
                    text="regression",
                    start_ms=1_000,
                    end_ms=2_000,
                ),
                committed_through_ms=999,
            )

        state = session.snapshot()
        self.assertEqual(state.version, 1)
        self.assertEqual(state.committed_through_ms, 1_000)
        self.assertEqual(len(state.windows), 1)
        self.assertIs(regressing.status, TransactionStatus.ABORTED)
        self.assertEqual(budget.available, self.capacity)

    def test_history_eviction_does_not_erase_the_committed_prefix(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session", history_limit=1)

        for index in range(3):
            transaction = worker.prepare(
                session=session,
                request=self.request(f"request-{index}", "session", index),
                window_id=f"window-{index}",
                resources=self.cost,
            )
            transaction.start(ImmediateFence())
            transaction.commit(
                WindowResult(
                    window_id=f"window-{index}",
                    text=f"result-{index}",
                    start_ms=index * 1_000,
                    end_ms=(index + 1) * 1_000,
                ),
                committed_through_ms=1_000 if index == 0 else None,
            )

        state = session.snapshot()
        self.assertEqual(state.version, 3)
        self.assertEqual(len(state.windows), 1)
        self.assertEqual(state.windows[0].result.window_id, "window-2")
        self.assertEqual(state.committed_through_ms, 1_000)

    def test_committed_prefix_rejects_a_later_overlapping_result(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")

        first = worker.prepare(
            session=session,
            request=self.request("request-0", "session", 0),
            window_id="window-0",
            resources=self.cost,
        )
        first.start(ImmediateFence())
        first.commit(
            self.result("window-0", "first"),
            committed_through_ms=1_000,
        )

        overlapping = worker.prepare(
            session=session,
            request=self.request("request-1", "session", 1),
            window_id="window-1",
            resources=self.cost,
        )
        overlapping.start(ImmediateFence())
        with self.assertRaisesRegex(ValueError, "overlap committed audio"):
            overlapping.commit(
                WindowResult(
                    window_id="window-1",
                    text="replacement",
                    start_ms=500,
                    end_ms=1_500,
                )
            )

        state = session.snapshot()
        self.assertEqual(state.version, 1)
        self.assertEqual(state.windows[0].result.text, "first")
        self.assertEqual(state.committed_through_ms, 1_000)
        self.assertIs(overlapping.status, TransactionStatus.ABORTED)

    def test_competing_commits_cannot_both_advance_the_committed_prefix(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=2)
        session = Session("session")
        first = worker.prepare(
            session=session,
            request=self.request("request-0", "session", 0),
            window_id="window-0",
            resources=self.cost,
        )
        competing = worker.prepare(
            session=session,
            request=self.request("request-1", "session", 1),
            window_id="window-1",
            resources=self.cost,
        )
        first.start(ImmediateFence())
        competing.start(ImmediateFence())

        first.commit(
            self.result("window-0", "winner"),
            committed_through_ms=500,
        )
        with self.assertRaises(StaleSessionError):
            competing.commit(
                self.result("window-1", "stale"),
                committed_through_ms=900,
            )

        state = session.snapshot()
        self.assertEqual(state.version, 1)
        self.assertEqual(state.committed_through_ms, 500)
        self.assertEqual(state.windows[0].result.text, "winner")
        self.assertIs(competing.status, TransactionStatus.ABORTED)
        self.assertEqual(budget.available, self.capacity)

    def test_committed_prefix_cannot_exceed_its_result(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        transaction = worker.prepare(
            session=session,
            request=self.request("request", "session", 0),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(ImmediateFence())

        with self.assertRaisesRegex(ValueError, "cannot exceed the result end"):
            transaction.commit(
                self.result("window", "invalid"),
                committed_through_ms=1_001,
            )

        self.assertEqual(session.snapshot().version, 0)
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(budget.available, self.capacity)

    def test_windows_remain_revisable_until_a_prefix_is_committed(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        later = worker.prepare(
            session=session,
            request=self.request("request-0", "session", 0),
            window_id="window-later",
            resources=self.cost,
        )
        later.start(ImmediateFence())
        later.commit(
            WindowResult(
                window_id="window-later",
                text="later hypothesis",
                start_ms=1_000,
                end_ms=2_000,
            )
        )

        earlier = worker.prepare(
            session=session,
            request=self.request("request-1", "session", 1),
            window_id="window-earlier",
            resources=self.cost,
        )
        earlier.start(ImmediateFence())
        state = earlier.commit(
            WindowResult(
                window_id="window-earlier",
                text="earlier revision",
                start_ms=0,
                end_ms=1_500,
            )
        )

        self.assertEqual(state.version, 2)
        self.assertIsNone(state.committed_through_ms)
        self.assertEqual(
            [record.result.window_id for record in state.windows],
            ["window-later", "window-earlier"],
        )
        self.assertEqual(budget.available, self.capacity)

    def test_committed_transaction_rejects_a_different_prefix(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        transaction = worker.prepare(
            session=session,
            request=self.request("request", "session", 0),
            window_id="window",
            resources=self.cost,
        )
        result = self.result("window", "result")

        transaction.start(ImmediateFence())
        first = transaction.commit(result, committed_through_ms=500)
        self.assertIs(
            transaction.commit(result, committed_through_ms=500),
            first,
        )
        with self.assertRaisesRegex(TransactionStateError, "different result"):
            transaction.commit(result, committed_through_ms=501)

        for invalid in (True, 500.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    transaction.commit(
                        result,
                        committed_through_ms=invalid,  # type: ignore[arg-type]
                    )
        self.assertEqual(session.snapshot().committed_through_ms, 500)

    def test_expiry_reaps_a_lost_transaction(self) -> None:
        now = [100.0]
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=5.0,
            clock=lambda: now[0],
        )
        session = Session("session")
        request = self.request("request", "session", 10)
        transaction = worker.prepare(
            session=session,
            request=request,
            window_id="window",
            resources=self.cost,
        )

        now[0] = 104.999
        self.assertEqual(worker.reap_expired(), ReapReport())
        self.assertEqual(worker.queue_depth, 1)
        now[0] = 105.0
        self.assertEqual(worker.reap_expired(), ReapReport(released=1))

        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertIs(request.status, RequestStatus.EXPIRED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(budget.lease_count, 0)

    def test_cancelled_request_is_rejected_before_admission(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 11)
        self.assertTrue(request.cancel())

        with self.assertRaises(RequestCancelledError):
            worker.prepare(
                session=session,
                request=request,
                window_id="window",
                resources=self.cost,
            )

        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_failed_transaction_construction_releases_its_lease(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)

        with self.assertRaises(ValueError):
            worker.prepare(
                session=Session("session"),
                request=self.request("request", "session", 13),
                window_id=" ",
                resources=self.cost,
            )

        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(budget.lease_count, 0)

    def test_failed_second_admission_does_not_abort_the_owner(self) -> None:
        first_budget = Budget(self.capacity)
        second_budget = Budget(self.capacity)
        first_worker = Worker("gpu-0", self.model, first_budget, queue_capacity=1)
        second_worker = Worker("gpu-1", self.model, second_budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 14)
        owner = first_worker.prepare(
            session=session,
            request=request,
            window_id="window",
            resources=self.cost,
        )

        with self.assertRaises(RequestTerminalError):
            second_worker.prepare(
                session=session,
                request=request,
                window_id="window",
                resources=self.cost,
            )

        self.assertIs(request.status, RequestStatus.PREPARED)
        self.assertEqual(second_worker.queue_depth, 0)
        self.assertEqual(second_budget.available, self.capacity)
        owner.start(ImmediateFence())
        owner.commit(self.result("window", "committed"))
        self.assertIs(request.status, RequestStatus.COMMITTED)
        self.assertEqual(first_worker.queue_depth, 0)
        self.assertEqual(first_budget.available, self.capacity)

    def test_aborted_random_draws_do_not_change_a_retry_stream(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        first = worker.prepare(
            session=session,
            request=self.request("attempt-1", "session", 12),
            window_id="window",
            resources=self.cost,
        )
        first.start(ImmediateFence())
        first_draw = first.randrange(1_000_000)
        first.abort()

        retry = worker.prepare(
            session=session,
            request=self.request("attempt-2", "session", 12),
            window_id="window",
            resources=self.cost,
        )
        retry.start(ImmediateFence())
        retry_draw = retry.randrange(1_000_000)
        retry.abort()

        self.assertEqual(first_draw, retry_draw)

    def test_running_deadline_requests_stop_before_releasing_capacity(self) -> None:
        now = [0.0]
        fence = RecordingFence()
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        request = self.request("request", "session", 15)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(fence)

        now[0] = 2.0
        self.assertEqual(worker.reap_expired(), ReapReport(stop_requested=1))
        self.assertEqual(worker.queue_depth, 1)
        self.assertEqual(budget.in_use, self.cost)
        self.assertEqual(fence.stop_requests, 1)

        with self.assertRaises(TransactionExpiredError):
            transaction.checkpoint()

        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertIs(request.status, RequestStatus.EXPIRED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(fence.stop_requests, 1)
        self.assertEqual(fence.waits, 1)

    def test_commit_cannot_publish_after_fence_crosses_deadline(self) -> None:
        now = [0.0]
        fence = AdvancingFence(lambda: now.__setitem__(0, 2.0))
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        session = Session("session")
        request = self.request("request", "session", 16)
        transaction = worker.prepare(
            session=session,
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(fence)

        with self.assertRaises(TransactionExpiredError):
            transaction.commit(self.result("window", "too late"))

        self.assertEqual(session.snapshot().version, 0)
        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(worker.queue_depth, 0)

    def test_external_abort_waits_for_owner_safe_point(self) -> None:
        fence = RecordingFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 17),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(fence)
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(transaction.abort()))
        thread.start()
        thread.join()

        self.assertEqual(result, [True])
        self.assertIs(transaction.status, TransactionStatus.RUNNING)
        self.assertEqual(budget.in_use, self.cost)
        self.assertEqual(worker.queue_depth, 1)

        with self.assertRaises(TransactionStateError):
            transaction.checkpoint()

        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(worker.queue_depth, 0)

    def test_supervisor_can_fence_work_after_owner_thread_exits(self) -> None:
        fence = RecordingFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 18),
            window_id="window",
            resources=self.cost,
        )
        owner = threading.Thread(target=lambda: transaction.start(fence))
        owner.start()
        owner.join()

        self.assertIs(transaction.status, TransactionStatus.RUNNING)
        self.assertTrue(worker.stop(transaction))
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(fence.waits, 1)
        self.assertGreaterEqual(fence.stop_requests, 1)
        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(worker.queue_depth, 0)

    def test_supervisor_does_not_release_work_owned_by_a_live_thread(self) -> None:
        fence = RecordingFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 181),
            window_id="window",
            resources=self.cost,
        )
        running = threading.Event()
        finish = threading.Event()
        stopped: list[bool] = []
        owner_errors: list[str] = []

        def execute() -> None:
            transaction.start(fence)
            running.set()
            finish.wait()
            try:
                transaction.checkpoint()
            except TransactionStateError:
                pass
            else:
                owner_errors.append("checkpoint accepted an abort request")

        owner = threading.Thread(target=execute)
        owner.start()
        running.wait()
        stopped.append(worker.stop(transaction))

        self.assertEqual(stopped, [True])
        self.assertTrue(owner.is_alive())
        self.assertIs(transaction.status, TransactionStatus.RUNNING)
        self.assertEqual(worker.queue_depth, 1)
        self.assertEqual(budget.in_use, self.cost)

        finish.set()
        owner.join()
        self.assertFalse(owner_errors)
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_failed_fence_quarantines_capacity_until_recovery(self) -> None:
        fence = FailOnceFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        transaction = worker.prepare(
            session=session,
            request=self.request("request", "session", 19),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(fence)

        with self.assertRaisesRegex(RuntimeError, "backend fence failed"):
            transaction.commit(self.result("window", "unpublished"))

        self.assertIs(transaction.status, TransactionStatus.QUARANTINED)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(worker.quarantined_count, 1)
        self.assertEqual(worker.queue_depth, 1)
        self.assertEqual(budget.in_use, self.cost)

        self.assertTrue(worker.recover(transaction))
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(worker.quarantined_count, 0)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_execute_relinquishes_owner_lease_on_a_persistent_pool_thread(self) -> None:
        scope = FailOnceFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        request = self.request("request", "session", 191)
        transactions: list[WindowTransaction] = []

        def operation(transaction: WindowTransaction) -> WindowResult:
            transactions.append(transaction)
            return self.result("window", "unpublished")

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                worker.execute,
                session=Session("session"),
                request=request,
                window_id="window",
                resources=self.cost,
                execution=scope,
                operation=operation,
            )
            with self.assertRaisesRegex(RuntimeError, "backend fence failed"):
                future.result()

            transaction = transactions[0]
            self.assertIs(transaction.status, TransactionStatus.ABORTED)

        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_execute_releases_capacity_when_operation_fails(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        request = self.request("request", "session", 20)

        def fail(_: object) -> WindowResult:
            raise ValueError("decoder failed")

        with self.assertRaisesRegex(ValueError, "decoder failed"):
            worker.execute(
                session=Session("session"),
                request=request,
                window_id="window",
                resources=self.cost,
                execution=ImmediateFence(),
                operation=fail,
            )

        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_budget_acquire_is_atomic_when_lease_construction_fails(self) -> None:
        budget = Budget(self.capacity)

        with patch(
            "whisper_runtime.resources.Lease",
            side_effect=MemoryError("allocation failed"),
        ):
            with self.assertRaises(MemoryError):
                budget.acquire(self.cost)

        self.assertEqual(budget.available, self.capacity)
        self.assertEqual(budget.lease_count, 0)

    def test_budget_release_is_atomic_when_vector_addition_fails(self) -> None:
        budget = Budget(self.capacity)
        lease = budget.acquire(self.cost)

        with patch.object(
            ResourceVector,
            "__add__",
            side_effect=MemoryError("allocation failed"),
        ):
            with self.assertRaises(MemoryError):
                lease.release()

        self.assertEqual(budget.in_use, self.cost)
        self.assertEqual(budget.lease_count, 1)
        self.assertFalse(lease.released)
        self.assertTrue(lease.release())
        self.assertEqual(budget.available, self.capacity)

    def test_lease_resource_vector_is_read_only(self) -> None:
        budget = Budget(self.capacity)
        lease = budget.acquire(self.cost)

        with self.assertRaises(AttributeError):
            lease.resources = ResourceVector(memory_bytes=1)

        self.assertEqual(lease.resources, self.cost)
        self.assertTrue(lease.release())
        self.assertEqual(budget.available, self.capacity)

    def test_worker_can_retry_terminal_cleanup_without_repeating_outcome(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 21),
            window_id="window",
            resources=self.cost,
        )

        with patch.object(
            ResourceVector,
            "__add__",
            side_effect=MemoryError("allocation failed"),
        ):
            self.assertTrue(transaction.abort())

        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertIsInstance(transaction.cleanup_error, MemoryError)
        self.assertEqual(worker.queue_depth, 1)
        self.assertEqual(worker.quarantined_count, 1)
        self.assertEqual(budget.lease_count, 1)

        self.assertTrue(worker.recover(transaction))
        self.assertIsNone(transaction.cleanup_error)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(worker.quarantined_count, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_rng_publication_failure_cannot_partially_commit_session(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 22)
        transaction = worker.prepare(
            session=session,
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(ImmediateFence())
        transaction.random()

        with patch(
            "whisper_runtime.state.random.Random",
            side_effect=MemoryError("allocation failed"),
        ):
            with self.assertRaises(MemoryError):
                transaction.commit(self.result("window", "not committed"))

        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(session.snapshot().windows, ())
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_commit_preparation_failure_aborts_without_leaking_capacity(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 23)
        transaction = worker.prepare(
            session=session,
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(ImmediateFence())

        with patch(
            "whisper_runtime.transaction.WindowRecord",
            side_effect=MemoryError("record allocation failed"),
        ):
            with self.assertRaisesRegex(MemoryError, "record allocation failed"):
                transaction.commit(self.result("window", "not committed"))

        self.assertEqual(session.snapshot().version, 0)
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_inactive_lease_is_rejected_before_session_publication(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 24)
        transaction = worker.prepare(
            session=session,
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(ImmediateFence())

        self.assertTrue(transaction._lease.release())
        with self.assertRaisesRegex(Exception, "lease is not active"):
            transaction.commit(self.result("window", "forged"))

        self.assertEqual(session.snapshot().version, 0)
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_submission_gate_drains_before_minting_the_final_fence(self) -> None:
        registered = threading.Event()
        entered = threading.Event()
        release = threading.Event()

        class RegisteredScope(RecordingFence):
            def completion_fence(inner_self) -> "RegisteredScope":
                if not registered.is_set():
                    raise AssertionError("fence was minted before submission drained")
                return inner_self

        scope = RegisteredScope()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 25),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(scope)

        def register_backend_work() -> None:
            entered.set()
            release.wait()
            registered.set()

        submitter = threading.Thread(
            target=lambda: transaction.submit(register_backend_work)
        )
        submitter.start()
        entered.wait()

        releaser = threading.Thread(target=lambda: (time.sleep(0.02), release.set()))
        releaser.start()
        state = transaction.commit(self.result("window", "committed"))
        submitter.join()
        releaser.join()

        self.assertEqual(state.version, 1)
        self.assertEqual(scope.waits, 1)
        with self.assertRaisesRegex(TransactionStateError, "submission is closed"):
            transaction.submit(lambda: None)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_submit_callback_defers_close_without_accepting_more_work(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 251),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(ImmediateFence())

        self.assertTrue(transaction.submit(transaction.abort))

        self.assertIs(transaction.status, TransactionStatus.RUNNING)
        with self.assertRaisesRegex(TransactionStateError, "submission is closed"):
            transaction.submit(lambda: None)
        with self.assertRaises(TransactionStateError):
            transaction.checkpoint()
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_submission_gate_rejects_async_callbacks(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 252),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(ImmediateFence())

        async def register_later() -> None:
            return None

        with self.assertRaisesRegex(TypeError, "deferred submit callbacks"):
            transaction.submit(register_later)

        def register_generator() -> object:
            yield None

        async def register_async_generator() -> object:
            yield None

        with self.assertRaisesRegex(TypeError, "deferred submit callbacks"):
            transaction.submit(register_generator)
        with self.assertRaisesRegex(TypeError, "deferred submit callbacks"):
            transaction.submit(register_async_generator)

        def return_awaitable() -> object:
            return register_later()

        with self.assertRaisesRegex(TypeError, "finish backend registration"):
            transaction.submit(return_awaitable)

        self.assertTrue(transaction.abort())
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_execution_scope_has_one_live_transaction_owner(self) -> None:
        scope = ImmediateFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=2)
        first = worker.prepare(
            session=Session("session-a"),
            request=self.request("request-a", "session-a", 253),
            window_id="window-a",
            resources=self.cost,
        )
        second = worker.prepare(
            session=Session("session-b"),
            request=self.request("request-b", "session-b", 254),
            window_id="window-b",
            resources=self.cost,
        )

        first.start(scope)
        with self.assertRaisesRegex(
            TransactionStateError,
            "cannot serve two live transactions",
        ):
            second.start(scope)

        self.assertTrue(second.abort())
        self.assertTrue(first.abort())

        third = worker.prepare(
            session=Session("session-c"),
            request=self.request("request-c", "session-c", 255),
            window_id="window-c",
            resources=self.cost,
        )
        third.start(scope)
        self.assertTrue(third.abort())
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_stop_signal_failure_can_be_retried_without_duplicate_delivery(
        self,
    ) -> None:
        scope = FailFirstStopFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 26),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(scope)

        first: list[bool] = []
        thread = threading.Thread(target=lambda: first.append(worker.stop(transaction)))
        thread.start()
        thread.join()
        self.assertEqual(first, [True])
        self.assertEqual(scope.stop_requests, 1)
        self.assertIsInstance(transaction.stop_signal_error, RuntimeError)
        self.assertIs(transaction.status, TransactionStatus.RUNNING)

        second: list[bool] = []
        thread = threading.Thread(
            target=lambda: second.append(worker.stop(transaction))
        )
        thread.start()
        thread.join()
        self.assertEqual(second, [True])
        self.assertEqual(scope.stop_requests, 2)
        self.assertIsNone(transaction.stop_signal_error)

        with self.assertRaises(TransactionStateError):
            transaction.checkpoint()
        self.assertEqual(scope.stop_requests, 2)
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(budget.available, self.capacity)

    def test_concurrent_stop_waiter_retries_a_failed_in_flight_signal(self) -> None:
        scope = BlockingFailFirstStopFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 261),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(scope)

        results: list[bool] = []
        first = threading.Thread(
            target=lambda: results.append(worker.stop(transaction))
        )
        second = threading.Thread(
            target=lambda: results.append(worker.stop(transaction))
        )
        first.start()
        scope.first_started.wait()
        second.start()
        scope.release_first.set()
        first.join()
        second.join()

        self.assertEqual(results, [True, True])
        self.assertEqual(scope.stop_requests, 2)
        with self.assertRaises(TransactionStateError):
            transaction.checkpoint()
        self.assertEqual(scope.stop_requests, 2)
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(budget.available, self.capacity)

    def test_close_waits_for_an_in_flight_stop_before_releasing_capacity(self) -> None:
        scope = BlockingStopFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 27),
            window_id="window",
            resources=self.cost,
        )
        transaction.start(scope)

        stopper = threading.Thread(target=lambda: worker.stop(transaction))
        stopper.start()
        scope.started.wait()
        self.assertTrue(stopper.is_alive())

        release_signal = threading.Thread(
            target=lambda: (time.sleep(0.02), scope.release.set())
        )
        release_signal.start()
        with self.assertRaises(TransactionStateError):
            transaction.checkpoint()
        stopper.join()
        release_signal.join()

        self.assertEqual(scope.stop_requests, 1)
        self.assertFalse(stopper.is_alive())
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_stop_signal_can_reenter_cancellation_without_deadlock(self) -> None:
        scope = ReentrantStopFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        transaction = worker.prepare(
            session=Session("session"),
            request=self.request("request", "session", 271),
            window_id="window",
            resources=self.cost,
        )
        scope.transaction = transaction
        transaction.start(scope)

        result: list[bool] = []
        stopper = threading.Thread(
            target=lambda: result.append(worker.stop(transaction))
        )
        stopper.start()
        stopper.join(timeout=1.0)

        self.assertFalse(stopper.is_alive())
        self.assertEqual(result, [True])
        self.assertEqual(scope.stop_requests, 1)
        with self.assertRaises(TransactionStateError):
            transaction.checkpoint()
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_supervisor_recovers_an_orphaned_quiescing_transaction(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        request = self.request("request", "session", 28)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )

        def abandon_during_close() -> None:
            transaction.start(ImmediateFence())
            with patch.object(
                transaction,
                "_finalize_backend",
                side_effect=SystemExit("owner exited"),
            ):
                transaction.commit(self.result("window", "unpublished"))

        owner = threading.Thread(target=abandon_during_close)
        owner.start()
        owner.join()

        self.assertIs(transaction.status, TransactionStatus.QUIESCING)
        self.assertEqual(budget.in_use, self.cost)
        self.assertTrue(worker.stop(transaction))
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_supervisor_finishes_after_owner_dies_post_fence(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        request = self.request("request", "session", 281)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )
        original_finalize = transaction._finalize_backend

        def finalize_then_exit(*args: object, **kwargs: object) -> None:
            original_finalize(*args, **kwargs)
            raise SystemExit("owner exited after the fence")

        def abandon_after_fence() -> None:
            transaction.start(ImmediateFence())
            with patch.object(
                transaction,
                "_finalize_backend",
                side_effect=finalize_then_exit,
            ):
                transaction.commit(self.result("window", "unpublished"))

        owner = threading.Thread(target=abandon_after_fence)
        owner.start()
        owner.join()

        self.assertIs(transaction.status, TransactionStatus.QUIESCING)
        self.assertEqual(budget.in_use, self.cost)
        self.assertTrue(worker.stop(transaction))
        self.assertIs(transaction.status, TransactionStatus.ABORTED)
        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_supervisor_preserves_expiry_as_the_terminal_cause(self) -> None:
        now = [0.0]
        scope = RecordingFence()
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        request = self.request("request", "session", 29)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )
        owner = threading.Thread(target=lambda: transaction.start(scope))
        owner.start()
        owner.join()

        now[0] = 2.0
        self.assertEqual(worker.reap_expired(), ReapReport(stop_requested=1))
        self.assertTrue(worker.stop(transaction))
        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertIs(request.status, RequestStatus.EXPIRED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_first_stop_cause_remains_stable(self) -> None:
        now = [0.0]
        scope = RecordingFence()
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        request = self.request("request", "session", 291)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(scope)

        now[0] = 2.0
        self.assertEqual(worker.reap_expired(), ReapReport(stop_requested=1))
        self.assertEqual(worker.reap_expired(), ReapReport())
        external_result: list[bool] = []
        thread = threading.Thread(
            target=lambda: external_result.append(transaction.abort())
        )
        thread.start()
        thread.join()
        self.assertEqual(external_result, [False])
        self.assertFalse(request.cancel())

        with self.assertRaises(TransactionExpiredError):
            transaction.checkpoint()
        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertIs(request.status, RequestStatus.EXPIRED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_owner_abort_cannot_replace_an_existing_expiry(self) -> None:
        now = [0.0]
        scope = RecordingFence()
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        request = self.request("request", "session", 292)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(scope)

        now[0] = 2.0
        worker.reap_expired()
        self.assertTrue(transaction.abort())

        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertIs(request.status, RequestStatus.EXPIRED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_quarantine_recovery_preserves_expiry(self) -> None:
        now = [0.0]
        scope = FailOnceFence()
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        request = self.request("request", "session", 30)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(scope)
        now[0] = 2.0
        worker.reap_expired()

        with self.assertRaisesRegex(RuntimeError, "backend fence failed"):
            transaction.checkpoint()
        self.assertIs(transaction.status, TransactionStatus.QUARANTINED)

        self.assertTrue(worker.recover(transaction))
        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertIs(request.status, RequestStatus.EXPIRED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_abort_after_an_unobserved_deadline_preserves_expiry(self) -> None:
        now = [0.0]
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        request = self.request("request", "session", 31)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(ImmediateFence())

        now[0] = 2.0
        self.assertTrue(transaction.abort())

        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertIs(request.status, RequestStatus.EXPIRED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_submit_rejects_work_after_the_deadline(self) -> None:
        now = [0.0]
        scope = RecordingFence()
        budget = Budget(self.capacity)
        worker = Worker(
            "gpu-0",
            self.model,
            budget,
            queue_capacity=1,
            transaction_ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        request = self.request("request", "session", 32)
        transaction = worker.prepare(
            session=Session("session"),
            request=request,
            window_id="window",
            resources=self.cost,
        )
        transaction.start(scope)
        invoked: list[bool] = []
        errors: list[BaseException] = []

        now[0] = 2.0

        def submit_after_deadline() -> None:
            try:
                transaction.submit(lambda: invoked.append(True))
            except BaseException as exc:
                errors.append(exc)

        submitter = threading.Thread(target=submit_after_deadline)
        submitter.start()
        submitter.join()

        self.assertFalse(invoked)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TransactionExpiredError)
        with self.assertRaises(TransactionExpiredError):
            transaction.checkpoint()
        self.assertIs(transaction.status, TransactionStatus.EXPIRED)
        self.assertIs(request.status, RequestStatus.EXPIRED)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_cancel_during_commit_fence_prevents_publication_and_joins_stop(
        self,
    ) -> None:
        scope = BlockingFenceAndStop()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 33)
        transactions: list[WindowTransaction] = []
        errors: list[BaseException] = []

        def operation(transaction: WindowTransaction) -> WindowResult:
            transactions.append(transaction)
            return self.result("window", "must not publish")

        def execute() -> None:
            try:
                worker.execute(
                    session=session,
                    request=request,
                    window_id="window",
                    resources=self.cost,
                    execution=scope,
                    operation=operation,
                )
            except BaseException as exc:
                errors.append(exc)

        owner = threading.Thread(target=execute)
        owner.start()
        scope.fence_started.wait()
        canceller = threading.Thread(target=request.cancel)
        canceller.start()
        scope.stop_started.wait()

        scope.release_fence.set()
        time.sleep(0.02)
        self.assertTrue(owner.is_alive())
        self.assertTrue(canceller.is_alive())
        self.assertEqual(budget.in_use, self.cost)

        scope.release_stop.set()
        owner.join()
        canceller.join()

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RequestCancelledError)
        self.assertIs(transactions[0].status, TransactionStatus.ABORTED)
        self.assertIs(request.status, RequestStatus.CANCELLED)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_supervisor_stop_during_commit_fence_prevents_publication(self) -> None:
        scope = BlockingFenceAndStop()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")
        request = self.request("request", "session", 34)
        transactions: list[WindowTransaction] = []
        errors: list[BaseException] = []

        def operation(transaction: WindowTransaction) -> WindowResult:
            transactions.append(transaction)
            return self.result("window", "must not publish")

        owner = threading.Thread(
            target=lambda: self._capture_execute_error(
                errors,
                worker,
                session,
                request,
                scope,
                operation,
            )
        )
        owner.start()
        scope.fence_started.wait()
        stopper = threading.Thread(target=lambda: worker.stop(transactions[0]))
        stopper.start()
        scope.stop_started.wait()
        scope.release_fence.set()
        time.sleep(0.02)
        self.assertTrue(owner.is_alive())
        self.assertEqual(budget.in_use, self.cost)
        scope.release_stop.set()
        owner.join()
        stopper.join()

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TransactionStateError)
        self.assertIs(transactions[0].status, TransactionStatus.ABORTED)
        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assertEqual(session.snapshot().version, 0)
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    @staticmethod
    def _capture_execute_error(
        errors: list[BaseException],
        worker: Worker,
        session: Session,
        request: RequestState,
        scope: BlockingFenceAndStop,
        operation: Callable[[WindowTransaction], WindowResult],
    ) -> None:
        try:
            worker.execute(
                session=session,
                request=request,
                window_id="window",
                resources=ResourceVector(
                    memory_bytes=100,
                    compute_units=1,
                    stream_slots=1,
                ),
                execution=scope,
                operation=operation,
            )
        except BaseException as exc:
            errors.append(exc)

    def test_execute_exposes_a_persistently_quarantined_transaction(self) -> None:
        scope = SwitchableFailFence()
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        request = self.request("request", "session", 35)

        with self.assertRaises(TransactionRetainedError) as raised:
            worker.execute(
                session=Session("session"),
                request=request,
                window_id="window",
                resources=self.cost,
                execution=scope,
                operation=lambda _: self.result("window", "unpublished"),
            )

        error = raised.exception
        self.assertIsNone(error.committed_state)
        self.assertIsInstance(error.operation_error, RuntimeError)
        self.assertIsInstance(error.retention_error, RuntimeError)
        self.assertIs(error.transaction.status, TransactionStatus.QUARANTINED)
        self.assertEqual(worker.queue_depth, 1)
        self.assertEqual(budget.in_use, self.cost)

        scope.fail = False
        self.assertTrue(worker.recover(error.transaction))
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_execute_exposes_committed_state_when_cleanup_stays_retained(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)
        session = Session("session")

        with patch.object(
            ResourceVector,
            "__add__",
            side_effect=MemoryError("allocation failed"),
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                worker.execute(
                    session=session,
                    request=self.request("request", "session", 36),
                    window_id="window",
                    resources=self.cost,
                    execution=ImmediateFence(),
                    operation=lambda _: self.result("window", "published"),
                )

        error = raised.exception
        self.assertIsNotNone(error.committed_state)
        assert error.committed_state is not None
        self.assertEqual(error.committed_state.version, 1)
        self.assertIsNone(error.operation_error)
        self.assertIs(error.transaction.status, TransactionStatus.COMMITTED)
        self.assertEqual(session.snapshot().version, 1)
        self.assertEqual(worker.queue_depth, 1)
        self.assertTrue(worker.recover(error.transaction))
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_execute_exposes_failed_operation_when_cleanup_stays_retained(self) -> None:
        budget = Budget(self.capacity)
        worker = Worker("gpu-0", self.model, budget, queue_capacity=1)

        def fail(_: WindowTransaction) -> WindowResult:
            raise ValueError("decoder failed")

        with patch.object(
            ResourceVector,
            "__add__",
            side_effect=MemoryError("allocation failed"),
        ):
            with self.assertRaises(TransactionRetainedError) as raised:
                worker.execute(
                    session=Session("session"),
                    request=self.request("request", "session", 37),
                    window_id="window",
                    resources=self.cost,
                    execution=ImmediateFence(),
                    operation=fail,
                )

        error = raised.exception
        self.assertIsNone(error.committed_state)
        self.assertIsInstance(error.operation_error, ValueError)
        self.assertIsInstance(error.retention_error, MemoryError)
        self.assertIs(error.transaction.status, TransactionStatus.ABORTED)
        self.assertEqual(worker.queue_depth, 1)
        self.assertTrue(worker.recover(error.transaction))
        self.assertEqual(worker.queue_depth, 0)
        self.assertEqual(budget.available, self.capacity)

    def test_nested_retained_error_does_not_leak_the_outer_transaction(self) -> None:
        outer_budget = Budget(self.capacity)
        inner_budget = Budget(self.capacity)
        outer_worker = Worker("outer", self.model, outer_budget, queue_capacity=1)
        inner_worker = Worker("inner", self.model, inner_budget, queue_capacity=1)
        inner_scope = SwitchableFailFence()

        def nested(_: WindowTransaction) -> WindowResult:
            inner_worker.execute(
                session=Session("inner-session"),
                request=self.request("inner-request", "inner-session", 38),
                window_id="inner-window",
                resources=self.cost,
                execution=inner_scope,
                operation=lambda __: self.result("inner-window", "unpublished"),
            )
            raise AssertionError("the nested execution must fail")

        with self.assertRaises(TransactionRetainedError) as raised:
            outer_worker.execute(
                session=Session("outer-session"),
                request=self.request("outer-request", "outer-session", 39),
                window_id="outer-window",
                resources=self.cost,
                execution=ImmediateFence(),
                operation=nested,
            )

        self.assertEqual(outer_worker.queue_depth, 0)
        self.assertEqual(outer_budget.available, self.capacity)
        self.assertEqual(inner_worker.queue_depth, 1)
        self.assertEqual(inner_budget.in_use, self.cost)
        self.assertIs(
            raised.exception.transaction.status,
            TransactionStatus.QUARANTINED,
        )

        inner_scope.fail = False
        self.assertTrue(inner_worker.recover(raised.exception.transaction))
        self.assertEqual(inner_worker.queue_depth, 0)
        self.assertEqual(inner_budget.available, self.capacity)


if __name__ == "__main__":
    unittest.main()
