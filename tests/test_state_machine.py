"""Deterministic adversarial traces for the runtime state machine."""

import random
import unittest

from whisper_runtime import (
    Budget,
    ImmediateFence,
    ModelSnapshot,
    QueueFullError,
    RequestCancelledError,
    RequestState,
    ResourceUnavailableError,
    ResourceVector,
    Session,
    StaleSessionError,
    TransactionExpiredError,
    TransactionStateError,
    TransactionStatus,
    WindowResult,
    WindowTransaction,
    Worker,
)


class RuntimeStateMachineTests(unittest.TestCase):
    def test_randomized_lifecycle_preserves_capacity_and_queue_bounds(self) -> None:
        generator = random.Random(0xACC0_4D10)
        model = ModelSnapshot(
            model_id="whisper-test",
            revision="1",
            backend="reference",
            fingerprint="sha256:test",
        )
        capacity = ResourceVector(
            memory_bytes=1_000,
            compute_units=8,
            stream_slots=5,
        )
        budget = Budget(capacity)
        now = [0.0]
        worker = Worker(
            "worker",
            model,
            budget,
            queue_capacity=5,
            transaction_ttl_seconds=10.0,
            clock=lambda: now[0],
        )
        sessions = [Session(f"session-{index}", history_limit=4) for index in range(3)]
        transactions: list[tuple[WindowTransaction, RequestState, str]] = []
        next_request = 0

        for _ in range(2_000):
            operation = generator.randrange(6)
            active = [
                entry
                for entry in transactions
                if entry[0].status
                in (TransactionStatus.PREPARED, TransactionStatus.RUNNING)
            ]

            if operation == 0 or not active:
                session = generator.choice(sessions)
                request_id = f"request-{next_request}"
                next_request += 1
                request = RequestState(
                    request_id=request_id,
                    session_id=session.session_id,
                    model=model,
                    rng_seed=next_request,
                )
                window_id = f"window-{next_request}"
                resources = ResourceVector(
                    memory_bytes=generator.choice((100, 200, 400)),
                    compute_units=generator.choice((1, 2, 4)),
                    stream_slots=1,
                )
                try:
                    transaction = worker.prepare(
                        session=session,
                        request=request,
                        window_id=window_id,
                        resources=resources,
                    )
                except (QueueFullError, ResourceUnavailableError):
                    pass
                else:
                    transactions.append((transaction, request, window_id))
                    if generator.randrange(2):
                        transaction.start(ImmediateFence())
            elif operation == 1:
                runnable = [
                    entry
                    for entry in active
                    if entry[0].status is TransactionStatus.RUNNING
                ]
                if runnable:
                    transaction, _, window_id = generator.choice(runnable)
                    try:
                        transaction.commit(
                            WindowResult(
                                window_id=window_id,
                                text="committed",
                                start_ms=0,
                                end_ms=1,
                            )
                        )
                    except (
                        RequestCancelledError,
                        StaleSessionError,
                        TransactionExpiredError,
                        TransactionStateError,
                    ):
                        pass
                else:
                    transaction, _, _ = generator.choice(active)
                    transaction.start(ImmediateFence())
            elif operation == 2:
                transaction, _, _ = generator.choice(active)
                transaction.abort()
            elif operation == 3:
                transaction, _, _ = generator.choice(active)
                worker.cancel(transaction)
            elif operation == 4:
                _, request, _ = generator.choice(active)
                request.cancel()
            else:
                now[0] += generator.random() * 12.0
                worker.reap_expired()

            self.assertTrue(budget.available.fits_within(capacity))
            self.assertEqual(budget.in_use + budget.available, capacity)
            self.assertEqual(budget.lease_count, worker.queue_depth)
            self.assertLessEqual(worker.queue_depth, worker.queue_capacity)
            for session in sessions:
                self.assertLessEqual(
                    len(session.snapshot().windows), session.history_limit
                )

        for transaction, _, _ in transactions:
            if transaction.status in (
                TransactionStatus.PREPARED,
                TransactionStatus.RUNNING,
            ):
                transaction.abort()
        self.assertEqual(budget.available, capacity)
        self.assertEqual(budget.lease_count, 0)
        self.assertEqual(worker.queue_depth, 0)


if __name__ == "__main__":
    unittest.main()
