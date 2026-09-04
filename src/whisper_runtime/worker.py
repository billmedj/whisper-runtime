"""Bounded worker admission."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import RLock
from typing import Callable

from .errors import (
    DuplicateRequestError,
    ModelMismatchError,
    QueueFullError,
    SessionMismatchError,
    TransactionRetainedError,
    TransactionStateError,
)
from .execution import ExecutionScope
from .model import ModelSnapshot
from .resources import Budget, ResourceVector
from .state import RequestState, Session, SessionState, WindowResult
from .transaction import ExpirationAction, TransactionStatus, WindowTransaction


@dataclass(frozen=True, slots=True)
class ReapReport:
    """Results from one expiration sweep."""

    released: int = 0
    stop_requested: int = 0


class Worker:
    """Admit window transactions within fixed queue and resource limits."""

    def __init__(
        self,
        worker_id: str,
        model: ModelSnapshot,
        budget: Budget,
        *,
        queue_capacity: int,
        transaction_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not worker_id or worker_id.isspace():
            raise ValueError("worker_id must not be empty")
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int):
            raise TypeError("queue_capacity must be an integer")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if isinstance(transaction_ttl_seconds, bool) or not isinstance(
            transaction_ttl_seconds, (int, float)
        ):
            raise TypeError("transaction_ttl_seconds must be a number")
        if not math.isfinite(transaction_ttl_seconds):
            raise ValueError("transaction_ttl_seconds must be finite")
        if transaction_ttl_seconds <= 0:
            raise ValueError("transaction_ttl_seconds must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._worker_id = worker_id
        self._model = model
        self._budget = budget
        self._queue_capacity = queue_capacity
        self._transaction_ttl_seconds = float(transaction_ttl_seconds)
        self._clock = clock
        self._transactions: dict[str, WindowTransaction] = {}
        self._lock = RLock()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def model(self) -> ModelSnapshot:
        return self._model

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def queue_capacity(self) -> int:
        return self._queue_capacity

    @property
    def transaction_ttl_seconds(self) -> float:
        return self._transaction_ttl_seconds

    @property
    def queue_depth(self) -> int:
        """Return the number of admitted transactions not yet closed."""

        with self._lock:
            return len(self._transactions)

    @property
    def quarantined_count(self) -> int:
        """Return leases retained because backend quiescence was not proven."""

        with self._lock:
            transactions = tuple(self._transactions.values())
        return sum(
            transaction.status is TransactionStatus.QUARANTINED
            or transaction.cleanup_error is not None
            for transaction in transactions
        )

    def prepare(
        self,
        *,
        session: Session,
        request: RequestState,
        window_id: str,
        resources: ResourceVector,
    ) -> WindowTransaction:
        """Atomically reserve capacity and admit one window transaction."""

        self.reap_expired()
        if request.model != self._model:
            raise ModelMismatchError("the request model does not match the worker")
        if request.session_id != session.session_id:
            raise SessionMismatchError(
                "the request session does not match the target session"
            )
        request.require_active()

        # A concurrent session commit can make this snapshot stale. The session
        # compare-and-swap remains the commit authority.
        expected_version = session.snapshot().version
        admission_key = request.request_id

        with self._lock:
            request.require_active()
            if admission_key in self._transactions:
                raise DuplicateRequestError(
                    f"request {admission_key!r} already has an admitted window"
                )
            if len(self._transactions) >= self._queue_capacity:
                raise QueueFullError(f"worker queue capacity is {self._queue_capacity}")

            lease = self._budget.acquire(resources)
            transaction: WindowTransaction | None = None
            try:
                transaction = WindowTransaction._create(
                    session=session,
                    request=request,
                    model=self._model,
                    window_id=window_id,
                    expected_version=expected_version,
                    lease=lease,
                    admission_key=admission_key,
                    expires_at=self._clock() + self._transaction_ttl_seconds,
                    clock=self._clock,
                    on_close=self._retire,
                )
                self._transactions[admission_key] = transaction
                transaction._activate()
            except BaseException:
                if transaction is None:
                    lease.release()
                else:
                    transaction.abort()
                raise
            assert transaction is not None
            return transaction

    def cancel(self, transaction: WindowTransaction) -> bool:
        """Cancel an exact transaction handle.

        A request identifier is not sufficient because identifiers can be reused
        after a terminal transaction leaves the queue.
        """

        if not isinstance(transaction, WindowTransaction):
            raise TypeError("transaction must be a WindowTransaction")
        with self._lock:
            current = self._transactions.get(transaction.request_id)
            if current is not transaction:
                return False
        return transaction.cancel()

    def recover(self, transaction: WindowTransaction) -> bool:
        """Retry quiescence for one exact quarantined transaction."""

        if not isinstance(transaction, WindowTransaction):
            raise TypeError("transaction must be a WindowTransaction")
        with self._lock:
            current = self._transactions.get(transaction.request_id)
            if current is not transaction:
                return False
        if transaction.status is TransactionStatus.QUARANTINED:
            return transaction.recover_quarantine()
        return transaction._retry_cleanup()

    def stop(self, transaction: WindowTransaction) -> bool:
        """Fence and abort an exact transaction, including orphaned work."""

        if not isinstance(transaction, WindowTransaction):
            raise TypeError("transaction must be a WindowTransaction")
        with self._lock:
            current = self._transactions.get(transaction.request_id)
            if current is not transaction:
                return False
        return transaction._supervisor_stop()

    def execute(
        self,
        *,
        session: Session,
        request: RequestState,
        window_id: str,
        resources: ResourceVector,
        execution: ExecutionScope,
        operation: Callable[[WindowTransaction], WindowResult],
    ) -> SessionState:
        """Run one operation with failure-safe transaction cleanup."""

        transaction = self.prepare(
            session=session,
            request=request,
            window_id=window_id,
            resources=resources,
        )
        committed_state: SessionState | None = None
        try:
            try:
                transaction.start(execution)
                result = operation(transaction)
                committed_state = transaction.commit(result)
            except BaseException as operation_error:
                self._finish_execution(
                    transaction,
                    operation_error=operation_error,
                    committed_state=committed_state,
                )
                raise

            self._finish_execution(
                transaction,
                operation_error=None,
                committed_state=committed_state,
            )
            return committed_state
        finally:
            transaction._owner_departed()

    def _finish_execution(
        self,
        transaction: WindowTransaction,
        *,
        operation_error: BaseException | None,
        committed_state: SessionState | None,
    ) -> None:
        """Close or recover one owner-driven execution.

        ``execute`` and adapter-managed run handles share this path so they
        cannot diverge on lease retention or backend quiescence.
        """

        close_error: BaseException | None = None
        if operation_error is not None:
            try:
                if transaction.status in (
                    TransactionStatus.PREPARED,
                    TransactionStatus.RUNNING,
                ):
                    transaction.abort()
                elif transaction.status is TransactionStatus.QUIESCING:
                    transaction._supervisor_stop()
            except BaseException as exc:
                close_error = exc

        retained, recovery_error = self._recover_retained_once(transaction)
        if not retained:
            return

        retention_error = (
            recovery_error
            or transaction.cleanup_error
            or close_error
            or TransactionStateError("the transaction remained admitted after recovery")
        )
        raise TransactionRetainedError(
            transaction,
            operation_error=operation_error,
            retention_error=retention_error,
            committed_state=committed_state,
        ) from retention_error

    def _recover_retained_once(
        self,
        transaction: WindowTransaction,
    ) -> tuple[bool, BaseException | None]:
        """Try one bounded recovery and report whether the handle remains owned."""

        if not self._owns(transaction):
            return False, None

        recovery_error: BaseException | None = None
        try:
            self.recover(transaction)
        except BaseException as exc:
            recovery_error = exc

        if not self._owns(transaction):
            return False, None
        return True, recovery_error or transaction.cleanup_error

    def _owns(self, transaction: WindowTransaction) -> bool:
        with self._lock:
            return self._transactions.get(transaction.request_id) is transaction

    def reap_expired(self) -> ReapReport:
        """Sweep deadlines without releasing capacity used by running work."""

        now = self._clock()
        with self._lock:
            transactions = tuple(self._transactions.values())

        released = 0
        stop_requested = 0
        for transaction in transactions:
            action = transaction.expire(now)
            if action is ExpirationAction.RELEASED:
                released += 1
            elif action is ExpirationAction.STOP_REQUESTED:
                stop_requested += 1
        return ReapReport(released=released, stop_requested=stop_requested)

    def _retire(self, transaction: WindowTransaction) -> None:
        with self._lock:
            current = self._transactions.get(transaction.request_id)
            if current is transaction and transaction.cleanup_error is None:
                del self._transactions[transaction.request_id]
