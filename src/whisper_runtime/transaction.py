"""Window transaction lifecycle."""

from __future__ import annotations

from enum import Enum
from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction
from random import Random
from threading import Condition, RLock, Thread, current_thread
from types import TracebackType
from typing import Callable, TypeVar

from .errors import (
    ModelMismatchError,
    RequestCancelledError,
    SessionMismatchError,
    TransactionExpiredError,
    TransactionStateError,
)
from .execution import (
    CompletionFence,
    ExecutionScope,
    SubmissionGate,
    _claim_execution_scope,
    _release_execution_scope,
)
from .model import ModelSnapshot
from .resources import Lease
from .state import (
    RequestState,
    Session,
    SessionState,
    WindowRecord,
    WindowResult,
    _validate_committed_through,
)


class TransactionStatus(str, Enum):
    PREPARED = "prepared"
    RUNNING = "running"
    QUIESCING = "quiescing"
    COMMITTED = "committed"
    ABORTED = "aborted"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class ExpirationAction(str, Enum):
    """Action taken by an expiration sweep."""

    NONE = "none"
    RELEASED = "released"
    STOP_REQUESTED = "stop_requested"


class _CloseOutcome(str, Enum):
    COMMIT = "commit"
    ABORT = "abort"
    CANCEL = "cancel"
    EXPIRE = "expire"


class _StopSignalResult(str, Enum):
    DELIVERED = "delivered"
    JOINED = "joined"
    ALREADY_DELIVERED = "already_delivered"
    REENTRANT = "reentrant"
    FAILED = "failed"
    CLOSED = "closed"


_CONSTRUCTION_TOKEN = object()
T = TypeVar("T")


class WindowTransaction:
    """A versioned state transition backed by an exclusive resource lease.

    A transaction has two release paths. Idle prepared work can release at once.
    Running work must first cross a backend completion fence. A failed fence
    quarantines the lease instead of returning capacity that may still be in use.
    """

    def __init__(
        self,
        *,
        session: Session,
        request: RequestState,
        model: ModelSnapshot,
        window_id: str,
        expected_version: int,
        lease: Lease,
        admission_key: str,
        expires_at: float,
        clock: Callable[[], float],
        on_close: Callable[[WindowTransaction], None] | None,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("WindowTransaction instances are created by Worker")
        if not window_id or window_id.isspace():
            raise ValueError("window_id must not be empty")
        if expected_version < 0:
            raise ValueError("expected_version must not be negative")
        if request.session_id != session.session_id:
            raise SessionMismatchError(
                "the request session does not match the target session"
            )
        if request.model != model:
            raise ModelMismatchError(
                "the request model does not match the transaction model"
            )
        if admission_key != request.request_id:
            raise ValueError("the admission key must match the request identity")
        if lease.released:
            raise TransactionStateError("the transaction lease is already released")
        lease._require_active()

        self._session = session
        self._request = request
        self._model = model
        self._window_id = window_id
        self._expected_version = expected_version
        self._lease = lease
        self._admission_key = admission_key
        self._expires_at = expires_at
        self._clock = clock
        self._on_close = on_close
        self._status = TransactionStatus.PREPARED
        self._close_outcome: _CloseOutcome | None = None
        self._committed_result: WindowResult | None = None
        self._committed_through_ms: int | None = None
        self._committed_state: SessionState | None = None
        self._cleanup_error: BaseException | None = None
        self._stop_signal_error: BaseException | None = None
        self._closed = False
        self._activated = False
        self._abort_requested = False
        self._deadline_requested = False
        self._execution: ExecutionScope | None = None
        self._execution_claimed = False
        self._submission_gate = SubmissionGate()
        self._quiescent = False
        self._backend_finalize_in_flight = False
        self._owner_thread: Thread | None = None
        self._owner_active = False
        self._rng: Random | None = None
        self._lock = RLock()
        self._stop_condition = Condition(self._lock)
        self._stop_signal_in_flight = False
        self._stop_signal_delivered = False
        self._stop_signal_owner: Thread | None = None
        self._stop_signals_closed = False

    @classmethod
    def _create(
        cls,
        *,
        session: Session,
        request: RequestState,
        model: ModelSnapshot,
        window_id: str,
        expected_version: int,
        lease: Lease,
        admission_key: str,
        expires_at: float,
        clock: Callable[[], float],
        on_close: Callable[[WindowTransaction], None] | None,
    ) -> WindowTransaction:
        """Create an unexposed transaction for worker admission."""

        return cls(
            session=session,
            request=request,
            model=model,
            window_id=window_id,
            expected_version=expected_version,
            lease=lease,
            admission_key=admission_key,
            expires_at=expires_at,
            clock=clock,
            on_close=on_close,
            _token=_CONSTRUCTION_TOKEN,
        )

    def _activate(self) -> None:
        """Bind the request after the worker installs its queue entry."""

        with self._lock:
            if self._activated:
                raise TransactionStateError("the transaction is already active")
            random_state = self._request._activate(self._cancel_from_request)
            self._activated = True
            try:
                generator = Random()
                generator.setstate(random_state)
                self._rng = generator
            except BaseException:
                self._request._abort_active()
                raise

    @property
    def request_id(self) -> str:
        return self._admission_key

    @property
    def expected_version(self) -> int:
        return self._expected_version

    @property
    def expires_at(self) -> float:
        return self._expires_at

    @property
    def cleanup_error(self) -> BaseException | None:
        """Return a cleanup error without changing a decided outcome."""

        with self._lock:
            return self._cleanup_error

    @property
    def capacity_released(self) -> bool:
        """Return whether cleanup and lease retirement completed."""

        with self._lock:
            return self._closed

    @property
    def stop_signal_error(self) -> BaseException | None:
        """Return the last non-fatal backend stop-signal error."""

        with self._lock:
            return self._stop_signal_error

    @property
    def status(self) -> TransactionStatus:
        with self._lock:
            return self._status

    def start(self, execution: ExecutionScope) -> None:
        """Bind a backend execution scope and enter the running state."""

        with self._lock:
            if self._status is not TransactionStatus.PREPARED:
                raise TransactionStateError(
                    f"a {self._status.value} transaction cannot start"
                )
            if self._clock() >= self._expires_at:
                self._request._expire_active()
                self._status = TransactionStatus.EXPIRED
                self._finish_locked()
                raise TransactionExpiredError(
                    f"request {self._admission_key!r} expired before execution"
                )
            if not isinstance(execution, ExecutionScope):
                raise TypeError("execution must implement ExecutionScope")
            owner = current_thread()
            if type(owner).__name__ == "_DummyThread":
                raise TransactionStateError(
                    "execution must start on a threading-managed thread"
                )
            _claim_execution_scope(execution, self)
            self._execution_claimed = True
            try:
                self._request._start_prepared()
                self._submission_gate.open()
                self._execution = execution
            except BaseException:
                _release_execution_scope(execution, self)
                self._execution_claimed = False
                raise
            self._owner_thread = owner
            self._owner_active = True
            self._status = TransactionStatus.RUNNING

    def submit(self, operation: Callable[[], T]) -> T:
        """Register one backend submission through the transaction gate.

        The callback must return only after the backend operation belongs to the
        execution scope. Calls admitted before close are drained. Later calls are
        rejected.
        """

        if not callable(operation):
            raise TypeError("operation must be callable")
        if (
            iscoroutinefunction(operation)
            or isgeneratorfunction(operation)
            or isasyncgenfunction(operation)
        ):
            raise TypeError(
                "deferred submit callbacks require a compatible execution adapter"
            )

        def checked_operation() -> T:
            execution_to_signal: ExecutionScope | None = None
            outcome: _CloseOutcome | None = None
            with self._lock:
                self._require_running_locked()
                outcome = self._pending_stop_locked()
                if outcome is not None:
                    self._submission_gate.seal()
                    execution_to_signal = self._execution
            if outcome is not None:
                if execution_to_signal is not None:
                    self._signal_stop(execution_to_signal)
                self._raise_stop(outcome)
            return operation()

        return self._submission_gate.submit(checked_operation)

    def checkpoint(self) -> None:
        """Acknowledge a pending stop at a backend-safe boundary."""

        execution: ExecutionScope | None = None
        with self._lock:
            self._require_running_owner_locked()
            outcome = self._pending_stop_locked()
            if outcome is not None:
                execution = self._begin_quiescence_locked(outcome)
        if outcome is not None:
            assert execution is not None
            self._complete_stop(outcome, execution)
            self._raise_stop(outcome)

    def random(self) -> float:
        """Draw from the transaction-local random stream."""

        self.checkpoint()
        with self._lock:
            return self._require_running_rng_locked().random()

    def randrange(self, stop: int) -> int:
        """Draw an integer from the transaction-local random stream."""

        if stop <= 0:
            raise ValueError("stop must be positive")
        self.checkpoint()
        with self._lock:
            return self._require_running_rng_locked().randrange(stop)

    def commit(
        self,
        result: WindowResult,
        *,
        committed_through_ms: int | None = None,
    ) -> SessionState:
        """Publish one result and an optional committed-prefix boundary."""

        try:
            _validate_committed_through(committed_through_ms)
            with self._lock:
                if self._status is TransactionStatus.COMMITTED:
                    if (
                        result == self._committed_result
                        and committed_through_ms == self._committed_through_ms
                    ):
                        assert self._committed_state is not None
                        return self._committed_state
                    raise TransactionStateError(
                        "the transaction already committed a different result or "
                        "committed prefix"
                    )
            if (
                committed_through_ms is not None
                and committed_through_ms > result.end_ms
            ):
                raise ValueError("committed_through_ms cannot exceed the result end")
            with self._lock:
                self._require_running_owner_locked()
                if result.window_id != self._window_id:
                    raise ValueError(
                        f"expected window_id {self._window_id!r}; "
                        f"received {result.window_id!r}"
                    )
                stop = self._pending_stop_locked()
                close_outcome = _CloseOutcome.COMMIT if stop is None else stop
                record: WindowRecord | None = None
                random_state = None
                if stop is None:
                    self._lease._require_active()
                    generator = self._require_rng_locked()
                    record = WindowRecord(
                        request_id=self._admission_key,
                        model=self._model,
                        result=result,
                        committed_through_ms=committed_through_ms,
                    )
                    random_state = generator.getstate()
                execution = self._begin_quiescence_locked(close_outcome)
        except BaseException as primary_error:
            try:
                if self.status is TransactionStatus.RUNNING:
                    self.abort()
            except BaseException as cleanup_error:
                raise cleanup_error from primary_error
            raise

        if stop is not None:
            self._complete_stop(stop, execution)
            self._raise_stop(stop)

        self._finalize_backend(execution, request_stop=False)

        post_fence_stop: _CloseOutcome | None = None
        with self._lock:
            if self._status is not TransactionStatus.QUIESCING:
                raise TransactionStateError("the transaction is not quiescing")
            if not self._quiescent:
                raise TransactionStateError("the backend has not reached quiescence")
            if self._close_outcome is not _CloseOutcome.COMMIT:
                post_fence_stop = self._close_outcome
            elif self._clock() >= self._expires_at:
                self._close_outcome = _CloseOutcome.EXPIRE
                self._deadline_requested = True
                post_fence_stop = _CloseOutcome.EXPIRE

            if post_fence_stop is not None:
                execution_after_fence = self._execution
            else:
                execution_after_fence = None

        if post_fence_stop is not None:
            if execution_after_fence is None:
                raise TransactionStateError("quiescing work has no execution scope")
            self._complete_stop(post_fence_stop, execution_after_fence)
            self._raise_stop(post_fence_stop)

        with self._lock:
            self._require_quiescing_locked(_CloseOutcome.COMMIT)
            assert record is not None
            assert random_state is not None
            try:
                self._lease._require_active()
                committed_state = self._request._commit_running(
                    lambda: self._session._commit(self._expected_version, record),
                    random_state,
                )
            except BaseException:
                self._request._abort_active()
                self._status = TransactionStatus.ABORTED
                self._finish_locked()
                raise

            self._committed_result = result
            self._committed_through_ms = committed_through_ms
            self._committed_state = committed_state
            self._status = TransactionStatus.COMMITTED
            self._finish_locked()
            return committed_state

    def abort(self) -> bool:
        """Abort idle work or request a stop from outside its execution owner."""

        execution_to_signal: ExecutionScope | None = None
        owner_execution: ExecutionScope | None = None
        owner_outcome = _CloseOutcome.ABORT
        changed = False
        with self._lock:
            if self._status in (TransactionStatus.ABORTED, TransactionStatus.EXPIRED):
                return False
            if self._status is TransactionStatus.COMMITTED:
                raise TransactionStateError("a committed transaction cannot abort")
            if self._status is TransactionStatus.QUIESCING:
                if self._close_outcome is not _CloseOutcome.COMMIT:
                    return False
                self._close_outcome = _CloseOutcome.ABORT
                self._abort_requested = True
                execution_to_signal = self._execution
                changed = True
            elif self._status is TransactionStatus.QUARANTINED:
                return False
            if self._status is TransactionStatus.PREPARED:
                if self._activated:
                    self._request._abort_active()
                self._status = TransactionStatus.ABORTED
                self._finish_locked()
                return True
            elif (
                self._status is TransactionStatus.RUNNING
                and self._owner_thread is current_thread()
            ):
                pending_outcome = self._pending_stop_locked()
                if pending_outcome is None:
                    pending_outcome = self._latch_stop_locked(_CloseOutcome.ABORT)
                owner_outcome = pending_outcome
                if owner_outcome is _CloseOutcome.ABORT:
                    changed = not self._abort_requested
                    self._abort_requested = True
                if self._submission_gate.inside_submit_callback():
                    self._submission_gate.seal()
                    execution_to_signal = self._execution
                else:
                    owner_execution = self._begin_quiescence_locked(owner_outcome)
            elif self._status is TransactionStatus.RUNNING:
                latched = self._pending_stop_locked()
                if latched is None:
                    latched = self._latch_stop_locked(_CloseOutcome.ABORT)
                if latched is _CloseOutcome.ABORT:
                    changed = not self._abort_requested
                    self._abort_requested = True
                self._submission_gate.seal()
                execution_to_signal = self._execution

        if owner_execution is not None:
            self._complete_stop(owner_outcome, owner_execution)
            return True
        if execution_to_signal is None:
            return changed
        signal = self._signal_stop(execution_to_signal)
        return changed or signal is _StopSignalResult.DELIVERED

    def cancel(self) -> bool:
        """Cancel idle work or request a cooperative stop for running work."""

        execution_to_signal: ExecutionScope | None = None
        with self._lock:
            if self._status not in (
                TransactionStatus.PREPARED,
                TransactionStatus.RUNNING,
                TransactionStatus.QUIESCING,
                TransactionStatus.QUARANTINED,
            ):
                return False
            if self._status is TransactionStatus.PREPARED:
                if not self._request._cancel_active():
                    return False
                self._close_outcome = _CloseOutcome.CANCEL
                self._status = TransactionStatus.ABORTED
                self._finish_locked()
                return True
            if self._status in (
                TransactionStatus.QUIESCING,
                TransactionStatus.QUARANTINED,
            ):
                if self._close_outcome is not _CloseOutcome.COMMIT:
                    return False
                changed = self._request._cancel_active()
                if not changed:
                    return False
                self._close_outcome = _CloseOutcome.CANCEL
                self._submission_gate.seal()
                execution_to_signal = self._execution
            else:
                existing = self._pending_stop_locked()
                if existing is None or existing is _CloseOutcome.CANCEL:
                    changed = self._request._cancel_active()
                    if changed:
                        self._latch_stop_locked(_CloseOutcome.CANCEL)
                else:
                    changed = False
                if not changed and self._pending_stop_locked() is None:
                    return False
                self._submission_gate.seal()
                execution_to_signal = self._execution

        if execution_to_signal is None:
            return changed
        signal = self._signal_stop(execution_to_signal)
        return changed or signal is _StopSignalResult.DELIVERED

    def expire(self, now: float) -> ExpirationAction:
        """Expire idle work or request a cooperative stop for running work."""

        execution_to_signal: ExecutionScope | None = None
        with self._lock:
            if now < self._expires_at:
                return ExpirationAction.NONE
            if self._status is TransactionStatus.PREPARED:
                self._request._expire_active()
                self._status = TransactionStatus.EXPIRED
                self._finish_locked()
                return ExpirationAction.RELEASED
            if self._status is TransactionStatus.RUNNING:
                latched = self._latch_stop_locked(_CloseOutcome.EXPIRE)
                changed = (
                    latched is _CloseOutcome.EXPIRE and not self._deadline_requested
                )
                if latched is _CloseOutcome.EXPIRE:
                    self._deadline_requested = True
                self._submission_gate.seal()
                execution_to_signal = self._execution
            elif (
                self._status
                in (
                    TransactionStatus.QUIESCING,
                    TransactionStatus.QUARANTINED,
                )
                and self._close_outcome is _CloseOutcome.COMMIT
            ):
                self._close_outcome = _CloseOutcome.EXPIRE
                self._deadline_requested = True
                self._submission_gate.seal()
                execution_to_signal = self._execution
                changed = True
            else:
                return ExpirationAction.NONE

        signal = (
            self._signal_stop(execution_to_signal)
            if execution_to_signal is not None
            else _StopSignalResult.ALREADY_DELIVERED
        )
        if changed or signal is _StopSignalResult.DELIVERED:
            return ExpirationAction.STOP_REQUESTED
        return ExpirationAction.NONE

    def recover_quarantine(self) -> bool:
        """Retry backend quiescence, then abort and release a quarantined lease."""

        with self._lock:
            if self._status is not TransactionStatus.QUARANTINED:
                return False
            if (
                self._owner_is_live_locked()
                and self._owner_thread is not current_thread()
            ):
                return False
            self._submission_gate.require_close_safe()
            if self._execution is None:
                raise TransactionStateError("quarantined work has no execution scope")
            execution = self._execution
            outcome = self._close_outcome
            if outcome is None or outcome is _CloseOutcome.COMMIT:
                outcome = _CloseOutcome.ABORT
            self._status = TransactionStatus.QUIESCING
            self._close_outcome = outcome
            self._cleanup_error = None
            self._quiescent = False

        self._complete_stop(outcome, execution)
        return True

    def _retry_cleanup(self) -> bool:
        """Retry release and retirement without changing the terminal outcome."""

        with self._lock:
            if self._cleanup_error is None or self._closed:
                return False
            if self._status not in (
                TransactionStatus.COMMITTED,
                TransactionStatus.ABORTED,
                TransactionStatus.EXPIRED,
            ):
                return False
            self._cleanup_error = None
            self._finish_locked()
            return self._cleanup_error is None and self._closed

    def _supervisor_stop(self) -> bool:
        """Fence and abort running work without relying on its original thread."""

        recover = False
        execution: ExecutionScope | None = None
        signal_only = False
        outcome = _CloseOutcome.ABORT
        changed = False
        with self._lock:
            if self._status is TransactionStatus.PREPARED:
                if self._activated:
                    self._request._abort_active()
                self._status = TransactionStatus.ABORTED
                self._finish_locked()
                return True
            if self._status is TransactionStatus.QUARANTINED:
                recover = True
            elif self._status is TransactionStatus.RUNNING:
                if self._execution is None:
                    raise TransactionStateError("running work has no execution scope")
                pending = self._pending_stop_locked()
                if pending is None:
                    self._abort_requested = True
                    pending = self._latch_stop_locked(_CloseOutcome.ABORT)
                    changed = True
                outcome = pending
                self._submission_gate.seal()
                execution = self._execution
                if (
                    self._owner_is_live_locked()
                    and self._owner_thread is not current_thread()
                ):
                    signal_only = True
                else:
                    self._submission_gate.require_close_safe()
                    self._status = TransactionStatus.QUIESCING
                    self._close_outcome = outcome
            elif self._status is TransactionStatus.QUIESCING:
                if self._close_outcome is _CloseOutcome.COMMIT:
                    self._abort_requested = True
                    self._close_outcome = _CloseOutcome.ABORT
                    changed = True
                if self._backend_finalize_in_flight or (
                    self._owner_is_live_locked()
                    and self._owner_thread is not current_thread()
                ):
                    execution = self._execution
                    signal_only = True
                else:
                    self._submission_gate.require_close_safe()
                    if self._execution is None:
                        raise TransactionStateError(
                            "quiescing work has no execution scope"
                        )
                    execution = self._execution
                    outcome = self._close_outcome or _CloseOutcome.ABORT
                    if outcome is _CloseOutcome.COMMIT:
                        outcome = _CloseOutcome.ABORT
                    self._close_outcome = outcome
            else:
                return False

        if recover:
            return self.recover_quarantine()

        if signal_only:
            if execution is None:
                return changed
            signal = self._signal_stop(execution)
            return changed or signal is _StopSignalResult.DELIVERED
        assert execution is not None
        self._complete_stop(outcome, execution)
        return True

    def _cancel_from_request(self) -> bool:
        return self.cancel()

    def _pending_stop_locked(self) -> _CloseOutcome | None:
        if self._close_outcome in (
            _CloseOutcome.ABORT,
            _CloseOutcome.CANCEL,
            _CloseOutcome.EXPIRE,
        ):
            return self._close_outcome
        if self._request.cancelled:
            return self._latch_stop_locked(_CloseOutcome.CANCEL)
        if self._abort_requested:
            return self._latch_stop_locked(_CloseOutcome.ABORT)
        if self._deadline_requested or self._clock() >= self._expires_at:
            return self._latch_stop_locked(_CloseOutcome.EXPIRE)
        return None

    def _latch_stop_locked(self, outcome: _CloseOutcome) -> _CloseOutcome:
        if self._close_outcome is None:
            self._close_outcome = outcome
        return self._close_outcome

    def _complete_stop(
        self,
        outcome: _CloseOutcome,
        execution: ExecutionScope,
    ) -> None:
        with self._lock:
            already_quiescent = self._quiescent
        if not already_quiescent:
            self._finalize_backend(execution, request_stop=True)

        with self._lock:
            self._require_quiescing_locked(outcome)
            if outcome is _CloseOutcome.EXPIRE:
                self._request._expire_active()
                self._status = TransactionStatus.EXPIRED
            else:
                self._request._abort_active()
                self._status = TransactionStatus.ABORTED
            self._finish_locked()

    def _raise_stop(self, outcome: _CloseOutcome) -> None:
        if outcome is _CloseOutcome.CANCEL:
            raise RequestCancelledError(f"request {self._admission_key!r} is cancelled")
        if outcome is _CloseOutcome.EXPIRE:
            raise TransactionExpiredError(
                f"request {self._admission_key!r} reached its deadline"
            )
        if outcome is _CloseOutcome.ABORT:
            raise TransactionStateError(
                f"request {self._admission_key!r} received an abort request"
            )
        raise AssertionError("commit is finalized by commit()")

    def _begin_quiescence_locked(
        self,
        outcome: _CloseOutcome,
    ) -> ExecutionScope:
        self._require_running_owner_locked()
        self._submission_gate.require_close_safe()
        if self._execution is None:
            raise TransactionStateError("running work has no execution scope")
        self._submission_gate.seal()
        self._status = TransactionStatus.QUIESCING
        self._close_outcome = outcome
        return self._execution

    def _finalize_backend(
        self,
        execution: ExecutionScope,
        *,
        request_stop: bool,
    ) -> None:
        with self._lock:
            if self._backend_finalize_in_flight:
                raise TransactionStateError(
                    "backend finalization is already in progress"
                )
            self._backend_finalize_in_flight = True
        try:
            self._submission_gate.require_close_safe()
            self._submission_gate.drain()
            if request_stop:
                self._ensure_stop_signal(execution)
            else:
                self._wait_for_stop_signal()
            fence = execution.completion_fence()
            if not isinstance(fence, CompletionFence):
                raise TypeError(
                    "execution completion_fence() must return CompletionFence"
                )
            fence.wait()
            self._close_stop_signals()
            self._submission_gate.close()
        except BaseException as exc:
            with self._lock:
                if self._status is TransactionStatus.QUIESCING:
                    self._status = TransactionStatus.QUARANTINED
                    self._cleanup_error = exc
            raise
        else:
            with self._lock:
                self._quiescent = True
        finally:
            with self._lock:
                self._backend_finalize_in_flight = False

    def _signal_stop(self, execution: ExecutionScope) -> _StopSignalResult:
        """Join or run one stop attempt without holding the transaction lock."""

        with self._stop_condition:
            joined = False
            while self._stop_signal_in_flight:
                if self._stop_signal_owner is current_thread():
                    return _StopSignalResult.REENTRANT
                joined = True
                self._stop_condition.wait()
            if self._stop_signals_closed:
                return _StopSignalResult.CLOSED
            if self._stop_signal_delivered:
                if joined:
                    return _StopSignalResult.JOINED
                return _StopSignalResult.ALREADY_DELIVERED
            self._stop_signal_in_flight = True
            self._stop_signal_owner = current_thread()

        try:
            execution.request_stop()
        except BaseException as exc:
            with self._stop_condition:
                self._stop_signal_error = exc
                self._stop_signal_in_flight = False
                self._stop_signal_owner = None
                self._stop_condition.notify_all()
            return _StopSignalResult.FAILED
        else:
            with self._stop_condition:
                self._stop_signal_error = None
                self._stop_signal_delivered = True
                self._stop_signal_in_flight = False
                self._stop_signal_owner = None
                self._stop_condition.notify_all()
            return _StopSignalResult.DELIVERED

    def _ensure_stop_signal(self, execution: ExecutionScope) -> None:
        """Wait for, or retry, the single stop signal before final fencing."""

        signal = self._signal_stop(execution)
        if signal is not _StopSignalResult.FAILED:
            return
        with self._lock:
            error = self._stop_signal_error
        if error is None:
            raise TransactionStateError("the backend stop signal failed")
        raise error

    def _wait_for_stop_signal(self) -> None:
        """Prevent a previously claimed stop call from outliving close."""

        with self._stop_condition:
            while self._stop_signal_in_flight:
                self._stop_condition.wait()

    def _close_stop_signals(self) -> None:
        """Join earlier stop calls and reject calls after the final fence."""

        with self._stop_condition:
            while self._stop_signal_in_flight:
                self._stop_condition.wait()
            self._stop_signals_closed = True

    def _require_running_locked(self) -> None:
        if self._status is not TransactionStatus.RUNNING:
            raise TransactionStateError(
                f"a {self._status.value} transaction cannot submit backend work"
            )

    def _require_running_owner_locked(self) -> None:
        if self._status is not TransactionStatus.RUNNING:
            raise TransactionStateError(
                f"a {self._status.value} transaction is not running"
            )
        if self._owner_thread is not current_thread() or not self._owner_active:
            raise TransactionStateError(
                "only the execution owner can cross a running safe boundary"
            )

    def _owner_is_live_locked(self) -> bool:
        return bool(
            self._owner_active
            and self._owner_thread is not None
            and self._owner_thread.is_alive()
        )

    def _owner_departed(self) -> None:
        """Release task ownership without treating a pool thread as live work."""

        with self._lock:
            if self._owner_thread is current_thread():
                self._owner_active = False

    def _require_quiescing_locked(self, outcome: _CloseOutcome) -> None:
        if self._status is not TransactionStatus.QUIESCING:
            raise TransactionStateError("the transaction is not quiescing")
        if self._close_outcome is not outcome:
            raise TransactionStateError("the transaction close outcome changed")
        if not self._quiescent:
            raise TransactionStateError("the backend has not reached quiescence")

    def _require_running_rng_locked(self) -> Random:
        self._require_running_owner_locked()
        return self._require_rng_locked()

    def _require_rng_locked(self) -> Random:
        if not self._activated or self._rng is None:
            raise TransactionStateError("the transaction random stream is unavailable")
        return self._rng

    def _finish_locked(self) -> None:
        if self._closed:
            return
        self._owner_active = False
        try:
            self._submission_gate.close()
            self._lease.release()
        except BaseException as exc:
            self._cleanup_error = exc
            return

        if self._execution_claimed:
            if self._execution is None:
                self._cleanup_error = TransactionStateError(
                    "the claimed execution scope is unavailable"
                )
                return
            try:
                _release_execution_scope(self._execution, self)
            except BaseException as exc:
                self._cleanup_error = exc
                return
            self._execution_claimed = False
        if self._on_close is not None:
            try:
                self._on_close(self)
            except BaseException as exc:
                self._cleanup_error = exc
                return
        self._closed = True

    def __enter__(self) -> WindowTransaction:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.status in (TransactionStatus.PREPARED, TransactionStatus.RUNNING):
            self.abort()
