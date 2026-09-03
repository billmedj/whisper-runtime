"""Backend execution boundaries for safe resource release."""

from __future__ import annotations

from enum import Enum
from inspect import (
    isasyncgen,
    isasyncgenfunction,
    isawaitable,
    iscoroutinefunction,
    isgenerator,
    isgeneratorfunction,
)
from threading import Condition, RLock, local
from typing import Callable, Protocol, TypeVar, runtime_checkable

from .errors import TransactionStateError

T = TypeVar("T")
_SCOPE_CLAIMS: dict[int, tuple[object, object]] = {}
_SCOPE_CLAIMS_LOCK = RLock()


def _claim_execution_scope(scope: object, owner: object) -> None:
    """Bind one scope object to one live transaction by identity."""

    key = id(scope)
    with _SCOPE_CLAIMS_LOCK:
        existing = _SCOPE_CLAIMS.get(key)
        if existing is not None:
            existing_scope, _ = existing
            if existing_scope is scope:
                raise TransactionStateError(
                    "an execution scope cannot serve two live transactions"
                )
            raise TransactionStateError("execution scope identity collision")
        _SCOPE_CLAIMS[key] = (scope, owner)


def _release_execution_scope(scope: object, owner: object) -> None:
    """Release the exact scope claim held by one terminal transaction."""

    key = id(scope)
    with _SCOPE_CLAIMS_LOCK:
        existing = _SCOPE_CLAIMS.get(key)
        if existing is None or existing[0] is not scope or existing[1] is not owner:
            raise TransactionStateError("the execution scope claim does not match")
        del _SCOPE_CLAIMS[key]


@runtime_checkable
class CompletionFence(Protocol):
    """Prove that submitted backend work no longer uses a transaction lease.

    A CUDA adapter can synchronize a recorded event. A remote backend can wait
    for its cancellation or completion acknowledgement. Returning from ``wait``
    is the authority that permits the runtime to release compute resources.
    """

    def wait(self) -> None:
        """Block until all work covered by this fence is quiescent."""


@runtime_checkable
class ExecutionScope(Protocol):
    """Own backend work submitted for one transaction.

    ``completion_fence`` is called only after the transaction submission gate is
    sealed and all admitted host-side submissions have returned. The returned
    fence must cover every backend operation registered before that call.
    ``request_stop`` must be idempotent because recovery can retry it.
    """

    def request_stop(self) -> None:
        """Ask all work in this scope to stop without waiting for completion."""

    def completion_fence(self) -> CompletionFence:
        """Create an authoritative fence for all registered scope work."""


class _GateState(str, Enum):
    NEW = "new"
    OPEN = "open"
    SEALED = "sealed"
    CLOSED = "closed"


class SubmissionGate:
    """Linearize backend submission against transaction close.

    A submission accepted while the gate is open is counted until its callback
    returns. The callback must not return before the backend has registered the
    submitted operation in the associated ``ExecutionScope``. Sealing is
    non-blocking. Draining waits only for callbacks admitted before the seal.
    """

    __slots__ = ("_active_submitters", "_condition", "_local", "_state")

    def __init__(self) -> None:
        self._active_submitters = 0
        self._condition = Condition(RLock())
        self._local = local()
        self._state = _GateState.NEW

    def open(self) -> None:
        with self._condition:
            if self._state is not _GateState.NEW:
                raise TransactionStateError("the submission gate cannot open")
            self._state = _GateState.OPEN

    def submit(self, operation: Callable[[], T]) -> T:
        """Run one backend-registration callback while holding admission."""

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
        with self._condition:
            if self._state is not _GateState.OPEN:
                raise TransactionStateError("backend submission is closed")
            self._active_submitters += 1
            self._local.depth = getattr(self._local, "depth", 0) + 1

        try:
            result = operation()
            if isawaitable(result) or isgenerator(result) or isasyncgen(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                else:
                    cancel = getattr(result, "cancel", None)
                    if callable(cancel):
                        cancel()
                raise TypeError(
                    "submit callbacks must finish backend registration before return"
                )
            return result
        finally:
            with self._condition:
                self._local.depth -= 1
                self._active_submitters -= 1
                if self._active_submitters == 0:
                    self._condition.notify_all()

    def require_close_safe(self) -> None:
        """Reject a close that would wait for the calling submit callback."""

        if self.inside_submit_callback():
            raise TransactionStateError(
                "a backend submit callback cannot close its own transaction"
            )

    def inside_submit_callback(self) -> bool:
        """Return whether the caller owns an active submission token."""

        return bool(getattr(self._local, "depth", 0))

    def seal(self) -> None:
        """Prevent new submissions without waiting for admitted callbacks."""

        with self._condition:
            if self._state is _GateState.OPEN:
                self._state = _GateState.SEALED
                return
            if self._state is _GateState.SEALED:
                return
            raise TransactionStateError(
                f"a {self._state.value} submission gate cannot seal"
            )

    def drain(self) -> None:
        """Wait until all callbacks admitted before sealing have returned."""

        with self._condition:
            if self._state is _GateState.CLOSED:
                return
            if self._state is not _GateState.SEALED:
                raise TransactionStateError("the submission gate is not sealed")
            while self._active_submitters:
                self._condition.wait()

    def close(self) -> None:
        """Make a drained gate terminal."""

        with self._condition:
            if self._state is _GateState.CLOSED:
                return
            if self._state is _GateState.NEW:
                self._state = _GateState.CLOSED
                return
            if self._state is not _GateState.SEALED:
                raise TransactionStateError("the submission gate is still open")
            if self._active_submitters:
                raise TransactionStateError("the submission gate is not drained")
            self._state = _GateState.CLOSED


class ImmediateFence:
    """Execution scope for work completed synchronously by its submit callback."""

    __slots__ = ()

    def request_stop(self) -> None:
        return None

    def completion_fence(self) -> CompletionFence:
        return self

    def wait(self) -> None:
        return None
