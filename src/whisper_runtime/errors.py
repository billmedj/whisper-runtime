"""Runtime error types."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import SessionState
    from .transaction import WindowTransaction


class RuntimeStateError(RuntimeError):
    """Base class for runtime state errors."""


class TransactionRetainedError(RuntimeStateError):
    """An execution ended, but its exact transaction still owns resources.

    The attached transaction is the recovery authority.  If ``committed_state``
    is present, the result was published before cleanup failed and the caller
    must not repeat the inference operation.
    """

    def __init__(
        self,
        transaction: WindowTransaction,
        *,
        operation_error: BaseException | None,
        retention_error: BaseException,
        committed_state: SessionState | None = None,
    ) -> None:
        outcome = "committed" if committed_state is not None else "did not commit"
        super().__init__(
            f"request {transaction.request_id!r} {outcome}, but its transaction "
            "still owns runtime capacity; recover the attached transaction"
        )
        self.transaction = transaction
        self.operation_error = operation_error
        self.retention_error = retention_error
        self.committed_state = committed_state


class ResourceUnavailableError(RuntimeStateError):
    """The worker cannot reserve the requested resources."""


class QueueFullError(RuntimeStateError):
    """The worker has reached its admission limit."""


class DuplicateRequestError(RuntimeStateError):
    """The worker already has a prepared window for the request."""


class ModelMismatchError(RuntimeStateError):
    """The request targets a different model snapshot."""


class SessionMismatchError(RuntimeStateError):
    """The request targets a different session."""


class RequestCancelledError(RuntimeStateError):
    """The request was cancelled before the operation committed."""


class RequestTerminalError(RuntimeStateError):
    """The request has already reached a terminal state."""


class StaleSessionError(RuntimeStateError):
    """The session changed after the transaction was prepared."""


class TransactionStateError(RuntimeStateError):
    """The requested transition is not valid for this transaction."""


class TransactionExpiredError(RuntimeStateError):
    """The transaction reached its deadline before it could commit."""
