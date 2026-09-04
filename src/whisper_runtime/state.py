"""Immutable session values and request-local mutable state."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Callable

from .errors import (
    RequestCancelledError,
    RequestTerminalError,
    StaleSessionError,
)
from .model import ModelSnapshot

RandomState = tuple[int, tuple[int, ...], float | None]


@dataclass(frozen=True, slots=True)
class WindowResult:
    """The committed output for one audio window."""

    window_id: str
    text: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.window_id or self.window_id.isspace():
            raise ValueError("window_id must not be empty")
        if isinstance(self.start_ms, bool) or not isinstance(self.start_ms, int):
            raise TypeError("start_ms must be an integer")
        if isinstance(self.end_ms, bool) or not isinstance(self.end_ms, int):
            raise TypeError("end_ms must be an integer")
        if self.start_ms < 0:
            raise ValueError("start_ms must not be negative")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must not precede start_ms")


@dataclass(frozen=True, slots=True)
class WindowRecord:
    """A result and optional committed prefix bound to one request."""

    request_id: str
    model: ModelSnapshot
    result: WindowResult
    committed_through_ms: int | None = None

    def __post_init__(self) -> None:
        _validate_committed_through(self.committed_through_ms)


@dataclass(frozen=True, slots=True)
class SessionState:
    """An immutable, versioned snapshot with a committed audio prefix."""

    session_id: str
    version: int = 0
    windows: tuple[WindowRecord, ...] = ()
    committed_through_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.session_id or self.session_id.isspace():
            raise ValueError("session_id must not be empty")
        if self.version < 0:
            raise ValueError("version must not be negative")
        _validate_committed_through(self.committed_through_ms)


class Session:
    """A compare-and-swap holder for ``SessionState``."""

    def __init__(self, session_id: str, *, history_limit: int = 1_024) -> None:
        if isinstance(history_limit, bool) or not isinstance(history_limit, int):
            raise TypeError("history_limit must be an integer")
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self._state = SessionState(session_id=session_id)
        self._history_limit = history_limit
        self._lock = RLock()

    @property
    def session_id(self) -> str:
        return self._state.session_id

    @property
    def history_limit(self) -> int:
        """Return the maximum number of committed windows retained in memory."""

        return self._history_limit

    def snapshot(self) -> SessionState:
        with self._lock:
            return self._state

    def _commit(self, expected_version: int, record: WindowRecord) -> SessionState:
        with self._lock:
            current = self._state
            if current.version != expected_version:
                raise StaleSessionError(
                    f"expected session version {expected_version}; "
                    f"current version is {current.version}"
                )

            committed_through = current.committed_through_ms
            if (
                committed_through is not None
                and record.result.start_ms < committed_through
            ):
                raise ValueError("a result cannot overlap committed audio")

            if record.committed_through_ms is not None:
                if (
                    committed_through is not None
                    and record.committed_through_ms < committed_through
                ):
                    raise ValueError("committed_through_ms must not regress")
                if record.committed_through_ms > record.result.end_ms:
                    raise ValueError(
                        "committed_through_ms cannot exceed the result end"
                    )
                committed_through = record.committed_through_ms

            retained = current.windows
            if len(retained) >= self._history_limit:
                if self._history_limit == 1:
                    retained = ()
                else:
                    retained = retained[-(self._history_limit - 1) :]

            next_state = SessionState(
                session_id=current.session_id,
                version=current.version + 1,
                windows=retained + (record,),
                committed_through_ms=committed_through,
            )
            self._state = next_state
            return next_state


def _validate_committed_through(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("committed_through_ms must be an integer or None")
    if value < 0:
        raise ValueError("committed_through_ms must not be negative")


class RequestStatus(str, Enum):
    """Lifecycle of a request admitted for one window transaction."""

    CREATED = "created"
    PREPARED = "prepared"
    RUNNING = "running"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    EXPIRED = "expired"


class RequestState:
    """Request identity, lifecycle, and rollback-safe random state."""

    __slots__ = (
        "_cancel_controller",
        "_lock",
        "_model",
        "_request_id",
        "_rng",
        "_session_id",
        "_status",
    )

    def __init__(
        self,
        request_id: str,
        session_id: str,
        model: ModelSnapshot,
        *,
        rng_seed: int,
    ) -> None:
        if not request_id or request_id.isspace():
            raise ValueError("request_id must not be empty")
        if not session_id or session_id.isspace():
            raise ValueError("session_id must not be empty")
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise TypeError("rng_seed must be an integer")

        self._request_id = request_id
        self._session_id = session_id
        self._model = model
        self._rng = random.Random(rng_seed)
        self._status = RequestStatus.CREATED
        self._cancel_controller: Callable[[], bool] | None = None
        self._lock = RLock()

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def model(self) -> ModelSnapshot:
        return self._model

    @property
    def status(self) -> RequestStatus:
        with self._lock:
            return self._status

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._status is RequestStatus.CANCELLED

    def cancel(self) -> bool:
        """Mark the request as cancelled once."""

        controller: Callable[[], bool] | None
        with self._lock:
            if self._status is RequestStatus.CREATED:
                self._status = RequestStatus.CANCELLED
                return True
            if self._status not in (RequestStatus.PREPARED, RequestStatus.RUNNING):
                return False
            controller = self._cancel_controller

        # The transaction owns cancellation after admission. Do not hold the
        # request lock while entering it; commit uses the inverse lock order.
        if controller is None:
            return False
        return controller()

    def require_active(self) -> None:
        with self._lock:
            self._require_nonterminal_locked()

    def _activate(self, controller: Callable[[], bool]) -> RandomState:
        """Bind cancellation to one transaction and snapshot random state."""

        with self._lock:
            if self._status is not RequestStatus.CREATED:
                self._require_nonterminal_locked()
                raise RequestTerminalError(
                    f"request {self._request_id!r} is already prepared"
                )
            self._status = RequestStatus.PREPARED
            self._cancel_controller = controller
            return self._rng.getstate()

    def _start_prepared(self) -> None:
        with self._lock:
            if self._status is not RequestStatus.PREPARED:
                self._require_nonterminal_locked()
                raise RequestTerminalError(
                    f"request {self._request_id!r} is not prepared"
                )
            self._status = RequestStatus.RUNNING

    def _commit_running(
        self,
        action: Callable[[], SessionState],
        random_state: RandomState,
    ) -> SessionState:
        """Commit state and RNG under the same cancellation guard."""

        with self._lock:
            if self._status is not RequestStatus.RUNNING:
                self._require_nonterminal_locked()
                raise RequestTerminalError(
                    f"request {self._request_id!r} is not running"
                )
            # Prepare the next generator before the session compare-and-swap.
            # Publishing it afterwards is a non-fallible reference assignment.
            next_generator = random.Random()
            next_generator.setstate(random_state)
            committed = action()
            self._rng = next_generator
            self._status = RequestStatus.COMMITTED
            self._cancel_controller = None
            return committed

    def _abort_active(self) -> bool:
        with self._lock:
            if self._status not in (RequestStatus.PREPARED, RequestStatus.RUNNING):
                return False
            self._status = RequestStatus.ABORTED
            self._cancel_controller = None
            return True

    def _cancel_active(self) -> bool:
        with self._lock:
            if self._status not in (RequestStatus.PREPARED, RequestStatus.RUNNING):
                return False
            self._status = RequestStatus.CANCELLED
            self._cancel_controller = None
            return True

    def _expire_active(self) -> bool:
        with self._lock:
            if self._status not in (RequestStatus.PREPARED, RequestStatus.RUNNING):
                return False
            self._status = RequestStatus.EXPIRED
            self._cancel_controller = None
            return True

    def _require_nonterminal_locked(self) -> None:
        if self._status is RequestStatus.CANCELLED:
            raise RequestCancelledError(f"request {self._request_id!r} is cancelled")
        if self._status in (
            RequestStatus.COMMITTED,
            RequestStatus.ABORTED,
            RequestStatus.EXPIRED,
        ):
            raise RequestTerminalError(
                f"request {self._request_id!r} is {self._status.value}"
            )
