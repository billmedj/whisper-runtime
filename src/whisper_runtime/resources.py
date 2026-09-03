"""Resource accounting for bounded workers."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, RLock
from types import TracebackType

from .errors import ResourceUnavailableError, RuntimeStateError

_LEASE_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ResourceVector:
    """A component-wise resource quantity."""

    memory_bytes: int = 0
    compute_units: int = 0
    stream_slots: int = 0

    def __post_init__(self) -> None:
        for field_name in ("memory_bytes", "compute_units", "stream_slots"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")

    def __add__(self, other: ResourceVector) -> ResourceVector:
        if not isinstance(other, ResourceVector):
            return NotImplemented
        return ResourceVector(
            self.memory_bytes + other.memory_bytes,
            self.compute_units + other.compute_units,
            self.stream_slots + other.stream_slots,
        )

    def __sub__(self, other: ResourceVector) -> ResourceVector:
        if not isinstance(other, ResourceVector):
            return NotImplemented
        if not other.fits_within(self):
            raise ValueError("resource subtraction would produce a negative value")
        return ResourceVector(
            self.memory_bytes - other.memory_bytes,
            self.compute_units - other.compute_units,
            self.stream_slots - other.stream_slots,
        )

    def fits_within(self, other: ResourceVector) -> bool:
        """Return whether every component fits within ``other``."""

        return (
            self.memory_bytes <= other.memory_bytes
            and self.compute_units <= other.compute_units
            and self.stream_slots <= other.stream_slots
        )


class Lease:
    """An exclusive reservation from a resource budget."""

    __slots__ = ("_budget", "_lease_id", "_lock", "_released", "_resources")

    def __init__(
        self,
        budget: Budget,
        lease_id: int,
        resources: ResourceVector,
        *,
        _token: object,
    ) -> None:
        if _token is not _LEASE_CONSTRUCTION_TOKEN:
            raise TypeError("resource leases are created by Budget.acquire")
        self._budget = budget
        self._lease_id = lease_id
        self._resources = resources
        self._released = False
        self._lock = Lock()

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    @property
    def resources(self) -> ResourceVector:
        return self._resources

    def release(self) -> bool:
        """Release the reservation once.

        Returns ``True`` for the release and ``False`` after it has already run.
        """

        with self._lock:
            if self._released:
                return False
            self._budget._release(self)
            self._released = True
            return True

    def _require_active(self) -> None:
        """Reject a lease that is not the exact object in its budget ledger."""

        self._budget._require_active(self)

    def __enter__(self) -> Lease:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class Budget:
    """A thread-safe resource budget with exact lease accounting."""

    def __init__(self, capacity: ResourceVector) -> None:
        self._capacity = capacity
        self._available = capacity
        self._active: dict[int, tuple[Lease, ResourceVector]] = {}
        self._next_lease_id = 1
        self._lock = RLock()

    @property
    def capacity(self) -> ResourceVector:
        return self._capacity

    @property
    def available(self) -> ResourceVector:
        with self._lock:
            return self._available

    @property
    def in_use(self) -> ResourceVector:
        with self._lock:
            return self._capacity - self._available

    @property
    def lease_count(self) -> int:
        with self._lock:
            return len(self._active)

    def acquire(self, resources: ResourceVector) -> Lease:
        """Reserve resources or raise ``ResourceUnavailableError``."""

        with self._lock:
            if not resources.fits_within(self._available):
                raise ResourceUnavailableError(
                    f"requested {resources!r}; available {self._available!r}"
                )

            lease_id = self._next_lease_id
            # Construct every fallible value before publishing ledger changes.
            lease = Lease(
                self,
                lease_id,
                resources,
                _token=_LEASE_CONSTRUCTION_TOKEN,
            )
            next_available = self._available - resources
            next_lease_id = lease_id + 1
            self._active[lease_id] = (lease, resources)
            self._available = next_available
            self._next_lease_id = next_lease_id
            return lease

    def _require_active(self, lease: Lease) -> None:
        with self._lock:
            active = self._active.get(lease._lease_id)
            if active is None or active[0] is not lease:
                raise RuntimeStateError("the lease is not active")
            if active[1] != lease.resources:
                raise RuntimeStateError("the lease resource vector does not match")

    def _release(self, lease: Lease) -> None:
        with self._lock:
            active = self._active.get(lease._lease_id)
            if active is None or active[0] is not lease:
                raise RuntimeStateError("the lease is not active")
            active_lease, reserved = active
            if active_lease.resources != reserved:
                raise RuntimeStateError("the lease resource vector does not match")

            available = self._available + reserved
            if not available.fits_within(self._capacity):
                raise RuntimeStateError("resource release exceeds the budget capacity")
            del self._active[lease._lease_id]
            self._available = available
