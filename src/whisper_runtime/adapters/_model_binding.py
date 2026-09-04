"""Shared ownership for adapter-bound model objects."""

from __future__ import annotations

from threading import Lock, RLock
from weakref import ReferenceType, ref

from ..errors import RuntimeStateError
from ..resources import ResourceVector
from ..transaction import TransactionStatus, WindowTransaction
from ..worker import Worker

_MODEL_BINDINGS_GUARD = Lock()


class ModelBinding:
    """Coordinate one live model object across all runtime adapters."""

    __slots__ = (
        "_cleanup_failures",
        "_retained_errors",
        "adapter_kind",
        "execution_profile",
        "lock",
        "worker",
    )

    def __init__(self) -> None:
        self.lock = RLock()
        self.worker: Worker | None = None
        self.execution_profile: tuple[str, ResourceVector, int] | None = None
        self.adapter_kind: str | None = None
        self._cleanup_failures: set[int] = set()
        self._retained_errors: dict[int, BaseException] = {}

    @property
    def retained_error(self) -> BaseException | None:
        """Return the oldest retained error, if one still owns capacity."""

        with self.lock:
            return next(iter(self._retained_errors.values()), None)

    @retained_error.setter
    def retained_error(self, error: BaseException | None) -> None:
        """Keep compatibility with single-lane adapters."""

        with self.lock:
            if error is None:
                self._retained_errors.clear()
                return
            transaction = getattr(error, "transaction", None)
            if not isinstance(transaction, WindowTransaction):
                raise TypeError("a retained model error must expose its transaction")
            self._retained_errors = {id(transaction): error}

    def record_retained_error(self, error: BaseException) -> None:
        """Record one exact retained transaction without replacing its peers."""

        transaction = getattr(error, "transaction", None)
        if not isinstance(transaction, WindowTransaction):
            raise TypeError("a retained model error must expose its transaction")
        with self.lock:
            self._retained_errors[id(transaction)] = error

    def record_cleanup_failure(self, scope: object) -> None:
        """Block new model work as soon as a completion fence fails."""

        with self.lock:
            self._cleanup_failures.add(id(scope))

    def clear_cleanup_failure(self, scope: object) -> None:
        """Clear one completion-fence failure after cleanup succeeds."""

        with self.lock:
            self._cleanup_failures.discard(id(scope))

    def unresolved_retained_error(self) -> BaseException | None:
        """Discard recovered entries and return the oldest unresolved error."""

        with self.lock:
            for key, error in tuple(self._retained_errors.items()):
                transaction = getattr(error, "transaction", None)
                if not isinstance(transaction, WindowTransaction):
                    raise RuntimeStateError(
                        "a retained model error has no transaction handle"
                    )
                if transaction_is_fully_recovered(transaction):
                    del self._retained_errors[key]
            retained = next(iter(self._retained_errors.values()), None)
            if retained is not None:
                return retained
            if self._cleanup_failures:
                raise RuntimeStateError(
                    "model cleanup failed; new work is blocked pending recovery"
                )
            return None


_MODEL_BINDINGS: dict[int, tuple[ReferenceType[object], ModelBinding]] = {}


def _drop_model_binding(key: int, dead_reference: ReferenceType[object]) -> None:
    with _MODEL_BINDINGS_GUARD:
        current = _MODEL_BINDINGS.get(key)
        if current is not None and current[0] is dead_reference:
            del _MODEL_BINDINGS[key]


def get_model_binding(model: object, *, subject: str) -> ModelBinding:
    """Return the process-wide binding for one live model object."""

    key = id(model)

    def remove(dead_reference: ReferenceType[object]) -> None:
        _drop_model_binding(key, dead_reference)

    try:
        reference = ref(model, remove)
    except TypeError as exc:
        raise TypeError(f"a {subject} must support weak references") from exc

    with _MODEL_BINDINGS_GUARD:
        current = _MODEL_BINDINGS.get(key)
        if current is not None and current[0]() is model:
            return current[1]
        binding = ModelBinding()
        _MODEL_BINDINGS[key] = (reference, binding)
        return binding


def bind_model(
    binding: ModelBinding,
    *,
    adapter_kind: str,
    worker: Worker,
    profile_id: str,
    resources: ResourceVector,
    subject: str,
    concurrency: int = 1,
) -> None:
    """Bind one model to one adapter kind, worker, and fixed profile."""

    if binding.adapter_kind is not None and binding.adapter_kind != adapter_kind:
        raise ValueError(
            f"one {subject} cannot change adapter kind during its lifetime"
        )
    if binding.worker is not None and binding.worker is not worker:
        raise ValueError(f"one {subject} cannot use multiple workers")
    profile = (profile_id, resources, concurrency)
    if binding.execution_profile is not None and binding.execution_profile != profile:
        raise ValueError(f"one {subject} cannot use multiple execution profiles")
    binding.adapter_kind = adapter_kind
    binding.worker = worker
    binding.execution_profile = profile


def transaction_is_fully_recovered(transaction: WindowTransaction) -> bool:
    """Return whether a retained transaction no longer owns runtime capacity."""

    return bool(
        transaction.status
        in (
            TransactionStatus.COMMITTED,
            TransactionStatus.ABORTED,
            TransactionStatus.EXPIRED,
        )
        and transaction.cleanup_error is None
    )


def require_model_available(binding: ModelBinding) -> None:
    """Reject a model whose previous transaction still owns capacity."""

    retained = binding.unresolved_retained_error()
    if retained is None:
        return
    raise retained
