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
    """Serialize one live model object across all runtime adapters."""

    __slots__ = (
        "adapter_kind",
        "execution_profile",
        "lock",
        "retained_error",
        "worker",
    )

    def __init__(self) -> None:
        self.lock = RLock()
        self.worker: Worker | None = None
        self.execution_profile: tuple[str, ResourceVector] | None = None
        self.adapter_kind: str | None = None
        self.retained_error: BaseException | None = None


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
) -> None:
    """Bind one model to one adapter kind, worker, and fixed profile."""

    if binding.adapter_kind is not None and binding.adapter_kind != adapter_kind:
        raise ValueError(
            f"one {subject} cannot change adapter kind during its lifetime"
        )
    if binding.worker is not None and binding.worker is not worker:
        raise ValueError(f"one {subject} cannot use multiple workers")
    profile = (profile_id, resources)
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

    retained = binding.retained_error
    if retained is None:
        return
    transaction = getattr(retained, "transaction", None)
    if not isinstance(transaction, WindowTransaction):
        raise RuntimeStateError("a retained model error has no transaction handle")
    if transaction_is_fully_recovered(transaction):
        binding.retained_error = None
        return
    raise retained
