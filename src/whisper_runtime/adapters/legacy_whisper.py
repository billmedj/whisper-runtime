"""Transactional bridge for the historical OpenAI Whisper Python API.

The bridge calls ``model.transcribe(audio, **options)`` without importing
``whisper``, NumPy, or PyTorch. It keeps the returned Whisper dictionary intact
inside a versioned envelope while committing the text and declared time span to
the runtime session.

This is a compatibility adapter, not a CUDA isolation layer. The supplied
``ExecutionScope`` must provide a completion fence that covers work started by
``transcribe``. ``ImmediateFence`` is valid only when the backend is fully
synchronous. A strong identity probe must bind the loaded weights to the
``ModelSnapshot`` fingerprint. Metadata-only probes do not detect weight
mutation. Cancellation cannot interrupt a blocking historical ``transcribe``
call; it takes effect when that call returns and the transaction closes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock, RLock
from time import perf_counter_ns
from typing import Protocol, cast, runtime_checkable
from weakref import ReferenceType, ref

from ..errors import (
    ModelMismatchError,
    RuntimeStateError,
    TransactionRetainedError,
)
from ..execution import ExecutionScope
from ..model import ModelSnapshot
from ..resources import ResourceVector
from ..state import RequestState, Session, SessionState, WindowResult
from ..transaction import TransactionStatus, WindowTransaction
from ..worker import Worker

LEGACY_WHISPER_ENVELOPE_VERSION = "whisper-runtime/legacy-transcription/v1"

_MODEL_BINDINGS_GUARD = Lock()


class _ModelBinding:
    __slots__ = ("execution_profile", "lock", "retained_error", "worker")

    def __init__(self) -> None:
        self.lock = RLock()
        self.worker: Worker | None = None
        self.execution_profile: LegacyExecutionProfile | None = None
        self.retained_error: LegacyTranscriptionRetainedError | None = None


_MODEL_BINDINGS: dict[int, tuple[ReferenceType[object], _ModelBinding]] = {}


class LegacyAdapterError(RuntimeStateError):
    """Base error for a violated historical Whisper adapter contract."""


class LegacyPayloadError(LegacyAdapterError):
    """The historical backend returned an invalid transcription payload."""


class LegacyOptionsMutationError(LegacyAdapterError):
    """The backend mutated an option value supplied for one transaction."""


class LegacyTranscriptionRetainedError(LegacyAdapterError):
    """The runtime retained capacity after a legacy transcription ended.

    ``runtime_error.transaction`` is the exact recovery authority. When
    ``envelope`` is present, the session commit already happened and the
    caller must not repeat inference. ``payload`` is only an attempted result
    when ``envelope`` is absent.
    """

    def __init__(
        self,
        runtime_error: TransactionRetainedError,
        *,
        envelope: LegacyTranscriptionEnvelope | None,
        payload_items: tuple[tuple[str, _FrozenValue], ...] | None,
        payload_digest: str | None,
        model_binding: _ModelBinding,
        worker: Worker,
    ) -> None:
        state = (
            "committed"
            if runtime_error.committed_state is not None
            else "did not commit"
        )
        super().__init__(
            f"legacy transcription {state}, but its transaction still owns "
            "runtime capacity; recover runtime_error.transaction"
        )
        self.runtime_error = runtime_error
        self.envelope = envelope
        self._payload_items = payload_items
        self.payload_digest = payload_digest
        self._model_binding = model_binding
        self._worker = worker

    @property
    def transaction(self) -> WindowTransaction:
        """Return the exact retained transaction accepted by ``Worker.recover``."""

        return self.runtime_error.transaction

    @property
    def committed_state(self) -> SessionState | None:
        """Return the published session state, if publication occurred."""

        return self.runtime_error.committed_state

    @property
    def payload(self) -> dict[str, object] | None:
        """Return a detached attempted payload, if the backend produced one."""

        if self._payload_items is None:
            return None
        return _thaw_mapping(self._payload_items)

    def recover(self) -> bool:
        """Retry exact recovery and reopen this model only after full cleanup."""

        with self._model_binding.lock:
            current = self._model_binding.retained_error
            if current is not None and current is not self:
                return False
            self._worker.recover(self.transaction)
            if not _transaction_is_fully_recovered(self.transaction):
                return False
            if self._model_binding.retained_error is self:
                self._model_binding.retained_error = None
            return True


@runtime_checkable
class LegacyWhisperModel(Protocol):
    """Structural type for ``whisper.model.Whisper`` and compatible models."""

    def transcribe(
        self,
        audio: object,
        **decode_options: object,
    ) -> Mapping[str, object]:
        """Return the historical Whisper transcription dictionary."""


ModelIdentityProbe = Callable[[LegacyWhisperModel], ModelSnapshot]


def _drop_model_binding(key: int, dead_reference: ReferenceType[object]) -> None:
    with _MODEL_BINDINGS_GUARD:
        current = _MODEL_BINDINGS.get(key)
        if current is not None and current[0] is dead_reference:
            del _MODEL_BINDINGS[key]


def _model_binding(model: object) -> _ModelBinding:
    """Return one shared execution binding for a live model object.

    Weak references let the registry preserve object identity without keeping
    unused models alive. Models that cannot be weak-referenced are rejected.
    """

    key = id(model)

    def remove(dead_reference: ReferenceType[object]) -> None:
        _drop_model_binding(key, dead_reference)

    try:
        reference = ref(model, remove)
    except TypeError as exc:
        raise TypeError("a legacy model must support weak references") from exc

    with _MODEL_BINDINGS_GUARD:
        current = _MODEL_BINDINGS.get(key)
        if current is not None and current[0]() is model:
            return current[1]
        binding = _ModelBinding()
        _MODEL_BINDINGS[key] = (reference, binding)
        return binding


def _transaction_is_fully_recovered(transaction: WindowTransaction) -> bool:
    return bool(
        transaction.status
        in (
            TransactionStatus.COMMITTED,
            TransactionStatus.ABORTED,
            TransactionStatus.EXPIRED,
        )
        and transaction.cleanup_error is None
    )


def _require_model_available(binding: _ModelBinding) -> None:
    retained = binding.retained_error
    if retained is None:
        return
    if _transaction_is_fully_recovered(retained.transaction):
        binding.retained_error = None
        return
    raise retained


@dataclass(frozen=True, slots=True)
class LegacyExecutionProfile:
    """Trusted resource configuration for one legacy model binding.

    The profile is fixed when the adapter is built. Per-call users cannot
    under-declare the resources consumed by inference.
    """

    profile_id: str
    resources: ResourceVector

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_id.isspace():
            raise ValueError("profile_id must not be empty")
        if not isinstance(self.resources, ResourceVector):
            raise TypeError("resources must be a ResourceVector")
        if self.resources == ResourceVector():
            raise ValueError("a legacy execution profile must reserve resources")


@dataclass(frozen=True, slots=True)
class LegacyInputProvenance:
    """Caller-supplied immutable identity for an audio input.

    The adapter does not inspect or hash arbitrary audio objects. The caller
    that creates this value is responsible for binding the metadata to the
    supplied audio. A digest, when present, must be a SHA-256 value.
    """

    input_id: str
    digest: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.input_id or self.input_id.isspace():
            raise ValueError("input_id must not be empty")
        if self.digest is not None:
            prefix, separator, value = self.digest.partition(":")
            if (
                prefix != "sha256"
                or separator != ":"
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("digest must use canonical sha256:<64 lowercase hex>")
        if self.media_type is not None and (
            not self.media_type or self.media_type.isspace()
        ):
            raise ValueError("media_type must not be empty")
        if self.size_bytes is not None:
            if isinstance(self.size_bytes, bool) or not isinstance(
                self.size_bytes, int
            ):
                raise TypeError("size_bytes must be an integer")
            if self.size_bytes < 0:
                raise ValueError("size_bytes must not be negative")


@dataclass(frozen=True, slots=True)
class LegacyExecutionMeasurements:
    """Measurements available without backend-specific instrumentation."""

    model_lock_wait_ns: int | None
    backend_call_ns: int | None
    runtime_call_ns: int
    device_time_ns: int | None = None
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("runtime_call_ns",):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        for field_name in (
            "backend_call_ns",
            "model_lock_wait_ns",
            "device_time_ns",
            "peak_memory_bytes",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True, slots=True)
class _FrozenValue:
    kind: str
    value: object = None


def _freeze_value(value: object, *, context: str) -> _FrozenValue:
    if value is None:
        return _FrozenValue("none")
    if isinstance(value, bool):
        return _FrozenValue("bool", value)
    if isinstance(value, int):
        return _FrozenValue("int", str(value))
    if isinstance(value, float):
        return _FrozenValue("float", value.hex())
    if isinstance(value, str):
        return _FrozenValue("str", value)
    if isinstance(value, bytes):
        return _FrozenValue("bytes", value.hex())
    if isinstance(value, list):
        return _FrozenValue(
            "list",
            tuple(_freeze_value(item, context=context) for item in value),
        )
    if isinstance(value, tuple):
        return _FrozenValue(
            "tuple",
            tuple(_freeze_value(item, context=context) for item in value),
        )
    if isinstance(value, Mapping):
        items: list[tuple[str, _FrozenValue]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"nested {context} keys must be strings")
            items.append((key, _freeze_value(item, context=context)))
        items.sort(key=lambda pair: pair[0])
        return _FrozenValue("mapping", tuple(items))
    raise TypeError(
        f"{context} must contain only None, booleans, numbers, strings, "
        "bytes, lists, tuples, and string-keyed mappings"
    )


def _thaw_value(option: _FrozenValue) -> object:
    if option.kind == "none":
        return None
    if option.kind == "bool":
        return cast(bool, option.value)
    if option.kind == "int":
        return int(cast(str, option.value))
    if option.kind == "float":
        return float.fromhex(cast(str, option.value))
    if option.kind == "str":
        return cast(str, option.value)
    if option.kind == "bytes":
        return bytes.fromhex(cast(str, option.value))
    if option.kind in ("list", "tuple"):
        values = cast(tuple[_FrozenValue, ...], option.value)
        thawed = tuple(_thaw_value(item) for item in values)
        return list(thawed) if option.kind == "list" else thawed
    if option.kind == "mapping":
        items = cast(tuple[tuple[str, _FrozenValue], ...], option.value)
        return {key: _thaw_value(item) for key, item in items}
    raise AssertionError(f"unknown frozen option kind: {option.kind}")


def _canonical_value(option: _FrozenValue) -> object:
    if option.kind in ("none", "bool", "int", "float", "str", "bytes"):
        return [option.kind, option.value]
    if option.kind in ("list", "tuple"):
        values = cast(tuple[_FrozenValue, ...], option.value)
        return [option.kind, [_canonical_value(item) for item in values]]
    if option.kind == "mapping":
        items = cast(tuple[tuple[str, _FrozenValue], ...], option.value)
        return [
            option.kind,
            [[key, _canonical_value(item)] for key, item in items],
        ]
    raise AssertionError(f"unknown frozen option kind: {option.kind}")


def _mapping_fingerprint(items: tuple[tuple[str, _FrozenValue], ...]) -> str:
    canonical = [[key, _canonical_value(option)] for key, option in items]
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _thaw_mapping(
    items: tuple[tuple[str, _FrozenValue], ...],
) -> dict[str, object]:
    return {key: _thaw_value(value) for key, value in items}


def _freeze_mapping(
    source: Mapping[str, object],
    *,
    context: str,
) -> tuple[tuple[str, _FrozenValue], ...]:
    items: list[tuple[str, _FrozenValue]] = []
    for key, value in source.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} keys must be strings")
        items.append((key, _freeze_value(value, context=context)))
    items.sort(key=lambda pair: pair[0])
    return tuple(items)


@dataclass(frozen=True, slots=True, init=False)
class LegacyTranscribeOptions:
    """A deep, immutable snapshot of historical ``transcribe`` options.

    The accepted value domain covers the options in OpenAI Whisper's public
    Python API. Backend-specific objects are rejected because they cannot be
    copied or fingerprinted with stable semantics.
    """

    _items: tuple[tuple[str, _FrozenValue], ...]
    fingerprint: str

    def __init__(self, options: Mapping[str, object] | None = None) -> None:
        source: Mapping[str, object] = {} if options is None else options
        items: list[tuple[str, _FrozenValue]] = []
        for key, value in source.items():
            if not isinstance(key, str):
                raise TypeError("transcription option keys must be strings")
            if not key:
                raise ValueError("transcription option keys must not be empty")
            if key in ("audio", "model"):
                raise ValueError(f"{key!r} is not a transcription option")
            items.append((key, _freeze_value(value, context="transcription options")))
        items.sort(key=lambda pair: pair[0])
        frozen_items = tuple(items)
        object.__setattr__(self, "_items", frozen_items)
        object.__setattr__(self, "fingerprint", _mapping_fingerprint(frozen_items))

    def to_kwargs(self) -> dict[str, object]:
        """Return a new mutable keyword dictionary for one backend call."""

        return _thaw_mapping(self._items)

    def _require_intact(self) -> None:
        if _mapping_fingerprint(self._items) != self.fingerprint:
            raise LegacyOptionsMutationError(
                "the immutable transcription option snapshot changed"
            )


@dataclass(frozen=True, slots=True, init=False)
class LegacyTranscriptionEnvelope:
    """Versioned result bound to an immutable historical Whisper payload.

    The canonical payload uses immutable internal values. ``payload`` and
    ``to_legacy_payload`` return detached dictionaries in the historical shape.
    ``payload_digest`` binds those values to the transaction metadata.
    """

    schema_version: str
    session_id: str
    request_id: str
    window_id: str
    start_ms: int
    end_ms: int
    model: ModelSnapshot
    options_fingerprint: str
    execution_profile: LegacyExecutionProfile
    input_provenance: LegacyInputProvenance
    measurements: LegacyExecutionMeasurements
    payload_digest: str
    committed_state: SessionState
    _payload_items: tuple[tuple[str, _FrozenValue], ...]

    def __init__(
        self,
        *,
        schema_version: str,
        session_id: str,
        request_id: str,
        window_id: str,
        start_ms: int,
        end_ms: int,
        model: ModelSnapshot,
        options_fingerprint: str,
        execution_profile: LegacyExecutionProfile,
        input_provenance: LegacyInputProvenance,
        measurements: LegacyExecutionMeasurements,
        payload_items: tuple[tuple[str, _FrozenValue], ...],
        payload_digest: str,
        committed_state: SessionState,
    ) -> None:
        payload = _thaw_mapping(payload_items)
        text = payload.get("text")
        if not isinstance(text, str):
            raise LegacyPayloadError(
                "the historical Whisper payload must contain string field 'text'"
            )
        if _mapping_fingerprint(payload_items) != payload_digest:
            raise LegacyPayloadError("the historical payload digest does not match")
        if committed_state.session_id != session_id or not committed_state.windows:
            raise LegacyPayloadError(
                "the committed session does not match the envelope"
            )
        record = committed_state.windows[-1]
        result = record.result
        if (
            record.request_id != request_id
            or record.model != model
            or result.window_id != window_id
            or result.start_ms != start_ms
            or result.end_ms != end_ms
            or result.text != text
        ):
            raise LegacyPayloadError("the committed window does not match the envelope")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "options_fingerprint", options_fingerprint)
        object.__setattr__(self, "execution_profile", execution_profile)
        object.__setattr__(self, "input_provenance", input_provenance)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "payload_digest", payload_digest)
        object.__setattr__(self, "committed_state", committed_state)
        object.__setattr__(self, "_payload_items", payload_items)

    @property
    def payload(self) -> dict[str, object]:
        """Return a detached payload in the historical Whisper shape."""

        return _thaw_mapping(self._payload_items)

    def to_legacy_payload(self) -> dict[str, object]:
        """Return a detached payload in the historical Whisper shape."""

        return self.payload

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary envelope without changing the payload."""

        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "window_id": self.window_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "model": {
                "model_id": self.model.model_id,
                "revision": self.model.revision,
                "backend": self.model.backend,
                "fingerprint": self.model.fingerprint,
            },
            "options_fingerprint": self.options_fingerprint,
            "execution_profile": {
                "profile_id": self.execution_profile.profile_id,
                "resources": {
                    "memory_bytes": self.execution_profile.resources.memory_bytes,
                    "compute_units": self.execution_profile.resources.compute_units,
                    "stream_slots": self.execution_profile.resources.stream_slots,
                },
                "serialization": "per_model",
            },
            "input": {
                "input_id": self.input_provenance.input_id,
                "digest": self.input_provenance.digest,
                "media_type": self.input_provenance.media_type,
                "size_bytes": self.input_provenance.size_bytes,
            },
            "measurements": {
                "model_lock_wait_ns": self.measurements.model_lock_wait_ns,
                "backend_call_ns": self.measurements.backend_call_ns,
                "runtime_call_ns": self.measurements.runtime_call_ns,
                "device_time_ns": self.measurements.device_time_ns,
                "peak_memory_bytes": self.measurements.peak_memory_bytes,
            },
            "payload_digest": self.payload_digest,
            "session_version": self.committed_state.version,
            "payload": self.to_legacy_payload(),
        }


@dataclass(frozen=True, slots=True, init=False)
class LegacyWhisperAdapter:
    """Bind one historical Whisper model to one transactional worker.

    The identity probe runs at bind time and around each transcription. It must
    return the exact checkpoint identity represented by ``worker.model``. One
    model-object binding covers admission, inference, the completion fence,
    and commit. The binding is shared across adapter instances. A legacy model
    has one worker, one full-capacity profile, and one admitted transaction.

    This boundary assumes exclusive, trusted ownership of the model object.
    Code that mutates weights or calls the model without using this adapter can
    bypass the lock. An in-process identity probe cannot prevent hostile writes
    or detect a swap that is restored between observations.

    This first bridge treats one historical ``transcribe`` call as one runtime
    window. It does not yet split the internal 30-second decode loop into
    cancellable transactions or assign separate resource leases to its stages.
    """

    worker: Worker
    execution_profile: LegacyExecutionProfile
    _model: LegacyWhisperModel
    _identity_probe: ModelIdentityProbe
    _model_identity: ModelSnapshot
    _model_binding: _ModelBinding

    def __init__(
        self,
        worker: Worker,
        model: LegacyWhisperModel,
        identity_probe: ModelIdentityProbe,
        execution_profile: LegacyExecutionProfile,
    ) -> None:
        if not isinstance(worker, Worker):
            raise TypeError("worker must be a Worker")
        if not isinstance(model, LegacyWhisperModel):
            raise TypeError("model must implement the historical Whisper API")
        if not callable(identity_probe):
            raise TypeError("identity_probe must be callable")
        if not isinstance(execution_profile, LegacyExecutionProfile):
            raise TypeError("execution_profile must be a LegacyExecutionProfile")
        if worker.queue_capacity != 1:
            raise ValueError("a legacy model worker must have queue_capacity=1")
        if execution_profile.resources != worker.budget.capacity:
            raise ValueError(
                "a legacy execution profile must reserve the full worker capacity"
            )

        model_binding = _model_binding(model)
        with model_binding.lock:
            _require_model_available(model_binding)
            if model_binding.worker is not None and model_binding.worker is not worker:
                raise ValueError("one legacy model object cannot use multiple workers")
            if (
                model_binding.execution_profile is not None
                and model_binding.execution_profile != execution_profile
            ):
                raise ValueError(
                    "one legacy model object cannot use multiple execution profiles"
                )
            identity = identity_probe(model)
            if not isinstance(identity, ModelSnapshot):
                raise TypeError("identity_probe must return ModelSnapshot")
            if identity != worker.model:
                raise ModelMismatchError(
                    "the loaded historical model does not match the worker snapshot"
                )
            model_binding.worker = worker
            model_binding.execution_profile = execution_profile

        object.__setattr__(self, "worker", worker)
        object.__setattr__(self, "execution_profile", execution_profile)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_identity_probe", identity_probe)
        object.__setattr__(self, "_model_identity", identity)
        object.__setattr__(self, "_model_binding", model_binding)

    @property
    def model_identity(self) -> ModelSnapshot:
        """Return the checkpoint identity bound at adapter construction."""

        return self._model_identity

    def transcribe(
        self,
        *,
        session: Session,
        request: RequestState,
        window_id: str,
        execution: ExecutionScope,
        audio: object,
        input_provenance: LegacyInputProvenance,
        start_ms: int,
        end_ms: int,
        options: LegacyTranscribeOptions | None = None,
    ) -> LegacyTranscriptionEnvelope:
        """Run one historical transcription as an atomic session transition.

        The method passes ``audio`` through by identity and passes a fresh thawed
        option dictionary to the backend. It does not inspect or hash ``audio``;
        ``input_provenance`` is the caller's explicit identity claim. The fixed
        execution profile supplies the resource cost. ``start_ms`` and
        ``end_ms`` describe the input span and are not inferred from segments.
        """

        if request.model != self._model_identity:
            raise ModelMismatchError(
                "the request model does not match the bound historical model"
            )
        if self.worker.model != self._model_identity:
            raise ModelMismatchError(
                "the worker model changed after the adapter was bound"
            )
        if not isinstance(input_provenance, LegacyInputProvenance):
            raise TypeError("input_provenance must be LegacyInputProvenance")

        frozen_options = options or LegacyTranscribeOptions()
        frozen_options._require_intact()
        payload_holder: list[tuple[tuple[str, _FrozenValue], ...]] = []
        payload_digest_holder: list[str] = []
        backend_call_ns_holder: list[int] = []
        model_lock_wait_ns_holder: list[int] = []
        model_lock_acquired = False
        call_started_ns = perf_counter_ns()

        def operation(transaction: WindowTransaction) -> WindowResult:
            nonlocal model_lock_acquired
            lock_wait_started_ns = perf_counter_ns()
            self._model_binding.lock.acquire()
            model_lock_acquired = True
            model_lock_wait_ns_holder.append(perf_counter_ns() - lock_wait_started_ns)
            _require_model_available(self._model_binding)
            kwargs = frozen_options.to_kwargs()
            initial_kwargs_fingerprint = LegacyTranscribeOptions(kwargs).fingerprint

            def invoke_backend() -> tuple[tuple[str, _FrozenValue], ...]:
                self._require_model_identity()
                backend_started_ns = perf_counter_ns()
                try:
                    raw_payload = self._model.transcribe(audio, **kwargs)
                finally:
                    backend_call_ns_holder.append(
                        perf_counter_ns() - backend_started_ns
                    )
                self._require_model_identity()
                if (
                    LegacyTranscribeOptions(kwargs).fingerprint
                    != initial_kwargs_fingerprint
                ):
                    raise LegacyOptionsMutationError(
                        "the historical backend mutated a transcription option"
                    )
                return _freeze_legacy_payload(raw_payload)

            payload_items = transaction.submit(invoke_backend)
            payload_holder.append(payload_items)
            payload_digest_holder.append(_mapping_fingerprint(payload_items))
            payload = _thaw_mapping(payload_items)
            return WindowResult(
                window_id=window_id,
                text=cast(str, payload["text"]),
                start_ms=start_ms,
                end_ms=end_ms,
            )

        try:
            try:
                committed = self.worker.execute(
                    session=session,
                    request=request,
                    window_id=window_id,
                    resources=self.execution_profile.resources,
                    execution=execution,
                    operation=operation,
                )
            except TransactionRetainedError as runtime_error:
                measurements = self._measurements(
                    call_started_ns,
                    model_lock_wait_ns_holder,
                    backend_call_ns_holder,
                )
                retained_error = LegacyTranscriptionRetainedError(
                    runtime_error,
                    envelope=None,
                    payload_items=(
                        payload_holder[0] if len(payload_holder) == 1 else None
                    ),
                    payload_digest=(
                        payload_digest_holder[0]
                        if len(payload_digest_holder) == 1
                        else None
                    ),
                    model_binding=self._model_binding,
                    worker=self.worker,
                )
                with self._model_binding.lock:
                    self._model_binding.retained_error = retained_error
                if runtime_error.committed_state is not None:
                    try:
                        retained_error.envelope = self._build_envelope(
                            session=session,
                            request=request,
                            window_id=window_id,
                            input_provenance=input_provenance,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            options=frozen_options,
                            measurements=measurements,
                            payload_holder=payload_holder,
                            payload_digest_holder=payload_digest_holder,
                            committed=runtime_error.committed_state,
                        )
                    except BaseException as envelope_error:
                        raise retained_error from envelope_error
                raise retained_error from runtime_error

            measurements = self._measurements(
                call_started_ns,
                model_lock_wait_ns_holder,
                backend_call_ns_holder,
            )
            return self._build_envelope(
                session=session,
                request=request,
                window_id=window_id,
                input_provenance=input_provenance,
                start_ms=start_ms,
                end_ms=end_ms,
                options=frozen_options,
                measurements=measurements,
                payload_holder=payload_holder,
                payload_digest_holder=payload_digest_holder,
                committed=committed,
            )
        finally:
            if model_lock_acquired:
                self._model_binding.lock.release()

    def _measurements(
        self,
        call_started_ns: int,
        model_lock_wait_ns_holder: list[int],
        backend_call_ns_holder: list[int],
    ) -> LegacyExecutionMeasurements:
        if len(model_lock_wait_ns_holder) > 1:
            raise LegacyAdapterError(
                "a transcription cannot acquire the model binding multiple times"
            )
        if len(backend_call_ns_holder) > 1:
            raise LegacyAdapterError(
                "a transcription cannot contain multiple backend measurements"
            )
        return LegacyExecutionMeasurements(
            model_lock_wait_ns=(
                model_lock_wait_ns_holder[0] if model_lock_wait_ns_holder else None
            ),
            backend_call_ns=(
                backend_call_ns_holder[0] if backend_call_ns_holder else None
            ),
            runtime_call_ns=perf_counter_ns() - call_started_ns,
        )

    def _build_envelope(
        self,
        *,
        session: Session,
        request: RequestState,
        window_id: str,
        input_provenance: LegacyInputProvenance,
        start_ms: int,
        end_ms: int,
        options: LegacyTranscribeOptions,
        measurements: LegacyExecutionMeasurements,
        payload_holder: list[tuple[tuple[str, _FrozenValue], ...]],
        payload_digest_holder: list[str],
        committed: SessionState,
    ) -> LegacyTranscriptionEnvelope:
        if len(payload_holder) != 1 or len(payload_digest_holder) != 1:
            raise LegacyAdapterError(
                "a committed transcription must contain exactly one payload"
            )
        return LegacyTranscriptionEnvelope(
            schema_version=LEGACY_WHISPER_ENVELOPE_VERSION,
            session_id=session.session_id,
            request_id=request.request_id,
            window_id=window_id,
            start_ms=start_ms,
            end_ms=end_ms,
            model=self._model_identity,
            options_fingerprint=options.fingerprint,
            execution_profile=self.execution_profile,
            input_provenance=input_provenance,
            measurements=measurements,
            payload_items=payload_holder[0],
            payload_digest=payload_digest_holder[0],
            committed_state=committed,
        )

    def _require_model_identity(self) -> None:
        observed = self._identity_probe(self._model)
        if not isinstance(observed, ModelSnapshot):
            raise TypeError("identity_probe must return ModelSnapshot")
        if observed != self._model_identity:
            raise ModelMismatchError(
                "the historical model identity changed during execution"
            )


def _freeze_legacy_payload(
    payload: Mapping[str, object],
) -> tuple[tuple[str, _FrozenValue], ...]:
    if not isinstance(payload, Mapping):
        raise LegacyPayloadError("the historical Whisper backend must return a mapping")
    try:
        frozen = _freeze_mapping(payload, context="historical Whisper payload")
    except (TypeError, ValueError) as exc:
        raise LegacyPayloadError(str(exc)) from exc
    copied = _thaw_mapping(frozen)
    if not isinstance(copied.get("text"), str):
        raise LegacyPayloadError(
            "the historical Whisper payload must contain string field 'text'"
        )
    return frozen
