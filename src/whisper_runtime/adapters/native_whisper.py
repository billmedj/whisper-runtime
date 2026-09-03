"""Token-step transaction bridge for the suspendable Whisper decoder.

This adapter targets the opt-in ``DecodingTask._start_run`` API implemented by
the companion Whisper prototype. PyTorch and Whisper are loaded only when a
decode starts. The first implementation is deliberately CPU-only, accepts one
unbatched 30-second mel window, and reserves one complete worker.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from threading import Condition, RLock
from typing import Protocol, cast, runtime_checkable

from ..errors import ModelMismatchError, RuntimeStateError, TransactionRetainedError
from ..model import ModelSnapshot
from ..resources import ResourceVector
from ..state import RequestState, Session, SessionState, WindowResult
from ..transaction import WindowTransaction
from ..worker import Worker
from ._model_binding import (
    ModelBinding,
    bind_model,
    get_model_binding,
    require_model_available,
)


class NativeAdapterError(RuntimeStateError):
    """Base error for a violated native decoder adapter contract."""


class NativeDependencyError(NativeAdapterError):
    """The required suspendable Whisper implementation is unavailable."""


class NativeDecodeContractError(NativeAdapterError):
    """The native decoder returned a value outside the supported contract."""


@runtime_checkable
class NativeDecodeRun(Protocol):
    """Mutable state owned by one suspendable decoder invocation."""

    @property
    def complete(self) -> bool:
        """Return whether token generation has ended."""

    def prefill(self) -> None:
        """Build the initial decoder cache without selecting a token."""

    def step(self) -> bool:
        """Select exactly one token for every active hypothesis."""

    def finalize(self) -> list[object]:
        """Build the single public decode result."""

    def cleanup(self) -> None:
        """Release request-local inference state idempotently."""


class _TorchGenerator(Protocol):
    def manual_seed(self, seed: int) -> object:
        """Seed this generator and return an implementation-defined value."""


class _NativeDecodingTask(Protocol):
    def _start_run(self, mel: object) -> object:
        """Start one request-local suspendable decode run."""


@dataclass(frozen=True, slots=True)
class _NativeComponents:
    generator_type: Callable[..., _TorchGenerator]
    options_type: Callable[..., object]
    task_type: Callable[[object, object], object]
    n_frames: int


def _load_native_components() -> _NativeComponents:
    """Load the optional native backend without an import-time dependency."""

    try:
        torch_module = import_module("torch")
        audio_module = import_module("whisper.audio")
        decoding_module = import_module("whisper.decoding")
    except (ImportError, OSError) as exc:
        raise NativeDependencyError(
            "native decoding requires PyTorch and the suspendable Whisper fork"
        ) from exc

    generator_type = getattr(torch_module, "Generator", None)
    options_type = getattr(decoding_module, "DecodingOptions", None)
    task_type = getattr(decoding_module, "DecodingTask", None)
    n_frames = getattr(audio_module, "N_FRAMES", None)
    if (
        not callable(generator_type)
        or not callable(options_type)
        or not callable(task_type)
        or isinstance(n_frames, bool)
        or not isinstance(n_frames, int)
        or n_frames <= 0
    ):
        raise NativeDependencyError(
            "the installed PyTorch or Whisper package lacks native decode types"
        )

    return _NativeComponents(
        generator_type=cast(Callable[..., _TorchGenerator], generator_type),
        options_type=cast(Callable[..., object], options_type),
        task_type=cast(Callable[[object, object], object], task_type),
        n_frames=n_frames,
    )


def _validate_token_input(
    name: str,
    value: str | tuple[int, ...] | None,
) -> None:
    if value is None or isinstance(value, str):
        return
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a string, tuple of integers, or None")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise TypeError(f"{name} tokens must be integers")


def _validate_optional_positive_integer(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_optional_finite_number(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_cpu_device(value: object, *, subject: str) -> None:
    device = getattr(value, "device", None)
    device_type = getattr(device, "type", None)
    if device_type != "cpu":
        raise ValueError(
            f"the native adapter requires an explicit CPU device for {subject}"
        )


def _require_mel_shape(mel: object, model: object, *, n_frames: int) -> None:
    dims = getattr(model, "dims", None)
    n_mels = getattr(dims, "n_mels", None)
    if isinstance(n_mels, bool) or not isinstance(n_mels, int) or n_mels <= 0:
        raise NativeDecodeContractError(
            "the native model must declare a positive dims.n_mels value"
        )

    raw_shape = getattr(mel, "shape", None)
    if not isinstance(raw_shape, tuple) or any(
        isinstance(axis, bool) or not isinstance(axis, int) for axis in raw_shape
    ):
        raise TypeError("mel must expose a concrete two-dimensional shape")
    shape = tuple(raw_shape)
    expected = (n_mels, n_frames)
    if shape != expected:
        raise ValueError(
            f"mel shape must be {expected!r} for the bound model; received {shape!r}"
        )


@dataclass(frozen=True, slots=True)
class NativeDecodeOptions:
    """Dependency-free options for one CPU decode window.

    The adapter fixes ``fp16`` to ``False`` and creates the PyTorch generator.
    Callers cannot supply shared random state.
    """

    task: str = "transcribe"
    language: str | None = None
    temperature: float = 0.0
    sample_len: int | None = None
    best_of: int | None = None
    beam_size: int | None = None
    patience: float | None = None
    length_penalty: float | None = None
    prompt: str | tuple[int, ...] | None = None
    prefix: str | tuple[int, ...] | None = None
    suppress_tokens: str | tuple[int, ...] | None = "-1"
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: float | None = 1.0

    def __post_init__(self) -> None:
        if self.task not in ("transcribe", "translate"):
            raise ValueError("task must be 'transcribe' or 'translate'")
        if self.language is not None and (
            not isinstance(self.language, str) or not self.language
        ):
            raise ValueError("language must be a non-empty string or None")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError("temperature must be a number")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and not negative")
        _validate_optional_positive_integer("sample_len", self.sample_len)
        _validate_optional_positive_integer("best_of", self.best_of)
        _validate_optional_positive_integer("beam_size", self.beam_size)
        _validate_optional_finite_number("patience", self.patience)
        _validate_optional_finite_number("length_penalty", self.length_penalty)
        _validate_optional_finite_number(
            "max_initial_timestamp", self.max_initial_timestamp
        )
        _validate_token_input("prompt", self.prompt)
        _validate_token_input("prefix", self.prefix)
        _validate_token_input("suppress_tokens", self.suppress_tokens)
        if not isinstance(self.suppress_blank, bool):
            raise TypeError("suppress_blank must be a boolean")
        if not isinstance(self.without_timestamps, bool):
            raise TypeError("without_timestamps must be a boolean")
        if self.beam_size is not None and self.best_of is not None:
            raise ValueError("beam_size and best_of cannot be used together")
        if self.temperature == 0 and self.best_of is not None:
            raise ValueError("best_of requires a temperature greater than zero")
        if self.patience is not None and self.beam_size is None:
            raise ValueError("patience requires beam_size")
        if self.length_penalty is not None and not 0 <= self.length_penalty <= 1:
            raise ValueError("length_penalty must be between zero and one")

    def _to_kwargs(self, generator: _TorchGenerator) -> dict[str, object]:
        return {
            "task": self.task,
            "language": self.language,
            "temperature": self.temperature,
            "sample_len": self.sample_len,
            "best_of": self.best_of,
            "beam_size": self.beam_size,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "prompt": self.prompt,
            "prefix": self.prefix,
            "suppress_tokens": self.suppress_tokens,
            "suppress_blank": self.suppress_blank,
            "without_timestamps": self.without_timestamps,
            "max_initial_timestamp": self.max_initial_timestamp,
            "fp16": False,
            "generator": generator,
        }


@dataclass(frozen=True, slots=True)
class NativeExecutionProfile:
    """Fixed full-worker resource reservation for one native model."""

    profile_id: str
    resources: ResourceVector

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_id.isspace():
            raise ValueError("profile_id must not be empty")
        if not isinstance(self.resources, ResourceVector):
            raise TypeError("resources must be a ResourceVector")
        if self.resources == ResourceVector():
            raise ValueError("a native execution profile must reserve resources")


class _CpuDecodeScope:
    """Fence one synchronous CPU run and clean it before lease release."""

    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._run: object | None = None
        self._bound = False
        self._cleanup_in_flight = False
        self._cleaned = False
        self._stop_requested = False

    def bind(self, run: object) -> None:
        """Register the run before its creating submission returns."""

        with self._condition:
            if self._bound:
                raise NativeDecodeContractError(
                    "an execution scope cannot bind multiple decode runs"
                )
            if self._cleaned:
                raise NativeDecodeContractError(
                    "a closed execution scope cannot bind a decode run"
                )
            self._run = run
            self._bound = True

    def request_stop(self) -> None:
        """Latch a stop; the owner reaches cleanup at the next checkpoint."""

        with self._condition:
            self._stop_requested = True

    def completion_fence(self) -> _CpuDecodeScope:
        return self

    def wait(self) -> None:
        """Clean the bound run once after admitted callbacks have drained."""

        with self._condition:
            while self._cleanup_in_flight:
                self._condition.wait()
            if self._cleaned:
                return
            run = self._run
            self._cleanup_in_flight = True

        try:
            if not self._bound:
                pass
            elif run is None:
                raise NativeDecodeContractError(
                    "the decode handle has no callable cleanup method; "
                    "quiescence cannot be proven"
                )
            else:
                cleanup = getattr(run, "cleanup", None)
                if not callable(cleanup):
                    raise NativeDecodeContractError(
                        "the decode handle has no callable cleanup method; "
                        "quiescence cannot be proven"
                    )
                cleanup_result = cleanup()
                if cleanup_result is not None:
                    close = getattr(cleanup_result, "close", None)
                    if callable(close):
                        close()
                    raise NativeDecodeContractError(
                        "decode cleanup did not complete synchronously; "
                        "quiescence cannot be proven"
                    )
        except BaseException:
            with self._condition:
                self._cleanup_in_flight = False
                self._condition.notify_all()
            raise
        else:
            with self._condition:
                self._cleaned = True
                self._cleanup_in_flight = False
                self._condition.notify_all()


NativeModelIdentityProbe = Callable[[object], ModelSnapshot]


@dataclass(frozen=True, slots=True, init=False)
class NativeWhisperAdapter:
    """Bind a suspendable CPU decoder to one transactional worker."""

    worker: Worker
    execution_profile: NativeExecutionProfile
    _model: object
    _identity_probe: NativeModelIdentityProbe
    _model_identity: ModelSnapshot
    _model_binding: ModelBinding

    def __init__(
        self,
        worker: Worker,
        model: object,
        identity_probe: NativeModelIdentityProbe,
        execution_profile: NativeExecutionProfile,
    ) -> None:
        if not isinstance(worker, Worker):
            raise TypeError("worker must be a Worker")
        if not callable(identity_probe):
            raise TypeError("identity_probe must be callable")
        if not isinstance(execution_profile, NativeExecutionProfile):
            raise TypeError("execution_profile must be a NativeExecutionProfile")
        if worker.queue_capacity != 1:
            raise ValueError("a native model worker must have queue_capacity=1")
        if execution_profile.resources != worker.budget.capacity:
            raise ValueError(
                "a native execution profile must reserve the full worker capacity"
            )

        binding = get_model_binding(model, subject="native model")
        with binding.lock:
            require_model_available(binding)
            identity = identity_probe(model)
            if not isinstance(identity, ModelSnapshot):
                raise TypeError("identity_probe must return ModelSnapshot")
            if identity != worker.model:
                raise ModelMismatchError(
                    "the loaded native model does not match the worker snapshot"
                )
            bind_model(
                binding,
                adapter_kind="native",
                worker=worker,
                profile_id=execution_profile.profile_id,
                resources=execution_profile.resources,
                subject="native model object",
            )

        object.__setattr__(self, "worker", worker)
        object.__setattr__(self, "execution_profile", execution_profile)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_identity_probe", identity_probe)
        object.__setattr__(self, "_model_identity", identity)
        object.__setattr__(self, "_model_binding", binding)

    @property
    def model_identity(self) -> ModelSnapshot:
        """Return the checkpoint identity fixed at adapter construction."""

        return self._model_identity

    def decode_window(
        self,
        *,
        session: Session,
        request: RequestState,
        window_id: str,
        mel: object,
        start_ms: int,
        end_ms: int,
        options: NativeDecodeOptions | None = None,
    ) -> SessionState:
        """Decode and atomically commit one unbatched 30-second CPU window."""

        if request.model != self._model_identity:
            raise ModelMismatchError(
                "the request model does not match the bound native model"
            )
        if self.worker.model != self._model_identity:
            raise ModelMismatchError(
                "the worker model changed after the adapter was bound"
            )
        if options is not None and not isinstance(options, NativeDecodeOptions):
            raise TypeError("options must be a NativeDecodeOptions or None")
        decode_options = options or NativeDecodeOptions()
        WindowResult(window_id=window_id, text="", start_ms=start_ms, end_ms=end_ms)
        if end_ms - start_ms > 30_000:
            raise ValueError("a native decode window cannot exceed 30 seconds")
        ndim = getattr(mel, "ndim", None)
        unsqueeze = getattr(mel, "unsqueeze", None)
        if ndim != 2 or not callable(unsqueeze):
            raise TypeError("mel must be one unbatched two-dimensional tensor")
        _require_cpu_device(mel, subject="the mel tensor")
        _require_cpu_device(self._model, subject="the model")

        components = _load_native_components()
        _require_mel_shape(mel, self._model, n_frames=components.n_frames)
        execution = _CpuDecodeScope()

        def operation(transaction: WindowTransaction) -> WindowResult:
            require_model_available(self._model_binding)
            seed = transaction.randrange(1 << 63)

            def start_run() -> NativeDecodeRun:
                self._require_model_identity()
                task, start = self._build_native_task(
                    components,
                    decode_options,
                    seed=seed,
                )
                del task
                batched_mel = unsqueeze(0)
                run_value = start(batched_mel)
                execution.bind(run_value)
                if not isinstance(run_value, NativeDecodeRun):
                    raise NativeDecodeContractError(
                        "_start_run returned an incompatible decode handle"
                    )
                return run_value

            run = transaction.submit(start_run)
            transaction.checkpoint()
            if run.complete:
                raise NativeDecodeContractError(
                    "the decode run completed before its prefill stage"
                )

            transaction.submit(run.prefill)
            transaction.checkpoint()

            while not run.complete:
                reported_complete = transaction.submit(run.step)
                if not isinstance(reported_complete, bool):
                    raise NativeDecodeContractError("decode step must return a boolean")
                if reported_complete != run.complete:
                    raise NativeDecodeContractError(
                        "decode step completion disagrees with the run state"
                    )
                transaction.checkpoint()

            results = transaction.submit(run.finalize)
            transaction.checkpoint()
            if not isinstance(results, list) or len(results) != 1:
                raise NativeDecodeContractError(
                    "native decoding must return exactly one result"
                )
            text = getattr(results[0], "text", None)
            if not isinstance(text, str):
                raise NativeDecodeContractError(
                    "the native decode result must contain text"
                )
            self._require_model_identity()
            return WindowResult(
                window_id=window_id,
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
            )

        with self._model_binding.lock:
            require_model_available(self._model_binding)
            self._require_model_identity()
            self._preflight_native_backend(components, decode_options)
            try:
                return self.worker.execute(
                    session=session,
                    request=request,
                    window_id=window_id,
                    resources=self.execution_profile.resources,
                    execution=execution,
                    operation=operation,
                )
            except TransactionRetainedError as retained:
                self._model_binding.retained_error = retained
                raise

    def _require_model_identity(self) -> None:
        _require_cpu_device(self._model, subject="the model")
        identity = self._identity_probe(self._model)
        if not isinstance(identity, ModelSnapshot):
            raise TypeError("identity_probe must return ModelSnapshot")
        if identity != self._model_identity:
            raise ModelMismatchError("the loaded native model changed during decoding")

    def _preflight_native_backend(
        self,
        components: _NativeComponents,
        options: NativeDecodeOptions,
    ) -> None:
        """Validate the staged API without starting encoder or decoder kernels."""

        self._build_native_task(components, options, seed=0)

    def _build_native_task(
        self,
        components: _NativeComponents,
        options: NativeDecodeOptions,
        *,
        seed: int,
    ) -> tuple[_NativeDecodingTask, Callable[[object], object]]:
        try:
            generator = components.generator_type(device="cpu")
            manual_seed = getattr(generator, "manual_seed", None)
            if not callable(manual_seed):
                raise TypeError("torch.Generator has no callable manual_seed")
            manual_seed(seed)
            native_options = components.options_type(**options._to_kwargs(generator))
            task_value = components.task_type(self._model, native_options)
        except (AttributeError, TypeError) as exc:
            raise NativeDependencyError(
                "the installed Whisper build does not implement the required "
                "transactional decode options"
            ) from exc

        task = cast(_NativeDecodingTask, task_value)
        start = getattr(task, "_start_run", None)
        if not callable(start):
            raise NativeDependencyError(
                "the installed Whisper decoder has no suspendable run API"
            )
        return task, cast(Callable[[object], object], start)
