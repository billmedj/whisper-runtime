"""Token-step transaction bridge for the suspendable Whisper decoder.

This adapter targets the opt-in ``DecodingTask._start_run`` API implemented by
the companion Whisper prototype. PyTorch and Whisper are loaded only when a
decode starts. The adapter accepts one unbatched 30-second mel window. CPU is
the default. An explicit CUDA profile borrows one model-binding-owned stream
with a fresh transaction scope and event-backed completion fence. The
experimental two-lane profile remains CPU only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from numbers import Real
from threading import Condition, RLock, current_thread
from typing import Protocol, TypeVar, cast, runtime_checkable

from ..errors import (
    ModelMismatchError,
    QueueFullError,
    RuntimeStateError,
    TransactionRetainedError,
    TransactionStateError,
)
from ..model import ModelSnapshot
from ..resources import ResourceVector
from ..state import (
    AudioSpan,
    RequestState,
    Session,
    SessionState,
    WindowResult,
    _validate_committed_through,
)
from ..transaction import TransactionStatus, WindowTransaction
from ..worker import Worker
from ._model_binding import (
    ModelBinding,
    bind_model,
    get_model_binding,
    require_model_available,
)
from .audio_evidence import SilencePublication
from .native_result import (
    NativeTimestampSegment,
    NativeTokenizer,
    NativeWindowResult,
    build_native_window_result,
    select_native_publication,
)
from .word_policy import AlignedPublication, NativeWordAlignment


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
    device: object

    def manual_seed(self, seed: int) -> object:
        """Seed this generator and return an implementation-defined value."""


class _NativeDecodingTask(Protocol):
    def _start_run(self, mel: object) -> object:
        """Start one request-local suspendable decode run."""


class _CudaEvent(Protocol):
    def record(self, stream: object) -> None:
        """Record this event on ``stream``."""

    def synchronize(self) -> None:
        """Wait until all work before this event has completed."""


class _CudaRuntime(Protocol):
    def is_available(self) -> bool:
        """Return whether CUDA is available in this process."""

    def device_count(self) -> int:
        """Return the number of visible CUDA devices."""

    def synchronize(self, device: object | None = None) -> None:
        """Wait for prior work while establishing the initial stream boundary."""

    def Stream(self, *, device: object | None = None) -> object:
        """Create a CUDA stream."""

    def Event(self, *, enable_timing: bool = False) -> _CudaEvent:
        """Create a CUDA event."""

    def device(self, device: object) -> AbstractContextManager[object]:
        """Select one CUDA device for the current host thread."""

    def stream(self, stream: object) -> AbstractContextManager[object]:
        """Select one CUDA stream for the current host thread."""


class _TorchModule(Protocol):
    cuda: _CudaRuntime
    float32: object


@dataclass(frozen=True, slots=True)
class _NativeComponents:
    generator_type: Callable[..., _TorchGenerator]
    options_type: Callable[..., object]
    task_type: Callable[[object, object], object]
    n_frames: int
    torch_module: _TorchModule | None = None


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
        torch_module=cast(_TorchModule, torch_module),
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


_CUDA_DEVICE_PATTERN = re.compile(r"cuda:(0|[1-9][0-9]*)\Z")


def _validate_profile_device(device: str) -> None:
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    if device == "cpu" or _CUDA_DEVICE_PATTERN.fullmatch(device) is not None:
        return
    raise ValueError("device must be 'cpu' or a canonical 'cuda:N' value")


def _value_device_name(value: object) -> str | None:
    device = getattr(value, "device", None)
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_type == "cpu" and device_index is None:
        return "cpu"
    if (
        device_type == "cuda"
        and not isinstance(device_index, bool)
        and isinstance(device_index, int)
        and device_index >= 0
    ):
        return f"cuda:{device_index}"
    return None


def _require_exact_device(value: object, *, subject: str, expected: str) -> None:
    observed = _value_device_name(value)
    if observed != expected:
        raise ValueError(
            f"the native adapter requires device {expected!r} for {subject}; "
            f"received {observed!r}"
        )


def _cuda_device_index(device: str) -> int:
    match = _CUDA_DEVICE_PATTERN.fullmatch(device)
    if match is None:
        raise ValueError("a CUDA profile requires a canonical 'cuda:N' device")
    return int(match.group(1))


def _require_cuda_runtime(
    components: _NativeComponents,
    *,
    device: str,
) -> _TorchModule:
    torch_module = components.torch_module
    if torch_module is None:
        raise NativeDependencyError("the loaded native components omit PyTorch")
    cuda = getattr(torch_module, "cuda", None)
    required = (
        "is_available",
        "device_count",
        "synchronize",
        "Stream",
        "Event",
        "device",
        "stream",
    )
    if cuda is None or any(
        not callable(getattr(cuda, name, None)) for name in required
    ):
        raise NativeDependencyError("PyTorch does not expose the required CUDA API")
    runtime = cast(_CudaRuntime, cuda)
    try:
        available = runtime.is_available()
        count = runtime.device_count()
    except (RuntimeError, TypeError) as exc:
        raise NativeDependencyError(
            "PyTorch could not inspect the CUDA runtime"
        ) from exc
    index = _cuda_device_index(device)
    if available is not True or isinstance(count, bool) or not isinstance(count, int):
        raise NativeDependencyError("CUDA is not available in this process")
    if index >= count:
        raise NativeDependencyError(
            f"CUDA device index {index} is not visible; device_count is {count}"
        )
    return torch_module


def _require_cuda_input_mel(mel: object, torch_module: _TorchModule) -> None:
    _require_exact_device(mel, subject="the mel tensor", expected="cpu")
    if getattr(mel, "dtype", None) != torch_module.float32:
        raise ValueError("a CUDA profile requires a CPU float32 mel tensor")
    if not callable(getattr(mel, "to", None)):
        raise TypeError("a CUDA mel tensor must expose a callable to method")


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
    """Dependency-free options for one native decode window.

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
    """Fixed native execution settings; feature reuse is experimental and opt in."""

    profile_id: str
    resources: ResourceVector
    max_concurrent_decodes: int = 1
    device: str = "cpu"
    reuse_alignment_features: bool = False

    def __post_init__(self) -> None:
        if type(self.reuse_alignment_features) is not bool:
            raise TypeError("reuse_alignment_features must be a boolean")
        if not self.profile_id or self.profile_id.isspace():
            raise ValueError("profile_id must not be empty")
        if not isinstance(self.resources, ResourceVector):
            raise TypeError("resources must be a ResourceVector")
        if self.resources == ResourceVector():
            raise ValueError("a native execution profile must reserve resources")
        if isinstance(self.max_concurrent_decodes, bool) or not isinstance(
            self.max_concurrent_decodes, int
        ):
            raise TypeError("max_concurrent_decodes must be an integer")
        if self.max_concurrent_decodes not in (1, 2):
            raise ValueError("max_concurrent_decodes must be 1 or 2")
        _validate_profile_device(self.device)
        if self.device != "cpu":
            if self.max_concurrent_decodes != 1:
                raise ValueError("a CUDA profile must use one concurrent decode")
            if self.resources.memory_bytes <= 0:
                raise ValueError("a CUDA profile must reserve device memory")
            if self.resources.compute_units <= 0:
                raise ValueError("a CUDA profile must reserve compute capacity")
            if self.resources.stream_slots != 1:
                raise ValueError("a CUDA profile must reserve exactly one stream")

    @property
    def worker_capacity(self) -> ResourceVector:
        """Return the exact capacity required by this execution profile."""

        lanes = self.max_concurrent_decodes
        return ResourceVector(
            memory_bytes=self.resources.memory_bytes * lanes,
            compute_units=self.resources.compute_units * lanes,
            stream_slots=self.resources.stream_slots * lanes,
        )


def _require_isolated_task(task: object) -> None:
    """Require the patched built-in path with request-local decoder cache."""

    extension_probe = getattr(task, "_uses_legacy_extension", None)
    if not callable(extension_probe) or extension_probe() is not False:
        raise NativeDependencyError(
            "this execution profile requires the built-in suspendable decoder path"
        )
    inference = getattr(task, "inference", None)
    if getattr(inference, "_use_legacy_cache", None) is not False:
        raise NativeDependencyError(
            "this execution profile requires request-local decoder cache support"
        )


def _require_isolated_run(run: object) -> None:
    """Reject a run that fell back to shared hook-based cache state."""

    inference = getattr(run, "inference", None)
    if getattr(inference, "_use_legacy_cache", None) is not False:
        raise NativeDecodeContractError(
            "this execution profile requires request-local decoder cache state"
        )
    if getattr(run, "_legacy_cache_lock", None) is not None:
        raise NativeDecodeContractError(
            "this execution profile cannot use the legacy decoder cache lock"
        )


def _require_run_complete(run: NativeDecodeRun) -> bool:
    """Read the backend completion flag without accepting truthy substitutes."""

    value: object = run.complete
    if type(value) is not bool:
        raise NativeDecodeContractError("decode run complete must be a boolean")
    return value


def _validate_committed_boundary(value: int | None, *, end_ms: int) -> None:
    """Validate one caller-supplied finality boundary for a known window."""

    _validate_committed_through(value)
    if value is not None and value > end_ms:
        raise ValueError("committed_through_ms cannot exceed the result end")


def _require_model_available_without_wait(binding: ModelBinding) -> None:
    """Report retained work when visible without waiting behind live work."""

    if not binding.lock.acquire(blocking=False):
        return
    try:
        require_model_available(binding)
    finally:
        binding.lock.release()


_ResultT = TypeVar("_ResultT")


class _CpuDecodeScope:
    """Fence one synchronous CPU run and clean it before lease release."""

    def __init__(self, model_binding: ModelBinding) -> None:
        self._condition = Condition(RLock())
        self._model_binding = model_binding
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

    def invoke(self, operation: Callable[[], _ResultT]) -> _ResultT:
        """Run one synchronous CPU operation."""

        return operation()

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
            # A failed fence must become visible before another encoder
            # preparation can enter the model binding.
            with self._model_binding.lock:
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
                    self._model_binding.record_cleanup_failure(self)
                    raise
                else:
                    self._model_binding.clear_cleanup_failure(self)
        except BaseException:
            with self._condition:
                self._cleanup_in_flight = False
                self._condition.notify_all()
            raise
        else:
            with self._condition:
                self._run = None
                self._cleaned = True
                self._cleanup_in_flight = False
                self._condition.notify_all()


class _CudaLane:
    """Keep one stream for a model binding, with at most one exact borrower.

    The binding lock guards ownership. A successful transaction fence returns
    the lane, not its physical CUDA allocations; stream-local workspaces may
    remain resident for the binding's lifetime. Decoder state stays in scopes.
    """

    def __init__(self, cuda: _CudaRuntime, device: str, stream: object) -> None:
        self.cuda = cuda
        self.device = device
        self.stream = stream
        self.owner: object | None = None

    def require_owner(self, owner: object) -> None:
        if self.owner is not owner:
            raise NativeDecodeContractError("the CUDA lane has a different borrower")


class _CudaDecodeScope:
    """Exclusively borrow a CUDA lane until this transaction's fence succeeds."""

    def __init__(
        self,
        model_binding: ModelBinding,
        torch_module: _TorchModule,
        device: str,
    ) -> None:
        self._condition = Condition(RLock())
        self._model_binding = model_binding
        self._torch_module = torch_module
        self._device = device
        self._lease_token = object()
        self._lane: _CudaLane | None = None
        self._stream: object | None = None
        self._event: _CudaEvent | None = None
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
        """Latch a stop without calling CUDA from the cancelling thread."""

        with self._condition:
            self._stop_requested = True

    def completion_fence(self) -> _CudaDecodeScope:
        return self

    def _require_stream(self) -> object:
        with self._model_binding.lock:
            with self._condition:
                if self._cleaned:
                    raise NativeDecodeContractError(
                        "a closed execution scope cannot submit CUDA work"
                    )
                if self._lane is not None:
                    self._lane.require_owner(self._lease_token)
                    return self._lane.stream

            cuda = self._torch_module.cuda
            lane = self._model_binding._cuda_lane
            if lane is None:
                # Only the first borrower needs the model-initialization
                # barrier. Later borrowers inherit the prior successful fence
                # on this same stream; input mel is always copied from CPU.
                # Do not hold the condition while synchronizing: cancellation
                # only latches local state and must never wait on CUDA here.
                with cuda.device(self._device):
                    cuda.synchronize(self._device)
                    stream = cuda.Stream(device=self._device)
                lane = _CudaLane(cuda, self._device, stream)
                self._model_binding._cuda_lane = lane
            if not isinstance(lane, _CudaLane):
                raise NativeDecodeContractError("the model has an invalid CUDA lane")
            if lane.cuda is not cuda or lane.device != self._device:
                raise NativeDecodeContractError(
                    "the CUDA lane runtime and device cannot change"
                )
            if lane.owner is not None:
                raise NativeDecodeContractError("the CUDA lane is already borrowed")
            with self._condition:
                if self._cleaned:
                    raise NativeDecodeContractError(
                        "a closed execution scope cannot submit CUDA work"
                    )
                lane.owner = self._lease_token
                self._lane = lane
                self._stream = lane.stream
                return lane.stream

    def invoke(self, operation: Callable[[], _ResultT]) -> _ResultT:
        """Run one operation on this transaction's exact CUDA stream."""

        cuda = self._torch_module.cuda
        stream = self._require_stream()
        with cuda.device(self._device), cuda.stream(stream):
            return operation()

    def wait(self) -> None:
        """Clean the run, record its final event, and wait for device completion."""

        with self._condition:
            while self._cleanup_in_flight:
                self._condition.wait()
            if self._cleaned:
                return
            run = self._run
            stream = self._stream
            self._cleanup_in_flight = True

        try:
            with self._model_binding.lock:
                try:
                    if self._bound:
                        if run is None or stream is None:
                            raise NativeDecodeContractError(
                                "the CUDA decode handle cannot be fenced"
                            )
                        cleanup = getattr(run, "cleanup", None)
                        if not callable(cleanup):
                            raise NativeDecodeContractError(
                                "the decode handle has no callable cleanup method; "
                                "quiescence cannot be proven"
                            )
                        cleanup_result = self.invoke(cleanup)
                        if cleanup_result is not None:
                            close = getattr(cleanup_result, "close", None)
                            if callable(close):
                                close()
                            raise NativeDecodeContractError(
                                "decode cleanup did not complete synchronously; "
                                "quiescence cannot be proven"
                            )

                    if stream is not None:
                        lane = self._lane
                        if lane is None or lane.stream is not stream:
                            raise NativeDecodeContractError(
                                "the CUDA stream has no matching lane"
                            )
                        lane.require_owner(self._lease_token)
                        cuda = self._torch_module.cuda
                        with cuda.device(self._device), cuda.stream(stream):
                            event = cuda.Event(enable_timing=False)
                            self._event = event
                            event.record(stream)
                        event.synchronize()
                        # The lease is returned only after proven quiescence.
                        # Failed cleanup/record/wait keeps this exact borrower
                        # for recovery, never handing its stream to a successor.
                        lane.owner = None
                except BaseException:
                    self._model_binding.record_cleanup_failure(self)
                    raise
                else:
                    self._model_binding.clear_cleanup_failure(self)
                    with self._condition:
                        self._run = None
                        self._event = None
                        self._stream = None
                        self._lane = None
                        self._cleaned = True
                        self._cleanup_in_flight = False
                        self._condition.notify_all()
        except BaseException:
            with self._condition:
                self._cleanup_in_flight = False
                self._condition.notify_all()
            raise


NativeModelIdentityProbe = Callable[[object], ModelSnapshot]


def _alignment_number(name: str, value: object) -> float:
    """Return one finite backend alignment value without coercing strings."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"word alignment {name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"word alignment {name} must be finite")
    return number


def _alignment_words(
    raw_words: object,
    *,
    native: NativeWindowResult,
    text_tokens: tuple[int, ...],
) -> NativeWordAlignment:
    """Snapshot mutable backend word timings as bounded immutable values."""

    if not isinstance(raw_words, list):
        raise TypeError("find_alignment must return a list")
    duration_seconds = (native.end_ms - native.start_ms) / 1_000
    words: list[NativeTimestampSegment] = []
    previous_end_seconds = 0.0
    previous_end_ms = native.start_ms
    for raw in raw_words:
        word = getattr(raw, "word", None)
        tokens = getattr(raw, "tokens", None)
        if not isinstance(word, str):
            raise TypeError("word alignment text must be a string")
        if not isinstance(tokens, (tuple, list)):
            raise TypeError("word alignment tokens must be a list or tuple")
        token_tuple = tuple(tokens)
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_tuple
        ):
            raise ValueError(
                "word alignment tokens must be nonnegative integers, not booleans"
            )
        start_seconds = _alignment_number("start", getattr(raw, "start", None))
        end_seconds = _alignment_number("end", getattr(raw, "end", None))
        probability = _alignment_number(
            "probability", getattr(raw, "probability", None)
        )
        if not 0 <= probability <= 1:
            raise ValueError("word alignment probability must be between zero and one")
        if (
            start_seconds < 0
            or end_seconds < start_seconds
            or end_seconds > duration_seconds
        ):
            raise ValueError("word alignment times must stay within the analysis span")
        if start_seconds < previous_end_seconds:
            raise ValueError("word alignment times must not overlap")
        start_ms = native.start_ms + round(start_seconds * 1_000)
        end_ms = native.start_ms + round(end_seconds * 1_000)
        if start_ms < previous_end_ms:
            raise ValueError("rounded word alignment times must not overlap")
        words.append(
            NativeTimestampSegment(
                span=AudioSpan(start_ms, end_ms),
                text=word,
                tokens=token_tuple,
            )
        )
        previous_end_seconds = end_seconds
        previous_end_ms = end_ms

    if tuple(token for word in words for token in word.tokens) != text_tokens:
        raise ValueError(
            "word alignment tokens do not preserve the decoded token stream"
        )
    if "".join(word.text for word in words).strip() != native.text:
        raise ValueError("word alignment text does not match the native result")
    return NativeWordAlignment(native=native, words=tuple(words))


class NativeWindowRun:
    """A transaction-owned native decode that advances one token at a time.

    The thread that calls :meth:`NativeWhisperAdapter.start_window` owns the
    run and must call ``step()``, ``finish()``, and ``close()``. ``cancel()``
    and ``stop()`` may be called from another thread. ``cancel()`` records
    cooperative intent. ``stop()`` also reclaims the transaction when its
    owner has exited. Use the run as a context manager so an unfinished
    transaction is normally closed by its owner.
    """

    def __init__(
        self,
        *,
        worker: Worker,
        model_binding: ModelBinding,
        transaction: WindowTransaction,
        execution: _CpuDecodeScope | _CudaDecodeScope,
        backend_run: NativeDecodeRun,
        cuda_profile: bool,
        require_model_identity: Callable[[], None],
        complete: bool,
        window_id: str,
        start_ms: int,
        end_ms: int,
        tokenizer: NativeTokenizer | None = None,
        alignment_model: object | None = None,
        alignment_batched_mel: object | None = None,
        alignment_max_frames: int = 3_000,
        reuse_alignment_features: bool = False,
        alignment_float32: object | None = None,
    ) -> None:
        self._worker = worker
        self._model_binding = model_binding
        self._transaction = transaction
        self._execution = execution
        self._backend_run: NativeDecodeRun | None = backend_run
        self._cuda_profile = cuda_profile
        self._require_model_identity = require_model_identity
        self._window_id = window_id
        self._start_ms = start_ms
        self._end_ms = end_ms
        self._tokenizer = tokenizer
        self._alignment_model: object | None = alignment_model
        self._alignment_batched_mel: object | None = alignment_batched_mel
        self._alignment_max_frames = alignment_max_frames
        self._reuse_alignment_features = reuse_alignment_features
        self._alignment_float32 = alignment_float32
        self._alignment_audio_features: object | None = None
        self._alignment_reused_features: bool | None = None
        self._prepared_alignment: NativeWordAlignment | None = None
        self._prepared_result: NativeWindowResult | None = None
        self._step_count = 0
        self._complete = complete
        self._closed = False
        self._owner_thread = current_thread()

    def __enter__(self) -> NativeWindowRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, traceback
        if self.closed:
            return
        self._require_owner()
        if exc_value is None:
            self.close()
        else:
            self._close_owner(
                operation_error=exc_value,
                committed_state=None,
            )

    @property
    def request_id(self) -> str:
        """Return the request that owns this run."""

        return self._transaction.request_id

    @property
    def complete(self) -> bool:
        """Return whether the backend has produced its final token."""

        return self._complete

    @property
    def closed(self) -> bool:
        """Return whether driver methods can no longer advance this run."""

        if self._closed:
            return True
        return self._transaction.status in (
            TransactionStatus.COMMITTED,
            TransactionStatus.ABORTED,
            TransactionStatus.EXPIRED,
        )

    @property
    def capacity_released(self) -> bool:
        """Return whether this run no longer owns worker capacity."""

        return self._transaction.capacity_released

    @property
    def step_count(self) -> int:
        """Return the number of token steps submitted through this handle."""

        return self._step_count

    def step(self) -> bool:
        """Run at most one token-generation step and stop at a safe boundary."""

        self._require_open()
        if self.complete:
            return True
        try:
            # A caller can pause the run until the transaction deadline.
            # Recheck cancellation and expiry before admitting new work.
            self._transaction.checkpoint()
            backend_run = self._require_backend_run()
            reported_value: object = self._submit(backend_run.step)
            if type(reported_value) is not bool:
                raise NativeDecodeContractError("decode step must return a boolean")
            reported_complete = reported_value
            actual_complete = _require_run_complete(backend_run)
            if reported_complete is not actual_complete:
                raise NativeDecodeContractError(
                    "decode step completion disagrees with the run state"
                )
            self._step_count += 1
            self._complete = actual_complete
            self._transaction.checkpoint()
            return actual_complete
        except BaseException as operation_error:
            self._close_owner(
                operation_error=operation_error,
                committed_state=None,
            )
            raise

    def cancel(self) -> bool:
        """Return whether cancellation changed state or delivered its signal."""

        if self.closed:
            return False
        return self._worker.cancel(self._transaction)

    def stop(self) -> bool:
        """Return whether stop changed state, delivered a signal, or recovered."""

        if self.capacity_released:
            self._release_alignment_inputs()
            return False
        changed = self._worker.stop(self._transaction)
        if not self.capacity_released:
            changed = self._worker.recover(self._transaction) or changed
        if self.capacity_released:
            self._release_alignment_inputs()
        return changed

    def prepare_result(self) -> NativeWindowResult:
        """Inspect completed decoding before choosing what to publish.

        Finalization runs once. This snapshot does not publish text or release
        capacity. Finish or close the run before its existing deadline.
        """

        self._require_open()
        if not self.complete:
            raise NativeDecodeContractError(
                "a native result cannot be prepared before token generation completes"
            )
        try:
            self._transaction.checkpoint()
            result = self._prepare_result()
            if self._cuda_profile:
                self._submit(self._require_model_identity)
                self._transaction.checkpoint()
            else:
                with self._model_binding.lock:
                    self._require_model_identity()
            return result
        except BaseException as operation_error:
            self._close_owner(operation_error=operation_error, committed_state=None)
            raise

    def prepare_word_alignment(
        self, *, reuse_alignment_features: bool | None = None
    ) -> NativeWordAlignment:
        """Estimate immutable word bounds once for a completed native result.

        Alignment is opt in. It reuses the cached native result, runs on the
        transaction-owned execution scope, and never invokes native result
        finalization more than once.

        An opted-in profile may disable feature reuse for a control window.
        A default profile cannot enable it here. A cached alignment cannot
        change modes. Reuse does not imply parity with the legacy encoder path.
        """

        self._require_open()
        if (
            reuse_alignment_features is not None
            and type(reuse_alignment_features) is not bool
        ):
            raise TypeError("reuse_alignment_features must be a boolean or None")
        reuse = (
            self._reuse_alignment_features
            if reuse_alignment_features is None
            else reuse_alignment_features
        )
        if reuse and not self._reuse_alignment_features:
            raise ValueError("alignment feature reuse requires an opted-in profile")
        if (
            self._prepared_alignment is not None
            and reuse != self._alignment_reused_features
        ):
            raise ValueError("a cached word alignment cannot change feature reuse mode")
        if not self.complete:
            raise NativeDecodeContractError(
                "word alignment requires completed token generation"
            )
        try:
            self._transaction.checkpoint()
            native = self._prepare_result()
            if self._prepared_alignment is None:
                self._prepared_alignment = self._prepare_word_alignment(
                    native, reuse=reuse
                )
                self._alignment_reused_features = reuse
            if self._cuda_profile:
                self._submit(self._require_model_identity)
                self._transaction.checkpoint()
            else:
                with self._model_binding.lock:
                    self._require_model_identity()
            return self._prepared_alignment
        except BaseException as operation_error:
            self._close_owner(operation_error=operation_error, committed_state=None)
            raise

    def _prepare_word_alignment(
        self, native: NativeWindowResult, *, reuse: bool
    ) -> NativeWordAlignment:
        metadata = native.metadata
        tokenizer = self._tokenizer
        if metadata is None or tokenizer is None:
            raise NativeDecodeContractError(
                "word alignment requires native tokens and a tokenizer"
            )
        eot = getattr(tokenizer, "eot", None)
        if isinstance(eot, bool) or not isinstance(eot, int) or eot < 0:
            raise NativeDecodeContractError(
                "word alignment requires a nonnegative tokenizer eot token"
            )
        text_tokens = tuple(token for token in metadata.tokens if token < eot)
        decoded = tokenizer.decode(list(text_tokens))
        if not isinstance(decoded, str):
            raise NativeDecodeContractError("tokenizer.decode must return a string")
        if decoded.strip() != native.text:
            raise NativeDecodeContractError(
                "decoded text tokens do not match the native result"
            )
        duration_ms = self._end_ms - self._start_ms
        num_frames = min(self._alignment_max_frames, 3_000, (duration_ms + 9) // 10)
        if not text_tokens:
            if native.text:
                raise NativeDecodeContractError(
                    "an empty token stream must produce an empty native result"
                )
            return NativeWordAlignment(native=native, words=())
        if num_frames == 0:
            raise NativeDecodeContractError(
                "nonempty text cannot be aligned to an empty audio span"
            )
        model = self._alignment_model
        batched_mel = self._alignment_batched_mel
        if model is None or batched_mel is None:
            raise NativeDecodeContractError("word alignment input is unavailable")

        def align() -> NativeWordAlignment:
            with self._model_binding.lock:
                require_model_available(self._model_binding)
                self._require_model_identity()
                try:
                    model_module = import_module("whisper.model")
                    timing_module = import_module("whisper.timing")
                except (ImportError, OSError) as exc:
                    raise NativeDependencyError(
                        "word alignment requires the patched Whisper timing backend"
                    ) from exc
                local_probe = getattr(
                    model_module, "_uses_request_local_alignment", None
                )
                finder = getattr(timing_module, "find_alignment", None)
                if not callable(local_probe) or not callable(finder):
                    raise NativeDependencyError(
                        "the Whisper backend lacks request-local word alignment"
                    )
                if local_probe(model) is not True:
                    raise NativeDependencyError(
                        "word alignment requires request-local attention capture"
                    )
                feature_arguments: dict[str, object] = {}
                if reuse:
                    try:
                        parameter = signature(finder).parameters.get("audio_features")
                    except (TypeError, ValueError) as exc:
                        raise NativeDependencyError(
                            "alignment feature reuse requires the optional audio_features backend"
                        ) from exc
                    if parameter is None or parameter.kind not in (
                        Parameter.KEYWORD_ONLY,
                        Parameter.POSITIONAL_OR_KEYWORD,
                    ):
                        raise NativeDependencyError(
                            "alignment feature reuse requires the optional audio_features backend"
                        )
                    feature_arguments["audio_features"] = (
                        self._require_alignment_features(model)
                    )
                try:
                    mel = batched_mel[0]  # type: ignore[index]
                except (IndexError, KeyError, TypeError) as exc:
                    raise NativeDecodeContractError(
                        "the retained mel batch cannot provide one alignment input"
                    ) from exc
                raw_words = finder(
                    model,
                    tokenizer,
                    list(text_tokens),
                    mel,
                    num_frames,
                    **feature_arguments,
                )
                self._require_model_identity()
                try:
                    return _alignment_words(
                        raw_words,
                        native=native,
                        text_tokens=text_tokens,
                    )
                except (TypeError, ValueError) as exc:
                    raise NativeDecodeContractError(str(exc)) from exc

        aligned = self._submit(align)
        self._transaction.checkpoint()
        return aligned

    def _prepare_result(self) -> NativeWindowResult:
        if self._prepared_result is None:
            results = self._submit(self._require_backend_run().finalize)
            self._transaction.checkpoint()
            if not isinstance(results, list) or len(results) != 1:
                raise NativeDecodeContractError(
                    "native decoding must return exactly one result"
                )
            try:
                self._prepared_result = build_native_window_result(
                    results[0],
                    window_id=self._window_id,
                    analysis_span=AudioSpan(self._start_ms, self._end_ms),
                    tokenizer=self._tokenizer,
                )
                if self._reuse_alignment_features:
                    # Only the exact finalized result of this owned backend run
                    # can supply features. Never accept a caller-provided tensor.
                    self._alignment_audio_features = getattr(
                        results[0], "audio_features", None
                    )
            except (TypeError, ValueError) as exc:
                raise NativeDecodeContractError(str(exc)) from exc
        return self._prepared_result

    def _require_alignment_features(self, model: object) -> object:
        features = self._alignment_audio_features
        if features is None:
            raise NativeDecodeContractError(
                "the finalized result has no alignment audio_features"
            )
        dims = getattr(model, "dims", None)
        expected = (
            getattr(dims, "n_audio_ctx", None),
            getattr(dims, "n_audio_state", None),
        )
        if any(type(axis) is not int or axis <= 0 for axis in expected):
            raise NativeDecodeContractError(
                "the model lacks concrete encoder feature dimensions"
            )
        if getattr(features, "shape", None) != expected:
            raise NativeDecodeContractError(
                "alignment audio_features must match one unbatched model window"
            )
        # NativeDecodeOptions fixes FP32. The companion API also validates the
        # real tensor type; these owner-side checks bind precision and device.
        if (
            self._alignment_float32 is None
            or getattr(features, "dtype", None) != self._alignment_float32
        ):
            raise NativeDecodeContractError(
                "alignment audio_features must use native float32 precision"
            )
        try:
            _require_exact_device(
                features,
                subject="alignment audio_features",
                expected=str(getattr(model, "device", None)),
            )
        except ValueError as exc:
            raise NativeDecodeContractError(str(exc)) from exc
        return features

    def _require_backend_run(self) -> NativeDecodeRun:
        run = self._backend_run
        if run is None:
            raise NativeDecodeContractError("the native backend run has been released")
        return run

    def _release_alignment_inputs(self) -> None:
        """Drop borrowed inputs only after the transaction's fence has released."""

        self._alignment_model = None
        self._alignment_batched_mel = None
        self._alignment_audio_features = None
        if self._reuse_alignment_features:
            # Completed backend results themselves retain encoder tensors.
            # The execution scope has already dropped its run after fencing.
            self._backend_run = None

    def finish(
        self,
        *,
        committed_through_ms: int | None = None,
        publication_span: AudioSpan | None = None,
        aligned_publication: AlignedPublication | None = None,
        silence_publication: SilencePublication | None = None,
    ) -> SessionState:
        """Commit a complete run, optionally selecting whole timed segments.

        Analysis may include previously committed audio. A selected publication
        must match explicit segment bounds; it cannot rewrite committed output.
        Exact digital silence may instead publish empty, source-backed coverage
        while retaining this run's full native result as provenance.
        """

        self._require_open()
        if publication_span is not None and not isinstance(publication_span, AudioSpan):
            raise TypeError("publication_span must be an AudioSpan or None")
        if aligned_publication is not None and not isinstance(
            aligned_publication, AlignedPublication
        ):
            raise TypeError("aligned_publication must be an AlignedPublication or None")
        if silence_publication is not None and not isinstance(
            silence_publication, SilencePublication
        ):
            raise TypeError("silence_publication must be a SilencePublication or None")
        if (
            sum(
                value is not None
                for value in (
                    publication_span,
                    aligned_publication,
                    silence_publication,
                )
            )
            > 1
        ):
            raise ValueError(
                "publication_span, aligned_publication, and silence_publication "
                "cannot be used together"
            )
        if aligned_publication is not None:
            if aligned_publication.alignment is not self._prepared_alignment:
                raise ValueError(
                    "aligned_publication must use this run's cached word alignment"
                )
            if aligned_publication.window_id != self._window_id:
                raise ValueError("aligned_publication must match this run's window")
        if silence_publication is not None:
            if silence_publication.native is not self._prepared_result:
                raise ValueError(
                    "silence_publication must use this run's cached native result"
                )
            if silence_publication.window_id != self._window_id:
                raise ValueError("silence_publication must match this run's window")
        _validate_committed_boundary(
            committed_through_ms,
            end_ms=(
                silence_publication.end_ms
                if silence_publication is not None
                else aligned_publication.end_ms
                if aligned_publication is not None
                else self._end_ms
                if publication_span is None
                else publication_span.end_ms
            ),
        )
        try:
            self._transaction.checkpoint()
        except BaseException as operation_error:
            self._close_owner(
                operation_error=operation_error,
                committed_state=None,
            )
            raise
        if not self.complete:
            raise NativeDecodeContractError(
                "a native window cannot finish before token generation completes"
            )

        committed_state: SessionState | None = None
        try:
            native_result = self._prepare_result()
            result: WindowResult = native_result
            if silence_publication is not None:
                result = silence_publication
            elif aligned_publication is not None:
                result = aligned_publication
            elif publication_span is not None:
                try:
                    result = select_native_publication(native_result, publication_span)
                except (TypeError, ValueError) as exc:
                    raise NativeDecodeContractError(str(exc)) from exc
            if self._cuda_profile:
                self._submit(self._require_model_identity)
                self._transaction.checkpoint()
            else:
                with self._model_binding.lock:
                    self._require_model_identity()
            committed_state = self._transaction.commit(
                result,
                committed_through_ms=committed_through_ms,
            )
        except BaseException as operation_error:
            self._close_owner(
                operation_error=operation_error,
                committed_state=committed_state,
            )
            raise

        self._close_owner(
            operation_error=None,
            committed_state=committed_state,
        )
        return committed_state

    def close(self) -> bool:
        """Abort an unfinished run and release its resources after fencing."""

        if self.closed:
            return False
        self._require_owner()
        try:
            changed = self._transaction.abort()
        except BaseException as operation_error:
            self._close_owner(
                operation_error=operation_error,
                committed_state=None,
            )
            raise
        self._close_owner(operation_error=None, committed_state=None)
        return changed

    def _submit(self, operation: Callable[[], _ResultT]) -> _ResultT:
        return self._transaction.submit(lambda: self._execution.invoke(operation))

    def _require_open(self) -> None:
        if self.closed:
            raise NativeDecodeContractError("the native window run is closed")
        self._require_owner()

    def _require_owner(self) -> None:
        if current_thread() is not self._owner_thread:
            raise TransactionStateError(
                "the native window run must be driven by its starting thread"
            )

    def _close_owner(
        self,
        *,
        operation_error: BaseException | None,
        committed_state: SessionState | None,
    ) -> None:
        try:
            self._worker._finish_execution(
                self._transaction,
                operation_error=operation_error,
                committed_state=committed_state,
            )
        except TransactionRetainedError as retained:
            self._model_binding.record_retained_error(retained)
            raise
        finally:
            self._closed = True
            self._transaction._owner_departed()
            if self._transaction.capacity_released:
                self._release_alignment_inputs()


@dataclass(frozen=True, slots=True, init=False)
class NativeWhisperAdapter:
    """Bind a suspendable native decoder to one transactional worker."""

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
        concurrency = execution_profile.max_concurrent_decodes
        if worker.queue_capacity != concurrency:
            raise ValueError("a native worker queue must match max_concurrent_decodes")
        if execution_profile.worker_capacity != worker.budget.capacity:
            raise ValueError(
                "the worker capacity must equal the per-transaction resources "
                "times max_concurrent_decodes"
            )
        if execution_profile.device != "cpu":
            _require_exact_device(
                model,
                subject="the model",
                expected=execution_profile.device,
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
                concurrency=concurrency,
                device=execution_profile.device,
                execution_variant=(
                    "alignment_features/v1"
                    if execution_profile.reuse_alignment_features
                    else None
                ),
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

    def start_window(
        self,
        *,
        session: Session,
        request: RequestState,
        window_id: str,
        mel: object,
        start_ms: int,
        end_ms: int,
        options: NativeDecodeOptions | None = None,
    ) -> NativeWindowRun:
        """Start one native window and return after decoder prefill.

        The returned handle owns the admitted transaction. Drive and close it
        from this thread. Another thread may call ``cancel()`` for cooperative
        cancellation or ``stop()`` to fence and reclaim an orphaned run.
        """

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

        cuda_profile = self.execution_profile.device != "cpu"
        if not cuda_profile:
            _require_cpu_device(mel, subject="the mel tensor")
            _require_cpu_device(self._model, subject="the model")

        components = _load_native_components()
        if cuda_profile:
            torch_module = _require_cuda_runtime(
                components,
                device=self.execution_profile.device,
            )
            _require_exact_device(
                self._model,
                subject="the model",
                expected=self.execution_profile.device,
            )
            _require_cuda_input_mel(mel, torch_module)
            execution: _CpuDecodeScope | _CudaDecodeScope = _CudaDecodeScope(
                self._model_binding,
                torch_module,
                self.execution_profile.device,
            )
        else:
            execution = _CpuDecodeScope(self._model_binding)
        _require_mel_shape(mel, self._model, n_frames=components.n_frames)

        _require_model_available_without_wait(self._model_binding)
        try:
            transaction = self.worker.prepare(
                session=session,
                request=request,
                window_id=window_id,
                resources=self.execution_profile.resources,
            )
        except QueueFullError:
            # A quarantined model carries a more useful recovery handle than a
            # generic full-queue error. Never wait behind admitted model work
            # merely to refine this rejection.
            _require_model_available_without_wait(self._model_binding)
            raise
        owner_transferred = False
        try:
            transaction.start(execution)

            def submit_native(callback: Callable[[], _ResultT]) -> _ResultT:
                return transaction.submit(lambda: execution.invoke(callback))

            seed = transaction.randrange(1 << 63)

            def prepare_run() -> tuple[
                Callable[[object], object], object, NativeTokenizer | None
            ]:
                with self._model_binding.lock:
                    require_model_available(self._model_binding)
                    self._require_model_identity()
                    task, start = self._build_native_task(
                        components,
                        decode_options,
                        seed=seed,
                    )
                    if (
                        getattr(task, "tokenizer", None) is not None
                        and not decode_options.without_timestamps
                        and (
                            components.n_frames != 3_000
                            or getattr(
                                getattr(self._model, "dims", None), "n_audio_ctx", None
                            )
                            != 1_500
                        )
                    ):
                        raise NativeDependencyError(
                            "timestamp extraction requires the standard Whisper "
                            "3000-frame input and 1500-position audio context"
                        )
                    if (
                        cuda_profile
                        or self.execution_profile.max_concurrent_decodes == 2
                    ):
                        _require_isolated_task(task)
                    batched_mel = unsqueeze(0)
                    if cuda_profile:
                        to_device = getattr(batched_mel, "to", None)
                        if not callable(to_device):
                            raise NativeDecodeContractError(
                                "the batched mel tensor cannot be copied to CUDA"
                            )
                        batched_mel = to_device(
                            device=self.execution_profile.device,
                            non_blocking=False,
                        )
                        _require_exact_device(
                            batched_mel,
                            subject="the copied mel tensor",
                            expected=self.execution_profile.device,
                        )
                    return (
                        start,
                        batched_mel,
                        cast(NativeTokenizer | None, getattr(task, "tokenizer", None)),
                    )

            start, batched_mel, tokenizer = submit_native(prepare_run)
            transaction.checkpoint()

            def start_run() -> NativeDecodeRun:
                # _start_run performs encoder preparation. Keep that stage
                # serialized even when request-local decoder runs may overlap.
                with self._model_binding.lock:
                    require_model_available(self._model_binding)
                    run_value = start(batched_mel)
                    execution.bind(run_value)
                    if not isinstance(run_value, NativeDecodeRun):
                        raise NativeDecodeContractError(
                            "_start_run returned an incompatible decode handle"
                        )
                    if (
                        cuda_profile
                        or self.execution_profile.max_concurrent_decodes == 2
                    ):
                        _require_isolated_run(run_value)
                    return run_value

            run = submit_native(start_run)
            transaction.checkpoint()
            if _require_run_complete(run):
                raise NativeDecodeContractError(
                    "the decode run completed before its prefill stage"
                )

            submit_native(run.prefill)
            transaction.checkpoint()
            complete = _require_run_complete(run)
            handle = NativeWindowRun(
                worker=self.worker,
                model_binding=self._model_binding,
                transaction=transaction,
                execution=execution,
                backend_run=run,
                cuda_profile=cuda_profile,
                require_model_identity=self._require_model_identity,
                complete=complete,
                window_id=window_id,
                start_ms=start_ms,
                end_ms=end_ms,
                tokenizer=tokenizer,
                alignment_model=self._model,
                alignment_batched_mel=batched_mel,
                alignment_max_frames=components.n_frames,
                reuse_alignment_features=self.execution_profile.reuse_alignment_features,
                alignment_float32=getattr(components.torch_module, "float32", None),
            )
            owner_transferred = True
            return handle
        except BaseException as operation_error:
            try:
                self.worker._finish_execution(
                    transaction,
                    operation_error=operation_error,
                    committed_state=None,
                )
            except TransactionRetainedError as retained:
                self._model_binding.record_retained_error(retained)
                raise
            raise
        finally:
            if not owner_transferred:
                transaction._owner_departed()

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
        committed_through_ms: int | None = None,
        publication_span: AudioSpan | None = None,
    ) -> SessionState:
        """Decode and atomically commit one unbatched 30-second mel window."""

        WindowResult(window_id=window_id, text="", start_ms=start_ms, end_ms=end_ms)
        if publication_span is not None:
            if not isinstance(publication_span, AudioSpan):
                raise TypeError("publication_span must be an AudioSpan or None")
            WindowResult(
                window_id=window_id,
                text="",
                start_ms=publication_span.start_ms,
                end_ms=publication_span.end_ms,
                analysis_span=AudioSpan(start_ms, end_ms),
            )
        _validate_committed_boundary(
            committed_through_ms,
            end_ms=end_ms if publication_span is None else publication_span.end_ms,
        )
        with self.start_window(
            session=session,
            request=request,
            window_id=window_id,
            mel=mel,
            start_ms=start_ms,
            end_ms=end_ms,
            options=options,
        ) as run:
            while not run.complete:
                run.step()
            return run.finish(
                committed_through_ms=committed_through_ms,
                publication_span=publication_span,
            )

    def _require_model_identity(self) -> None:
        if self.execution_profile.device == "cpu":
            _require_cpu_device(self._model, subject="the model")
        else:
            _require_exact_device(
                self._model,
                subject="the model",
                expected=self.execution_profile.device,
            )
        identity = self._identity_probe(self._model)
        if not isinstance(identity, ModelSnapshot):
            raise TypeError("identity_probe must return ModelSnapshot")
        if identity != self._model_identity:
            raise ModelMismatchError("the loaded native model changed during decoding")

    def _build_native_task(
        self,
        components: _NativeComponents,
        options: NativeDecodeOptions,
        *,
        seed: int,
    ) -> tuple[_NativeDecodingTask, Callable[[object], object]]:
        try:
            generator = components.generator_type(device=self.execution_profile.device)
            generator_device = str(getattr(generator, "device", ""))
            if generator_device != self.execution_profile.device:
                raise NativeDependencyError(
                    "torch.Generator did not use the execution profile device"
                )
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
