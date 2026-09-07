"""Load the pinned local backend for the single-stream command.

The bootstrap manifest identifies backend source and dependency versions. It
does not certify this installed runtime or make the process a security sandbox.
Only an existing, checksum-verified tiny.en checkpoint is accepted. No setup,
package installation, model download, or subprocess inference is performed.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from . import Budget, ModelSnapshot, ResourceVector, Worker
from .adapters import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    NativeDecodeOptions,
    NativeExecutionProfile,
    NativeWhisperAdapter,
)
from .profiles import DEFAULT_PROFILE, get_profile

BACKEND_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
BACKEND_REUSE_TREE = "32163d5cdb87babc1cd415a86cc5a58116c86a16"
BACKEND_BASE = "86098128c0b4f24f0e2aa2994de830614b474227"
CHECKPOINT_SHA256 = "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
MODEL_FINGERPRINT = (
    "sha256:8041a80119a588f542472da35e97d0372fce1d9709ed9874475e9c03deac5de6"
)
DEPENDENCIES = {
    "torch": "2.6.0",
    "numpy": "2.5.2",
    "numba": "0.67.0",
    "tiktoken": "0.14.0",
    "more-itertools": "11.1.0",
    "tqdm": "4.70.0",
}
# Compatibility alias for existing experiment tools; settings remain unchanged.
CLI_STREAM_CONFIG = get_profile(DEFAULT_PROFILE).stream_config


class NativeSetupError(RuntimeError):
    """The supplied native setup is unavailable or differs from this profile."""


@dataclass(frozen=True)
class BackendSetup:
    path: Path
    revision: str


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeSetupError("setup manifest contains a duplicate field")
        result[key] = value
    return result


def _git(path: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeSetupError("cannot verify the local Whisper Git checkout") from exc


def validate_backend(
    manifest: Path, *, reuse_alignment_features: bool = False
) -> BackendSetup:
    """Check the explicitly selected source pin before importing the checkout.

    Feature reuse requires its own pinned tree. Neither mode applies a patch or
    accepts the other mode's tree, keeping the default source profile unchanged.
    """
    if type(reuse_alignment_features) is not bool:
        raise TypeError("reuse_alignment_features must be a boolean")
    expected_tree = BACKEND_REUSE_TREE if reuse_alignment_features else BACKEND_TREE
    try:
        with manifest.open("rb") as source:
            content = source.read(1_048_577)
        if len(content) > 1_048_576:
            raise NativeSetupError("setup manifest exceeds 1 MiB")
        document = json.loads(content, object_pairs_hook=_object)
    except (OSError, ValueError, UnicodeError) as exc:
        raise NativeSetupError("cannot read the setup manifest") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "2":
        raise NativeSetupError("expected a bootstrap version-2 setup manifest")
    record = document.get("backend")
    if not isinstance(record, dict) or (
        record.get("tree") != expected_tree
        or record.get("base_commit") != BACKEND_BASE
        or record.get("url") != "https://github.com/openai/whisper.git"
        or record.get("clean") is not True
    ):
        raise NativeSetupError("setup does not identify the pinned patched backend")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise NativeSetupError("backend path must be absolute")
    path = Path(path_value).resolve()
    if path != (manifest.resolve().parent / "backend").resolve():
        raise NativeSetupError("backend must belong to this bootstrap directory")
    if Path(_git(path, "rev-parse", "--show-toplevel")).resolve() != path:
        raise NativeSetupError("backend is not its own Git worktree")
    revision = _git(path, "rev-parse", "HEAD")
    if revision != record.get("applied_commit"):
        raise NativeSetupError("backend revision differs from the setup manifest")
    if _git(path, "rev-parse", "HEAD^{tree}") != expected_tree:
        raise NativeSetupError("backend source tree differs from the pinned tree")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise NativeSetupError("backend source contains uncommitted changes")
    ignored = _git(
        path, "ls-files", "--others", "--ignored", "--exclude-standard", "--", "whisper"
    )
    for name in ignored.splitlines():
        extra = Path(name)
        if "__pycache__" not in extra.parts or extra.suffix != ".pyc":
            raise NativeSetupError("backend contains untracked importable files")
    if not (path / "whisper" / "__init__.py").is_file():
        raise NativeSetupError("backend package is missing")
    return BackendSetup(path, revision)


def validate_dependencies() -> None:
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise NativeSetupError("the native profile requires Python 3.12 or 3.13")
    for name, expected in DEPENDENCIES.items():
        try:
            observed = importlib.metadata.version(name).split("+", 1)[0]
        except importlib.metadata.PackageNotFoundError as exc:
            raise NativeSetupError(
                f"missing {name}; run this command in the native setup environment"
            ) from exc
        if observed != expected:
            raise NativeSetupError(f"native profile requires {name}=={expected}")


def validate_checkpoint(path: Path) -> None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise NativeSetupError("provide an existing local tiny.en checkpoint") from exc
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise NativeSetupError(
            "checkpoint does not match the supported tiny.en weights"
        )


def _import_backend(setup: BackendSetup, *, eager_cuda: bool = False) -> ModuleType:
    module_path = setup.path / "whisper" / "__init__.py"
    if any(name == "whisper" or name.startswith("whisper.") for name in sys.modules):
        raise NativeSetupError(
            "start a fresh process before loading the native backend"
        )
    spec = importlib.util.spec_from_file_location(
        "whisper", module_path, submodule_search_locations=[str(module_path.parent)]
    )
    if spec is None or spec.loader is None:
        raise NativeSetupError("cannot load the verified Whisper package")
    module = importlib.util.module_from_spec(spec)
    previous_prefix, previous_write = sys.pycache_prefix, sys.dont_write_bytecode
    try:
        # Do not execute stale bytecode from an earlier checkout or installation.
        with tempfile.TemporaryDirectory(prefix="whisper-cli-import-") as cache:
            sys.pycache_prefix, sys.dont_write_bytecode = cache, True
            sys.modules["whisper"] = module
            spec.loader.exec_module(module)
            if eager_cuda:
                # Alignment otherwise imports this module lazily after the fresh
                # cache guard has ended. Missing CUDA prerequisites fail before
                # loading any checkpoint; no numerical fallback is substituted.
                importlib.import_module("whisper.triton_ops")
    except BaseException:
        for name in tuple(sys.modules):
            if name == "whisper" or name.startswith("whisper."):
                del sys.modules[name]
        raise
    finally:
        sys.pycache_prefix, sys.dont_write_bytecode = previous_prefix, previous_write
    return module


def _fingerprint(model: Any) -> str:
    """Use the same field encoding as the recorded native model fingerprints."""
    state = model.state_dict()
    if not isinstance(state, Mapping):
        raise NativeSetupError("loaded model has no tensor state mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        for field in (str(name), str(value.dtype), repr(tuple(value.shape))):
            encoded = field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(value.numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _load_model(whisper: ModuleType, checkpoint: Path, device: str) -> Any:
    """Keep path-only loading and the named tiny.en legacy alignment mask."""
    masks = getattr(whisper, "_ALIGNMENT_HEADS", None)
    if not isinstance(masks, Mapping) or not isinstance(masks.get("tiny.en"), bytes):
        raise NativeSetupError("verified backend lacks the tiny.en alignment mask")
    # The backend's local-path branch deliberately does not install named-model
    # alignment heads. They are a nonpersistent buffer, absent from state_dict
    # fingerprints. Restore the verified mask before moving every buffer to CUDA.
    loaded = whisper.load_model(str(checkpoint.resolve()), device="cpu")
    loaded.set_alignment_heads(masks["tiny.en"])
    return loaded.to(device).float().eval()


def _prepare_alignment_backtrace() -> dict[str, object]:
    """Compile the pinned CUDA aligner's CPU backtrace before accepting audio.

    CUDA DTW returns a two-dimensional int32 trace. Its CPU copy can be dense
    or strided. Compile both layouts without running inference, allocating GPU
    tensors or changing the decoder. Triton kernels remain lazy.
    """
    timing = sys.modules.get("whisper.timing")
    compile_trace = getattr(getattr(timing, "backtrace", None), "compile", None)
    if not callable(compile_trace):
        raise NativeSetupError(
            "verified backend lacks the alignment backtrace compiler"
        )
    started = time.perf_counter()
    numba = importlib.import_module("numba")
    for layout in ("C", "A"):
        compile_trace((numba.types.Array(numba.types.int32, 2, layout),))
    return {
        "id": "cuda-backtrace-init-v1",
        "signatures": ["int32[:,::1]", "int32[:,:]"],
        "wall_seconds": time.perf_counter() - started,
        "model_decodes": 0,
        "triton_kernels_warmed": False,
    }


def create_stream(
    *,
    manifest: Path,
    model: Path,
    device: str = "cpu",
    profile: str | None = None,
    reuse_alignment_features: bool = False,
    config: ContinuousStreamConfig | None = None,
) -> ContinuousTranscriptStream:
    """Construct the English, FP32, single-lane CLI profile without downloads.

    The resource vector is admission accounting, not an enforced VRAM limit.
    The caller owns the stream and must close it, including on input failures.
    Omitted profile selects conservative-v1. An explicit registered profile may
    not be combined with a custom config or enabled feature-reuse override.
    Legacy custom config/reuse arguments remain experimental when profile is
    omitted. Reuse requires the separate, existing pinned backend in both APIs.
    """
    if type(reuse_alignment_features) is not bool:
        raise TypeError("reuse_alignment_features must be a boolean")
    if config is not None and not isinstance(config, ContinuousStreamConfig):
        raise TypeError("config must be ContinuousStreamConfig or None")
    selected = get_profile(DEFAULT_PROFILE if profile is None else profile)
    if profile is not None and (config is not None or reuse_alignment_features):
        raise ValueError(
            "profile excludes config and reuse_alignment_features overrides"
        )
    if profile is not None:
        reuse_alignment_features = selected.reuse_alignment_features
    stream_config = selected.stream_config if config is None else config
    native_profile_id = selected.native_profile_id
    if device not in ("cpu", "cuda:0"):
        raise NativeSetupError("supported devices are cpu and cuda:0")
    setup = validate_backend(
        manifest, reuse_alignment_features=reuse_alignment_features
    )
    validate_checkpoint(model)
    validate_dependencies()
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    if device == "cuda:0" and not torch.cuda.is_available():
        raise NativeSetupError("CUDA is unavailable; select cpu or a GPU worker")
    whisper = _import_backend(setup, eager_cuda=device == "cuda:0")
    torch.set_num_threads(1)
    loaded = _load_model(whisper, model, device)
    fingerprint = _fingerprint(loaded)
    if fingerprint != MODEL_FINGERPRINT:
        raise NativeSetupError(
            "loaded model state differs from the recorded tiny.en model"
        )
    if device == "cuda:0":
        _prepare_alignment_backtrace()
    identity = ModelSnapshot(
        "tiny.en", setup.revision, f"pytorch-{device}", fingerprint
    )
    capacity = ResourceVector(
        memory_bytes=2_147_483_648, compute_units=1, stream_slots=1
    )
    worker = Worker(
        "caption-cli",
        identity,
        Budget(capacity),
        queue_capacity=1,
        transaction_ttl_seconds=180,
    )

    def probe(observed: object) -> ModelSnapshot:
        if observed is not loaded:
            raise NativeSetupError("native model identity changed")
        # CUDA's binding also checks tensor layout/version stamps in the adapter.
        if device == "cpu" and _fingerprint(loaded) != identity.fingerprint:
            raise NativeSetupError("native model state changed")
        return identity

    def mel_builder(content: bytes) -> object:
        samples = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768.0
        return whisper.log_mel_spectrogram(
            whisper.pad_or_trim(samples), n_mels=loaded.dims.n_mels
        ).contiguous()

    adapter = NativeWhisperAdapter(
        worker,
        loaded,
        probe,
        NativeExecutionProfile(
            native_profile_id,
            capacity,
            device=device,
            reuse_alignment_features=reuse_alignment_features,
        ),
    )
    return ContinuousTranscriptStream(
        adapter,
        stream_id="caption-cli",
        mel_builder=mel_builder,
        options=NativeDecodeOptions(language="en", without_timestamps=False),
        rng_seed=7,
        config=stream_config,
    )
