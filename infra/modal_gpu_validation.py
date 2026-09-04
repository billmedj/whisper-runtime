"""Run the bounded CUDA-readiness check on a Modal T4.

This check does not exercise the transactional runtime adapter. It verifies one
pinned patched Whisper backend, checkpoint, and audio fixture on CUDA. See
``docs/MODAL_GPU_VALIDATION.md`` before starting a paid run.
"""

from __future__ import annotations

import datetime as dt
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "whisper-runtime-cuda-readiness"
MODAL_SDK_VERSION = "1.5.5"
RUNTIME_REPOSITORY = "https://github.com/billmedj/whisper-runtime.git"
BACKEND_REPOSITORY = "https://github.com/openai/whisper.git"
BACKEND_BASE_COMMIT = "86098128c0b4f24f0e2aa2994de830614b474227"
BACKEND_BASE_TREE = "f7b3cb8e12a2e84dccacc4c858c33d5a9c114688"
BACKEND_PATCHED_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
PATCH_MANIFEST_SHA256 = (
    "0fa1a833b0c489056d77da21188519c7e16fde7825d06bc5b902ba23a01abeb5"
)
NATIVE_ADAPTER_SHA256 = (
    "3388992843384d2a4259588e9bc0e22dd971b7fd2fe4162e515330ef8b480d4c"
)
MODEL_NAME = "tiny.en"
MODEL_CHECKPOINT_SHA256 = (
    "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
)
MODEL_STATE_SHA256 = (
    "sha256:8041a80119a588f542472da35e97d0372fce1d9709ed9874475e9c03deac5de6"
)
MODEL_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    f"{MODEL_CHECKPOINT_SHA256}/tiny.en.pt"
)
MODEL_CACHE_NAME = "whisper-runtime-model-cache-v1"
MODEL_CACHE_GENERATION = 1
MODEL_CACHE_MOUNT = "/models"
MODEL_CHECKPOINT_PATH = Path(MODEL_CACHE_MOUNT) / "tiny.en.pt"
AUDIO_FIXTURE_ID = "openai-whisper-jfk-flac"
AUDIO_RELATIVE_PATH = "openai-whisper/tests/jfk.flac"
AUDIO_PATH = Path("/opt") / AUDIO_RELATIVE_PATH
AUDIO_SHA256 = "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
AUDIO_SIZE_BYTES = 1_152_693
AUDIO_SAMPLE_RATE_HZ = 16_000
AUDIO_SAMPLE_COUNT = 176_000
EXPECTED_TEXT = (
    "And so my fellow Americans ask not what your country can do for you, "
    "ask what you can do for your country."
)
GPU_REQUEST = "T4"
EXPECTED_COMPUTE_CAPABILITY = (7, 5)
RNG_SEED = 7
PATCH_FILES = tuple(f"{index:04d}-" for index in range(1, 8))


def _required_runtime_commit() -> str:
    value = os.environ.get("WHISPER_RUNTIME_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError(
            "WHISPER_RUNTIME_COMMIT must be the full public commit to validate"
        )
    return value


RUNTIME_COMMIT = _required_runtime_commit()


def _require_definition_opt_in() -> None:
    if os.environ.get("WHISPER_MODAL_ENABLE_REMOTE_RESOURCES") != "1":
        raise RuntimeError(
            "set WHISPER_MODAL_ENABLE_REMOTE_RESOURCES=1 before constructing "
            "the Modal app"
        )


def _require_matching_clean_checkout() -> None:
    root = Path(__file__).resolve().parents[1]

    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError("the validation source Git identity is unavailable")
        return result.stdout.strip()

    if run("rev-parse", "HEAD") != RUNTIME_COMMIT:
        raise RuntimeError("the local checkout does not match WHISPER_RUNTIME_COMMIT")
    if run("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("the validation source checkout must be clean")


if os.environ.get("MODAL_IS_REMOTE") != "1":
    _require_definition_opt_in()
_require_matching_clean_checkout()


def _build_command() -> str:
    patch_directory = "/opt/whisper-runtime/patches/openai-whisper"
    apply_commands = " && ".join(
        f"git am --committer-date-is-author-date {patch_directory}/{prefix}*.patch"
        for prefix in PATCH_FILES
    )
    return " && ".join(
        (
            "git init --quiet /opt/whisper-runtime",
            "git -C /opt/whisper-runtime remote add origin " + RUNTIME_REPOSITORY,
            "git -C /opt/whisper-runtime fetch --depth=1 origin " + RUNTIME_COMMIT,
            "git -C /opt/whisper-runtime checkout --detach FETCH_HEAD",
            'test "$(git -C /opt/whisper-runtime rev-parse HEAD)" = ' + RUNTIME_COMMIT,
            f"cd {patch_directory} && sha256sum --check SHA256SUMS",
            'test "$(sha256sum /opt/whisper-runtime/patches/openai-whisper/'
            "SHA256SUMS | cut -d' ' -f1)\" = " + PATCH_MANIFEST_SHA256,
            "git init --quiet /opt/openai-whisper",
            "git -C /opt/openai-whisper remote add origin " + BACKEND_REPOSITORY,
            "git -C /opt/openai-whisper fetch --depth=1 origin " + BACKEND_BASE_COMMIT,
            "git -C /opt/openai-whisper checkout --detach FETCH_HEAD",
            "test \"$(git -C /opt/openai-whisper rev-parse 'HEAD^{tree}')\" = "
            + BACKEND_BASE_TREE,
            "git -C /opt/openai-whisper config user.name 'Whisper Runtime Check'",
            "git -C /opt/openai-whisper config user.email "
            "'whisper-runtime-check@users.noreply.github.com'",
            f"cd /opt/openai-whisper && {apply_commands}",
            "test \"$(git -C /opt/openai-whisper rev-parse 'HEAD^{tree}')\" = "
            + BACKEND_PATCHED_TREE,
            'test -z "$(git -C /opt/openai-whisper status --porcelain '
            '--untracked-files=all)"',
            "python -m pip install --no-deps --no-build-isolation /opt/whisper-runtime",
            "python -m pip check",
        )
    )


image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ca-certificates", "ffmpeg", "git")
    .pip_install(
        "torch==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .uv_pip_install(
        "jsonschema==4.25.1",
        "more-itertools==11.1.0",
        "numba==0.67.0",
        "numpy==2.5.2",
        "setuptools==82.0.1",
        "tiktoken==0.14.0",
        "tqdm==4.70.0",
    )
    .run_commands(_build_command())
    .env(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": (
                "/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime"
            ),
            "PYTHONUTF8": "1",
            "WHISPER_RUNTIME_COMMIT": RUNTIME_COMMIT,
        }
    )
)

model_cache = modal.Volume.from_name(MODEL_CACHE_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str, root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git identity check failed for {root.name}")
    return result.stdout.strip()


def _source_identity(root: Path) -> dict[str, object]:
    status = _git("status", "--porcelain", "--untracked-files=all", root=root)
    if status:
        raise RuntimeError(f"the {root.name} source tree is not clean")
    return {
        "git_commit": _git("rev-parse", "HEAD", root=root),
        "git_tree": _git("rev-parse", "HEAD^{tree}", root=root),
        "clean": True,
    }


def _model_fingerprint(model: object) -> str:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("the model does not expose state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("model.state_dict() did not return a mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        for field in (str(name), str(value.dtype), repr(tuple(value.shape))):
            encoded = field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(value.numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _hook_fingerprint(model: object) -> tuple[tuple[str, str, str, int], ...]:
    records: list[tuple[str, str, str, int]] = []
    for module_name, module in model.named_modules():
        for category, attribute in (
            ("forward", "_forward_hooks"),
            ("forward_pre", "_forward_pre_hooks"),
            ("backward", "_backward_hooks"),
        ):
            records.extend(
                (module_name, category, repr(key), id(callback))
                for key, callback in getattr(module, attribute, {}).items()
            )
    return tuple(sorted(records))


def _finish_staged_run(task: object, mel: object) -> tuple[object, int]:
    run = task._start_run(mel.unsqueeze(0))
    cleanup_calls = 0
    cleanup_complete = False
    try:
        if not run.complete:
            run.prefill()
        while not run.complete:
            run.step()
        results = run.finalize()
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError("the staged decoder did not return one result")
        run.cleanup()
        cleanup_calls += 1
        run.cleanup()
        cleanup_calls += 1
        cleanup_complete = True
        return results[0], cleanup_calls
    finally:
        if not cleanup_complete:
            run.cleanup()


def _probe_blocked_network() -> dict[str, object]:
    try:
        connection = socket.create_connection(("1.1.1.1", 443), timeout=1.0)
    except OSError as error:
        return {
            "target": "1.1.1.1:443",
            "denied": True,
            "exception_type": type(error).__name__,
            "errno": error.errno,
        }
    connection.close()
    raise RuntimeError("the GPU function could open an outbound network connection")


def _probe_read_only_model_cache() -> dict[str, object]:
    probe = Path(MODEL_CACHE_MOUNT) / ".read-only-probe"
    try:
        probe.write_bytes(b"probe")
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
            raise RuntimeError(
                "the model-cache write probe failed for an unexpected reason"
            ) from error
        return {
            "denied": True,
            "exception_type": type(error).__name__,
            "errno": error.errno,
        }
    probe.unlink(missing_ok=True)
    raise RuntimeError("the GPU function could write to the model-cache volume")


def _command_first_line(*arguments: str) -> str:
    result = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or not lines or not lines[0].strip():
        raise RuntimeError(f"version command failed: {arguments[0]}")
    return lines[0].strip()


def _decoded_pcm_fingerprint(audio: object) -> str:
    value = audio.copy(order="C")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _resource_vector(value: object) -> dict[str, int]:
    return {
        "memory_bytes": value.memory_bytes,
        "compute_units": value.compute_units,
        "stream_slots": value.stream_slots,
    }


@app.function(
    image=image,
    volumes={MODEL_CACHE_MOUNT: model_cache},
    cpu=2.0,
    memory=4096,
    timeout=600,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    include_source=False,
)
def prime_model_cache() -> dict[str, object]:
    """Populate the versioned cache and verify the checkpoint before commit."""

    destination = MODEL_CHECKPOINT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256_file(destination) == MODEL_CHECKPOINT_SHA256:
        return {
            "cache_generation": MODEL_CACHE_GENERATION,
            "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
            "downloaded": False,
        }

    temporary = destination.with_suffix(".pt.partial")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"checkpoint download returned HTTP {response.status}"
                )
            with temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        observed = _sha256_file(temporary)
        if observed != MODEL_CHECKPOINT_SHA256:
            raise RuntimeError(
                "checkpoint digest mismatch: "
                f"expected {MODEL_CHECKPOINT_SHA256}, observed {observed}"
            )
        os.replace(temporary, destination)
        model_cache.commit()
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "cache_generation": MODEL_CACHE_GENERATION,
        "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "downloaded": True,
    }


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    volumes={
        MODEL_CACHE_MOUNT: model_cache.with_mount_options(read_only=True),
    },
    cpu=2.0,
    memory=4096,
    timeout=900,
    startup_timeout=900,
    retries=0,
    max_containers=1,
    block_network=True,
    restrict_modal_access=True,
    single_use_containers=True,
    include_source=False,
)
def validate_cuda_backend() -> dict[str, object]:
    """Return one closed evidence record for the direct patched CUDA backend."""

    import numpy
    import torch
    import whisper
    from whisper.audio import SAMPLE_RATE
    from whisper.decoding import DecodingOptions, DecodingTask

    from whisper_runtime import (
        Budget,
        ModelSnapshot,
        RequestState,
        ResourceVector,
        Session,
        Worker,
    )
    from whisper_runtime.adapters import NativeExecutionProfile, NativeWhisperAdapter

    started = time.perf_counter()
    network_probe = _probe_blocked_network()
    model_cache_probe = _probe_read_only_model_cache()

    runtime_root = Path("/opt/whisper-runtime")
    backend_root = Path("/opt/openai-whisper")
    runtime = _source_identity(runtime_root)
    backend = _source_identity(backend_root)
    if runtime["git_commit"] != RUNTIME_COMMIT:
        raise RuntimeError(
            "the runtime commit differs from the requested public commit"
        )
    if backend["git_tree"] != BACKEND_PATCHED_TREE:
        raise RuntimeError("the patched backend tree differs from the pinned tree")
    if (
        not _git(
            "merge-base",
            "--is-ancestor",
            BACKEND_BASE_COMMIT,
            str(backend["git_commit"]),
            root=backend_root,
        )
        == ""
    ):
        raise RuntimeError("unexpected output from the backend ancestry check")
    patch_manifest = runtime_root / "patches/openai-whisper/SHA256SUMS"
    if _sha256_file(patch_manifest) != PATCH_MANIFEST_SHA256:
        raise RuntimeError("the patch manifest differs from its pinned digest")
    native_adapter = runtime_root / "src/whisper_runtime/adapters/native_whisper.py"
    if _sha256_file(native_adapter) != NATIVE_ADAPTER_SHA256:
        raise RuntimeError("the native adapter differs from the pinned CPU-only source")
    if _sha256_file(MODEL_CHECKPOINT_PATH) != MODEL_CHECKPOINT_SHA256:
        raise RuntimeError("the cached checkpoint differs from its pinned digest")
    if _sha256_file(AUDIO_PATH) != AUDIO_SHA256:
        raise RuntimeError("the audio fixture differs from its pinned digest")
    if AUDIO_PATH.stat().st_size != AUDIO_SIZE_BYTES:
        raise RuntimeError("the audio fixture size differs from its manifest")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the check requires exactly one visible CUDA device")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    if "T4" not in device_name or capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "the allocated GPU is not the requested T4: "
            f"name={device_name!r}, capability={capability!r}"
        )

    model = whisper.load_model(
        MODEL_NAME,
        device="cuda",
        download_root=MODEL_CACHE_MOUNT,
    ).eval()
    if str(model.device) != "cuda:0":
        raise RuntimeError(f"the model loaded on an unexpected device: {model.device}")
    state_before = _model_fingerprint(model)
    if state_before != MODEL_STATE_SHA256:
        raise RuntimeError("the loaded model state differs from its pinned fingerprint")
    hooks_before = _hook_fingerprint(model)

    audio = whisper.load_audio(str(AUDIO_PATH))
    if SAMPLE_RATE != AUDIO_SAMPLE_RATE_HZ or len(audio) != AUDIO_SAMPLE_COUNT:
        raise RuntimeError("the decoded audio differs from its manifest")
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
    ).to("cuda")

    def options() -> DecodingOptions:
        return DecodingOptions(
            language="en",
            task="transcribe",
            temperature=0.0,
            without_timestamps=True,
            fp16=True,
            generator=torch.Generator(device="cuda").manual_seed(RNG_SEED),
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()

    staged_start = torch.cuda.Event(enable_timing=True)
    staged_end = torch.cuda.Event(enable_timing=True)
    staged_start.record()
    result, cleanup_calls = _finish_staged_run(DecodingTask(model, options()), mel)
    staged_end.record()
    staged_end.synchronize()
    staged_seconds = staged_start.elapsed_time(staged_end) / 1000.0
    if result.text != EXPECTED_TEXT:
        raise RuntimeError(
            f"unexpected transcript: expected {EXPECTED_TEXT!r}, observed {result.text!r}"
        )

    reuse_start = torch.cuda.Event(enable_timing=True)
    reuse_end = torch.cuda.Event(enable_timing=True)
    reuse_start.record()
    reuse, _reuse_cleanup_calls = _finish_staged_run(
        DecodingTask(model, options()), mel
    )
    reuse_end.record()
    reuse_end.synchronize()
    reuse_seconds = reuse_start.elapsed_time(reuse_end) / 1000.0
    if reuse.text != EXPECTED_TEXT:
        raise RuntimeError("the model did not reproduce the pinned transcript")
    if _reuse_cleanup_calls != 2:
        raise RuntimeError("the reuse decode did not complete idempotent cleanup")

    state_after = _model_fingerprint(model)
    hooks_after = _hook_fingerprint(model)
    if state_after != state_before or hooks_after != hooks_before:
        raise RuntimeError("the staged decode changed persistent model state or hooks")

    snapshot = ModelSnapshot(
        model_id=MODEL_NAME,
        revision=str(backend["git_commit"]),
        backend="pytorch-cuda-boundary",
        fingerprint=state_before,
    )
    capacity = ResourceVector(
        memory_bytes=1_000_000_000,
        compute_units=1,
        stream_slots=1,
    )
    budget = Budget(capacity)
    worker = Worker(
        "modal-cuda-boundary",
        snapshot,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=300,
    )

    def identity_probe(observed: object) -> ModelSnapshot:
        return ModelSnapshot(
            model_id=MODEL_NAME,
            revision=str(backend["git_commit"]),
            backend="pytorch-cuda-boundary",
            fingerprint=_model_fingerprint(observed),
        )

    adapter = NativeWhisperAdapter(
        worker,
        model,
        identity_probe,
        NativeExecutionProfile("tiny.en/cuda-boundary", capacity),
    )
    session = Session("modal-cuda-boundary")
    request = RequestState(
        "modal-cuda-boundary-1",
        session.session_id,
        snapshot,
        rng_seed=RNG_SEED,
    )
    queue_depth_before = worker.queue_depth
    budget_before = budget.available
    rejection_type = ""
    rejection_message = ""
    try:
        adapter.decode_window(
            session=session,
            request=request,
            window_id="window-0",
            mel=mel,
            start_ms=0,
            end_ms=11_000,
        )
    except ValueError as error:
        rejection_type = type(error).__name__
        rejection_message = str(error)
    else:
        raise RuntimeError("the CPU-only adapter accepted a CUDA tensor")
    expected_rejection = (
        "the native adapter requires an explicit CPU device for the mel tensor"
    )
    if rejection_message != expected_rejection:
        raise RuntimeError(
            f"the adapter rejected for an unexpected reason: {rejection_message}"
        )
    if worker.queue_depth != queue_depth_before or budget.available != budget_before:
        raise RuntimeError("the rejected adapter call changed admission state")

    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    recorded_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    assertions = {
        "runtime_source_pinned": True,
        "backend_source_pinned": True,
        "patch_manifest_verified": True,
        "checkpoint_verified_before_load": True,
        "input_fixture_verified": True,
        "torch_cuda_available": True,
        "requested_t4_observed": True,
        "model_loaded_on_cuda": True,
        "staged_decode_completed": True,
        "decode_result_matches_expected": True,
        "cleanup_idempotent": cleanup_calls == 2 and _reuse_cleanup_calls == 2,
        "persistent_model_state_unchanged": state_after == state_before,
        "model_hooks_unchanged": hooks_after == hooks_before,
        "model_reusable_after_decode": reuse.text == EXPECTED_TEXT,
        "cuda_timing_synchronized": True,
        "torch_cuda_memory_measured": peak_allocated >= allocated_before,
        "native_runtime_adapter_rejected_before_admission": True,
        "network_block_configured": True,
        "outbound_network_probe_denied": network_probe["denied"] is True,
        "model_cache_read_only": model_cache_probe["denied"] is True,
    }
    if any(value is not True for value in assertions.values()):
        failed = sorted(name for name, value in assertions.items() if value is not True)
        raise RuntimeError(f"CUDA readiness assertions failed: {', '.join(failed)}")

    function_call_id = modal.current_function_call_id()
    if not isinstance(function_call_id, str) or not function_call_id:
        raise RuntimeError("Modal did not expose a function-call identifier")
    observed_modal_sdk = str(modal.__version__)
    if observed_modal_sdk != MODAL_SDK_VERSION:
        raise RuntimeError(
            f"Modal SDK mismatch: expected {MODAL_SDK_VERSION}, observed {observed_modal_sdk}"
        )

    return {
        "schema_version": "1",
        "recorded_at": recorded_at,
        "status": "passed",
        "scope": {
            "evidence_kind": "patched-whisper-cuda-readiness",
            "statement": (
                "One pinned patched Whisper backend decoded one pinned fixture "
                "on one Modal T4. No runtime transaction was admitted or executed."
            ),
            "runtime_adapter_exercised": False,
        },
        "claims": {
            "patched_backend_cuda_decode": True,
            "runtime_adapter_exercised": False,
            "worker_admission_exercised": False,
            "transaction_lifecycle_exercised": False,
            "cuda_completion_fence_exercised": False,
            "performance_benchmark": False,
        },
        "runtime": {
            "repository": RUNTIME_REPOSITORY,
            "native_adapter_path": "src/whisper_runtime/adapters/native_whisper.py",
            "native_adapter_sha256": NATIVE_ADAPTER_SHA256,
            **runtime,
        },
        "backend": {
            "repository": BACKEND_REPOSITORY,
            "base_commit": BACKEND_BASE_COMMIT,
            "base_tree": BACKEND_BASE_TREE,
            "applied_commit": backend["git_commit"],
            "git_tree": backend["git_tree"],
            "clean": backend["clean"],
            "patch_manifest": "patches/openai-whisper/SHA256SUMS",
            "patch_manifest_sha256": PATCH_MANIFEST_SHA256,
        },
        "modal": {
            "sdk_version": observed_modal_sdk,
            "function_call_id": function_call_id,
            "image_id": os.environ.get("MODAL_IMAGE_ID"),
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "environment": os.environ.get("MODAL_ENVIRONMENT"),
            "cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
            "region": os.environ.get("MODAL_REGION"),
            "network_blocked": True,
            "modal_access_restricted": True,
            "model_cache": {
                "name": MODEL_CACHE_NAME,
                "generation": MODEL_CACHE_GENERATION,
                "mount_path": MODEL_CACHE_MOUNT,
                "read_only": True,
                "write_probe": model_cache_probe,
            },
            "network_probe": network_probe,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
            "nvidia_driver": _command_first_line(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ),
            "ffmpeg": _command_first_line("ffmpeg", "-version"),
            "numpy": str(numpy.__version__),
            "tiktoken": str(importlib.metadata.version("tiktoken")),
            "numba": str(importlib.metadata.version("numba")),
            "tqdm": str(importlib.metadata.version("tqdm")),
            "more_itertools": str(importlib.metadata.version("more-itertools")),
        },
        "gpu": {
            "requested": GPU_REQUEST,
            "visible_device_count": int(torch.cuda.device_count()),
            "device_index": int(device_index),
            "name": str(device_name),
            "capability_major": int(capability[0]),
            "capability_minor": int(capability[1]),
            "total_memory_bytes": int(properties.total_memory),
        },
        "model": {
            "name": MODEL_NAME,
            "device": str(model.device),
            "dtype": str(next(model.parameters()).dtype),
            "checkpoint_path": "model-cache-v1/tiny.en.pt",
            "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
            "loaded_state_sha256_before": state_before,
            "loaded_state_sha256_after": state_after,
        },
        "input": {
            "fixture_id": AUDIO_FIXTURE_ID,
            "path": AUDIO_RELATIVE_PATH,
            "sha256": AUDIO_SHA256,
            "size_bytes": AUDIO_SIZE_BYTES,
            "sample_rate_hz": SAMPLE_RATE,
            "sample_count": len(audio),
            "decoded_pcm_sha256": _decoded_pcm_fingerprint(audio),
        },
        "decode": {
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,
            "without_timestamps": True,
            "fp16": True,
            "rng_seed": RNG_SEED,
            "staged_result_count": 1,
            "text": result.text,
            "expected_text": EXPECTED_TEXT,
            "expected_text_matched": result.text == EXPECTED_TEXT,
            "reuse_text": reuse.text,
            "reuse_matched": reuse.text == EXPECTED_TEXT,
            "cleanup_calls": cleanup_calls,
            "reuse_cleanup_calls": _reuse_cleanup_calls,
        },
        "timing": {
            "staged_decode_seconds": float(staged_seconds),
            "reuse_decode_seconds": float(reuse_seconds),
            "total_seconds": float(time.perf_counter() - started),
            "synchronized": True,
        },
        "memory": {
            "allocated_before_bytes": int(allocated_before),
            "peak_allocated_bytes": int(peak_allocated),
            "peak_reserved_bytes": int(peak_reserved),
            "allocated_after_bytes": int(allocated_after),
            "measured": True,
            "enforced": False,
        },
        "adapter_boundary": {
            "adapter": "NativeWhisperAdapter",
            "device": "cuda:0",
            "attempted": True,
            "rejected_before_admission": True,
            "exception_type": rejection_type,
            "message": rejection_message,
            "queue_depth_before": queue_depth_before,
            "queue_depth_after": worker.queue_depth,
            "budget_available_before": _resource_vector(budget_before),
            "budget_available_after": _resource_vector(budget.available),
        },
        "assertions": assertions,
    }


def _write_record(path: Path, record: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_path(value: str) -> Path:
    relative = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or ":" in value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("artifacts", "modal")
        or relative.suffix != ".json"
    ):
        raise ValueError("--output must be a safe .json path below artifacts/modal")
    return Path(*relative.parts)


@app.local_entrypoint()
def main(
    output: str = "",
    skip_cache_prime: bool = False,
    confirm_paid_gpu: bool = False,
) -> None:
    """Prime the cache, run the GPU check, and validate the local record."""

    if not confirm_paid_gpu:
        raise SystemExit(
            "No cache or GPU function was dispatched. Pass --confirm-paid-gpu "
            "to allocate the T4."
        )
    destination = _output_path(output)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite an existing path: {destination}")
    if not skip_cache_prime:
        prime_model_cache.remote()
    record = validate_cuda_backend.remote()
    _write_record(destination, record)
    validator = (
        Path(__file__).resolve().parents[1] / "tools/validate_modal_cuda_record.py"
    )
    runtime_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", RUNTIME_COMMIT],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(validator),
            str(destination),
            "--expected-runtime-commit",
            RUNTIME_COMMIT,
            "--expected-runtime-tree",
            runtime_tree,
        ],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(destination.resolve())
