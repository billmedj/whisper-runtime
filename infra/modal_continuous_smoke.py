"""Run one short continuous-transcript diagnostic on a Modal T4.

The diagnostic has a two-call experiment budget and no automatic retry. It is
not a qualification, benchmark, or production-readiness result. Importing this
module does not import Modal, Torch, or Whisper and does not define remote
resources unless ``WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE=1`` is set.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path("/opt/whisper-runtime")
BACKEND_ROOT = Path("/opt/openai-whisper")
MANIFEST_PATH = "experiments/modal-continuous-smoke-v1.json"
PRODUCER_PATH = "infra/modal_continuous_smoke.py"
APP_NAME = "whisper-runtime-continuous-smoke-v1"
MODEL_CACHE_NAME = "whisper-runtime-model-cache-v1"
MODEL_CACHE_MOUNT = "/models"
MODEL_CHECKPOINT_PATH = Path(MODEL_CACHE_MOUNT) / "tiny.en.pt"
REMOTE_RESOURCES_ENV = "WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE"
MODAL_SDK_VERSION = "1.5.5"
MAX_DRIVER_STEPS = 20_000
_GIT_HASH = re.compile(r"[0-9a-f]{40}\Z")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(payload)


def _source_files(root: Path) -> tuple[Path, ...]:
    package = root / "src" / "whisper_runtime"
    files = [path for path in package.rglob("*.py") if path.is_file()]
    files.extend((root / PRODUCER_PATH, root / MANIFEST_PATH))
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError("the continuous smoke source snapshot is incomplete")
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _source_snapshot(root: Path = ROOT) -> dict[str, object]:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _source_files(root)
    ]
    return {
        "algorithm": "sha256-file-manifest-v1",
        "digest": _canonical_sha256(entries),
        "files": entries,
    }


def _read_registration(root: Path = ROOT) -> dict[str, Any]:
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("the continuous smoke registration must be an object")
    _validate_registration(manifest)
    return manifest


def _validate_registration(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("manifest_version") != "1"
        or manifest.get("manifest_id") != "modal-continuous-smoke-v1"
        or manifest.get("state") != "diagnostic"
    ):
        raise ValueError("unexpected continuous smoke registration identity")
    claims = manifest.get("claim_boundary")
    if claims != {
        "qualification": False,
        "performance_benchmark": False,
        "production_readiness": False,
        "public_commit_reproducibility": False,
    }:
        raise ValueError("the diagnostic claim boundary is not closed")
    source = manifest.get("source_policy")
    if (
        not isinstance(source, Mapping)
        or _GIT_HASH.fullmatch(str(source.get("public_base_commit", ""))) is None
    ):
        raise ValueError("the source policy requires one full public base commit")
    budget = manifest.get("paid_budget")
    if not isinstance(budget, Mapping) or any(
        budget.get(name) != value
        for name, value in (
            ("maximum_gpu_function_calls", 2),
            ("gpu_seconds_per_call", 600),
            ("maximum_gpu_seconds", 1_200),
            ("automatic_retries", 0),
            ("maximum_containers", 1),
            ("minimum_containers", 0),
            ("startup_timeout_seconds", 300),
            ("scaledown_window_seconds", 2),
        )
    ):
        raise ValueError("the paid diagnostic budget is not the registered budget")
    cell = manifest.get("cell")
    if not isinstance(cell, Mapping):
        raise ValueError("the diagnostic cell is missing")
    if any(
        cell.get(name) != value
        for name, value in (
            ("gpu", "T4"),
            ("device", "cuda:0"),
            ("model", "tiny.en"),
            ("sample_rate_hz", 16_000),
            ("duration_ms", 11_000),
            ("chunk_ms", 1_000),
        )
    ):
        raise ValueError("the diagnostic cell differs from the short T4 cell")
    stream = cell.get("stream_config")
    expected_stream = {
        "preview_interval_ms": 1_000,
        "max_window_ms": 30_000,
        "max_buffer_ms": 40_000,
        "holdback_ms": 1_000,
        "timestamp_tolerance_ms": 200,
    }
    if stream != expected_stream:
        raise ValueError("the continuous stream configuration is not registered")


def _attempt_paths(root: Path, attempt: int) -> tuple[Path, Path]:
    maximum = int(_read_registration(root)["paid_budget"]["maximum_gpu_function_calls"])
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise TypeError("attempt must be an integer")
    if not 1 <= attempt <= maximum:
        raise ValueError(f"attempt must be between 1 and {maximum}")
    stem = root / "artifacts" / "modal" / f"continuous-smoke-v1-attempt-{attempt}"
    return stem.with_suffix(".json"), stem.with_suffix(".attempt.jsonl")


def _append_receipt(path: Path, event: Mapping[str, Any], *, create: bool) -> None:
    payload = json.dumps(
        event,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if "\n" in payload or "\r" in payload:
        raise ValueError("a receipt event must fit on one line")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if create else "a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_paid_confirmation(value: bool) -> None:
    if value is not True:
        raise RuntimeError("pass --confirm-paid-gpu to use one registered T4 call")


def _command_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        raise RuntimeError(f"git {arguments[0]} failed")
    return value


def _decoded_float32_fingerprint(audio: object) -> str:
    value = audio.copy(order="C")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _event_payload(event: object) -> dict[str, Any]:
    value = asdict(event)
    kind = getattr(getattr(event, "kind", None), "value", None)
    if not isinstance(kind, str):
        raise RuntimeError("the stream emitted an event with no string kind")
    value["kind"] = kind
    return value


def _normalized_committed_text(events: list[Mapping[str, Any]]) -> str:
    revisions: dict[tuple[str, int], str] = {}
    parts: list[str] = []
    for event in events:
        kind = event.get("kind")
        segment = event.get("segment_id")
        revision = event.get("revision")
        if kind in {"provisional", "replace"}:
            if isinstance(segment, str) and isinstance(revision, int):
                text = event.get("text")
                if isinstance(text, str):
                    revisions[(segment, revision)] = text
        elif (
            kind == "commit" and isinstance(segment, str) and isinstance(revision, int)
        ):
            text = revisions.get((segment, revision))
            if text is not None:
                parts.append(text.strip())
    return " ".join(" ".join(parts).split())


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _evaluate_events(
    events: list[Mapping[str, Any]],
    *,
    pre_eof_commit_count: int,
    input_samples: int,
    full_window_control_text: str,
    reference_untimed_transcript_sha256: str,
    state: object,
    metrics: object,
    capacity: object,
    budget: object,
    worker: object,
    max_buffer_samples: int,
) -> tuple[dict[str, bool], str]:
    sequence = [event.get("sequence_number") for event in events]
    ordered = sequence == list(range(1, len(events) + 1))
    revisions: dict[tuple[str, int], str] = {}
    committed_segments: set[str] = set()
    commits: list[Mapping[str, Any]] = []
    immutable = True
    for event in events:
        kind = event.get("kind")
        segment = event.get("segment_id")
        revision = event.get("revision")
        if kind in {"provisional", "replace"}:
            if not isinstance(segment, str) or not isinstance(revision, int):
                immutable = False
                continue
            if segment in committed_segments:
                immutable = False
            text = event.get("text")
            if not isinstance(text, str):
                immutable = False
            else:
                revisions[(segment, revision)] = text
        elif kind == "commit":
            commits.append(event)
            if (
                not isinstance(segment, str)
                or not isinstance(revision, int)
                or (segment, revision) not in revisions
                or segment in committed_segments
            ):
                immutable = False
            elif segment is not None:
                committed_segments.add(segment)
    watermarks = [event.get("committed_through_sample") for event in commits]
    monotonic = bool(watermarks) and all(
        isinstance(value, int) and value > (watermarks[index - 1] if index else -1)
        for index, value in enumerate(watermarks)
    )
    finals = [event for event in events if event.get("kind") == "final"]
    final_once = len(finals) == 1 and bool(events) and events[-1].get("kind") == "final"
    transcript = _normalized_committed_text(events)
    control = _normalized_text(full_window_control_text)
    checks = {
        "ordered_events": ordered,
        "immutable_committed_revisions": immutable,
        "monotonic_commit_watermarks": monotonic,
        "final_event_once": final_once,
        "full_input_committed_at_eof": (
            bool(watermarks)
            and watermarks[-1] == input_samples
            and getattr(metrics, "committed_samples", None) == input_samples
            and getattr(state, "committed_through_ms", None) == input_samples // 16
        ),
        "bounded_audio_buffer": (
            getattr(metrics, "buffered_samples", None) == 0
            and isinstance(getattr(metrics, "peak_buffered_samples", None), int)
            and getattr(metrics, "peak_buffered_samples") <= max_buffer_samples
        ),
        "runtime_capacity_restored": (
            getattr(worker, "queue_depth", None) == 0
            and getattr(budget, "lease_count", None) == 0
            and getattr(budget, "available", None) == capacity
        ),
        "nonempty_final_transcript": bool(transcript),
        "nonempty_full_window_control": bool(control),
        "pre_eof_commit_observed": pre_eof_commit_count > 0,
        "continuous_matches_full_window_control": transcript == control,
        "reference_untimed_text_observed": reference_untimed_transcript_sha256
        in {_sha256_text(transcript), _sha256_text(control)},
    }
    return checks, transcript


def _run_worker(
    expected_snapshot: Mapping[str, Any],
    *,
    attempt: int,
    registration_sha256: str,
    modal_module: Any,
) -> dict[str, Any]:
    manifest = _read_registration(RUNTIME_ROOT)
    if _sha256_file(RUNTIME_ROOT / MANIFEST_PATH) != registration_sha256:
        raise RuntimeError(
            "the remote registration differs from the local registration"
        )
    observed_snapshot = _source_snapshot(RUNTIME_ROOT)
    if observed_snapshot != expected_snapshot:
        raise RuntimeError(
            "the remote source snapshot differs from the submitted snapshot"
        )
    output_path, _ = _attempt_paths(RUNTIME_ROOT, attempt)
    del output_path  # Validate the attempt budget without writing in the worker.

    cell = manifest["cell"]
    base_commit = manifest["source_policy"]["public_base_commit"]
    if _command_output(RUNTIME_ROOT, "rev-parse", "HEAD") != base_commit:
        raise RuntimeError("the image base checkout differs from the registration")

    torch = importlib.import_module("torch")
    whisper = importlib.import_module("whisper")
    np = importlib.import_module("numpy")
    runtime = importlib.import_module("whisper_runtime")
    adapters = importlib.import_module("whisper_runtime.adapters")
    continuous = importlib.import_module("whisper_runtime.adapters.continuous_stream")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the diagnostic requires one visible CUDA device")
    device_index = int(torch.cuda.current_device())
    device_name = str(torch.cuda.get_device_name(device_index))
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if device_index != 0 or device_name != "Tesla T4" or capability != (7, 5):
        raise RuntimeError("the allocated device differs from the registered T4")
    if _sha256_file(MODEL_CHECKPOINT_PATH) != cell["checkpoint_sha256"]:
        raise RuntimeError("the cached model checkpoint differs from the registration")
    audio_path = Path(str(cell["input_path"]))
    if _sha256_file(audio_path) != cell["input_sha256"]:
        raise RuntimeError("the audio fixture differs from the registration")

    model = whisper.load_model(
        str(cell["model"]), device=str(cell["device"]), download_root=MODEL_CACHE_MOUNT
    ).eval()
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    floating = tuple(tensor for tensor in tensors if tensor.is_floating_point())
    if (
        model.training is not False
        or not floating
        or any(str(tensor.device) != cell["device"] for tensor in tensors)
        or any(tensor.dtype != torch.float32 for tensor in floating)
    ):
        raise RuntimeError("the loaded model differs from the registered FP32 profile")
    audio = whisper.load_audio(str(audio_path))
    expected_samples = int(cell["sample_rate_hz"]) * int(cell["duration_ms"]) // 1_000
    if (
        len(audio) != expected_samples
        or _decoded_float32_fingerprint(audio) != cell["decoded_float32_fingerprint"]
    ):
        raise RuntimeError("the decoded fixture differs from the registration")
    pcm = np.rint(audio * 32768.0).clip(-32768, 32767).astype("<i2").tobytes()
    if hashlib.sha256(pcm).hexdigest() != cell["converted_pcm_s16le_sha256"]:
        raise RuntimeError("the registered float-to-s16 conversion differs")

    capacity = runtime.ResourceVector(
        memory_bytes=2_147_483_648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    snapshot = runtime.ModelSnapshot(
        model_id=str(cell["model"]),
        revision=_command_output(BACKEND_ROOT, "rev-parse", "HEAD"),
        backend="pytorch-cuda-continuous-diagnostic",
        fingerprint=str(cell["model_state_sha256"]),
    )
    worker = runtime.Worker(
        "modal-continuous-smoke",
        snapshot,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=300,
    )

    def identity_probe(observed: object) -> object:
        if (
            observed is not model
            or str(getattr(observed, "device", "")) != cell["device"]
        ):
            raise RuntimeError("the adapter model identity changed")
        return snapshot

    profile = adapters.NativeExecutionProfile(
        "tiny.en/cuda-continuous-diagnostic-v1", capacity, device=str(cell["device"])
    )
    adapter = adapters.NativeWhisperAdapter(worker, model, identity_probe, profile)

    def mel_builder(content: bytes) -> object:
        decoded = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768.0
        return whisper.log_mel_spectrogram(
            whisper.pad_or_trim(decoded), n_mels=model.dims.n_mels
        ).contiguous()

    stream_config = continuous.ContinuousStreamConfig(**cell["stream_config"])
    options = adapters.NativeDecodeOptions(**cell["decode_options"])
    control_steps = 0
    control_session = runtime.Session(f"modal-continuous-smoke-{attempt}:control")
    with adapter.start_window(
        session=control_session,
        request=runtime.RequestState(
            f"modal-continuous-smoke-{attempt}:control:request",
            control_session.session_id,
            snapshot,
            rng_seed=7,
        ),
        window_id=f"modal-continuous-smoke-{attempt}:control:window",
        mel=mel_builder(pcm),
        start_ms=0,
        end_ms=int(cell["duration_ms"]),
        options=options,
    ) as control_run:
        while not control_run.complete:
            control_run.step()
            control_steps += 1
            if control_steps > MAX_DRIVER_STEPS:
                raise RuntimeError("the full-window control exceeded its step bound")
        full_window_control = control_run.prepare_result()
    full_window_control_text = _normalized_text(full_window_control.text)
    if budget.available != capacity or worker.queue_depth != 0:
        raise RuntimeError("the full-window control retained runtime capacity")

    events: list[dict[str, Any]] = []
    pre_eof_commits = 0
    driver_steps = 0
    torch.cuda.reset_peak_memory_stats(0)
    started_ns = time.perf_counter_ns()
    with continuous.ContinuousTranscriptStream(
        adapter,
        stream_id=f"modal-continuous-smoke-{attempt}",
        mel_builder=mel_builder,
        options=options,
        rng_seed=7,
        config=stream_config,
    ) as stream:
        chunk_bytes = int(cell["sample_rate_hz"]) * int(cell["chunk_ms"]) // 1_000 * 2
        for sequence, offset in enumerate(range(0, len(pcm), chunk_bytes)):
            stream.push(sequence, pcm[offset : offset + chunk_bytes])
            while stream.ready:
                batch = stream.step()
                driver_steps += 1
                if driver_steps > MAX_DRIVER_STEPS:
                    raise RuntimeError("the continuous driver exceeded its step bound")
                payloads = [_event_payload(event) for event in batch]
                pre_eof_commits += sum(item["kind"] == "commit" for item in payloads)
                events.extend(payloads)
        stream.finish_input()
        while stream.ready:
            batch = stream.step()
            driver_steps += 1
            if driver_steps > MAX_DRIVER_STEPS:
                raise RuntimeError("the continuous driver exceeded its step bound")
            events.extend(_event_payload(event) for event in batch)
    torch.cuda.synchronize(0)
    elapsed_ns = time.perf_counter_ns() - started_ns
    state = stream.state
    metrics = stream.metrics
    checks, transcript = _evaluate_events(
        events,
        pre_eof_commit_count=pre_eof_commits,
        input_samples=expected_samples,
        full_window_control_text=full_window_control_text,
        reference_untimed_transcript_sha256=str(
            cell["reference_untimed_transcript_sha256"]
        ),
        state=state,
        metrics=metrics,
        capacity=capacity,
        budget=budget,
        worker=worker,
        max_buffer_samples=int(cell["stream_config"]["max_buffer_ms"]) * 16,
    )
    required_names = tuple(manifest["observations"]["required"])
    required_passed = all(checks.get(name) is True for name in required_names)
    if not required_passed:
        status = "failed"
    elif not checks["pre_eof_commit_observed"]:
        status = "completed_without_pre_eof_commit"
    else:
        status = "passed"
    function_call_id = str(modal_module.current_function_call_id())
    return {
        "schema_version": "1-diagnostic",
        "recorded_at": _utc_now(),
        "status": status,
        "attempt": attempt,
        "claim_boundary": manifest["claim_boundary"],
        "source": {
            "public_base_commit": base_commit,
            "snapshot": observed_snapshot,
            "registration_sha256": registration_sha256,
            "public_commit_reproducibility": False,
        },
        "worker": {
            "function_call_id_sha256": _sha256_text(function_call_id),
            "python": sys.version.split()[0],
            "modal": str(modal_module.__version__),
            "torch": str(torch.__version__),
        },
        "gpu": {
            "name": device_name,
            "compute_capability": list(capability),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        },
        "model": {
            "name": cell["model"],
            "device": cell["device"],
            "checkpoint_sha256": cell["checkpoint_sha256"],
        },
        "input": {
            "fixture": cell["fixture"],
            "sample_rate_hz": cell["sample_rate_hz"],
            "sample_count": expected_samples,
            "pcm_s16le_sha256": hashlib.sha256(pcm).hexdigest(),
            "decoded_float32_fingerprint": cell["decoded_float32_fingerprint"],
        },
        "full_window_control": {
            "scope": "One native decode of the full fixture with the same model and decode options. Exact text equality is diagnostic because rolling windows change context.",
            "step_count": control_steps,
            "text": full_window_control_text,
            "text_sha256": _sha256_text(full_window_control_text),
            "reference_untimed_transcript_sha256": cell[
                "reference_untimed_transcript_sha256"
            ],
        },
        "stream": {
            "profile_id": stream.profile_id,
            "config": cell["stream_config"],
            "driver_steps": driver_steps,
            "events": events,
            "metrics": asdict(metrics),
            "final_transcript": transcript,
            "final_transcript_sha256": _sha256_text(transcript),
            "pre_eof_commit_count": pre_eof_commits,
            "session_version": state.version,
            "committed_through_ms": state.committed_through_ms,
        },
        "checks": checks,
        "timing": {
            "wall_ns": elapsed_ns,
            "benchmark": False,
        },
    }


def _execute_local_attempt(
    *,
    root: Path,
    attempt: int,
    confirm_paid_gpu: bool,
    remote_function: object,
) -> Path:
    _require_paid_confirmation(confirm_paid_gpu)
    manifest = _read_registration(root)
    output_path, receipt_path = _attempt_paths(root, attempt)
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("this diagnostic attempt already exists")
    snapshot = _source_snapshot(root)
    registration_sha256 = _sha256_file(root / MANIFEST_PATH)
    common = {
        "receipt_version": "1",
        "manifest_id": manifest["manifest_id"],
        "attempt": attempt,
        "source_snapshot_sha256": snapshot["digest"],
        "registration_sha256": registration_sha256,
    }
    _append_receipt(
        receipt_path,
        {**common, "sequence": 0, "event": "attempt-started", "at": _utc_now()},
        create=True,
    )
    try:
        remote = getattr(remote_function, "remote")
        record = remote(snapshot, attempt, registration_sha256)
        if not isinstance(record, dict):
            raise TypeError("the Modal worker returned a non-object record")
        _write_json_exclusive(output_path, record)
    except BaseException as error:
        if isinstance(error, Exception):
            _append_receipt(
                receipt_path,
                {
                    **common,
                    "sequence": 1,
                    "event": "attempt-failed",
                    "at": _utc_now(),
                    "error_type": type(error).__name__,
                    "error_message_sha256": _sha256_text(str(error)),
                },
                create=False,
            )
        raise
    _append_receipt(
        receipt_path,
        {
            **common,
            "sequence": 1,
            "event": "record-written",
            "at": _utc_now(),
            "record_sha256": _sha256_file(output_path),
            "status": record.get("status"),
        },
        create=False,
    )
    return output_path


def _modal_main(attempt: int = 1, confirm_paid_gpu: bool = False) -> None:
    """Use one of the two manually selected diagnostic GPU calls."""

    if run_continuous_smoke is None:
        raise RuntimeError(f"set {REMOTE_RESOURCES_ENV}=1 before modal run")
    output = _execute_local_attempt(
        root=ROOT,
        attempt=attempt,
        confirm_paid_gpu=confirm_paid_gpu,
        remote_function=run_continuous_smoke,
    )
    print(f"Wrote diagnostic record to {output}")


def _definition_enabled() -> bool:
    return os.environ.get(REMOTE_RESOURCES_ENV) == "1"


def _define_modal_resources() -> tuple[Any, Any, Any]:
    modal = importlib.import_module("modal")
    if str(modal.__version__) != MODAL_SDK_VERSION:
        raise RuntimeError(
            f"Modal SDK mismatch: expected {MODAL_SDK_VERSION}, observed {modal.__version__}"
        )
    manifest = _read_registration()
    budget = manifest["paid_budget"]
    cell = manifest["cell"]
    base_commit = manifest["source_policy"]["public_base_commit"]
    qualification = importlib.import_module("infra.modal_native_cuda_qualification")
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("ca-certificates", "ffmpeg", "git")
        .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
        .uv_pip_install(*qualification.DIRECT_IMAGE_PACKAGES)
        .run_commands(qualification._build_command(base_commit))
        .add_local_dir(
            ROOT / "src" / "whisper_runtime",
            "/opt/whisper-runtime/src/whisper_runtime",
            copy=True,
        )
        .add_local_file(
            ROOT / PRODUCER_PATH, f"/opt/whisper-runtime/{PRODUCER_PATH}", copy=True
        )
        .add_local_file(
            ROOT / MANIFEST_PATH, f"/opt/whisper-runtime/{MANIFEST_PATH}", copy=True
        )
        .env(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": "/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime",
                "PYTHONUTF8": "1",
                REMOTE_RESOURCES_ENV: "0",
            }
        )
    )
    model_cache = modal.Volume.from_name(MODEL_CACHE_NAME, create_if_missing=False)
    app = modal.App(APP_NAME)

    @app.function(
        image=image,
        serialized=True,
        gpu=str(cell["gpu"]),
        cloud=str(cell["cloud"]),
        region=str(cell["region"]),
        volumes={MODEL_CACHE_MOUNT: model_cache.with_mount_options(read_only=True)},
        cpu=float(budget["cpu_cores"]),
        memory=int(float(budget["memory_gib"]) * 1_024),
        min_containers=int(budget["minimum_containers"]),
        max_containers=int(budget["maximum_containers"]),
        scaledown_window=int(budget["scaledown_window_seconds"]),
        retries=int(budget["automatic_retries"]),
        timeout=int(budget["gpu_seconds_per_call"]),
        startup_timeout=int(budget["startup_timeout_seconds"]),
        block_network=True,
        restrict_modal_access=True,
        single_use_containers=True,
        include_source=False,
    )
    def run_continuous_smoke(
        expected_snapshot: Mapping[str, Any], attempt: int, registration_sha256: str
    ) -> dict[str, Any]:
        producer = importlib.import_module("infra.modal_continuous_smoke")
        return producer._run_worker(
            expected_snapshot,
            attempt=attempt,
            registration_sha256=registration_sha256,
            modal_module=modal,
        )

    main = app.local_entrypoint(name="main")(_modal_main)
    return app, run_continuous_smoke, main


if _definition_enabled():
    app, run_continuous_smoke, main = _define_modal_resources()
else:
    app = None
    run_continuous_smoke = None
    main = None
