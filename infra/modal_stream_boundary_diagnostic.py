"""Run one bounded T4 comparison of continuous-stream boundary policies.

This is a single-input diagnostic, not a benchmark or a quality claim. Importing
the module does not load Modal, Torch, or Whisper and cannot allocate a worker.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path("/opt/whisper-runtime")
BACKEND_ROOT = Path("/opt/openai-whisper")
PRODUCER_PATH = "infra/modal_stream_boundary_diagnostic.py"
MANIFEST_PATH = "experiments/modal-stream-boundary-diagnostic-v1.json"
MANIFEST_ID = "modal-stream-boundary-diagnostic-v1"
APP_NAME = "whisper-runtime-stream-boundary-diagnostic-v1"
REMOTE_RESOURCES_ENV = "WHISPER_MODAL_ENABLE_STREAM_BOUNDARY_DIAGNOSTIC"
MODEL_CACHE_NAME = "whisper-runtime-model-cache-v1"
MODEL_CACHE_MOUNT = "/models"
MODEL_CHECKPOINT_PATH = Path(MODEL_CACHE_MOUNT) / "tiny.en.pt"
MODAL_SDK_VERSION = "1.5.5"
MAX_DRIVER_STEPS = 20_000
_GIT_HASH = re.compile(r"[0-9a-f]{40}\Z")
COMMON_PRODUCER_PATH = "infra/modal_continuous_smoke.py"

_old_remote_setting = os.environ.get("WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE")
os.environ["WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE"] = "0"
try:
    _common = importlib.import_module("infra.modal_continuous_smoke")
finally:
    if _old_remote_setting is None:
        os.environ.pop("WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE", None)
    else:
        os.environ["WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE"] = _old_remote_setting

_append_receipt = _common._append_receipt
_command_output = _common._command_output
_decoded_float32_fingerprint = _common._decoded_float32_fingerprint
_normalized_committed_text = _common._normalized_committed_text
_require_paid_confirmation = _common._require_paid_confirmation
_sha256_file = _common._sha256_file
_sha256_text = _common._sha256_text
_utc_now = _common._utc_now
_write_json_exclusive = _common._write_json_exclusive


def _source_snapshot(root: Path = ROOT) -> dict[str, object]:
    package = root / "src" / "whisper_runtime"
    paths = [path for path in package.rglob("*.py") if path.is_file()]
    paths.extend(
        (root / PRODUCER_PATH, root / MANIFEST_PATH, root / COMMON_PRODUCER_PATH)
    )
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError("the diagnostic source snapshot is incomplete")
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _common._sha256_file(path),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    return {
        "algorithm": "sha256-file-manifest-v1",
        "digest": _common._canonical_sha256(entries),
        "files": entries,
    }


def _read_registration(root: Path = ROOT) -> dict[str, Any]:
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("the diagnostic registration must be an object")
    _validate_registration(manifest)
    return manifest


def _validate_registration(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("manifest_version") != "1"
        or manifest.get("manifest_id") != MANIFEST_ID
        or manifest.get("state") != "diagnostic"
    ):
        raise ValueError("unexpected stream-boundary registration identity")
    if manifest.get("claim_boundary") != {
        "recognition_improvement": False,
        "performance_benchmark": False,
        "production_readiness": False,
        "public_commit_reproducibility": False,
        "generalization_beyond_this_input": False,
    }:
        raise ValueError("the diagnostic claim boundary is not closed")
    source = manifest.get("source_policy")
    if (
        not isinstance(source, Mapping)
        or _GIT_HASH.fullmatch(str(source.get("image_base_commit", ""))) is None
    ):
        raise ValueError("source_policy requires one full image base commit")
    if source.get("snapshot_paths") != [
        "src/whisper_runtime/**/*.py",
        PRODUCER_PATH,
        COMMON_PRODUCER_PATH,
        MANIFEST_PATH,
    ]:
        raise ValueError("the source snapshot declaration is not fixed")
    if manifest.get("replay_pacing") != "unpaced-source-time":
        raise ValueError("the diagnostic must remain an unpaced source-time replay")
    if manifest.get("rng_seed") != 7:
        raise ValueError("the diagnostic seed is not fixed")
    budget = manifest.get("paid_budget")
    if not isinstance(budget, Mapping) or any(
        budget.get(name) != value
        for name, value in (
            ("maximum_gpu_function_calls", 1),
            ("gpu_seconds_per_call", 600),
            ("maximum_gpu_seconds", 600),
            ("automatic_retries", 0),
            ("maximum_containers", 1),
            ("minimum_containers", 0),
            ("startup_timeout_seconds", 300),
            ("scaledown_window_seconds", 2),
        )
    ):
        raise ValueError("the paid diagnostic budget is not fixed")
    audio = manifest.get("input")
    if not isinstance(audio, Mapping) or any(
        audio.get(name) != value
        for name, value in (
            ("fixture_repetitions", 3),
            ("fixture_duration_ms", 11_000),
            ("duration_ms", 33_000),
            ("sample_rate_hz", 16_000),
            ("chunk_ms", 1_000),
        )
    ):
        raise ValueError("the registered input is not the fixed 33-second stream")
    cells = manifest.get("cells")
    expected_cells = [
        ("baseline", 0, 1_000),
        ("left-context-2000", 2_000, 1_000),
        ("holdback-2000", 0, 2_000),
        ("left-context-2000-holdback-2000", 2_000, 2_000),
    ]
    observed_cells = []
    if isinstance(cells, list):
        observed_cells = [
            (cell.get("cell_id"), cell.get("left_context_ms"), cell.get("holdback_ms"))
            for cell in cells
            if isinstance(cell, Mapping)
        ]
    if observed_cells != expected_cells:
        raise ValueError("the four boundary-policy cells are not fixed")


def _attempt_paths(root: Path, attempt: int) -> tuple[Path, Path]:
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise TypeError("attempt must be an integer")
    if attempt != 1:
        raise ValueError("this diagnostic permits exactly attempt 1")
    stem = root / "artifacts" / "modal" / "stream-boundary-diagnostic-v1-attempt-1"
    return stem.with_suffix(".json"), stem.with_suffix(".attempt.jsonl")


def _plain(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("diagnostic records require finite numbers")
        return value
    if isinstance(value, Enum):
        return _plain(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("diagnostic mapping keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    raise TypeError(f"unsupported diagnostic value type: {type(value).__name__}")


def _capture_trace(stream: object, traces: list[dict[str, Any]]) -> None:
    trace = getattr(stream, "last_trace", None)
    if trace is None:
        return
    payload = _plain(trace)
    if not isinstance(payload, dict):
        raise TypeError("last_trace must serialize to an object")
    index = payload.get("decode_index")
    if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
        raise ValueError("last_trace.decode_index must be positive")
    if traces and index == traces[-1]["decode_index"]:
        if payload != traces[-1]:
            raise RuntimeError("an emitted decision trace changed in place")
        return
    if traces and index <= traces[-1]["decode_index"]:
        raise RuntimeError("decision trace indices must increase")
    traces.append(payload)


def _safe_stream_error(error: Exception) -> dict[str, object]:
    name = type(error).__name__
    policy = name == "StreamNeedsResolutionError"
    reasons = {
        "StreamNeedsResolutionError": (
            "stable_prefix_not_found",
            "The bounded window reached its limit without a publishable prefix.",
        ),
        "AudioBufferFullError": (
            "audio_buffer_full",
            "The bounded input buffer could not admit the next chunk.",
        ),
        "TransactionRetainedError": (
            "transaction_retained",
            "The runtime retained recovery authority after a transaction failure.",
        ),
    }
    reason_code, reason = reasons.get(
        name,
        ("unexpected_stream_error", "The stream stopped before it completed."),
    )
    safe_name = (
        name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", name) else "Exception"
    )
    return {
        "category": "policy_resolution" if policy else "runtime_failure",
        "error_class": safe_name,
        "reason_code": reason_code,
        "reason": reason,
    }


def _drive_stream(
    stream: Any, pcm: bytes, *, chunk_bytes: int
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], int, int, dict[str, object] | None
]:
    events: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    driver_steps = accepted_chunks = 0
    error_record: dict[str, object] | None = None
    try:
        with stream:
            for sequence, offset in enumerate(range(0, len(pcm), chunk_bytes)):
                stream.push(sequence, pcm[offset : offset + chunk_bytes])
                accepted_chunks += 1
                while stream.ready:
                    try:
                        batch = stream.step()
                    except Exception:
                        _capture_trace(stream, traces)
                        raise
                    driver_steps += 1
                    if driver_steps > MAX_DRIVER_STEPS:
                        raise RuntimeError("the driver exceeded its fixed step bound")
                    _capture_trace(stream, traces)
                    events.extend(_plain(event) for event in batch)
            stream.finish_input()
            while stream.ready:
                try:
                    batch = stream.step()
                except Exception:
                    _capture_trace(stream, traces)
                    raise
                driver_steps += 1
                if driver_steps > MAX_DRIVER_STEPS:
                    raise RuntimeError("the driver exceeded its fixed step bound")
                _capture_trace(stream, traces)
                events.extend(_plain(event) for event in batch)
    except Exception as error:
        error_record = _safe_stream_error(error)
    if any(not isinstance(event, dict) for event in events):
        raise TypeError("transcript events must serialize to objects")
    return events, traces, driver_steps, accepted_chunks, error_record


def _event_checks(
    events: list[Mapping[str, Any]],
    traces: list[Mapping[str, Any]],
    *,
    accepted_samples: int,
    total_samples: int,
    state: object,
    metrics: object,
    budget: object,
    worker: object,
    capacity: object,
    retained_from_sample: int,
    max_buffer_samples: int,
    error_record: Mapping[str, object] | None,
) -> dict[str, bool]:
    sequence = [event.get("sequence_number") for event in events]
    revisions: dict[str, int] = {}
    revision_spans: dict[str, tuple[int, int]] = {}
    committed: set[str] = set()
    commits: list[Mapping[str, Any]] = []
    immutable = True
    publication_within_input = True
    for event in events:
        kind = event.get("kind")
        segment = event.get("segment_id")
        revision = event.get("revision")
        if kind in {"provisional", "replace"}:
            if (
                not isinstance(segment, str)
                or not isinstance(revision, int)
                or segment in committed
                or revision != revisions.get(segment, 0) + 1
            ):
                immutable = False
            else:
                revisions[segment] = revision
                start = event.get("start_sample")
                end = event.get("end_sample")
                if (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or start < 0
                    or end <= start
                ):
                    immutable = False
                else:
                    revision_spans[segment] = (start, end)
        elif kind == "commit":
            commits.append(event)
            if (
                not isinstance(segment, str)
                or not isinstance(revision, int)
                or segment in committed
                or revisions.get(segment) != revision
                or revision_spans.get(segment)
                != (event.get("start_sample"), event.get("end_sample"))
            ):
                immutable = False
            else:
                committed.add(segment)
        for field_name in ("start_sample", "end_sample", "committed_through_sample"):
            value = event.get(field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= accepted_samples
            ):
                publication_within_input = False
    watermarks = [event.get("committed_through_sample") for event in commits]
    commit_coverage = True
    prior_watermark = 0
    for commit in commits:
        watermark = commit.get("committed_through_sample")
        if (
            commit.get("start_sample") != prior_watermark
            or commit.get("end_sample") != watermark
            or isinstance(watermark, bool)
            or not isinstance(watermark, int)
        ):
            commit_coverage = False
            break
        prior_watermark = watermark
    policy_resolution = (
        error_record is not None and error_record.get("category") == "policy_resolution"
    )
    finals = [event for event in events if event.get("kind") == "final"]
    committed_samples = getattr(metrics, "committed_samples", None)
    buffered_samples = getattr(metrics, "buffered_samples", None)
    state_committed_ms = getattr(state, "committed_through_ms", None)
    state_matches_metrics = (
        state_committed_ms is None and committed_samples == 0 and not commits
    ) or (
        isinstance(state_committed_ms, int)
        and isinstance(committed_samples, int)
        and state_committed_ms * 16 == committed_samples
    )
    commit_watermark_matches_runtime = (
        isinstance(committed_samples, int)
        and (
            (bool(commits) and prior_watermark == committed_samples)
            or (not commits and committed_samples == 0)
        )
        and state_matches_metrics
    )
    common = {
        "ordered_events": sequence == list(range(1, len(events) + 1)),
        "immutable_committed_revisions": immutable,
        "monotonic_commit_watermarks": all(
            isinstance(value, int) and value > (watermarks[index - 1] if index else -1)
            for index, value in enumerate(watermarks)
        ),
        "publication_within_accepted_input": publication_within_input,
        "contiguous_commit_coverage": commit_coverage,
        "commit_watermark_matches_runtime": commit_watermark_matches_runtime,
        "bounded_audio_buffer": (
            isinstance(getattr(metrics, "peak_buffered_samples", None), int)
            and getattr(metrics, "peak_buffered_samples") <= max_buffer_samples
        ),
        "runtime_capacity_restored": (
            getattr(worker, "queue_depth", None) == 0
            and getattr(budget, "lease_count", None) == 0
            and getattr(budget, "available", None) == capacity
        ),
        "accepted_input_accounted": (
            getattr(metrics, "accepted_samples", None) == accepted_samples
            and isinstance(committed_samples, int)
            and isinstance(buffered_samples, int)
            and 0 <= retained_from_sample <= committed_samples <= accepted_samples
            and retained_from_sample + buffered_samples == accepted_samples
            and state_matches_metrics
            and accepted_samples <= total_samples
        ),
        "decision_trace_ordered": (
            [trace.get("decode_index") for trace in traces]
            == list(range(1, len(traces) + 1))
            and len(traces) == getattr(metrics, "decode_count", None)
        ),
        "final_event_once": len(finals) == 1 and bool(events) and events[-1] in finals,
        "full_input_committed_at_eof": (
            getattr(metrics, "accepted_samples", None) == total_samples
            and getattr(metrics, "committed_samples", None) == total_samples
            and getattr(state, "committed_through_ms", None) == total_samples // 16
            and getattr(metrics, "buffered_samples", None) == 0
            and retained_from_sample == total_samples
        ),
    }
    common["completion_required"] = not policy_resolution
    return common


def _word_difference(observed: str, reference: str) -> dict[str, object]:
    left = re.findall(r"\w+", observed.casefold())
    right = re.findall(r"\w+", reference.casefold())
    previous = list(range(len(right) + 1))
    for row, token in enumerate(left, 1):
        current = [row]
        for column, target in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (token != target),
                )
            )
        previous = current
    distance = previous[-1]
    return {
        "exact_text_match": observed == reference,
        "word_edit_distance": distance,
        "reference_word_count": len(right),
        "word_edit_rate": None if not right else distance / len(right),
    }


def _offline_result(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not isinstance(value.get("text"), str):
        raise RuntimeError("offline transcription returned an invalid result")
    segments = []
    for segment in value.get("segments", []):
        if not isinstance(segment, Mapping):
            raise RuntimeError("offline transcription returned an invalid segment")
        segments.append(
            {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
            }
        )
    text = " ".join(str(value["text"]).split())
    return {
        "language": value.get("language"),
        "text": text,
        "text_sha256": _sha256_text(text),
        "segments": _plain(segments),
    }


def _run_worker(
    expected_snapshot: Mapping[str, Any],
    *,
    registration_sha256: str,
    modal_module: Any,
) -> dict[str, Any]:
    manifest = _read_registration(RUNTIME_ROOT)
    if _sha256_file(RUNTIME_ROOT / MANIFEST_PATH) != registration_sha256:
        raise RuntimeError("the remote registration differs from the local one")
    snapshot_record = _source_snapshot(RUNTIME_ROOT)
    if snapshot_record != expected_snapshot:
        raise RuntimeError("the remote source snapshot differs from the submitted one")
    base_commit = manifest["source_policy"]["image_base_commit"]
    if _command_output(RUNTIME_ROOT, "rev-parse", "HEAD") != base_commit:
        raise RuntimeError("the image checkout differs from the registration")

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
    capability = tuple(int(item) for item in torch.cuda.get_device_capability(0))
    if device_index != 0 or device_name != "Tesla T4" or capability != (7, 5):
        raise RuntimeError("the allocated device differs from the registered T4")

    model_config = manifest["model"]
    input_config = manifest["input"]
    if _sha256_file(MODEL_CHECKPOINT_PATH) != model_config["checkpoint_sha256"]:
        raise RuntimeError("the cached checkpoint differs from the registration")
    audio_path = Path(str(input_config["input_path"]))
    if _sha256_file(audio_path) != input_config["input_sha256"]:
        raise RuntimeError("the audio fixture differs from the registration")
    model = whisper.load_model(
        str(model_config["name"]),
        device=str(model_config["device"]),
        download_root=MODEL_CACHE_MOUNT,
    ).eval()
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    floating = tuple(tensor for tensor in tensors if tensor.is_floating_point())
    if (
        model.training is not False
        or not floating
        or any(str(tensor.device) != model_config["device"] for tensor in tensors)
        or any(tensor.dtype != torch.float32 for tensor in floating)
    ):
        raise RuntimeError("the loaded model differs from the registered FP32 profile")

    audio = whisper.load_audio(str(audio_path))
    fixture_samples = (
        int(input_config["sample_rate_hz"])
        * int(input_config["fixture_duration_ms"])
        // 1_000
    )
    if (
        len(audio) != fixture_samples
        or _decoded_float32_fingerprint(audio)
        != input_config["decoded_float32_fingerprint"]
    ):
        raise RuntimeError("the decoded fixture differs from the registration")
    fixture_pcm = np.rint(audio * 32768.0).clip(-32768, 32767).astype("<i2").tobytes()
    if (
        hashlib.sha256(fixture_pcm).hexdigest()
        != input_config["converted_fixture_pcm_s16le_sha256"]
    ):
        raise RuntimeError("the float-to-s16 conversion differs from the registration")
    pcm = fixture_pcm * int(input_config["fixture_repetitions"])
    if hashlib.sha256(pcm).hexdigest() != input_config["stream_pcm_s16le_sha256"]:
        raise RuntimeError("the constructed stream differs from the registration")
    total_samples = len(pcm) // 2
    stream_audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    fixture_audio = np.frombuffer(fixture_pcm, dtype="<i2").astype(np.float32) / 32768.0

    offline_options = dict(manifest["offline_control_options"])
    rng_seed = int(manifest["rng_seed"])
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    full_control = _offline_result(model.transcribe(stream_audio, **offline_options))
    segmented_runs = []
    for _ in range(int(input_config["fixture_repetitions"])):
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed_all(rng_seed)
        segmented_runs.append(
            _offline_result(model.transcribe(fixture_audio, **offline_options))
        )
    segmented_text = " ".join(str(run["text"]) for run in segmented_runs)
    segmented_control = {
        "scope": "Three independent 11-second offline transcriptions; not a 33-second full-window decode.",
        "runs": segmented_runs,
        "text": segmented_text,
        "text_sha256": _sha256_text(segmented_text),
    }
    torch.cuda.synchronize(0)

    capacity = runtime.ResourceVector(
        memory_bytes=2_147_483_648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    model_snapshot = runtime.ModelSnapshot(
        model_id=str(model_config["name"]),
        revision=_command_output(BACKEND_ROOT, "rev-parse", "HEAD"),
        backend="pytorch-cuda-stream-boundary-diagnostic",
        fingerprint=str(model_config["model_state_sha256"]),
    )
    worker = runtime.Worker(
        "modal-stream-boundary-diagnostic",
        model_snapshot,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=300,
    )

    def identity_probe(observed: object) -> object:
        if (
            observed is not model
            or str(getattr(observed, "device", "")) != model_config["device"]
        ):
            raise RuntimeError("the adapter model identity changed")
        return model_snapshot

    profile = adapters.NativeExecutionProfile(
        "tiny.en/cuda-stream-boundary-diagnostic-v1",
        capacity,
        device=str(model_config["device"]),
    )
    adapter = adapters.NativeWhisperAdapter(worker, model, identity_probe, profile)

    def mel_builder(content: bytes) -> object:
        decoded = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768.0
        return whisper.log_mel_spectrogram(
            whisper.pad_or_trim(decoded), n_mels=model.dims.n_mels
        ).contiguous()

    common_config = manifest["common_stream_config"]
    decode_options = adapters.NativeDecodeOptions(**manifest["decode_options"])
    native_control_steps = 0
    native_session = runtime.Session(f"{MANIFEST_ID}:native-control")
    with adapter.start_window(
        session=native_session,
        request=runtime.RequestState(
            f"{MANIFEST_ID}:native-control:request",
            native_session.session_id,
            model_snapshot,
            rng_seed=rng_seed,
        ),
        window_id=f"{MANIFEST_ID}:native-control:window",
        mel=mel_builder(fixture_pcm),
        start_ms=0,
        end_ms=int(input_config["fixture_duration_ms"]),
        options=decode_options,
    ) as native_run:
        while not native_run.complete:
            native_run.step()
            native_control_steps += 1
            if native_control_steps > MAX_DRIVER_STEPS:
                raise RuntimeError("the native control exceeded its step bound")
        native_result = native_run.prepare_result()
    if budget.available != capacity or worker.queue_depth != 0:
        raise RuntimeError("the native control retained runtime capacity")
    native_fixture_text = " ".join(native_result.text.split())
    native_segmented_text = " ".join(
        native_fixture_text for _ in range(int(input_config["fixture_repetitions"]))
    )
    native_segmented_control = {
        "scope": "One 11-second native decode repeated three times; not a 33-second full-window decode.",
        "step_count": native_control_steps,
        "single_fixture_text": native_fixture_text,
        "single_fixture_text_sha256": _sha256_text(native_fixture_text),
        "text": native_segmented_text,
        "text_sha256": _sha256_text(native_segmented_text),
        "raw_result": _plain(native_result),
    }
    chunk_bytes = (
        int(input_config["sample_rate_hz"]) * int(input_config["chunk_ms"]) // 1_000 * 2
    )
    cells: list[dict[str, object]] = []
    stopped_after: str | None = None
    overall_started_ns = time.perf_counter_ns()
    for cell in manifest["cells"]:
        cell_id = str(cell["cell_id"])
        config = continuous.ContinuousStreamConfig(
            **common_config,
            left_context_ms=int(cell["left_context_ms"]),
            holdback_ms=int(cell["holdback_ms"]),
        )
        stream = continuous.ContinuousTranscriptStream(
            adapter,
            stream_id=f"{MANIFEST_ID}:{cell_id}",
            mel_builder=mel_builder,
            options=decode_options,
            rng_seed=rng_seed,
            config=config,
        )
        torch.cuda.reset_peak_memory_stats(0)
        started_ns = time.perf_counter_ns()
        events, traces, driver_steps, accepted_chunks, error_record = _drive_stream(
            stream, pcm, chunk_bytes=chunk_bytes
        )
        torch.cuda.synchronize(0)
        elapsed_ns = time.perf_counter_ns() - started_ns
        state = stream.state
        metrics = stream.metrics
        accepted_samples = min(accepted_chunks * chunk_bytes // 2, total_samples)
        checks = _event_checks(
            events,
            traces,
            accepted_samples=accepted_samples,
            total_samples=total_samples,
            state=state,
            metrics=metrics,
            budget=budget,
            worker=worker,
            capacity=capacity,
            retained_from_sample=stream.retained_from_sample,
            max_buffer_samples=int(config.max_buffer_ms) * 16,
            error_record=error_record,
        )
        lifecycle_names = manifest["outcomes"]["lifecycle_required"]
        lifecycle_passed = all(checks.get(name) is True for name in lifecycle_names)
        policy_resolution = (
            error_record is not None
            and error_record.get("category") == "policy_resolution"
        )
        if not policy_resolution:
            lifecycle_passed = lifecycle_passed and all(
                checks.get(name) is True
                for name in manifest["outcomes"][
                    "complete_stream_required_only_when_no_policy_resolution_error"
                ]
            )
        if error_record is not None and not policy_resolution:
            lifecycle_passed = False
        transcript = _normalized_committed_text(events)
        cells.append(
            {
                "cell_id": cell_id,
                "config": _plain(config),
                "status": (
                    "policy_resolution"
                    if policy_resolution and lifecycle_passed
                    else "completed"
                    if lifecycle_passed
                    else "lifecycle_failure"
                ),
                "lifecycle_passed": lifecycle_passed,
                "error": error_record,
                "driver_steps": driver_steps,
                "accepted_chunks": accepted_chunks,
                "state": _plain(state),
                "metrics": _plain(metrics),
                "events": events,
                "decision_traces": traces,
                "checks": checks,
                "recognition": {
                    "comparison_scope": (
                        "partial_committed_transcript"
                        if policy_resolution
                        else "complete_committed_transcript"
                    ),
                    "text": transcript,
                    "text_sha256": _sha256_text(transcript),
                    "against_offline_full_stream": _word_difference(
                        transcript, str(full_control["text"])
                    ),
                    "against_segmented_control": _word_difference(
                        transcript, segmented_text
                    ),
                    "against_native_segmented_control": _word_difference(
                        transcript, native_segmented_text
                    ),
                },
                "cuda_memory": {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                },
                "timing": {"wall_ns": elapsed_ns, "benchmark": False},
            }
        )
        if not lifecycle_passed:
            stopped_after = cell_id
            break

    function_call_id = str(modal_module.current_function_call_id())
    attempted_all = len(cells) == len(manifest["cells"])
    return {
        "schema_version": "1-diagnostic",
        "recorded_at": _utc_now(),
        "status": "completed" if attempted_all and stopped_after is None else "stopped",
        "claim_boundary": manifest["claim_boundary"],
        "source": {
            "image_base_commit": base_commit,
            "snapshot": snapshot_record,
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
        },
        "model": model_config,
        "input": {
            **input_config,
            "sample_count": total_samples,
            "pcm_s16le_sha256": hashlib.sha256(pcm).hexdigest(),
        },
        "controls": {
            "offline_full_stream": full_control,
            "offline_segmented": segmented_control,
            "native_segmented": native_segmented_control,
            "options": offline_options,
            "rng_seed": rng_seed,
        },
        "cells": cells,
        "run_summary": {
            "cells_registered": len(manifest["cells"]),
            "cells_attempted": len(cells),
            "stopped_after_cell": stopped_after,
            "recognition_differences_are_gates": False,
        },
        "timing": {
            "cell_loop_wall_ns": time.perf_counter_ns() - overall_started_ns,
            "benchmark": False,
        },
    }


def _execute_local_attempt(
    *, root: Path, attempt: int, confirm_paid_gpu: bool, remote_function: object
) -> Path:
    _require_paid_confirmation(confirm_paid_gpu)
    manifest = _read_registration(root)
    output_path, receipt_path = _attempt_paths(root, attempt)
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("the one registered diagnostic attempt already exists")
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
        record = getattr(remote_function, "remote")(snapshot, registration_sha256)
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
    """Use the one manually confirmed diagnostic GPU call."""

    if run_stream_boundary_diagnostic is None:
        raise RuntimeError(f"set {REMOTE_RESOURCES_ENV}=1 before modal run")
    output = _execute_local_attempt(
        root=ROOT,
        attempt=attempt,
        confirm_paid_gpu=confirm_paid_gpu,
        remote_function=run_stream_boundary_diagnostic,
    )
    print(f"Wrote diagnostic record to {output}")


def _define_modal_resources() -> tuple[Any, Any, Any]:
    modal = importlib.import_module("modal")
    if str(modal.__version__) != MODAL_SDK_VERSION:
        raise RuntimeError(
            f"Modal SDK mismatch: expected {MODAL_SDK_VERSION}, observed {modal.__version__}"
        )
    manifest = _read_registration()
    budget = manifest["paid_budget"]
    model_config = manifest["model"]
    base_commit = manifest["source_policy"]["image_base_commit"]
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
            ROOT / COMMON_PRODUCER_PATH,
            f"/opt/whisper-runtime/{COMMON_PRODUCER_PATH}",
            copy=True,
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
        gpu=str(model_config["gpu"]),
        cloud=str(model_config["cloud"]),
        region=str(model_config["region"]),
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
    def run_stream_boundary_diagnostic(
        expected_snapshot: Mapping[str, Any], registration_sha256: str
    ) -> dict[str, Any]:
        producer = importlib.import_module("infra.modal_stream_boundary_diagnostic")
        return producer._run_worker(
            expected_snapshot,
            registration_sha256=registration_sha256,
            modal_module=modal,
        )

    main = app.local_entrypoint(name="main")(_modal_main)
    return app, run_stream_boundary_diagnostic, main


if os.environ.get(REMOTE_RESOURCES_ENV) == "1":
    app, run_stream_boundary_diagnostic, main = _define_modal_resources()
else:
    app = None
    run_stream_boundary_diagnostic = None
    main = None
