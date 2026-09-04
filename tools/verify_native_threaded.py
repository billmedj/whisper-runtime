"""Verify same-model staged decode isolation across two operating-system threads."""

from __future__ import annotations

import argparse
import datetime as dt
import faulthandler
import importlib.metadata
import json
import math
import os
import platform
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TextIO

from smoke_native_whisper import (
    fingerprint_loaded_model,
    verify_loaded_model_fingerprint,
    verify_source_revision,
)
from verify_native_interleaving import (
    PINNED_WHISPER_TREE,
    _checkpoint_path,
    _expect_rejection,
    _finish,
    command_version,
    count_model_hooks,
    decode_results_match,
    git_source,
    model_execution_state_fingerprint,
    model_hook_fingerprint,
    result_record,
    sha256_file,
    source_tree_for_module,
    tensor_fingerprint,
    tensor_storage_identity,
    verify_audio_manifest_binding,
    verify_base_revision,
)

ROOT = Path(__file__).resolve().parents[1]
CHILD_TOKEN_ENV = "WHISPER_RUNTIME_THREADED_CHILD_TOKEN"
EXPECTED_THREADED_ASSERTIONS = frozenset(
    {
        "two_distinct_worker_threads",
        "instrumented_forwards_ran_on_owner_threads",
        "decoder_forward_lifetimes_overlap",
        "forward_instrumentation_restored",
        "request_local_cache_path",
        "state_objects_distinct",
        "kv_cache_storage_disjoint",
        "both_runs_stepped_before_cancellation",
        "cancelled_cleanup_idempotent",
        "cancelled_tensor_and_decoder_state_released",
        "cancelled_run_rejects_step",
        "cancelled_run_rejects_finalize",
        "survivor_cache_unchanged_by_cancellation",
        "survivor_matches_isolated_baseline",
        "survivor_cache_cleanup_complete",
        "survivor_rejects_second_finalize",
        "model_reusable_after_cleanup",
        "model_state_unchanged",
        "model_execution_hooks_unchanged",
        "input_unchanged",
        "prepared_features_unchanged",
        "no_worker_errors",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two staged decodes on one model in distinct operating-system "
            "threads and verify request-state isolation."
        )
    )
    parser.add_argument("audio", help="Path to an audio file accepted by Whisper")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument(
        "--input-label",
        required=True,
        help="Portable repository-relative label for the audio input",
    )
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--download-root")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--expected-text")
    parser.add_argument(
        "--expected-model-fingerprint",
        help="Expected sha256:<hex> fingerprint of the loaded model state",
    )
    parser.add_argument(
        "--numeric-absolute-tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for scalar result comparisons",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for each synchronization point and worker join",
    )
    parser.add_argument(
        "--process-timeout-seconds",
        type=float,
        default=300.0,
        help="Hard timeout enforced by the parent process",
    )
    parser.add_argument("--_child-token", help=argparse.SUPPRESS)
    return parser.parse_args()


def verify_passed_threaded_assertions(assertions: Mapping[str, object]) -> None:
    missing = EXPECTED_THREADED_ASSERTIONS - set(assertions)
    unexpected = set(assertions) - EXPECTED_THREADED_ASSERTIONS
    if missing:
        raise RuntimeError(f"missing check assertions: {', '.join(sorted(missing))}")
    if unexpected:
        raise RuntimeError(
            f"unexpected check assertions: {', '.join(sorted(unexpected))}"
        )
    failed = sorted(name for name, passed in assertions.items() if passed is not True)
    if failed:
        raise RuntimeError(f"check assertions failed: {', '.join(failed)}")


def callable_fingerprint(value: object) -> tuple[int, int]:
    """Identify a bound method without relying on transient bound-method identity."""

    target = getattr(value, "__func__", value)
    owner = getattr(value, "__self__", None)
    return id(owner), id(target)


def verified_child_mode(
    argument_token: str | None,
    environment_token: str | None,
) -> bool:
    """Accept child mode only when the parent supplied the same nonce twice."""

    if argument_token is None:
        return False
    if environment_token is None or not secrets.compare_digest(
        argument_token,
        environment_token,
    ):
        raise RuntimeError("the internal child-process token is invalid")
    return True


def _child_python_executable(environment: dict[str, str]) -> str:
    """Bypass the Windows venv launcher and preserve its import paths."""

    if os.name != "nt" or sys.prefix == sys.base_prefix:
        return sys.executable
    candidate = getattr(sys, "_base_executable", None)
    if not isinstance(candidate, str) or not Path(candidate).is_file():
        return sys.executable
    prefix = Path(sys.prefix).resolve()
    inherited_paths = environment.get("PYTHONPATH", "").split(os.pathsep)
    for entry in sys.path:
        if not entry:
            continue
        resolved = Path(entry).resolve()
        if resolved == prefix or prefix in resolved.parents:
            inherited_paths.append(str(resolved))
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys(path for path in inherited_paths if path)
    )
    return candidate


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> bool:
    """Request tree termination and wait for the verifier child for a bounded time."""

    if process.poll() is not None:
        return True
    tree_termination_requested = False
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=10.0,
            )
            tree_termination_requested = completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            tree_termination_requested = True
        except (OSError, ProcessLookupError):
            pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        return False
    return process.poll() is not None and tree_termination_requested


def _read_capture(stream: BinaryIO) -> str:
    """Read one binary child capture as strict UTF-8."""

    stream.flush()
    stream.seek(0)
    value = stream.read()
    return value.decode("utf-8", errors="strict")


def _write_utf8(stream: TextIO, value: str) -> None:
    """Write UTF-8 without depending on the Windows console code page."""

    encoded = value.encode("utf-8", errors="strict")
    binary_stream = getattr(stream, "buffer", None)
    if binary_stream is None:
        stream.write(value)
        stream.flush()
        return
    binary_stream.write(encoded)
    binary_stream.flush()


def _write_captured_stderr(value: str | None) -> None:
    if value:
        _write_utf8(sys.stderr, value if value.endswith("\n") else f"{value}\n")


class ControlledForwardGate:
    """Prove overlap by rendezvousing inside two live decoder calls."""

    def __init__(
        self,
        decoder_original: Callable[..., object],
        first_block_original: Callable[..., object],
        role_context: threading.local,
        timeout_seconds: float,
    ) -> None:
        self._decoder_original = decoder_original
        self._first_block_original = first_block_original
        self._role_context = role_context
        self._timeout_seconds = timeout_seconds
        self._inner_barrier = threading.Barrier(2)
        self._lock = threading.Lock()
        self._seen_roles: set[str] = set()
        self._inner_barrier_entries: dict[str, int] = {}
        self._active = 0
        self.max_active = 0
        self.records: list[dict[str, object]] = []

    def abort(self) -> None:
        self._inner_barrier.abort()

    def __call__(self, *args: object, **kwargs: object) -> object:
        role = getattr(self._role_context, "role", None)
        if role not in {"cancelled", "survivor"}:
            return self._decoder_original(*args, **kwargs)

        with self._lock:
            if role in self._seen_roles:
                should_gate = False
            else:
                self._seen_roles.add(role)
                should_gate = True

        if not should_gate:
            return self._decoder_original(*args, **kwargs)

        forward_started_ns = time.perf_counter_ns()
        python_ident = threading.get_ident()
        native_id = threading.get_native_id()
        self._role_context.measure_decoder_forward = True
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            return self._decoder_original(*args, **kwargs)
        finally:
            forward_finished_ns = time.perf_counter_ns()
            self._role_context.measure_decoder_forward = False
            with self._lock:
                self._active -= 1
                inner_barrier_entered_ns = self._inner_barrier_entries.get(role)
                self.records.append(
                    {
                        "role": role,
                        "python_thread_id": python_ident,
                        "native_thread_id": native_id,
                        "inner_barrier_entered_ns": inner_barrier_entered_ns,
                        "forward_started_ns": forward_started_ns,
                        "forward_finished_ns": forward_finished_ns,
                    }
                )

    def first_block_forward(self, *args: object, **kwargs: object) -> object:
        """Rendezvous after both workers have entered decoder.forward."""

        role = getattr(self._role_context, "role", None)
        should_wait = role in {"cancelled", "survivor"} and getattr(
            self._role_context, "measure_decoder_forward", False
        )
        if not should_wait:
            return self._first_block_original(*args, **kwargs)

        entered_ns = time.perf_counter_ns()
        with self._lock:
            if role in self._inner_barrier_entries:
                raise RuntimeError("a worker entered the inner forward gate twice")
            self._inner_barrier_entries[role] = entered_ns
        try:
            self._inner_barrier.wait(timeout=self._timeout_seconds)
        except threading.BrokenBarrierError as error:
            raise RuntimeError("the inner decoder-forward barrier failed") from error
        return self._first_block_original(*args, **kwargs)


@contextmanager
def controlled_decoder_forward(
    decoder: object,
    role_context: threading.local,
    timeout_seconds: float,
):
    """Install and then fully remove temporary forward-call instrumentation."""

    first_block = next(iter(getattr(decoder, "blocks")))
    decoder_attributes = vars(decoder)
    block_attributes = vars(first_block)
    had_decoder_forward = "forward" in decoder_attributes
    had_block_forward = "forward" in block_attributes
    previous_decoder_forward = decoder_attributes.get("forward")
    previous_block_forward = block_attributes.get("forward")
    decoder_original = getattr(decoder, "forward")
    first_block_original = getattr(first_block, "forward")
    before = {
        "decoder": callable_fingerprint(decoder_original),
        "first_block": callable_fingerprint(first_block_original),
    }
    gate = ControlledForwardGate(
        decoder_original,
        first_block_original,
        role_context,
        timeout_seconds,
    )
    setattr(decoder, "forward", gate)
    setattr(first_block, "forward", gate.first_block_forward)
    try:
        yield gate, before
    finally:
        gate.abort()
        if had_block_forward:
            setattr(first_block, "forward", previous_block_forward)
        else:
            delattr(first_block, "forward")
        if had_decoder_forward:
            setattr(decoder, "forward", previous_decoder_forward)
        else:
            delattr(decoder, "forward")


def forward_lifetimes_overlap(records: list[dict[str, object]]) -> bool:
    """Return true when the measured decoder call-body intervals overlap."""

    if len(records) != 2 or {record.get("role") for record in records} != {
        "cancelled",
        "survivor",
    }:
        return False
    barriers = [record.get("inner_barrier_entered_ns") for record in records]
    started = [record.get("forward_started_ns") for record in records]
    finished = [record.get("forward_finished_ns") for record in records]
    if not all(isinstance(value, int) for value in (*barriers, *started, *finished)):
        return False
    return (
        max(started) < min(finished)
        and max(barriers) < min(finished)
        and all(
            start <= barrier < finish
            for barrier, start, finish in zip(barriers, started, finished, strict=True)
        )
    )


def _wait_event(event: threading.Event, timeout_seconds: float, label: str) -> None:
    if not event.wait(timeout_seconds):
        raise RuntimeError(f"timed out waiting for {label}")


def _wait_barrier(
    barrier: threading.Barrier,
    timeout_seconds: float,
    label: str,
) -> None:
    try:
        barrier.wait(timeout=timeout_seconds)
    except threading.BrokenBarrierError as error:
        raise RuntimeError(f"the {label} barrier failed") from error


def _abort_barriers(barriers: tuple[threading.Barrier, ...]) -> None:
    for barrier in barriers:
        try:
            barrier.abort()
        except threading.BrokenBarrierError:
            pass


def _thread_record(role: str, started_ns: int) -> dict[str, object]:
    return {
        "role": role,
        "python_thread_id": threading.get_ident(),
        "native_thread_id": threading.get_native_id(),
        "started_ns": started_ns,
        "finished_ns": 0,
        "events": [],
    }


def _event(record: dict[str, object], name: str) -> None:
    events = record["events"]
    if not isinstance(events, list):
        raise TypeError("worker event storage is invalid")
    events.append({"name": name, "at_ns": time.perf_counter_ns()})


def _worker_error_message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def run_payload(args: argparse.Namespace) -> int:
    if (
        not math.isfinite(args.numeric_absolute_tolerance)
        or args.numeric_absolute_tolerance < 0
    ):
        raise ValueError("--numeric-absolute-tolerance must be finite and non-negative")
    if (
        not math.isfinite(args.worker_timeout_seconds)
        or args.worker_timeout_seconds <= 0
    ):
        raise ValueError("--worker-timeout-seconds must be finite and positive")
    raw_input_label = args.input_label
    input_label = PurePosixPath(raw_input_label)
    if (
        input_label.is_absolute()
        or not raw_input_label
        or "\\" in raw_input_label
        or ":" in raw_input_label
        or input_label.as_posix() != raw_input_label
        or ".." in input_label.parts
        or not input_label.parts
    ):
        raise ValueError("--input-label must be a safe repository-relative path")

    import numpy
    import torch
    import whisper
    from whisper.audio import SAMPLE_RATE
    from whisper.decoding import DecodingOptions, DecodingTask, PyTorchInference

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    revision = verify_source_revision(whisper.__file__, args.revision)
    verify_base_revision(whisper.__file__, args.base_revision, revision)
    observed_tree = source_tree_for_module(whisper.__file__)
    if observed_tree != PINNED_WHISPER_TREE:
        raise RuntimeError(
            "the imported Whisper tree does not match the pinned patch series: "
            f"expected {PINNED_WHISPER_TREE}, observed {observed_tree}"
        )
    runtime_source = git_source(ROOT)

    model = whisper.load_model(
        args.model,
        device="cpu",
        download_root=args.download_root,
    ).eval()
    model_fingerprint_before = fingerprint_loaded_model(model)
    execution_state_before = model_execution_state_fingerprint(model)
    verify_loaded_model_fingerprint(
        model_fingerprint_before, args.expected_model_fingerprint
    )
    hooks_before = count_model_hooks(model)
    hook_fingerprint_before = model_hook_fingerprint(model)
    first_decoder_block = next(iter(model.decoder.blocks))
    decoder_forward_before = callable_fingerprint(model.decoder.forward)
    first_block_forward_before = callable_fingerprint(first_decoder_block.forward)

    audio_path = Path(args.audio)
    audio_file_sha256 = sha256_file(audio_path)
    audio_size_bytes = audio_path.stat().st_size
    audio = whisper.load_audio(str(audio_path))
    if len(audio) < 2:
        raise ValueError("the audio fixture must contain at least two samples")
    verify_audio_manifest_binding(
        fixture_id=args.fixture_id,
        file_sha256=audio_file_sha256,
        size_bytes=audio_size_bytes,
        sample_rate_hz=SAMPLE_RATE,
        sample_count=len(audio),
    )
    split = len(audio) // 2
    cancelled_audio = audio[:split]
    survivor_audio = audio
    cancelled_mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(cancelled_audio), n_mels=model.dims.n_mels
    )
    survivor_mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(survivor_audio), n_mels=model.dims.n_mels
    )
    language = "en" if args.model.endswith(".en") else None

    def make_options() -> DecodingOptions:
        return DecodingOptions(
            language=language,
            temperature=0.0,
            without_timestamps=True,
            fp16=False,
        )

    total_started = time.perf_counter()
    encoder_preparation_started = time.perf_counter()
    with torch.no_grad():
        cancelled_features = model.encoder(cancelled_mel.unsqueeze(0))
        survivor_features = model.encoder(survivor_mel.unsqueeze(0))
    encoder_preparation_elapsed = time.perf_counter() - encoder_preparation_started
    if cancelled_features.data_ptr() == survivor_features.data_ptr():
        raise RuntimeError("the prepared audio features share tensor storage")
    cancelled_features_sha256 = tensor_fingerprint(cancelled_features[0])
    survivor_features_sha256 = tensor_fingerprint(survivor_features[0])

    baseline_started = time.perf_counter()
    baseline_run = DecodingTask(model, make_options())._start_run(
        survivor_features.clone()
    )
    try:
        baseline = _finish(baseline_run)
    finally:
        baseline_run.cleanup()
    baseline_elapsed = time.perf_counter() - baseline_started

    worker_tasks = {
        "cancelled": DecodingTask(model, make_options()),
        "survivor": DecodingTask(model, make_options()),
    }
    role_context = threading.local()
    shared_lock = threading.Lock()
    runs: dict[str, object] = {}
    worker_records: dict[str, dict[str, object]] = {}
    worker_results: dict[str, object] = {}
    worker_errors: dict[str, str] = {}
    shared_checks: dict[str, object] = {}

    runs_ready = threading.Barrier(2)
    prefills_ready = threading.Barrier(2)
    steps_ready = threading.Barrier(2)
    first_steps_done = threading.Barrier(2)
    barriers = (runs_ready, prefills_ready, steps_ready, first_steps_done)
    pair_checked = threading.Event()
    prefill_checked = threading.Event()
    cache_checked = threading.Event()
    cancellation_done = threading.Event()

    threaded_started = time.perf_counter()
    with controlled_decoder_forward(
        model.decoder,
        role_context,
        args.worker_timeout_seconds,
    ) as (forward_gate, installed_forward_fingerprints):

        def worker(role: str, audio_features: object) -> None:
            started_ns = time.perf_counter_ns()
            record = _thread_record(role, started_ns)
            role_context.role = role
            run = None
            try:
                run = worker_tasks[role]._start_run(audio_features)
                _event(record, "start")
                with shared_lock:
                    runs[role] = run
                    worker_records[role] = record
                _wait_barrier(runs_ready, args.worker_timeout_seconds, "runs-ready")

                if role == "survivor":
                    cancelled_run = runs["cancelled"]
                    survivor_run = runs["survivor"]
                    if not isinstance(
                        cancelled_run.inference, PyTorchInference
                    ) or not isinstance(survivor_run.inference, PyTorchInference):
                        raise RuntimeError(
                            "the check requires the built-in PyTorch decoder"
                        )
                    request_local = not (
                        cancelled_run.inference._use_legacy_cache
                        or survivor_run.inference._use_legacy_cache
                    )
                    if not request_local:
                        raise RuntimeError(
                            "the check requires the request-local cache path"
                        )
                    state_distinct = all(
                        (
                            cancelled_run is not survivor_run,
                            cancelled_run.task is not survivor_run.task,
                            cancelled_run.inference is not survivor_run.inference,
                            cancelled_run.decoder is not survivor_run.decoder,
                            cancelled_run.inference.kv_cache
                            is not survivor_run.inference.kv_cache,
                            cancelled_run.audio_features.data_ptr()
                            != survivor_run.audio_features.data_ptr(),
                            cancelled_run.tokens.data_ptr()
                            != survivor_run.tokens.data_ptr(),
                            cancelled_run.sum_logprobs.data_ptr()
                            != survivor_run.sum_logprobs.data_ptr(),
                            cancelled_run.no_speech_probs
                            is not survivor_run.no_speech_probs,
                        )
                    )
                    if not state_distinct:
                        raise RuntimeError("the two threaded runs share mutable state")
                    shared_checks["request_local_cache_path"] = request_local
                    shared_checks["state_objects_distinct"] = state_distinct
                    pair_checked.set()
                else:
                    _wait_event(
                        pair_checked,
                        args.worker_timeout_seconds,
                        "run-pair inspection",
                    )

                run.prefill()
                _event(record, "prefill")
                _wait_barrier(
                    prefills_ready,
                    args.worker_timeout_seconds,
                    "prefills-ready",
                )
                if role == "survivor":
                    cancelled_run = runs["cancelled"]
                    survivor_run = runs["survivor"]
                    pending_logits_distinct = (
                        cancelled_run._pending_logits is not None
                        and survivor_run._pending_logits is not None
                        and cancelled_run._pending_logits.data_ptr()
                        != survivor_run._pending_logits.data_ptr()
                    )
                    if not pending_logits_distinct:
                        raise RuntimeError("the two threaded runs share pending logits")
                    shared_checks["pending_logits_distinct"] = True
                    prefill_checked.set()
                else:
                    _wait_event(
                        prefill_checked,
                        args.worker_timeout_seconds,
                        "prefill inspection",
                    )

                _wait_barrier(steps_ready, args.worker_timeout_seconds, "steps-ready")
                if run.step():
                    raise RuntimeError(
                        f"the {role} run completed before isolation was observed"
                    )
                _event(record, "step:1")
                _wait_barrier(
                    first_steps_done,
                    args.worker_timeout_seconds,
                    "first-steps-done",
                )

                if role == "cancelled":
                    _wait_event(
                        cache_checked,
                        args.worker_timeout_seconds,
                        "survivor cache snapshot",
                    )
                    cancelled_inference = run.inference
                    run.cleanup()
                    _event(record, "cleanup")
                    run.cleanup()
                    _event(record, "cleanup:idempotent")
                    if cancelled_inference.kv_cache:
                        raise RuntimeError("the cancelled run retained cache entries")
                    state_released = all(
                        getattr(run, field, None) is None
                        for field in (
                            "audio_features",
                            "tokens",
                            "sum_logprobs",
                            "inference",
                            "decoder",
                            "_pending_logits",
                        )
                    )
                    if not state_released:
                        raise RuntimeError(
                            "the cancelled run retained request-owned state"
                        )
                    step_rejection = _expect_rejection(run.step, "cancelled")
                    _event(record, "step:rejected")
                    finalize_rejection = _expect_rejection(run.finalize, "cancelled")
                    _event(record, "finalize:rejected")
                    with shared_lock:
                        worker_results[role] = {
                            "cleanup_idempotent": True,
                            "state_released": state_released,
                            "step_rejected": "cancelled" in step_rejection,
                            "finalize_rejected": "cancelled" in finalize_rejection,
                            "steps": 1,
                        }
                    cancellation_done.set()
                    return

                cancelled_run = runs["cancelled"]
                cancelled_inference = cancelled_run.inference
                survivor_inference = run.inference
                if not cancelled_inference.kv_cache or not survivor_inference.kv_cache:
                    raise RuntimeError("both request-local caches must be populated")
                cancelled_storage = {
                    tensor_storage_identity(value)
                    for value in cancelled_inference.kv_cache.values()
                }
                survivor_storage = {
                    tensor_storage_identity(value)
                    for value in survivor_inference.kv_cache.values()
                }
                cache_storage_disjoint = cancelled_storage.isdisjoint(survivor_storage)
                if not cache_storage_disjoint:
                    raise RuntimeError(
                        "the two threaded runs share decoder cache storage"
                    )
                survivor_cache_identity = id(survivor_inference.kv_cache)
                survivor_cache_snapshot = {
                    key: (value, tensor_storage_identity(value), value.clone())
                    for key, value in survivor_inference.kv_cache.items()
                }
                survivor_cache_keys = tuple(survivor_inference.kv_cache)
                shared_checks["kv_cache_storage_disjoint"] = cache_storage_disjoint
                shared_checks["survivor_cache_entries_at_cancellation"] = len(
                    survivor_cache_snapshot
                )
                cache_checked.set()

                _wait_event(
                    cancellation_done,
                    args.worker_timeout_seconds,
                    "cancelled-run cleanup",
                )
                survivor_cache_unchanged = (
                    id(survivor_inference.kv_cache) == survivor_cache_identity
                    and tuple(survivor_inference.kv_cache) == survivor_cache_keys
                    and all(
                        survivor_inference.kv_cache[key] is original
                        and tensor_storage_identity(survivor_inference.kv_cache[key])
                        == storage
                        and torch.equal(survivor_inference.kv_cache[key], snapshot)
                        for key, (original, storage, snapshot) in (
                            survivor_cache_snapshot.items()
                        )
                    )
                )
                if not survivor_cache_unchanged:
                    raise RuntimeError("cancelling one run changed the survivor cache")

                while not run.complete:
                    run.step()
                    _event(record, f"step:{run.step_index}")
                results = run.finalize()
                _event(record, "finalize")
                if not isinstance(results, list) or len(results) != 1:
                    raise RuntimeError("the survivor did not return exactly one result")
                survivor = results[0]
                cache_cleanup_complete = not survivor_inference.kv_cache
                if not cache_cleanup_complete:
                    raise RuntimeError("the survivor retained cache entries")
                finalize_rejection = _expect_rejection(run.finalize, "finalized")
                _event(record, "finalize:rejected")
                run.cleanup()
                _event(record, "cleanup:idempotent")
                with shared_lock:
                    worker_results[role] = {
                        "result": survivor,
                        "cache_unchanged": survivor_cache_unchanged,
                        "cache_cleanup_complete": cache_cleanup_complete,
                        "finalize_rejected": "finalized" in finalize_rejection,
                        "steps": run.step_index,
                    }
            except BaseException as error:
                with shared_lock:
                    worker_errors[role] = _worker_error_message(error)
                _abort_barriers(barriers)
                pair_checked.set()
                prefill_checked.set()
                cache_checked.set()
                cancellation_done.set()
                forward_gate.abort()
            finally:
                if run is not None:
                    try:
                        run.cleanup()
                    except BaseException as cleanup_error:
                        with shared_lock:
                            worker_errors.setdefault(
                                role,
                                _worker_error_message(cleanup_error),
                            )
                record["finished_ns"] = time.perf_counter_ns()
                with shared_lock:
                    worker_records[role] = record

        workers = [
            threading.Thread(
                target=worker,
                args=("cancelled", cancelled_features),
                name="whisper-cancelled-run",
                daemon=True,
            ),
            threading.Thread(
                target=worker,
                args=("survivor", survivor_features),
                name="whisper-survivor-run",
                daemon=True,
            ),
        ]
        for thread in workers:
            thread.start()
        join_deadline = time.monotonic() + args.worker_timeout_seconds
        for thread in workers:
            thread.join(timeout=max(0.0, join_deadline - time.monotonic()))
        alive = [thread.name for thread in workers if thread.is_alive()]
        if alive:
            _abort_barriers(barriers)
            forward_gate.abort()
            pair_checked.set()
            prefill_checked.set()
            cache_checked.set()
            cancellation_done.set()
            for thread in workers:
                thread.join(timeout=1.0)
            raise RuntimeError(f"worker thread did not finish: {', '.join(alive)}")
        if worker_errors:
            details = "; ".join(
                f"{role}: {message}" for role, message in sorted(worker_errors.items())
            )
            raise RuntimeError(f"threaded decode failed: {details}")

        forward_records = sorted(
            forward_gate.records,
            key=lambda record: str(record["role"]),
        )
        max_forward_calls = forward_gate.max_active

    threaded_elapsed = time.perf_counter() - threaded_started
    decoder_forward_after = callable_fingerprint(model.decoder.forward)
    first_block_forward_after = callable_fingerprint(first_decoder_block.forward)
    instrumentation_restored = (
        decoder_forward_before == installed_forward_fingerprints["decoder"]
        and first_block_forward_before == installed_forward_fingerprints["first_block"]
        and decoder_forward_after == decoder_forward_before
        and first_block_forward_after == first_block_forward_before
    )
    if not instrumentation_restored:
        raise RuntimeError("decoder forward instrumentation was not fully restored")

    survivor_record = worker_results.get("survivor")
    cancelled_record = worker_results.get("cancelled")
    if not isinstance(survivor_record, dict) or not isinstance(cancelled_record, dict):
        raise RuntimeError("one threaded worker did not publish a terminal result")
    survivor = survivor_record.get("result")
    if survivor is None or not decode_results_match(
        baseline,
        survivor,
        absolute_tolerance=args.numeric_absolute_tolerance,
    ):
        raise RuntimeError("the threaded survivor differs from its isolated baseline")
    if args.expected_text is not None and survivor.text != args.expected_text:
        raise RuntimeError(
            "the surviving transcript does not match --expected-text: "
            f"expected {args.expected_text!r}, observed {survivor.text!r}"
        )

    reuse_started = time.perf_counter()
    reuse_run = DecodingTask(model, make_options())._start_run(
        survivor_features.clone()
    )
    try:
        reuse = _finish(reuse_run)
    finally:
        reuse_run.cleanup()
    reuse_elapsed = time.perf_counter() - reuse_started
    reusable = decode_results_match(
        baseline,
        reuse,
        absolute_tolerance=args.numeric_absolute_tolerance,
    )
    if not reusable:
        raise RuntimeError("the model changed after threaded run cleanup")

    model_fingerprint_after = fingerprint_loaded_model(model)
    execution_state_after = model_execution_state_fingerprint(model)
    hooks_after = count_model_hooks(model)
    hook_fingerprint_after = model_hook_fingerprint(model)
    thread_records = sorted(
        worker_records.values(), key=lambda record: str(record["role"])
    )
    thread_ids_distinct = (
        len(thread_records) == 2
        and len({record["python_thread_id"] for record in thread_records}) == 2
        and len({record["native_thread_id"] for record in thread_records}) == 2
    )
    overlap = forward_lifetimes_overlap(forward_records)
    events_by_role = {
        str(thread["role"]): {
            str(event["name"]): event["at_ns"]
            for event in thread["events"]
            if isinstance(event, dict)
        }
        for thread in thread_records
    }
    both_runs_stepped_before_cancellation = False
    if set(events_by_role) == {"cancelled", "survivor"}:
        cancelled_step_ns = events_by_role["cancelled"].get("step:1")
        survivor_step_ns = events_by_role["survivor"].get("step:1")
        cleanup_ns = events_by_role["cancelled"].get("cleanup")
        both_runs_stepped_before_cancellation = (
            isinstance(cancelled_step_ns, int)
            and isinstance(survivor_step_ns, int)
            and isinstance(cleanup_ns, int)
            and max(cancelled_step_ns, survivor_step_ns) < cleanup_ns
        )
    input_unchanged = (
        sha256_file(audio_path) == audio_file_sha256
        and audio_path.stat().st_size == audio_size_bytes
    )
    prepared_features_unchanged = (
        tensor_fingerprint(cancelled_features[0]) == cancelled_features_sha256
        and tensor_fingerprint(survivor_features[0]) == survivor_features_sha256
    )
    assertions = {
        "two_distinct_worker_threads": thread_ids_distinct,
        "instrumented_forwards_ran_on_owner_threads": all(
            any(
                call["role"] == thread["role"]
                and call["python_thread_id"] == thread["python_thread_id"]
                and call["native_thread_id"] == thread["native_thread_id"]
                for call in forward_records
            )
            for thread in thread_records
        ),
        "decoder_forward_lifetimes_overlap": overlap and max_forward_calls == 2,
        "forward_instrumentation_restored": instrumentation_restored,
        "request_local_cache_path": shared_checks.get("request_local_cache_path"),
        "state_objects_distinct": (
            shared_checks.get("state_objects_distinct") is True
            and shared_checks.get("pending_logits_distinct") is True
        ),
        "kv_cache_storage_disjoint": shared_checks.get("kv_cache_storage_disjoint"),
        "both_runs_stepped_before_cancellation": (
            both_runs_stepped_before_cancellation
        ),
        "cancelled_cleanup_idempotent": cancelled_record.get("cleanup_idempotent"),
        "cancelled_tensor_and_decoder_state_released": cancelled_record.get(
            "state_released"
        ),
        "cancelled_run_rejects_step": cancelled_record.get("step_rejected"),
        "cancelled_run_rejects_finalize": cancelled_record.get("finalize_rejected"),
        "survivor_cache_unchanged_by_cancellation": survivor_record.get(
            "cache_unchanged"
        ),
        "survivor_matches_isolated_baseline": True,
        "survivor_cache_cleanup_complete": survivor_record.get(
            "cache_cleanup_complete"
        ),
        "survivor_rejects_second_finalize": survivor_record.get("finalize_rejected"),
        "model_reusable_after_cleanup": reusable,
        "model_state_unchanged": (
            model_fingerprint_after == model_fingerprint_before
            and execution_state_after == execution_state_before
        ),
        "model_execution_hooks_unchanged": (
            hooks_after == hooks_before
            and hook_fingerprint_after == hook_fingerprint_before
        ),
        "input_unchanged": input_unchanged,
        "prepared_features_unchanged": prepared_features_unchanged,
        "no_worker_errors": not worker_errors,
    }
    verify_passed_threaded_assertions(assertions)

    checkpoint = _checkpoint_path(args.model, args.download_root)
    manifest = ROOT / "patches" / "openai-whisper" / "SHA256SUMS"
    record: dict[str, Any] = {
        "schema_version": "1",
        "recorded_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "passed",
        "scope": "patched_backend",
        "subject": {
            "layer": "patched_openai_whisper_backend",
            "entrypoint": "whisper.decoding.DecodingTask._start_run",
            "runtime_adapter_exercised": False,
            "scheduler_exercised": False,
            "encoder_concurrency_exercised": False,
        },
        "runtime": {
            "version": importlib.metadata.version("whisper-execution-runtime"),
            **runtime_source,
        },
        "backend": {
            "name": "openai-whisper-suspendable",
            "base_commit": args.base_revision,
            "applied_commit": revision,
            "git_tree": observed_tree,
            "clean": True,
            "patch_manifest": "patches/openai-whisper/SHA256SUMS",
            "patch_manifest_sha256": sha256_file(manifest),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "process_id": __import__("os").getpid(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "tiktoken": importlib.metadata.version("tiktoken"),
            "numba": importlib.metadata.version("numba"),
            "tqdm": importlib.metadata.version("tqdm"),
            "more_itertools": importlib.metadata.version("more-itertools"),
            "jsonschema": importlib.metadata.version("jsonschema"),
            "ffmpeg": command_version("ffmpeg"),
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "model": {
            "name": args.model,
            "device": "cpu",
            "checkpoint_sha256": sha256_file(checkpoint),
            "loaded_state_before": model_fingerprint_before,
            "loaded_state_after": model_fingerprint_after,
            "execution_state_before": execution_state_before,
            "execution_state_after": execution_state_after,
        },
        "explicit_decode_options": {
            "language": language,
            "temperature": 0.0,
            "without_timestamps": True,
            "fp16": False,
        },
        "input": {
            "fixture_id": args.fixture_id,
            "path": input_label.as_posix(),
            "file_sha256": audio_file_sha256,
            "size_bytes": audio_size_bytes,
            "sample_rate_hz": SAMPLE_RATE,
            "source_sample_count": len(audio),
            "cancelled": {
                "sample_start": 0,
                "sample_end": split,
                "pcm_sha256": tensor_fingerprint(cancelled_audio),
                "mel_sha256": tensor_fingerprint(cancelled_mel),
                "encoded_features_sha256": cancelled_features_sha256,
            },
            "survivor": {
                "sample_start": 0,
                "sample_end": len(audio),
                "pcm_sha256": tensor_fingerprint(survivor_audio),
                "mel_sha256": tensor_fingerprint(survivor_mel),
                "encoded_features_sha256": survivor_features_sha256,
            },
        },
        "execution": {
            "mode": "two_os_threads_decoder_body_overlap",
            "thread_count": 2,
            "synchronization": "first_decoder_block_barrier_and_events",
            "encoder_preparation": "sequential_encoder_passes",
            "controlled_forward_calls": len(forward_records),
            "maximum_instrumented_forward_calls_live": max_forward_calls,
            "kernel_overlap_measured": False,
            "parallel_kernel_execution_claimed": False,
            "cancellation": "owner_thread_cleanup_after_first_step",
            "cancelled_steps": cancelled_record["steps"],
            "survivor_steps": survivor_record["steps"],
            "survivor_cache_entries_at_cancellation": shared_checks[
                "survivor_cache_entries_at_cancellation"
            ],
            "numeric_absolute_tolerance": args.numeric_absolute_tolerance,
            "worker_timeout_seconds": args.worker_timeout_seconds,
            "process_timeout_seconds": args.process_timeout_seconds,
            "timing_is_benchmark": False,
            "elapsed_seconds": {
                "encoder_preparation": encoder_preparation_elapsed,
                "baseline": baseline_elapsed,
                "threaded": threaded_elapsed,
                "reuse_control": reuse_elapsed,
                "total": time.perf_counter() - total_started,
            },
            "threads": thread_records,
            "controlled_forward_lifetimes": forward_records,
        },
        "assertions": assertions,
        "results": {
            "isolated_baseline": result_record(baseline),
            "survivor": result_record(survivor),
            "reuse_control": result_record(reuse),
        },
    }
    from validate_threaded_record import validate_threaded_record

    validation_failures = validate_threaded_record(record, "generated record")
    if validation_failures:
        raise RuntimeError("; ".join(validation_failures))
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def main() -> int:
    args = parse_args()
    if (
        not math.isfinite(args.process_timeout_seconds)
        or args.process_timeout_seconds <= 0
    ):
        raise ValueError("--process-timeout-seconds must be finite and positive")
    if verified_child_mode(args._child_token, os.environ.get(CHILD_TOKEN_ENV)):
        faulthandler.dump_traceback_later(
            max(0.001, args.process_timeout_seconds * 0.9),
            repeat=False,
            file=sys.stderr,
        )
        try:
            return run_payload(args)
        finally:
            faulthandler.cancel_dump_traceback_later()

    child_token = secrets.token_urlsafe(32)
    child_environment = os.environ.copy()
    child_environment[CHILD_TOKEN_ENV] = child_token
    child_environment["PYTHONIOENCODING"] = "utf-8:strict"
    command = [
        _child_python_executable(child_environment),
        "-X",
        "utf8",
        "-B",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--_child-token",
        child_token,
    ]
    process_options: dict[str, object] = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True
    with (
        tempfile.TemporaryFile(mode="w+b") as child_stdout_capture,
        tempfile.TemporaryFile(mode="w+b") as child_stderr_capture,
    ):
        process = subprocess.Popen(
            command,
            stdout=child_stdout_capture,
            stderr=child_stderr_capture,
            env=child_environment,
            **process_options,
        )
        try:
            process.wait(timeout=args.process_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            terminated = _terminate_process_tree(process)
            child_stderr = _read_capture(child_stderr_capture)
            _write_captured_stderr(child_stderr)
            detail = ""
            if not terminated:
                detail = "; process-tree termination could not be confirmed"
            raise RuntimeError(
                "the threaded verifier exceeded its process timeout" + detail
            ) from error
        child_stdout = _read_capture(child_stdout_capture)
        child_stderr = _read_capture(child_stderr_capture)
    _write_captured_stderr(child_stderr)
    if process.returncode != 0:
        detail = child_stderr.strip() or child_stdout.strip()
        raise RuntimeError(
            "the threaded verifier child failed with exit code "
            f"{process.returncode}: {detail}"
        )
    try:
        record = json.loads(child_stdout)
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "the threaded verifier child did not emit one finite JSON record"
        ) from error
    _write_utf8(sys.stdout, f"{serialized}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
