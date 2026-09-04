"""Verify two real-model transactions through the native runtime adapter."""

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
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from smoke_native_whisper import (
    fingerprint_loaded_model,
    verify_loaded_model_fingerprint,
    verify_source_revision,
)
from verify_native_interleaving import (
    PINNED_WHISPER_TREE,
    _checkpoint_path,
    command_version,
    git_source,
    model_execution_state_fingerprint,
    model_hook_fingerprint,
    sha256_file,
    source_tree_for_module,
    tensor_fingerprint,
    tensor_storage_identity,
    verify_audio_manifest_binding,
    verify_base_revision,
)
from verify_native_threaded import (
    _child_python_executable,
    _read_capture,
    _terminate_process_tree,
    _write_captured_stderr,
    _write_utf8,
    callable_fingerprint,
    controlled_decoder_forward,
    forward_lifetimes_overlap,
    verified_child_mode,
)

ROOT = Path(__file__).resolve().parents[1]
CHILD_TOKEN_ENV = "WHISPER_RUNTIME_ADAPTER_CONCURRENCY_CHILD_TOKEN"
ROLES = ("cancelled", "survivor")
EXPECTED_RUNTIME_ASSERTIONS = frozenset(
    {
        "two_distinct_caller_threads",
        "runtime_adapter_exercised",
        "two_transactions_admitted",
        "full_declared_capacity_reserved_at_overlap",
        "start_run_serialized",
        "instrumented_forwards_ran_on_owner_threads",
        "decoder_forward_lifetimes_overlap",
        "request_local_cache_path",
        "state_objects_distinct",
        "kv_cache_storage_disjoint",
        "both_runs_stepped_before_cancellation",
        "cancel_request_accepted",
        "cancelled_request_did_not_commit",
        "cancelled_state_released",
        "cancelled_lease_released_after_cleanup",
        "survivor_cache_unchanged_by_cancelled_cleanup",
        "survivor_committed_once",
        "survivor_matches_isolated_baseline",
        "final_queue_empty",
        "final_budget_restored",
        "model_state_unchanged",
        "model_execution_hooks_unchanged",
        "instrumentation_restored",
        "model_reusable_after_transactions",
        "input_unchanged",
        "mel_inputs_unchanged",
        "no_unexpected_worker_errors",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def verify_passed_runtime_assertions(assertions: Mapping[str, object]) -> None:
    missing = EXPECTED_RUNTIME_ASSERTIONS - set(assertions)
    unexpected = set(assertions) - EXPECTED_RUNTIME_ASSERTIONS
    if missing:
        raise RuntimeError(f"missing check assertions: {', '.join(sorted(missing))}")
    if unexpected:
        raise RuntimeError(
            f"unexpected check assertions: {', '.join(sorted(unexpected))}"
        )
    failed = sorted(name for name, value in assertions.items() if value is not True)
    if failed:
        raise RuntimeError(f"check assertions failed: {', '.join(failed)}")


def _wait_event(event: threading.Event, timeout: float, label: str) -> None:
    if not event.wait(timeout):
        raise RuntimeError(f"timed out waiting for {label}")


def _wait_until(predicate: Callable[[], bool], timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for {label}")
        time.sleep(0.01)


def _event(record: dict[str, object], name: str) -> None:
    events = record.get("events")
    if not isinstance(events, list):
        raise TypeError("worker event storage is invalid")
    events.append({"name": name, "at_ns": time.perf_counter_ns()})


def _thread_record(role: str) -> dict[str, object]:
    return {
        "role": role,
        "python_thread_id": threading.get_ident(),
        "native_thread_id": threading.get_native_id(),
        "started_ns": time.perf_counter_ns(),
        "finished_ns": 0,
        "events": [],
    }


def _resource_record(resources: object) -> dict[str, int]:
    return {
        "memory_bytes": int(getattr(resources, "memory_bytes")),
        "compute_units": int(getattr(resources, "compute_units")),
        "stream_slots": int(getattr(resources, "stream_slots")),
    }


def _request_session_record(request: object, session: object) -> dict[str, object]:
    state = session.snapshot()
    return {
        "request_status": request.status.value,
        "session_version": state.version,
        "window_count": len(state.windows),
    }


def _runtime_snapshot(
    label: str,
    worker: object,
    budget: object,
    requests: Mapping[str, object],
    sessions: Mapping[str, object],
) -> dict[str, object]:
    return {
        "label": label,
        "at_ns": time.perf_counter_ns(),
        "queue_depth": worker.queue_depth,
        "lease_count": budget.lease_count,
        "capacity": _resource_record(budget.capacity),
        "available": _resource_record(budget.available),
        "in_use": _resource_record(budget.in_use),
        "cancelled": _request_session_record(
            requests["cancelled"], sessions["cancelled"]
        ),
        "survivor": _request_session_record(requests["survivor"], sessions["survivor"]),
    }


class RuntimeRunProbe:
    """Observe native runs created by two adapter calls without changing results."""

    def __init__(
        self,
        role_context: threading.local,
        worker_records: dict[str, dict[str, object]],
        shared_lock: threading.Lock,
        timeout_seconds: float,
    ) -> None:
        self.role_context = role_context
        self.worker_records = worker_records
        self.shared_lock = shared_lock
        self.timeout_seconds = timeout_seconds
        self.runs: dict[str, object] = {}
        self.feature_fingerprints: dict[str, str] = {}
        self.start_intervals: list[dict[str, object]] = []
        self.first_steps = {role: threading.Event() for role in ROLES}
        self.first_step_barrier = threading.Barrier(2)
        self.release_cancelled = threading.Event()
        self.release_survivor = threading.Event()
        self.cancelled_cleanup_complete = threading.Event()
        self._start_active = 0
        self.max_start_active = 0

    def abort(self) -> None:
        self.first_step_barrier.abort()
        self.release_cancelled.set()
        self.release_survivor.set()
        self.cancelled_cleanup_complete.set()

    def _record_event(self, role: str, name: str) -> None:
        with self.shared_lock:
            record = self.worker_records.get(role)
            if record is None:
                raise RuntimeError(f"the {role} worker record is unavailable")
            _event(record, name)

    def start_run(self, original: Callable[..., object], *args: object) -> object:
        role = getattr(self.role_context, "role", None)
        if role not in ROLES:
            return original(*args)
        self._record_event(role, "start_run:begin")
        started_ns = time.perf_counter_ns()
        with self.shared_lock:
            self._start_active += 1
            self.max_start_active = max(self.max_start_active, self._start_active)
        try:
            run = original(*args)
        finally:
            finished_ns = time.perf_counter_ns()
            with self.shared_lock:
                self._start_active -= 1
                self.start_intervals.append(
                    {
                        "role": role,
                        "python_thread_id": threading.get_ident(),
                        "native_thread_id": threading.get_native_id(),
                        "started_ns": started_ns,
                        "finished_ns": finished_ns,
                    }
                )
        self._record_event(role, "start_run:end")
        with self.shared_lock:
            self.runs[role] = run
            self.feature_fingerprints[role] = tensor_fingerprint(
                getattr(run, "audio_features")[0]
            )
        self._instrument_run(role, run)
        return run

    def _instrument_run(self, role: str, run: object) -> None:
        original_prefill = getattr(run, "prefill")
        original_step = getattr(run, "step")
        original_finalize = getattr(run, "finalize")
        original_cleanup = getattr(run, "cleanup")
        step_count = 0

        def prefill() -> object:
            self._record_event(role, "prefill:begin")
            result = original_prefill()
            self._record_event(role, "prefill:end")
            return result

        def step() -> object:
            nonlocal step_count
            result = original_step()
            step_count += 1
            self._record_event(role, f"step:{step_count}")
            if step_count != 1:
                return result
            if result is not False:
                raise RuntimeError(
                    f"the {role} run completed before cancellation isolation "
                    "could be observed"
                )
            self.first_steps[role].set()
            try:
                self.first_step_barrier.wait(timeout=self.timeout_seconds)
            except threading.BrokenBarrierError as error:
                raise RuntimeError("the first-step barrier failed") from error
            release = (
                self.release_cancelled if role == "cancelled" else self.release_survivor
            )
            _wait_event(
                release,
                self.timeout_seconds,
                f"release of the {role} first step",
            )
            return result

        def finalize() -> object:
            result = original_finalize()
            self._record_event(role, "finalize")
            return result

        def cleanup() -> object:
            self._record_event(role, "cleanup:begin")
            try:
                return original_cleanup()
            finally:
                self._record_event(role, "cleanup:end")
                if role == "cancelled":
                    self.cancelled_cleanup_complete.set()

        setattr(run, "prefill", prefill)
        setattr(run, "step", step)
        setattr(run, "finalize", finalize)
        setattr(run, "cleanup", cleanup)


@contextmanager
def controlled_start_runs(task_type: type, probe: RuntimeRunProbe):
    """Install temporary DecodingTask._start_run instrumentation."""

    attributes = vars(task_type)
    previous = attributes.get("_start_run")
    if previous is None or not callable(previous):
        raise RuntimeError("DecodingTask._start_run is unavailable")
    before = callable_fingerprint(previous)

    def instrumented(task: object, mel: object) -> object:
        return probe.start_run(previous, task, mel)

    setattr(task_type, "_start_run", instrumented)
    try:
        yield before
    finally:
        probe.abort()
        setattr(task_type, "_start_run", previous)


def _cache_snapshot(cache: Mapping[object, object]) -> dict[str, object]:
    return {
        "identity": id(cache),
        "keys": tuple(cache),
        "values": {
            key: (
                value,
                tensor_storage_identity(value),
                value.detach().clone(),
            )
            for key, value in cache.items()
        },
    }


def _cache_matches(
    cache: Mapping[object, object], snapshot: Mapping[str, object]
) -> bool:
    values = snapshot.get("values")
    keys = snapshot.get("keys")
    if not isinstance(values, dict) or not isinstance(keys, tuple):
        return False
    try:
        import torch

        return (
            id(cache) == snapshot.get("identity")
            and tuple(cache) == keys
            and all(
                cache[key] is original
                and tensor_storage_identity(cache[key]) == storage
                and torch.equal(cache[key], copy)
                for key, (original, storage, copy) in values.items()
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _state_isolation(runs: Mapping[str, object]) -> tuple[bool, bool, bool, int]:
    cancelled = runs["cancelled"]
    survivor = runs["survivor"]
    cancelled_inference = cancelled.inference
    survivor_inference = survivor.inference
    request_local = (
        getattr(cancelled_inference, "_use_legacy_cache", None) is False
        and getattr(survivor_inference, "_use_legacy_cache", None) is False
        and getattr(cancelled, "_legacy_cache_lock", None) is None
        and getattr(survivor, "_legacy_cache_lock", None) is None
    )
    distinct = all(
        (
            cancelled is not survivor,
            cancelled.task is not survivor.task,
            cancelled_inference is not survivor_inference,
            cancelled.decoder is not survivor.decoder,
            cancelled_inference.kv_cache is not survivor_inference.kv_cache,
            cancelled.audio_features.data_ptr() != survivor.audio_features.data_ptr(),
            cancelled.tokens.data_ptr() != survivor.tokens.data_ptr(),
            cancelled.sum_logprobs.data_ptr() != survivor.sum_logprobs.data_ptr(),
            cancelled.no_speech_probs is not survivor.no_speech_probs,
        )
    )
    cancelled_storage = {
        tensor_storage_identity(value)
        for value in cancelled_inference.kv_cache.values()
    }
    survivor_storage = {
        tensor_storage_identity(value) for value in survivor_inference.kv_cache.values()
    }
    disjoint = (
        bool(cancelled_storage)
        and bool(survivor_storage)
        and (cancelled_storage.isdisjoint(survivor_storage))
    )
    return request_local, distinct, disjoint, len(survivor_storage)


def _validate_input_label(raw_label: str) -> str:
    label = PurePosixPath(raw_label)
    if (
        label.is_absolute()
        or not raw_label
        or "\\" in raw_label
        or ":" in raw_label
        or label.as_posix() != raw_label
        or ".." in label.parts
        or not label.parts
    ):
        raise ValueError("--input-label must be a safe repository-relative path")
    return label.as_posix()


def run_payload(args: argparse.Namespace) -> int:
    if args.model != "tiny.en":
        raise ValueError("this verifier requires --model tiny.en")
    if (
        not math.isfinite(args.worker_timeout_seconds)
        or args.worker_timeout_seconds <= 0
    ):
        raise ValueError("--worker-timeout-seconds must be finite and positive")
    input_label = _validate_input_label(args.input_label)

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
    from whisper_runtime.adapters import (
        NativeDecodeOptions,
        NativeExecutionProfile,
        NativeWhisperAdapter,
    )
    from whisper_runtime.errors import RequestCancelledError

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
    verify_loaded_model_fingerprint(
        model_fingerprint_before, args.expected_model_fingerprint
    )
    execution_state_before = model_execution_state_fingerprint(model)
    hook_fingerprint_before = model_hook_fingerprint(model)
    decoder_forward_before = callable_fingerprint(model.decoder.forward)
    first_block = next(iter(model.decoder.blocks))
    first_block_forward_before = callable_fingerprint(first_block.forward)
    start_run_before = callable_fingerprint(DecodingTask._start_run)

    snapshot = ModelSnapshot(
        model_id=args.model,
        revision=revision,
        backend="pytorch-cpu",
        fingerprint=model_fingerprint_before,
    )

    def identity_probe(observed: object) -> ModelSnapshot:
        return ModelSnapshot(
            model_id=args.model,
            revision=revision,
            backend="pytorch-cpu",
            fingerprint=fingerprint_loaded_model(observed),
        )

    audio_path = Path(args.audio)
    audio_sha256 = sha256_file(audio_path)
    audio_size = audio_path.stat().st_size
    audio = whisper.load_audio(str(audio_path))
    if len(audio) < 2:
        raise ValueError("the audio fixture must contain at least two samples")
    verify_audio_manifest_binding(
        fixture_id=args.fixture_id,
        file_sha256=audio_sha256,
        size_bytes=audio_size,
        sample_rate_hz=SAMPLE_RATE,
        sample_count=len(audio),
    )
    split = len(audio) // 2
    total_started = time.perf_counter()
    input_started = time.perf_counter()
    cancelled_audio = audio[:split]
    survivor_audio = audio
    cancelled_mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(cancelled_audio), n_mels=model.dims.n_mels
    )
    survivor_mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(survivor_audio), n_mels=model.dims.n_mels
    )
    cancelled_mel_sha256 = tensor_fingerprint(cancelled_mel)
    survivor_mel_sha256 = tensor_fingerprint(survivor_mel)
    input_elapsed = time.perf_counter() - input_started

    def direct_options() -> DecodingOptions:
        return DecodingOptions(
            language="en",
            temperature=0.0,
            without_timestamps=True,
            fp16=False,
        )

    baseline_started = time.perf_counter()
    baseline_results = DecodingTask(model, direct_options()).run(
        survivor_mel.unsqueeze(0)
    )
    baseline_elapsed = time.perf_counter() - baseline_started
    if len(baseline_results) != 1:
        raise RuntimeError("the isolated baseline did not return one result")
    baseline_text = baseline_results[0].text

    per_transaction = ResourceVector(
        memory_bytes=1_000_000_000,
        compute_units=1,
        stream_slots=1,
    )
    profile = NativeExecutionProfile(
        "tiny.en/cpu/two-lane",
        per_transaction,
        max_concurrent_decodes=2,
    )
    budget = Budget(profile.worker_capacity)
    worker = Worker(
        "native-runtime-concurrency",
        snapshot,
        budget,
        queue_capacity=2,
        transaction_ttl_seconds=max(300.0, args.worker_timeout_seconds * 2),
    )
    adapter = NativeWhisperAdapter(worker, model, identity_probe, profile)
    sessions = {role: Session(f"runtime-{role}") for role in ROLES}
    requests = {
        role: RequestState(
            f"runtime-{role}-1",
            sessions[role].session_id,
            snapshot,
            rng_seed=7,
        )
        for role in ROLES
    }
    options = NativeDecodeOptions(
        language="en",
        temperature=0.0,
        without_timestamps=True,
    )
    mels = {"cancelled": cancelled_mel, "survivor": survivor_mel}
    durations = {
        "cancelled": min(round(split * 1_000 / SAMPLE_RATE), 30_000),
        "survivor": min(round(len(audio) * 1_000 / SAMPLE_RATE), 30_000),
    }

    role_context = threading.local()
    shared_lock = threading.Lock()
    worker_records: dict[str, dict[str, object]] = {}
    worker_states: dict[str, object] = {}
    worker_errors: dict[str, BaseException] = {}
    probe = RuntimeRunProbe(
        role_context,
        worker_records,
        shared_lock,
        args.worker_timeout_seconds,
    )
    snapshots = [_runtime_snapshot("initial", worker, budget, requests, sessions)]

    def caller(role: str) -> None:
        record = _thread_record(role)
        role_context.role = role
        with shared_lock:
            worker_records[role] = record
        try:
            _event(record, "adapter:enter")
            state = adapter.decode_window(
                session=sessions[role],
                request=requests[role],
                window_id=f"window-{role}",
                mel=mels[role],
                start_ms=0,
                end_ms=durations[role],
                options=options,
            )
            _event(record, "adapter:committed")
            with shared_lock:
                worker_states[role] = state
        except RequestCancelledError as error:
            _event(record, "adapter:cancelled")
            with shared_lock:
                worker_errors[role] = error
        except BaseException as error:
            with shared_lock:
                worker_errors[role] = error
            probe.abort()
        finally:
            record["finished_ns"] = time.perf_counter_ns()

    concurrent_started = time.perf_counter()
    with (
        controlled_decoder_forward(
            model.decoder,
            role_context,
            args.worker_timeout_seconds,
        ) as (forward_gate, installed_forwards),
        controlled_start_runs(DecodingTask, probe) as installed_start_run,
    ):
        threads = [
            threading.Thread(
                target=caller,
                args=(role,),
                name=f"runtime-{role}",
                daemon=True,
            )
            for role in ROLES
        ]
        for thread in threads:
            thread.start()
        try:
            for role in ROLES:
                _wait_event(
                    probe.first_steps[role],
                    args.worker_timeout_seconds,
                    f"the {role} first step",
                )
            _wait_until(
                lambda: worker.queue_depth == 2 and budget.lease_count == 2,
                args.worker_timeout_seconds,
                "two admitted transactions",
            )
            both_admitted = _runtime_snapshot(
                "both_admitted", worker, budget, requests, sessions
            )
            snapshots.append(both_admitted)

            with shared_lock:
                runs = dict(probe.runs)
            if set(runs) != set(ROLES):
                raise RuntimeError("the adapter did not expose both native runs")
            request_local, state_distinct, storage_disjoint, cache_entries = (
                _state_isolation(runs)
            )
            cancelled_inference = runs["cancelled"].inference
            survivor_inference = runs["survivor"].inference
            survivor_cache = _cache_snapshot(survivor_inference.kv_cache)

            cancel_started_ns = time.perf_counter_ns()
            cancel_accepted = requests["cancelled"].cancel()
            cancel_finished_ns = time.perf_counter_ns()
            if not cancel_accepted:
                raise RuntimeError("the running request did not accept cancellation")
            probe.release_cancelled.set()
            _wait_event(
                probe.cancelled_cleanup_complete,
                args.worker_timeout_seconds,
                "cancelled transaction cleanup",
            )
            _wait_until(
                lambda: worker.queue_depth == 1 and budget.lease_count == 1,
                args.worker_timeout_seconds,
                "cancelled lease release",
            )
            cancelled_released = _runtime_snapshot(
                "cancelled_released", worker, budget, requests, sessions
            )
            snapshots.append(cancelled_released)
            survivor_cache_unchanged = _cache_matches(
                survivor_inference.kv_cache, survivor_cache
            )
            cancelled_state_released = not cancelled_inference.kv_cache and all(
                getattr(runs["cancelled"], name, None) is None
                for name in (
                    "audio_features",
                    "tokens",
                    "sum_logprobs",
                    "inference",
                    "decoder",
                    "_pending_logits",
                )
            )
            probe.release_survivor.set()
        except BaseException:
            probe.abort()
            raise
        finally:
            deadline = time.monotonic() + args.worker_timeout_seconds
            for thread in threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            alive = [thread.name for thread in threads if thread.is_alive()]
            if alive:
                probe.abort()
                raise RuntimeError(
                    f"runtime caller thread did not finish: {', '.join(alive)}"
                )
        forward_records = sorted(forward_gate.records, key=lambda item: item["role"])
        maximum_forward_calls = forward_gate.max_active
    concurrent_elapsed = time.perf_counter() - concurrent_started

    start_run_after = callable_fingerprint(DecodingTask._start_run)
    decoder_forward_after = callable_fingerprint(model.decoder.forward)
    first_block_forward_after = callable_fingerprint(first_block.forward)
    instrumentation_restored = (
        start_run_before == installed_start_run == start_run_after
        and decoder_forward_before == installed_forwards["decoder"]
        and first_block_forward_before == installed_forwards["first_block"]
        and decoder_forward_after == decoder_forward_before
        and first_block_forward_after == first_block_forward_before
    )

    unexpected_errors = {
        role: error
        for role, error in worker_errors.items()
        if role != "cancelled" or type(error).__name__ != "RequestCancelledError"
    }
    if unexpected_errors:
        detail = "; ".join(
            f"{role}: {type(error).__name__}: {error}"
            for role, error in sorted(unexpected_errors.items())
        )
        raise RuntimeError(f"runtime adapter concurrency check failed: {detail}")
    cancelled_error = worker_errors.get("cancelled")
    survivor_state = worker_states.get("survivor")
    if type(cancelled_error).__name__ != "RequestCancelledError":
        raise RuntimeError("the cancelled adapter call had no cancellation outcome")
    if survivor_state is None:
        raise RuntimeError("the survivor adapter call did not commit")
    survivor_text = survivor_state.windows[-1].result.text
    if survivor_text != baseline_text:
        raise RuntimeError("the surviving adapter result differs from the baseline")
    if args.expected_text is not None and survivor_text != args.expected_text:
        raise RuntimeError(
            "the surviving transcript does not match --expected-text: "
            f"expected {args.expected_text!r}, observed {survivor_text!r}"
        )

    reuse_session = Session("runtime-reuse")
    reuse_request = RequestState(
        "runtime-reuse-1",
        reuse_session.session_id,
        snapshot,
        rng_seed=7,
    )
    reuse_started = time.perf_counter()
    reuse_state = adapter.decode_window(
        session=reuse_session,
        request=reuse_request,
        window_id="window-reuse",
        mel=survivor_mel,
        start_ms=0,
        end_ms=durations["survivor"],
        options=options,
    )
    reuse_elapsed = time.perf_counter() - reuse_started
    reuse_text = reuse_state.windows[-1].result.text
    snapshots.append(_runtime_snapshot("final", worker, budget, requests, sessions))

    model_fingerprint_after = fingerprint_loaded_model(model)
    execution_state_after = model_execution_state_fingerprint(model)
    hook_fingerprint_after = model_hook_fingerprint(model)
    thread_records = sorted(worker_records.values(), key=lambda item: item["role"])
    start_intervals = sorted(probe.start_intervals, key=lambda item: item["started_ns"])
    start_run_serialized = (
        probe.max_start_active == 1
        and len(start_intervals) == 2
        and start_intervals[0]["finished_ns"] <= start_intervals[1]["started_ns"]
    )
    thread_ids_distinct = (
        len(thread_records) == 2
        and len({item["python_thread_id"] for item in thread_records}) == 2
        and len({item["native_thread_id"] for item in thread_records}) == 2
    )
    overlap = forward_lifetimes_overlap(forward_records)
    owner_match = all(
        any(
            call["role"] == thread["role"]
            and call["python_thread_id"] == thread["python_thread_id"]
            and call["native_thread_id"] == thread["native_thread_id"]
            for call in forward_records
        )
        for thread in thread_records
    )
    both_capacity = snapshots[1]
    released_capacity = snapshots[2]
    final_capacity = snapshots[3]
    zero = _resource_record(ResourceVector())
    one_lane = _resource_record(per_transaction)
    full_capacity = _resource_record(profile.worker_capacity)
    survivor_session_state = sessions["survivor"].snapshot()
    cancelled_session_state = sessions["cancelled"].snapshot()
    assertions = {
        "two_distinct_caller_threads": thread_ids_distinct,
        "runtime_adapter_exercised": True,
        "two_transactions_admitted": (
            both_capacity["queue_depth"] == 2 and both_capacity["lease_count"] == 2
        ),
        "full_declared_capacity_reserved_at_overlap": (
            both_capacity["available"] == zero
            and both_capacity["in_use"] == full_capacity
        ),
        "start_run_serialized": start_run_serialized,
        "instrumented_forwards_ran_on_owner_threads": owner_match,
        "decoder_forward_lifetimes_overlap": (overlap and maximum_forward_calls == 2),
        "request_local_cache_path": request_local,
        "state_objects_distinct": state_distinct,
        "kv_cache_storage_disjoint": storage_disjoint,
        "both_runs_stepped_before_cancellation": all(
            event.is_set() for event in probe.first_steps.values()
        ),
        "cancel_request_accepted": cancel_accepted,
        "cancelled_request_did_not_commit": (
            requests["cancelled"].status.value == "cancelled"
            and cancelled_session_state.version == 0
            and not cancelled_session_state.windows
        ),
        "cancelled_state_released": cancelled_state_released,
        "cancelled_lease_released_after_cleanup": (
            released_capacity["queue_depth"] == 1
            and released_capacity["lease_count"] == 1
            and released_capacity["available"] == one_lane
            and released_capacity["in_use"] == one_lane
        ),
        "survivor_cache_unchanged_by_cancelled_cleanup": survivor_cache_unchanged,
        "survivor_committed_once": (
            requests["survivor"].status.value == "committed"
            and survivor_session_state.version == 1
            and len(survivor_session_state.windows) == 1
        ),
        "survivor_matches_isolated_baseline": survivor_text == baseline_text,
        "final_queue_empty": final_capacity["queue_depth"] == 0,
        "final_budget_restored": (
            final_capacity["lease_count"] == 0
            and final_capacity["available"] == full_capacity
            and final_capacity["in_use"] == zero
        ),
        "model_state_unchanged": (
            model_fingerprint_after == model_fingerprint_before
            and execution_state_after == execution_state_before
        ),
        "model_execution_hooks_unchanged": (
            hook_fingerprint_after == hook_fingerprint_before
        ),
        "instrumentation_restored": instrumentation_restored,
        "model_reusable_after_transactions": (
            reuse_request.status.value == "committed"
            and reuse_state.version == 1
            and reuse_text == baseline_text
        ),
        "input_unchanged": (
            sha256_file(audio_path) == audio_sha256
            and audio_path.stat().st_size == audio_size
        ),
        "mel_inputs_unchanged": (
            tensor_fingerprint(cancelled_mel) == cancelled_mel_sha256
            and tensor_fingerprint(survivor_mel) == survivor_mel_sha256
        ),
        "no_unexpected_worker_errors": not unexpected_errors,
    }
    verify_passed_runtime_assertions(assertions)

    checkpoint = _checkpoint_path(args.model, args.download_root)
    manifest = ROOT / "patches" / "openai-whisper" / "SHA256SUMS"
    record: dict[str, Any] = {
        "schema_version": "1",
        "recorded_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "passed",
        "scope": "native_runtime_adapter",
        "subject": {
            "layer": "whisper_runtime.adapters.NativeWhisperAdapter",
            "entrypoint": "NativeWhisperAdapter.decode_window",
            "runtime_adapter_exercised": True,
            "worker_admission_exercised": True,
            "transaction_lifecycle_exercised": True,
            "scheduler_exercised": False,
            "caller_threads": 2,
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
            "process_id": os.getpid(),
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
            "language": "en",
            "temperature": 0.0,
            "without_timestamps": True,
            "fp16": False,
        },
        "input": {
            "fixture_id": args.fixture_id,
            "path": input_label,
            "file_sha256": audio_sha256,
            "size_bytes": audio_size,
            "sample_rate_hz": SAMPLE_RATE,
            "source_sample_count": len(audio),
            "cancelled": {
                "sample_start": 0,
                "sample_end": split,
                "pcm_sha256": tensor_fingerprint(cancelled_audio),
                "mel_sha256": cancelled_mel_sha256,
                "observed_runtime_features_sha256": probe.feature_fingerprints[
                    "cancelled"
                ],
            },
            "survivor": {
                "sample_start": 0,
                "sample_end": len(audio),
                "pcm_sha256": tensor_fingerprint(survivor_audio),
                "mel_sha256": survivor_mel_sha256,
                "observed_runtime_features_sha256": probe.feature_fingerprints[
                    "survivor"
                ],
            },
        },
        "resources": {
            "accounting": "declared_in_process_admission_ledger",
            "per_transaction": one_lane,
            "capacity": full_capacity,
            "os_memory_enforced": False,
            "device_memory_measured": False,
            "snapshots": snapshots,
        },
        "execution": {
            "mode": "two_runtime_admitted_transactions",
            "thread_count": 2,
            "worker_queue_capacity": worker.queue_capacity,
            "max_concurrent_decodes": profile.max_concurrent_decodes,
            "encoder_policy": "serialized_start_run",
            "decoder_overlap_observation": "outer_call_lifetimes",
            "kernel_overlap_measured": False,
            "parallel_kernel_execution_claimed": False,
            "throughput_measured": False,
            "timing_is_benchmark": False,
            "cancellation": (
                "external_request_cancel_after_both_first_steps_before_"
                "cancelled_checkpoint"
            ),
            "cancelled_steps": 1,
            "survivor_steps": int(getattr(runs["survivor"], "step_index")),
            "survivor_cache_entries_at_cancellation": cache_entries,
            "worker_timeout_seconds": args.worker_timeout_seconds,
            "process_timeout_seconds": args.process_timeout_seconds,
            "elapsed_seconds": {
                "input_preparation": input_elapsed,
                "baseline": baseline_elapsed,
                "concurrent_adapter": concurrent_elapsed,
                "reuse_control": reuse_elapsed,
                "total": time.perf_counter() - total_started,
            },
            "controller": {
                "cancel_started_ns": cancel_started_ns,
                "cancel_finished_ns": cancel_finished_ns,
                "cancel_accepted": cancel_accepted,
            },
            "threads": thread_records,
            "start_run_intervals": start_intervals,
            "maximum_start_run_calls_live": probe.max_start_active,
            "controlled_forward_lifetimes": forward_records,
            "maximum_instrumented_forward_calls_live": maximum_forward_calls,
        },
        "assertions": assertions,
        "results": {
            "isolated_baseline": {"text": baseline_text},
            "cancelled": {
                "error_type": type(cancelled_error).__name__,
                "request_status": requests["cancelled"].status.value,
                "session_version": cancelled_session_state.version,
                "window_count": len(cancelled_session_state.windows),
            },
            "survivor": {
                "text": survivor_text,
                "request_status": requests["survivor"].status.value,
                "session_version": survivor_session_state.version,
                "window_count": len(survivor_session_state.windows),
                "window_id": survivor_session_state.windows[0].result.window_id,
            },
            "reuse_control": {
                "text": reuse_text,
                "request_status": reuse_request.status.value,
                "session_version": reuse_state.version,
                "window_count": len(reuse_state.windows),
                "window_id": reuse_state.windows[0].result.window_id,
            },
        },
    }
    from validate_runtime_concurrency_record import validate_runtime_concurrency_record

    failures = validate_runtime_concurrency_record(record, "generated record")
    if failures:
        raise RuntimeError("; ".join(failures))
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
    environment = os.environ.copy()
    environment[CHILD_TOKEN_ENV] = child_token
    environment["PYTHONIOENCODING"] = "utf-8:strict"
    command = [
        _child_python_executable(environment),
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
        tempfile.TemporaryFile(mode="w+b") as stdout_capture,
        tempfile.TemporaryFile(mode="w+b") as stderr_capture,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout_capture,
            stderr=stderr_capture,
            env=environment,
            **process_options,
        )
        try:
            process.wait(timeout=args.process_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            terminated = _terminate_process_tree(process)
            child_stderr = _read_capture(stderr_capture)
            _write_captured_stderr(child_stderr)
            detail = "" if terminated else "; process-tree termination is unconfirmed"
            raise RuntimeError(
                "the runtime concurrency verifier exceeded its process timeout" + detail
            ) from error
        child_stdout = _read_capture(stdout_capture)
        child_stderr = _read_capture(stderr_capture)
    _write_captured_stderr(child_stderr)
    if process.returncode != 0:
        detail = child_stderr.strip() or child_stdout.strip()
        raise RuntimeError(
            "the runtime concurrency verifier child failed with exit code "
            f"{process.returncode}: {detail}"
        )
    try:
        record = json.loads(child_stdout)
        serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "the runtime concurrency verifier child did not emit finite JSON"
        ) from error
    _write_utf8(sys.stdout, f"{serialized}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
