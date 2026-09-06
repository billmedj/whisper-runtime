"""Compare fresh and reused CUDA lanes on one fixed input. No import-time work."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import time
from dataclasses import asdict

from infra import modal_acoustic_diagnostic as shared

ROOT = shared.ROOT
REMOTE_ROOT = shared.REMOTE_ROOT
PRODUCER = "infra/modal_cuda_lane.py"
ROUNDS = 6
ORDERS = {
    "alternating-v1": ["reused", "fresh"] * ROUNDS,
    "blocked-v2": ["reused"] * 4 + ["fresh"] * 4 + ["reused"] * 4,
}
ACTIVE_ORDER = "blocked-v2"
CLAIMS = dict.fromkeys(
    ("full_stream_recovery", "general_speedup", "production_readiness"), False
)


def _corpus():
    return shared._corpus()


def snapshot(root=ROOT):
    base = shared.snapshot(root)
    names = {item["path"] for item in base["files"]}
    names.update((PRODUCER, "infra/modal_native_cuda_qualification.py"))
    files = [
        dict(path=name, size_bytes=len(data), sha256=shared._sha(data))
        for name in sorted(names)
        for data in [(root / name).read_bytes()]
    ]
    return dict(commit=base["commit"], digest=shared._hash(files), files=files)


def summarize(cells, *, order="alternating-v1"):
    """Evaluate matched output and post-close allocation, not allocator reserve."""
    if len(cells) != ROUNDS * 2:
        raise ValueError("twelve completed observations are required")
    if order not in ORDERS or [cell["arm"] for cell in cells] != ORDERS[order]:
        raise ValueError("observations must preserve registered order")
    if any(
        cell["call_index"] != index
        or cell["closed"] is not True
        or cell["capacity_restored"] is not True
        or cell["stale_cancel_changed_state"] is not False
        or len(cell["streams"]) != 1
        for index, cell in enumerate(cells, 1)
    ):
        raise ValueError("incomplete lifecycle or stream observation")
    # Transaction identity must differ; every decode value must remain equal.
    hashes = {
        shared._hash(
            {
                **cell["alignment"],
                "native": {**cell["alignment"]["native"], "window_id": "same-input"},
            }
        )
        for cell in cells
    }
    reuse = [cell for cell in cells if cell["arm"] == "reused"]
    fresh = [cell for cell in cells if cell["arm"] == "fresh"]

    def delta(cell):
        return (
            cell["after_close"]["allocated_bytes"] - cell["before"]["allocated_bytes"]
        )

    result = {
        "exact_alignment_equal": len(hashes) == 1,
        "reused_stream_count": len({cell["streams"][0] for cell in reuse}),
        "fresh_stream_count": len({cell["streams"][0] for cell in fresh}),
        "post_close_allocated_delta_bytes": {
            "reused": [delta(cell) for cell in reuse],
            "fresh": [delta(cell) for cell in fresh],
        },
        "warm_reused_allocation_flat": all(delta(cell) == 0 for cell in reuse[1:]),
        "encoder_calls": [len(cell["encoder_frames"]) for cell in cells],
    }
    if order == "blocked-v2":
        consecutive = [
            right["after_handle_release"]["allocated_bytes"]
            - left["after_handle_release"]["allocated_bytes"]
            for left, right in zip(cells, cells[1:])
            if left["arm"] == right["arm"] == "reused"
        ]
        result["post_handle_release_allocated_bytes"] = [
            cell["after_handle_release"]["allocated_bytes"] for cell in cells
        ]
        result["consecutive_reused_allocated_deltas_bytes"] = consecutive
        result["consecutive_reused_allocation_flat"] = len(consecutive) == 6 and all(
            value == 0 for value in consecutive
        )
    return result


def run_worker(expected_snapshot):
    shared.verify_snapshot(expected_snapshot)
    corpus = _corpus()
    b, c = corpus.b, corpus.c
    base = corpus.read_registration(REMOTE_ROOT)
    q = corpus._helper("infra.modal_native_cuda_qualification")
    torch, np, whisper, runtime, adapters = (
        importlib.import_module(name)
        for name in (
            "torch",
            "numpy",
            "whisper",
            "whisper_runtime",
            "whisper_runtime.adapters",
        )
    )
    if (
        str(torch.__version__).split("+")[0] != "2.6.0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "Tesla T4"
    ):
        raise RuntimeError("registered PyTorch 2.6 / single T4 unavailable")
    if c._sha256_file(b.MODEL_CHECKPOINT_PATH) != base["model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint mismatch")
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=b.MODEL_CACHE_MOUNT
    ).eval()
    initial = q._model_fingerprint(model)
    if initial != base["model"]["model_state_sha256"]:
        raise RuntimeError("model fingerprint mismatch")
    fixture = base["fixtures"][0]
    pcm = (corpus.REMOTE_ASSETS / fixture["filename"]).read_bytes()
    if (
        len(pcm) != fixture["sample_count"] * 2
        or shared._sha(pcm) != fixture["pcm_sha256"]
    ):
        raise RuntimeError("registered PCM differs")
    # Pad only to the runtime's 20 ms source clock; record the padded identity.
    pcm += b"\x00\x00" * (-fixture["sample_count"] % 320)
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
    ).contiguous()
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=c._command_output(b.BACKEND_ROOT, "rev-parse", "HEAD"),
        backend="pytorch-cuda-lane-diagnostic",
        fingerprint=initial,
    )
    worker = runtime.Worker(
        "cuda-lane", identity, budget, queue_capacity=1, transaction_ttl_seconds=180
    )

    def probe(observed):
        if observed is not model:
            raise RuntimeError("model identity changed")
        return identity

    adapter = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/cuda-lane-v1", capacity, device="cuda:0"
        ),
    )
    binding = adapter._model_binding
    options = adapters.NativeDecodeOptions(**base["decode_options"])

    def available():
        return (
            budget.available == capacity
            and budget.lease_count == 0
            and worker.queue_depth == 0
        )

    def memory():
        gc.collect()
        torch.cuda.synchronize(0)
        return dict(
            allocated_bytes=int(torch.cuda.memory_allocated(0)),
            reserved_bytes=int(torch.cuda.memory_reserved(0)),
        )

    def measure(cell, previous):
        index = cell["call_index"]
        torch.cuda.reset_peak_memory_stats(0)

        def observe(module, args):
            cell["encoder_frames"].append(int(args[0].shape[-1]))
            stream = int(torch.cuda.current_stream(0).cuda_stream)
            if stream not in cell["streams"]:
                cell["streams"].append(stream)

        hook = model.encoder.register_forward_pre_hook(observe)
        started = time.perf_counter_ns()
        try:
            session = runtime.Session(f"lane:{index}")
            with adapter.start_window(
                session=session,
                request=runtime.RequestState(
                    f"lane:{index}:request", session.session_id, identity, rng_seed=7
                ),
                window_id=session.session_id,
                mel=mel,
                start_ms=0,
                end_ms=len(pcm) // 32,
                options=options,
            ) as run:
                cell["stale_cancel_changed_state"] = (
                    previous.cancel() if previous is not None else False
                )
                steps = 0
                while not run.complete:
                    if steps >= b.MAX_DRIVER_STEPS:
                        raise RuntimeError("native step bound exceeded")
                    run.step()
                    steps += 1
                cell["alignment"] = asdict(run.prepare_word_alignment())
            cell.update(
                closed=run.closed,
                capacity_restored=run.capacity_released and available(),
            )
        finally:
            hook.remove()
        cell.update(
            wall_ns=time.perf_counter_ns() - started,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
        )
        return run

    cells, persistent, previous, failure = [], None, None, None
    for index, arm in enumerate(ORDERS[ACTIVE_ORDER], 1):
        cell = dict(call_index=index, arm=arm, encoder_frames=[], streams=[])
        cells.append(cell)
        try:
            if not available():
                raise RuntimeError("previous window did not release capacity")
            # Diagnostic-only lifetime ablation. Never replace an active lane.
            # The fresh arm recreates the previous stream lifetime.
            with binding.lock:
                if binding.retained_error is not None:
                    raise RuntimeError("cannot replace a quarantined lane")
                if (
                    binding._cuda_lane is not None
                    and binding._cuda_lane.owner is not None
                ):
                    raise RuntimeError("cannot replace a borrowed lane")
                binding._cuda_lane = persistent if arm == "reused" else None
            cell["before"] = memory()
            previous = measure(cell, previous)
            if not cell["closed"] or not cell["capacity_restored"]:
                raise RuntimeError("window retained capacity")
            if arm == "reused":
                persistent = binding._cuda_lane
            with binding.lock:
                binding._cuda_lane = persistent
            cell["after_close"] = memory()
            # V1 tested stale cancellation and retained a closed handle.
            # V2 isolates physical residency by dropping it before each sample.
            previous = None
            cell["after_handle_release"] = memory()
        except Exception as error:
            failure = b._safe_stream_error(error)
            cell["error"] = failure
            break
    summary = summarize(cells, order=ACTIVE_ORDER) if failure is None else None
    # The cached backend retains its finalized feature tensor in a closed handle.
    # Per-call samples contain one such handle; measure its release separately.
    previous = None
    after_handle_release = memory() if failure is None else None
    final = q._model_fingerprint(model) if failure is None else None
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    return {
        "schema_version": "1-diagnostic",
        "experiment_id": "modal-cuda-lane-v2",
        "status": "completed" if failure is None else "failed",
        "error": failure,
        "qualified": False,
        "claim_boundary": CLAIMS,
        "cells": cells,
        "summary": summary,
        "after_handle_release": after_handle_release,
        "model": dict(
            initial_sha256=initial,
            final_sha256=final,
            unchanged=initial == final,
            backend_revision=identity.revision,
        ),
        "input": dict(
            fixture_id=fixture["id"],
            pcm_sha256=shared._sha(pcm),
            sample_count=len(pcm) // 2,
        ),
        "scope": dict(
            single_host_thread=True,
            order=ACTIVE_ORDER,
            alternating_order=False,
            first_pair_cold=True,
            same_pcm_mel_options=True,
            full_stream=False,
            cache_clearing=False,
            reused_lane_held_between_arms=True,
            maximum_native_windows=ROUNDS * 2,
            post_close_retains_one_handle=True,
            drops_handle_before_next_window=True,
            stale_cancellation_tested=False,
            timing_excludes_preprocessing_and_collection=True,
        ),
        "source": dict(
            snapshot=expected_snapshot,
            registration_sha256=shared._sha((REMOTE_ROOT / PRODUCER).read_bytes()),
        ),
        "worker": dict(
            function_call_id=call_id,
            function_call_id_sha256=shared._sha(call_id.encode()),
            modal=str(modal.__version__),
        ),
    }


def validate_record(record, expected):
    """Recompute the lane summary and bind it to the dispatched source."""
    if record["scope"]["order"] != ACTIVE_ORDER:
        raise ValueError("worker schedule differs")
    summary = (
        summarize(record["cells"], order=ACTIVE_ORDER)
        if record["status"] == "completed"
        else None
    )
    if record["source"]["snapshot"] != expected or record["summary"] != summary:
        raise ValueError("worker source or recomputed summary differs")


def run(*, replay_id, confirm_paid_gpu=False, root=ROOT, producer=None):
    """Launch one producer with CPU preflight and exclusive evidence files.

    A producer module supplies ``snapshot``, ``PRODUCER``, ``CLAIMS``,
    ``_corpus``, and ``validate_record(record, expected)``. Its remote entrypoint
    remains ``run_worker(expected_snapshot)`` through the shared resource helper.
    """
    if confirm_paid_gpu is not True:
        raise ValueError("GPU work requires --confirm-paid-gpu")
    if producer is None:
        producer = importlib.import_module("infra.modal_cuda_lane")
    output, receipt, raw = shared._paths(root, replay_id, False)
    if any(path.exists() for path in (output, receipt, raw)):
        raise FileExistsError("attempt exists; no overwrite or retry")
    expected = producer.snapshot(root)
    corpus = producer._corpus()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(event="attempt-started", source_digest=expected["digest"]))
            + "\n"
        )
    try:
        app, echo, execute = shared.resources(
            expected, root, worker_module=producer.__name__
        )
        with app.run(detach=False):
            probe = corpus.b._transport_probe_payload()
            if echo.remote(probe) != probe:
                raise RuntimeError("CPU transport preflight failed")
            payload = execute.remote()
            corpus.b._write_bytes_exclusive(raw, payload)
            record = corpus.b._decode_worker_record(
                payload,
                expected_snapshot=expected,
                registration_sha256=shared._sha(
                    (root / producer.PRODUCER).read_bytes()
                ),
                manifest=dict(claim_boundary=producer.CLAIMS),
            )
            producer.validate_record(record, expected)
            record["app_id"] = app.app_id
        corpus.c._write_json_exclusive(output, record)
        event = dict(event="record-written", sha256=shared._sha(output.read_bytes()))
    except BaseException as error:
        with receipt.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    dict(event="attempt-failed", error_type=type(error).__name__)
                )
                + "\n"
            )
        raise
    with receipt.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(run(**vars(parser.parse_args())))
