"""One cached-model, seven-window diagnostic. No stream publication authority."""

from __future__ import annotations

import importlib
import importlib.metadata
import time
from dataclasses import asdict


def run_worker(expected_snapshot):
    p = importlib.import_module("infra.modal_resolution_handoff")
    shared, features, paced = p.shared, p.features, p.paced
    corpus = p._corpus()
    b, c = corpus.b, corpus.c
    root = shared.REMOTE_ROOT
    shared.verify_snapshot(expected_snapshot, root)
    cases = p.cases(root)
    base = corpus.read_registration(root)
    backend = features.apply_backend_patch(root, b.BACKEND_ROOT)
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
    q = corpus._helper("infra.modal_native_cuda_qualification")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "Tesla T4"
    ):
        raise RuntimeError("registered single T4 unavailable")
    checkpoint = c._sha256_file(b.MODEL_CHECKPOINT_PATH)
    if checkpoint != base["model"]["checkpoint_sha256"]:
        raise RuntimeError("cached checkpoint differs")
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=b.MODEL_CACHE_MOUNT
    ).eval()
    initial = q._model_fingerprint(model)
    if (
        initial != base["model"]["model_state_sha256"]
        or model.training
        or any(
            str(value.device) != "cuda:0"
            or (value.is_floating_point() and value.dtype != torch.float32)
            for value in (*model.parameters(), *model.buffers())
        )
    ):
        raise RuntimeError("registered FP32 model differs")
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=features.PATCHED_TREE,
        backend="pytorch-cuda-resolution-handoff",
        fingerprint=initial,
    )
    worker = runtime.Worker(
        "resolution-handoff",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=180,
    )

    def probe(observed):
        if observed is not model:
            raise RuntimeError("model instance differs")
        return identity

    native = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/resolution-handoff-v1", capacity, device="cuda:0"
        ),
    )
    adapter = paced._MeasuredAdapter(native)
    adapter.reuse = False
    options = adapters.NativeDecodeOptions(**base["decode_options"])
    effective = dict(
        backend_artifacts={
            name: c._sha256_file(b.BACKEND_ROOT / name) for name in p.BACKEND_ARTIFACTS
        },
        decode_options=asdict(options),
        rng_seed=7,
        alignment_mode="legacy",
        preprocessing="s16le-to-float32-div32768/pad-or-trim-480000/log-mel-80x3000/contiguous",
        model_sha256=initial,
        checkpoint_sha256=checkpoint,
        precision="float32",
        torch=torch.__version__,
        numpy=np.__version__,
        tiktoken=importlib.metadata.version("tiktoken"),
        attention=dict(decode="backend-default-sdpa", alignment="explicit-non-sdpa"),
    )

    def available():
        return (
            budget.available == capacity
            and budget.lease_count == 0
            and worker.queue_depth == 0
        )

    hook = model.encoder.register_forward_pre_hook(
        adapter.observe_encoder, with_kwargs=True
    )
    cells, stop = [], None
    started_all = time.perf_counter_ns()
    try:
        for case in cases:
            cell = dict(
                id=case["id"],
                status="pending",
                archive=case["archive"],
                original_source_trace=case["source_trace"],
                original_anchor_diagnostic=case["source_diagnostic"],
                fresh_windows=[],
                raw_alignments={},
                summary=None,
                comparability=p.provenance_checks(case, effective, backend, root),
                publication_authorized=False,
                error=None,
            )
            cells.append(cell)
            if not cell["comparability"]["comparable"]:
                cell["status"] = "blocked"
                continue
            if case["archived_current"] is not None:
                cell["raw_alignments"].update(
                    current=case["archived_current"],
                    candidate=case["archived_candidate"],
                )
            try:
                for window in case["windows"]:
                    if adapter.count >= 7 or not available():
                        raise RuntimeError("seven-window bound or capacity unavailable")
                    if (time.perf_counter_ns() - started_all) / 1e9 >= 150:
                        raise TimeoutError(
                            "diagnostic reserve reached before native admission"
                        )
                    pcm = p._slice(case, window["start_ms"], window["end_ms"])
                    observation = dict(
                        **window,
                        pcm_sha256=shared._sha(pcm),
                        sample_count=len(pcm) // 2,
                        first_call_no_warmup=adapter.count == 0,
                        measurement=None,
                        completed=False,
                        error=None,
                    )
                    cell["fresh_windows"].append(observation)
                    run = None
                    torch.cuda.synchronize(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    before = time.perf_counter_ns()
                    observation["memory_before_bytes"] = int(
                        torch.cuda.memory_allocated(0)
                    )
                    try:
                        audio = (
                            np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                        )
                        mel = whisper.log_mel_spectrogram(
                            whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
                        ).contiguous()
                        observation.update(
                            mel_shape=list(mel.shape),
                            mel_wall_ns=time.perf_counter_ns() - before,
                        )
                        session = runtime.Session(
                            f"handoff:{case['id']}:{window['arm']}"
                        )
                        previous_count = adapter.count
                        try:
                            run = adapter.start_window(
                                session=session,
                                request=runtime.RequestState(
                                    session.session_id + ":request",
                                    session.session_id,
                                    identity,
                                    rng_seed=7,
                                ),
                                window_id=session.session_id,
                                mel=mel,
                                start_ms=window["start_ms"],
                                end_ms=window["end_ms"],
                                options=options,
                            )
                        finally:
                            if adapter.count > previous_count:
                                observation["measurement"] = adapter.records[-1]
                        steps = 0
                        while not run.complete:
                            if steps >= b.MAX_DRIVER_STEPS:
                                raise RuntimeError("native decoder step bound exceeded")
                            run.step()
                            steps += 1
                        run.prepare_result()
                        alignment = run.prepare_word_alignment()
                        cell["raw_alignments"][window["arm"]] = b._plain(alignment)
                        observation.update(completed=True, decoder_steps=steps)
                    except Exception as error:
                        observation["error"] = b._safe_stream_error(error)
                        raise
                    finally:
                        try:
                            if run is not None:
                                run.close()
                        finally:
                            torch.cuda.synchronize(0)
                            observation.update(
                                wall_ns=time.perf_counter_ns() - before,
                                peak_allocated_bytes=int(
                                    torch.cuda.max_memory_allocated(0)
                                ),
                                peak_reserved_bytes=int(
                                    torch.cuda.max_memory_reserved(0)
                                ),
                                memory_after_close_bytes=int(
                                    torch.cuda.memory_allocated(0)
                                ),
                                capacity_restored=available(),
                            )
                    if not run.closed or not run.capacity_released or not available():
                        raise RuntimeError("native close did not restore capacity")
                    run = None
                    observation["memory_after_handle_release_bytes"] = int(
                        torch.cuda.memory_allocated(0)
                    )
                try:
                    cell["summary"] = p.summarize_cell(case, cell["raw_alignments"])
                    cell["status"] = (
                        "evaluated"
                        if cell["summary"]["control_native_reproduced"]
                        else "blocked"
                    )
                except Exception as error:
                    cell.update(
                        status="assessment_failed", error=b._safe_stream_error(error)
                    )
            except Exception as error:
                cell.update(
                    status="lifecycle_failure", error=b._safe_stream_error(error)
                )
                stop = dict(reason="native_lifecycle_failure", case_id=case["id"])
                break
    finally:
        hook.remove()
    final = q._model_fingerprint(model) if available() else None
    complete = (
        len(cells) == 5
        and adapter.count == 7
        and available()
        and final == initial
        and all(cell["status"] == "evaluated" for cell in cells)
    )
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    if not isinstance(call_id, str) or not call_id:
        raise RuntimeError("worker call identity unavailable")
    return dict(
        schema_version="1-diagnostic",
        experiment_id="modal-resolution-handoff-v1",
        status="completed" if complete else "failed",
        qualified=False,
        claim_boundary=p.CLAIMS,
        scope=p.scope(),
        source=dict(
            snapshot=expected_snapshot,
            registration_sha256=shared._sha((root / p.PRODUCER).read_bytes()),
        ),
        inputs=p.input_records(root),
        backend=backend,
        effective_identity=effective,
        model=dict(
            initial_sha256=initial,
            final_sha256=final,
            unchanged=final == initial,
            backend_revision=features.PATCHED_TREE,
            checkpoint_sha256=checkpoint,
        ),
        cells=cells,
        native_window_count=adapter.count,
        capacity_restored=available(),
        elapsed_ns=time.perf_counter_ns() - started_all,
        stop=stop,
        worker=dict(
            function_call_id=call_id,
            function_call_id_sha256=shared._sha(call_id.encode()),
            modal=str(modal.__version__),
            torch=str(torch.__version__),
        ),
    )
