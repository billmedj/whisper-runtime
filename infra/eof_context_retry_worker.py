"""Real incremental controller execution for the fixed EOF-retry registration."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import time
from dataclasses import asdict

from infra import modal_eof_context_retry as p


class _MeasuredRun:
    def __init__(self, owner, raw, record):
        self.owner, self.raw, self.record = owner, raw, record

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _call(self, phase, method, *args, **kwargs):
        try:
            return self.owner.invoke(self.record, phase, method, *args, **kwargs)
        finally:
            self.record.update(
                closed=self.raw.closed, capacity_restored=self.raw.capacity_released
            )

    def step(self):
        self.record["driver_steps"] += 1
        return self._call("decode", self.raw.step)

    def prepare_result(self):
        return self._call("result", self.raw.prepare_result)

    def prepare_word_alignment(self):
        return self._call("alignment", self.raw.prepare_word_alignment)

    def finish(self, *args, **kwargs):
        return self._call("finish", self.raw.finish, *args, **kwargs)

    def close(self):
        return self._call("close", self.raw.close)


class _MeasuredAdapter:
    """Bound admissions and count every actual encode/decode model forward."""

    def __init__(self, native, started_ns, clock=time.perf_counter_ns, *, producer=p):
        self.native, self.started_ns, self.clock = native, started_ns, clock
        self.producer = producer
        self.model_identity = native.model_identity
        self.count, self.records, self.active = 0, [], None
        self.pcm, self.mel_wall_ns = b"", 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def invoke(self, record, phase, method, *args, **kwargs):
        previous, self.active = self.active, (record, phase)
        started = self.clock()
        try:
            return method(*args, **kwargs)
        finally:
            timings = record["operation_wall_ns"]
            timings[phase] = timings.get(phase, 0) + self.clock() - started
            self.active = previous

    def start_window(self, **kwargs):
        p = self.producer
        elapsed = self.clock() - self.started_ns
        if self.count >= p.MAX_NATIVE_WINDOWS or len(self.records) >= getattr(
            p, "MAX_NATIVE_WINDOWS_PER_CELL", 20
        ):
            raise RuntimeError("registered native-window bound reached")
        if elapsed >= (p.GPU_TIMEOUT_SECONDS - p.CLEANUP_RESERVE_SECONDS) * 1000000000:
            raise TimeoutError("cleanup reserve reached before native admission")
        start, end = kwargs["start_ms"], kwargs["end_ms"]
        if not 0 <= start < end or end - start > 30000 or end * 32 > len(self.pcm):
            raise ValueError("native window outside registered source")
        self.count += 1
        record = dict(
            call_index=self.count,
            window_id=kwargs["window_id"],
            start_ms=start,
            end_ms=end,
            pcm_sha256=p.shared._sha(self.pcm[start * 32 : end * 32]),
            admitted_elapsed_ns=elapsed,
            mel_shape=list(kwargs["mel"].shape),
            mel_wall_ns=self.mel_wall_ns,
            operation_wall_ns={},
            encoder_calls=[],
            decoder_forwards=[],
            driver_steps=0,
            closed=False,
            capacity_restored=False,
        )
        self.records.append(record)
        raw = self.invoke(record, "decode", self.native.start_window, **kwargs)
        return _MeasuredRun(self, raw, record)

    def observe_encoder(self, module, args, kwargs):
        if self.active is None:
            raise RuntimeError("unmeasured encoder forward")
        record, phase = self.active
        record["encoder_calls"].append(
            dict(
                phase=phase,
                input_frames=int(args[0].shape[-1]),
                use_sdpa=kwargs.get("use_sdpa"),
            )
        )

    def observe_decoder(self, module, args, kwargs):
        if self.active is None:
            raise RuntimeError("unmeasured decoder forward")
        record, phase = self.active
        record["decoder_forwards"].append(
            dict(phase=phase, token_count=int(args[0].shape[-1]))
        )


def _paced_budget_stop(pcm, *, producer, corpus, elapsed_ns, next_cell):
    """Reserve source time, bounded EOF drain, and cleanup before a paced cell."""
    remaining = producer.GPU_TIMEOUT_SECONDS * 1_000_000_000 - elapsed_ns
    required = (
        len(pcm) // 2 * 62500
        + corpus.PACED_REPLAY["config"]["max_drain_ms"] * 1_000_000
        + producer.CLEANUP_RESERVE_SECONDS * 1_000_000_000
    )
    if remaining < required:
        return dict(
            reason="budget_stop",
            next_cell=next_cell,
            remaining_ns=remaining,
            required_ns=required,
        )
    return None


def _paced_execution_failed(pacing, error):
    return not (
        (pacing["status"] == "completed" and error is None)
        or (
            pacing["status"] == "runtime_error"
            and error is not None
            and error.get("category") == "policy_resolution"
        )
    )


def run_worker(expected_snapshot, *, producer=p, paced=False):
    """Run the registered schedule through the unpaced or existing paced driver."""
    p = producer
    started_all = time.perf_counter_ns()
    shared, root = p.shared, p.REMOTE_ROOT
    shared.verify_snapshot(expected_snapshot, root)
    corpus = p._corpus()
    b, c = corpus.b, corpus.c
    inputs = p.cases(root)
    inventory = {case["id"]: case for case in inputs}
    registered = corpus.read_registration(root)
    backend = dict(
        commit=c._command_output(b.BACKEND_ROOT, "rev-parse", "HEAD"),
        tree=c._command_output(b.BACKEND_ROOT, "rev-parse", "HEAD^{tree}"),
    )
    p._require(
        backend == dict(commit=p.BACKEND_COMMIT, tree=p.BACKEND_TREE),
        "cached image backend identity differs",
    )
    p._require(
        not p._git(b.BACKEND_ROOT, "status", "--porcelain"),
        "backend checkout is dirty",
    )
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
    p._require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and torch.cuda.get_device_name(0) == "Tesla T4",
        "registered single T4 unavailable",
    )
    checkpoint = c._sha256_file(b.MODEL_CHECKPOINT_PATH)
    p._require(
        checkpoint == registered["model"]["checkpoint_sha256"],
        "cached checkpoint differs",
    )
    # Hash and existence are checked before load_model; the volume is read-only,
    # and the Modal function blocks network. There is no download fallback.
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=b.MODEL_CACHE_MOUNT
    ).eval()
    initial = q._model_fingerprint(model)
    p._require(
        initial == registered["model"]["model_state_sha256"]
        and not model.training
        and all(
            str(v.device) == "cuda:0"
            and (not v.is_floating_point() or v.dtype == torch.float32)
            for v in (*model.parameters(), *model.buffers())
        ),
        "registered FP32 model differs",
    )
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=p.BACKEND_COMMIT,
        backend="pytorch-cuda-eof-context-retry",
        fingerprint=initial,
    )
    worker = runtime.Worker(
        "eof-context-retry",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=p.GPU_TIMEOUT_SECONDS,
    )

    def probe(observed):
        if observed is not model:
            raise RuntimeError("model owner changed")
        return identity

    native = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/eof-context-retry-v1", capacity, device="cuda:0"
        ),
    )
    measured = _MeasuredAdapter(native, started_all, producer=p)
    options = adapters.NativeDecodeOptions(**registered["decode_options"])

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

    def mel(pcm):
        started = time.perf_counter_ns()
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        value = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
        ).contiguous()
        measured.mel_wall_ns = time.perf_counter_ns() - started
        return value

    effective = dict(
        decode_options=asdict(options),
        rng_seed=7,
        alignment_mode="legacy",
        precision="float32",
        model_sha256=initial,
        checkpoint_sha256=checkpoint,
        torch=torch.__version__,
        numpy=np.__version__,
        tiktoken=importlib.metadata.version("tiktoken"),
        attention=dict(decode="backend-default-sdpa", alignment="explicit-non-sdpa"),
        preprocessing="s16le-to-float32-div32768/pad-or-trim-480000/log-mel-80x3000/contiguous",
        backend_artifacts={
            name: c._sha256_file(b.BACKEND_ROOT / name)
            for name in (
                "whisper/model.py",
                "whisper/decoding.py",
                "whisper/timing.py",
                "whisper/audio.py",
                "whisper/tokenizer.py",
                "whisper/assets/gpt2.tiktoken",
                "whisper/assets/mel_filters.npz",
            )
        },
    )
    hooks = [
        model.encoder.register_forward_pre_hook(
            measured.observe_encoder, with_kwargs=True
        ),
        model.decoder.register_forward_pre_hook(
            measured.observe_decoder, with_kwargs=True
        ),
    ]
    cells, stop = [], None
    try:
        for case_id, arm in p.SCHEDULE:
            case, stream = inventory[case_id], None
            if paced:
                stop = _paced_budget_stop(
                    case["pcm"],
                    producer=p,
                    corpus=corpus,
                    elapsed_ns=time.perf_counter_ns() - started_all,
                    next_cell=len(cells),
                )
                if stop is not None:
                    break
            measured.records, measured.pcm = [], case["pcm"]
            config = p.stream_config(arm, case_id)
            cell = dict(
                case_id=case_id,
                arm=arm,
                config=config,
                windows=measured.records,
                events=[],
                decision_traces=[],
                stream_status="failed",
                done=False,
                error=None,
                capacity_restored=False,
                context_retry_observation=None,
            )
            cells.append(cell)
            cell_started = time.perf_counter_ns()
            try:
                p._require(available(), "previous cell retained capacity")
                cell["memory_before"] = memory()
                torch.cuda.reset_peak_memory_stats(0)
                endpoints = importlib.import_module(
                    "whisper_runtime.adapters.audio_endpoints"
                )
                configured = {
                    **config,
                    "endpointing": endpoints.QuietEndpointConfig(
                        **config["endpointing"]
                    ),
                }
                stream = adapters.ContinuousTranscriptStream(
                    measured,
                    stream_id=f"eof-context-retry:{case_id}:{arm}",
                    mel_builder=mel,
                    options=options,
                    rng_seed=7,
                    config=adapters.ContinuousStreamConfig(**configured),
                )
                if paced:
                    events, traces, pacing, error = corpus._drive_paced(
                        stream, case["pcm"]
                    )
                    cell["pacing"] = pacing
                    steps, chunks = pacing["driver_steps"], len(pacing["admissions"])
                else:
                    events, traces, steps, chunks, error = b._drive_stream(
                        stream, case["pcm"], chunk_bytes=32000
                    )
                cell.update(
                    events=events,
                    decision_traces=traces,
                    driver_steps=steps,
                    accepted_chunks=chunks,
                    error=error,
                    done=stream.done,
                    metrics=b._plain(stream.metrics),
                    state=b._plain(stream.state),
                    retained_from_sample=stream.retained_from_sample,
                    profile_id=stream.profile_id,
                    capacity_restored=available(),
                )
                cell["context_retry_observation"] = b._plain(
                    stream.context_retry_observation
                )
                cell["checks"] = p.cell_checks(cell, case["pcm"])
                text = c._normalized_committed_text(events)
                cell["recognition"] = dict(
                    text=text,
                    against_human_reference=b._word_difference(
                        text, case["reference_text"]
                    ),
                )
                cell["stream_status"] = (
                    "completed"
                    if (error is None and stream.done and all(cell["checks"].values()))
                    else "failed"
                )
                if error is not None and error.get("category") != "policy_resolution":
                    stop = dict(reason="non_policy_error", cell=len(cells) - 1)
                if paced and _paced_execution_failed(pacing, error):
                    stop = dict(
                        reason="paced_execution_failed",
                        cell=len(cells) - 1,
                        pacing_status=pacing["status"],
                    )
            except Exception as error:
                cell["error"] = b._safe_stream_error(error)
                stop = dict(reason="cell_error", cell=len(cells) - 1)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as error:
                        cell["cleanup_error"] = b._safe_stream_error(error)
                        stop = dict(reason="cleanup_error", cell=len(cells) - 1)
                    del stream
                cell["capacity_restored"] = available()
                if available():
                    cell["memory_after_close_and_handle_release"] = memory()
                    cell["memory_peak"] = dict(
                        allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                        reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
                    )
                else:
                    stop = dict(reason="capacity_retained", cell=len(cells) - 1)
                cell["wall_ns"] = time.perf_counter_ns() - cell_started
            if stop is not None:
                break
    finally:
        for hook in hooks:
            hook.remove()
    final = q._model_fingerprint(model) if available() else None
    summary = p.summarize(
        cells,
        model_unchanged=initial == final,
        capacity_restored=available(),
        root=root,
    )
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    return dict(
        schema_version="1-diagnostic",
        experiment_id=p.EXPERIMENT_ID,
        status="completed"
        if summary["development_recovery_demonstrated"]
        else "failed",
        qualified=False,
        claim_boundary=p.CLAIMS,
        scope=p.scope(),
        backend=backend,
        effective_identity=effective,
        inputs=p.input_records(root),
        schedule=[dict(case_id=c, arm=a) for c, a in p.SCHEDULE],
        cells=cells,
        native_window_count=measured.count,
        capacity_restored=available(),
        summary=summary,
        stop=stop,
        elapsed_ns=time.perf_counter_ns() - started_all,
        model=dict(
            initial_sha256=initial,
            final_sha256=final,
            unchanged=initial == final,
            checkpoint_sha256=checkpoint,
        ),
        source=dict(
            snapshot=expected_snapshot,
            registration_sha256=c._sha256_file(root / p.PRODUCER),
        ),
        worker=dict(
            function_call_id=call_id,
            function_call_id_sha256=shared._sha(call_id.encode()),
            modal=str(modal.__version__),
        ),
    )
