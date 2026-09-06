"""Bounded source-paced comparison of same-window alignment feature reuse."""

from __future__ import annotations

import argparse
import gc
import importlib
import time
from types import SimpleNamespace

from infra import modal_acoustic_diagnostic as shared
from infra import modal_alignment_features as features

ROOT = shared.ROOT
REMOTE_ROOT = shared.REMOTE_ROOT
PRODUCER = "infra/modal_paced_features.py"
PREREGISTRATION = "docs/research/2026-09-06-paced-feature-reuse.md"
MAX_NATIVE_WINDOWS = 128
WORKER_SECONDS = 180
RESERVE_SECONDS = 15
SHORT_SAMPLES = 174240
SCHEDULE = [
    ("mixed-control", "baseline"),
    ("mixed-control", "reuse"),
    ("attenuated-prefix", "reuse"),
    ("attenuated-prefix", "baseline"),
    ("noisy-prefix", "baseline"),
    ("noisy-prefix", "reuse"),
]
CLAIMS = dict.fromkeys(
    (
        "general_speedup",
        "production_readiness",
        "cross_window_reuse",
        "gpu_device_timing",
    ),
    False,
)


def _corpus():
    return shared._corpus()


def stream_config():
    return {
        **_corpus().AUTOMATIC_ENDPOINTS_PROFILE["stream_config"],
        "word_boundary_fallback": True,
        "left_context_ms": 2000,
    }


def cases(root=ROOT):
    corpus = _corpus()
    assets = corpus.REMOTE_ASSETS if root == REMOTE_ROOT else root / corpus.ASSET_PATH
    metadata, pcms = shared._cases(root, assets)
    base = corpus.read_registration(root)
    short_reference = " ".join(item["reference_text"] for item in base["fixtures"][:2])
    source_metadata = {item["id"]: item for item in metadata["cases"]}
    result = []
    for case_id, source_id, count in (
        ("mixed-control", "mixed-control", 698560),
        ("attenuated-prefix", "mixed-attenuated32", SHORT_SAMPLES),
        ("noisy-prefix", "mixed-continuous-noise64", SHORT_SAMPLES),
    ):
        pcm = pcms[source_id][: count * 2]
        if len(pcm) != count * 2:
            raise ValueError("registered PCM prefix is incomplete")
        result.append(
            dict(
                id=case_id,
                pcm=pcm,
                sample_count=count,
                duration_seconds=count / 16000,
                pcm_sha256=shared._sha(pcm),
                source_case_id=source_id,
                source_pcm_sha256=source_metadata[source_id]["pcm_sha256"],
                recipe=source_metadata[source_id]["recipe"],
                prefix_samples=count,
                reference_text=(
                    source_metadata[source_id]["reference_text"]
                    if case_id == "mixed-control"
                    else short_reference
                ),
            )
        )
    return result


def input_records(root=ROOT):
    return [
        {key: value for key, value in item.items() if key != "pcm"}
        for item in cases(root)
    ]


def snapshot(root=ROOT):
    base = features.snapshot(root)
    names = {item["path"] for item in base["files"]}
    names.update((PRODUCER, PREREGISTRATION, "infra/modal_cuda_lane.py"))
    files = [
        dict(path=name, size_bytes=len(data), sha256=shared._sha(data))
        for name in sorted(names)
        for data in [(root / name).read_bytes()]
    ]
    return dict(commit=base["commit"], digest=shared._hash(files), files=files)


class _MeasuredRun:
    """Diagnostic delegation only; native lifecycle remains authoritative."""

    def __init__(self, owner, raw, record, reuse):
        self.owner, self.raw, self.record, self.reuse = owner, raw, record, reuse

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
        return self._call("decode", self.raw.step)

    def prepare_result(self):
        return self._call("result", self.raw.prepare_result)

    def prepare_word_alignment(self):
        return self._call(
            "alignment",
            self.raw.prepare_word_alignment,
            reuse_alignment_features=self.reuse,
        )

    def finish(self, *args, **kwargs):
        return self._call("finish", self.raw.finish, *args, **kwargs)

    def close(self):
        return self._call("close", self.raw.close)


class _MeasuredAdapter:
    """One native adapter, with per-run control and symmetric host measurements."""

    def __init__(self, native):
        self.native = native
        self.model_identity = native.model_identity
        self.reuse = False
        self.records = []
        self.count = 0
        self.active = None

    def invoke(self, record, phase, method, *args, **kwargs):
        previous, self.active = self.active, (record, phase)
        started = time.perf_counter_ns()
        try:
            return method(*args, **kwargs)
        finally:
            timings = record["operation_wall_ns"]
            timings[phase] = timings.get(phase, 0) + time.perf_counter_ns() - started
            self.active = previous

    def start_window(self, **kwargs):
        if self.count >= MAX_NATIVE_WINDOWS:
            raise RuntimeError("aggregate native window limit reached")
        self.count += 1
        record = dict(
            call_index=self.count,
            window_id=kwargs["window_id"],
            start_ms=kwargs["start_ms"],
            end_ms=kwargs["end_ms"],
            reuse_alignment_features=self.reuse,
            encoder_calls=[],
            operation_wall_ns={},
            gpu_device_time_ms=None,
            closed=False,
            capacity_restored=False,
        )
        self.records.append(record)
        raw = self.invoke(record, "decode", self.native.start_window, **kwargs)
        return _MeasuredRun(self, raw, record, self.reuse)

    def observe_encoder(self, module, args, kwargs):
        if self.active is None:
            raise RuntimeError("encoder call escaped measured native execution")
        record, phase = self.active
        record["encoder_calls"].append(
            dict(
                phase=phase,
                input_frames=int(args[0].shape[-1]),
                use_sdpa=kwargs.get("use_sdpa"),
            )
        )


def validate_record(record, expected):
    """Recompute clocks, publication accounting and scores before accepting evidence."""
    corpus = _corpus()
    b, c = corpus.b, corpus.c
    inputs = cases()
    inventory = {item["id"]: item for item in inputs}

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(record["source"]["snapshot"] == expected, "worker source differs")
    require(record["experiment_id"] == "modal-paced-features-v1", "experiment differs")
    require(record["schema_version"] == "1-diagnostic", "schema differs")
    require(record["qualified"] is False, "diagnostic cannot qualify a release")
    require(record["claim_boundary"] == CLAIMS, "claim boundary differs")
    require(
        record["inputs"]
        == [{k: v for k, v in item.items() if k != "pcm"} for item in inputs],
        "worker inputs differ",
    )
    schedule = [dict(case_id=case_id, arm=arm) for case_id, arm in SCHEDULE]
    require(record["schedule"] == schedule, "schedule differs")
    for key, value in (
        ("base_commit", features.BASE_COMMIT),
        ("base_tree", features.BASE_TREE),
        ("patched_tree", features.PATCHED_TREE),
        ("patch_sha256", features.PATCH_SHA),
    ):
        require(record["backend"][key] == value, "backend identity differs")
    registered = corpus.read_registration()["model"]
    model = record["model"]
    require(
        model["initial_sha256"] == registered["model_state_sha256"], "model differs"
    )
    require(
        model["checkpoint_sha256"] == registered["checkpoint_sha256"],
        "checkpoint differs",
    )
    require(model["backend_revision"] == features.PATCHED_TREE, "model backend differs")
    require(
        model["unchanged"] is (model["final_sha256"] == model["initial_sha256"]),
        "model verdict differs",
    )
    require(record["scope"] == scope(), "execution scope differs")
    cells, warmup = record["cells"], record["warmup"]
    require(len(cells) <= len(schedule) and len(warmup) <= 2, "too many cells")
    require(
        [dict(case_id=x["case_id"], arm=x["arm"]) for x in cells]
        == schedule[: len(cells)],
        "cell order differs",
    )
    require(
        [x["arm"] for x in warmup] == ["baseline", "reuse"][: len(warmup)],
        "warmup order differs",
    )
    if cells:
        require(
            len(warmup) == 2 and all(x["capacity_restored"] is True for x in warmup),
            "warmup did not finish",
        )
    windows = []
    for owner in [*warmup, *cells]:
        for window in owner["windows"]:
            windows.append(window)
            require(
                window["reuse_alignment_features"] is (owner["arm"] == "reuse"),
                "alignment mode differs",
            )
            require(window["gpu_device_time_ms"] is None, "unmeasured device time")
            require(
                0 <= window["start_ms"] < window["end_ms"]
                and window["end_ms"] - window["start_ms"] <= 30000,
                "invalid native span",
            )
            require(
                all(
                    type(t) is int and t >= 0
                    for t in window["operation_wall_ns"].values()
                ),
                "invalid host time",
            )
            for call in window["encoder_calls"]:
                require(call["input_frames"] == 3000, "encoder input differs")
                require(
                    (call["phase"], call["use_sdpa"])
                    in (("decode", None), ("alignment", False)),
                    "encoder path differs",
                )
                require(
                    not window["reuse_alignment_features"] or call["phase"] == "decode",
                    "reuse repeated encoder",
                )
    require(
        record["native_window_count"] == len(windows) <= MAX_NATIVE_WINDOWS,
        "native window count differs",
    )
    require(
        [x["call_index"] for x in windows] == list(range(1, len(windows) + 1)),
        "native call order differs",
    )
    for warm in warmup:
        require(
            warm["sample_count"] == 32000
            and warm["pcm_sha256"] == shared._sha(inputs[0]["pcm"][:64000]),
            "warmup input differs",
        )
    for cell in cells:
        if cell["pacing"] is None:
            require(
                cell["stream_status"] == "failed" and cell["error"] is not None,
                "incomplete cell cannot pass",
            )
            continue
        item = inventory[cell["case_id"]]
        events, traces = cell["events"], cell["decision_traces"]
        checks = dict(shared.publication_checks(events, traces, item["pcm"]))
        checks.update(
            corpus._paced_checks(events, traces, cell["pacing"], item["sample_count"])
        )
        metrics = SimpleNamespace(**cell["metrics"])
        checks.update(
            b._event_checks(
                events,
                traces,
                accepted_samples=metrics.accepted_samples,
                total_samples=item["sample_count"],
                state=SimpleNamespace(**cell["state"]),
                metrics=metrics,
                budget=SimpleNamespace(available=1, lease_count=0),
                worker=SimpleNamespace(queue_depth=0),
                capacity=1,
                retained_from_sample=cell["retained_from_sample"],
                max_buffer_samples=stream_config()["max_buffer_ms"] * 16,
                error_record=cell["error"],
            )
        )
        # Actual budget state is worker telemetry, not reconstructible from JSON.
        checks["runtime_capacity_restored"] = cell["checks"][
            "runtime_capacity_restored"
        ]
        checks["profile_identity"] = (
            cell["profile_id"]
            == "word_boundary_quiet_endpoint_stream/v1+input_evidence/v1"
        )
        require(checks == cell["checks"], "recomputed stream checks differ")
        text = c._normalized_committed_text(events)
        require(
            cell["recognition"]
            == dict(
                text=text,
                against_human_reference=b._word_difference(
                    text, item["reference_text"]
                ),
            ),
            "recognition score differs",
        )
        passed = (
            cell["error"] is None
            and all(checks.values())
            and cell.get("cleanup_error") is None
            and cell["capacity_restored"] is True
        )
        require(
            cell["stream_status"] == ("completed" if passed else "failed"),
            "stream verdict differs",
        )
        if passed:
            require(
                len(cell["windows"]) == cell["metrics"]["decode_count"],
                "completed decode count differs",
            )
            require(
                all(
                    sum(call["phase"] == "decode" for call in window["encoder_calls"])
                    == 1
                    for window in cell["windows"]
                ),
                "completed window lacks its encoder",
            )
            require(
                cell["capacity_restored"] is True
                and all(
                    x["closed"] is True and x["capacity_restored"] is True
                    for x in cell["windows"]
                ),
                "completed cell retained native work",
            )
    completed = (
        record["stop"] is None
        and len(cells) == len(schedule)
        and all(x["stream_status"] == "completed" for x in cells)
        and model["unchanged"]
    )
    require(
        record["status"] == ("completed" if completed else "failed"),
        "experiment verdict differs",
    )
    if completed:
        require(
            record["capacity_restored"] is True,
            "completed experiment retained capacity",
        )


def scope():
    return dict(
        stream_config=stream_config(),
        pacing=_corpus().PACED_REPLAY,
        warmup_arms=["baseline", "reuse"],
        warmup_sample_count=32000,
        maximum_native_windows=MAX_NATIVE_WINDOWS,
        worker_seconds=WORKER_SECONDS,
        reserve_seconds=RESERVE_SECONDS,
        same_model_and_lane=True,
        single_inference_owner=True,
        source_thread_only_admits_pcm=True,
        gpu_device_time_ms=None,
        host_timings_include_instrumentation=True,
        identical_windows_or_event_boundaries_required=False,
    )


def comparison_summary(record):
    """Describe every observed pair, including failures and unpaired windows."""
    pairs = []
    for item in record["inputs"]:
        arms = {
            cell["arm"]: cell
            for cell in record["cells"]
            if cell["case_id"] == item["id"]
        }
        pair = dict(case_id=item["id"], arms={})
        for name, cell in arms.items():
            windows = cell["windows"]
            pair["arms"][name] = dict(
                status=cell["stream_status"],
                native_windows=len(windows),
                encoder_forwards=sum(len(x["encoder_calls"]) for x in windows),
                native_host_ms=sum(
                    sum(x["operation_wall_ns"].values()) for x in windows
                )
                / 1e6,
                committed_samples=cell.get("metrics", {}).get("committed_samples"),
                word_errors=cell.get("recognition", {}).get("against_human_reference"),
            )
        if len(arms) == 2 and all(x.get("recognition") for x in arms.values()):
            left, right = arms["baseline"], arms["reuse"]

            def commits(cell):
                return [
                    (x["start_sample"], x["end_sample"], x["text"])
                    for x in cell["events"]
                    if x["kind"] == "commit"
                ]

            def aligned(cell):
                grouped = {}
                for trace in cell["decision_traces"]:
                    alignment = trace.get("word_alignment")
                    if alignment is not None:
                        key = (
                            trace["analysis_start_sample"],
                            trace["analysis_end_sample"],
                        )
                        grouped.setdefault(key, []).append(alignment["words"])
                return grouped

            a, b = aligned(left), aligned(right)
            common = sorted(a.keys() & b.keys())
            unique = [key for key in common if len(a[key]) == len(b[key]) == 1]
            pair.update(
                both_completed=all(
                    x["stream_status"] == "completed" for x in arms.values()
                ),
                committed_text_equal=left["recognition"]["text"]
                == right["recognition"]["text"],
                commit_spans_and_text_equal=commits(left) == commits(right),
                common_aligned_windows=len(unique),
                ambiguous_common_windows=len(common) - len(unique),
                unpaired_aligned_windows=len(a.keys() ^ b.keys()),
                common_word_outputs_equal=(
                    all(a[key] == b[key] for key in unique) if unique else None
                ),
                word_mismatch_windows=[list(key) for key in unique if a[key] != b[key]],
            )
        pairs.append(pair)
    return pairs


def run_worker(expected_snapshot):
    started = time.monotonic_ns()
    shared.verify_snapshot(expected_snapshot)
    corpus = _corpus()
    b, c = corpus.b, corpus.c
    backend = features.apply_backend_patch(REMOTE_ROOT, b.BACKEND_ROOT)
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
    base = corpus.read_registration(REMOTE_ROOT)
    q = corpus._helper("infra.modal_native_cuda_qualification")
    if (
        str(torch.__version__).split("+")[0] != "2.6.0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "Tesla T4"
    ):
        raise RuntimeError("registered PyTorch 2.6 / single T4 unavailable")
    checkpoint = c._sha256_file(b.MODEL_CHECKPOINT_PATH)
    if checkpoint != base["model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint mismatch")
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=b.MODEL_CACHE_MOUNT
    ).eval()
    initial = q._model_fingerprint(model)
    if initial != base["model"]["model_state_sha256"]:
        raise RuntimeError("model fingerprint mismatch")
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=features.PATCHED_TREE,
        backend="pytorch-cuda-paced-features",
        fingerprint=initial,
    )
    worker = runtime.Worker(
        "paced-features",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=180,
    )

    def probe(observed):
        if observed is not model:
            raise RuntimeError("model binding changed")
        return identity

    native = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/paced-features-v1",
            capacity,
            device="cuda:0",
            reuse_alignment_features=True,
        ),
    )
    measured = _MeasuredAdapter(native)
    options = adapters.NativeDecodeOptions(**base["decode_options"])
    inputs = cases(REMOTE_ROOT)
    inventory = {item["id"]: item for item in inputs}
    configuration = stream_config()
    endpoint_module = importlib.import_module(
        "whisper_runtime.adapters.audio_endpoints"
    )
    config = adapters.ContinuousStreamConfig(
        **{
            **configuration,
            "endpointing": endpoint_module.QuietEndpointConfig(
                **configuration["endpointing"]
            ),
        }
    )

    def mel(pcm):
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
        ).contiguous()

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

    def elapsed():
        return time.monotonic_ns() - started

    warmup, cells, stop = [], [], None
    hook = model.encoder.register_forward_pre_hook(
        measured.observe_encoder, with_kwargs=True
    )
    try:
        warm_pcm = inventory["mixed-control"]["pcm"][: 2000 * 32]
        for arm in ("baseline", "reuse"):
            measured.reuse = arm == "reuse"
            measured.records = []
            warm = dict(
                arm=arm,
                sample_count=32000,
                pcm_sha256=shared._sha(warm_pcm),
                windows=measured.records,
            )
            warmup.append(warm)
            phase_started = time.monotonic_ns()
            session = runtime.Session(f"paced-features:warmup:{arm}")
            run = measured.start_window(
                session=session,
                request=runtime.RequestState(
                    f"{session.session_id}:request",
                    session.session_id,
                    identity,
                    rng_seed=7,
                ),
                window_id=session.session_id,
                mel=mel(warm_pcm),
                start_ms=0,
                end_ms=2000,
                options=options,
            )
            try:
                for _ in range(b.MAX_DRIVER_STEPS):
                    if run.complete:
                        break
                    run.step()
                run.prepare_word_alignment()
            finally:
                run.close()
            del run
            warm.update(
                wall_ns=time.monotonic_ns() - phase_started,
                capacity_restored=available(),
            )
            if not available():
                raise RuntimeError("warmup retained capacity")

        for index, (case_id, arm) in enumerate(SCHEDULE):
            case = inventory[case_id]
            remaining_ns = WORKER_SECONDS * 1000000000 - elapsed()
            required_ns = int((case["duration_seconds"] + RESERVE_SECONDS) * 1000000000)
            required_ns += corpus.PACED_REPLAY["config"]["max_drain_ms"] * 1000000
            if remaining_ns < required_ns:
                stop = dict(
                    reason="budget_stop",
                    next_cell=index,
                    remaining_ns=remaining_ns,
                    required_ns=required_ns,
                )
                break
            measured.reuse = arm == "reuse"
            measured.records = []
            cell = dict(
                case_id=case_id,
                arm=arm,
                stream_status="failed",
                windows=measured.records,
                events=[],
                decision_traces=[],
                pacing=None,
                checks={},
                error=None,
            )
            cells.append(cell)
            stream = None
            try:
                if not available():
                    raise RuntimeError("previous cell retained capacity")
                cell["memory_before"] = memory()
                torch.cuda.reset_peak_memory_stats(0)
                stream = adapters.ContinuousTranscriptStream(
                    measured,
                    stream_id=f"paced-features:{case_id}:{arm}",
                    mel_builder=mel,
                    options=options,
                    rng_seed=7,
                    config=config,
                )
                events, traces, pacing, error = corpus._drive_paced(stream, case["pcm"])
                cell.update(
                    events=events, decision_traces=traces, pacing=pacing, error=error
                )
                metrics = stream.metrics
                cell.update(
                    metrics=b._plain(metrics),
                    state=b._plain(stream.state),
                    retained_from_sample=stream.retained_from_sample,
                    profile_id=stream.profile_id,
                )
                checks = b._event_checks(
                    events,
                    traces,
                    accepted_samples=metrics.accepted_samples,
                    total_samples=case["sample_count"],
                    state=stream.state,
                    metrics=metrics,
                    budget=budget,
                    worker=worker,
                    capacity=capacity,
                    retained_from_sample=stream.retained_from_sample,
                    max_buffer_samples=config.max_buffer_ms * 16,
                    error_record=error,
                )
                checks.update(shared.publication_checks(events, traces, case["pcm"]))
                checks.update(
                    corpus._paced_checks(events, traces, pacing, case["sample_count"])
                )
                checks["profile_identity"] = (
                    stream.profile_id
                    == "word_boundary_quiet_endpoint_stream/v1+input_evidence/v1"
                )
                cell["checks"] = checks
                text = c._normalized_committed_text(events)
                cell["recognition"] = dict(
                    text=text,
                    against_human_reference=b._word_difference(
                        text, case["reference_text"]
                    ),
                )
                cell["stream_status"] = (
                    "completed"
                    if error is None and stream.done and all(checks.values())
                    else "failed"
                )
            except Exception as error:
                cell["error"] = b._safe_stream_error(error)
                stop = dict(reason="cell_error", cell=index)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as error:
                        cell["cleanup_error"] = b._safe_stream_error(error)
                        cell["stream_status"] = "failed"
                        stop = dict(reason="cleanup_error", cell=index)
                    del stream
                cell["capacity_restored"] = available()
                if available():
                    cell["memory_after_close"] = memory()
                    cell["memory_peak"] = dict(
                        allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                        reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
                    )
                else:
                    cell["stream_status"] = "failed"
                    stop = dict(reason="capacity_retained", cell=index)
            if stop is not None:
                break
    except Exception as error:
        stop = dict(reason="worker_error", error=b._safe_stream_error(error))
    finally:
        hook.remove()

    final = None
    if available():
        try:
            final = q._model_fingerprint(model)
        except Exception as error:
            stop = dict(
                reason="final_model_verification_failed",
                error=b._safe_stream_error(error),
            )
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    completed = (
        stop is None
        and len(cells) == len(SCHEDULE)
        and all(cell["stream_status"] == "completed" for cell in cells)
        and final == initial
    )
    return dict(
        schema_version="1-diagnostic",
        experiment_id="modal-paced-features-v1",
        status="completed" if completed else "failed",
        qualified=False,
        claim_boundary=CLAIMS,
        inputs=[
            {key: value for key, value in item.items() if key != "pcm"}
            for item in inputs
        ],
        schedule=[dict(case_id=case_id, arm=arm) for case_id, arm in SCHEDULE],
        cells=cells,
        warmup=warmup,
        stop=stop,
        native_window_count=measured.count,
        backend=backend,
        capacity_restored=available(),
        elapsed_ns=elapsed(),
        model=dict(
            initial_sha256=initial,
            final_sha256=final,
            unchanged=final == initial,
            checkpoint_sha256=checkpoint,
            backend_revision=identity.revision,
        ),
        scope=scope(),
        source=dict(
            snapshot=expected_snapshot,
            registration_sha256=shared._sha((REMOTE_ROOT / PRODUCER).read_bytes()),
        ),
        worker=dict(
            function_call_id=call_id,
            function_call_id_sha256=shared._sha(call_id.encode()),
            modal=str(modal.__version__),
            torch=str(torch.__version__),
        ),
    )


if __name__ == "__main__":
    from infra import modal_cuda_lane as launcher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(
        launcher.run(
            producer=importlib.import_module("infra.modal_paced_features"),
            **vars(parser.parse_args()),
        )
    )
