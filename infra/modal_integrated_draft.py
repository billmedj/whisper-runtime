"""One bounded T4 test of the integrated draft path. Preflight is local only."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import time
import uuid
from dataclasses import asdict, replace

from infra import modal_draft_features as prior

composed, shared, features, live, paced = (
    prior.composed,
    prior.shared,
    prior.features,
    prior.live,
    prior.paced,
)
ROOT, REMOTE_ROOT = prior.ROOT, prior.REMOTE_ROOT
PRODUCER = "infra/modal_integrated_draft.py"
TEST = "tools/test_modal_integrated_draft.py"
PREREGISTRATION = "docs/research/2026-09-07-integrated-draft-gpu-plan.md"
GPU_TIMEOUT_SECONDS, RESERVE_SECONDS, MAX_NATIVE_WINDOWS = 480, 20, 256
SCHEDULE = (
    ("a1", "fast-reuse"),
    ("b1", "fast-draft32"),
    ("b2", "fast-draft32"),
    ("a2", "fast-reuse"),
)
CLAIMS = dict(
    prior.CLAIMS,
    sustained_live=False,
    held_out_accuracy=False,
    integrated_gpu_qualification=False,
)
_require, _corpus = prior._require, prior._corpus


def configurations():
    return {
        arm: replace(config, max_draft_tokens=0 if arm == prior.ARMS[0] else 32)
        for arm, config in prior.configurations().items()
    }


def input_plan(root=ROOT):
    seed, original = composed.input_plan(root)
    pcm = seed * 2
    spans = [
        {
            "cycle": cycle,
            "kind": name,
            **dict(
                span,
                start_sample=span["start_sample"] + cycle * 744800,
                end_sample=span["end_sample"] + cycle * 744800,
            ),
        }
        for cycle in range(2)
        for name, span in original["spans"].items()
    ]
    return pcm, dict(
        sample_count=len(pcm) // 2,
        duration_seconds=len(pcm) / 32000,
        pcm_sha256=shared._sha(pcm),
        cycles=2,
        cycle_samples=744800,
        cycle_sha256=original["pcm_sha256"],
        spans=spans,
        reference_text=" ".join([original["reference_text"]] * 2),
        recipe="Repeat the registered speech/silence/noise/silence cycle twice",
    )


def scope():
    rate = (
        live.PRICES["t4_per_second"]
        + 2 * live.PRICES["cpu_core_per_second"]
        + 4 * live.PRICES["gib_per_second"]
    )
    estimate = GPU_TIMEOUT_SECONDS * rate * live.PRICES["regional_multiplier_ceiling"]
    _require(
        GPU_TIMEOUT_SECONDS == 480 and RESERVE_SECONDS == 20 and estimate < 0.18,
        "registered resource bound changed",
    )
    return dict(
        gpu="T4",
        gpu_calls=1,
        timeout_seconds=GPU_TIMEOUT_SECONDS,
        reserve_seconds=RESERVE_SECONDS,
        min_containers=0,
        max_containers=1,
        retries=0,
        maximum_native_windows=MAX_NATIVE_WINDOWS,
        schedule=[list(s) for s in SCHEDULE],
        configurations={a: asdict(c) for a, c in configurations().items()},
        warmup_windows_per_arm=2,
        warmup_samples=128000,
        cancellation_probe_windows=1,
        prices=live.PRICES,
        prices_checked="2026-09-07",
        prices_source="https://modal.com/pricing",
        planning_compute_usd=round(estimate, 6),
        physical_spending_cap=False,
        cost_exclusions="build/startup/storage/network/taxes/provider rescheduling",
        model_download=False,
        same_model_and_lane=True,
        inference_owners=1,
        backend_method_patching=False,
        pacing=_corpus().PACED_REPLAY,
    )


def snapshot(root=ROOT):
    inherited = prior.snapshot(root)
    names = {item["path"] for item in inherited["files"]} | {
        PRODUCER,
        TEST,
        PREREGISTRATION,
    }
    files = [
        dict(path=name, size_bytes=len(raw), sha256=shared._sha(raw))
        for name in sorted(names)
        for raw in [(root / name).read_bytes()]
    ]
    return dict(commit=inherited["commit"], files=files, digest=shared._hash(files))


def admit(started, seconds):
    return time.monotonic() - started + seconds + RESERVE_SECONDS < GPU_TIMEOUT_SECONDS


class Measured(composed.Measured):
    """Observe owned native operations; never replace decoder methods."""

    def start_window(self, **kwargs):
        _require(self.count < MAX_NATIVE_WINDOWS, "native window bound reached")
        self.count += 1
        record = dict(
            call_index=self.count,
            window_id=kwargs["window_id"],
            start_ms=kwargs["start_ms"],
            end_ms=kwargs["end_ms"],
            draft_tokens=list(kwargs.get("draft_tokens", ())),
            reuse_alignment_features=self.reuse,
            encoder_calls=[],
            operation_wall_ns={},
            gpu_device_time_ms=None,
            closed=False,
            capacity_restored=False,
        )
        self.records.append(record)
        raw = self.invoke(record, "decode", self.native.start_window, **kwargs)
        return paced._MeasuredRun(self, raw, record, self.reuse)

    def before(self, kind, module, args, kwargs):
        super().before(kind, module, args, kwargs)
        if kind == "decoder":
            tokens = args[0] if args else kwargs["x"]
            _require(len(tokens.shape) == 2, "invalid decoder token axes")
            self.active[0]["forwards"][-1]["input_tokens"] = int(
                tokens.shape[0] * tokens.shape[1]
            )

    def invoke(self, record, phase, method, *args, **kwargs):
        owner = getattr(method, "__self__", None)
        backend = getattr(owner, "_backend_run", None)
        inference = getattr(backend, "inference", None)
        try:
            result = super().invoke(record, phase, method, *args, **kwargs)
            if phase == "result":
                metadata = result.metadata
                _require(metadata is not None, "native result lacks metadata")
                record["decode_observation"] = dict(
                    call_index=record["call_index"],
                    start_ms=record["start_ms"],
                    end_ms=record["end_ms"],
                    result=dict(
                        text=result.text,
                        **{
                            key: getattr(metadata, key)
                            for key in (
                                "tokens",
                                "language",
                                "avg_logprob",
                                "no_speech_prob",
                                "temperature",
                                "compression_ratio",
                            )
                        },
                    ),
                    stats=dict(inference.stats)
                    if hasattr(inference, "stats")
                    else None,
                )
            return result
        finally:
            if inference is not None and (
                phase in ("close", "finish")
                or (
                    getattr(owner, "closed", False)
                    and getattr(owner, "capacity_released", False)
                )
            ):
                original = getattr(inference, "original", inference)
                record["native_cleanup"] = dict(
                    cache_empty=not original.kv_cache,
                    hooks_empty=not original.hooks,
                    rows_cleared=getattr(inference, "rows", None) is None,
                    audio_cleared=getattr(inference, "audio_features", None) is None,
                    draft_cleared=not getattr(inference, "draft", ()),
                )


def analyze(cells):
    if len(cells) != 4 or any(c["stream_status"] != "completed" for c in cells):
        return dict(complete=False, accepted=False)
    pairs = [prior.comparison([cells[a], cells[b]]) for a, b in ((0, 1), (3, 2))]
    events_exact = all(c["events"] == cells[0]["events"] for c in cells[1:])
    groups = {arm: [c for c in cells if c["arm"] == arm] for arm in prior.ARMS}
    totals = {
        arm: dict(
            decode_wall_ns=sum(
                c["summary"]["operation_wall_ns"].get("decode", 0) for c in group
            ),
            total_wall_ns=sum(c["summary"]["total_operation_wall_ns"] for c in group),
            forward_interval_ms=sum(
                sum(c["summary"]["forward_interval_ms"].values()) for c in group
            ),
        )
        for arm, group in groups.items()
    }
    control, candidate = (totals[a] for a in prior.ARMS)
    faster = {key: candidate[key] < control[key] for key in control}
    accepted = events_exact and all(p["accepted"] for p in pairs)
    return dict(
        complete=True,
        accepted=accepted,
        pairs=pairs,
        full_events_exact=events_exact,
        pooled_totals=totals,
        pooled_faster=faster,
        efficiency_gate=accepted and all(faster.values()),
        scope="One order-balanced screening run; not independent repetitions or held-out audio",
    )


def run_worker(expected):
    started = time.monotonic()
    shared.verify_snapshot(expected)
    corpus, b = _corpus(), _corpus().b
    backend = features.apply_backend_patch(REMOTE_ROOT, b.BACKEND_ROOT)
    torch, np, whisper, runtime, adapters, setup = (
        importlib.import_module(name)
        for name in (
            "torch",
            "numpy",
            "whisper",
            "whisper_runtime",
            "whisper_runtime.adapters",
            "whisper_runtime.native_setup",
        )
    )
    _require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and torch.cuda.get_device_name(0) == "Tesla T4",
        "registered T4 unavailable",
    )
    setup.validate_dependencies()
    setup.validate_checkpoint(b.MODEL_CHECKPOINT_PATH)
    torch.set_num_threads(1)
    model = setup._load_model(whisper, b.MODEL_CHECKPOINT_PATH, "cuda:0")
    initial = setup._fingerprint(model)
    _require(
        initial == setup.MODEL_FINGERPRINT and not model.training,
        "model identity differs",
    )
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        "tiny.en", features.PATCHED_TREE, "pytorch-cuda-integrated-draft", initial
    )
    worker = runtime.Worker(
        "integrated-draft",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=180,
    )

    def probe(observed):
        _require(observed is model, "model binding changed")
        return identity

    native = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/integrated-draft-v1",
            capacity,
            device="cuda:0",
            reuse_alignment_features=True,
        ),
    )
    measured = Measured(
        native, torch.cuda, started + GPU_TIMEOUT_SECONDS - RESERVE_SECONDS
    )
    measured.reuse = True
    options = adapters.NativeDecodeOptions(
        **corpus.read_registration(REMOTE_ROOT)["decode_options"]
    )
    pcm, plan = input_plan(REMOTE_ROOT)

    def mel(content):
        audio = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768.0
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
        return dict(
            allocated_bytes=int(torch.cuda.memory_allocated(0)),
            reserved_bytes=int(torch.cuda.memory_reserved(0)),
        )

    def observations():
        return [
            w["decode_observation"]
            for w in measured.records
            if "decode_observation" in w
        ]

    hooks, warmup, cells, probes, stop = [], [], [], [], None
    before_hooks = [
        (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in model.modules()
    ]
    for kind in ("encoder", "decoder"):
        module = getattr(model, kind)
        hooks.append(
            module.register_forward_pre_hook(
                lambda m, a, kw, k=kind: measured.before(k, m, a, kw), with_kwargs=True
            )
        )
        hooks.append(
            module.register_forward_hook(
                lambda m, a, kw, out, k=kind: measured.after(k, m, a, kw, out),
                with_kwargs=True,
                always_call=True,
            )
        )
    try:
        for arm in prior.ARMS:
            _require(admit(started, 10), "budget_stop before warmup")
            measured.records, hint = [], ()
            warm = dict(arm=arm, windows=measured.records)
            warmup.append(warm)
            for index in range(2):
                session = runtime.Session(f"warm:{arm}:{index}")
                run = measured.start_window(
                    session=session,
                    request=runtime.RequestState(
                        session.session_id + ":request",
                        session.session_id,
                        identity,
                        rng_seed=7,
                    ),
                    window_id=session.session_id,
                    mel=mel(pcm[:256000]),
                    start_ms=0,
                    end_ms=8000,
                    options=options,
                    draft_tokens=hint if arm == prior.ARMS[1] else (),
                )
                try:
                    for _ in range(224):
                        if run.complete:
                            break
                        run.step()
                    _require(run.complete, "warmup token bound reached")
                    hint = tuple(run.prepare_result().metadata.tokens[:32])
                    run.prepare_word_alignment()
                finally:
                    run.close()
                del run
            measured.resolve()
            warm["capacity_restored"] = available()
            _require(available(), "warmup retained capacity")
        _require(
            len(hint) == 32 and admit(started, 5), "cancellation probe unavailable"
        )
        measured.records = []
        session = runtime.Session("integrated-cancel")
        cancelled = dict(windows=measured.records)
        probes.append(cancelled)
        run = measured.start_window(
            session=session,
            request=runtime.RequestState(
                "cancel", session.session_id, identity, rng_seed=7
            ),
            window_id="cancel",
            mel=mel(pcm[:256000]),
            start_ms=0,
            end_ms=8000,
            options=options,
            draft_tokens=hint,
        )
        try:
            inference = run.raw._backend_run.inference
            _require(inference.rows is not None, "probe lacks speculative prefill")
            cancelled["requested"] = run.cancel()
            try:
                run.step()
            except runtime.RequestCancelledError:
                cancelled["observed"] = True
            else:
                cancelled["observed"] = False
        finally:
            run.close()
        cancelled.update(
            rows_cleared=inference.rows is None,
            audio_cleared=inference.audio_features is None,
            hint_cleared=not inference.draft,
            cache_empty=not inference.original.kv_cache,
            session_unchanged=session.snapshot().version == 0,
            capacity_restored=available(),
        )
        del run, inference
        measured.resolve()
        _require(
            all(v for k, v in cancelled.items() if k != "windows"),
            "cancellation failed",
        )
        for position, arm in SCHEDULE:
            if not admit(
                started,
                plan["duration_seconds"]
                + corpus.PACED_REPLAY["config"]["max_drain_ms"] / 1000,
            ):
                stop = dict(reason="budget_stop", next_position=position)
                break
            measured.records = []
            cell = dict(
                position=position,
                arm=arm,
                config=asdict(configurations()[arm]),
                windows=measured.records,
                events=[],
                decision_traces=[],
                pacing=None,
                error=None,
                input_sha256=plan["pcm_sha256"],
                memory_before=memory(),
                admitted_elapsed_ns=int((time.monotonic() - started) * 1e9),
            )
            cells.append(cell)
            torch.cuda.reset_peak_memory_stats(0)
            stream = None
            try:
                _require(available(), "previous cell retained capacity")
                stream = adapters.ContinuousTranscriptStream(
                    measured,
                    stream_id="integrated-draft",
                    mel_builder=mel,
                    options=options,
                    rng_seed=7,
                    config=configurations()[arm],
                )
                events, traces, pacing, error = prior.drive_paced(stream, pcm)
                cell.update(
                    events=events,
                    decision_traces=traces,
                    pacing=pacing,
                    error=error,
                    metrics=b._plain(stream.metrics),
                    state=b._plain(stream.state),
                    done=stream.done,
                    profile_id=stream.profile_id,
                )
            except Exception as error:
                cell["error"] = b._safe_stream_error(error)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                        cell["hint_cleared"] = not stream._draft_tokens
                    except Exception as error:
                        cell["cleanup_error"] = b._safe_stream_error(error)
                del stream
                if available() and cell.get("cleanup_error") is None:
                    try:
                        measured.resolve()
                    except Exception as error:
                        cell["cleanup_error"] = b._safe_stream_error(error)
                cell.update(
                    capacity_restored=available(),
                    memory_after_close=memory(),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
                    draft_observations=observations(),
                )
                cell["checks"] = dict(
                    composed.cell_checks(cell, pcm),
                    hint_cleared=cell.get("hint_cleared") is True,
                )
                cell["stream_status"] = (
                    "completed" if all(cell["checks"].values()) else "failed"
                )
                cell["summary"] = prior.summary(cell, plan)
            if cell["stream_status"] != "completed":
                stop = dict(reason="cell_failed", position=position)
                break
    except Exception as error:
        stop = dict(reason="worker_error", error=b._safe_stream_error(error))
    finally:
        for hook in reversed(hooks):
            hook.remove()
    final = None
    if available():
        try:
            final = setup._fingerprint(model)
        except Exception as error:
            stop = dict(
                reason="final_model_verification_failed",
                error=b._safe_stream_error(error),
            )
    hooks_unchanged = before_hooks == [
        (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in model.modules()
    ]
    call_id = importlib.import_module("modal").current_function_call_id()
    record = dict(
        schema_version="1-diagnostic",
        experiment_id="modal-integrated-draft-v1",
        qualified=False,
        source=dict(
            snapshot=expected,
            registration_sha256=shared._sha((REMOTE_ROOT / PRODUCER).read_bytes()),
        ),
        claim_boundary=CLAIMS,
        scope=scope(),
        input=plan,
        backend=backend,
        warmup=warmup,
        probes=probes,
        cells=cells,
        stop=stop,
        native_window_count=measured.count,
        capacity_restored=available(),
        hooks_unchanged=hooks_unchanged,
        elapsed_ns=int((time.monotonic() - started) * 1e9),
        model=dict(
            initial_sha256=initial,
            final_sha256=final,
            unchanged=initial == final,
            checkpoint_sha256=setup.CHECKPOINT_SHA256,
            loads=1,
        ),
        effective_identity=dict(
            decode_options=asdict(options),
            rng_seed=7,
            profile=asdict(native.execution_profile),
        ),
        worker=dict(
            function_call_id=call_id,
            function_call_id_sha256=shared._sha(call_id.encode()),
            torch=str(torch.__version__),
            modal=str(importlib.import_module("modal").__version__),
        ),
    )
    record["status"] = (
        "completed"
        if stop is None
        and len(cells) == 4
        and available()
        and initial == final
        and hooks_unchanged
        else "partial"
    )
    record["comparison"] = (
        analyze(cells)
        if record["status"] == "completed"
        else dict(complete=False, accepted=False)
    )
    return record


def validate_record(record, expected, root=ROOT):
    pcm, plan = input_plan(root)
    _require(
        record["experiment_id"] == "modal-integrated-draft-v1"
        and record["scope"] == scope()
        and record["input"] == plan
        and record["source"]["snapshot"] == expected
        and record["claim_boundary"] == CLAIMS
        and record["qualified"] is False,
        "record identity differs",
    )
    _require(record["status"] in ("completed", "partial"), "invalid terminal status")
    _require(
        record["backend"]["patched_tree"] == features.PATCHED_TREE,
        "backend identity differs",
    )
    cells = record["cells"]
    _require(
        [(c["position"], c["arm"]) for c in cells] == list(SCHEDULE[: len(cells)])
        and len(cells) <= 4,
        "cell order differs",
    )
    windows = [
        w
        for group in (record["warmup"], record["probes"], cells)
        for c in group
        for w in c["windows"]
    ]
    _require(
        record["native_window_count"] == len(windows) <= MAX_NATIVE_WINDOWS
        and [w["call_index"] for w in windows] == list(range(1, len(windows) + 1)),
        "native window accounting differs",
    )
    for window in windows:
        _require(
            0 <= window["start_ms"] < window["end_ms"] <= 93100
            and window["end_ms"] - window["start_ms"] <= 30000,
            "native audio span differs",
        )
        for forward in window.get("forwards", []):
            value = forward["interval_ms"]
            _require(
                value is None
                or (
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                ),
                "invalid forward interval",
            )
            if forward["kind"] == "decoder":
                _require(
                    type(forward["input_tokens"]) is int
                    and forward["input_tokens"] > 0,
                    "invalid token count",
                )
        observation = window.get("decode_observation")
        if observation:
            _require(
                (observation["stats"] is not None) == bool(window["draft_tokens"]),
                "native draft activation differs",
            )
            if observation["stats"]:
                _require(
                    observation["stats"]["proposed_tokens"]
                    == len(window["draft_tokens"])
                    and observation["stats"]["parallel_prefills"] == 1,
                    "native draft work differs",
                )
    for cell in cells:
        checks = dict(
            composed.cell_checks(cell, pcm),
            hint_cleared=cell.get("hint_cleared") is True,
        )
        _require(
            cell["config"] == asdict(configurations()[cell["arm"]])
            and cell["input_sha256"] == plan["pcm_sha256"],
            "cell input/config differs",
        )
        _require(
            cell["checks"] == checks and cell["summary"] == prior.summary(cell, plan),
            "cell summary differs",
        )
        _require(
            cell["draft_observations"]
            == [
                w["decode_observation"]
                for w in cell["windows"]
                if "decode_observation" in w
            ],
            "observations differ",
        )
        _require(
            cell["stream_status"]
            == ("completed" if all(checks.values()) else "failed"),
            "cell status differs",
        )
        if cell["arm"] == prior.ARMS[0]:
            _require(
                not any(w["draft_tokens"] for w in cell["windows"]),
                "control used a draft",
            )
    if record["status"] == "completed":
        _require(
            record["stop"] is None
            and len(cells) == 4
            and all(c["stream_status"] == "completed" for c in cells)
            and len(record["warmup"]) == 2
            and len(record["probes"]) == 1
            and all(
                len(w["windows"]) == 2 and w["capacity_restored"]
                for w in record["warmup"]
            )
            and all(v for k, v in record["probes"][0].items() if k != "windows")
            and record["capacity_restored"]
            and record["hooks_unchanged"]
            and record["model"]["unchanged"],
            "invalid completion",
        )
        _require(
            record["model"]["initial_sha256"]
            == record["model"]["final_sha256"]
            == importlib.import_module(
                "whisper_runtime.native_setup"
            ).MODEL_FINGERPRINT,
            "final model identity differs",
        )
        _require(
            all(
                w["closed"]
                and w["capacity_restored"]
                and w.get("native_cleanup")
                and all(w["native_cleanup"].values())
                and all(f["interval_ms"] is not None for f in w.get("forwards", []))
                for w in windows
            ),
            "incomplete native cleanup or timing",
        )
        _require(
            record["warmup"][1]["windows"][1]["decode_observation"]["stats"][
                "parallel_prefills"
            ]
            == 1,
            "candidate warmup did not use the integrated draft",
        )
    expected_comparison = (
        analyze(cells)
        if record["status"] == "completed"
        else dict(complete=False, accepted=False)
    )
    _require(record["comparison"] == expected_comparison, "comparison differs")


def resources(expected, root=ROOT):
    modal = importlib.import_module("modal")
    _require(str(modal.__version__) == "1.5.5", "registered Modal SDK required")
    corpus = _corpus()
    q = corpus._helper("infra.modal_native_cuda_qualification")
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("ca-certificates", "ffmpeg", "git")
        .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
        .uv_pip_install(*q.DIRECT_IMAGE_PACKAGES)
        .run_commands(q._build_command(corpus.BASE_COMMIT))
    )
    image = image.env(
        dict(
            PYTHONPATH="/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
            WHISPER_MODAL_ENABLE_WORD_CORPUS="0",
            WHISPER_MODAL_ENABLE_REMOTE_RESOURCES="0",
        )
    )
    for item in expected["files"]:
        image = image.add_local_file(
            root / item["path"], (REMOTE_ROOT / item["path"]).as_posix(), copy=False
        )
    for fixture in corpus.read_registration(root)["fixtures"]:
        image = image.add_local_file(
            root / corpus.ASSET_PATH / fixture["filename"],
            (REMOTE_ROOT / corpus.ASSET_PATH / fixture["filename"]).as_posix(),
            copy=False,
        )
    app = modal.App("wr-integrated-draft-" + uuid.uuid4().hex[:12])
    common = dict(
        image=image,
        serialized=True,
        min_containers=0,
        max_containers=1,
        retries=0,
        startup_timeout=300,
        scaledown_window=2,
        block_network=True,
        restrict_modal_access=True,
        single_use_containers=True,
        include_source=False,
    )

    @app.function(**common, cpu=0.125, memory=128, timeout=30)
    def echo(payload):
        return payload

    volume = modal.Volume.from_name(corpus.b.MODEL_CACHE_NAME, create_if_missing=False)

    @app.function(
        **common,
        cpu=2,
        memory=4096,
        gpu="T4",
        cloud="aws",
        region="us-west",
        timeout=GPU_TIMEOUT_SECONDS,
        volumes={corpus.b.MODEL_CACHE_MOUNT: volume.with_mount_options(read_only=True)},
    )
    def execute():
        producer = importlib.import_module("infra.modal_integrated_draft")
        return producer._corpus().b._encode_worker_record(producer.run_worker(expected))

    return app, echo, execute


def run(*, replay_id, preflight=False, confirm_paid_gpu=False, root=ROOT):
    output, journal, raw = shared._paths(root, replay_id, False)
    frozen = output.parent / "integrated-draft-preflight.json"
    _require(preflight or confirm_paid_gpu is True, "requires --confirm-paid-gpu")
    _require(
        not any(p.exists() for p in (output, journal, raw)),
        "attempt exists; no retry/overwrite",
    )
    expected, (_, plan) = snapshot(root), input_plan(root)
    spec = dict(source=expected, input=plan, scope=scope())
    if preflight:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        live.prior._write_result(frozen, spec)
        return frozen
    _require(
        json.loads(frozen.read_text(encoding="utf-8")) == spec,
        "frozen source or plan changed",
    )
    live.prior._journal(
        journal, "attempt-started", first=True, source_digest=expected["digest"]
    )
    try:
        app, echo, execute = resources(expected, root)
        with app.run(detach=False):
            payload = _corpus().b._transport_probe_payload()
            _require(echo.remote(payload) == payload, "CPU transport preflight failed")
            payload = execute.remote()
            _corpus().b._write_bytes_exclusive(raw, payload)
            record = _corpus().b._decode_worker_record(
                payload,
                expected_snapshot=expected,
                registration_sha256=shared._sha((root / PRODUCER).read_bytes()),
                manifest=dict(claim_boundary=CLAIMS),
            )
            validate_record(record, expected, root)
            record["app_id"] = app.app_id
        record["ephemeral_app_exit_completed"] = True
        live.prior._write_result(output, record)
        live.prior._journal(journal, "attempt-finished", status=record["status"])
        return output
    except BaseException as error:
        live.prior._journal(journal, "attempt-failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(run(**vars(parser.parse_args())))
