"""Frozen three-arm, same-model T4 diagnostic. Imports/preflight never launch GPU."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import subprocess
import time
import uuid
from dataclasses import asdict, replace

from infra import modal_live_qualification as live
from infra import modal_paced_features as paced
from tools.analyze_stream_text_agreement import _commits

shared, features = paced.shared, paced.features
ROOT, REMOTE_ROOT = shared.ROOT, shared.REMOTE_ROOT
PRODUCER = "infra/modal_composed_features.py"
TEST = "tools/test_modal_composed_features.py"
PREREGISTRATION = "docs/research/2026-09-06-composed-inference.md"
GPU_TIMEOUT_SECONDS, RESERVE_SECONDS = 300, 20
ARMS = ("cli-legacy", "fast-legacy", "fast-reuse")
CLAIMS = dict.fromkeys(
    (
        "production_readiness",
        "general_speedup",
        "gpu_occupancy",
        "energy",
        "word_latency",
        "cross_window_feature_reuse",
    ),
    False,
)
_corpus, _require = paced._corpus, live._require


def configurations():
    cli = importlib.import_module("whisper_runtime.native_setup").CLI_STREAM_CONFIG
    _require(
        (cli.left_context_ms, cli.word_context_limit_ms) == (2000, 6000),
        "conservative CLI configuration changed",
    )
    return {
        arm: replace(
            cli,
            left_context_ms=2000 if arm == ARMS[0] else 20000,
            word_context_limit_ms=6000 if arm == ARMS[0] else 24000,
        )
        for arm in ARMS
    }


def input_plan(root=ROOT):
    pcm, plan = live.input_plan(root, short_only=True)
    _require(len(pcm) == 744800 * 2, "registered 46.55 second recipe changed")
    return pcm, dict(
        sample_count=len(pcm) // 2,
        pcm_sha256=shared._sha(pcm),
        recipe=plan["recipe"],
        reference_text=plan["smoke_reference"],
        spans=plan["arms"],
        main_sha256=plan["main_sha256"],
        noisy_prefix_sha256=plan["noisy_prefix_sha256"],
    )


def scope():
    rate = sum(
        (
            live.PRICES["t4_per_second"],
            2 * live.PRICES["cpu_core_per_second"],
            4 * live.PRICES["gib_per_second"],
        )
    )
    estimate = GPU_TIMEOUT_SECONDS * rate * live.PRICES["regional_multiplier_ceiling"]
    _require(0 < GPU_TIMEOUT_SECONDS <= 300 and estimate < 1, "budget exceeds plan")
    return dict(
        gpu="T4",
        gpu_calls=1,
        timeout_seconds=GPU_TIMEOUT_SECONDS,
        min_containers=0,
        max_containers=1,
        retries=0,
        reserve_seconds=RESERVE_SECONDS,
        planning_compute_usd=round(estimate, 6),
        physical_spending_cap=False,
        cost_exclusions="build/startup/storage/network/taxes/provider rescheduling",
        prices=live.PRICES,
        prices_checked="2026-09-06",
        prices_source="https://modal.com/pricing",
        same_model_and_lane=True,
        inference_owners=1,
        model_download=False,
        warmup_arms=list(ARMS),
        warmup_samples=32000,
        maximum_native_windows=paced.MAX_NATIVE_WINDOWS,
        configurations={a: asdict(c) for a, c in configurations().items()},
        alignment={a: a == ARMS[2] for a in ARMS},
        common_profile_opt_in=True,
        pacing=_corpus().PACED_REPLAY,
        timing="CUDA-event forward intervals; not occupancy/energy",
        host_timings_include_instrumentation=True,
    )


def snapshot(root=ROOT):
    corpus = _corpus()
    names = {
        p.relative_to(root).as_posix()
        for p in (root / "src/whisper_runtime").rglob("*.py")
    }
    names.update(
        (
            PRODUCER,
            TEST,
            PREREGISTRATION,
            shared.PRODUCER,
            shared.MANIFEST,
            *shared.BUILDERS,
            corpus.PRODUCER_PATH,
            corpus.MANIFEST_PATH,
            *corpus.HELPER_PATHS,
            paced.PRODUCER,
            features.PRODUCER,
            features.PATCH,
            live.PRODUCER,
            "infra/__init__.py",
            "infra/modal_network_replay.py",
            "infra/modal_word_resolution.py",
            "infra/modal_native_cuda_qualification.py",
            "infra/native_cuda_trace.py",
            "infra/modal-native-cuda-image-inputs.lock",
            "tools/analyze_stream_text_agreement.py",
            "tools/analyze_word_resolution.py",
            "tools/word_anchor_reconciliation.py",
        )
    )
    files = [
        dict(path=n, size_bytes=len(raw), sha256=shared._sha(raw))
        for n in sorted(names)
        for raw in [(root / n).read_bytes()]
    ]
    return dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        files=files,
        digest=shared._hash(files),
    )


class Measured(paced._MeasuredAdapter):
    """Events bracket real forwards on the already-owned lane; never sync tokens."""

    def __init__(self, native, cuda, deadline):
        super().__init__(native)
        self.cuda, self.deadline, self.pending, self.stack = cuda, deadline, [], {}

    def invoke(self, record, phase, method, *args, **kwargs):
        if phase not in {"close", "finish"} and time.monotonic() >= self.deadline:
            raise TimeoutError("worker cleanup reserve reached")
        return super().invoke(record, phase, method, *args, **kwargs)

    def before(self, kind, module, args, kwargs):
        _require(self.active is not None, "forward escaped measured native execution")
        lane = self.native._model_binding._cuda_lane
        stream = self.cuda.current_stream(0)
        _require(
            lane is not None
            and lane.owner is not None
            and stream.cuda_stream == lane.stream.cuda_stream,
            "forward escaped owned stream",
        )
        if kind == "encoder":
            self.observe_encoder(module, args, kwargs)
        record, phase = self.active
        call = dict(
            kind=kind, phase=phase, stream=int(stream.cuda_stream), interval_ms=None
        )
        record.setdefault("forwards", []).append(call)
        begin, end = (
            self.cuda.Event(enable_timing=True),
            self.cuda.Event(enable_timing=True),
        )
        begin.record(stream)
        self.stack.setdefault(kind, []).append((call, begin, end, stream))

    def after(self, kind, module, args, kwargs, output):
        if self.stack.get(kind):
            call, begin, end, stream = self.stack[kind].pop()
            end.record(stream)
            self.pending.append((call, begin, end))

    def resolve(self):
        lane = self.native._model_binding._cuda_lane
        _require(
            all(w["closed"] and w["capacity_restored"] for w in self.records)
            and (lane is None or lane.owner is None)
            and not any(self.stack.values()),
            "timing requires finished native completion fence",
        )
        for call, begin, end in self.pending:
            _require(end.query(), "completion fence did not finish timing events")
            call["interval_ms"] = float(begin.elapsed_time(end))
        self.pending.clear()


def summary(cell, plan):
    commits = [
        dict(start_sample=e.start_sample, end_sample=e.end_sample, text=t)
        for e, t in _commits(cell["events"])
    ]
    text = " ".join(c["text"] for c in commits)
    forwards = [f for w in cell["windows"] for f in w.get("forwards", [])]
    return dict(
        commits=commits,
        text=text,
        word_errors=_corpus().b._word_difference(text, plan["reference_text"]),
        final_count=sum(e["kind"] == "final" for e in cell["events"]),
        final_coverage=cell.get("metrics", {}).get("committed_samples", 0)
        == plan["sample_count"],
        committed_samples=cell.get("metrics", {}).get("committed_samples", 0),
        first_commit_ns=(cell.get("pacing") or {}).get("first_commit_ns"),
        forwards={
            k: sum(f["kind"] == k for f in forwards) for k in ("encoder", "decoder")
        },
        forward_interval_ms={
            k: None
            if any(f["interval_ms"] is None for f in forwards if f["kind"] == k)
            else sum(f["interval_ms"] for f in forwards if f["kind"] == k)
            for k in ("encoder", "decoder")
        },
    )


def cell_checks(cell, pcm):
    checks = dict(
        shared.publication_checks(cell["events"], cell["decision_traces"], pcm)
    )
    if cell.get("pacing") is not None:
        checks.update(
            _corpus()._paced_checks(
                cell["events"], cell["decision_traces"], cell["pacing"], len(pcm) // 2
            )
        )
    checks.update(
        full_input=cell.get("metrics", {}).get("accepted_samples") == len(pcm) // 2,
        final_coverage=cell.get("metrics", {}).get("committed_samples")
        == len(pcm) // 2,
        final_once=sum(e["kind"] == "final" for e in cell["events"]) == 1,
        done=cell.get("done") is True,
        no_error=cell.get("error") is None,
        cleanup=cell.get("capacity_restored") is True
        and cell.get("cleanup_error") is None
        and all(w["closed"] and w["capacity_restored"] for w in cell["windows"]),
    )
    return checks


def admit(started, seconds):
    return time.monotonic() - started + seconds + RESERVE_SECONDS < GPU_TIMEOUT_SECONDS


def run_worker(expected_snapshot):
    started = time.monotonic()
    shared.verify_snapshot(expected_snapshot)
    corpus = _corpus()
    b = corpus.b
    backend = features.apply_backend_patch(REMOTE_ROOT, b.BACKEND_ROOT)
    torch, np, whisper, runtime, adapters, setup = (
        importlib.import_module(n)
        for n in (
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
        "registered single T4 unavailable",
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
    _require(
        all(
            str(v.device) == "cuda:0"
            and (not v.is_floating_point() or v.dtype == torch.float32)
            for v in (*model.parameters(), *model.buffers())
        ),
        "model device/precision differs",
    )
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        "tiny.en", features.PATCHED_TREE, "pytorch-cuda-composed", initial
    )
    worker = runtime.Worker(
        "composed", identity, budget, queue_capacity=1, transaction_ttl_seconds=180
    )

    def probe(observed):
        _require(observed is model, "model binding changed")
        return identity

    native = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/composed-v1",
            capacity,
            device="cuda:0",
            reuse_alignment_features=True,
        ),
    )
    measured = Measured(
        native, torch.cuda, started + GPU_TIMEOUT_SECONDS - RESERVE_SECONDS
    )
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
        gc.collect()  # Native close already fenced; do not add a device synchronization.
        return dict(
            allocated_bytes=int(torch.cuda.memory_allocated(0)),
            reserved_bytes=int(torch.cuda.memory_reserved(0)),
        )

    hooks, warmup, cells, stop = [], [], [], None
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
        for arm in ARMS:
            _require(admit(started, 5), "budget_stop before warmup")
            measured.reuse, measured.records = arm == ARMS[2], []
            warm = dict(
                arm=arm,
                windows=measured.records,
                sample_count=32000,
                pcm_sha256=shared._sha(pcm[:64000]),
            )
            warmup.append(warm)
            session = runtime.Session("composed:warm:" + arm)
            tick = time.monotonic_ns()
            run = measured.start_window(
                session=session,
                request=runtime.RequestState(
                    session.session_id + ":request",
                    session.session_id,
                    identity,
                    rng_seed=7,
                ),
                window_id=session.session_id,
                mel=mel(pcm[:64000]),
                start_ms=0,
                end_ms=2000,
                options=options,
            )
            try:
                for _ in range(224):
                    if run.complete:
                        break
                    run.step()
                _require(run.complete, "warmup token bound reached")
                run.prepare_word_alignment()
            finally:
                run.close()
            del run
            measured.resolve()
            warm.update(
                elapsed_ns=time.monotonic_ns() - tick, capacity_restored=available()
            )
            _require(available(), "warmup retained capacity")
        for arm, config in configurations().items():
            if not admit(
                started, 46.55 + corpus.PACED_REPLAY["config"]["max_drain_ms"] / 1000
            ):
                stop = dict(reason="budget_stop", next_arm=arm)
                break
            measured.reuse, measured.records = arm == ARMS[2], []
            cell = dict(
                arm=arm,
                config=asdict(config),
                windows=measured.records,
                events=[],
                decision_traces=[],
                pacing=None,
                error=None,
                input_sha256=plan["pcm_sha256"],
                admitted_elapsed_ns=int((time.monotonic() - started) * 1e9),
                memory_before=memory(),
            )
            cells.append(cell)
            torch.cuda.reset_peak_memory_stats(0)
            stream, tick = None, time.monotonic_ns()
            try:
                _require(available(), "previous cell retained capacity")
                stream = adapters.ContinuousTranscriptStream(
                    measured,
                    stream_id="composed:" + arm,
                    mel_builder=mel,
                    options=options,
                    rng_seed=7,
                    config=config,
                )
                events, traces, pacing, error = corpus._drive_paced(stream, pcm)
                cell.update(
                    events=events,
                    decision_traces=traces,
                    pacing=pacing,
                    error=error,
                    metrics=b._plain(stream.metrics),
                    state=b._plain(stream.state),
                    profile_id=stream.profile_id,
                    done=stream.done,
                )
            except Exception as error:
                cell["error"] = b._safe_stream_error(error)
                stop = dict(reason="cell_error", arm=arm)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as error:
                        cell["cleanup_error"] = b._safe_stream_error(error)
                del stream
                cell["capacity_restored"] = available()
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
                    elapsed_ns=time.monotonic_ns() - tick,
                )
                cell["checks"] = cell_checks(cell, pcm)
                cell["stream_status"] = (
                    "completed" if all(cell["checks"].values()) else "failed"
                )
                cell["summary"] = summary(cell, plan)
                _require(
                    available() and cell.get("cleanup_error") is None,
                    "cell cleanup failed",
                )
            if stop is not None:
                break
    except Exception as error:
        stop = dict(reason="worker_error", error=b._safe_stream_error(error))
    finally:
        for hook in hooks:
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
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    return dict(
        schema_version="1-diagnostic",
        experiment_id="modal-composed-features-v1",
        qualified=False,
        claim_boundary=CLAIMS,
        source=dict(
            snapshot=expected_snapshot,
            registration_sha256=shared._sha((REMOTE_ROOT / PRODUCER).read_bytes()),
        ),
        scope=scope(),
        input=plan,
        backend=backend,
        warmup=warmup,
        cells=cells,
        stop=stop,
        native_window_count=measured.count,
        capacity_restored=available(),
        elapsed_ns=int((time.monotonic() - started) * 1e9),
        status="completed"
        if stop is None
        and len(cells) == 3
        and final == initial
        and all(c.get("stream_status") == "completed" for c in cells)
        else "partial",
        model=dict(
            initial_sha256=initial,
            final_sha256=final,
            checkpoint_sha256=setup.CHECKPOINT_SHA256,
            backend_revision=identity.revision,
            unchanged=initial == final,
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
            modal=str(modal.__version__),
        ),
    )


def validate_record(record, expected, root=ROOT):
    pcm, plan = input_plan(root)
    _require(
        record["source"]["snapshot"] == expected
        and record["scope"] == scope()
        and record["input"] == plan
        and record["claim_boundary"] == CLAIMS
        and not record["qualified"],
        "identity differs",
    )
    registered = _corpus().read_registration(root)
    runtime, adapters = (
        importlib.import_module(n)
        for n in ("whisper_runtime", "whisper_runtime.adapters")
    )
    profile = adapters.NativeExecutionProfile(
        "tiny.en/composed-v1",
        runtime.ResourceVector(
            memory_bytes=2147483648, compute_units=1, stream_slots=1
        ),
        device="cuda:0",
        reuse_alignment_features=True,
    )
    _require(
        record["effective_identity"]
        == dict(
            rng_seed=7,
            profile=asdict(profile),
            decode_options=asdict(
                adapters.NativeDecodeOptions(**registered["decode_options"])
            ),
        ),
        "effective options/profile differs",
    )
    _require(
        record["experiment_id"] == "modal-composed-features-v1"
        and record["model"]["initial_sha256"]
        == registered["model"]["model_state_sha256"]
        and record["model"]["checkpoint_sha256"]
        == registered["model"]["checkpoint_sha256"]
        and record["model"]["loads"] == 1
        and record["model"]["unchanged"]
        is (record["model"]["initial_sha256"] == record["model"]["final_sha256"])
        and record["model"]["backend_revision"] == features.PATCHED_TREE
        and all(
            record["backend"][k] == v
            for k, v in dict(
                base_commit=features.BASE_COMMIT,
                base_tree=features.BASE_TREE,
                patched_tree=features.PATCHED_TREE,
                patch_sha256=features.PATCH_SHA,
            ).items()
        ),
        "backend/model differs",
    )
    windows = []
    for group in (record["warmup"], record["cells"]):
        _require(
            [x["arm"] for x in group] == list(ARMS[: len(group)]) and len(group) <= 3,
            "schedule differs",
        )
        for cell in group:
            windows.extend(cell["windows"])
            for window in cell["windows"]:
                _require(
                    window["reuse_alignment_features"] is (cell["arm"] == ARMS[2]),
                    "alignment mode differs",
                )
                calls = window.get("forwards", [])
                _require(
                    sum(f["kind"] == "encoder" for f in calls)
                    == len(window["encoder_calls"]),
                    "encoder count differs",
                )
                _require(
                    0 <= window["start_ms"] < window["end_ms"] <= 46550
                    and window["end_ms"] - window["start_ms"] <= 30000,
                    "native window span differs",
                )
                _require(
                    not window["reuse_alignment_features"]
                    or all(f["phase"] == "decode" for f in window["encoder_calls"]),
                    "reuse repeated encoder",
                )
                for f in calls:
                    _require(
                        f["kind"] in {"encoder", "decoder"}
                        and f["phase"] in {"decode", "alignment"}
                        and (
                            f["interval_ms"] is None
                            or (
                                window["closed"]
                                and window["capacity_restored"]
                                and math.isfinite(f["interval_ms"])
                                and f["interval_ms"] >= 0
                            )
                        ),
                        "invalid forward timing",
                    )
    _require(
        record["native_window_count"] == len(windows) <= paced.MAX_NATIVE_WINDOWS
        and [w["call_index"] for w in windows] == list(range(1, len(windows) + 1)),
        "window bound differs",
    )
    _require(
        len({f["stream"] for w in windows for f in w.get("forwards", [])}) <= 1,
        "owned lane identity changed",
    )
    for warm in record["warmup"]:
        _require(
            warm["sample_count"] == 32000
            and warm["pcm_sha256"] == shared._sha(pcm[:64000]),
            "warmup input differs",
        )
    if record["cells"]:
        _require(
            len(record["warmup"]) == 3
            and all(w.get("capacity_restored") for w in record["warmup"]),
            "warmup incomplete",
        )
    for cell in record["cells"]:
        checks = cell_checks(cell, pcm)
        _require(
            cell["config"] == asdict(configurations()[cell["arm"]])
            and cell["summary"] == summary(cell, plan)
            and cell["checks"] == checks
            and cell["input_sha256"] == plan["pcm_sha256"]
            and cell["stream_status"]
            == ("completed" if all(checks.values()) else "failed"),
            "cell summary/checks differ",
        )
        _require(
            0
            <= cell["admitted_elapsed_ns"]
            < (GPU_TIMEOUT_SECONDS - RESERVE_SECONDS - 61.55) * 1e9,
            "admission crossed budget",
        )
        if cell["stream_status"] == "completed":
            _require(
                all(
                    f["interval_ms"] is not None
                    for w in cell["windows"]
                    for f in w.get("forwards", [])
                ),
                "completed timing unresolved",
            )
    complete = (
        record["stop"] is None
        and len(record["cells"]) == len(record["warmup"]) == 3
        and record["model"]["unchanged"]
        and record["capacity_restored"]
        and all(c["stream_status"] == "completed" for c in record["cells"])
    )
    _require(
        record["status"] == ("completed" if complete else "partial"),
        "completion verdict differs",
    )


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
    for item in expected["files"]:
        image = image.add_local_file(
            root / item["path"], (REMOTE_ROOT / item["path"]).as_posix(), copy=True
        )
    for f in corpus.read_registration(root)["fixtures"]:
        image = image.add_local_file(
            root / corpus.ASSET_PATH / f["filename"],
            (REMOTE_ROOT / corpus.ASSET_PATH / f["filename"]).as_posix(),
            copy=True,
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
    app = modal.App("wr-composed-" + uuid.uuid4().hex[:12])
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
        producer = importlib.import_module("infra.modal_composed_features")
        return producer._corpus().b._encode_worker_record(producer.run_worker(expected))

    return app, echo, execute


def run(*, replay_id, preflight=False, confirm_paid_gpu=False, root=ROOT):
    output, receipt, raw = shared._paths(root, replay_id, False)
    frozen = output.parent / "composed-preflight.json"
    _require(preflight or confirm_paid_gpu is True, "requires --confirm-paid-gpu")
    _require(
        not any(p.exists() for p in (output, receipt, raw)),
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
        "preflight no longer matches frozen bytes/plan",
    )
    live.prior._journal(
        receipt, "attempt-started", first=True, source_digest=expected["digest"]
    )
    try:
        app, echo, execute = resources(expected, root)
        with app.run(detach=False):
            probe = _corpus().b._transport_probe_payload()
            _require(echo.remote(probe) == probe, "CPU transport preflight failed")
            payload = execute.remote()  # Exactly one paid call; never retry.
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
        live.prior._journal(receipt, "attempt-finished", status=record["status"])
        return output
    except BaseException as error:
        live.prior._journal(receipt, "attempt-failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(run(**vars(parser.parse_args())))
