"""Bounded native draft screen on new public audio. Preflight is local only."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from infra import modal_integrated_draft as previous

ROOT, REMOTE_ROOT = previous.ROOT, previous.REMOTE_ROOT
shared, prior, composed = previous.shared, previous.prior, previous.composed
PRODUCER = "infra/modal_draft_holdout.py"
TEST = "tools/test_modal_draft_holdout.py"
PLAN = "docs/research/2026-09-07-draft-holdout-plan.md"
MANIFEST = "experiments/draft-holdout-20260907.json"
ASSETS = "artifacts/draft-holdout-20260907"
TIMEOUT, RESERVE = 300, 20
SCHEDULE = previous.SCHEDULE
CLAIMS = dict(previous.CLAIMS, held_out_population_accuracy=False)
require = previous._require


def input_plan(root=ROOT):
    manifest = json.loads((root / MANIFEST).read_bytes())
    require(len(manifest["fixtures"]) == 2, "two registered fixtures required")
    parts, spans, head = [], [], 0
    for fixture in manifest["fixtures"]:
        name = fixture["filename"]
        require(
            Path(name).name == name and "\\" not in name and name.endswith(".pcm"),
            "unsafe fixture name",
        )
        raw = (root / ASSETS / name).read_bytes()
        require(
            len(raw) == fixture["sample_count"] * 2
            and shared._sha(raw) == fixture["pcm_sha256"],
            "fixture differs",
        )
        require(64000 <= fixture["sample_count"] <= 320000, "clip outside 4–20 seconds")
        spans.append(
            dict(
                id=fixture["id"],
                start_sample=head,
                end_sample=head + fixture["sample_count"],
            )
        )
        parts.extend((raw, bytes(64000)))
        head += fixture["sample_count"] + 32000
    pcm = b"".join(parts)
    require(len(pcm) <= 45 * 32000, "input exceeds 45 seconds")
    return pcm, dict(
        sample_count=len(pcm) // 2,
        duration_seconds=len(pcm) / 32000,
        pcm_sha256=shared._sha(pcm),
        spans=spans,
        reference_text=" ".join(f["reference_text"] for f in manifest["fixtures"]),
        manifest_sha256=shared._sha((root / MANIFEST).read_bytes()),
        recipe="Two unchanged clips; two seconds digital silence after each",
    )


def snapshot(root=ROOT):
    inherited = previous.snapshot(root)
    names = {f["path"] for f in inherited["files"]} | {PRODUCER, TEST, PLAN, MANIFEST}
    files = [
        dict(path=n, size_bytes=len(raw), sha256=shared._sha(raw))
        for n in sorted(names)
        for raw in [(root / n).read_bytes()]
    ]
    return dict(commit=inherited["commit"], files=files, digest=shared._hash(files))


def scope():
    rate = sum(
        (
            previous.live.PRICES["t4_per_second"],
            2 * previous.live.PRICES["cpu_core_per_second"],
            4 * previous.live.PRICES["gib_per_second"],
        )
    )
    return dict(
        gpu="T4",
        gpu_calls=1,
        timeout_seconds=TIMEOUT,
        reserve_seconds=RESERVE,
        maximum_native_windows=previous.MAX_NATIVE_WINDOWS,
        retries=0,
        min_containers=0,
        max_containers=1,
        source_paced=True,
        configurations={a: asdict(c) for a, c in previous.configurations().items()},
        schedule=[list(s) for s in SCHEDULE],
        model_download=False,
        planning_compute_usd=round(TIMEOUT * rate * 1.75, 6),
        cost_exclusions="startup/build/storage/network/taxes/provider rescheduling",
        physical_spending_cap=False,
        prices_source="https://modal.com/pricing",
        prices_checked="2026-09-07",
    )


def admitted(started, duration):
    return time.monotonic() - started + duration + RESERVE < TIMEOUT


def comparison(cells):
    result = previous.analyze(cells)
    result["scope"] = (
        "One ABBA screen on two project-new speakers; not population or production qualification"
    )
    return result


def run_worker(expected):
    started = time.monotonic()
    shared.verify_snapshot(expected)
    corpus = previous._corpus()
    b = corpus.b
    backend = previous.features.apply_backend_patch(REMOTE_ROOT, b.BACKEND_ROOT)
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
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and torch.cuda.get_device_name(0) == "Tesla T4",
        "registered T4 required",
    )
    setup.validate_dependencies()
    setup.validate_checkpoint(b.MODEL_CHECKPOINT_PATH)
    torch.set_num_threads(1)
    model = setup._load_model(whisper, b.MODEL_CHECKPOINT_PATH, "cuda:0")
    initial = setup._fingerprint(model)
    require(initial == setup.MODEL_FINGERPRINT and not model.training, "model differs")
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        "tiny.en", previous.features.PATCHED_TREE, "native-draft-holdout", initial
    )
    worker = runtime.Worker(
        "draft-holdout", identity, budget, queue_capacity=1, transaction_ttl_seconds=180
    )

    def probe(observed):
        require(observed is model, "model binding changed")
        return identity

    native = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/draft-holdout-v1",
            capacity,
            device="cuda:0",
            reuse_alignment_features=True,
        ),
    )
    measured = previous.Measured(native, torch.cuda, started + TIMEOUT - RESERVE)
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

    before = [
        (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in model.modules()
    ]
    hooks, warmup, cells, stop = [], [], [], None
    try:
        for kind in ("encoder", "decoder"):
            module = getattr(model, kind)
            hooks.extend(
                (
                    module.register_forward_pre_hook(
                        lambda m, a, kw, k=kind: measured.before(k, m, a, kw),
                        with_kwargs=True,
                    ),
                    module.register_forward_hook(
                        lambda m, a, kw, out, k=kind: measured.after(k, m, a, kw, out),
                        with_kwargs=True,
                        always_call=True,
                    ),
                )
            )
        for arm in prior.ARMS:
            require(admitted(started, 10), "no warmup budget")
            measured.records, hint = [], ()
            warm = dict(arm=arm, windows=measured.records)
            warmup.append(warm)
            for index in range(2):
                session = runtime.Session(f"warm:{arm}:{index}")
                run = measured.start_window(
                    session=session,
                    request=runtime.RequestState(
                        session.session_id + ":r",
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
                    require(run.complete, "warmup token bound")
                    hint = tuple(run.prepare_result().metadata.tokens[:32])
                    run.prepare_word_alignment()
                finally:
                    run.close()
                del run
            measured.resolve()
            warm["capacity_restored"] = available()
            require(available(), "warmup leaked capacity")
        require(
            warmup[1]["windows"][1]["decode_observation"]["stats"] is not None,
            "warmup did not activate draft",
        )
        for position, arm in SCHEDULE:
            if not admitted(started, plan["duration_seconds"] + 15):
                stop = dict(reason="budget_stop", position=position)
                break
            require(available(), "previous stream retained capacity")
            measured.records = []
            cell = dict(
                position=position,
                arm=arm,
                config=asdict(previous.configurations()[arm]),
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
                stream = adapters.ContinuousTranscriptStream(
                    measured,
                    stream_id="draft-holdout",
                    mel_builder=mel,
                    options=options,
                    rng_seed=7,
                    config=previous.configurations()[arm],
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
                if available() and not cell.get("cleanup_error"):
                    try:
                        measured.resolve()
                    except Exception as error:
                        cell["cleanup_error"] = b._safe_stream_error(error)
                cell.update(
                    capacity_restored=available(),
                    memory_after_close=memory(),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
                    draft_observations=[
                        w["decode_observation"]
                        for w in measured.records
                        if "decode_observation" in w
                    ],
                )
                cell["checks"] = dict(
                    composed.cell_checks(cell, pcm),
                    hint_cleared=cell.get("hint_cleared") is True,
                )
                cell["stream_status"] = (
                    "completed" if all(cell["checks"].values()) else "failed"
                )
                cell["summary"] = prior.summary(cell, plan)
            # Retain a normal blocked stream and run its matched arm; never retry it.
            # Infrastructure or cleanup failures stop the worker immediately.
            error_type = (cell.get("error") or {}).get("error_class")
            if (
                not available()
                or cell.get("cleanup_error")
                or (cell["error"] and error_type != "StreamNeedsResolutionError")
            ):
                stop = dict(reason="cell_error", position=position)
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
                reason="model_verification_error", error=b._safe_stream_error(error)
            )
    hooks_unchanged = before == [
        (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in model.modules()
    ]
    complete = (
        stop is None
        and len(cells) == 4
        and available()
        and final == initial
        and hooks_unchanged
    )
    call_id = importlib.import_module("modal").current_function_call_id()
    return dict(
        schema_version="1-diagnostic",
        experiment_id="modal-draft-holdout-v1",
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
        ),
        status="completed" if complete else "partial",
        comparison=comparison(cells)
        if complete
        else dict(complete=False, accepted=False),
    )


def validate_record(record, expected, root=ROOT):
    pcm, plan = input_plan(root)
    require(
        record["experiment_id"] == "modal-draft-holdout-v1"
        and record["source"]["snapshot"] == expected
        and record["scope"] == scope()
        and record["input"] == plan
        and record["claim_boundary"] == CLAIMS
        and record["qualified"] is False,
        "record identity differs",
    )
    require(record["status"] in ("completed", "partial"), "unknown status")
    require(
        record["backend"]["patched_tree"] == previous.features.PATCHED_TREE,
        "backend differs",
    )
    cells = record["cells"]
    require(
        [(c["position"], c["arm"]) for c in cells] == list(SCHEDULE[: len(cells)])
        and len(cells) <= 4,
        "schedule differs",
    )
    windows = [w for c in record["warmup"] + cells for w in c["windows"]]
    require(
        len(windows) == record["native_window_count"] <= previous.MAX_NATIVE_WINDOWS
        and [w["call_index"] for w in windows] == list(range(1, len(windows) + 1)),
        "native accounting differs",
    )
    for window in windows:
        require(
            0
            <= window["start_ms"]
            < window["end_ms"]
            <= plan["duration_seconds"] * 1000
            and window["end_ms"] - window["start_ms"] <= 30000,
            "window span differs",
        )
        for forward in window.get("forwards", []):
            value = forward["interval_ms"]
            require(
                value is None
                or (
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                ),
                "invalid forward interval",
            )
            if forward["kind"] == "decoder":
                require(
                    type(forward["input_tokens"]) is int
                    and forward["input_tokens"] > 0,
                    "invalid decoder input count",
                )
        observation = window.get("decode_observation")
        if observation:
            require(
                (observation["stats"] is not None) == bool(window["draft_tokens"]),
                "native draft activation differs",
            )
            if observation["stats"]:
                require(
                    observation["stats"]["proposed_tokens"]
                    == len(window["draft_tokens"])
                    and observation["stats"]["parallel_prefills"] == 1,
                    "draft work differs",
                )
    for c in cells:
        require(
            c["config"] == asdict(previous.configurations()[c["arm"]])
            and c["input_sha256"] == plan["pcm_sha256"],
            "cell config/input differs",
        )
        require(
            c["checks"]
            == dict(
                composed.cell_checks(c, pcm), hint_cleared=c.get("hint_cleared") is True
            )
            and c["summary"] == prior.summary(c, plan),
            "cell summary differs",
        )
        require(
            c["stream_status"]
            == ("completed" if all(c["checks"].values()) else "failed"),
            "stream status differs",
        )
        require(
            c["draft_observations"]
            == [
                w["decode_observation"]
                for w in c["windows"]
                if "decode_observation" in w
            ],
            "observation differs",
        )
        if c["arm"] == prior.ARMS[0]:
            require(
                not any(w["draft_tokens"] for w in c["windows"]), "control used draft"
            )
    if record["status"] == "completed":
        require(
            record["stop"] is None
            and len(cells) == 4
            and record["capacity_restored"]
            and record["hooks_unchanged"]
            and record["model"]["initial_sha256"]
            == record["model"]["final_sha256"]
            == importlib.import_module(
                "whisper_runtime.native_setup"
            ).MODEL_FINGERPRINT,
            "invalid completion",
        )
        require(
            len(record["warmup"]) == 2
            and all(
                len(c["windows"]) == 2 and c["capacity_restored"]
                for c in record["warmup"]
            ),
            "invalid warmup",
        )
        require(
            record["warmup"][1]["windows"][1]["decode_observation"]["stats"]
            is not None,
            "candidate warmup lacks draft",
        )
        require(
            all(
                w["closed"]
                and w["capacity_restored"]
                and w.get("native_cleanup")
                and all(w["native_cleanup"].values())
                and all(f["interval_ms"] is not None for f in w.get("forwards", []))
                for w in windows
            ),
            "native cleanup/timing differs",
        )
    require(
        record["comparison"]
        == (
            comparison(cells)
            if record["status"] == "completed"
            else dict(complete=False, accepted=False)
        ),
        "comparison differs",
    )


def resources(expected, root=ROOT):
    modal = importlib.import_module("modal")
    require(str(modal.__version__) == "1.5.5", "registered Modal SDK required")
    corpus = previous._corpus()
    q = corpus._helper("infra.modal_native_cuda_qualification")
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("ca-certificates", "ffmpeg", "git")
        .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
        .uv_pip_install(*q.DIRECT_IMAGE_PACKAGES)
        .run_commands(q._build_command(corpus.BASE_COMMIT))
        .env(
            dict(
                PYTHONPATH="/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONNOUSERSITE="1",
                WHISPER_MODAL_ENABLE_WORD_CORPUS="0",
                WHISPER_MODAL_ENABLE_REMOTE_RESOURCES="0",
            )
        )
    )
    for item in expected["files"]:
        image = image.add_local_file(
            root / item["path"], (REMOTE_ROOT / item["path"]).as_posix(), copy=False
        )
    for f in json.loads((root / MANIFEST).read_bytes())["fixtures"]:
        image = image.add_local_file(
            root / ASSETS / f["filename"],
            (REMOTE_ROOT / ASSETS / f["filename"]).as_posix(),
            copy=False,
        )
    app = modal.App("wr-draft-holdout-" + uuid.uuid4().hex[:12])
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
        timeout=TIMEOUT,
        volumes={corpus.b.MODEL_CACHE_MOUNT: volume.with_mount_options(read_only=True)},
    )
    def execute():
        module = importlib.import_module("infra.modal_draft_holdout")
        return module.previous._corpus().b._encode_worker_record(
            module.run_worker(expected)
        )

    return app, echo, execute


def run(*, replay_id, preflight=False, confirm_paid_gpu=False, root=ROOT):
    output, journal, raw = shared._paths(root, replay_id, False)
    frozen = output.parent / "draft-holdout-preflight.json"
    require(preflight or confirm_paid_gpu is True, "requires --confirm-paid-gpu")
    require(
        not any(p.exists() for p in (output, journal, raw)),
        "attempt exists; no overwrite/retry",
    )
    expected, (_, plan) = snapshot(root), input_plan(root)
    spec = dict(source=expected, input=plan, scope=scope())
    if preflight:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        previous.live.prior._write_result(frozen, spec)
        return frozen
    require(json.loads(frozen.read_bytes()) == spec, "frozen source or plan changed")
    previous.live.prior._journal(
        journal, "attempt-started", first=True, source_digest=expected["digest"]
    )
    try:
        app, echo, execute = resources(expected, root)
        with app.run(detach=False):
            b = previous._corpus().b
            payload = b._transport_probe_payload()
            require(echo.remote(payload) == payload, "CPU transport preflight failed")
            payload = execute.remote()
            b._write_bytes_exclusive(raw, payload)
            record = b._decode_worker_record(
                payload,
                expected_snapshot=expected,
                registration_sha256=shared._sha((root / PRODUCER).read_bytes()),
                manifest=dict(claim_boundary=CLAIMS),
            )
            validate_record(record, expected, root)
            record["app_id"] = app.app_id
        record["ephemeral_app_exit_completed"] = True
        previous.live.prior._write_result(output, record)
        previous.live.prior._journal(
            journal, "attempt-finished", status=record["status"]
        )
        return output
    except BaseException as error:
        previous.live.prior._journal(
            journal, "attempt-failed", error_type=type(error).__name__
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(run(**vars(parser.parse_args())))
