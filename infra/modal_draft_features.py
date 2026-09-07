"""Bounded, paired T4 measurement of untrusted decoder drafts.

Import and preflight are local. One explicit paid call runs both arms on one
model and lane. All backend, alignment and publication decisions stay native.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict, replace
from unittest.mock import patch

from infra import modal_composed_features as composed
from tools.verify_decoder_draft import result_record
from tools.verify_stream_draft import StreamDraftInference

shared, features, live, paced = (
    composed.shared,
    composed.features,
    composed.live,
    composed.paced,
)
ROOT, REMOTE_ROOT = composed.ROOT, composed.REMOTE_ROOT
PRODUCER = "infra/modal_draft_features.py"
TEST = "tools/test_modal_draft_features.py"
PREREGISTRATION = "docs/research/2026-09-07-draft-gpu-plan.md"
GPU_TIMEOUT_SECONDS, RESERVE_SECONDS = 240, 20
ARMS = ("fast-reuse", "fast-draft32")
MAX_DRAFT_TOKENS, MAX_NATIVE_WINDOWS = 32, 128
CLAIMS = dict.fromkeys(
    (
        "production_readiness",
        "general_speedup",
        "gpu_occupancy",
        "energy",
        "word_latency",
        "cross_window_feature_reuse",
        "numerical_identity",
    ),
    False,
)
_require, _corpus, input_plan = composed._require, composed._corpus, composed.input_plan


def configurations():
    cli = importlib.import_module("whisper_runtime.native_setup").CLI_STREAM_CONFIG
    config = replace(cli, left_context_ms=20000, word_context_limit_ms=24000)
    return dict.fromkeys(ARMS, config)


def scope():
    rate = sum(
        (
            live.PRICES["t4_per_second"],
            2 * live.PRICES["cpu_core_per_second"],
            4 * live.PRICES["gib_per_second"],
        )
    )
    estimate = GPU_TIMEOUT_SECONDS * rate * live.PRICES["regional_multiplier_ceiling"]
    _require(
        0 < GPU_TIMEOUT_SECONDS <= 240 and RESERVE_SECONDS == 20 and estimate < 1,
        "budget exceeds the registered plan",
    )
    return dict(
        gpu="T4",
        gpu_calls=1,
        timeout_seconds=GPU_TIMEOUT_SECONDS,
        reserve_seconds=RESERVE_SECONDS,
        min_containers=0,
        max_containers=1,
        retries=0,
        planning_compute_usd=round(estimate, 6),
        physical_spending_cap=False,
        cost_exclusions="build/startup/storage/network/taxes/provider rescheduling",
        prices=live.PRICES,
        prices_checked="2026-09-07",
        prices_source="https://modal.com/pricing",
        same_model_and_lane=True,
        model_download=False,
        inference_owners=1,
        maximum_native_windows=MAX_NATIVE_WINDOWS,
        warmup_arms=list(ARMS),
        warmup_samples=128000,
        warmup_windows_per_arm=2,
        cancellation_probe_windows=1,
        configurations={a: asdict(c) for a, c in configurations().items()},
        alignment_reuse=dict.fromkeys(ARMS, True),
        draft_tokens=dict(zip(ARMS, (0, 32))),
        draft_state_reset="before and after each warmup arm, cleanup probe and measured cell",
        current_audio_prefill=True,
        prior_audio_or_kv_reused=False,
        pacing=_corpus().PACED_REPLAY,
        timing="CUDA-event forward intervals; not occupancy/energy/FLOPs",
        host_timings_include_instrumentation=True,
    )


def snapshot(root=ROOT):
    inherited = composed.snapshot(root)
    names = {item["path"] for item in inherited["files"]}
    names.update(
        (
            PRODUCER,
            TEST,
            PREREGISTRATION,
            "tools/verify_decoder_draft.py",
            "tools/verify_stream_draft.py",
        )
    )
    files = [
        dict(path=name, size_bytes=len(raw), sha256=shared._sha(raw))
        for name in sorted(names)
        for raw in [(root / name).read_bytes()]
    ]
    return dict(commit=inherited["commit"], files=files, digest=shared._hash(files))


def admit(started, seconds):
    return time.monotonic() - started + seconds + RESERVE_SECONDS < GPU_TIMEOUT_SECONDS


class Measured(composed.Measured):
    """Count submitted token positions as well as module invocations."""

    def start_window(self, **kwargs):
        _require(self.count < MAX_NATIVE_WINDOWS, "native window bound reached")
        return super().start_window(**kwargs)

    def before(self, kind, module, args, kwargs):
        super().before(kind, module, args, kwargs)
        if kind == "decoder":
            value = args[0] if args else kwargs["x"]
            _require(len(value.shape) == 2, "decoder token input must have two axes")
            self.active[0]["forwards"][-1]["input_tokens"] = int(
                value.shape[0] * value.shape[1]
            )


class DraftHook:
    """One isolated worker hook; retain token IDs, never old model state."""

    def __init__(self, measured):
        self.measured = measured
        self.enabled = False
        self.draft = ()
        self.observations = []
        self.last_inference = None
        self.stack = ExitStack()

    def reset(self, enabled):
        self.enabled, self.draft, self.observations = enabled, (), []
        self.last_inference = None

    def __enter__(self):
        decoding = importlib.import_module("whisper.decoding")
        start, finalize = (
            decoding.DecodingTask._start_run,
            decoding._DecodingRun.finalize,
        )

        def start_run(task, mel):
            options = task.options
            _require(
                options.temperature == 0
                and options.beam_size is None
                and options.best_of is None
                and options.language == "en"
                and not options.fp16
                and not options.without_timestamps,
                "draft diagnostic requires registered greedy FP32 options",
            )
            run = start(task, mel)
            try:
                if self.enabled and self.draft:
                    run.inference = StreamDraftInference(run.inference, self.draft)
                self.last_inference = run.inference
            except BaseException:
                run.cleanup()
                raise
            return run

        def finish_run(run):
            results = finalize(run)
            _require(
                len(results) == 1 and self.measured.active is not None,
                "result escaped the measured single-sequence run",
            )
            wrapped = run.inference
            is_draft = isinstance(wrapped, StreamDraftInference)
            original = wrapped.original if is_draft else wrapped
            _require(
                not original.kv_cache and not original.hooks,
                "decoder retained cache or hooks after finalization",
            )
            record = self.measured.active[0]
            observation = dict(
                call_index=record["call_index"],
                start_ms=record["start_ms"],
                end_ms=record["end_ms"],
                token_steps=run.step_index,
                result=result_record(results[0]),
                stats=dict(wrapped.stats) if is_draft else None,
                cache_empty=True,
            )
            self.observations.append(observation)
            record["decode_observation"] = observation
            self.draft = (
                tuple(results[0].tokens[:MAX_DRAFT_TOKENS]) if self.enabled else ()
            )
            return results

        self.stack.enter_context(
            patch.object(decoding.DecodingTask, "_start_run", start_run)
        )
        self.stack.enter_context(
            patch.object(decoding._DecodingRun, "finalize", finish_run)
        )
        return self

    def __exit__(self, *exception):
        self.reset(False)
        return self.stack.__exit__(*exception)


def drive_paced(stream, pcm):
    module = importlib.import_module("whisper_runtime.adapters.paced_replay")
    events, traces, event_times, trace_times, publication_inputs = [], [], [], [], []
    plain = _corpus().b._plain

    def event(event, elapsed):
        record = plain(event)
        events.append(record)
        event_times.append(elapsed)
        if record["kind"] == "commit":
            publication_inputs.append(
                dict(
                    start_sample=record["start_sample"],
                    end_sample=record["end_sample"],
                    accepted_samples=stream.metrics.accepted_samples,
                )
            )

    def trace(trace, elapsed):
        traces.append(plain(trace))
        trace_times.append(elapsed)

    result = module.drive_paced(
        stream,
        pcm,
        config=module.PacedReplayConfig(**_corpus().PACED_REPLAY["config"]),
        on_event=event,
        on_trace=trace,
    )
    pacing = dict(_corpus().PACED_REPLAY)
    pacing.update(
        {
            name: plain(getattr(result, name))
            for name in (
                "status",
                "elapsed_ns",
                "input_finished_ns",
                "offered_samples",
                "accepted_samples",
                "driver_steps",
                "max_source_lag_ns",
                "max_admission_lag_ns",
                "admissions",
                "undelivered_events",
                "undelivered_event_ns",
            )
        }
    )
    pacing.update(
        event_elapsed_ns=event_times,
        trace_elapsed_ns=trace_times,
        publication_inputs=publication_inputs,
        latency_scope="same worker paced input; excludes loading and network",
    )
    pacing["output_lags"] = _corpus()._paced_output_lags(events, traces, pacing)
    pacing.update(_corpus()._paced_milestones(events, pacing))
    error = _corpus().b._safe_stream_error(result.error) if result.error else None
    return events, traces, pacing, error


def summary(cell, plan):
    result = composed.summary(cell, plan)
    result["decoder_input_tokens"] = sum(
        f["input_tokens"]
        for w in cell["windows"]
        for f in w.get("forwards", [])
        if f["kind"] == "decoder"
    )
    result["draft_matched_tokens"] = sum(
        o["stats"]["matched_tokens"] for o in cell["draft_observations"] if o["stats"]
    )
    phases = {p for w in cell["windows"] for p in w["operation_wall_ns"]}
    result["operation_wall_ns"] = {
        p: sum(w["operation_wall_ns"].get(p, 0) for w in cell["windows"])
        for p in sorted(phases)
    }
    result["total_operation_wall_ns"] = sum(result["operation_wall_ns"].values())
    return result


def comparison(cells):
    if len(cells) != 2:
        return dict(paired=False, accepted=False)
    control, candidate = cells
    _require([c["arm"] for c in cells] == list(ARMS), "paired arm order differs")
    observations = [c["draft_observations"] for c in cells]

    def spans(group):
        return [(o["start_ms"], o["end_ms"]) for o in group]

    def reasons(cell):
        records = [
            {
                k: t.get(k)
                for k in (
                    "analysis_start_sample",
                    "analysis_end_sample",
                    "action",
                    "reason",
                    "committed_before_sample",
                    "retained_from_sample",
                    "publication_span",
                    "eof",
                )
            }
            for t in cell["decision_traces"]
        ]
        for record, trace in zip(records, cell["decision_traces"], strict=True):
            evidence = trace.get("audio_evidence") or {}
            record["audio_evidence"] = {
                k: evidence.get(k) for k in ("state", "reason", "observation")
            }
            word = trace.get("word_publication") or {}
            record["word_publication"] = {
                k: word.get(k)
                for k in (
                    "start_ms",
                    "end_ms",
                    "text",
                    "word_start",
                    "word_end",
                    "boundary_tolerance_ms",
                    "final",
                )
            }
            silence = trace.get("silence_publication")
            record["silence_publication"] = (
                None
                if silence is None
                else {
                    k: silence.get(k)
                    for k in (
                        "analysis_span",
                        "analysis_start_sample",
                        "start_ms",
                        "end_ms",
                        "text",
                        "observation",
                    )
                }
            )
        return records

    def commit_sources(cell):
        return [
            t["analysis_end_sample"]
            for t in cell["decision_traces"]
            if t["action"] == "commit"
        ]

    exact = dict(
        commit_spans_and_text=control["summary"]["commits"]
        == candidate["summary"]["commits"],
        word_errors=control["summary"]["word_errors"]
        == candidate["summary"]["word_errors"],
        window_spans=spans(observations[0]) == spans(observations[1]),
        native_window_spans=spans(control["windows"]) == spans(candidate["windows"]),
        window_tokens=[o["result"]["tokens"] for o in observations[0]]
        == [o["result"]["tokens"] for o in observations[1]],
        decision_reasons=reasons(control) == reasons(candidate),
        commit_source_endpoints=commit_sources(control) == commit_sources(candidate),
        checks_pass=all(c["checks"] and all(c["checks"].values()) for c in cells),
    )
    numerical = []
    if exact["window_spans"]:
        for before, after in zip(*observations, strict=True):
            numerical.append(
                dict(
                    start_ms=before["start_ms"],
                    end_ms=before["end_ms"],
                    deltas={
                        k: after["result"][k] - before["result"][k]
                        for k in ("avg_logprob", "no_speech_prob", "compression_ratio")
                    },
                )
            )
    before_time = sum(control["summary"]["forward_interval_ms"].values())
    after_time = sum(candidate["summary"]["forward_interval_ms"].values())
    faster = dict(
        forward_intervals=after_time < before_time,
        inclusive_native_operations=candidate["summary"]["total_operation_wall_ns"]
        < control["summary"]["total_operation_wall_ns"],
    )
    return dict(
        paired=True,
        accepted=all(exact.values()),
        exact=exact,
        numeric_deltas=numerical,
        decision_comparison_scope="source spans, action/reason, retained/committed cursors, audio disposition, publication and word boundaries; raw scored traces retained separately",
        measured_efficiency=faster,
        measured_efficiency_gate=all(exact.values()) and all(faster.values()),
        numerical_identity_qualified=False,
        publication_inputs={c["arm"]: c["pacing"]["publication_inputs"] for c in cells},
        publication_input_positions_exact=control["pacing"]["publication_inputs"]
        == candidate["pacing"]["publication_inputs"],
        publication_clock_boundary="admission positions may differ with measured execution time",
        decoder_forwards_saved=control["summary"]["forwards"]["decoder"]
        - candidate["summary"]["forwards"]["decoder"],
        decoder_input_tokens_delta=candidate["summary"]["decoder_input_tokens"]
        - control["summary"]["decoder_input_tokens"],
        total_operation_wall_ns_delta=candidate["summary"]["total_operation_wall_ns"]
        - control["summary"]["total_operation_wall_ns"],
        operation_wall_ns={c["arm"]: c["summary"]["operation_wall_ns"] for c in cells},
        forward_interval_ms_delta={
            k: candidate["summary"]["forward_interval_ms"][k]
            - control["summary"]["forward_interval_ms"][k]
            for k in ("encoder", "decoder")
        },
    )


def run_worker(expected_snapshot):
    started = time.monotonic()
    shared.verify_snapshot(expected_snapshot)
    corpus, b = _corpus(), _corpus().b
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
    _require(
        all(
            str(v.device) == "cuda:0"
            and (not v.is_floating_point() or v.dtype == torch.float32)
            for v in (*model.parameters(), *model.buffers())
        ),
        "model precision/device differs",
    )
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        "tiny.en", features.PATCHED_TREE, "pytorch-cuda-draft", initial
    )
    worker = runtime.Worker(
        "draft", identity, budget, queue_capacity=1, transaction_ttl_seconds=180
    )

    def probe(observed):
        _require(observed is model, "model binding changed")
        return identity

    native = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/draft-v1", capacity, device="cuda:0", reuse_alignment_features=True
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
        with DraftHook(measured) as draft:
            for arm in ARMS:
                _require(admit(started, 10), "budget_stop before warmup")
                measured.records = []
                draft.reset(arm == ARMS[1])
                warm = dict(
                    arm=arm,
                    windows=measured.records,
                    draft_observations=draft.observations,
                    sample_count=128000,
                    pcm_sha256=shared._sha(pcm[:256000]),
                )
                warmup.append(warm)
                for index in range(2):
                    session = runtime.Session(f"draft:warm:{arm}:{index}")
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
                warm["capacity_restored"] = available()
                probe_draft = draft.draft
                draft.reset(False)
                _require(available(), "warmup retained capacity")
            _require(
                admit(started, 5) and len(probe_draft) == 32,
                "cancellation probe unavailable",
            )
            measured.records = []
            draft.reset(True)
            draft.draft = probe_draft
            probe_record = dict(
                kind="cancel_after_speculative_prefill", windows=measured.records
            )
            probes.append(probe_record)
            session = runtime.Session("draft:cancel-probe")
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
            )
            wrapped = draft.last_inference
            try:
                _require(
                    isinstance(wrapped, StreamDraftInference)
                    and wrapped.rows is not None,
                    "probe did not execute speculative prefill",
                )
                probe_record["cancellation_changed_state"] = measured.invoke(
                    run.record, "cancel", run.raw.cancel
                )
            finally:
                run.close()
            del run
            measured.resolve()
            probe_record.update(
                capacity_restored=available(),
                rows_cleared=wrapped.rows is None,
                features_cleared=wrapped.audio_features is None,
                cache_empty=not wrapped.original.kv_cache,
                hooks_empty=not wrapped.original.hooks,
                stats=dict(wrapped.stats),
                published_events=0,
            )
            _require(
                all(
                    probe_record[k]
                    for k in (
                        "cancellation_changed_state",
                        "capacity_restored",
                        "rows_cleared",
                        "features_cleared",
                        "cache_empty",
                        "hooks_empty",
                    )
                ),
                "cancellation cleanup failed",
            )
            draft.reset(False)
            for arm, config in configurations().items():
                if not admit(
                    started,
                    46.55 + corpus.PACED_REPLAY["config"]["max_drain_ms"] / 1000,
                ):
                    stop = dict(reason="budget_stop", next_arm=arm)
                    break
                measured.records = []
                draft.reset(arm == ARMS[1])
                cell = dict(
                    arm=arm,
                    config=asdict(config),
                    windows=measured.records,
                    draft_observations=draft.observations,
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
                stream, tick = None, time.monotonic_ns()
                try:
                    _require(available(), "previous cell retained capacity")
                    stream = adapters.ContinuousTranscriptStream(
                        measured,
                        stream_id="draft:" + arm,
                        mel_builder=mel,
                        options=options,
                        rng_seed=7,
                        config=config,
                    )
                    events, traces, pacing, error = drive_paced(stream, pcm)
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
                    cell["checks"] = composed.cell_checks(cell, pcm)
                    cell["stream_status"] = (
                        "completed" if all(cell["checks"].values()) else "failed"
                    )
                    cell["summary"] = summary(cell, plan)
                    draft.reset(False)
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
    hooks_unchanged = before_hooks == [
        (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in model.modules()
    ]
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
    record = dict(
        schema_version="1-diagnostic",
        experiment_id="modal-draft-features-v1",
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
    record["status"] = (
        "completed"
        if (
            stop is None
            and len(cells) == 2
            and final == initial
            and hooks_unchanged
            and all(c["stream_status"] == "completed" for c in cells)
        )
        else "partial"
    )
    record["comparison"] = (
        comparison(cells)
        if record["status"] == "completed"
        else dict(paired=False, accepted=False)
    )
    return record


def validate_record(record, expected, root=ROOT):
    pcm, plan = input_plan(root)
    _require(
        record["schema_version"] == "1-diagnostic"
        and record["experiment_id"] == "modal-draft-features-v1"
        and record["source"]["snapshot"] == expected
        and record["scope"] == scope()
        and record["input"] == plan
        and record["claim_boundary"] == CLAIMS
        and record["qualified"] is False,
        "experiment identity differs",
    )
    registered = _corpus().read_registration(root)
    runtime, adapters = (
        importlib.import_module(n)
        for n in ("whisper_runtime", "whisper_runtime.adapters")
    )
    profile = adapters.NativeExecutionProfile(
        "tiny.en/draft-v1",
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
        record["model"]["initial_sha256"] == registered["model"]["model_state_sha256"]
        and record["model"]["checkpoint_sha256"]
        == registered["model"]["checkpoint_sha256"]
        and record["model"]["loads"] == 1
        and record["model"]["backend_revision"] == features.PATCHED_TREE
        and all(
            record["backend"][k] == v
            for k, v in dict(
                base_commit=features.BASE_COMMIT,
                base_tree=features.BASE_TREE,
                patched_tree=features.PATCHED_TREE,
                patch_sha256=features.PATCH_SHA,
            ).items()
        )
        and record["model"]["unchanged"]
        is (record["model"]["initial_sha256"] == record["model"]["final_sha256"]),
        "model identity differs",
    )
    windows = []
    for group in (record["warmup"], record["cells"]):
        _require(
            [c["arm"] for c in group] == list(ARMS[: len(group)]) and len(group) <= 2,
            "schedule differs",
        )
        for cell in group:
            windows.extend(cell["windows"])
            observations = [
                w["decode_observation"]
                for w in cell["windows"]
                if "decode_observation" in w
            ]
            _require(
                observations == cell["draft_observations"], "decode observations differ"
            )
            for index, observation in enumerate(observations):
                stats = observation["stats"]
                _require(observation["cache_empty"] is True, "decoder cleanup failed")
                _require(
                    (stats is None) if cell["arm"] == ARMS[0] or index == 0 else True,
                    "draft leaked across arm boundaries",
                )
                if stats is not None:
                    _require(
                        0
                        <= stats["matched_tokens"]
                        <= stats["proposed_tokens"]
                        <= MAX_DRAFT_TOKENS
                        and stats["cleanup_calls"] >= 1,
                        "invalid draft counts",
                    )
            for window in cell["windows"]:
                _require(
                    window["reuse_alignment_features"] is True,
                    "alignment reuse disabled",
                )
                calls = window.get("forwards", [])
                _require(
                    sum(f["kind"] == "encoder" for f in calls)
                    == len(window["encoder_calls"]),
                    "encoder count differs",
                )
                for f in calls:
                    _require(
                        f["kind"] in {"encoder", "decoder"}
                        and f["phase"] in {"decode", "alignment"}
                        and (
                            f["interval_ms"] is None
                            or math.isfinite(f["interval_ms"])
                            and f["interval_ms"] >= 0
                        ),
                        "invalid forward interval",
                    )
                    _require(
                        f["kind"] != "decoder"
                        or type(f["input_tokens"]) is int
                        and f["input_tokens"] > 0,
                        "invalid decoder token count",
                    )
    probe_windows = [w for probe in record["probes"] for w in probe["windows"]]
    _require(
        len(record["probes"]) <= 1 and len(probe_windows) <= 1,
        "cleanup probe bound differs",
    )
    windows = sorted(windows + probe_windows, key=lambda w: w["call_index"])
    _require(
        record["native_window_count"] == len(windows) <= MAX_NATIVE_WINDOWS
        and [w["call_index"] for w in windows] == list(range(1, len(windows) + 1)),
        "window count differs",
    )
    _require(
        len({f["stream"] for w in windows for f in w.get("forwards", [])}) <= 1,
        "owned lane changed",
    )
    for cell in record["cells"]:
        checks = composed.cell_checks(cell, pcm)
        _require(
            cell["config"] == asdict(configurations()[cell["arm"]])
            and cell["summary"] == summary(cell, plan)
            and cell["checks"] == checks
            and cell["input_sha256"] == plan["pcm_sha256"]
            and cell["stream_status"]
            == ("completed" if all(checks.values()) else "failed"),
            "cell verdict differs",
        )
        _require(
            0
            <= cell["admitted_elapsed_ns"]
            < (GPU_TIMEOUT_SECONDS - RESERVE_SECONDS - 61.55) * 1e9,
            "cell admission exceeded budget",
        )
        if cell["stream_status"] == "completed":
            _require(
                len(cell["draft_observations"]) == len(cell["windows"])
                and all(
                    f["interval_ms"] is not None
                    for w in cell["windows"]
                    for f in w.get("forwards", [])
                ),
                "completed window lacks decode result or timing",
            )
    for warm in record["warmup"]:
        _require(
            warm["sample_count"] == 128000
            and warm["pcm_sha256"] == shared._sha(pcm[:256000])
            and len(warm["windows"]) <= 2,
            "warmup differs",
        )
    if record["cells"]:
        _require(
            len(record["warmup"]) == 2
            and all(
                w["capacity_restored"] and len(w["windows"]) == 2
                for w in record["warmup"]
            )
            and record["warmup"][1]["draft_observations"][1]["stats"][
                "parallel_prefills"
            ]
            == 1,
            "candidate prefill was not warmed",
        )
        _require(
            len(record["probes"]) == 1
            and all(
                record["probes"][0][key]
                for key in (
                    "cancellation_changed_state",
                    "capacity_restored",
                    "rows_cleared",
                    "features_cleared",
                    "cache_empty",
                    "hooks_empty",
                )
            )
            and record["probes"][0]["stats"]["parallel_prefills"] == 1
            and record["probes"][0]["stats"]["cleanup_calls"] >= 1
            and record["probes"][0]["kind"] == "cancel_after_speculative_prefill"
            and record["probes"][0]["published_events"] == 0
            and len(probe_windows) == 1
            and probe_windows[0]["closed"]
            and probe_windows[0]["capacity_restored"]
            and "decode_observation" not in probe_windows[0],
            "cancellation probe failed",
        )
    complete = (
        record["stop"] is None
        and len(record["cells"]) == len(record["warmup"]) == 2
        and len(record["probes"]) == 1
        and record["model"]["unchanged"]
        and record["capacity_restored"]
        and record["hooks_unchanged"]
        and all(c["stream_status"] == "completed" for c in record["cells"])
    )
    _require(
        record["status"] == ("completed" if complete else "partial"),
        "completion verdict differs",
    )
    expected_comparison = (
        comparison(record["cells"]) if complete else dict(paired=False, accepted=False)
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
    app = modal.App("wr-draft-" + uuid.uuid4().hex[:12])
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
        producer = importlib.import_module("infra.modal_draft_features")
        return producer._corpus().b._encode_worker_record(producer.run_worker(expected))

    return app, echo, execute


def run(*, replay_id, preflight=False, confirm_paid_gpu=False, root=ROOT):
    output, receipt, raw = shared._paths(root, replay_id, False)
    frozen = output.parent / "draft-preflight.json"
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
