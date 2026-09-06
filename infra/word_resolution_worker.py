"""Bounded native-window counterfactuals; no stream commits or remote entrypoint.

Importing this module loads no GPU/native backend. All output texts are proposed
concatenations, including held-out prefixes constructed from a bootstrap decode.
Neither local matching nor a head-only decode proves omission-free recognition.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import asdict

from tools.word_anchor_reconciliation import (
    plan_bounded_resolution,
    propose_terminal_end_reconciliation,
)
from whisper_runtime.adapters.word_policy import (
    compare_word_hypotheses,
    diagnose_word_anchor,
)

LEGACY_ROUTING_POLICY = "diagnostic-v1"
BOUNDED_ROUTING_POLICY = "bounded-fallback-v2"


def _evaluate(
    case,
    current,
    alternative,
    parameters,
    difference,
    *,
    routing_policy=LEGACY_ROUTING_POLICY,
):
    """Choose from observations first; human-reference metrics never route arms."""
    if routing_policy not in (LEGACY_ROUTING_POLICY, BOUNDED_ROUTING_POLICY):
        raise ValueError("unknown counterfactual routing policy")
    head, anchor = case["head_ms"], case["frozen_anchor"]
    matching = dict(
        committed_through_ms=head,
        anchor=anchor,
        timestamp_tolerance_ms=parameters["timestamp_tolerance_ms"],
    )
    routing_started = time.perf_counter_ns()
    started = time.perf_counter_ns()
    strict = compare_word_hypotheses(None, current, final=True, **matching)
    strict_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    diagnostic = diagnose_word_anchor(current, **matching)
    diagnostic_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    shadow = propose_terminal_end_reconciliation(
        current,
        final=True,
        max_end_extension_ms=parameters["max_end_extension_ms"],
        **matching,
    )
    shadow_ns = time.perf_counter_ns() - started
    baseline_text = "" if strict.publication is None else strict.publication.text
    shadow_text, shadow_available = "", False
    if shadow.status == "eligible":
        assert shadow.word_end is not None
        shadow_text = "".join(
            word.text for word in current.words[shadow.word_end :]
        ).strip()
        shadow_available = True
    elif shadow.status == "not_needed" and strict.publication is not None:
        shadow_text, shadow_available = baseline_text, True
    selected = (
        "alternative"
        if diagnostic.status in {"lexical_missing", "token_mismatch"}
        else "shadow"
        if diagnostic.status == "timing_mismatch" and shadow.status == "eligible"
        else "baseline"
    )
    route = None
    if routing_policy == BOUNDED_ROUTING_POLICY:
        span = current.native.analyzed_span
        alternative_span = alternative.native.analyzed_span
        route = plan_bounded_resolution(
            baseline_available=strict.publication is not None,
            reconciliation_eligible=shadow.status == "eligible",
            current_window=(span.start_ms, span.end_ms),
            alternative_window=(alternative_span.start_ms, alternative_span.end_ms),
        )
        selected = route.arm
    routing_ns = time.perf_counter_ns() - routing_started
    arms = {}
    for name, suffix, available in (
        ("baseline", baseline_text, strict.publication is not None),
        ("shadow", shadow_text, shadow_available),
        ("alternative", alternative.native.text, bool(alternative.native.text.strip())),
    ):
        text = " ".join(
            part for part in (case["committed_text"].strip(), suffix.strip()) if part
        )
        arms[name] = {
            "available": available,
            "suffix_text": suffix,
            "text": text,
            "against_human_reference": difference(text, case["reference_text"]),
            "publication_authority": False,
        }
    if selected == "unresolved":
        arms["unresolved"] = dict(arms["baseline"])
    # Reproduction is evaluated only after the observation-based route is fixed.
    # Canonical JSON normalizes tuple/list tokens but preserves JSON scalar types.
    reproduced = (
        None
        if "expected_native_text" not in case
        else current.native.text == case["expected_native_text"]
    )
    alignment_reproduced = (
        None
        if "expected_words" not in case
        else json.dumps(case["expected_words"], sort_keys=True, allow_nan=False)
        == json.dumps(
            [asdict(word) for word in current.words], sort_keys=True, allow_nan=False
        )
    )
    reason_reproduced = (
        None
        if "expected_strict_reason" not in case
        else strict.reason == case["expected_strict_reason"]
    )
    baseline_edits = arms["baseline"]["against_human_reference"]["word_edit_distance"]
    routed_edits = arms[selected]["against_human_reference"]["word_edit_distance"]
    return {
        "arms": arms,
        "routed": {
            "arm": selected,
            "reason": diagnostic.status if route is None else route.reason,
            "uses_reference": False,
            **({} if route is None else {"policy": routing_policy}),
        },
        "outcome": {
            "strict_reason": strict.reason,
            "expected_native_text_reproduced": reproduced,
            "expected_alignment_reproduced": alignment_reproduced,
            "strict_reason_reproduced": reason_reproduced,
            "comparison_valid": all(
                match is not False
                for match in (reproduced, alignment_reproduced, reason_reproduced)
            ),
            "routed_word_edit_delta": routed_edits - baseline_edits,
            "routed_non_regression": routed_edits <= baseline_edits,
            "full_stream_completion": False,
        },
        "current_diagnostic": asdict(diagnostic),
        "shadow_proposal": asdict(shadow),
        "local_timing": {
            "total_routing_wall_ns": routing_ns,
            "strict_wall_ns": strict_ns,
            "diagnostic_wall_ns": diagnostic_ns,
            "shadow_wall_ns": shadow_ns,
        },
    }


def _bootstrap(case, alignment, parameters):
    """Create a synthetic prefix from raw model words, never a human reference."""
    cutoff = case["bootstrap_ms"] - parameters["bootstrap_holdback_ms"]
    selected = []
    for word in alignment.words:
        if word.span.end_ms > cutoff:
            break
        selected.append(word)
    while selected and not any(char.isalnum() for char in selected[-1].text):
        selected.pop()
    if not selected or selected[-1].span.end_ms <= 0:
        return None
    anchor = selected[-4:]
    while anchor and not any(char.isalnum() for char in anchor[0].text):
        anchor.pop(0)
    head = selected[-1].span.end_ms
    retained = max(
        0, min(head - parameters["left_context_ms"], anchor[0].span.start_ms)
    )
    return {
        **case,
        "head_ms": head,
        "retained_ms": retained // 20 * 20,
        "frozen_anchor": tuple(anchor),
        "committed_text": "".join(word.text for word in selected).strip(),
    }


def run_worker(expected_snapshot):
    producer = importlib.import_module("infra.modal_word_resolution")
    acoustic = importlib.import_module("infra.modal_acoustic_diagnostic")
    corpus = acoustic._corpus()
    b, c = corpus.b, corpus.c
    root = acoustic.REMOTE_ROOT
    producer.verify_snapshot(expected_snapshot)
    settings, cases = producer.registration(root), producer.build_cases(root)
    parameters = settings["parameters"]
    base = corpus.read_registration(root)
    torch, np, whisper = (
        importlib.import_module(name) for name in ("torch", "numpy", "whisper")
    )
    runtime, adapters = (
        importlib.import_module(name)
        for name in ("whisper_runtime", "whisper_runtime.adapters")
    )
    q = corpus._helper("infra.modal_native_cuda_qualification")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "Tesla T4"
    ):
        raise RuntimeError("registered single T4 unavailable")
    if c._sha256_file(b.MODEL_CHECKPOINT_PATH) != base["model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint mismatch")
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=b.MODEL_CACHE_MOUNT
    ).eval()
    initial = q._model_fingerprint(model)
    if (
        initial != base["model"]["model_state_sha256"]
        or model.training
        or any(
            str(t.device) != "cuda:0"
            or (t.is_floating_point() and t.dtype != torch.float32)
            for t in (*model.parameters(), *model.buffers())
        )
    ):
        raise RuntimeError("registered FP32 model mismatch")
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=c._command_output(b.BACKEND_ROOT, "rev-parse", "HEAD"),
        backend="pytorch-cuda-word-resolution",
        fingerprint=initial,
    )
    worker = runtime.Worker(
        "word-resolution",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=180,
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
            "tiny.en/word-resolution-v1", capacity, device="cuda:0"
        ),
    )
    options = adapters.NativeDecodeOptions(**base["decode_options"])
    window_count = 0

    def available():
        return (
            budget.available == capacity
            and budget.lease_count == 0
            and worker.queue_depth == 0
        )

    def decode(case, name, start_ms, end_ms, measurements):
        nonlocal window_count
        if not available() or window_count >= 10:
            raise RuntimeError("capacity unavailable or ten-window limit reached")
        pcm = case["pcm"][start_ms * 32 : end_ms * 32]
        if (
            not 0 <= start_ms < end_ms <= len(case["pcm"]) // 32
            or end_ms - start_ms > 30_000
        ):
            raise ValueError("counterfactual slice is outside retained bounded input")
        window_count += 1
        measurement = {
            "call_index": window_count,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "sample_count": len(pcm) // 2,
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "first_call_no_warmup": window_count == 1,
            "encoder_input_frames": [],
            "decoder_forward_calls": 0,
            "decoder_steps": 0,
            "completed": False,
        }
        measurements[name] = measurement
        hooks, run = [], None
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter_ns()
        try:
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            mel = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
            ).contiguous()
            measurement["mel_shape"] = list(mel.shape)
            torch.cuda.synchronize(0)
            measurement["mel_wall_ns"] = time.perf_counter_ns() - started

            def encoder_hook(module, args):
                measurement["encoder_input_frames"].append(int(args[0].shape[-1]))

            def decoder_hook(module, args):
                measurement["decoder_forward_calls"] += 1

            hooks.append(model.encoder.register_forward_pre_hook(encoder_hook))
            hooks.append(model.decoder.register_forward_pre_hook(decoder_hook))
            session = runtime.Session(f"resolution:{case['id']}:{name}")
            phase = time.perf_counter_ns()
            with adapter.start_window(
                session=session,
                request=runtime.RequestState(
                    f"{session.session_id}:request",
                    session.session_id,
                    identity,
                    rng_seed=7,
                ),
                window_id=session.session_id,
                mel=mel,
                start_ms=start_ms,
                end_ms=end_ms,
                options=options,
            ) as run:
                while not run.complete:
                    if measurement["decoder_steps"] >= b.MAX_DRIVER_STEPS:
                        raise RuntimeError("native driver step bound exceeded")
                    run.step()
                    measurement["decoder_steps"] += 1
                run.prepare_result()
                torch.cuda.synchronize(0)
                measurement["decode_wall_ns"] = time.perf_counter_ns() - phase
                phase = time.perf_counter_ns()
                alignment = run.prepare_word_alignment()
                torch.cuda.synchronize(0)
                measurement["alignment_wall_ns"] = time.perf_counter_ns() - phase
                phase = time.perf_counter_ns()
            torch.cuda.synchronize(0)
            measurement["close_wall_ns"] = time.perf_counter_ns() - phase
            if not run.capacity_released or not run.closed or not available():
                raise RuntimeError("native window retained capacity")
            measurement["completed"] = True
            return alignment
        finally:
            for hook in hooks:
                hook.remove()
            torch.cuda.synchronize(0)
            measurement.update(
                wall_ns=time.perf_counter_ns() - started,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
                capacity_restored=available(),
                encoder_forward_calls=len(measurement["encoder_input_frames"]),
                padded_encoder_ms=sum(measurement["encoder_input_frames"]) * 10,
            )

    cells = []
    for index, original in enumerate(cases):
        case = dict(original)
        cell = {
            "id": case["id"],
            "split": case["split"],
            "timings": {},
            "raw_alignments": {},
            "state_authority": "synthetic_prefix_state"
            if "bootstrap_ms" in case
            else "saved_terminal_state",
        }
        cells.append(cell)
        try:
            if "bootstrap_ms" in case:
                bootstrap = decode(
                    case, "bootstrap", 0, case["bootstrap_ms"], cell["timings"]
                )
                cell["raw_alignments"]["bootstrap"] = asdict(bootstrap)
                case = _bootstrap(case, bootstrap, parameters)
                if case is None:
                    cell.update(
                        status="bootstrap_unresolved", capacity_restored=available()
                    )
                    continue
            cell["frozen_state"] = {
                key: case[key]
                for key in ("head_ms", "retained_ms", "end_ms", "committed_text")
            }
            cell["frozen_state"]["anchor"] = [
                asdict(word) for word in case["frozen_anchor"]
            ]
            order = (
                ["current", "alternative"]
                if index % 2 == 0
                else ["alternative", "current"]
            )
            cell["decode_order"] = order
            alignments = {}
            for name in order:
                start_ms = case["retained_ms"] if name == "current" else case["head_ms"]
                alignments[name] = decode(
                    case, name, start_ms, case["end_ms"], cell["timings"]
                )
                cell["raw_alignments"][name] = asdict(alignments[name])
            cell.update(
                _evaluate(
                    case,
                    alignments["current"],
                    alignments["alternative"],
                    parameters,
                    b._word_difference,
                )
            )
            cell.update(status="evaluated", capacity_restored=available())
        except Exception as error:
            cell.update(
                status="lifecycle_failure",
                error=b._safe_stream_error(error),
                capacity_restored=available(),
            )
            break
    final = q._model_fingerprint(model) if available() else None
    return {
        "model": {
            "initial_sha256": initial,
            "final_sha256": final,
            "unchanged": final == initial,
            "backend_revision": identity.revision,
        },
        "capacity_restored": available(),
        "cells": cells,
        "native_window_count": window_count,
        "measurement_scope": {
            "no_warmup": True,
            "alternating_arm_order": True,
            "forward_hook_overhead_included": True,
            "performance_benchmark": False,
            "padded_encoder_ms_is_not_flops": True,
            "actual_stream_commits": 0,
        },
    }
