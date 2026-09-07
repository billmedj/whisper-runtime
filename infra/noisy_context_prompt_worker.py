"""Compare two exact crops with and without the published decoder history.

This worker collects observations. It never commits a transcript or changes
the source stream. Human reference text is used only for post-hoc scoring.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import time
from dataclasses import asdict

from infra import modal_noisy_context_prompt as p
from infra.eof_context_retry_worker import _MeasuredAdapter


def analyze(alignment, source, crop):
    """Replay the existing selector against the frozen head and anchor."""
    from tools.analyze_word_resolution import _alignment
    from whisper_runtime.adapters.audio_evidence import (
        AudioEvidenceDecision,
        AudioObservation,
        assess_audio,
    )
    from whisper_runtime.adapters.native_result import NativeTimestampSegment
    from whisper_runtime.adapters.word_policy import compare_word_hypotheses
    from whisper_runtime.state import AudioSpan

    current = _alignment(alignment)
    pcm = source["pcm"][crop["start_sample"] * 2 : crop["end_sample"] * 2]
    p._require(p.shared._sha(pcm) == crop["pcm_sha256"], "analysis PCM differs")
    p._require(
        current.native.analyzed_span == AudioSpan(crop["start_ms"], crop["end_ms"]),
        "analysis span differs from its crop",
    )
    anchor = tuple(
        NativeTimestampSegment(
            span=AudioSpan(**word["span"]),
            text=word["text"],
            tokens=tuple(word["tokens"]),
        )
        for word in source["anchor"]
    )
    decision = compare_word_hypotheses(
        None,
        current,
        committed_through_ms=source["head_sample"] // 16,
        anchor=anchor,
        holdback_ms=source["config"]["holdback_ms"],
        timestamp_tolerance_ms=source["config"]["timestamp_tolerance_ms"],
        final=True,
    )
    evidence = assess_audio(AudioObservation.from_pcm(pcm), current.native)
    publication = decision.publication
    lexical = publication is not None and any(c.isalnum() for c in publication.text)
    if publication is not None and not lexical:
        evidence = AudioEvidenceDecision(
            evidence.observation,
            "uncertain",
            "no_lexical_text",
            evidence.no_speech_prob,
        )
    # A head-only observation does not validate its join to the prior words.
    # Even a retained prompted match is a diagnostic candidate, not independent
    # corroboration or permission to publish.
    candidate = (
        crop["id"] == "retained" and lexical and evidence.state == "speech_candidate"
    )
    suffix = (
        current.native.text
        if crop["id"] == "head-only"
        else (publication.text if publication is not None else None)
    )
    full_text = (
        None if suffix is None else " ".join((source["prompt"] + " " + suffix).split())
    )
    return dict(
        audio_evidence=asdict(evidence),
        word_decision=asdict(decision),
        strict_candidate=bool(candidate),
        publication_authorized=False,
        recognition=dict(
            hypothetical_full_text=full_text,
            against_human_reference=None
            if full_text is None
            else p._corpus().b._word_difference(full_text, source["reference_text"]),
        ),
    )


def check_budget(started_ns):
    if (
        time.perf_counter_ns() - started_ns
        >= (p.GPU_TIMEOUT_SECONDS - p.CLEANUP_RESERVE_SECONDS) * 1_000_000_000
    ):
        raise TimeoutError("cleanup reserve reached before native work")


def drive(run, started_ns):
    """Bound each decode while retaining the worker's cleanup reserve."""
    initial = run.step_count
    while not run.complete:
        if run.step_count - initial >= p.MAX_DRIVER_STEPS:
            raise RuntimeError("registered decoder-step bound reached")
        check_budget(started_ns)
        run.step()
    return run.step_count - initial


def run_worker(expected_snapshot):
    started_all = time.perf_counter_ns()
    root, shared = p.REMOTE_ROOT, p.shared
    shared.verify_snapshot(expected_snapshot, root)
    source = p.source_case(root)
    corpus = p._corpus()
    b, c = corpus.b, corpus.c
    registered = corpus.read_registration(root)
    backend = dict(
        commit=p._git(b.BACKEND_ROOT, "rev-parse", "HEAD"),
        tree=p._git(b.BACKEND_ROOT, "rev-parse", "HEAD^{tree}"),
    )
    p._require(
        backend == dict(commit=p.BACKEND_COMMIT, tree=p.BACKEND_TREE)
        and not p._git(b.BACKEND_ROOT, "status", "--porcelain"),
        "cached backend identity differs",
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
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=b.MODEL_CACHE_MOUNT
    ).eval()
    initial = q._model_fingerprint(model)
    p._require(
        initial == registered["model"]["model_state_sha256"]
        and not model.training
        and model.dims.n_text_ctx // 2 == p.MAX_DRIVER_STEPS
        and all(
            str(v.device) == "cuda:0"
            and (not v.is_floating_point() or v.dtype == torch.float32)
            for v in (*model.parameters(), *model.buffers())
        ),
        "registered FP32 model differs",
    )
    options = adapters.NativeDecodeOptions(**source["decode_options"])
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=options.language,
        task=options.task,
    )
    p._require(tokenizer.eot == p.TOKENIZER_EOT, "registered tokenizer EOT differs")
    effective = dict(
        decode_options=asdict(options),
        rng_seed=p.SEED,
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
            for name in source["effective_identity"]["backend_artifacts"]
        },
    )
    p._require(
        effective == source["effective_identity"], "effective archived identity differs"
    )
    effective["reuse_decode_features"] = True
    effective["tokenizer_eot"] = tokenizer.eot
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=p.BACKEND_COMMIT,
        backend="pytorch-cuda-noisy-context-prompt",
        fingerprint=initial,
    )
    worker = runtime.Worker(
        "noisy-context-prompt",
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
            "tiny.en/noisy-context-prompt-v1",
            capacity,
            device="cuda:0",
            reuse_decode_features=True,
            reuse_alignment_features=False,
        ),
    )
    measured = _MeasuredAdapter(native, started_all, producer=p)

    def available():
        return (
            budget.available == capacity
            and budget.lease_count == 0
            and worker.queue_depth == 0
        )

    hooks = [
        model.encoder.register_forward_pre_hook(
            measured.observe_encoder, with_kwargs=True
        ),
        model.decoder.register_forward_pre_hook(
            measured.observe_decoder, with_kwargs=True
        ),
    ]
    cells, stop, error = [], None, None
    try:
        for crop in source["crops"]:
            run = first = first_alignment = None
            session = runtime.Session(crop["id"])
            cell = dict(
                **crop,
                window=None,
                attempts=[],
                decode_attempt_count=0,
                first_snapshot_unchanged=False,
                same_lease=False,
                closed=False,
                capacity_restored=False,
                session_version=0,
                error=None,
            )
            cells.append(cell)
            measured.records, measured.pcm = [], source["pcm"]
            try:
                started = time.perf_counter_ns()
                pcm = source["pcm"][crop["start_sample"] * 2 : crop["end_sample"] * 2]
                audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                mel = whisper.log_mel_spectrogram(
                    whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
                ).contiguous()
                measured.mel_wall_ns = time.perf_counter_ns() - started
                request = runtime.RequestState(
                    crop["id"], crop["id"], identity, rng_seed=p.SEED
                )
                cell["decode_attempt_count"] = 1
                run = measured.start_window(
                    session=session,
                    request=request,
                    window_id="noisy-" + crop["id"],
                    mel=mel,
                    start_ms=crop["start_ms"],
                    end_ms=crop["end_ms"],
                    options=options,
                )
                transaction = run._transaction
                lane = run._model_binding._cuda_lane
                owner = lane.owner
                p._require(owner is not None, "missing CUDA owner")
                for attempt_id, prompt in (
                    ("null", None),
                    ("published-prefix", source["prompt"]),
                ):
                    if prompt is not None:
                        check_budget(started_all)
                        cell["decode_attempt_count"] = 2
                        measured.invoke(
                            measured.records[0], "redecode", run.redecode, prompt=prompt
                        )
                    steps = drive(run, started_all)
                    result = run.prepare_result()
                    check_budget(started_all)
                    alignment = measured.invoke(
                        measured.records[0],
                        "alignment",
                        run.raw.prepare_word_alignment,
                        reuse_alignment_features=False,
                    )
                    p._require(
                        alignment.native is result, "alignment native result differs"
                    )
                    observation = dict(
                        id=attempt_id,
                        prompt=prompt,
                        decode_attempt=run.decode_attempt,
                        decode_options={**asdict(options), "prompt": prompt},
                        result=asdict(result),
                        alignment=asdict(alignment),
                        session_version=session.snapshot().version,
                        driver_steps=steps,
                        encoder_forwards_so_far=len(
                            measured.records[0]["encoder_calls"]
                        ),
                        **analyze(asdict(alignment), source, crop),
                    )
                    cell["attempts"].append(observation)
                    if first is None:
                        first, first_alignment = result, alignment
                    cell["same_lease"] = (
                        run._transaction is transaction
                        and lane.owner is owner
                        and not run.capacity_released
                        and budget.lease_count == 1
                        and worker.queue_depth == 1
                        and budget.available == runtime.ResourceVector()
                    )
                    p._require(
                        cell["same_lease"], "observation changed the native lease"
                    )
                cell["first_snapshot_unchanged"] = (
                    asdict(first) == cell["attempts"][0]["result"]
                    and asdict(first_alignment) == cell["attempts"][0]["alignment"]
                    and first is not result
                    and first_alignment is not alignment
                )
                p._require(cell["first_snapshot_unchanged"], "old observation changed")
            except Exception as caught:
                cell["error"] = error = b._safe_stream_error(caught)
                stop = dict(reason="cell_error", cell=len(cells) - 1)
            finally:
                if run is not None:
                    try:
                        run.close()  # Deliberately no finish() or stream mutation.
                    except Exception as caught:
                        cell["cleanup_error"] = error = b._safe_stream_error(caught)
                        stop = dict(reason="cleanup_error", cell=len(cells) - 1)
                    cell["closed"] = run.closed
                if measured.records:
                    cell["window"] = measured.records[0]
                cell["session_version"] = session.snapshot().version
                cell["capacity_restored"] = available()
                if not available() or cell["session_version"] != 0:
                    stop = dict(reason="lifecycle_invariant", cell=len(cells) - 1)
            if stop is not None:
                break
    finally:
        for hook in hooks:
            hook.remove()
    final = q._model_fingerprint(model) if available() else None
    if initial != final and stop is None:
        stop = dict(reason="model_changed")
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    complete = (
        stop is None
        and len(cells) == 2
        and all(len(cell["attempts"]) == 2 for cell in cells)
        and available()
        and initial == final
    )
    return dict(
        schema_version="1-diagnostic",
        experiment_id=p.EXPERIMENT_ID,
        status="completed" if complete else "failed",
        qualified=False,
        claim_boundary=p.CLAIMS,
        scope=p.scope(),
        input=p.input_record(root),
        backend=backend,
        effective_identity=effective,
        cells=cells,
        control=p.control(cells, source),
        native_window_count=measured.count,
        decode_attempt_count=sum(cell["decode_attempt_count"] for cell in cells),
        capacity_restored=available(),
        stop=stop,
        error=error,
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
