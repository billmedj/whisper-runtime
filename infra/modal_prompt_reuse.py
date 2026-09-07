"""One bounded T4 diagnostic of exact-window, two-prompt encoder reuse.

No live-stream policy or recognition improvement is qualified by this fixture.
The existing launcher requires explicit paid-work consent and never retries.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

from infra import modal_acoustic_diagnostic as shared

ROOT = shared.ROOT
REMOTE_ROOT = shared.REMOTE_ROOT
PRODUCER = "infra/modal_prompt_reuse.py"
GPU_TIMEOUT_SECONDS = 120
SAMPLE_LEN = 96
SEED = 7
PROFILES = ("greedy", "beam", "sample")
PROMPTS = (
    "An address to the American people.",
    "A speech about citizenship and service.",
)
BACKEND_COMMIT = "a0b9695ae1cc52bad4b8626fe9fb6ea4ac0ee650"
BACKEND_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
WORK_BOUNDS = dict(
    gpu="T4",
    gpu_calls=1,
    max_containers=1,
    retries=0,
    gpu_timeout_seconds=GPU_TIMEOUT_SECONDS,
    startup_timeout_seconds=300,
    decode_attempts=14,
    max_driver_steps_per_attempt=SAMPLE_LEN,
    max_hypotheses_per_step=2,
    sample_len=SAMPLE_LEN,
    numeric_precision="float32",
    model_download=False,
)
CLAIMS = dict.fromkeys(
    (
        "general_speedup",
        "recognition_improvement",
        "full_stream_recovery",
        "production_readiness",
        "cross_window_reuse",
        "physical_spending_cap",
    ),
    False,
)
# Only this task's implementation and its two launcher changes may be uncommitted.
# Every executed file is still hashed and checked in the disposable worker.
ALLOWED_DIRTY_SOURCE = frozenset(
    (
        PRODUCER,
        "src/whisper_runtime/adapters/native_whisper.py",
        "infra/modal_acoustic_diagnostic.py",
        "infra/modal_cuda_lane.py",
    )
)


def _corpus():
    return shared._corpus()


def _git(root, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def snapshot(root=ROOT):
    """Bind the uploaded working bytes, allowing only declared dirty source."""
    root = Path(root).resolve()
    corpus = _corpus()
    helpers = {
        PRODUCER,
        shared.PRODUCER,
        shared.MANIFEST,
        *shared.BUILDERS,
        corpus.MANIFEST_PATH,
        corpus.PRODUCER_PATH,
        *corpus.HELPER_PATHS,
        "infra/modal_cuda_lane.py",
        "infra/modal_native_cuda_qualification.py",
        "infra/native_cuda_trace.py",
        "conformance/audio-manifest.json",
    }
    names = helpers | {
        path.relative_to(root).as_posix()
        for path in (root / "src/whisper_runtime").rglob("*.py")
    }
    scope = ["src/whisper_runtime", *sorted(helpers)]
    dirty = set(_git(root, "diff", "--name-only", "HEAD", "--", *scope).splitlines())
    dirty.update(
        _git(
            root, "ls-files", "--others", "--exclude-standard", "--", *scope
        ).splitlines()
    )
    if dirty - ALLOWED_DIRTY_SOURCE:
        raise ValueError(
            "unreviewed dirty executed source: " + ", ".join(sorted(dirty))
        )
    files = [
        dict(path=name, size_bytes=len(data), sha256=shared._sha(data))
        for name in sorted(names)
        for data in [(root / name).read_bytes()]
    ]
    return dict(
        commit=_git(root, "rev-parse", "HEAD"),
        tracked_tree=_git(root, "rev-parse", "HEAD^{tree}"),
        dirty_source_paths=sorted(dirty),
        digest=shared._hash(files),
        files=files,
    )


def profile_options(profile):
    if profile not in PROFILES:
        raise ValueError("unknown decode profile")
    return dict(
        task="transcribe",
        language="en",
        sample_len=SAMPLE_LEN,
        temperature=0.4 if profile == "sample" else 0.0,
        beam_size=2 if profile == "beam" else None,
        best_of=2 if profile == "sample" else None,
        without_timestamps=False,
    )


def prompt_pair(profile):
    if profile not in PROFILES:
        raise ValueError("unknown decode profile")
    return (PROMPTS[0], None if profile == "greedy" else PROMPTS[1])


def input_record(root=ROOT):
    manifest = json.loads((root / "conformance/audio-manifest.json").read_text())
    fixture = next(
        item for item in manifest["fixtures"] if item["id"] == "openai-whisper-jfk-flac"
    )
    return {
        key: fixture[key]
        for key in (
            "id",
            "sha256",
            "size_bytes",
            "decoded_sample_rate_hz",
            "decoded_sample_count",
        )
    }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _result(value):
    metadata = value.metadata
    _require(metadata is not None, "missing native metadata")
    _require(isinstance(metadata.tokens, tuple), "mutable token provenance")
    _require(
        value.__dataclass_params__.frozen and metadata.__dataclass_params__.frozen,
        "mutable result provenance",
    )
    for scalar in (metadata.avg_logprob, metadata.no_speech_prob):
        _require(scalar is not None and math.isfinite(scalar), "missing finite score")
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def summarize(cells, cancellation):
    _require(
        [cell["id"] for cell in cells] == list(PROFILES),
        "three fixed profiles required",
    )
    outcomes, streams = [], set()
    for cell in cells:
        _require(
            cell["decode_options"] == profile_options(cell["id"]), "profile differs"
        )
        _require(
            cell["prompts"] == list(prompt_pair(cell["id"])), "cell prompts differ"
        )
        for name, attempts in (("baseline_a", 1), ("baseline_b", 1), ("reuse_a_b", 2)):
            arm = cell[name]
            _require(
                arm["decode_attempts"] == attempts and arm["encoder_forwards"] == 1,
                "decode or encoder count differs",
            )
            _require(
                len(arm["decoder_steps"]) == attempts
                and all(
                    type(steps) is int and 1 <= steps <= SAMPLE_LEN
                    for steps in arm["decoder_steps"]
                ),
                "driver step bound differs",
            )
            _require(
                all(
                    arm[key] is True
                    for key in (
                        "same_lease",
                        "held_before_commit",
                        "immutable_provenance",
                        "committed_final",
                        "resources_released",
                        "lane_released",
                        "closed",
                    )
                ),
                "unsafe completed lifecycle",
            )
            _require(
                arm["session_version"] == 1 and arm["request_status"] == "committed",
                "unexpected completed state",
            )
            _require(
                type(arm["lane_stream"]) is int
                and arm["lane_stream"] > 0
                and arm["stream_ids"] == [arm["lane_stream"]]
                and type(arm["lane_owner"]) is int
                and arm["lane_owner"] > 0,
                "execution escaped the leased CUDA lane",
            )
            streams.add(arm["lane_stream"])
            for result in (arm["first_result"], arm["committed_result"]):
                metadata = result["metadata"]
                _require(
                    isinstance(metadata["tokens"], list)
                    and all(type(token) is int for token in metadata["tokens"]),
                    "missing exact tokens",
                )
                _require(
                    all(
                        type(metadata[key]) in (int, float)
                        and math.isfinite(metadata[key])
                        for key in ("avg_logprob", "no_speech_prob")
                    ),
                    "missing finite scores",
                )
        a, b, reuse = (cell[name] for name in ("baseline_a", "baseline_b", "reuse_a_b"))
        _require(
            a["first_result"] == a["committed_result"]
            and b["first_result"] == b["committed_result"],
            "control result changed",
        )
        outcomes.append(
            dict(
                id=cell["id"],
                first_result_exact=a["committed_result"] == reuse["first_result"],
                second_result_exact=b["committed_result"] == reuse["committed_result"],
                baseline_encoder_forwards=2,
                reuse_encoder_forwards=1,
            )
        )
    _require(
        cancellation["decode_attempts"] == 2 and cancellation["encoder_forwards"] == 1,
        "cancellation attempt counts differ",
    )
    steps = cancellation["decoder_steps"]
    _require(
        len(steps) == 2
        and type(steps[0]) is int
        and 1 <= steps[0] <= SAMPLE_LEN
        and type(steps[1]) is int
        and steps[1] == 0,
        "cancellation step bound differs",
    )
    _require(
        all(
            cancellation[key] is True
            for key in (
                "same_lease",
                "held_before_cancel",
                "cancel_observed",
                "immutable_provenance",
                "resources_released",
                "lane_released",
                "closed",
            )
        ),
        "unsafe cancellation lifecycle",
    )
    _require(
        cancellation["session_version"] == 0
        and cancellation["request_status"] == "cancelled",
        "cancelled candidate was published",
    )
    _require(
        cancellation["stream_ids"] == [cancellation["lane_stream"]]
        and type(cancellation["lane_owner"]) is int
        and cancellation["lane_owner"] > 0,
        "cancellation escaped the leased CUDA lane",
    )
    streams.add(cancellation["lane_stream"])
    _require(len(streams) == 1, "more than one model-binding CUDA stream used")
    return dict(
        cells=outcomes,
        all_first_results_exact=all(item["first_result_exact"] for item in outcomes),
        all_second_results_exact=all(item["second_result_exact"] for item in outcomes),
        baseline_encoder_forwards=6,
        reuse_encoder_forwards=3,
        decode_attempts=14,
        all_lifecycle_checks=True,
        cuda_stream_count=len(streams),
    )


def validate_record(record, expected):
    _require(record["source"]["snapshot"] == expected, "worker source differs")
    _require(
        record["schema_version"] == "1-diagnostic"
        and record["status"] == "completed"
        and record["qualified"] is False
        and record["error"] is None,
        "worker did not complete",
    )
    _require(
        record["claim_boundary"] == CLAIMS and record["work_bounds"] == WORK_BOUNDS,
        "claim or work bounds differ",
    )
    _require(record["input"] == input_record(), "input binding differs")
    _require(
        record["configuration"]
        == dict(
            rng_seed=SEED,
            prompts={profile: list(prompt_pair(profile)) for profile in PROFILES},
        ),
        "context differs",
    )
    _require(
        record["backend"] == dict(commit=BACKEND_COMMIT, tree=BACKEND_TREE, clean=True),
        "backend differs",
    )
    registered = _corpus().read_registration()["model"]
    model = record["model"]
    _require(
        model["checkpoint_sha256"] == registered["checkpoint_sha256"]
        and model["initial_sha256"]
        == model["final_sha256"]
        == registered["model_state_sha256"]
        and model["unchanged"] is True,
        "model binding changed",
    )
    _require(
        record["environment"]
        == dict(
            torch="2.6.0",
            cuda_devices=1,
            gpu="Tesla T4",
            device="cuda:0",
            precision="float32",
        ),
        "GPU environment differs",
    )
    expected_summary = summarize(record["cells"], record["cancellation"])
    _require(record["summary"] == expected_summary, "summary differs")
    _require(
        expected_summary["all_first_results_exact"]
        and expected_summary["all_second_results_exact"],
        "exact prompt parity failed",
    )


def run_worker(expected_snapshot):
    shared.verify_snapshot(expected_snapshot)
    corpus = _corpus()
    b, c = corpus.b, corpus.c
    q = corpus._helper("infra.modal_native_cuda_qualification")
    backend = dict(
        commit=_git(b.BACKEND_ROOT, "rev-parse", "HEAD"),
        tree=_git(b.BACKEND_ROOT, "rev-parse", "HEAD^{tree}"),
        clean=not bool(_git(b.BACKEND_ROOT, "status", "--porcelain")),
    )
    _require(
        backend == dict(commit=BACKEND_COMMIT, tree=BACKEND_TREE, clean=True),
        "cached backend identity differs",
    )
    torch, whisper, runtime, adapters = (
        importlib.import_module(name)
        for name in ("torch", "whisper", "whisper_runtime", "whisper_runtime.adapters")
    )
    _require(
        Path(whisper.__file__).resolve().is_relative_to(b.BACKEND_ROOT),
        "unexpected backend import",
    )
    _require(
        Path(runtime.__file__).resolve().is_relative_to(REMOTE_ROOT / "src"),
        "runtime did not import the uploaded working tree",
    )
    environment = dict(
        torch=str(torch.__version__).split("+")[0],
        cuda_devices=torch.cuda.device_count(),
        gpu=torch.cuda.get_device_name(0),
        device="cuda:0",
        precision="float32",
    )
    _require(
        environment
        == dict(
            torch="2.6.0",
            cuda_devices=1,
            gpu="Tesla T4",
            device="cuda:0",
            precision="float32",
        ),
        "single T4 unavailable",
    )
    registered = corpus.read_registration(REMOTE_ROOT)["model"]
    checkpoint_sha = c._sha256_file(b.MODEL_CHECKPOINT_PATH)
    _require(checkpoint_sha == registered["checkpoint_sha256"], "cached model differs")
    # An existing absolute filename cannot take the named-model download branch.
    model = (
        whisper.load_model(str(b.MODEL_CHECKPOINT_PATH), device="cuda:0").float().eval()
    )
    initial = q._model_fingerprint(model)
    _require(initial == registered["model_state_sha256"], "loaded model differs")
    fixture = input_record(REMOTE_ROOT)
    audio_path = b.BACKEND_ROOT / "tests/jfk.flac"
    _require(
        audio_path.stat().st_size == fixture["size_bytes"]
        and c._sha256_file(audio_path) == fixture["sha256"],
        "cached audio differs",
    )
    audio = whisper.load_audio(str(audio_path))
    _require(
        fixture["decoded_sample_rate_hz"] == 16000
        and len(audio) == fixture["decoded_sample_count"] == 176000,
        "decoded fixture differs",
    )
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
    ).float()
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot("tiny.en", BACKEND_COMMIT, "pytorch-cuda", initial)
    worker = runtime.Worker(
        "prompt-reuse",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=GPU_TIMEOUT_SECONDS,
    )

    def probe(observed):
        _require(observed is model, "model object changed")
        return identity

    adapter = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/prompt-reuse-v1",
            capacity,
            device="cuda:0",
            reuse_decode_features=True,
        ),
    )

    def available():
        return (
            budget.available == capacity
            and budget.lease_count == 0
            and worker.queue_depth == 0
        )

    attempt_count = 0

    def execute(label, profile, prompt, *, replacement=False, cancel=False):
        nonlocal attempt_count
        _require(available(), "capacity not restored before next run")
        attempts = 2 if replacement else 1
        _require(
            attempt_count + attempts <= WORK_BOUNDS["decode_attempts"],
            "attempt budget exceeded",
        )
        attempt_count += attempts
        observed = dict(
            encoder_forwards=0,
            decode_attempts=attempts,
            decoder_steps=[],
            stream_ids=[],
            same_lease=True,
        )
        stream_ids = set()

        def encoder_hook(_module, _args, _output):
            observed["encoder_forwards"] += 1
            stream_ids.add(int(torch.cuda.current_stream(0).cuda_stream))

        def decoder_hook(_module, _args):
            stream_ids.add(int(torch.cuda.current_stream(0).cuda_stream))

        def drive(run):
            steps = 0
            while not run.complete:
                _require(steps < SAMPLE_LEN, "native driver step bound exceeded")
                run.step()
                steps += 1
            observed["decoder_steps"].append(steps)

        session = runtime.Session(label)
        request = runtime.RequestState(label, label, identity, rng_seed=SEED)
        hooks = [
            model.encoder.register_forward_hook(encoder_hook),
            model.decoder.register_forward_pre_hook(decoder_hook),
        ]
        try:
            with adapter.start_window(
                session=session,
                request=request,
                window_id="matched-jfk-window",
                mel=mel,
                start_ms=0,
                end_ms=11000,
                options=adapters.NativeDecodeOptions(
                    prompt=prompt, **profile_options(profile)
                ),
            ) as run:
                transaction = run._transaction
                lane = run._model_binding._cuda_lane
                owner = lane.owner
                observed.update(
                    lane_stream=int(lane.stream.cuda_stream), lane_owner=id(owner)
                )
                _require(owner is not None, "missing CUDA lease owner")
                drive(run)
                first = run.prepare_result()
                first_record = _result(first)
                if replacement:
                    run.redecode(prompt=prompt_pair(profile)[1])
                    observed["same_lease"] = (
                        run._transaction is transaction and lane.owner is owner
                    )
                    _require(
                        run.decode_attempt == 2, "replacement attempt not recorded"
                    )
                held = (
                    not run.capacity_released
                    and budget.available == runtime.ResourceVector()
                    and budget.lease_count == 1
                    and worker.queue_depth == 1
                    and session.snapshot().version == 0
                    and lane.owner is owner
                )
                if cancel:
                    observed["held_before_cancel"] = held
                    _require(
                        not run.complete, "replacement completed before cancellation"
                    )
                    before_cancel_steps = run.step_count
                    run.cancel()
                    observed["cancel_observed"] = False
                    try:
                        run.step()
                    except runtime.RequestCancelledError:
                        observed["cancel_observed"] = True
                    observed["decoder_steps"].append(
                        run.step_count - before_cancel_steps
                    )
                else:
                    observed["held_before_commit"] = held
                    if replacement:
                        drive(run)
                    final = run.prepare_result()
                    observed.update(
                        first_result=first_record, committed_result=_result(final)
                    )
                    state = run.finish(committed_through_ms=11000)
                    observed["committed_final"] = state.windows[-1].result is final
                observed["immutable_provenance"] = _result(first) == first_record
            observed.update(
                decode_attempts=run.decode_attempt,
                immutable_provenance=observed["immutable_provenance"]
                and _result(first) == first_record,
                closed=run.closed,
                resources_released=run.capacity_released and available(),
                lane_released=lane.owner is None,
                session_version=session.snapshot().version,
                request_status=request.status.value,
                stream_ids=sorted(stream_ids),
            )
        finally:
            for hook in hooks:
                hook.remove()
        return observed

    cells = []
    for profile in PROFILES:
        prompt_a, prompt_b = prompt_pair(profile)
        cells.append(
            dict(
                id=profile,
                decode_options=profile_options(profile),
                prompts=list(prompt_pair(profile)),
                baseline_a=execute(f"{profile}:baseline-a", profile, prompt_a),
                baseline_b=execute(f"{profile}:baseline-b", profile, prompt_b),
                reuse_a_b=execute(
                    f"{profile}:reuse-a-b", profile, prompt_a, replacement=True
                ),
            )
        )
    cancellation = execute(
        "cancel:reuse-a-b", "greedy", PROMPTS[0], replacement=True, cancel=True
    )
    torch.cuda.synchronize(0)
    final = q._model_fingerprint(model)
    _require(
        attempt_count == WORK_BOUNDS["decode_attempts"] and available(),
        "final budget differs",
    )
    shared.verify_snapshot(expected_snapshot)
    call_id = importlib.import_module("modal").current_function_call_id()
    return dict(
        schema_version="1-diagnostic",
        experiment_id="modal-prompt-reuse-v1",
        status="completed",
        qualified=False,
        error=None,
        claim_boundary=CLAIMS,
        work_bounds=WORK_BOUNDS,
        configuration=dict(
            rng_seed=SEED,
            prompts={profile: list(prompt_pair(profile)) for profile in PROFILES},
        ),
        input=fixture,
        backend=backend,
        environment=environment,
        model=dict(
            checkpoint_sha256=checkpoint_sha,
            initial_sha256=initial,
            final_sha256=final,
            unchanged=initial == final,
        ),
        cells=cells,
        cancellation=cancellation,
        summary=summarize(cells, cancellation),
        source=dict(
            snapshot=expected_snapshot,
            registration_sha256=shared._sha((REMOTE_ROOT / PRODUCER).read_bytes()),
        ),
        worker=dict(
            function_call_id=call_id,
            function_call_id_sha256=shared._sha(call_id.encode()),
            modal="1.5.5",
        ),
    )


def run(*, replay_id, confirm_paid_gpu=False, root=ROOT):
    from infra import modal_cuda_lane

    return modal_cuda_lane.run(
        replay_id=replay_id,
        confirm_paid_gpu=confirm_paid_gpu,
        root=root,
        producer=importlib.import_module("infra.modal_prompt_reuse"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(run(**vars(parser.parse_args())))
