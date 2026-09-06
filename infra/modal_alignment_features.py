"""One bounded T4 comparison of legacy and same-window feature alignment."""

from __future__ import annotations

import argparse
import gc
import importlib
import subprocess
import sys
import time
from dataclasses import asdict

from infra import modal_acoustic_diagnostic as shared
from infra import modal_word_resolution as inputs
from tools.prepare_acoustic_cases import _attenuate

ROOT = shared.ROOT
REMOTE_ROOT = shared.REMOTE_ROOT
PRODUCER = "infra/modal_alignment_features.py"
PATCH = "patches/openai-whisper/experimental/0008-Add-optional-alignment-audio-features.patch"
PATCH_SHA = "e665bca7abea5ab273c34ecf4d63c050f8c7120c7fdd9c411038b4f7511cd169"
BASE_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
BASE_COMMIT = "a0b9695ae1cc52bad4b8626fe9fb6ea4ac0ee650"
PATCHED_TREE = "32163d5cdb87babc1cd415a86cc5a58116c86a16"
CLAIMS = dict.fromkeys(
    (
        "general_speedup",
        "full_stream_recovery",
        "production_readiness",
        "cross_window_reuse",
    ),
    False,
)


def _corpus():
    return shared._corpus()


def cases(root=ROOT):
    saved = inputs.build_cases(root)
    result = []
    for item in saved[:2]:
        for mode, start in (
            ("current", item["retained_ms"]),
            ("alternative", item["head_ms"]),
        ):
            result.append(
                dict(
                    id=f"{item['id']}:{mode}",
                    start_ms=start,
                    end_ms=item["end_ms"],
                    pcm=item["pcm"][start * 32 : item["end_ms"] * 32],
                )
            )
    for item in saved[2:]:
        result.append(
            dict(
                id=f"previously-{item['id']}:full",
                start_ms=0,
                end_ms=item["end_ms"],
                pcm=item["pcm"],
            )
        )
    result.append(
        dict(id="digital-silence", start_ms=0, end_ms=2000, pcm=bytes(2000 * 32))
    )
    result.append(
        dict(
            id="attenuated32:2961-960-0004",
            start_ms=0,
            end_ms=saved[2]["end_ms"],
            pcm=_attenuate(saved[2]["pcm"]),
        )
    )
    if len(result) != 8 or any(
        not 0 < item["end_ms"] - item["start_ms"] <= 30000
        or len(item["pcm"]) != (item["end_ms"] - item["start_ms"]) * 32
        for item in result
    ):
        raise ValueError("eight complete bounded PCM windows required")
    return result


def input_records(root=ROOT):
    return [
        {
            **{key: value for key, value in item.items() if key != "pcm"},
            "pcm_sha256": shared._sha(item["pcm"]),
            "sample_count": len(item["pcm"]) // 2,
        }
        for item in cases(root)
    ]


def snapshot(root=ROOT):
    base = inputs.snapshot(root)
    names = {item["path"] for item in base["files"]}
    names.update((PRODUCER, PATCH))
    files = [
        dict(path=name, size_bytes=len(data), sha256=shared._sha(data))
        for name in sorted(names)
        for data in [(root / name).read_bytes()]
    ]
    return dict(commit=base["commit"], digest=shared._hash(files), files=files)


def apply_backend_patch(root, backend):
    """Patch only this disposable worker checkout, before importing Whisper."""
    if any(name == "whisper" or name.startswith("whisper.") for name in sys.modules):
        raise RuntimeError("Whisper must not be imported before patch verification")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=backend, text=True).strip()

    raw = (root / PATCH).read_bytes()
    before = git("rev-parse", "HEAD^{tree}")
    if (
        shared._sha(raw) != PATCH_SHA
        or before != BASE_TREE
        or git("status", "--porcelain")
    ):
        raise ValueError("backend or optional patch identity differs")
    git("apply", "--check", str(root / PATCH))
    git("apply", str(root / PATCH))
    git("add", "whisper/timing.py", "tests/test_alignment_audio_features.py")
    after = git("write-tree")
    if after != PATCHED_TREE:
        raise ValueError("patched backend tree differs")
    return dict(
        base_commit=git("rev-parse", "HEAD"),
        base_tree=before,
        patched_tree=after,
        patch_sha256=PATCH_SHA,
        timing_sha256=shared._sha((backend / "whisper/timing.py").read_bytes()),
    )


def summarize(cells):
    if len(cells) != 8 or any(
        set(cell["arms"]) != {"baseline", "reuse"} for cell in cells
    ):
        raise ValueError("eight paired observations required")
    outcomes = []
    for cell in cells:
        baseline, reuse = (cell["arms"][name] for name in ("baseline", "reuse"))
        if any(
            arm["closed"] is not True or arm["capacity_restored"] is not True
            for arm in (baseline, reuse)
        ):
            raise ValueError("each native run must close and restore capacity")

        def native(arm):
            return {**arm["alignment"]["native"], "window_id": "matched-input"}

        nonempty = bool(baseline["alignment"]["native"]["text"])
        decode_path = [dict(phase="decode", input_frames=3000, use_sdpa=None)]
        alignment_path = (
            [dict(phase="alignment", input_frames=3000, use_sdpa=False)]
            if nonempty
            else []
        )
        outcomes.append(
            dict(
                id=cell["id"],
                native_equal=shared._hash(native(baseline))
                == shared._hash(native(reuse)),
                words_equal=shared._hash(baseline["alignment"]["words"])
                == shared._hash(reuse["alignment"]["words"]),
                baseline_encoder_calls=len(baseline["encoder_calls"]),
                reuse_encoder_calls=len(reuse["encoder_calls"]),
                expected_encoder_counts=(
                    len(baseline["encoder_calls"]) == (2 if nonempty else 1)
                    and len(reuse["encoder_calls"]) == 1
                ),
                expected_encoder_paths=(
                    baseline["encoder_calls"] == decode_path + alignment_path
                    and reuse["encoder_calls"] == decode_path
                ),
            )
        )
    return dict(
        cells=outcomes,
        all_native_equal=all(item["native_equal"] for item in outcomes),
        all_words_equal=all(item["words_equal"] for item in outcomes),
        all_expected_encoder_counts=all(
            item["expected_encoder_counts"] for item in outcomes
        ),
        all_expected_encoder_paths=all(
            item["expected_encoder_paths"] for item in outcomes
        ),
        baseline_encoder_calls=sum(item["baseline_encoder_calls"] for item in outcomes),
        reuse_encoder_calls=sum(item["reuse_encoder_calls"] for item in outcomes),
    )


def validate_record(record, expected):
    if record["source"]["snapshot"] != expected or record["inputs"] != input_records():
        raise ValueError("worker source or inputs differ")
    if (
        record["backend"]["patched_tree"] != PATCHED_TREE
        or record["backend"]["patch_sha256"] != PATCH_SHA
        or record["backend"]["base_tree"] != BASE_TREE
        or record["backend"]["base_commit"] != BASE_COMMIT
    ):
        raise ValueError("worker backend differs")
    model = record["model"]
    expected_model = _corpus().read_registration()["model"]["model_state_sha256"]
    if model["initial_sha256"] != expected_model or model["unchanged"] is not (
        model["final_sha256"] == model["initial_sha256"]
    ):
        raise ValueError("worker model identity differs")
    if record["status"] == "completed" and model["unchanged"] is not True:
        raise ValueError("completed comparison changed model state")
    expected_summary = (
        summarize(record["cells"]) if record["status"] == "completed" else None
    )
    if record["status"] == "completed" and (
        record["native_window_count"] != 16
        or [cell["id"] for cell in record["cells"]]
        != [item["id"] for item in record["inputs"]]
        or sorted(
            arm["call_index"]
            for cell in record["cells"]
            for arm in cell["arms"].values()
        )
        != list(range(1, 17))
    ):
        raise ValueError("paired call identities differ")
    if record["summary"] != expected_summary or record["qualified"] is not False:
        raise ValueError("worker summary differs")


def run_worker(expected_snapshot):
    shared.verify_snapshot(expected_snapshot)
    corpus = _corpus()
    b, c = corpus.b, corpus.c
    # The cache volume is read-only; only the disposable checkout is patched.
    backend = apply_backend_patch(REMOTE_ROOT, b.BACKEND_ROOT)
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
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "Tesla T4"
    ):
        raise RuntimeError("registered PyTorch 2.6 / single T4 unavailable")
    if c._sha256_file(b.MODEL_CHECKPOINT_PATH) != base["model"]["checkpoint_sha256"]:
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
        revision=PATCHED_TREE,
        backend="pytorch-cuda-alignment-features",
        fingerprint=initial,
    )
    worker = runtime.Worker(
        "alignment-features",
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
            "tiny.en/alignment-features-v1",
            capacity,
            device="cuda:0",
            reuse_alignment_features=True,
        ),
    )
    options = adapters.NativeDecodeOptions(**base["decode_options"])

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

    def difference(left, right):
        delta = (left - right).double()
        return dict(
            exact=bool(torch.equal(left, right)),
            max_abs=float(delta.abs().max()),
            rms=float(delta.square().mean().sqrt()),
            allclose=bool(torch.allclose(left, right, atol=1e-6, rtol=1e-5)),
            atol=1e-6,
            rtol=1e-5,
        )

    count = 0

    def decode(case, arm, mel, observed):
        nonlocal count
        if count >= 16 or not available():
            raise RuntimeError("sixteen-window limit or unavailable capacity")
        count += 1
        features, phase = [], "decode"
        observed.update(call_index=count, before=memory(), encoder_calls=[])

        def encoder_hook(module, args, kwargs, output):
            observed["encoder_calls"].append(
                dict(
                    phase=phase,
                    input_frames=int(args[0].shape[-1]),
                    use_sdpa=kwargs.get("use_sdpa"),
                )
            )
            # CPU snapshots only: diagnostic storage must not keep GPU tensors.
            features.append(output.detach().float().cpu())

        hook = model.encoder.register_forward_hook(encoder_hook, with_kwargs=True)
        started = time.perf_counter_ns()
        torch.cuda.reset_peak_memory_stats(0)
        try:
            session = runtime.Session(f"features:{count}")
            with adapter.start_window(
                session=session,
                request=runtime.RequestState(
                    f"features:{count}:request",
                    session.session_id,
                    identity,
                    rng_seed=7,
                ),
                window_id=session.session_id,
                mel=mel,
                start_ms=case["start_ms"],
                end_ms=case["end_ms"],
                options=options,
            ) as run:
                steps = 0
                while not run.complete:
                    if steps >= b.MAX_DRIVER_STEPS:
                        raise RuntimeError("native step bound exceeded")
                    run.step()
                    steps += 1
                run.prepare_result()
                torch.cuda.synchronize(0)
                observed["decode_wall_ns"] = time.perf_counter_ns() - started
                phase, phase_started = "alignment", time.perf_counter_ns()
                observed["alignment"] = asdict(
                    run.prepare_word_alignment(reuse_alignment_features=arm == "reuse")
                )
                torch.cuda.synchronize(0)
                observed["alignment_wall_ns"] = time.perf_counter_ns() - phase_started
            observed.update(
                closed=run.closed,
                capacity_restored=run.capacity_released and available(),
            )
        finally:
            hook.remove()
        observed.update(
            wall_ns=time.perf_counter_ns() - started,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
        )
        del run
        observed["after_handle_release"] = memory()
        return features

    cells, failure = [], None
    for index, case in enumerate(cases(REMOTE_ROOT)):
        cell = dict(
            id=case["id"],
            arms={},
            order=["baseline", "reuse"] if index % 2 == 0 else ["reuse", "baseline"],
        )
        cells.append(cell)
        try:
            audio = np.frombuffer(case["pcm"], dtype="<i2").astype(np.float32) / 32768.0
            mel = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
            ).contiguous()
            features = {}
            for arm in cell["order"]:
                cell["arms"][arm] = {}
                features[arm] = decode(case, arm, mel, cell["arms"][arm])
            cell["feature_comparison"] = dict(
                decode_between_runs=difference(
                    features["baseline"][0], features["reuse"][0]
                ),
                decode_vs_legacy_alignment=difference(
                    features["baseline"][0], features["baseline"][1]
                )
                if len(features["baseline"]) == 2
                else None,
            )
            del features
        except Exception as error:
            failure = b._safe_stream_error(error)
            cell["error"] = failure
            break
    final = q._model_fingerprint(model) if available() else None
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    return dict(
        schema_version="1-diagnostic",
        experiment_id="modal-alignment-features-v1",
        status="completed" if failure is None else "failed",
        qualified=False,
        error=failure,
        claim_boundary=CLAIMS,
        inputs=input_records(REMOTE_ROOT),
        cells=cells,
        summary=summarize(cells) if failure is None else None,
        native_window_count=count,
        backend=backend,
        model=dict(
            initial_sha256=initial, final_sha256=final, unchanged=initial == final
        ),
        scope=dict(
            maximum_native_windows=16,
            cached_model=True,
            reference_routing=False,
            actual_stream_commits=0,
            default_alignment_unchanged=True,
            same_lane=True,
            single_host_thread=True,
            feature_snapshots_on_cpu=True,
            forward_hook_overhead_included=True,
            timing_excludes_preprocessing_and_collection=True,
        ),
        source=dict(
            snapshot=expected_snapshot,
            registration_sha256=shared._sha((REMOTE_ROOT / PRODUCER).read_bytes()),
        ),
        worker=dict(
            function_call_id=call_id,
            function_call_id_sha256=shared._sha(call_id.encode()),
            modal=str(modal.__version__),
        ),
    )


if __name__ == "__main__":
    from infra.modal_cuda_lane import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(
        run(
            producer=importlib.import_module("infra.modal_alignment_features"),
            **vars(parser.parse_args()),
        )
    )
