"""One grouped acoustic diagnostic. Importing starts no remote resources."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/whisper-runtime")
MANIFEST = "experiments/modal-acoustic-diagnostic-v1.json"
PRODUCER = "infra/modal_acoustic_diagnostic.py"
BUILDERS = ("tools/prepare_acoustic_cases.py", "tools/prepare_speech_corpus.py")
CASES = [
    "mixed-control",
    "concatenated-no-added-pauses",
    "mixed-continuous-noise64",
    "mixed-attenuated32",
]
CELLS = [("quiet", CASES[0]), *(("hybrid", case) for case in CASES)]
ACCEPTANCE = {
    "real_final_and_full_coverage": True,
    "native_capacity_restored": True,
    "hybrid_human_edits_at_most_offline": True,
    "legacy_normalized_control_edits": 0,
    "legacy_human_edits": 6,
    "legacy_reference_words": 88,
}
CHECK_NAMES = frozenset(
    (
        "ordered_events",
        "immutable_committed_revisions",
        "monotonic_commit_watermarks",
        "publication_within_accepted_input",
        "contiguous_commit_coverage",
        "commit_watermark_matches_runtime",
        "bounded_audio_buffer",
        "runtime_capacity_restored",
        "accepted_input_accounted",
        "decision_trace_ordered",
        "final_event_once",
        "full_input_committed_at_eof",
        "completion_required",
        "native_pcm_binding",
        "aligned_word_identity",
        "selected_text_matches_commit",
        "profile_identity",
    )
)


def _corpus():
    old = os.environ.get("WHISPER_MODAL_ENABLE_WORD_CORPUS")
    try:
        os.environ["WHISPER_MODAL_ENABLE_WORD_CORPUS"] = "0"
        return importlib.import_module("infra.modal_word_corpus")
    finally:
        if old is None:
            os.environ.pop("WHISPER_MODAL_ENABLE_WORD_CORPUS", None)
        else:
            os.environ["WHISPER_MODAL_ENABLE_WORD_CORPUS"] = old


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value) -> str:
    return _sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def registration(root=ROOT):
    value = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1-diagnostic",
        "id": "modal-acoustic-diagnostic-v1",
        "modal_sdk": "1.5.5",
        "case_ids": CASES,
        "cells": [f"{profile}:{case}" for profile, case in CELLS],
        "hybrid_changes": {"word_boundary_fallback": True, "left_context_ms": 2000},
        "chunk_ms": 1000,
        "rng_seed": 7,
        "acceptance": ACCEPTANCE,
        "execution": {
            "gpu": "T4",
            "cpu": 2,
            "memory_mib": 4096,
            "timeout_seconds": 180,
            "startup_timeout_seconds": 300,
            "max_containers": 1,
            "min_containers": 0,
            "configured_retries": 0,
            "gpu_client_calls": 1,
        },
        "claim_boundary": {
            "recognition_improvement": False,
            "performance_benchmark": False,
            "production_readiness": False,
            "generalization_beyond_this_input": False,
            "physical_execution_or_spending_cap": False,
        },
    }
    for key, item in expected.items():
        if _hash(value.get(key)) != _hash(item):
            raise ValueError(f"unregistered {key}")
    return value


def snapshot(root=ROOT):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("commit local changes before the diagnostic")
    commit = git("rev-parse", "HEAD")
    corpus = _corpus()
    names = sorted(
        [
            p.relative_to(root).as_posix()
            for p in (root / "src/whisper_runtime").rglob("*.py")
        ]
        + [
            MANIFEST,
            PRODUCER,
            *BUILDERS,
            corpus.MANIFEST_PATH,
            corpus.PRODUCER_PATH,
            *corpus.HELPER_PATHS,
        ]
    )
    files = []
    for name in names:
        data = (root / name).read_bytes()
        files.append({"path": name, "size_bytes": len(data), "sha256": _sha(data)})
    return {"commit": commit, "digest": _hash(files), "files": files}


def verify_snapshot(expected, root=REMOTE_ROOT):
    if _hash(expected["files"]) != expected["digest"]:
        raise ValueError("snapshot digest mismatch")
    for item in expected["files"]:
        path = root / item["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("source path escapes root")
        data = path.read_bytes()
        if len(data) != item["size_bytes"] or _sha(data) != item["sha256"]:
            raise ValueError("source file mismatch")


def _cases(root=ROOT, asset_dir=None):
    return importlib.import_module("tools.prepare_acoustic_cases").build_cases(
        root=root, asset_dir=asset_dir
    )


def publication_checks(events, traces, pcm):
    """Bind native observations and selected text; do not certify word timing."""
    observed = words_valid = publication_valid = True
    expected = {}
    try:
        for trace in traces:
            start, end = trace["analysis_start_sample"], trace["analysis_end_sample"]
            content = pcm[start * 2 : end * 2]
            observation = trace["audio_evidence"]["observation"]
            observed &= (
                0 <= start < end <= len(pcm) // 2
                and observation["sample_count"] == end - start
                and observation["pcm_sha256"] == _sha(content)
                and observation["digital_silence"] is (not any(content))
            )
            native = trace["result"]
            observed &= (
                native["analysis_span"]
                or {"start_ms": native["start_ms"], "end_ms": native["end_ms"]}
            ) == {"start_ms": start // 16, "end_ms": end // 16}
            alignment, publication = (
                trace.get("word_alignment"),
                trace.get("word_publication"),
            )
            if alignment is not None:
                words_valid &= (
                    alignment["native"] == native
                    and "".join(word["text"] for word in alignment["words"]).strip()
                    == native["text"]
                )
            if publication is not None:
                selected = alignment["words"][
                    publication["word_start"] : publication["word_end"]
                ]
                words_valid &= (
                    publication["alignment"] == alignment
                    and 0
                    <= publication["word_start"]
                    <= publication["word_end"]
                    <= len(alignment["words"])
                    and publication["text"]
                    == "".join(word["text"] for word in selected).strip()
                )
            unit = trace.get("source_unit")
            if unit is not None and unit["origin"] == "quiet_run":
                endpoint = unit["endpoint"]
                quiet = pcm[
                    endpoint["quiet_start_sample"] * 2 : endpoint["end_sample"] * 2
                ]
                peak = max(
                    (abs(item[0]) for item in struct.iter_unpack("<h", quiet)),
                    default=32768,
                )
                observed &= (
                    peak == endpoint["peak"] <= 32
                    and endpoint["end_sample"] <= end
                    and endpoint["end_sample"] - endpoint["quiet_start_sample"] >= 9600
                )
            if trace["action"] == "commit":
                if publication is not None:
                    span = (publication["start_ms"] * 16, publication["end_ms"] * 16)
                    text = publication["text"]
                elif trace.get("silence_publication") is not None:
                    silent = trace["silence_publication"]
                    span, text = (silent["start_ms"] * 16, silent["end_ms"] * 16), ""
                else:
                    span, text = (trace["committed_before_sample"], end), native["text"]
                publication_valid &= span not in expected
                expected[span] = text
        revisions = {}
        for event in events:
            if event["kind"] in {"provisional", "replace"}:
                revisions[event["segment_id"]] = event
            elif event["kind"] == "commit":
                revision = revisions[event["segment_id"]]
                publication_valid &= revision["revision"] == event[
                    "revision"
                ] and revision["text"] == expected.get(
                    (event["start_sample"], event["end_sample"])
                )
    except (KeyError, TypeError, ValueError):
        observed = words_valid = publication_valid = False
    return {
        "native_pcm_binding": bool(observed),
        "aligned_word_identity": bool(words_valid),
        "selected_text_matches_commit": bool(publication_valid),
    }


def quality_gate(profile, recognition, offline):
    human = recognition["against_human_reference"]
    if profile == "hybrid":
        return (
            human["word_edit_distance"]
            <= offline["against_human_reference"]["word_edit_distance"]
        )
    return (
        recognition["against_offline_control"]["word_edit_distance"] == 0
        and human["word_edit_distance"] == 6
        and human["reference_word_count"] == 88
    )


def run_worker(expected_snapshot):
    corpus = _corpus()
    b, c = corpus.b, corpus.c
    verify_snapshot(expected_snapshot)
    settings = registration(REMOTE_ROOT)
    base = corpus.read_registration(REMOTE_ROOT)
    inputs, pcms = _cases(REMOTE_ROOT, corpus.REMOTE_ASSETS)
    if [case["id"] for case in inputs["cases"]] != CASES:
        raise ValueError("case order differs from registration")
    metadata = {case["id"]: case for case in inputs["cases"]}
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
        raise RuntimeError("registered T4 unavailable")
    if c._sha256_file(b.MODEL_CHECKPOINT_PATH) != base["model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint mismatch")
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=b.MODEL_CACHE_MOUNT
    ).eval()
    initial = q._model_fingerprint(model)
    if initial != base["model"]["model_state_sha256"] or any(
        str(t.device) != "cuda:0"
        or (t.is_floating_point() and t.dtype != torch.float32)
        for t in (*model.parameters(), *model.buffers())
    ):
        raise RuntimeError("registered model mismatch")
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=c._command_output(b.BACKEND_ROOT, "rev-parse", "HEAD"),
        backend="pytorch-cuda-acoustic-diagnostic",
        fingerprint=initial,
    )
    worker = runtime.Worker(
        "acoustic-diagnostic",
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
            "tiny.en/acoustic-diagnostic-v1", capacity, device="cuda:0"
        ),
    )
    options = adapters.NativeDecodeOptions(**base["decode_options"])

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

    controls, cells = {}, []
    for profile, case_id in CELLS:
        pcm, case = pcms[case_id], metadata[case_id]
        stream = None
        try:
            if case_id not in controls:
                torch.manual_seed(7)
                torch.cuda.manual_seed_all(7)
                control = b._offline_result(
                    model.transcribe(
                        np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0,
                        **base["offline_control_options"],
                    )
                )
                control["against_human_reference"] = b._word_difference(
                    control["text"], case["reference_text"]
                )
                controls[case_id] = control
            config = dict(corpus.AUTOMATIC_ENDPOINTS_PROFILE["stream_config"])
            if profile == "hybrid":
                config.update(settings["hybrid_changes"])
            endpoints = importlib.import_module(
                "whisper_runtime.adapters.audio_endpoints"
            )
            config["endpointing"] = endpoints.QuietEndpointConfig(
                **config["endpointing"]
            )
            stream = adapters.ContinuousTranscriptStream(
                adapter,
                stream_id=f"acoustic:{profile}:{case_id}",
                mel_builder=mel,
                options=options,
                rng_seed=7,
                config=adapters.ContinuousStreamConfig(**config),
            )
            torch.cuda.synchronize(0)
            started = time.perf_counter_ns()
            events, traces, steps, chunks, error = b._drive_stream(
                stream, pcm, chunk_bytes=32000
            )
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter_ns() - started
            metrics = stream.metrics
            checks = b._event_checks(
                events,
                traces,
                accepted_samples=metrics.accepted_samples,
                total_samples=len(pcm) // 2,
                state=stream.state,
                metrics=metrics,
                budget=budget,
                worker=worker,
                capacity=capacity,
                retained_from_sample=stream.retained_from_sample,
                max_buffer_samples=40000 * 16,
                error_record=error,
            )
            checks.update(publication_checks(events, traces, pcm))
            expected_profile = (
                "word_boundary_quiet_endpoint_stream/v1+input_evidence/v1"
                if profile == "hybrid"
                else corpus.AUTOMATIC_ENDPOINTS_PROFILE["profile_id"]
            )
            checks["profile_identity"] = stream.profile_id == expected_profile
            recognition = {"text": c._normalized_committed_text(events)}
            recognition["against_human_reference"] = b._word_difference(
                recognition["text"], case["reference_text"]
            )
            recognition["against_offline_control"] = b._word_difference(
                recognition["text"], controls[case_id]["text"]
            )
            completed = error is None and stream.done and all(checks.values())
            quality = quality_gate(profile, recognition, controls[case_id])
            lifecycle = all(
                value
                for key, value in checks.items()
                if key
                not in {
                    "completion_required",
                    "final_event_once",
                    "full_input_committed_at_eof",
                }
            )
            policy_failure = (
                error is not None
                and error.get("category") == "policy_resolution"
                and lifecycle
            )
            cell = {
                "id": f"{profile}:{case_id}",
                "profile": profile,
                "case": case,
                "profile_id": stream.profile_id,
                "stream_status": "completed"
                if completed
                else "policy_failure"
                if policy_failure
                else "lifecycle_failure",
                "qualified": completed and quality,
                "quality_gate_passed": quality,
                "checks": checks,
                "error": error,
                "events": events,
                "decision_traces": traces,
                "metrics": b._plain(metrics),
                "state": b._plain(stream.state),
                "retained_from_sample": stream.retained_from_sample,
                "driver_steps": steps,
                "accepted_chunks": chunks,
                "recognition": recognition,
                "timing": {"wall_ns": elapsed, "unpaced": True, "benchmark": False},
            }
            cells.append(cell)
            stream.close()
            cell["capacity_restored_after_close"] = available()
            if not available() or cell["stream_status"] == "lifecycle_failure":
                cell["qualified"] = False
                break
        except Exception as error:
            failure = {
                "id": f"{profile}:{case_id}",
                "profile": profile,
                "case": case,
                "stream_status": "lifecycle_failure",
                "qualified": False,
                "error": b._safe_stream_error(error),
            }
            if cells and cells[-1]["id"] == failure["id"]:
                cells[-1].update(qualified=False, cleanup_error=failure["error"])
            else:
                cells.append(failure)
            break
    final = q._model_fingerprint(model) if available() else None
    qualified = (
        len(cells) == len(CELLS)
        and all(cell["qualified"] for cell in cells)
        and final == initial
    )
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    return {
        "schema_version": "1-diagnostic",
        "status": "completed" if qualified else "failed",
        "qualified": qualified,
        "claim_boundary": settings["claim_boundary"],
        "source": {
            "snapshot": expected_snapshot,
            "registration_sha256": c._sha256_file(REMOTE_ROOT / MANIFEST),
        },
        "worker": {
            "function_call_id": call_id,
            "function_call_id_sha256": _sha(call_id.encode()),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "modal": str(modal.__version__),
        },
        "inputs": inputs,
        "cells": cells,
        "controls": controls,
        "model": {
            "initial_sha256": initial,
            "final_sha256": final,
            "unchanged": final == initial,
            "backend_revision": identity.revision,
        },
        "capacity_restored": available(),
    }


def resources(expected_snapshot, root=ROOT):
    corpus = _corpus()
    modal = importlib.import_module("modal")
    if str(modal.__version__) != "1.5.5":
        raise ValueError("registered Modal SDK required")
    q = corpus._helper("infra.modal_native_cuda_qualification")
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("ca-certificates", "ffmpeg", "git")
        .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
        .uv_pip_install(*q.DIRECT_IMAGE_PACKAGES)
        .run_commands(q._build_command(corpus.BASE_COMMIT))
        .add_local_dir(
            root / "src/whisper_runtime",
            "/opt/whisper-runtime/src/whisper_runtime",
            copy=True,
        )
    )
    for item in expected_snapshot["files"]:
        if not item["path"].startswith("src/whisper_runtime/"):
            image = image.add_local_file(
                root / item["path"], (REMOTE_ROOT / item["path"]).as_posix(), copy=True
            )
    for fixture in corpus.read_registration(root)["fixtures"]:
        image = image.add_local_file(
            root / corpus.ASSET_PATH / fixture["filename"],
            (corpus.REMOTE_ASSETS / fixture["filename"]).as_posix(),
            copy=True,
        )
    image = image.env(
        {
            "PYTHONPATH": "/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "WHISPER_MODAL_ENABLE_WORD_CORPUS": "0",
            "WHISPER_MODAL_ENABLE_REMOTE_RESOURCES": "0",
        }
    )
    app = modal.App(f"whisper-acoustic-{uuid.uuid4().hex[:12]}")
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
    def echo(payload: bytes) -> bytes:
        return payload

    volume = modal.Volume.from_name(corpus.b.MODEL_CACHE_NAME, create_if_missing=False)

    @app.function(
        **common,
        cpu=2,
        memory=4096,
        gpu="T4",
        cloud="aws",
        region="us-west",
        timeout=180,
        volumes={corpus.b.MODEL_CACHE_MOUNT: volume.with_mount_options(read_only=True)},
    )
    def execute() -> bytes:
        producer = importlib.import_module("infra.modal_acoustic_diagnostic")
        return producer._corpus().b._encode_worker_record(
            producer.run_worker(expected_snapshot)
        )

    return app, echo, execute


def _paths(root, replay_id, preflight):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", replay_id) or re.fullmatch(
        r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", replay_id, re.I
    ):
        raise ValueError("invalid replay id")
    directory = root / "artifacts/modal" / replay_id
    if directory.resolve().parent != (root / "artifacts/modal").resolve():
        raise ValueError("replay path redirects")
    stem = "acoustic-preflight" if preflight else "acoustic-diagnostic"
    return (
        directory / f"{stem}.json",
        directory / f"{stem}.attempt.jsonl",
        directory / f"{stem}.result.zlib",
    )


def validate_record(record, expected_snapshot):
    if record["source"]["snapshot"] != expected_snapshot:
        raise ValueError("worker source differs")
    seen = [cell["id"] for cell in record["cells"]]
    planned = [f"{p}:{c}" for p, c in CELLS]
    if not seen or seen != planned[: len(seen)]:
        raise ValueError("worker cell order differs")
    for cell in record["cells"]:
        if cell["qualified"] and (
            cell.get("stream_status") != "completed"
            or set(cell.get("checks", {})) != CHECK_NAMES
            or any(type(value) is not bool for value in cell["checks"].values())
            or not all(cell["checks"].values())
            or not cell.get("quality_gate_passed")
            or cell.get("capacity_restored_after_close") is not True
        ):
            raise ValueError("failed cell cannot qualify")
        if "recognition" in cell:
            case_id = cell["id"].split(":", 1)[1]
            if cell["quality_gate_passed"] is not quality_gate(
                cell["profile"], cell["recognition"], record["controls"][case_id]
            ):
                raise ValueError("quality decision differs")
    qualified = (
        len(seen) == len(planned)
        and all(c["qualified"] for c in record["cells"])
        and record["model"]["unchanged"] is True
        and record["model"]["initial_sha256"] == record["model"]["final_sha256"]
        and record["capacity_restored"] is True
    )
    if record["qualified"] is not qualified or record["status"] != (
        "completed" if qualified else "failed"
    ):
        raise ValueError("false overall success")


def run(*, replay_id, preflight=False, confirm_paid_gpu=False, root=ROOT):
    if not preflight and confirm_paid_gpu is not True:
        raise ValueError("GPU work requires --confirm-paid-gpu")
    settings = registration(root)
    output, receipt, raw = _paths(root, replay_id, preflight)
    if any(path.exists() for path in (output, receipt, raw)):
        raise FileExistsError("this attempt already exists")
    expected = snapshot(root)
    inputs, _ = _cases(root)
    if not preflight:
        probe, _, _ = _paths(root, replay_id, True)
        proof = json.loads(probe.read_text())
        if (
            proof.get("status") != "completed"
            or proof.get("source_snapshot") != expected
            or proof.get("payload_sha256")
            != _sha(_corpus().b._transport_probe_payload())
        ):
            raise ValueError("matching CPU transport preflight required")
    corpus = _corpus()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "attempt-started",
                    "preflight": preflight,
                    "source_commit": expected["commit"],
                    "source_digest": expected["digest"],
                }
            )
            + "\n"
        )
    try:
        app, echo, execute = resources(expected, root)
        with app.run(detach=False):
            if preflight:
                payload = corpus.b._transport_probe_payload()
                received = echo.remote(payload)
                if received != payload:
                    raise ValueError("transport payload differs")
                record = {
                    "status": "completed",
                    "source_snapshot": expected,
                    "payload_sha256": _sha(payload),
                    "app_id": app.app_id,
                }
            else:
                payload = execute.remote()
                corpus.b._write_bytes_exclusive(raw, payload)
                record = corpus.b._decode_worker_record(
                    payload,
                    expected_snapshot=expected,
                    registration_sha256=corpus.c._sha256_file(root / MANIFEST),
                    manifest=settings,
                )
                record["app_id"] = app.app_id
                if record.get("inputs") != inputs:
                    raise ValueError("worker cases differ from local recipes")
                validate_record(record, expected)
        corpus.c._write_json_exclusive(output, record)
        with receipt.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "record-written",
                        "status": record["status"],
                        "sha256": corpus.c._sha256_file(output),
                    }
                )
                + "\n"
            )
    except BaseException as error:
        with receipt.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": "attempt-failed", "error_type": type(error).__name__}
                )
                + "\n"
            )
        raise RuntimeError("acoustic diagnostic failed; see its receipt") from None
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    args = parser.parse_args()
    print(run(**vars(args)))


if __name__ == "__main__":
    main()
