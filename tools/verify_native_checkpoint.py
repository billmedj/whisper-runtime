"""Verify a publication-boundary savepoint across two fresh CPU processes.

Requires existing tiny.en weights and an existing audio fixture of at most 15
seconds. Four greedy window decodes compare two known source units with a saved
first-unit boundary. No downloads, GPU work or exact-token restore are involved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path

from native_backend_setup import (
    BACKEND_BASE_COMMIT,
    BACKEND_PATCHED_TREE,
    sha256_file,
)
from smoke_native_whisper import fingerprint_loaded_model
from verify_native_interleaving import source_tree_for_module, verify_base_revision
from verify_native_prompt_reuse import require, result_record, verify_revision

from whisper_runtime import Budget, ModelSnapshot, ResourceVector, Worker
from whisper_runtime.adapters import (
    NativeDecodeOptions,
    NativeExecutionProfile,
    NativeWhisperAdapter,
    NativeWindowResult,
)
from whisper_runtime.adapters.continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
)

ROOT = Path(__file__).resolve().parents[1]
STREAM_ID = "cpu-checkpoint-jfk-two-units"
SAMPLE_LEN = 96
RNG_SEED = 7
SOURCE_FILES = (
    "src/whisper_runtime/state.py",
    "src/whisper_runtime/adapters/audio_endpoints.py",
    "src/whisper_runtime/adapters/continuous_stream.py",
    "src/whisper_runtime/adapters/stream_checkpoint.py",
    "src/whisper_runtime/adapters/_checkpoint_io.py",
    "src/whisper_runtime/adapters/native_whisper.py",
    "tools/verify_native_checkpoint.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Existing fixture, at most 15 seconds")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--phase", choices=("producer", "consumer"), help=argparse.SUPPRESS
    )
    parser.add_argument("--savepoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.phase is None) != (args.savepoint is None):
        parser.error("--phase and --savepoint must be supplied together")
    return args


def typed_value(value: object) -> object:
    """Copy every scalar/field and concrete dataclass type for exact comparison."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: typed_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Enum):
        return {"type": type(value).__qualname__, "value": value.value}
    if isinstance(value, tuple):
        return {"tuple": [typed_value(item) for item in value]}
    if isinstance(value, list):
        return [typed_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def drain_until(stream: ContinuousTranscriptStream, version: int) -> list[object]:
    events: list[object] = []
    for _ in range(SAMPLE_LEN + 8):
        if stream.state.version == version:
            require(not stream.active, "publication boundary retained a native run")
            return events
        require(stream.state.version < version, "stream passed the requested boundary")
        require(stream.ready, "stream stopped before the requested publication")
        events.extend(stream.step())
    raise RuntimeError("the fixed greedy step budget was exceeded")


def complete_state(stream: ContinuousTranscriptStream) -> object:
    for record in stream.state.windows:
        require(
            isinstance(record.result, NativeWindowResult), "expected speech publication"
        )
        require(bool(record.result.text.strip()), "fixture produced no speech text")
        result_record(record.result)
    return typed_value(stream.state)


def run_phase(args: argparse.Namespace, incoming: dict | None = None) -> dict:
    audio_path, checkpoint = args.audio.resolve(), args.checkpoint.resolve()
    for label, path in (("audio", audio_path), ("checkpoint", checkpoint)):
        require(path.is_file(), f"existing {label} was not found: {path}")
    require(checkpoint.stem == "tiny.en", "this bounded diagnostic requires tiny.en.pt")
    initial_sources = source_hashes()
    import numpy as np
    import torch

    # Isolate fresh source imports from pre-existing cached bytecode.
    previous_prefix = sys.pycache_prefix
    with tempfile.TemporaryDirectory(prefix="whisper-checkpoint-import-") as cache:
        sys.pycache_prefix = cache
        try:
            import whisper
            from whisper.audio import SAMPLE_RATE
        finally:
            sys.pycache_prefix = previous_prefix

    torch.set_num_threads(1)
    revision = verify_revision(whisper.__file__, args.revision)
    verify_base_revision(whisper.__file__, BACKEND_BASE_COMMIT, revision)
    tree = source_tree_for_module(whisper.__file__)
    require(
        tree == BACKEND_PATCHED_TREE, "backend differs from the pinned patched tree"
    )
    audio_digest, checkpoint_digest = sha256_file(audio_path), sha256_file(checkpoint)
    audio = whisper.load_audio(str(audio_path))
    require(0 < len(audio) <= 15 * SAMPLE_RATE, "fixture must contain 0 < audio <= 15s")
    require(bool(np.isfinite(audio).all()), "fixture contains nonfinite samples")
    pcm = np.rint(np.clip(audio, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes()
    unit_samples = len(pcm) // 2
    model = whisper.load_model(str(checkpoint), device="cpu").float().eval()
    fingerprint = fingerprint_loaded_model(model)
    identity = ModelSnapshot(checkpoint.stem, revision, "pytorch-cpu", fingerprint)

    def identity_probe(observed: object) -> ModelSnapshot:
        return ModelSnapshot(
            checkpoint.stem, revision, "pytorch-cpu", fingerprint_loaded_model(observed)
        )

    def mel_builder(content: bytes):
        samples = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768.0
        return whisper.log_mel_spectrogram(
            whisper.pad_or_trim(samples), n_mels=model.dims.n_mels
        ).float()

    pipeline = {
        "profile": "cpu-fp32-known-source-unit-checkpoint/v1",
        "backend_revision": revision,
        "backend_tree": tree,
        "tokenizer": "tiny.en English tokenizer from the verified backend tree",
        "preprocessing": "16k mono float clip/rint s16le; float32 /32768; pad_or_trim mel",
        "n_mels": model.dims.n_mels,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "source_sha256": initial_sources,
    }
    pipeline_identity = (
        "sha256:"
        + sha256(
            json.dumps(pipeline, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
    )
    capacity = ResourceVector(
        memory_bytes=1_000_000_000, compute_units=1, stream_slots=1
    )
    budget = Budget(capacity)
    worker = Worker(
        "checkpoint-cpu",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=180,
    )
    adapter = NativeWhisperAdapter(
        worker,
        model,
        identity_probe,
        NativeExecutionProfile("checkpoint-cpu", capacity),
    )
    options = NativeDecodeOptions(language="en", sample_len=SAMPLE_LEN, temperature=0.0)
    config = ContinuousStreamConfig(source_units=True, input_evidence=True)
    encoder_calls = 0

    def count_encoder(_module: object, _inputs: object, _output: object) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    def require_released() -> None:
        require(budget.available == capacity, "native capacity was not fully restored")
        require(budget.lease_count == 0, "native budget retained an active lease")
        require(worker.queue_depth == 0, "native admission queue was not released")

    def start_stream() -> ContinuousTranscriptStream:
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id=STREAM_ID,
            mel_builder=mel_builder,
            options=options,
            config=config,
            rng_seed=RNG_SEED,
        )
        stream.push(0, pcm + pcm)
        stream.seal_unit(unit_samples)
        return stream

    hook = model.encoder.register_forward_hook(count_encoder)
    try:
        if args.phase == "producer":
            with start_stream() as control:
                control_events = drain_until(control, 1)
                require_released()
                control.seal_unit(unit_samples * 2)
                control.finish_input()
                control_events.extend(drain_until(control, 2))
                require(control.done, "uninterrupted control did not finish")
                control_state = complete_state(control)
                require_released()
            with start_stream() as source:
                saved_events = drain_until(source, 1)
                require_released()
                require(
                    source.accepted_samples == unit_samples * 2, "pending PCM absent"
                )
                require(
                    source.retained_from_sample == unit_samples, "wrong replay origin"
                )
                saved_state = complete_state(source)
                digest = source.save_checkpoint(
                    args.savepoint, pipeline_identity=pipeline_identity
                )
                phase = {
                    "control_state": control_state,
                    "control_events": typed_value(control_events),
                    "saved_state": saved_state,
                    "saved_events": typed_value(saved_events),
                    "savepoint_sha256": digest,
                    "savepoint_bytes": args.savepoint.stat().st_size,
                    "saved_accepted_samples": source.accepted_samples,
                    "saved_retained_from_sample": source.retained_from_sample,
                }
            require_released()
            require(encoder_calls == 3, "producer must perform exactly three encodes")
        else:
            require(
                isinstance(incoming, dict), "consumer requires the producer receipt"
            )
            require(
                incoming["pipeline_identity"] == pipeline_identity, "pipeline changed"
            )
            with ContinuousTranscriptStream.from_checkpoint(
                adapter,
                path=args.savepoint,
                expected_sha256=incoming["savepoint_sha256"],
                pipeline_identity=pipeline_identity,
                mel_builder=mel_builder,
            ) as restored:
                require(
                    typed_value(restored.state) == incoming["saved_state"],
                    "saved state changed",
                )
                require(
                    restored.accepted_samples == unit_samples * 2,
                    "saved input was lost",
                )
                require(
                    restored.retained_from_sample == unit_samples,
                    "saved origin changed",
                )
                require(restored.expected_chunk == 1, "saved input sequence changed")
                require(not restored.input_finished, "saved EOF flag changed")
                decoded_before = restored.metrics.decoded_source_samples
                restored.seal_unit(unit_samples * 2)
                restored.finish_input()
                new_events = drain_until(restored, 2)
                require(restored.done, "restored stream did not finish")
                phase = {
                    "restored_state": complete_state(restored),
                    "new_events": typed_value(new_events),
                    "redecoded_samples": (
                        restored.metrics.decoded_source_samples - decoded_before
                    ),
                    "replay_interval_samples": [unit_samples, unit_samples * 2],
                    "final_accepted_samples": restored.accepted_samples,
                    "final_committed_samples": restored.metrics.committed_samples,
                    "final_buffered_samples": restored.metrics.buffered_samples,
                }
                require_released()
            require(encoder_calls == 1, "consumer must perform exactly one encode")
    finally:
        hook.remove()
    require(fingerprint_loaded_model(model) == fingerprint, "model state changed")
    require(sha256_file(audio_path) == audio_digest, "fixture file changed")
    require(sha256_file(checkpoint) == checkpoint_digest, "weights file changed")
    require(
        source_hashes() == initial_sources, "diagnostic source changed during execution"
    )
    require_released()
    return {
        "status": "passed",
        "phase": args.phase,
        "pid": os.getpid(),
        "encoder_forwards": encoder_calls,
        "resources_released": True,
        "backend": {"revision": revision, "tree": tree, "tracked_source_clean": True},
        "model": {
            "path": str(checkpoint),
            "sha256": checkpoint_digest,
            "fingerprint": fingerprint,
        },
        "audio": {
            "path": str(audio_path),
            "sha256": audio_digest,
            "unit_samples": unit_samples,
            "pcm_sha256": sha256(pcm).hexdigest(),
        },
        "pipeline_identity": pipeline_identity,
        "pipeline": pipeline,
        "configuration": {
            "options": typed_value(options),
            "config": typed_value(config),
            "rng_seed": RNG_SEED,
            "cpu_threads": 1,
        },
        **phase,
    }


def combine_receipts(producer: dict, consumer: dict) -> dict:
    checks = {
        "different_processes": producer["pid"] != consumer["pid"],
        "same_pipeline": producer["pipeline_identity"] == consumer["pipeline_identity"],
        "same_model_and_audio": producer["model"] == consumer["model"]
        and producer["audio"] == consumer["audio"],
        "exact_final_state_and_nested_provenance": producer["control_state"]
        == consumer["restored_state"],
        "exact_events": producer["saved_events"] + consumer["new_events"]
        == producer["control_events"],
        "four_encoder_forwards": producer["encoder_forwards"] == 3
        and consumer["encoder_forwards"] == 1,
        "bounded_retained_replay": consumer["redecoded_samples"]
        == producer["audio"]["unit_samples"],
        "all_input_committed": consumer["final_committed_samples"]
        == consumer["final_accepted_samples"]
        and consumer["final_buffered_samples"] == 0,
        "all_capacity_released": producer["resources_released"]
        and consumer["resources_released"],
    }
    return {
        "schema_version": "1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "cpu_publication_boundary_new_process_bounded_replay",
        "checks": checks,
        "producer": producer,
        "consumer": consumer,
        "source_process_exited_before_restore": True,
        "model_downloaded": False,
        "gpu_used": False,
        "exact_token_state_restore": False,
        "durable_input_acknowledgement": False,
        "timing_is_benchmark": False,
        "recognition_accuracy_qualification": False,
    }


def verify(args: argparse.Namespace) -> dict:
    for path in (args.audio, args.checkpoint):
        require(path.resolve().is_file(), f"existing input was not found: {path}")
    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="whisper-checkpoint-cpu-") as directory:
        savepoint = Path(directory) / "publication.savepoint"
        common = [
            sys.executable,
            "-B",
            script,
            str(args.audio.resolve()),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--revision",
            args.revision,
            "--savepoint",
            str(savepoint),
        ]
        receipts: list[dict] = []
        for phase in ("producer", "consumer"):
            completed = subprocess.run(
                [*common, "--phase", phase],
                input=json.dumps(receipts[0], allow_nan=False) if receipts else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=600,
            )
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            require(
                completed.returncode == 0, f"{phase} process failed: {completed.stdout}"
            )
            receipt = json.loads(completed.stdout)
            require(receipt.get("status") == "passed", f"{phase} receipt failed")
            receipts.append(receipt)
        return combine_receipts(*receipts)


def main() -> int:
    args = parse_args()
    try:
        if args.phase:
            incoming = json.load(sys.stdin) if args.phase == "consumer" else None
            report = run_phase(args, incoming)
        else:
            report = verify(args)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        report = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
