"""Decode one local audio file with the verified native CPU backend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from tools.native_backend_setup import (
    NativeSetupError,
    installed_versions,
    load_validated_setup,
    require_cached_checkpoint,
    verify_dependency_versions,
)
from tools.smoke_native_whisper import (
    fingerprint_loaded_model,
    verify_loaded_model_fingerprint,
    verify_source_revision,
    verify_terminal_invariants,
)
from whisper_runtime import (
    Budget,
    ModelSnapshot,
    RequestState,
    ResourceVector,
    Session,
    Worker,
)
from whisper_runtime.adapters import (
    NativeDecodeOptions,
    NativeExecutionProfile,
    NativeWhisperAdapter,
)

MODEL_NAME = "tiny.en"
MODEL_STATE_FINGERPRINT = (
    "sha256:8041a80119a588f542472da35e97d0372fce1d9709ed9874475e9c03deac5de6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--expected-text")
    return parser.parse_args()


def model_url(models: object, name: str) -> str:
    if not isinstance(models, Mapping):
        raise NativeSetupError("Whisper does not expose its model registry")
    value = models.get(name)
    if not isinstance(value, str):
        raise NativeSetupError(f"Whisper does not define the {name!r} checkpoint")
    return value


def main() -> int:
    args = parse_args()
    setup = load_validated_setup(args.manifest.resolve())
    verify_dependency_versions(installed_versions())
    sys.path.insert(0, str(setup.paths.backend))

    import whisper
    from whisper.audio import SAMPLE_RATE

    revision = verify_source_revision(whisper.__file__, setup.backend.commit)
    url = model_url(getattr(whisper, "_MODELS", None), MODEL_NAME)
    checkpoint = require_cached_checkpoint(
        MODEL_NAME,
        url,
        args.model_cache,
        allow_download=args.allow_model_download,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(
        MODEL_NAME,
        device="cpu",
        download_root=str(checkpoint.parent),
    ).eval()
    fingerprint = fingerprint_loaded_model(model)
    verify_loaded_model_fingerprint(fingerprint, MODEL_STATE_FINGERPRINT)
    snapshot = ModelSnapshot(
        model_id=MODEL_NAME,
        revision=revision,
        backend="pytorch-cpu",
        fingerprint=fingerprint,
    )

    def identity_probe(observed: object) -> ModelSnapshot:
        return ModelSnapshot(
            model_id=MODEL_NAME,
            revision=revision,
            backend="pytorch-cpu",
            fingerprint=fingerprint_loaded_model(observed),
        )

    audio = whisper.load_audio(str(args.audio))
    duration_ms = min(round(len(audio) * 1_000 / SAMPLE_RATE), 30_000)
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(audio),
        n_mels=model.dims.n_mels,
    )
    capacity = ResourceVector(
        memory_bytes=1_000_000_000,
        compute_units=1,
        stream_slots=1,
    )
    budget = Budget(capacity)
    worker = Worker(
        "native-example",
        snapshot,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=300,
    )
    adapter = NativeWhisperAdapter(
        worker,
        model,
        identity_probe,
        NativeExecutionProfile("tiny.en/cpu", capacity),
    )
    session = Session("native-example")
    request = RequestState(
        "native-example-1",
        session.session_id,
        snapshot,
        rng_seed=7,
    )

    started = time.perf_counter()
    state = adapter.decode_window(
        session=session,
        request=request,
        window_id="window-0",
        mel=mel,
        start_ms=0,
        end_ms=duration_ms,
        options=NativeDecodeOptions(
            language="en",
            temperature=0.0,
            without_timestamps=True,
        ),
    )
    elapsed = time.perf_counter() - started
    result = state.windows[-1].result
    if args.expected_text is not None and result.text != args.expected_text:
        raise RuntimeError(
            f"transcript mismatch: expected {args.expected_text!r}, "
            f"observed {result.text!r}"
        )
    verify_terminal_invariants(
        request_status=request.status.value,
        session_version=state.version,
        queue_depth=worker.queue_depth,
        available=budget.available,
        capacity=capacity,
    )
    print(
        json.dumps(
            {
                "backend_commit": revision,
                "backend_tree": setup.backend.tree,
                "model": MODEL_NAME,
                "model_fingerprint": fingerprint,
                "request_status": request.status.value,
                "session_version": state.version,
                "text": result.text,
                "elapsed_seconds": elapsed,
                "queue_depth_after": worker.queue_depth,
                "resources_released": budget.available == capacity,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NativeSetupError as error:
        raise SystemExit(f"native example failed: {error}") from error
