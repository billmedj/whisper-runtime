"""Decode one local audio file with the verified native CPU backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
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
    SessionState,
    Worker,
)
from whisper_runtime.adapters import (
    NativeDecodeOptions,
    NativeExecutionProfile,
    NativeStreamConfig,
    NativeTranscriptStream,
    NativeWhisperAdapter,
    TranscriptEvent,
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
    parser.add_argument(
        "--stream-preview-ms",
        type=int,
        help="Replay the file as PCM and revise the transcript at this interval",
    )
    parser.add_argument("--stream-chunk-ms", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, help="Write the run report to this JSON file"
    )
    return parser.parse_args()


def model_url(models: object, name: str) -> str:
    if not isinstance(models, Mapping):
        raise NativeSetupError("Whisper does not expose its model registry")
    value = models.get(name)
    if not isinstance(value, str):
        raise NativeSetupError(f"Whisper does not define the {name!r} checkpoint")
    return value


def run_stream(
    *,
    adapter: NativeWhisperAdapter,
    pcm_s16le: bytes,
    mel_builder: Callable[[bytes], object],
    preview_interval_ms: int,
    chunk_ms: int,
) -> tuple[SessionState, list[dict[str, object]], dict[str, object]]:
    """Replay decoded audio through the bounded PCM stream profile."""

    config = NativeStreamConfig(preview_interval_ms=preview_interval_ms)
    if not 0 < chunk_ms <= config.max_audio_ms:
        raise ValueError("stream-chunk-ms must be between 1 and 30000")
    sample_count = len(pcm_s16le) // 2
    if not pcm_s16le or len(pcm_s16le) % 2 or sample_count > config.max_audio_samples:
        raise ValueError("stream input must contain 1 sample to 30 seconds of PCM")
    options = NativeDecodeOptions(
        language="en", temperature=0.0, without_timestamps=True
    )
    events: list[TranscriptEvent] = []
    with NativeTranscriptStream(
        adapter,
        stream_id="native-stream-example",
        mel_builder=mel_builder,
        options=options,
        rng_seed=7,
        config=config,
    ) as stream:
        chunk_bytes = config.sample_rate_hz * chunk_ms // 1000 * 2
        for sequence, offset in enumerate(range(0, len(pcm_s16le), chunk_bytes)):
            stream.push(sequence, pcm_s16le[offset : offset + chunk_bytes])
            while stream.ready:
                events.extend(stream.step())
        stream.finish_input()
        while stream.ready:
            events.extend(stream.step())
    state = stream.state
    metrics = stream.metrics

    # Use the same PCM, preprocessing, options, and seed for the final control.
    control_session = Session("native-stream-control")
    control_request = RequestState(
        "native-stream-control",
        control_session.session_id,
        adapter.model_identity,
        rng_seed=7,
    )
    end_ms = sample_count * 1000 // config.sample_rate_hz
    control_state = adapter.decode_window(
        session=control_session,
        request=control_request,
        window_id="control",
        mel=mel_builder(pcm_s16le),
        start_ms=0,
        end_ms=end_ms,
        options=options,
        committed_through_ms=end_ms,
    )
    if state.windows[-1].result.text != control_state.windows[-1].result.text:
        raise RuntimeError("stream final text differs from the same-PCM control")
    return (
        state,
        [asdict(event) for event in events],
        {
            "profile_id": stream.profile_id,
            "pcm_sha256": hashlib.sha256(pcm_s16le).hexdigest(),
            "sample_count": sample_count,
            "chunk_ms": chunk_ms,
            "preview_interval_ms": preview_interval_ms,
            "accepted_audio_ms": metrics.accepted_audio_ms,
            "decoded_source_audio_ms": metrics.decoded_source_audio_ms,
            "source_reprocessing_factor": metrics.source_reprocessing_factor,
            "decode_count": metrics.decode_count,
            "same_pcm_control_text": control_state.windows[-1].result.text,
            "same_pcm_control_matches": True,
        },
    )


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
    duration_ms = min(len(audio) * 1_000 // SAMPLE_RATE, 30_000)
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
    started = time.perf_counter()
    stream_events: list[dict[str, object]] | None = None
    stream_metrics: dict[str, object] | None = None
    if args.stream_preview_ms is None:
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio),
            n_mels=model.dims.n_mels,
        )
        session = Session("native-example")
        request = RequestState(
            "native-example-1",
            session.session_id,
            snapshot,
            rng_seed=7,
        )
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
        request_status = request.status.value
    else:
        import numpy as np

        pcm_s16le = np.rint(audio * 32768.0).clip(-32768, 32767).astype("<i2").tobytes()

        def mel_builder(content: bytes) -> object:
            decoded = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768.0
            return whisper.log_mel_spectrogram(
                whisper.pad_or_trim(decoded),
                n_mels=model.dims.n_mels,
            )

        state, stream_events, stream_metrics = run_stream(
            adapter=adapter,
            pcm_s16le=pcm_s16le,
            mel_builder=mel_builder,
            preview_interval_ms=args.stream_preview_ms,
            chunk_ms=args.stream_chunk_ms,
        )
        request_status = "committed"
    elapsed = time.perf_counter() - started
    result = state.windows[-1].result
    if args.expected_text is not None and result.text != args.expected_text:
        raise RuntimeError(
            f"transcript mismatch: expected {args.expected_text!r}, "
            f"observed {result.text!r}"
        )
    if args.stream_preview_ms is None:
        verify_terminal_invariants(
            request_status=request_status,
            session_version=state.version,
            queue_depth=worker.queue_depth,
            available=budget.available,
            capacity=capacity,
        )
    else:
        if state.committed_through_ms != duration_ms:
            raise RuntimeError("the stream did not commit the full input")
        if worker.queue_depth != 0 or budget.available != capacity:
            raise RuntimeError("the stream retained runtime capacity")
    report = json.dumps(
        {
            "runtime_commit": setup.runtime.commit,
            "backend_commit": revision,
            "backend_tree": setup.backend.tree,
            "model": MODEL_NAME,
            "model_fingerprint": fingerprint,
            "request_status": request_status,
            "session_version": state.version,
            "text": result.text,
            "stream_events": stream_events,
            "stream_metrics": stream_metrics,
            "elapsed_seconds": elapsed,
            "queue_depth_after": worker.queue_depth,
            "resources_released": budget.available == capacity,
        },
        ensure_ascii=False,
        indent=2,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NativeSetupError as error:
        raise SystemExit(f"native example failed: {error}") from error
