"""Run one real Whisper window through the native transactional adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _update_field(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def fingerprint_loaded_model(model: object) -> str:
    """Hash the names, layouts, and bytes of the live model state."""

    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("the loaded model must provide state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("model.state_dict() must return a mapping")

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        detach = getattr(tensor, "detach", None)
        if not callable(detach):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        value = detach().cpu().contiguous()
        _update_field(digest, str(name))
        _update_field(digest, str(value.dtype))
        _update_field(digest, repr(tuple(value.shape)))
        digest.update(value.numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def verify_loaded_model_fingerprint(observed: str, expected: str | None) -> None:
    """Reject a loaded model state that differs from the declared identity."""

    if expected is not None and observed != expected:
        raise RuntimeError(
            "the loaded model fingerprint does not match "
            "--expected-model-fingerprint: "
            f"expected {expected}, observed {observed}"
        )


def verify_terminal_invariants(
    *,
    request_status: str,
    session_version: int,
    queue_depth: int,
    available: ResourceVector,
    capacity: ResourceVector,
) -> None:
    """Require one committed publication and complete runtime cleanup."""

    if request_status != "committed":
        raise RuntimeError(
            f"the request did not reach committed status: observed {request_status}"
        )
    if session_version != 1:
        raise RuntimeError(
            f"the first window did not publish session version 1: {session_version}"
        )
    if queue_depth != 0:
        raise RuntimeError(
            "the worker retained an admission slot after commit: "
            f"queue depth {queue_depth}"
        )
    if available != capacity:
        raise RuntimeError(
            "the declared resource budget was not restored after commit: "
            f"expected {capacity!r}, observed {available!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode one local audio file through the native CPU adapter."
    )
    parser.add_argument("audio", help="Path to an audio file accepted by Whisper")
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--download-root")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--expected-text")
    parser.add_argument(
        "--expected-model-fingerprint",
        help="Expected sha256:<hex> fingerprint of the loaded model state",
    )
    return parser.parse_args()


def verify_source_revision(module_file: str, expected_revision: str) -> str:
    """Bind the reported revision to the imported Whisper worktree."""

    module_path = Path(module_file).resolve()
    source_root = module_path.parent.parent
    top_level = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if top_level.returncode != 0:
        raise RuntimeError(
            "the imported Whisper source is not in a readable Git worktree"
        )
    observed_root = Path(top_level.stdout.strip()).resolve()
    if observed_root != source_root:
        raise RuntimeError(
            "the imported Whisper package is not rooted at the detected Git worktree"
        )

    relative_module = module_path.relative_to(source_root).as_posix()
    tracked_module = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "--error-unmatch",
            "--",
            relative_module,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if tracked_module.returncode != 0:
        raise RuntimeError("the imported Whisper module is not tracked by Git")

    package_path = module_path.parent.relative_to(source_root).as_posix()
    ignored_package_files = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            package_path,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if ignored_package_files.returncode != 0:
        raise RuntimeError("the imported Whisper package ignore state is unavailable")
    if ignored_package_files.stdout.strip():
        raise RuntimeError(
            "the imported Whisper package contains ignored files; its Git revision "
            "does not fully identify the executed source"
        )

    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if revision.returncode != 0:
        raise RuntimeError(
            "the imported Whisper source is not in a readable Git worktree"
        )
    observed_revision = revision.stdout.strip()
    if observed_revision != expected_revision:
        raise RuntimeError(
            "the imported Whisper revision does not match --revision: "
            f"expected {expected_revision}, observed {observed_revision}"
        )

    status = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if status.returncode != 0:
        raise RuntimeError("the imported Whisper worktree status is unavailable")
    if status.stdout.strip():
        raise RuntimeError(
            "the imported Whisper worktree has source changes; its Git revision "
            "does not fully identify the executed source"
        )
    return observed_revision


def main() -> int:
    args = parse_args()

    import whisper
    from whisper.audio import SAMPLE_RATE

    revision = verify_source_revision(whisper.__file__, args.revision)

    model = whisper.load_model(
        args.model,
        device="cpu",
        download_root=args.download_root,
    ).eval()
    fingerprint = fingerprint_loaded_model(model)
    verify_loaded_model_fingerprint(fingerprint, args.expected_model_fingerprint)
    snapshot = ModelSnapshot(
        model_id=args.model,
        revision=revision,
        backend="pytorch-cpu",
        fingerprint=fingerprint,
    )

    def identity_probe(observed: object) -> ModelSnapshot:
        return ModelSnapshot(
            model_id=args.model,
            revision=revision,
            backend="pytorch-cpu",
            fingerprint=fingerprint_loaded_model(observed),
        )

    audio = whisper.load_audio(args.audio)
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
        "native-cpu-smoke",
        snapshot,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=120,
    )
    adapter = NativeWhisperAdapter(
        worker,
        model,
        identity_probe,
        NativeExecutionProfile("native-cpu-smoke", capacity),
    )
    session = Session("native-cpu-smoke")
    request = RequestState(
        "native-cpu-smoke-1",
        session.session_id,
        snapshot,
        rng_seed=args.rng_seed,
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
            language="en" if args.model.endswith(".en") else None,
            temperature=0.0,
            without_timestamps=True,
        ),
    )
    elapsed = time.perf_counter() - started
    result = state.windows[-1].result
    if args.expected_text is not None and result.text != args.expected_text:
        raise RuntimeError(
            "the committed transcript does not match --expected-text: "
            f"expected {args.expected_text!r}, observed {result.text!r}"
        )
    available = budget.available
    verify_terminal_invariants(
        request_status=request.status.value,
        session_version=state.version,
        queue_depth=worker.queue_depth,
        available=available,
        capacity=capacity,
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "revision": revision,
                "fingerprint": fingerprint,
                "request_status": request.status.value,
                "session_version": state.version,
                "window_id": result.window_id,
                "text": result.text,
                "expected_text_matched": (None if args.expected_text is None else True),
                "elapsed_seconds": elapsed,
                "queue_depth_after": 0,
                "resources_released": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
