"""Run the native example in the environment created by the bootstrap command."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from native_backend_setup import (
    DEFAULT_SETUP_ROOT,
    RUNTIME_ROOT,
    NativeSetupError,
    SetupPaths,
    isolated_command_environment,
    load_validated_setup,
)

JFK_TRANSCRIPT = (
    "And so my fellow Americans ask not what your country can do for you, "
    "ask what you can do for your country."
)


def native_example_environment(
    backend: Path,
    *,
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build an import path without ambient Python or pip injection settings."""

    environment = isolated_command_environment(source)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(backend), str(RUNTIME_ROOT / "src"), str(RUNTIME_ROOT))
    )
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one tiny.en CPU transaction with the verified native backend. "
            "The default audio is the JFK file in the cloned Whisper test tree."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_SETUP_ROOT,
        help="Setup directory created by bootstrap_native_backend.py",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Local audio file (default: backend/tests/jfk.flac)",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        help="Checkpoint directory (default: <root>/models)",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Whisper to download tiny.en into the model cache if needed",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stream-preview-ms",
        type=int,
        help="Replay PCM through the bounded stream profile at this interval",
    )
    mode.add_argument(
        "--segment-publication-check",
        action="store_true",
        help="Verify whole-segment publication against a timestamp-enabled control",
    )
    parser.add_argument("--stream-chunk-ms", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, help="Write the run report to this JSON file"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = SetupPaths.from_root(args.root)
    try:
        setup = load_validated_setup(paths.manifest)
    except NativeSetupError as error:
        raise SystemExit(f"cannot run native example: {error}") from error

    default_audio = setup.paths.backend / "tests" / "jfk.flac"
    audio = (args.audio or default_audio).expanduser().resolve()
    if not audio.is_file():
        raise SystemExit(f"audio file not found: {audio}")
    model_cache = (
        (args.model_cache or (setup.paths.root / "models")).expanduser().resolve()
    )

    command = [
        str(setup.python),
        "-B",
        str(RUNTIME_ROOT / "examples" / "native_transaction.py"),
        "--manifest",
        str(paths.manifest),
        "--audio",
        str(audio),
        "--model-cache",
        str(model_cache),
    ]
    if args.allow_model_download:
        command.append("--allow-model-download")
    if args.stream_preview_ms is not None:
        command.extend(("--stream-preview-ms", str(args.stream_preview_ms)))
        command.extend(("--stream-chunk-ms", str(args.stream_chunk_ms)))
    if args.segment_publication_check:
        command.append("--segment-publication-check")
    if args.output is not None:
        command.extend(("--output", str(args.output.resolve())))
    if audio == default_audio.resolve() and not args.segment_publication_check:
        command.extend(("--expected-text", JFK_TRANSCRIPT))

    environment = native_example_environment(setup.paths.backend)
    try:
        completed = subprocess.run(
            command,
            cwd=RUNTIME_ROOT,
            env=environment,
            check=False,
        )
    except OSError as error:
        raise SystemExit(f"cannot start setup Python: {error}") from error
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
