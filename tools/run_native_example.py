"""Run the native example in the environment created by the bootstrap command."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from native_backend_setup import (
    RUNTIME_ROOT,
    NativeSetupError,
    SetupPaths,
    load_validated_setup,
)

JFK_TRANSCRIPT = (
    "And so my fellow Americans ask not what your country can do for you, "
    "ask what you can do for your country."
)


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
        default=RUNTIME_ROOT / ".tmp-native",
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
    if audio == default_audio.resolve():
        command.extend(("--expected-text", JFK_TRANSCRIPT))

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(setup.paths.backend),
                    str(RUNTIME_ROOT / "src"),
                    str(RUNTIME_ROOT),
                )
            ),
        }
    )
    completed = subprocess.run(command, cwd=RUNTIME_ROOT, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
