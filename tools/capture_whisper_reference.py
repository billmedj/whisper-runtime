from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one OpenAI Whisper result as a conformance fixture."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument(
        "--profile", choices=("reference", "optimized"), default="reference"
    )
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_paths(root: Path, *arguments: str) -> set[str]:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-c",
            "core.quotepath=false",
            "-C",
            str(root),
            *arguments,
        ],
        check=True,
        capture_output=True,
    )
    return {
        entry.decode("utf-8", errors="surrogateescape")
        for entry in result.stdout.split(b"\0")
        if entry
    }


def source_tree_digest(root: Path, relative_paths: set[str]) -> str:
    """Hash a portable inventory of source paths and file content."""
    digest = hashlib.sha256(b"whisper-runtime-source-tree-v1\0")
    for relative in sorted(
        relative_paths,
        key=lambda value: value.encode("utf-8", errors="surrogateescape"),
    ):
        path = root / Path(relative)
        name = relative.replace("\\", "/").encode("utf-8", errors="surrogateescape")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        if path.is_symlink():
            content = os.readlink(path).encode("utf-8", errors="surrogateescape")
            digest.update(b"L")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(hashlib.sha256(content).digest())
        elif path.is_file():
            digest.update(b"F")
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(sha256_file(path)))
        else:
            digest.update(b"M")
            digest.update((0).to_bytes(8, "big"))
            digest.update(hashlib.sha256(b"").digest())
    return digest.hexdigest()


def _relative_if_within(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _path_key(path: str) -> str:
    portable = path.replace("\\", "/")
    return portable.casefold() if os.name == "nt" else portable


def git_metadata(
    root: Path,
    *,
    excluded_paths: set[str] | None = None,
) -> tuple[str | None, bool | None, str]:
    excluded = {_path_key(path) for path in (excluded_paths or set())}
    try:
        commit = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked = _git_paths(root, "ls-files", "-z")
        untracked = _git_paths(root, "ls-files", "--others", "--exclude-standard", "-z")
        changed = _git_paths(root, "diff", "HEAD", "--name-only", "-z")
        included = {
            path.replace("\\", "/")
            for path in tracked | untracked
            if _path_key(path) not in excluded
        }
        relevant_changes = {
            path.replace("\\", "/")
            for path in changed | untracked
            if _path_key(path) not in excluded
        }
        return commit, bool(relevant_changes), source_tree_digest(root, included)
    except (OSError, subprocess.CalledProcessError):
        included = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and ".git" not in path.relative_to(root).parts
            and _path_key(path.relative_to(root).as_posix()) not in excluded
        }
        return None, None, source_tree_digest(root, included)


def json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def checkpoint_digest(model_name: str) -> str | None:
    cache_root = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
    )
    checkpoint = cache_root / f"{model_name}.pt"
    return sha256_file(checkpoint) if checkpoint.is_file() else None


def portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve(strict=True)
    audio = args.audio.resolve(strict=True)
    output = args.output.resolve()

    sys.path.insert(0, str(source_root))
    import torch  # type: ignore[import-not-found]
    import whisper  # type: ignore[import-not-found]
    from whisper.audio import SAMPLE_RATE  # type: ignore[import-not-found]

    torch.manual_seed(args.seed)
    model = whisper.load_model(args.model, device=args.device)
    options: dict[str, Any] = {
        "language": args.language,
        "task": args.task,
        "temperature": args.temperature,
        "word_timestamps": args.word_timestamps,
    }
    if args.beam_size is not None:
        options["beam_size"] = args.beam_size

    started = time.perf_counter()
    result = model.transcribe(str(audio), **options)
    wall_seconds = time.perf_counter() - started
    audio_samples = whisper.load_audio(str(audio))
    audio_seconds = len(audio_samples) / SAMPLE_RATE
    model_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")

    audio_relative = _relative_if_within(audio, source_root)
    excluded_paths = {audio_relative} if audio_relative is not None else set()
    commit, dirty, tree_sha256 = git_metadata(
        source_root,
        excluded_paths=excluded_paths,
    )
    fixture = {
        "schema_version": "1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "outcome": "success",
        "profile": args.profile,
        "source": {
            "root": source_root.name,
            "git_commit": commit,
            "dirty": dirty,
            "tree_sha256": tree_sha256,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "whisper_module": portable_path(
                Path(whisper.__file__).resolve(), source_root
            ),
        },
        "model": {
            "name": args.model,
            "device": str(model.device),
            "dtype": model_dtype,
            "checkpoint_sha256": checkpoint_digest(args.model),
            "dimensions": json_value(model.dims),
        },
        "audio": {
            "fixture_id": args.fixture_id,
            "path": portable_path(audio, source_root),
            "sha256": sha256_file(audio),
            "size_bytes": audio.stat().st_size,
            "sample_rate_hz": SAMPLE_RATE,
            "sample_start": 0,
            "sample_end": len(audio_samples),
        },
        "options": {**options, "seed": args.seed},
        "comparison": {
            "timestamp_abs_tol": 1e-6,
            "numeric_abs_tol": 1e-6,
        },
        "measurement": {
            "wall_seconds": wall_seconds,
            "real_time_factor": wall_seconds / audio_seconds if audio_seconds else None,
            "peak_host_memory_bytes": None,
            "peak_device_memory_bytes": None,
            "queue_delay_seconds": 0.0,
            "execution_seconds": wall_seconds,
            "encoder_calls": None,
            "decoder_steps": None,
            "fallback_attempts": None,
            "peak_hypothesis_count": None,
            "alignment_seconds": None,
            "alignment_capture_bytes": None,
            "encoder_batch_size_peak": None,
            "decoder_batch_size_peak": None,
            "active_requests_peak": None,
            "state_isolation_failures": None,
            "admission_delay_seconds": None,
            "cancellation_delay_seconds": None,
            "resources_held_after_bytes": None,
        },
        "result": json_value(result),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")
    print(f"wall_seconds={wall_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
