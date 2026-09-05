"""Prepare the frozen speech corpus used by the word-publication diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "modal-word-corpus-v1.json"
DEFAULT_SOURCE_DIR = ROOT / "artifacts" / "corpus" / "librispeech-test-clean-mini"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "speech-corpus-v1"

DATASET_ID = "openslr/librispeech_asr"
DATASET_CONFIGURATION = "clean"
DATASET_SPLIT = "test"
DATASET_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
ROWS_HOST = "datasets-server.huggingface.co"
MAX_METADATA_BYTES = 256 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
ID_PATTERN = re.compile(r"[0-9]+-[0-9]+-[0-9]+\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class CorpusError(RuntimeError):
    """A corpus input does not match its frozen registration."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    filename: str
    sha256: str
    size_bytes: int
    metadata_url: str
    row_offset: int
    speaker_id: int
    chapter_id: int


@dataclass(frozen=True, slots=True)
class Fixture:
    fixture_id: str
    filename: str
    pcm_sha256: str
    sample_count: int
    reference_text: str
    source: SourceRecord


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CorpusError(f"{label} must be an object")
    return value


def _string(record: dict[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise CorpusError(f"{label}.{key} must be a non-empty string")
    return value


def _integer(record: dict[str, Any], key: str, label: str, *, minimum: int) -> int:
    value = record.get(key)
    if type(value) is not int or value < minimum:
        raise CorpusError(f"{label}.{key} must be an integer >= {minimum}")
    return value


def _sha256(value: str, label: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise CorpusError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_metadata_url(url: str, row_offset: int) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise CorpusError("source.metadata_url is not a valid URL") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != ROWS_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path != "/rows"
        or parsed.fragment
    ):
        raise CorpusError("source.metadata_url must use the registered HTTPS rows API")
    query = urllib.parse.parse_qs(
        parsed.query, keep_blank_values=True, strict_parsing=True
    )
    expected = {
        "dataset": [DATASET_ID],
        "config": [DATASET_CONFIGURATION],
        "split": [DATASET_SPLIT],
        "offset": [str(row_offset)],
        "length": ["1"],
    }
    if query != expected:
        raise CorpusError("source.metadata_url query does not match the fixture row")


def load_manifest(path: Path) -> tuple[Fixture, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read manifest {path}: {error}") from error
    root = _object(document, "manifest")
    if root.get("manifest_version") != "1":
        raise CorpusError("manifest_version must be '1'")
    if root.get("manifest_id") != "modal-word-corpus-v1":
        raise CorpusError("manifest_id must be 'modal-word-corpus-v1'")

    dataset = _object(root.get("dataset"), "dataset")
    required_dataset = {
        "id": DATASET_ID,
        "configuration": DATASET_CONFIGURATION,
        "split": DATASET_SPLIT,
        "revision": DATASET_REVISION,
        "license": "CC BY 4.0",
    }
    for key, expected in required_dataset.items():
        if dataset.get(key) != expected:
            raise CorpusError(f"dataset.{key} does not match the frozen corpus")

    audio_format = _object(root.get("audio_format"), "audio_format")
    if (
        audio_format.get("sample_rate_hz") != 16_000
        or audio_format.get("channels") != 1
        or audio_format.get("encoding") != "pcm_s16le"
    ):
        raise CorpusError("audio_format must be mono 16 kHz PCM s16le")

    raw_fixtures = root.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise CorpusError("fixtures must be a non-empty array")
    fixtures: list[Fixture] = []
    seen: set[str] = set()
    for index, raw_fixture in enumerate(raw_fixtures):
        label = f"fixtures[{index}]"
        item = _object(raw_fixture, label)
        fixture_id = _string(item, "id", label)
        if ID_PATTERN.fullmatch(fixture_id) is None or fixture_id in seen:
            raise CorpusError(f"{label}.id is invalid or duplicated")
        seen.add(fixture_id)
        filename = _string(item, "filename", label)
        if filename != f"{fixture_id}.pcm":
            raise CorpusError(f"{label}.filename must be {fixture_id}.pcm")
        reference = _string(item, "reference_text", label)
        if reference != reference.strip():
            raise CorpusError(f"{label}.reference_text has surrounding whitespace")

        raw_source = _object(item.get("source"), f"{label}.source")
        source_filename = _string(raw_source, "filename", f"{label}.source")
        if source_filename != f"{fixture_id}.flac":
            raise CorpusError(f"{label}.source.filename must be {fixture_id}.flac")
        row_offset = _integer(raw_source, "row_offset", f"{label}.source", minimum=0)
        metadata_url = _string(raw_source, "metadata_url", f"{label}.source")
        _validate_metadata_url(metadata_url, row_offset)
        source = SourceRecord(
            filename=source_filename,
            sha256=_sha256(
                _string(raw_source, "sha256", f"{label}.source"),
                f"{label}.source.sha256",
            ),
            size_bytes=_integer(raw_source, "size_bytes", f"{label}.source", minimum=1),
            metadata_url=metadata_url,
            row_offset=row_offset,
            speaker_id=_integer(raw_source, "speaker_id", f"{label}.source", minimum=0),
            chapter_id=_integer(raw_source, "chapter_id", f"{label}.source", minimum=0),
        )
        if source.size_bytes > MAX_SOURCE_BYTES:
            raise CorpusError(f"{label}.source.size_bytes exceeds the 1 MiB limit")
        fixtures.append(
            Fixture(
                fixture_id=fixture_id,
                filename=filename,
                pcm_sha256=_sha256(
                    _string(item, "pcm_sha256", label), f"{label}.pcm_sha256"
                ),
                sample_count=_integer(item, "sample_count", label, minimum=1),
                reference_text=reference,
                source=source,
            )
        )
    return tuple(fixtures)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _verify_file(path: Path, size: int, digest: str, label: str) -> None:
    if not path.is_file():
        raise CorpusError(f"{label} is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise CorpusError(f"{label} size mismatch: expected {size}, got {actual_size}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(128 * 1024), b""):
            hasher.update(block)
    actual_digest = hasher.hexdigest()
    if actual_digest != digest:
        raise CorpusError(
            f"{label} digest mismatch: expected {digest}, got {actual_digest}"
        )


def _run(command: list[str], label: str) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=60)
    except FileNotFoundError as error:
        raise CorpusError(f"{command[0]} is required to {label}") from error
    except subprocess.TimeoutExpired as error:
        raise CorpusError(f"{label} timed out") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise CorpusError(f"{label} failed: {detail or 'no diagnostic output'}")
    return result


def _probe_audio(path: Path, expected_samples: int) -> None:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name,codec_type,sample_rate,channels,time_base,duration_ts",
            "-of",
            "json",
            str(path),
        ],
        "inspect source audio",
    )
    try:
        report = json.loads(result.stdout.decode("utf-8"))
        streams = report["streams"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CorpusError("ffprobe returned an invalid report") from error
    if not isinstance(streams, list) or len(streams) != 1:
        raise CorpusError("source audio must contain exactly one audio stream")
    stream = _object(streams[0], "ffprobe.stream")
    expected = {
        "codec_name": "flac",
        "codec_type": "audio",
        "sample_rate": "16000",
        "channels": 1,
        "time_base": "1/16000",
        "duration_ts": expected_samples,
    }
    for key, value in expected.items():
        actual = stream.get(key)
        if key == "duration_ts" and isinstance(actual, str) and actual.isdecimal():
            actual = int(actual)
        if actual != value:
            raise CorpusError(
                f"source audio {key} mismatch: expected {value}, got {actual}"
            )


def _convert_audio(path: Path) -> bytes:
    result = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "-1",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ],
        "decode source audio",
    )
    return result.stdout


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    created = False
    try:
        descriptor = os.open(path, flags, 0o644)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def _fetch_bytes(url: str, limit: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "whisper-runtime-corpus/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > limit:
                raise CorpusError("download exceeds its size limit")
            content = response.read(limit + 1)
    except CorpusError:
        raise
    except Exception as error:
        raise CorpusError("download failed") from error
    if len(content) > limit:
        raise CorpusError("download exceeds its size limit")
    return content


def _signed_audio_url(document: bytes, fixture: Fixture) -> str:
    try:
        payload = json.loads(document.decode("utf-8"))
        rows = payload["rows"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CorpusError("rows API returned invalid JSON") from error
    if not isinstance(rows, list) or len(rows) != 1:
        raise CorpusError("rows API must return exactly one row")
    entry = _object(rows[0], "rows[0]")
    if entry.get("row_idx") != fixture.source.row_offset:
        raise CorpusError("rows API returned a different row offset")
    row = _object(entry.get("row"), "rows[0].row")
    expected = {
        "id": fixture.fixture_id,
        "text": fixture.reference_text,
        "speaker_id": fixture.source.speaker_id,
        "chapter_id": fixture.source.chapter_id,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise CorpusError(f"rows API returned a different {key}")
    audio = row.get("audio")
    if isinstance(audio, list) and len(audio) == 1:
        audio = audio[0]
    audio_record = _object(audio, "rows[0].row.audio")
    url = _string(audio_record, "src", "rows[0].row.audio")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise CorpusError("rows API returned an invalid audio URL") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != ROWS_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.fragment
    ):
        raise CorpusError("rows API returned an unapproved audio URL")
    return url


def _download_source(fixture: Fixture, path: Path) -> None:
    metadata = _fetch_bytes(fixture.source.metadata_url, MAX_METADATA_BYTES)
    audio_url = _signed_audio_url(metadata, fixture)
    content = _fetch_bytes(audio_url, MAX_SOURCE_BYTES)
    if len(content) != fixture.source.size_bytes:
        raise CorpusError("downloaded source size does not match the manifest")
    if _digest(content) != fixture.source.sha256:
        raise CorpusError("downloaded source digest does not match the manifest")
    try:
        _write_exclusive(path, content)
    except FileExistsError as error:
        raise CorpusError(
            f"source appeared during download; verify it: {path}"
        ) from error


def prepare_fixture(
    fixture: Fixture, source_dir: Path, output_dir: Path, *, download: bool
) -> str:
    source_path = source_dir / fixture.source.filename
    if not source_path.exists():
        if not download:
            raise CorpusError(
                f"missing source fixture {source_path}; place the registered FLAC there "
                "or rerun with --download"
            )
        _download_source(fixture, source_path)
    _verify_file(
        source_path,
        fixture.source.size_bytes,
        fixture.source.sha256,
        "source fixture",
    )
    _probe_audio(source_path, fixture.sample_count)

    output_path = output_dir / fixture.filename
    expected_size = fixture.sample_count * 2
    if output_path.exists():
        _verify_file(output_path, expected_size, fixture.pcm_sha256, "PCM fixture")
        return "verified existing"

    content = _convert_audio(source_path)
    if len(content) != expected_size:
        raise CorpusError(
            f"decoded PCM size mismatch: expected {expected_size}, got {len(content)}"
        )
    if _digest(content) != fixture.pcm_sha256:
        raise CorpusError("decoded PCM digest does not match the manifest")
    try:
        _write_exclusive(output_path, content)
    except FileExistsError as error:
        raise CorpusError(
            f"PCM fixture appeared during conversion: {output_path}"
        ) from error
    return "wrote"


def prepare_corpus(
    manifest: Path, source_dir: Path, output_dir: Path, *, download: bool = False
) -> tuple[tuple[str, str], ...]:
    fixtures = load_manifest(manifest)
    return tuple(
        (
            fixture.fixture_id,
            prepare_fixture(fixture, source_dir, output_dir, download=download),
        )
        for fixture in fixtures
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and prepare the frozen LibriSpeech diagnostic corpus."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch a missing registered FLAC through its fixed rows endpoint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = prepare_corpus(
            args.manifest, args.source_dir, args.output_dir, download=args.download
        )
    except CorpusError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for fixture_id, status in results:
        print(f"{fixture_id}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
