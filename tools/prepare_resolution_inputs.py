"""Verify two frozen held-out PCM assets; downloading requires --download.

Selection is already frozen. This command never advances rows, invokes a model,
alters the original corpus, or overwrites an existing asset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_speech_corpus import (
    DATASET_ID,
    DATASET_REVISION,
    MAX_METADATA_BYTES,
    MAX_SOURCE_BYTES,
    CorpusError,
    Fixture,
    SourceRecord,
    _digest,
    _fetch_bytes,
    _signed_audio_url,
    _validate_metadata_url,
    _verify_file,
    _write_exclusive,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "experiments/modal-word-resolution-inputs-v1.json"
FIXTURES = (("2961-960-0004", 501), ("8455-210777-0028", 1000))


def prepare(root: Path = ROOT, *, download: bool = False) -> list[dict]:
    value = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    dataset = value["dataset"]
    if (
        value["schema_version"] != "1"
        or value["id"] != "modal-word-resolution-inputs-v1"
        or dataset["id"] != DATASET_ID
        or dataset["revision"] != DATASET_REVISION
        or dataset["configuration"] != "clean"
        or dataset["split"] != "test"
        or dataset["license"] != "CC BY 4.0"
        or [(item["id"], item["source"]["row_offset"]) for item in value["fixtures"]]
        != list(FIXTURES)
    ):
        raise CorpusError("held-out registration identity differs")
    results = []
    for item in value["fixtures"]:
        relative = f"artifacts/resolution-inputs-v1/{item['id']}.pcm"
        path = root / relative
        if item["pcm_path"] != relative or not path.resolve().is_relative_to(
            root.resolve()
        ):
            raise CorpusError("asset path differs or escapes the workspace")
        source = item["source"]
        _validate_metadata_url(source["metadata_url"], source["row_offset"])
        fixture = Fixture(
            item["id"],
            path.name,
            item["pcm_sha256"],
            item["sample_count"],
            item["reference_text"],
            SourceRecord(
                f"{item['id']}.flac",
                source["sha256"],
                source["size_bytes"],
                source["metadata_url"],
                source["row_offset"],
                source["speaker_id"],
                source["chapter_id"],
            ),
        )
        if (
            type(fixture.sample_count) is not int
            or not 80000 <= fixture.sample_count <= 192000
            or fixture.sample_count / 16000 != item["duration_seconds"]
            or not 0 < fixture.source.size_bytes <= MAX_SOURCE_BYTES
        ):
            raise CorpusError("source size or duration exceeds the selection bounds")
        if path.exists():
            _verify_file(
                path, fixture.sample_count * 2, fixture.pcm_sha256, "held-out PCM"
            )
            results.append({"id": fixture.fixture_id, "status": "verified existing"})
            continue
        if not download:
            raise CorpusError(f"missing held-out PCM {path}; use --download")
        metadata = _fetch_bytes(fixture.source.metadata_url, MAX_METADATA_BYTES)
        url = _signed_audio_url(metadata, fixture)
        parsed = urllib.parse.urlsplit(url)
        expected_path = (
            f"/cached-assets/{DATASET_ID}/--/{DATASET_REVISION}/--/clean/test/"
            f"{fixture.source.row_offset}/audio/audio.flac"
        )
        if parsed.path != expected_path:
            raise CorpusError("audio URL does not bind the frozen dataset revision")
        audio = _fetch_bytes(url, MAX_SOURCE_BYTES)
        if (
            len(audio) != fixture.source.size_bytes
            or _digest(audio) != fixture.source.sha256
        ):
            raise CorpusError("source FLAC differs from its frozen identity")
        if (
            audio[:4] != b"fLaC"
            or audio[4] & 127
            or int.from_bytes(audio[5:8], "big") != 34
        ):
            raise CorpusError("expected a FLAC STREAMINFO header")
        packed = int.from_bytes(audio[18:26], "big")
        if (
            packed >> 44 != 16000
            or ((packed >> 41) & 7) != 0
            or ((packed >> 36) & 31) != 15
            or packed & ((1 << 36) - 1) != fixture.sample_count
        ):
            raise CorpusError("source format or full sample count differs")
        converted = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
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
                "-t",
                "12.001",
                "-c:a",
                "pcm_s16le",
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=audio,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
        if (
            len(converted) != fixture.sample_count * 2
            or _digest(converted) != fixture.pcm_sha256
        ):
            raise CorpusError("converted PCM differs from its frozen identity")
        _write_exclusive(path, converted)
        results.append({"id": fixture.fixture_id, "status": "wrote"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    for result in prepare(download=args.download):
        print(json.dumps(result))


if __name__ == "__main__":
    main()
