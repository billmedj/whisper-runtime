"""Tests for the frozen speech-corpus preparation tool."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import prepare_speech_corpus as corpus

SOURCE = b"registered flac"
PCM = b"\x01\x00\x02\x00\x03\x00"
FIXTURE_ID = "1-2-3"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _manifest() -> dict:
    return {
        "manifest_version": "1",
        "manifest_id": "modal-word-corpus-v1",
        "dataset": {
            "id": corpus.DATASET_ID,
            "configuration": corpus.DATASET_CONFIGURATION,
            "split": corpus.DATASET_SPLIT,
            "revision": corpus.DATASET_REVISION,
            "license": "CC BY 4.0",
        },
        "audio_format": {
            "sample_rate_hz": 16_000,
            "channels": 1,
            "encoding": "pcm_s16le",
        },
        "fixtures": [
            {
                "id": FIXTURE_ID,
                "filename": f"{FIXTURE_ID}.pcm",
                "pcm_sha256": _sha256(PCM),
                "sample_count": len(PCM) // 2,
                "reference_text": "A FROZEN REFERENCE",
                "source": {
                    "filename": f"{FIXTURE_ID}.flac",
                    "sha256": _sha256(SOURCE),
                    "size_bytes": len(SOURCE),
                    "metadata_url": (
                        "https://datasets-server.huggingface.co/rows?"
                        "dataset=openslr%2Flibrispeech_asr&config=clean&split=test&"
                        "offset=7&length=1"
                    ),
                    "row_offset": 7,
                    "speaker_id": 1,
                    "chapter_id": 2,
                },
            }
        ],
    }


class SpeechCorpusPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.manifest = root / "manifest.json"
        self.source_dir = root / "source"
        self.output_dir = root / "output"
        self.source_dir.mkdir()
        self.manifest.write_text(json.dumps(_manifest()), encoding="utf-8")

    @property
    def source(self) -> Path:
        return self.source_dir / f"{FIXTURE_ID}.flac"

    @property
    def output(self) -> Path:
        return self.output_dir / f"{FIXTURE_ID}.pcm"

    def prepare(self, *, download: bool = False) -> tuple[tuple[str, str], ...]:
        return corpus.prepare_corpus(
            self.manifest,
            self.source_dir,
            self.output_dir,
            download=download,
        )

    def test_existing_registered_pcm_is_verified_without_conversion(self) -> None:
        self.source.write_bytes(SOURCE)
        self.output_dir.mkdir()
        self.output.write_bytes(PCM)
        with (
            mock.patch.object(corpus, "_probe_audio") as probe,
            mock.patch.object(corpus, "_convert_audio") as convert,
        ):
            result = self.prepare()
        self.assertEqual(result, ((FIXTURE_ID, "verified existing"),))
        probe.assert_called_once()
        convert.assert_not_called()
        self.assertEqual(self.output.read_bytes(), PCM)

    def test_existing_pcm_mismatch_is_rejected_without_overwrite(self) -> None:
        self.source.write_bytes(SOURCE)
        self.output_dir.mkdir()
        original = b"wrong"
        self.output.write_bytes(original)
        with mock.patch.object(corpus, "_probe_audio"):
            with self.assertRaisesRegex(
                corpus.CorpusError, "PCM fixture size mismatch"
            ):
                self.prepare()
        self.assertEqual(self.output.read_bytes(), original)

    def test_missing_source_requires_explicit_download(self) -> None:
        with mock.patch.object(corpus, "_fetch_bytes") as fetch:
            with self.assertRaisesRegex(corpus.CorpusError, "rerun with --download"):
                self.prepare()
        fetch.assert_not_called()

    def test_conversion_writes_once_then_skips(self) -> None:
        self.source.write_bytes(SOURCE)
        with (
            mock.patch.object(corpus, "_probe_audio"),
            mock.patch.object(corpus, "_convert_audio", return_value=PCM) as convert,
        ):
            self.assertEqual(self.prepare(), ((FIXTURE_ID, "wrote"),))
            self.assertEqual(self.prepare(), ((FIXTURE_ID, "verified existing"),))
        convert.assert_called_once()
        self.assertEqual(self.output.read_bytes(), PCM)

    def test_download_checks_row_identity_and_source_digest_before_write(self) -> None:
        metadata = json.dumps(
            {
                "rows": [
                    {
                        "row_idx": 7,
                        "row": {
                            "id": FIXTURE_ID,
                            "text": "A FROZEN REFERENCE",
                            "speaker_id": 1,
                            "chapter_id": 2,
                            "audio": [
                                {
                                    "src": (
                                        "https://datasets-server.huggingface.co/"
                                        "cached-assets/frozen.flac"
                                    )
                                }
                            ],
                        },
                    }
                ]
            }
        ).encode()
        with (
            mock.patch.object(corpus, "_fetch_bytes", side_effect=[metadata, SOURCE]),
            mock.patch.object(corpus, "_probe_audio"),
            mock.patch.object(corpus, "_convert_audio", return_value=PCM),
        ):
            self.assertEqual(self.prepare(download=True), ((FIXTURE_ID, "wrote"),))
        self.assertEqual(self.source.read_bytes(), SOURCE)
        self.assertEqual(self.output.read_bytes(), PCM)

    def test_bad_download_identity_writes_nothing(self) -> None:
        metadata = json.dumps(
            {
                "rows": [
                    {
                        "row_idx": 7,
                        "row": {
                            "id": "9-9-9",
                            "text": "A FROZEN REFERENCE",
                            "speaker_id": 1,
                            "chapter_id": 2,
                            "audio": [
                                {
                                    "src": (
                                        "https://datasets-server.huggingface.co/"
                                        "cached-assets/frozen.flac"
                                    )
                                }
                            ],
                        },
                    }
                ]
            }
        ).encode()
        with mock.patch.object(corpus, "_fetch_bytes", return_value=metadata):
            with self.assertRaisesRegex(corpus.CorpusError, "different id"):
                self.prepare(download=True)
        self.assertFalse(self.source.exists())
        self.assertFalse(self.output.exists())

    def test_changed_rows_query_and_boolean_integer_are_rejected(self) -> None:
        for mutate, expected in (
            (
                lambda data: data["fixtures"][0]["source"].__setitem__(
                    "row_offset", True
                ),
                "row_offset must be an integer",
            ),
            (
                lambda data: data["fixtures"][0]["source"].__setitem__(
                    "metadata_url",
                    "https://datasets-server.huggingface.co/rows?"
                    "dataset=openslr%2Flibrispeech_asr&config=clean&split=test&"
                    "offset=8&length=1",
                ),
                "query does not match",
            ),
        ):
            with self.subTest(expected=expected):
                document = copy.deepcopy(_manifest())
                mutate(document)
                self.manifest.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(corpus.CorpusError, expected):
                    corpus.load_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
