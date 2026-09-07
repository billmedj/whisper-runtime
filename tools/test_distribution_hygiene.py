from __future__ import annotations

import tarfile
import tempfile
import unittest
from pathlib import Path

from check_distribution import SDIST_REQUIRED_SUFFIXES, _check_forbidden, check_sdist


class DistributionHygieneTests(unittest.TestCase):
    def test_alpha_release_notes_and_diagrams_cannot_be_omitted(self):
        required = {
            "docs/releases/0.1.0a1.md",
            "docs/assets/transcription-flow.svg",
            "docs/assets/execution-lifecycle.svg",
        }
        self.assertTrue(required <= SDIST_REQUIRED_SUFFIXES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "package.tar.gz"
            for omitted in sorted(required):
                with self.subTest(omitted=omitted):
                    with tarfile.open(path, "w:gz") as archive:
                        for name in sorted(SDIST_REQUIRED_SUFFIXES - {omitted}):
                            archive.addfile(tarfile.TarInfo("package/" + name))
                    self.assertEqual(
                        check_sdist(path),
                        [f"source distribution is missing {omitted}"],
                    )

    def test_each_published_evidence_archive_is_required(self):
        archives = {name for name in SDIST_REQUIRED_SUFFIXES if name.endswith(".zip")}
        self.assertEqual(len(archives), 16)
        self.assertIn("evidence/modal-low-latency-v2-2026-09-07.zip", archives)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "package.tar.gz"
            for omitted in sorted(archives):
                with self.subTest(omitted=omitted):
                    with tarfile.open(path, "w:gz") as archive:
                        for name in sorted(SDIST_REQUIRED_SUFFIXES - {omitted}):
                            archive.addfile(tarfile.TarInfo("package/" + name))
                    self.assertEqual(
                        check_sdist(path),
                        [f"source distribution is missing {omitted}"],
                    )

    def test_local_outputs_and_environments_are_never_publishable(self):
        for relative in (
            "artifacts/report.json",
            ".tmp-native/backend/whisper/model.py",
            ".venv/lib/site-packages/package.py",
            ".git/config",
            ".ruff_cache/cache",
            "build/lib/package.py",
            "conformance/cache/audio.wav",
        ):
            with self.subTest(relative=relative):
                self.assertTrue(_check_forbidden({f"package/{relative}"}, "sdist"))

    def test_published_evidence_and_source_are_allowed(self):
        self.assertEqual(
            _check_forbidden(
                {"package/evidence/report.json", "whisper_runtime/cli.py"}, "sdist"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
