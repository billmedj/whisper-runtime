"""Current alpha metadata and source-distribution inclusions stay consistent."""

import json
import re
import unittest
from pathlib import Path

from check_distribution import SDIST_REQUIRED_SUFFIXES

ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_alpha_version_and_citation_agree(self):
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        project_versions = re.findall(r'^version = ("[^"\n]+")$', project, re.MULTILINE)
        citation_versions = re.findall(
            r'^version: ("[^"\n]+")$', citation, re.MULTILINE
        )
        self.assertEqual(len(project_versions), 1)
        self.assertEqual(len(citation_versions), 1)
        self.assertEqual(json.loads(project_versions[0]), "0.1.0a1")
        self.assertEqual(
            json.loads(citation_versions[0]), json.loads(project_versions[0])
        )
        self.assertIn('"Development Status :: 3 - Alpha"', project)
        self.assertNotIn('"Development Status :: 2 - Pre-Alpha"', project)
        security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("This project is alpha.", security)
        self.assertIn("No release\nis approved for production use.", security)

    def test_every_required_evidence_archive_is_explicitly_in_the_manifest(self):
        manifest = set((ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines())
        for name in SDIST_REQUIRED_SUFFIXES:
            if name.endswith(".zip"):
                with self.subTest(name=name):
                    self.assertIn(f"include {name}", manifest)

    def test_release_notes_and_diagrams_are_included_in_the_source_distribution(self):
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        self.assertIn("recursive-include docs *.md", manifest)
        self.assertIn("recursive-include docs/assets *.svg", manifest)
        release = "docs/releases/0.1.0a1.md"
        self.assertIn(release, SDIST_REQUIRED_SUFFIXES)
        self.assertTrue((ROOT / release).is_file())


if __name__ == "__main__":
    unittest.main()
