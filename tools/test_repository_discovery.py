from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_repository


class RepositoryDiscoveryTests(unittest.TestCase):
    def test_generated_outputs_are_pruned_but_publication_evidence_is_checked(self):
        with tempfile.TemporaryDirectory(prefix=".tmp-discovery-") as temporary:
            root = Path(temporary) / "repo"
            included = {
                "README.md",
                "src/package.py",
                "evidence/record.json",
                "docs/artifacts/explanation.md",
            }
            excluded = {
                "artifacts/local/report.json",
                ".tmp-native/venv/site-packages/dependency.py",
                "build/lib/package.py",
                ".git/config.json",
                "src/__pycache__/cache.py",
                "conformance/fixtures/generated/record.json",
                "audio.wav",
            }
            for relative in included | excluded:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("portable", encoding="utf-8")
            with patch.object(check_repository, "ROOT", root):
                actual = {
                    path.relative_to(root).as_posix()
                    for path in check_repository.tracked_text_files()
                }
            self.assertEqual(actual, included)

    def test_walk_prunes_before_descending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [".tmp-native", "artifacts", "evidence", "src"]
            with (
                patch.object(check_repository, "ROOT", root),
                patch.object(
                    check_repository.os,
                    "walk",
                    return_value=iter([(str(root), directories, ["README.md"])]),
                ),
            ):
                self.assertEqual(
                    check_repository.tracked_text_files(), [root / "README.md"]
                )
            self.assertEqual(directories, ["evidence", "src"])


if __name__ == "__main__":
    unittest.main()
