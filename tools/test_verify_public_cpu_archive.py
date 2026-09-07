"""Public derivative integrity checks; no private original, native stack or network."""

import json
import unittest
import zipfile
from pathlib import Path

from tools import verify_public_cpu_archive as v


class PublicArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = (
            Path(__file__).resolve().parents[1]
            / "evidence/native-draft-fresh-process-cpu-2026-09-07.zip"
        )
        with zipfile.ZipFile(cls.archive) as archive:
            cls.files = {name: archive.read(name) for name in archive.namelist()}

    def test_public_archive_recomputes_without_original_private_paths(self):
        result = v.verify(self.archive)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["during_phase_source_files"], 65)
        self.assertEqual(result["payload_files"], 79)
        self.assertFalse(result["raw_report_byte_identity_preserved"])

    def test_member_tampering_is_detected_by_public_manifest(self):
        for name in (
            "report.json",
            "report.savepoint.json",
            "src/whisper_runtime/state.py",
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "public payload manifest"),
            ):
                v.verify_payload({**self.files, name: self.files[name] + b" "})

    def test_portable_private_path_detection_is_not_disabled(self):
        for prefix in (
            "C:" + chr(92) + "Users" + chr(92),
            chr(47) + "Users" + chr(47),
            chr(47) + "home" + chr(47),
        ):
            with self.subTest(prefix=prefix):
                self.assertIsNotNone(v.PERSONAL_PATH.search(prefix + "example/data"))

    def test_refreshed_public_manifest_cannot_hide_source_or_original_hash_changes(
        self,
    ):
        for name in ("src/whisper_runtime/state.py", v.DERIVATION):
            changed = dict(self.files)
            if name == v.DERIVATION:
                value = json.loads(changed[name])
                value["original_archive_sha256"] = "0" * 64
                changed[name] = v.encode(value)
            else:
                changed[name] += b" "
            changed[v.PUBLIC_MANIFEST] = v.encode(
                {
                    "schema": "public-cpu-archive-manifest/v1",
                    "files": {
                        key: {"sha256": v.sha(raw), "size_bytes": len(raw)}
                        for key, raw in sorted(changed.items())
                        if key != v.PUBLIC_MANIFEST
                    },
                }
            )
            with self.subTest(name=name), self.assertRaises(ValueError):
                v.verify_payload(changed)


if __name__ == "__main__":
    unittest.main()
