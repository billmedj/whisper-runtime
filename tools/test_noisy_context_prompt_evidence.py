"""Replay the T4 observations and the explicitly bounded report correction."""

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from infra import modal_noisy_context_prompt as p
from tools import prepare_acoustic_cases as acoustic
from tools.verify_noisy_context_prompt import KNOWN_CONTROL_ERROR_SHA, verify


class NoisyContextPromptEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(
            (
                p.ROOT
                / "evidence/modal-t4-tiny-en-noisy-context-prompt-2026-09-06.json"
            ).read_text(encoding="utf-8")
        )

    def test_receipt_is_preserved_and_all_other_checks_replay(self):
        before = copy.deepcopy(self.record)
        result = verify(self.record)
        self.assertEqual(self.record, before)
        self.assertEqual(p.shared._hash(self.record), KNOWN_CONTROL_ERROR_SHA)
        self.assertTrue(result["record_checks_pass"])
        self.assertTrue(result["report_only_correction"])
        self.assertFalse(result["original_control"]["retained_null_alignment_equal"])
        self.assertTrue(result["replayed_control"]["retained_null_alignment_equal"])
        self.assertFalse(result["inference_run"])

    def clean_root(self, directory):
        root = Path(directory)
        for name in (
            acoustic.MANIFEST_PATH,
            acoustic.ARCHIVED_ASSETS,
            p.ARCHIVE,
            p.PRODUCER,
        ):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((p.ROOT / name).read_bytes())
        return root

    def test_clean_checkout_replays_shipped_pcm_without_creating_cache_or_mutating_receipt(
        self,
    ):
        before = copy.deepcopy(self.record)
        with tempfile.TemporaryDirectory() as directory:
            root = self.clean_root(directory)
            result = verify(self.record, root)
            self.assertTrue(result["record_checks_pass"])
            self.assertTrue(result["report_only_correction"])
            self.assertFalse((root / acoustic.ASSET_PATH).exists())
        self.assertEqual(self.record, before)
        self.assertEqual(p.shared._hash(self.record), KNOWN_CONTROL_ERROR_SHA)

    def test_corrupt_archive_pcm_is_rejected_without_weakening_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.clean_root(directory)
            path = root / acoustic.ARCHIVED_ASSETS
            with zipfile.ZipFile(path) as archive:
                entries = {name: archive.read(name) for name in archive.namelist()}
            name = acoustic.ASSET_PATH + "/6930-75918-0000.pcm"
            entries[name] = bytes([entries[name][0] ^ 1]) + entries[name][1:]
            with zipfile.ZipFile(
                path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name, raw in entries.items():
                    archive.writestr(name, raw)
            with self.assertRaisesRegex(ValueError, "PCM digest mismatch"):
                verify(self.record, root)

    def test_memory_diagnostic_registration_uses_same_cache_free_source(self):
        from infra import modal_memory_diagnostic as memory

        with tempfile.TemporaryDirectory() as directory:
            root = self.clean_root(directory)
            seed, registration = memory.input_plan(root, capacity_smoke=True)
            self.assertEqual(len(seed) // 2, 744800)
            self.assertEqual(registration["seed_sha256"], p.shared._sha(seed))
            self.assertEqual(
                registration["phases"][0]["sha256"],
                registration["phases"][1]["sha256"],
            )
            self.assertEqual(registration["profile"]["name"], "low-latency-v2")
            self.assertFalse((root / acoustic.ASSET_PATH).exists())

    def test_missing_alternate_assets_and_corrupt_existing_cache_do_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.clean_root(directory)
            with self.assertRaises(FileNotFoundError):
                acoustic.build_cases(root, root / "explicit-assets")
            cache = root / acoustic.ASSET_PATH
            cache.mkdir(parents=True)
            (cache / "6930-75918-0000.pcm").write_bytes(bytes(112160))
            with self.assertRaisesRegex(ValueError, "PCM digest mismatch"):
                verify(self.record, root)

    def test_unknown_receipt_cannot_use_the_report_correction(self):
        for key, value in (("elapsed_ns", 1), ("qualified", True)):
            changed = copy.deepcopy(self.record)
            changed[key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "unrecognized"),
            ):
                verify(changed)

    def test_control_correction_does_not_bypass_record_validation(self):
        with patch.object(
            p, "validate_record", side_effect=ValueError("bad native result")
        ):
            with self.assertRaisesRegex(ValueError, "bad native result"):
                verify(self.record)

    def test_four_observations_share_two_encoder_admissions(self):
        self.assertEqual(self.record["native_window_count"], 2)
        self.assertEqual(self.record["decode_attempt_count"], 4)
        self.assertEqual(
            sum(len(c["window"]["encoder_calls"]) for c in self.record["cells"]), 6
        )
        for cell in self.record["cells"]:
            self.assertEqual(cell["session_version"], 0)
            self.assertTrue(cell["capacity_restored"])
            self.assertTrue(cell["first_snapshot_unchanged"])
            self.assertTrue(cell["same_lease"])
            self.assertTrue(all(not a["strict_candidate"] for a in cell["attempts"]))

    def test_prompt_changes_the_retained_text_not_its_join_authority(self):
        retained, head = self.record["cells"]
        original, prompted = retained["attempts"]
        self.assertEqual(original["audio_evidence"]["reason"], "no_lexical_text")
        self.assertEqual(prompted["word_decision"]["reason"], "anchor_missing")
        self.assertIsNone(prompted["recognition"]["hypothetical_full_text"])
        self.assertEqual(
            prompted["result"]["text"], head["attempts"][0]["result"]["text"]
        )
        self.assertEqual(
            head["attempts"][0]["alignment"]["words"],
            head["attempts"][1]["alignment"]["words"],
        )
        self.assertEqual(
            head["attempts"][0]["recognition"]["against_human_reference"][
                "word_edit_distance"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
