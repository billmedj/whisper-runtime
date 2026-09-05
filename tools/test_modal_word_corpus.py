"""Local checks for the bounded corpus experiment. No model or GPU is required."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.dict(os.environ, {"WHISPER_MODAL_ENABLE_WORD_CORPUS": "0"}):
    from infra import modal_word_corpus as corpus


class Echo:
    def remote(self, payload: bytes) -> bytes:
        return payload


class CorpusGuards(unittest.TestCase):
    def root(self, path: Path) -> dict:
        for relative in (
            corpus.MANIFEST_PATH,
            corpus.PRODUCER_PATH,
            *corpus.HELPER_PATHS,
        ):
            destination = path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((corpus.ROOT / relative).read_bytes())
        source = path / "src/whisper_runtime/example.py"
        source.parent.mkdir(parents=True)
        source.write_text("VALUE = 1\n", encoding="utf-8")
        return corpus.read_registration(path)

    def test_import_needs_no_modal_or_torch(self) -> None:
        env = dict(os.environ, WHISPER_MODAL_ENABLE_WORD_CORPUS="0")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import infra.modal_word_corpus as c; "
                "assert 'modal' not in sys.modules; assert 'torch' not in sys.modules; "
                "assert c.app is None",
            ],
            cwd=corpus.ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registration_keeps_cost_and_claim_bounds(self) -> None:
        manifest = corpus.read_registration()
        self.assertEqual(len(manifest["fixtures"]), 3)
        self.assertEqual(len(manifest["cases"]), 5)
        self.assertEqual(sum(c["sample_count"] for c in manifest["cases"]), 1479840)
        self.assertEqual(manifest["paid_budget"]["maximum_gpu_seconds"], 180)
        self.assertEqual(set(manifest["claim_boundary"].values()), {False})

    def test_modified_budget_paths_and_total_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.root(root)
            for change in (
                "budget",
                "path",
                "boolean_samples",
                "duration",
                "duplicate",
            ):
                with self.subTest(change=change):
                    manifest = copy.deepcopy(original)
                    if change == "budget":
                        manifest["paid_budget"]["maximum_gpu_function_calls"] = 2
                    elif change == "path":
                        manifest["fixtures"][0]["filename"] = "../outside.pcm"
                    elif change == "boolean_samples":
                        manifest["cases"][0]["sample_count"] = True
                    elif change == "duration":
                        manifest["cases"][0]["sample_count"] = 180 * 16000
                    else:
                        manifest["fixtures"][1]["id"] = manifest["fixtures"][0]["id"]
                    (root / corpus.MANIFEST_PATH).write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        corpus.read_registration(root)

    def test_case_preserves_exact_samples_and_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            pcm = b"\x01\x00" * 7
            (assets / "short.pcm").write_bytes(pcm)
            fixture = {
                "id": "short",
                "filename": "short.pcm",
                "sample_count": 7,
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "reference_text": "ONE",
            }
            expected = pcm + bytes(32) + pcm
            case = {
                "parts": [
                    {"fixture_id": "short"},
                    {"silence_ms": 1},
                    {"fixture_id": "short"},
                ],
                "sample_count": 30,
                "pcm_sha256": hashlib.sha256(expected).hexdigest(),
                "reference_text": "ONE ONE",
            }
            self.assertEqual(corpus.build_case(case, [fixture], assets), expected)
            for key, value in (
                ("sample_count", 29),
                ("reference_text", "ONE"),
                ("pcm_sha256", "0" * 64),
            ):
                changed = dict(case, **{key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    corpus.build_case(changed, [fixture], assets)
            (assets / "short.pcm").write_bytes(pcm[:-2])
            with self.assertRaises(ValueError):
                corpus.build_case(case, [fixture], assets)

    def test_unknown_part_does_not_become_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                corpus.build_case({"parts": [{"gap": 1000}]}, [], Path(temporary))

    def test_silence_has_no_word_error_rate(self) -> None:
        measured = corpus.b._word_difference("invented words", "")
        self.assertEqual(measured["word_edit_distance"], 2)
        self.assertIsNone(measured["word_edit_rate"])

    def test_snapshot_binds_helpers_and_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            initial = corpus.source_snapshot(root)
            paths = {entry["path"] for entry in initial["files"]}
            self.assertTrue(set(corpus.HELPER_PATHS).issubset(paths))
            self.assertIn(corpus.MANIFEST_PATH, paths)
            helper = root / corpus.HELPER_PATHS[0]
            helper.write_text("# changed\n", encoding="utf-8")
            self.assertNotEqual(
                initial["digest"], corpus.source_snapshot(root)["digest"]
            )

    def test_preflight_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            receipt = corpus._execute_transport_probe(root=root, remote_function=Echo())
            rows = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(rows[-1]["event"], "transport-passed")
            self.assertGreater(rows[-1]["payload_bytes"], 8192)
            with self.assertRaises(FileExistsError):
                corpus._execute_transport_probe(root=root, remote_function=Echo())

    def test_paid_call_requires_confirmation(self) -> None:
        with self.assertRaises((ValueError, RuntimeError)):
            corpus._execute_local_attempt(remote_function=Echo())

    def test_receipt_prevents_paid_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            receipt = corpus._paths(root)[1]
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text("failed attempt\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                corpus._execute_local_attempt(
                    root=root, confirm_paid_gpu=True, remote_function=Echo()
                )


if __name__ == "__main__":
    unittest.main()
