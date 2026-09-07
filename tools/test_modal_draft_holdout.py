"""Local input, resource and record guards for the new-audio screen."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infra import modal_draft_holdout as x


class HoldoutTests(unittest.TestCase):
    def test_import_does_not_start_or_import_gpu(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys; import infra.modal_draft_holdout; "
                "assert not {'modal','torch','whisper'} & set(sys.modules)",
            ],
            cwd=x.ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def fixture_root(self, directory):
        root = Path(directory)
        (root / x.ASSETS).mkdir(parents=True)
        (root / x.MANIFEST).parent.mkdir(parents=True)
        fixtures = []
        for i in range(2):
            raw = bytes((i + 1, 0)) * 64000
            name = f"clip-{i}.pcm"
            (root / x.ASSETS / name).write_bytes(raw)
            fixtures.append(
                dict(
                    id=str(i),
                    filename=name,
                    sample_count=64000,
                    pcm_sha256=x.shared._sha(raw),
                    reference_text=f"clip {i}",
                )
            )
        (root / x.MANIFEST).write_text(
            json.dumps(dict(fixtures=fixtures)), encoding="utf-8"
        )
        return root, fixtures

    def test_exact_input_and_silence_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            root, _ = self.fixture_root(d)
            pcm, plan = x.input_plan(root)
            self.assertEqual(plan["sample_count"], 192000)
            self.assertEqual(plan["duration_seconds"], 12)
            self.assertEqual(plan["reference_text"], "clip 0 clip 1")
            self.assertEqual(pcm[128000:192000], bytes(64000))
            self.assertEqual(pcm[-64000:], bytes(64000))
            self.assertEqual(plan["spans"][1]["start_sample"], 96000)

    def test_input_hash_and_unsafe_filename_rejected(self):
        for change in ("hash", "path"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root, fixtures = self.fixture_root(d)
                fixtures[0]["pcm_sha256" if change == "hash" else "filename"] = (
                    "0" * 64 if change == "hash" else "../escape.pcm"
                )
                (root / x.MANIFEST).write_text(
                    json.dumps(dict(fixtures=fixtures)), encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    x.input_plan(root)

    def test_registered_downloaded_input(self):
        if not (x.ROOT / x.ASSETS).exists():
            self.skipTest("optional public PCM absent")
        pcm, plan = x.input_plan()
        self.assertEqual(plan["sample_count"], 247760)
        self.assertEqual(len(pcm), 495520)
        self.assertEqual(plan["duration_seconds"], 15.485)

    def test_resource_and_config_bounds(self):
        scope = x.scope()
        self.assertEqual(
            (scope["gpu_calls"], scope["timeout_seconds"], scope["retries"]),
            (1, 300, 0),
        )
        self.assertEqual((scope["min_containers"], scope["max_containers"]), (0, 1))
        self.assertLess(scope["planning_compute_usd"], 0.11)
        configs = list(scope["configurations"].values())
        self.assertEqual(
            {k for k in configs[0] if configs[0][k] != configs[1][k]},
            {"max_draft_tokens"},
        )
        self.assertFalse(any(x.CLAIMS.values()))
        with patch.object(x.time, "monotonic", return_value=295):
            self.assertTrue(x.admitted(100, 80))
            self.assertFalse(x.admitted(100, 85))

    def test_existing_attempt_prevents_resource_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output, journal, _ = x.shared._paths(root, "holdout-test", False)
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text("existing", encoding="utf-8")
            with patch.object(x, "resources") as resources:
                with self.assertRaisesRegex(ValueError, "attempt exists"):
                    x.run(replay_id="holdout-test", confirm_paid_gpu=True, root=root)
                resources.assert_not_called()
            self.assertFalse(output.exists())

    def test_partial_record_identity_and_comparison_are_verified(self):
        with tempfile.TemporaryDirectory() as d:
            root, _ = self.fixture_root(d)
            _, plan = x.input_plan(root)
            record = dict(
                experiment_id="modal-draft-holdout-v1",
                source=dict(snapshot={}),
                scope=x.scope(),
                input=plan,
                claim_boundary=x.CLAIMS,
                qualified=False,
                status="partial",
                backend=dict(patched_tree=x.previous.features.PATCHED_TREE),
                cells=[],
                warmup=[],
                native_window_count=0,
                comparison=dict(complete=False, accepted=False),
            )
            x.validate_record(record, {}, root)
            for field, value in (
                ("qualified", True),
                ("status", "completed"),
                ("comparison", dict(complete=True, accepted=True)),
                ("native_window_count", 1),
            ):
                bad = copy.deepcopy(record)
                bad[field] = value
                with (
                    self.subTest(field=field),
                    self.assertRaises((ValueError, KeyError)),
                ):
                    x.validate_record(bad, {}, root)

    def test_partial_record_roundtrip_uses_transport_identity(self):
        b = x.previous._corpus().b
        call_id = "fc-test123"
        record = dict(
            schema_version="1-diagnostic",
            claim_boundary=x.CLAIMS,
            source=dict(snapshot=dict(digest="d"), registration_sha256="p"),
            worker=dict(
                function_call_id=call_id,
                function_call_id_sha256=x.shared._sha(call_id.encode()),
            ),
        )
        payload = b._encode_worker_record(record)
        decoded = b._decode_worker_record(
            payload,
            expected_snapshot=dict(digest="d"),
            registration_sha256="p",
            manifest=dict(claim_boundary=x.CLAIMS),
        )
        self.assertEqual(decoded, record)


if __name__ == "__main__":
    unittest.main()
