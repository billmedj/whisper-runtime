"""CPU checkpoint diagnostic orchestration tests; no model or device execution."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import verify_native_checkpoint as diagnostic


def receipts():
    producer = {
        "status": "passed",
        "pid": 101,
        "pipeline_identity": "same-pipeline",
        "model": {"fingerprint": "same-model"},
        "audio": {"unit_samples": 160},
        "control_state": {"full": "state"},
        "control_events": ["first", "second"],
        "saved_events": ["first"],
        "encoder_forwards": 3,
        "resources_released": True,
    }
    consumer = {
        "status": "passed",
        "pid": 102,
        "pipeline_identity": "same-pipeline",
        "model": producer["model"],
        "audio": producer["audio"],
        "restored_state": producer["control_state"],
        "new_events": ["second"],
        "encoder_forwards": 1,
        "redecoded_samples": 160,
        "final_committed_samples": 320,
        "final_accepted_samples": 320,
        "final_buffered_samples": 0,
        "resources_released": True,
    }
    return producer, consumer


class NativeCheckpointDiagnosticTests(unittest.TestCase):
    def test_archived_native_cpu_run_preserves_full_state_and_events(self):
        raw = (
            diagnostic.ROOT / "evidence/native-cpu-tiny-en-checkpoint-2026-09-06.json"
        ).read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "0d94faf718c9f0c5a1b07b5ea765aa84f99f9c55cd04432e081874a9660c251e",
        )
        record = json.loads(raw)
        replay = diagnostic.combine_receipts(record["producer"], record["consumer"])
        self.assertEqual(replay["status"], "passed")
        self.assertEqual(replay["checks"], record["checks"])
        self.assertEqual(record["producer"]["savepoint_bytes"], 473593)
        self.assertEqual(
            record["consumer"]["replay_interval_samples"], [176000, 352000]
        )
        self.assertFalse(record["gpu_used"])
        self.assertFalse(record["exact_token_state_restore"])

    def test_cli_requires_phase_and_savepoint_together(self):
        base = ["fixture.flac", "--checkpoint", "tiny.en.pt", "--revision", "revision"]
        self.assertIsNone(diagnostic.parse_args(base).phase)
        self.assertEqual(
            diagnostic.parse_args(
                base + ["--phase", "producer", "--savepoint", "saved"]
            ).phase,
            "producer",
        )
        for extra in (["--phase", "producer"], ["--savepoint", "saved"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                diagnostic.parse_args(base + extra)

    def test_typed_comparison_preserves_nested_types_and_tuples(self):
        @dataclass(frozen=True)
        class Nested:
            values: tuple[int, ...]

        @dataclass(frozen=True)
        class Publication:
            text: str
            metadata: Nested

        encoded = diagnostic.typed_value(Publication("words", Nested((1, 2))))
        self.assertTrue(encoded["type"].endswith("Publication"))
        nested = encoded["fields"]["metadata"]
        self.assertTrue(nested["type"].endswith("Nested"))
        self.assertEqual(nested["fields"]["values"], {"tuple": [1, 2]})
        self.assertNotEqual(
            diagnostic.typed_value((1, 2)), diagnostic.typed_value([1, 2])
        )
        with self.assertRaises(TypeError):
            diagnostic.typed_value(object())

    def test_report_requires_every_equivalence_and_resource_check(self):
        producer, consumer = receipts()
        report = diagnostic.combine_receipts(producer, consumer)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["source_process_exited_before_restore"])
        for name in (
            "model_downloaded",
            "gpu_used",
            "exact_token_state_restore",
            "durable_input_acknowledgement",
            "timing_is_benchmark",
            "recognition_accuracy_qualification",
        ):
            self.assertIs(report[name], False)
        for field, value in (
            ("pid", producer["pid"]),
            ("pipeline_identity", "changed"),
            ("model", {"fingerprint": "changed"}),
            ("restored_state", {"lost": "metadata"}),
            ("new_events", ["first", "second"]),
            ("encoder_forwards", 2),
            ("redecoded_samples", 320),
            ("final_committed_samples", 319),
            ("final_buffered_samples", 1),
            ("resources_released", False),
        ):
            changed = copy.deepcopy(consumer)
            changed[field] = value
            with self.subTest(field=field):
                self.assertEqual(
                    diagnostic.combine_receipts(producer, changed)["status"], "failed"
                )

    def test_subprocesses_are_sequential_and_second_receives_first_receipt(self):
        producer, consumer = receipts()
        commands = []

        def completed(command, **kwargs):
            commands.append((command, kwargs))
            phase = command[-1]
            self.assertEqual(command[-2], "--phase")
            if phase == "producer":
                self.assertEqual(len(commands), 1)
                self.assertIsNone(kwargs["input"])
                receipt = producer
            else:
                self.assertEqual(len(commands), 2)
                self.assertEqual(json.loads(kwargs["input"]), producer)
                receipt = consumer
            return SimpleNamespace(returncode=0, stdout=json.dumps(receipt), stderr="")

        with tempfile.TemporaryDirectory(
            prefix="checkpoint-command-test-"
        ) as directory:
            audio = Path(directory) / "fixture.flac"
            weights = Path(directory) / "tiny.en.pt"
            audio.write_bytes(b"fixture")
            weights.write_bytes(b"weights")
            args = SimpleNamespace(audio=audio, checkpoint=weights, revision="revision")
            with patch.object(diagnostic.subprocess, "run", side_effect=completed):
                report = diagnostic.verify(args)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(commands), 2)
        for command, kwargs in commands:
            self.assertIn("-B", command)
            self.assertEqual(kwargs["timeout"], 600)
            self.assertTrue(kwargs["capture_output"])
        savepoint = Path(commands[0][0][commands[0][0].index("--savepoint") + 1])
        self.assertFalse(savepoint.parent.exists())

    def test_failed_source_never_launches_consumer(self):
        with tempfile.TemporaryDirectory(
            prefix="checkpoint-failure-test-"
        ) as directory:
            audio = Path(directory) / "fixture.flac"
            weights = Path(directory) / "tiny.en.pt"
            audio.write_bytes(b"fixture")
            weights.write_bytes(b"weights")
            args = SimpleNamespace(audio=audio, checkpoint=weights, revision="revision")
            failed = SimpleNamespace(
                returncode=1, stdout='{"status":"failed"}', stderr=""
            )
            with (
                patch.object(diagnostic.subprocess, "run", return_value=failed) as run,
                self.assertRaisesRegex(RuntimeError, "producer process failed"),
            ):
                diagnostic.verify(args)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
