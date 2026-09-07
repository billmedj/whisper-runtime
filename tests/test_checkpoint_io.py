"""CPU-only hostile-input and no-clobber contracts for private savepoint I/O."""

import hashlib
import json
import math
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

from whisper_runtime.adapters import _checkpoint_io as checkpoint
from whisper_runtime.adapters.continuous_stream import ContinuousStreamConfig


@dataclass(frozen=True)
class Child:
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("negative count")


@dataclass(frozen=True)
class Snapshot:
    child: Child | None
    samples: tuple[int, ...]
    pair: tuple[str, bool]
    state: Literal["ready", "closed"]
    fraction: float


@dataclass(frozen=True)
class ChildVariant(Child):
    label: str


REGISTRY = (Child, Snapshot)


def envelope(payload: object) -> bytes:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return json.dumps(
        {
            "schema": "continuous-savepoint/v1",
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "payload": payload,
        }
    ).encode()


class CheckpointCodecTests(unittest.TestCase):
    def test_known_legacy_config_fields_only_default_to_false(self):
        for missing in (
            ("defer_word_commits",),
            ("defer_word_commits", "eof_context_retry"),
            ("previous_holdback_ms",),
            (
                "previous_holdback_ms",
                "defer_word_commits",
                "eof_context_retry",
                "max_draft_tokens",
            ),
        ):
            with self.subTest(missing=missing):
                config = ContinuousStreamConfig()
                registry = (ContinuousStreamConfig,)
                payload = json.loads(checkpoint.encode({"config": config}, registry))[
                    "payload"
                ]
                fields = payload["items"]["config"]["fields"]
                for name in missing:
                    fields.pop(name)
                restored = checkpoint.decode(envelope(payload), registry)["config"]
                self.assertEqual(restored, config)
                fields.pop("preview_interval_ms")
                with self.assertRaises(ValueError):
                    checkpoint.decode(envelope(payload), registry)

    def test_v1_config_missing_only_eof_context_retry_defaults_to_false(self) -> None:
        config = ContinuousStreamConfig()
        registry = (ContinuousStreamConfig,)
        payload = json.loads(checkpoint.encode({"config": config}, registry))["payload"]
        self.assertIs(
            payload["items"]["config"]["fields"].pop("eof_context_retry"), False
        )
        restored = checkpoint.decode(envelope(payload), registry)["config"]
        self.assertEqual(restored, config)
        self.assertIs(restored.eof_context_retry, False)
        rewritten = json.loads(checkpoint.encode({"config": restored}, registry))
        self.assertIs(
            rewritten["payload"]["items"]["config"]["fields"]["eof_context_retry"],
            False,
        )

    def test_v1_config_compatibility_keeps_other_fields_and_types_strict(self) -> None:
        registry = (ContinuousStreamConfig,)
        original = json.loads(
            checkpoint.encode({"config": ContinuousStreamConfig()}, registry)
        )["payload"]
        original["items"]["config"]["fields"].pop("eof_context_retry")
        for missing in original["items"]["config"]["fields"]:
            if missing in (
                "defer_word_commits",
                "max_draft_tokens",
                "previous_holdback_ms",
            ):
                continue  # These optional features postdate older v1 savepoints.
            payload = json.loads(json.dumps(original))
            payload["items"]["config"]["fields"].pop(missing)
            with (
                self.subTest(missing=missing),
                self.assertRaisesRegex(
                    ValueError, "fields do not match ContinuousStreamConfig"
                ),
            ):
                checkpoint.decode(envelope(payload), registry)
        for name, value in (
            ("unknown", False),
            ("preview_interval_ms", True),
            ("eof_context_retry", 0),
            ("eof_context_retry", "false"),
            ("eof_context_retry", None),
            ("max_draft_tokens", False),
            ("max_draft_tokens", "32"),
            ("max_draft_tokens", 33),
        ):
            payload = json.loads(json.dumps(original))
            payload["items"]["config"]["fields"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                checkpoint.decode(envelope(payload), registry)

    def test_v1_config_migration_never_bypasses_checksum(self) -> None:
        registry = (ContinuousStreamConfig,)
        raw = json.loads(
            checkpoint.encode({"config": ContinuousStreamConfig()}, registry)
        )
        raw["payload"]["items"]["config"]["fields"].pop("eof_context_retry")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            checkpoint.decode(json.dumps(raw).encode(), registry)

    def test_roundtrip_is_deterministic_and_preserves_types(self) -> None:
        payload = {
            "snapshot": Snapshot(Child(1), (2, 3), ("x", True), "ready", 0.2),
            "bytes": b"\x00\xff",
            "tuple": (),
            "dict": {"tag": "bytes", "data": "ordinary"},
            "none": None,
        }
        raw = checkpoint.encode(payload, REGISTRY)
        self.assertEqual(checkpoint.decode(raw, REGISTRY), payload)
        self.assertEqual(
            raw, checkpoint.encode(dict(reversed(tuple(payload.items()))), REGISTRY)
        )

    def test_corruption_and_envelope_validation(self) -> None:
        raw = checkpoint.encode({"value": "before"}, ())
        for bad in (
            raw.replace(b"before", b"after"),
            b"{}",
            raw + b"!",
            raw.replace(b"continuous-savepoint/v1", b"continuous-savepoint/v2"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checkpoint.decode(bad, ())

    def test_duplicate_keys_and_nonfinite_numbers(self) -> None:
        for raw in (
            b'{"a":1,"a":2}',
            b'{"a":NaN}',
            b'{"a":Infinity}',
            b'{"a":-Infinity}',
            envelope({"tag": "dict", "items": {"bad": float("inf")}}),
            envelope({"tag": "dict", "items": {"bad": 1.0}}).replace(b"1.0", b"1e999"),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                checkpoint.decode(raw, ())

    def test_unsafe_types_tags_and_fields(self) -> None:
        for value in ([], {1: "x"}, object(), math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint.encode({"value": value}, ())
        for value in (
            {"tag": "pickle", "data": "x"},
            {"tag": "dataclass", "class": "Path", "fields": {}},
            {"tag": "dataclass", "class": "Child", "fields": {}},
            {"tag": "dataclass", "class": "Child", "fields": {"count": 1, "extra": 2}},
            {"tag": "dataclass", "class": "Child", "fields": {"count": True}},
            {"tag": "dataclass", "class": "Child", "fields": {"count": -1}},
            {"tag": "bytes", "data": "???"},
            {"tag": "bytes", "data": "AB=="},
            {"tag": "tuple", "items": {}},
            {"tag": "dict", "items": [], "extra": 1},
        ):
            raw = envelope({"tag": "dict", "items": {"value": value}})
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint.decode(raw, REGISTRY)

    def test_field_annotation_types_are_enforced(self) -> None:
        original = json.loads(
            checkpoint.encode(
                {"snapshot": Snapshot(Child(1), (1,), ("x", True), "ready", 0.5)},
                REGISTRY,
            )
        )
        for field, replacement in (
            ("child", 1),
            ("samples", {"tag": "tuple", "items": [True]}),
            ("pair", {"tag": "tuple", "items": ["x", 1]}),
            ("state", "unknown"),
            ("fraction", True),
        ):
            payload = json.loads(json.dumps(original["payload"]))
            payload["items"]["snapshot"]["fields"][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                checkpoint.decode(envelope(payload), REGISTRY)
        with self.assertRaises(ValueError):
            checkpoint.encode({"child": Child(True)}, REGISTRY)

    def test_registry_cannot_expand_from_input(self) -> None:
        raw = checkpoint.encode({"child": Child(1)}, REGISTRY)
        with self.assertRaises(ValueError):
            checkpoint.decode(raw, ())
        with self.assertRaises(ValueError):
            checkpoint.encode({}, (Child, Child))
        with self.assertRaises(ValueError):
            checkpoint.encode({}, (str,))

    def test_dataclass_subtype_retains_its_concrete_provenance(self) -> None:
        payload = {
            "snapshot": Snapshot(ChildVariant(1, "native"), (), ("x", True), "ready", 0)
        }
        registry = (*REGISTRY, ChildVariant)
        self.assertEqual(
            checkpoint.decode(checkpoint.encode(payload, registry), registry), payload
        )
        with self.assertRaises(ValueError):
            checkpoint.encode(payload, REGISTRY)

    def test_bad_digest_and_decode_failure_have_no_disk_effects(self) -> None:
        raw = json.loads(checkpoint.encode({}, ()))
        raw["sha256"] = "\u00e9" * 64
        with patch.object(checkpoint.os, "open") as opened:
            with self.assertRaises(ValueError):
                checkpoint.decode(json.dumps(raw).encode(), ())
            opened.assert_not_called()

    def test_size_and_nesting_limits(self) -> None:
        with patch.object(checkpoint, "_MAX_BYTES", 256):
            for action in (
                lambda: checkpoint.decode(b" " * 257, ()),
                lambda: checkpoint.encode({"data": "x" * 257}, ()),
                lambda: checkpoint.encode({"data": b"x" * 257}, ()),
                lambda: checkpoint.encode({"data": "\\" * 180}, ()),
            ):
                with self.assertRaises(ValueError):
                    action()
        with self.assertRaises(ValueError):
            checkpoint.decode(b"[" * 33 + b"]" * 33, ())
        recursive = {}
        recursive["self"] = recursive
        with self.assertRaises(ValueError):
            checkpoint.encode(recursive, ())


class CheckpointFilesystemTests(unittest.TestCase):
    # Resolve platform aliases in the fixture root, never the checkpoint under test.
    def test_exclusive_roundtrip_and_existing_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            path = Path(root) / "savepoint.json"
            checkpoint.write_new(path, b"first")
            self.assertEqual(checkpoint.read(path), b"first")
            with self.assertRaises(FileExistsError):
                checkpoint.write_new(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual(list(Path(root).iterdir()), [path])

    def test_atomic_publication_failure_leaves_no_destination_or_temporary(
        self,
    ) -> None:
        for operation in ("os.link", "os.fsync"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as root,
            ):
                root = Path(root).resolve()
                path = Path(root) / "savepoint.json"
                with patch(
                    f"whisper_runtime.adapters._checkpoint_io.{operation}",
                    side_effect=OSError("injected"),
                ):
                    with self.assertRaises(OSError):
                        checkpoint.write_new(path, b"payload")
                self.assertEqual(list(Path(root).iterdir()), [])

    def test_concurrent_writers_never_replace_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            path = Path(root) / "savepoint.json"

            def attempt(value: bytes) -> bytes | None:
                try:
                    checkpoint.write_new(path, value)
                except FileExistsError:
                    return None
                return value

            with ThreadPoolExecutor(max_workers=8) as executor:
                outcomes = tuple(
                    executor.map(attempt, [bytes([index]) * 4096 for index in range(8)])
                )
            winners = [value for value in outcomes if value is not None]
            self.assertEqual(len(winners), 1)
            self.assertEqual(checkpoint.read(path), winners[0])
            self.assertEqual(list(Path(root).iterdir()), [path])

    def test_temp_collision_does_not_delete_an_unowned_file(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            path = Path(root) / "savepoint.json"
            existing = Path(root) / ".savepoint-fixed.tmp"
            existing.write_bytes(b"unowned")
            with patch.object(
                checkpoint.uuid, "uuid4", return_value=SimpleNamespace(hex="fixed")
            ):
                with self.assertRaises(FileExistsError):
                    checkpoint.write_new(path, b"payload")
            self.assertEqual(existing.read_bytes(), b"unowned")
            self.assertFalse(path.exists())

    def test_file_growth_after_stat_is_still_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            path = Path(root) / "savepoint.json"
            path.write_bytes(b"small")
            original_fstat = os.fstat

            def grow_after_stat(descriptor: int) -> os.stat_result:
                result = original_fstat(descriptor)
                with path.open("ab") as stream:
                    stream.write(b"x" * 32)
                return result

            with (
                patch.object(checkpoint, "_MAX_BYTES", 16),
                patch.object(checkpoint.os, "fstat", side_effect=grow_after_stat),
            ):
                with self.assertRaises(ValueError):
                    checkpoint.read(path)

    def test_oversized_read_write_and_directory_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            path = Path(root) / "savepoint.json"
            with patch.object(checkpoint, "_MAX_BYTES", 16):
                with self.assertRaises(ValueError):
                    checkpoint.write_new(path, b"x" * 17)
                self.assertFalse(path.exists())
                path.write_bytes(b"x" * 17)
                with self.assertRaises(ValueError):
                    checkpoint.read(path)
            with self.assertRaises(OSError):
                checkpoint.read(root)

    def test_symlink_file_and_parent_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            directory = Path(root)
            actual = directory / "actual"
            actual.mkdir()
            original = actual / "saved.json"
            original.write_bytes(b"untouched")
            self.assertEqual(checkpoint.read(original), b"untouched")
            linked_file, linked_dir = directory / "file-link", directory / "dir-link"
            try:
                linked_file.symlink_to(original)
                linked_dir.symlink_to(actual, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation unavailable")
            for path in (linked_file, linked_dir / "saved.json"):
                with self.subTest(path=path), self.assertRaises(OSError):
                    checkpoint.read(path)
            with self.assertRaises(OSError):
                checkpoint.write_new(linked_file, b"replacement")
            with self.assertRaises(OSError):
                checkpoint.write_new(linked_dir / "new.json", b"replacement")
            self.assertEqual(original.read_bytes(), b"untouched")
            self.assertFalse((actual / "new.json").exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX named pipes only")
    def test_fifo_read_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            path = Path(root) / "pipe"
            os.mkfifo(path)
            with self.assertRaises(OSError):
                checkpoint.read(path)


if __name__ == "__main__":
    unittest.main()
