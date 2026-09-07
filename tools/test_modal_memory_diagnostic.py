"""Local-only memory diagnostic tests: no native inference or remote work."""

from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infra import modal_memory_diagnostic as p
from tools.test_modal_release_soak import ScriptedOwner


def memory():
    return {
        "errors": [],
        "statm": {"rss_bytes": 6 * 1024**3},
        "cuda": {
            "initialized": True,
            "allocated_bytes": 100,
            "reserved_bytes": 200,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        },
    }


@contextmanager
def scripted_factory(observer):
    for phase in p.REQUIRED_SNAPSHOTS[1:5]:
        if phase not in observer.seen:
            observer.capture(phase)
    yield


def plan(*, capacity_smoke=False):
    seed = bytes(1280)
    return seed, {
        "mode": p.CAPACITY_MODE if capacity_smoke else p.DIAGNOSTIC_MODE,
        "seed_sha256": p.prior._sha(seed),
        "profile": p.soak.profile_contract(),
        "outer_seconds": 120,
        "smoke_reference": "scripted",
        "max_smoke_word_edit_rate": 0.1,
        "phases": [
            {
                "phase": name,
                "sample_count": 640,
                "sha256": p.prior._sha(seed),
                "timeout_seconds": 3,
            }
            for name in p.SESSIONS
        ],
    }


def messages(*, owner=None, sampler=memory, score=None, capacity_smoke=False):
    seed, registration = plan(capacity_smoke=capacity_smoke)
    owner = owner or ScriptedOwner()
    with patch.object(p.Observer, "factory_scope", scripted_factory):
        rows = list(
            p.run_diagnostic(
                seed,
                registration,
                {"digest": "source"},
                memory=sampler,
                owner_factory=lambda expected: owner,
                score=score or (lambda text, reference: {"word_edit_rate": 0.0}),
            )
        )
    return rows, registration, owner


class SamplingTests(unittest.TestCase):
    def test_capacity_sampler_does_not_read_detailed_smaps_or_import_native(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    p, "_smaps_metrics", side_effect=AssertionError("not lean")
                ) as detailed,
                patch.object(
                    p.importlib,
                    "import_module",
                    side_effect=AssertionError("native import"),
                ),
            ):
                sample = p.memory_sample(Path(directory), detailed=False)
        detailed.assert_not_called()
        self.assertIsNone(sample["detailed_memory"])
        self.assertEqual(sample["disabled_metrics"], ["detailed_memory"])
        self.assertFalse(p.sampling_failed(sample, True))
        self.assertTrue(p.sampling_failed(sample, False))

    def test_sampler_never_imports_native_or_initializes_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / "smaps_rollup").write_text(
                "Rss: 100 kB\nPss: 90 kB\nPrivate_Clean: 20 kB\nPrivate_Dirty: 30 kB\nShared_Clean: 40 kB\nShared_Dirty: 10 kB\n"
            )
            (proc / "statm").write_text("30 25 10 1 0 5 0\n")
            (proc / "status").write_text("Name:\ttest\nThreads:\t7\n")
            with (
                patch.object(p.os, "sysconf", return_value=4096, create=True),
                patch.object(
                    p.importlib,
                    "import_module",
                    side_effect=AssertionError("unexpected import"),
                ),
            ):
                result = p.memory_sample(proc)
            self.assertEqual(result["errors"], [])
            self.assertFalse(result["torch_imported"])
            self.assertEqual(result["cuda"], {"initialized": False})
            self.assertEqual(result["statm"]["rss_bytes"], 102400)
            self.assertEqual(result["detailed_memory"]["private_bytes"], 51200)
            self.assertEqual(result["detailed_memory"]["shared_bytes"], 51200)
            self.assertEqual(result["detailed_memory"]["source"], "smaps_rollup")
            self.assertEqual(result["missing_capabilities"], [])
            self.assertEqual(result["schema_version"], p.MEMORY_SAMPLE_SCHEMA)
            self.assertEqual(result["threads"], 7)
            cuda = Mock()
            cuda.is_initialized.return_value = False
            torch = SimpleNamespace(
                cuda=cuda, get_num_threads=lambda: 1, get_num_interop_threads=lambda: 2
            )
            with patch.dict(sys.modules, {"torch": torch}):
                result = p.memory_sample(proc)
            self.assertEqual(result["cuda"], {"initialized": False})
            self.assertEqual(result["torch_threads"], {"intra_op": 1, "inter_op": 2})
            cuda.synchronize.assert_not_called()
            cuda.memory_allocated.assert_not_called()
            cuda.is_available.assert_not_called()

    def test_partial_sampling_errors_are_fixed_components_not_exception_text(self):
        with tempfile.TemporaryDirectory() as directory:
            result = p.memory_sample(Path(directory))
        self.assertEqual(
            {error["component"] for error in result["errors"]},
            {"detailed_memory", "statm", "threads"},
        )
        self.assertIsNone(result["detailed_memory"])
        self.assertEqual(result["missing_capabilities"], ["smaps_rollup", "smaps"])
        self.assertNotIn(directory, json.dumps(result))

    def test_rollup_and_streamed_mapping_aggregation_have_identical_attribution(self):
        one = (
            "Rss: 50 kB\nPss: 45 kB\nPrivate_Clean: 10 kB\nPrivate_Dirty: 15 kB\n"
            "Shared_Clean: 20 kB\nShared_Dirty: 5 kB\nPrivate_Hugetlb: 1 kB\nShared_Hugetlb: 2 kB\n"
        )
        rollup = (
            "Rss: 100 kB\nPss: 90 kB\nPrivate_Clean: 20 kB\nPrivate_Dirty: 30 kB\n"
            "Shared_Clean: 40 kB\nShared_Dirty: 10 kB\nPrivate_Hugetlb: 2 kB\nShared_Hugetlb: 4 kB\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / "smaps_rollup").write_text(
                "1000-3000 ---p 0 00:00 0 [rollup]\n" + rollup
            )
            (proc / "smaps").write_text(
                "1000-2000 r--p 0 00:00 0 /first\n"
                + one
                + "KernelPageSize: 4 kB\n"
                + "2000-3000 rw-p 0 00:00 0 /second\n"
                + one
                + "VmFlags: rd wr\n"
            )
            native = p._smaps_metrics(proc / "smaps_rollup", aggregate=False)
            aggregate = p._smaps_metrics(proc / "smaps", aggregate=True)
            for key in ("fields_bytes", "private_bytes", "shared_bytes"):
                self.assertEqual(native[key], aggregate[key])
            self.assertNotIn("KernelPageSize", aggregate["fields_bytes"])
            self.assertEqual(aggregate["mapping_count"], 2)
            self.assertFalse(aggregate["host_physical_memory_measured"])
            self.assertFalse(aggregate["host_physical_memory_verified"])
            self.assertFalse(aggregate["private_shared_attribution_verified"])
            self.assertTrue(aggregate["reported_guest_estimates"])
            self.assertFalse(aggregate["atomic_snapshot"])
            original_open = Path.open

            def missing_rollup(path, *args, **kwargs):
                if path.name == "smaps_rollup":
                    raise FileNotFoundError()
                return original_open(path, *args, **kwargs)

            with (
                patch.object(Path, "open", missing_rollup),
                patch.object(
                    p.importlib,
                    "import_module",
                    side_effect=AssertionError("native import"),
                ),
            ):
                result = p.memory_sample(proc)
            self.assertEqual(
                result["detailed_memory"]["fields_bytes"], native["fields_bytes"]
            )
            self.assertEqual(result["detailed_memory"]["source"], "smaps_aggregate")
            self.assertEqual(result["missing_capabilities"], ["smaps_rollup"])
            self.assertNotIn(
                "detailed_memory", {error["component"] for error in result["errors"]}
            )

    def test_rollup_permission_or_parse_errors_do_not_trigger_fallback(self):
        for failure in (
            PermissionError("secret"),
            p.soak.SoakError("invalid_smaps_field"),
        ):
            with patch.object(p, "_smaps_metrics", side_effect=failure) as read:
                result = p.memory_sample(Path("unused"))
            read.assert_called_once_with(Path("unused/smaps_rollup"), aggregate=False)
            self.assertEqual(result["missing_capabilities"], [])
            self.assertIsNone(result["detailed_memory"])
            self.assertIn(
                {"component": "detailed_memory", "error_type": type(failure).__name__},
                result["errors"],
            )
            self.assertNotIn("secret", json.dumps(result))
            with patch.object(
                p, "_smaps_metrics", side_effect=[FileNotFoundError(), failure]
            ) as read:
                result = p.memory_sample(Path("unused"))
            self.assertEqual(read.call_count, 2)
            self.assertEqual(result["missing_capabilities"], ["smaps_rollup"])
            self.assertIsNone(result["detailed_memory"])
            self.assertIn(
                {"component": "detailed_memory", "error_type": type(failure).__name__},
                result["errors"],
            )

    def test_malformed_mapping_fields_and_stream_limits_are_rejected(self):
        fields = "".join(f"{name}: 1 kB\n" for name in p.SMAPS_REQUIRED)
        header = "1000-2000 rw-p 0 00:00 0\n"
        changes = (
            fields,  # smaps must identify a mapping, not silently assume rollup.
            header + fields.replace("Rss: 1 kB", "Rss: -1 kB"),
            header + fields.replace("Rss: 1 kB", "Rss: 1 MB"),
            header + fields + "Rss: 1 kB\n",
            header + fields + "2000-3000 rw-p 0 00:00 0\nRss: 1 kB\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smaps"
            for content in changes:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    p._smaps_metrics(path, aggregate=True)
            path.write_text(header + fields)
            with (
                patch.object(p, "SMAPS_MAX_BYTES", 16),
                self.assertRaisesRegex(ValueError, "smaps_read_limit"),
            ):
                p._smaps_metrics(path, aggregate=True)
            with (
                patch.object(p, "SMAPS_MAX_LINE_BYTES", 8),
                self.assertRaisesRegex(ValueError, "smaps_read_limit"),
            ):
                p._smaps_metrics(path, aggregate=True)

    def test_missing_detailed_capabilities_still_fail_completed_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / "statm").write_text("30 25 10 1 0 5 0\n")
            (proc / "status").write_text("Threads: 2\n")
            with patch.object(p.os, "sysconf", return_value=4096, create=True):
                rows, _, owner = messages(sampler=lambda: p.memory_sample(proc))
        self.assertEqual(owner.sessions, 2)
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertEqual(rows[-1]["error_code"], "memory_sampling_incomplete")
        sample = next(row for row in rows if row["type"] == "memory")["sample"]
        self.assertIsNone(sample["detailed_memory"])
        self.assertEqual(sample["missing_capabilities"], ["smaps_rollup", "smaps"])
        self.assertEqual(
            sample["errors"],
            [{"component": "detailed_memory", "error_type": "FileNotFoundError"}],
        )

    def test_already_initialized_cuda_is_fenced_without_peak_reset_or_empty_cache(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        for method, value in (
            ("memory_allocated", 10),
            ("memory_reserved", 20),
            ("max_memory_allocated", 30),
            ("max_memory_reserved", 40),
        ):
            getattr(cuda, method).return_value = value
        torch = SimpleNamespace(
            cuda=cuda, get_num_threads=lambda: 1, get_num_interop_threads=lambda: 2
        )
        with (
            patch.dict(sys.modules, {"torch": torch}),
            tempfile.TemporaryDirectory() as directory,
        ):
            result = p.memory_sample(Path(directory))
        self.assertEqual(result["cuda"]["peak_reserved_bytes"], 40)
        cuda.synchronize.assert_called_once_with(0)
        cuda.empty_cache.assert_not_called()
        cuda.reset_peak_memory_stats.assert_not_called()

    def test_scoped_lifecycle_and_dtw_wrappers_restore_success_and_failure(self):
        from whisper_runtime import native_setup as setup

        rows, value = [], object()
        cpu = Mock(return_value=value)
        gpu = Mock(side_effect=RuntimeError("do not expose secret"))
        timing = SimpleNamespace(dtw_cpu=cpu, dtw_cuda=gpu)
        imports = SimpleNamespace(import_module=Mock(return_value=SimpleNamespace()))
        backend = Mock(return_value=value)
        load, fingerprint = Mock(return_value=value), Mock(return_value="fingerprint")
        observer = p.Observer(rows.append, memory)
        with (
            patch.object(p.live, "importlib", imports),
            patch.object(setup, "_import_backend", backend),
            patch.object(setup, "_load_model", load),
            patch.object(setup, "_fingerprint", fingerprint),
            patch.dict(sys.modules, {"whisper.timing": timing}),
        ):
            with observer.factory_scope():
                p.live.importlib.import_module("torch")
                self.assertIs(setup._import_backend("verified", eager_cuda=True), value)
                backend.assert_called_once_with("verified", eager_cuda=True)
                # Legacy fixtures may also request whisper; do not double-wrap.
                p.live.importlib.import_module("whisper")
                self.assertIs(setup._load_model("a"), value)
                self.assertEqual(setup._fingerprint(value), "fingerprint")
                matrix = SimpleNamespace(shape=(2, 3))
                self.assertIs(timing.dtw_cpu(matrix), value)
                self.assertIs(timing.dtw_cpu(matrix), value)
                with self.assertRaisesRegex(RuntimeError, "secret"):
                    timing.dtw_cuda(matrix)
            self.assertIs(p.live.importlib, imports)
            self.assertIs(setup._import_backend, backend)
            self.assertIs(setup._load_model, load)
            self.assertIs(setup._fingerprint, fingerprint)
            observer.close()
            self.assertIs(timing.dtw_cpu, cpu)
            self.assertIs(timing.dtw_cuda, gpu)
        self.assertEqual(observer.counts, {"dtw_cpu": 2, "dtw_cuda": 1})
        operations = [row for row in rows if row["type"] == "operation"]
        self.assertEqual([row["status"] for row in operations], ["completed", "failed"])
        self.assertEqual(operations[0]["shape"], [2, 3])
        for operation in operations:
            self.assertEqual(operation["clock"], "time.monotonic")
            self.assertGreaterEqual(
                operation["end_monotonic_seconds"],
                operation["start_monotonic_seconds"],
            )
        self.assertNotIn("secret", json.dumps(rows))
        with patch.object(p.live, "importlib", imports):
            with self.assertRaises(RuntimeError), observer.factory_scope():
                raise RuntimeError()
            self.assertIs(p.live.importlib, imports)

    def test_operation_interval_uses_source_clock_reference(self):
        rows = []
        timing = SimpleNamespace(dtw_cuda=lambda matrix: matrix)
        observer = p.Observer(rows.append, memory)
        clock = p.soak.SourceClock(100.0, now=lambda: 100.0)
        with (
            patch.dict(sys.modules, {"whisper.timing": timing}),
            patch.object(p.time, "monotonic", side_effect=[101.125, 101.375]),
        ):
            observer.wrap_dtw()
            try:
                timing.dtw_cuda(SimpleNamespace(shape=(2, 3)))
            finally:
                observer.close()
        operation = next(row for row in rows if row["type"] == "operation")
        origin = clock.receipt()["origin_monotonic_seconds"]
        self.assertEqual(operation["start_monotonic_seconds"] - origin, 1.125)
        self.assertEqual(operation["end_monotonic_seconds"] - origin, 1.375)


class DiagnosticTests(unittest.TestCase):
    def test_named_low_latency_profile_reaches_owner_and_collector(self):
        for name in p.soak.CAPACITY_PROFILES:
            with self.subTest(profile=name):
                self.check_named_low_latency_profile(name)

    def check_named_low_latency_profile(self, name):
        from whisper_runtime.native_setup import BACKEND_REUSE_TREE
        from whisper_runtime.profiles import get_profile

        profile = get_profile(name)

        class LowOwner(ScriptedOwner):
            def factory(self, phase):
                stream, close = super().factory(phase)
                stream.config = profile.stream_config

                def finalize():
                    return {
                        **close(),
                        "stream_config": asdict(profile.stream_config),
                        "execution_profile": asdict(profile),
                        "native_profile_id": profile.native_profile_id,
                        "reuse_alignment_features": True,
                        "backend_tree": BACKEND_REUSE_TREE,
                        "backend_revision": "1" * 40,
                    }

                return stream, finalize

        seed, registration = plan(capacity_smoke=True)
        registration.update(profile=asdict(profile), max_first_commit_seconds=8)
        owner = LowOwner()
        factory = Mock(return_value=owner)
        with patch.object(p.Observer, "factory_scope", scripted_factory):
            rows = list(
                p.run_diagnostic(
                    seed,
                    registration,
                    {"digest": "source"},
                    memory=memory,
                    owner_factory=factory,
                    score=lambda text, reference: {"word_edit_rate": 0.0},
                )
            )
        factory.assert_called_once_with({"digest": "source"}, profile=name)
        self.assertEqual(rows[-1]["status"], "completed")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                p.collect(
                    iter(rows), Path(directory), registration, {"digest": "source"}
                )["status"],
                "completed",
            )
        for field in (
            "execution_profile",
            "reuse_alignment_features",
            "backend_tree",
            "native_profile_id",
        ):
            altered = copy.deepcopy(rows)
            next(row for row in altered if row["type"] == "session_done")["metadata"][
                field
            ] = None
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                p.collect(
                    iter(altered), Path(directory), registration, {"digest": "source"}
                )

    def test_profile_plan_preserves_v1_missing_zero_and_requires_explicit_v2_holdback(
        self,
    ):
        for name, holdback in p.soak.CAPACITY_PROFILES.items():
            _, registration = plan(capacity_smoke=True)
            registration.update(
                profile=p.soak.profile_contract(name), max_first_commit_seconds=8
            )
            self.assertEqual(p.profile_plan(registration), name)
            for limit in (None, True, 8.000000000000002):
                with (
                    self.subTest(name=name, limit=limit),
                    self.assertRaises(ValueError),
                ):
                    p.profile_plan({**registration, "max_first_commit_seconds": limit})
            del registration["profile"]["stream_config"]["previous_holdback_ms"]
            original = copy.deepcopy(registration)
            if holdback == 0:
                self.assertEqual(p.profile_plan(registration), name)
            else:
                with self.assertRaisesRegex(ValueError, "stale_profile"):
                    p.profile_plan(registration)
            self.assertEqual(registration, original)

    def test_first_commit_is_measured_from_committed_caption_and_checked_independently(
        self,
    ):
        rows, registration, _ = messages(capacity_smoke=True)
        registration["max_first_commit_seconds"] = 8
        first = {}
        for row in rows:
            if row["type"] == "event" and row["event"]["kind"] == "commit":
                self.assertIsNone(row["event"]["text"])
                first.setdefault(row["phase"], row["elapsed_seconds"])
            elif row["type"] == "session_done":
                self.assertEqual(row["first_commit_seconds"], first[row["phase"]])
        self.assertEqual(len(first), 2)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                p.collect(
                    iter(rows), Path(directory), registration, {"digest": "source"}
                )["status"],
                "completed",
            )
        for mutation in ("missing", "nonfinite", "backward", "late", "forged"):
            altered = copy.deepcopy(rows)
            event = next(
                row
                for row in altered
                if row["type"] == "event" and row["event"]["kind"] == "commit"
            )
            if mutation == "forged":
                next(row for row in altered if row["type"] == "session_done")[
                    "first_commit_seconds"
                ] = 0
            elif mutation == "missing":
                del event["elapsed_seconds"]
            else:
                event["elapsed_seconds"] = {
                    "nonfinite": float("nan"),
                    "backward": -1,
                    "late": 9,
                }[mutation]
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                p.collect(
                    iter(altered), Path(directory), registration, {"digest": "source"}
                )

    def test_worker_first_commit_deadline_stops_before_second_session(self):
        seed, registration = plan(capacity_smoke=True)
        registration["max_first_commit_seconds"] = 0
        owner = ScriptedOwner()
        with patch.object(p.Observer, "factory_scope", scripted_factory):
            rows = list(
                p.run_diagnostic(
                    seed,
                    registration,
                    {"digest": "source"},
                    memory=memory,
                    owner_factory=lambda expected: owner,
                    score=lambda text, reference: {"word_edit_rate": 0.0},
                )
            )
        self.assertEqual(owner.sessions, 1)
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertEqual(rows[-1]["error_code"], "first_commit_deadline")

    def test_capacity_mode_ignores_optional_guest_stats_but_keeps_gpu_and_receipt_gates(
        self,
    ):
        def unavailable_guest():
            return {
                **memory(),
                "errors": [
                    {"component": name, "error_type": "FileNotFoundError"}
                    for name in ("statm", "threads", "detailed_memory")
                ],
            }

        rows, registration, owner = messages(
            capacity_smoke=True, sampler=unavailable_guest
        )
        self.assertEqual(owner.sessions, 2)
        self.assertEqual(rows[-1]["status"], "completed")
        self.assertEqual(rows[0]["type"], "resource_capabilities")
        self.assertEqual(
            rows[0]["submitted_resource_contract"]["memory_limit_mib"], 4096
        )
        self.assertFalse(rows[0]["guest_attests_provider_enforcement"])
        with tempfile.TemporaryDirectory() as directory:
            report = p.collect(
                iter(rows), Path(directory), registration, {"digest": "source"}
            )
            self.assertEqual(report["status"], "completed")
        for mutate in (
            "capability",
            "missing_capability",
            "empty_instance",
            "null_instance",
            "exports",
            "cuda",
            "missing_cuda",
        ):
            altered = copy.deepcopy(rows)
            if mutate == "capability":
                altered[0]["submitted_resource_contract"]["memory_limit_mib"] = 0
            elif mutate == "missing_capability":
                altered = altered[1:]
                for number, row in enumerate(altered, 1):
                    row["sequence"] = number
            elif mutate in ("empty_instance", "null_instance"):
                altered[0]["instance_id"] = "" if mutate == "empty_instance" else None
            elif mutate == "exports":
                next(row for row in altered if row["type"] == "session_done")[
                    "export_sha256"
                ]["srt"] = "changed"
            else:
                sample = next(
                    row["sample"]
                    for row in altered
                    if row.get("phase") == "session-2_closed_gc"
                )
                if mutate == "cuda":
                    sample["cuda"]["allocated_bytes"] += 1
                else:
                    sample["cuda"] = None
            with (
                self.subTest(mutate=mutate),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                p.collect(
                    iter(altered), Path(directory), registration, {"digest": "source"}
                )
        with tempfile.TemporaryDirectory() as directory:
            report = p.collect(
                iter(rows[:-1]), Path(directory), registration, {"digest": "source"}
            )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_code"], "missing_terminal")

    def test_capacity_required_cuda_or_first_semantic_failure_stops_next_session(self):
        for cuda in (
            None,
            {"initialized": False},
            {"initialized": True, "allocated_bytes": 1},
        ):
            rows, _, owner = messages(
                capacity_smoke=True, sampler=lambda: {**memory(), "cuda": cuda}
            )
            self.assertEqual(owner.sessions, 1)
            self.assertEqual(rows[-1]["error_code"], "capacity_cuda_gate")
        rows, _, owner = messages(
            capacity_smoke=True, score=lambda text, reference: {"word_edit_rate": 1.0}
        )
        self.assertEqual(owner.sessions, 1)
        self.assertEqual(rows[-1]["error_code"], "quality_gate")
        baseline = memory()["cuda"]
        self.assertTrue(p.capacity_cuda_valid(memory(), baseline))
        for key, value in (
            ("allocated_bytes", 101),
            ("reserved_bytes", p.soak.MEMORY_POLICY["reserved_growth_bytes"] + 201),
            ("reserved_bytes", p.soak.MEMORY_POLICY["reserved_ceiling_bytes"] + 1),
            ("allocated_bytes", True),
        ):
            sample = memory()
            sample["cuda"][key] = value
            self.assertFalse(p.capacity_cuda_valid(sample, baseline))

    def test_consumer_closing_early_stops_owner_without_a_second_session(self):
        import threading

        seed, registration = plan()
        owner = ScriptedOwner()
        with patch.object(p.Observer, "factory_scope", scripted_factory):
            output = p.run_diagnostic(
                seed,
                registration,
                {"digest": "source"},
                memory=memory,
                owner_factory=lambda expected: owner,
                score=lambda text, reference: {"word_edit_rate": 0.0},
            )
            self.assertEqual(next(output)["phase"], "before_native_imports")
            output.close()
        self.assertLessEqual(owner.sessions, 1)
        self.assertTrue(all(stream.closed for stream in owner.streams))
        self.assertFalse(
            any(
                thread.name == "memory-diagnostic-native-owner"
                for thread in threading.enumerate()
            )
        )

    def test_owner_constructor_failure_propagates_without_starting_thread(self):
        seed, registration = plan()
        with (
            patch.object(p.threading.Thread, "start") as start,
            self.assertRaisesRegex(RuntimeError, "constructor"),
        ):
            list(
                p.run_diagnostic(
                    seed,
                    registration,
                    {"digest": "source"},
                    owner_factory=Mock(side_effect=RuntimeError("constructor")),
                )
            )
        start.assert_not_called()

    def test_two_equal_closed_sessions_keep_old_ram_gates_observational(self):
        original = dict(p.soak.MEMORY_POLICY)
        rows, registration, owner = messages()
        self.assertEqual(rows[-1]["status"], "completed")
        self.assertEqual(owner.sessions, 2)
        self.assertEqual(len(set(owner.threads)), 1)
        self.assertTrue(all(stream.closed for stream in owner.streams))
        with tempfile.TemporaryDirectory() as directory:
            report = p.collect(
                iter(rows), Path(directory), registration, {"digest": "source"}
            )
            self.assertEqual(report["status"], "completed")
            self.assertEqual(
                [row["phase"] for row in report["snapshots"]],
                list(p.REQUIRED_SNAPSHOTS),
            )
            self.assertEqual(
                report["sessions"][0]["text"], report["sessions"][1]["text"]
            )
        self.assertEqual(original, p.soak.MEMORY_POLICY)

    def test_first_semantic_failure_stops_second_session_and_sanitizes_failure(self):
        rows, _, owner = messages(score=lambda text, reference: {"word_edit_rate": 1.0})
        self.assertEqual(owner.sessions, 1)
        self.assertEqual(rows[-1]["error_code"], "quality_gate")
        self.assertEqual(rows[-1]["status"], "failed")

    def test_driver_failure_preserves_cleanup_metadata_and_failed_collector_status(
        self,
    ):
        profile = p.soak.profile_contract()

        class FailedDriverOwner(ScriptedOwner):
            def factory(self, phase):
                stream, finish = super().factory(phase)
                stream.step = Mock(side_effect=RuntimeError("private driver detail"))

                def finalize():
                    return {
                        **finish(),
                        "execution_profile": profile,
                        "native_profile_id": profile["native_profile_id"],
                        "reuse_alignment_features": False,
                    }

                return stream, finalize

        rows, registration, owner = messages(
            owner=FailedDriverOwner(), capacity_smoke=True
        )
        failure = rows[-1]
        self.assertEqual(owner.sessions, 1)
        self.assertEqual(failure["status"], "failed")
        self.assertFalse(failure["qualified"])
        self.assertTrue(failure["cleanup_verified"])
        self.assertIsNone(failure["terminal_before_gate"])
        metadata = failure["terminal_after_cleanup"]
        self.assertTrue(metadata["capacity_restored"])
        self.assertTrue(metadata["model_unchanged"])
        self.assertEqual(metadata["execution_profile"], profile)
        self.assertEqual(metadata["stream_config"], profile["stream_config"])
        self.assertEqual(metadata["source_digest"], "source")
        self.assertEqual(metadata["instance_id"], owner.instance_id)
        self.assertEqual(failure["source_prefix"]["native_accepted_samples"], 640)
        self.assertEqual(owner.streams[0].metrics.committed_samples, 0)
        self.assertFalse(any(row["type"] == "event" for row in rows))
        self.assertNotIn("private driver detail", json.dumps(rows))
        with tempfile.TemporaryDirectory() as directory:
            report = p.collect(
                iter(rows), Path(directory), registration, {"digest": "source"}
            )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["terminal"], failure)
        self.assertEqual(report["sessions"], [])

    def test_cleanup_exception_leaves_no_postcleanup_metadata_or_success(self):
        for failure_point in ("close", "finalize"):
            finalizer = Mock(side_effect=RuntimeError("private finalizer detail"))

            class FailedCleanupOwner(ScriptedOwner):
                def factory(self, phase):
                    stream, _ = super().factory(phase)
                    stream.step = Mock(
                        side_effect=RuntimeError("private driver detail")
                    )
                    if failure_point == "close":
                        stream.close = Mock(
                            side_effect=RuntimeError("private close detail")
                        )
                    return stream, finalizer

            with self.subTest(failure_point=failure_point):
                rows, _, owner = messages(
                    owner=FailedCleanupOwner(), capacity_smoke=True
                )
                failure = rows[-1]
                self.assertEqual(owner.sessions, 1)
                self.assertEqual(failure["status"], "failed")
                self.assertFalse(failure["qualified"])
                self.assertFalse(failure["cleanup_verified"])
                self.assertIsNone(failure["terminal_before_gate"])
                self.assertIsNone(failure["terminal_after_cleanup"])
                self.assertNotIn("private ", json.dumps(rows))
                if failure_point == "close":
                    finalizer.assert_not_called()
                else:
                    finalizer.assert_called_once_with()

    def test_postcleanup_metadata_does_not_replace_original_pre_gate_terminal(self):
        class CountedOwner(ScriptedOwner):
            def __init__(self):
                super().__init__()
                self.finalizations = 0

            def factory(self, phase):
                stream, finish = super().factory(phase)

                def finalize():
                    self.finalizations += 1
                    return {**finish(), "finalization_number": self.finalizations}

                return stream, finalize

        rows, registration, owner = messages(
            owner=CountedOwner(), score=lambda text, reference: {"word_edit_rate": 1.0}
        )
        failure = rows[-1]
        self.assertEqual(owner.finalizations, 2)
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["error_code"], "quality_gate")
        self.assertEqual(
            failure["terminal_before_gate"]["metadata"]["finalization_number"], 1
        )
        self.assertEqual(failure["terminal_after_cleanup"]["finalization_number"], 2)
        self.assertTrue(failure["cleanup_verified"])
        with tempfile.TemporaryDirectory() as directory:
            report = p.collect(
                iter(rows), Path(directory), registration, {"digest": "source"}
            )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["terminal"], failure)

    def test_source_late_preserves_clock_evidence_and_stops_second_session(self):
        original = p.soak.SourceClock

        def late_clock(origin):
            return original(origin, now=lambda: origin + 1.0)

        with patch.object(p.soak, "SourceClock", side_effect=late_clock):
            rows, _, owner = messages(capacity_smoke=True)
        failure = rows[-1]
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["error_code"], "source_late")
        self.assertEqual(failure["source_clock"]["failed_frame"]["sequence"], 0)
        self.assertEqual(failure["source_clock"]["offered_samples"], 0)
        self.assertEqual(failure["source_prefix"]["offered_samples"], 0)
        self.assertEqual(failure["source_prefix"]["native_accepted_samples"], 0)
        self.assertEqual(failure["source_prefix"]["offered_sha256"], p.prior._sha(b""))
        self.assertTrue(failure["cleanup_verified"])
        self.assertEqual(owner.sessions, 1)

    def test_sampler_failure_is_visible_and_never_success(self):
        def failed_sample():
            raise RuntimeError("secret path")

        rows, _, owner = messages(sampler=failed_sample)
        self.assertEqual(owner.sessions, 2)
        self.assertEqual(rows[-1]["error_code"], "memory_sampling_incomplete")
        self.assertNotIn("secret path", json.dumps(rows))
        self.assertTrue(
            next(row for row in rows if row["type"] == "memory")["sample"]["errors"]
        )

    def test_preimported_native_baseline_fails_before_model_work(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            rows, _, owner = messages()
        self.assertEqual(owner.sessions, 0)
        self.assertEqual(rows[-1]["error_code"], "baseline_native_already_imported")

    def test_collector_refuses_gaps_text_drift_and_false_success(self):
        rows, registration, _ = messages()
        variations = []
        changed = copy.deepcopy(rows)
        changed[0]["source_digest"] = "stale"
        variations.append(changed)
        changed = copy.deepcopy(rows)
        next(row for row in changed if row["type"] == "session_done")["text"] = (
            "changed"
        )
        variations.append(changed)
        changed = copy.deepcopy(rows)
        changed[-1]["model_loads"] = 2
        variations.append(changed)
        variations.append(rows[:2] + rows[3:])
        for changed in variations:
            with (
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                p.collect(
                    iter(changed), Path(directory), registration, {"digest": "source"}
                )


class PreflightTests(unittest.TestCase):
    def test_import_is_local_without_native_or_modal_dependencies(self):
        script = f"import sys; sys.path[:0] = [{str(p.ROOT)!r}, {str(p.ROOT / 'src')!r}]; import infra.modal_memory_diagnostic; assert not {{'torch', 'whisper', 'numpy', 'numba', 'modal'}} & sys.modules.keys()"
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_preflight_frozen_source_and_seed_fail_closed_without_modal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_bytes(b"reviewed")
            (root / "review.py").write_bytes(b"testbytes")
            reviewed = [
                {
                    "path": "review.py",
                    "size_bytes": 9,
                    "sha256": p.prior._sha(b"testbytes"),
                }
            ]
            expected = {
                "digest": p.prior._canonical(
                    [
                        {
                            "path": "source.py",
                            "size_bytes": 8,
                            "sha256": p.prior._sha(b"reviewed"),
                        }
                    ]
                ),
                "files": [
                    {
                        "path": "source.py",
                        "size_bytes": 8,
                        "sha256": p.prior._sha(b"reviewed"),
                    }
                ],
                "review_files_not_uploaded": reviewed,
                "review_digest": p.prior._canonical(reviewed),
            }
            seed, registration = plan()
            with (
                patch.object(p, "snapshot", return_value=expected),
                patch.object(p, "input_plan", return_value=(seed, registration)),
                patch.object(
                    p.importlib,
                    "import_module",
                    side_effect=AssertionError("must not load Modal"),
                ),
            ):
                output = p.run(replay_id="local-test", preflight=True, root=root)
                self.assertEqual(
                    json.loads(output.read_text())["status"], "local-preflight-passed"
                )
                self.assertEqual(
                    (output.parent / "source/source.py").read_bytes(), b"reviewed"
                )
                with self.assertRaisesRegex(ValueError, "preflight_no_longer_matches"):
                    p.run(
                        replay_id="local-test",
                        confirm_paid_gpu=True,
                        capacity_smoke=True,
                        root=root,
                    )
                self.assertEqual(
                    (output.parent / "review/review.py").read_bytes(), b"testbytes"
                )
                (output.parent / "review/review.py").write_bytes(b"different")
                with self.assertRaisesRegex(ValueError, "source file mismatch"):
                    p.run(replay_id="local-test", confirm_paid_gpu=True, root=root)
                (output.parent / "review/review.py").write_bytes(b"testbytes")
                with self.assertRaisesRegex(ValueError, "no_retry"):
                    p.run(replay_id="local-test", preflight=True, root=root)
                (output.parent / "source/source.py").write_bytes(b"changed!")
                with self.assertRaisesRegex(ValueError, "source file mismatch"):
                    p.run(replay_id="local-test", confirm_paid_gpu=True, root=root)
                (output.parent / "source/source.py").write_bytes(b"reviewed")
                (output.parent / "seed.pcm").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "frozen_seed_changed"):
                    p.run(replay_id="local-test", confirm_paid_gpu=True, root=root)
                expected["review_digest"] = "changed"
                with self.assertRaisesRegex(ValueError, "preflight_no_longer_matches"):
                    p.run(replay_id="local-test", confirm_paid_gpu=True, root=root)

    def test_no_authority_unsafe_namespace_or_existing_receipt_reaches_resources(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(p, "resources") as resources,
        ):
            for arguments in ({}, {"preflight": True, "confirm_paid_gpu": True}):
                with self.assertRaises(ValueError):
                    p.run(replay_id="test", root=Path(directory), **arguments)
            for name in ("..", "../escape", "CON", "a/b"):
                with self.assertRaises(ValueError):
                    p.run(replay_id=name, preflight=True, root=Path(directory))
            target = Path(directory) / "artifacts/modal-memory-diagnostic/used"
            target.mkdir(parents=True)
            (target / "attempt.jsonl").write_text("already used")
            with self.assertRaisesRegex(ValueError, "no_retry"):
                p.run(replay_id="used", confirm_paid_gpu=True, root=Path(directory))
            resources.assert_not_called()

    def test_cli_exit_code_is_nonzero_for_failed_or_unclean_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            for record, mode, expected in (
                ({"status": "failed"}, "--preflight", 1),
                (
                    {"status": "completed", "ephemeral_context_exit_completed": False},
                    "--confirm-paid-gpu",
                    1,
                ),
                (
                    {"status": "completed", "ephemeral_context_exit_completed": True},
                    "--confirm-paid-gpu",
                    0,
                ),
                ({"status": "local-preflight-passed"}, "--preflight", 0),
                ({"status": "local-preflight-passed"}, "--confirm-paid-gpu", 1),
                (
                    {"status": "completed", "ephemeral_context_exit_completed": True},
                    "--preflight",
                    1,
                ),
            ):
                output.write_text(json.dumps(record))
                with (
                    patch.object(p, "run", return_value=output),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(p.main(["--replay-id", "test", mode]), expected)
            with (
                patch.object(p, "run", side_effect=RuntimeError("secret")),
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(p.main(["--replay-id", "test", "--preflight"]), 1)
            self.assertNotIn("secret", stderr.getvalue())
            output.write_text(json.dumps({"status": "failed"}))
            with (
                patch.object(p, "run", return_value=output) as run,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    p.main(
                        [
                            "--replay-id",
                            "test",
                            "--confirm-paid-gpu",
                            "--capacity-smoke",
                        ]
                    ),
                    1,
                )
            self.assertTrue(run.call_args.kwargs["capacity_smoke"])

    def test_resources_are_single_bounded_readonly_no_retry_or_pool_settings(self):
        modal, captured = Mock(), {}

        def register(**kwargs):
            captured.update(kwargs)
            return lambda function: function

        modal.App.return_value.function.side_effect = register
        corpus = SimpleNamespace(
            BASE_COMMIT="pinned",
            b=SimpleNamespace(MODEL_CACHE_NAME="existing", MODEL_CACHE_MOUNT="/model"),
            _helper=lambda name: SimpleNamespace(
                DIRECT_IMAGE_PACKAGES=(),
                _build_command=lambda revision: "checked-build",
            ),
        )
        with patch.object(p.prior, "_corpus", return_value=corpus):
            p.resources(modal, {"files": []}, {}, Path("unused"))
        for name, expected in {
            "gpu": "T4",
            "cpu": 2,
            "memory": 4096,
            "timeout": 300,
            "startup_timeout": 180,
            "min_containers": 0,
            "max_containers": 1,
            "buffer_containers": 0,
            "block_network": True,
            "restrict_modal_access": True,
            "single_use_containers": True,
            "include_source": False,
        }.items():
            self.assertEqual(captured[name], expected)
        self.assertNotIn("retries", captured)
        modal.Volume.from_name.assert_called_once_with(
            "existing", create_if_missing=False
        )
        modal.Volume.from_name.return_value.with_mount_options.assert_called_once_with(
            read_only=True
        )
        self.assertEqual(p.budget_plan()["configured_retries"], 0)
        self.assertEqual(p.budget_plan()["planning_compute_usd"], 0.104517)

    def test_registration_keeps_exact_input_and_two_equal_sessions(self):
        seed, registration = p.input_plan()
        self.assertEqual(len(seed) // 2, 744800)
        self.assertEqual(registration["seed_sha256"], p.prior._sha(seed))
        self.assertEqual(
            [phase["phase"] for phase in registration["phases"]], list(p.SESSIONS)
        )
        self.assertEqual(
            registration["phases"][0]["sha256"], registration["phases"][1]["sha256"]
        )
        self.assertEqual(
            registration["release_memory_gates_unchanged"], p.soak.MEMORY_POLICY
        )
        self.assertTrue(registration["memory_is_observational"])
        capacity_seed, capacity = p.input_plan(capacity_smoke=True)
        self.assertEqual(capacity["profile"]["name"], "low-latency-v2")
        self.assertEqual(
            capacity["profile"]["stream_config"]["previous_holdback_ms"], 2000
        )
        self.assertEqual(capacity["max_first_commit_seconds"], 8)
        _, legacy = p.input_plan(capacity_smoke=True, profile="low-latency-v1")
        self.assertEqual(legacy["profile"]["name"], "low-latency-v1")
        self.assertEqual(legacy["profile"]["stream_config"]["previous_holdback_ms"], 0)
        self.assertEqual(legacy["max_first_commit_seconds"], 8)
        self.assertEqual(seed, capacity_seed)
        self.assertEqual(capacity["phases"], registration["phases"])
        self.assertEqual(capacity["mode"], p.CAPACITY_MODE)
        self.assertTrue(capacity["guest_memory_is_optional"])
        self.assertFalse(capacity["guest_rss_is_physical_capacity_gate"])
        self.assertFalse(capacity["detailed_memory_policy"]["enabled"])
        self.assertEqual(
            capacity["submitted_resource_contract"]["memory_limit_mib"], 4096
        )
        self.assertEqual(
            p.budget_plan()["submitted_resource_contract"]["memory_limit_mib"], 0
        )
        self.assertEqual(
            p.budget_plan(capacity_smoke=True)["submitted_resource_contract"][
                "memory_limit_mib"
            ],
            4096,
        )
        frozen = p.snapshot()
        self.assertIn(p.PRODUCER, {entry["path"] for entry in frozen["files"]})
        self.assertNotIn(p.TEST, {entry["path"] for entry in frozen["files"]})
        self.assertIn(
            p.CAPACITY_PLAN,
            {entry["path"] for entry in frozen["review_files_not_uploaded"]},
        )

    def test_pinned_sdk_resource_definition_without_hydration_or_network(self):
        try:
            import modal
        except ImportError:
            self.skipTest("optional Modal SDK is not installed")
        if modal.__version__ != "1.5.5":
            self.skipTest("registered Modal SDK 1.5.5 required")
        forbidden = AssertionError("resource definition attempted remote work")
        with (
            patch.object(modal.App, "run", side_effect=forbidden),
            patch.object(modal.App, "deploy", side_effect=forbidden),
            patch.object(modal.Volume, "hydrate", side_effect=forbidden),
            patch.object(modal.Function, "hydrate", side_effect=forbidden),
            patch.object(modal.Function, "remote_gen", side_effect=forbidden),
            patch("socket.socket.connect", side_effect=forbidden),
            patch("socket.socket.connect_ex", side_effect=forbidden),
        ):
            from modal._resources import convert_fn_config_to_resources_config

            for mode, memory, maximum in (
                (p.DIAGNOSTIC_MODE, 4096, 0),
                (p.CAPACITY_MODE, (4096, 4096), 4096),
            ):
                app, endpoint = p.resources(modal, p.snapshot(), {"mode": mode}, p.ROOT)
                self.assertIsNone(app.app_id)
                self.assertFalse(endpoint.is_hydrated)
                self.assertEqual(endpoint._spec_.memory, memory)
                serialized = convert_fn_config_to_resources_config(
                    cpu=endpoint._spec_.cpu,
                    memory=endpoint._spec_.memory,
                    gpu=endpoint._spec_.gpus,
                )
                self.assertEqual(serialized.memory_mb, 4096)
                self.assertEqual(serialized.memory_mb_max, maximum)


if __name__ == "__main__":
    unittest.main()
