"""Local scripted sanity only: these tests do not run or qualify a native soak."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infra import modal_release_soak as p
from whisper_runtime.adapters import StreamEventKind, TranscriptEvent
from whisper_runtime.native_setup import CLI_STREAM_CONFIG, MODEL_FINGERPRINT


@dataclass
class Metrics:
    accepted_samples: int = 0
    committed_samples: int = 0
    buffered_samples: int = 0
    peak_buffered_samples: int = 0


class ScriptedStream:
    def __init__(self):
        self.config = CLI_STREAM_CONFIG
        self.metrics = Metrics()
        self.ready = self.done = self.closed = False
        self.accepted_samples = 0

    def push(self, sequence, pcm):
        self.accepted_samples += len(pcm) // 2
        self.metrics.accepted_samples = self.accepted_samples
        self.metrics.buffered_samples = self.accepted_samples
        self.metrics.peak_buffered_samples = self.accepted_samples

    def finish_input(self):
        self.ready = True

    def step(self):
        self.done = True
        self.metrics.committed_samples = self.accepted_samples
        self.metrics.buffered_samples = 0
        common = dict(
            segment_id="scripted",
            revision=1,
            start_sample=0,
            end_sample=self.accepted_samples,
            sample_rate_hz=16000,
            session_version=1,
        )
        return [
            TranscriptEvent(1, StreamEventKind.PROVISIONAL, text="scripted", **common),
            TranscriptEvent(
                2,
                StreamEventKind.COMMIT,
                committed_through_sample=self.accepted_samples,
                committed_through_ms=self.accepted_samples // 16,
                **common,
            ),
            TranscriptEvent(3, StreamEventKind.FINAL, session_version=1),
        ]

    def close(self):
        self.closed = True


class ScriptedOwner:
    def __init__(self):
        self.instance_id, self.loads, self.sessions = "scripted-owner", 1, 0
        self.streams, self.threads = [], []

    def factory(self, phase):
        import threading

        stream = ScriptedStream()
        self.streams.append(stream)
        self.threads.append(threading.get_ident())
        self.sessions += 1
        number = self.sessions

        def finalize():
            return {
                "capacity_restored": stream.closed,
                "model_unchanged": True,
                "model_loads": self.loads,
                "source_digest": "source",
                "instance_id": self.instance_id,
                "session_number": number,
                "model_initial_sha256": MODEL_FINGERPRINT,
                "model_final_sha256": MODEL_FINGERPRINT,
                "stream_config": asdict(CLI_STREAM_CONFIG),
                "config_overrides": {},
                "peak_buffered_samples": stream.metrics.peak_buffered_samples,
                "max_buffer_samples": CLI_STREAM_CONFIG.max_buffer_ms * 16,
            }

        return stream, finalize


def memory():
    return {
        "rss_bytes": 1024 * p.MIB,
        "allocated_bytes": 200 * p.MIB,
        "reserved_bytes": 512 * p.MIB,
        "peak_allocated_bytes": 300 * p.MIB,
        "peak_reserved_bytes": 512 * p.MIB,
    }


def short_plan():
    seed = b"\x01\x00" * 320
    return seed, {
        "seed_sha256": p.prior._sha(seed),
        "profile": p.profile_contract(),
        "outer_seconds": 10,
        "smoke_reference": "scripted",
        "max_smoke_word_edit_rate": 0.1,
        "phases": [
            {
                "phase": name,
                "sample_count": 640,
                "sha256": p.source_hash(seed, 640),
                "timeout_seconds": 3,
            }
            for name in p.PHASES
        ],
    }


def messages(owner=None):
    seed, plan = short_plan()
    return list(
        p.run_suite(
            owner or ScriptedOwner(),
            seed,
            plan,
            {"digest": "source"},
            memory=memory,
            score=lambda text, reference: {"word_edit_rate": 0},
        )
    ), plan


class PlanTests(unittest.TestCase):
    def test_incremental_source_and_hour_cap(self):
        seed = b"\x01\x00" * 337
        chunks = list(p.frames(seed, 337 * 2 + 7))
        self.assertEqual(b"".join(chunks), seed * 2 + bytes(14))
        self.assertTrue(all(len(chunk) == 640 for chunk in chunks[:-1]))
        for seed, samples in (
            (b"", 1),
            (b"x", 1),
            (b"ab", True),
            (b"ab", 0),
            (b"ab", 3600 * 16000 + 1),
        ):
            with self.assertRaises(ValueError):
                list(p.frames(seed, samples))

    def test_exact_four_hours_plus_smoke_and_budget(self):
        seed = bytes(744800 * 2)
        original = {
            key: "registered"
            for key in (
                "recipe",
                "main_case",
                "main_sha256",
                "noisy_source_case",
                "noisy_prefix_sha256",
                "smoke_reference",
                "quality_calibration",
            )
        }
        original.update(noisy_prefix_samples=174240, max_smoke_word_edit_rate=0.1)
        with patch.object(p.live, "input_plan", return_value=(seed, original)):
            _, plan = p.input_plan()
        self.assertEqual([stage["phase"] for stage in plan["phases"]], list(p.PHASES))
        self.assertEqual(
            [stage["sample_count"] / 16000 for stage in plan["phases"]],
            [46.55, 3600, 3600, 3600, 3600],
        )
        budget = p.budget_plan()
        self.assertEqual(budget["outer_seconds"], 14800)
        self.assertLess(budget["planning_compute_usd"], 5.17)
        self.assertFalse(budget["physical_spending_cap"])
        self.assertEqual(budget["configured_retries"], 0)

    def test_preflight_has_no_modal_import_no_audio_files_and_rejects_drift(self):
        seed, plan = short_plan()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(p, "input_plan", return_value=(seed, plan)),
            patch.object(p, "snapshot", return_value={"digest": "frozen"}),
            patch.object(p, "resources") as resources,
            patch.dict(sys.modules, {"modal": None}),
        ):
            options = dict(root=Path(directory), replay_id="scripted")
            path = p.run(preflight=True, **options)
            self.assertEqual(
                json.loads(path.read_text())["status"], "local-preflight-passed"
            )
            self.assertFalse(list(path.parent.glob("*.pcm")))
            resources.assert_not_called()
            with self.assertRaisesRegex(ValueError, "choose_preflight"):
                p.run(**options)
            with (
                patch.object(p, "snapshot", return_value={"digest": "changed"}),
                self.assertRaisesRegex(ValueError, "preflight_no_longer_matches"),
            ):
                p.run(confirm_paid_gpu=True, **options)
            resources.assert_not_called()

    def test_named_profile_and_stale_cli_configuration_rejected(self):
        self.assertEqual(p.profile_contract()["name"], "conservative-v1")
        with patch(
            "whisper_runtime.native_setup.CLI_STREAM_CONFIG",
            replace(CLI_STREAM_CONFIG, left_context_ms=4000),
        ):
            with self.assertRaisesRegex(ValueError, "stale_or_nonconservative"):
                p.profile_contract()
        seed, plan = short_plan()
        plan["profile"]["stream_config"]["max_draft_tokens"] = 32
        with self.assertRaisesRegex(ValueError, "stale_profile"):
            list(p.run_suite(ScriptedOwner(), seed, plan, {"digest": "source"}))

    def test_paid_collector_failure_preserves_app_id_and_journal(self):
        seed, plan = short_plan()
        app = SimpleNamespace(app_id="ap-scripted", run=lambda **kwargs: nullcontext())
        endpoint = Mock()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(p, "input_plan", return_value=(seed, plan)),
            patch.object(p, "snapshot", return_value={"digest": "frozen"}),
            patch.object(p, "resources", return_value=(app, endpoint)),
            patch.object(p, "collect", side_effect=p.SoakError("memory_gate")),
            patch.dict(sys.modules, {"modal": SimpleNamespace(__version__="1.5.5")}),
        ):
            options = dict(root=Path(directory), replay_id="scripted-app-failure")
            p.run(preflight=True, **options)
            output = p.run(confirm_paid_gpu=True, **options)
            record = json.loads(output.read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["app_id"], "ap-scripted")
            self.assertEqual(record["error_code"], "memory_gate")
            journal = [
                json.loads(row)
                for row in output.with_name("release-soak.attempt.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(journal[1]["event"], "app-started")
            self.assertEqual(journal[1]["app_id"], "ap-scripted")

    def test_native_exception_text_is_not_a_harness_reason_code(self):
        self.assertEqual(
            p.failure_code(p.SoakError("source_late"), "fallback"), "source_late"
        )
        self.assertEqual(
            p.failure_code(ValueError("secret-from-provider"), "fallback"), "fallback"
        )


class DriverTests(unittest.TestCase):
    def test_clock_argmax_records_survive_later_frames_and_ties(self):
        timer = SimpleNamespace(value=100.0, late=0.0)

        def wait(delay):
            timer.value += delay + timer.late
            return False

        clock = p.SourceClock(100.0, now=lambda: timer.value)
        for sequence, (late, gap) in enumerate(
            ((0.125, 0.25), (0.0625, 0.125), (0.125, 0.25))
        ):
            timer.late = late
            clock.wait(
                sequence,
                sequence * 16000,
                (sequence + 1) * 16000,
                SimpleNamespace(wait=wait),
            )
            clock.offered(sequence, (sequence + 1) * 16000)
            timer.value += gap
            clock.resumed()
        receipt = clock.receipt()
        self.assertEqual(receipt["schema_version"], "source-clock-v2")
        self.assertEqual(receipt["clock"], "time.monotonic")
        self.assertEqual(receipt["origin_monotonic_seconds"], 100.0)
        self.assertEqual(receipt["last_attempt"]["sequence"], 2)
        wake = receipt["max_wake_frame"]
        self.assertEqual(wake["sequence"], 0)
        self.assertEqual(wake["wake_lateness_seconds"], 0.125)
        self.assertEqual(wake["due_elapsed_seconds"], 1.0)
        self.assertEqual(wake["wake_elapsed_seconds"], 1.125)
        gap = receipt["max_yield_resume_interval"]
        self.assertEqual(gap["sequence"], 0)
        self.assertEqual((gap["start_sample"], gap["end_sample"]), (0, 16000))
        self.assertEqual(gap["gap_seconds"], 0.25)
        self.assertEqual(gap["yield_elapsed_seconds"], 1.125)
        self.assertEqual(gap["resume_elapsed_seconds"], 1.375)
        self.assertEqual(receipt["max_yield_resume_seconds"], gap["gap_seconds"])
        self.assertLess(len(p.encode(receipt)), 2048)

    def test_prefix_receipt_distinguishes_offered_from_native_accepted(self):
        clock = p.SourceClock(0.0, now=lambda: 1.0)
        clock.offered(1, 640)
        digest = hashlib.sha256(bytes(1280))
        stream = SimpleNamespace(metrics=Metrics(accepted_samples=320))
        receipt = p.source_prefix_receipt(clock, digest, stream)
        self.assertEqual(receipt["offered_samples"], 640)
        self.assertEqual(receipt["offered_sha256"], digest.hexdigest())
        self.assertEqual(receipt["native_accepted_samples"], 320)
        self.assertNotIn("accepted_sha256", receipt)
        terminal = {"metrics": {"accepted_samples": 640}}
        self.assertEqual(
            p.source_prefix_receipt(clock, digest, None, terminal)[
                "native_accepted_samples"
            ],
            640,
        )
        self.assertIsNone(p.source_prefix_receipt(None, None, None))

    def test_source_clock_deadline_and_gap_are_separate_without_rebasing(self):
        timer = SimpleNamespace(value=0.0)

        def wait(delay):
            timer.value += delay
            return False

        cancelled = SimpleNamespace(wait=wait)
        clock = p.SourceClock(0.0, now=lambda: timer.value)
        clock.wait(0, 0, 320, cancelled)
        clock.offered(0, 320)
        timer.value += 0.30  # The caller/push/scheduling interval, not a sleep.
        clock.resumed()
        with self.assertRaisesRegex(p.SoakError, "source_late"):
            clock.wait(1, 320, 640, cancelled)
        receipt = clock.receipt()
        self.assertAlmostEqual(receipt["max_yield_resume_seconds"], 0.30)
        self.assertAlmostEqual(receipt["max_wake_lateness_seconds"], 0.28)
        self.assertEqual(receipt["offered_frames"], 1)
        self.assertEqual(receipt["offered_samples"], 320)
        self.assertEqual(receipt["last_yield_sequence"], 0)
        self.assertEqual(receipt["last_yield_end_sample"], 320)
        failed = receipt["failed_frame"]
        self.assertEqual(failed["sequence"], 1)
        self.assertEqual((failed["start_sample"], failed["end_sample"]), (320, 640))
        self.assertEqual(failed["due_elapsed_seconds"], 0.04)
        self.assertEqual(failed["wake_elapsed_seconds"], 0.32)
        self.assertEqual(failed, receipt["last_attempt"])
        self.assertLess(len(p.encode(receipt)), 2048)

    def test_source_clock_exact_quarter_second_threshold_and_finite_receipt(self):
        for lateness, fails in ((0.25, False), (0.250001, True)):
            timer = SimpleNamespace(value=0.0)

            def wait(delay):
                timer.value += delay + lateness
                return False

            clock = p.SourceClock(0.0, now=lambda: timer.value)
            if fails:
                with self.assertRaisesRegex(p.SoakError, "source_late"):
                    clock.wait(0, 0, 16000, SimpleNamespace(wait=wait))
            else:
                clock.wait(0, 0, 16000, SimpleNamespace(wait=wait))
            self.assertEqual(clock.receipt()["failed_frame"] is not None, fails)
            self.assertEqual(clock.receipt()["max_yield_resume_seconds"], 0.0)
        for invalid in (float("nan"), float("inf"), -1.0):
            clock = p.SourceClock(0.0, now=lambda: invalid)
            with self.assertRaisesRegex(p.SoakError, "invalid_source_clock"):
                clock.wait(0, 0, 320, SimpleNamespace(wait=lambda _: False))
            p.encode(clock.receipt())

    def test_source_late_failure_preserves_receipt_and_stops_next_phase(self):
        original = p.SourceClock

        def late_clock(origin):
            return original(origin, now=lambda: origin + 1.0)

        owner = ScriptedOwner()
        with patch.object(p, "SourceClock", side_effect=late_clock):
            rows, _ = messages(owner)
        failure = next(row for row in rows if row["type"] == "phase_failed")
        self.assertEqual(failure["error_code"], "source_late")
        self.assertEqual(failure["source_clock"]["failed_frame"]["sequence"], 0)
        self.assertEqual(failure["source_clock"]["offered_samples"], 0)
        self.assertEqual(failure["source_prefix"]["offered_samples"], 0)
        self.assertEqual(failure["source_prefix"]["native_accepted_samples"], 0)
        self.assertEqual(failure["source_prefix"]["offered_sha256"], p.prior._sha(b""))
        self.assertTrue(failure["cleanup_verified"])
        self.assertEqual(owner.sessions, 1)

    def test_scripted_five_sessions_same_owner_full_events_hash_and_eof(self):
        owner = ScriptedOwner()
        rows, plan = messages(owner)
        self.assertEqual(rows[-1]["status"], "completed")
        self.assertEqual(owner.sessions, 5)
        self.assertEqual(len(set(owner.threads)), 1)
        self.assertTrue(all(stream.closed for stream in owner.streams))
        with tempfile.TemporaryDirectory() as directory:
            result = p.collect(iter(rows), Path(directory), plan, {"digest": "source"})
            self.assertEqual(result["status"], "completed")
            for report in result["phases"]:
                receipt = report["event_receipt"]
                raw = (Path(directory) / receipt["path"]).read_bytes()
                self.assertEqual(p.prior._sha(raw), receipt["sha256"])
                self.assertEqual(len(raw.splitlines()), 3)
                self.assertEqual(report["offered_samples"], 640)
                self.assertEqual(report["eof_chunks"], 2)
                self.assertEqual(report["source_clock"]["offered_samples"], 640)
                self.assertIsNone(report["source_clock"]["failed_frame"])

    def test_smoke_failure_does_not_start_hour_one(self):
        owner = ScriptedOwner()
        seed, plan = short_plan()
        rows = list(
            p.run_suite(
                owner,
                seed,
                plan,
                {"digest": "source"},
                memory=memory,
                score=lambda text, reference: {"word_edit_rate": 0.11},
            )
        )
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertEqual(rows[-1]["error_code"], "smoke_quality_gate")
        self.assertEqual(owner.sessions, 1)
        with tempfile.TemporaryDirectory() as directory:
            result = p.collect(iter(rows), Path(directory), plan, {"digest": "source"})
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["failure"]["cleanup_verified"])
            self.assertTrue(result["failure"]["event_receipt"]["partial"])
            self.assertIn("memory", result["failure"])
            self.assertTrue((Path(directory) / "smoke.terminal.json").is_file())

    def test_truncated_delivery_is_failed_even_after_real_final(self):
        rows, plan = messages()
        truncated = rows[
            : next(
                index for index, row in enumerate(rows) if row["type"] == "phase_done"
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = p.collect(
                iter(truncated), Path(directory), plan, {"digest": "source"}
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_code"], "missing_terminal")
            self.assertFalse((Path(directory) / "smoke.terminal.json").exists())

    def test_memory_gate_during_work_cannot_be_hidden_by_terminal_recovery(self):
        rows, plan = messages()
        bad = memory()
        bad["rss_bytes"] = p.MEMORY_POLICY["rss_ceiling_bytes"] + 1
        rows.insert(
            1,
            {
                "type": "memory",
                "phase": "smoke",
                "sample": bad,
                "terminal": False,
                "elapsed_seconds": 300,
            },
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "memory_gate"),
        ):
            p.collect(iter(rows), Path(directory), plan, {"digest": "source"})

    def test_bad_identity_coverage_or_exact_profile_cannot_pass(self):
        rows, plan = messages()
        for path, value in (
            (("metadata", "instance_id"), "replacement"),
            (("metadata", "model_loads"), 2),
            (("metadata", "source_digest"), "stale"),
            (("metadata", "model_final_sha256"), "bad"),
            (("metadata", "capacity_restored"), False),
            (("metrics", "committed_samples"), 639),
            (("metrics", "accepted_sha256"), "bad"),
            (("eof_chunks",), 1),
            (("final_count",), 0),
        ):
            changed = copy.deepcopy(rows)
            terminal = next(row for row in changed if row["type"] == "phase_done")
            target = terminal
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with (
                self.subTest(path=path),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                p.collect(iter(changed), Path(directory), plan, {"digest": "source"})

    def test_missing_event_and_duplicate_worker_start_fail(self):
        rows, plan = messages()
        changes = [rows[:2] + rows[3:], rows[:1] + rows]
        for changed in changes:
            with (
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                p.collect(iter(changed), Path(directory), plan, {"digest": "source"})

    def test_memory_gates_fixed_baseline_not_posthoc_ratchet(self):
        baseline = memory()
        self.assertTrue(p.memory_valid(memory(), baseline, terminal=True))
        for field, excess in (
            ("allocated_bytes", 1),
            ("rss_bytes", p.MEMORY_POLICY["rss_growth_bytes"] + 1),
            ("reserved_bytes", p.MEMORY_POLICY["reserved_growth_bytes"] + 1),
        ):
            sample = memory()
            sample[field] += excess
            self.assertFalse(p.memory_valid(sample, baseline, terminal=True))
        rows, plan = messages()
        terminal = next(
            row
            for row in rows
            if row["type"] == "phase_done" and row["phase"] == "hour-1"
        )
        terminal["memory"]["allocated_bytes"] += 1
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "memory_gate"),
        ):
            p.collect(iter(rows), Path(directory), plan, {"digest": "source"})


class CliExitTests(unittest.TestCase):
    def test_exit_code_matches_record_and_requested_mode(self):
        for mode, status, expected in (
            ("--preflight", "local-preflight-passed", 0),
            ("--confirm-paid-gpu", "completed", 0),
            ("--confirm-paid-gpu", "failed", 1),
            ("--confirm-paid-gpu", "local-preflight-passed", 1),
            ("--confirm-paid-gpu", None, 1),
            ("--preflight", "failed", 1),
        ):
            with (
                self.subTest(mode=mode, status=status),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "result.json"
                output.write_text(json.dumps({"status": status}), encoding="utf-8")
                with (
                    patch.object(p, "run", return_value=output),
                    patch("builtins.print"),
                ):
                    self.assertEqual(p.main(["--replay-id", "test", mode]), expected)


class CapacityTests(unittest.TestCase):
    def registration(self, profile=p.CAPACITY_PROFILE):
        seed, plan = short_plan()
        plan.update(
            mode=p.CAPACITY_MODE,
            capacity_limit=True,
            profile=p.profile_contract(profile),
            memory_policy=dict(p.CAPACITY_MEMORY_POLICY),
            submitted_resource_contract=p.resource_contract(),
            first_nonempty_commit_max_seconds=8.0,
        )
        return seed, plan

    def run_scripted(self, *, empty=False, sample=None, profile=p.CAPACITY_PROFILE):
        from whisper_runtime.native_setup import BACKEND_REUSE_TREE
        from whisper_runtime.profiles import get_profile

        selected = get_profile(profile)

        class Owner(ScriptedOwner):
            def factory(self, phase):
                stream, original_finalize = super().factory(phase)
                stream.config = selected.stream_config
                if empty:
                    original_step = stream.step

                    def step():
                        values = original_step()
                        values[0] = replace(values[0], text="   ")
                        return values

                    stream.step = step

                def finalize():
                    return {
                        **original_finalize(),
                        "stream_config": asdict(selected.stream_config),
                        "execution_profile": asdict(selected),
                        "native_profile_id": selected.native_profile_id,
                        "reuse_alignment_features": True,
                        "backend_tree": BACKEND_REUSE_TREE,
                        "backend_revision": "a" * 40,
                    }

                return stream, finalize

        seed, plan = self.registration(profile)
        owner = Owner()
        rows = list(
            p.run_suite(
                owner,
                seed,
                plan,
                {"digest": "source"},
                memory=lambda: (
                    sample
                    if sample is not None
                    else {**memory(), "rss_bytes": 6 * p.MIB * 1024}
                ),
                score=lambda text, reference: {"word_edit_rate": 0},
            )
        )
        return rows, plan, owner

    def test_capacity_defaults_to_v2_but_explicit_v1_keeps_zero_holdback(self):
        self.assertEqual(p.select_profile(), p.PROFILE)
        self.assertEqual(p.select_profile(capacity_limit=True), "low-latency-v2")
        for name, holdback in p.CAPACITY_PROFILES.items():
            self.assertEqual(p.select_profile(name, capacity_limit=True), name)
            rows, plan, _ = self.run_scripted(profile=name)
            self.assertEqual(
                plan["profile"]["stream_config"]["previous_holdback_ms"], holdback
            )
            with tempfile.TemporaryDirectory() as directory:
                report = p.collect(
                    iter(rows), Path(directory), plan, {"digest": "source"}
                )
            self.assertEqual(report["status"], "completed")
            changed = copy.deepcopy(rows)
            next(row for row in changed if row["type"] == "phase_done")["metadata"][
                "reuse_alignment_features"
            ] = False
            with (
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                p.collect(iter(changed), Path(directory), plan, {"digest": "source"})

    def test_legacy_missing_holdback_is_zero_without_mutating_receipt(self):
        for name, holdback in p.CAPACITY_PROFILES.items():
            _, plan = self.registration(name)
            del plan["profile"]["stream_config"]["previous_holdback_ms"]
            original = copy.deepcopy(plan)
            if holdback == 0:
                self.assertTrue(p.capacity_mode(plan))
            else:
                with self.assertRaisesRegex(ValueError, "stale_profile"):
                    p.capacity_mode(plan)
            self.assertEqual(plan, original)

    def test_new_contract_reuses_gates_without_changing_historical_rss_policy(self):
        original = dict(p.MEMORY_POLICY)
        sample = {**memory(), "rss_bytes": 6 * 1024 * p.MIB}
        self.assertFalse(p.memory_valid(sample, None, terminal=True))
        self.assertTrue(
            p.memory_valid(sample, None, terminal=True, capacity_limit=True)
        )
        baseline = dict(sample)
        sample["rss_bytes"] += p.MEMORY_POLICY["rss_growth_bytes"]
        self.assertTrue(
            p.memory_valid(sample, baseline, terminal=True, capacity_limit=True)
        )
        sample["rss_bytes"] += 1
        self.assertFalse(
            p.memory_valid(sample, baseline, terminal=True, capacity_limit=True)
        )
        for missing in ("rss_bytes", "allocated_bytes", "reserved_bytes"):
            candidate = dict(baseline)
            del candidate[missing]
            self.assertFalse(
                p.memory_valid(candidate, baseline, terminal=True, capacity_limit=True)
            )
        self.assertEqual(p.MEMORY_POLICY, original)
        self.assertNotIn("rss_ceiling_bytes", p.CAPACITY_MEMORY_POLICY)

    def test_five_sessions_have_exact_profile_and_recomputed_first_commit(self):
        rows, plan, owner = self.run_scripted()
        self.assertEqual(owner.sessions, 5)
        self.assertEqual(rows[0]["type"], "resource_contract")
        with tempfile.TemporaryDirectory() as directory:
            report = p.collect(iter(rows), Path(directory), plan, {"digest": "source"})
        self.assertEqual(report["status"], "completed")
        self.assertLessEqual(report["phases"][0]["first_nonempty_commit_seconds"], 8.0)
        self.assertTrue(
            all(
                row["metadata"]["execution_profile"] == plan["profile"]
                for row in report["phases"]
            )
        )

    def test_empty_commit_or_missing_guest_measurement_stops_after_smoke(self):
        for options, reason in (
            ({"empty": True}, "first_commit_gate"),
            (
                {
                    "sample": {
                        key: value
                        for key, value in memory().items()
                        if key != "rss_bytes"
                    }
                },
                "memory_gate",
            ),
        ):
            rows, _, owner = self.run_scripted(**options)
            self.assertEqual(owner.sessions, 1)
            self.assertEqual(rows[-1]["error_code"], reason)
            self.assertTrue(all(stream.closed for stream in owner.streams))

    def test_collector_rejects_forged_late_or_malformed_first_commit_evidence(self):
        rows, plan, _ = self.run_scripted()
        for mutation in (
            "summary",
            "late",
            "missing_clock",
            "bad_clock",
            "profile",
            "resource",
        ):
            changed = copy.deepcopy(rows)
            terminal = next(row for row in changed if row["type"] == "phase_done")
            if mutation == "summary":
                terminal["first_nonempty_commit_seconds"] = 0.0
            elif mutation == "late":
                for row in changed:
                    if row["type"] == "event" and row["phase"] == "smoke":
                        row["worker_elapsed_ns"] += 9_000_000_000
                terminal["first_nonempty_commit_seconds"] += 9.0
            elif mutation in ("missing_clock", "bad_clock"):
                event = next(row for row in changed if row["type"] == "event")
                if mutation == "missing_clock":
                    del event["worker_elapsed_ns"]
                else:
                    event["worker_elapsed_ns"] = True
            elif mutation == "profile":
                terminal["metadata"]["reuse_alignment_features"] = False
            else:
                changed[0]["instance_id"] = ""
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                p.collect(iter(changed), Path(directory), plan, {"digest": "source"})

    def test_capacity_preflight_freezes_bytes_and_paid_rejects_drift_or_mode(self):
        seed, plan = self.registration()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, review = [], []
            for name, target in (("source.py", files), ("plan.md", review)):
                raw = name.encode()
                (root / name).write_bytes(raw)
                target.append(
                    {"path": name, "size_bytes": len(raw), "sha256": p.prior._sha(raw)}
                )
            expected = {
                "files": files,
                "digest": p.prior._canonical(files),
                "review_files_not_uploaded": review,
                "review_digest": p.prior._canonical(review),
            }
            with (
                patch.object(p, "input_plan", return_value=(seed, plan)),
                patch.object(p, "snapshot", return_value=expected),
                patch.object(p, "resources") as resources,
            ):
                options = dict(
                    root=root,
                    replay_id="new-capacity",
                    profile=p.CAPACITY_PROFILE,
                    capacity_limit=True,
                )
                output = p.run(preflight=True, **options)
                self.assertEqual(
                    (output.parent / "source/source.py").read_bytes(), b"source.py"
                )
                self.assertEqual(
                    (output.parent / "review/plan.md").read_bytes(), b"plan.md"
                )
                self.assertEqual((output.parent / "seed.pcm").read_bytes(), seed)
                with self.assertRaises(ValueError):
                    p.run(confirm_paid_gpu=True, **{**options, "capacity_limit": False})
                (output.parent / "review/plan.md").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    p.run(confirm_paid_gpu=True, **options)
                resources.assert_not_called()

    def test_input_and_snapshot_require_explicit_candidate_and_capture_reuse_patch(
        self,
    ):
        for profile, capacity in ((p.PROFILE, True), (p.CAPACITY_PROFILE, False)):
            with self.assertRaisesRegex(
                ValueError, "profile_requires_registered_capacity_mode"
            ):
                p.input_plan(profile=profile, capacity_limit=capacity)
        observed = p.snapshot(profile=p.CAPACITY_PROFILE, capacity_limit=True)
        self.assertIn(p.live.REUSE_PATCH, {row["path"] for row in observed["files"]})
        self.assertIn(
            p.CAPACITY_PLAN_DOC,
            {row["path"] for row in observed["review_files_not_uploaded"]},
        )


class ResourceTests(unittest.TestCase):
    def test_resource_registration_is_single_bounded_readonly_function(self):
        modal = Mock()
        captured = {}

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
        for key, value in {
            "gpu": "T4",
            "cpu": 2,
            "memory": 4096,
            "timeout": 14800,
            "min_containers": 0,
            "max_containers": 1,
            "buffer_containers": 0,
            "block_network": True,
            "restrict_modal_access": True,
            "include_source": False,
        }.items():
            self.assertEqual(captured[key], value)
        self.assertNotIn("retries", captured)
        modal.Volume.from_name.assert_called_once_with(
            "existing", create_if_missing=False
        )
        modal.Volume.from_name.return_value.with_mount_options.assert_called_once_with(
            read_only=True
        )

    def test_pinned_sdk_accepts_real_resource_definition_without_hydration(self):
        try:
            import modal
        except ImportError:
            self.skipTest("optional pinned Modal SDK is not installed")
        if modal.__version__ != "1.5.5":
            self.skipTest("this registration requires Modal SDK 1.5.5")
        # Exercise the actual SDK decorators, images and read-only volume
        # handles, not a permissive mock. Nothing is hydrated or run remotely.
        forbidden = AssertionError("resource-definition test attempted remote work")
        with (
            patch.object(modal.App, "run", side_effect=forbidden),
            patch.object(modal.App, "deploy", side_effect=forbidden),
            patch.object(modal.Volume, "hydrate", side_effect=forbidden),
            patch.object(modal.Function, "hydrate", side_effect=forbidden),
            patch.object(modal.Function, "remote_gen", side_effect=forbidden),
            patch("socket.socket.connect", side_effect=forbidden),
            patch("socket.socket.connect_ex", side_effect=forbidden),
        ):
            app, endpoint = p.resources(modal, p.snapshot(), {})
            from modal._resources import convert_fn_config_to_resources_config

            def serialized(function):
                spec = function._spec_
                return convert_fn_config_to_resources_config(
                    cpu=spec.cpu, memory=spec.memory, gpu=spec.gpus
                )

            original_resources = serialized(endpoint)
            _, registration = CapacityTests().registration()
            _, capacity_endpoint = p.resources(
                modal,
                p.snapshot(profile=p.CAPACITY_PROFILE, capacity_limit=True),
                registration,
            )
            capacity_resources = serialized(capacity_endpoint)
        self.assertIsNone(app.app_id)
        self.assertFalse(endpoint.is_hydrated)
        self.assertFalse(capacity_endpoint.is_hydrated)
        self.assertEqual(
            (original_resources.memory_mb, original_resources.memory_mb_max), (4096, 0)
        )
        self.assertEqual(
            (capacity_resources.memory_mb, capacity_resources.memory_mb_max),
            (4096, 4096),
        )
        self.assertEqual(
            (capacity_resources.milli_cpu, capacity_resources.milli_cpu_max), (2000, 0)
        )
        self.assertEqual(p.budget_plan()["configured_retries"], 0)
        self.assertIsNone(p.budget_plan()["sdk_retry_policy"])


if __name__ == "__main__":
    unittest.main()
