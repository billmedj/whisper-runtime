"""Offline-only, synthetic coverage/identity/finality counterexamples."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from tools import summarize_live_qualification as audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = dict(
            commit="synthetic",
            files=[dict(path="fake.py", sha256="0" * 64, size_bytes=0)],
        )
        source["digest"] = audit.registration.prior._canonical(source["files"])
        self.plan = dict(
            cycle_samples=640,
            long_tail="whole cycles then digital silence",
            smoke_reference="one two",
            arms=dict(
                main=dict(start_sample=0, end_sample=160, reference="one"),
                noisy=dict(start_sample=320, end_sample=480, reference="two"),
            ),
            stages={
                phase: dict(sample_count=count, cycles=cycles, sha256=str(cycles) * 64)
                for phase, count, cycles in (("smoke", 640, 1), ("long", 1440, 2))
            },
        )
        self.specification = dict(source=source, input=self.plan, budget={})
        self.preflight = dict(status="local-preflight-passed", **self.specification)
        self.events, reports = {}, {}
        for phase, stage in self.plan["stages"].items():
            spans = audit._intervals(self.plan, stage)
            self.events[phase] = self.event_rows(
                [
                    (start, end, {"main": "one", "noisy": "two"}.get(arm, ""))
                    for start, end, _, arm in spans
                ]
            )
            done = dict(
                type="done",
                protocol=audit.LIVE_PROTOCOL,
                status="completed",
                source_eof_received=True,
                metrics=dict(
                    accepted_samples=stage["sample_count"],
                    committed_samples=stage["sample_count"],
                    accepted_sha256=stage["sha256"],
                    buffered_samples=0,
                    trace_count=0,
                    accepted_chunks=(stage["sample_count"] + 319) // 320,
                    event_count=0,
                ),
                metadata=dict(
                    capacity_restored=True,
                    model_unchanged=True,
                    model_loads=1,
                    source_digest=source["digest"],
                    instance_id="worker",
                    session_number=1 if phase == "smoke" else 2,
                    peak_buffered_samples=640,
                    max_buffer_samples=1280,
                ),
            )
            reports[phase] = dict(
                phase=phase,
                passed=True,
                committed_events=0,
                result=dict(
                    status="completed",
                    protocol=audit.LIVE_PROTOCOL,
                    sample_count=stage["sample_count"],
                    sha256=stage["sha256"],
                    server_done=done,
                    client_metrics=dict(
                        received_events=0,
                        sent_samples=stage["sample_count"],
                        sent_chunks=done["metrics"]["accepted_chunks"],
                        sender_finished=True,
                        eof_sent=True,
                    ),
                    qualification_receipt=dict(
                        status="available",
                        sha256="a" * 64,
                        record=dict(
                            done=done,
                            phase=phase,
                            source_digest=source["digest"],
                            instance_id="worker",
                        ),
                    ),
                ),
            )
        self.final = dict(
            status="completed",
            qualified=False,
            specification=self.specification,
            cleanup=dict(
                ephemeral_context_exit_completed=True, proxy_token_deleted=True
            ),
            qualification=dict(
                status="completed",
                long_gate_passed=True,
                same_worker_model=True,
                **reports,
            ),
        )
        self.write()

    @staticmethod
    def event_rows(spans):
        rows = []
        for version, (start, end, text) in enumerate(spans, 1):
            common = dict(
                segment_id=f"segment-{version}",
                revision=1,
                start_sample=start,
                end_sample=end,
                sample_rate_hz=16000,
                session_version=version,
            )
            for kind in (
                audit.StreamEventKind.PROVISIONAL,
                audit.StreamEventKind.COMMIT,
            ):
                extra = (
                    dict(text=text)
                    if kind is audit.StreamEventKind.PROVISIONAL
                    else dict(
                        committed_through_sample=end, committed_through_ms=end // 16
                    )
                )
                event = audit.TranscriptEvent(
                    sequence_number=len(rows) + 1, kind=kind, **common, **extra
                )
                rows.append(
                    dict(event=asdict(event), client_elapsed_ns=end * 62500 - 1_000_000)
                )
        rows.append(
            dict(
                event=asdict(
                    audit.TranscriptEvent(
                        sequence_number=len(rows) + 1,
                        kind=audit.StreamEventKind.FINAL,
                        session_version=len(spans),
                    )
                ),
                client_elapsed_ns=spans[-1][1] * 62500,
            )
        )
        return rows

    def write(self):
        for phase, events in self.events.items():
            raw = b"".join(
                (json.dumps(row, sort_keys=True) + "\n").encode() for row in events
            )
            name = f"live-v01-{phase}.events.jsonl"
            (self.root / name).write_bytes(raw)
            report = self.final["qualification"][phase]
            report["event_receipt"] = dict(
                path=name, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
            )
            report["committed_events"] = sum(
                row["event"]["kind"] == "commit" for row in events
            )
            report["result"]["client_metrics"]["received_events"] = len(events)
            report["result"]["server_done"]["metrics"]["event_count"] = len(events)
        (self.root / "live-v01.json").write_text(
            json.dumps(self.final), encoding="utf-8"
        )
        (self.root / "live-v01-preflight.json").write_text(
            json.dumps(self.preflight), encoding="utf-8"
        )

    def test_complete_repeated_reference_is_not_deduplicated_and_timing_is_signed(self):
        report = audit.summarize(self.root)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["registered_qualified"])
        long = report["stages"]["long"]
        self.assertEqual(long["committed_text"], "one two one two")
        self.assertEqual(long["whole_stage_score"]["reference_word_count"], 4)
        self.assertEqual(long["whole_stage_score"]["word_edit_distance"], 0)
        self.assertEqual(long["arrival_minus_source_end_ms"]["median"], -1)
        self.assertIn("NOT word latency", report["timing_scope"])
        self.assertEqual(long["commits"][-1]["attribution"]["arm"], "tail_silence")
        self.assertEqual(report["source"], self.specification["source"])

    def make_short(self):
        self.plan.update(short_only=True, config_overrides={"left_context_ms": 20000})
        self.plan["stages"].pop("long")
        config = {"left_context_ms": 20000, "word_context_limit_ms": 24000}
        self.plan["stages"]["smoke"]["stream_config"] = config
        self.final.update(status="completed-short")
        qualification = self.final["qualification"]
        qualification.update(
            status="completed-short", long=None, long_gate_passed=False
        )
        qualification.pop("same_worker_model")
        qualification["smoke"]["result"]["server_done"]["metadata"].update(
            stream_config=config,
            config_overrides=dict(self.plan["config_overrides"]),
        )
        self.events.pop("long")
        (self.root / "live-v01-long.events.jsonl").unlink()
        self.write()

    def test_completed_short_needs_no_long_log_and_is_not_full_qualification(self):
        self.make_short()
        report = audit.summarize(self.root)
        self.assertEqual(report["status"], "complete-short")
        self.assertEqual(report["registered_status"], "completed-short")
        self.assertFalse(report["registered_qualified"])
        self.assertEqual(set(report["stages"]), {"smoke"})
        self.assertIn("not full V0.1 qualification", report["scope"])

    def test_short_rejects_long_claims_and_changed_config(self):
        self.make_short()
        original = copy.deepcopy(self.final)
        for change in (
            lambda final: final.update(qualified=True),
            lambda final: final["qualification"].update(long={}),
            lambda final: final["qualification"].update(long_gate_passed=True),
            lambda final: final["qualification"].update(status="completed"),
            lambda final: final["qualification"]["smoke"]["result"]["server_done"][
                "metadata"
            ].update(config_overrides={}),
            lambda final: final["qualification"]["smoke"]["result"]["server_done"][
                "metadata"
            ].update(stream_config={}),
        ):
            self.final = copy.deepcopy(original)
            change(self.final)
            self.write()
            with self.assertRaises(ValueError):
                audit.summarize(self.root)

    def test_short_preserves_event_digest_and_source_checks(self):
        self.make_short()
        path = self.root / "live-v01-smoke.events.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"one"', b'"two"', 1))
        with self.assertRaisesRegex(ValueError, "changed event log"):
            audit.summarize(self.root)
        self.specification["source"]["files"][0]["sha256"] = "1" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "inventory digest"):
            audit.summarize(self.root)

    def test_cross_arm_and_cycle_commit_is_unattributable_but_text_counted_once(self):
        self.events["long"] = self.event_rows(
            [(0, 800, "one two one"), (800, 1440, "two")]
        )
        self.write()
        result = audit.summarize(self.root)["stages"]["long"]
        self.assertEqual(result["unattributable_commits"], 2)
        self.assertEqual(result["whole_stage_score"]["word_edit_distance"], 0)
        self.assertTrue(all(c["attribution"] is None for c in result["commits"]))

    def test_only_exact_committed_revision_is_scored(self):
        rows = self.events["long"]
        replacement = copy.deepcopy(rows[0])
        rows[0]["event"]["text"] = "wrong preview never published"
        replacement["event"].update(kind="replace", revision=2, supersedes_revision=1)
        rows[1]["event"]["revision"] = 2
        rows.insert(1, replacement)
        for sequence, row in enumerate(rows, 1):
            row["event"]["sequence_number"] = sequence
        self.write()
        self.assertEqual(
            audit.summarize(self.root)["stages"]["long"]["committed_text"],
            "one two one two",
        )

    def test_no_final_is_incomplete_even_with_completed_done(self):
        self.events["long"].pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "no FINAL"):
            audit.summarize(self.root)

    def test_repeated_sequence_is_rejected(self):
        self.events["long"][1]["event"]["sequence_number"] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            audit.summarize(self.root)

    def test_identical_committed_range_is_rejected(self):
        row = copy.deepcopy(self.events["long"][1])
        row["event"]["sequence_number"] = 3
        self.events["long"].insert(2, row)
        self.write()
        with self.assertRaisesRegex(ValueError, "repeated identical"):
            audit.summarize(self.root)

    def test_changed_input_and_incomplete_done_are_rejected(self):
        original = copy.deepcopy(self.final)
        for key, value in (
            ("accepted_sha256", "changed"),
            ("committed_samples", 1441),
            ("buffered_samples", 1),
        ):
            with self.subTest(key=key):
                self.final = copy.deepcopy(original)
                self.final["qualification"]["long"]["result"]["server_done"]["metrics"][
                    key
                ] = value
                self.write()
                with self.assertRaises((ValueError, audit.ReplayError)):
                    audit.summarize(self.root)

    def test_preflight_drift_and_source_inventory_drift_are_rejected(self):
        self.preflight = copy.deepcopy(self.preflight)
        self.preflight["input"]["smoke_reference"] = "invented"
        self.write()
        with self.assertRaisesRegex(ValueError, "original preflight"):
            audit.summarize(self.root)
        self.preflight = dict(status="local-preflight-passed", **self.specification)
        self.specification["source"]["files"][0]["sha256"] = "1" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "inventory digest"):
            audit.summarize(self.root)

    def test_changed_log_bytes_fail_receipt_hash(self):
        path = self.root / "live-v01-long.events.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"one"', b'"two"', 1))
        with self.assertRaisesRegex(ValueError, "changed event log"):
            audit.summarize(self.root)

    def test_missing_log_and_unterminated_log_are_incomplete(self):
        path = self.root / "live-v01-long.events.jsonl"
        raw = path.read_bytes().rstrip(b"\n")
        path.unlink()
        with self.assertRaises(OSError):
            audit.summarize(self.root)
        path.write_bytes(raw)
        receipt = self.final["qualification"]["long"]["event_receipt"]
        receipt.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        (self.root / "live-v01.json").write_text(
            json.dumps(self.final), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "incomplete or changed"):
            audit.summarize(self.root)

    def test_thirty_eight_cycles_have_one_whole_reference_and_silence_tail(self):
        stage = self.plan["stages"]["long"]
        stage.update(cycles=38, sample_count=38 * 640 + 160)
        spans = audit._intervals(self.plan, stage)
        self.events["long"] = self.event_rows(
            [
                (start, end, {"main": "one", "noisy": "two"}.get(arm, ""))
                for start, end, _, arm in spans
            ]
        )
        result = self.final["qualification"]["long"]["result"]
        result["sample_count"] = stage["sample_count"]
        result["server_done"]["metrics"].update(
            accepted_samples=stage["sample_count"],
            committed_samples=stage["sample_count"],
        )
        result["client_metrics"]["sent_samples"] = stage["sample_count"]
        self.write()
        report = audit.summarize(self.root)["stages"]["long"]
        self.assertEqual(report["whole_stage_score"]["reference_word_count"], 76)
        self.assertEqual(report["whole_stage_score"]["word_edit_distance"], 0)
        self.assertEqual(report["committed_text"], " ".join(["one two"] * 38))

    def test_receipt_owner_and_event_count_must_match(self):
        result = self.final["qualification"]["long"]["result"]
        result["qualification_receipt"]["record"]["instance_id"] = "other-worker"
        self.write()
        with self.assertRaisesRegex(ValueError, "terminal receipt differs"):
            audit.summarize(self.root)
        result["qualification_receipt"]["record"]["instance_id"] = "worker"
        self.write()
        result["client_metrics"]["received_events"] += 1
        (self.root / "live-v01.json").write_text(
            json.dumps(self.final), encoding="utf-8"
        )
        with self.assertRaises(audit.ReplayError):
            audit.summarize(self.root)

    def test_malformed_arrival_or_event_are_rejected(self):
        original = copy.deepcopy(self.events)
        for change in (
            lambda row: row.update(client_elapsed_ns=True),
            lambda row: row.update(client_elapsed_ns=-1),
            lambda row: row["event"].update(extra_field=True),
        ):
            self.events = copy.deepcopy(original)
            change(self.events["long"][1])
            self.write()
            with self.assertRaises((TypeError, ValueError)):
                audit.summarize(self.root)

    def test_missing_final_file_and_malformed_json_report_nonzero_incomplete(self):
        (self.root / "live-v01.json").unlink()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(audit.main([str(self.root)]), 1)
        self.assertEqual(
            json.loads(output.getvalue())["status"], "incomplete_or_invalid"
        )
        for raw in ('{"status":"a","status":"b"}', '{"value":NaN}', "{"):
            with self.assertRaises(ValueError):
                audit._json(raw)


if __name__ == "__main__":
    unittest.main()
