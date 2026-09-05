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
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"WHISPER_MODAL_ENABLE_WORD_CORPUS": "0"}):
    from infra import modal_word_corpus as corpus


class Echo:
    def remote(self, payload: bytes) -> bytes:
        return payload


class CorpusGuards(unittest.TestCase):
    def paced_observations(self, total_samples=640):
        admissions = []
        for index, start in enumerate(range(0, total_samples, 320)):
            end = min(start + 320, total_samples)
            scheduled = end * 62500
            admissions.append(
                {
                    "sequence_number": index,
                    "start_sample": start,
                    "end_sample": end,
                    "scheduled_ns": scheduled,
                    "offered_ns": scheduled + 1_000_000,
                    "accepted_ns": scheduled + 1_000_001,
                    "buffered_samples": end - start,
                }
            )
        end_ns = total_samples * 62500
        events = [
            {
                "sequence_number": 1,
                "kind": "commit",
                "start_sample": 0,
                "end_sample": 320,
                "committed_through_sample": 320,
                "committed_through_ms": 20,
            },
            {
                "sequence_number": 2,
                "kind": "commit",
                "start_sample": 320,
                "end_sample": total_samples,
                "committed_through_sample": total_samples,
                "committed_through_ms": total_samples // 16,
            },
            {"sequence_number": 3, "kind": "final"},
        ]
        traces = [
            {"analysis_end_sample": 320},
            {"analysis_end_sample": total_samples},
        ]
        pacing = {
            **copy.deepcopy(corpus.PACED_REPLAY),
            "status": "completed",
            "elapsed_ns": end_ns + 15_000_000,
            "input_finished_ns": end_ns + 5_000_000,
            "offered_samples": total_samples,
            "accepted_samples": total_samples,
            "driver_steps": 4,
            "max_source_lag_ns": 1_000_000,
            "max_admission_lag_ns": 1_000_001,
            "admissions": admissions,
            "undelivered_events": [],
            "undelivered_event_ns": None,
            "event_elapsed_ns": [30_000_000, end_ns + 10_000_000, end_ns + 10_000_000],
            "trace_elapsed_ns": [30_000_000, end_ns + 10_000_000],
        }
        pacing["output_lags"] = corpus._paced_output_lags(events, traces, pacing)
        pacing.update(corpus._paced_milestones(events, pacing))
        return events, traces, pacing

    def test_paced_registration_and_profile_are_explicit(self) -> None:
        manifest = corpus.read_registration()
        self.assertEqual(manifest["paced_replay"], corpus.PACED_REPLAY)
        self.assertEqual(
            corpus._stream_profile(manifest, False, paced_replay=True),
            corpus.AUTOMATIC_ENDPOINTS_PROFILE,
        )
        for flags in (
            {"paced_replay": 1},
            {"paced_replay": True, "source_units": True},
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                corpus._stream_profile(manifest, False, **flags)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            for change in ("missing", "pace", "lag"):
                altered = copy.deepcopy(manifest)
                if change == "missing":
                    altered.pop("paced_replay")
                elif change == "pace":
                    altered["paced_replay"]["pacing_id"] = "unpaced"
                else:
                    altered["paced_replay"]["config"]["max_source_lag_ms"] = 1000
                (root / corpus.MANIFEST_PATH).write_text(
                    json.dumps(altered), encoding="utf-8"
                )
                with self.subTest(change=change), self.assertRaises(ValueError):
                    corpus.read_registration(root)

    def test_paced_metrics_reject_early_missing_and_malformed_observations(
        self,
    ) -> None:
        events, traces, pacing = self.paced_observations(657)
        self.assertTrue(all(corpus._paced_checks(events, traces, pacing, 657).values()))
        self.assertEqual(pacing["admissions"][-1]["end_sample"], 657)
        for change in (
            "early",
            "gap",
            "partial",
            "boolean",
            "late",
            "missing_time",
            "false_lag",
            "exaggerated_lag",
            "false_acceptance",
            "no_live_commit",
            "source_overload",
            "corrupt_distribution",
            "future_trace",
            "negative_time",
            "admission_lag",
            "false_admission_lag",
            "undelivered",
            "first_text",
            "first_commit",
            "drain",
        ):
            altered = copy.deepcopy(pacing)
            if change == "early":
                altered["admissions"][0]["offered_ns"] = 1
            elif change == "gap":
                altered["admissions"][1]["start_sample"] += 1
            elif change == "partial":
                altered["admissions"].pop()
            elif change == "boolean":
                altered["admissions"][0]["scheduled_ns"] = True
            elif change == "late":
                altered["max_source_lag_ns"] = 251_000_000
            elif change == "missing_time":
                altered["event_elapsed_ns"].pop()
            elif change == "false_lag":
                altered["max_source_lag_ns"] = 0
            elif change == "exaggerated_lag":
                altered["max_source_lag_ns"] = 1_000_001
            elif change == "false_acceptance":
                altered["accepted_samples"] = 656
            elif change == "no_live_commit":
                altered["input_finished_ns"] = 1
            elif change == "source_overload":
                altered["status"] = "source_overload"
            elif change == "corrupt_distribution":
                altered["output_lags"]["commit"]["min_ns"] = 0
            elif change == "future_trace":
                altered["trace_elapsed_ns"][0] = 1
            elif change == "admission_lag":
                altered["max_admission_lag_ns"] = 251_000_000
            elif change == "false_admission_lag":
                altered["max_admission_lag_ns"] = 0
            elif change == "undelivered":
                altered["undelivered_events"] = [{"kind": "commit"}]
            elif change == "first_text":
                altered["first_nonempty_text_ns"] = 1
            elif change == "first_commit":
                altered["first_commit_ns"] = 1
            elif change == "drain":
                altered["drain_after_input_finished_ns"] += 1
            else:
                altered["event_elapsed_ns"][0] = -1
            with self.subTest(change=change):
                self.assertFalse(
                    all(corpus._paced_checks(events, traces, altered, 657).values())
                )

    def test_paced_flag_receipts_and_worker_result_are_bound(self) -> None:
        with self.assertRaises(ValueError):
            corpus._execute_local_attempt(
                paced_replay=True, confirm_paid_gpu=True, remote_function=Echo()
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            total = corpus.SOURCE_UNIT_BOUNDARIES["end_samples"][-1]
            events, traces, pacing = self.paced_observations(total)
            case = {
                "case_id": corpus.SOURCE_UNIT_BOUNDARIES["case_id"],
                "status": "completed",
                "events": events,
                "decision_traces": traces,
                "pacing": pacing,
                "checks": corpus._paced_checks(events, traces, pacing, total),
                "accepted_samples": total,
                "accepted_chunks": len(pacing["admissions"]),
                "driver_steps": pacing["driver_steps"],
            }
            record = {
                "status": "completed",
                "worker": {"function_call_id": "fc-test"},
                "input_evidence": True,
                "source_units": False,
                "automatic_endpoints": True,
                "paced_replay": True,
                "replay_pacing": corpus.PACED_REPLAY["pacing_id"],
                **corpus.AUTOMATIC_ENDPOINTS_PROFILE,
                "cases": [case],
            }
            for variant in ("valid", "wrong_variant", "false_success", "false_metric"):
                corpus._execute_transport_probe(
                    root=root, replay_id=variant, remote_function=Echo()
                )
                altered = copy.deepcopy(record)
                if variant == "wrong_variant":
                    altered["paced_replay"] = False
                elif variant == "false_success":
                    altered["cases"][0]["pacing"]["status"] = "source_lag"
                    altered["cases"][0]["checks"]["paced_completed"] = False
                elif variant == "false_metric":
                    altered["cases"][0]["accepted_chunks"] += 1
                with (
                    patch.object(corpus, "_inputs", return_value=[]),
                    patch.object(
                        corpus.b, "_decode_worker_record", return_value=altered
                    ),
                    patch.object(Echo, "remote", return_value=b"payload") as remote,
                ):
                    kwargs = dict(
                        root=root,
                        replay_id=variant,
                        paced_replay=True,
                        confirm_paid_gpu=True,
                        remote_function=Echo(),
                    )
                    if variant == "valid":
                        output = corpus._execute_local_attempt(**kwargs)
                        self.assertEqual(json.loads(output.read_text()), record)
                        with self.assertRaises(FileExistsError):
                            corpus._execute_local_attempt(**kwargs)
                    else:
                        with self.assertRaises(ValueError):
                            corpus._execute_local_attempt(**kwargs)
                        self.assertFalse(corpus._paths(root, variant)[0].exists())
                    self.assertEqual(remote.call_count, 1)
                    self.assertEqual(
                        remote.call_args.args[2:], (True, False, True, True)
                    )
                rows = [
                    json.loads(line)
                    for line in corpus._paths(root, variant)[1].read_text().splitlines()
                ]
                self.assertTrue(all(row["paced_replay"] is True for row in rows))

    def test_paced_terminal_guard_rejects_self_consistent_incomplete_publication(
        self,
    ) -> None:
        for change in (
            "missing_final",
            "short_commit",
            "wrong_watermark",
            "commit_gap",
        ):
            events, traces, pacing = self.paced_observations(657)
            if change == "missing_final":
                events.pop()
                pacing["event_elapsed_ns"].pop()
            elif change == "short_commit":
                events[1]["end_sample"] -= 1
                events[1]["committed_through_sample"] -= 1
                events[1]["committed_through_ms"] = events[1]["end_sample"] // 16
            elif change == "wrong_watermark":
                events[1]["committed_through_sample"] -= 1
            else:
                events[1]["start_sample"] += 1
            pacing["output_lags"] = corpus._paced_output_lags(events, traces, pacing)
            pacing.update(corpus._paced_milestones(events, pacing))
            checks = corpus._paced_checks(events, traces, pacing, 657)
            with self.subTest(change=change):
                self.assertFalse(checks["paced_terminal_coverage"])
                self.assertTrue(
                    all(
                        value
                        for key, value in checks.items()
                        if key != "paced_terminal_coverage"
                    )
                )

    def test_paced_driver_runs_existing_runtime_with_partial_last_chunk(self) -> None:
        with patch.object(
            sys,
            "path",
            [str(corpus.ROOT / "src"), str(corpus.ROOT / "tests"), *sys.path],
        ):
            fixture = corpus.importlib.import_module("test_continuous_evidence")
            endpoints = corpus.importlib.import_module(
                "whisper_runtime.adapters.audio_endpoints"
            )
        adapter = fixture.EvidenceNativeAdapter(fixture.you_result)
        stream = fixture.ContinuousTranscriptStream(
            adapter,
            stream_id="paced-harness",
            mel_builder=lambda content: content,
            config=fixture.ContinuousStreamConfig(
                preview_interval_ms=100,
                max_window_ms=1000,
                max_buffer_ms=1200,
                holdback_ms=100,
                input_evidence=True,
                source_units=True,
                endpointing=endpoints.QuietEndpointConfig(
                    quiet_ms=60, min_unit_ms=100, max_quiet_unit_ms=500
                ),
            ),
        )
        pcm = b"\xe8\x03" * 1600 + bytes(3200) + b"\xe8\x03" * 1617
        with patch.object(
            stream, "seal_unit", side_effect=AssertionError("oracle boundary called")
        ):
            events, traces, pacing, error = corpus._drive_paced(stream, pcm)
        self.assertIsNone(error)
        self.assertEqual(pacing["status"], "completed")
        self.assertTrue(
            all(corpus._paced_checks(events, traces, pacing, len(pcm) // 2).values())
        )
        self.assertEqual(pacing["accepted_samples"], 4817)
        self.assertEqual(len(pacing["admissions"]), 16)
        self.assertEqual(
            pacing["admissions"][-1]["end_sample"]
            - pacing["admissions"][-1]["start_sample"],
            17,
        )
        self.assertEqual(stream.metrics.buffered_samples, 0)
        self.assertTrue(all(corpus._unit_publication_checks(events, traces).values()))
        self.assertEqual(adapter.budget.available, adapter.capacity)
        stream.close()

    def test_modal_mount_destinations_use_posix_paths(self) -> None:
        destinations = []

        class Image:
            def add_local_file(self, source, destination, **kwargs):
                self_test.assertTrue(destination.startswith("/"), destination)
                self_test.assertNotIn("\\", destination)
                destinations.append(destination)
                return self

            def __getattr__(self, name):
                return lambda *args, **kwargs: self

        self_test = self
        fake_app = SimpleNamespace(
            function=lambda **kwargs: lambda function: function,
            local_entrypoint=lambda **kwargs: lambda function: function,
        )
        fake = SimpleNamespace(
            __version__="1.5.5",
            Image=SimpleNamespace(debian_slim=lambda **kwargs: Image()),
            Volume=SimpleNamespace(from_name=lambda *args, **kwargs: Image()),
            App=lambda name: fake_app,
        )
        original = corpus.importlib.import_module
        with (
            patch.object(corpus, "_inputs", return_value=[]),
            patch.object(
                corpus.importlib,
                "import_module",
                side_effect=lambda name: fake if name == "modal" else original(name),
            ),
        ):
            _, execute, _, _ = corpus._define_modal_resources()
        self.assertEqual(
            sum(path.startswith("/opt/speech-corpus/") for path in destinations), 3
        )
        with (
            patch.object(corpus, "_run_worker", return_value={}) as worker,
            patch.object(corpus.b, "_encode_worker_record", return_value=b"record"),
        ):
            self.assertEqual(execute({}, "registration"), b"record")
            self.assertIs(worker.call_args.kwargs["input_evidence"], False)
            self.assertEqual(execute({}, "registration", True), b"record")
            self.assertIs(worker.call_args.kwargs["input_evidence"], True)
            self.assertEqual(execute({}, "registration", True, True), b"record")
            self.assertIs(worker.call_args.kwargs["source_units"], True)
            self.assertEqual(execute({}, "registration", True, False, True), b"record")
            self.assertIs(worker.call_args.kwargs["automatic_endpoints"], True)
            self.assertEqual(
                execute({}, "registration", True, False, True, True), b"record"
            )
            self.assertIs(worker.call_args.kwargs["paced_replay"], True)

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
        self.assertEqual(
            manifest["input_evidence_profile"], corpus.INPUT_EVIDENCE_PROFILE
        )

    def test_input_evidence_registration_is_required_and_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.root(root)
            for change in ("absent", "flag", "profile", "holdback"):
                manifest = copy.deepcopy(original)
                if change == "absent":
                    manifest.pop("input_evidence_profile")
                elif change == "profile":
                    manifest["input_evidence_profile"]["profile_id"] = "invented"
                else:
                    key, value = (
                        ("input_evidence", False)
                        if change == "flag"
                        else ("holdback_ms", 1)
                    )
                    manifest["input_evidence_profile"]["stream_config"][key] = value
                (root / corpus.MANIFEST_PATH).write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.subTest(change=change), self.assertRaises(ValueError):
                    corpus.read_registration(root)

    def test_source_unit_registration_binds_exact_oracle_endpoints(self) -> None:
        manifest = corpus.read_registration()
        units = corpus._source_unit_plan(manifest)
        self.assertEqual(len(units), 11)
        self.assertEqual(
            [unit["end_sample"] for unit in units],
            corpus.SOURCE_UNIT_BOUNDARIES["end_samples"],
        )
        self.assertEqual(
            [unit["start_sample"] for unit in units],
            [0, *corpus.SOURCE_UNIT_BOUNDARIES["end_samples"][:-1]],
        )
        self.assertEqual(sum(unit["kind"] == "fixture" for unit in units), 6)
        self.assertEqual(sum(unit["kind"] == "digital_silence" for unit in units), 5)
        self.assertEqual(
            [unit["occurrence"] for unit in units if unit["kind"] == "fixture"],
            [1, 1, 1, 2, 2, 2],
        )
        self.assertIs(manifest["source_unit_boundaries"]["acoustic_detection"], False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            for change in ("absent", "endpoint", "alignment", "detector"):
                changed = copy.deepcopy(manifest)
                if change == "absent":
                    changed.pop("source_units_profile")
                elif change == "endpoint":
                    changed["source_unit_boundaries"]["end_samples"][0] -= 1
                elif change == "alignment":
                    changed["source_units_profile"]["stream_config"][
                        "word_alignment"
                    ] = True
                else:
                    changed["source_unit_boundaries"]["acoustic_detection"] = True
                (root / corpus.MANIFEST_PATH).write_text(
                    json.dumps(changed), encoding="utf-8"
                )
                with self.subTest(change=change), self.assertRaises(ValueError):
                    corpus.read_registration(root)

    def test_source_units_driver_keeps_one_global_owner_and_exact_chunk_counts(
        self,
    ) -> None:
        with patch.object(
            sys,
            "path",
            [str(corpus.ROOT / "src"), str(corpus.ROOT / "tests"), *sys.path],
        ):
            fixture = corpus.importlib.import_module("test_continuous_evidence")
        adapter = fixture.EvidenceNativeAdapter(fixture.you_result)
        stream = fixture.ContinuousTranscriptStream(
            adapter,
            stream_id="oracle-driver",
            mel_builder=lambda content: content,
            config=fixture.ContinuousStreamConfig(
                preview_interval_ms=100,
                max_window_ms=500,
                max_buffer_ms=600,
                holdback_ms=100,
                input_evidence=True,
                source_units=True,
            ),
        )
        pieces = [b"\x01\x00" * 4000, bytes(8000), b"\x01\x00" * 4000]
        units = []
        for index, content in enumerate(pieces):
            units.append(
                {
                    "index": index,
                    "start_sample": index * 4000,
                    "end_sample": (index + 1) * 4000,
                    "pcm_sha256": hashlib.sha256(content).hexdigest(),
                    "kind": "digital_silence" if index == 1 else "fixture",
                    "fixture_id": "repeat",
                    "reference_text": "" if index == 1 else "you you",
                    "occurrence": 2 if index == 2 else 1,
                }
            )
        events, traces, steps, chunks, error, accepted = corpus._drive_source_units(
            stream, b"".join(pieces), units, chunk_bytes=3200
        )
        self.assertIsNone(error)
        self.assertEqual((chunks, accepted), (9, 12000))
        self.assertGreater(steps, 0)
        self.assertTrue(stream.done)
        commits = [event for event in events if event["kind"] == "commit"]
        self.assertEqual(
            [(event["start_sample"], event["end_sample"]) for event in commits],
            [(0, 4000), (4000, 8000), (8000, 12000)],
        )
        self.assertEqual(sum(event["kind"] == "final" for event in events), 1)
        self.assertEqual(
            [event["sequence_number"] for event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertTrue(
            all(
                corpus._trace_checks(
                    traces,
                    input_evidence=True,
                    word_alignment=False,
                    source_units=True,
                    registered_units=units,
                ).values()
            )
        )
        self.assertEqual(sum(bool(trace["eof"]) for trace in traces), 1)
        for key, value in (
            ("start_sample", 1),
            ("end_sample", 999),
            ("origin", "detected"),
        ):
            changed = copy.deepcopy(traces[-1])
            changed["source_unit"][key] = value
            with self.subTest(key=key):
                self.assertFalse(
                    corpus._trace_checks(
                        [changed],
                        input_evidence=True,
                        word_alignment=False,
                        source_units=True,
                        registered_units=units,
                    )["source_unit_trace_contract"]
                )
        self.assertFalse(
            corpus._trace_checks(
                [traces[-1]],
                input_evidence=True,
                word_alignment=False,
                source_units=True,
                registered_units=units[:-1],
            )["source_unit_trace_contract"]
        )
        progress = corpus._source_progress(events, traces, accepted)
        self.assertEqual(progress["eof_commit_coverage_ms"], 250)
        self.assertEqual(progress["pre_eof_commit_count"], 2)
        results = corpus._source_unit_results(
            events, units, {"repeat": {"text": "you you"}}
        )
        self.assertEqual(
            [result["text"] for result in results], ["you you", "", "you you"]
        )
        self.assertTrue(all(result["completed"] for result in results))
        self.assertEqual(adapter.budget.available, adapter.capacity)
        self.assertEqual(adapter.worker.queue_depth, 0)
        with self.assertRaises(ValueError):
            corpus._drive_source_units(stream, b"wrong", units, chunk_bytes=3200)
        self.assertTrue(all(corpus._unit_publication_checks(events, traces).values()))
        for commit in commits:
            changed = copy.deepcopy(events)
            revision = next(
                event
                for event in changed
                if event["kind"] in {"provisional", "replace"}
                and (event["segment_id"], event["revision"])
                == (commit["segment_id"], commit["revision"])
            )
            revision["text"] = "changed published text"
            with self.subTest(span=(commit["start_sample"], commit["end_sample"])):
                self.assertFalse(
                    corpus._unit_publication_checks(changed, traces)[
                        "unit_publication_matches_trace"
                    ]
                )
        self.assertFalse(
            corpus._unit_publication_checks(events, [])[
                "unit_publication_matches_trace"
            ]
        )
        self.assertTrue(
            corpus._unit_publication_checks([], traces)[
                "unit_publication_matches_trace"
            ]
        )
        self.assertFalse(
            corpus._unit_publication_checks([events[-1]], traces)[
                "unit_publication_matches_trace"
            ]
        )

    def test_source_units_variant_binds_arguments_receipts_and_config(self) -> None:
        for value, replay_id in ((1, "units"), (True, "")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                corpus._execute_local_attempt(
                    source_units=value,
                    replay_id=replay_id,
                    confirm_paid_gpu=True,
                    remote_function=Echo(),
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            corpus._execute_transport_probe(
                root=root, replay_id="units", remote_function=Echo()
            )
            record = {
                "status": "diagnostic",
                "worker": {"function_call_id": "fc-test"},
                "source_units": True,
                "input_evidence": True,
                **corpus.SOURCE_UNITS_PROFILE,
            }
            with (
                patch.object(corpus, "_inputs", return_value=[]),
                patch.object(corpus.b, "_decode_worker_record", return_value=record),
                patch.object(Echo, "remote", return_value=b"payload") as remote,
            ):
                output = corpus._execute_local_attempt(
                    root=root,
                    replay_id="units",
                    source_units=True,
                    confirm_paid_gpu=True,
                    remote_function=Echo(),
                )
            self.assertEqual(remote.call_args.args[2:], (True, True))
            self.assertEqual(json.loads(output.read_text()), record)
            rows = [
                json.loads(line)
                for line in corpus._paths(root, "units")[1].read_text().splitlines()
            ]
            self.assertTrue(
                all(
                    row["source_units"] is True and row["input_evidence"] is True
                    for row in rows
                )
            )
            self.assertEqual(
                rows[0]["profile_id"], corpus.SOURCE_UNITS_PROFILE["profile_id"]
            )

    def test_automatic_endpoint_registration_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.root(root)
            self.assertEqual(
                original["automatic_endpoints_profile"],
                corpus.AUTOMATIC_ENDPOINTS_PROFILE,
            )
            for change in (
                "absent",
                "quiet_peak",
                "quiet_ms",
                "profile",
                "source_units",
            ):
                manifest = copy.deepcopy(original)
                profile = manifest["automatic_endpoints_profile"]
                if change == "absent":
                    manifest.pop("automatic_endpoints_profile")
                elif change == "profile":
                    profile["profile_id"] = "unregistered"
                elif change == "source_units":
                    profile["stream_config"]["source_units"] = False
                else:
                    profile["stream_config"]["endpointing"][change] += 1
                (root / corpus.MANIFEST_PATH).write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.subTest(change=change), self.assertRaises(ValueError):
                    corpus.read_registration(root)

    def test_automatic_endpoints_bind_flags_and_exclusive_receipts(self) -> None:
        for flags in (
            {"automatic_endpoints": 1, "replay_id": "automatic"},
            {"automatic_endpoints": True},
            {
                "automatic_endpoints": True,
                "source_units": True,
                "replay_id": "automatic",
            },
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                corpus._execute_local_attempt(
                    **flags, confirm_paid_gpu=True, remote_function=Echo()
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            corpus._execute_transport_probe(
                root=root, replay_id="automatic", remote_function=Echo()
            )
            record = {
                "status": "diagnostic",
                "worker": {"function_call_id": "fc-test"},
                "source_units": False,
                "input_evidence": True,
                "automatic_endpoints": True,
                **corpus.AUTOMATIC_ENDPOINTS_PROFILE,
            }
            with (
                patch.object(corpus, "_inputs", return_value=[]),
                patch.object(corpus.b, "_decode_worker_record", return_value=record),
                patch.object(Echo, "remote", return_value=b"payload") as remote,
            ):
                output = corpus._execute_local_attempt(
                    root=root,
                    replay_id="automatic",
                    automatic_endpoints=True,
                    confirm_paid_gpu=True,
                    remote_function=Echo(),
                )
                with self.assertRaises(FileExistsError):
                    corpus._execute_local_attempt(
                        root=root,
                        replay_id="automatic",
                        automatic_endpoints=True,
                        confirm_paid_gpu=True,
                        remote_function=Echo(),
                    )
            self.assertEqual(remote.call_args.args[2:], (True, False, True))
            self.assertEqual(remote.call_count, 1)
            self.assertEqual(json.loads(output.read_text()), record)
            rows = [
                json.loads(line)
                for line in corpus._paths(root, "automatic")[1].read_text().splitlines()
            ]
            self.assertTrue(all(row["automatic_endpoints"] is True for row in rows))
            self.assertTrue(all(row["source_units"] is False for row in rows))
            self.assertEqual(
                rows[0]["stream_config"],
                corpus.AUTOMATIC_ENDPOINTS_PROFILE["stream_config"],
            )

    def test_automatic_endpoints_reject_wrong_worker_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            corpus._execute_transport_probe(
                root=root, replay_id="automatic", remote_function=Echo()
            )
            record = {
                "input_evidence": True,
                "source_units": False,
                "automatic_endpoints": False,
                **corpus.AUTOMATIC_ENDPOINTS_PROFILE,
            }
            with (
                patch.object(corpus, "_inputs", return_value=[]),
                patch.object(corpus.b, "_decode_worker_record", return_value=record),
                patch.object(Echo, "remote", return_value=b"payload"),
                self.assertRaises(ValueError),
            ):
                corpus._execute_local_attempt(
                    root=root,
                    replay_id="automatic",
                    automatic_endpoints=True,
                    confirm_paid_gpu=True,
                    remote_function=Echo(),
                )
            self.assertFalse(corpus._paths(root, "automatic")[0].exists())
            rows = [
                json.loads(line)
                for line in corpus._paths(root, "automatic")[1].read_text().splitlines()
            ]
            self.assertEqual(rows[-1]["event"], "attempt-failed")
            self.assertIs(rows[-1]["automatic_endpoints"], True)

    def test_automatic_driver_uses_causal_audio_not_oracle_boundaries(self) -> None:
        with patch.object(
            sys,
            "path",
            [str(corpus.ROOT / "src"), str(corpus.ROOT / "tests"), *sys.path],
        ):
            fixture = corpus.importlib.import_module("test_continuous_evidence")
            endpoints = corpus.importlib.import_module(
                "whisper_runtime.adapters.audio_endpoints"
            )
        adapter = fixture.EvidenceNativeAdapter(fixture.you_result)
        config = dict(corpus.AUTOMATIC_ENDPOINTS_PROFILE["stream_config"])
        config["endpointing"] = endpoints.QuietEndpointConfig(**config["endpointing"])
        stream = fixture.ContinuousTranscriptStream(
            adapter,
            stream_id="automatic-driver",
            mel_builder=lambda content: content,
            config=fixture.ContinuousStreamConfig(**config),
        )
        speech, quiet = b"\xe8\x03" * 16000, b"\x14\x00" * 12800
        pcm = speech + quiet + speech + quiet + speech + b"\xe8\x03" * 17
        with patch.object(
            stream, "seal_unit", side_effect=AssertionError("oracle boundary called")
        ):
            events, traces, steps, chunks, error = corpus.b._drive_stream(
                stream, pcm, chunk_bytes=1994
            )
        self.assertIsNone(error)
        self.assertTrue(stream.done)
        self.assertGreater(steps, 0)
        self.assertEqual(chunks, (len(pcm) + 1993) // 1994)
        self.assertEqual(stream.accepted_samples, len(pcm) // 2)
        self.assertEqual(stream.metrics.buffered_samples, 0)
        commits = [event for event in events if event["kind"] == "commit"]
        self.assertEqual(
            [(event["start_sample"], event["end_sample"]) for event in commits],
            [(0, 25600), (25600, 54400), (54400, 73617)],
        )
        self.assertEqual(sum(event["kind"] == "final" for event in events), 1)
        self.assertEqual(
            [event["sequence_number"] for event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertTrue(
            all(
                corpus._trace_checks(
                    traces,
                    input_evidence=True,
                    word_alignment=False,
                    source_units=True,
                    automatic_endpoints=True,
                ).values()
            )
        )
        self.assertTrue(all(corpus._automatic_endpoint_checks(traces, pcm).values()))
        self.assertTrue(all(corpus._unit_publication_checks(events, traces).values()))
        changed_events = copy.deepcopy(events)
        committed = commits[-1]
        published_revision = next(
            event
            for event in changed_events
            if event["kind"] in {"provisional", "replace"}
            and (event["segment_id"], event["revision"])
            == (committed["segment_id"], committed["revision"])
        )
        published_revision["text"] += " fabricated"
        self.assertFalse(
            corpus._unit_publication_checks(changed_events, traces)[
                "unit_publication_matches_trace"
            ]
        )
        self.assertTrue(
            all(trace.get("silence_publication") is None for trace in traces)
        )
        closed = [trace for trace in traces if trace.get("source_unit") is not None]
        self.assertEqual(
            [trace["source_unit"]["origin"] for trace in closed],
            ["quiet_run", "quiet_run", "end_of_input"],
        )
        for change in (
            "hash",
            "quiet_start",
            "peak",
            "origin",
            "silence_authority",
            "eof",
            "horizon",
        ):
            changed = copy.deepcopy(closed[-1] if change == "eof" else closed[0])
            if change == "hash":
                changed["audio_evidence"]["observation"]["pcm_sha256"] = "0" * 64
            elif change in {"quiet_start", "peak"}:
                key = "quiet_start_sample" if change == "quiet_start" else "peak"
                changed["source_unit"]["endpoint"][key] += 1
            elif change == "origin":
                changed["source_unit"]["origin"] = "caller"
            elif change == "silence_authority":
                changed["silence_publication"] = {"text": ""}
            elif change == "horizon":
                changed["accepted_through_sample"] = changed["analysis_end_sample"] - 1
            else:
                changed["eof"] = False
            with self.subTest(change=change):
                self.assertFalse(
                    all(corpus._automatic_endpoint_checks([changed], pcm).values())
                )
        self.assertFalse(
            corpus._automatic_endpoint_checks(closed[1:], pcm)[
                "automatic_endpoint_trace_contract"
            ]
        )
        self.assertEqual(adapter.budget.available, adapter.capacity)
        self.assertEqual(adapter.worker.queue_depth, 0)

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
            self.assertIn(corpus.PRODUCER_PATH, paths)
            helper = root / corpus.HELPER_PATHS[0]
            helper.write_text("# changed\n", encoding="utf-8")
            self.assertNotEqual(
                initial["digest"], corpus.source_snapshot(root)["digest"]
            )
            changed_helper = corpus.source_snapshot(root)
            (root / corpus.PRODUCER_PATH).write_text(
                "# producer changed\n", encoding="utf-8"
            )
            self.assertNotEqual(
                changed_helper["digest"], corpus.source_snapshot(root)["digest"]
            )

    def test_replay_paths_keep_legacy_default_and_reject_unsafe_names(self) -> None:
        root = Path("example")
        legacy = corpus._paths(root)
        self.assertEqual(legacy, corpus._paths(root, ""))
        self.assertEqual(
            legacy[0], root / "artifacts/modal/word-corpus-v1-attempt-1.json"
        )
        named = corpus._paths(root, "repair-20260905_A")
        self.assertEqual([path.name for path in legacy], [path.name for path in named])
        self.assertTrue(
            all(
                path.parent == root / "artifacts/modal/repair-20260905_A"
                for path in named
            )
        )
        for replay_id in (
            ".",
            "..",
            "../outside",
            "a/b",
            "a\\b",
            "/absolute",
            "C:\\absolute",
            " space",
            "name.txt",
            "caf\u00e9",
            "a" * 65,
            "CON",
            "nul",
            "LPT1",
            None,
            True,
        ):
            with self.subTest(replay_id=replay_id), self.assertRaises(ValueError):
                corpus._paths(root, replay_id)

    def test_named_replay_requires_its_own_bound_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            legacy_probe = corpus._execute_transport_probe(
                root=root, remote_function=Echo()
            )
            named_probe = corpus._paths(root, "repair")[3]
            remote = SimpleNamespace(remote=lambda *args: self.fail("GPU must not run"))
            with patch.object(corpus, "_inputs", return_value=[]):
                with self.assertRaises(FileNotFoundError):
                    corpus._execute_local_attempt(
                        root=root,
                        replay_id="repair",
                        confirm_paid_gpu=True,
                        remote_function=remote,
                    )
                named_probe.parent.mkdir(parents=True, exist_ok=True)
                for foreign_id in ("", "other-replay"):
                    rows = [
                        json.loads(line)
                        for line in legacy_probe.read_text().splitlines()
                    ]
                    if foreign_id:
                        for row in rows:
                            row["replay_id"] = foreign_id
                    named_probe.write_text(
                        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
                    )
                    with (
                        self.subTest(foreign_id=foreign_id),
                        self.assertRaises(ValueError),
                    ):
                        corpus._execute_local_attempt(
                            root=root,
                            replay_id="repair",
                            confirm_paid_gpu=True,
                            remote_function=remote,
                        )
            self.assertFalse(corpus._paths(root, "repair")[1].exists())

    def test_named_replay_is_exclusive_and_preserves_legacy_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            legacy_paths = corpus._paths(root)
            legacy_paths[0].parent.mkdir(parents=True, exist_ok=True)
            for path in legacy_paths:
                path.write_bytes(b"untouched legacy evidence")
            replay_id = "repair"
            probe = corpus._execute_transport_probe(
                root=root, replay_id=replay_id, remote_function=Echo()
            )
            with self.assertRaises(FileExistsError):
                corpus._execute_transport_probe(
                    root=root, replay_id=replay_id, remote_function=Echo()
                )
            calls = []

            def remote(snapshot, registration_sha256):
                calls.append((snapshot, registration_sha256))
                return b"compressed test payload"

            record = {
                "status": "diagnostic",
                "worker": {"function_call_id": "fc-test"},
                "input_evidence": False,
                "source_units": False,
                **corpus._stream_profile(corpus.read_registration(root), False),
            }
            with (
                patch.object(corpus, "_inputs", return_value=[]),
                patch.object(corpus.b, "_decode_worker_record", return_value=record),
            ):
                output = corpus._execute_local_attempt(
                    root=root,
                    replay_id=replay_id,
                    confirm_paid_gpu=True,
                    remote_function=SimpleNamespace(remote=remote),
                )
                with self.assertRaises(FileExistsError):
                    corpus._execute_local_attempt(
                        root=root,
                        replay_id=replay_id,
                        confirm_paid_gpu=True,
                        remote_function=SimpleNamespace(remote=remote),
                    )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], corpus.source_snapshot(root))
            self.assertEqual(output, corpus._paths(root, replay_id)[0])
            self.assertEqual(json.loads(output.read_text()), record)
            for path in (probe, corpus._paths(root, replay_id)[1]):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertTrue(all(row["replay_id"] == replay_id for row in rows))
            for path in legacy_paths:
                self.assertEqual(path.read_bytes(), b"untouched legacy evidence")

    def test_input_evidence_variant_binds_remote_arguments_receipts_and_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            corpus._execute_transport_probe(
                root=root, replay_id="evidence", remote_function=Echo()
            )
            record = {
                "status": "diagnostic",
                "worker": {"function_call_id": "fc-test"},
                "input_evidence": True,
                "source_units": False,
                **corpus.INPUT_EVIDENCE_PROFILE,
            }
            with (
                patch.object(corpus, "_inputs", return_value=[]),
                patch.object(corpus.b, "_decode_worker_record", return_value=record),
                patch.object(Echo, "remote", return_value=b"test payload") as remote,
            ):
                output = corpus._execute_local_attempt(
                    root=root,
                    replay_id="evidence",
                    input_evidence=True,
                    confirm_paid_gpu=True,
                    remote_function=Echo(),
                )
            self.assertEqual(len(remote.call_args.args), 3)
            self.assertIs(remote.call_args.args[2], True)
            self.assertEqual(json.loads(output.read_text()), record)
            rows = [
                json.loads(line)
                for line in corpus._paths(root, "evidence")[1].read_text().splitlines()
            ]
            self.assertTrue(all(row["input_evidence"] is True for row in rows))
            self.assertEqual(
                rows[0]["stream_config"], corpus.INPUT_EVIDENCE_PROFILE["stream_config"]
            )
            self.assertEqual(
                rows[0]["profile_id"], corpus.INPUT_EVIDENCE_PROFILE["profile_id"]
            )

    def test_input_evidence_requires_boolean_named_variant_and_matching_result(
        self,
    ) -> None:
        for flag, replay_id in ((1, "evidence"), ("true", "evidence"), (True, "")):
            with (
                self.subTest(flag=flag, replay_id=replay_id),
                self.assertRaises(ValueError),
            ):
                corpus._execute_local_attempt(
                    input_evidence=flag,
                    replay_id=replay_id,
                    confirm_paid_gpu=True,
                    remote_function=Echo(),
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            corpus._execute_transport_probe(
                root=root, replay_id="evidence", remote_function=Echo()
            )
            with (
                patch.object(corpus, "_inputs", return_value=[]),
                patch.object(
                    corpus.b,
                    "_decode_worker_record",
                    return_value={"input_evidence": False},
                ),
                patch.object(Echo, "remote", return_value=b"test payload"),
                self.assertRaises(ValueError),
            ):
                corpus._execute_local_attempt(
                    root=root,
                    replay_id="evidence",
                    input_evidence=True,
                    confirm_paid_gpu=True,
                    remote_function=Echo(),
                )
            self.assertFalse(corpus._paths(root, "evidence")[0].exists())

    def test_modal_main_forwards_replay_id_for_both_modes(self) -> None:
        with (
            patch.object(corpus, "run_word_corpus", Echo()),
            patch.object(corpus, "run_transport_probe", Echo()),
            patch.object(corpus, "_execute_transport_probe") as probe,
            patch.object(corpus, "_execute_local_attempt") as attempt,
            patch("builtins.print"),
        ):
            corpus._modal_main(replay_id="repair", transport_preflight_only=True)
            self.assertEqual(probe.call_args.kwargs["replay_id"], "repair")
            attempt.assert_not_called()
            corpus._modal_main(
                replay_id="repair", confirm_paid_gpu=True, input_evidence=True
            )
            self.assertEqual(attempt.call_args.kwargs["replay_id"], "repair")
            self.assertTrue(attempt.call_args.kwargs["confirm_paid_gpu"])
            self.assertTrue(attempt.call_args.kwargs["input_evidence"])
            corpus._modal_main(
                replay_id="units", confirm_paid_gpu=True, source_units=True
            )
            self.assertTrue(attempt.call_args.kwargs["source_units"])
            corpus._modal_main(
                replay_id="automatic", confirm_paid_gpu=True, automatic_endpoints=True
            )
            self.assertTrue(attempt.call_args.kwargs["automatic_endpoints"])
            corpus._modal_main(
                replay_id="paced", confirm_paid_gpu=True, paced_replay=True
            )
            self.assertTrue(attempt.call_args.kwargs["paced_replay"])

    def test_rejected_evidence_trace_contract_and_silence_eof_coverage(self) -> None:
        native = {
            "window_id": "w",
            "text": "invented",
            "metadata": {"language": "en"},
            "analysis_span": {"start_ms": 0, "end_ms": 1000},
        }
        observation = {
            "sample_count": 16000,
            "pcm_sha256": "0" * 64,
            "digital_silence": True,
        }
        silence = {
            "window_id": "w",
            "text": "",
            "native": native,
            "observation": observation,
            "analysis_span": native["analysis_span"],
            "start_ms": 0,
            "end_ms": 1000,
        }
        trace = {
            "result": native,
            "audio_evidence": {"state": "non_speech", "observation": observation},
            "analysis_start_sample": 0,
            "analysis_end_sample": 16000,
            "committed_before_sample": 0,
            "action": "commit",
            "eof": True,
            "silence_publication": silence,
        }
        with patch.object(
            corpus.b, "_trace_profile_checks", wraps=corpus.b._trace_profile_checks
        ) as normal:
            self.assertTrue(
                all(corpus._trace_checks([trace], input_evidence=True).values())
            )
            normal.assert_called_once_with([], word_alignment=True)
        self.assertFalse(
            corpus._trace_checks([trace], input_evidence=False)[
                "input_evidence_trace_contract"
            ]
        )
        events = [{"kind": "commit", "start_sample": 0, "end_sample": 16000}]
        progress = corpus._source_progress(events, [trace], 16000)
        self.assertEqual(progress["pre_eof_commit_count"], 0)
        self.assertEqual(progress["eof_commit_coverage_ms"], 1000)
        for key, value in (
            ("text", "invented"),
            ("end_ms", 2000),
            ("native", {}),
            ("observation", {}),
        ):
            changed = copy.deepcopy(trace)
            changed["silence_publication"][key] = value
            with self.subTest(key=key):
                self.assertFalse(
                    corpus._trace_checks([changed], input_evidence=True)[
                        "input_evidence_trace_contract"
                    ]
                )
        uncertain = copy.deepcopy(trace)
        uncertain["audio_evidence"]["state"] = "uncertain"
        uncertain["audio_evidence"]["observation"]["digital_silence"] = False
        uncertain["silence_publication"] = None
        uncertain["action"] = "wait_for_input"
        self.assertTrue(
            all(corpus._trace_checks([uncertain], input_evidence=True).values())
        )
        for key, value in (
            ("action", "commit"),
            ("word_publication", {}),
            ("word_alignment", {}),
            ("publication_span", {}),
        ):
            changed = {**uncertain, key: value}
            with self.subTest(key=key):
                self.assertFalse(
                    corpus._trace_checks([changed], input_evidence=True)[
                        "input_evidence_trace_contract"
                    ]
                )

    def test_preflight_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.root(root)
            receipt = corpus._execute_transport_probe(root=root, remote_function=Echo())
            rows = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(rows[-1]["event"], "transport-passed")
            self.assertTrue(all("replay_id" not in row for row in rows))
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
