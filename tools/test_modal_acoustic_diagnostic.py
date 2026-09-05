"""Local guards for grouped acoustic evidence. No account or GPU is required."""

import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infra import modal_acoustic_diagnostic as diagnostic


class AcousticDiagnosticTests(unittest.TestCase):
    def fixture(self):
        pcm = b"\x01\0" * 32
        native = {"analysis_span": {"start_ms": 0, "end_ms": 2}, "text": "alpha beta"}
        alignment = {"native": native, "words": [{"text": "alpha"}, {"text": " beta"}]}
        publication = {
            "alignment": alignment,
            "word_start": 1,
            "word_end": 2,
            "text": "beta",
            "start_ms": 1,
            "end_ms": 2,
        }
        trace = {
            "analysis_start_sample": 0,
            "analysis_end_sample": 32,
            "committed_before_sample": 16,
            "audio_evidence": {
                "observation": {
                    "sample_count": 32,
                    "pcm_sha256": diagnostic._sha(pcm),
                    "digital_silence": False,
                }
            },
            "result": native,
            "word_alignment": alignment,
            "word_publication": publication,
            "source_unit": None,
            "action": "commit",
        }
        events = [
            {"kind": "provisional", "segment_id": "one", "revision": 1, "text": "beta"},
            {
                "kind": "commit",
                "segment_id": "one",
                "revision": 1,
                "start_sample": 16,
                "end_sample": 32,
            },
        ]
        return events, [trace], pcm

    def test_alignment_suffix_not_full_native_text_is_the_publication(self):
        events, traces, pcm = self.fixture()
        self.assertTrue(
            all(diagnostic.publication_checks(events, traces, pcm).values())
        )
        events[0]["text"] = "alpha beta"
        self.assertFalse(
            diagnostic.publication_checks(events, traces, pcm)[
                "selected_text_matches_commit"
            ]
        )

    def test_trace_checks_reject_modified_pcm_alignment_words_and_slice(self):
        for field in ("pcm", "native", "slice", "publication", "revision"):
            events, traces, pcm = self.fixture()
            if field == "pcm":
                pcm = b"\x02\0" * 32
            elif field == "native":
                traces[0]["word_alignment"] = copy.deepcopy(traces[0]["word_alignment"])
                traces[0]["word_alignment"]["native"]["text"] = "other"
            elif field == "slice":
                traces[0]["word_publication"]["word_start"] = 0
            elif field == "publication":
                traces[0]["word_publication"]["text"] = "invented"
            else:
                events[1]["revision"] = 2
            self.assertFalse(
                all(diagnostic.publication_checks(events, traces, pcm).values()), field
            )

    def test_quality_gate_is_relative_to_each_offline_control(self):
        offline = {"against_human_reference": {"word_edit_distance": 6}}
        recognition = {
            "against_human_reference": {
                "word_edit_distance": 6,
                "reference_word_count": 88,
            },
            "against_offline_control": {"word_edit_distance": 0},
        }
        self.assertTrue(diagnostic.quality_gate("hybrid", recognition, offline))
        self.assertTrue(diagnostic.quality_gate("context", recognition, offline))
        self.assertTrue(diagnostic.quality_gate("quiet", recognition, offline))
        recognition["against_human_reference"]["word_edit_distance"] = 7
        self.assertFalse(diagnostic.quality_gate("hybrid", recognition, offline))
        self.assertFalse(diagnostic.quality_gate("context", recognition, offline))
        self.assertFalse(diagnostic.quality_gate("quiet", recognition, offline))
        recognition["against_human_reference"]["word_edit_distance"] = 4
        self.assertTrue(diagnostic.quality_gate("hybrid", recognition, offline))
        self.assertTrue(diagnostic.quality_gate("context", recognition, offline))
        self.assertFalse(diagnostic.quality_gate("quiet", recognition, offline))
        with self.assertRaises(ValueError):
            diagnostic.quality_gate("unknown", recognition, offline)

    def test_registration_freezes_cases_profiles_acceptance_and_resources(self):
        original = diagnostic.registration()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / diagnostic.MANIFEST
            target.parent.mkdir(parents=True)
            for field in (
                "case_ids",
                "cells",
                "candidate_cells",
                "control_cells",
                "control_expectations",
                "qualification_scope",
                "execution",
                "acceptance",
                "hybrid_changes",
                "context_changes",
                "claim_boundary",
            ):
                mutated = copy.deepcopy(original)
                mutated[field] = {} if isinstance(original[field], dict) else []
                target.write_text(json.dumps(mutated))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    diagnostic.registration(root)

    def record_fixture(self):
        cells, controls = [], {}
        for profile, case_id in diagnostic.CELLS:
            controls[case_id] = {"against_human_reference": {"word_edit_distance": 6}}
            cells.append(
                {
                    "id": f"{profile}:{case_id}",
                    "profile": profile,
                    "qualified": True,
                    "stream_status": "completed",
                    "error": None,
                    "checks": dict.fromkeys(diagnostic.CHECK_NAMES, True),
                    "quality_gate_passed": True,
                    "capacity_restored_after_close": True,
                    "recognition": {
                        "against_human_reference": {
                            "word_edit_distance": 6,
                            "reference_word_count": 88,
                        },
                        "against_offline_control": {"word_edit_distance": 0},
                    },
                }
            )
            if profile == "hybrid":
                cells[-1].update(
                    qualified=False,
                    stream_status="policy_failure",
                    quality_gate_passed=False,
                    error={"category": "policy_resolution"},
                )
                cells[-1]["checks"].update(
                    dict.fromkeys(diagnostic.COMPLETION_CHECKS, False)
                )
                cells[-1]["recognition"]["against_human_reference"][
                    "word_edit_distance"
                ] = 7
        source = {"digest": "a"}
        record = {
            "source": {"snapshot": source},
            "cells": cells,
            "controls": controls,
            "model": {"unchanged": True, "initial_sha256": "x", "final_sha256": "x"},
            "capacity_restored": True,
            "qualified": True,
            "status": "completed",
            "qualification_scope": "context_candidates_with_matched_controls",
        }
        record.update(
            diagnostic.qualification_summary(
                cells, model_unchanged=True, capacity_restored=True
            )
        )
        return record, source

    def test_quality_failure_cannot_be_relabelled_qualified(self):
        record, source = self.record_fixture()
        diagnostic.validate_record(record, source)
        for field in (
            "quality",
            "runtime",
            "capacity",
            "model",
            "order",
            "empty_checks",
            "missing_check",
            "numeric_check",
        ):
            changed = copy.deepcopy(record)
            if field == "quality":
                changed["cells"][3]["recognition"]["against_human_reference"][
                    "word_edit_distance"
                ] = 7
            elif field == "runtime":
                changed["cells"][3]["stream_status"] = "policy_failure"
            elif field == "capacity":
                changed["cells"][3]["capacity_restored_after_close"] = False
            elif field == "model":
                changed["model"]["final_sha256"] = "y"
            elif field == "empty_checks":
                changed["cells"][3]["checks"] = {}
            elif field == "missing_check":
                changed["cells"][3]["checks"].pop("native_pcm_binding")
            elif field == "numeric_check":
                changed["cells"][3]["checks"]["native_pcm_binding"] = 1
            else:
                changed["cells"].reverse()
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostic.validate_record(changed, source)

    def test_expected_control_failures_remain_unqualified_without_failing_candidate(
        self,
    ):
        record, source = self.record_fixture()
        diagnostic.validate_record(record, source)
        self.assertTrue(record["candidate_qualified"])
        self.assertTrue(record["qualified"])
        self.assertFalse(record["all_cells_qualified"])
        self.assertTrue(record["controls_match_expected"])
        self.assertEqual(record["control_outcomes"], diagnostic.CONTROL_EXPECTATIONS)
        self.assertEqual(
            [cell["qualified"] for cell in record["cells"]],
            [True, False, False, True, True, True, True],
        )

    def test_control_integrity_and_reproduction_cannot_be_bypassed(self):
        record, source = self.record_fixture()
        for field in (
            "error",
            "check",
            "missing_check",
            "numeric_check",
            "capacity",
            "status",
            "quiet_quality",
            "profile",
            "missing_cell",
        ):
            changed = copy.deepcopy(record)
            cell = changed["cells"][1]
            if field == "error":
                cell["error"] = {"category": "native_execution"}
            elif field == "check":
                cell["checks"]["native_pcm_binding"] = False
            elif field == "missing_check":
                cell["checks"].pop("aligned_word_identity")
            elif field == "numeric_check":
                cell["checks"]["bounded_audio_buffer"] = 1
            elif field == "capacity":
                cell["capacity_restored_after_close"] = False
            elif field == "status":
                cell["stream_status"] = "completed"
            elif field == "quiet_quality":
                changed["cells"][0]["recognition"]["against_offline_control"][
                    "word_edit_distance"
                ] = 1
            elif field == "profile":
                cell["profile"] = "context"
            else:
                changed["cells"].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostic.validate_record(changed, source)

    def test_summary_cannot_relabel_failed_controls_or_candidate(self):
        record, source = self.record_fixture()
        for field in (
            "candidate_qualified",
            "all_cells_qualified",
            "controls_match_expected",
            "control_outcomes",
            "qualification_scope",
        ):
            changed = copy.deepcopy(record)
            changed[field] = (
                not changed[field] if isinstance(changed[field], bool) else {}
            )
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostic.validate_record(changed, source)

    def test_unexpected_passing_controls_invalidate_matched_comparison(self):
        record, source = self.record_fixture()
        for cell in record["cells"][1:3]:
            cell.update(
                qualified=True,
                stream_status="completed",
                error=None,
                quality_gate_passed=True,
            )
            cell["checks"].update(dict.fromkeys(diagnostic.COMPLETION_CHECKS, True))
            cell["recognition"]["against_human_reference"]["word_edit_distance"] = 6
        record.update(
            diagnostic.qualification_summary(
                record["cells"], model_unchanged=True, capacity_restored=True
            )
        )
        self.assertTrue(record["all_cells_qualified"])
        self.assertFalse(record["candidate_qualified"])
        self.assertFalse(record["controls_match_expected"])
        with self.assertRaises(ValueError):
            diagnostic.validate_record(record, source)
        record.update(qualified=False, status="failed")
        diagnostic.validate_record(record, source)

    def test_candidate_policy_failure_is_not_waived_by_expected_control_failures(self):
        record, source = self.record_fixture()
        candidate = copy.deepcopy(record["cells"][1])
        candidate.update(id="context:mixed-control", profile="context")
        record["cells"][3] = candidate
        record.update(
            diagnostic.qualification_summary(
                record["cells"], model_unchanged=True, capacity_restored=True
            )
        )
        record.update(qualified=False, status="failed")
        diagnostic.validate_record(record, source)
        self.assertTrue(record["controls_match_expected"])
        self.assertFalse(record["candidate_qualified"])
        self.assertFalse(record["all_cells_qualified"])
        candidate["qualified"] = True
        with self.assertRaises(ValueError):
            diagnostic.validate_record(record, source)

    def test_real_hybrid_driver_checks_closed_quiet_without_false_stream_eof(self):
        self._assert_real_driver(0)

    def test_real_context_driver_preserves_publication_integrity(self):
        self._assert_real_driver(600)

    def _assert_real_driver(self, context_limit):
        with patch.object(
            sys,
            "path",
            [str(diagnostic.ROOT / "tests"), str(diagnostic.ROOT / "src"), *sys.path],
        ):
            from test_continuous_endpoint_words import word_result
            from test_continuous_evidence import EvidenceNativeAdapter
            from test_continuous_stream import pcm_ms

            from whisper_runtime.adapters.audio_endpoints import QuietEndpointConfig
            from whisper_runtime.adapters.continuous_stream import (
                ContinuousStreamConfig,
                ContinuousTranscriptStream,
            )

        adapter = EvidenceNativeAdapter(word_result)
        config = ContinuousStreamConfig(
            preview_interval_ms=200,
            max_window_ms=2000,
            max_buffer_ms=2400,
            holdback_ms=200,
            timestamp_tolerance_ms=0,
            left_context_ms=200,
            input_evidence=True,
            source_units=True,
            word_boundary_fallback=True,
            word_context_limit_ms=context_limit,
            endpointing=QuietEndpointConfig(max_quiet_unit_ms=2000),
        )
        stream = ContinuousTranscriptStream(
            adapter,
            stream_id="acoustic-driver-hybrid",
            mel_builder=lambda content: content,
            config=config,
        )
        self.addCleanup(stream.close)
        pcm = pcm_ms(1200, 900) + pcm_ms(800, 1)
        helper = diagnostic._corpus().b
        events, traces, steps, chunks, error = helper._drive_stream(
            stream, pcm, chunk_bytes=200 * 32
        )
        self.assertIsNone(error)
        self.assertGreater(steps, 0)
        self.assertEqual(chunks, 10)
        checks = helper._event_checks(
            events,
            traces,
            accepted_samples=len(pcm) // 2,
            total_samples=len(pcm) // 2,
            state=stream.state,
            metrics=stream.metrics,
            budget=adapter.budget,
            worker=adapter.worker,
            capacity=adapter.capacity,
            retained_from_sample=stream.retained_from_sample,
            max_buffer_samples=config.max_buffer_ms * 16,
            error_record=error,
        )
        checks.update(diagnostic.publication_checks(events, traces, pcm))
        checks["profile_identity"] = stream.profile_id == (
            "word_boundary_quiet_endpoint_stream/v1"
            + ("+word_context/v1" if context_limit else "")
            + "+input_evidence/v1"
        )
        self.assertEqual(set(checks), diagnostic.CHECK_NAMES)
        self.assertTrue(all(checks.values()), checks)
        quiet = [
            trace
            for trace in traces
            if trace["source_unit"] is not None
            and trace["source_unit"]["origin"] == "quiet_run"
        ]
        self.assertEqual(len(quiet), 1)
        self.assertTrue(quiet[0]["word_publication"]["final"])
        self.assertFalse(quiet[0]["eof"])
        self.assertGreater(quiet[0]["word_publication"]["word_start"], 0)
        self.assertNotEqual(
            quiet[0]["word_publication"]["text"], quiet[0]["result"]["text"]
        )
        self.assertEqual(sum(event["kind"] == "final" for event in events), 1)
        self.assertEqual(events[-1]["kind"], "final")
        self.assertEqual(adapter.budget.available, adapter.capacity)
        self.assertEqual(adapter.worker.queue_depth, 0)

    def test_real_transport_roundtrip_binds_new_manifest_worker_and_source(self):
        settings = diagnostic.registration()
        helper = diagnostic._corpus().b
        snapshot = {"commit": "a" * 40, "digest": "b" * 64, "files": []}
        registration_hash = diagnostic._sha(
            (diagnostic.ROOT / diagnostic.MANIFEST).read_bytes()
        )
        call_id = "fc-AcousticLocal123"
        record = {
            "schema_version": "1-diagnostic",
            "claim_boundary": settings["claim_boundary"],
            "source": {
                "snapshot": snapshot,
                "registration_sha256": registration_hash,
            },
            "worker": {
                "function_call_id": call_id,
                "function_call_id_sha256": diagnostic._sha(call_id.encode()),
            },
            "status": "failed",
            "qualified": False,
            "cells": [],
        }
        payload = helper._encode_worker_record(record)
        decoded = helper._decode_worker_record(
            payload,
            expected_snapshot=snapshot,
            registration_sha256=registration_hash,
            manifest=settings,
        )
        self.assertEqual(decoded, record)
        self.assertIsInstance(payload, bytes)
        for changed_field in (
            "function_call_id_sha256",
            "registration_sha256",
            "snapshot",
            "claim_boundary",
        ):
            changed = copy.deepcopy(record)
            if changed_field == "function_call_id_sha256":
                changed["worker"][changed_field] = "0" * 64
            elif changed_field == "registration_sha256":
                changed["source"][changed_field] = "0" * 64
            elif changed_field == "snapshot":
                changed["source"]["snapshot"]["digest"] = "0" * 64
            else:
                changed["claim_boundary"] = {}
            with (
                self.subTest(changed_field=changed_field),
                self.assertRaises(ValueError),
            ):
                helper._decode_worker_record(
                    helper._encode_worker_record(changed),
                    expected_snapshot=snapshot,
                    registration_sha256=registration_hash,
                    manifest=settings,
                )

    def test_paid_permission_and_preflight_are_required_before_resource_calls(self):
        with patch.object(diagnostic, "resources") as resources:
            with self.assertRaises(ValueError):
                diagnostic.run(replay_id="not-authorized")
            resources.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(diagnostic, "registration", return_value={}),
                patch.object(diagnostic, "snapshot", return_value={}),
                patch.object(diagnostic, "_cases", return_value=({}, {})),
                patch.object(diagnostic, "resources") as resources,
            ):
                with self.assertRaises(FileNotFoundError):
                    diagnostic.run(
                        replay_id="no-preflight", confirm_paid_gpu=True, root=Path(temp)
                    )
                resources.assert_not_called()

    def test_paths_reject_escapes_and_existing_receipt_blocks_rerun(self):
        for value in ("", "../a", "a/b", "a\\b", "CON", "NUL", "LPT1"):
            with self.assertRaises(ValueError):
                diagnostic._paths(diagnostic.ROOT, value, True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, receipt, _ = diagnostic._paths(root, "existing", True)
            receipt.parent.mkdir(parents=True)
            receipt.write_text("preserved")
            with patch.object(diagnostic, "registration", return_value={}):
                with self.assertRaises(FileExistsError):
                    diagnostic.run(replay_id="existing", preflight=True, root=root)
            self.assertEqual(receipt.read_text(), "preserved")

    def test_actual_sdk_defines_functions_without_app_run(self):
        try:
            modal = importlib.import_module("modal")
        except ImportError:
            self.skipTest("Modal SDK is not installed")
        if modal.__version__ != "1.5.5":
            self.skipTest("Registered Modal SDK unavailable")
        with patch.object(
            modal.App, "run", side_effect=AssertionError("remote execution")
        ):
            app, echo, execute = diagnostic.resources(
                {
                    "files": [
                        {"path": diagnostic.MANIFEST},
                        {"path": diagnostic.PRODUCER},
                    ]
                }
            )
        self.assertIsNotNone(app)
        self.assertIsNotNone(echo)
        self.assertIsNotNone(execute)


if __name__ == "__main__":
    unittest.main()
