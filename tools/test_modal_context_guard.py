"""CPU-only fixtures: synthetic outputs test the protocol, not model recognition."""

import copy
import json
import runpy
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from infra import modal_context_guard as p
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeTimestampSegment
from whisper_runtime.adapters.native_result import NativeWindowResult
from whisper_runtime.adapters.word_policy import NativeWordAlignment


def synthetic_cases():
    """Use saved state and outputs, but invented zero PCM; never dispatch this."""
    archive = p._seven()
    inventory = []
    for saved_input, saved, plan in zip(
        archive["inputs"], archive["cells"], p.REGISTERED_WINDOWS
    ):
        case_id, start, end, kind = plan
        pcm = b"\x00" * (saved_input["sample_count"] * 2)
        raw = copy.deepcopy(saved["raw_alignments"])
        inventory.append(
            dict(
                id=case_id,
                pcm=pcm,
                reference_text=saved_input["reference_text"],
                frozen_state=saved_input["frozen_state"],
                session_version=saved_input["session_version"],
                source_reason=saved_input["source_reason"],
                source_native=raw["current"]["native"],
                source_trace=saved["original_source_trace"],
                source_diagnostic=saved["original_anchor_diagnostic"],
                archive=saved_input["archive"],
                context_archive=dict(
                    path=p.ARCHIVE, sha256=p.ARCHIVE_SHA, cell_id=case_id
                ),
                historical_alignments=raw,
                historical_summary=saved["summary"],
                historical_fresh_windows=saved["fresh_windows"],
                guard_observation=dict(
                    kind=kind,
                    arm="guard",
                    start_ms=start,
                    end_ms=end,
                    pcm_sha256=p.shared._sha(pcm[start * 32 : end * 32]),
                    sample_count=(end - start) * 16,
                    requested_guard_ms=500,
                    historical_source="synthetic-fixture-only",
                    historical_cost=None,
                ),
                windows=[]
                if kind == "reused"
                else [dict(arm="guard", start_ms=start, end_ms=end)],
            )
        )
    return inventory


def synthetic_record(inventory, inputs):
    archive = p._seven()
    cells, count = [], 0
    for case in inventory:
        cell = dict(
            id=case["id"],
            status="evaluated",
            archive=case["archive"],
            original_source_trace=case["source_trace"],
            original_anchor_diagnostic=case["source_diagnostic"],
            raw_alignments=p.initial_alignments(case),
            fresh_windows=[],
            publication_authorized=False,
            error=None,
            comparability=p.provenance_checks(
                case, archive["effective_identity"], archive["backend"]
            ),
        )
        assert cell["comparability"]["comparable"]
        for window in case["windows"]:
            count += 1
            start, end = window["start_ms"], window["end_ms"]
            span = AudioSpan(start, end)
            window_id = f"handoff:{case['id']}:guard"
            native = NativeWindowResult(
                window_id=window_id,
                text="synthetic",
                start_ms=start,
                end_ms=end,
                analysis_span=span,
            )
            cell["raw_alignments"]["guard"] = asdict(
                NativeWordAlignment(
                    native,
                    (NativeTimestampSegment(span, "synthetic", (1,)),),
                )
            )
            cell["fresh_windows"].append(
                dict(
                    **window,
                    pcm_sha256=p.shared._sha(p._slice(case, start, end)),
                    sample_count=(end - start) * 16,
                    first_call_no_warmup=count == 1,
                    completed=True,
                    error=None,
                    mel_shape=[80, 3000],
                    decoder_steps=1,
                    capacity_restored=True,
                    memory_before_bytes=0,
                    mel_wall_ns=1,
                    wall_ns=10,
                    peak_allocated_bytes=0,
                    peak_reserved_bytes=0,
                    memory_after_close_bytes=0,
                    memory_after_handle_release_bytes=0,
                    measurement=dict(
                        call_index=count,
                        window_id=window_id,
                        start_ms=start,
                        end_ms=end,
                        reuse_alignment_features=False,
                        gpu_device_time_ms=None,
                        operation_wall_ns={
                            "decode": 1,
                            "result": 1,
                            "alignment": 1,
                            "close": 1,
                        },
                        closed=True,
                        capacity_restored=True,
                        encoder_calls=[
                            dict(phase="decode", input_frames=3000, use_sdpa=None),
                            dict(phase="alignment", input_frames=3000, use_sdpa=False),
                        ],
                    ),
                )
            )
        cell["summary"] = p.summarize_cell(case, cell["raw_alignments"])
        cells.append(cell)
    return p._canonical(
        dict(
            schema_version="1-diagnostic",
            experiment_id=p.EXPERIMENT_ID,
            source=dict(snapshot=dict(fixture="synthetic-not-a-measurement")),
            status="completed",
            qualified=False,
            scope=p.scope(),
            claim_boundary=p.CLAIMS,
            effective_identity=archive["effective_identity"],
            backend=archive["backend"],
            model=archive["model"],
            inputs=inputs,
            cells=cells,
            native_window_count=count,
            capacity_restored=True,
            elapsed_ns=1,
            stop=None,
        )
    )


class ContextGuardProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = synthetic_cases()
        with patch.object(p, "cases", return_value=cls.inventory):
            cls.inputs = p.input_records()
        cls.record = synthetic_record(cls.inventory, cls.inputs)

    def validate(self, record):
        with (
            patch.object(p, "cases", return_value=self.inventory),
            patch.object(p, "input_records", return_value=self.inputs),
        ):
            p.validate_record(record, self.record["source"]["snapshot"])

    def test_json_roundtrip_preserves_three_new_and_two_reused_observations(self):
        self.validate(json.loads(json.dumps(self.record)))
        self.assertEqual(self.record["native_window_count"], 3)
        self.assertEqual(
            [len(c["fresh_windows"]) for c in self.record["cells"]], [0, 1, 1, 1, 0]
        )
        for case, cell in zip(self.inventory, self.record["cells"]):
            self.assertEqual(
                cell["summary"]["historical_summary"], case["historical_summary"]
            )
            self.assertFalse(cell["summary"]["fresh_control_reproduction"])
            self.assertNotIn("control_native_reproduced", cell["summary"])
            self.assertFalse(cell["summary"]["publication_authorized"])

    def test_initial_alignments_copy_history_without_aliasing(self):
        for case in self.inventory:
            before = json.dumps(case["historical_alignments"], sort_keys=True)
            raw = p.initial_alignments(case)
            self.assertEqual(
                "guard" in raw, case["guard_observation"]["kind"] == "reused"
            )
            raw["current"]["native"]["text"] = "changed fixture copy"
            self.assertEqual(
                before, json.dumps(case["historical_alignments"], sort_keys=True)
            )

    def test_reference_neither_selects_nor_changes_diagnostic(self):
        for case, cell in zip(self.inventory, self.record["cells"]):
            changed = {**case, "reference_text": "intentionally unrelated reference"}
            self.assertEqual(
                p.summarize_cell(case, cell["raw_alignments"]),
                p.summarize_cell(changed, cell["raw_alignments"]),
            )

    def test_adversarial_mutations_are_rejected(self):
        def change_path(record, path, value):
            target = record
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value

        mutations = [
            (("experiment_id",), "other"),
            (("native_window_count",), 4),
            (("native_window_count",), True),
            (("qualified",), True),
            (("status",), "failed"),
            (("cells", 0, "publication_authorized"), True),
            (
                ("cells", 0, "raw_alignments", "current", "native", "text"),
                "invented history",
            ),
            (
                ("cells", 0, "raw_alignments", "guard", "native", "window_id"),
                "pretend-fresh",
            ),
            (
                ("cells", 1, "raw_alignments", "guard", "native", "window_id"),
                "unmeasured",
            ),
            (
                ("cells", 1, "raw_alignments", "overlap", "words", 0, "text"),
                "changed history",
            ),
            (("cells", 1, "fresh_windows", 0, "start_ms"), 20200),
            (("cells", 1, "fresh_windows", 0, "pcm_sha256"), "0" * 64),
            (("cells", 1, "fresh_windows", 0, "completed"), False),
            (("cells", 1, "fresh_windows", 0, "measurement"), None),
            (("cells", 1, "fresh_windows", 0, "first_call_no_warmup"), False),
            (("cells", 1, "fresh_windows", 0, "measurement", "call_index"), 2),
            (
                (
                    "cells",
                    1,
                    "fresh_windows",
                    0,
                    "measurement",
                    "reuse_alignment_features",
                ),
                True,
            ),
            (("cells", 1, "fresh_windows", 0, "measurement", "closed"), False),
            (
                (
                    "cells",
                    1,
                    "fresh_windows",
                    0,
                    "measurement",
                    "encoder_calls",
                    1,
                    "use_sdpa",
                ),
                True,
            ),
            (
                (
                    "cells",
                    1,
                    "fresh_windows",
                    0,
                    "measurement",
                    "encoder_calls",
                    0,
                    "input_frames",
                ),
                2999,
            ),
            (
                ("cells", 1, "summary", "guard_correspondence", "reason"),
                "strict_overlap_and_suffix_agree",
            ),
            (
                ("cells", 1, "summary", "historical_summary", "publication_authorized"),
                True,
            ),
            (("effective_identity", "rng_seed"), 8),
            (("effective_identity", "torch"), "unmeasured-version"),
            (("backend", "patch_sha256"), "0" * 64),
            (("model", "final_sha256"), "0" * 64),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                bad = copy.deepcopy(self.record)
                change_path(bad, path, value)
                with self.assertRaises(ValueError):
                    self.validate(bad)

    def test_unplanned_or_unmeasured_alignment_rejected_even_with_replayed_summary(
        self,
    ):
        for change in (
            "new_guard_from_history",
            "reused_guard_counted_fresh",
            "extra_arm",
            "changed_historical_anchor",
        ):
            with self.subTest(change=change):
                bad = copy.deepcopy(self.record)
                if change == "new_guard_from_history":
                    bad["cells"][1]["raw_alignments"]["guard"] = copy.deepcopy(
                        bad["cells"][1]["raw_alignments"]["overlap"]
                    )
                elif change == "reused_guard_counted_fresh":
                    bad["cells"][0]["fresh_windows"] = copy.deepcopy(
                        bad["cells"][1]["fresh_windows"]
                    )
                elif change == "extra_arm":
                    bad["cells"][1]["raw_alignments"]["retry"] = copy.deepcopy(
                        bad["cells"][1]["raw_alignments"]["guard"]
                    )
                else:
                    bad["cells"][1]["raw_alignments"]["overlap"]["words"][0][
                        "tokens"
                    ] = [999]
                    bad["cells"][1]["summary"] = p.summarize_cell(
                        self.inventory[1], bad["cells"][1]["raw_alignments"]
                    )
                with self.assertRaises(ValueError):
                    self.validate(bad)

    def test_partial_native_failure_is_recordable_and_cannot_claim_completion(self):
        bad = copy.deepcopy(self.record)
        bad["cells"] = bad["cells"][:2]
        cell = bad["cells"][-1]
        cell.update(
            status="lifecycle_failure", summary=None, error={"type": "SyntheticFailure"}
        )
        del cell["raw_alignments"]["guard"]
        observation = cell["fresh_windows"][0]
        observation.update(completed=False, error={"type": "SyntheticFailure"})
        observation["measurement"].update(closed=True, capacity_restored=True)
        observation["measurement"]["encoder_calls"] = []
        bad.update(
            status="failed",
            native_window_count=1,
            stop=dict(reason="native_lifecycle_failure", case_id=cell["id"]),
        )
        self.validate(bad)
        bad["status"] = "completed"
        with self.assertRaises(ValueError):
            self.validate(bad)

    def test_failure_before_native_admission_can_have_zero_calls(self):
        record = copy.deepcopy(self.record)
        record["cells"] = record["cells"][:2]
        cell = record["cells"][-1]
        cell.update(
            status="lifecycle_failure",
            summary=None,
            fresh_windows=[],
            error={"type": "SyntheticTimeout"},
        )
        del cell["raw_alignments"]["guard"]
        record.update(
            status="failed",
            native_window_count=0,
            stop=dict(reason="native_lifecycle_failure", case_id=cell["id"]),
        )
        self.validate(record)
        record["stop"] = None
        with self.assertRaises(ValueError):
            self.validate(record)

    def test_incompatible_identity_blocks_all_cells_without_native_work(self):
        record = copy.deepcopy(self.record)
        record["effective_identity"]["numpy"] = "different"
        record.update(status="failed", native_window_count=0)
        for case, cell in zip(self.inventory, record["cells"]):
            cell.update(
                status="blocked",
                raw_alignments={},
                fresh_windows=[],
                summary=None,
                comparability=p.provenance_checks(
                    case, record["effective_identity"], record["backend"]
                ),
            )
        self.validate(record)

    def test_legacy_bridge_is_not_relabelled_as_measured_old_packages(self):
        self.assertFalse(
            self.record["cells"][-1]["comparability"][
                "historical_package_identities_measured"
            ]
        )
        self.assertTrue(
            self.record["cells"][-1]["comparability"]["unused_optional_path_bridge"]
        )
        self.assertFalse(
            self.record["cells"][-1]["comparability"]["backend_trees_equal"]
        )

    def test_fixed_archive_corruption_rejected_before_model_work(self):
        with patch.object(Path, "read_bytes", return_value=b"{}"):
            with self.assertRaisesRegex(ValueError, "digest"):
                p._seven()

    def test_cli_and_shared_worker_keep_the_importable_producer_identity(self):
        with (
            patch(
                "infra.modal_cuda_lane.run", return_value="not-dispatched"
            ) as dispatch,
            patch(
                "sys.argv", ["context-guard", "--replay-id", "synthetic-never-dispatch"]
            ),
            patch("builtins.print"),
            patch("warnings.warn"),
        ):
            runpy.run_module("infra.modal_context_guard", run_name="__main__")
        self.assertIs(dispatch.call_args.kwargs["producer"], p)
        self.assertFalse(dispatch.call_args.kwargs["confirm_paid_gpu"])
        with patch(
            "infra.resolution_handoff_worker.run_worker", return_value={}
        ) as worker:
            p.run_worker({"fixture": True})
        worker.assert_called_once_with(
            {"fixture": True}, producer_module="infra.modal_context_guard"
        )


class ContextGuardLocalPCMTests(unittest.TestCase):
    def test_fixed_guards_rebuild_and_bind_existing_costs(self):
        try:
            inventory = p.cases()
        except FileNotFoundError as error:
            self.skipTest(f"optional local PCM assets unavailable: {error}")
        self.assertEqual(
            [
                (
                    c["id"],
                    c["guard_observation"]["start_ms"],
                    c["guard_observation"]["end_ms"],
                    c["guard_observation"]["kind"],
                )
                for c in inventory
            ],
            list(p.REGISTERED_WINDOWS),
        )
        self.assertEqual(sum(len(c["windows"]) for c in inventory), 3)
        for case in inventory:
            guard = case["guard_observation"]
            self.assertEqual(
                guard["pcm_sha256"],
                p.shared._sha(p._slice(case, guard["start_ms"], guard["end_ms"])),
            )
            self.assertGreaterEqual(
                guard["start_ms"], case["frozen_state"]["retained_ms"]
            )
            if guard["kind"] == "reused":
                self.assertEqual(
                    guard["historical_cost"]["pcm_sha256"], guard["pcm_sha256"]
                )
                self.assertIsNotNone(guard["historical_source"])
            else:
                self.assertIsNone(guard["historical_source"])


if __name__ == "__main__":
    unittest.main()
