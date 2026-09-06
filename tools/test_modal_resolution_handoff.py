"""CPU checks for the fixed overlap experiment; imports start no remote work."""

import copy
import hashlib
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from infra import modal_resolution_handoff as experiment
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeDecodeOptions, NativeTimestampSegment
from whisper_runtime.adapters.native_result import NativeWindowResult
from whisper_runtime.adapters.word_policy import NativeWordAlignment


def synthetic_record(cases, inputs):
    """Protocol fixture only: the generated words and timings are not measurements."""
    p = experiment
    base = p._corpus().read_registration(p.ROOT)
    model = base["model"]
    effective = dict(
        backend_artifacts=p.BACKEND_ARTIFACTS,
        decode_options=asdict(NativeDecodeOptions(**base["decode_options"])),
        rng_seed=7,
        alignment_mode="legacy",
        preprocessing="s16le-to-float32-div32768/pad-or-trim-480000/log-mel-80x3000/contiguous",
        model_sha256=model["model_state_sha256"],
        checkpoint_sha256=model["checkpoint_sha256"],
        precision="float32",
        torch="2.6.0+cu124",
        numpy="2.5.2",
        tiktoken="0.14.0",
        attention=dict(decode="backend-default-sdpa", alignment="explicit-non-sdpa"),
    )
    backend = dict(
        base_commit=p.features.BASE_COMMIT,
        base_tree=p.features.BASE_TREE,
        patched_tree=p.features.PATCHED_TREE,
        patch_sha256=p.features.PATCH_SHA,
        timing_sha256=p.BACKEND_ARTIFACTS["whisper/timing.py"],
    )
    cells, count = [], 0
    for case in cases:
        cell = dict(
            id=case["id"],
            status="evaluated",
            archive=case["archive"],
            original_source_trace=case["source_trace"],
            original_anchor_diagnostic=case["source_diagnostic"],
            comparability=p.provenance_checks(case, effective, backend),
            raw_alignments={},
            fresh_windows=[],
            publication_authorized=False,
            error=None,
        )
        assert cell["comparability"]["comparable"]
        if case["archived_current"] is not None:
            cell["raw_alignments"].update(
                current=case["archived_current"], candidate=case["archived_candidate"]
            )
        for window in case["windows"]:
            count += 1
            start, end, arm = window["start_ms"], window["end_ms"], window["arm"]
            span = AudioSpan(start, end)
            window_id = f"handoff:{case['id']}:{arm}"
            if arm == "current":
                native = copy.deepcopy(case["source_native"])
                native["window_id"] = window_id
                aligned = dict(
                    native=native,
                    words=[dict(span=asdict(span), text=native["text"], tokens=[1])],
                )
            else:
                native = NativeWindowResult(
                    window_id=window_id,
                    text="synthetic",
                    start_ms=start,
                    end_ms=end,
                    analysis_span=span,
                )
                aligned = asdict(
                    NativeWordAlignment(
                        native, (NativeTimestampSegment(span, "synthetic", (1,)),)
                    )
                )
            cell["raw_alignments"][arm] = p._canonical(aligned)
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
                    measurement=dict(
                        call_index=count,
                        window_id=window_id,
                        start_ms=start,
                        end_ms=end,
                        reuse_alignment_features=False,
                        gpu_device_time_ms=None,
                        operation_wall_ns={"decode": 1, "alignment": 1, "close": 1},
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
            experiment_id="modal-resolution-handoff-v1",
            status="completed",
            qualified=False,
            source={"snapshot": {"fixture": "synthetic"}},
            scope=p.scope(),
            claim_boundary=p.CLAIMS,
            effective_identity=effective,
            backend=backend,
            inputs=inputs,
            cells=cells,
            native_window_count=7,
            capacity_restored=True,
            elapsed_ns=1,
            stop=None,
            model=dict(
                initial_sha256=model["model_state_sha256"],
                final_sha256=model["model_state_sha256"],
                unchanged=True,
                checkpoint_sha256=model["checkpoint_sha256"],
                backend_revision=p.features.PATCHED_TREE,
            ),
        )
    )


class ResolutionHandoffInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cases = experiment.cases()
        except FileNotFoundError as error:
            raise unittest.SkipTest(
                "optional local PCM assets are unavailable"
            ) from error

    def test_seven_fixed_windows_use_retained_pcm_without_changing_the_plan(self):
        expected = (
            (1680, 10890),
            (3680, 10890),
            (2020, 10890),
            (20720, 33660),
            (32400, 43660),
            (1120, 8220),
            (1100, 7740),
        )
        windows = [
            (window["start_ms"], window["end_ms"])
            for case in self.cases
            for window in case["windows"]
        ]
        self.assertEqual(tuple(windows), expected)
        self.assertEqual(len(self.cases), 5)
        for case in self.cases:
            frozen = case["frozen_state"]
            anchor = frozen["anchor"]
            self.assertEqual(anchor[-1]["span"]["end_ms"], frozen["head_ms"])
            overlap = next(w for w in case["windows"] if w["arm"] == "overlap")
            self.assertEqual(
                overlap["start_ms"], anchor[0]["span"]["start_ms"] // 20 * 20
            )
            for window in case["windows"]:
                start, end = window["start_ms"], window["end_ms"]
                self.assertGreaterEqual(start, frozen["retained_ms"])
                self.assertEqual(end, frozen["end_ms"])
                self.assertLess(start, end)
                self.assertLessEqual(end - start, 30_000)
                self.assertEqual(
                    len(case["pcm"][start * 32 : end * 32]), (end - start) * 32
                )

    def test_archive_identity_and_old_candidate_pcm_reproduce_exactly(self):
        for case in self.cases:
            archive = case["archive"]
            path = experiment.ROOT / archive["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), archive["sha256"]
            )
        old = json.loads(
            (
                experiment.ROOT
                / "evidence/modal-t4-tiny-en-word-resolution-2026-09-06.json"
            ).read_bytes()
        )
        for case, saved in zip(self.cases[1:], old["cells"]):
            self.assertEqual(case["frozen_state"], saved["frozen_state"])
            self.assertEqual(
                case["archived_candidate"], saved["raw_alignments"]["alternative"]
            )
            timing = saved["timings"]["alternative"]
            pcm = case["pcm"][timing["start_ms"] * 32 : timing["end_ms"] * 32]
            self.assertEqual(hashlib.sha256(pcm).hexdigest(), timing["pcm_sha256"])

    def test_input_manifest_is_deterministic_and_contains_no_pcm_or_tensors(self):
        records = experiment.input_records()
        self.assertEqual(records, experiment.input_records())
        plain = json.dumps(records, sort_keys=True)
        self.assertNotIn('"pcm":', plain)
        self.assertNotIn('"audio_features":', plain)
        self.assertNotIn('"qualified": true', plain)

    def test_reference_text_cannot_change_the_assessment_or_authorize_publication(self):
        # The constructed overlap is not a claimed model observation. It uses
        # archived words solely to exercise the pure assessor without a GPU.
        case = copy.deepcopy(self.cases[1])
        overlap = copy.deepcopy(case["archived_candidate"])
        overlap["native"]["start_ms"] = case["windows"][0]["start_ms"]
        overlap["native"]["analysis_span"]["start_ms"] = case["windows"][0]["start_ms"]
        raw = dict(
            current=case["archived_current"],
            candidate=case["archived_candidate"],
            overlap=overlap,
        )
        before = json.dumps(raw, sort_keys=True)
        first = experiment.summarize_cell(case, raw)
        case["reference_text"] = "deliberately unrelated reference words"
        second = experiment.summarize_cell(case, raw)
        self.assertEqual(first["assessment"], second["assessment"])
        self.assertEqual(first["proposed_text"], second["proposed_text"])
        self.assertNotEqual(
            first["against_human_reference"], second["against_human_reference"]
        )
        self.assertFalse(first["publication_authorized"])
        self.assertFalse(first["assessment"]["publication_authorized"])
        self.assertIn(
            "acoustic_boundary_coverage", first["assessment"]["missing_evidence"]
        )
        self.assertEqual(before, json.dumps(raw, sort_keys=True))

    def test_json_roundtrip_and_adversarial_record_joins(self):
        inputs = experiment.input_records()
        record = synthetic_record(self.cases, inputs)
        expected = record["source"]["snapshot"]
        with (
            patch.object(experiment, "cases", return_value=self.cases),
            patch.object(experiment, "input_records", return_value=inputs),
        ):
            experiment.validate_record(json.loads(json.dumps(record)), expected)
            for change in (
                "archive",
                "window",
                "count",
                "index",
                "mode",
                "attention",
                "fingerprint",
                "schema",
                "publication",
            ):
                with self.subTest(change=change):
                    bad = copy.deepcopy(record)
                    first = bad["cells"][0]
                    if change == "archive":
                        cell = bad["cells"][1]
                        cell["raw_alignments"]["candidate"]["native"]["window_id"] = (
                            "invented-history"
                        )
                        cell["summary"] = experiment.summarize_cell(
                            self.cases[1], cell["raw_alignments"]
                        )
                    elif change == "window":
                        first["raw_alignments"]["overlap"]["native"]["window_id"] = (
                            "unmeasured-window"
                        )
                        first["summary"] = experiment.summarize_cell(
                            self.cases[0], first["raw_alignments"]
                        )
                    elif change == "count":
                        bad["native_window_count"] = 8
                    elif change == "index":
                        first["fresh_windows"][0]["measurement"]["call_index"] = 2
                    elif change == "mode":
                        first["fresh_windows"][0]["measurement"][
                            "reuse_alignment_features"
                        ] = True
                    elif change == "attention":
                        first["fresh_windows"][0]["measurement"]["encoder_calls"][1][
                            "use_sdpa"
                        ] = True
                    elif change == "fingerprint":
                        bad["effective_identity"]["backend_artifacts"][
                            "whisper/tokenizer.py"
                        ] = "0" * 64
                    elif change == "schema":
                        bad["schema_version"] = "unregistered"
                    else:
                        first["publication_authorized"] = True
                    with self.assertRaises(ValueError):
                        experiment.validate_record(bad, expected)


class ResolutionHandoffGuardTests(unittest.TestCase):
    def test_archive_corruption_is_rejected_before_model_work(self):
        with patch.object(Path, "read_bytes", return_value=b"{}"):
            with self.assertRaisesRegex(ValueError, "digest"):
                experiment._archive(
                    experiment.ROOT, experiment.OLD_ARCHIVE, experiment.OLD_SHA
                )


if __name__ == "__main__":
    unittest.main()
