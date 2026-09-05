"""Local experiment checks; no Modal client or native model is loaded."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infra import modal_word_resolution as experiment
from infra.word_resolution_worker import _bootstrap, _evaluate
from tools.analyze_word_resolution import _alignment
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeTimestampSegment


class WordResolutionTests(unittest.TestCase):
    def test_recorded_t4_choices_and_scores_replay_without_audio_or_model(self):
        path = (
            experiment.ROOT
            / "evidence/modal-t4-tiny-en-word-resolution-2026-09-06.json"
        )
        record = json.loads(path.read_bytes())
        self.assertTrue(experiment.experiment_complete(record, record["inputs"]))
        self.assertEqual(record["status"], "completed")
        self.assertFalse(record["qualified"])
        differences = experiment._corpus().b._word_difference
        for saved, item in zip(record["cells"], record["inputs"]):
            case = dict(item)
            raw = saved["raw_alignments"]
            if item["split"] == "heldout":
                case = _bootstrap(
                    case, _alignment(raw["bootstrap"]), experiment.PARAMETERS
                )
                self.assertIsNotNone(case)
            else:
                case["frozen_anchor"] = tuple(
                    NativeTimestampSegment(
                        AudioSpan(**word["span"]), word["text"], tuple(word["tokens"])
                    )
                    for word in item["frozen_anchor"]
                )
            replay = _evaluate(
                case,
                _alignment(raw["current"]),
                _alignment(raw["alternative"]),
                experiment.PARAMETERS,
                differences,
            )
            for key in (
                "arms",
                "routed",
                "outcome",
                "current_diagnostic",
                "shadow_proposal",
            ):
                self.assertEqual(
                    json.loads(json.dumps(replay[key])), saved[key], (item["id"], key)
                )
        self.assertEqual(
            [cell["routed"]["arm"] for cell in record["cells"]],
            ["alternative", "shadow", "baseline", "alternative"],
        )
        self.assertEqual(
            [cell["outcome"]["routed_word_edit_delta"] for cell in record["cells"]],
            [-30, -16, 0, -11],
        )

    def test_registration_is_fixed(self):
        settings = experiment.registration()
        self.assertEqual(settings["execution"]["maximum_native_windows"], 10)
        self.assertFalse(any(settings["claim_boundary"].values()))
        for key, replacement in (
            ("parameters", {}),
            ("execution", {}),
            ("claim_boundary", {}),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / experiment.MANIFEST
                path.parent.mkdir()
                changed = {**settings, key: replacement}
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    experiment.registration(root)

    def test_development_inputs_reproduce_archive_without_model(self):
        if not (experiment.ROOT / "artifacts/speech-corpus-v1").exists():
            self.skipTest("optional local PCM corpus is absent")
        if not (
            experiment.ROOT / "artifacts/resolution-inputs-v1/2961-960-0004.pcm"
        ).exists():
            self.skipTest("optional held-out PCM is absent")
        values = experiment.input_records()
        self.assertEqual([item["id"] for item in values[:2]], experiment.DEVELOPMENT)
        self.assertEqual([item["head_ms"] for item in values[:2]], [21260, 33540])
        self.assertEqual(values[0]["expected_native_text"], "It's the tents.")
        self.assertEqual([item["end_ms"] for item in values[2:]], [8220, 7740])
        self.assertTrue(all("pcm" not in item for item in values))
        self.assertTrue(all("frozen_anchor" not in item for item in values[2:]))

    def test_heldout_speakers_are_distinct_and_unused(self):
        original = json.loads((experiment.ROOT / experiment.INPUTS).read_text())
        for speaker in (6930, original["fixtures"][0]["source"]["speaker_id"]):
            changed = copy.deepcopy(original)
            changed["fixtures"][1]["source"]["speaker_id"] = speaker
            with patch.object(experiment.json, "loads", return_value=changed):
                with self.assertRaises(ValueError):
                    experiment._heldout(experiment.ROOT)

    def fixture(self):
        inputs, cells, count = [], [], 0
        for index in range(4):
            item = {
                "id": str(index),
                "split": "development" if index < 2 else "heldout",
                "head_ms": 100,
                "retained_ms": 0,
                "end_ms": 1000,
                "committed_text": "hello",
            }
            inputs.append(item)
            timings = {}
            for name in (
                ["current", "alternative"]
                if index < 2
                else ["bootstrap", "current", "alternative"]
            ):
                count += 1
                timings[name] = {
                    "call_index": count,
                    "completed": True,
                    "capacity_restored": True,
                }
            cells.append(
                {
                    "id": str(index),
                    "status": "evaluated",
                    "capacity_restored": True,
                    "outcome": {
                        "comparison_valid": True,
                        "expected_native_text_reproduced": True,
                        "routed_non_regression": False,
                    },
                    "routed": {"uses_reference": False},
                    "arms": {"baseline": {"publication_authority": False}},
                    "frozen_state": item,
                    "timings": timings,
                }
            )
        return {
            "model": {
                "unchanged": True,
                "initial_sha256": "abc",
                "final_sha256": "abc",
            },
            "capacity_restored": True,
            "native_window_count": 10,
            "cells": cells,
        }, inputs

    def test_completion_does_not_require_favourable_recognition(self):
        record, inputs = self.fixture()
        self.assertTrue(experiment.experiment_complete(record, inputs))

    def test_completion_requires_all_states_and_resource_cleanup(self):
        for fault in (
            "missing",
            "model",
            "count",
            "state",
            "capacity",
            "reproduction",
            "reference",
            "authority",
            "duplicate_call",
        ):
            with self.subTest(fault=fault):
                record, inputs = self.fixture()
                cell = record["cells"][0]
                if fault == "missing":
                    record["cells"].pop()
                elif fault == "model":
                    record["model"]["unchanged"] = False
                elif fault == "count":
                    record["native_window_count"] = 11
                elif fault == "state":
                    cell["status"] = "bootstrap_unresolved"
                elif fault == "capacity":
                    cell["timings"]["current"]["capacity_restored"] = False
                elif fault == "reproduction":
                    cell["outcome"]["expected_native_text_reproduced"] = False
                elif fault == "reference":
                    cell["routed"]["uses_reference"] = True
                elif fault == "authority":
                    cell["arms"]["baseline"]["publication_authority"] = True
                else:
                    cell["timings"]["current"]["call_index"] = 2
                self.assertFalse(experiment.experiment_complete(record, inputs))

    def test_paid_call_requires_explicit_flag_before_resources(self):
        with patch.object(experiment.shared, "resources") as resources:
            with self.assertRaises(ValueError):
                experiment.run(replay_id="not-started")
            resources.assert_not_called()


if __name__ == "__main__":
    unittest.main()
