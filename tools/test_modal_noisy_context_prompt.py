"""CPU-only archive joins, immutable registration and bounded receipt checks."""

from __future__ import annotations

import copy
import importlib
import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infra import modal_noisy_context_prompt as p
from infra.eof_context_retry_worker import _MeasuredAdapter


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = p.source_case()

    def test_prompt_is_exact_actual_commits_not_preview_or_human_reference(self):
        source = self.source
        self.assertEqual(
            source["prompt"], "Concord returned to its place amidst the tents"
        )
        self.assertFalse(source["prompt"].endswith("."))
        self.assertEqual(
            [item["text"] for item in source["prompt_commits"]],
            ["Concord returned to its", "place amidst the tents"],
        )
        self.assertEqual(source["prompt_commits"][-1]["end_sample"], 58880)
        self.assertNotEqual(source["prompt"], source["reference_text"])
        self.assertEqual(
            source["prompt_sha256"], p.shared._sha(source["prompt"].encode())
        )

    def test_fixed_crops_old_anchor_and_original_default_options(self):
        source = self.source
        self.assertEqual(source["pcm_sha256"], p.PCM_SHA)
        self.assertEqual(source["head_sample"], 58880)
        self.assertEqual(source["retained_sample"], 26880)
        self.assertEqual(source["accepted_sample"], 174240)
        self.assertEqual(source["source_trace"]["reason"], "no_lexical_text")
        self.assertEqual(
            [(c["id"], c["start_ms"], c["end_ms"]) for c in source["crops"]],
            list(p.CROPS),
        )
        self.assertEqual(
            [word["text"] for word in source["anchor"]],
            [" place", " amidst", " the", " tents"],
        )
        self.assertEqual(source["anchor"][-1]["span"]["end_ms"], 3680)
        for crop in source["crops"]:
            pcm = source["pcm"][crop["start_ms"] * 32 : crop["end_ms"] * 32]
            self.assertEqual(len(pcm) // 2, crop["sample_count"])
            self.assertEqual(p.shared._sha(pcm), crop["pcm_sha256"])
        self.assertEqual(
            source["decode_options"], source["effective_identity"]["decode_options"]
        )
        self.assertIsNone(source["decode_options"]["sample_len"])
        self.assertIsNone(source["decode_options"]["prompt"])

    def test_input_is_lossless_json_and_excludes_pcm(self):
        with patch.object(p, "source_case", return_value=self.source):
            record = p.input_record()
        self.assertNotIn("pcm", record)
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False)), record)
        self.assertEqual(record["archive"]["sha256"], p.ARCHIVE_SHA)

    def test_archive_byte_change_blocks_before_pcm_loading(self):
        with (
            patch("pathlib.Path.read_bytes", return_value=b"{}"),
            patch.object(p.shared, "_cases") as build,
        ):
            with self.assertRaisesRegex(ValueError, "archived paced evidence changed"):
                p.source_case()
        build.assert_not_called()

    def test_pcm_change_cannot_be_hidden_by_saved_archive_metadata(self):
        original = p.shared._cases

        def changed(*args):
            metadata, pcms = original(*args)
            raw = pcms["mixed-continuous-noise64"]
            pcms["mixed-continuous-noise64"] = bytes([raw[0] ^ 1]) + raw[1:]
            return metadata, pcms

        with patch.object(p.shared, "_cases", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "registered noisy PCM"):
                p.source_case()

    def test_work_bounds_are_registered_without_qualification(self):
        scope = p.scope()
        self.assertEqual(p.GPU_TIMEOUT_SECONDS, 90)
        self.assertEqual(p.CLEANUP_RESERVE_SECONDS, 15)
        self.assertEqual(p.MAX_NATIVE_WINDOWS, 2)
        self.assertEqual(p.MAX_DRIVER_STEPS, 224)
        self.assertEqual(p.SEED, 7)
        self.assertEqual(scope["max_decode_attempts"], 4)
        self.assertEqual(scope["expected_encoder_forwards"], 6)
        self.assertEqual(scope["independent_decode_alignment_encoder_forwards"], 8)
        self.assertTrue(scope["nonpublishing"])
        self.assertTrue(scope["head_only_has_no_join_authority"])
        self.assertTrue(scope["reuse_decode_features"])
        self.assertFalse(scope["borrowed_alignment_features"])
        self.assertFalse(scope["reference_used_for_selection"])
        self.assertTrue(all(value is False for value in p.CLAIMS.values()))

    def test_control_ignores_only_window_identity_and_does_not_gate_result(self):
        raw = copy.deepcopy(self.source["archived_alignment"])
        raw["native"]["window_id"] = "new-independent-observation"
        cells = [dict(attempts=[dict(alignment=raw)])]
        result = p.control(cells, self.source)
        self.assertTrue(result["retained_null_alignment_equal"])
        self.assertTrue(result["mismatch_is_observation_not_launch_failure"])
        self.assertEqual(result["head_only_control"], "not_registered")
        raw["words"][0]["tokens"][0] += 1
        self.assertFalse(p.control(cells, self.source)["retained_null_alignment_equal"])
        self.assertIsNone(p.control([], self.source)["retained_null_alignment_equal"])

    def test_control_matches_live_dataclass_tuples_and_json_roundtrip(self):
        from tools.analyze_word_resolution import _alignment

        live = asdict(_alignment(self.source["archived_alignment"]))
        live["native"]["window_id"] = "fresh-native-window"
        self.assertIsInstance(live["words"], tuple)
        self.assertIsInstance(live["native"]["metadata"]["tokens"], tuple)
        cells = [dict(attempts=[dict(alignment=live)])]
        saved = json.loads(json.dumps(cells, allow_nan=False))
        before = p.control(cells, self.source)
        after = p.control(saved, self.source)
        self.assertTrue(before["retained_null_alignment_equal"])
        self.assertEqual(before, after)
        # Representation normalization must not hide any actual native difference.
        live["native"]["metadata"]["avg_logprob"] += 0.01
        self.assertFalse(p.control(cells, self.source)["retained_null_alignment_equal"])

    def test_snapshot_covers_executed_helpers_not_old_archive_chain(self):
        def git(root, *args):
            return (
                p.PRODUCER
                if args[0] == "diff"
                else ""
                if args[0] == "ls-files"
                else "git-id"
            )

        with patch.object(p, "input_record"), patch.object(p, "_git", side_effect=git):
            saved = p.snapshot()
        names = {item["path"] for item in saved["files"]}
        self.assertTrue(
            {p.PRODUCER, p.WORKER, p.TEST, p.ARCHIVE, p.prior.WORKER} <= names
        )
        self.assertNotIn(p.prior.ARCHIVE, names)
        self.assertNotIn("infra/modal_paced_context_retry.py", names)
        self.assertEqual(saved["digest"], p.shared._hash(saved["files"]))
        self.assertEqual(saved["dirty_source_paths"], [p.PRODUCER])

    def test_snapshot_still_rejects_unreviewed_dirty_executed_source(self):
        with (
            patch.object(p, "input_record"),
            patch.object(
                p, "_git", return_value="src/whisper_runtime/adapters/word_policy.py"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unreviewed dirty executed source"):
                p.snapshot()

    def test_worker_wrapper_passes_snapshot_without_mutating_prior(self):
        module = importlib.import_module("infra.noisy_context_prompt_worker")
        old_timeout = p.prior.GPU_TIMEOUT_SECONDS
        with patch.object(module, "run_worker", return_value="record") as run:
            self.assertEqual(p.run_worker({"digest": "bound"}), "record")
        run.assert_called_once_with({"digest": "bound"})
        self.assertEqual(p.prior.GPU_TIMEOUT_SECONDS, old_timeout)


class AdmissionTests(unittest.TestCase):
    def measured(self, now=1):
        native = SimpleNamespace(model_identity="model", start_window=Mock())
        measured = _MeasuredAdapter(native, 0, clock=lambda: now, producer=p)
        measured.pcm = bytes(174240 * 2)
        return measured

    def args(self):
        return dict(
            start_ms=1680,
            end_ms=10890,
            window_id="w",
            mel=SimpleNamespace(shape=(80, 3000)),
        )

    def test_two_window_bound_stops_before_native_admission(self):
        measured = self.measured()
        measured.count = 2
        with self.assertRaisesRegex(RuntimeError, "native-window bound"):
            measured.start_window(**self.args())
        measured.native.start_window.assert_not_called()

    def test_cleanup_reserve_stops_at_75_seconds_before_native_admission(self):
        measured = self.measured(75_000_000_000)
        with self.assertRaisesRegex(TimeoutError, "cleanup reserve"):
            measured.start_window(**self.args())
        measured.native.start_window.assert_not_called()
        self.assertEqual(measured.count, 0)


class ReceiptTests(unittest.TestCase):
    """Synthetic observations exercise receipt validation, not decoder behavior."""

    @classmethod
    def setUpClass(cls):
        cls.source = p.source_case()

    def fixture(self, *, empty_head=False):
        from infra.noisy_context_prompt_worker import analyze

        source = self.source
        expected = dict(files=[dict(path=p.PRODUCER, sha256="producer-bytes")])
        cells = []
        for index, crop in enumerate(source["crops"], 1):
            raw = copy.deepcopy(source["archived_alignment"])
            native = raw["native"]
            native.update(
                start_ms=crop["start_ms"],
                end_ms=crop["end_ms"],
                analysis_span=dict(start_ms=crop["start_ms"], end_ms=crop["end_ms"]),
                window_id="synthetic-" + crop["id"],
            )
            if crop["id"] == "head-only":
                native["text"] = "" if empty_head else "after"
                native["metadata"].update(
                    tokens=[] if empty_head else [706],
                    segments=[],
                    timestamps_complete=False,
                )
                raw["words"] = (
                    []
                    if empty_head
                    else [
                        dict(
                            text=" after",
                            tokens=[706],
                            span=dict(start_ms=3800, end_ms=4100),
                        )
                    ]
                )
            has_tokens = any(
                token < p.TOKENIZER_EOT for token in native["metadata"]["tokens"]
            )
            attempts = [
                dict(
                    id="null" if attempt == 1 else "published-prefix",
                    prompt=None if attempt == 1 else source["prompt"],
                    decode_attempt=attempt,
                    decode_options={
                        **source["decode_options"],
                        "prompt": None if attempt == 1 else source["prompt"],
                    },
                    result=native,
                    alignment=raw,
                    session_version=0,
                    driver_steps=8,
                    encoder_forwards_so_far=1 + attempt * int(has_tokens),
                    **analyze(raw, source, crop),
                )
                for attempt in (1, 2)
            ]
            window = dict(
                call_index=index,
                window_id=native["window_id"],
                start_ms=crop["start_ms"],
                end_ms=crop["end_ms"],
                pcm_sha256=crop["pcm_sha256"],
                admitted_elapsed_ns=index,
                mel_shape=[80, 3000],
                encoder_calls=[dict(phase="decode", input_frames=3000, use_sdpa=None)]
                + [dict(phase="alignment", input_frames=3000, use_sdpa=False)]
                * (2 * int(has_tokens)),
                driver_steps=16,
                operation_wall_ns=dict(decode=200),
                closed=True,
                capacity_restored=True,
            )
            cells.append(
                dict(
                    **crop,
                    window=window,
                    attempts=attempts,
                    decode_attempt_count=2,
                    first_snapshot_unchanged=True,
                    same_lease=True,
                    closed=True,
                    capacity_restored=True,
                    session_version=0,
                    error=None,
                )
            )
        effective = source["effective_identity"]
        record = dict(
            schema_version="1-diagnostic",
            experiment_id=p.EXPERIMENT_ID,
            scope=p.scope(),
            qualified=False,
            claim_boundary=p.CLAIMS,
            source=dict(snapshot=expected, registration_sha256="producer-bytes"),
            input={key: value for key, value in source.items() if key != "pcm"},
            backend=dict(commit=p.BACKEND_COMMIT, tree=p.BACKEND_TREE),
            effective_identity={
                **effective,
                "reuse_decode_features": True,
                "tokenizer_eot": p.TOKENIZER_EOT,
            },
            model=dict(
                initial_sha256=effective["model_sha256"],
                final_sha256=effective["model_sha256"],
                unchanged=True,
                checkpoint_sha256=effective["checkpoint_sha256"],
            ),
            cells=cells,
            native_window_count=2,
            decode_attempt_count=4,
            capacity_restored=True,
            control=p.control(cells, source),
            status="completed",
            error=None,
            stop=None,
        )
        return p._canonical(record), expected

    def validate(self, record, expected):
        with patch.object(p, "source_case", return_value=self.source):
            p.validate_record(record, expected)

    def test_complete_json_receipt_replays_without_gpu(self):
        record, expected = self.fixture()
        self.validate(record, expected)
        self.assertEqual(
            sum(len(c["window"]["encoder_calls"]) for c in record["cells"]), 6
        )

    def test_empty_token_alignment_does_not_force_fictitious_encoder_work(self):
        record, expected = self.fixture(empty_head=True)
        self.validate(record, expected)
        self.assertEqual(len(record["cells"][1]["window"]["encoder_calls"]), 1)

    def test_mutated_schedule_pcm_counts_authority_or_cleanup_is_rejected(self):
        original, expected = self.fixture()
        mutations = (
            (("qualified",), True),
            (("input", "prompt"), "human reference substituted"),
            (("input", "anchor", 0, "tokens", 0), 999),
            (("cells", 0, "pcm_sha256"), "wrong"),
            (("cells", 0, "attempts", 1, "prompt"), None),
            (("cells", 0, "attempts", 1, "decode_options", "sample_len"), 96),
            (("cells", 0, "attempts", 0, "driver_steps"), 225),
            (("cells", 0, "window", "admitted_elapsed_ns"), 75_000_000_000),
            (("cells", 0, "window", "encoder_calls", 1, "use_sdpa"), True),
            (("cells", 0, "attempts", 1, "encoder_forwards_so_far"), 4),
            (("cells", 0, "first_snapshot_unchanged"), False),
            (("cells", 0, "same_lease"), False),
            (("cells", 0, "window", "capacity_restored"), False),
            (("cells", 1, "attempts", 1, "strict_candidate"), True),
            (("cells", 1, "attempts", 1, "publication_authorized"), True),
            (("cells", 1, "session_version"), 1),
            (("native_window_count",), 3),
            (("decode_attempt_count",), 5),
            (("control", "retained_null_alignment_equal"), False),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                record = copy.deepcopy(original)
                target = record
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    self.validate(record, expected)

    def test_failed_admission_keeps_started_attempt_and_explicit_failure(self):
        record, expected = self.fixture()
        cell = record["cells"][0]
        cell.update(
            window=None,
            attempts=[],
            decode_attempt_count=1,
            error={"category": "runtime"},
        )
        record.update(
            cells=[cell],
            native_window_count=0,
            decode_attempt_count=1,
            status="failed",
            error={"category": "runtime"},
            stop={"reason": "cell_error"},
        )
        record["control"] = p.control(record["cells"], self.source)
        self.validate(record, expected)
        record.update(error=None, stop=None)
        with self.assertRaisesRegex(ValueError, "explicit failure"):
            self.validate(record, expected)


if __name__ == "__main__":
    unittest.main()
