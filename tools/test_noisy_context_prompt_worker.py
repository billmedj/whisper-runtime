"""CPU-only checks for frozen observations; never load a model or launch work."""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import sys
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch

from infra import noisy_context_prompt_worker as worker
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeDecodeOptions, native_whisper
from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from whisper_runtime.adapters.word_policy import NativeWordAlignment


def _word(token, start, end, text):
    return NativeTimestampSegment(AudioSpan(start, end), text, (token,))


def _alignment(words, *, start=0, score=0.01):
    return NativeWordAlignment(
        NativeWindowResult(
            window_id="synthetic",
            text="".join(word.text for word in words).strip(),
            start_ms=start,
            end_ms=3000,
            metadata=NativeDecodeMetadata(
                language="en",
                tokens=tuple(token for word in words for token in word.tokens),
                no_speech_prob=score,
            ),
        ),
        words,
    )


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.anchor = (_word(1, 100, 500, " old"),)
        self.suffix = (_word(2, 1200, 1600, " next"),)
        self.current = _alignment(self.anchor + self.suffix)
        self.source = dict(
            pcm=b"\x01\x00" * 48000,
            anchor=[asdict(word) for word in self.anchor],
            head_sample=16000,
            prompt="old",
            config=dict(holdback_ms=2000, timestamp_tolerance_ms=200),
            reference_text="old next",
        )
        corpus = SimpleNamespace(
            b=SimpleNamespace(
                _word_difference=lambda text, reference: {
                    "word_edit_distance": int(text != reference)
                }
            )
        )
        self.corpus_patch = patch.object(worker.p, "_corpus", return_value=corpus)
        self.corpus_patch.start()
        self.addCleanup(self.corpus_patch.stop)

    def crop(self, *, name="retained", start=0):
        return dict(
            id=name,
            start_ms=start,
            end_ms=3000,
            start_sample=start * 16,
            end_sample=48000,
            pcm_sha256=worker.p.shared._sha(self.source["pcm"][start * 32 :]),
        )

    def analyze(self, alignment=None, *, crop=None, source=None):
        return worker.analyze(
            asdict(self.current if alignment is None else alignment),
            self.source if source is None else source,
            self.crop() if crop is None else crop,
        )

    def test_strict_retained_match_is_only_an_unpublished_candidate(self):
        before = copy.deepcopy(self.source)
        result = self.analyze()
        self.assertTrue(result["strict_candidate"])
        self.assertFalse(result["publication_authorized"])
        self.assertEqual(result["word_decision"]["reason"], "eof")
        self.assertEqual(result["audio_evidence"]["state"], "speech_candidate")
        self.assertEqual(result["recognition"]["hypothetical_full_text"], "old next")
        self.assertEqual(self.source, before)

    def test_human_reference_changes_scoring_not_selection(self):
        first = self.analyze()
        changed = self.analyze(
            source={**self.source, "reference_text": "unrelated oracle text"}
        )
        for name in (
            "strict_candidate",
            "publication_authorized",
            "word_decision",
            "audio_evidence",
        ):
            self.assertEqual(first[name], changed[name], name)
        self.assertEqual(
            first["recognition"]["hypothetical_full_text"],
            changed["recognition"]["hypothetical_full_text"],
        )
        self.assertNotEqual(
            first["recognition"]["against_human_reference"],
            changed["recognition"]["against_human_reference"],
        )

    def test_head_only_lexical_output_never_validates_the_join(self):
        result = self.analyze(
            _alignment(self.suffix, start=1000),
            crop=self.crop(name="head-only", start=1000),
        )
        self.assertEqual(result["word_decision"]["reason"], "eof")
        self.assertFalse(result["strict_candidate"])
        self.assertFalse(result["publication_authorized"])
        self.assertEqual(result["recognition"]["hypothetical_full_text"], "old next")

    def test_missing_anchor_preserves_refusal_and_has_no_reference_score(self):
        result = self.analyze(_alignment(self.suffix))
        self.assertEqual(result["word_decision"]["reason"], "anchor_missing")
        self.assertFalse(result["strict_candidate"])
        self.assertIsNone(result["recognition"]["hypothetical_full_text"])
        self.assertIsNone(result["recognition"]["against_human_reference"])

    def test_nonlexical_suffix_cannot_borrow_speech_from_lexical_anchor(self):
        punctuation = (_word(3, 1200, 1600, " ."),)
        result = self.analyze(_alignment(self.anchor + punctuation))
        self.assertIsNotNone(result["word_decision"]["publication"])
        self.assertEqual(result["audio_evidence"]["state"], "uncertain")
        self.assertEqual(result["audio_evidence"]["reason"], "no_lexical_text")
        self.assertFalse(result["strict_candidate"])

    def test_conflicting_speech_score_cannot_authorize_a_strict_match(self):
        result = self.analyze(_alignment(self.anchor + self.suffix, score=0.8))
        self.assertFalse(result["strict_candidate"])
        self.assertFalse(result["publication_authorized"])
        self.assertEqual(result["audio_evidence"]["state"], "uncertain")

    def test_pcm_and_span_mismatches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "PCM differs"):
            self.analyze(crop={**self.crop(), "pcm_sha256": "wrong"})
        changed = replace(
            self.current,
            native=replace(self.current.native, end_ms=3001),
        )
        with self.assertRaisesRegex(ValueError, "span differs"):
            self.analyze(changed)


class DriveTests(unittest.TestCase):
    @staticmethod
    def fake_run(*, count=40, required=2):
        run = SimpleNamespace(step_count=count, complete=required == 0)

        def step():
            run.step_count += 1
            run.complete = run.step_count >= count + required

        run.step = step
        return run

    def test_counts_each_attempt_from_its_existing_step_count(self):
        run = self.fake_run()
        with patch.object(worker.time, "perf_counter_ns", return_value=1):
            self.assertEqual(worker.drive(run, 0), 2)
        self.assertEqual(run.step_count, 42)

    def test_step_bound_stops_before_excess_work(self):
        run = self.fake_run(required=worker.p.MAX_DRIVER_STEPS + 1)
        with (
            patch.object(worker.time, "perf_counter_ns", return_value=1),
            self.assertRaisesRegex(RuntimeError, "decoder-step bound"),
        ):
            worker.drive(run, 0)
        self.assertEqual(run.step_count, 40 + worker.p.MAX_DRIVER_STEPS)

    def test_cleanup_reserve_stops_before_the_next_step(self):
        run = self.fake_run()
        deadline = (
            worker.p.GPU_TIMEOUT_SECONDS - worker.p.CLEANUP_RESERVE_SECONDS
        ) * 1_000_000_000
        with (
            patch.object(worker.time, "perf_counter_ns", return_value=deadline),
            self.assertRaisesRegex(TimeoutError, "cleanup reserve"),
        ):
            worker.drive(run, 0)
        self.assertEqual(run.step_count, 40)

    def test_completed_prefill_needs_no_driver_step(self):
        run = self.fake_run(required=0)
        with patch.object(worker.time, "perf_counter_ns") as clock:
            self.assertEqual(worker.drive(run, 0), 0)
        clock.assert_not_called()


class NativeObservationSequenceTests(unittest.TestCase):
    def test_two_windows_keep_four_snapshots_without_publication(self):
        # Reuse the project's dependency-free native backend fixtures. They
        # model the owned encoder dispatch, not numerical model inference.
        with patch.object(sys, "path", [str(worker.p.ROOT / "tests"), *sys.path]):
            fixtures = importlib.import_module("test_native_prompt_reuse")
            alignment_fixtures = importlib.import_module("test_native_word_alignment")
        encoder_count = 0
        for _ in range(2):
            case = fixtures.NativePromptReuseTests()
            case.setUp()
            calls = []

            def legacy_finder(*args):
                calls.append(args)
                return case.finder(*args)

            try:
                raw = case.start(complete=False, options=NativeDecodeOptions())
                measured = worker._MeasuredAdapter(
                    case.adapter, 0, clock=lambda: 1, producer=worker.p
                )
                record = dict(operation_wall_ns={}, driver_steps=0)
                facade_module = importlib.import_module(
                    "infra.eof_context_retry_worker"
                )
                run = facade_module._MeasuredRun(measured, raw, record)
                with (
                    patch.object(worker.time, "perf_counter_ns", return_value=1),
                    patch.object(
                        native_whisper,
                        "import_module",
                        side_effect=alignment_fixtures.ImportHarness(legacy_finder),
                    ),
                    patch.object(raw, "finish", side_effect=AssertionError("publish")),
                ):
                    transaction = run._transaction
                    self.assertEqual(worker.drive(run, 0), 1)
                    a = run.prepare_result()
                    wa = measured.invoke(
                        record,
                        "alignment",
                        run.raw.prepare_word_alignment,
                        reuse_alignment_features=False,
                    )
                    snapshot = asdict(wa)
                    measured.invoke(
                        record, "redecode", run.redecode, prompt="published prefix"
                    )
                    self.assertIsNone(raw._prepared_alignment)
                    self.assertEqual(worker.drive(run, 0), 1)
                    b = run.prepare_result()
                    wb = measured.invoke(
                        record,
                        "alignment",
                        run.raw.prepare_word_alignment,
                        reuse_alignment_features=False,
                    )
                    self.assertIsNot(a, b)
                    self.assertIsNot(wa, wb)
                    self.assertIs(wa.native, a)
                    self.assertIs(wb.native, b)
                    self.assertEqual(asdict(wa), snapshot)
                    self.assertIs(run._transaction, transaction)
                    self.assertIsNone(measured.active)
                    self.assertEqual(record["driver_steps"], 2)
                    self.assertEqual(case.session.snapshot().version, 0)
                    run.close()
                    case.assert_released(raw)
                    self.assertTrue(record["closed"])
                    self.assertTrue(record["capacity_restored"])
                    self.assertEqual(case.first.finalize_calls, 1)
                    self.assertEqual(case.second.finalize_calls, 1)
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(case.harness.encoder_calls, 1)
                    encoder_count += case.harness.encoder_calls + len(calls)
            finally:
                case.doCleanups()
        self.assertEqual(encoder_count, 6)

    def test_worker_source_has_no_finish_or_stream_commit_call(self):
        tree = ast.parse(inspect.getsource(worker.run_worker))
        invoked = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(invoked & {"finish", "commit", "decode_window"})


if __name__ == "__main__":
    unittest.main()
