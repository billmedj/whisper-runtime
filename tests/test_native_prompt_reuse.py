"""Bounded prompt replacement preserves one transaction and its feature owner."""

import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import test_native_adapter as fixtures
import test_native_word_alignment as alignment

from whisper_runtime import RequestCancelledError, RequestStatus, Session
from whisper_runtime.adapters import (
    NativeDecodeContractError,
    NativeDecodeOptions,
    NativeExecutionProfile,
    NativeWhisperAdapter,
    native_whisper,
)
from whisper_runtime.adapters.audio_evidence import AudioObservation, SilencePublication
from whisper_runtime.adapters.word_policy import select_word_publication


class Features(alignment.AudioFeatures):
    def __init__(self, *, device=None):
        super().__init__(device=device)
        self.batch = SimpleNamespace(source=self, shape=(1, 1_500, 4))
        self.unsqueeze_calls = []

    def unsqueeze(self, dimension):
        self.unsqueeze_calls.append(dimension)
        return self.batch


class ReuseHarness(alignment.Harness):
    def __init__(self, runs, *, features, torch_module):
        super().__init__(runs, torch_module=torch_module)
        self.features = features
        self.encoder_calls = 0
        self.tasks = []
        self.fail_build = False

    def task_type(self, model, options):
        if self.fail_build:
            raise RuntimeError("task build failed")
        task = super().task_type(model, options)
        original_start = task._start_run

        def start(value):
            # Model the native backend's pre-encoded-input dispatch. Identity
            # assertions below ensure the adapter sends the owned tensor.
            if value is not self.features.batch:
                self.encoder_calls += 1
            return original_start(value)

        task._start_run = start
        self.tasks.append(task)
        return task


class NativePromptReuseTests(unittest.TestCase):
    request = fixtures.NativeWhisperAdapterTests.request

    def setUp(self):
        fixtures.NativeWhisperAdapterTests.setUp(self)
        self.configure()

    def configure(self, *, enabled=True, cuda=False):
        self.events = []
        self.runtime = fixtures.FakeCudaRuntime(self.events)
        device = fixtures.FakeDevice("cuda", 1) if cuda else fixtures.FakeDevice("cpu")
        self.model = fixtures.FakeNativeModel(self.identity, device=device)
        self.model.dims.n_audio_state = 4
        self.adapter = NativeWhisperAdapter(
            self.worker,
            self.model,
            fixtures.probe,
            NativeExecutionProfile(
                "prompt-reuse",
                self.capacity,
                device="cuda:1" if cuda else "cpu",
                reuse_decode_features=enabled,
                reuse_alignment_features=True,
            ),
        )
        self.features = Features(device=device)
        self.first = alignment.ResultRun(
            alignment.raw_result(audio_features=self.features)
        )
        self.second = alignment.ResultRun(
            alignment.raw_result(text="world", tokens=[2], audio_features=self.features)
        )
        self.harness = ReuseHarness(
            [self.first, self.second],
            features=self.features,
            torch_module=fixtures.FakeTorchModule(self.runtime),
        )
        self.mel = (
            alignment.AlignableCudaMel(self.runtime, self.events)
            if cuda
            else alignment.AlignableMel()
        )
        self.session = Session("session-1")
        self.native_request = self.request()

    def start(self, *, complete=True, options=None):
        with patch.object(
            native_whisper,
            "_load_native_components",
            return_value=self.harness.components(),
        ):
            run = self.adapter.start_window(
                session=self.session,
                request=self.native_request,
                window_id="window-1",
                mel=self.mel,
                start_ms=0,
                end_ms=1_000,
                options=options or NativeDecodeOptions(prompt="original"),
            )
        self.addCleanup(run.close)
        if complete:
            self.assertTrue(run.step())
        return run

    def assert_released(self, run, *, published=False):
        self.assertTrue(run.closed)
        self.assertTrue(run.capacity_released)
        self.assertEqual(self.budget.available, self.capacity)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.session.snapshot().version, int(published))
        self.assertIsNone(run._alignment_audio_features)
        self.assertIsNone(run._backend_run)

    @staticmethod
    def finder(model, tokenizer, tokens, mel, frames, *, audio_features=None):
        del model, tokenizer, mel, frames, audio_features
        if tokens == [2]:
            return [alignment.timing(" world", [2], 0.4, 0.9)]
        return alignment.NativeAlignmentFeatureReuseTests.words()

    def test_profile_is_explicit_and_strictly_boolean(self):
        self.assertFalse(
            NativeExecutionProfile("default", self.capacity).reuse_decode_features
        )
        for value in (0, 1, "true", None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(TypeError, "boolean"),
            ):
                NativeExecutionProfile(
                    "bad", self.capacity, reuse_decode_features=value
                )

    def test_reuses_exact_features_but_fresh_task_generator_and_same_options(self):
        options = NativeDecodeOptions(
            prompt="original",
            language="en",
            temperature=0.4,
            sample_len=12,
            prefix=(1,),
            suppress_tokens=(4,),
            without_timestamps=True,
        )
        run = self.start(options=options)
        transaction = run._transaction
        before_budget = self.budget.available
        snapshot = run.prepare_result()
        self.assertEqual(run.decode_attempt, 1)
        run.redecode(prompt="replacement")
        self.assertEqual(run.decode_attempt, 2)
        self.assertIs(run._transaction, transaction)
        self.assertEqual(self.budget.available, before_budget)
        self.assertEqual(self.harness.encoder_calls, 1)
        self.assertIs(self.harness.batched_mels[1], self.features.batch)
        self.assertEqual(self.features.unsqueeze_calls, [0])
        self.assertEqual(self.mel.unsqueeze_calls, [0])
        self.assertIsNot(self.harness.tasks[0], self.harness.tasks[1])
        first_generator, second_generator = self.harness.generators
        self.assertIsNot(first_generator, second_generator)
        self.assertEqual(first_generator.seed, second_generator.seed)
        self.assertEqual(first_generator.device, second_generator.device)
        old_options, new_options = self.harness.option_kwargs
        for key in old_options.keys() - {"prompt", "generator"}:
            self.assertEqual(old_options[key], new_options[key], key)
        self.assertEqual(new_options["prompt"], "replacement")
        self.assertIsNone(run._prepared_result)
        self.assertFalse(run.complete)
        self.assertEqual(self.second.prefill_calls, 1)
        self.assertEqual(snapshot.text, "hello world")
        with self.assertRaises(FrozenInstanceError):
            snapshot.text = "changed"

    def test_finish_publishes_only_replacement_and_keeps_first_snapshot(self):
        run = self.start()
        original = run.prepare_result()
        run.redecode(prompt=None)
        self.assertEqual(self.session.snapshot().version, 0)
        with self.assertRaisesRegex(NativeDecodeContractError, "before token"):
            run.finish()
        run.step()
        replacement = run.prepare_result()
        self.assertIs(run.prepare_result(), replacement)
        self.assertIsNot(replacement, original)
        state = run.finish()
        self.assertIs(state.windows[-1].result, replacement)
        self.assertEqual(replacement.text, "world")
        self.assertEqual(original.text, "hello world")
        self.assertEqual(original.metadata.tokens, (1, 2))
        self.assertEqual(run.step_count, 2)
        self.assertEqual(self.first.finalize_calls, 1)
        self.assertEqual(self.second.finalize_calls, 1)
        self.assertEqual((self.first.cleanup_calls, self.second.cleanup_calls), (1, 1))
        self.assertEqual(self.native_request.status, RequestStatus.COMMITTED)
        self.assert_released(run, published=True)

    def test_alignment_is_invalidated_recomputed_and_old_publication_rejected(self):
        run = self.start()
        with patch.object(
            native_whisper,
            "import_module",
            side_effect=alignment.ImportHarness(self.finder),
        ):
            first_alignment = run.prepare_word_alignment()
            old_publication = select_word_publication(
                first_alignment, 0, 2, first_alignment.native.analyzed_span, final=True
            )
            run.redecode(prompt="new")
            self.assertIsNone(run._prepared_alignment)
            run.step()
            second_alignment = run.prepare_word_alignment()
            self.assertIsNot(second_alignment, first_alignment)
            self.assertEqual(first_alignment.native.text, "hello world")
            self.assertEqual(second_alignment.native.text, "world")
            with self.assertRaisesRegex(ValueError, "cached word alignment"):
                run.finish(aligned_publication=old_publication)
            self.assertFalse(run.closed)
            current = select_word_publication(
                second_alignment,
                0,
                1,
                second_alignment.native.analyzed_span,
                final=True,
            )
            run.finish(aligned_publication=current)
        self.assert_released(run, published=True)

    def test_old_silence_publication_cannot_publish_after_redecode(self):
        run = self.start()
        first = run.prepare_result()
        publication = SilencePublication(
            window_id=first.window_id,
            text="",
            start_ms=0,
            end_ms=1_000,
            analysis_span=first.analyzed_span,
            native=first,
            observation=AudioObservation.from_pcm(bytes(32_000)),
            analysis_start_sample=0,
        )
        run.redecode(prompt="new")
        run.step()
        run.prepare_result()
        with self.assertRaisesRegex(ValueError, "cached native result"):
            run.finish(silence_publication=publication)
        self.assertEqual(self.session.snapshot().version, 0)
        run.finish()
        self.assert_released(run, published=True)

    def test_opt_out_rejects_without_touching_original_candidate(self):
        self.configure(enabled=False)
        run = self.start()
        with self.assertRaisesRegex(ValueError, "opted-in"):
            run.redecode(prompt="new")
        self.assertEqual(self.first.finalize_calls, 0)
        self.assertEqual(len(self.harness.tasks), 1)
        self.assertEqual(run.decode_attempt, 1)
        self.assertEqual(run.finish().windows[-1].result.text, "hello world")

    def test_unchanged_and_invalid_prompt_leave_attempt_available(self):
        run = self.start()
        with self.assertRaisesRegex(ValueError, "must differ"):
            run.redecode(prompt="original")
        for value in (42, [1], (True,)):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                run.redecode(prompt=value)
        self.assertEqual(run.decode_attempt, 1)
        self.assertEqual(self.first.finalize_calls, 0)
        run.redecode(prompt=(2,))
        self.assertEqual(self.harness.option_kwargs[-1]["prompt"], (2,))

    def test_incomplete_and_closed_runs_cannot_redecode(self):
        run = self.start(complete=False)
        with self.assertRaisesRegex(NativeDecodeContractError, "completed decode"):
            run.redecode(prompt="new")
        self.assertFalse(run.closed)
        run.close()
        with self.assertRaisesRegex(NativeDecodeContractError, "closed"):
            run.redecode(prompt="new")
        self.assert_released(run)

    def test_attempt_limit_is_one_even_when_second_candidate_is_complete(self):
        run = self.start()
        run.redecode(prompt="new")
        run.step()
        with self.assertRaisesRegex(NativeDecodeContractError, "only one"):
            run.redecode(prompt="third")
        self.assertEqual(len(self.harness.tasks), 2)
        self.assertEqual(run.finish().windows[-1].result.text, "world")

    def test_features_must_have_owned_shape_precision_device_and_batching(self):
        for invalid in ("missing", "shape", "dtype", "device", "batching"):
            with self.subTest(invalid=invalid):
                self.setUp()
                if invalid == "missing":
                    self.first.result.audio_features = None
                elif invalid == "shape":
                    self.features.shape = (1, 1_500, 4)
                elif invalid == "dtype":
                    self.features.dtype = object()
                elif invalid == "device":
                    self.features.device = fixtures.FakeDevice("cuda", 0)
                else:
                    self.features.unsqueeze = None
                run = self.start()
                with self.assertRaisesRegex(
                    NativeDecodeContractError, "audio_features"
                ):
                    run.redecode(prompt="new")
                self.assertEqual(len(self.harness.tasks), 1)
                self.assertEqual(self.second.prefill_calls, 0)
                self.assert_released(run)

    def test_task_build_start_and_prefill_failures_do_not_publish_or_leak(self):
        for stage in ("build", "start", "prefill"):
            with self.subTest(stage=stage):
                self.setUp()
                run = self.start()
                if stage == "build":
                    self.harness.fail_build = True
                else:
                    self.second.fail_stage = stage
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    run.redecode(prompt="new")
                self.assertGreaterEqual(self.first.cleanup_calls, 1)
                self.assertEqual(self.second.cleanup_calls, int(stage == "prefill"))
                self.assertEqual(self.native_request.status, RequestStatus.ABORTED)
                self.assert_released(run)

    def test_finalize_failure_of_either_attempt_never_publishes(self):
        for second in (False, True):
            with self.subTest(second=second):
                self.setUp()
                run = self.start()
                if second:
                    run.redecode(prompt="new")
                    run.step()
                    self.second.fail_stage = "finalize"
                else:
                    self.first.fail_stage = "finalize"
                with self.assertRaisesRegex(RuntimeError, "finalize failed"):
                    run.finish() if second else run.redecode(prompt="new")
                self.assert_released(run)

    def test_old_cleanup_failure_retains_owner_until_recovery(self):
        run = self.start()
        run.prepare_result()
        self.first.fail_cleanup = True
        with self.assertRaises(fixtures.TransactionRetainedError):
            run.redecode(prompt="new")
        self.assertFalse(run.capacity_released)
        self.assertIs(run._execution._run, self.first)
        self.assertIs(run._alignment_audio_features, self.features)
        self.assertEqual(self.second.prefill_calls, 0)
        self.assertEqual(len(self.harness.batched_mels), 1)
        self.first.fail_cleanup = False
        self.assertTrue(run.stop())
        self.assert_released(run)

    def test_new_cleanup_failure_retains_replacement_until_recovery(self):
        run = self.start()
        run.redecode(prompt="new")
        self.second.fail_cleanup = True
        with self.assertRaises(fixtures.TransactionRetainedError):
            run.close()
        self.assertFalse(run.capacity_released)
        self.assertIs(run._execution._run, self.second)
        self.assertEqual(self.first.cleanup_calls, 1)
        self.second.fail_cleanup = False
        self.assertTrue(run.stop())
        self.assert_released(run)

    def test_cancel_before_redecode_never_starts_replacement(self):
        run = self.start()
        run.cancel()
        with self.assertRaises(RequestCancelledError):
            run.redecode(prompt="new")
        self.assertEqual(len(self.harness.tasks), 1)
        self.assertEqual(self.second.cleanup_calls, 0)
        self.assert_released(run)

    def test_cancel_during_child_creation_cleans_registered_child(self):
        run = self.start()
        self.harness.on_start = run.cancel
        with self.assertRaises(RequestCancelledError):
            run.redecode(prompt="new")
        self.assertEqual(self.second.prefill_calls, 0)
        self.assertEqual((self.first.cleanup_calls, self.second.cleanup_calls), (1, 1))
        self.assert_released(run)

    def test_cancel_during_child_prefill_cleans_registered_child(self):
        run = self.start()
        self.second.on_prefill = run.cancel
        with self.assertRaises(RequestCancelledError):
            run.redecode(prompt="new")
        self.assertEqual(self.second.prefill_calls, 1)
        self.assertEqual((self.first.cleanup_calls, self.second.cleanup_calls), (1, 1))
        self.assert_released(run)

    def test_cancel_after_handoff_closes_only_current_owner(self):
        run = self.start()
        run.redecode(prompt="new")
        run.cancel()
        with self.assertRaises(RequestCancelledError):
            run.step()
        self.assertEqual(self.second.step_calls, 0)
        self.assertEqual((self.first.cleanup_calls, self.second.cleanup_calls), (1, 1))
        self.assert_released(run)

    def test_cuda_handoff_uses_same_stream_without_second_audio_copy(self):
        self.configure(cuda=True)
        run = self.start()
        streams = []
        self.first.on_cleanup = lambda: streams.append(self.runtime.active_stream)
        self.second.on_prefill = lambda: streams.append(self.runtime.active_stream)
        self.second.on_step = lambda: streams.append(self.runtime.active_stream)
        self.second.on_cleanup = lambda: streams.append(self.runtime.active_stream)
        run.redecode(prompt="new")
        self.assertNotIn("event:synchronize", self.events)
        run.step()
        run.finish()
        self.assertEqual(len(self.runtime.streams), 1)
        self.assertEqual(streams, [self.runtime.streams[0]] * 4)
        self.assertEqual(self.mel.copy_arguments, [("cuda:1", False)])
        self.assertEqual(self.harness.encoder_calls, 1)
        self.assertEqual(self.events.count("event:synchronize"), 1)
        self.assert_released(run, published=True)

    def test_cuda_failed_fence_keeps_replacement_until_recovery(self):
        self.configure(cuda=True)
        run = self.start()
        run.redecode(prompt="new")
        run.step()
        run.prepare_result()
        self.runtime.fail_event_synchronize = True
        with self.assertRaises(fixtures.TransactionRetainedError):
            run.close()
        self.assertFalse(run.capacity_released)
        self.assertIs(run._backend_run, self.second)
        self.assertIs(run._alignment_audio_features, self.features)
        self.runtime.fail_event_synchronize = False
        self.assertTrue(run.stop())
        self.assertEqual(self.first.cleanup_calls, 1)
        self.assert_released(run)


if __name__ == "__main__":
    unittest.main()
