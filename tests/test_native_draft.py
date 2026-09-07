"""Draft hints remain opt-in and obey native run ownership without PyTorch."""

import unittest
from unittest.mock import patch

import test_native_adapter as fixtures
import test_native_prompt_reuse as reuse

from whisper_runtime import RequestCancelledError, RequestStatus, Session
from whisper_runtime.adapters import (
    NativeDecodeContractError,
    NativeDecodeOptions,
    NativeDependencyError,
    NativeExecutionProfile,
    NativeWhisperAdapter,
    native_whisper,
)

WRAPPER = "whisper_runtime.adapters._draft_inference.VerifiedDraftInference"
OMITTED = object()


class WrapperSpy:
    """Represent the wrapper while leaving inference semantics to its unit tests."""

    _use_legacy_cache = False

    def __init__(self, inference, draft_tokens):
        self.inner = inference
        self.draft_tokens = draft_tokens


class TrackingHarness(fixtures.BackendHarness):
    def __init__(self, runs, **kwargs):
        super().__init__(runs, **kwargs)
        self.tasks = []

    def task_type(self, model, options):
        task = super().task_type(model, options)
        self.tasks.append(task)
        return task


class NativeDraftTests(unittest.TestCase):
    request = fixtures.NativeWhisperAdapterTests.request

    def setUp(self):
        fixtures.NativeWhisperAdapterTests.setUp(self)
        self.session = Session("session-1")
        self.native_request = self.request()
        self.backend = fixtures.FakeRun(complete_after=1)
        self.harness = TrackingHarness([self.backend])
        self.mel = fixtures.FakeMel()

    def start(self, *, draft_tokens=OMITTED, options=None):
        kwargs = {} if draft_tokens is OMITTED else {"draft_tokens": draft_tokens}
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
                options=options,
                **kwargs,
            )
        self.addCleanup(run.close)
        return run

    def assert_released(self, *, published=False):
        self.assertEqual(self.budget.available, self.capacity)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.session.snapshot().version, int(published))

    def assert_rejected_before_admission(self, draft_tokens, options=None):
        with (
            patch.object(native_whisper, "_load_native_components") as load,
            patch.object(self.worker, "prepare") as prepare,
            self.assertRaises((TypeError, ValueError)),
        ):
            self.adapter.start_window(
                session=self.session,
                request=self.native_request,
                window_id="window-1",
                mel=self.mel,
                start_ms=0,
                end_ms=1_000,
                options=options,
                draft_tokens=draft_tokens,
            )
        load.assert_not_called()
        prepare.assert_not_called()
        self.assertEqual(self.native_request.status, RequestStatus.CREATED)
        self.assertEqual(self.mel.unsqueeze_calls, [])
        self.assert_released()

    def test_hint_validation_precedes_loading_and_admission(self):
        class IntegerSubclass(int):
            pass

        invalid = (
            None,
            [],
            [1],
            "1",
            1,
            (True,),
            (False,),
            (1.0,),
            ("1",),
            (-1,),
            (IntegerSubclass(1),),
            tuple(range(33)),
        )
        for value in invalid:
            with self.subTest(value=value):
                self.assert_rejected_before_admission(value)

    def test_nonempty_hint_rejects_nongreedy_modes_before_admission(self):
        for options in (
            NativeDecodeOptions(beam_size=1),
            NativeDecodeOptions(beam_size=3),
            NativeDecodeOptions(temperature=0.1),
            NativeDecodeOptions(temperature=0.4, best_of=2),
        ):
            with self.subTest(options=options):
                self.assert_rejected_before_admission((1,), options)

    def test_omitted_and_empty_hint_leave_ordinary_inference_unchanged(self):
        for hint in (OMITTED, ()):
            for options in (
                NativeDecodeOptions(),
                NativeDecodeOptions(beam_size=2),
                NativeDecodeOptions(temperature=0.4, best_of=2),
            ):
                with self.subTest(hint=hint, options=options):
                    self.setUp()
                    # A default CPU profile does not impose isolation unless
                    # another feature, including a nonempty draft, requires it.
                    self.harness.use_legacy_cache = True
                    self.harness.use_legacy_extension = True
                    original = self.backend.inference
                    original._use_legacy_cache = True
                    with patch(WRAPPER, side_effect=WrapperSpy) as wrap:
                        run = self.start(draft_tokens=hint, options=options)
                    wrap.assert_not_called()
                    self.assertIs(self.backend.inference, original)
                    self.assertEqual(self.backend.prefill_calls, 1)
                    self.assertTrue(run.step())
                    self.assertEqual(
                        run.finish().windows[-1].result.text, " decoded text"
                    )
                    self.assertEqual(self.backend.cleanup_calls, 1)
                    self.assert_released(published=True)

    def test_zero_id_and_full_32_token_hint_are_accepted_without_option_changes(self):
        hint = tuple(range(32))
        options = NativeDecodeOptions(prompt=(9,), prefix=(8,), suppress_tokens=(7,))
        original = self.backend.inference
        with patch(WRAPPER, side_effect=WrapperSpy) as wrap:
            run = self.start(draft_tokens=hint, options=options)
        wrap.assert_called_once_with(original, hint)
        self.assertIsInstance(self.backend.inference, WrapperSpy)
        self.assertIs(self.backend.inference.inner, original)
        self.assertIs(self.backend.inference.draft_tokens, hint)
        self.assertIsNot(self.harness.tasks[0].inference, self.backend.inference)
        self.assertEqual(self.harness.option_kwargs[0]["prompt"], (9,))
        self.assertEqual(self.harness.option_kwargs[0]["prefix"], (8,))
        self.assertEqual(self.harness.option_kwargs[0]["suppress_tokens"], (7,))
        self.assertNotIn("draft_tokens", self.harness.option_kwargs[0])
        run.step()
        run.finish()
        self.assert_released(published=True)

    def test_wrapper_is_installed_after_binding_and_isolation_before_prefill(self):
        events = []
        original = self.backend.inference
        bind = native_whisper._CpuDecodeScope.bind
        isolated = native_whisper._require_isolated_run

        def tracked_bind(scope, backend):
            bind(scope, backend)
            self.assertIs(scope._run, self.backend)
            events.append("bind")

        def tracked_isolation(backend):
            self.assertIs(backend.inference, original)
            isolated(backend)
            events.append("isolated")

        def wrap(inference, hint):
            self.assertEqual(events, ["bind", "isolated"])
            self.assertEqual(self.worker.queue_depth, 1)
            self.assertNotEqual(self.budget.available, self.capacity)
            events.append("wrap")
            return WrapperSpy(inference, hint)

        def prefill():
            self.assertIsInstance(self.backend.inference, WrapperSpy)
            events.append("prefill")

        self.backend.on_prefill = prefill
        with (
            patch.object(native_whisper._CpuDecodeScope, "bind", new=tracked_bind),
            patch.object(
                native_whisper, "_require_isolated_run", new=tracked_isolation
            ),
            patch(WRAPPER, side_effect=wrap),
        ):
            run = self.start(draft_tokens=(1, 2))
        self.assertEqual(events, ["bind", "isolated", "wrap", "prefill"])
        run.close()
        self.assertEqual(self.backend.cleanup_calls, 1)
        self.assert_released()

    def test_nonisolated_task_is_rejected_before_start_or_wrapper(self):
        for attribute in ("use_legacy_cache", "use_legacy_extension"):
            with self.subTest(attribute=attribute):
                self.setUp()
                setattr(self.harness, attribute, True)
                with (
                    patch(WRAPPER, side_effect=WrapperSpy) as wrap,
                    self.assertRaises(NativeDependencyError),
                ):
                    self.start(draft_tokens=(1,))
                wrap.assert_not_called()
                self.assertEqual(self.harness.batched_mels, [])
                self.assertEqual(self.backend.prefill_calls, 0)
                self.assertEqual(self.backend.cleanup_calls, 0)
                self.assert_released()

    def test_nonisolated_owned_run_is_cleaned_without_wrapping(self):
        for violation in ("legacy_cache", "legacy_lock"):
            with self.subTest(violation=violation):
                self.setUp()
                if violation == "legacy_cache":
                    self.backend.inference._use_legacy_cache = True
                else:
                    self.backend._legacy_cache_lock = object()
                original = self.backend.inference
                with (
                    patch(WRAPPER, side_effect=WrapperSpy) as wrap,
                    self.assertRaises(NativeDecodeContractError),
                ):
                    self.start(draft_tokens=(1,))
                wrap.assert_not_called()
                self.assertIs(self.backend.inference, original)
                self.assertEqual(self.backend.prefill_calls, 0)
                self.assertEqual(self.backend.cleanup_calls, 1)
                self.assertEqual(self.native_request.status, RequestStatus.ABORTED)
                self.assert_released()

    def test_wrapper_constructor_failure_cleans_owned_run_and_releases_lease(self):
        original = self.backend.inference
        with (
            patch(WRAPPER, side_effect=RuntimeError("draft constructor failed")),
            self.assertRaisesRegex(RuntimeError, "draft constructor failed"),
        ):
            self.start(draft_tokens=(1,))
        self.assertIs(self.backend.inference, original)
        self.assertEqual(self.backend.prefill_calls, 0)
        self.assertEqual(self.backend.cleanup_calls, 1)
        self.assertEqual(self.native_request.status, RequestStatus.ABORTED)
        self.assert_released()

    def test_prefill_failure_cleans_wrapped_owner_and_releases_lease(self):
        self.backend.fail_stage = "prefill"
        cleaned = []
        self.backend.on_cleanup = lambda: cleaned.append(self.backend.inference)
        with (
            patch(WRAPPER, side_effect=WrapperSpy),
            self.assertRaisesRegex(RuntimeError, "prefill failed"),
        ):
            self.start(draft_tokens=(1,))
        self.assertEqual(self.backend.prefill_calls, 1)
        self.assertEqual(self.backend.cleanup_calls, 1)
        self.assertEqual(cleaned, [self.backend.inference])
        self.assertIsInstance(cleaned[0], WrapperSpy)
        self.assertEqual(self.native_request.status, RequestStatus.ABORTED)
        self.assert_released()

    def test_cancel_during_wrapper_construction_cleans_before_prefill(self):
        def wrap(inference, hint):
            self.native_request.cancel()
            return WrapperSpy(inference, hint)

        with (
            patch(WRAPPER, side_effect=wrap),
            self.assertRaises(RequestCancelledError),
        ):
            self.start(draft_tokens=(1,))
        self.assertEqual(self.backend.prefill_calls, 0)
        self.assertEqual(self.backend.cleanup_calls, 1)
        self.assertEqual(self.native_request.status, RequestStatus.CANCELLED)
        self.assert_released()

    def test_cancel_during_prefill_cleans_wrapped_owner(self):
        self.backend.on_prefill = self.native_request.cancel
        with (
            patch(WRAPPER, side_effect=WrapperSpy),
            self.assertRaises(RequestCancelledError),
        ):
            self.start(draft_tokens=(1,))
        self.assertEqual(self.backend.prefill_calls, 1)
        self.assertEqual(self.backend.cleanup_calls, 1)
        self.assertEqual(self.native_request.status, RequestStatus.CANCELLED)
        self.assert_released()

    def test_cancel_or_close_after_start_cleans_once_without_publication(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                self.setUp()
                with patch(WRAPPER, side_effect=WrapperSpy):
                    run = self.start(draft_tokens=(1,))
                if cancel:
                    self.assertTrue(run.cancel())
                    with self.assertRaises(RequestCancelledError):
                        run.step()
                else:
                    run.close()
                run.close()
                self.assertTrue(run.closed)
                self.assertTrue(run.capacity_released)
                self.assertEqual(self.backend.step_calls, 0)
                self.assertEqual(self.backend.cleanup_calls, 1)
                self.assert_released()

    def test_fake_cuda_wrapper_and_prefill_share_owned_device_and_stream(self):
        events = []
        runtime = fixtures.FakeCudaRuntime(events)
        self.model = fixtures.FakeNativeModel(
            self.identity, device=fixtures.FakeDevice("cuda", 1)
        )
        self.adapter = NativeWhisperAdapter(
            self.worker,
            self.model,
            fixtures.probe,
            NativeExecutionProfile("draft-cuda", self.capacity, device="cuda:1"),
        )
        self.mel = fixtures.FakeCudaMel(runtime, events)
        self.harness.torch_module = fixtures.FakeTorchModule(runtime)
        scopes = []

        def capture_scope():
            scopes.append((runtime.active_device, runtime.active_stream))

        def wrap(inference, hint):
            capture_scope()
            return WrapperSpy(inference, hint)

        self.backend.on_prefill = capture_scope
        self.backend.on_cleanup = capture_scope
        with patch(WRAPPER, side_effect=wrap):
            run = self.start(draft_tokens=(1,))
        run.close()
        self.assertEqual(len(runtime.streams), 1)
        self.assertEqual(scopes, [("cuda:1", runtime.streams[0])] * 3)
        self.assertEqual(self.backend.cleanup_calls, 1)
        self.assertEqual(events.count("event:synchronize"), 1)
        self.assert_released()


class NativeDraftPromptReuseTests(unittest.TestCase):
    request = fixtures.NativeWhisperAdapterTests.request
    configure = reuse.NativePromptReuseTests.configure
    assert_released = reuse.NativePromptReuseTests.assert_released

    def setUp(self):
        reuse.NativePromptReuseTests.setUp(self)

    def test_redecode_has_fresh_ordinary_inference_and_no_replayed_hint(self):
        first_inference = self.first.inference
        second_inference = self.second.inference
        with (
            patch.object(
                native_whisper,
                "_load_native_components",
                return_value=self.harness.components(),
            ),
            patch(WRAPPER, side_effect=WrapperSpy) as wrap,
        ):
            run = self.adapter.start_window(
                session=self.session,
                request=self.native_request,
                window_id="window-1",
                mel=self.mel,
                start_ms=0,
                end_ms=1_000,
                options=NativeDecodeOptions(prompt="original"),
                draft_tokens=(1, 2),
            )
            self.addCleanup(run.close)
            self.assertIsInstance(self.first.inference, WrapperSpy)
            self.assertTrue(run.step())
            transaction = run._transaction
            available = self.budget.available
            run.redecode(prompt="replacement")
            wrap.assert_called_once_with(first_inference, (1, 2))
            self.assertIs(self.second.inference, second_inference)
            self.assertIs(run._transaction, transaction)
            self.assertEqual(self.budget.available, available)
            self.assertEqual(run.decode_attempt, 2)
            self.assertEqual(self.first.cleanup_calls, 1)
            self.assertEqual(self.second.prefill_calls, 1)
            self.assertEqual(self.harness.encoder_calls, 1)
            self.assertTrue(run.step())
            self.assertEqual(run.finish().windows[-1].result.text, "world")
        self.assertEqual(self.second.cleanup_calls, 1)
        self.assert_released(run, published=True)


if __name__ == "__main__":
    unittest.main()
