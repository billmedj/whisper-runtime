import unittest
from collections.abc import Callable, Mapping
from copy import deepcopy
from threading import Event, Lock, Thread
from typing import cast
from unittest.mock import patch

from whisper_runtime import (
    Budget,
    ImmediateFence,
    ModelMismatchError,
    ModelSnapshot,
    QueueFullError,
    RequestState,
    RequestStatus,
    ResourceVector,
    Session,
    WindowTransaction,
    Worker,
)
from whisper_runtime.adapters import (
    LEGACY_WHISPER_ENVELOPE_VERSION,
    LegacyExecutionProfile,
    LegacyInputProvenance,
    LegacyOptionsMutationError,
    LegacyPayloadError,
    LegacyTranscribeOptions,
    LegacyTranscriptionEnvelope,
    LegacyTranscriptionRetainedError,
    LegacyWhisperAdapter,
)


class FakeLegacyWhisper:
    def __init__(
        self,
        identity: ModelSnapshot,
        payload: Mapping[str, object],
    ) -> None:
        self.identity = identity
        self.payload = dict(payload)
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.identity_after_call: ModelSnapshot | None = None
        self.mutate_list_option = False
        self.error: BaseException | None = None

    def transcribe(
        self,
        audio: object,
        **decode_options: object,
    ) -> Mapping[str, object]:
        self.calls.append((audio, deepcopy(decode_options)))
        if self.mutate_list_option:
            clips = cast(list[object], decode_options["clip_timestamps"])
            clips.append(30.0)
        if self.identity_after_call is not None:
            self.identity = self.identity_after_call
        if self.error is not None:
            raise self.error
        return self.payload


class SerialProbeWhisper:
    """Expose overlapping calls without relying on timing for correctness."""

    def __init__(self, identity: ModelSnapshot) -> None:
        self.identity = identity
        self.first_entered = Event()
        self.second_entered = Event()
        self.release_first = Event()
        self._lock = Lock()
        self._active = 0
        self._calls = 0
        self.max_active = 0

    def transcribe(
        self,
        audio: object,
        **decode_options: object,
    ) -> Mapping[str, object]:
        del decode_options
        with self._lock:
            self._active += 1
            self._calls += 1
            call_number = self._calls
            self.max_active = max(self.max_active, self._active)
        try:
            if call_number == 1:
                self.first_entered.set()
                if not self.release_first.wait(timeout=2):
                    raise TimeoutError(
                        "the serialization test did not release call one"
                    )
            else:
                self.second_entered.set()
            return {"text": cast(str, audio), "segments": []}
        finally:
            with self._lock:
                self._active -= 1


class SwitchableFailScope:
    def __init__(self) -> None:
        self.fail = True
        self.stop_calls = 0

    def request_stop(self) -> None:
        self.stop_calls += 1

    def completion_fence(self) -> "SwitchableFailScope":
        return self

    def wait(self) -> None:
        if self.fail:
            raise RuntimeError("backend is still active")


def probe(model: object) -> ModelSnapshot:
    return cast(ModelSnapshot, getattr(model, "identity"))


class LegacyWhisperAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_identity = ModelSnapshot(
            model_id="large-v3-turbo",
            revision="openai-whisper-main",
            backend="pytorch",
            fingerprint="sha256:test-checkpoint",
        )
        self.capacity = ResourceVector(
            memory_bytes=4_096,
            compute_units=4,
            stream_slots=2,
        )
        self.cost = self.capacity
        self.budget = Budget(self.capacity)
        self.worker = Worker(
            "legacy-worker",
            self.model_identity,
            self.budget,
            queue_capacity=1,
        )
        self.execution_profile = LegacyExecutionProfile(
            profile_id="large-v3-turbo/test",
            resources=self.cost,
        )
        self.input_provenance = LegacyInputProvenance(
            input_id="fixture:sample.flac",
            digest=f"sha256:{'0' * 64}",
            media_type="audio/flac",
            size_bytes=1_024,
        )
        self.payload: dict[str, object] = {
            "text": " Hello world.",
            "language": "en",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.5,
                    "text": " Hello world.",
                    "tokens": [50364, 2425, 1002, 13, 50439],
                }
            ],
            "vendor_extension": {"kept": True},
        }
        self.model = FakeLegacyWhisper(self.model_identity, self.payload)
        self.adapter = LegacyWhisperAdapter(
            self.worker,
            self.model,
            probe,
            self.execution_profile,
        )

    def request(self, request_id: str = "request-1") -> RequestState:
        return RequestState(
            request_id=request_id,
            session_id="session-1",
            model=self.model_identity,
            rng_seed=7,
        )

    def transcribe(
        self,
        request: RequestState,
        *,
        options: LegacyTranscribeOptions | None = None,
        audio: object = "sample.flac",
    ) -> LegacyTranscriptionEnvelope:
        return self.adapter.transcribe(
            session=Session("session-1"),
            request=request,
            window_id="window-1",
            execution=ImmediateFence(),
            audio=audio,
            input_provenance=self.input_provenance,
            start_ms=0,
            end_ms=1_500,
            options=options,
        )

    def test_preserves_historical_payload_and_commits_versioned_envelope(self) -> None:
        source_options: dict[str, object] = {
            "language": "en",
            "temperature": (0.0, 0.2),
            "clip_timestamps": [0.0, 1.5],
            "verbose": None,
            "word_timestamps": True,
        }
        options = LegacyTranscribeOptions(source_options)
        cast(list[float], source_options["clip_timestamps"]).append(99.0)
        audio = object()
        request = self.request()

        submit_calls: list[str] = []
        original_submit = WindowTransaction.submit

        def tracked_submit(
            transaction: WindowTransaction,
            operation: Callable[[], object],
        ) -> object:
            submit_calls.append(transaction.request_id)
            return original_submit(transaction, operation)

        with patch.object(WindowTransaction, "submit", new=tracked_submit):
            envelope = self.transcribe(request, options=options, audio=audio)

        self.assertEqual(submit_calls, ["request-1"])
        self.assertEqual(envelope.schema_version, LEGACY_WHISPER_ENVELOPE_VERSION)
        self.assertEqual(envelope.model, self.model_identity)
        self.assertEqual(envelope.options_fingerprint, options.fingerprint)
        self.assertEqual(envelope.payload, self.payload)
        self.assertIsNot(envelope.payload, self.model.payload)
        self.assertEqual(envelope.to_legacy_payload(), self.payload)
        self.assertEqual(envelope.committed_state.version, 1)
        self.assertEqual(
            envelope.committed_state.windows[0].result.text,
            self.payload["text"],
        )
        self.assertEqual(request.status, RequestStatus.COMMITTED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

        called_audio, called_options = self.model.calls[0]
        self.assertIs(called_audio, audio)
        self.assertEqual(called_options["clip_timestamps"], [0.0, 1.5])
        self.assertEqual(called_options["temperature"], (0.0, 0.2))
        self.assertIsNone(called_options["verbose"])

        serialized = envelope.as_dict()
        self.assertEqual(serialized["session_version"], 1)
        self.assertEqual(serialized["payload"], self.payload)

    def test_rejects_binding_to_a_different_loaded_model(self) -> None:
        other = ModelSnapshot(
            model_id="large-v3-turbo",
            revision="other",
            backend="pytorch",
            fingerprint="sha256:other",
        )
        model = FakeLegacyWhisper(other, self.payload)
        with self.assertRaises(ModelMismatchError):
            LegacyWhisperAdapter(
                self.worker,
                model,
                probe,
                self.execution_profile,
            )

    def test_identity_change_during_transcribe_aborts_and_releases(self) -> None:
        self.model.identity_after_call = ModelSnapshot(
            model_id="large-v3-turbo",
            revision="changed",
            backend="pytorch",
            fingerprint="sha256:changed",
        )
        request = self.request()

        with self.assertRaises(ModelMismatchError):
            self.transcribe(request)

        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_backend_option_mutation_is_detected_without_changing_snapshot(
        self,
    ) -> None:
        self.model.mutate_list_option = True
        request = self.request()
        options = LegacyTranscribeOptions({"clip_timestamps": [0.0, 1.5]})

        with self.assertRaises(LegacyOptionsMutationError):
            self.transcribe(request, options=options)

        self.assertEqual(options.to_kwargs()["clip_timestamps"], [0.0, 1.5])
        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(self.budget.available, self.capacity)

    def test_backend_error_aborts_and_releases(self) -> None:
        self.model.error = RuntimeError("decode failed")
        request = self.request()

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            self.transcribe(request)

        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_invalid_payload_aborts_without_projection(self) -> None:
        self.model.payload = {"language": "en", "segments": []}
        request = self.request()

        with self.assertRaises(LegacyPayloadError):
            self.transcribe(request)

        self.assertEqual(request.status, RequestStatus.ABORTED)
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_options_reject_nonportable_values_and_reserved_arguments(self) -> None:
        with self.assertRaises(TypeError):
            LegacyTranscribeOptions({"temperature": object()})
        with self.assertRaises(ValueError):
            LegacyTranscribeOptions({"audio": "different.wav"})

    def test_payload_is_immutable_and_bound_to_the_committed_projection(self) -> None:
        envelope = self.transcribe(self.request())
        original_digest = envelope.payload_digest

        detached = envelope.payload
        detached["text"] = "forged"
        cast(dict[str, object], detached["vendor_extension"])["kept"] = False

        self.assertEqual(envelope.payload["text"], " Hello world.")
        self.assertEqual(
            cast(dict[str, object], envelope.payload["vendor_extension"])["kept"],
            True,
        )
        self.assertEqual(envelope.payload_digest, original_digest)
        self.assertEqual(envelope.as_dict()["payload_digest"], original_digest)
        self.assertEqual(
            cast(dict[str, object], envelope.as_dict()["payload"])["text"],
            envelope.committed_state.windows[-1].result.text,
        )

    def test_profile_is_fixed_nonzero_and_reported_with_honest_measurements(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            LegacyExecutionProfile("zero", ResourceVector())
        too_large = LegacyExecutionProfile(
            "too-large",
            ResourceVector(memory_bytes=self.capacity.memory_bytes + 1),
        )
        with self.assertRaises(ValueError):
            LegacyWhisperAdapter(self.worker, self.model, probe, too_large)

        envelope = self.transcribe(self.request())
        data = envelope.as_dict()
        profile = cast(dict[str, object], data["execution_profile"])
        measurements = cast(dict[str, object], data["measurements"])
        self.assertEqual(profile["profile_id"], "large-v3-turbo/test")
        self.assertEqual(profile["serialization"], "per_model")
        self.assertGreaterEqual(cast(int, measurements["backend_call_ns"]), 0)
        self.assertIsNone(measurements["device_time_ns"])
        self.assertIsNone(measurements["peak_memory_bytes"])
        self.assertEqual(self.budget.available, self.capacity)

    def test_input_provenance_is_validated_and_audio_is_not_implicitly_hashed(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            LegacyInputProvenance("audio", digest="sha256:ABC")

        class OpaqueAudio:
            def __bytes__(self) -> bytes:
                raise AssertionError("the adapter must not inspect arbitrary audio")

        audio = OpaqueAudio()
        envelope = self.transcribe(self.request(), audio=audio)
        self.assertIs(self.model.calls[-1][0], audio)
        self.assertEqual(
            cast(dict[str, object], envelope.as_dict()["input"])["input_id"],
            "fixture:sample.flac",
        )

    def test_model_calls_use_one_bounded_worker_across_adapter_instances(self) -> None:
        model = SerialProbeWhisper(self.model_identity)
        first_adapter = LegacyWhisperAdapter(
            self.worker,
            model,
            probe,
            self.execution_profile,
        )
        second_adapter = LegacyWhisperAdapter(
            self.worker,
            model,
            probe,
            self.execution_profile,
        )
        results: dict[str, str] = {}
        errors: list[BaseException] = []

        def run(adapter: LegacyWhisperAdapter, suffix: str) -> None:
            try:
                envelope = adapter.transcribe(
                    session=Session(f"session-{suffix}"),
                    request=RequestState(
                        f"request-{suffix}",
                        f"session-{suffix}",
                        self.model_identity,
                        rng_seed=7,
                    ),
                    window_id=f"window-{suffix}",
                    execution=ImmediateFence(),
                    audio=suffix,
                    input_provenance=LegacyInputProvenance(f"input-{suffix}"),
                    start_ms=0,
                    end_ms=1,
                )
                results[suffix] = cast(str, envelope.payload["text"])
            except BaseException as exc:
                errors.append(exc)

        first = Thread(target=run, args=(first_adapter, "A"))
        first.start()
        self.assertTrue(model.first_entered.wait(timeout=1))
        with self.assertRaises(QueueFullError):
            second_adapter.transcribe(
                session=Session("session-B"),
                request=RequestState(
                    "request-B",
                    "session-B",
                    self.model_identity,
                    rng_seed=7,
                ),
                window_id="window-B",
                execution=ImmediateFence(),
                audio="B",
                input_provenance=LegacyInputProvenance("input-B"),
                start_ms=0,
                end_ms=1,
            )
        model.release_first.set()
        first.join(timeout=2)
        self.assertFalse(first.is_alive())

        run(second_adapter, "B")
        self.assertEqual(errors, [])
        self.assertEqual(results, {"A": "A", "B": "B"})
        self.assertEqual(model.max_active, 1)

    def test_one_model_object_cannot_be_bound_to_a_second_worker(self) -> None:
        second_worker = Worker(
            "other-worker",
            self.model_identity,
            Budget(self.capacity),
            queue_capacity=1,
        )
        with self.assertRaisesRegex(ValueError, "cannot use multiple workers"):
            LegacyWhisperAdapter(
                second_worker,
                self.model,
                probe,
                LegacyExecutionProfile("other", self.capacity),
            )

    def test_committed_retained_error_preserves_envelope_and_recovery_handle(
        self,
    ) -> None:
        request = self.request()
        with patch.object(
            ResourceVector,
            "__add__",
            side_effect=MemoryError("release failed"),
        ):
            with self.assertRaises(LegacyTranscriptionRetainedError) as raised:
                self.transcribe(request)

        error = raised.exception
        self.assertIsNotNone(error.envelope)
        assert error.envelope is not None
        self.assertIs(error.committed_state, error.envelope.committed_state)
        self.assertEqual(error.envelope.payload, self.payload)
        self.assertEqual(error.payload, self.payload)
        self.assertIs(error.transaction, error.runtime_error.transaction)
        self.assertEqual(request.status, RequestStatus.COMMITTED)
        self.assertEqual(self.worker.queue_depth, 1)
        self.assertTrue(error.recover())
        self.assertEqual(self.worker.queue_depth, 0)

    def test_uncommitted_retained_error_labels_payload_as_attempted_only(self) -> None:
        request = self.request()
        scope = SwitchableFailScope()

        with self.assertRaises(LegacyTranscriptionRetainedError) as raised:
            self.adapter.transcribe(
                session=Session("session-1"),
                request=request,
                window_id="window-1",
                execution=scope,
                audio="sample.flac",
                input_provenance=self.input_provenance,
                start_ms=0,
                end_ms=1_500,
            )

        error = raised.exception
        self.assertIsNone(error.envelope)
        self.assertIsNone(error.committed_state)
        self.assertEqual(error.payload, self.payload)
        self.assertEqual(request.status, RequestStatus.RUNNING)
        self.assertEqual(self.worker.queue_depth, 1)
        with self.assertRaises(QueueFullError):
            self.adapter.transcribe(
                session=Session("other-session"),
                request=RequestState(
                    "other-request",
                    "other-session",
                    self.model_identity,
                    rng_seed=9,
                ),
                window_id="other-window",
                execution=ImmediateFence(),
                audio="other.flac",
                input_provenance=LegacyInputProvenance("other-input"),
                start_ms=0,
                end_ms=1,
            )
        scope.fail = False
        self.assertTrue(error.recover())
        self.assertEqual(self.worker.queue_depth, 0)


if __name__ == "__main__":
    unittest.main()
