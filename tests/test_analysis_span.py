"""Published output may exclude overlap analyzed as decoder context."""

import unittest
from dataclasses import FrozenInstanceError, dataclass

from whisper_runtime import (
    AudioSpan,
    Budget,
    ImmediateFence,
    ModelSnapshot,
    RequestState,
    RequestStatus,
    ResourceVector,
    Session,
    StaleSessionError,
    TransactionStatus,
    WindowResult,
    WindowTransaction,
    Worker,
)


class AudioSpanTests(unittest.TestCase):
    def test_span_is_frozen_slotted_and_accepts_empty_intervals(self) -> None:
        span = AudioSpan(0, 0)
        self.assertEqual(span, AudioSpan(start_ms=0, end_ms=0))
        self.assertFalse(hasattr(span, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            span.start_ms = 1  # type: ignore[misc]

    def test_span_requires_non_boolean_integers(self) -> None:
        for field in ("start_ms", "end_ms"):
            for invalid in (True, False, 0.0, "0", None):
                with self.subTest(field=field, invalid=invalid):
                    arguments = {"start_ms": 0, "end_ms": 1}
                    arguments[field] = invalid  # type: ignore[assignment]
                    with self.assertRaisesRegex(
                        TypeError, f"{field} must be an integer"
                    ):
                        AudioSpan(**arguments)

    def test_span_rejects_negative_and_reversed_bounds(self) -> None:
        for start_ms, end_ms in ((-1, 0), (-2, -1), (0, -1), (2, 1)):
            with self.subTest(start_ms=start_ms, end_ms=end_ms):
                with self.assertRaises(ValueError):
                    AudioSpan(start_ms, end_ms)

    def test_legacy_result_construction_preserves_defaults_and_equality(self) -> None:
        result = WindowResult("window", "text", 1_000, 2_000)
        self.assertEqual(
            result,
            WindowResult(
                window_id="window",
                text="text",
                start_ms=1_000,
                end_ms=2_000,
                analysis_span=None,
            ),
        )
        self.assertIsNone(result.analysis_span)
        self.assertEqual(result.analyzed_span, AudioSpan(1_000, 2_000))

    def test_explicit_analysis_contains_output_including_boundaries(self) -> None:
        span = AudioSpan(17_000, 26_000)
        for start_ms, end_ms in ((20_000, 24_000), (17_000, 26_000), (26_000, 26_000)):
            with self.subTest(start_ms=start_ms, end_ms=end_ms):
                result = WindowResult("window", "text", start_ms, end_ms, span)
                self.assertIs(result.analyzed_span, span)
                self.assertEqual((result.start_ms, result.end_ms), (start_ms, end_ms))

    def test_result_rejects_noncontaining_analysis(self) -> None:
        for span in (AudioSpan(20_001, 26_000), AudioSpan(17_000, 23_999)):
            with self.subTest(span=span):
                with self.assertRaisesRegex(ValueError, "must contain"):
                    WindowResult("window", "text", 20_000, 24_000, span)

    def test_result_rejects_untyped_analysis(self) -> None:
        for invalid in ((0, 1), {"start_ms": 0, "end_ms": 1}, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "analysis_span must be"):
                    WindowResult("window", "text", 0, 1, invalid)  # type: ignore[arg-type]

    def test_frozen_slotted_subclass_can_add_required_keyword_metadata(self) -> None:
        @dataclass(frozen=True, slots=True, kw_only=True)
        class TypedResult(WindowResult):
            metadata: tuple[int, ...]

        result = TypedResult("window", "text", 0, 1, metadata=(1, 2))
        self.assertEqual(result.analyzed_span, AudioSpan(0, 1))
        self.assertEqual(result.metadata, (1, 2))
        self.assertFalse(hasattr(result, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            result.metadata = ()  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "must contain"):
            TypedResult("window", "text", 0, 2, AudioSpan(0, 1), metadata=())


class AnalysisSpanTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ModelSnapshot("whisper-test", "1", "reference", "sha256:test")
        self.capacity = ResourceVector(
            memory_bytes=1_000, compute_units=2, stream_slots=2
        )
        self.cost = ResourceVector(memory_bytes=100, compute_units=1, stream_slots=1)
        self.budget = Budget(self.capacity)
        self.worker = Worker("worker", self.model, self.budget, queue_capacity=2)

    def prepare(
        self, session: Session, name: str
    ) -> tuple[WindowTransaction, RequestState]:
        request = RequestState(name, session.session_id, self.model, rng_seed=1)
        transaction = self.worker.prepare(
            session=session, request=request, window_id=name, resources=self.cost
        )
        transaction.start(ImmediateFence())
        return transaction, request

    def establish_watermark(self, session: Session) -> None:
        transaction, _ = self.prepare(session, "first")
        transaction.commit(
            WindowResult("first", "first", 0, 20_000), committed_through_ms=20_000
        )

    def assert_released(self) -> None:
        self.assertEqual(self.worker.queue_depth, 0)
        self.assertEqual(self.budget.lease_count, 0)
        self.assertEqual(self.budget.available, self.capacity)

    def test_analysis_can_overlap_committed_audio_without_publishing_it(self) -> None:
        session = Session("session")
        self.establish_watermark(session)
        transaction, request = self.prepare(session, "overlap")
        result = WindowResult(
            "overlap", "new output", 20_000, 24_000, AudioSpan(17_000, 26_000)
        )
        state = transaction.commit(result, committed_through_ms=24_000)
        self.assertIs(state.windows[-1].result, result)
        self.assertEqual(state.committed_through_ms, 24_000)
        self.assertIs(request.status, RequestStatus.COMMITTED)
        self.assert_released()

    def test_publishing_below_watermark_still_aborts_and_releases(self) -> None:
        for history_limit in (1, 2):
            with self.subTest(history_limit=history_limit):
                session = Session("session", history_limit=history_limit)
                self.establish_watermark(session)
                # Evict the watermark-setting record without updating its value.
                for index in range(history_limit):
                    name = f"retained-{index}"
                    transaction, _ = self.prepare(session, name)
                    transaction.commit(
                        WindowResult(
                            name, "tail", 20_000, 24_000, AudioSpan(17_000, 26_000)
                        )
                    )
                before = session.snapshot()
                self.assertEqual(len(before.windows), history_limit)
                self.assertEqual(before.committed_through_ms, 20_000)
                self.assertTrue(
                    all(
                        record.committed_through_ms is None for record in before.windows
                    )
                )
                transaction, request = self.prepare(session, "invalid")
                with self.assertRaisesRegex(
                    ValueError, "cannot overlap committed audio"
                ):
                    transaction.commit(
                        WindowResult(
                            "invalid",
                            "old output",
                            19_999,
                            24_000,
                            AudioSpan(17_000, 26_000),
                        )
                    )
                self.assertIs(session.snapshot(), before)
                self.assertIs(transaction.status, TransactionStatus.ABORTED)
                self.assertIs(request.status, RequestStatus.ABORTED)
                self.assert_released()

    def test_analysis_does_not_extend_allowed_watermark(self) -> None:
        session = Session("session")
        self.establish_watermark(session)
        before = session.snapshot()
        transaction, request = self.prepare(session, "invalid")
        with self.assertRaisesRegex(ValueError, "cannot exceed the result end"):
            transaction.commit(
                WindowResult(
                    "invalid", "text", 20_000, 24_000, AudioSpan(17_000, 26_000)
                ),
                committed_through_ms=25_000,
            )
        self.assertIs(session.snapshot(), before)
        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assert_released()

    def test_explicit_analysis_preserves_compare_and_swap_protection(self) -> None:
        session = Session("session")
        self.establish_watermark(session)
        first, _ = self.prepare(session, "first-new")
        stale, request = self.prepare(session, "stale")
        state = first.commit(
            WindowResult(
                "first-new", "text", 20_000, 24_000, AudioSpan(17_000, 26_000)
            ),
            committed_through_ms=24_000,
        )
        with self.assertRaises(StaleSessionError):
            stale.commit(
                WindowResult("stale", "text", 24_000, 26_000, AudioSpan(17_000, 26_000))
            )
        self.assertIs(session.snapshot(), state)
        self.assertIs(stale.status, TransactionStatus.ABORTED)
        self.assertIs(request.status, RequestStatus.ABORTED)
        self.assert_released()


if __name__ == "__main__":
    unittest.main()
