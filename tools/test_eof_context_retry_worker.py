"""CPU-only regressions for injected plans; no native model or remote launch."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from infra import eof_context_retry_worker as worker
from infra import modal_eof_context_retry as unpaced


def _paced_producer():
    return SimpleNamespace(
        GPU_TIMEOUT_SECONDS=150,
        CLEANUP_RESERVE_SECONDS=20,
        MAX_NATIVE_WINDOWS=80,
        MAX_NATIVE_WINDOWS_PER_CELL=30,
        shared=SimpleNamespace(_sha=lambda pcm: "injected-source-hash"),
    )


class InjectedAdmissionTests(unittest.TestCase):
    def adapter(self, *, producer=None, elapsed_ns=1):
        native = SimpleNamespace(model_identity="model", start_window=Mock())
        self.clock = SimpleNamespace(now=elapsed_ns)
        supplied = {} if producer is None else {"producer": producer}
        measured = worker._MeasuredAdapter(
            native, 0, clock=lambda: self.clock.now, **supplied
        )
        measured.pcm = bytes(32000)
        return measured

    @staticmethod
    def args():
        return dict(
            start_ms=0,
            end_ms=1000,
            window_id="synthetic-window",
            mel=SimpleNamespace(shape=(80, 3000)),
        )

    def admit(self, measured, count):
        for _ in range(count):
            measured.start_window(**self.args())

    def assert_bound_rejects_before_native(self, measured):
        before = (measured.count, len(measured.records))
        native_calls = measured.native.start_window.call_count
        with self.assertRaisesRegex(RuntimeError, "native-window bound"):
            measured.start_window(**self.args())
        self.assertEqual((measured.count, len(measured.records)), before)
        self.assertEqual(measured.native.start_window.call_count, native_calls)

    def test_run_worker_retains_unpaced_defaults(self):
        parameters = inspect.signature(worker.run_worker).parameters
        self.assertIs(parameters["producer"].default, unpaced)
        self.assertIs(parameters["paced"].default, False)

    def test_default_producer_keeps_twenty_window_cell_limit(self):
        measured = self.adapter()
        self.assertIs(measured.producer, unpaced)
        self.admit(measured, 20)
        self.assert_bound_rejects_before_native(measured)
        self.assertEqual(measured.count, 20)

    def test_injected_plan_allows_thirty_per_cell_but_eighty_total(self):
        measured = self.adapter(producer=_paced_producer())
        self.admit(measured, 30)
        self.assert_bound_rejects_before_native(measured)
        measured.records = []
        self.admit(measured, 30)
        self.assert_bound_rejects_before_native(measured)
        measured.records = []
        self.admit(measured, 20)
        self.assert_bound_rejects_before_native(measured)
        self.assertEqual(measured.count, 80)
        self.assertEqual(measured.native.start_window.call_count, 80)
        self.assertEqual(measured.records[-1]["call_index"], 80)

    def test_injected_legacy_plan_without_cell_limit_keeps_twenty(self):
        producer = _paced_producer()
        del producer.MAX_NATIVE_WINDOWS_PER_CELL
        measured = self.adapter(producer=producer)
        self.admit(measured, 20)
        self.assert_bound_rejects_before_native(measured)

    def test_timeout_uses_each_plan_and_preserves_cleanup_reserve(self):
        for producer, deadline_seconds in ((None, 100), (_paced_producer(), 130)):
            with self.subTest(deadline_seconds=deadline_seconds):
                deadline_ns = deadline_seconds * 1_000_000_000
                measured = self.adapter(producer=producer, elapsed_ns=deadline_ns - 1)
                self.admit(measured, 1)
                self.clock.now = deadline_ns
                with self.assertRaisesRegex(TimeoutError, "cleanup reserve"):
                    measured.start_window(**self.args())
                self.assertEqual(measured.native.start_window.call_count, 1)
                self.assertEqual(measured.count, 1)
                self.assertEqual(len(measured.records), 1)

    def test_injected_hash_helper_is_used_for_observed_source(self):
        measured = self.adapter(producer=_paced_producer())
        self.admit(measured, 1)
        self.assertEqual(measured.records[0]["pcm_sha256"], "injected-source-hash")


class PacedCellBudgetTests(unittest.TestCase):
    def decision(self, *, elapsed_ns, producer=None, samples=538560, drain_ms=15000):
        return worker._paced_budget_stop(
            bytes(samples * 2),
            producer=_paced_producer() if producer is None else producer,
            corpus=SimpleNamespace(PACED_REPLAY={"config": {"max_drain_ms": drain_ms}}),
            elapsed_ns=elapsed_ns,
            next_cell=2,
        )

    def test_registered_source_duration_drain_and_cleanup_are_reserved(self):
        # 538560 signed-16-bit samples / 16 kHz = 33.66 seconds.
        self.assertEqual(
            self.decision(elapsed_ns=100_000_000_000),
            dict(
                reason="budget_stop",
                next_cell=2,
                remaining_ns=50_000_000_000,
                required_ns=68_660_000_000,
            ),
        )

    def test_exact_budget_boundary_is_allowed_but_one_ns_less_is_not(self):
        self.assertIsNone(self.decision(elapsed_ns=81_340_000_000))
        stop = self.decision(elapsed_ns=81_340_000_001)
        self.assertEqual(stop["remaining_ns"], stop["required_ns"] - 1)

    def test_budget_uses_injected_timeout_drain_and_cleanup_values(self):
        self.assertIsNone(self.decision(elapsed_ns=80_000_000_000))
        stop = self.decision(elapsed_ns=80_000_000_000, producer=unpaced)
        self.assertEqual(stop["remaining_ns"], 40_000_000_000)
        producer = _paced_producer()
        producer.CLEANUP_RESERVE_SECONDS = 25
        stop = self.decision(
            elapsed_ns=150_000_000_000,
            producer=producer,
            samples=16000,
            drain_ms=2500,
        )
        self.assertEqual(stop["remaining_ns"], 0)
        self.assertEqual(stop["required_ns"], 28_500_000_000)


class PacedExecutionOutcomeTests(unittest.TestCase):
    def test_nonexception_pacing_failures_stop_execution(self):
        for status in ("drain_timeout", "step_limit", "source_lag", "admission_lag"):
            with self.subTest(status=status):
                self.assertTrue(
                    worker._paced_execution_failed({"status": status}, None)
                )

    def test_runtime_error_only_exempts_explicit_policy_resolution(self):
        pacing = {"status": "runtime_error"}
        self.assertFalse(
            worker._paced_execution_failed(pacing, {"category": "policy_resolution"})
        )
        for error in (None, {}, {"category": "runtime_error"}):
            with self.subTest(error=error):
                self.assertTrue(worker._paced_execution_failed(pacing, error))

    def test_completed_replay_does_not_stop_the_next_cell(self):
        self.assertFalse(worker._paced_execution_failed({"status": "completed"}, None))


if __name__ == "__main__":
    unittest.main()
