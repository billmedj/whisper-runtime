"""Dependency-free checks for compile-only native alignment initialization."""

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from whisper_runtime import native_setup as setup


class AlignmentInitializationTests(unittest.TestCase):
    def fixtures(self):
        scalar = object()
        array = Mock(
            side_effect=lambda dtype, dimensions, layout: (dtype, dimensions, layout)
        )
        numba = SimpleNamespace(types=SimpleNamespace(int32=scalar, Array=array))
        backtrace = Mock()
        timing = SimpleNamespace(backtrace=backtrace)
        return scalar, array, numba, backtrace, timing

    def test_compiles_exact_int32_two_dimensional_c_and_a_layouts_in_order(self):
        scalar, array, numba, backtrace, timing = self.fixtures()
        with (
            patch.dict(sys.modules, {"whisper.timing": timing}),
            patch.object(
                setup.importlib, "import_module", return_value=numba
            ) as imports,
            patch.object(setup.time, "perf_counter", side_effect=(10.0, 10.25)),
        ):
            receipt = setup._prepare_alignment_backtrace()
        imports.assert_called_once_with("numba")
        self.assertEqual(
            array.call_args_list, [call(scalar, 2, "C"), call(scalar, 2, "A")]
        )
        self.assertEqual(
            backtrace.compile.call_args_list,
            [call(((scalar, 2, "C"),)), call(((scalar, 2, "A"),))],
        )
        self.assertEqual(
            receipt,
            {
                "id": "cuda-backtrace-init-v1",
                "signatures": ["int32[:,::1]", "int32[:,:]"],
                "wall_seconds": 0.25,
                "model_decodes": 0,
                "triton_kernels_warmed": False,
            },
        )

    def test_success_receipt_is_plain_and_preparation_executes_no_backtrace_or_model(
        self,
    ):
        _, _, numba, backtrace, timing = self.fixtures()
        with (
            patch.dict(sys.modules, {"whisper.timing": timing}),
            patch.object(setup.importlib, "import_module", return_value=numba),
            patch.object(setup, "_load_model") as model,
            patch.object(setup, "_fingerprint") as fingerprint,
        ):
            receipt = setup._prepare_alignment_backtrace()
        backtrace.assert_not_called()
        model.assert_not_called()
        fingerprint.assert_not_called()
        self.assertEqual(json.loads(json.dumps(receipt, allow_nan=False)), receipt)
        self.assertGreaterEqual(receipt["wall_seconds"], 0)
        self.assertEqual(receipt["model_decodes"], 0)
        self.assertIs(receipt["triton_kernels_warmed"], False)

    def test_absent_or_incompatible_backend_refuses_before_importing_numba(self):
        for timing in (
            None,
            SimpleNamespace(),
            SimpleNamespace(backtrace=None),
            SimpleNamespace(backtrace=SimpleNamespace(compile=None)),
            SimpleNamespace(backtrace=SimpleNamespace(compile=3)),
        ):
            with (
                self.subTest(timing=timing),
                patch.dict(sys.modules, {"whisper.timing": timing}),
                patch.object(setup.importlib, "import_module") as imports,
                self.assertRaisesRegex(setup.NativeSetupError, "backtrace compiler"),
            ):
                setup._prepare_alignment_backtrace()
            imports.assert_not_called()

    def test_numba_import_error_propagates_without_compiling_or_receipt(self):
        _, _, _, backtrace, timing = self.fixtures()
        failure = ImportError("compiler unavailable")
        receipt = None
        with (
            patch.dict(sys.modules, {"whisper.timing": timing}),
            patch.object(setup.importlib, "import_module", side_effect=failure),
            self.assertRaises(ImportError) as observed,
        ):
            receipt = setup._prepare_alignment_backtrace()
        self.assertIs(observed.exception, failure)
        self.assertIsNone(receipt)
        backtrace.compile.assert_not_called()
        backtrace.assert_not_called()

    def test_either_signature_compile_failure_propagates_without_success_receipt(self):
        for failed_signature in (1, 2):
            _, _, numba, backtrace, timing = self.fixtures()
            failure = RuntimeError("compiler rejected signature")
            backtrace.compile.side_effect = [None] * (failed_signature - 1) + [failure]
            receipt = None
            with (
                self.subTest(failed_signature=failed_signature),
                patch.dict(sys.modules, {"whisper.timing": timing}),
                patch.object(setup.importlib, "import_module", return_value=numba),
                self.assertRaises(RuntimeError) as observed,
            ):
                receipt = setup._prepare_alignment_backtrace()
            self.assertIs(observed.exception, failure)
            self.assertIsNone(receipt)
            self.assertEqual(backtrace.compile.call_count, failed_signature)
            backtrace.assert_not_called()

    def test_signature_construction_failure_does_not_emit_a_success_receipt(self):
        _, array, numba, backtrace, timing = self.fixtures()
        failure = TypeError("unsupported compiler type")
        array.side_effect = failure
        receipt = None
        with (
            patch.dict(sys.modules, {"whisper.timing": timing}),
            patch.object(setup.importlib, "import_module", return_value=numba),
            self.assertRaises(TypeError) as observed,
        ):
            receipt = setup._prepare_alignment_backtrace()
        self.assertIs(observed.exception, failure)
        self.assertIsNone(receipt)
        backtrace.compile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
