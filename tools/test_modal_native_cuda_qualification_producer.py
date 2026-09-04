"""Pure tests for the Modal native CUDA qualification producer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra import modal_native_cuda_qualification as producer  # noqa: E402


def _manifest() -> dict[str, object]:
    return json.loads(
        (ROOT / producer.QUALIFICATION_MANIFEST_PATH).read_text(encoding="utf-8")
    )


class _Distribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.name = name
        self.version = version


class ModalNativeCudaQualificationProducerTests(unittest.TestCase):
    def test_fresh_import_does_not_load_remote_dependencies(self) -> None:
        script = f"""
import importlib
import os
import sys
os.environ.pop('WHISPER_MODAL_ENABLE_REMOTE_RESOURCES', None)
os.environ.pop('MODAL_IS_REMOTE', None)
sys.path.insert(0, {str(ROOT)!r})
importlib.import_module('infra.modal_native_cuda_qualification')
assert 'modal' not in sys.modules
assert 'torch' not in sys.modules
assert 'whisper' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_modal_invocation_must_use_the_canonical_module_name(self) -> None:
        producer._require_canonical_modal_invocation()
        invalid_identities = (
            ("modal_native_cuda_qualification", "", "modal_native_cuda_qualification"),
            ("__main__", "infra", "infra.modal_native_cuda_qualification"),
            (
                producer.CANONICAL_MODULE_NAME,
                "infra",
                "modal_native_cuda_qualification",
            ),
        )
        for name, package, spec_name in invalid_identities:
            with (
                self.subTest(name=name, package=package, spec_name=spec_name),
                patch.dict(
                    producer.__dict__,
                    {
                        "__name__": name,
                        "__package__": package,
                        "__spec__": SimpleNamespace(name=spec_name),
                    },
                    clear=False,
                ),
                self.assertRaisesRegex(SystemExit, "modal run -m"),
            ):
                producer._require_canonical_modal_invocation()

    def test_invalid_invocation_stops_before_attempt_creation(self) -> None:
        with (
            patch.object(
                producer,
                "_require_canonical_modal_invocation",
                side_effect=SystemExit("invalid module identity"),
            ),
            patch.object(producer, "_execute_registered_attempt") as execute,
            self.assertRaisesRegex(SystemExit, "invalid module identity"),
        ):
            producer._modal_main(confirm_paid_gpu=True)
        execute.assert_not_called()

    def test_paid_dispatch_requires_explicit_confirmation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "No cache or GPU function"):
            producer._require_paid_confirmation(False)
        producer._require_paid_confirmation(True)

    def test_remote_context_does_not_redefine_modal_resources(self) -> None:
        with patch.dict(
            os.environ,
            {"MODAL_IS_REMOTE": "1", "WHISPER_MODAL_ENABLE_REMOTE_RESOURCES": "0"},
            clear=False,
        ):
            self.assertFalse(producer._definition_enabled())

    def test_remote_wrapper_delegates_to_bound_producer(self) -> None:
        observed: dict[str, object] = {}

        def run(runtime_commit: str, *, modal_module: object) -> dict[str, object]:
            observed.update(commit=runtime_commit, modal=modal_module)
            return {"status": "passed"}

        bound = SimpleNamespace(_run_qualification_worker=run)
        modal = object()
        with patch.object(
            producer.importlib, "import_module", return_value=bound
        ) as load:
            result = producer._run_bound_worker("a" * 40, modal_module=modal)
        load.assert_called_once_with("infra.modal_native_cuda_qualification")
        self.assertEqual(result, {"status": "passed"})
        self.assertEqual(observed, {"commit": "a" * 40, "modal": modal})

    def test_output_path_is_closed_and_repository_relative(self) -> None:
        self.assertEqual(
            producer._output_path(producer.REGISTERED_OUTPUT_PATH),
            Path(producer.REGISTERED_OUTPUT_PATH),
        )
        for value in (
            "",
            "run.json",
            "artifacts/modal/another-run.json",
            "artifacts/modal/../run.json",
            "artifacts/modal/run.txt",
            "C:/artifacts/modal/run.json",
            "artifacts\\modal\\run.json",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                producer._output_path(value)

    def test_registered_campaign_order_is_exact(self) -> None:
        order = producer._campaign_order(_manifest())
        self.assertEqual(len(order), 33)
        self.assertEqual(
            order[:4],
            (
                ("control-warmup", 0, None),
                ("warmup", 0, None),
                ("control-warmup", 1, None),
                ("warmup", 1, None),
            ),
        )
        self.assertEqual(
            order[-2:],
            (("fault", 1, "event-synchronize"), ("reuse", 1, "event-synchronize")),
        )

    def test_fault_trace_counts_cover_each_registered_fault(self) -> None:
        self.assertEqual(
            producer.FAULT_TRACE_COUNTS,
            {
                "cleanup": (3, 1),
                "event-create": (3, 3),
                "event-record": (3, 3),
                "event-synchronize": (3, 3),
            },
        )

    def test_fault_is_armed_before_resource_admission(self) -> None:
        self.assertEqual(producer.FAULT_EVENTS, producer._validator().FAULT_EVENTS)
        self.assertEqual(
            producer.FAULT_EVENTS[:3],
            ("run-start", "fault-armed", "lease-acquired"),
        )

        context = producer.RunContext(
            run_id="fault-cleanup-0",
            run_kind="fault",
            iteration=0,
            session_id="session",
            request_id="request",
            transaction_id="transaction",
            lease_id="lease",
            fault_point="cleanup",
        )
        events = producer.QualificationEventLog("a" * 64)
        trace = producer._prepare_fault_trace(context, events)
        self.assertEqual(trace.fault_plan.remaining(producer.FaultPoint.CLEANUP), 2)
        self.assertEqual(tuple(item["event"] for item in events.events), ("run-start",))
        router = producer.TraceRouter()
        with router.activate(trace):
            producer._record_fault_armed(context, events, router, trace)
        self.assertEqual(
            tuple(item["event"] for item in events.events),
            ("run-start", "fault-armed"),
        )

    def test_modal_image_id_is_observed_not_invented(self) -> None:
        with patch.dict(os.environ, {"MODAL_IMAGE_ID": "im-12345678"}, clear=False):
            self.assertEqual(producer._required_modal_image_id(), "im-12345678")
        for value in ("", "sha256:" + "a" * 64, "im-short"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"MODAL_IMAGE_ID": value}, clear=False),
                self.assertRaises(RuntimeError),
            ):
                producer._required_modal_image_id()

    def test_modal_location_is_observed_and_exact(self) -> None:
        manifest = _manifest()
        with patch.dict(
            os.environ,
            {
                "MODAL_CLOUD_PROVIDER": "CLOUD_PROVIDER_AWS",
                "MODAL_REGION": "us-west-2",
            },
            clear=False,
        ):
            self.assertEqual(
                producer._required_modal_location(manifest),
                ("CLOUD_PROVIDER_AWS", "us-west-2"),
            )
        with (
            patch.dict(
                os.environ,
                {
                    "MODAL_CLOUD_PROVIDER": "CLOUD_PROVIDER_GCP",
                    "MODAL_REGION": "us-west-2",
                },
                clear=False,
            ),
            self.assertRaisesRegex(RuntimeError, "differs from the registration"),
        ):
            producer._required_modal_location(manifest)

    def test_shadow_module_origin_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            checkout = parent / "checkout"
            expected = checkout / "package" / "module.py"
            expected.parent.mkdir(parents=True)
            expected.write_text("", encoding="utf-8")
            producer._require_module_origin(
                SimpleNamespace(__file__=str(expected)),
                checkout,
                "package/module.py",
                "fixture module",
            )
            shadow = parent / "shadow" / "module.py"
            shadow.parent.mkdir()
            shadow.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "outside the bound checkout"):
                producer._require_module_origin(
                    SimpleNamespace(__file__=str(shadow)),
                    checkout,
                    "package/module.py",
                    "fixture module",
                )

    def test_wall_time_uses_registered_terminal_event(self) -> None:
        context = producer.RunContext(
            run_id="run",
            run_kind="measured",
            iteration=0,
            session_id="session",
            request_id="request",
            transaction_id="transaction",
            lease_id="lease",
        )
        events = SimpleNamespace(
            events_for=lambda _: [
                {"event": "run-start", "offset_ns": 10},
                {"event": "budget-restored", "offset_ns": 40},
                {"event": "run-complete", "offset_ns": 90},
            ]
        )
        self.assertEqual(
            producer._run_wall_ns(events, context, end_event="budget-restored"),
            30,
        )

    def test_image_inputs_are_exact_and_sorted(self) -> None:
        values = producer._locked_image_inputs()
        self.assertEqual(values, tuple(sorted(values)))
        self.assertIn("modal==1.5.5", values)
        self.assertIn("torch==2.6.0", values)

    def test_input_manifest_is_content_bound_to_the_registered_fixture(self) -> None:
        cell = _manifest()["cell"]
        producer._verify_input_manifest(ROOT / producer.INPUT_MANIFEST_PATH, cell)

        with tempfile.TemporaryDirectory() as directory:
            altered_path = Path(directory) / "audio-manifest.json"
            altered = json.loads(
                (ROOT / producer.INPUT_MANIFEST_PATH).read_text(encoding="utf-8")
            )
            altered["fixtures"][0]["source_url"] = "https://example.invalid/audio.flac"
            altered_path.write_text(json.dumps(altered), encoding="utf-8")
            altered_cell = dict(cell)
            altered_cell["input_manifest_sha256"] = producer._sha256_file(altered_path)
            with self.assertRaisesRegex(RuntimeError, "fixture differs"):
                producer._verify_input_manifest(altered_path, altered_cell)

    def test_resolved_inventory_is_sorted_normalized_and_complete(self) -> None:
        distributions = [
            _Distribution("Torch", "2.6.0"),
            _Distribution("example_pkg", "1.0"),
            _Distribution("Modal", "1.5.5"),
            _Distribution("Example-Pkg", "0.9-shadowed"),
        ]
        selected = {
            "example-pkg": _Distribution("example_pkg", "1.0"),
            "modal": _Distribution("Modal", "1.5.5"),
            "torch": _Distribution("Torch", "2.6.0"),
        }
        with (
            patch.object(
                producer.importlib.metadata,
                "distributions",
                return_value=distributions,
            ),
            patch.object(
                producer.importlib.metadata,
                "distribution",
                side_effect=selected.__getitem__,
            ) as resolve,
        ):
            self.assertEqual(
                producer._resolved_dependencies(),
                [
                    {"name": "example-pkg", "version": "1.0"},
                    {"name": "modal", "version": "1.5.5"},
                    {"name": "torch", "version": "2.6.0"},
                ],
            )
        self.assertEqual(
            [call.args[0] for call in resolve.call_args_list],
            ["example-pkg", "modal", "torch"],
        )

    def test_resolved_inventory_uses_the_resolver_selected_version(self) -> None:
        visible = [
            _Distribution("idna", "3.10"),
            _Distribution("IDNA", "3.11-shadowed"),
        ]
        with (
            patch.object(
                producer.importlib.metadata, "distributions", return_value=visible
            ),
            patch.object(
                producer.importlib.metadata,
                "distribution",
                return_value=_Distribution("idna", "3.10"),
            ) as resolve,
        ):
            self.assertEqual(
                producer._resolved_dependencies(),
                [{"name": "idna", "version": "3.10"}],
            )
        resolve.assert_called_once_with("idna")

    def test_resolved_inventory_rejects_a_disappearing_distribution(self) -> None:
        with (
            patch.object(
                producer.importlib.metadata,
                "distributions",
                return_value=[_Distribution("idna", "3.10")],
            ),
            patch.object(
                producer.importlib.metadata,
                "distribution",
                side_effect=producer.importlib.metadata.PackageNotFoundError("idna"),
            ),
            self.assertRaisesRegex(RuntimeError, "disappeared during inventory"),
        ):
            producer._resolved_dependencies()

    def test_resolved_inventory_rejects_invalid_selected_metadata(self) -> None:
        cases = (
            (_Distribution("other", "3.10"), "different package"),
            (_Distribution("idna", ""), "no version"),
        )
        for selected, message in cases:
            with (
                self.subTest(message=message),
                patch.object(
                    producer.importlib.metadata,
                    "distributions",
                    return_value=[_Distribution("idna", "3.10")],
                ),
                patch.object(
                    producer.importlib.metadata,
                    "distribution",
                    return_value=selected,
                ),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                producer._resolved_dependencies()

    def test_record_write_is_atomic_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            producer._write_record(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            with self.assertRaises(FileExistsError):
                producer._write_record(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})

    def test_failed_campaign_retains_one_attempt(self) -> None:
        calls = 0

        def fail() -> dict[str, object]:
            nonlocal calls
            calls += 1
            raise RuntimeError("simulated campaign failure")

        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                destination = Path("artifacts/modal/failure.json")
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    producer._execute_registered_attempt(
                        destination,
                        runtime_commit="a" * 40,
                        manifest=_manifest(),
                        manifest_sha256="b" * 64,
                        prime_cache=None,
                        run_campaign=fail,
                    )
                events = [
                    json.loads(line)
                    for line in producer._attempt_receipt_path(destination)
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual(
                    [event["event"] for event in events],
                    ["attempt-started", "attempt-failed"],
                )
                self.assertEqual([event["sequence"] for event in events], [0, 1])
                self.assertEqual({event["attempt"] for event in events}, {1})
                self.assertEqual(calls, 1)
                self.assertFalse(destination.exists())
                with self.assertRaises(FileExistsError):
                    producer._execute_registered_attempt(
                        destination,
                        runtime_commit="a" * 40,
                        manifest=_manifest(),
                        manifest_sha256="b" * 64,
                        prime_cache=None,
                        run_campaign=fail,
                    )
                self.assertEqual(calls, 1)
            finally:
                os.chdir(previous)

    def test_base_exception_is_not_disguised(self) -> None:
        def interrupt() -> dict[str, object]:
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                destination = Path("artifacts/modal/interrupted.json")
                with self.assertRaises(KeyboardInterrupt):
                    producer._execute_registered_attempt(
                        destination,
                        runtime_commit="a" * 40,
                        manifest=_manifest(),
                        manifest_sha256="b" * 64,
                        prime_cache=None,
                        run_campaign=interrupt,
                    )
                lines = (
                    producer._attempt_receipt_path(destination)
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                self.assertEqual(len(lines), 1)
                self.assertEqual(json.loads(lines[0])["event"], "attempt-started")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
