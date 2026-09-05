"""Network diagnostic guards with local fakes; no account or GPU is used."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import inspect
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infra import modal_network_replay as network


class NetworkDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "commit": "a" * 40,
            "files": [],
            "digest": network._canonical([]),
        }
        self.pcm = b"\x01\x00" * network.SAMPLES

    def preflight_record(self):
        return {
            "status": "completed",
            "preflight": True,
            "source": self.snapshot,
            "cleanup": {
                "ephemeral_context_exit_completed": True,
                "proxy_token_deleted": True,
            },
            "result": {
                "status": "completed",
                "protocol": "pcm-echo/v1",
                "unauthorized_status": 401,
                "sample_count": 16000,
                "chunks": 50,
                "sha256": network._sha(self.pcm[:32000]),
                "server_done": {"source_snapshot": self.snapshot},
            },
        }

    def fake_modal(
        self,
        *,
        scoped=True,
        client_error=None,
        enter_error=False,
        exit_error=False,
        delete_error=False,
        allow_error=False,
    ):
        order = []
        token = SimpleNamespace(token_id="wk-private", token_secret="ws-private")

        def create():
            order.append("create")
            return token

        def delete(identifier):
            self.assertEqual(identifier, token.token_id)
            order.append("delete")
            if delete_error:
                raise RuntimeError("ws-private")

        def allow(identifier, environment):
            self.assertEqual(
                (identifier, environment), (token.token_id, "test-environment")
            )
            order.append("allow")
            if allow_error:
                raise RuntimeError("wk-private")

        manager = SimpleNamespace(
            create=create,
            delete=delete,
            allow=allow,
            list=lambda: [SimpleNamespace(token_id=token.token_id, scoped=scoped)],
        )

        @contextmanager
        def run(**options):
            self.assertEqual(
                options, {"detach": False, "environment_name": "test-environment"}
            )
            order.append("enter")
            if enter_error:
                raise RuntimeError("startup ws-private")
            try:
                yield
            finally:
                order.append("exit")
                if exit_error:
                    raise RuntimeError("teardown ws-private")

        app = SimpleNamespace(run=run, app_id="ap-test")
        endpoint = SimpleNamespace(
            get_web_url=lambda: "https://private-label.modal.run/"
        )
        modal = SimpleNamespace(
            Workspace=SimpleNamespace(
                from_context=lambda: SimpleNamespace(proxy_tokens=manager)
            ),
            Environment=SimpleNamespace(
                from_context=lambda: SimpleNamespace(
                    name="test-environment", hydrate=lambda: None
                )
            ),
        )

        async def client(url, pcm, *, headers):
            self.assertEqual(
                headers,
                {"Modal-Key": token.token_id, "Modal-Secret": token.token_secret},
            )
            self.assertEqual(url, "wss://private-label.modal.run/")
            order.append("client")
            if client_error:
                raise client_error
            return {"status": "completed", "server_done": {}}

        return modal, lambda *a, **k: (app, endpoint), client, order

    def test_registration_rejects_relaxed_auth_budget_claims_and_pcm(self):
        original = network.read_registration()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / network.MANIFEST_PATH
            target.parent.mkdir(parents=True)
            for key, value in (
                ("proxy_auth_required", False),
                ("sample_count", 1),
                ("claims", {}),
                ("modal_sdk", "other"),
                ("execution", {**original["execution"], "configured_retries": 1}),
            ):
                changed = {**original, key: value}
                target.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    network.read_registration(root)

    def test_paths_reject_traversal_windows_devices_and_overwrite(self):
        for name in ("", "../x", "a/b", "a\\b", "CON", "nul", "LPT9", "x" * 49):
            with self.subTest(name=name), self.assertRaises(ValueError):
                network._paths(network.ROOT, name, True)
        with tempfile.TemporaryDirectory() as temp:
            output, journal = network._paths(Path(temp), "valid-id", True)
            network._journal(journal, "start", first=True)
            with self.assertRaises(FileExistsError):
                network._journal(journal, "start", first=True)
            network._write_result(output, {"status": "failed"})
            with self.assertRaises(FileExistsError):
                network._write_result(output, {})

    def test_endpoint_url_never_accepts_credentials_redirect_targets_or_queries(self):
        self.assertEqual(
            network._websocket_url("https://x.modal.run/path"), "wss://x.modal.run/path"
        )
        for url in (
            "http://x.modal.run",
            "https://evil.test",
            "https://x.modal.run.evil.test",
            "https://user:password@x.modal.run",
            "https://x.modal.run/?token=x",
            "https://x.modal.run/#x",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                network._websocket_url(url)

    def test_source_binding_requires_clean_commit_and_exact_file_bytes(self):
        with patch.object(
            network.subprocess, "check_output", return_value=" M source.py"
        ):
            with self.assertRaises(ValueError):
                network.source_snapshot()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = "src/whisper_runtime/test.py"
            (root / name).parent.mkdir(parents=True)
            (root / name).write_bytes(b"pass\n")
            with (
                patch.object(network, "_source_paths", return_value=[name]),
                patch.object(
                    network.subprocess, "check_output", side_effect=["", "a" * 40]
                ),
            ):
                snapshot = network.source_snapshot(root)
            network._verify_source(snapshot, root)
            (root / name).write_bytes(b"changed\n")
            with self.assertRaises(ValueError):
                network._verify_source(snapshot, root)

    def test_proxy_token_cleanup_order_for_scoped_and_unscoped_tokens(self):
        for scoped in (False, True):
            modal, resources, client, order = self.fake_modal(scoped=scoped)
            with patch.object(
                network.importlib,
                "import_module",
                return_value=SimpleNamespace(replay=client),
            ):
                record = network._attempt(
                    modal=modal,
                    snapshot=self.snapshot,
                    pcm=self.pcm,
                    preflight=False,
                    resource_factory=resources,
                )
            self.assertEqual(record["status"], "completed")
            self.assertEqual(
                order,
                [
                    "create",
                    *(["allow"] if scoped else []),
                    "enter",
                    "client",
                    "exit",
                    "delete",
                ],
            )
            self.assertTrue(all(record["cleanup"].values()))
            serialized = json.dumps(record)
            for private in ("wk-private", "ws-private", "private-label"):
                self.assertNotIn(private, serialized)

    def test_client_error_or_cancel_stops_context_then_revokes_token(self):
        for error in (
            RuntimeError("https://private-label.modal.run/?ws-private"),
            KeyboardInterrupt(),
        ):
            modal, resources, client, order = self.fake_modal(client_error=error)
            with patch.object(
                network.importlib,
                "import_module",
                return_value=SimpleNamespace(replay=client),
            ):
                record = network._attempt(
                    modal=modal,
                    snapshot=self.snapshot,
                    pcm=self.pcm,
                    preflight=False,
                    resource_factory=resources,
                )
            self.assertEqual(
                record["status"],
                "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            )
            self.assertEqual(order[-2:], ["exit", "delete"])
            self.assertTrue(all(record["cleanup"].values()))
            self.assertNotIn("ws-private", json.dumps(record))

    def test_startup_allow_teardown_and_deletion_failures_never_pass(self):
        for field in ("enter_error", "exit_error", "delete_error", "allow_error"):
            modal, resources, client, order = self.fake_modal(**{field: True})
            with patch.object(
                network.importlib,
                "import_module",
                return_value=SimpleNamespace(replay=client),
            ):
                record = network._attempt(
                    modal=modal,
                    snapshot=self.snapshot,
                    pcm=self.pcm,
                    preflight=False,
                    resource_factory=resources,
                )
            self.assertEqual(record["status"], "failed")
            self.assertEqual(order[-1], "delete")
            self.assertNotIn("ws-private", json.dumps(record))

    def test_preflight_gate_rejects_missing_cleanup_and_false_auth_or_source(self):
        original = self.preflight_record()
        self.assertTrue(network._preflight_valid(original, self.snapshot, self.pcm))
        for field in (
            "missing_cleanup",
            "token_not_deleted",
            "unauthorized_accepted",
            "wrong_hash",
            "wrong_source",
            "not_preflight",
        ):
            changed = copy.deepcopy(original)
            if field == "missing_cleanup":
                changed["cleanup"] = {}
            elif field == "token_not_deleted":
                changed["cleanup"]["proxy_token_deleted"] = False
            elif field == "unauthorized_accepted":
                changed["result"]["unauthorized_status"] = 101
            elif field == "wrong_hash":
                changed["result"]["sha256"] = "0" * 64
            elif field == "wrong_source":
                changed["source"]["commit"] = "b" * 40
            else:
                changed["preflight"] = False
            self.assertFalse(network._preflight_valid(changed, self.snapshot, self.pcm))

    def test_paid_gate_runs_before_modal_or_attempt_journal(self):
        with patch.object(network, "_attempt") as attempt:
            with self.assertRaises(ValueError):
                network.run(replay_id="no-authority")
            attempt.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch.object(network, "read_registration", return_value={}),
                patch.object(network, "source_snapshot", return_value=self.snapshot),
                patch.object(network, "_input", return_value=self.pcm),
                patch.object(network, "_attempt") as attempt,
            ):
                with self.assertRaises(FileNotFoundError):
                    network.run(
                        replay_id="no-preflight", root=root, confirm_paid_gpu=True
                    )
                attempt.assert_not_called()
                self.assertFalse(
                    network._paths(root, "no-preflight", False)[1].exists()
                )

    def test_validation_failure_preserves_received_record_and_blocks_rerun(self):
        record = self.preflight_record()
        record["result"]["server_done"]["source_snapshot"] = {"wrong": True}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch.object(network, "read_registration", return_value={}),
                patch.object(network, "source_snapshot", return_value=self.snapshot),
                patch.object(network, "_input", return_value=self.pcm),
                patch.object(network, "_attempt", return_value=record) as attempt,
                patch.object(
                    network.importlib,
                    "import_module",
                    return_value=SimpleNamespace(__version__="1.5.5"),
                ),
            ):
                output = network.run(replay_id="preserve", root=root, preflight=True)
                saved = json.loads(output.read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["error_code"], "worker_validation_failed")
                self.assertEqual(
                    saved["result"]["server_done"]["source_snapshot"], {"wrong": True}
                )
                with self.assertRaises(FileExistsError):
                    network.run(replay_id="preserve", root=root, preflight=True)
                self.assertEqual(attempt.call_count, 1)

    def test_cpu_echo_validates_binary_sequence_digest_and_never_fakes_asr(self):
        pcm = self.pcm[:32000]
        digest = hashlib.sha256(pcm).hexdigest()

        async def exercise(corrupt=False):
            incoming = asyncio.Queue()
            incoming.put_nowait({"type": "websocket.connect"})
            incoming.put_nowait(
                {
                    "type": "websocket.receive",
                    "text": json.dumps(
                        {
                            "type": "start",
                            "protocol": "pcm-echo/v1",
                            "sample_count": 16000,
                            "sha256": digest,
                        }
                    ),
                }
            )
            for sequence in range(50):
                frame = (
                    network.HEADER.pack(sequence + int(corrupt), sequence * 320)
                    + pcm[sequence * 640 : (sequence + 1) * 640]
                )
                incoming.put_nowait({"type": "websocket.receive", "bytes": frame})
            incoming.put_nowait(
                {
                    "type": "websocket.receive",
                    "text": json.dumps(
                        {
                            "type": "eof",
                            "samples": 16000,
                            "chunks": 50,
                            "sha256": digest,
                        }
                    ),
                }
            )
            output = []

            async def send(message):
                output.append(message)

            with patch.object(network, "_verify_source"):
                await network.make_echo_app(self.snapshot, digest)(
                    {"type": "websocket"}, incoming.get, send
                )
            return output

        output = asyncio.run(exercise())
        self.assertEqual(sum("bytes" in item for item in output), 50)
        done = json.loads(output[-2]["text"])
        self.assertEqual((done["status"], done["sha256"]), ("completed", digest))
        self.assertNotIn("event", " ".join(item.get("text", "") for item in output))
        self.assertEqual(
            asyncio.run(exercise(True))[-1], {"type": "websocket.close", "code": 1008}
        )

    def test_gpu_receive_capture_is_bounded_ordered_and_one_connection_only(self):
        captured = []

        def make_app(factory, **options):
            self.assertEqual(
                options,
                {
                    "expected_samples": network.SAMPLES,
                    "expected_sha256": network.PCM_SHA256,
                },
            )

            async def handler(scope, receive, send):
                factory()
                for _ in range(4):
                    await receive()

            return handler

        async def exercise():
            with (
                patch.object(network, "_verify_source"),
                patch.object(
                    network.importlib,
                    "import_module",
                    return_value=SimpleNamespace(make_app=make_app),
                ),
                patch.object(
                    network,
                    "_native_factory",
                    side_effect=lambda snapshot, data: captured.append(data),
                ),
            ):
                app = network.make_gpu_app(self.snapshot)
                inputs = asyncio.Queue()
                good = network.HEADER.pack(0, 0) + b"\x01\x00" * 320
                for frame in (
                    good,
                    good,
                    network.HEADER.pack(2, 320) + b"\x02\x00" * 320,
                    network.HEADER.pack(1, 320) + b"\x03\x00" * 320,
                ):
                    inputs.put_nowait({"type": "websocket.receive", "bytes": frame})
                output = []

                async def send(message):
                    output.append(message)

                await app({"type": "websocket"}, inputs.get, send)
                await app({"type": "websocket"}, inputs.get, send)
                return output

        output = asyncio.run(exercise())
        self.assertEqual(len(captured), 1)
        self.assertEqual(bytes(captured[0]), b"\x01\x00" * 320 + b"\x03\x00" * 320)
        self.assertEqual(output, [{"type": "websocket.close", "code": 1008}])

    def test_resource_definition_uses_proxy_gate_no_retries_and_readonly_cache(self):
        for preflight in (True, False):
            image = Mock()
            for method in (
                "apt_install",
                "pip_install",
                "uv_pip_install",
                "run_commands",
                "add_local_dir",
                "add_local_file",
                "env",
            ):
                getattr(image, method).return_value = image
            options, auth = {}, {}

            class App:
                def __init__(self, name):
                    self.name = name

                def function(self, **kwargs):
                    options.update(kwargs)
                    return lambda function: function

            def asgi_app(**kwargs):
                auth.update(kwargs)
                return lambda function: function

            volume = SimpleNamespace(with_mount_options=Mock(return_value="readonly"))
            modal = SimpleNamespace(
                Image=SimpleNamespace(debian_slim=lambda **kw: image),
                App=App,
                asgi_app=asgi_app,
                Volume=SimpleNamespace(from_name=Mock(return_value=volume)),
            )
            corpus = SimpleNamespace(
                BASE_COMMIT="a" * 40,
                _helper=lambda name: SimpleNamespace(
                    DIRECT_IMAGE_PACKAGES=(), _build_command=lambda commit: "cached"
                ),
                b=SimpleNamespace(
                    MODEL_CACHE_NAME="cache", MODEL_CACHE_MOUNT="/models"
                ),
            )
            with patch.object(network, "_corpus", return_value=corpus):
                _, endpoint = network._resources(
                    modal, self.snapshot, preflight=preflight, pcm=self.pcm
                )
            self.assertFalse(inspect.iscoroutinefunction(endpoint))
            self.assertEqual(len(inspect.signature(endpoint).parameters), 0)
            self.assertIs(auth["requires_proxy_auth"], True)
            for key, value in {
                "min_containers": 0,
                "max_containers": 1,
                "buffer_containers": 0,
                "single_use_containers": True,
                "startup_timeout": 180,
            }.items():
                self.assertEqual(options[key], value)
            self.assertNotIn("retries", options)
            self.assertEqual(options["timeout"], 30 if preflight else 180)
            self.assertEqual("gpu" in options, not preflight)
            if not preflight:
                volume.with_mount_options.assert_called_once_with(read_only=True)

    def test_invalid_resource_definition_cannot_create_a_proxy_token(self):
        modal, _, _, order = self.fake_modal()
        resources = Mock(side_effect=ValueError("invalid ASGI settings"))
        record = network._attempt(
            modal=modal,
            snapshot=self.snapshot,
            pcm=self.pcm,
            preflight=True,
            resource_factory=resources,
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(order, [])

    def test_pinned_sdk_accepts_cpu_and_gpu_asgi_definitions_without_running(self):
        try:
            modal = importlib.import_module("modal")
        except ImportError:
            self.skipTest("Modal SDK is not installed")
        if str(modal.__version__) != network.SDK_VERSION:
            self.skipTest("The registered Modal SDK is not installed")
        snapshot = {"files": [{"path": network.MANIFEST_PATH}]}
        with (
            patch.object(modal.App, "run", side_effect=AssertionError("remote run")),
            patch.object(
                modal.Workspace,
                "from_context",
                side_effect=AssertionError("credential lookup"),
            ),
        ):
            for preflight in (True, False):
                with self.subTest(preflight=preflight):
                    app, endpoint = network._resources(
                        modal,
                        snapshot,
                        preflight=preflight,
                        pcm=self.pcm,
                    )
                    self.assertIsNotNone(app)
                    self.assertIsNotNone(endpoint)


if __name__ == "__main__":
    unittest.main()
