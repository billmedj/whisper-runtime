"""Offline controls and removable-hook checks for the bounded CPU verifier."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import verify_deferred_word_commits as verifier


class FakeModule:
    def __init__(self):
        self._forward_pre_hooks = {0: lambda *args: None}

    def register_forward_pre_hook(self, observer, *, with_kwargs):
        if not with_kwargs:
            raise AssertionError("the observer requires keyword-aware hooks")
        key = max(self._forward_pre_hooks, default=0) + 1
        self._forward_pre_hooks[key] = observer
        return SimpleNamespace(remove=lambda: self._forward_pre_hooks.pop(key))

    def forward(self, *inputs):
        for observer in self._forward_pre_hooks.values():
            observer(self, inputs, {})


class ForwardCounterTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(encoder=FakeModule(), decoder=FakeModule())

    def test_real_hook_inputs_are_counted_and_prior_hooks_remain(self):
        with verifier._count_forwards(self.model) as counts:
            self.model.encoder.forward(SimpleNamespace(shape=(1, 80, 3000)))
            self.model.decoder.forward(
                SimpleNamespace(shape=(1, 7)), SimpleNamespace(shape=(1, 1500, 384))
            )
            self.model.decoder.forward(
                SimpleNamespace(shape=(1, 1)), SimpleNamespace(shape=(1, 1500, 384))
            )
            self.assertFalse(counts["hooks_removed"])
        self.assertEqual(
            counts,
            dict(
                encoder_forwards=1,
                encoder_input_frames=3000,
                decoder_forwards=2,
                decoder_input_frames=3000,
                decoder_input_tokens=8,
                hooks_removed=True,
            ),
        )
        self.assertEqual(tuple(self.model.encoder._forward_pre_hooks), (0,))
        self.assertEqual(tuple(self.model.decoder._forward_pre_hooks), (0,))

    def test_worker_failure_still_removes_only_owned_hooks(self):
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            with verifier._count_forwards(self.model) as counts:
                raise RuntimeError("decode failed")
        self.assertTrue(counts["hooks_removed"])
        self.assertEqual(tuple(self.model.encoder._forward_pre_hooks), (0,))
        self.assertEqual(tuple(self.model.decoder._forward_pre_hooks), (0,))

    def test_second_registration_failure_removes_first_hook(self):
        with (
            patch.object(
                self.model.decoder,
                "register_forward_pre_hook",
                side_effect=RuntimeError("registration failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "registration failed"),
        ):
            with verifier._count_forwards(self.model):
                self.fail("the context must not start")
        self.assertEqual(tuple(self.model.encoder._forward_pre_hooks), (0,))
        self.assertEqual(tuple(self.model.decoder._forward_pre_hooks), (0,))


class VerifierCliTests(unittest.TestCase):
    def test_default_worker_profile_and_deadline_remain_unchanged(self):
        with (
            patch("sys.argv", ["verify", "--model", "existing.pt"]),
            patch.object(verifier.subprocess, "run") as run,
        ):
            verifier.main()
        command = run.call_args.args[0]
        for name, value in (
            ("--arm", "both"),
            ("--left-context-ms", "2000"),
            ("--word-context-limit-ms", "6000"),
            ("--case", "both"),
            ("--cell-timeout-seconds", "80"),
            ("--manifest", str(verifier.ROOT / ".tmp-native/manifest.json")),
        ):
            self.assertEqual(command[command.index(name) + 1], value)
        self.assertNotIn("--reuse-alignment-features", command)
        self.assertNotIn("--legacy-alignment", command)
        self.assertEqual(
            run.call_args.kwargs, dict(cwd=verifier.ROOT, timeout=240, check=True)
        )

    def test_explicit_control_flags_and_manifest_reach_child(self):
        with (
            patch(
                "sys.argv",
                [
                    "verify",
                    "--model",
                    "existing.pt",
                    "--manifest",
                    "optional.json",
                    "--arm",
                    "candidate",
                    "--reuse-alignment-features",
                    "--legacy-alignment",
                ],
            ),
            patch.object(verifier.subprocess, "run") as run,
        ):
            verifier.main()
        command = run.call_args.args[0]
        self.assertIn("--reuse-alignment-features", command)
        self.assertIn("--legacy-alignment", command)
        self.assertEqual(command[command.index("--arm") + 1], "candidate")
        self.assertEqual(
            command[command.index("--manifest") + 1],
            str(Path("optional.json").resolve()),
        )

    def test_worker_dispatch_uses_explicit_options(self):
        with (
            patch(
                "sys.argv",
                [
                    "verify",
                    "--worker",
                    "--model",
                    "existing.pt",
                    "--manifest",
                    "optional.json",
                    "--arm",
                    "candidate",
                    "--reuse-alignment-features",
                    "--legacy-alignment",
                    "--left-context-ms",
                    "20000",
                    "--word-context-limit-ms",
                    "24000",
                    "--case",
                    "continuous",
                    "--cell-timeout-seconds",
                    "140",
                ],
            ),
            patch.object(verifier, "run_worker") as worker,
        ):
            verifier.main()
        worker.assert_called_once_with(
            Path("existing.pt"),
            "candidate",
            manifest=Path("optional.json"),
            reuse_alignment_features=True,
            legacy_alignment=True,
            case="continuous",
            cell_timeout_seconds=140,
            left_context_ms=20000,
            word_context_limit_ms=24000,
        )

    def test_control_without_factory_opt_in_refused_before_launch(self):
        with (
            patch(
                "sys.argv", ["verify", "--model", "existing.pt", "--legacy-alignment"]
            ),
            patch.object(verifier.subprocess, "run") as run,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            verifier.main()
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()
        with self.assertRaisesRegex(ValueError, "feature-reuse opt-in"):
            verifier.run_worker(Path("existing.pt"), "candidate", legacy_alignment=True)

    def test_cell_deadline_and_case_bounds_fail_before_launch(self):
        for arguments in (
            ["--cell-timeout-seconds", "0"],
            ["--cell-timeout-seconds", "141"],
            ["--case", "unknown"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("sys.argv", ["verify", "--model", "existing.pt", *arguments]),
                patch.object(verifier.subprocess, "run") as run,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                verifier.main()
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()
        for seconds in (0, 141, True, "80"):
            with (
                self.subTest(seconds=seconds),
                self.assertRaisesRegex(ValueError, "cell timeout"),
            ):
                verifier.run_worker(
                    Path("existing.pt"), "candidate", cell_timeout_seconds=seconds
                )
        with self.assertRaisesRegex(ValueError, "case must"):
            verifier.run_worker(Path("existing.pt"), "candidate", case="unknown")


if __name__ == "__main__":
    unittest.main()
