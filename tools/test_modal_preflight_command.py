"""Regression test for the remote qualification preflight command."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra import modal_native_cuda_qualification as producer  # noqa: E402


class ModalPreflightCommandTests(unittest.TestCase):
    def test_discovery_command_imports_the_remote_contract_tests(self) -> None:
        environment = os.environ.copy()
        python_path = [str(ROOT / "src"), str(ROOT)]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        result = subprocess.run(
            [
                sys.executable,
                *producer.PREFLIGHT_TEST_ARGUMENTS,
                "-k",
                "test_platform_modal_module_does_not_require_distribution_metadata",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
