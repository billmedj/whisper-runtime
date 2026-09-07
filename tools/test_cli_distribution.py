from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from check_distribution import check_wheel


class CliDistributionTests(unittest.TestCase):
    def wheel(self, root, declaration, *, omit=None, version="1.2.3a4"):
        path = Path(root) / "runtime.whl"
        # Synthetic metadata must not be coupled to this checkout's release.
        prefix = f"whisper_execution_runtime-{version}.dist-info/"
        files = {
            "whisper_runtime/__init__.py": "",
            "whisper_runtime/py.typed": "",
            "whisper_runtime/cli.py": "",
            "whisper_runtime/captions.py": "",
            "whisper_runtime/audio_source.py": "",
            "whisper_runtime/native_setup.py": "",
            "whisper_runtime/profiles.py": "",
            "whisper_runtime/remote.py": "",
            "whisper_runtime/remote_transport.py": "",
            "whisper_runtime/pcm_live.py": "",
            prefix + "METADATA": (
                "License-Expression: Apache-2.0 AND MIT\nRequires-Python: >=3.10\n"
            ),
        }
        for name in (
            "LICENSE",
            "NOTICE",
            "THIRD_PARTY_NOTICES.md",
            "patches/openai-whisper/LICENSE",
        ):
            files[prefix + "licenses/" + name] = "test license"
        if declaration is not None:
            files[prefix + "entry_points.txt"] = declaration
        if omit is not None:
            del files[omit]
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in files.items():
                archive.writestr(name, value)
        return path

    def test_installed_command_and_modules_are_required(self):
        declaration = "[console_scripts]\nwhisper-runtime = whisper_runtime.cli:main\n"
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(check_wheel(self.wheel(root, declaration)), [])
            for module in (
                "native_setup",
                "profiles",
                "remote",
                "remote_transport",
                "pcm_live",
            ):
                with self.subTest(module=module):
                    filename = f"whisper_runtime/{module}.py"
                    failures = check_wheel(self.wheel(root, declaration, omit=filename))
                    self.assertIn(f"wheel is missing {filename}", failures)

    def test_missing_wrong_and_malformed_entrypoints_fail(self):
        for declaration in (None, "[console_scripts]\nother = other:main", "bad ini"):
            with (
                self.subTest(declaration=declaration),
                tempfile.TemporaryDirectory() as root,
            ):
                self.assertTrue(check_wheel(self.wheel(root, declaration)))

    def test_prerelease_and_stable_dist_info_names_are_supported(self):
        declaration = "[console_scripts]\nwhisper-runtime = whisper_runtime.cli:main\n"
        for version in ("0.1.0a1", "1.2.3", "2.0.0rc1"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as root:
                self.assertEqual(
                    check_wheel(self.wheel(root, declaration, version=version)), []
                )


if __name__ == "__main__":
    unittest.main()
