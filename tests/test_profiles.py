"""Frozen installed profile catalog: no model, device, or network prerequisites."""

import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

from whisper_runtime.profiles import DEFAULT_PROFILE, get_profile, list_profiles


class ProfileTests(unittest.TestCase):
    def test_catalog_is_versioned_exact_and_read_only(self):
        catalog = list_profiles()
        self.assertIsInstance(catalog, tuple)
        self.assertEqual(
            tuple(profile.name for profile in catalog),
            (
                "conservative-v1",
                "low-latency-v1",
                "low-latency-v2",
                "experimental-optimized-v1",
            ),
        )
        self.assertEqual(DEFAULT_PROFILE, "conservative-v1")
        self.assertEqual(
            tuple(profile.label for profile in catalog),
            (
                "Standard",
                "Low latency (experimental)",
                "Low latency v2 (experimental)",
                "Optimized (experimental)",
            ),
        )
        for profile in catalog:
            self.assertIs(get_profile(profile.name), profile)
            with self.assertRaises(FrozenInstanceError):
                profile.name = "changed"
            with self.assertRaises(FrozenInstanceError):
                profile.stream_config.max_draft_tokens = 12
            with self.assertRaises(FrozenInstanceError):
                profile.stream_config.endpointing.quiet_peak = 64

    def test_standard_definition_remains_the_conservative_cli_defaults(self):
        profile = get_profile(DEFAULT_PROFILE)
        self.assertFalse(profile.experimental)
        self.assertFalse(profile.reuse_alignment_features)
        self.assertEqual(profile.native_profile_id, "tiny.en/cli-fp32-v1")
        self.assertEqual(
            asdict(profile.stream_config),
            dict(
                preview_interval_ms=2000,
                max_window_ms=30000,
                max_buffer_ms=40000,
                holdback_ms=2000,
                timestamp_tolerance_ms=200,
                left_context_ms=2000,
                coalesce_previews=False,
                word_alignment=False,
                input_evidence=True,
                source_units=True,
                endpointing=dict(
                    frame_ms=20,
                    quiet_ms=600,
                    min_unit_ms=1000,
                    max_quiet_unit_ms=10000,
                    quiet_peak=32,
                ),
                word_boundary_fallback=True,
                word_context_limit_ms=6000,
                resolution_probe=False,
                eof_context_retry=True,
                defer_word_commits=True,
                max_draft_tokens=0,
                previous_holdback_ms=0,
            ),
        )

    def test_optimized_changes_only_named_experimental_settings(self):
        standard = get_profile(DEFAULT_PROFILE)
        optimized = get_profile("experimental-optimized-v1")
        self.assertTrue(optimized.experimental)
        self.assertTrue(optimized.reuse_alignment_features)
        self.assertNotEqual(optimized.native_profile_id, standard.native_profile_id)
        expected = asdict(standard.stream_config)
        expected.update(
            left_context_ms=20000, word_context_limit_ms=24000, max_draft_tokens=32
        )
        self.assertEqual(asdict(optimized.stream_config), expected)

    def test_low_latency_removes_deferral_without_drafts_or_looser_evidence(self):
        standard = get_profile(DEFAULT_PROFILE)
        live = get_profile("low-latency-v1")
        self.assertTrue(live.experimental)
        self.assertTrue(live.reuse_alignment_features)
        self.assertEqual(live.native_profile_id, "tiny.en/cli-low-latency-fp32-v1")
        expected = asdict(standard.stream_config)
        expected.update(
            left_context_ms=20000,
            word_context_limit_ms=24000,
            defer_word_commits=False,
        )
        self.assertEqual(asdict(live.stream_config), expected)
        self.assertEqual(live.stream_config.max_draft_tokens, 0)

    def test_unknown_and_invalid_names_do_not_fall_back(self):
        for name in ("", "latest", "standard", "CONSERVATIVE-V1", "conservative-v2"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unknown"):
                get_profile(name)
        for name in (None, 1, False, {}, []):
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, "string"):
                get_profile(name)

    def test_low_latency_v2_adds_only_previous_witness_holdback(self):
        older = get_profile("low-latency-v1")
        candidate = get_profile("low-latency-v2")
        self.assertTrue(candidate.experimental)
        self.assertTrue(candidate.reuse_alignment_features)
        self.assertEqual(candidate.native_profile_id, "tiny.en/cli-low-latency-fp32-v2")
        expected = asdict(older.stream_config)
        expected["previous_holdback_ms"] = 2000
        self.assertEqual(asdict(candidate.stream_config), expected)
        for profile in list_profiles():
            if profile.name != candidate.name:
                self.assertEqual(profile.stream_config.previous_holdback_ms, 0)

    def test_catalog_and_factory_import_without_site_packages_or_ml_modules(self):
        source = Path(__file__).resolve().parents[1] / "src"
        script = (
            f"import sys; sys.path.insert(0, {str(source)!r}); "
            "from whisper_runtime.profiles import list_profiles; "
            "from whisper_runtime.native_setup import create_stream; "
            "assert len(list_profiles()) == 4; "
            "assert not {'torch', 'whisper', 'numpy', 'sounddevice'} & sys.modules.keys()"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
