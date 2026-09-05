"""Pure local tests for deterministic, bounded acoustic stress recipes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import prepare_acoustic_cases as acoustic


def _pack(samples):
    return struct.pack("<" + "h" * len(samples), *samples)


def _unpack(pcm):
    return [sample[0] for sample in struct.iter_unpack("<h", pcm)]


class AcousticCasesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "cached-assets"
        self.assets.mkdir()
        self.manifest = json.loads(
            (acoustic.ROOT / acoustic.MANIFEST_PATH).read_text(encoding="utf-8")
        )
        self.source = {}
        pattern = [0, 0, 1, -1, 31, -31, 64, -64, 1000, -1000, 32767, -32768]
        for index, fixture in enumerate(self.manifest["fixtures"]):
            values = (pattern * 40)[: 321 + index * 19]
            pcm = _pack(values)
            fixture["sample_count"] = len(values)
            fixture["pcm_sha256"] = hashlib.sha256(pcm).hexdigest()
            self.source[fixture["id"]] = pcm
            (self.assets / fixture["filename"]).write_bytes(pcm)
        original = copy.deepcopy(
            next(
                c for c in self.manifest["cases"] if c["id"] == acoustic.SOURCE_CASE_ID
            )
        )
        chunks = []
        for part in original["parts"]:
            if "silence_ms" in part:
                part["silence_ms"] = 2
                chunks.append(bytes(64))
            else:
                chunks.append(self.source[part["fixture_id"]])
        self.mixed = b"".join(chunks)
        original.update(
            sample_count=len(self.mixed) // 2,
            pcm_sha256=hashlib.sha256(self.mixed).hexdigest(),
        )
        self.manifest["cases"] = [original]
        self.manifest_path = self.root / acoustic.MANIFEST_PATH
        self.manifest_path.parent.mkdir()
        self._save_manifest()

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def _build(self):
        return acoustic.build_cases(self.root, self.assets)

    def test_determinism_metadata_hashes_references_and_bounds(self):
        first, pcm = self._build()
        second, second_pcm = self._build()
        self.assertEqual(first, second)
        self.assertEqual(pcm, second_pcm)
        self.assertEqual(tuple(pcm), acoustic.CASE_IDS)
        self.assertEqual(len(first["cases"]), 4)
        self.assertTrue(first["synthetic"])
        self.assertFalse(first["natural_no_pause_speech"])
        self.assertFalse(first["recognition_improvement_claim"])
        self.assertLessEqual(first["total_samples"], 175 * 16_000)
        self.assertEqual(first["total_samples"], sum(len(p) // 2 for p in pcm.values()))
        self.assertEqual(
            first["source_manifest"]["sha256"],
            hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        )
        reference = self.manifest["cases"][0]["reference_text"]
        for case in first["cases"]:
            self.assertEqual(case["reference_text"], reference)
            self.assertEqual(
                case["pcm_sha256"], hashlib.sha256(pcm[case["id"]]).hexdigest()
            )
            self.assertEqual(
                case["input_sha256"], hashlib.sha256(self.mixed).hexdigest()
            )
            self.assertEqual(case["sample_count"], len(pcm[case["id"]]) // 2)
            self.assertLessEqual(case["sample_count"], acoustic.MAX_CASE_SAMPLES)
            self.assertEqual(
                case["signal_stats"], acoustic.signal_stats(pcm[case["id"]])
            )
        json.dumps(first, allow_nan=False)

    def test_control_and_concat_preserve_every_fixture_sample_and_order(self):
        metadata, pcm = self._build()
        expected = b"".join(
            self.source[part["fixture_id"]]
            for part in self.manifest["cases"][0]["parts"]
            if "fixture_id" in part
        )
        self.assertEqual(pcm[acoustic.CASE_IDS[0]], self.mixed)
        self.assertEqual(pcm[acoustic.CASE_IDS[1]], expected)
        self.assertEqual(len(self.mixed) - len(expected), 5 * 2 * 32)
        control, concat = metadata["cases"][:2]
        self.assertEqual(
            control["signal_stats"]["sum_squares_s16"],
            concat["signal_stats"]["sum_squares_s16"],
        )
        self.assertEqual(
            control["signal_stats"]["zero_samples"]
            - concat["signal_stats"]["zero_samples"],
            160,
        )
        self.assertTrue(concat["recipe"]["fixture_internal_silence_preserved"])
        self.assertFalse(concat["recipe"]["natural_continuous_speech"])
        self.assertEqual(concat["recipe"]["removed_silence_samples"], 160)

    def test_noise_matches_recorded_integer_recipe_without_truncation(self):
        metadata, pcms = self._build()
        source = _unpack(self.mixed)
        noisy = _unpack(pcms[acoustic.CASE_IDS[2]])
        state, clipping = acoustic.NOISE_SEED, 0
        expected = []
        for sample in source:
            state = (1664525 * state + 1013904223) % (2**32)
            addition = 64 if state >= 2**31 else -64
            value = sample + addition
            clipped = min(32767, max(-32768, value))
            clipping += clipped != value
            expected.append(clipped)
        self.assertEqual(noisy, expected)
        self.assertEqual(len(noisy), len(source))
        self.assertEqual(metadata["cases"][2]["clipping_count"], clipping)
        self.assertGreater(clipping, 0)
        zero_noisy, zero_clipped = acoustic._add_noise(bytes(640 * 2))
        self.assertEqual(set(map(abs, _unpack(zero_noisy))), {64})
        self.assertEqual(zero_clipped, 0)
        self.assertEqual(acoustic.signal_stats(zero_noisy)["quiet_frame_count"], 0)

    def test_attenuation_rounds_toward_zero_and_never_truncates_time(self):
        values = [
            -32768,
            -65,
            -64,
            -63,
            -33,
            -32,
            -31,
            -1,
            0,
            1,
            31,
            32,
            33,
            63,
            64,
            65,
            32767,
        ]
        attenuated = _unpack(acoustic._attenuate(_pack(values)))
        self.assertEqual(attenuated, [math.trunc(value / 32) for value in values])
        metadata, pcm = self._build()
        self.assertEqual(len(pcm[acoustic.CASE_IDS[3]]), len(self.mixed))
        self.assertEqual(metadata["cases"][3]["clipping_count"], 0)
        self.assertLess(
            metadata["cases"][3]["signal_stats"]["rms_s16"],
            metadata["cases"][0]["signal_stats"]["rms_s16"],
        )

    def test_signal_stats_are_exact_and_account_for_partial_frame(self):
        stats = acoustic.signal_stats(_pack([-32768, 0, 32767]))
        self.assertEqual(stats["sample_count"], 3)
        self.assertEqual(stats["sum_s16"], -1)
        self.assertEqual(stats["sum_squares_s16"], 32768**2 + 32767**2)
        self.assertEqual(stats["peak_abs_s16"], 32768)
        self.assertEqual(stats["zero_samples"], 1)
        self.assertEqual(stats["full_scale_samples"], 2)
        self.assertEqual(stats["frame_count"], 1)
        self.assertTrue(stats["partial_tail_included"])
        quiet = acoustic.signal_stats(bytes(321 * 2))
        self.assertEqual(quiet["quiet_frame_count"], 2)
        self.assertEqual(quiet["rms_s16"], 0)
        self.assertIsNone(quiet["rms_dbfs"])
        for invalid in (b"", b"x", "not bytes"):
            with self.assertRaises(ValueError):
                acoustic.signal_stats(invalid)

    def test_asset_digest_and_length_mismatches_are_rejected(self):
        path = self.assets / self.manifest["fixtures"][0]["filename"]
        registered = path.read_bytes()
        path.write_bytes(bytes([registered[0] ^ 1]) + registered[1:])
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self._build()
        path.write_bytes(registered[:-2])
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            self._build()

    def test_registered_mixture_hash_and_reference_are_required(self):
        original = self.manifest["cases"][0]
        original["pcm_sha256"] = "0" * 64
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "differs from registration"):
            self._build()
        original["pcm_sha256"] = hashlib.sha256(self.mixed).hexdigest()
        original["reference_text"] = "A DIFFERENT REFERENCE"
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "differs from registration"):
            self._build()

    def test_total_and_individual_audio_bounds_are_checked_before_transforms(self):
        for silence_ms, expected in ((12000, "175-second"), (120001, "input bound")):
            for part in self.manifest["cases"][0]["parts"]:
                if "silence_ms" in part:
                    part["silence_ms"] = silence_ms
            self._save_manifest()
            with (
                self.subTest(silence_ms=silence_ms),
                mock.patch.object(
                    acoustic,
                    "_add_noise",
                    side_effect=AssertionError("must fail first"),
                ),
                self.assertRaisesRegex(ValueError, expected),
            ):
                self._build()

    def test_build_is_read_only_and_never_invokes_network_or_processes(self):
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("network forbidden"),
            ),
            mock.patch(
                "subprocess.run", side_effect=AssertionError("process forbidden")
            ),
        ):
            self._build()
        after = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        self.assertEqual(before, after)


class FrozenAcousticCorpusTests(unittest.TestCase):
    @unittest.skipUnless(
        (acoustic.ROOT / acoustic.ASSET_PATH / "6930-75918-0000.pcm").is_file(),
        "frozen PCM cache is not available locally",
    )
    def test_existing_three_clip_cache_has_frozen_outputs_and_measured_stress(self):
        record, pcm = acoustic.build_cases()
        self.assertEqual(record["total_duration_seconds"], 164.64)
        self.assertEqual(
            [case["sample_count"] for case in record["cases"]],
            [698560, 538560, 698560, 698560],
        )
        self.assertEqual(
            [case["pcm_sha256"] for case in record["cases"]],
            [
                "b6f8c541b55d2da4c9494dda2e5daebf8674c29286a3ea983206b9a99e9badad",
                "f170f383aaefea4057c1d9a8ee2b105f6192b9aaa5b2995a9f0c01fac86f21aa",
                "b98ad164a8fbe46c1f7832154c1422b0733cda0ec8fc8e69bb0fc597fb4946e2",
                "681023b9857c442f8a26db6bb132d8d75cf77945e095c8dc399d17886df68bec",
            ],
        )
        self.assertEqual([case["clipping_count"] for case in record["cases"]], [0] * 4)
        self.assertEqual(record["cases"][2]["signal_stats"]["quiet_frame_count"], 0)
        self.assertGreater(
            record["cases"][3]["signal_stats"]["quiet_frame_count"],
            record["cases"][0]["signal_stats"]["quiet_frame_count"],
        )
        self.assertEqual(
            len(set(case["reference_text"] for case in record["cases"])), 1
        )
        self.assertEqual(
            sum(len(data) // 2 for data in pcm.values()), record["total_samples"]
        )


if __name__ == "__main__":
    unittest.main()
