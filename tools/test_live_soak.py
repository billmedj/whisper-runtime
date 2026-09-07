import unittest

from tools.verify_live_soak import run


class LiveSoakTests(unittest.TestCase):
    def test_small_soak_uses_real_controller_and_releases_capacity(self):
        result = run(seconds=8)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["committed_samples"], 8 * 16_000)
        self.assertEqual(result["final_events"], 1)
        self.assertTrue(result["capacity_restored"])
        self.assertFalse(result["real_audio"])
        self.assertFalse(result["native_inference"])
        self.assertLessEqual(result["retained_publications"], 4)

    def test_invalid_scope_rejected(self):
        for args in ({"seconds": 0}, {"seconds": 14_401}, {"chunk_ms": 80}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                run(**args)
