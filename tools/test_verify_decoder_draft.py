"""Lightweight CPU-only tests: no Torch/model/GPU imports required."""

import unittest
from types import SimpleNamespace

from tools.verify_decoder_draft import compare_results, draft_variants, trim_self_cache


class FakeTensor:
    def __init__(self, length):
        self.shape = (1, length, 2)

    def __getitem__(self, indices):
        return FakeTensor(indices[1].stop)

    def detach(self):
        return self


class DecoderDraftTests(unittest.TestCase):
    def test_drafts_preserve_raw_prefix_and_corruption_is_explicit(self):
        tokens = tuple(range(40))
        variants = draft_variants(tokens, 99)
        self.assertEqual([len(v) for v in variants.values()], [16, 32, 16, 8])
        self.assertEqual(variants["draft32"], tokens[:32])
        self.assertEqual(variants["corrupt_first"], (99, *tokens[1:16]))
        self.assertEqual(variants["truncated8"], tokens[:8])

    def test_short_or_invalid_source_is_rejected(self):
        for tokens in ((1,) * 31, (True,) * 40, (-1,) * 40):
            with self.assertRaises(ValueError):
                draft_variants(tokens, 99)

    def test_trim_changes_only_self_keys_and_values(self):
        cross = FakeTensor(1500)
        inference = SimpleNamespace(
            kv_modules=("self_k", "self_v"),
            kv_cache={
                "self_k": FakeTensor(35),
                "self_v": FakeTensor(35),
                "cross": cross,
            },
        )
        trim_self_cache(inference, 7)
        self.assertEqual(inference.kv_cache["self_k"].shape[1], 7)
        self.assertEqual(inference.kv_cache["self_v"].shape[1], 7)
        self.assertIs(inference.kv_cache["cross"], cross)
        with self.assertRaises(RuntimeError):
            trim_self_cache(inference, 8)

    def test_numeric_difference_is_not_called_exact_result(self):
        reference = dict(
            tokens=[1, 2],
            text="same",
            avg_logprob=-0.5,
            no_speech_prob=0.1,
            compression_ratio=1.0,
        )
        candidate = {**reference, "avg_logprob": -0.500001}
        compared = compare_results(reference, candidate)
        self.assertTrue(compared["exact_tokens"])
        self.assertTrue(compared["exact_text"])
        self.assertFalse(compared["exact_result"])
        self.assertNotEqual(compared["numeric_delta"]["avg_logprob"], 0)


if __name__ == "__main__":
    unittest.main()
