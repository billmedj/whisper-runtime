"""Tiny causal decoder/cache tests; no model, backend import, or GPU required."""

import copy
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from whisper_runtime.adapters._draft_inference import VerifiedDraftInference


class Tensor:
    """Just the indexing/copy operations needed by the inference contract."""

    device = "cpu"
    dtype = "long"

    def __init__(self, values):
        self.values = copy.deepcopy(values)
        shape = []
        current = values
        while isinstance(current, (list, tuple)):
            shape.append(len(current))
            current = current[0] if current else None
        self.shape = tuple(shape)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        def select(values, indices):
            if not indices:
                return values
            first, *rest = indices
            selected = values[first]
            if isinstance(first, slice):
                return [select(value, rest) for value in selected]
            return select(selected, rest)

        values = select(self.values, indices)
        return Tensor(values) if isinstance(values, (list, tuple)) else values

    def clone(self):
        return Tensor(self.values)

    def detach(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)


class TinyTorch:
    @staticmethod
    def tensor(values, **_kwargs):
        return Tensor(values)

    @staticmethod
    def cat(tensors, dim):
        assert dim == 1
        return Tensor([sum((list(value.values[0]) for value in tensors), [])])


class Inference:
    """Causal rows depend on current audio and the actual cached token prefix."""

    _use_legacy_cache = False
    initial_token_length = 2

    def __init__(self, tensor_type=Tensor, context=16):
        self.tensor_type = tensor_type
        self.kv_modules = (object(), object())
        self.cross_modules = (object(), object())
        self.kv_cache = {}
        self.hooks = []
        self.calls = []
        self.cleanup_calls = 0
        self.fail_cleanup = False
        self.model = SimpleNamespace(
            dims=SimpleNamespace(n_vocab=16, n_text_ctx=context), decoder=self.decoder
        )

    def decoder(self, tokens, audio, *, kv_cache, _update_kv_cache):
        assert _update_kv_cache is True
        assert tokens.device == "cpu" or tokens.device.type == "cpu"
        cached = kv_cache.get(self.kv_modules[0])
        prefix = [] if cached is None else cached.tolist()[0]
        new_tokens = tokens.tolist()[0]
        self.calls.append((tuple(new_tokens), audio, len(prefix)))
        rows = []
        for token in new_tokens:
            prefix.append(token)
            rows.append([sum(prefix) + audio.value + index for index in range(16)])
        for module in self.kv_modules:
            kv_cache[module] = self.tensor_type([prefix])
        for module in self.cross_modules:
            if module not in kv_cache:
                kv_cache[module] = self.tensor_type([[audio.value] * 4])
        return self.tensor_type([rows])

    def logits(self, tokens, audio):
        if tokens.shape[1] > self.initial_token_length:
            tokens = tokens[:, -1:]
        return self.decoder(
            tokens, audio, kv_cache=self.kv_cache, _update_kv_cache=True
        )

    def cleanup_caching(self):
        self.cleanup_calls += 1
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")
        self.kv_cache = {}
        self.hooks = []


class DraftInferenceTests(unittest.TestCase):
    def setUp(self):
        self.audio = SimpleNamespace(value=40)
        self.original = Inference()
        self.prefix = Tensor([[1, 2]])
        self.torch_patch = patch(
            "whisper_runtime.adapters._draft_inference.import_module",
            return_value=TinyTorch,
        )
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def wrap(self, draft=(3, 4, 5)):
        wrapped = VerifiedDraftInference(self.original, draft)
        self.addCleanup(wrapped.cleanup_caching)
        return wrapped

    def assert_rows_equal(self, actual, expected):
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_constructor_rejects_invalid_proposals_before_forward(self):
        for invalid in ([3], None, (True,), (1.0,), ("1",)):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                VerifiedDraftInference(self.original, invalid)
        for invalid in ((-1,), (16,), tuple([1] * 33)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                VerifiedDraftInference(self.original, invalid)
        self.assertEqual(self.original.calls, [])
        self.assertEqual(self.original.kv_cache, {})
        self.wrap(tuple([0] * 32))
        self.wrap((15,))

    def test_legacy_cache_is_rejected(self):
        self.original._use_legacy_cache = True
        with self.assertRaisesRegex(ValueError, "request-local"):
            self.wrap()

    def test_prefill_uses_current_audio_and_preserves_initial_rows(self):
        baseline = Inference()
        expected = baseline.logits(self.prefix, self.audio)
        wrapped = self.wrap()
        result = wrapped.logits(self.prefix, self.audio)
        self.assert_rows_equal(result, expected)
        self.assertEqual(self.original.calls, [((1, 2, 3, 4, 5), self.audio, 0)])
        self.assertEqual(wrapped.stats["parallel_prefills"], 1)
        self.assertFalse(wrapped._use_legacy_cache)
        self.assertEqual(self.prefix.tolist(), [[1, 2]])
        result.values[0][1][0] = -999
        self.assertNotEqual(wrapped.rows.values[0][1][0], -999)

    def test_matching_rows_and_exhaustion_equal_ordinary_causal_decode(self):
        baseline = Inference()
        wrapped = self.wrap()
        self.assert_rows_equal(
            wrapped.logits(self.prefix, self.audio),
            baseline.logits(self.prefix, self.audio),
        )
        tokens = [1, 2]
        for token in (3, 4, 5, 8, 9):
            tokens.append(token)
            value = Tensor([tokens])
            self.assert_rows_equal(
                wrapped.logits(value, self.audio), baseline.logits(value, self.audio)
            )
        self.assertEqual(wrapped.stats["matched_tokens"], 3)
        self.assertEqual(wrapped.stats["saved_row_calls"], 3)
        self.assertEqual(wrapped.stats["ordinary_suffix_forwards"], 2)
        self.assertIsNone(wrapped.stats["crop_length"])
        self.assertIsNone(wrapped.rows)
        self.assertEqual(len(self.original.calls), 3)

    def test_each_mismatch_position_crops_only_self_cache(self):
        for mismatch in range(3):
            with self.subTest(mismatch=mismatch):
                self.original = Inference()
                baseline = Inference()
                wrapped = self.wrap()
                wrapped.logits(self.prefix, self.audio)
                baseline.logits(self.prefix, self.audio)
                cross_values = [
                    self.original.kv_cache[m] for m in self.original.cross_modules
                ]
                tokens = [1, 2]
                for token in (*wrapped.draft[:mismatch], 12, 13):
                    tokens.append(token)
                    value = Tensor([tokens])
                    self.assert_rows_equal(
                        wrapped.logits(value, self.audio),
                        baseline.logits(value, self.audio),
                    )
                self.assertEqual(wrapped.stats["mismatch_index"], mismatch)
                self.assertEqual(wrapped.stats["crop_length"], 2 + mismatch)
                self.assertEqual(
                    self.original.calls[1], ((12,), self.audio, 2 + mismatch)
                )
                self.assertEqual(wrapped.stats["saved_row_calls"], mismatch)
                for module, value in zip(self.original.cross_modules, cross_values):
                    self.assertIs(self.original.kv_cache[module], value)

    def test_filtered_choice_not_raw_argmax_controls_acceptance(self):
        wrapped = self.wrap((15, 4))
        logits = wrapped.logits(self.prefix, self.audio)
        self.assertEqual(logits.values[0][-1].index(max(logits.values[0][-1])), 15)
        # Model a standard filter suppressing the raw winner before selection.
        logits.values[0][-1][15] = -999
        selected = logits.values[0][-1].index(max(logits.values[0][-1]))
        wrapped.logits(Tensor([[1, 2, selected]]), self.audio)
        self.assertEqual(wrapped.stats["matched_tokens"], 0)
        self.assertEqual(wrapped.stats["mismatch_index"], 0)
        self.assertEqual(self.original.calls[1][0], (14,))

    def test_empty_and_full_context_fall_back_without_speculative_forward(self):
        for draft, context, reason in (
            ((), 16, "empty_draft"),
            ((3,), 2, "context_full"),
        ):
            with self.subTest(reason=reason):
                self.original = Inference(context=context)
                wrapped = self.wrap(draft)
                wrapped.logits(self.prefix, self.audio)
                self.assertEqual(self.original.calls[0][0], (1, 2))
                self.assertEqual(wrapped.stats["parallel_prefills"], 0)
                self.assertEqual(wrapped.stats["fallback_reason"], reason)

    def test_context_crop_uses_only_available_proposals(self):
        self.original = Inference(context=4)
        wrapped = self.wrap()
        wrapped.logits(self.prefix, self.audio)
        self.assertEqual(self.original.calls[0][0], (1, 2, 3, 4))
        self.assertEqual(wrapped.draft, (3, 4))
        self.assertEqual(wrapped.stats["effective_draft_tokens"], 2)
        self.assertEqual(wrapped.stats["context_cropped_tokens"], 1)

    def test_repeated_skipped_and_backward_cursors_are_rejected(self):
        for tokens in ([1, 2], [1, 2, 3, 4], [1]):
            with self.subTest(tokens=tokens):
                self.original = Inference()
                wrapped = self.wrap()
                wrapped.logits(self.prefix, self.audio)
                with self.assertRaisesRegex(RuntimeError, "cursor"):
                    wrapped.logits(Tensor([tokens]), self.audio)
                self.assertEqual(wrapped.stats["matched_tokens"], 0)
                self.assertEqual(len(self.original.calls), 1)
        wrapped.logits(Tensor([[1, 2, 3]]), self.audio)
        with self.assertRaisesRegex(RuntimeError, "cursor"):
            wrapped.logits(Tensor([[1, 2, 3]]), self.audio)
        self.assertEqual(wrapped.stats["matched_tokens"], 1)

    def test_changed_audio_and_beam_rearrangement_are_rejected(self):
        wrapped = self.wrap()
        wrapped.logits(self.prefix, self.audio)
        with self.assertRaisesRegex(RuntimeError, "audio"):
            wrapped.logits(Tensor([[1, 2, 3]]), SimpleNamespace(value=40))
        with self.assertRaisesRegex(RuntimeError, "beam"):
            wrapped.rearrange_kv_cache([0])

    def test_nonfresh_cache_batch_and_wrong_initial_cursor_are_rejected(self):
        wrapped = self.wrap()
        for value in (Tensor([[1, 2], [1, 2]]), Tensor([[1]]), Tensor([[]])):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                wrapped.logits(value, self.audio)
        self.original.kv_cache[object()] = Tensor([[1]])
        with self.assertRaisesRegex(ValueError, "fresh"):
            wrapped.logits(self.prefix, self.audio)
        self.assertEqual(self.original.calls, [])

    def test_bad_self_cache_does_not_partially_crop_other_entries(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                self.original = Inference()
                wrapped = self.wrap()
                wrapped.logits(self.prefix, self.audio)
                first, second = self.original.kv_modules
                first_value = self.original.kv_cache[first]
                if missing:
                    del self.original.kv_cache[second]
                else:
                    self.original.kv_cache[second] = Tensor([[1]])
                with self.assertRaises(RuntimeError):
                    wrapped.logits(Tensor([[1, 2, 12]]), self.audio)
                self.assertIs(self.original.kv_cache[first], first_value)
                self.assertEqual(first_value.shape[1], 5)
                self.assertEqual(len(self.original.calls), 1)

    def test_cleanup_releases_owned_tensors_and_is_idempotent(self):
        wrapped = self.wrap()
        wrapped.logits(self.prefix, self.audio)
        wrapped.cleanup_caching()
        wrapped.cleanup_caching()
        self.assertIsNone(wrapped.rows)
        self.assertIsNone(wrapped.audio_features)
        self.assertEqual(wrapped.draft, ())
        self.assertEqual(wrapped.stats["proposed_tokens"], 3)
        self.assertEqual(self.original.kv_cache, {})
        self.assertEqual(self.original.cleanup_calls, 1)
        self.assertEqual(wrapped.stats["cleanup_calls"], 1)
        with self.assertRaisesRegex(RuntimeError, "cleaned up"):
            wrapped.logits(self.prefix, self.audio)

    def test_cleanup_failure_still_releases_rows_audio_and_allows_retry(self):
        wrapped = self.wrap()
        wrapped.logits(self.prefix, self.audio)
        self.original.fail_cleanup = True
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            wrapped.cleanup_caching()
        self.assertIsNone(wrapped.rows)
        self.assertIsNone(wrapped.audio_features)
        self.original.fail_cleanup = False
        wrapped.cleanup_caching()
        self.assertEqual(self.original.kv_cache, {})

    def test_tiny_real_cpu_tensor_parity_without_a_model(self):
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            self.skipTest("optional CPU Torch is not installed")
        self.torch_patch.stop()
        original = Inference(tensor_type=torch.tensor)
        baseline = Inference(tensor_type=torch.tensor)
        wrapped = VerifiedDraftInference(original, (3, 4, 5))
        self.addCleanup(wrapped.cleanup_caching)
        for sequence in ([1, 2], [1, 2, 3], [1, 2, 3, 12], [1, 2, 3, 12, 13]):
            tokens = torch.tensor([sequence], device="cpu", dtype=torch.long)
            self.assert_rows_equal(
                wrapped.logits(tokens, self.audio), baseline.logits(tokens, self.audio)
            )
        self.assertEqual(wrapped.stats["saved_row_calls"], 1)
        self.assertEqual(wrapped.stats["crop_length"], 3)


if __name__ == "__main__":
    unittest.main()
