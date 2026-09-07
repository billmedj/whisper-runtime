"""Request-local, bounded greedy draft verification with no eager Torch import.

Draft IDs are proposals, not prompt context or accepted output. The unchanged
backend step loop applies its filters and selects each token. Only that selected
token can unlock the corresponding causally computed current-audio logits row.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TypedDict


class _DraftStats(TypedDict):
    proposed_tokens: int
    effective_draft_tokens: int
    context_cropped_tokens: int
    matched_tokens: int
    mismatch_index: int | None
    crop_length: int | None
    parallel_prefills: int
    saved_row_calls: int
    ordinary_suffix_forwards: int
    cleanup_calls: int
    fallback_reason: str | None


def _trim_self_cache(inference: Any, length: int) -> None:
    """Validate the entire crop before modifying only explicit self-attention KV."""
    if type(length) is not int or length < 1:
        raise ValueError("cache prefix length must be a positive integer")
    modules = tuple(inference.kv_modules)
    if not modules or any(module not in inference.kv_cache for module in modules):
        raise RuntimeError("draft self-attention cache is incomplete")
    if any(inference.kv_cache[module].shape[1] < length for module in modules):
        raise RuntimeError("cannot extend a draft cache by truncating it")
    for module in modules:
        inference.kv_cache[module] = inference.kv_cache[module][:, :length].detach()


class VerifiedDraftInference:
    """Wrap exactly one fresh, request-local, single-sequence greedy inference.

    Callers retain the standard prefill/step/filter/finalize implementation and
    own terminal cleanup, including after errors. Each logits call must advance
    the token cursor by exactly one; this wrapper cannot be reused for a new run.
    """

    _use_legacy_cache = False

    def __init__(self, original: Any, draft_tokens: tuple[int, ...]) -> None:
        if getattr(original, "_use_legacy_cache", True):
            raise ValueError("draft verification requires request-local caching")
        if type(draft_tokens) is not tuple:
            raise TypeError("draft_tokens must be a tuple of integer token IDs")
        if len(draft_tokens) > 32:
            raise ValueError("draft_tokens must contain at most 32 token IDs")
        if any(type(token) is not int for token in draft_tokens):
            raise TypeError("draft token IDs must be integers, not booleans")
        if draft_tokens:
            vocabulary = original.model.dims.n_vocab
            if type(vocabulary) is not int or vocabulary < 1:
                raise ValueError("draft verification requires a valid vocabulary size")
            if any(token < 0 or token >= vocabulary for token in draft_tokens):
                raise ValueError("draft token ID is outside the model vocabulary")

        self.original = original
        self.draft = draft_tokens
        self.rows: Any = None
        self.initial_length: int | None = None
        self.started = False
        self.audio_features: Any = None
        self._next_length: int | None = None
        self._closed = False
        self._cleanup_done = False
        self.stats: _DraftStats = dict(
            proposed_tokens=len(draft_tokens),
            effective_draft_tokens=0,
            context_cropped_tokens=0,
            matched_tokens=0,
            mismatch_index=None,
            crop_length=None,
            parallel_prefills=0,
            saved_row_calls=0,
            ordinary_suffix_forwards=0,
            cleanup_calls=0,
            fallback_reason=None,
        )

    def logits(self, tokens: Any, audio_features: Any) -> Any:
        if self._closed:
            raise RuntimeError("draft inference has been cleaned up")
        if len(tokens.shape) != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError("draft verification requires one nonempty greedy sequence")
        length = int(tokens.shape[1])
        if not self.started:
            if self.original.kv_cache or getattr(self.original, "hooks", ()):
                raise ValueError("draft prefill requires a fresh request-local cache")
            if length != self.original.initial_token_length:
                raise ValueError("draft prefill must start at the initial token cursor")
            self.started = True
            self.initial_length = length
            self._next_length = length + 1
            self.audio_features = audio_features
            available = max(0, int(self.original.model.dims.n_text_ctx) - length)
            self.draft = self.draft[:available]
            self.stats["effective_draft_tokens"] = len(self.draft)
            self.stats["context_cropped_tokens"] = self.stats["proposed_tokens"] - len(
                self.draft
            )
            if not self.draft:
                self.stats["fallback_reason"] = (
                    "context_full" if self.stats["proposed_tokens"] else "empty_draft"
                )
                return self._ordinary_logits(tokens, audio_features)

            torch = import_module("torch")
            draft = torch.tensor([self.draft], device=tokens.device, dtype=tokens.dtype)
            proposed = torch.cat((tokens, draft), dim=1)
            self.rows = self.original.model.decoder(
                proposed,
                audio_features,
                kv_cache=self.original.kv_cache,
                _update_kv_cache=True,
            )
            self.stats["parallel_prefills"] += 1
            # Preserve every initial row for the backend's SOT/no-speech lookup.
            # Filters mutate returned logits, so never return a speculative view.
            return self.rows[:, :length].clone()

        if audio_features is not self.audio_features:
            raise RuntimeError(
                "current audio features changed during draft verification"
            )
        if length != self._next_length:
            raise RuntimeError("unexpected token cursor during draft verification")
        self._next_length = length + 1
        if self.rows is not None:
            assert self.initial_length is not None
            generated = length - self.initial_length
            if generated <= len(self.draft):
                if int(tokens[0, -1]) == self.draft[generated - 1]:
                    self.stats["matched_tokens"] += 1
                    self.stats["saved_row_calls"] += 1
                    index = self.initial_length + generated - 1
                    return self.rows[:, index : index + 1].clone()
                # Exclude the correcting token: ordinary inference consumes it.
                crop_length = length - 1
                _trim_self_cache(self.original, crop_length)
                self.stats["mismatch_index"] = generated - 1
                self.stats["crop_length"] = crop_length
            # Exhaustion needs no crop: all speculative self KV was accepted.
            self.rows = None
        return self._ordinary_logits(tokens, audio_features)

    def _ordinary_logits(self, tokens: Any, audio_features: Any) -> Any:
        self.stats["ordinary_suffix_forwards"] += 1
        return self.original.logits(tokens, audio_features)

    def cleanup_caching(self) -> None:
        """Release rows/audio even if backend cleanup fails; allow cleanup retry."""
        self._closed = True
        self.rows = None
        self.audio_features = None
        self.draft = ()
        if not self._cleanup_done:
            self.stats["cleanup_calls"] += 1
            self.original.cleanup_caching()
            self._cleanup_done = True

    def rearrange_kv_cache(self, _indices: Any) -> None:
        raise RuntimeError("beam search is outside greedy draft verification")
