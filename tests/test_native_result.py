import gc
import unittest
import weakref
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

from whisper_runtime.adapters.native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
    build_native_window_result,
    select_native_publication,
)
from whisper_runtime.state import AudioSpan, WindowResult


class Tokenizer:
    eot = 100
    timestamp_begin = 200
    text = {1: " Hello", 2: " world", 3: " unfinished", 4: "!"}

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.text[token] for token in tokens)


class NativeResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = AudioSpan(5_000, 7_000)
        self.tokenizer = Tokenizer()

    def build(self, raw: object, **kwargs: object) -> NativeWindowResult:
        return build_native_window_result(
            raw,
            window_id="window-1",
            analysis_span=self.analysis,
            tokenizer=self.tokenizer,
            **kwargs,
        )

    def raw(self, **kwargs: object) -> SimpleNamespace:
        values = {
            "text": "Hello world",
            "language": "en",
            "tokens": [200, 1, 225, 225, 2, 250],
        }
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_legacy_text_result_keeps_original_bounds(self) -> None:
        result = self.build(SimpleNamespace(text=" legacy text "))
        self.assertIsInstance(result, WindowResult)
        self.assertIsNone(result.metadata)
        self.assertIsNone(result.publication_segment_indices)
        self.assertEqual(result.text, " legacy text ")
        self.assertEqual((result.start_ms, result.end_ms), (5_000, 7_000))
        self.assertEqual(result.analyzed_span, self.analysis)

    def test_raw_text_is_required(self) -> None:
        for raw in (object(), SimpleNamespace(text=None), SimpleNamespace(text=12)):
            with self.subTest(raw=raw), self.assertRaises(TypeError):
                self.build(raw)

    def test_any_metadata_requires_language_and_tokens(self) -> None:
        for attributes in (
            {"language": "en"},
            {"tokens": [1]},
            {"avg_logprob": None},
            {"no_speech_prob": float("nan")},
            {"language": None, "tokens": [1]},
            {"language": "  ", "tokens": [1]},
        ):
            with (
                self.subTest(attributes=attributes),
                self.assertRaises((TypeError, ValueError)),
            ):
                self.build(SimpleNamespace(text="Hello", **attributes))

    def test_metadata_and_segments_are_frozen_snapshots(self) -> None:
        raw = self.raw(avg_logprob=-0.5, no_speech_prob=0.1, temperature=0)
        result = self.build(raw)
        metadata = result.metadata
        self.assertIsNotNone(metadata)
        raw.tokens[:] = [3]
        raw.language = "fr"
        raw.avg_logprob = -2.0
        self.assertEqual(metadata.tokens, (200, 1, 225, 225, 2, 250))
        self.assertEqual(metadata.language, "en")
        self.assertEqual(metadata.avg_logprob, -0.5)
        self.assertEqual(metadata.segments[0].tokens, (1,))
        self.assertFalse(hasattr(result, "__dict__"))
        self.assertFalse(hasattr(metadata, "__dict__"))
        self.assertFalse(hasattr(metadata.segments[0], "__dict__"))
        for value, name in (
            (result, "text"),
            (metadata, "language"),
            (metadata.segments[0], "text"),
        ):
            with self.subTest(value=value), self.assertRaises(FrozenInstanceError):
                setattr(value, name, "changed")

    def test_metadata_constructor_copies_mutable_inputs(self) -> None:
        tokens = [1]
        segment = NativeTimestampSegment(AudioSpan(0, 20), "Hello", tokens)
        segments = [segment]
        metadata = NativeDecodeMetadata("en", tokens, segments=segments)
        tokens.append(2)
        segments.clear()
        self.assertEqual(segment.tokens, (1,))
        self.assertEqual(metadata.tokens, (1,))
        self.assertEqual(metadata.segments, (segment,))

    def test_does_not_read_or_retain_model_features(self) -> None:
        class UnreadableFeatures:
            text = "Hello"

            @property
            def audio_features(self) -> object:
                raise AssertionError("audio features must not be read")

        self.assertIsNone(self.build(UnreadableFeatures()).metadata)

        class Features:
            pass

        features = Features()
        reference = weakref.ref(features)
        raw = self.raw(audio_features=features)
        result = self.build(raw)
        del features, raw
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(hasattr(result.metadata, "audio_features"))

    def test_absent_and_nan_scores_become_none(self) -> None:
        metadata = self.build(self.raw()).metadata
        for name in (
            "avg_logprob",
            "no_speech_prob",
            "temperature",
            "compression_ratio",
        ):
            with self.subTest(name=name):
                self.assertIsNone(getattr(metadata, name))
                result = self.build(self.raw(**{name: float("nan")}))
                self.assertIsNone(getattr(result.metadata, name))

    def test_finite_scores_are_copied_as_floats(self) -> None:
        metadata = self.build(
            self.raw(
                avg_logprob=-1,
                no_speech_prob=1,
                temperature=0,
                compression_ratio=2,
            )
        ).metadata
        for name, expected in (
            ("avg_logprob", -1.0),
            ("no_speech_prob", 1.0),
            ("temperature", 0.0),
            ("compression_ratio", 2.0),
        ):
            self.assertEqual(getattr(metadata, name), expected)
            self.assertIsInstance(getattr(metadata, name), float)

    def test_invalid_scores_are_rejected(self) -> None:
        for name in (
            "avg_logprob",
            "no_speech_prob",
            "temperature",
            "compression_ratio",
        ):
            for value in (True, False, "0.2", float("inf"), float("-inf"), object()):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    self.build(self.raw(**{name: value}))
        for name, value in (
            ("no_speech_prob", -0.01),
            ("no_speech_prob", 1.01),
            ("temperature", -0.01),
            ("compression_ratio", -0.01),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.build(self.raw(**{name: value}))

    def test_invalid_tokens_are_rejected_without_accepting_tensor_like_values(
        self,
    ) -> None:
        for tokens in (None, "1", iter([1]), [True], [False], [1.0], [-1], [object()]):
            with (
                self.subTest(tokens=tokens),
                self.assertRaises((TypeError, ValueError)),
            ):
                self.build(self.raw(tokens=tokens))

    def test_complete_pairs_keep_nonzero_audio_offset_and_exact_text_spacing(
        self,
    ) -> None:
        result = self.build(self.raw())
        self.assertEqual(result.text, "Hello world")
        self.assertIsNone(result.publication_segment_indices)
        self.assertEqual((result.start_ms, result.end_ms), (5_000, 7_000))
        self.assertTrue(result.metadata.timestamps_complete)
        self.assertEqual(
            result.metadata.segments,
            (
                NativeTimestampSegment(AudioSpan(5_000, 5_500), " Hello", (1,)),
                NativeTimestampSegment(AudioSpan(5_500, 6_000), " world", (2,)),
            ),
        )

    def test_timestamp_free_or_untimed_prefix_never_invents_spans(self) -> None:
        for tokens in ([], [1, 2], [1, 200, 2, 225], [100, 200, 1, 225]):
            with self.subTest(tokens=tokens):
                result = self.build(self.raw(tokens=tokens))
                self.assertFalse(result.metadata.timestamps_complete)
                self.assertEqual(result.metadata.segments, ())

    def test_without_tokenizer_metadata_stays_untimed(self) -> None:
        result = build_native_window_result(
            self.raw(), window_id="window-1", analysis_span=self.analysis
        )
        self.assertEqual(result.metadata.tokens, (200, 1, 225, 225, 2, 250))
        self.assertEqual(result.metadata.segments, ())
        self.assertFalse(result.metadata.timestamps_complete)

    def test_unfinished_or_malformed_tail_keeps_only_complete_prefix(self) -> None:
        for tail in ([225], [225, 2], [2, 250], [225, 100, 250], [220, 2, 250]):
            with self.subTest(tail=tail):
                result = self.build(self.raw(tokens=[200, 1, 225, *tail]))
                self.assertFalse(result.metadata.timestamps_complete)
                self.assertEqual(len(result.metadata.segments), 1)
                self.assertEqual(result.metadata.segments[0].text, " Hello")
                self.assertEqual(result.text, "Hello world")

    def test_invalid_first_timestamp_pair_is_not_published(self) -> None:
        for tokens in (
            [250, 1, 225],
            [225, 1, 225],
            [200, 1, 301],
            [301, 1, 302],
            [200, 1, 1_701],
            [200, 1, 100],
            [200, 1, 199],
            [200, 225],
            [200],
            [200, 1],
        ):
            with self.subTest(tokens=tokens):
                result = self.build(self.raw(tokens=tokens))
                self.assertFalse(result.metadata.timestamps_complete)
                self.assertEqual(result.metadata.segments, ())

    def test_timestamp_range_is_bounded_even_if_analysis_is_longer(self) -> None:
        result = build_native_window_result(
            self.raw(text="Hello", tokens=[200, 1, 1_701]),
            window_id="window-1",
            analysis_span=AudioSpan(5_000, 45_000),
            tokenizer=self.tokenizer,
        )
        self.assertFalse(result.metadata.timestamps_complete)
        self.assertEqual(result.metadata.segments, ())

    def test_exact_end_of_analysis_and_max_timestamp_are_allowed(self) -> None:
        for span, closing_token in (
            (self.analysis, 300),
            (AudioSpan(0, 30_000), 1_700),
        ):
            with self.subTest(span=span):
                result = build_native_window_result(
                    self.raw(text="Hello", tokens=[200, 1, closing_token]),
                    window_id="window-1",
                    analysis_span=span,
                    tokenizer=self.tokenizer,
                )
                self.assertTrue(result.metadata.timestamps_complete)
                self.assertEqual(result.metadata.segments[0].span, span)

    def test_complete_timestamps_must_reconstruct_native_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.build(self.raw(text="Different transcript"))

    def test_selects_whole_segment_with_full_analysis_provenance(self) -> None:
        result = self.build(self.raw(), publication_span=AudioSpan(5_500, 6_000))
        self.assertEqual(result.text, "world")
        self.assertEqual((result.start_ms, result.end_ms), (5_500, 6_000))
        self.assertEqual(result.analyzed_span, self.analysis)
        self.assertEqual(result.metadata.tokens, (200, 1, 225, 225, 2, 250))
        self.assertEqual(len(result.metadata.segments), 2)
        self.assertEqual(result.publication_segment_indices, (1,))

    def test_selects_multiple_complete_segments_and_preserves_join_spacing(
        self,
    ) -> None:
        result = self.build(self.raw(), publication_span=AudioSpan(5_000, 6_000))
        self.assertEqual(result.text, "Hello world")
        self.assertEqual((result.start_ms, result.end_ms), (5_000, 6_000))
        self.assertEqual(result.publication_segment_indices, (0, 1))

    def test_separate_selection_preserves_metadata_identity_and_original_result(
        self,
    ) -> None:
        full = self.build(self.raw(avg_logprob=-0.5))
        selected = select_native_publication(full, AudioSpan(5_500, 6_000))
        self.assertIs(selected.metadata, full.metadata)
        self.assertIs(selected.analysis_span, full.analysis_span)
        self.assertEqual(selected.publication_segment_indices, (1,))
        self.assertEqual(selected.text, "world")
        self.assertEqual((selected.start_ms, selected.end_ms), (5_500, 6_000))
        self.assertEqual(selected.metadata.avg_logprob, -0.5)
        self.assertIsNone(full.publication_segment_indices)
        self.assertEqual(full.text, "Hello world")
        self.assertEqual((full.start_ms, full.end_ms), (5_000, 7_000))

    def test_reselection_uses_full_analysis_not_previous_publication_bounds(
        self,
    ) -> None:
        full = self.build(self.raw())
        first = select_native_publication(full, AudioSpan(5_000, 5_500))
        second = select_native_publication(first, AudioSpan(5_500, 6_000))
        self.assertEqual(second.publication_segment_indices, (1,))
        self.assertIs(second.metadata, full.metadata)
        self.assertEqual(second.analyzed_span, self.analysis)

    def test_selection_resolves_implicit_full_analysis_before_narrowing(self) -> None:
        full = replace(self.build(self.raw()), analysis_span=None)
        selected = select_native_publication(full, AudioSpan(5_500, 6_000))
        self.assertEqual(selected.analysis_span, self.analysis)
        self.assertEqual(selected.analyzed_span, full.analyzed_span)

    def test_explicit_indices_are_immutable_copies(self) -> None:
        selected = self.build(self.raw(), publication_span=AudioSpan(5_500, 6_000))
        indices = [1]
        copied = replace(selected, publication_segment_indices=indices)
        indices.clear()
        self.assertEqual(copied.publication_segment_indices, (1,))
        with self.assertRaises(FrozenInstanceError):
            copied.publication_segment_indices = (0,)

    def test_explicit_indices_must_be_valid_ordered_segment_indices(self) -> None:
        selected = self.build(self.raw(), publication_span=AudioSpan(5_000, 6_000))
        for indices in ([], [-1], [True], [1.0], [2], [0, 0], [1, 0], "0", iter([0])):
            with (
                self.subTest(indices=indices),
                self.assertRaises((TypeError, ValueError)),
            ):
                replace(selected, publication_segment_indices=indices)
        with self.assertRaises(ValueError):
            replace(selected, metadata=None)

    def test_explicit_indices_may_not_skip_an_intervening_segment(self) -> None:
        full = self.build(
            self.raw(
                text="Hello world!", tokens=[200, 1, 225, 225, 2, 250, 250, 4, 275]
            )
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            replace(
                full,
                text="Hello!",
                end_ms=6_500,
                publication_segment_indices=(0, 2),
            )

    def test_explicit_indices_must_match_published_text_and_bounds(self) -> None:
        selected = self.build(self.raw(), publication_span=AudioSpan(5_500, 6_000))
        for changes in (
            {"text": "Hello"},
            {"start_ms": 5_499},
            {"end_ms": 6_001},
            {"publication_segment_indices": (0,)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(selected, **changes)

    def test_separate_selection_requires_typed_native_result_and_span(self) -> None:
        with self.assertRaises(TypeError):
            select_native_publication(WindowResult("w", "text", 0, 1), AudioSpan(0, 1))
        with self.assertRaises(TypeError):
            select_native_publication(self.build(self.raw()), (5_000, 6_000))

    def test_selection_may_include_silence_between_complete_segments(self) -> None:
        result = self.build(
            self.raw(tokens=[200, 1, 225, 230, 2, 250]),
            publication_span=AudioSpan(5_000, 6_000),
        )
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.metadata.segments[1].span.start_ms, 5_600)

    def test_complete_prefix_can_be_selected_despite_unfinished_tail(self) -> None:
        result = self.build(
            self.raw(tokens=[200, 1, 225, 225, 2]),
            publication_span=AudioSpan(5_000, 5_500),
        )
        self.assertEqual(result.text, "Hello")
        self.assertFalse(result.metadata.timestamps_complete)
        self.assertEqual(result.metadata.tokens, (200, 1, 225, 225, 2))
        self.assertEqual(result.publication_segment_indices, (0,))

    def test_multiple_complete_segments_do_not_hide_unfinished_last_segment(
        self,
    ) -> None:
        tokens = [200, 1, 350, 350, 2, 550, 550, 3]
        result = build_native_window_result(
            self.raw(text="Hello world unfinished", tokens=tokens),
            window_id="window-1",
            analysis_span=AudioSpan(17_000, 26_000),
            tokenizer=self.tokenizer,
            publication_span=AudioSpan(17_000, 24_000),
        )
        self.assertEqual(result.text, "Hello world")
        self.assertFalse(result.metadata.timestamps_complete)
        self.assertEqual(len(result.metadata.segments), 2)
        self.assertEqual(result.metadata.tokens, tuple(tokens))

    def test_selection_rejects_straddled_segments_and_missing_boundaries(self) -> None:
        for span in (
            AudioSpan(5_001, 6_000),
            AudioSpan(5_000, 5_999),
            AudioSpan(5_001, 5_499),
            AudioSpan(5_000, 7_000),
            AudioSpan(6_000, 6_500),
            AudioSpan(4_999, 6_000),
            AudioSpan(5_000, 7_001),
        ):
            with self.subTest(span=span), self.assertRaises(ValueError):
                self.build(self.raw(), publication_span=span)

    def test_selection_requires_times_including_for_legacy_and_untimed_prefix(
        self,
    ) -> None:
        for raw in (
            SimpleNamespace(text="Hello"),
            self.raw(tokens=[1, 2]),
            self.raw(tokens=[1, 200, 2, 225]),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.build(raw, publication_span=AudioSpan(5_000, 5_500))

    def test_span_arguments_are_typed(self) -> None:
        with self.assertRaises(TypeError):
            self.build(self.raw(), publication_span=(5_000, 6_000))
        with self.assertRaises(TypeError):
            build_native_window_result(
                self.raw(), window_id="window-1", analysis_span=(5_000, 7_000)
            )

    def test_invalid_tokenizer_metadata_and_decode_text_are_rejected(self) -> None:
        for eot, timestamp_begin in ((True, 200), (100, True), (-1, 200), (200, 100)):
            tokenizer = Tokenizer()
            tokenizer.eot = eot
            tokenizer.timestamp_begin = timestamp_begin
            with (
                self.subTest(eot=eot, timestamp_begin=timestamp_begin),
                self.assertRaises(ValueError),
            ):
                build_native_window_result(
                    self.raw(),
                    window_id="window-1",
                    analysis_span=self.analysis,
                    tokenizer=tokenizer,
                )

        class NonStringTokenizer(Tokenizer):
            def decode(self, tokens: list[int]) -> object:
                return object()

        with self.assertRaises(TypeError):
            build_native_window_result(
                self.raw(),
                window_id="window-1",
                analysis_span=self.analysis,
                tokenizer=NonStringTokenizer(),
            )


if __name__ == "__main__":
    unittest.main()
