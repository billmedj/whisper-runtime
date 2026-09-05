"""Bounded rolling audio with conservative, timestamped prefix commits.

Input admission is thread-safe. One owner drives model work and consumes events.
If a prefix cannot be resolved within the window, processing stops explicitly;
the stream never evicts uncommitted audio to make room.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock, current_thread
from typing import Callable

from ..errors import TransactionRetainedError
from ..state import AudioSpan, RequestState, Session, SessionState
from .native_result import (
    NativeTimestampSegment,
    NativeWindowResult,
    select_native_publication,
)
from .native_stream import (
    AudioBufferFullError,
    AudioSequenceError,
    NativeStreamError,
    StreamEventKind,
    TranscriptEvent,
)
from .native_whisper import NativeDecodeOptions, NativeWhisperAdapter, NativeWindowRun
from .stream_policy import compare_hypotheses
from .word_policy import (
    AlignedPublication,
    NativeWordAlignment,
    WordAgreementDecision,
    compare_word_hypotheses,
)

CONTINUOUS_PROFILE = "timestamp_agreement_stream/v1"
CONTEXT_CONTINUOUS_PROFILE = "context_agreement_stream/v1"
COALESCED_CONTINUOUS_PROFILE = "coalesced_timestamp_agreement_stream/v1"
COALESCED_CONTEXT_CONTINUOUS_PROFILE = "coalesced_context_agreement_stream/v1"
WORD_CONTINUOUS_PROFILE = "word_agreement_stream/v1"
COALESCED_WORD_CONTINUOUS_PROFILE = "coalesced_word_agreement_stream/v1"
_SAMPLES_PER_MS = 16


class StreamNeedsResolutionError(NativeStreamError):
    """The bounded window contains an unresolved prefix; accepted PCM is retained."""


@dataclass(frozen=True, slots=True)
class ContinuousStreamConfig:
    preview_interval_ms: int = 1_000
    max_window_ms: int = 30_000
    max_buffer_ms: int = 40_000
    holdback_ms: int = 1_000
    timestamp_tolerance_ms: int = 200
    left_context_ms: int = 0
    coalesce_previews: bool = False
    word_alignment: bool = False

    def __post_init__(self) -> None:
        for name in ("coalesce_previews", "word_alignment"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in self.__dataclass_fields__:
            if name in ("coalesce_previews", "word_alignment"):
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if not 0 < self.preview_interval_ms < self.max_window_ms <= 30_000:
            raise ValueError("require 0 < preview_interval_ms < max_window_ms <= 30000")
        if not self.max_window_ms <= self.max_buffer_ms <= 120_000:
            raise ValueError("max_buffer_ms must be between max_window_ms and 120000")
        if self.holdback_ms >= self.max_window_ms:
            raise ValueError("holdback_ms must be less than max_window_ms")
        if self.left_context_ms + self.preview_interval_ms >= self.max_window_ms:
            raise ValueError("left context must leave room for growing analyses")
        if self.left_context_ms % 20:
            raise ValueError("left_context_ms must be a multiple of 20 ms")


@dataclass(frozen=True, slots=True)
class ContinuousStreamMetrics:
    accepted_samples: int
    committed_samples: int
    buffered_samples: int
    peak_buffered_samples: int
    decode_count: int
    decoded_source_samples: int
    events_emitted: int
    accepted_chunks: int


@dataclass(frozen=True, slots=True)
class ContinuousDecodeTrace:
    """Latest prepared decision, not proof of publication or resource release.

    The caller may persist these immutable values. Only one record is retained;
    raw segment times describe the analysis, not the emitted event boundary.
    ``action`` describes the intended operation. Inspect events and run state to
    determine whether it committed, failed, or awaits resource recovery.
    """

    decode_index: int
    analysis_start_sample: int
    analysis_end_sample: int
    committed_before_sample: int
    retained_from_sample: int
    eof: bool
    reason: str
    publication_span: AudioSpan | None
    result: NativeWindowResult
    action: str
    word_alignment: NativeWordAlignment | None = None
    word_publication: AlignedPublication | None = None


class ContinuousTranscriptStream:
    """Experimental mono 16 kHz s16le stream; no hidden threads or event queue.

    ``push`` and ``finish_input`` may run on a producer thread. ``step`` returns
    bounded event batches to its owner. The caller must consume or persist them;
    the runtime retains only four session records, not a complete transcript.
    Pause keeps native resources until the transaction deadline. Cancellation of
    a decode does not discard its input; ``close`` abandons the whole stream.
    Opt-in preview coalescing skips obsolete endpoints under backlog. Its
    hypotheses depend on input arrival timing and can differ from the default.
    Word alignment is also opt-in; it binds whole-word output to explicit
    processed coverage. Both options preserve admitted input across retries.
    """

    def __init__(
        self,
        adapter: NativeWhisperAdapter,
        *,
        stream_id: str,
        mel_builder: Callable[[bytes], object],
        options: NativeDecodeOptions | None = None,
        rng_seed: int = 0,
        config: ContinuousStreamConfig | None = None,
    ) -> None:
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("stream_id must not be empty")
        if not callable(mel_builder):
            raise TypeError("mel_builder must be callable")
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise TypeError("rng_seed must be an integer")
        if config is not None and not isinstance(config, ContinuousStreamConfig):
            raise TypeError("config must be ContinuousStreamConfig or None")
        if options is not None and not isinstance(options, NativeDecodeOptions):
            raise TypeError("options must be NativeDecodeOptions or None")
        self._config = config or ContinuousStreamConfig()
        self._options = options or NativeDecodeOptions(without_timestamps=False)
        if self._options.without_timestamps:
            raise ValueError("continuous transcription requires timestamp tokens")
        if self.config.word_alignment and self._options.task != "transcribe":
            raise ValueError("word-aligned streaming supports transcription only")
        self._adapter = adapter
        self._model = adapter.model_identity
        self._mel_builder = mel_builder
        self._seed = rng_seed
        self._id = stream_id
        self._owner = current_thread()
        self._lock = RLock()
        self._session = Session(f"{stream_id}:session", history_limit=4)
        self._audio = bytearray()
        self._accepted = self._head = self._chunk = self._peak = 0
        self._retained = 0
        self._eof = self._done = False
        self._unresolved_eof = False
        self._next_endpoint = self.config.preview_interval_ms * _SAMPLES_PER_MS
        self._last_endpoint = 0
        self._previous: NativeWindowResult | None = None
        self._word_previous: NativeWordAlignment | None = None
        self._word_anchor: tuple[NativeTimestampSegment, ...] = ()
        self._word_pending: WordAgreementDecision | None = None
        self._retry_analysis: tuple[int, int, bool] | None = None
        self._run: NativeWindowRun | None = None
        self._run_end = 0
        self._run_start = 0
        self._run_window_id = ""
        self._run_final = False
        self._pending_state: SessionState | None = None
        self._pending_end = 0
        self._pending_final = False
        self._sequence = self._segment = self._revision = 0
        self._decode_count = self._decoded_samples = 0
        self._trace_count = 0
        self._last_trace: ContinuousDecodeTrace | None = None

    @property
    def config(self) -> ContinuousStreamConfig:
        return self._config

    @property
    def profile_id(self) -> str:
        if self.config.word_alignment:
            return (
                COALESCED_WORD_CONTINUOUS_PROFILE
                if self.config.coalesce_previews
                else WORD_CONTINUOUS_PROFILE
            )
        if self.config.coalesce_previews:
            return (
                COALESCED_CONTEXT_CONTINUOUS_PROFILE
                if self.config.left_context_ms
                else COALESCED_CONTINUOUS_PROFILE
            )
        return (
            CONTEXT_CONTINUOUS_PROFILE
            if self.config.left_context_ms
            else CONTINUOUS_PROFILE
        )

    @property
    def retained_from_sample(self) -> int:
        with self._lock:
            return self._retained

    @property
    def last_trace(self) -> ContinuousDecodeTrace | None:
        """Inspect the latest decode decision on the model-work owner thread."""
        self._require_owner()
        return self._last_trace

    @property
    def expected_chunk(self) -> int:
        with self._lock:
            return self._chunk

    @property
    def accepted_samples(self) -> int:
        with self._lock:
            return self._accepted

    @property
    def input_finished(self) -> bool:
        with self._lock:
            return self._eof

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def ready(self) -> bool:
        with self._lock:
            return not self._done and (
                self.active
                or self._eof
                or self._accepted
                >= min(
                    self._next_endpoint,
                    self._retained + self.config.max_window_ms * _SAMPLES_PER_MS,
                )
            )

    @property
    def state(self) -> SessionState:
        return self._session.snapshot()

    @property
    def metrics(self) -> ContinuousStreamMetrics:
        with self._lock:
            return ContinuousStreamMetrics(
                self._accepted,
                self._head,
                len(self._audio) // 2,
                self._peak,
                self._decode_count,
                self._decoded_samples,
                self._sequence,
                self._chunk,
            )

    def push(self, sequence_number: int, pcm_s16le: bytes) -> int:
        """Admit one chunk atomically, or reject it without consuming its sequence."""
        with self._lock:
            if self._eof or self._done:
                raise NativeStreamError("input is closed")
            if isinstance(sequence_number, bool) or not isinstance(
                sequence_number, int
            ):
                raise TypeError("sequence_number must be an integer")
            if sequence_number != self._chunk:
                raise AudioSequenceError(f"expected PCM chunk {self._chunk}")
            if not isinstance(pcm_s16le, bytes):
                raise TypeError("pcm_s16le must be bytes")
            if not pcm_s16le or len(pcm_s16le) % 2:
                raise ValueError("PCM must contain one or more complete 16-bit samples")
            limit = self.config.max_buffer_ms * _SAMPLES_PER_MS
            if len(self._audio) // 2 + len(pcm_s16le) // 2 > limit:
                raise AudioBufferFullError(
                    "audio buffer is full; retry after processing"
                )
            self._audio.extend(pcm_s16le)
            self._accepted += len(pcm_s16le) // 2
            self._chunk += 1
            self._peak = max(self._peak, len(self._audio) // 2)
            return self._accepted

    def finish_input(self) -> bool:
        with self._lock:
            if self._eof:
                return False
            self._eof = True
            return True

    def cancel_active(self) -> bool:
        run = self._run
        return False if run is None else run.cancel()

    def stop_active(self) -> bool:
        run = self._run
        return False if run is None else run.stop()

    def step(self) -> tuple[TranscriptEvent, ...]:
        """Start, advance or resolve one native operation, on the owner thread."""
        self._require_owner()
        if not self.ready:
            return ()
        run = self._run
        if run is not None:
            if run.closed:
                if not run.capacity_released:
                    raise NativeStreamError(
                        "native capacity is retained; recover before retry"
                    )
                self._run = None
                if self._pending_state is not None:
                    return self._publish_commit()
                if self._unresolved_eof:
                    self._record_decode()
                    self._retry_analysis = None
                return ()
            try:
                if not run.complete:
                    run.step()
                    return ()
                return self._resolve(run)
            except TransactionRetainedError as error:
                self._pending_state = error.committed_state
                raise
            except BaseException:
                if run.capacity_released:
                    self._run = None
                raise
        if self._unresolved_eof:
            raise StreamNeedsResolutionError(
                "EOF boundary is unresolved; close the stream before retrying "
                "with a different policy"
            )
        with self._lock:
            if self._eof and self._head == self._accepted:
                self._audio.clear()
                self._retained = self._head
                self._done = True
                return (self._final_event(),)
            bound = self._retained + self.config.max_window_ms * _SAMPLES_PER_MS
            final = self._eof and self._accepted <= bound
            endpoint = self._accepted if final else min(self._next_endpoint, bound)
            if self.config.coalesce_previews and not final:
                ceiling = bound
                if self._previous is None:
                    # Reserve a later observation at this origin. Preserve the
                    # scheduled minimum even when the interval exceeds half a
                    # window; it is already strictly below the hard bound.
                    ceiling = max(
                        endpoint,
                        bound - self.config.preview_interval_ms * _SAMPLES_PER_MS,
                    )
                endpoint = min(self._accepted, ceiling)
            start = self._retained
            if self._retry_analysis is not None:
                start, endpoint, final = self._retry_analysis
            if not final and endpoint <= self._last_endpoint:
                raise StreamNeedsResolutionError(
                    "no stable contiguous prefix within the audio window; input retained"
                )
            if (
                self.config.coalesce_previews or self.config.word_alignment
            ) and self._retry_analysis is None:
                # Admission is immutable even if preprocessing/startup fails,
                # or more PCM/EOF arrives while a failed run is recovered.
                self._retry_analysis = (start, endpoint, final)
            pcm = bytes(memoryview(self._audio)[: (endpoint - start) * 2])
        window_id = f"{self._id}:window:{start}:{endpoint}"
        self._run = self._adapter.start_window(
            session=self._session,
            request=RequestState(
                f"{window_id}:request",
                self._session.session_id,
                self._model,
                rng_seed=self._seed,
            ),
            window_id=window_id,
            mel=self._mel_builder(pcm),
            start_ms=start // _SAMPLES_PER_MS,
            end_ms=endpoint // _SAMPLES_PER_MS,
            options=self._options,
        )
        self._run_start, self._run_end, self._run_final = start, endpoint, final
        self._run_window_id = window_id
        return ()

    def _resolve(self, run: NativeWindowRun) -> tuple[TranscriptEvent, ...]:
        result = run.prepare_result()
        if (
            result.window_id != self._run_window_id
            or result.analyzed_span.start_ms != self._run_start // _SAMPLES_PER_MS
            or result.analyzed_span.end_ms != self._run_end // _SAMPLES_PER_MS
        ):
            run.close()
            raise NativeStreamError("native result does not match the admitted audio")
        if self.config.word_alignment:
            return self._resolve_words(run, result)
        decision = compare_hypotheses(
            self._previous,
            result,
            committed_through_ms=self._head // _SAMPLES_PER_MS,
            holdback_ms=self.config.holdback_ms,
            timestamp_tolerance_ms=self.config.timestamp_tolerance_ms,
        )
        span = decision.publication_span
        reason = decision.reason.value
        if self._run_final:
            reason = "eof"
            if self._run_start < self._head:
                span = self._context_span(result, final=True)
                if span is None:
                    self._trace(result, None, "eof_unresolved", "unresolved")
                    self._unresolved_eof = True
                    run.close()
                    self._run = None
                    self._record_decode()
                    self._retry_analysis = None
                    raise StreamNeedsResolutionError(
                        "EOF has no exact contiguous suffix after retained context; "
                        "input retained"
                    )
            else:
                span = None  # Preserve the default native full-result EOF contract.
        self._trace(
            result, span, reason, "commit" if self._run_final or span else "preview"
        )
        if self._run_final or span is not None:
            end = self._run_end
            if not self._run_final:
                assert span is not None
                end = span.end_ms * _SAMPLES_PER_MS
            self._pending_end, self._pending_final = end, self._run_final
            self._pending_state = run.finish(
                committed_through_ms=end // _SAMPLES_PER_MS,
                publication_span=span,
            )
            self._run = None
            return self._publish_commit()
        run.close()  # Fence before exposing a provisional result or retaining it.
        self._previous = result
        text = result.text
        if self._run_start < self._head:
            preview_span = self._context_span(result, final=False)
            text = (
                select_native_publication(result, preview_span).text
                if preview_span is not None
                else ""
            )
        return self._publish_preview(text)

    def _resolve_words(
        self, run: NativeWindowRun, result: NativeWindowResult
    ) -> tuple[TranscriptEvent, ...]:
        alignment = run.prepare_word_alignment()
        if alignment.native != result:
            run.close()
            raise NativeStreamError("word alignment does not match the native result")
        decision = compare_word_hypotheses(
            self._word_previous,
            alignment,
            committed_through_ms=self._head // _SAMPLES_PER_MS,
            anchor=self._word_anchor,
            holdback_ms=self.config.holdback_ms,
            timestamp_tolerance_ms=self.config.timestamp_tolerance_ms,
            final=self._run_final,
        )
        publication = decision.publication
        self._trace(
            result,
            None,
            decision.reason,
            "commit"
            if publication is not None
            else "unresolved"
            if self._run_final
            else "preview",
            word_alignment=alignment,
            word_publication=publication,
        )
        if publication is not None:
            self._word_pending = decision
            self._pending_end = (
                self._run_end
                if self._run_final
                else publication.end_ms * _SAMPLES_PER_MS
            )
            self._pending_final = self._run_final
            self._pending_state = run.finish(
                committed_through_ms=self._pending_end // _SAMPLES_PER_MS,
                aligned_publication=publication,
            )
            self._run = None
            return self._publish_commit()
        if self._run_final:
            self._unresolved_eof = True
            run.close()
            self._run = None
            self._retry_analysis = None
            self._record_decode()
            raise StreamNeedsResolutionError(
                f"EOF word alignment is unresolved ({decision.reason}); input retained"
            )
        run.close()
        self._word_previous = alignment
        self._previous = result
        # Provisional text uses estimated positions only. It has no commit authority.
        text = "".join(
            word.text
            for word in alignment.words
            if word.span.start_ms >= self._head // _SAMPLES_PER_MS
        ).strip()
        return self._publish_preview(text)

    def _publish_preview(self, text: str) -> tuple[TranscriptEvent, ...]:
        """Advance the preview cursor only after native cleanup has succeeded."""
        self._run = None
        self._retry_analysis = None
        self._record_decode()
        self._last_endpoint = self._run_end
        self._next_endpoint = self._run_end + self.config.preview_interval_ms * 16
        return (self._text_event(text, self._head, self._run_end),)

    def _context_span(
        self, result: NativeWindowResult, *, final: bool
    ) -> AudioSpan | None:
        """Select only complete, contiguous new segments; never clip a straddle."""
        metadata = result.metadata
        if metadata is None or (final and not metadata.timestamps_complete):
            return None
        start = end = self._head // _SAMPLES_PER_MS
        for segment in metadata.segments:
            if segment.span.end_ms <= start:
                continue
            if (
                segment.span.start_ms != end
                or segment.span.end_ms <= end
                or segment.span.end_ms > result.analyzed_span.end_ms
                or not segment.tokens
                or not segment.text.strip()
            ):
                break
            end = segment.span.end_ms
        if end <= start or (final and end != result.analyzed_span.end_ms):
            return None
        return AudioSpan(start, end)

    def _trace(
        self,
        result: NativeWindowResult,
        span: AudioSpan | None,
        reason: str,
        action: str,
        *,
        word_alignment: NativeWordAlignment | None = None,
        word_publication: AlignedPublication | None = None,
    ) -> None:
        self._trace_count += 1
        self._last_trace = ContinuousDecodeTrace(
            self._trace_count,
            self._run_start,
            self._run_end,
            self._head,
            self._retained,
            self._run_final,
            reason,
            span,
            result,
            action,
            word_alignment,
            word_publication,
        )

    def _publish_commit(self) -> tuple[TranscriptEvent, ...]:
        state = self._pending_state
        assert state is not None
        if state != self._session.snapshot() or not state.windows:
            raise NativeStreamError("committed result does not belong to this session")
        result = state.windows[-1].result
        if self.config.word_alignment and (
            self._word_pending is None or self._word_pending.publication != result
        ):
            raise NativeStreamError("committed result does not match the word decision")
        record = state.windows[-1]
        end = self._pending_end
        if (
            result.window_id != self._run_window_id
            or record.request_id != f"{self._run_window_id}:request"
            or record.model != self._model
            or result.start_ms != self._head // 16
            or state.committed_through_ms != end // 16
        ):
            raise NativeStreamError(
                "committed result has an unexpected source boundary"
            )
        self._record_decode()
        text = self._text_event(result.text, self._head, end)
        self._sequence += 1
        commit = TranscriptEvent(
            self._sequence,
            StreamEventKind.COMMIT,
            segment_id=text.segment_id,
            revision=text.revision,
            start_sample=self._head,
            end_sample=end,
            sample_rate_hz=16_000,
            committed_through_sample=end,
            committed_through_ms=end // 16,
            session_version=state.version,
        )
        with self._lock:
            retained = (
                end
                if self._pending_final
                else max(self._retained, end - self.config.left_context_ms * 16)
            )
            del self._audio[: (retained - self._retained) * 2]
            self._retained = retained
            self._head = end
            self._done = self._pending_final
        # Establish a fresh growing pair after rebasing. Starting immediately at
        # the old endpoint can exhaust a full window before a second observation.
        self._last_endpoint = end
        interval = self.config.preview_interval_ms * 16
        self._next_endpoint = min(
            self._run_end + interval,
            self._retained + self.config.max_window_ms * 16 - interval,
        )
        self._previous = None
        self._word_previous = None
        if self._word_pending is not None:
            self._word_anchor = self._word_pending.next_anchor
            self._word_pending = None
        self._pending_state = None
        self._retry_analysis = None
        self._segment += 1
        self._revision = 0
        return (text, commit, self._final_event()) if self._done else (text, commit)

    def _record_decode(self) -> None:
        self._decode_count += 1
        self._decoded_samples += self._run_end - self._run_start

    def _text_event(self, text: str, start: int, end: int) -> TranscriptEvent:
        previous = self._revision
        self._revision += 1
        self._sequence += 1
        return TranscriptEvent(
            self._sequence,
            StreamEventKind.REPLACE if previous else StreamEventKind.PROVISIONAL,
            segment_id=f"{self._id}:segment:{self._segment}",
            revision=self._revision,
            start_sample=start,
            end_sample=end,
            sample_rate_hz=16_000,
            text=text,
            supersedes_revision=previous or None,
            session_version=self.state.version,
        )

    def _final_event(self) -> TranscriptEvent:
        self._sequence += 1
        return TranscriptEvent(
            self._sequence,
            StreamEventKind.FINAL,
            session_version=self.state.version,
        )

    def close(self) -> bool:
        self._require_owner()
        if self.done:
            return False
        if self._run is not None:
            self._run.close()
            if not self._run.capacity_released:
                raise NativeStreamError(
                    "native capacity is retained; recover before closing"
                )
            self._run = None
        with self._lock:
            self._retry_analysis = None
            self._eof = self._done = True
        return True

    def __enter__(self) -> ContinuousTranscriptStream:
        return self

    def __exit__(self, *exc: object) -> None:
        if exc and isinstance(exc[1], TransactionRetainedError):
            return  # Preserve recovery authority; never replace the original error.
        self.close()

    def _require_owner(self) -> None:
        if current_thread() is not self._owner:
            raise NativeStreamError("model work must run on the creating thread")
