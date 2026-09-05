"""Bounded rolling audio with conservative, timestamped prefix commits.

Input admission is thread-safe. One owner drives model work and consumes events.
If a prefix cannot be resolved within the window, processing stops explicitly;
the stream never evicts uncommitted audio to make room.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock, current_thread
from typing import Callable, Literal

from ..errors import TransactionRetainedError
from ..state import AudioSpan, RequestState, Session, SessionState
from .audio_endpoints import (
    QuietEndpointConfig,
    QuietEndpointDetector,
    QuietEndpointProposal,
)
from .audio_evidence import (
    AudioEvidenceDecision,
    AudioObservation,
    SilencePublication,
    assess_audio,
)
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
    AnchorDiagnostic,
    NativeWordAlignment,
    WordAgreementDecision,
    compare_word_hypotheses,
)

CONTINUOUS_PROFILE = "timestamp_agreement_stream/v1"
CONTEXT_CONTINUOUS_PROFILE = "context_agreement_stream/v1"
COALESCED_CONTINUOUS_PROFILE = "coalesced_timestamp_agreement_stream/v1"
COALESCED_CONTEXT_CONTINUOUS_PROFILE = "coalesced_context_agreement_stream/v1"
WORD_CONTINUOUS_PROFILE = "word_agreement_stream/v1"
SOURCE_UNIT_PROFILE = "source_unit_stream/v1"
COALESCED_SOURCE_UNIT_PROFILE = "coalesced_source_unit_stream/v1"
QUIET_ENDPOINT_PROFILE = "quiet_endpoint_stream/v1"
COALESCED_QUIET_ENDPOINT_PROFILE = "coalesced_quiet_endpoint_stream/v1"
WORD_BOUNDARY_QUIET_ENDPOINT_PROFILE = "word_boundary_quiet_endpoint_stream/v1"
COALESCED_WORD_BOUNDARY_QUIET_ENDPOINT_PROFILE = (
    "coalesced_word_boundary_quiet_endpoint_stream/v1"
)
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
    input_evidence: bool = False
    source_units: bool = False
    endpointing: QuietEndpointConfig | None = None
    word_boundary_fallback: bool = False
    word_context_limit_ms: int = 0

    def __post_init__(self) -> None:
        flags = (
            "coalesce_previews",
            "word_alignment",
            "input_evidence",
            "source_units",
            "word_boundary_fallback",
        )
        for name in flags:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in self.__dataclass_fields__:
            if name in flags or name == "endpointing":
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
        if self.source_units and (
            not self.input_evidence
            or self.word_alignment
            or (self.left_context_ms and not self.word_boundary_fallback)
        ):
            raise ValueError(
                "source_units requires input_evidence, no word alignment, "
                "and zero left context unless word_boundary_fallback is selected"
            )
        if self.word_boundary_fallback and (
            not self.source_units
            or not self.input_evidence
            or self.endpointing is None
            or not self.left_context_ms
        ):
            raise ValueError(
                "word_boundary_fallback requires source_units, input_evidence, "
                "endpointing, and positive left context"
            )
        if self.word_context_limit_ms:
            if not self.word_boundary_fallback:
                raise ValueError("word context requires word_boundary_fallback")
            if self.word_context_limit_ms < self.left_context_ms:
                raise ValueError(
                    "word context limit must cover the desired left context"
                )
            if self.word_context_limit_ms % 20:
                raise ValueError("word_context_limit_ms must be a multiple of 20 ms")
            if (
                self.word_context_limit_ms + 2 * self.preview_interval_ms
                >= self.max_window_ms
            ):
                raise ValueError(
                    "word context must leave room for two growing analyses"
                )
        if self.endpointing is not None:
            if not isinstance(self.endpointing, QuietEndpointConfig):
                raise TypeError("endpointing must be QuietEndpointConfig or None")
            if not self.source_units:
                raise ValueError("endpointing requires source_units")
            if self.endpointing.max_quiet_unit_ms > self.max_window_ms:
                raise ValueError("quiet units must fit the analysis window")


@dataclass(frozen=True, slots=True)
class SourceUnit:
    """A closed input range, not a claim of correct recognition.

    The origin records who closed the range. It does not certify silence or
    authorize publication without the selected input-evidence policy.
    The hybrid word profile may already have committed a prefix of this range;
    the original quiet observation is preserved while alignment selects only
    the unpublished suffix.
    """

    start_sample: int
    end_sample: int
    origin: Literal["caller", "end_of_input", "quiet_run"]
    endpoint: QuietEndpointProposal | None = None

    def __post_init__(self) -> None:
        for value in (self.start_sample, self.end_sample):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("source unit boundaries must be integers")
        if not 0 <= self.start_sample < self.end_sample:
            raise ValueError("source unit must have a nonempty nonnegative range")
        if self.origin not in ("caller", "end_of_input", "quiet_run"):
            raise ValueError("unknown source unit origin")
        if self.origin == "quiet_run":
            if not isinstance(self.endpoint, QuietEndpointProposal):
                raise TypeError("quiet_run requires a QuietEndpointProposal")
            if (
                self.endpoint.end_sample != self.end_sample
                or self.endpoint.quiet_start_sample < self.start_sample
            ):
                raise ValueError("endpoint observation must be inside its source unit")
        elif self.endpoint is not None:
            raise ValueError("only quiet_run units can contain endpoint observations")


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
    audio_evidence: AudioEvidenceDecision | None = None
    silence_publication: SilencePublication | None = None
    source_unit: SourceUnit | None = None
    accepted_through_sample: int | None = None
    anchor_diagnostic: AnchorDiagnostic | None = None


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
        if (
            self.config.word_alignment or self.config.source_units
        ) and self._options.task != "transcribe":
            raise ValueError(
                "word-aligned and source-unit streaming support transcription only"
            )
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
        self._previous_eligible = True
        self._run_observation: AudioObservation | None = None
        self._audio_decision: AudioEvidenceDecision | None = None
        self._silence_pending: SilencePublication | None = None
        self._word_previous: NativeWordAlignment | None = None
        self._word_anchor: tuple[NativeTimestampSegment, ...] = ()
        self._word_pending: WordAgreementDecision | None = None
        self._word_retained_pending: int | None = None
        self._retry_analysis: tuple[int, int, bool] | None = None
        self._unit: SourceUnit | None = None
        self._endpoints: deque[QuietEndpointProposal] = deque()
        self._endpoint_detector = (
            QuietEndpointDetector(self.config.endpointing)
            if self.config.endpointing is not None
            else None
        )
        self._retry_unit: SourceUnit | None = None
        self._run_unit: SourceUnit | None = None
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
        base = self._base_profile_id()
        return f"{base}+input_evidence/v1" if self.config.input_evidence else base

    def _base_profile_id(self) -> str:
        if self.config.word_boundary_fallback:
            return (
                COALESCED_WORD_BOUNDARY_QUIET_ENDPOINT_PROFILE
                if self.config.coalesce_previews
                else WORD_BOUNDARY_QUIET_ENDPOINT_PROFILE
            ) + ("+word_context/v1" if self.config.word_context_limit_ms else "")
        if self.config.endpointing is not None:
            return (
                COALESCED_QUIET_ENDPOINT_PROFILE
                if self.config.coalesce_previews
                else QUIET_ENDPOINT_PROFILE
            )
        if self.config.source_units:
            return (
                COALESCED_SOURCE_UNIT_PROFILE
                if self.config.coalesce_previews
                else SOURCE_UNIT_PROFILE
            )
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
                or self._unit is not None
                or bool(self._endpoints)
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
            if self._endpoint_detector is not None:
                self._endpoints.extend(
                    self._endpoint_detector.observe(self._accepted, pcm_s16le)
                )
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

    def seal_unit(self, end_sample: int) -> SourceUnit:
        """Close one admitted input range without ending the stream.

        Call on the model-work owner before processing beyond the desired end.
        Only one boundary can be pending. A running preview is not promoted to
        a final result: its immutable operation must finish before the sealed
        range is decoded. This method neither removes PCM nor commits text.
        """
        self._require_owner()
        with self._lock:
            if not self.config.source_units:
                raise NativeStreamError("seal_unit requires source_units")
            if self.config.endpointing is not None:
                raise NativeStreamError("automatic endpointing owns source boundaries")
            if self._done:
                raise NativeStreamError("input is closed")
            if (
                self._unit is not None
                or self._unresolved_eof
                or (self._run is not None and self._run_unit is not None)
                or (self._retry_analysis is not None and self._retry_unit is not None)
            ):
                raise NativeStreamError("a source unit is already pending")
            unit = SourceUnit(self._head, end_sample, "caller")
            if end_sample > self._accepted:
                raise ValueError("source unit exceeds admitted audio")
            if end_sample - self._head > self.config.max_window_ms * 16:
                raise ValueError("source unit exceeds the analysis window")
            analyzed_end = self._last_endpoint
            if self._run is not None:
                analyzed_end = max(analyzed_end, self._run_end)
            if self._retry_analysis is not None:
                analyzed_end = max(analyzed_end, self._retry_analysis[1])
            if end_sample < analyzed_end:
                raise ValueError("source unit ends before an admitted analysis")
            self._unit = unit
            return unit

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
                "input boundary is unresolved; close the stream before retrying "
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
            unit = None
            if self.config.source_units:
                unit = self._unit
                if unit is None and self._endpoints and self._retry_analysis is None:
                    proposal = self._endpoints[0]
                    if proposal.end_sample > bound:
                        if not self.config.word_boundary_fallback:
                            raise StreamNeedsResolutionError(
                                "automatic endpoint exceeds the analysis window; input retained"
                            )
                        # Keep the observation queued. Only a supported word
                        # prefix may rebase the window; never manufacture a cut.
                    else:
                        unit = SourceUnit(
                            min(self._head, proposal.quiet_start_sample)
                            if self.config.word_boundary_fallback
                            else self._head,
                            proposal.end_sample,
                            "quiet_run",
                            proposal,
                        )
                if unit is None and final:
                    unit = SourceUnit(self._head, self._accepted, "end_of_input")
                if unit is not None:
                    endpoint = unit.end_sample
                    final = self._eof and endpoint == self._accepted
            if self.config.coalesce_previews and not final and unit is None:
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
                unit = self._retry_unit
            if not final and unit is None and endpoint <= self._last_endpoint:
                raise StreamNeedsResolutionError(
                    "no stable contiguous prefix within the audio window; input retained"
                )
            if (
                self.config.coalesce_previews
                or self.config.word_alignment
                or self.config.input_evidence
            ) and self._retry_analysis is None:
                # Admission is immutable even if preprocessing/startup fails,
                # or more PCM/EOF arrives while a failed run is recovered.
                self._retry_analysis = (start, endpoint, final)
                self._retry_unit = unit
            pcm = bytes(memoryview(self._audio)[: (endpoint - start) * 2])
        observation = (
            AudioObservation.from_pcm(pcm) if self.config.input_evidence else None
        )
        window_id = f"{self._id}:window:{start}:{endpoint}"
        if self.config.source_units:
            window_id += ":unit" if unit is not None else ":preview"
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
        self._run_unit = unit
        self._run_window_id = window_id
        self._run_observation = observation
        self._audio_decision = None
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
        if self.config.input_evidence:
            observation = self._run_observation
            if observation is None or observation.sample_count != (
                self._run_end - self._run_start
            ):
                run.close()
                raise NativeStreamError("input evidence does not match admitted audio")
            self._audio_decision = assess_audio(observation, result)
            if self._audio_decision.state == "non_speech":
                if self.config.source_units and self._run_unit is None:
                    self._trace(result, None, "source_unit_open", "preview")
                    run.close()
                    if self.config.word_boundary_fallback:
                        self._word_previous = None
                    return self._publish_preview("")
                return self._resolve_silence(run, result, observation)
            if self._audio_decision.state != "speech_candidate":
                return self._defer_audio(run, result)
        if self.config.word_boundary_fallback:
            return self._resolve_words(run, result)
        if self.config.source_units:
            return self._resolve_unit(run, result)
        if self.config.word_alignment:
            return self._resolve_words(run, result)
        decision = compare_hypotheses(
            self._previous if self._previous_eligible else None,
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
        if (self._run_final or span is not None) and not self._publication_supported(
            result.text
            if span is None
            else select_native_publication(result, span).text
        ):
            return self._defer_audio(run, result)
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
        self._previous_eligible = True
        text = result.text
        if self._run_start < self._head:
            preview_span = self._context_span(result, final=False)
            text = (
                select_native_publication(result, preview_span).text
                if preview_span is not None
                else ""
            )
        return self._publish_preview(text)

    def _publication_supported(self, text: str) -> bool:
        if not self.config.input_evidence or any(char.isalnum() for char in text):
            return True
        assert self._audio_decision is not None
        self._audio_decision = AudioEvidenceDecision(
            self._audio_decision.observation,
            "uncertain",
            "no_lexical_text",
            self._audio_decision.no_speech_prob,
        )
        return False

    def _resolve_unit(
        self, run: NativeWindowRun, result: NativeWindowResult
    ) -> tuple[TranscriptEvent, ...]:
        """Publish a closed range under a full-result recognition contract.

        Input closure does not establish word accuracy. Unlike the word profile,
        this profile makes no per-word finality or timing claim. Open-range text
        can change until an input boundary closes it and its native run completes.
        """
        unit = self._run_unit
        if unit is None:
            self._trace(result, None, "source_unit_open", "preview")
            run.close()
            return self._publish_preview(result.text)
        if unit.start_sample != self._head or unit.end_sample != self._run_end:
            run.close()
            raise NativeStreamError("source unit does not match admitted analysis")
        self._trace(result, None, "source_unit", "commit")
        self._pending_end, self._pending_final = unit.end_sample, self._run_final
        self._pending_state = run.finish(committed_through_ms=unit.end_sample // 16)
        self._run = None
        return self._publish_commit()

    def _resolve_silence(
        self,
        run: NativeWindowRun,
        result: NativeWindowResult,
        observation: AudioObservation,
    ) -> tuple[TranscriptEvent, ...]:
        publication = SilencePublication(
            window_id=result.window_id,
            text="",
            start_ms=self._head // 16,
            end_ms=self._run_end // 16,
            analysis_span=result.analyzed_span,
            native=result,
            observation=observation,
            analysis_start_sample=self._run_start,
        )
        self._trace(
            result, None, "digital_silence", "commit", silence_publication=publication
        )
        self._silence_pending = publication
        self._pending_end, self._pending_final = self._run_end, self._run_final
        self._pending_state = run.finish(
            committed_through_ms=self._run_end // 16,
            silence_publication=publication,
        )
        self._run = None
        return self._publish_commit()

    def _defer_audio(
        self, run: NativeWindowRun, result: NativeWindowResult
    ) -> tuple[TranscriptEvent, ...]:
        """Suppress unsupported text without granting audio-discard authority."""
        assert self._audio_decision is not None
        closed = self._run_final or self._run_unit is not None
        self._trace(
            result,
            None,
            self._audio_decision.reason,
            "unresolved" if closed else "wait_for_input",
        )
        # Preserve the scheduling observation, but never use a rejected result
        # as an agreement witness. Close/fence before exposing any event.
        if closed:
            self._unresolved_eof = True
        run.close()
        self._previous = result
        self._previous_eligible = False
        self._word_previous = None
        if closed:
            self._run = None
            self._record_decode()
            self._retry_analysis = None
            boundary = "source unit" if self._run_unit is not None else "EOF"
            raise StreamNeedsResolutionError(
                f"{boundary} input evidence is unresolved ({self._audio_decision.reason}); "
                "input retained"
            )
        return self._publish_preview("")

    def _resolve_words(
        self, run: NativeWindowRun, result: NativeWindowResult
    ) -> tuple[TranscriptEvent, ...]:
        # The word selector's final=True closes this analysis suffix, not the
        # stream. Trace reason maps it to source_unit; EOF/FINAL stay source EOF.
        # Alignment still validates/excludes retained committed words.
        unit = self._run_unit
        closed_unit = self.config.word_boundary_fallback and unit is not None
        closed = self._run_final or closed_unit
        if (
            closed_unit
            and unit is not None
            and (unit.start_sample > self._head or unit.end_sample != self._run_end)
        ):
            run.close()
            raise NativeStreamError("source unit does not match admitted analysis")
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
            final=closed,
        )
        publication = decision.publication
        if publication is not None and not self._publication_supported(
            publication.text
        ):
            return self._defer_audio(run, result)
        retained = None
        reason = decision.reason
        if publication is not None and not closed and self.config.word_context_limit_ms:
            # A new window need not regenerate punctuation that led into its
            # first spoken word. Keep the exact lexical suffix as the witness;
            # publication text, internal punctuation and frozen times stay intact.
            anchor = decision.next_anchor
            while anchor and not any(c.isalnum() for c in anchor[0].text):
                anchor = anchor[1:]
            decision = WordAgreementDecision(
                decision.reason, publication, anchor, decision.anchor_diagnostic
            )
            retained = self._word_context_start(decision)
            if retained is None:
                publication = None
                reason = "context_unresolved"
        self._trace(
            result,
            None,
            "source_unit" if closed_unit and publication is not None else reason,
            "commit"
            if publication is not None
            else "unresolved"
            if closed
            else "preview",
            word_alignment=alignment,
            word_publication=publication,
            anchor_diagnostic=decision.anchor_diagnostic,
        )
        if publication is not None:
            self._word_pending = decision
            self._word_retained_pending = retained
            self._pending_end = (
                self._run_end if closed else publication.end_ms * _SAMPLES_PER_MS
            )
            self._pending_final = self._run_final
            self._pending_state = run.finish(
                committed_through_ms=self._pending_end // _SAMPLES_PER_MS,
                aligned_publication=publication,
            )
            self._run = None
            return self._publish_commit()
        if closed:
            self._unresolved_eof = True
            run.close()
            self._run = None
            self._retry_analysis = None
            self._record_decode()
            boundary = "source unit" if closed_unit else "EOF"
            raise StreamNeedsResolutionError(
                f"{boundary} word alignment is unresolved ({decision.reason}); input retained"
            )
        run.close()
        self._word_previous = alignment
        self._previous = result
        self._previous_eligible = True
        # Provisional text uses estimated positions only. It has no commit authority.
        text = "".join(
            word.text
            for word in alignment.words
            if word.span.start_ms >= self._head // _SAMPLES_PER_MS
        ).strip()
        return self._publish_preview(text)

    def _word_context_start(self, decision: WordAgreementDecision) -> int | None:
        """Select bounded context from this alignment before committing any output.

        Word times are estimates. This preserves the matching witnesses and avoids
        cutting through observed words; it does not prove sufficient model context.
        """
        publication = decision.publication
        assert publication is not None
        anchor = decision.next_anchor
        if sum(any(c.isalnum() for c in word.text) for word in anchor) < 2:
            return None
        end = publication.end_ms * _SAMPLES_PER_MS
        origin = max(
            self._retained,
            min(
                end - self.config.left_context_ms * _SAMPLES_PER_MS,
                anchor[0].span.start_ms * _SAMPLES_PER_MS,
            ),
        )
        grid = 20 * _SAMPLES_PER_MS
        origin -= (origin - self._retained) % grid
        # Iterate backward: rounding one word's start can enter its predecessor.
        for word in reversed(publication.alignment.words[: publication.word_end]):
            start, stop = word.span.start_ms * 16, word.span.end_ms * 16
            if start < origin < stop:
                origin = max(self._retained, start)
                origin -= (origin - self._retained) % grid
        if (
            end - origin > self.config.word_context_limit_ms * _SAMPLES_PER_MS
            or any(word.span.start_ms * 16 < origin for word in anchor)
            or origin
            + (self.config.max_window_ms - self.config.preview_interval_ms) * 16
            <= end
        ):
            return None
        return origin

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
        silence_publication: SilencePublication | None = None,
        anchor_diagnostic: AnchorDiagnostic | None = None,
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
            self._audio_decision,
            silence_publication,
            self._run_unit,
            self.accepted_samples,
            anchor_diagnostic,
        )

    def _publish_commit(self) -> tuple[TranscriptEvent, ...]:
        state = self._pending_state
        assert state is not None
        if state != self._session.snapshot() or not state.windows:
            raise NativeStreamError("committed result does not belong to this session")
        result = state.windows[-1].result
        if (
            self._run_unit is not None
            and self._unit is not None
            and self._unit != self._run_unit
        ):
            raise NativeStreamError("committed source unit changed")
        if self._silence_pending is not None and result != self._silence_pending:
            raise NativeStreamError(
                "committed result does not match the silence decision"
            )
        if self._run_unit is not None and self._run_unit.origin == "quiet_run":
            with self._lock:
                if not self._endpoints or self._endpoints[0] != self._run_unit.endpoint:
                    raise NativeStreamError("committed endpoint observation changed")
        if (
            (self.config.word_alignment or self.config.word_boundary_fallback)
            and self._silence_pending is None
            and (self._word_pending is None or self._word_pending.publication != result)
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
                or self._silence_pending is not None
                or self._run_unit is not None
                else self._word_retained_pending
                if self._word_retained_pending is not None
                else max(self._retained, end - self.config.left_context_ms * 16)
            )
            del self._audio[: (retained - self._retained) * 2]
            self._retained = retained
            self._head = end
            self._done = self._pending_final
            if self._run_unit is not None:
                self._unit = None
                if self._run_unit.origin == "quiet_run":
                    self._endpoints.popleft()
        # Establish a fresh growing pair after rebasing. Starting immediately at
        # the old endpoint can exhaust a full window before a second observation.
        self._last_endpoint = end
        interval = self.config.preview_interval_ms * 16
        self._next_endpoint = min(
            self._run_end + interval,
            self._retained + self.config.max_window_ms * 16 - interval,
        )
        if self.config.source_units and (
            not self.config.word_boundary_fallback or self._run_unit is not None
        ):
            self._next_endpoint = end + interval
        self._previous = None
        self._previous_eligible = True
        self._word_previous = None
        if self._silence_pending is not None:
            self._word_anchor = ()
            self._silence_pending = None
        if self._word_pending is not None:
            self._word_anchor = (
                ()
                if self.config.word_boundary_fallback and self._run_unit is not None
                else self._word_pending.next_anchor
            )
            self._word_pending = None
        self._word_retained_pending = None
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
