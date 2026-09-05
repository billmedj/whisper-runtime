"""Bounded PCM ingestion and revision events for the native adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import current_thread
from typing import Callable, Protocol

from ..errors import RuntimeStateError, TransactionRetainedError
from ..model import ModelSnapshot
from ..state import RequestState, Session, SessionState
from .native_whisper import NativeDecodeOptions

_PCM_SAMPLE_BYTES = 2
_WHISPER_SAMPLE_RATE_HZ = 16_000
BOUNDED_PREFIX_PROFILE = "bounded_prefix_preview/v1"


class NativeStreamError(RuntimeStateError):
    """Base error for an invalid native stream operation."""


class AudioSequenceError(NativeStreamError):
    """The next PCM chunk does not have the expected sequence number."""


class AudioBufferFullError(NativeStreamError):
    """A PCM chunk would exceed the configured stream bound."""


@dataclass(frozen=True, slots=True)
class NativeStreamConfig:
    """Fixed limits for one mono signed 16-bit little-endian PCM stream."""

    sample_rate_hz: int = _WHISPER_SAMPLE_RATE_HZ
    preview_interval_ms: int = 1_000
    max_audio_ms: int = 30_000

    def __post_init__(self) -> None:
        for name, value in (
            ("sample_rate_hz", self.sample_rate_hz),
            ("preview_interval_ms", self.preview_interval_ms),
            ("max_audio_ms", self.max_audio_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sample_rate_hz != _WHISPER_SAMPLE_RATE_HZ:
            raise ValueError("sample_rate_hz must be 16000")
        if self.max_audio_ms > 30_000:
            raise ValueError("max_audio_ms cannot exceed 30000")
        if self.preview_interval_ms > self.max_audio_ms:
            raise ValueError("preview_interval_ms cannot exceed max_audio_ms")

    @property
    def preview_interval_samples(self) -> int:
        return self.sample_rate_hz * self.preview_interval_ms // 1_000

    @property
    def max_audio_samples(self) -> int:
        return self.sample_rate_hz * self.max_audio_ms // 1_000


class StreamEventKind(str, Enum):
    """Public transcript event types."""

    PROVISIONAL = "provisional"
    REPLACE = "replace"
    COMMIT = "commit"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """One ordered transcript change or terminal session marker."""

    sequence_number: int
    kind: StreamEventKind
    segment_id: str | None = None
    revision: int | None = None
    start_sample: int | None = None
    end_sample: int | None = None
    sample_rate_hz: int | None = None
    text: str | None = None
    supersedes_revision: int | None = None
    committed_through_sample: int | None = None
    committed_through_ms: int | None = None
    session_version: int | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            "sequence_number", self.sequence_number, positive=True
        )
        if not isinstance(self.kind, StreamEventKind):
            raise TypeError("kind must be a StreamEventKind")
        if self.kind is StreamEventKind.FINAL:
            _require_nonnegative_integer("session_version", self.session_version)
            for name, value in (
                ("segment_id", self.segment_id),
                ("revision", self.revision),
                ("start_sample", self.start_sample),
                ("end_sample", self.end_sample),
                ("sample_rate_hz", self.sample_rate_hz),
                ("text", self.text),
                ("supersedes_revision", self.supersedes_revision),
                ("committed_through_sample", self.committed_through_sample),
                ("committed_through_ms", self.committed_through_ms),
            ):
                if value is not None:
                    raise ValueError(f"a final event cannot contain {name}")
            return

        if (
            not isinstance(self.segment_id, str)
            or not self.segment_id
            or self.segment_id.isspace()
        ):
            raise ValueError("segment_id must not be empty")
        _require_nonnegative_integer("revision", self.revision, positive=True)
        _require_nonnegative_integer("start_sample", self.start_sample)
        _require_nonnegative_integer("end_sample", self.end_sample)
        _require_nonnegative_integer(
            "sample_rate_hz", self.sample_rate_hz, positive=True
        )
        assert self.start_sample is not None
        assert self.end_sample is not None
        if self.end_sample < self.start_sample:
            raise ValueError("end_sample must not precede start_sample")

        if self.kind in (StreamEventKind.PROVISIONAL, StreamEventKind.REPLACE):
            if not isinstance(self.text, str):
                raise TypeError("a provisional or replace event must contain text")
            if self.committed_through_sample is not None:
                raise ValueError("a provisional event cannot commit source audio")
            if self.committed_through_ms is not None:
                raise ValueError("a provisional event cannot commit source audio")
            assert self.revision is not None
            if self.revision <= 0:
                raise ValueError("a provisional revision must be positive")
            if self.kind is StreamEventKind.PROVISIONAL:
                if self.supersedes_revision is not None:
                    raise ValueError(
                        "a first provisional event cannot supersede a revision"
                    )
            else:
                _require_nonnegative_integer(
                    "supersedes_revision", self.supersedes_revision, positive=True
                )
                if self.supersedes_revision != self.revision - 1:
                    raise ValueError(
                        "a replace event must supersede the prior revision"
                    )
            if self.session_version is not None:
                _require_nonnegative_integer("session_version", self.session_version)
            return

        if self.text is not None:
            raise ValueError(
                "a commit event identifies a revision, not transcript text"
            )
        if self.supersedes_revision is not None:
            raise ValueError("a commit event cannot supersede a revision")
        _require_nonnegative_integer(
            "committed_through_sample", self.committed_through_sample
        )
        _require_nonnegative_integer("committed_through_ms", self.committed_through_ms)
        _require_nonnegative_integer(
            "session_version", self.session_version, positive=True
        )
        assert self.committed_through_sample is not None
        if self.committed_through_sample > self.end_sample:
            raise ValueError("a commit cannot exceed its source span")
        assert self.sample_rate_hz is not None
        assert self.committed_through_ms is not None
        if self.committed_through_ms != _samples_to_floor_ms(
            self.committed_through_sample,
            self.sample_rate_hz,
        ):
            raise ValueError(
                "the millisecond watermark must be the sample watermark floor"
            )

    @property
    def start_ms(self) -> float | None:
        """Return the exact source start in milliseconds when the event has a span."""

        if self.start_sample is None or self.sample_rate_hz is None:
            return None
        return self.start_sample * 1_000 / self.sample_rate_hz

    @property
    def end_ms(self) -> float | None:
        """Return the exact source end in milliseconds when the event has a span."""

        if self.end_sample is None or self.sample_rate_hz is None:
            return None
        return self.end_sample * 1_000 / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class StreamMetrics:
    """A point-in-time account of admitted and decoded source audio."""

    sample_rate_hz: int
    accepted_chunks: int
    accepted_samples: int
    decode_count: int
    decoded_source_samples: int
    events_emitted: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            "sample_rate_hz", self.sample_rate_hz, positive=True
        )
        for name, value in (
            ("accepted_chunks", self.accepted_chunks),
            ("accepted_samples", self.accepted_samples),
            ("decode_count", self.decode_count),
            ("decoded_source_samples", self.decoded_source_samples),
            ("events_emitted", self.events_emitted),
        ):
            _require_nonnegative_integer(name, value)

    @property
    def accepted_audio_ms(self) -> float:
        return self.accepted_samples * 1_000 / self.sample_rate_hz

    @property
    def decoded_source_audio_ms(self) -> float:
        return self.decoded_source_samples * 1_000 / self.sample_rate_hz

    @property
    def source_reprocessing_factor(self) -> float:
        """Return cumulative prefix input divided by unique admitted input."""

        if self.accepted_samples == 0:
            return 0.0
        return self.decoded_source_samples / self.accepted_samples


class _WindowRun(Protocol):
    @property
    def complete(self) -> bool:
        """Return whether token generation has ended."""

    @property
    def closed(self) -> bool:
        """Return whether owner operations are no longer permitted."""

    @property
    def capacity_released(self) -> bool:
        """Return whether the run no longer owns worker capacity."""

    def step(self) -> bool:
        """Advance at most one decoder token step."""

    def finish(self, *, committed_through_ms: int | None = None) -> SessionState:
        """Finalize and publish the completed run."""

    def cancel(self) -> bool:
        """Request cooperative cancellation."""

    def stop(self) -> bool:
        """Fence and reclaim an abandoned run."""

    def close(self) -> bool:
        """Abort an unfinished run from its owner thread."""


class _WindowAdapter(Protocol):
    @property
    def model_identity(self) -> ModelSnapshot:
        """Return the model snapshot bound to the adapter."""

    def start_window(
        self,
        *,
        session: Session,
        request: RequestState,
        window_id: str,
        mel: object,
        start_ms: int,
        end_ms: int,
        options: NativeDecodeOptions | None = None,
    ) -> _WindowRun:
        """Start one governed native decode."""


class NativeTranscriptStream:
    """Drive bounded prefix previews through one native adapter.

    ``push`` only admits PCM. Each call to ``step`` starts a scheduled window,
    advances at most one decoder token step, or publishes a completed window.
    The caller can pause by withholding ``step`` and can cancel or stop the
    active run. Text stays provisional until the final successful publication.

    This profile retains at most 30 seconds and is not continuous streaming.
    The caller supplies preprocessing, so it also makes no offline-equivalence
    claim. Instances are single-owner except for ``cancel_active`` and
    ``stop_active``, which delegate to the thread-safe native run controls.
    A pause cannot extend past the native transaction deadline.
    """

    def __init__(
        self,
        adapter: _WindowAdapter,
        *,
        stream_id: str,
        mel_builder: Callable[[bytes], object],
        options: NativeDecodeOptions | None = None,
        rng_seed: int = 0,
        config: NativeStreamConfig | None = None,
    ) -> None:
        if not isinstance(stream_id, str) or not stream_id or stream_id.isspace():
            raise ValueError("stream_id must not be empty")
        if not callable(mel_builder):
            raise TypeError("mel_builder must be callable")
        if options is not None and not isinstance(options, NativeDecodeOptions):
            raise TypeError("options must be a NativeDecodeOptions or None")
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise TypeError("rng_seed must be an integer")
        if config is not None and not isinstance(config, NativeStreamConfig):
            raise TypeError("config must be a NativeStreamConfig or None")
        model = adapter.model_identity
        if not isinstance(model, ModelSnapshot):
            raise TypeError("adapter.model_identity must be a ModelSnapshot")

        self._adapter = adapter
        self._owner_thread = current_thread()
        self._stream_id = stream_id
        self._segment_id = f"{stream_id}:segment-0"
        self._mel_builder = mel_builder
        self._options = options
        self._rng_seed = rng_seed
        self._config = config or NativeStreamConfig()
        self._session = Session(f"{stream_id}:session")
        self._model = model
        self._audio = bytearray()
        self._expected_chunk = 0
        self._next_preview_sample = self._config.preview_interval_samples
        self._input_finished = False
        self._done = False
        self._revision = 0
        self._current_text: str | None = None
        self._current_endpoint = -1
        self._event_sequence = 0
        self._decode_count = 0
        self._decoded_source_samples = 0
        self._active_run: _WindowRun | None = None
        self._active_endpoint = 0
        self._active_end_ms = 0
        self._active_final = False
        self._active_window_id = ""
        self._completed_state: SessionState | None = None

    def __enter__(self) -> NativeTranscriptStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, traceback
        run = self._active_run
        if (
            isinstance(exc_value, TransactionRetainedError)
            and run is not None
            and run.closed
            and not run.capacity_released
        ):
            # Keep the recovery authority and any committed result on the error.
            return
        self.close()

    @property
    def config(self) -> NativeStreamConfig:
        return self._config

    @property
    def profile_id(self) -> str:
        """Return the machine-readable contract implemented by this stream."""

        return BOUNDED_PREFIX_PROFILE

    @property
    def expected_chunk(self) -> int:
        return self._expected_chunk

    @property
    def accepted_samples(self) -> int:
        return len(self._audio) // _PCM_SAMPLE_BYTES

    @property
    def input_finished(self) -> bool:
        return self._input_finished

    @property
    def done(self) -> bool:
        return self._done

    @property
    def active(self) -> bool:
        return self._active_run is not None

    @property
    def ready(self) -> bool:
        """Return whether ``step`` can make progress."""

        if self._done:
            return False
        if self._active_run is not None:
            return True
        if self._input_finished:
            return True
        return self._next_preview_sample <= self.accepted_samples

    @property
    def state(self) -> SessionState:
        return self._session.snapshot()

    @property
    def metrics(self) -> StreamMetrics:
        return StreamMetrics(
            sample_rate_hz=self._config.sample_rate_hz,
            accepted_chunks=self._expected_chunk,
            accepted_samples=self.accepted_samples,
            decode_count=self._decode_count,
            decoded_source_samples=self._decoded_source_samples,
            events_emitted=self._event_sequence,
        )

    def push(self, sequence_number: int, pcm_s16le: bytes) -> int:
        """Atomically admit PCM and return the accepted-through sample index."""

        self._require_owner()
        if self._input_finished:
            raise NativeStreamError("input is already finished")
        if isinstance(sequence_number, bool) or not isinstance(sequence_number, int):
            raise TypeError("sequence_number must be an integer")
        if sequence_number != self._expected_chunk:
            raise AudioSequenceError(
                f"expected PCM chunk {self._expected_chunk}; received {sequence_number}"
            )
        if not isinstance(pcm_s16le, bytes):
            raise TypeError("pcm_s16le must be bytes")
        if not pcm_s16le:
            raise ValueError("pcm_s16le must not be empty")
        if len(pcm_s16le) % _PCM_SAMPLE_BYTES:
            raise ValueError("pcm_s16le must contain complete 16-bit samples")

        chunk_samples = len(pcm_s16le) // _PCM_SAMPLE_BYTES
        if self.accepted_samples + chunk_samples > self._config.max_audio_samples:
            raise AudioBufferFullError(
                f"PCM chunk would exceed the {self._config.max_audio_ms} ms bound"
            )

        self._audio.extend(pcm_s16le)
        self._expected_chunk += 1
        return self.accepted_samples

    def finish_input(self) -> bool:
        """Mark end of input once. Final publication still requires ``step``."""

        self._require_owner()
        if self._input_finished:
            return False
        self._input_finished = True
        return True

    def cancel_active(self) -> bool:
        """Request cooperative cancellation of the active native run."""

        run = self._active_run
        return False if run is None else run.cancel()

    def stop_active(self) -> bool:
        """Fence and recover the active native run when possible."""

        run = self._active_run
        return False if run is None else run.stop()

    def close(self) -> bool:
        """Abandon the stream and release an active run from its owner thread."""

        self._require_owner()
        if self._done:
            return False
        changed = True
        run = self._active_run
        if run is not None:
            try:
                changed = run.close()
            finally:
                if run.capacity_released:
                    self._clear_active()
            if not run.capacity_released:
                raise NativeStreamError(
                    "the closed native run still owns capacity; call stop_active to recover"
                )
        self._input_finished = True
        self._done = True
        return changed

    def step(self) -> tuple[TranscriptEvent, ...]:
        """Start, advance, or publish at most one governed window operation."""

        self._require_owner()
        if not self.ready:
            return ()
        run = self._active_run
        if run is not None:
            if run.closed:
                if not run.capacity_released:
                    raise NativeStreamError("the closed native run still owns capacity")
                if self._completed_state is not None:
                    return self._publish_active(self._completed_state)
                self._clear_active()
                return ()
            if not run.complete:
                try:
                    run.step()
                except BaseException:
                    if run.capacity_released:
                        self._clear_active()
                    raise
                return ()
            try:
                state = run.finish(
                    committed_through_ms=(
                        self._active_end_ms if self._active_final else None
                    )
                )
            except TransactionRetainedError as error:
                self._completed_state = error.committed_state
                raise
            except BaseException:
                if run.capacity_released:
                    self._clear_active()
                raise
            return self._publish_active(state)

        samples = self.accepted_samples
        if self._input_finished and samples == 0:
            self._done = True
            return (self._final_event(session_version=0),)

        final = self._input_finished
        endpoint = samples if final else self._next_preview_sample
        self._start_decode(endpoint, final=final)
        return ()

    def _start_decode(self, endpoint: int, *, final: bool) -> None:
        pcm = bytes(memoryview(self._audio)[: endpoint * _PCM_SAMPLE_BYTES])
        mel = self._mel_builder(pcm)
        label = "final" if final else "preview"
        window_id = f"{self._stream_id}:{label}:{endpoint}"
        end_ms = _samples_to_floor_ms(endpoint, self._config.sample_rate_hz)
        request = RequestState(
            f"{window_id}:request",
            self._session.session_id,
            self._model,
            rng_seed=self._rng_seed if final else self._rng_seed + endpoint,
        )
        run = self._adapter.start_window(
            session=self._session,
            request=request,
            window_id=window_id,
            mel=mel,
            start_ms=0,
            end_ms=end_ms,
            options=self._options,
        )
        self._active_run = run
        self._active_endpoint = endpoint
        self._active_end_ms = end_ms
        self._active_final = final
        self._active_window_id = window_id

    def _publish_active(self, state: SessionState) -> tuple[TranscriptEvent, ...]:
        endpoint = self._active_endpoint
        end_ms = self._active_end_ms
        final = self._active_final
        window_id = self._active_window_id
        if (
            state.session_id != self._session.session_id
            or state != self._session.snapshot()
            or not state.windows
        ):
            self._done = True
            raise NativeStreamError("the native result does not belong to this session")
        record = state.windows[-1]
        result = record.result
        self._clear_active()
        if (
            result.window_id != window_id
            or record.request_id != f"{window_id}:request"
            or record.model != self._model
            or result.start_ms != 0
            or result.end_ms != end_ms
        ):
            self._done = True
            raise NativeStreamError(
                "the native result does not match its source window"
            )
        if final and state.committed_through_ms != end_ms:
            self._done = True
            raise NativeStreamError(
                "the native result did not commit its source window"
            )

        self._decode_count += 1
        self._decoded_source_samples += endpoint
        if not final:
            self._next_preview_sample += self._config.preview_interval_samples

        events: list[TranscriptEvent] = []
        hypothesis = self._hypothesis_event(
            result.text,
            endpoint=endpoint,
            session_version=state.version,
        )
        if hypothesis is not None:
            events.append(hypothesis)

        if final:
            committed_sample = _milliseconds_to_samples(
                end_ms,
                self._config.sample_rate_hz,
            )
            events.append(
                self._new_event(
                    StreamEventKind.COMMIT,
                    segment_id=self._segment_id,
                    revision=self._revision,
                    start_sample=0,
                    end_sample=endpoint,
                    sample_rate_hz=self._config.sample_rate_hz,
                    committed_through_sample=committed_sample,
                    committed_through_ms=state.committed_through_ms,
                    session_version=state.version,
                )
            )
            events.append(self._final_event(session_version=state.version))
            self._done = True
        return tuple(events)

    def _hypothesis_event(
        self,
        text: str,
        *,
        endpoint: int,
        session_version: int,
    ) -> TranscriptEvent | None:
        if text == self._current_text and endpoint == self._current_endpoint:
            return None
        previous = self._revision
        self._revision += 1
        self._current_text = text
        self._current_endpoint = endpoint
        kind = StreamEventKind.PROVISIONAL if previous == 0 else StreamEventKind.REPLACE
        return self._new_event(
            kind,
            segment_id=self._segment_id,
            revision=self._revision,
            start_sample=0,
            end_sample=endpoint,
            sample_rate_hz=self._config.sample_rate_hz,
            text=text,
            supersedes_revision=previous if previous else None,
            session_version=session_version,
        )

    def _final_event(self, *, session_version: int) -> TranscriptEvent:
        return self._new_event(
            StreamEventKind.FINAL,
            session_version=session_version,
        )

    def _new_event(
        self,
        kind: StreamEventKind,
        *,
        segment_id: str | None = None,
        revision: int | None = None,
        start_sample: int | None = None,
        end_sample: int | None = None,
        sample_rate_hz: int | None = None,
        text: str | None = None,
        supersedes_revision: int | None = None,
        committed_through_sample: int | None = None,
        committed_through_ms: int | None = None,
        session_version: int | None = None,
    ) -> TranscriptEvent:
        event = TranscriptEvent(
            sequence_number=self._event_sequence + 1,
            kind=kind,
            segment_id=segment_id,
            revision=revision,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate_hz=sample_rate_hz,
            text=text,
            supersedes_revision=supersedes_revision,
            committed_through_sample=committed_through_sample,
            committed_through_ms=committed_through_ms,
            session_version=session_version,
        )
        self._event_sequence += 1
        return event

    def _clear_active(self) -> None:
        self._active_run = None
        self._active_endpoint = 0
        self._active_end_ms = 0
        self._active_final = False
        self._active_window_id = ""
        self._completed_state = None

    def _require_owner(self) -> None:
        if current_thread() is not self._owner_thread:
            raise NativeStreamError("the stream must be driven by its creating thread")


def _samples_to_floor_ms(samples: int, sample_rate_hz: int) -> int:
    return samples * 1_000 // sample_rate_hz


def _milliseconds_to_samples(milliseconds: int, sample_rate_hz: int) -> int:
    return milliseconds * sample_rate_hz // 1_000


def _require_nonnegative_integer(
    name: str,
    value: object,
    *,
    positive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "not negative"
        raise ValueError(f"{name} must be {qualifier}")
