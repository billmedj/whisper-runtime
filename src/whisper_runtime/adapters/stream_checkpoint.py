"""Local publication-boundary savepoints; no tensors or executable state.

Only an explicit completed save is recoverable. This module does not implement
durable input acknowledgement, output delivery, a multi-writer lease, or token
cache migration. Checksums are not authentication. Keep the expected file hash
and the caller's pipeline identity in trusted storage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Callable

from ..model import ModelSnapshot
from ..resources import ResourceVector
from ..state import AudioSpan, Session, SessionState, WindowRecord, WindowResult
from . import _checkpoint_io as storage
from .audio_endpoints import (
    QuietEndpointConfig,
    QuietEndpointDetector,
    QuietEndpointProposal,
    QuietEndpointState,
)
from .audio_evidence import AudioObservation, SilencePublication
from .continuous_stream import (
    ContinuousStreamConfig,
    ContinuousTranscriptStream,
    SourceUnit,
)
from .native_result import (
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWindowResult,
)
from .native_stream import NativeStreamError
from .native_whisper import (
    NativeDecodeOptions,
    NativeExecutionProfile,
    NativeWhisperAdapter,
)
from .word_policy import AlignedPublication, NativeWordAlignment


@dataclass(frozen=True, slots=True)
class _Position:
    accepted: int
    head: int
    retained: int
    chunk: int
    peak: int
    eof: bool
    done: bool
    next_endpoint: int
    last_endpoint: int
    sequence: int
    segment: int
    revision: int
    decode_count: int
    decoded_samples: int
    trace_count: int


@dataclass(frozen=True, slots=True)
class _Savepoint:
    stream_id: str
    pipeline_identity: str
    profile_id: str
    model: ModelSnapshot
    execution_profile: NativeExecutionProfile | None
    config: ContinuousStreamConfig
    options: NativeDecodeOptions
    rng_seed: int
    history_limit: int
    session: SessionState
    position: _Position
    pcm: bytes
    word_anchor: tuple[NativeTimestampSegment, ...]
    unit: SourceUnit | None
    endpoints: tuple[QuietEndpointProposal, ...]
    detector: QuietEndpointState | None


_TYPES = (
    _Savepoint,
    _Position,
    ModelSnapshot,
    ResourceVector,
    SessionState,
    WindowRecord,
    WindowResult,
    AudioSpan,
    NativeWindowResult,
    NativeDecodeMetadata,
    NativeTimestampSegment,
    NativeWordAlignment,
    AlignedPublication,
    AudioObservation,
    SilencePublication,
    NativeDecodeOptions,
    NativeExecutionProfile,
    ContinuousStreamConfig,
    SourceUnit,
    QuietEndpointConfig,
    QuietEndpointProposal,
    QuietEndpointState,
)


def _digest(value: str, *, prefixed: bool = False) -> None:
    prefix = "sha256:" if prefixed else ""
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 64
        or any(c not in "0123456789abcdef" for c in value[len(prefix) :])
    ):
        raise ValueError("expected a lowercase SHA-256 identity")


def _same_origin_witness(
    config: ContinuousStreamConfig, state: SessionState, position: _Position
) -> NativeWordAlignment | None:
    """Derive a released growing witness from already serialized publication data."""
    result = state.windows[-1].result if state.windows else None
    if (
        config.defer_word_commits
        and isinstance(result, AlignedPublication)
        and not result.final
        and not position.done
        and position.retained == result.analyzed_span.start_ms * 16
        and position.head == result.end_ms * 16
        and position.last_endpoint == result.analyzed_span.end_ms * 16
        and position.head <= position.last_endpoint <= position.accepted
        and position.last_endpoint <= position.retained + config.max_window_ms * 16
        and position.next_endpoint
        == min(
            position.last_endpoint + config.preview_interval_ms * 16,
            position.retained + config.max_window_ms * 16,
        )
    ):
        return result.alignment
    return None


def _require_boundary(stream: ContinuousTranscriptStream) -> None:
    # An aborted/retained transaction and an unresolved EOF are not savepoints.
    if (
        any(
            getattr(stream, name) is not None
            for name in (
                "_run",
                "_pending_state",
                "_resolution_retained",
                "_retry_analysis",
                "_word_pending",
                "_silence_pending",
                "_resolution",
                "_word_retained_pending",
            )
        )
        or stream._unresolved_eof
        or stream._resolution_active
        or stream._context_retry_active
        or (
            stream._context_retry is not None
            and stream._context_retry.status != "recovered"
        )
        or stream._revision
    ):
        raise NativeStreamError("save requires an idle publication boundary")
    if stream._previous is not None or stream._word_previous is not None:
        witness = _same_origin_witness(
            stream.config,
            stream.state,
            _Position(
                **{f.name: getattr(stream, "_" + f.name) for f in fields(_Position)}
            ),
        )
        if (
            witness is None
            or stream._word_previous is not witness
            or stream._previous is not witness.native
        ):
            raise NativeStreamError("save requires an idle publication boundary")
    if stream._closed or (
        stream._done and (not stream._eof or stream._head != stream._accepted)
    ):
        raise NativeStreamError("an abandoned stream is not a completed savepoint")


def _validate(saved: _Savepoint) -> Session:
    """Check cross-field invariants before constructing an executable stream."""
    _digest(saved.pipeline_identity, prefixed=True)
    p, config, state = saved.position, saved.config, saved.session
    for field in fields(p):
        value = getattr(p, field.name)
        if field.name in ("eof", "done"):
            if type(value) is not bool:
                raise ValueError("invalid checkpoint flag")
        elif type(value) is not int or value < 0:
            raise ValueError("invalid checkpoint counter")
    if type(saved.history_limit) is not int or not 1 <= saved.history_limit <= 4096:
        raise ValueError("invalid checkpoint history limit")
    if type(saved.rng_seed) is not int:
        raise ValueError("invalid checkpoint seed")
    if not saved.stream_id.strip() or state.session_id != f"{saved.stream_id}:session":
        raise ValueError("checkpoint session identity differs")
    session = Session.from_snapshot(state, history_limit=saved.history_limit)
    if (
        not 0 <= p.retained <= p.head <= p.accepted
        or len(saved.pcm) != 2 * (p.accepted - p.retained)
        or not p.accepted - p.retained <= p.peak <= config.max_buffer_ms * 16
        or p.peak > p.accepted
        or p.chunk > p.accepted
        or (p.chunk == 0) != (p.accepted == 0)
        or p.head // 16 != (state.committed_through_ms or 0)
        or p.segment != state.version
        or p.sequence < 2 * state.version
        or p.revision != 0
        or (
            p.last_endpoint != p.head and _same_origin_witness(config, state, p) is None
        )
        or (not p.done and p.next_endpoint <= p.head)
        or p.decode_count < state.version
        or (p.done and (not p.eof or p.head != p.accepted))
    ):
        raise ValueError("inconsistent checkpoint coverage or event cursors")
    if state.version == 0 and (p.head or p.retained):
        raise ValueError("uncommitted session has a committed boundary")
    previous_end = 0
    for record in state.windows:
        if (
            record.model != saved.model
            or record.result.start_ms != previous_end
            or record.committed_through_ms != record.result.end_ms
        ):
            raise ValueError("checkpoint committed history is not contiguous")
        previous_end = record.result.end_ms
    if saved.word_anchor:
        if (
            not (config.word_alignment or config.word_boundary_fallback)
            or not state.windows
        ):
            raise ValueError("checkpoint has an unexpected word anchor")
        result = state.windows[-1].result
        if not isinstance(result, AlignedPublication):
            raise ValueError("checkpoint anchor lacks its committed word provenance")
        selected = result.alignment.words[result.word_start : result.word_end][-4:]
        if not any(saved.word_anchor == selected[i:] for i in range(len(selected))):
            raise ValueError("checkpoint anchor differs from committed words")
    if saved.unit is not None and (
        not config.source_units
        or not saved.unit.start_sample <= p.head < saved.unit.end_sample <= p.accepted
    ):
        raise ValueError("checkpoint source unit differs from retained coverage")
    if config.endpointing is None:
        if saved.detector is not None or saved.endpoints:
            raise ValueError("checkpoint has unexpected endpoint state")
    else:
        if saved.detector is None or saved.detector.accepted_samples != p.accepted:
            raise ValueError("checkpoint detector input position differs")
        QuietEndpointDetector.from_snapshot(config.endpointing, saved.detector)
        last = p.head
        for proposal in saved.endpoints:
            if not last < proposal.end_sample <= p.accepted:
                raise ValueError(
                    "checkpoint endpoints are not ordered retained boundaries"
                )
            last = proposal.end_sample
    return session


def save_checkpoint(
    stream: ContinuousTranscriptStream, path: str | Path, *, pipeline_identity: str
) -> str:
    stream._require_owner()
    _digest(pipeline_identity, prefixed=True)
    with stream._lock:
        _require_boundary(stream)
        saved = _Savepoint(
            stream._id,
            pipeline_identity,
            stream.profile_id,
            stream._model,
            getattr(stream._adapter, "execution_profile", None),
            stream.config,
            stream._options,
            stream._seed,
            stream._session.history_limit,
            stream.state,
            _Position(
                **{f.name: getattr(stream, "_" + f.name) for f in fields(_Position)}
            ),
            bytes(stream._audio),
            stream._word_anchor,
            stream._unit,
            tuple(stream._endpoints),
            stream._endpoint_detector.snapshot() if stream._endpoint_detector else None,
        )
        _validate(saved)
        raw = storage.encode({"checkpoint": saved}, _TYPES)
        # Keep admission frozen until the complete savepoint is published.
        storage.write_new(path, raw)
    return sha256(raw).hexdigest()


def load_checkpoint(
    adapter: NativeWhisperAdapter,
    *,
    path: str | Path,
    expected_sha256: str,
    pipeline_identity: str,
    mel_builder: Callable[[bytes], object],
) -> ContinuousTranscriptStream:
    _digest(expected_sha256)
    _digest(pipeline_identity, prefixed=True)
    raw = storage.read(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("checkpoint differs from its trusted file digest")
    payload = storage.decode(raw, _TYPES)
    saved = payload.get("checkpoint")
    if set(payload) != {"checkpoint"} or not isinstance(saved, _Savepoint):
        raise ValueError("invalid stream checkpoint payload")
    if (
        saved.pipeline_identity != pipeline_identity
        or saved.model != adapter.model_identity
        or saved.execution_profile != getattr(adapter, "execution_profile", None)
    ):
        raise ValueError("checkpoint model, execution profile or pipeline differs")
    session = _validate(saved)
    stream = ContinuousTranscriptStream(
        adapter,
        stream_id=saved.stream_id,
        mel_builder=mel_builder,
        config=saved.config,
        options=saved.options,
        rng_seed=saved.rng_seed,
        history_limit=saved.history_limit,
    )
    if saved.profile_id != stream.profile_id:
        raise ValueError("checkpoint streaming profile differs")
    stream._session = session
    for field in fields(saved.position):
        setattr(stream, "_" + field.name, getattr(saved.position, field.name))
    stream._audio = bytearray(saved.pcm)
    stream._word_anchor = saved.word_anchor
    witness = _same_origin_witness(saved.config, saved.session, saved.position)
    if witness is not None:
        stream._word_previous = witness
        stream._previous = witness.native
    stream._unit = saved.unit
    stream._endpoints = deque(saved.endpoints)
    if saved.config.endpointing is not None and saved.detector is not None:
        stream._endpoint_detector = QuietEndpointDetector.from_snapshot(
            saved.config.endpointing, saved.detector
        )
    # No native task, generator, cache, lease or pending publication is restored.
    return stream
