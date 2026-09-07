"""Read-only correspondence with an earlier, jointly observed continuation.

No result here selects text, changes an anchor, or authorizes publication or
audio retirement. Matching declared identities is not computation attestation;
matching two recognitions cannot exclude a shared acoustic omission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..state import SessionState
from .audio_evidence import AudioObservation
from .word_policy import (
    AlignedPublication,
    AnchorDiagnostic,
    NativeWordAlignment,
    WordSequenceDiagnostic,
    diagnose_word_anchor,
    diagnose_word_sequence,
)

if TYPE_CHECKING:
    from .continuous_stream import ContinuousAnalysisIdentity


@dataclass(frozen=True, slots=True)
class ContinuityWitness:
    """Separate refusals and diagnostics; no selected slice or new boundary.

    Empty ``reasons`` means observed prefix correspondence only. ``lexical``
    omits standalone nonlexical units solely to explain representation changes;
    ``sequence`` retains every unit and still refuses such differences.
    Deltas are candidate minus joint, in complete lexical-sequence order.
    """

    reasons: tuple[str, ...]
    anchor: AnchorDiagnostic | None = None
    sequence: WordSequenceDiagnostic | None = None
    lexical: WordSequenceDiagnostic | None = None
    lexical_deltas_ms: tuple[tuple[str, int, int], ...] = ()


def assess_continuity_witness(
    *,
    joint: NativeWordAlignment,
    candidate: NativeWordAlignment,
    published: SessionState,
    current: SessionState,
    joint_audio: AudioObservation,
    candidate_audio: AudioObservation,
    joint_identity: ContinuousAnalysisIdentity | None,
    candidate_identity: ContinuousAnalysisIdentity | None,
    pcm: bytes,
    pcm_start_sample: int = 0,
    timestamp_tolerance_ms: int = 200,
) -> ContinuityWitness:
    """Compare one frozen joint observation to one later head-only observation.

    The authoritative anchor is derived only from the last actual aligned
    publication in ``published``; no candidate word is promoted into it.
    Source windows must lie on the millisecond grid, overlap, and extend from
    before the published head (joint) or exactly at it (candidate). All joint
    suffix units and all candidate units starting before the joint end are
    compared, including boundary-straddling words. There is no match search,
    punctuation repair, fitted tolerance, or human-reference input.
    """
    from .continuous_stream import ContinuousAnalysisIdentity

    for value, expected in (
        (joint, NativeWordAlignment),
        (candidate, NativeWordAlignment),
        (published, SessionState),
        (current, SessionState),
        (joint_audio, AudioObservation),
        (candidate_audio, AudioObservation),
    ):
        if not isinstance(value, expected):
            raise TypeError("invalid continuity observation type")
    if not isinstance(pcm, bytes):
        raise TypeError("pcm must be immutable s16le bytes")
    for number in (pcm_start_sample, timestamp_tolerance_ms):
        if type(number) is not int or number < 0:
            raise ValueError("sample origin and tolerance must be nonnegative integers")
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM must contain complete nonempty s16le samples")
    for identity in (joint_identity, candidate_identity):
        if identity is not None and not isinstance(
            identity, ContinuousAnalysisIdentity
        ):
            raise TypeError("identity must be ContinuousAnalysisIdentity or None")

    reasons = []
    if current != published:
        reasons.append("stale_published_state")
    if not published.windows or not isinstance(
        published.windows[-1].result, AlignedPublication
    ):
        return ContinuityWitness(tuple(reasons + ["missing_aligned_publication"]))
    record = published.windows[-1]
    publication = record.result
    assert isinstance(publication, AlignedPublication)
    head = published.committed_through_ms
    anchor = publication.alignment.words[publication.word_start : publication.word_end][
        -4:
    ]
    if (
        type(head) is not int
        or head != publication.end_ms
        or record.committed_through_ms != head
        or not anchor
        or anchor[-1].span.end_ms != head
    ):
        return ContinuityWitness(tuple(reasons + ["unbound_published_boundary"]))
    before, after = joint.native.analyzed_span, candidate.native.analyzed_span
    if not before.start_ms < head == after.start_ms < before.end_ms <= after.end_ms:
        return ContinuityWitness(tuple(reasons + ["incompatible_observation_windows"]))

    for label, alignment, audio in (
        ("joint", joint, joint_audio),
        ("candidate", candidate, candidate_audio),
    ):
        span = alignment.native.analyzed_span
        lo, hi = (
            span.start_ms * 16 - pcm_start_sample,
            span.end_ms * 16 - pcm_start_sample,
        )
        if not 0 <= lo < hi <= len(pcm) // 2 or audio != AudioObservation.from_pcm(
            pcm[lo * 2 : hi * 2]
        ):
            reasons.append(label + "_pcm_mismatch")
    if joint_identity is None or candidate_identity is None:
        reasons.append("missing_analysis_identity")
    else:
        if joint_identity != candidate_identity:
            reasons.append("analysis_identity_mismatch")
        if any(
            i.declared_model != record.model
            for i in (joint_identity, candidate_identity)
        ):
            reasons.append("published_model_mismatch")
        for identity in (joint_identity, candidate_identity):
            if any(
                not v
                for v in (
                    identity.tokenizer_artifact_identity,
                    identity.preprocessing_identity,
                    identity.backend_artifact_identity,
                    identity.effective_alignment_mode,
                )
            ):
                reasons.append("incomplete_analysis_identity")
            if identity.requested_decode_options.prompt is not None or (
                identity.requested_decode_options.prefix is not None
            ):
                reasons.append("prompt_conditioned_observation")

    diagnostic = diagnose_word_anchor(
        joint,
        anchor=anchor,
        committed_through_ms=head,
        timestamp_tolerance_ms=timestamp_tolerance_ms,
        observation="previous",
    )
    if diagnostic.status != "matched" or diagnostic.lexical_match_count != 1:
        reason = (
            "joint_anchor_ambiguous"
            if diagnostic.lexical_match_count > 1
            else ("joint_anchor_" + diagnostic.status)
        )
        return ContinuityWitness(tuple(dict.fromkeys(reasons + [reason])), diagnostic)
    assert diagnostic.word_end is not None
    left = joint.words[diagnostic.word_end :]
    right = tuple(w for w in candidate.words if w.span.start_ms < before.end_ms)
    sequence = diagnose_word_sequence(left, right)
    left_lex = tuple(w for w in left if any(c.isalnum() for c in w.text))
    right_lex = tuple(w for w in right if any(c.isalnum() for c in w.text))
    lexical = diagnose_word_sequence(left_lex, right_lex)
    if not left_lex or not right_lex:
        reasons.append("no_lexical_continuation")
    if sequence.text_relation != "exact_units" or not sequence.unit_tokens_equal:
        reasons.append(
            "representation_mismatch"
            if (
                lexical.text_relation != "different_units" and lexical.unit_tokens_equal
            )
            else "continuation_sequence_mismatch"
        )
    if any(
        v is not None and v > timestamp_tolerance_ms
        for v in (sequence.max_start_delta_ms, sequence.max_end_delta_ms)
    ):
        reasons.append("unit_timing_mismatch")
    deltas = (
        tuple(
            (
                old.text,
                new.span.start_ms - old.span.start_ms,
                new.span.end_ms - old.span.end_ms,
            )
            for old, new in zip(left_lex, right_lex)
        )
        if lexical.text_relation != "different_units"
        else ()
    )
    if any(
        max(abs(start), abs(end)) > timestamp_tolerance_ms for _, start, end in deltas
    ):
        reasons.append("lexical_timing_mismatch")
    if any(w.span.end_ms > before.end_ms for w in right):
        reasons.append("overlap_cuts_candidate_word")
    if any(w.span.start_ms == w.span.end_ms for w in left + right):
        reasons.append("zero_duration_continuation_unit")
    observed_anchor_end = joint.words[diagnostic.word_end - 1].span.end_ms
    if any(
        words and words[0].span.start_ms < max(head, observed_anchor_end)
        for words in (left, right)
    ):
        reasons.append("continuation_crosses_published_boundary")
    return ContinuityWitness(
        tuple(dict.fromkeys(reasons)), diagnostic, sequence, lexical, deltas
    )
