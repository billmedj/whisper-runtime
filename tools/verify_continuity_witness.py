"""Replay one source-bound archived bridge on CPU; stdout only, no inference.

This consumes immutable alignments, actual commits, exact cached PCM and recorded
identities. No human transcript or post-hoc recognition score is used by the
witness. The fixed receipt hashes deliberately keep this a narrow reproducer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra import modal_noisy_context_prompt as p
from tools.analyze_stream_text_agreement import _commits
from tools.analyze_word_resolution import _alignment, _publication
from whisper_runtime import ModelSnapshot, Session, SessionState, WindowRecord
from whisper_runtime.adapters.audio_evidence import AudioObservation
from whisper_runtime.adapters.continuity_witness import assess_continuity_witness
from whisper_runtime.adapters.continuous_stream import ContinuousAnalysisIdentity
from whisper_runtime.adapters.native_whisper import NativeDecodeOptions

PROMPT_ARCHIVE = "evidence/modal-t4-tiny-en-noisy-context-prompt-2026-09-06.json"
PROMPT_SHA = "ac48e91c068d5d57ee746a04cb3cecbdf9cea2ecc493775048aba430992e9832"


def load_observations(root=p.ROOT):
    """Bind only the frozen retry trace 3 and head-only/null attempt 0."""
    root = Path(root)
    source = p.source_case(root)  # Existing cached-input/hash validation; no download.
    old_raw, new_raw = (
        (root / p.ARCHIVE).read_bytes(),
        (root / PROMPT_ARCHIVE).read_bytes(),
    )
    p._require(p.shared._sha(old_raw) == p.ARCHIVE_SHA, "paced archive changed")
    p._require(p.shared._sha(new_raw) == PROMPT_SHA, "prompt archive changed")
    old, new = json.loads(old_raw), json.loads(new_raw)
    cell = next(
        c for c in old["cells"] if (c["case_id"], c["arm"]) == ("noisy-prefix", "retry")
    )
    trace, attempt = cell["decision_traces"][3], new["cells"][1]["attempts"][0]
    joint, candidate = (
        _alignment(trace["word_alignment"]),
        _alignment(attempt["alignment"]),
    )
    state = cell["state"]
    records = tuple(
        WindowRecord(
            row["request_id"],
            ModelSnapshot(**row["model"]),
            _publication(row["result"], _alignment(row["result"]["alignment"])),
            row["committed_through_ms"],
        )
        for row in state["windows"]
    )
    published = SessionState(
        state["session_id"], state["version"], records, state["committed_through_ms"]
    )
    p._require(
        Session.from_snapshot(published).snapshot() == published,
        "publication replay differs",
    )
    commits = _commits(cell["events"])
    p._require(len(commits) == len(records) == 2, "actual commit count differs")
    for (event, text), record in zip(commits, records):
        p._require(
            (event.start_sample, event.end_sample, text)
            == (
                record.result.start_ms * 16,
                record.result.end_ms * 16,
                record.result.text,
            ),
            "published record differs from confirmed event",
        )
    p._require(
        records[-1].result == _publication(trace["word_publication"], joint),
        "joint publication differs",
    )
    p._require(
        attempt["id"] == "null" and new["cells"][1]["id"] == "head-only",
        "candidate differs",
    )
    p._require(
        attempt["session_version"] == 0 and not attempt["publication_authorized"],
        "diagnostic was published",
    )
    effective = old["effective_identity"]
    additions = {
        k: v for k, v in new["effective_identity"].items() if k not in effective
    }
    p._require(
        additions == {"reuse_decode_features": True, "tokenizer_eot": 50256},
        "new identity fields differ",
    )
    p._require(
        effective["alignment_mode"] == "legacy" and attempt["decode_attempt"] == 1,
        "initial legacy-aligned observation required",
    )
    p._require(
        effective
        == new["input"]["effective_identity"]
        == {k: new["effective_identity"][k] for k in effective}
        and old["backend"] == new["backend"]
        and old["model"] == new["model"],
        "recorded effective identities differ",
    )
    old_files, new_files = (
        {item["path"]: item for item in record["source"]["snapshot"]["files"]}
        for record in (old, new)
    )
    common = old_files.keys() & new_files.keys()
    p._require(
        all(old_files[path] == new_files[path] for path in common),
        "shared dispatched sources differ",
    )
    model = records[-1].model
    p._require(
        model.fingerprint == effective["model_sha256"]
        and model.revision == old["backend"]["commit"],
        "published model differs",
    )
    artifacts = effective["backend_artifacts"]
    # Existing identity fields: hashes of the recorded relevant metadata, not
    # new attestations of execution or ownership of the later diagnostic.
    identity = ContinuousAnalysisIdentity(
        declared_model=model,
        requested_decode_options=NativeDecodeOptions(**effective["decode_options"]),
        request_rng_seed=effective["rng_seed"],
        tokenizer_artifact_identity="sha256:"
        + p.shared._hash(
            [
                effective["tiktoken"],
                artifacts["whisper/tokenizer.py"],
                artifacts["whisper/assets/gpt2.tiktoken"],
            ]
        ),
        preprocessing_identity="sha256:"
        + p.shared._hash(
            [
                effective["preprocessing"],
                effective["numpy"],
                artifacts["whisper/audio.py"],
                artifacts["whisper/assets/mel_filters.npz"],
            ]
        ),
        backend_artifact_identity="sha256:"
        + p.shared._hash([old["backend"], effective]),
        effective_alignment_mode="legacy_encoder",
    )
    worker_sha = "f4e4e6ca348fe9d4613e3a1fe3fc76d11f6e67c7344d172117464754dd3875d7"
    p._require(
        new_files[p.WORKER]["sha256"]
        == worker_sha
        == p.shared._sha((root / p.WORKER).read_bytes()),
        "diagnostic declaration source differs",
    )
    # The hash-pinned worker declares this different backend label. Matching
    # numerical provenance must not silently rebind it to the old session.
    candidate_identity = replace(
        identity,
        declared_model=replace(model, backend="pytorch-cuda-noisy-context-prompt"),
    )
    arguments = dict(
        joint=joint,
        candidate=candidate,
        published=published,
        current=published,
        joint_audio=AudioObservation(**trace["audio_evidence"]["observation"]),
        candidate_audio=AudioObservation(**attempt["audio_evidence"]["observation"]),
        joint_identity=identity,
        candidate_identity=candidate_identity,
        pcm=source["pcm"],
    )
    provenance = dict(
        paced_archive_sha256=p.ARCHIVE_SHA,
        prompt_archive_sha256=PROMPT_SHA,
        full_pcm_sha256=source["pcm_sha256"],
        source_pcm_sha256=source["source_pcm_sha256"],
        overlap_pcm_sha256=p.shared._sha(source["pcm"][3680 * 32 : 8000 * 32]),
        joint_trace_index_zero_based=3,
        joint_decode_index=trace["decode_index"],
        candidate_cell="head-only",
        candidate_attempt="null",
        dispatched_snapshot_digests=[
            r["source"]["snapshot"]["digest"] for r in (old, new)
        ],
        identical_shared_dispatched_files=len(common),
        declared_models=[
            asdict(i.declared_model) for i in (identity, candidate_identity)
        ],
        diagnostic_declaration_source_sha256=worker_sha,
    )
    return arguments, provenance


def verify(root=p.ROOT):
    arguments, provenance = load_observations(root)
    result = assess_continuity_witness(**arguments)
    anchor_end = result.anchor.word_end if result.anchor is not None else None
    joint, candidate = arguments["joint"], arguments["candidate"]
    return dict(
        schema_version="1-read-only-continuity-witness",
        provenance=provenance,
        published_version=arguments["published"].version,
        published_head_ms=arguments["published"].committed_through_ms,
        joint_span=asdict(joint.native.analyzed_span),
        candidate_span=asdict(candidate.native.analyzed_span),
        witness=asdict(result),
        joint_suffix=[asdict(w) for w in joint.words[anchor_end:]]
        if anchor_end is not None
        else [],
        candidate_prefix=[
            asdict(w) for w in candidate.words if w.span.start_ms < joint.native.end_ms
        ],
        human_reference_used=False,
        inference_run=False,
        publication_authorized=False,
        audio_retirement_authorized=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=p.ROOT, help="repository with cached corpus"
    )
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
