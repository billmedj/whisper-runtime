"""Seven native observations for a read-only, terminal overlap diagnostic.

Historical candidates remain historical evidence. No candidate in this experiment
can publish text, evict PCM, or resume a stream.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict

from infra import modal_acoustic_diagnostic as shared
from infra import modal_alignment_features as features
from infra import modal_paced_features as paced
from infra import modal_word_resolution as old
from tools.analyze_word_resolution import _alignment

ROOT = shared.ROOT
REMOTE_ROOT = shared.REMOTE_ROOT
PRODUCER = "infra/modal_resolution_handoff.py"
WORKER = "infra/resolution_handoff_worker.py"
PREREGISTRATION = "docs/research/2026-09-06-overlap-observations.md"
OLD_ARCHIVE = "evidence/modal-t4-tiny-en-word-resolution-2026-09-06.json"
OLD_SHA = "6415da1704b84663780ddd1eda8e18b6daffaae72c5c2501207474ed81078b4f"
PACED_ARCHIVE = "evidence/modal-t4-tiny-en-paced-features-2026-09-06.json"
PACED_SHA = "ada0b2584f60edbb38bce9086d897a244694936d2704ee70c452ec1bd564a4b2"
MAX_NATIVE_WINDOWS = 7
EXPERIMENT_ID = "modal-resolution-handoff-v1"
BACKEND_ARTIFACTS = {
    "whisper/tokenizer.py": "3b48e361a7e95b4ec0356ca6d72bba635778aa10269153136ee7bc34cae30b85",
    "whisper/assets/gpt2.tiktoken": "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    "whisper/model.py": "7bba626157d964910c553cf13db3ea3302dbe2e9fc20966384cad16e3f805360",
    "whisper/audio.py": "36ed287b2de3eaf40207cd2a44a8d45d57ea5c212a39df07401d4fcb4d5a5a1a",
    "whisper/decoding.py": "9bd6138ebeae0cbd5dad911a31330c89e9402d8838b0ef1880877465cdde63e5",
    "whisper/assets/mel_filters.npz": "7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c",
    "whisper/timing.py": "25c9bca776064721b5f8fbe1fff7a4d62cfc9e4f65fa5012d1c5076affc39b93",
}
CLAIMS = dict.fromkeys(
    (
        "publication_authorized",
        "full_stream_recovery",
        "omission_free_recognition",
        "general_speedup",
        "production_readiness",
    ),
    False,
)


def _corpus():
    return shared._corpus()


def _archive(root, path, digest):
    raw = (root / path).read_bytes()
    if shared._sha(raw) != digest:
        raise ValueError("historical evidence digest differs")
    return json.loads(raw)


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _slice(case, start, end):
    pcm = case["pcm"][start * 32 : end * 32]
    if not 0 <= start < end or end - start > 30000 or len(pcm) != (end - start) * 32:
        raise ValueError("invalid native window")
    return pcm


def cases(root=ROOT):
    """Rebuild inputs and bind every reused candidate to its archived PCM slice."""
    previous = _archive(root, OLD_ARCHIVE, OLD_SHA)
    paced_record = _archive(root, PACED_ARCHIVE, PACED_SHA)
    saved = old.build_cases(root)
    if _canonical(old.input_records(root)) != previous["inputs"]:
        raise ValueError("historical resolution inputs differ")
    short = paced.cases(root)
    if _canonical(paced.input_records(root)) != paced_record["inputs"]:
        raise ValueError("historical paced inputs differ")
    noisy = next(item for item in short if item["id"] == "noisy-prefix")
    cell = next(
        item
        for item in paced_record["cells"]
        if item["case_id"] == "noisy-prefix" and item["arm"] == "baseline"
    )
    trace = cell["decision_traces"][-1]
    state = cell["state"]
    last = state["windows"][-1]["result"]
    anchor = last["alignment"]["words"][last["word_start"] : last["word_end"]][-4:]
    head, retained, end = (
        trace[key] // 16
        for key in (
            "committed_before_sample",
            "retained_from_sample",
            "analysis_end_sample",
        )
    )
    if (head, retained, end) != (3680, 1680, 10890) or not trace["eof"]:
        raise ValueError("registered terminal paced state differs")
    commits = _corpus().c._normalized_committed_text(cell["events"])
    result = [
        dict(
            id="noisy-prefix",
            pcm=noisy["pcm"],
            reference_text=noisy["reference_text"],
            frozen_state=dict(
                head_ms=head,
                retained_ms=retained,
                end_ms=end,
                committed_text=commits,
                anchor=anchor,
            ),
            session_version=state["version"],
            source_reason=trace["reason"],
            source_native=trace["result"],
            archived_current=None,
            archived_candidate=None,
            archive=dict(
                path=PACED_ARCHIVE,
                sha256=PACED_SHA,
                cell_id="noisy-prefix:baseline",
                state_authority="saved_terminal_state",
                backend_tree=features.PATCHED_TREE,
            ),
            source_trace=trace,
            source_diagnostic=trace["anchor_diagnostic"],
            windows=[
                dict(arm=arm, start_ms=start, end_ms=end)
                for arm, start in (
                    ("current", 1680),
                    ("candidate", 3680),
                    ("overlap", 2020),
                )
            ],
        )
    ]
    for item, archived, overlap_start in zip(
        saved, previous["cells"], (20720, 32400, 1120, 1100)
    ):
        if item["id"] != archived["id"]:
            raise ValueError("historical case order differs")
        frozen = archived["frozen_state"]
        if "bootstrap_ms" in item:
            from infra.word_resolution_worker import _bootstrap

            reconstructed = _bootstrap(
                item,
                _alignment(archived["raw_alignments"]["bootstrap"]),
                old.PARAMETERS,
            )
            rebuilt = {
                key: reconstructed[key]
                for key in ("head_ms", "retained_ms", "end_ms", "committed_text")
            }
            rebuilt["anchor"] = [
                asdict(word) for word in reconstructed["frozen_anchor"]
            ]
            if _canonical(rebuilt) != frozen:
                raise ValueError("historical synthetic prefix differs")
        for arm, start in (
            ("current", frozen["retained_ms"]),
            ("alternative", frozen["head_ms"]),
        ):
            timing = archived["timings"][arm]
            if (
                timing["start_ms"] != start
                or timing["end_ms"] != frozen["end_ms"]
                or timing["pcm_sha256"]
                != shared._sha(_slice(item, start, frozen["end_ms"]))
                or timing["sample_count"] != (frozen["end_ms"] - start) * 16
            ):
                raise ValueError("historical native input differs")
        result.append(
            dict(
                id=item["id"],
                pcm=item["pcm"],
                reference_text=item["reference_text"],
                frozen_state=frozen,
                session_version=1,
                source_reason=archived["outcome"]["strict_reason"],
                source_native=archived["raw_alignments"]["current"]["native"],
                archived_current=archived["raw_alignments"]["current"],
                archived_candidate=archived["raw_alignments"]["alternative"],
                source_trace=None,
                source_diagnostic=archived["current_diagnostic"],
                archive=dict(
                    path=OLD_ARCHIVE,
                    sha256=OLD_SHA,
                    cell_id=item["id"],
                    state_authority=archived["state_authority"],
                    backend_tree=features.BASE_TREE,
                ),
                windows=[
                    dict(arm="overlap", start_ms=overlap_start, end_ms=frozen["end_ms"])
                ],
            )
        )
    for item in result:
        expected_overlap = (
            item["frozen_state"]["anchor"][0]["span"]["start_ms"] // 20 * 20
        )
        if item["windows"][-1]["start_ms"] != expected_overlap:
            raise ValueError("registered overlap is not the fixed anchor onset")
        for window in item["windows"]:
            _slice(item, window["start_ms"], window["end_ms"])
    return result


def input_records(root=ROOT):
    return [
        dict(
            id=item["id"],
            sample_count=len(item["pcm"]) // 2,
            pcm_sha256=shared._sha(item["pcm"]),
            archive=item["archive"],
            frozen_state=item["frozen_state"],
            session_version=item["session_version"],
            source_reason=item["source_reason"],
            source_native_sha256=shared._hash(item["source_native"]),
            archived_candidate_sha256=None
            if item["archived_candidate"] is None
            else shared._hash(item["archived_candidate"]),
            reference_text=item["reference_text"],
            windows=[
                {
                    **window,
                    "pcm_sha256": shared._sha(
                        _slice(item, window["start_ms"], window["end_ms"])
                    ),
                    "sample_count": (window["end_ms"] - window["start_ms"]) * 16,
                }
                for window in item["windows"]
            ],
        )
        for item in cases(root)
    ]


def snapshot(root=ROOT):
    # Detect missing PCM or a stale historical-state join before paid dispatch.
    input_records(root)
    base = paced.snapshot(root)
    names = {item["path"] for item in base["files"]}
    names.update((PRODUCER, WORKER, PREREGISTRATION, OLD_ARCHIVE, PACED_ARCHIVE))
    files = [
        dict(path=name, size_bytes=len(raw), sha256=shared._sha(raw))
        for name in sorted(names)
        for raw in [(root / name).read_bytes()]
    ]
    return dict(commit=base["commit"], digest=shared._hash(files), files=files)


def scope():
    return dict(
        max_native_windows=7,
        model="tiny.en",
        dtype="float32",
        device="Tesla T4",
        alignment="legacy",
        borrowed_features=False,
        inference_owners=1,
        cuda_lanes=1,
        warmup_windows=0,
        worker_timeout_seconds=180,
        actual_stream_commits=0,
        historical_candidate_reuse=True,
        gpu_device_time_measured=False,
        recognition_reference_used_for_routing=False,
        performance_benchmark=False,
    )


def provenance_checks(case, effective, backend, root=ROOT):
    """Check actual worker identities against a source-reconstructed legacy recipe.

    The historical runs did not measure each package/artifact identity separately.
    Their immutable backend tree and source snapshot identify that recipe instead.
    The optional patch changes the tree; its unselected default branch was audited.
    This is a declared, source-bound comparison, not attestation of GPU execution.
    """
    from whisper_runtime.adapters.native_whisper import NativeDecodeOptions

    archive = _archive(root, case["archive"]["path"], case["archive"]["sha256"])
    base = _corpus().read_registration(root)
    historical_files = {
        item["path"]: item["sha256"] for item in archive["source"]["snapshot"]["files"]
    }
    checks = dict(
        model_state=(
            effective.get("model_sha256")
            == archive["model"]["initial_sha256"]
            == archive["model"]["final_sha256"]
            == base["model"]["model_state_sha256"]
        ),
        checkpoint=effective.get("checkpoint_sha256")
        == base["model"]["checkpoint_sha256"],
        shared_recipe=(
            historical_files.get("experiments/modal-word-corpus-v1.json")
            == shared._sha(
                (root / "experiments/modal-word-corpus-v1.json").read_bytes()
            )
        ),
        backend_artifacts=effective.get("backend_artifacts") == BACKEND_ARTIFACTS,
        patch_bridge=(
            backend.get("base_commit") == features.BASE_COMMIT
            and backend.get("base_tree") == features.BASE_TREE
            and backend.get("patched_tree") == features.PATCHED_TREE
            and backend.get("patch_sha256") == features.PATCH_SHA
            and backend.get("timing_sha256") == BACKEND_ARTIFACTS["whisper/timing.py"]
        ),
        options=effective.get("decode_options")
        == _canonical(asdict(NativeDecodeOptions(**base["decode_options"]))),
        rng_seed=effective.get("rng_seed") == 7,
        precision=effective.get("precision") == "float32",
        legacy_alignment=effective.get("alignment_mode") == "legacy",
        attention=effective.get("attention")
        == dict(decode="backend-default-sdpa", alignment="explicit-non-sdpa"),
        preprocessing=effective.get("preprocessing")
        == "s16le-to-float32-div32768/pad-or-trim-480000/log-mel-80x3000/contiguous",
        package_recipe=(
            effective.get("torch") == "2.6.0+cu124"
            and effective.get("numpy") == "2.5.2"
            and effective.get("tiktoken") == "0.14.0"
        ),
    )
    return dict(
        comparable=all(checks.values()),
        checks=checks,
        basis="source-reconstructed-legacy-pipeline",
        historical_package_identities_measured=False,
        backend_trees_equal=case["archive"]["backend_tree"] == features.PATCHED_TREE,
        historical_candidate_is_fresh=False,
        unused_optional_path_bridge=case["archive"]["backend_tree"]
        == features.BASE_TREE,
    )


def assess_case(case, raw_alignments):
    """Recompute structural correspondence only; model provenance is separate."""
    from tools.word_anchor_reconciliation import assess_resolution_handoff
    from whisper_runtime.adapters.audio_evidence import (
        AudioEvidenceDecision,
        AudioObservation,
        assess_audio,
    )
    from whisper_runtime.adapters.continuous_stream import (
        ContinuousDecodeTrace,
        ContinuousResolutionObservation,
    )
    from whisper_runtime.adapters.native_result import NativeTimestampSegment
    from whisper_runtime.adapters.word_policy import AnchorDiagnostic
    from whisper_runtime.state import AudioSpan

    current, candidate, overlap = (
        _alignment(raw_alignments[arm]) for arm in ("current", "candidate", "overlap")
    )
    frozen = case["frozen_state"]
    start, head, end = frozen["retained_ms"], frozen["head_ms"], frozen["end_ms"]
    anchor = tuple(
        NativeTimestampSegment(
            AudioSpan(**word["span"]), word["text"], tuple(word["tokens"])
        )
        for word in frozen["anchor"]
    )
    saved_evidence = (
        None if case["source_trace"] is None else case["source_trace"]["audio_evidence"]
    )
    evidence = (
        assess_audio(
            AudioObservation.from_pcm(_slice(case, start, end)), current.native
        )
        if saved_evidence is None
        else AudioEvidenceDecision(
            **{
                key: value
                for key, value in saved_evidence.items()
                if key != "observation"
            },
            observation=AudioObservation(**saved_evidence["observation"]),
        )
    )
    source = ContinuousDecodeTrace(
        decode_index=1,
        analysis_start_sample=start * 16,
        analysis_end_sample=end * 16,
        committed_before_sample=head * 16,
        retained_from_sample=start * 16,
        eof=True,
        reason=case["source_reason"],
        publication_span=None,
        result=current.native,
        action="unresolved",
        word_alignment=current,
        audio_evidence=evidence,
        accepted_through_sample=end * 16,
        anchor_diagnostic=None
        if case["source_diagnostic"] is None
        else AnchorDiagnostic(**case["source_diagnostic"]),
    )

    def observation(alignment):
        span = alignment.native.analyzed_span
        return ContinuousResolutionObservation(
            source=source,
            anchor=anchor,
            session_version=case["session_version"],
            analysis_start_sample=span.start_ms * 16,
            analysis_end_sample=span.end_ms * 16,
            status="observed",
            reason="diagnostic_native_observation",
            pcm_sha256=shared._sha(_slice(case, span.start_ms, span.end_ms)),
            candidate=alignment,
        )

    result = assess_resolution_handoff(
        observation(candidate),
        current_session_version=case["session_version"],
        committed_through_sample=head * 16,
        retained_from_sample=start * 16,
        retained_pcm=_slice(case, start, end),
        overlap=observation(overlap),
        overlap_attempted=True,
    )
    return _canonical(asdict(result))


def native_equal(left, right):
    return shared._hash({**left, "window_id": "same-input"}) == shared._hash(
        {**right, "window_id": "same-input"}
    )


def summarize_cell(case, raw_alignments):
    """Preserve refusal evidence and report proposed text without authorizing it."""
    assessment = assess_case(case, raw_alignments)
    text = " ".join(
        (
            case["frozen_state"]["committed_text"].strip(),
            raw_alignments["candidate"]["native"]["text"].strip(),
        )
    ).strip()
    return dict(
        assessment=assessment,
        control_native_reproduced=native_equal(
            raw_alignments["current"]["native"], case["source_native"]
        ),
        proposed_text=text,
        against_human_reference=_corpus().b._word_difference(
            text, case["reference_text"]
        ),
        publication_authorized=False,
        source_observation_kind="synthetic_assessment_join_of_fresh_control_and_saved_refusal"
        if case["archived_current"] is None
        else "reconstructed_saved_state_with_archived_alignment",
    )


def _validate_record_basic(record, expected):
    """Validate frozen inputs and independently replay each complete observation."""
    if (
        record["source"]["snapshot"] != expected
        or record["scope"] != scope()
        or record["claim_boundary"] != CLAIMS
        or record["qualified"] is not False
        or record["inputs"] != input_records()
        or record["experiment_id"] != "modal-resolution-handoff-v1"
    ):
        raise ValueError("diagnostic identity differs")
    inventory = cases()
    cells = record["cells"]
    if len(cells) > 5 or [item["id"] for item in cells] != [
        item["id"] for item in inventory[: len(cells)]
    ]:
        raise ValueError("cell order differs")
    windows = [window for cell in cells for window in cell["fresh_windows"]]
    if (
        record["native_window_count"]
        != sum(w["measurement"] is not None for w in windows)
        or len(windows) > 7
    ):
        raise ValueError("native window bound differs")
    for case, cell in zip(inventory, cells):
        if cell["status"] == "evaluated":
            if cell["summary"] != summarize_cell(case, cell["raw_alignments"]):
                raise ValueError("structural replay differs")
            if len(cell["fresh_windows"]) != len(case["windows"]):
                raise ValueError("evaluated cell lacks its windows")
    completed = (
        len(cells) == 5
        and len(windows) == 7
        and record["capacity_restored"] is True
        and record["model"]["unchanged"] is True
        and all(cell["status"] == "evaluated" for cell in cells)
    )
    if record["status"] != ("completed" if completed else "failed"):
        raise ValueError("completion claim differs")


def validate_record(record, expected):
    """Bind archived and fresh outputs separately before interpreting a summary."""
    if record["schema_version"] != "1-diagnostic":
        raise ValueError("record schema differs")
    _validate_record_basic(record, expected)

    def integer(value):
        return type(value) is int and value >= 0

    model, effective = record["model"], record["effective_identity"]
    registered_model = _corpus().read_registration(ROOT)["model"]
    if (
        model["initial_sha256"] != registered_model["model_state_sha256"]
        or model["initial_sha256"] != effective["model_sha256"]
        or model["checkpoint_sha256"] != registered_model["checkpoint_sha256"]
        or model["checkpoint_sha256"] != effective["checkpoint_sha256"]
        or model["backend_revision"] != features.PATCHED_TREE
        or model["unchanged"] is not (model["final_sha256"] == model["initial_sha256"])
        or type(record["capacity_restored"]) is not bool
        or not integer(record["elapsed_ns"])
        or not integer(record["native_window_count"])
    ):
        raise ValueError("model or lifecycle identity differs")
    calls = []
    for case, cell in zip(cases(), record["cells"]):
        comparison = provenance_checks(case, effective, record["backend"])
        if (
            cell["comparability"] != comparison
            or cell["archive"] != case["archive"]
            or cell["original_source_trace"] != case["source_trace"]
            or cell["original_anchor_diagnostic"] != case["source_diagnostic"]
            or cell["publication_authorized"] is not False
            or cell["status"]
            not in {"evaluated", "blocked", "assessment_failed", "lifecycle_failure"}
        ):
            raise ValueError("cell provenance or authority differs")
        raw, observed, planned = (
            cell["raw_alignments"],
            cell["fresh_windows"],
            case["windows"],
        )
        if len(observed) > len(planned):
            raise ValueError("extra native window")
        if not comparison["comparable"]:
            if (
                cell["status"] != "blocked"
                or observed
                or raw
                or cell["summary"] is not None
            ):
                raise ValueError("incompatible evidence was used")
            continue
        if case["archived_current"] is not None:
            for arm, archived in (
                ("current", case["archived_current"]),
                ("candidate", case["archived_candidate"]),
            ):
                if raw.get(arm) != archived:
                    raise ValueError("historical alignment changed")
        allowed = {window["arm"] for window in observed}
        if case["archived_current"] is not None:
            allowed.update(("current", "candidate"))
        if set(raw) - allowed:
            raise ValueError("unmeasured alignment")
        for window, plan in zip(observed, planned):
            start, end, arm = plan["start_ms"], plan["end_ms"], plan["arm"]
            if (
                any(window[key] != plan[key] for key in ("arm", "start_ms", "end_ms"))
                or window["pcm_sha256"] != shared._sha(_slice(case, start, end))
                or window["sample_count"] != (end - start) * 16
                or type(window["completed"]) is not bool
            ):
                raise ValueError("fresh native input differs")
            measurement = window["measurement"]
            if measurement is not None:
                calls.append(measurement["call_index"])
                if (
                    not integer(calls[-1])
                    or calls[-1] != len(calls)
                    or measurement["start_ms"] != start
                    or measurement["end_ms"] != end
                    or measurement["window_id"] != f"handoff:{case['id']}:{arm}"
                    or measurement["reuse_alignment_features"] is not False
                    or measurement["gpu_device_time_ms"] is not None
                    or not all(
                        integer(value)
                        for value in measurement["operation_wall_ns"].values()
                    )
                    or "finish" in measurement["operation_wall_ns"]
                ):
                    raise ValueError("native measurement identity differs")
            if arm in raw:
                alignment = _alignment(raw[arm])
                if (
                    alignment.native.analyzed_span.start_ms != start
                    or alignment.native.analyzed_span.end_ms != end
                    or alignment.native.window_id != f"handoff:{case['id']}:{arm}"
                ):
                    raise ValueError(
                        "fresh alignment does not match its measured window"
                    )
            if window["completed"]:
                if measurement is None or arm not in raw:
                    raise ValueError("completed observation lacks native work")
                if (
                    window["error"] is not None
                    or measurement["closed"] is not True
                    or measurement["capacity_restored"] is not True
                    or window["capacity_restored"] is not True
                    or window["mel_shape"] != [80, 3000]
                    or not integer(window["decoder_steps"])
                ) and cell["status"] != "lifecycle_failure":
                    raise ValueError("completed native lifecycle differs")
                phases = [item["phase"] for item in measurement["encoder_calls"]]
                if phases not in (["decode"], ["decode", "alignment"]) or any(
                    item["input_frames"] != 3000
                    or item["use_sdpa"]
                    is not (False if item["phase"] == "alignment" else None)
                    for item in measurement["encoder_calls"]
                ):
                    raise ValueError("legacy encoder work differs")
        if cell["summary"] is not None:
            if set(raw) != {"current", "candidate", "overlap"} or cell[
                "summary"
            ] != _canonical(summarize_cell(case, raw)):
                raise ValueError("structural replay differs")
            expected_status = (
                "evaluated"
                if cell["summary"]["control_native_reproduced"]
                else "blocked"
            )
            if cell["status"] != expected_status:
                raise ValueError("control reproduction claim differs")
        if cell["status"] in {"evaluated", "assessment_failed"} and (
            len(observed) != len(planned)
            or any(not window["completed"] for window in observed)
            or (cell["status"] == "evaluated" and cell["summary"] is None)
        ):
            raise ValueError("evaluated cell lacks complete observations")
    if record["native_window_count"] != len(calls) or len(calls) > MAX_NATIVE_WINDOWS:
        raise ValueError("native window bound differs")
    if record["status"] == "completed" and record["stop"] is not None:
        raise ValueError("completed record reports an early stop")


def initial_alignments(case):
    """Return recorded inputs separately from observations made by this worker."""
    if case["archived_current"] is None:
        return {}
    return dict(current=case["archived_current"], candidate=case["archived_candidate"])


def summary_is_comparable(summary):
    return summary["control_native_reproduced"]


def run_worker(expected_snapshot):
    return importlib.import_module("infra.resolution_handoff_worker").run_worker(
        expected_snapshot
    )


if __name__ == "__main__":
    from infra.modal_cuda_lane import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(
        run(
            producer=importlib.import_module("infra.modal_resolution_handoff"),
            **vars(parser.parse_args()),
        )
    )
