"""Three new fixed guard windows plus two byte-bound historical observations.

Read-only legacy-alignment diagnostic. No retry, publication, or acceptance-rule
change is authorized by an evaluated cell or a completed record.
"""

from __future__ import annotations

import argparse
import importlib

from infra import modal_resolution_handoff as old
from tools.analyze_word_resolution import _alignment

shared, features, paced = old.shared, old.features, old.paced
ROOT, REMOTE_ROOT = old.ROOT, old.REMOTE_ROOT
PRODUCER = "infra/modal_context_guard.py"
WORKER = old.WORKER
PREREGISTRATION = "docs/research/2026-09-06-context-guard-observations.md"
ARCHIVE = "evidence/modal-t4-tiny-en-resolution-handoff-2026-09-06.json"
ARCHIVE_SHA = "545e6e8d0d420539b5e94333e5aeeba2095886577f9daca0382cc183344b9a6c"
EXPERIMENT_ID = "modal-context-guard-v1"
GUARD_MS = 500
MAX_NATIVE_WINDOWS = 3
BACKEND_ARTIFACTS = old.BACKEND_ARTIFACTS
CLAIMS = dict(old.CLAIMS)
_canonical, _slice, _corpus = old._canonical, old._slice, old._corpus
REGISTERED_WINDOWS = (
    ("noisy-prefix", 1680, 10890, "reused"),
    ("context:concatenated-no-added-pauses", 20220, 33660, "new"),
    ("context:mixed-continuous-noise64", 31900, 43660, "new"),
    ("heldout:2961-960-0004:clean", 620, 8220, "new"),
    ("heldout:8455-210777-0028:noise64", 640, 7740, "reused"),
)


def _seven(root=ROOT):
    return old._archive(root, ARCHIVE, ARCHIVE_SHA)


def cases(root=ROOT):
    """Bind the fixed schedule, every reused raw output, and actual PCM bytes."""
    archive = _seven(root)
    inventory = old.cases(root)
    if (
        _canonical(old.input_records(root)) != archive["inputs"]
        or len(inventory) != len(REGISTERED_WINDOWS)
        or len(archive["cells"]) != len(REGISTERED_WINDOWS)
    ):
        raise ValueError("seven-window historical input join differs")
    result = []
    for case, saved, registered in zip(inventory, archive["cells"], REGISTERED_WINDOWS):
        case_id, start, end, kind = registered
        frozen = case["frozen_state"]
        calculated = max(
            frozen["retained_ms"],
            frozen["anchor"][0]["span"]["start_ms"] // 20 * 20 - GUARD_MS,
        )
        if (
            case["id"] != saved["id"]
            or case["id"] != case_id
            or (calculated, frozen["end_ms"]) != (start, end)
            or saved["status"] != "evaluated"
            or not saved["comparability"]["comparable"]
        ):
            raise ValueError("registered guard or archived observation differs")
        raw = _canonical(saved["raw_alignments"])
        if set(raw) != {"current", "candidate", "overlap"}:
            raise ValueError("archived alignment inventory differs")
        for window in saved["fresh_windows"]:
            if window["pcm_sha256"] != shared._sha(
                _slice(case, window["start_ms"], window["end_ms"])
            ):
                raise ValueError("seven-window PCM differs")
        pcm_hash = shared._sha(_slice(case, start, end))
        guard = dict(
            kind=kind,
            arm="guard",
            start_ms=start,
            end_ms=end,
            pcm_sha256=pcm_hash,
            sample_count=(end - start) * 16,
            requested_guard_ms=GUARD_MS,
            added_context_ms=frozen["anchor"][0]["span"]["start_ms"] // 20 * 20 - start,
            historical_source=None,
            historical_cost=None,
        )
        if kind == "reused":
            span = _alignment(raw["current"]).native.analyzed_span
            if (span.start_ms, span.end_ms) != (start, end):
                raise ValueError("reused guard span differs")
            if case_id == "noisy-prefix":
                cost = next(w for w in saved["fresh_windows"] if w["arm"] == "current")
                source = dict(
                    path=ARCHIVE, sha256=ARCHIVE_SHA, cell_id=case_id, arm="current"
                )
            else:
                previous = old._archive(root, old.OLD_ARCHIVE, old.OLD_SHA)
                source_cell = next(c for c in previous["cells"] if c["id"] == case_id)
                cost = source_cell["timings"]["current"]
                if raw["current"] != source_cell["raw_alignments"]["current"]:
                    raise ValueError("legacy reused guard raw output differs")
                source = dict(
                    path=old.OLD_ARCHIVE,
                    sha256=old.OLD_SHA,
                    cell_id=case_id,
                    arm="current",
                )
            if (
                cost["pcm_sha256"] != pcm_hash
                or (cost["start_ms"], cost["end_ms"]) != (start, end)
                or cost["sample_count"] != guard["sample_count"]
            ):
                raise ValueError("reused guard PCM or interval differs")
            guard.update(historical_source=source, historical_cost=cost)
        result.append(
            {
                **case,
                "context_archive": dict(
                    path=ARCHIVE, sha256=ARCHIVE_SHA, cell_id=case_id
                ),
                "historical_alignments": raw,
                "historical_summary": saved["summary"],
                "historical_fresh_windows": saved["fresh_windows"],
                "guard_observation": guard,
                "windows": []
                if kind == "reused"
                else [dict(arm="guard", start_ms=start, end_ms=end)],
            }
        )
    return result


def initial_alignments(case):
    raw = _canonical(case["historical_alignments"])
    if case["guard_observation"]["kind"] == "reused":
        raw["guard"] = _canonical(raw["current"])
    return raw


def input_records(root=ROOT):
    return [
        dict(
            id=case["id"],
            sample_count=len(case["pcm"]) // 2,
            pcm_sha256=shared._sha(case["pcm"]),
            archive=case["archive"],
            context_archive=case["context_archive"],
            frozen_state=case["frozen_state"],
            session_version=case["session_version"],
            source_reason=case["source_reason"],
            source_native_sha256=shared._hash(case["source_native"]),
            historical_alignment_sha256={
                arm: shared._hash(raw)
                for arm, raw in case["historical_alignments"].items()
            },
            historical_summary=case["historical_summary"],
            historical_fresh_windows=case["historical_fresh_windows"],
            guard_observation=case["guard_observation"],
            reference_text=case["reference_text"],
            windows=[
                {
                    **window,
                    "pcm_sha256": shared._sha(
                        _slice(case, window["start_ms"], window["end_ms"])
                    ),
                    "sample_count": (window["end_ms"] - window["start_ms"]) * 16,
                }
                for window in case["windows"]
            ],
        )
        for case in cases(root)
    ]


def snapshot(root=ROOT):
    input_records(root)
    base = old.snapshot(root)
    names = {item["path"] for item in base["files"]}
    names.update(
        (
            PRODUCER,
            WORKER,
            PREREGISTRATION,
            ARCHIVE,
            "docs/research/2026-09-06-resolution-disagreements.md",
            "tools/test_modal_context_guard.py",
            "tools/analyze_resolution_disagreements.py",
            "tools/analyze_word_resolution.py",
            "tools/analyze_stream_text_agreement.py",
        )
    )
    files = [
        dict(path=name, size_bytes=len(raw), sha256=shared._sha(raw))
        for name in sorted(names)
        for raw in [(root / name).read_bytes()]
    ]
    return dict(commit=base["commit"], digest=shared._hash(files), files=files)


def scope():
    return {
        **old.scope(),
        "max_native_windows": MAX_NATIVE_WINDOWS,
        "requested_guard_ms": GUARD_MS,
        "guard_observations": 5,
        "reused_guard_observations": 2,
        "fresh_control_reproduction": False,
        "retries": 0,
        "timing_tolerance_ms": 200,
        "historical_refusals_preserved": True,
        "casefold_acceptance": False,
    }


def provenance_checks(case, effective, backend, root=ROOT):
    previous = _seven(root)
    source = old.provenance_checks(case, effective, backend, root)
    guard = case["guard_observation"]
    checks = {
        **source["checks"],
        "seven_window_effective_identity": _canonical(effective)
        == previous["effective_identity"],
        "seven_window_backend_identity": backend == previous["backend"],
        "guard_pcm": guard["pcm_sha256"]
        == shared._sha(_slice(case, guard["start_ms"], guard["end_ms"])),
    }
    return {
        **source,
        "checks": checks,
        "comparable": all(checks.values()),
        "basis": "source-reconstructed-legacy-pipeline-and-exact-seven-window-effective-identity",
        "context_archive": case["context_archive"],
        "historical_native_control_reused": True,
        "fresh_control_reproduction": False,
        "guard_observation_kind": guard["kind"],
    }


def summarize_cell(case, raw_alignments):
    from tools.analyze_resolution_disagreements import summarize_guard_correspondence

    preserved = all(
        raw_alignments.get(arm) == value
        for arm, value in case["historical_alignments"].items()
    )
    correspondence = summarize_guard_correspondence(
        case["frozen_state"], raw_alignments["guard"], raw_alignments["candidate"]
    )
    return _canonical(
        dict(
            historical_summary=case["historical_summary"],
            historical_native_control_preserved=preserved,
            historical_native_control_reused=True,
            fresh_control_reproduction=False,
            guard_observation=case["guard_observation"],
            guard_correspondence=correspondence,
            publication_authorized=False,
        )
    )


def summary_is_comparable(summary):
    return summary["historical_native_control_preserved"] is True


def _nonnegative_integer(value):
    return type(value) is int and value >= 0


def _validate_window(case, window, plan, raw, call_count, status):
    start, end, arm = plan["start_ms"], plan["end_ms"], plan["arm"]
    if (
        any(window[key] != plan[key] for key in ("arm", "start_ms", "end_ms"))
        or window["pcm_sha256"] != shared._sha(_slice(case, start, end))
        or window["sample_count"] != (end - start) * 16
        or type(window["completed"]) is not bool
        or window["first_call_no_warmup"] is not (call_count == 0)
    ):
        raise ValueError("fresh native input differs")
    measurement = window["measurement"]
    if measurement is not None:
        call_count += 1
        if (
            not _nonnegative_integer(measurement["call_index"])
            or measurement["call_index"] != call_count
            or measurement["start_ms"] != start
            or measurement["end_ms"] != end
            or measurement["window_id"] != f"handoff:{case['id']}:{arm}"
            or measurement["reuse_alignment_features"] is not False
            or measurement["gpu_device_time_ms"] is not None
            or not all(
                _nonnegative_integer(v)
                for v in measurement["operation_wall_ns"].values()
            )
            or "finish" in measurement["operation_wall_ns"]
        ):
            raise ValueError("native measurement identity differs")
    if arm in raw:
        native = _alignment(raw[arm]).native
        if (
            (native.analyzed_span.start_ms, native.analyzed_span.end_ms) != (start, end)
            or native.window_id != f"handoff:{case['id']}:{arm}"
            or not window["completed"]
            or measurement is None
        ):
            raise ValueError("fresh alignment lacks its completed measured window")
    if window["completed"]:
        if measurement is None or arm not in raw:
            raise ValueError("completed observation lacks native work")
        if (
            window["error"] is not None
            or measurement["closed"] is not True
            or measurement["capacity_restored"] is not True
            or window["capacity_restored"] is not True
            or window["mel_shape"] != [80, 3000]
            or not _nonnegative_integer(window["decoder_steps"])
        ) and status != "lifecycle_failure":
            raise ValueError("completed native lifecycle differs")
        phases = [item["phase"] for item in measurement["encoder_calls"]]
        if phases not in (["decode"], ["decode", "alignment"]) or any(
            item["input_frames"] != 3000
            or item["use_sdpa"] is not (False if item["phase"] == "alignment" else None)
            for item in measurement["encoder_calls"]
        ):
            raise ValueError("legacy encoder work differs")
        if status != "lifecycle_failure" and (
            set(measurement["operation_wall_ns"])
            != {"decode", "result", "alignment", "close"}
            or not all(
                _nonnegative_integer(window.get(key))
                for key in (
                    "mel_wall_ns",
                    "wall_ns",
                    "memory_before_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "memory_after_close_bytes",
                    "memory_after_handle_release_bytes",
                )
            )
        ):
            raise ValueError("completed observation lacks measured cost")
    return call_count


def validate_record(record, expected):
    """Replay all diagnostics while keeping history distinct from three new calls."""
    if (
        record["schema_version"] != "1-diagnostic"
        or record["experiment_id"] != EXPERIMENT_ID
        or record["source"]["snapshot"] != expected
        or record["scope"] != scope()
        or record["claim_boundary"] != CLAIMS
        or record["qualified"] is not False
        or record["inputs"] != input_records()
    ):
        raise ValueError("diagnostic identity differs")
    inventory, cells = cases(), record["cells"]
    if len(cells) > len(inventory) or [c["id"] for c in cells] != [
        c["id"] for c in inventory[: len(cells)]
    ]:
        raise ValueError("cell order differs")
    model, effective = record["model"], record["effective_identity"]
    registered = _corpus().read_registration(ROOT)["model"]
    if (
        model["initial_sha256"] != registered["model_state_sha256"]
        or model["initial_sha256"] != effective["model_sha256"]
        or model["checkpoint_sha256"] != registered["checkpoint_sha256"]
        or model["checkpoint_sha256"] != effective["checkpoint_sha256"]
        or model["backend_revision"] != features.PATCHED_TREE
        or model["unchanged"] is not (model["final_sha256"] == model["initial_sha256"])
        or type(record["capacity_restored"]) is not bool
        or not _nonnegative_integer(record["elapsed_ns"])
        or not _nonnegative_integer(record["native_window_count"])
    ):
        raise ValueError("model or lifecycle identity differs")
    count, windows, failure = 0, [], None
    for case, cell in zip(inventory, cells):
        comparison = provenance_checks(case, effective, record["backend"])
        if (
            cell["comparability"] != comparison
            or cell["archive"] != case["archive"]
            or cell["original_source_trace"] != case["source_trace"]
            or cell["original_anchor_diagnostic"] != case["source_diagnostic"]
            or cell["publication_authorized"] is not False
            or cell["status"]
            not in {"evaluated", "blocked", "assessment_failed", "lifecycle_failure"}
            or failure is not None
        ):
            raise ValueError("cell provenance or authority differs")
        raw, observed, planned = (
            cell["raw_alignments"],
            cell["fresh_windows"],
            case["windows"],
        )
        if len(observed) > len(planned):
            raise ValueError("extra native window")
        windows.extend(observed)
        if not comparison["comparable"]:
            if (
                cell["status"] != "blocked"
                or observed
                or raw
                or cell["summary"] is not None
            ):
                raise ValueError("incompatible evidence was used")
            continue
        historical = initial_alignments(case)
        if any(raw.get(arm) != value for arm, value in historical.items()):
            raise ValueError("historical alignment changed")
        if set(raw) - (set(historical) | {w["arm"] for w in observed}):
            raise ValueError("unmeasured alignment")
        for window, plan in zip(observed, planned):
            count = _validate_window(case, window, plan, raw, count, cell["status"])
        if cell["summary"] is not None:
            if set(raw) != {"current", "candidate", "overlap", "guard"} or cell[
                "summary"
            ] != summarize_cell(case, raw):
                raise ValueError("guard summary replay differs")
            expected_status = (
                "evaluated" if summary_is_comparable(cell["summary"]) else "blocked"
            )
            if cell["status"] != expected_status:
                raise ValueError("historical control preservation claim differs")
        elif cell["status"] == "blocked":
            raise ValueError("comparable cell lacks its blocked summary")
        if cell["status"] in {"evaluated", "assessment_failed"} and (
            len(observed) != len(planned)
            or any(not w["completed"] for w in observed)
            or (
                cell["status"] == "evaluated"
                and (cell["summary"] is None or cell["error"] is not None)
            )
        ):
            raise ValueError("evaluated cell lacks complete observations")
        if cell["status"] == "lifecycle_failure":
            if cell["error"] is None or cell["summary"] is not None:
                raise ValueError("lifecycle failure lacks its error")
            failure = dict(reason="native_lifecycle_failure", case_id=case["id"])
        elif cell["status"] == "assessment_failed" and cell["error"] is None:
            raise ValueError("assessment failure lacks its error")
    if (
        count != record["native_window_count"]
        or count > MAX_NATIVE_WINDOWS
        or len(windows) > MAX_NATIVE_WINDOWS
    ):
        raise ValueError("native window bound differs")
    if record["stop"] != failure or (len(cells) != len(inventory) and failure is None):
        raise ValueError("partial diagnostic stop differs")
    complete = (
        len(cells) == len(inventory)
        and count == MAX_NATIVE_WINDOWS
        and len(windows) == MAX_NATIVE_WINDOWS
        and record["capacity_restored"] is True
        and model["unchanged"] is True
        and all(c["status"] == "evaluated" for c in cells)
    )
    if record["status"] != ("completed" if complete else "failed"):
        raise ValueError("completion claim differs")


def run_worker(expected_snapshot):
    return importlib.import_module("infra.resolution_handoff_worker").run_worker(
        expected_snapshot, producer_module="infra.modal_context_guard"
    )


if __name__ == "__main__":
    from infra.modal_cuda_lane import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(
        run(
            producer=importlib.import_module("infra.modal_context_guard"),
            **vars(parser.parse_args()),
        )
    )
