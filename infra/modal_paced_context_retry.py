"""One bounded T4 call: source-paced EOF retry and a noisy refusal control.

The baseline is observed in this same call, not forced to match an unpaced
archive. Human references score published output afterwards and cannot select a
crop, repair text, or authorize a commit. Importing this module launches nothing.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from infra import modal_eof_context_retry as prior

shared = prior.shared
ROOT, REMOTE_ROOT = prior.ROOT, prior.REMOTE_ROOT
PRODUCER = "infra/modal_paced_context_retry.py"
WORKER = prior.WORKER
TEST = "tools/test_modal_paced_context_retry.py"
EXPERIMENT_ID = "modal-paced-context-retry-v1"
GPU_TIMEOUT_SECONDS = 150
CLEANUP_RESERVE_SECONDS = 20
MAX_NATIVE_WINDOWS = 80
MAX_NATIVE_WINDOWS_PER_CELL = 30
BACKEND_COMMIT, BACKEND_TREE = prior.BACKEND_COMMIT, prior.BACKEND_TREE
SCHEDULE = (
    ("concatenated-no-added-pauses", "baseline"),
    ("concatenated-no-added-pauses", "retry"),
    ("noisy-prefix", "retry"),
)
CLAIMS = {
    **prior.CLAIMS,
    "noisy_stream_recovery": False,
    "word_or_subtitle_latency": False,
    "gpu_device_timing": False,
}
ALLOWED_DIRTY_SOURCE = prior.ALLOWED_DIRTY_SOURCE | {PRODUCER, TEST}
_require, _canonical, _git, _corpus = (
    prior._require,
    prior._canonical,
    prior._git,
    prior._corpus,
)
_COMPLETION_CHECKS = frozenset(
    (
        "final_event_once",
        "full_input_committed_at_eof",
        "completion_required",
        "paced_completed",
        "paced_terminal_coverage",
        "paced_pre_eof_commit",
    )
)


def cases(root=ROOT):
    corpus = _corpus()
    assets = corpus.REMOTE_ASSETS if root == REMOTE_ROOT else root / corpus.ASSET_PATH
    metadata, pcms = shared._cases(root, assets)
    inventory = {item["id"]: item for item in metadata["cases"]}
    base = corpus.read_registration(root)
    result = []
    for name, source, count in (
        ("concatenated-no-added-pauses", "concatenated-no-added-pauses", 538560),
        ("noisy-prefix", "mixed-continuous-noise64", 174240),
    ):
        pcm = pcms[source][: count * 2]
        _require(len(pcm) == count * 2, "registered PCM is incomplete")
        result.append(
            dict(
                id=name,
                pcm=pcm,
                sample_count=count,
                duration_seconds=count / 16000,
                pcm_sha256=shared._sha(pcm),
                source_case_id=source,
                source_pcm_sha256=inventory[source]["pcm_sha256"],
                recipe=inventory[source]["recipe"],
                prefix_samples=count,
                reference_text=inventory[source]["reference_text"]
                if name == source
                else " ".join(item["reference_text"] for item in base["fixtures"][:2]),
            )
        )
    return result


def input_records(root=ROOT):
    return [{k: v for k, v in case.items() if k != "pcm"} for case in cases(root)]


def stream_config(arm, case_id="concatenated-no-added-pauses"):
    _require(case_id in {case for case, _ in SCHEDULE}, "unregistered case")
    return prior.stream_config(arm, case_id)


def scope():
    return {
        **prior.scope(),
        "gpu_timeout_seconds": GPU_TIMEOUT_SECONDS,
        "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
        "max_native_windows": MAX_NATIVE_WINDOWS,
        "max_native_windows_per_cell": MAX_NATIVE_WINDOWS_PER_CELL,
        "pacing": _canonical(_corpus().PACED_REPLAY),
        "chunk_bytes": _corpus().PACED_REPLAY["config"]["chunk_ms"] * 32,
        "max_driver_steps_per_cell": _corpus().PACED_REPLAY["config"]["max_steps"],
        "source_thread_only_admits_pcm": True,
        "same_call_baseline": True,
        "historical_unpaced_baseline_required": False,
        "identical_cross_arm_native_observations_required": False,
        "noisy_policy_refusal_is_expected_not_recovery": True,
        "latency_scope": "same-worker-source-end-to-output; excludes endpoint quiet wait; not word or subtitle latency",
        "host_timings_include_instrumentation": True,
    }


def snapshot(root=ROOT):
    """Freeze executed working bytes under an explicit reviewed dirty allowlist."""
    root = Path(root).resolve()
    input_records(root)
    corpus = _corpus()
    helpers = {
        PRODUCER,
        WORKER,
        TEST,
        prior.PRODUCER,
        shared.PRODUCER,
        shared.MANIFEST,
        *shared.BUILDERS,
        corpus.MANIFEST_PATH,
        corpus.PRODUCER_PATH,
        *corpus.HELPER_PATHS,
        "infra/modal_cuda_lane.py",
        "infra/modal_native_cuda_qualification.py",
        "infra/native_cuda_trace.py",
    }
    names = helpers | {
        path.relative_to(root).as_posix()
        for path in (root / "src/whisper_runtime").rglob("*.py")
    }
    paths = ["src/whisper_runtime", *sorted(helpers)]
    dirty = set(_git(root, "diff", "--name-only", "HEAD", "--", *paths).splitlines())
    dirty.update(
        _git(
            root, "ls-files", "--others", "--exclude-standard", "--", *paths
        ).splitlines()
    )
    _require(
        not dirty - ALLOWED_DIRTY_SOURCE,
        "unreviewed dirty executed source: "
        + ", ".join(sorted(dirty - ALLOWED_DIRTY_SOURCE)),
    )
    files = [
        dict(path=name, size_bytes=len(raw), sha256=shared._sha(raw))
        for name in sorted(names)
        for raw in [(root / name).read_bytes()]
    ]
    return dict(
        commit=_git(root, "rev-parse", "HEAD"),
        tracked_tree=_git(root, "rev-parse", "HEAD^{tree}"),
        dirty_source_paths=sorted(dirty),
        digest=shared._hash(files),
        files=files,
    )


def cell_checks(cell, pcm):
    checks = prior.cell_checks(cell, pcm)
    checks.update(
        _corpus()._paced_checks(
            cell["events"], cell["decision_traces"], cell["pacing"], len(pcm) // 2
        )
    )
    return checks


def _integrity(cell):
    checks = cell.get("checks", {}) if cell else {}
    pacing_status = (cell.get("pacing") or {}).get("status") if cell else None
    policy_refusal = (
        (cell.get("error") or {}).get("category") == "policy_resolution"
        if cell
        else False
    )
    return bool(
        checks
        and cell.get("metrics")
        and cell.get("pacing")
        and (
            pacing_status == "completed"
            or (pacing_status == "runtime_error" and policy_refusal)
        )
        and cell.get("capacity_restored") is True
        and cell.get("cleanup_error") is None
        and all(
            value is True
            for key, value in checks.items()
            if key not in _COMPLETION_CHECKS
        )
    )


def outcome(cell):
    """Keep an already-complete baseline distinct from observed retry recovery."""
    if not cell or not cell.get("metrics"):
        return "not_observed"
    observation = cell.get("context_retry_observation")
    if cell.get("stream_status") == "completed":
        if observation is None:
            return "already_complete_without_retry"
        return (
            "recovered"
            if observation["status"] == "recovered"
            else "completed_without_recovery"
        )
    policy = (cell.get("error") or {}).get("category") == "policy_resolution"
    if policy:
        return "no_retry_policy_refusal" if observation is None else "retry_refused"
    return "execution_failed"


def summarize(cells, *, model_unchanged, capacity_restored, root=ROOT):
    del root  # No historical control can gate a source-paced observation.
    baseline, retry, noisy = (
        cells[index] if len(cells) > index else None for index in range(3)
    )
    baseline_outcome, retry_outcome, noisy_outcome = map(
        outcome, (baseline, retry, noisy)
    )
    observation = retry.get("context_retry_observation") if retry else None
    prefix = native_match = None
    if baseline and retry and observation is not None and baseline.get("metrics"):
        prefix = prior.commit_signature(retry["events"])[
            : len(prior.commit_signature(baseline["events"]))
        ] == prior.commit_signature(baseline["events"])
        before = {
            **retry,
            "decision_traces": retry["decision_traces"][
                : observation["source"]["decode_index"]
            ],
            "events": baseline["events"],
            "metrics": baseline["metrics"],
        }
        native_match = prior.control_signature(before) == prior.control_signature(
            baseline
        )
    integrity = [_integrity(cell) for cell in (baseline, retry, noisy)]
    baseline_valid = integrity[0] and baseline_outcome in {
        "already_complete_without_retry",
        "no_retry_policy_refusal",
    }
    full_stream = bool(
        retry
        and retry.get("stream_status") == "completed"
        and integrity[1]
        and all(retry["checks"].values())
    )
    recovered = full_stream and retry_outcome == "recovered"
    noisy_refusal = integrity[2] and noisy_outcome in {
        "no_retry_policy_refusal",
        "retry_refused",
    }
    noisy_completed = bool(
        noisy and noisy.get("stream_status") == "completed" and integrity[2]
    )
    observation_complete = all(
        (
            len(cells) == 3,
            baseline_valid,
            integrity[1],
            noisy_refusal or noisy_completed,
            model_unchanged,
            capacity_restored,
        )
    )
    scores = {}
    for name, cell in (("baseline", baseline), ("retry", retry), ("noisy", noisy)):
        scores[name] = (
            (
                cell.get("recognition", {})
                .get("against_human_reference", {})
                .get("word_edit_distance")
            )
            if cell
            else None
        )
    improved = (
        scores["retry"] < scores["baseline"]
        if scores["retry"] is not None and scores["baseline"] is not None
        else None
    )
    return dict(
        baseline_outcome=baseline_outcome,
        retry_outcome=retry_outcome,
        noisy_outcome=noisy_outcome,
        baseline_policy_refusal_observed=baseline_outcome == "no_retry_policy_refusal",
        full_stream_completed=full_stream,
        retry_stream_completed=recovered,
        frozen_prefix_unchanged=prefix,
        pre_retry_native_observations_match_baseline=native_match,
        cross_arm_comparison="not_available"
        if native_match is None
        else "matched"
        if native_match
        else "unmatched",
        cell_integrity=dict(
            zip(("baseline", "retry", "noisy"), integrity, strict=True)
        ),
        noisy_refusal_expected=noisy_refusal,
        noisy_stream_completed=noisy_completed,
        noisy_recovery_demonstrated=noisy_completed and noisy_outcome == "recovered",
        human_word_edits=scores,
        published_human_edits_improved=improved,
        experiment_observation_complete=observation_complete,
        development_recovery_demonstrated=bool(observation_complete and recovered),
    )


def validate_retry_observation(cell, pcm):
    observation = cell.get("context_retry_observation")
    retry_windows = [
        window for window in cell["windows"] if ":context-retry:" in window["window_id"]
    ]
    if observation is None:
        # Source-paced boundaries may complete naturally. Do not fabricate a retry
        # or require an unpaced failure just to preserve an earlier outcome.
        _require(not retry_windows, "retry native window lacks its observation")
        return
    prior.validate_retry_observation(cell, pcm)
    _require(
        observation["status"]
        in {"unavailable", "scheduled", "running", "recovered", "refused", "failed"},
        "unregistered retry status",
    )
    _require(len(retry_windows) <= 1, "more than one actual context retry")
    if observation["status"] == "unavailable":
        _require(
            not retry_windows
            and observation["candidate"] is None
            and observation["pcm_sha256"] is None
            and len(cell["windows"]) == observation["source"]["decode_index"],
            "unavailable retry performed native work",
        )


def validate_record(record, expected, root=ROOT):
    """Recompute source clocks, native PCM joins, receipt, output and verdict."""
    _require(record["source"]["snapshot"] == expected, "worker source differs")
    _require(record["schema_version"] == "1-diagnostic", "schema differs")
    _require(
        record["scope"] == scope() and record["experiment_id"] == EXPERIMENT_ID,
        "experiment scope differs",
    )
    _require(
        record["claim_boundary"] == CLAIMS and record["qualified"] is False,
        "release qualification is unsupported",
    )
    _require(record["inputs"] == input_records(root), "worker inputs differ")
    _require(
        record["schedule"] == [dict(case_id=c, arm=a) for c, a in SCHEDULE],
        "schedule differs",
    )
    _require(
        record["backend"] == dict(commit=BACKEND_COMMIT, tree=BACKEND_TREE),
        "backend identity differs",
    )
    registered, model = _corpus().read_registration(root), record["model"]
    _require(
        model["initial_sha256"] == registered["model"]["model_state_sha256"]
        and model["checkpoint_sha256"] == registered["model"]["checkpoint_sha256"]
        and model["unchanged"] is (model["initial_sha256"] == model["final_sha256"]),
        "model identity differs",
    )
    from dataclasses import asdict

    from whisper_runtime.adapters.native_whisper import NativeDecodeOptions

    effective = record["effective_identity"]
    _require(
        effective["decode_options"]
        == _canonical(asdict(NativeDecodeOptions(**registered["decode_options"])))
        and effective["rng_seed"] == 7
        and effective["alignment_mode"] == "legacy"
        and effective["precision"] == "float32"
        and effective["model_sha256"] == model["initial_sha256"]
        and effective["checkpoint_sha256"] == model["checkpoint_sha256"]
        and effective["torch"] == "2.6.0+cu124"
        and effective["numpy"] == "2.5.2"
        and effective["tiktoken"] == "0.14.0",
        "effective execution identity differs",
    )
    cells, count = record["cells"], 0
    inventory = {case["id"]: case for case in cases(root)}
    _require(len(cells) <= len(SCHEDULE), "extra cell")
    _require(
        len(cells) == len(SCHEDULE) or record["stop"] is not None,
        "partial run lacks an explicit stop",
    )
    for cell, (case_id, arm) in zip(cells, SCHEDULE):
        _require((cell["case_id"], cell["arm"]) == (case_id, arm), "cell order differs")
        _require(cell["config"] == stream_config(arm, case_id), "stream config differs")
        pcm = inventory[case_id]["pcm"]
        for window in cell["windows"]:
            count += 1
            start, end = window["start_ms"], window["end_ms"]
            _require(
                window["call_index"] == count
                and 0 <= start < end
                and end - start <= 30000
                and end * 32 <= len(pcm),
                "unregistered native window",
            )
            _require(
                window["pcm_sha256"] == shared._sha(pcm[start * 32 : end * 32]),
                "native PCM differs",
            )
            _require(
                0
                <= window["admitted_elapsed_ns"]
                < (GPU_TIMEOUT_SECONDS - CLEANUP_RESERVE_SECONDS) * 1000000000,
                "window admitted after cleanup reserve",
            )
            _require(window["mel_shape"] == [80, 3000], "encoder input differs")
            _require(
                all(e["input_frames"] == 3000 for e in window["encoder_calls"]),
                "encoder frame accounting differs",
            )
            if window["closed"]:
                _require(
                    [e["phase"] for e in window["encoder_calls"]]
                    in (["decode"], ["decode", "alignment"]),
                    "closed native window lacks legacy encoder work",
                )
            _require(
                all(
                    type(value) is int and value >= 0
                    for value in window["operation_wall_ns"].values()
                ),
                "invalid native operation time",
            )
        if cell.get("metrics"):
            checks = cell_checks(cell, pcm)
            _require(checks == cell["checks"], "event/PCM/pacing checks do not replay")
            pacing = cell["pacing"]
            _require(
                pacing["pacing_id"] == _corpus().PACED_REPLAY["pacing_id"]
                and pacing["config"] == _corpus().PACED_REPLAY["config"],
                "pacing configuration differs",
            )
            _require(
                cell["driver_steps"] == pacing["driver_steps"]
                and cell["accepted_chunks"] == len(pacing["admissions"])
                and cell["metrics"]["accepted_samples"] == pacing["accepted_samples"],
                "pacing accounting differs",
            )
            text = _corpus().c._normalized_committed_text(cell["events"])
            _require(
                cell["recognition"]
                == dict(
                    text=text,
                    against_human_reference=_corpus().b._word_difference(
                        text, inventory[case_id]["reference_text"]
                    ),
                ),
                "human score differs",
            )
            complete = cell["error"] is None and cell["done"] and all(checks.values())
            _require(
                (cell["stream_status"] == "completed") is bool(complete),
                "completion claim differs",
            )
            validate_retry_observation(cell, pcm)
            trace_ids = [
                trace["result"]["window_id"] for trace in cell["decision_traces"]
            ]
            window_ids = [window["window_id"] for window in cell["windows"]]
            _require(
                len(window_ids) == len(set(window_ids))
                and all(i in window_ids for i in trace_ids),
                "trace lacks a unique measured native window",
            )
            if cell["stream_status"] == "completed":
                _require(
                    trace_ids == window_ids
                    and all(
                        w["closed"] and w["capacity_restored"] for w in cell["windows"]
                    ),
                    "completed cell has unaccounted native work",
                )
        else:
            _require(
                cell["stream_status"] == "failed" and cell["error"] is not None,
                "incomplete cell cannot pass",
            )
        _require(
            len(cell["windows"]) <= MAX_NATIVE_WINDOWS_PER_CELL,
            "per-cell native bound exceeded",
        )
    _require(
        count == record["native_window_count"] and count <= MAX_NATIVE_WINDOWS,
        "native call accounting differs",
    )
    summary = summarize(
        cells,
        model_unchanged=model["unchanged"],
        capacity_restored=record["capacity_restored"],
        root=root,
    )
    _require(record["summary"] == summary, "summary does not replay")
    _require(
        record["status"]
        == ("completed" if summary["development_recovery_demonstrated"] else "failed"),
        "result claim differs",
    )


def run_worker(expected_snapshot):
    return importlib.import_module("infra.eof_context_retry_worker").run_worker(
        expected_snapshot, producer=importlib.import_module(__name__), paced=True
    )


if __name__ == "__main__":
    from infra.modal_cuda_lane import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(
        run(
            producer=importlib.import_module("infra.modal_paced_context_retry"),
            **vars(parser.parse_args()),
        )
    )
