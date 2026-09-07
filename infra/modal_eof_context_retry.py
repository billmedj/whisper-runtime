"""One bounded real-stream EOF context retry experiment; import launches nothing.

The development baseline must reproduce before recovery is reported. References
score published output afterwards and never select a crop or authorize a commit.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from infra import modal_acoustic_diagnostic as shared

ROOT = shared.ROOT
REMOTE_ROOT = shared.REMOTE_ROOT
PRODUCER = "infra/modal_eof_context_retry.py"
WORKER = "infra/eof_context_retry_worker.py"
TEST = "tools/test_modal_eof_context_retry.py"
ARCHIVE = "evidence/modal-t4-tiny-en-word-context-2026-09-06.json"
ARCHIVE_SHA = "f96e524e01574b6f2f8df3f9e09932b990313bba9354799a58703da043526b85"
ARCHIVE_CELL = "context:concatenated-no-added-pauses"
EXPERIMENT_ID = "modal-eof-context-retry-v1"
GPU_TIMEOUT_SECONDS = 120
MAX_NATIVE_WINDOWS = 60
CLEANUP_RESERVE_SECONDS = 20
BACKEND_COMMIT = "a0b9695ae1cc52bad4b8626fe9fb6ea4ac0ee650"
BACKEND_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
SCHEDULE = (
    ("concatenated-no-added-pauses", "baseline"),
    ("concatenated-no-added-pauses", "retry"),
    ("attenuated-prefix", "retry"),
)
CLAIMS = dict.fromkeys(
    (
        "production_readiness",
        "general_speedup",
        "omission_free_recognition",
        "generalization_beyond_development",
        "physical_spending_cap",
    ),
    False,
)
# This explicit execution-scope allowlist is reviewed, not a replacement of the
# shared launcher's clean-tree rule. Unrelated dirty docs/tests are not executed.
ALLOWED_DIRTY_SOURCE = frozenset(
    (
        PRODUCER,
        WORKER,
        TEST,
        "infra/modal_acoustic_diagnostic.py",
        "infra/modal_cuda_lane.py",
        "src/whisper_runtime/adapters/native_whisper.py",
        "src/whisper_runtime/adapters/__init__.py",
        "src/whisper_runtime/adapters/continuous_stream.py",
        "src/whisper_runtime/adapters/audio_endpoints.py",
        "src/whisper_runtime/adapters/_checkpoint_io.py",
        "src/whisper_runtime/adapters/stream_checkpoint.py",
        "src/whisper_runtime/state.py",
    )
)


def _corpus():
    return shared._corpus()


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def archive_cell(root=ROOT):
    raw = (root / ARCHIVE).read_bytes()
    _require(shared._sha(raw) == ARCHIVE_SHA, "archived development evidence changed")
    record = json.loads(raw)
    return next(c for c in record["cells"] if c["id"] == ARCHIVE_CELL)


def cases(root=ROOT):
    corpus = _corpus()
    assets = corpus.REMOTE_ASSETS if root == REMOTE_ROOT else root / corpus.ASSET_PATH
    metadata, pcms = shared._cases(root, assets)
    inventory = {item["id"]: item for item in metadata["cases"]}
    base = corpus.read_registration(root)
    result = []
    for name, source, count in (
        ("concatenated-no-added-pauses", "concatenated-no-added-pauses", 538560),
        ("attenuated-prefix", "mixed-attenuated32", 174240),
    ):
        pcm = pcms[source][: count * 2]
        _require(len(pcm) == count * 2, "registered PCM is incomplete")
        result.append(
            dict(
                id=name,
                pcm=pcm,
                sample_count=count,
                pcm_sha256=shared._sha(pcm),
                source_case_id=source,
                source_pcm_sha256=inventory[source]["pcm_sha256"],
                recipe=inventory[source]["recipe"],
                reference_text=inventory[source]["reference_text"]
                if name == source
                else " ".join(f["reference_text"] for f in base["fixtures"][:2]),
            )
        )
    return result


def input_records(root=ROOT):
    return [{k: v for k, v in case.items() if k != "pcm"} for case in cases(root)]


def stream_config(arm, case_id="concatenated-no-added-pauses"):
    _require(arm in {"baseline", "retry"}, "unregistered arm")
    config = {
        **_corpus().AUTOMATIC_ENDPOINTS_PROFILE["stream_config"],
        "word_boundary_fallback": True,
        "left_context_ms": 2000,
        "word_context_limit_ms": 6000
        if case_id == "concatenated-no-added-pauses"
        else 0,
        "eof_context_retry": arm == "retry",
    }
    return _canonical(config)


def scope():
    return dict(
        gpu="T4",
        gpu_calls=1,
        max_containers=1,
        configured_retries=0,
        gpu_timeout_seconds=GPU_TIMEOUT_SECONDS,
        max_native_windows=MAX_NATIVE_WINDOWS,
        max_native_windows_per_cell=20,
        cleanup_reserve_seconds=CLEANUP_RESERVE_SECONDS,
        warmup_windows=0,
        pacing="unpaced-incremental-pcm",
        chunk_bytes=32000,
        max_driver_steps_per_cell=20000,
        inference_owners=1,
        cuda_lanes=1,
        model="tiny.en",
        precision="float32",
        alignment="legacy",
        rng_seed=7,
        model_download=False,
        borrowed_alignment_features=False,
        reference_used_for_selection=False,
        benchmark=False,
    )


def _git(root, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def snapshot(root=ROOT):
    """Explicit content snapshot of exact executed files and declared dirty paths."""
    root = Path(root).resolve()
    input_records(root)
    archive_cell(root)
    corpus = _corpus()
    helpers = {
        PRODUCER,
        WORKER,
        TEST,
        ARCHIVE,
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


def _without_window_ids(value):
    if isinstance(value, dict):
        return {k: _without_window_ids(v) for k, v in value.items() if k != "window_id"}
    if isinstance(value, list):
        return [_without_window_ids(item) for item in value]
    return value


def commit_signature(events):
    revisions, result = {}, []
    for event in events:
        if event["kind"] in {"provisional", "replace"}:
            revisions[event["segment_id"]] = event
        elif event["kind"] == "commit":
            previous = revisions[event["segment_id"]]
            result.append(
                dict(
                    start=event["start_sample"],
                    end=event["end_sample"],
                    revision=event["revision"],
                    text=previous["text"],
                )
            )
    return result


def control_signature(cell):
    keys = (
        "analysis_start_sample",
        "analysis_end_sample",
        "committed_before_sample",
        "retained_from_sample",
        "eof",
        "reason",
        "action",
        "result",
        "word_alignment",
    )
    return dict(
        commits=commit_signature(cell["events"]),
        traces=[
            _without_window_ids({k: trace[k] for k in keys})
            for trace in cell["decision_traces"]
        ],
        committed_samples=cell["metrics"]["committed_samples"],
        accepted_samples=cell["metrics"]["accepted_samples"],
    )


def cell_checks(cell, pcm):
    b = _corpus().b
    metrics = SimpleNamespace(**cell["metrics"])
    capacity = "reported-capacity"
    checks = b._event_checks(
        cell["events"],
        cell["decision_traces"],
        accepted_samples=metrics.accepted_samples,
        total_samples=len(pcm) // 2,
        state=SimpleNamespace(**cell["state"]),
        metrics=metrics,
        budget=SimpleNamespace(
            available=capacity if cell["capacity_restored"] else None,
            lease_count=0 if cell["capacity_restored"] else 1,
        ),
        worker=SimpleNamespace(queue_depth=0),
        capacity=capacity,
        retained_from_sample=cell["retained_from_sample"],
        max_buffer_samples=640000,
        error_record=cell["error"],
    )
    checks.update(
        shared.publication_checks(cell["events"], cell["decision_traces"], pcm)
    )
    return checks


def summarize(cells, *, model_unchanged, capacity_restored, root=ROOT):
    baseline = cells[0] if cells else None
    retry = cells[1] if len(cells) > 1 else None
    control = cells[2] if len(cells) > 2 else None
    matched = bool(
        baseline
        and baseline.get("metrics")
        and control_signature(baseline) == control_signature(archive_cell(root))
        and baseline["error"]
        and baseline["error"]["category"] == "policy_resolution"
    )
    prefix = bool(
        baseline
        and retry
        and retry.get("metrics")
        and commit_signature(retry["events"])[
            : len(commit_signature(baseline["events"]))
        ]
        == commit_signature(baseline["events"])
    )
    retry_observation = (
        None if retry is None else retry.get("context_retry_observation")
    )
    recovered = bool(
        retry
        and retry.get("stream_status") == "completed"
        and retry_observation
        and retry_observation["status"] == "recovered"
    )
    same_observations = False
    if retry_observation is not None and baseline and retry:
        before = {
            **retry,
            "decision_traces": retry["decision_traces"][
                : retry_observation["source"]["decode_index"]
            ],
            "events": baseline["events"],
            "metrics": baseline["metrics"],
        }
        same_observations = control_signature(before) == control_signature(baseline)
    control_ok = bool(
        control
        and control.get("stream_status") == "completed"
        and control["recognition"]["against_human_reference"]["word_edit_distance"] <= 1
    )
    improved = bool(
        retry
        and baseline
        and retry.get("recognition")
        and retry["recognition"]["against_human_reference"]["word_edit_distance"]
        < baseline["recognition"]["against_human_reference"]["word_edit_distance"]
    )
    success = all(
        (
            matched,
            prefix,
            same_observations,
            recovered,
            control_ok,
            improved,
            model_unchanged,
            capacity_restored,
        )
    )
    return dict(
        baseline_failure_reproduced=matched,
        frozen_prefix_unchanged=prefix,
        pre_retry_native_observations_match_baseline=same_observations,
        retry_stream_completed=recovered,
        control_completed_without_quality_regression=control_ok,
        published_human_edits_improved=improved,
        development_recovery_demonstrated=success,
    )


def validate_retry_observation(cell, pcm):
    """Bind the sole crop to the frozen source, actual call and committed state."""
    observation = cell.get("context_retry_observation")
    if observation is None:
        _require(
            cell["arm"] == "baseline"
            or cell["case_id"] == "attenuated-prefix"
            or cell["stream_status"] != "completed",
            "retry success lacks its observation",
        )
        return
    _require(cell["arm"] == "retry", "baseline performed a retry")
    source, anchor = observation["source"], observation["anchor"]
    _require(
        source in cell["decision_traces"]
        and source["eof"]
        and source["word_publication"] is None,
        "retry lacks frozen refusal",
    )
    start = max(
        source["retained_from_sample"],
        (anchor[0]["span"]["start_ms"] // 20 * 20 - 500) * 16
        if anchor
        else source["retained_from_sample"],
    )
    end = source["analysis_end_sample"]
    _require(
        observation["analysis_start_sample"] == start
        and observation["analysis_end_sample"] == end
        and source["retained_from_sample"] <= start < end <= len(pcm) // 2,
        "retry is not the fixed retained guard",
    )
    _require(
        observation["session_version"] <= cell["state"]["version"],
        "retry version is stale",
    )
    matches = [
        w
        for w in cell["windows"]
        if (w["start_ms"] * 16, w["end_ms"] * 16) == (start, end)
    ]
    if observation["pcm_sha256"] is not None:
        _require(
            observation["pcm_sha256"] == shared._sha(pcm[start * 2 : end * 2]),
            "retry retained PCM identity differs",
        )
    if observation["candidate"] is not None:
        _require(len(matches) == 1, "candidate lacks its one native window")
        candidate = observation["candidate"]
        traces = [
            t for t in cell["decision_traces"] if t.get("word_alignment") == candidate
        ]
        _require(
            len(traces) == 1
            and traces[0]["result"]["window_id"] == matches[0]["window_id"],
            "retry candidate lacks its actual trace",
        )
    if observation["status"] == "recovered":
        _require(
            observation["candidate"] is not None
            and len(matches) == 1
            and matches[0]["closed"] is True
            and matches[0]["capacity_restored"] is True
            and observation["committed_version"] == observation["session_version"] + 1
            and observation["committed_version"] <= cell["state"]["version"]
            and traces[0]["action"] == "commit"
            and traces[0]["word_publication"] is not None
            and cell["metrics"]["committed_samples"] == end,
            "recovered retry lacks a normal released native transaction",
        )
        _require(
            len(cell["windows"]) == source["decode_index"] + 1,
            "recovery used more than one additional native window",
        )


def validate_record(record, expected, root=ROOT):
    _require(record["source"]["snapshot"] == expected, "worker source differs")
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
    registered = _corpus().read_registration(root)
    model = record["model"]
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
                window["admitted_elapsed_ns"]
                < (GPU_TIMEOUT_SECONDS - CLEANUP_RESERVE_SECONDS) * 1000000000
                and window["admitted_elapsed_ns"] >= 0,
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
            _require(checks == cell["checks"], "event/PCM checks do not replay")
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
        _require(len(cell["windows"]) <= 20, "per-cell native bound exceeded")
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
        expected_snapshot
    )


if __name__ == "__main__":
    from infra.modal_cuda_lane import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(
        run(
            producer=importlib.import_module("infra.modal_eof_context_retry"),
            **vars(parser.parse_args()),
        )
    )
