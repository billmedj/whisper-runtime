"""Frozen 2x2 noisy-window/published-prefix diagnostic; never publishes text.

The actual failed source-paced receipt fixes the PCM, head, old anchor and
prompt. Two native windows each decode with no prompt, then replace that decode
using the existing same-window API. Human references are post-hoc scores only.
Importing this module allocates no remote resources.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path

from infra import modal_eof_context_retry as prior
from tools.analyze_stream_text_agreement import _commits

shared = prior.shared
ROOT, REMOTE_ROOT = prior.ROOT, prior.REMOTE_ROOT
_corpus, _require, _canonical, _git = (
    prior._corpus,
    prior._require,
    prior._canonical,
    prior._git,
)
PRODUCER = "infra/modal_noisy_context_prompt.py"
WORKER = "infra/noisy_context_prompt_worker.py"
TEST = "tools/test_modal_noisy_context_prompt.py"
ARCHIVE = "evidence/modal-t4-tiny-en-paced-context-retry-2026-09-06.json"
ARCHIVE_SHA = "fa856cb4f0028109e7524b6e563fd008600d6fe84043d86c8f433784bdc2e3bf"
PCM_SHA = "5276e641f14eaae484a8da6f916e44f9373239fbe70c70ba41a3f1a36f747108"
EXPERIMENT_ID = "modal-noisy-context-prompt-v1"
GPU_TIMEOUT_SECONDS = 90
CLEANUP_RESERVE_SECONDS = 15
MAX_NATIVE_WINDOWS = MAX_NATIVE_WINDOWS_PER_CELL = 2
# Native tiny.en has n_text_ctx=448; sample_len=None retains its 224-step default.
# This is a driver safety bound, not a changed decode option.
MAX_DRIVER_STEPS = 224
SEED = 7
TOKENIZER_EOT = 50256
BACKEND_COMMIT, BACKEND_TREE = prior.BACKEND_COMMIT, prior.BACKEND_TREE
CROPS = (("retained", 1680, 10890), ("head-only", 3680, 10890))
CLAIMS = {
    **prior.CLAIMS,
    **dict.fromkeys(
        (
            "recognition_improvement",
            "full_stream_recovery",
            "noisy_stream_recovery",
            "publication_authority",
            "cross_window_feature_reuse",
            "word_or_subtitle_latency",
            "gpu_device_timing",
        ),
        False,
    ),
}
ALLOWED_DIRTY_SOURCE = prior.ALLOWED_DIRTY_SOURCE | {
    PRODUCER,
    WORKER,
    TEST,
    "tools/test_noisy_context_prompt_worker.py",
    prior.PRODUCER,
    ARCHIVE,
}


def source_case(root=ROOT):
    """Reconstruct the exact published prompt and both crops from frozen bytes."""
    root = Path(root)
    raw = (root / ARCHIVE).read_bytes()
    _require(shared._sha(raw) == ARCHIVE_SHA, "archived paced evidence changed")
    archived = json.loads(raw)
    cells = [
        cell
        for cell in archived["cells"]
        if (cell["case_id"], cell["arm"]) == ("noisy-prefix", "retry")
    ]
    _require(len(cells) == 1, "ambiguous noisy source cell")
    cell = cells[0]
    source = cell["decision_traces"][-1]
    receipt = cell["context_retry_observation"]
    _require(
        receipt["source"] == source
        and receipt["status"] == "unavailable"
        and receipt["reason"] == "no_distinct_anchor_window"
        and source["eof"] is True
        and source["action"] == "unresolved"
        and source["reason"] == "no_lexical_text"
        and source["word_publication"] is None,
        "frozen noisy refusal differs",
    )
    _require(
        source["committed_before_sample"] == 58880
        and source["retained_from_sample"] == source["analysis_start_sample"] == 26880
        and source["accepted_through_sample"] == source["analysis_end_sample"] == 174240
        and cell["metrics"]["committed_samples"] == 58880
        and cell["metrics"]["accepted_samples"] == 174240
        and receipt["session_version"] == 2,
        "frozen source clocks differ",
    )
    commits = [
        dict(
            sequence_number=event.sequence_number,
            segment_id=event.segment_id,
            revision=event.revision,
            start_sample=event.start_sample,
            end_sample=event.end_sample,
            text=text,
        )
        for event, text in _commits(cell["events"])
    ]
    _require(
        len(commits) == 2 and commits[-1]["end_sample"] == 58880,
        "published prefix does not reach the frozen head",
    )
    # Whitespace canonicalization only: do not borrow punctuation from a preview
    # or the human reference. The real last committed word has no trailing dot.
    prompt = " ".join(" ".join(item["text"] for item in commits).split())
    _require(
        prompt == "Concord returned to its place amidst the tents",
        "actual published prompt differs",
    )
    corpus = _corpus()
    registered = corpus.read_registration(root)
    from whisper_runtime.adapters.native_whisper import NativeDecodeOptions

    options = _canonical(asdict(NativeDecodeOptions(**registered["decode_options"])))
    _require(
        options == archived["effective_identity"]["decode_options"]
        and options == receipt["analysis_identity"]["requested_decode_options"]
        and options["prompt"] is None
        and options["sample_len"] is None
        and archived["effective_identity"]["rng_seed"] == SEED,
        "original registered decode options differ",
    )
    assets = corpus.REMOTE_ASSETS if root == REMOTE_ROOT else root / corpus.ASSET_PATH
    metadata, pcms = shared._cases(root, assets)
    inventory = {item["id"]: item for item in metadata["cases"]}
    source_id = "mixed-continuous-noise64"
    pcm = pcms[source_id][: 174240 * 2]
    saved_input = next(
        item for item in archived["inputs"] if item["id"] == "noisy-prefix"
    )
    _require(
        len(pcm) == 174240 * 2
        and shared._sha(pcm) == PCM_SHA == saved_input["pcm_sha256"]
        and inventory[source_id]["pcm_sha256"] == saved_input["source_pcm_sha256"]
        and inventory[source_id]["recipe"] == saved_input["recipe"],
        "registered noisy PCM or recipe differs",
    )
    crops = [
        dict(
            id=name,
            start_ms=start,
            end_ms=end,
            start_sample=start * 16,
            end_sample=end * 16,
            sample_count=(end - start) * 16,
            pcm_sha256=shared._sha(pcm[start * 32 : end * 32]),
        )
        for name, start, end in CROPS
    ]
    _require(
        crops[0]["pcm_sha256"] == source["audio_evidence"]["observation"]["pcm_sha256"]
        and source["word_alignment"]["native"] == source["result"],
        "frozen retained alignment or PCM differs",
    )
    return dict(
        id="noisy-prefix",
        pcm=pcm,
        sample_count=174240,
        pcm_sha256=PCM_SHA,
        source_case_id=source_id,
        source_pcm_sha256=saved_input["source_pcm_sha256"],
        recipe=saved_input["recipe"],
        archive=dict(
            path=ARCHIVE, sha256=ARCHIVE_SHA, cell_id="noisy-prefix", arm="retry"
        ),
        source_trace=source,
        archived_alignment=source["word_alignment"],
        archive_backend=archived["backend"],
        effective_identity=archived["effective_identity"],
        anchor=receipt["anchor"],
        prompt=prompt,
        prompt_sha256=shared._sha(prompt.encode("utf-8")),
        prompt_commits=commits,
        decode_options=options,
        session_version=2,
        head_sample=58880,
        retained_sample=26880,
        accepted_sample=174240,
        config=cell["config"],
        crops=crops,
        reference_text=saved_input["reference_text"],
    )


def input_record(root=ROOT):
    return {key: value for key, value in source_case(root).items() if key != "pcm"}


def scope():
    return dict(
        gpu="T4",
        gpu_calls=1,
        max_containers=1,
        configured_retries=0,
        gpu_timeout_seconds=GPU_TIMEOUT_SECONDS,
        cleanup_reserve_seconds=CLEANUP_RESERVE_SECONDS,
        max_native_windows=MAX_NATIVE_WINDOWS,
        max_decode_attempts=4,
        max_driver_steps_per_attempt=MAX_DRIVER_STEPS,
        expected_decode_encoder_forwards=2,
        expected_alignment_encoder_forwards=4,
        expected_encoder_forwards=6,
        expected_encoder_counts_assume_nonempty_text_tokens=True,
        independent_decode_alignment_encoder_forwards=8,
        warmup_windows=0,
        inference_owners=1,
        cuda_lanes=1,
        model="tiny.en",
        precision="float32",
        rng_seed=SEED,
        sample_len=None,
        reuse_decode_features=True,
        alignment="legacy",
        borrowed_alignment_features=False,
        model_download=False,
        nonpublishing=True,
        head_only_has_no_join_authority=True,
        frozen_anchor_unchanged=True,
        reference_used_for_selection=False,
        reference_scores_are_posthoc=True,
        host_timings_include_instrumentation=True,
        benchmark=False,
    )


def snapshot(root=ROOT):
    """Hash exact reviewed working bytes without mutating other launch guards."""
    root = Path(root).resolve()
    input_record(root)
    corpus = _corpus()
    helpers = {
        PRODUCER,
        WORKER,
        TEST,
        "tools/test_noisy_context_prompt_worker.py",
        ARCHIVE,
        prior.PRODUCER,
        prior.WORKER,
        shared.PRODUCER,
        shared.MANIFEST,
        *shared.BUILDERS,
        corpus.MANIFEST_PATH,
        corpus.PRODUCER_PATH,
        *corpus.HELPER_PATHS,
        "infra/modal_cuda_lane.py",
        "infra/modal_native_cuda_qualification.py",
        "infra/native_cuda_trace.py",
        "tools/analyze_stream_text_agreement.py",
        "tools/analyze_word_resolution.py",
        "tools/word_anchor_reconciliation.py",
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


def control(cells, source):
    """Report retained null-prompt comparability; never make it a success gate."""
    null = cells[0]["attempts"][0] if cells and cells[0]["attempts"] else None
    return dict(
        retained_null_alignment_equal=None
        if null is None or null.get("alignment") is None
        else _canonical(prior._without_window_ids(null["alignment"]))
        == _canonical(prior._without_window_ids(source["archived_alignment"])),
        ignored_fields=["window_id"],
        head_only_control="not_registered",
        mismatch_is_observation_not_launch_failure=True,
    )


def validate_record(record, expected, root=ROOT):
    """Recompute frozen joins, strict shadow decisions and actual work counts."""
    _canonical(record)  # Reject non-finite JSON rather than silently accepting NaN.
    source = source_case(root)
    registered = _corpus().read_registration(root)
    worker = importlib.import_module("infra.noisy_context_prompt_worker")
    _require(
        record["schema_version"] == "1-diagnostic"
        and record["experiment_id"] == EXPERIMENT_ID
        and record["scope"] == scope(),
        "diagnostic registration differs",
    )
    _require(
        record["qualified"] is False and record["claim_boundary"] == CLAIMS,
        "diagnostic cannot authorize publication or qualification",
    )
    _require(record["source"]["snapshot"] == expected, "worker source differs")
    producer_sha = next(
        item["sha256"] for item in expected["files"] if item["path"] == PRODUCER
    )
    _require(
        record["source"]["registration_sha256"] == producer_sha,
        "producer registration bytes differ",
    )
    _require(
        record["input"]
        == {key: value for key, value in source.items() if key != "pcm"},
        "frozen prompt, anchor, PCM or input differs",
    )
    effective, model = record["effective_identity"], record["model"]
    _require(
        record["backend"] == dict(commit=BACKEND_COMMIT, tree=BACKEND_TREE)
        and effective.get("reuse_decode_features") is True
        and effective.get("tokenizer_eot") == TOKENIZER_EOT
        and all(
            effective.get(key) == value
            for key, value in source["effective_identity"].items()
        ),
        "original effective execution identity differs",
    )
    _require(
        model["initial_sha256"] == registered["model"]["model_state_sha256"]
        and model["checkpoint_sha256"] == registered["model"]["checkpoint_sha256"]
        and model["unchanged"] is (model["initial_sha256"] == model["final_sha256"]),
        "model identity differs",
    )
    cells = record["cells"]
    _require(isinstance(cells, list) and len(cells) <= 2, "extra native window cell")
    window_count = completed_attempts = decode_count = 0
    all_safe = True
    seen_window_ids = set()
    for cell, crop in zip(cells, source["crops"]):
        _require(
            all(
                cell[key] == crop[key]
                for key in ("id", "start_ms", "end_ms", "pcm_sha256")
            ),
            "fixed crop or cell order differs",
        )
        _require(cell["session_version"] == 0, "diagnostic published a native result")
        attempts = cell["attempts"]
        count = cell["decode_attempt_count"]
        _require(
            isinstance(attempts, list)
            and len(attempts) <= 2
            and type(count) is int
            and len(attempts) <= count <= min(2, len(attempts) + 1),
            "decode attempt bound or accounting differs",
        )
        decode_count += count
        completed_attempts += len(attempts)
        encoders, steps = 1, 0
        for index, attempt in enumerate(attempts):
            prompt = None if index == 0 else source["prompt"]
            _require(
                attempt["id"] == ("null" if index == 0 else "published-prefix")
                and attempt["prompt"] == prompt
                and attempt["decode_attempt"] == index + 1
                and attempt["decode_options"]
                == {**source["decode_options"], "prompt": prompt},
                "same-window prompt schedule or original options differ",
            )
            _require(
                attempt["publication_authorized"] is False
                and attempt["session_version"] == 0,
                "shadow decision acquired publication authority",
            )
            step_count = attempt["driver_steps"]
            _require(
                type(step_count) is int and 1 <= step_count <= MAX_DRIVER_STEPS,
                "driver step bound differs",
            )
            steps += step_count
            alignment, result = attempt["alignment"], attempt["result"]
            _require(alignment["native"] == result, "alignment/native result differs")
            _require(
                result["start_ms"] == crop["start_ms"]
                and result["end_ms"] == crop["end_ms"]
                and result["analysis_span"]
                == dict(start_ms=crop["start_ms"], end_ms=crop["end_ms"]),
                "native result escaped its registered crop",
            )
            analysis = _canonical(worker.analyze(alignment, source, crop))
            _require(
                all(attempt[key] == value for key, value in analysis.items()),
                "strict shadow analysis or post-hoc score does not replay",
            )
            _require(
                crop["id"] != "head-only" or attempt["strict_candidate"] is False,
                "head-only crop has no join authority",
            )
            encoders += int(
                any(token < TOKENIZER_EOT for token in result["metadata"]["tokens"])
            )
            _require(
                attempt["encoder_forwards_so_far"] == encoders,
                "decode-feature or legacy alignment work differs",
            )
        window = cell["window"]
        if window is not None:
            window_count += 1
            _require(
                window["call_index"] == window_count
                and all(
                    window[key] == crop[key]
                    for key in ("start_ms", "end_ms", "pcm_sha256")
                )
                and window["window_id"] not in seen_window_ids,
                "native window identity or PCM differs",
            )
            seen_window_ids.add(window["window_id"])
            _require(
                all(
                    attempt["result"]["window_id"] == window["window_id"]
                    for attempt in attempts
                ),
                "replacement left its original native window",
            )
            elapsed = window["admitted_elapsed_ns"]
            _require(
                type(elapsed) is int
                and 0
                <= elapsed
                < (GPU_TIMEOUT_SECONDS - CLEANUP_RESERVE_SECONDS) * 1_000_000_000,
                "native admission crossed cleanup reserve",
            )
            _require(window["mel_shape"] == [80, 3000], "encoder input differs")
            calls = window["encoder_calls"]
            phases = [item["phase"] for item in calls]
            _require(
                len(calls) <= 3
                and (
                    not calls or phases == ["decode"] + ["alignment"] * (len(calls) - 1)
                )
                and all(item["input_frames"] == 3000 for item in calls)
                and all(
                    item["use_sdpa"] is False
                    for item in calls
                    if item["phase"] == "alignment"
                ),
                "encoder phases, frames or legacy attention differ",
            )
            _require(
                type(window["driver_steps"]) is int
                and steps <= window["driver_steps"] <= count * MAX_DRIVER_STEPS
                and all(
                    type(value) is int and value >= 0
                    for value in window["operation_wall_ns"].values()
                ),
                "native driver accounting or host timings differ",
            )
            if len(attempts) == 2:
                _require(
                    len(calls) == encoders and window["driver_steps"] == steps,
                    "completed window contains unaccounted native work",
                )
        else:
            _require(not attempts, "decoded attempt lacks its native window")
        safe = bool(
            window
            and window["closed"] is True
            and window["capacity_restored"] is True
            and all(
                cell[key] is True
                for key in (
                    "first_snapshot_unchanged",
                    "same_lease",
                    "closed",
                    "capacity_restored",
                )
            )
            and cell.get("error") is None
            and cell.get("cleanup_error") is None
        )
        all_safe &= safe
    _require(
        type(record["native_window_count"]) is int
        and record["native_window_count"] == window_count <= MAX_NATIVE_WINDOWS
        and type(record["decode_attempt_count"]) is int
        and record["decode_attempt_count"] == decode_count <= 4,
        "aggregate native work accounting differs",
    )
    _require(
        record["control"] == control(cells, source),
        "archived control comparison does not replay",
    )
    complete = bool(
        len(cells) == 2
        and completed_attempts == decode_count == 4
        and window_count == 2
        and all_safe
        and model["unchanged"] is True
        and record["capacity_restored"] is True
        and record["error"] is None
        and record["stop"] is None
    )
    _require(
        record["status"] == ("completed" if complete else "failed"),
        "diagnostic completion claim differs",
    )
    _require(
        complete or record["error"] is not None or record["stop"] is not None,
        "incomplete diagnostic lacks explicit failure",
    )


def run_worker(expected_snapshot):
    return importlib.import_module("infra.noisy_context_prompt_worker").run_worker(
        expected_snapshot
    )


if __name__ == "__main__":
    from infra.modal_cuda_lane import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(
        run(
            producer=importlib.import_module("infra.modal_noisy_context_prompt"),
            **vars(parser.parse_args()),
        )
    )
