"""One bounded experimental fast20/24 CPU stream with untrusted token drafts.

No production/backend edits or score tolerance changes. The existing native
factory, token-step API, alignment and publication policy remain authoritative.
Only an isolated worker temporarily wraps request-local decoder inference.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.verify_decoder_draft import DraftInference  # noqa: E402


class StreamDraftInference(DraftInference):
    # The adapter checks this declaration. The superclass rejects legacy cache
    # inference and every speculative cache is built fresh from current audio.
    _use_legacy_cache = False


PARITY_FIELDS = (
    "case",
    "candidate",
    "model_identity",
    "profile",
    "config",
    "pcm_sha256",
    "input_samples",
    "endpoint_samples",
    "done",
    "error",
    "timed_out",
    "metrics",
    "text",
    "word_edits",
    "reference_word_count",
    "final_count",
    "commit_source_ends",
    "commit_input_samples",
    "traces",
    "native_windows",
    "capacity_restored",
    "queue_depth_after_close",
    "budget_lease_count_after_close",
    "all_windows_closed",
    "all_window_capacity_restored",
)


def stream_comparison(reference, candidate):
    required = {*PARITY_FIELDS, "forward_counters", "final_trace"}
    for label, record in (("reference", reference), ("candidate", candidate)):
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be a receipt object")
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{label} is missing required fields: {sorted(missing)}")
    equal = {field: candidate[field] == reference[field] for field in PARITY_FIELDS}
    before = reference["forward_counters"]
    after = candidate["forward_counters"]
    return dict(
        exact_fields=equal,
        all_observable_stream_fields_exact=all(equal.values()),
        decoder_forwards_before=before["decoder_forwards"],
        decoder_forwards_after=after["decoder_forwards"],
        decoder_forwards_saved=before["decoder_forwards"] - after["decoder_forwards"],
        encoder_forwards_before=before["encoder_forwards"],
        encoder_forwards_after=after["encoder_forwards"],
        final_no_speech_delta=(
            candidate["final_trace"]["no_speech_prob"]
            - reference["final_trace"]["no_speech_prob"]
        )
        if candidate.get("final_trace") and reference.get("final_trace")
        else None,
        numerical_score_identity_qualified=False,
        full_event_payload_identity_qualified=False,
        comparison_scope="archived fields only; raw token scores are not in the control receipt",
    )


def run_worker(model_path, manifest, baseline):
    from tools.verify_deferred_word_commits import run_worker as stream_worker
    from whisper_runtime import native_setup

    captured = io.StringIO()
    observations = []
    draft = ()
    models = []
    before_hooks = []
    factory = native_setup.create_stream
    failure = None
    with ExitStack() as patches:

        def create_stream(**kwargs):
            stream = factory(**kwargs)
            model = stream._adapter._model
            if models:
                raise RuntimeError("only one model/factory is allowed")
            models.append((model, stream._adapter.model_identity.fingerprint))
            before_hooks.extend(
                (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
                for m in model.modules()
            )
            from whisper.decoding import DecodingTask, _DecodingRun

            start = DecodingTask._start_run
            finalize = _DecodingRun.finalize

            def start_run(task, mel):
                if len(observations) >= 20:
                    raise RuntimeError("20-window diagnostic limit exceeded")
                if (
                    task.options.temperature != 0
                    or task.options.beam_size is not None
                    or task.options.best_of is not None
                    or task.options.language != "en"
                    or task.options.fp16
                    or task.options.without_timestamps
                ):
                    raise RuntimeError(
                        "diagnostic requires the registered greedy FP32 profile"
                    )
                run = start(task, mel)
                if draft:
                    run.inference = StreamDraftInference(run.inference, draft)
                return run

            def finish_run(run):
                nonlocal draft
                result = finalize(run)
                if len(result) != 1:
                    raise RuntimeError("diagnostic requires exactly one result")
                wrapped = run.inference
                original = (
                    wrapped.original
                    if isinstance(wrapped, StreamDraftInference)
                    else wrapped
                )
                if original.kv_cache or original.hooks:
                    raise RuntimeError(
                        "completed decoder did not clean its owned cache"
                    )
                observations.append(
                    dict(
                        index=len(observations) + 1,
                        token_steps=run.step_index,
                        tokens=list(result[0].tokens),
                        avg_logprob=float(result[0].avg_logprob),
                        no_speech_prob=float(result[0].no_speech_prob),
                        stats=dict(wrapped.stats)
                        if isinstance(wrapped, StreamDraftInference)
                        else None,
                        cache_empty=True,
                    )
                )
                # Keep only token IDs. Previous encoder features, self/cross KV,
                # scores and word alignment never become decoder evidence.
                draft = tuple(result[0].tokens[:32])
                return result

            patches.enter_context(patch.object(DecodingTask, "_start_run", start_run))
            patches.enter_context(patch.object(_DecodingRun, "finalize", finish_run))
            return stream

        patches.enter_context(
            patch.object(native_setup, "create_stream", create_stream)
        )
        try:
            with redirect_stdout(captured):
                stream_worker(
                    model_path,
                    "candidate",
                    manifest=manifest,
                    reuse_alignment_features=True,
                    legacy_alignment=False,
                    case="continuous",
                    cell_timeout_seconds=140,
                    left_context_ms=20000,
                    word_context_limit_ms=24000,
                )
        except BaseException as exc:
            failure = dict(error=repr(exc), traceback=traceback.format_exc())
    lines = [
        json.loads(line) for line in captured.getvalue().splitlines() if line.strip()
    ]
    report = (
        lines[0]
        if len(lines) == 1
        else dict(done=False, unexpected_worker_stdout=captured.getvalue())
    )
    model_unchanged = (
        len(models) == 1 and native_setup._fingerprint(models[0][0]) == models[0][1]
    )
    hooks_unchanged = len(models) == 1 and before_hooks == [
        (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
        for m in models[0][0].modules()
    ]
    reference = json.loads(baseline.read_text(encoding="utf-8").strip())
    report["draft_experiment"] = dict(
        schema="stream-draft-cpu/v1",
        production_feature=False,
        single_cell=True,
        hard_parent_timeout_seconds=140,
        max_draft_tokens=32,
        previous_tokens_are_untrusted=True,
        old_audio_features_or_kv_reused=False,
        score_thresholds_changed=False,
        original_step_filters_alignment_publication=True,
        model_unchanged=model_unchanged,
        hooks_unchanged=hooks_unchanged,
        total_matched_draft_tokens=sum(
            o["stats"]["matched_tokens"] for o in observations if o["stats"]
        ),
        observations=observations,
        baseline_sha256=sha256(baseline.read_bytes()).hexdigest(),
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        draft_tool_sha256=sha256(
            (ROOT / "tools/verify_decoder_draft.py").read_bytes()
        ).hexdigest(),
        failure=failure,
    )
    if "forward_counters" in report:
        report["draft_experiment"]["comparison"] = stream_comparison(reference, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / ".tmp-composed-native/manifest.json"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT
        / ".tmp-composed-native/cpu-fast20k24k-reuse-continuous-140s.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        report = run_worker(
            args.model.resolve(), args.manifest.resolve(), args.baseline.resolve()
        )
        print(json.dumps(report, allow_nan=False))
        return (
            0 if report.get("done") and not report["draft_experiment"]["failure"] else 1
        )
    if args.output is None or args.output.exists():
        parser.error("--output must name a new receipt")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [
                str(ROOT / ".tmp-native/venv/Scripts/python.exe"),
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                "--model",
                str(args.model.resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--baseline",
                str(args.baseline.resolve()),
            ],
            cwd=ROOT,
            timeout=140,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError:
            report = dict(done=False, worker_stdout=completed.stdout)
        report["worker_stderr"] = completed.stderr
        report["worker_returncode"] = completed.returncode
    except subprocess.TimeoutExpired:
        report = dict(done=False, timed_out=True)
    report["parent_wall_seconds"] = time.monotonic() - started
    report["parent_timeout_seconds"] = 140
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report.get("done") else 1


if __name__ == "__main__":
    raise SystemExit(main())
