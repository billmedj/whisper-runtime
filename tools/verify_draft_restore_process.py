"""Bounded fresh-process CPU draft savepoint regression; no downloads or GPU.

Three sequential interpreters run an uninterrupted control, a checkpoint prefix,
and a restored suffix. Only JSON/PCM publication state crosses the process exit.
This is not exact-token-state migration, a GPU release test, or a timing claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
FROZEN = (
    ROOT
    / "artifacts/modal/integrated-draft-t4-20260907-v1/integrated-draft-preflight.json"
)
SCORES = frozenset(("avg_logprob", "no_speech_prob", "probability"))
CHUNK_BYTES = 64000
PHASE_TIMEOUT_SECONDS = 120


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def source_record():
    frozen = json.loads(FROZEN.read_bytes())["source"]["files"]
    expected = {item["path"]: item["sha256"] for item in frozen}
    names = sorted(
        set(expected)
        | {
            "tools/verify_draft_restore_process.py",
            "tools/test_verify_draft_restore_process.py",
        }
    )
    observed = {name: digest(ROOT / name) for name in names}
    mismatches = [name for name, value in expected.items() if observed[name] != value]
    require(not mismatches, f"frozen source changed: {mismatches}")
    return dict(frozen_preflight_sha256=digest(FROZEN), files=observed)


def without_scores(value):
    if isinstance(value, dict):
        return {k: without_scores(v) for k, v in value.items() if k not in SCORES}
    if isinstance(value, (list, tuple)):
        return [without_scores(v) for v in value]
    return value


def differences(left, right, path=""):
    """Return every differing value, without silently applying tolerances."""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            result.extend(differences(left.get(key), right.get(key), f"{path}/{key}"))
        return result
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [dict(path=path, control=left, restored=right)]
        return [
            item
            for i, (a, b) in enumerate(zip(left, right))
            for item in differences(a, b, f"{path}/{i}")
        ]
    if left == right:
        return []
    result = dict(path=path, control=left, restored=right)
    if type(left) is float and type(right) is float:
        result["absolute_delta"] = abs(left - right)
    return [result]


class ObservedAdapter:
    """Observe real admission arguments/owners without patching backend methods."""

    def __init__(self, adapter):
        self.adapter, self.admissions, self.owners = adapter, [], []

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def start_window(self, **kwargs):
        row = {key: kwargs[key] for key in ("window_id", "start_ms", "end_ms")}
        row["draft_tokens"] = list(kwargs.get("draft_tokens", ()))
        row["options"] = asdict(kwargs["options"])
        run = self.adapter.start_window(**kwargs)
        inference = run._backend_run.inference
        row["inference_type"] = type(inference).__name__
        row["parallel_prefills_at_start"] = getattr(inference, "stats", {}).get(
            "parallel_prefills", 0
        )
        self.admissions.append(row)
        self.owners.append((run, inference))
        return run

    def lifecycle(self):
        rows = []
        for run, inference in self.owners:
            original = getattr(inference, "original", inference)
            row = dict(
                closed=run.closed,
                capacity_released=run.capacity_released,
                cache_cleared=not original.kv_cache,
                hooks_cleared=not getattr(original, "hooks", ()),
            )
            if hasattr(inference, "stats"):
                row.update(
                    rows_cleared=inference.rows is None,
                    audio_cleared=inference.audio_features is None,
                    draft_cleared=not inference.draft,
                    exactly_one_cleanup=inference.stats["cleanup_calls"] == 1,
                )
            rows.append(row)
        return rows


def phase(args, incoming=None):
    from tools.prepare_acoustic_cases import build_cases
    from tools.verify_deferred_word_commits import _count_forwards
    from whisper_runtime import native_setup
    from whisper_runtime.adapters import ContinuousTranscriptStream

    started_ns, sources = time.time_ns(), source_record()
    audio_manifest, cases = build_cases()
    # One untrimmed occurrence of each of the three known utterances (16.83s),
    # rather than the repeated 33.66s registered no-added-pauses stress sequence.
    samples = sum(item["sample_count"] for item in audio_manifest["source_fixtures"])
    pcm = cases["concatenated-no-added-pauses"][: samples * 2]
    require(samples == 269280 and len(pcm) == samples * 2, "fixture recipe changed")
    audio = dict(
        pcm_sha256=sha256(pcm).hexdigest(),
        samples=samples,
        duration_seconds=samples / 16000,
        encoding="pcm_s16le",
        sample_rate_hz=16000,
        source_fixtures=audio_manifest["source_fixtures"],
        recipe="First three untrimmed registered utterances, no inserted pauses",
    )
    pipeline = (
        "sha256:" + sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    )
    config = replace(
        native_setup.CLI_STREAM_CONFIG,
        left_context_ms=20000,
        word_context_limit_ms=24000,
        max_draft_tokens=32,
    )
    initial = native_setup.create_stream(
        manifest=args.manifest,
        model=args.model,
        device="cpu",
        reuse_alignment_features=True,
        config=config,
    )
    raw_adapter, mel_builder, options = (
        initial._adapter,
        initial._mel_builder,
        initial._options,
    )
    initial.close()
    observed = ObservedAdapter(raw_adapter)
    model = raw_adapter._model

    def hooks():
        return [
            (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
            for m in model.modules()
        ]

    original_hooks = hooks()
    initial_restore = None
    if args.phase == "consumer":
        require(incoming["pipeline_identity"] == pipeline, "restore pipeline changed")
        stream = ContinuousTranscriptStream.from_checkpoint(
            observed,
            path=args.savepoint,
            expected_sha256=incoming["checkpoint_sha256"],
            pipeline_identity=pipeline,
            mel_builder=mel_builder,
        )
        initial_restore = dict(
            state_exact=json.loads(json.dumps(asdict(stream.state)))
            == incoming["state"],
            metrics_exact=asdict(stream.metrics) == incoming["metrics"],
            old_draft_absent=not stream._draft_tokens,
            inactive=not stream.active,
            no_native_admissions=not observed.admissions,
            no_lease=raw_adapter.worker.budget.lease_count == 0,
            accepted_samples=stream.accepted_samples,
            retained_from_sample=stream.retained_from_sample,
            expected_chunk=stream.expected_chunk,
        )
    else:
        stream = ContinuousTranscriptStream(
            observed,
            stream_id="native-draft-fresh-process",
            mel_builder=mel_builder,
            options=options,
            config=config,
            rng_seed=7,
        )
    require(stream.config == config, "configuration not preserved")
    events, traces, inputs, boundary = [], [], [], None
    checkpoint_sha256 = None
    deadline = time.monotonic() + 90
    try:
        with _count_forwards(model) as counts:
            for _ in range(10000):
                require(time.monotonic() <= deadline, "90-second phase driver limit")
                if stream.done:
                    break
                if not stream.ready:
                    offset = stream.accepted_samples * 2
                    require(offset < len(pcm), "driver stalled after all input")
                    content = pcm[offset : offset + CHUNK_BYTES]
                    sequence = stream.expected_chunk
                    stream.push(sequence, content)
                    inputs.append(
                        dict(
                            sequence=sequence,
                            start_sample=offset // 2,
                            end_sample=(offset + len(content)) // 2,
                        )
                    )
                    if offset + len(content) == len(pcm):
                        stream.finish_input()
                    continue
                batch = stream.step()
                events.extend(asdict(event) for event in batch)
                trace = stream.last_trace
                if trace is not None and (
                    not traces or traces[-1]["decode_index"] != trace.decode_index
                ):
                    traces.append(asdict(trace))
                if (
                    boundary is None
                    and not stream.done
                    and any(e.kind.value == "commit" for e in batch)
                ):
                    boundary = dict(
                        native_admissions=len(observed.admissions),
                        old_draft_tokens=list(stream._draft_tokens),
                        accepted_samples=stream.accepted_samples,
                        committed_samples=stream.metrics.committed_samples,
                        retained_from_sample=stream.retained_from_sample,
                        event_count=len(events),
                        inactive=not stream.active,
                    )
                    if args.phase == "producer":
                        require(
                            boundary["old_draft_tokens"], "no old hint at save boundary"
                        )
                        checkpoint_sha256 = stream.save_checkpoint(
                            args.savepoint, pipeline_identity=pipeline
                        )
                        break
            else:
                raise RuntimeError("10000-step guard exhausted")
            state, metrics, completed = (
                asdict(stream.state),
                asdict(stream.metrics),
                stream.done,
            )
    finally:
        stream.close()
    lifecycle = observed.lifecycle()
    cleanup = dict(
        capacity_restored=raw_adapter.worker.budget.available
        == raw_adapter.worker.budget.capacity,
        no_leases=raw_adapter.worker.budget.lease_count == 0,
        queue_empty=raw_adapter.worker.queue_depth == 0,
        old_hint_cleared=not stream._draft_tokens,
        all_native_owners_closed=all(all(row.values()) for row in lifecycle),
        hooks_unchanged=original_hooks == hooks(),
        model_unchanged=native_setup._fingerprint(model)
        == raw_adapter.model_identity.fingerprint,
        sources_unchanged=source_record() == sources,
    )
    require(all(cleanup.values()), f"cleanup failure: {cleanup}")
    require(
        checkpoint_sha256 is not None if args.phase == "producer" else completed,
        "phase did not reach its required terminal boundary",
    )
    checkpoint_fields = None
    if args.phase == "producer":
        checkpoint_fields = list(
            json.loads(args.savepoint.read_bytes())["payload"]["items"]["checkpoint"][
                "fields"
            ]
        )
    return dict(
        phase=args.phase,
        pid=os.getpid(),
        parent_pid=os.getppid(),
        started_ns=started_ns,
        ended_ns=time.time_ns(),
        executable=sys.executable,
        cpu_threads=1,
        gpu_used=False,
        sources=sources,
        model=asdict(raw_adapter.model_identity),
        model_file_sha256=digest(args.model),
        manifest_sha256=digest(args.manifest),
        audio=audio,
        pipeline_identity=pipeline,
        config=asdict(config),
        options=asdict(options),
        initial_restore=initial_restore,
        completed=completed,
        boundary=boundary,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_fields=checkpoint_fields,
        events=events,
        traces=traces,
        state=state,
        metrics=metrics,
        input_admissions=inputs,
        native_admissions=observed.admissions,
        forward_counts=counts,
        native_lifecycle=lifecycle,
        cleanup=cleanup,
    )


def combine(control, producer, consumer, orchestration):
    restored_events = producer["events"] + consumer["events"]
    restored_traces = producer["traces"] + consumer["traces"]
    restored_admissions = producer["native_admissions"] + consumer["native_admissions"]
    ignored = {"draft_tokens", "inference_type", "parallel_prefills_at_start"}

    def admission_core(rows):
        return [{k: v for k, v in row.items() if k not in ignored} for row in rows]

    def coverage(rows):
        return [
            [e["start_sample"], e["end_sample"], e["committed_through_sample"]]
            for e in rows
            if e["kind"] == "commit"
        ]

    diffs = differences(control["traces"], restored_traces)
    cursor = 0
    contiguous = True
    for start, end, committed in coverage(restored_events):
        contiguous = contiguous and start == cursor and end == committed and end > start
        cursor = end
    initial = consumer["initial_restore"]
    checks = dict(
        three_distinct_processes=len({r["pid"] for r in (control, producer, consumer)})
        == 3,
        source_exited_before_restore=all(
            row["returncode"] == 0 for row in orchestration
        )
        and orchestration[1]["returned_ns"] < orchestration[2]["launched_ns"]
        and producer["ended_ns"] < consumer["started_ns"],
        same_sources_model_audio_config=all(
            all(
                r[key] == control[key]
                for key in (
                    "sources",
                    "model",
                    "model_file_sha256",
                    "audio",
                    "config",
                    "options",
                    "pipeline_identity",
                    "manifest_sha256",
                )
            )
            for r in (producer, consumer)
        ),
        old_hint_existed=bool(producer["boundary"]["old_draft_tokens"]),
        checkpoint_omits_draft_field="draft_tokens"
        not in producer["checkpoint_fields"],
        restore_state_and_metrics_exact=initial["state_exact"]
        and initial["metrics_exact"],
        restored_native_state_absent=all(
            initial[key]
            for key in (
                "old_draft_absent",
                "inactive",
                "no_native_admissions",
                "no_lease",
            )
        ),
        first_restored_admission_no_draft=bool(consumer["native_admissions"])
        and consumer["native_admissions"][0]["draft_tokens"] == []
        and consumer["native_admissions"][0]["parallel_prefills_at_start"] == 0,
        actual_draft_prefills_observed=any(
            a["parallel_prefills_at_start"] == 1 for a in producer["native_admissions"]
        ),
        exact_events=control["events"] == restored_events,
        exact_tokens_decisions_and_provenance_except_scores=without_scores(
            control["traces"]
        )
        == without_scores(restored_traces),
        exact_source_coverage=coverage(control["events"]) == coverage(restored_events),
        contiguous_all_input_committed=contiguous
        and cursor == control["audio"]["samples"],
        exact_final_state_except_scores=without_scores(control["state"])
        == without_scores(consumer["state"]),
        exact_metrics=control["metrics"] == consumer["metrics"],
        exact_input_admission=control["input_admissions"]
        == producer["input_admissions"] + consumer["input_admissions"],
        exact_native_admission_except_draft=admission_core(control["native_admissions"])
        == admission_core(restored_admissions),
        one_final_event=sum(e["kind"] == "final" for e in restored_events) == 1,
        completed=control["completed"]
        and consumer["completed"]
        and not producer["completed"],
        lifecycle_cleanup=all(
            all(r["cleanup"].values()) for r in (control, producer, consumer)
        ),
    )
    return dict(
        schema="native-draft-fresh-process-cpu/v1",
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        trace_differences=diffs,
        exact_trace_scores=not diffs,
        state_differences=differences(control["state"], consumer["state"]),
        native_admission_differences=differences(
            control["native_admissions"], restored_admissions
        ),
        source_coverage=coverage(restored_events),
        orchestration=orchestration,
        records=[control, producer, consumer],
        gpu_used=False,
        model_downloaded=False,
        source_paced=False,
        exact_token_state_restore=False,
        gpu_release_claim=False,
        migration_claim=False,
        claim="One CPU publication-boundary restart on 16.83s of fixed known PCM. No GPU, timing, durable-delivery, or generalization claim.",
    )


def verify(args):
    require(not args.output.exists(), "existing evidence is never replaced")
    savepoint = args.output.with_suffix(".savepoint.json")
    require(not savepoint.exists(), "existing savepoint is never replaced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipts, orchestration = [], []
    for name in ("control", "producer", "consumer"):
        command = [
            str(ROOT / ".tmp-native/venv/Scripts/python.exe"),
            "-B",
            str(Path(__file__).resolve()),
            "--phase",
            name,
            "--model",
            str(args.model.resolve()),
            "--manifest",
            str(args.manifest.resolve()),
            "--savepoint",
            str(savepoint.resolve()),
        ]
        launched_ns = time.time_ns()
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            input=json.dumps(receipts[1]) if name == "consumer" else None,
            timeout=PHASE_TIMEOUT_SECONDS,
            check=False,
        )
        orchestration.append(
            dict(
                phase=name,
                command=command,
                launched_ns=launched_ns,
                returned_ns=time.time_ns(),
                returncode=result.returncode,
                stderr=result.stderr,
            )
        )
        if result.returncode:
            return dict(
                status="failed",
                phase=name,
                stdout=result.stdout,
                orchestration=orchestration,
                records=receipts,
            )
        receipts.append(json.loads(result.stdout))
    return combine(*receipts, orchestration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--phase", choices=("control", "producer", "consumer"), help=argparse.SUPPRESS
    )
    parser.add_argument("--savepoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.phase and (args.output is None or args.output.exists()):
        parser.error("provide a new --output path")
    try:
        report = (
            phase(args, json.load(sys.stdin) if args.phase == "consumer" else None)
            if args.phase
            else verify(args)
        )
    except BaseException:
        report = dict(status="failed", error=traceback.format_exc())
    if args.phase:
        print(json.dumps(report, allow_nan=False))
    else:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print(
            json.dumps(
                dict(
                    output=str(args.output),
                    status=report["status"],
                    checks=report.get("checks"),
                )
            )
        )
    return int(report.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
