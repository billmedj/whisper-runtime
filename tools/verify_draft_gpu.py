"""Read-only recomputation of the registered T4 draft comparison. No inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path


def commits(events):
    latest, seen, output, head = {}, set(), [], 0
    for sequence, event in enumerate(events, 1):
        if event["sequence_number"] != sequence:
            raise ValueError("event sequence differs")
        if event["kind"] == "final":
            if sequence != len(events):
                raise ValueError("FINAL is not last")
            continue
        if event["start_sample"] != head or event["segment_id"] in seen:
            raise ValueError("event revises committed audio or leaves a gap")
        if event["kind"] in ("provisional", "replace"):
            latest[event["segment_id"]] = event
            continue
        if event["kind"] != "commit":
            raise ValueError("unsupported event")
        prior = latest[event["segment_id"]]
        fields = ("revision", "start_sample", "end_sample", "session_version")
        if prior["sequence_number"] != sequence - 1 or any(
            prior[k] != event[k] for k in fields
        ):
            raise ValueError("commit does not select the preceding exact revision")
        output.append({k: event[k] for k in ("start_sample", "end_sample")})
        output[-1]["text"] = prior["text"]
        head = event["end_sample"]
        seen.add(event["segment_id"])
    return output


def policy(cell):
    fields = (
        "analysis_start_sample",
        "analysis_end_sample",
        "action",
        "reason",
        "committed_before_sample",
        "retained_from_sample",
        "publication_span",
        "eof",
    )
    selected = []
    for trace in cell["decision_traces"]:
        item = {key: trace.get(key) for key in fields}
        for name, keys in {
            "audio_evidence": ("state", "reason", "observation"),
            "word_publication": (
                "start_ms",
                "end_ms",
                "text",
                "word_start",
                "word_end",
                "boundary_tolerance_ms",
                "final",
            ),
            "silence_publication": (
                "analysis_span",
                "analysis_start_sample",
                "start_ms",
                "end_ms",
                "text",
                "observation",
            ),
        }.items():
            value = trace.get(name)
            item[name] = (
                None if value is None else {key: value.get(key) for key in keys}
            )
        selected.append(item)
    return selected


def measure(cell):
    forwards = [f for window in cell["windows"] for f in window["forwards"]]
    phases = {}
    for forward in forwards:
        key = f"{forward['kind']}/{forward['phase']}"
        row = phases.setdefault(key, dict(calls=0, interval_ms=0.0, input_tokens=0))
        row["calls"] += 1
        row["interval_ms"] += forward["interval_ms"]
        row["input_tokens"] += forward.get("input_tokens", 0)
    wall = {}
    for window in cell["windows"]:
        for phase, value in window["operation_wall_ns"].items():
            wall[phase] = wall.get(phase, 0) + value
    final_time = next(
        elapsed
        for event, elapsed in zip(
            cell["events"], cell["pacing"]["event_elapsed_ns"], strict=True
        )
        if event["kind"] == "final"
    )
    observations = [w["decode_observation"] for w in cell["windows"]]
    return dict(
        arm=cell["arm"],
        phases=phases,
        operation_wall_ns=wall,
        total_operation_wall_ns=sum(wall.values()),
        commits=commits(cell["events"]),
        native_windows=len(cell["windows"]),
        observations_match=observations == cell["draft_observations"],
        closed_and_released=all(
            w["closed"] and w["capacity_restored"] for w in cell["windows"]
        ),
        matched_tokens=sum(
            o["stats"]["matched_tokens"] for o in observations if o["stats"]
        ),
        proposed_tokens=sum(
            o["stats"]["proposed_tokens"] for o in observations if o["stats"]
        ),
        memory={
            k: cell[k]
            for k in (
                "memory_before",
                "memory_after_close",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        },
        clock=dict(
            first_commit_ns=cell["pacing"]["first_commit_ns"],
            final_event_ns=final_time,
            final_after_input_finished_ns=final_time
            - cell["pacing"]["input_finished_ns"],
        ),
    )


def audit(directory, historical, root):
    names = (
        "acoustic-diagnostic.json",
        "acoustic-diagnostic.result.zlib",
        "draft-preflight.json",
        "acoustic-diagnostic.attempt.jsonl",
    )
    files = {name: (directory / name).read_bytes() for name in names}
    result, raw, preflight = (
        json.loads(files[names[0]]),
        json.loads(zlib.decompress(files[names[1]])),
        json.loads(files[names[2]]),
    )
    provenance = dict(
        raw_matches_local_result=raw
        == {
            k: v
            for k, v in result.items()
            if k not in ("app_id", "ephemeral_app_exit_completed")
        },
        source_matches_preflight=result["source"]["snapshot"] == preflight["source"],
        current_frozen_files_match=all(
            hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
            and (root / item["path"]).stat().st_size == item["size_bytes"]
            for item in preflight["source"]["files"]
        ),
    )
    before, after = result["cells"]
    measured = [measure(c) for c in (before, after)]
    observations = [
        [w["decode_observation"] for w in c["windows"]] for c in (before, after)
    ]
    spans = lambda c: [(w["start_ms"], w["end_ms"]) for w in c["windows"]]  # noqa: E731
    exact = dict(
        committed_output=measured[0]["commits"] == measured[1]["commits"],
        window_tokens=[o["result"]["tokens"] for o in observations[0]]
        == [o["result"]["tokens"] for o in observations[1]],
        window_text=[o["result"]["text"] for o in observations[0]]
        == [o["result"]["text"] for o in observations[1]],
        native_window_spans=spans(before) == spans(after),
        policy=policy(before) == policy(after),
        metrics=before["metrics"] == after["metrics"],
    )
    prior = next(
        c
        for c in json.loads(historical.read_bytes())["cells"]
        if c["arm"] == "fast-reuse"
    )
    reproduction = dict(
        committed_output=commits(prior["events"]) == measured[0]["commits"],
        native_window_spans=spans(prior) == spans(before),
        policy=policy(prior) == policy(before),
        forward_counts=prior["summary"]["forwards"] == before["summary"]["forwards"],
        reference_errors=prior["summary"]["word_errors"]
        == before["summary"]["word_errors"],
    )
    totals = [sum(p["interval_ms"] for p in arm["phases"].values()) for arm in measured]
    efficiency = dict(
        total_forward_intervals_ms=totals,
        total_forward_intervals_change_percent=(totals[1] / totals[0] - 1) * 100,
        decode_operation_wall_change_percent=(
            measured[1]["operation_wall_ns"]["decode"]
            / measured[0]["operation_wall_ns"]["decode"]
            - 1
        )
        * 100,
        total_operation_wall_change_percent=(
            measured[1]["total_operation_wall_ns"]
            / measured[0]["total_operation_wall_ns"]
            - 1
        )
        * 100,
    )
    efficiency["gate_recomputed"] = (
        all(exact.values())
        and totals[1] < totals[0]
        and measured[1]["total_operation_wall_ns"]
        < measured[0]["total_operation_wall_ns"]
        and all(all(c["checks"].values()) for c in (before, after))
    )
    if (
        efficiency["gate_recomputed"]
        != result["comparison"]["measured_efficiency_gate"]
    ):
        raise ValueError("registered efficiency verdict differs")
    return dict(
        schema="draft-t4-independent-audit/v1",
        provenance=provenance,
        files={
            name: hashlib.sha256(value).hexdigest() for name, value in files.items()
        },
        source_file_count=len(preflight["source"]["files"]),
        arms=measured,
        exact=exact,
        historical_control_reproduction=reproduction,
        efficiency=efficiency,
        max_absolute_score_difference={
            k: max(
                abs(a["result"][k] - b["result"][k])
                for a, b in zip(*observations, strict=True)
            )
            for k in ("avg_logprob", "no_speech_prob", "compression_ratio")
        },
        cancellation_probe={
            k: v for k, v in result["probes"][0].items() if k != "windows"
        },
        lifecycle={
            k: result[k]
            for k in (
                "status",
                "capacity_restored",
                "hooks_unchanged",
                "native_window_count",
                "elapsed_ns",
            )
        },
        model_unchanged=result["model"]["initial_sha256"]
        == result["model"]["final_sha256"],
        registered_efficiency_gate=result["comparison"]["measured_efficiency_gate"],
        caveats=[
            "Single fixed-order paired screening run, not a general speed or energy estimate.",
            "Control alignment has six 190-351 ms host stalls versus about 10 ms in the candidate; shape/JIT warmup is a hypothesis, not established causality.",
            "Do not attribute the full 34.03 percent total-operation wall reduction to draft decoding. Decode-operation wall falls 16.87 percent on this run.",
            "Native-operation wall excludes mel construction, stream policy, driver work and source waiting.",
            "Numerical scores differ. Exact tokens and policy decisions here do not prove equality near all decision thresholds.",
            "Cancellation records actual native cleanup; published_events=0 is assigned, not a public-stream event counter.",
            "Peak allocated/reserved memory is unchanged between arms; the reserved allocator pool is not released to other applications.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("historical", type=Path)
    args = parser.parse_args()
    report = audit(args.directory, args.historical, Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2))
    if not all(report["provenance"].values()) or not all(report["exact"].values()):
        raise SystemExit(1)
