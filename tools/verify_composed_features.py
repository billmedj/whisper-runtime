"""Audit one frozen three-arm receipt offline; no Modal, inference or writes."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from infra import modal_composed_features as producer


def audit(record):
    arms, samples = {}, record["input"]["sample_count"]
    for cell in record["cells"]:
        events, windows, saved = cell["events"], cell["windows"], cell["summary"]
        indices = [i for i, e in enumerate(events) if e["kind"] == "commit"]
        commits = [
            (events[i]["start_sample"], events[i]["end_sample"], events[i - 1]["text"])
            for i in indices
        ]
        text = " ".join(c[2] for c in commits)
        calls = [f for w in windows for f in w.get("forwards", [])]
        counts = dict(Counter(f["kind"] for f in calls))
        times = {}
        for kind in ("encoder", "decoder"):
            counts.setdefault(kind, 0)
            values = [f["interval_ms"] for f in calls if f["kind"] == kind]
            times[kind] = None if None in values else sum(values)
        pacing = cell.get("pacing") or {}
        clock = pacing["event_elapsed_ns"][indices[0]] if indices and pacing else None
        saved_commits = [
            tuple(c[k] for k in ("start_sample", "end_sample", "text"))
            for c in saved["commits"]
        ]
        producer._require(
            (commits, text, counts, times, clock)
            == (
                saved_commits,
                saved["text"],
                saved["forwards"],
                saved["forward_interval_ms"],
                saved["first_commit_ns"],
            ),
            "raw events/forwards disagree with summary",
        )
        metrics = cell.get("metrics", {})
        full = {
            metrics.get("accepted_samples"),
            metrics.get("committed_samples"),
            samples,
        } == {samples}
        contiguous = (
            bool(commits)
            and commits[-1][1] == samples
            and ([c[0] for c in commits] == [0] + [c[1] for c in commits[:-1]])
        )
        final = [i for i, e in enumerate(events) if e["kind"] == "final"] == [
            len(events) - 1
        ]
        cleanup = (
            cell.get("capacity_restored") is True
            and cell.get("cleanup_error") is None
            and all(w["closed"] and w["capacity_restored"] for w in windows)
        )
        arms[cell["arm"]] = dict(
            commits=commits,
            text=text,
            forwards=counts,
            cuda_forward_ms=times,
            native_windows=len(windows),
            cuda_forward_sum_ms=None if None in times.values() else sum(times.values()),
            encoder_phases={
                p: sum(f["kind"] == "encoder" and f["phase"] == p for f in calls)
                for p in ("decode", "alignment")
            },
            first_commit_ns=clock,
            **{
                k: cell.get(k)
                for k in ("elapsed_ns", "peak_allocated_bytes", "peak_reserved_bytes")
            },
            context_ms=[
                cell["config"][k] for k in ("left_context_ms", "word_context_limit_ms")
            ],
            full_samples=full and contiguous,
            final_once_last=final,
            cleanup=cleanup,
            completed=cell["stream_status"] == "completed" and cell.get("done") is True,
        )
    checks = dict(
        receipt_completed=record["status"] == "completed",
        three_completed_arms=set(arms) == set(producer.ARMS)
        and all(
            all(
                a[k]
                for k in ("completed", "full_samples", "final_once_last", "cleanup")
            )
            for a in arms.values()
        ),
    )
    comparisons = {}
    if set(arms) == set(producer.ARMS):
        control, legacy, reuse = (arms[k] for k in producer.ARMS)
        checks.update(
            exact_fast_commit_parity=legacy["commits"] == reuse["commits"],
            exact_fast_text_parity=legacy["text"] == reuse["text"],
            same_fast_context=legacy["context_ms"]
            == reuse["context_ms"]
            == [20000, 24000],
            alignment_encoders_removed=legacy["encoder_phases"]["alignment"]
            > 0
            == reuse["encoder_phases"]["alignment"],
            fewer_encoder_forwards=reuse["forwards"]["encoder"]
            < legacy["forwards"]["encoder"],
            no_added_decoder_forwards=reuse["forwards"]["decoder"]
            <= legacy["forwards"]["decoder"],
            both_fast_commits_earlier=all(
                a["first_commit_ns"] is not None
                and control["first_commit_ns"] is not None
                and a["first_commit_ns"] < control["first_commit_ns"]
                for a in (legacy, reuse)
            ),
        )
        for label, baseline in (
            ("reuse_vs_fast_legacy", legacy),
            ("reuse_vs_cli_legacy", control),
        ):
            old, new = baseline["cuda_forward_sum_ms"], reuse["cuda_forward_sum_ms"]
            comparisons[label] = dict(
                encoder_call_delta=reuse["forwards"]["encoder"]
                - baseline["forwards"]["encoder"],
                decoder_call_delta=reuse["forwards"]["decoder"]
                - baseline["forwards"]["decoder"],
                cuda_forward_sum_ratio=None
                if old in (None, 0) or new is None
                else new / old,
            )
    return dict(
        arms=arms,
        checks=checks,
        comparisons=comparisons,
        preregistered_cross_arm_expectations_met=all(checks.values()),
        cpu_elapsed_comparison_applicable=False,
        general_speed_or_energy_claim=False,
        timing_scope="Per-forward CUDA-event sums; warmup excluded, not occupancy or word latency.",
    )


def verify(result, preflight):
    raw, frozen_raw = Path(result).read_bytes(), Path(preflight).read_bytes()
    record, frozen = json.loads(raw), json.loads(frozen_raw)
    producer._require(
        record["scope"] == frozen["scope"] and record["input"] == frozen["input"],
        "preflight differs",
    )
    producer.validate_record(record, frozen["source"])
    return dict(
        result_sha256=hashlib.sha256(raw).hexdigest(),
        preflight_sha256=hashlib.sha256(frozen_raw).hexdigest(),
        **audit(record),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("preflight", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.result, args.preflight)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["preregistered_cross_arm_expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
