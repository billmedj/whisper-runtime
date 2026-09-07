"""Independent, standard-library-only replay of one saved integrated GPU screen.

Read-only: emits JSON to stdout, never launches Modal/imports the producer.
Usage: python tools/audit_integrated_draft.py [saved-run-directory]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/modal/integrated-draft-t4-20260907-v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def commits(cell):
    latest, result, committed = {}, [], set()
    previous, head = None, 0
    for sequence, event in enumerate(cell["events"], 1):
        assert event["sequence_number"] == sequence
        assert previous is None or previous["kind"] != "final"
        if event["kind"] == "final":
            previous = event
            continue
        assert event["start_sample"] == head and event["sample_rate_hz"] == 16000
        assert event["segment_id"] not in committed
        if event["kind"] in {"provisional", "replace"}:
            old = latest.get(event["segment_id"])
            assert event["revision"] == (old["revision"] + 1 if old else 1)
            assert event["kind"] == ("replace" if old else "provisional")
            latest[event["segment_id"]] = event
        elif event["kind"] == "commit":
            revision = latest[event["segment_id"]]
            assert revision == previous and revision["revision"] == event["revision"]
            assert revision["start_sample"] == event["start_sample"]
            assert revision["end_sample"] == event["end_sample"]
            assert revision["session_version"] == event["session_version"]
            assert event["committed_through_sample"] == event["end_sample"] > head
            result.append(
                dict(
                    start_sample=event["start_sample"],
                    end_sample=event["end_sample"],
                    text=revision["text"],
                )
            )
            committed.add(event["segment_id"])
            head = event["end_sample"]
        previous = event
    return result


def source_clocks(cell, total_samples):
    pacing = cell["pacing"]
    head = last_accepted = max_source_lag = max_admission_lag = 0
    valid = no_early = True
    for sequence, admission in enumerate(pacing["admissions"]):
        start, end = admission["start_sample"], admission["end_sample"]
        scheduled, offered, accepted = (
            admission[k] for k in ("scheduled_ns", "offered_ns", "accepted_ns")
        )
        valid &= admission["sequence_number"] == sequence and start == head
        valid &= end == min(head + pacing["config"]["chunk_ms"] * 16, total_samples)
        valid &= (
            scheduled == end * 62500
            and last_accepted <= offered <= accepted <= pacing["elapsed_ns"]
        )
        no_early &= scheduled <= offered
        max_source_lag = max(max_source_lag, offered - scheduled)
        max_admission_lag = max(max_admission_lag, accepted - scheduled)
        head, last_accepted = end, accepted
    output_clocks_valid = True
    for objects, key, sample in (
        (cell["events"], "event_elapsed_ns", "end_sample"),
        (cell["decision_traces"], "trace_elapsed_ns", "analysis_end_sample"),
    ):
        clocks = pacing[key]
        output_clocks_valid &= len(objects) == len(clocks)
        previous = 0
        for obj, observed in zip(objects, clocks, strict=True):
            output_clocks_valid &= previous <= observed <= pacing["elapsed_ns"]
            if obj.get(sample) is not None:
                output_clocks_valid &= observed >= obj[sample] * 62500
            previous = observed
    return dict(
        admission_count=len(pacing["admissions"]),
        admission_contract=valid,
        source_not_early=no_early,
        full_admitted=head
        == pacing["accepted_samples"]
        == pacing["offered_samples"]
        == total_samples,
        max_source_lag_ns=max_source_lag,
        max_admission_lag_ns=max_admission_lag,
        stored_lag_values_exact=max_source_lag == pacing["max_source_lag_ns"]
        and max_admission_lag == pacing["max_admission_lag_ns"],
        source_lags_within_bound=max(max_source_lag, max_admission_lag)
        <= pacing["config"]["max_source_lag_ms"] * 1000000,
        output_clocks_valid=output_clocks_valid,
        all_events_delivered=pacing["undelivered_events"] == []
        and pacing["undelivered_event_ns"] is None,
    )


def selected_decisions(cell):
    result = []
    for trace in cell["decision_traces"]:
        selected = {
            k: trace.get(k)
            for k in (
                "analysis_start_sample",
                "analysis_end_sample",
                "action",
                "reason",
                "committed_before_sample",
                "retained_from_sample",
                "publication_span",
                "eof",
            )
        }
        for field, keys in (
            ("audio_evidence", ("state", "reason", "observation")),
            (
                "word_publication",
                (
                    "start_ms",
                    "end_ms",
                    "text",
                    "word_start",
                    "word_end",
                    "boundary_tolerance_ms",
                    "final",
                ),
            ),
            (
                "silence_publication",
                (
                    "analysis_span",
                    "analysis_start_sample",
                    "start_ms",
                    "end_ms",
                    "text",
                    "observation",
                ),
            ),
        ):
            value = trace.get(field)
            selected[field] = {k: (value or {}).get(k) for k in keys}
        result.append(selected)
    return result


def measure(cell):
    counts, positions, cuda_ms = Counter(), Counter(), defaultdict(float)
    wall_ns, stats = Counter(), Counter()
    for window in cell["windows"]:
        wall_ns.update(window["operation_wall_ns"])
        for forward in window.get("forwards", []):
            key = forward["phase"] + ":" + forward["kind"]
            counts[key] += 1
            value = forward["interval_ms"]
            assert type(value) in (int, float) and math.isfinite(value) and value >= 0
            cuda_ms[key] += value
            if forward["kind"] == "decoder":
                assert (
                    type(forward["input_tokens"]) is int and forward["input_tokens"] > 0
                )
                positions[key] += forward["input_tokens"]
        observed = window.get("decode_observation", {}).get("stats")
        if observed:
            stats.update(
                {
                    key: value
                    for key, value in observed.items()
                    if type(value) is int
                    and key not in {"mismatch_index", "crop_length"}
                }
            )
    return dict(
        windows=len(cell["windows"]),
        raw_forwards=dict(counts),
        decoder_input_positions=dict(positions),
        forward_interval_ms=dict(cuda_ms),
        operation_wall_ns=dict(wall_ns),
        total_operation_wall_ns=sum(wall_ns.values()),
        total_forward_interval_ms=sum(cuda_ms.values()),
        draft_stats=dict(stats),
    )


def reduction(control, candidate):
    return dict(
        control=control,
        candidate=candidate,
        delta=candidate - control,
        reduction_percent=100 * (control - candidate) / control if control else None,
    )


def audit(directory):
    output = directory / "acoustic-diagnostic.json"
    raw_path = directory / "acoustic-diagnostic.result.zlib"
    frozen_path = directory / "integrated-draft-preflight.json"
    record = json.loads(output.read_bytes())
    frozen = json.loads(frozen_path.read_bytes())
    raw = json.loads(zlib.decompress(raw_path.read_bytes()))
    snapshot = frozen["source"]
    source_checks = []
    for item in snapshot["files"]:
        content = (ROOT / item["path"]).read_bytes()
        source_checks.append(
            dict(
                path=item["path"],
                matches=len(content) == item["size_bytes"]
                and sha(content) == item["sha256"],
            )
        )
    provenance = dict(
        raw_sha256=sha(raw_path.read_bytes()),
        result_sha256=sha(output.read_bytes()),
        frozen_sha256=sha(frozen_path.read_bytes()),
        source_digest=snapshot["digest"],
        frozen_source_digest_recomputed=digest(snapshot["files"]) == snapshot["digest"],
        frozen_files_count=len(source_checks),
        frozen_files_match=all(s["matches"] for s in source_checks),
        frozen_file_mismatches=[s["path"] for s in source_checks if not s["matches"]],
        source_snapshot_exact=record["source"]["snapshot"] == snapshot,
        producer_hash_exact=record["source"]["registration_sha256"]
        == sha((ROOT / "infra/modal_integrated_draft.py").read_bytes()),
        input_and_scope_exact=record["input"] == frozen["input"]
        and record["scope"] == frozen["scope"],
        compressed_payload_exact=raw
        == {
            k: v
            for k, v in record.items()
            if k not in {"app_id", "ephemeral_app_exit_completed"}
        },
        function_call_id=record["worker"]["function_call_id"],
        function_call_id_hash_exact=record["worker"]["function_call_id_sha256"]
        == sha(record["worker"]["function_call_id"].encode()),
        app_id=record.get("app_id"),
        ephemeral_app_exit_completed=record.get("ephemeral_app_exit_completed"),
    )
    cells, measured = record["cells"], []
    for cell in cells:
        values = measure(cell)
        native_forwards = {
            kind: sum(
                count
                for key, count in values["raw_forwards"].items()
                if key.endswith(":" + kind)
            )
            for kind in ("encoder", "decoder")
        }
        summary = cell["summary"]
        raw_forward_intervals = {
            kind: sum(
                f["interval_ms"]
                for w in cell["windows"]
                for f in w.get("forwards", [])
                if f["kind"] == kind
            )
            for kind in ("encoder", "decoder")
        }
        values.update(
            position=cell["position"],
            arm=cell["arm"],
            event_count=len(cell["events"]),
            event_sha256=digest(cell["events"]),
            commit_count=len(commits(cell)),
            native_summary_matches=(
                native_forwards == summary["forwards"]
                and raw_forward_intervals == summary["forward_interval_ms"]
                and values["operation_wall_ns"] == summary["operation_wall_ns"]
                and values["total_operation_wall_ns"]
                == summary["total_operation_wall_ns"]
                and sum(values["decoder_input_positions"].values())
                == summary["decoder_input_tokens"]
                and commits(cell) == summary["commits"]
                and values["draft_stats"].get("matched_tokens", 0)
                == summary["draft_matched_tokens"]
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
            gpu_peak_increment_bytes=cell["peak_allocated_bytes"]
            - cell["memory_before"]["allocated_bytes"],
            admission_elapsed_ns=cell["admitted_elapsed_ns"],
            source_clock_audit=source_clocks(cell, record["input"]["sample_count"]),
            pacing={
                k: cell["pacing"].get(k)
                for k in (
                    "status",
                    "elapsed_ns",
                    "input_finished_ns",
                    "accepted_samples",
                    "offered_samples",
                    "max_source_lag_ns",
                    "max_admission_lag_ns",
                )
            },
            lifecycle=dict(
                full_input=cell["metrics"]["accepted_samples"]
                == record["input"]["sample_count"],
                full_coverage=cell["metrics"]["committed_samples"]
                == record["input"]["sample_count"],
                final_once=sum(e["kind"] == "final" for e in cell["events"]) == 1,
                done=cell["done"],
                no_error=cell["error"] is None and cell.get("cleanup_error") is None,
                capacity_restored=cell["capacity_restored"],
                hint_cleared=cell["hint_cleared"],
                all_reported_checks=all(cell["checks"].values()),
            ),
        )
        measured.append(values)
        values["largest_alignment_operations"] = [
            dict(
                call_index=w["call_index"],
                start_ms=w["start_ms"],
                end_ms=w["end_ms"],
                wall_ns=w["operation_wall_ns"].get("alignment", 0),
                forward_cuda_ms=sum(
                    f["interval_ms"]
                    for f in w.get("forwards", [])
                    if f["phase"] == "alignment"
                ),
            )
            for w in sorted(
                cell["windows"],
                key=lambda w: w["operation_wall_ns"].get("alignment", 0),
                reverse=True,
            )[:4]
        ]
    pairs = []
    for a, b in ((0, 1), (3, 2)):
        if max(a, b) >= len(cells):
            continue
        control, candidate = cells[a], cells[b]
        co, ca = measured[a], measured[b]
        obs = [
            [
                w["decode_observation"]
                for w in cell["windows"]
                if "decode_observation" in w
            ]
            for cell in (control, candidate)
        ]
        equal = dict(
            events=control["events"] == candidate["events"],
            commit_spans_text=commits(control) == commits(candidate),
            native_window_spans=[
                (w["start_ms"], w["end_ms"]) for w in control["windows"]
            ]
            == [(w["start_ms"], w["end_ms"]) for w in candidate["windows"]],
            current_audio_tokens=[o["result"]["tokens"] for o in obs[0]]
            == [o["result"]["tokens"] for o in obs[1]],
            native_text=[o["result"]["text"] for o in obs[0]]
            == [o["result"]["text"] for o in obs[1]],
            selected_decisions=selected_decisions(control)
            == selected_decisions(candidate),
            commit_source_endpoints=[
                t["analysis_end_sample"]
                for t in control["decision_traces"]
                if t["action"] == "commit"
            ]
            == [
                t["analysis_end_sample"]
                for t in candidate["decision_traces"]
                if t["action"] == "commit"
            ],
        )
        pairs.append(
            dict(
                control=control["position"],
                candidate=candidate["position"],
                exact=equal,
                publication_admission_positions_exact=control["pacing"][
                    "publication_inputs"
                ]
                == candidate["pacing"]["publication_inputs"],
                maximum_absolute_score_deltas={
                    key: max(
                        (
                            abs(x["result"][key] - y["result"][key])
                            for x, y in zip(*obs, strict=True)
                        ),
                        default=0,
                    )
                    for key in ("avg_logprob", "no_speech_prob", "compression_ratio")
                },
                decode_phase_wall_ns=reduction(
                    co["operation_wall_ns"]["decode"], ca["operation_wall_ns"]["decode"]
                ),
                decode_decoder_cuda_ms=reduction(
                    co["forward_interval_ms"]["decode:decoder"],
                    ca["forward_interval_ms"]["decode:decoder"],
                ),
                all_forward_cuda_ms=reduction(
                    co["total_forward_interval_ms"], ca["total_forward_interval_ms"]
                ),
                all_native_wall_ns=reduction(
                    co["total_operation_wall_ns"], ca["total_operation_wall_ns"]
                ),
            )
        )
    grouped = {
        arm: [m for m in measured if m["arm"] == arm]
        for arm in ("fast-reuse", "fast-draft32")
    }
    pooled = {}
    for arm, group in grouped.items():
        values = dict(cells=len(group))
        for key in (
            "raw_forwards",
            "decoder_input_positions",
            "operation_wall_ns",
            "forward_interval_ms",
        ):
            totals = defaultdict(float) if key == "forward_interval_ms" else Counter()
            for item in group:
                for phase, value in item[key].items():
                    totals[phase] += value
            values[key] = dict(totals)
        values["total_operation_wall_ns"] = sum(values["operation_wall_ns"].values())
        values["total_forward_interval_ms"] = sum(
            values["forward_interval_ms"].values()
        )
        pooled[arm] = values
    aa, bb = pooled.values()
    pooled_comparison = {
        key: reduction(aa[key], bb[key])
        for key in ("total_operation_wall_ns", "total_forward_interval_ms")
    }
    for key, field, phase in (
        ("decode_wall_ns", "operation_wall_ns", "decode"),
        ("decode_decoder_cuda_ms", "forward_interval_ms", "decode:decoder"),
        ("alignment_wall_ns", "operation_wall_ns", "alignment"),
        ("decode_decoder_forwards", "raw_forwards", "decode:decoder"),
    ):
        pooled_comparison[key] = reduction(
            aa[field].get(phase, 0), bb[field].get(phase, 0)
        )
    all_windows = [
        w
        for group in (record["warmup"], record["probes"], cells)
        for cell in group
        for w in cell["windows"]
    ]
    lifecycle = dict(
        total_windows=len(all_windows),
        count_and_indices_exact=len(all_windows) == record["native_window_count"] <= 256
        and [w["call_index"] for w in all_windows]
        == list(range(1, len(all_windows) + 1)),
        all_windows_closed=all(
            w["closed"] and w["capacity_restored"] for w in all_windows
        ),
        all_native_cleanup=all(
            w.get("native_cleanup") and all(w["native_cleanup"].values())
            for w in all_windows
        ),
        all_cuda_events_resolved=all(
            f["interval_ms"] is not None
            for w in all_windows
            for f in w.get("forwards", [])
        ),
        worker_capacity_restored=record["capacity_restored"],
        hooks_unchanged=record["hooks_unchanged"],
        model_unchanged=record["model"]["initial_sha256"]
        == record["model"]["final_sha256"],
        warmup_capacity_restored=all(w["capacity_restored"] for w in record["warmup"]),
        cancellation_probe={
            k: v for k, v in record["probes"][0].items() if k != "windows"
        },
    )
    all_tokens = [
        [
            w["decode_observation"]["result"]["tokens"]
            for w in c["windows"]
            if "decode_observation" in w
        ]
        for c in cells
    ]
    order_diagnostics = dict(
        same_control_a1_to_a2_alignment_wall_ns=reduction(
            measured[0]["operation_wall_ns"]["alignment"],
            measured[3]["operation_wall_ns"]["alignment"],
        ),
        same_control_a1_to_a2_decode_wall_ns=reduction(
            measured[0]["operation_wall_ns"]["decode"],
            measured[3]["operation_wall_ns"]["decode"],
        ),
        explanation="The first measured control has much larger alignment host wall time despite similar alignment forward CUDA intervals. This is an order-associated timing asymmetry, consistent with unmeasured shape-dependent initialization or compilation, but the exact cause was not instrumented. It cannot be credited to drafts. Later control alignment is slightly faster than both candidate cells.",
        all_cell_allocated_baselines_and_post_close_equal=len(
            {c["memory_before"]["allocated_bytes"] for c in cells}
            | {c["memory_after_close"]["allocated_bytes"] for c in cells}
        )
        == 1,
        first_control_peak_below_later_control=cells[0]["peak_allocated_bytes"]
        < cells[3]["peak_allocated_bytes"],
        candidate_and_later_control_peaks_equal=cells[1]["peak_allocated_bytes"]
        == cells[2]["peak_allocated_bytes"]
        == cells[3]["peak_allocated_bytes"],
    )
    return dict(
        schema_version=1,
        audit_date="2026-09-07",
        read_only_offline=True,
        input_directory=directory.relative_to(ROOT).as_posix(),
        status=record["status"],
        qualified=record["qualified"],
        elapsed_ns=record["elapsed_ns"],
        stop=record["stop"],
        input=record["input"],
        provenance=provenance,
        schedule=[[c["position"], c["arm"]] for c in cells],
        expected_abba_exact=[(c["position"], c["arm"]) for c in cells]
        == [
            ("a1", "fast-reuse"),
            ("b1", "fast-draft32"),
            ("b2", "fast-draft32"),
            ("a2", "fast-reuse"),
        ],
        warmup=[dict(arm=c["arm"], **measure(c)) for c in record["warmup"]],
        cells=measured,
        pairs=pairs,
        pooled=pooled,
        pooled_comparison=pooled_comparison,
        all_full_events_exact=all(c["events"] == cells[0]["events"] for c in cells[1:]),
        all_current_audio_tokens_exact=all(v == all_tokens[0] for v in all_tokens[1:]),
        observed_order_diagnostics=order_diagnostics,
        lifecycle=lifecycle,
        producer_verdict={
            k: record["comparison"].get(k)
            for k in (
                "complete",
                "accepted",
                "full_events_exact",
                "pooled_faster",
                "efficiency_gate",
            )
        },
        caveats=[
            "One fixed ABBA sequence on one worker and repeated known audio; no randomization, independent replicas, held-out accuracy, CI or production qualification.",
            "Two warmup windows per arm precede measurement; order is A,A,B(no hint),B(with hint), then cancellation. This does not establish steady-state thermal, clocks, allocator or kernel behavior for every later shape.",
            "ABBA balances a linear order trend but cannot remove nonlinear drift or middle-versus-edge effects. Report both adjacent contrasts and pooled equal-count totals.",
            "decode phase wall includes native start/encoder/setup and token stepping; decode:decoder CUDA intervals are decoder-specific. Alignment has its own decoder forwards.",
            "CUDA-event intervals are stream elapsed forward intervals, not occupancy, energy, FLOPs or kernel-only busy time. Host timings include measurement hooks.",
            "Peak memory is allocator-reported device allocation/reservation per cell, not total process GPU/host memory. Reserved allocator growth is not by itself a live-tensor leak.",
            "Publication source endpoints and token parity are exact requirements; score differences and admission-clock changes are reported separately, with no new numeric tolerance.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.directory.resolve()), indent=2, sort_keys=True, allow_nan=False
        )
    )
