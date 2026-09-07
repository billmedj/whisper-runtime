"""Offline supplemental lexical/coverage audit; never changes qualification gates.

Usage: python tools/summarize_live_qualification.py ARTIFACT_DIRECTORY
Requires the final live-v01.json, original preflight and each planned event log.
Short-only completion is audited separately and never implies full qualification.
No inference, network, reference-based routing or guessed per-arm word alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

# ruff: noqa: E402 -- This checkout-only tool resolves runtime imports below.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from infra import modal_live_qualification as registration
from infra.modal_stream_boundary_diagnostic import _word_difference
from whisper_runtime.adapters.native_stream import StreamEventKind, TranscriptEvent
from whisper_runtime.captions import CommittedTranscript
from whisper_runtime.remote_transport import LIVE_PROTOCOL, ReplayError, _check_done


def require(value, message):
    if not value:
        raise ValueError(message)


def _object(pairs):
    require(len(dict(pairs)) == len(pairs), "duplicate JSON object key")
    return dict(pairs)


def _json(raw):
    return json.loads(
        raw,
        object_pairs_hook=_object,
        parse_constant=lambda value: require(False, "nonfinite JSON"),
    )


def _read(path):
    require(
        0 < path.stat().st_size <= registration.MAX_EVENT_RECEIPT_BYTES,
        f"missing, empty or oversized artifact: {path.name}",
    )
    return path.read_bytes()


def _intervals(plan, stage):
    cycle, cycles, count = plan["cycle_samples"], stage["cycles"], stage["sample_count"]
    require(
        all(type(v) is int and v > 0 for v in (cycle, cycles, count))
        and cycles <= 1000
        and count <= registration.LONG_SECONDS * 16000
        and cycles == count // cycle,
        "invalid whole-cycle recipe",
    )
    require(
        plan["long_tail"] == "whole cycles then digital silence",
        "unsupported tail reference contract",
    )
    arms = sorted(plan["arms"].items(), key=lambda item: item[1]["start_sample"])
    require(arms and plan["smoke_reference"].strip(), "missing recipe reference")
    require(
        plan["smoke_reference"] == " ".join(arm["reference"] for _, arm in arms),
        "whole-cycle reference differs from preregistered arms",
    )
    spans = []
    for index in range(cycles):
        offset, head = index * cycle, 0
        for name, arm in arms:
            start, end = arm["start_sample"], arm["end_sample"]
            require(
                type(start) is int
                and type(end) is int
                and head <= start < end <= cycle,
                "invalid or overlapping recipe arms",
            )
            if head < start:
                spans.append((offset + head, offset + start, index + 1, "silence"))
            spans.append((offset + start, offset + end, index + 1, name))
            head = end
        if head < cycle:
            spans.append((offset + head, offset + cycle, index + 1, "silence"))
    if cycles * cycle < count:
        spans.append((cycles * cycle, count, None, "tail_silence"))
    return spans


def _stage(directory, phase, report, plan, digest):
    stage, result = plan["stages"][phase], report["result"]
    require(
        report["phase"] == phase
        and result["status"] == "completed"
        and result["protocol"] == LIVE_PROTOCOL
        and report["passed"] is True,
        f"{phase}: incomplete registered stage",
    )
    require(
        all(result[key] == stage[key] for key in ("sample_count", "sha256")),
        f"{phase}: input identity differs",
    )
    done, terminal = result["server_done"], result["qualification_receipt"]
    _check_done(done, result)
    require(
        registration.stage_valid(done, stage, expected_digest=digest),
        f"{phase}: registered terminal gate failed",
    )
    require(
        terminal["status"] == "available"
        and terminal["record"]["done"] == done
        and terminal["record"]["phase"] == phase
        and terminal["record"]["source_digest"] == digest
        and terminal["record"]["instance_id"] == done["metadata"]["instance_id"],
        f"{phase}: terminal receipt differs",
    )
    receipt = report["event_receipt"]
    require(receipt["path"] == f"live-v01-{phase}.events.jsonl", "unexpected log path")
    raw = _read(directory / receipt["path"])
    require(
        len(raw) == receipt["size_bytes"]
        and hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        and raw.endswith(b"\n"),
        f"{phase}: incomplete or changed event log",
    )
    projection, seen, commits, elapsed = CommittedTranscript(), set(), [], -1
    spans = _intervals(plan, stage)
    for line in raw.splitlines():
        entry = _json(line)
        require(set(entry) == {"event", "client_elapsed_ns"}, "invalid event record")
        arrived = entry["client_elapsed_ns"]
        require(
            type(arrived) is int and arrived >= max(0, elapsed),
            "invalid event arrival order",
        )
        elapsed = arrived
        event = TranscriptEvent(
            **{**entry["event"], "kind": StreamEventKind(entry["event"]["kind"])}
        )
        require(
            event.end_sample is None or event.end_sample <= stage["sample_count"],
            "event extends beyond registered input",
        )
        if event.kind is StreamEventKind.COMMIT:
            source_range = event.start_sample, event.end_sample
            require(
                source_range not in seen, "repeated identical committed source range"
            )
            seen.add(source_range)
        caption = projection.accept(event)
        if caption is not None:
            regions = [
                s
                for s in spans
                if s[0] < caption.end_sample and s[1] > caption.start_sample
            ]
            attribution = (
                {"cycle": regions[0][2], "arm": regions[0][3]}
                if len(regions) == 1
                else None
            )
            commits.append(
                dict(
                    sequence_number=event.sequence_number,
                    start_sample=caption.start_sample,
                    end_sample=caption.end_sample,
                    text=caption.text,
                    attribution=attribution,
                    attribution_reason=None
                    if attribution
                    else "crosses_recipe_boundary",
                    arrival_minus_source_end_ms=(arrived - caption.end_sample * 62500)
                    / 1e6,
                )
            )
    require(
        projection.final and projection.head == stage["sample_count"],
        f"{phase}: no FINAL or incomplete committed coverage",
    )
    metrics = result["client_metrics"]
    require(
        projection.sequence == metrics["received_events"]
        and len(commits) == report["committed_events"]
        and metrics["sent_samples"] == stage["sample_count"]
        and metrics["sender_finished"] is True
        and metrics["eof_sent"] is True,
        f"{phase}: event or source counts differ",
    )
    text = projection.render("txt").rstrip("\n")
    reference = " ".join([plan["smoke_reference"]] * stage["cycles"])
    delays = sorted(c["arrival_minus_source_end_ms"] for c in commits)
    return dict(
        status="complete",
        input=stage,
        event_receipt=receipt,
        terminal_receipt_sha256_reported=terminal["sha256"],
        final=True,
        committed_samples=projection.head,
        event_count=projection.sequence,
        committed_text=text,
        whole_stage_score=_word_difference(text, reference),
        commits=commits,
        unattributable_commits=sum(c["attribution"] is None for c in commits),
        arrival_minus_source_end_ms=dict(
            min=min(delays),
            median=statistics.median(delays),
            p95=delays[math.ceil(0.95 * len(delays)) - 1],
            max=max(delays),
        ),
    )


def summarize(directory):
    directory = Path(directory)
    final_raw, preflight_raw = (
        _read(directory / "live-v01.json"),
        _read(directory / "live-v01-preflight.json"),
    )
    final, preflight = _json(final_raw), _json(preflight_raw)
    require(
        preflight == {"status": "local-preflight-passed", **final["specification"]},
        "final specification differs from original preflight",
    )
    source, plan = preflight["source"], preflight["input"]
    require(
        registration.prior._canonical(source["files"]) == source["digest"],
        "recorded source inventory digest differs",
    )
    qualification = final["qualification"]
    short_only = plan.get("short_only", False)
    require(type(short_only) is bool, "invalid short-only mode")
    if short_only:
        require(
            final["status"] == qualification["status"] == "completed-short"
            and final["qualified"] is False
            and qualification["long"] is None
            and qualification["long_gate_passed"] is False
            and qualification.get("same_worker_model") is not True
            and set(plan["stages"]) == {"smoke"},
            "short-only report claims a long run or full qualification",
        )
        config = plan["stages"]["smoke"]["stream_config"]
        overrides = plan["config_overrides"]
        require(
            isinstance(config, dict)
            and config
            and isinstance(overrides, dict)
            and set(overrides) <= {"left_context_ms", "word_context_limit_ms"}
            and all(config.get(key) == value for key, value in overrides.items())
            and qualification["smoke"]["result"]["server_done"]["metadata"].get(
                "config_overrides"
            )
            == overrides,
            "short-only configuration differs from preflight",
        )
    else:
        require(
            final["status"] == qualification["status"] == "completed"
            and qualification["long_gate_passed"] is True
            and qualification["same_worker_model"] is True,
            "qualification is incomplete or failed",
        )
    stages = {
        phase: _stage(directory, phase, qualification[phase], plan, source["digest"])
        for phase in (("smoke",) if short_only else ("smoke", "long"))
    }
    owners = [qualification[p]["result"]["server_done"]["metadata"] for p in stages]
    require(
        owners[0]["instance_id"]
        and all(owner["instance_id"] == owners[0]["instance_id"] for owner in owners)
        and [o["session_number"] for o in owners] == ([1] if short_only else [1, 2]),
        "worker/session identity differs",
    )
    return dict(
        status="complete-short" if short_only else "complete",
        scope="short-only supplemental audit; not full V0.1 qualification"
        if short_only
        else "supplemental lexical coverage audit; no new qualification gate",
        registered_status=final["status"],
        registered_qualified=final["qualified"],
        cleanup=final["cleanup"],
        source=source,
        input_recipe=plan,
        stages=stages,
        final_sha256=hashlib.sha256(final_raw).hexdigest(),
        preflight_sha256=hashlib.sha256(preflight_raw).hexdigest(),
        identity_scope="input hashes cross-checked against DONE; PCM not rehashed; terminal raw hash preserved, not reverified",
        timing_scope="client post-READY event arrival minus committed-source-end; includes scheduling/transport; NOT word latency",
        attribution_scope="source-span containment only; crossing commits unattributable; no per-arm lexical score",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        report = summarize(args.directory)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        ReplayError,
    ) as error:
        print(json.dumps(dict(status="incomplete_or_invalid", reason=str(error))))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
