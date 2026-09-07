"""Prepare, then explicitly authorize one native T4 endurance attempt.

This is source-paced SDK inference, NOT a live-v2/WebSocket qualification.
Imports and --preflight are local-only. No retry, deployment or model download.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from infra import modal_live_qualification as live

ROOT, prior = live.ROOT, live.prior
PRODUCER = "infra/modal_release_soak.py"
TEST = "tools/test_modal_release_soak.py"
PLAN_DOC = "docs/research/2026-09-07-release-soak-plan.md"
CAPACITY_PLAN_DOC = "docs/research/2026-09-07-low-latency-qualification-plan.md"
PROFILE = "conservative-v1"
CAPACITY_PROFILE = "low-latency-v2"
CAPACITY_PROFILES = {"low-latency-v1": 0, "low-latency-v2": 2000}
CAPACITY_MODE = "provider-endurance-v1"
FIRST_COMMIT_SECONDS = 8.0
PHASES = ("smoke", "hour-1", "hour-2", "hour-3", "hour-4")
OUTER_SECONDS = 14800
MIB = 1024 * 1024
MEMORY_POLICY = {
    "checkpoint_interval_seconds": 300,
    "terminal_allocated_delta_bytes": 0,
    "reserved_growth_bytes": 256 * MIB,
    "reserved_ceiling_bytes": 2048 * MIB,
    "rss_growth_bytes": 512 * MIB,
    "rss_ceiling_bytes": 3584 * MIB,
    "baseline": "post-smoke, closed stream, GC and CUDA fence; never reset",
    "empty_cache": False,
}
CAPACITY_MEMORY_POLICY = {
    key: value for key, value in MEMORY_POLICY.items() if key != "rss_ceiling_bytes"
}
CAPACITY_MEMORY_POLICY.update(
    rss_scope="guest-reported current RSS growth sentinel, not physical host RAM",
    rss_required=True,
    host_physical_memory_verified=False,
)
MAX_PHASE_EVENTS = 100000
MAX_PHASE_LOG_BYTES = 32 * MIB
MAX_MESSAGE_BYTES = 512 * 1024


class SoakError(ValueError):
    """A fixed harness reason, never an interpolated native/provider message."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(value, code):
    if not value:
        raise SoakError(code)


def failure_code(error, fallback):
    return error.code if isinstance(error, SoakError) else fallback


def encode(value):
    raw = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    require(len(raw) <= MAX_MESSAGE_BYTES, "message_limit")
    return raw


class SourceClock:
    """Bounded producer observations, not a measurement of push time alone.

    Frame sequences are zero-based; times are seconds relative to the unchanged
    source origin, also recorded on the same worker's monotonic clock as native
    operation observations (not a cross-worker or physical-host time reference).
    The yield/resume gap includes push, caller work and scheduling.
    Only the input producer mutates this object; read it after drive_stream joins.
    """

    def __init__(self, origin, *, now=None):
        self.now = now or time.monotonic
        require(math.isfinite(origin), "invalid_source_clock")
        self.origin, self.yield_at, self.yield_frame = origin, None, None
        self.state = {
            "schema_version": "source-clock-v2",
            "clock": "time.monotonic",
            "origin_monotonic_seconds": origin,
            "max_wake_lateness_seconds": 0.0,
            "max_wake_frame": None,
            "max_yield_resume_seconds": 0.0,
            "max_yield_resume_interval": None,
            "offered_frames": 0,
            "offered_samples": 0,
            "last_yield_sequence": None,
            "last_yield_end_sample": None,
            "last_attempt": None,
            "failed_frame": None,
        }

    def timestamp(self):
        value = self.now()
        require(math.isfinite(value) and value >= self.origin, "invalid_source_clock")
        return value

    def elapsed(self):
        return self.timestamp() - self.origin

    def wait(self, sequence, start_sample, end_sample, cancelled):
        due = self.origin + end_sample / 16000
        require(not cancelled.wait(max(0, due - self.timestamp())), "cancelled")
        wake = self.timestamp()
        late = wake - due
        attempt = {
            "sequence": sequence,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "due_elapsed_seconds": due - self.origin,
            "wake_elapsed_seconds": wake - self.origin,
            "wake_lateness_seconds": late,
        }
        self.state["last_attempt"] = attempt
        if late >= 0 and (
            self.state["max_wake_frame"] is None
            or late > self.state["max_wake_lateness_seconds"]
        ):
            self.state["max_wake_lateness_seconds"] = late
            self.state["max_wake_frame"] = dict(attempt)
        if late > 0.25:
            self.state["failed_frame"] = dict(attempt)
            raise SoakError("source_late")

    def offered(self, sequence, end_sample):
        self.yield_at = self.elapsed()
        self.yield_frame = {
            "sequence": sequence,
            "start_sample": self.state["offered_samples"],
            "end_sample": end_sample,
        }
        self.state.update(
            offered_frames=sequence + 1,
            offered_samples=end_sample,
            last_yield_sequence=sequence,
            last_yield_end_sample=end_sample,
        )

    def resumed(self):
        if self.yield_at is not None:
            resumed = self.elapsed()
            gap = resumed - self.yield_at
            require(gap >= 0, "invalid_source_clock")
            if (
                self.state["max_yield_resume_interval"] is None
                or gap > self.state["max_yield_resume_seconds"]
            ):
                self.state["max_yield_resume_seconds"] = gap
                self.state["max_yield_resume_interval"] = {
                    **self.yield_frame,
                    "yield_elapsed_seconds": self.yield_at,
                    "resume_elapsed_seconds": resumed,
                    "gap_seconds": gap,
                }
            self.yield_at = self.yield_frame = None

    def receipt(self):
        # A JSON copy avoids later mutation and enforces finite, bounded output.
        return json.loads(encode(self.state))


def source_prefix_receipt(clock, digest, stream, terminal=None):
    """Offered hash is not an accepted hash; retain native admission separately."""
    if clock is None:
        return None
    result = {
        "offered_samples": clock.state["offered_samples"],
        "offered_sha256": digest.hexdigest(),
        "native_accepted_samples": None,
    }
    if stream is not None or terminal is not None:
        try:
            accepted = (
                stream.metrics.accepted_samples
                if stream is not None
                else terminal["metrics"]["accepted_samples"]
            )
            require(
                type(accepted) is int and 0 <= accepted <= result["offered_samples"],
                "invalid_accepted_count",
            )
            result["native_accepted_samples"] = accepted
        except BaseException as error:
            result["accepted_count_error_type"] = type(error).__name__
    return result


def profile_contract(profile_name=PROFILE):
    from whisper_runtime.native_setup import CLI_STREAM_CONFIG
    from whisper_runtime.profiles import DEFAULT_PROFILE, get_profile

    require(profile_name in (PROFILE, *CAPACITY_PROFILES), "unregistered_profile")
    profile = get_profile(profile_name)
    if profile_name != PROFILE:
        require(
            profile.stream_config.previous_holdback_ms
            == CAPACITY_PROFILES[profile_name],
            "stale_previous_holdback",
        )
        return asdict(profile)
    require(
        DEFAULT_PROFILE == PROFILE
        and profile.name == PROFILE
        and profile.native_profile_id == "tiny.en/cli-fp32-v1"
        and profile.experimental is False
        and profile.reuse_alignment_features is False
        and profile.stream_config == CLI_STREAM_CONFIG,
        "stale_or_nonconservative_profile",
    )
    return asdict(profile)


def select_profile(profile=None, *, capacity_limit=False, allow_standard=False):
    """Resolve only prospective defaults; receipts always retain the selected name."""
    require(type(capacity_limit) is bool, "invalid_capacity_mode")
    if profile is None:
        profile = CAPACITY_PROFILE if capacity_limit else PROFILE
    require(profile in (PROFILE, *CAPACITY_PROFILES), "unregistered_profile")
    require(
        (profile in CAPACITY_PROFILES and capacity_limit)
        or (profile == PROFILE and (not capacity_limit or allow_standard)),
        "profile_requires_registered_capacity_mode",
    )
    return profile


def config_comparison(config):
    """A legacy absent field means zero, never the v2 value; leave receipts intact."""
    if not isinstance(config, dict):
        return config
    return {"previous_holdback_ms": 0, **config}


def profile_comparison(profile):
    if not isinstance(profile, dict):
        return profile
    return {**profile, "stream_config": config_comparison(profile.get("stream_config"))}


def plan_profile(plan, *, capacity_limit=False):
    name = select_profile(plan["profile"]["name"], capacity_limit=capacity_limit)
    require(
        profile_comparison(plan["profile"])
        == profile_comparison(profile_contract(name)),
        "stale_profile",
    )
    return name


def frames(seed, samples):
    """Bounded one-hour source: whole speech cycles then digital silence."""
    require(isinstance(seed, bytes) and seed and len(seed) % 2 == 0, "invalid_seed")
    require(type(samples) is int and 0 < samples <= 3600 * 16000, "input_limit")
    total = samples * 2
    repeated = total // len(seed) * len(seed)
    for offset in range(0, total, 640):
        count = min(640, total - offset)
        take = min(count, max(0, repeated - offset))
        start = offset % len(seed)
        first = seed[start : start + take]
        missing = take - len(first)
        yield (
            first
            + (seed * ((missing + len(seed) - 1) // len(seed)))[:missing]
            + bytes(count - take)
        )


def source_hash(seed, samples):
    digest = hashlib.sha256()
    for frame in frames(seed, samples):
        digest.update(frame)
    return digest.hexdigest()


def resource_contract():
    return {
        "mode": CAPACITY_MODE,
        "sdk_version": prior.SDK_VERSION,
        "gpu": "T4",
        "cpu_request": 2,
        "cpu_limit": 0,
        "memory_request_mib": 4096,
        "memory_limit_mib": 4096,
        "maximum_containers": 1,
        "timeout_seconds": OUTER_SECONDS,
        "guest_attests_provider_enforcement": False,
    }


def capacity_mode(plan):
    require(
        plan.get("mode", "source-paced-native-sdk-not-websocket")
        in ("source-paced-native-sdk-not-websocket", CAPACITY_MODE),
        "unknown_endurance_mode",
    )
    capacity = plan.get("mode") == CAPACITY_MODE
    require(plan.get("capacity_limit", False) is capacity, "capacity_mode_mismatch")
    if capacity:
        plan_profile(plan, capacity_limit=True)
        require(
            plan.get("memory_policy") == CAPACITY_MEMORY_POLICY
            and plan.get("submitted_resource_contract") == resource_contract()
            and plan.get("first_nonempty_commit_max_seconds") == FIRST_COMMIT_SECONDS,
            "capacity_contract_mismatch",
        )
    return capacity


def native_profile_valid(metadata, profile):
    setup = importlib.import_module("whisper_runtime.native_setup")
    return (
        profile_comparison(metadata.get("execution_profile"))
        == profile_comparison(profile)
        and metadata.get("native_profile_id") == profile["native_profile_id"]
        and metadata.get("reuse_alignment_features")
        is profile["reuse_alignment_features"]
        and metadata.get("backend_tree")
        == (
            setup.BACKEND_REUSE_TREE
            if profile["reuse_alignment_features"]
            else setup.BACKEND_TREE
        )
        and isinstance(metadata.get("backend_revision"), str)
        and len(metadata["backend_revision"]) == 40
        and all(c in "0123456789abcdef" for c in metadata["backend_revision"])
    )


def input_plan(root=ROOT, *, profile=None, capacity_limit=False):
    profile = select_profile(profile, capacity_limit=capacity_limit)
    seed, original = live.input_plan(root, short_only=True)
    require(len(seed) // 2 == 744800, "registered_smoke_length_differs")
    hour_hash = source_hash(seed, 3600 * 16000)
    return seed, {
        "schema_version": CAPACITY_MODE if capacity_limit else "release-soak-native-v1",
        "mode": CAPACITY_MODE
        if capacity_limit
        else "source-paced-native-sdk-not-websocket",
        "profile": profile_contract(profile),
        "seed_sha256": prior._sha(seed),
        "seed_samples": len(seed) // 2,
        "recipe": original["recipe"],
        "input_allowlist": {
            key: original[key]
            for key in (
                "main_case",
                "main_sha256",
                "noisy_source_case",
                "noisy_prefix_samples",
                "noisy_prefix_sha256",
            )
        },
        "smoke_reference": original["smoke_reference"],
        "max_smoke_word_edit_rate": original["max_smoke_word_edit_rate"],
        "quality_calibration": original["quality_calibration"],
        "memory_policy": CAPACITY_MEMORY_POLICY if capacity_limit else MEMORY_POLICY,
        "phases": [
            {
                "phase": name,
                "sample_count": len(seed) // 2 if name == "smoke" else 3600 * 16000,
                "sha256": prior._sha(seed) if name == "smoke" else hour_hash,
                "timeout_seconds": 190 if name == "smoke" else 3690,
            }
            for name in PHASES
        ],
        "tail": "whole 46.55-second cycles then silence, independently per hour",
        "max_source_lateness_seconds": 0.25,
        "drain_timeout_seconds": 30,
        "max_phase_events": MAX_PHASE_EVENTS,
        "max_phase_log_bytes": MAX_PHASE_LOG_BYTES,
        "outer_seconds": OUTER_SECONDS,
        **(
            {
                "capacity_limit": True,
                "submitted_resource_contract": resource_contract(),
                "first_nonempty_commit_max_seconds": FIRST_COMMIT_SECONDS,
            }
            if capacity_limit
            else {}
        ),
    }


def snapshot(root=ROOT, *, profile=None, capacity_limit=False):
    profile = select_profile(profile, capacity_limit=capacity_limit)
    existing = live.snapshot(root)
    names = {item["path"] for item in existing["files"]} | {PRODUCER}
    if profile_contract(profile)["reuse_alignment_features"]:
        names.update(
            {
                live.REUSE_PATCH,
                (Path(live.REUSE_PATCH).parent / "SHA256SUMS").as_posix(),
            }
        )
    review_names = {name for name in names if name.startswith("tools/test_")} | {
        TEST,
        PLAN_DOC,
    }
    if capacity_limit or profile != PROFILE:
        review_names.add(CAPACITY_PLAN_DOC)
    names -= review_names
    files = [
        {"path": name, "size_bytes": len(data), "sha256": prior._sha(data)}
        for name in sorted(names)
        for data in [(Path(root) / name).read_bytes()]
    ]
    review = [
        {"path": name, "size_bytes": len(data), "sha256": prior._sha(data)}
        for name in sorted(review_names)
        for data in [(Path(root) / name).read_bytes()]
    ]
    return {
        **existing,
        "files": files,
        "digest": prior._canonical(files),
        "review_files_not_uploaded": review,
        "review_digest": prior._canonical(review),
    }


def budget_plan(*, capacity_limit=False):
    price = live.PRICES
    estimate = (
        OUTER_SECONDS
        * (
            price["t4_per_second"]
            + 2 * price["cpu_core_per_second"]
            + 4 * price["gib_per_second"]
        )
        * price["regional_multiplier_ceiling"]
    )
    return {
        "gpu": "T4",
        "cpu": 2,
        "memory_mib": 4096,
        "cloud": "aws",
        "region": "us-west",
        "outer_seconds": OUTER_SECONDS,
        "startup_timeout_seconds": 180,
        "maximum_gpu_containers": 1,
        "minimum_gpu_containers": 0,
        "buffer_containers": 0,
        "configured_retries": 0,
        "sdk_retry_policy": None,
        "generator_retries_supported": False,
        "persistent_deployment": False,
        "planning_compute_usd": round(estimate, 6),
        "prices": price,
        "prices_source": "https://modal.com/pricing",
        "prices_checked": "2026-09-07",
        "physical_spending_cap": False,
        "exclusions": "build/startup/storage/egress/taxes/provider crash rescheduling; not a spending cap",
        **(
            {"mode": CAPACITY_MODE, "submitted_resource_contract": resource_contract()}
            if capacity_limit
            else {}
        ),
    }


def memory_sample():
    """Current Linux RSS, not ru_maxrss; CUDA peak counters are observational."""
    torch = importlib.import_module("torch")
    torch.cuda.synchronize(0)
    rss_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return {
        "rss_bytes": rss_pages * os.sysconf("SC_PAGE_SIZE"),
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def memory_valid(sample, baseline, *, terminal, capacity_limit=False):
    fields = (
        "rss_bytes",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    if not all(type(sample.get(key)) is int and sample[key] >= 0 for key in fields):
        return False
    if (
        not capacity_limit and sample["rss_bytes"] > MEMORY_POLICY["rss_ceiling_bytes"]
    ) or sample["reserved_bytes"] > MEMORY_POLICY["reserved_ceiling_bytes"]:
        return False
    if baseline is None:
        return True
    return (
        sample["rss_bytes"] <= baseline["rss_bytes"] + MEMORY_POLICY["rss_growth_bytes"]
        and sample["reserved_bytes"]
        <= baseline["reserved_bytes"] + MEMORY_POLICY["reserved_growth_bytes"]
        and (not terminal or sample["allocated_bytes"] == baseline["allocated_bytes"])
    )


def stage_valid(done, stage, expected, profile, instance, session):
    metadata = done.get("metadata", {})
    return (
        live.stage_valid(done, stage, expected_digest=expected["digest"])
        and metadata.get("instance_id") == instance
        and metadata.get("session_number") == session
        and metadata.get("model_initial_sha256")
        == metadata.get("model_final_sha256")
        == importlib.import_module("whisper_runtime.native_setup").MODEL_FINGERPRINT
        and config_comparison(metadata.get("stream_config"))
        == config_comparison(profile["stream_config"])
        and metadata.get("config_overrides") == {}
        and (
            profile.get("name") not in CAPACITY_PROFILES
            or native_profile_valid(metadata, profile)
        )
        and profile_comparison(done.get("profile")) == profile_comparison(profile)
        and done.get("final_count") == 1
        and done.get("last_event_kind") == "final"
        and done.get("eof_chunks") == (stage["sample_count"] + 319) // 320
    )


def run_suite(owner, seed, plan, expected, *, memory=memory_sample, score=None):
    """One owner thread across all phases; bounded event mailbox to the caller.

    Injection points serve CPU-only scripted tests. The paid entry rejects any plan
    other than its frozen native registration; it exposes no short-duration flags.
    """
    from whisper_runtime.captions import CommittedTranscript
    from whisper_runtime.cli import drive_stream

    capacity_limit = capacity_mode(plan)
    plan_profile(plan, capacity_limit=capacity_limit)
    require(prior._sha(seed) == plan["seed_sha256"], "seed_identity")
    mailbox, cancelled = queue.Queue(maxsize=64), threading.Event()
    completed = threading.Event()
    started = time.monotonic()
    phase_deadline = [started + plan["outer_seconds"]]

    def emit(value):
        require(not cancelled.is_set(), "cancelled")
        encode(value)
        mailbox.put(value, timeout=0.25)

    def work():
        baseline = None
        stream = finalize = stage = terminal = None
        source_clock = None
        first_commit_ns = None
        try:
            if capacity_limit:
                emit(
                    {
                        "type": "resource_contract",
                        "instance_id": owner.instance_id,
                        "submitted_resource_contract": resource_contract(),
                    }
                )
            for number, stage in enumerate(plan["phases"], 1):
                terminal = None
                source_clock = None
                first_commit_ns = None
                phase_deadline[0] = time.monotonic() + stage["timeout_seconds"]
                require(not cancelled.is_set(), "cancelled")
                phase = stage["phase"]
                emit(
                    {
                        "type": "phase_started",
                        "phase": phase,
                        "instance_id": owner.instance_id,
                        "session_number": number,
                    }
                )
                stream, finalize = owner.factory(phase)
                require(
                    config_comparison(asdict(stream.config))
                    == config_comparison(plan["profile"]["stream_config"]),
                    "stale_stream_config",
                )
                phase_cancel = threading.Event()
                transcript = CommittedTranscript()
                digest = hashlib.sha256()
                counters = {
                    "samples": 0,
                    "chunks": 0,
                    "events": 0,
                    "final": 0,
                    "last": None,
                    "eof": False,
                    "event_bytes": 0,
                }
                smoke_words = []
                origin = time.monotonic()
                source_clock = SourceClock(origin)
                checkpoint_at = origin + MEMORY_POLICY["checkpoint_interval_seconds"]

                def source():
                    for frame in frames(seed, stage["sample_count"]):
                        end = counters["samples"] + len(frame) // 2
                        source_clock.wait(
                            counters["chunks"], counters["samples"], end, cancelled
                        )
                        digest.update(frame)
                        source_clock.offered(counters["chunks"], end)
                        counters["samples"], counters["chunks"] = (
                            end,
                            counters["chunks"] + 1,
                        )
                        try:
                            yield frame
                        finally:
                            source_clock.resumed()
                    counters["eof"] = True

                def event(value):
                    nonlocal checkpoint_at, first_commit_ns
                    event_ns = int((time.monotonic() - origin) * 1e9)
                    caption = transcript.accept(value)
                    if (
                        caption is not None
                        and caption.text.strip()
                        and first_commit_ns is None
                    ):
                        first_commit_ns = event_ns
                    # Persist every event outside the worker. Retain no caption
                    # history; the projection's segment IDs remain event-capped.
                    transcript.captions.clear()
                    if caption is not None and phase == "smoke":
                        smoke_words.append(caption.text)
                        require(sum(map(len, smoke_words)) <= 65536, "smoke_text_limit")
                    counters["events"] += 1
                    counters["last"] = value.kind.value
                    counters["final"] += value.kind.value == "final"
                    message = {
                        "type": "event",
                        "phase": phase,
                        "event": asdict(value),
                        "worker_elapsed_ns": event_ns,
                    }
                    counters["event_bytes"] += len(encode(message))
                    require(
                        counters["events"] <= MAX_PHASE_EVENTS
                        and counters["event_bytes"] <= MAX_PHASE_LOG_BYTES,
                        "event_limit",
                    )
                    emit(message)
                    if (
                        capacity_limit
                        and phase == "smoke"
                        and event_ns > FIRST_COMMIT_SECONDS * 1e9
                    ):
                        require(
                            first_commit_ns is not None
                            and first_commit_ns <= FIRST_COMMIT_SECONDS * 1e9,
                            "first_commit_gate",
                        )
                    now = time.monotonic()
                    require(now - origin <= stage["timeout_seconds"], "phase_timeout")
                    require(now - started <= plan["outer_seconds"], "outer_timeout")
                    if now >= checkpoint_at:
                        sample = memory()
                        emit(
                            {
                                "type": "memory",
                                "phase": phase,
                                "terminal": False,
                                "elapsed_seconds": now - origin,
                                "sample": sample,
                            }
                        )
                        require(
                            memory_valid(
                                sample,
                                baseline,
                                terminal=False,
                                capacity_limit=capacity_limit,
                            ),
                            "memory_gate",
                        )
                        checkpoint_at = (
                            now + MEMORY_POLICY["checkpoint_interval_seconds"]
                        )

                # The shipped CLI's producer/owner, backpressure, EOF coverage,
                # cooperative cancellation and native-close ordering are reused.
                drive_stream(
                    stream,
                    source(),
                    on_event=event,
                    cancel=phase_cancel,
                    live=True,
                    drain_timeout=30,
                )
                metadata = finalize()
                terminal = {
                    "type": "phase_done",
                    "phase": phase,
                    "status": "completed",
                    "source_eof_received": counters["eof"],
                    "profile": plan["profile"],
                    "metrics": {
                        **asdict(stream.metrics),
                        "accepted_sha256": digest.hexdigest(),
                    },
                    "metadata": metadata,
                    "final_count": counters["final"],
                    "last_event_kind": counters["last"],
                    "eof_chunks": counters["chunks"],
                    "event_count": counters["events"],
                    "offered_samples": counters["samples"],
                    "elapsed_seconds": time.monotonic() - origin,
                    "source_clock": source_clock.receipt(),
                    **(
                        {
                            "first_nonempty_commit_seconds": None
                            if first_commit_ns is None
                            else first_commit_ns / 1e9
                        }
                        if capacity_limit
                        else {}
                    ),
                }
                if capacity_limit:
                    require(
                        native_profile_valid(metadata, plan["profile"]),
                        "native_profile_gate",
                    )
                    if phase == "smoke":
                        require(
                            first_commit_ns is not None
                            and first_commit_ns <= FIRST_COMMIT_SECONDS * 1e9,
                            "first_commit_gate",
                        )
                require(
                    transcript.final
                    and transcript.head == stage["sample_count"]
                    and counters["samples"] == stage["sample_count"],
                    "coverage_gate",
                )
                require(
                    stage_valid(
                        terminal,
                        stage,
                        expected,
                        plan["profile"],
                        owner.instance_id,
                        number,
                    ),
                    "stage_gate",
                )
                if phase == "smoke":
                    difference = score or prior._corpus().b._word_difference
                    terminal["smoke_score"] = difference(
                        " ".join(smoke_words), plan["smoke_reference"]
                    )
                    rate = terminal["smoke_score"].get("word_edit_rate")
                    require(
                        isinstance(rate, (int, float))
                        and 0 <= rate <= plan["max_smoke_word_edit_rate"],
                        "smoke_quality_gate",
                    )
                # Remove closures before sampling; this keeps the measurement
                # boundary identical between warmup and every one-hour session.
                stream = finalize = None
                transcript = None
                gc.collect()
                sample = memory()
                terminal["memory"] = sample
                require(
                    memory_valid(
                        sample, baseline, terminal=True, capacity_limit=capacity_limit
                    ),
                    "memory_gate",
                )
                if baseline is None:
                    baseline = dict(sample)
                terminal["memory_baseline"] = baseline
                emit(terminal)
            emit(
                {
                    "type": "suite_done",
                    "status": "completed",
                    "instance_id": owner.instance_id,
                    "model_loads": owner.loads,
                    "sessions": owner.sessions,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
        except BaseException as error:
            # Native exceptions can contain paths or credentials; type only.
            try:
                failure = {
                    "type": "phase_failed",
                    "phase": stage["phase"] if stage else None,
                    "status": "failed",
                    "error_code": failure_code(error, "native_soak_failed"),
                    "error_type": type(error).__name__,
                    "instance_id": owner.instance_id,
                    "initialization_stage": str(
                        getattr(owner, "initialization_stage", "scripted")
                    )[:64],
                    "terminal_before_gate": terminal,
                    "source_clock": source_clock.receipt() if source_clock else None,
                    "source_prefix": source_prefix_receipt(
                        source_clock, digest if source_clock else None, stream, terminal
                    ),
                    "cleanup_verified": False,
                    **(
                        {
                            "first_nonempty_commit_seconds": None
                            if first_commit_ns is None
                            else first_commit_ns / 1e9
                        }
                        if capacity_limit
                        else {}
                    ),
                }
                if stream is not None:
                    try:
                        stream.close()
                        failure["metadata"] = finalize()
                        failure["metrics"] = asdict(stream.metrics)
                        failure["cleanup_verified"] = True
                    except BaseException as cleanup_error:
                        failure["cleanup_error_type"] = type(cleanup_error).__name__
                    stream = finalize = None
                elif terminal is not None:
                    failure["cleanup_verified"] = terminal["metadata"][
                        "capacity_restored"
                    ]
                gc.collect()
                try:
                    failure["memory"] = memory()
                except BaseException as memory_error:
                    failure["memory_error_type"] = type(memory_error).__name__
                encode(failure)
                mailbox.put(failure, timeout=0.25)
                mailbox.put(
                    {
                        "type": "suite_done",
                        "status": "failed",
                        "error_code": failure_code(error, "native_soak_failed"),
                        "error_type": type(error).__name__,
                        "instance_id": owner.instance_id,
                    },
                    timeout=0.25,
                )
            except queue.Full:
                pass
        finally:
            completed.set()

    thread = threading.Thread(
        target=work, name="release-soak-native-owner", daemon=True
    )
    thread.start()
    try:
        while not completed.is_set() or not mailbox.empty():
            require(
                time.monotonic() - started <= plan["outer_seconds"], "outer_timeout"
            )
            require(time.monotonic() <= phase_deadline[0], "phase_timeout")
            try:
                yield mailbox.get(timeout=0.1)
            except queue.Empty:
                continue
    finally:
        cancelled.set()
        thread.join(timeout=5)
        require(not thread.is_alive(), "native_owner_not_stopped")


def resources(modal, expected, plan, root=ROOT):
    """Called only after exact preflight matching and explicit paid authorization."""
    capacity_limit = capacity_mode(plan)
    corpus = prior._corpus()
    qualification = corpus._helper("infra.modal_native_cuda_qualification")
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("ca-certificates", "ffmpeg", "git")
        .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
        .uv_pip_install(*qualification.DIRECT_IMAGE_PACKAGES)
        .run_commands(qualification._build_command(corpus.BASE_COMMIT))
    )
    for item in expected["files"]:
        image = image.add_local_file(
            Path(root) / item["path"],
            (live.REMOTE_ROOT / item["path"]).as_posix(),
            copy=True,
        )
    if capacity_limit:
        image = live.prepare_profile_image(
            image, plan["profile"]["name"], expected, root
        )
    image = image.env(
        {
            "PYTHONPATH": "/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "WHISPER_MODAL_ENABLE_WORD_CORPUS": "0",
            "WHISPER_MODAL_ENABLE_REMOTE_RESOURCES": "0",
        }
    )
    volume = modal.Volume.from_name(corpus.b.MODEL_CACHE_NAME, create_if_missing=False)
    app = modal.App("wr-release-soak-" + uuid.uuid4().hex[:12])

    @app.function(
        image=image,
        serialized=True,
        cpu=2,
        memory=(4096, 4096) if capacity_limit else 4096,
        gpu="T4",
        cloud="aws",
        region="us-west",
        timeout=OUTER_SECONDS,
        startup_timeout=180,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=2,
        # Modal 1.5.5 rejects even retries=0 for generators. Omit the
        # retry policy; generator application retries are not supported.
        single_use_containers=True,
        block_network=True,
        restrict_modal_access=True,
        include_source=False,
        volumes={corpus.b.MODEL_CACHE_MOUNT: volume.with_mount_options(read_only=True)},
    )
    def endpoint(seed):
        module = importlib.import_module("infra.modal_release_soak")
        module.prior._verify_source(expected)
        module.plan_profile(plan, capacity_limit=capacity_limit)
        owner = (
            module.live.NativeOwner(expected, profile=plan["profile"]["name"])
            if capacity_limit
            else module.live.NativeOwner(expected)
        )
        yield from module.run_suite(owner, seed, plan, expected)

    return app, endpoint


def collect(messages, directory, plan, expected):
    """Hash/persist complete event JSONL incrementally; fail on replay or gaps."""
    from whisper_runtime.adapters import StreamEventKind, TranscriptEvent
    from whisper_runtime.captions import CommittedTranscript

    capacity_limit = capacity_mode(plan)
    plan_profile(plan, capacity_limit=capacity_limit)
    reports, instance, active, handle = [], None, None, None
    digest, size, events = None, 0, 0
    result = {"status": "failed", "phases": reports}
    try:
        for item in messages:
            require(result.get("suite_done") is None, "message_after_suite_done")
            raw = encode(item)
            kind = item.get("type")
            if kind == "resource_contract":
                require(
                    capacity_limit
                    and not reports
                    and active is None
                    and instance is None
                    and isinstance(item.get("instance_id"), str)
                    and bool(item["instance_id"].strip())
                    and item.get("submitted_resource_contract") == resource_contract(),
                    "resource_contract_gate",
                )
                instance = item["instance_id"]
                result["resource_contract_receipt"] = item
            elif kind == "phase_started":
                require(
                    not capacity_limit or "resource_contract_receipt" in result,
                    "missing_resource_contract",
                )
                require(
                    active is None and len(reports) < len(plan["phases"]),
                    "phase_replay",
                )
                stage = plan["phases"][len(reports)]
                require(
                    item["phase"] == stage["phase"]
                    and item["session_number"] == len(reports) + 1,
                    "phase_order",
                )
                instance = instance or item["instance_id"]
                require(item["instance_id"] == instance, "worker_changed")
                active = stage["phase"]
                projection = CommittedTranscript()
                first_commit_ns = None
                previous_event_ns = -1
                handle = (directory / f"{active}.events.jsonl").open("xb")
                digest, size, events = hashlib.sha256(), 0, 0
            elif kind in {"event", "memory"}:
                require(
                    active is not None and item["phase"] == active, "unexpected_event"
                )
                size += len(raw)
                events += kind == "event"
                require(
                    size <= MAX_PHASE_LOG_BYTES and events <= MAX_PHASE_EVENTS,
                    "event_limit",
                )
                if kind == "event":
                    event = dict(item["event"])
                    event["kind"] = StreamEventKind(event["kind"])
                    caption = projection.accept(TranscriptEvent(**event))
                    if capacity_limit:
                        observed_ns = item.get("worker_elapsed_ns")
                        require(
                            type(observed_ns) is int
                            and observed_ns >= previous_event_ns
                            and observed_ns >= 0,
                            "event_clock_gate",
                        )
                        previous_event_ns = observed_ns
                        if (
                            caption is not None
                            and caption.text.strip()
                            and first_commit_ns is None
                        ):
                            first_commit_ns = observed_ns
                    projection.captions.clear()
                else:
                    require(
                        memory_valid(
                            item["sample"],
                            reports[0]["memory"] if reports else None,
                            terminal=False,
                            capacity_limit=capacity_limit,
                        ),
                        "memory_gate",
                    )
                handle.write(raw)
                handle.flush()
                digest.update(raw)
            elif kind == "phase_done":
                if capacity_limit:
                    require(
                        native_profile_valid(item.get("metadata", {}), plan["profile"]),
                        "native_profile_gate",
                    )
                    expected_first = (
                        None if first_commit_ns is None else first_commit_ns / 1e9
                    )
                    observed_first = item.get("first_nonempty_commit_seconds")
                    require(
                        observed_first == expected_first
                        and (
                            observed_first is None
                            or (
                                type(observed_first) in (int, float)
                                and math.isfinite(observed_first)
                            )
                        ),
                        "first_commit_receipt_gate",
                    )
                    if active == "smoke":
                        require(
                            first_commit_ns is not None
                            and first_commit_ns <= FIRST_COMMIT_SECONDS * 1e9,
                            "first_commit_gate",
                        )
                require(
                    active == item["phase"]
                    and stage_valid(
                        item,
                        stage,
                        expected,
                        plan["profile"],
                        instance,
                        len(reports) + 1,
                    ),
                    "terminal_gate",
                )
                require(events == item["event_count"], "event_count")
                require(
                    projection.final and projection.head == stage["sample_count"],
                    "event_coverage",
                )
                if active == "smoke":
                    rate = item.get("smoke_score", {}).get("word_edit_rate")
                    require(
                        isinstance(rate, (int, float))
                        and 0 <= rate <= plan["max_smoke_word_edit_rate"],
                        "smoke_quality_gate",
                    )
                require(
                    memory_valid(
                        item["memory"],
                        reports[0]["memory"] if reports else None,
                        terminal=True,
                        capacity_limit=capacity_limit,
                    ),
                    "memory_gate",
                )
                handle.close()
                handle = None
                item = {
                    **item,
                    "event_receipt": {
                        "path": f"{active}.events.jsonl",
                        "sha256": digest.hexdigest(),
                        "size_bytes": size,
                        "event_count": events,
                    },
                }
                prior._write_result(directory / f"{active}.terminal.json", item)
                reports.append(item)
                active = None
            elif kind == "phase_failed":
                require(
                    active == item["phase"] and item["instance_id"] == instance,
                    "failure_identity",
                )
                handle.close()
                handle = None
                failure = {
                    **item,
                    "event_receipt": {
                        "path": f"{active}.events.jsonl",
                        "sha256": digest.hexdigest(),
                        "size_bytes": size,
                        "event_count": events,
                        "partial": True,
                    },
                }
                prior._write_result(directory / f"{active}.terminal.json", failure)
                result["failure"] = failure
                # Keep active set: a later completed suite must fail its gate.
            elif kind == "suite_done":
                require(result.get("suite_done") is None, "duplicate_suite_done")
                result["suite_done"] = item
                if item.get("status") == "completed":
                    require(
                        active is None
                        and len(reports) == len(plan["phases"])
                        and item["instance_id"] == instance
                        and item["model_loads"] == 1
                        and item["sessions"] == len(reports),
                        "suite_gate",
                    )
                    result["status"] = "completed"
            else:
                raise SoakError("unexpected_message")
        if result.get("suite_done") is None:
            result.update(status="failed", error_code="missing_terminal")
        return result
    finally:
        if handle is not None:
            handle.close()
        if hasattr(messages, "close"):
            messages.close()


def run(
    *,
    replay_id,
    preflight=False,
    confirm_paid_gpu=False,
    root=ROOT,
    profile=None,
    capacity_limit=False,
):
    require(preflight != confirm_paid_gpu, "choose_preflight_or_confirm_paid_gpu")
    directory = prior._paths(Path(root), replay_id, True)[0].parent
    output = directory / (
        "release-soak-preflight.json" if preflight else "release-soak.json"
    )
    receipt = directory / "release-soak.attempt.jsonl"
    require(
        not output.exists() and (preflight or not receipt.exists()),
        "attempt_exists_no_retry",
    )
    profile = select_profile(profile, capacity_limit=capacity_limit)
    seed, plan = input_plan(root, profile=profile, capacity_limit=capacity_limit)
    expected = snapshot(root, profile=profile, capacity_limit=capacity_limit)
    specification = {
        "source": expected,
        "input": plan,
        "budget": budget_plan(capacity_limit=capacity_limit),
    }
    frozen = directory / "source"
    if preflight:
        if capacity_limit:
            require(not directory.exists(), "namespace_exists_no_overwrite")
        directory.mkdir(parents=True, exist_ok=not capacity_limit)
        if capacity_limit:
            for target_root, entries in (
                (frozen, expected["files"]),
                (directory / "review", expected["review_files_not_uploaded"]),
            ):
                for item in entries:
                    data = (Path(root) / item["path"]).read_bytes()
                    require(
                        len(data) == item["size_bytes"]
                        and prior._sha(data) == item["sha256"],
                        "source_changed_during_freeze",
                    )
                    target = target_root / item["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as handle:
                        handle.write(data)
            with (directory / "seed.pcm").open("xb") as handle:
                handle.write(seed)
        prior._write_result(
            output, {"status": "local-preflight-passed", **specification}
        )
        return output
    previous = json.loads((directory / "release-soak-preflight.json").read_text())
    require(
        previous == {"status": "local-preflight-passed", **specification},
        "preflight_no_longer_matches",
    )
    if capacity_limit:
        prior._verify_source(expected, frozen)
        prior._verify_source(
            {
                "files": expected["review_files_not_uploaded"],
                "digest": expected["review_digest"],
            },
            directory / "review",
        )
        require((directory / "seed.pcm").read_bytes() == seed, "frozen_seed_changed")
    modal = importlib.import_module("modal")
    require(
        str(modal.__version__) == prior.SDK_VERSION, "registered_modal_sdk_required"
    )
    prior._journal(
        receipt, "attempt-started", first=True, source_digest=expected["digest"]
    )
    record = {"status": "failed", "ephemeral_context_exit_completed": False}
    try:
        app, endpoint = resources(
            modal, expected, plan, frozen if capacity_limit else root
        )
        with app.run(detach=False):
            record["app_id"] = app.app_id
            prior._journal(receipt, "app-started", app_id=app.app_id)
            record.update(collect(endpoint.remote_gen(seed), directory, plan, expected))
        record["ephemeral_context_exit_completed"] = True
    except BaseException as error:
        record.update(
            status="failed",
            error_code=failure_code(error, "soak_attempt_failed"),
            error_type=type(error).__name__,
        )
    record.update(
        specification=specification,
        qualified=False,
        claim_boundary="native repeated-input endurance only; no WebSocket, diverse four-hour speech, physical mic, or production guarantee",
    )
    prior._write_result(output, record)
    prior._journal(
        receipt,
        "attempt-finished",
        status=record["status"],
        output_sha256=prior._sha(output.read_bytes()),
    )
    return output


def main(argv=None):
    """Return failure to the shell when the persisted attempt did not pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument(
        "--profile", default=None, choices=(PROFILE, *CAPACITY_PROFILES)
    )
    parser.add_argument(
        "--capacity-limit",
        action="store_true",
        help="new low-latency provider-endurance-v1 contract with requested 4096 MiB limit",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--confirm-paid-gpu", action="store_true")
    arguments = vars(parser.parse_args(argv))
    output = run(**arguments)
    print(output)
    record = json.loads(output.read_text(encoding="utf-8"))
    expected_status = (
        "local-preflight-passed" if arguments["preflight"] else "completed"
    )
    return 0 if record.get("status") == expected_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
