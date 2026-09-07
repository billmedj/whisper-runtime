"""Prepare one explicitly authorized T4 memory attribution diagnostic.

Imports/preflight are local only. Two identical 46.55s source-paced sessions,
not release qualification. No allocator settings, RAM gates, or numerics change.
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
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from infra import modal_release_soak as soak

ROOT, live, prior = soak.ROOT, soak.live, soak.prior
PRODUCER = "infra/modal_memory_diagnostic.py"
TEST = "tools/test_modal_memory_diagnostic.py"
CAPACITY_PLAN = "docs/research/2026-09-07-capacity-smoke-plan.md"
TIMEOUT = 300
DIAGNOSTIC_MODE = "native-memory-diagnostic-v2"
CAPACITY_MODE = "provider-capacity-smoke-v1"
SESSIONS = ("session-1", "session-2")
REQUIRED_SNAPSHOTS = (
    "before_native_imports",
    "after_torch_import",
    "after_native_imports",
    "after_model_load",
    "after_model_fingerprint",
    "session-1_closed_gc",
    "session-2_closed_gc",
)
ENVIRONMENT = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_MODULE_LOADING",
)
require = soak.require
MEMORY_SAMPLE_SCHEMA = "native-memory-sample-v2"
PROC_MEMORY_SCOPE = "guest-reported procfs estimates; physical and private/shared attribution unverified"
SMAPS_MAX_BYTES = 16 * 1024 * 1024
SMAPS_MAX_LINE_BYTES = 8192
SMAPS_REQUIRED = (
    "Rss",
    "Pss",
    "Private_Clean",
    "Private_Dirty",
    "Shared_Clean",
    "Shared_Dirty",
)
SMAPS_ADDITIVE = frozenset(
    (
        *SMAPS_REQUIRED,
        "Pss_Anon",
        "Pss_File",
        "Pss_Shmem",
        "Pss_Dirty",
        "Private_Hugetlb",
        "Shared_Hugetlb",
        "Swap",
        "SwapPss",
    )
)


def _smaps_metrics(path, *, aggregate):
    """Stream a bounded procfs file; sum attribution, never page-size metadata."""
    totals, fields = {}, {}
    headers = records = consumed = 0

    def finish_mapping():
        nonlocal records
        require(all(name in fields for name in SMAPS_REQUIRED), "incomplete_smaps")
        for name, value in fields.items():
            totals[name] = totals.get(name, 0) + value
        records += 1
        fields.clear()

    with path.open("rb") as handle:
        while raw := handle.readline(SMAPS_MAX_LINE_BYTES + 1):
            consumed += len(raw)
            require(
                len(raw) <= SMAPS_MAX_LINE_BYTES and consumed <= SMAPS_MAX_BYTES,
                "smaps_read_limit",
            )
            if re.match(rb"^[0-9a-fA-F]+-[0-9a-fA-F]+\s", raw):
                if headers or fields:
                    finish_mapping()
                headers += 1
                require(aggregate or headers == 1, "multiple_rollup_records")
                continue
            key, separator, value = raw.partition(b":")
            name = key.decode("ascii", errors="replace")
            if separator and name in SMAPS_ADDITIVE:
                require(not aggregate or headers > 0, "smaps_missing_mapping_header")
                match = re.fullmatch(rb"\s*([0-9]+)\s+kB\s*", value)
                require(match is not None and name not in fields, "invalid_smaps_field")
                fields[name] = int(match.group(1)) * 1024
    finish_mapping()
    require(aggregate or records == 1, "multiple_rollup_records")
    return {
        "source": "smaps_aggregate" if aggregate else "smaps_rollup",
        "scope": PROC_MEMORY_SCOPE,
        "reported_guest_estimates": True,
        "host_physical_memory_measured": False,
        "host_physical_memory_verified": False,
        "private_shared_attribution_verified": False,
        "atomic_snapshot": False,
        "mapping_count": records if aggregate else None,
        "bytes_read": consumed,
        "fields_bytes": totals,
        "private_bytes": sum(
            totals.get(key, 0)
            for key in ("Private_Clean", "Private_Dirty", "Private_Hugetlb")
        ),
        "shared_bytes": sum(
            totals.get(key, 0)
            for key in ("Shared_Clean", "Shared_Dirty", "Shared_Hugetlb")
        ),
    }


def memory_sample(proc=Path("/proc/self"), *, detailed=True):
    """Observe Linux and *already imported/initialized* native state only.

    No Torch/Numba import, device discovery, thread-pool initialization, cache
    purge or peak-counter reset. Partial measurements explicitly retain errors.
    A missing rollup permits bounded smaps aggregation, not a physical-RAM claim.
    PSS/private/shared remain guest-reported estimates, not verified physical
    attribution. No claim is made about the deployed guest implementation.
    """
    result = {
        "schema_version": MEMORY_SAMPLE_SCHEMA,
        "proc_memory_scope": PROC_MEMORY_SCOPE,
        "errors": [],
        "missing_capabilities": [],
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT},
    }

    def observe(name, operation):
        try:
            result[name] = operation()
        except Exception as error:
            result[name] = None
            result["errors"].append(
                {"component": name, "error_type": type(error).__name__}
            )

    def detailed_memory():
        try:
            return _smaps_metrics(proc / "smaps_rollup", aggregate=False)
        except FileNotFoundError:
            result["missing_capabilities"].append("smaps_rollup")
        try:
            return _smaps_metrics(proc / "smaps", aggregate=True)
        except FileNotFoundError:
            result["missing_capabilities"].append("smaps")
            raise  # Detailed metrics remain required: unknown is never zero.

    def statm():
        values = [int(value) for value in (proc / "statm").read_text().split()]
        require(len(values) == 7 and min(values) >= 0, "invalid_statm")
        page = os.sysconf("SC_PAGE_SIZE")
        return {
            "source": "statm",
            "scope": PROC_MEMORY_SCOPE,
            "reported_guest_estimates": True,
            "host_physical_memory_verified": False,
            "page_size_bytes": page,
            "pages": values,
            "rss_bytes": values[1] * page,
            "shared_bytes": values[2] * page,
        }

    def threads():
        for line in (proc / "status").read_text().splitlines():
            if line.startswith("Threads:"):
                value = int(line.split(":", 1)[1])
                require(value > 0, "invalid_threads")
                return value
        raise soak.SoakError("missing_threads")

    if detailed:
        observe("detailed_memory", detailed_memory)
    else:
        result["detailed_memory"] = None
        result["disabled_metrics"] = ["detailed_memory"]
    observe("statm", statm)
    observe("threads", threads)
    torch = sys.modules.get("torch")
    result["torch_imported"] = torch is not None
    result["whisper_imported"] = "whisper" in sys.modules
    result["numba_imported"] = "numba" in sys.modules
    numba_config = sys.modules.get("numba.core.config")
    result["numba_config"] = (
        {
            name: getattr(numba_config, name, None)
            for name in (
                "NUMBA_NUM_THREADS",
                "NUMBA_DEFAULT_NUM_THREADS",
                "THREADING_LAYER",
            )
        }
        if numba_config is not None
        else None
    )
    result["cuda"] = {"initialized": False} if torch is None else None
    if torch is not None:
        observe(
            "torch_threads",
            lambda: {
                "intra_op": torch.get_num_threads(),
                "inter_op": torch.get_num_interop_threads(),
            },
        )

        def cuda():
            if not torch.cuda.is_initialized():
                return {"initialized": False}
            torch.cuda.synchronize(0)
            return {
                "initialized": True,
                **{
                    name: int(getattr(torch.cuda, method)(0))
                    for name, method in (
                        ("allocated_bytes", "memory_allocated"),
                        ("reserved_bytes", "memory_reserved"),
                        ("peak_allocated_bytes", "max_memory_allocated"),
                        ("peak_reserved_bytes", "max_memory_reserved"),
                    )
                },
            }

        observe("cuda", cuda)
    return result


def capacity_mode(plan):
    mode = plan.get("mode", DIAGNOSTIC_MODE)
    require(mode in (DIAGNOSTIC_MODE, CAPACITY_MODE), "unknown_mode")
    return mode == CAPACITY_MODE


def profile_plan(plan):
    profile = soak.select_profile(
        plan["profile"]["name"],
        capacity_limit=capacity_mode(plan),
        allow_standard=True,
    )
    require(
        soak.profile_comparison(plan["profile"])
        == soak.profile_comparison(soak.profile_contract(profile)),
        "stale_profile",
    )
    if profile in soak.CAPACITY_PROFILES:
        require(
            type(plan.get("max_first_commit_seconds")) in (int, float)
            and plan["max_first_commit_seconds"] == soak.FIRST_COMMIT_SECONDS,
            "registered_first_commit_gate",
        )
    return profile


def resource_contract(capacity_smoke):
    return {
        "mode": CAPACITY_MODE if capacity_smoke else DIAGNOSTIC_MODE,
        "sdk_version": prior.SDK_VERSION,
        "memory_request_mib": 4096,
        "memory_limit_mib": 4096 if capacity_smoke else 0,
        "gpu": "T4",
        "maximum_containers": 1,
        "timeout_seconds": TIMEOUT,
        "basis": "submitted resource configuration; pinned SDK serialization checked",
        "guest_attests_provider_enforcement": False,
    }


def sampling_failed(sample, capacity_smoke):
    optional = (
        {"detailed_memory", "statm", "threads", "torch_threads"}
        if capacity_smoke
        else set()
    )
    return any(
        error.get("component") not in optional for error in sample.get("errors", [])
    )


def capacity_cuda_valid(sample, baseline):
    cuda = sample.get("cuda")
    if not isinstance(cuda, dict) or cuda.get("initialized") is not True:
        return False
    fields = (
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    if not all(type(cuda.get(name)) is int and cuda[name] >= 0 for name in fields):
        return False
    policy = soak.MEMORY_POLICY
    return cuda["reserved_bytes"] <= policy["reserved_ceiling_bytes"] and (
        baseline is None
        or (
            cuda["allocated_bytes"] == baseline["allocated_bytes"]
            and cuda["reserved_bytes"]
            <= baseline["reserved_bytes"] + policy["reserved_growth_bytes"]
        )
    )


class Observer:
    """Scoped observation of the unmodified native factory and DTW callables."""

    def __init__(self, emit, memory, *, capacity_smoke=False):
        self.emit, self.memory = emit, memory
        self.capacity_smoke = capacity_smoke
        self.session = None
        self.seen, self.counts, self.restores = set(), {}, []
        self.sampling_errors = False

    def capture(self, phase):
        try:
            sample = self.memory()
        except Exception as error:
            sample = {
                "errors": [{"component": "sampler", "error_type": type(error).__name__}]
            }
        self.sampling_errors |= sampling_failed(sample, self.capacity_smoke)
        self.seen.add(phase)
        self.emit(
            {
                "type": "memory",
                "phase": phase,
                "session": self.session,
                "sample": sample,
            }
        )
        return sample

    def wrap_dtw(self):
        # Whisper imports timing itself. Do not import optional backend modules
        # or invoke Numba APIs merely to make observation available.
        timing = sys.modules.get("whisper.timing")
        for name in ("dtw_cpu", "dtw_cuda"):
            original = getattr(timing, name, None)
            if not callable(original):
                continue
            self.counts[name] = 0

            def measured(*args, _name=name, _original=original, **kwargs):
                self.counts[_name] += 1
                first = self.counts[_name] == 1
                if first:
                    self.capture("first_" + _name + "_before")
                start_monotonic = time.monotonic() if first else None
                started = time.perf_counter_ns()
                status = "failed"
                try:
                    result = _original(*args, **kwargs)
                    status = "completed"
                    return result
                finally:
                    elapsed = time.perf_counter_ns() - started
                    if first:
                        end_monotonic = time.monotonic()
                        matrix = args[0] if args else kwargs.get("x")
                        self.emit(
                            {
                                "type": "operation",
                                "name": _name,
                                "session": self.session,
                                "status": status,
                                "shape": [int(n) for n in getattr(matrix, "shape", ())],
                                "wall_ns": elapsed,
                                "clock": "time.monotonic",
                                "start_monotonic_seconds": start_monotonic,
                                "end_monotonic_seconds": end_monotonic,
                                "first_call_only": True,
                            }
                        )
                        self.capture("first_" + _name + "_after")

            setattr(timing, name, measured)
            self.restores.append((timing, name, original))

    @contextmanager
    def factory_scope(self):
        setup = importlib.import_module("whisper_runtime.native_setup")
        original_imports, import_backend, load, fingerprint = (
            live.importlib,
            setup._import_backend,
            setup._load_model,
            setup._fingerprint,
        )

        def imported(name, *args, **kwargs):
            module = original_imports.import_module(name, *args, **kwargs)
            if name == "torch" and "after_torch_import" not in self.seen:
                self.capture("after_torch_import")
            if name == "whisper" and "after_native_imports" not in self.seen:
                self.capture("after_native_imports")
                self.wrap_dtw()
            return module

        def backend_imported(*args, **kwargs):
            module = import_backend(*args, **kwargs)
            if "after_native_imports" not in self.seen:
                self.capture("after_native_imports")
                self.wrap_dtw()
            return module

        def loaded(*args, **kwargs):
            model = load(*args, **kwargs)
            self.capture("after_model_load")
            return model

        def fingerprinted(*args, **kwargs):
            value = fingerprint(*args, **kwargs)
            if "after_model_fingerprint" not in self.seen:
                self.capture("after_model_fingerprint")
            return value

        # Replace only the harness module's importlib reference, not the global
        # import system. The factory and inference have one owner thread.
        live.importlib = SimpleNamespace(import_module=imported)
        setup._import_backend = backend_imported
        setup._load_model, setup._fingerprint = loaded, fingerprinted
        try:
            yield
        finally:
            live.importlib = original_imports
            setup._import_backend = import_backend
            setup._load_model, setup._fingerprint = load, fingerprint

    def close(self):
        for module, name, original in reversed(self.restores):
            setattr(module, name, original)
        self.restores.clear()


def input_plan(root=ROOT, *, capacity_smoke=False, profile=None):
    profile = soak.select_profile(
        profile, capacity_limit=capacity_smoke, allow_standard=True
    )
    seed, original = live.input_plan(root, short_only=True)
    require(len(seed) // 2 == 744800, "registered_source_length")
    return seed, {
        "schema_version": CAPACITY_MODE if capacity_smoke else DIAGNOSTIC_MODE,
        "mode": CAPACITY_MODE if capacity_smoke else DIAGNOSTIC_MODE,
        "submitted_resource_contract": resource_contract(capacity_smoke),
        "memory_sample_schema": MEMORY_SAMPLE_SCHEMA,
        "proc_memory_scope": PROC_MEMORY_SCOPE,
        "detailed_memory_policy": {
            "preferred": "smaps_rollup",
            "fallback": "smaps_aggregate",
            "fallback_only_on": "FileNotFoundError",
            "max_bytes_per_file": SMAPS_MAX_BYTES,
            "max_line_bytes": SMAPS_MAX_LINE_BYTES,
            "enabled": not capacity_smoke,
            "both_missing_is_failure": not capacity_smoke,
            "host_physical_memory_measured": False,
            "host_physical_memory_verified": False,
            "private_shared_attribution_verified": False,
            "capacity_qualification": "requires independently verified provider-limit or host telemetry; these guest estimates are diagnostic only",
        },
        "profile": soak.profile_contract(profile),
        "max_first_commit_seconds": soak.FIRST_COMMIT_SECONDS
        if profile in soak.CAPACITY_PROFILES
        else None,
        "seed_sha256": prior._sha(seed),
        "seed_samples": len(seed) // 2,
        "recipe": original["recipe"],
        "smoke_reference": original["smoke_reference"],
        "max_smoke_word_edit_rate": original["max_smoke_word_edit_rate"],
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
        "phases": [
            {
                "phase": name,
                "sample_count": len(seed) // 2,
                "sha256": prior._sha(seed),
                "timeout_seconds": 190,
            }
            for name in SESSIONS
        ],
        "outer_seconds": TIMEOUT,
        "drain_timeout_seconds": 30,
        "max_source_lateness_seconds": 0.25,
        "memory_is_observational": not capacity_smoke,
        "capacity_cuda_policy": {
            key: soak.MEMORY_POLICY[key]
            for key in (
                "terminal_allocated_delta_bytes",
                "reserved_growth_bytes",
                "reserved_ceiling_bytes",
            )
        }
        if capacity_smoke
        else None,
        "guest_memory_is_optional": capacity_smoke,
        "guest_rss_is_physical_capacity_gate": False,
        "release_memory_gates_unchanged": dict(soak.MEMORY_POLICY),
        "required_snapshots": list(REQUIRED_SNAPSHOTS),
        "environment_allowlist": list(ENVIRONMENT),
    }


def snapshot(root=ROOT, *, profile=soak.PROFILE):
    existing = soak.snapshot(
        root, profile=profile, capacity_limit=profile != soak.PROFILE
    )
    files = [
        *existing["files"],
        {
            "path": PRODUCER,
            "size_bytes": (Path(root) / PRODUCER).stat().st_size,
            "sha256": prior._sha((Path(root) / PRODUCER).read_bytes()),
        },
    ]
    files.sort(key=lambda item: item["path"])
    review = [
        *existing["review_files_not_uploaded"],
        {
            "path": TEST,
            "size_bytes": (Path(root) / TEST).stat().st_size,
            "sha256": prior._sha((Path(root) / TEST).read_bytes()),
        },
        {
            "path": CAPACITY_PLAN,
            "size_bytes": (Path(root) / CAPACITY_PLAN).stat().st_size,
            "sha256": prior._sha((Path(root) / CAPACITY_PLAN).read_bytes()),
        },
    ]
    review.sort(key=lambda item: item["path"])
    return {
        **existing,
        "files": files,
        "digest": prior._canonical(files),
        "review_files_not_uploaded": review,
        "review_digest": prior._canonical(review),
    }


def budget_plan(*, capacity_smoke=False):
    result = {
        **soak.budget_plan(),
        "outer_seconds": TIMEOUT,
        "mode": CAPACITY_MODE if capacity_smoke else DIAGNOSTIC_MODE,
        "submitted_resource_contract": resource_contract(capacity_smoke),
    }
    price = live.PRICES
    result["planning_compute_usd"] = round(
        TIMEOUT
        * (
            price["t4_per_second"]
            + 2 * price["cpu_core_per_second"]
            + 4 * price["gib_per_second"]
        )
        * price["regional_multiplier_ceiling"],
        6,
    )
    return result


def run_diagnostic(
    seed,
    plan,
    expected,
    *,
    memory=None,
    owner_factory=live.NativeOwner,
    score=None,
):
    """Bounded mailbox keeps baseline/partial snapshots visible during native work."""
    from whisper_runtime.captions import CommittedTranscript
    from whisper_runtime.cli import drive_stream

    profile = profile_plan(plan)
    require(prior._sha(seed) == plan["seed_sha256"], "seed_identity")
    capacity_smoke = capacity_mode(plan)
    if memory is None:

        def memory():
            return memory_sample(detailed=not capacity_smoke)

    mailbox, cancelled, finished = (
        queue.Queue(maxsize=64),
        threading.Event(),
        threading.Event(),
    )
    started = time.monotonic()
    owner = (
        owner_factory(expected)
        if profile == soak.PROFILE
        else owner_factory(expected, profile=profile)
    )

    def work():
        sequence = 0

        def emit(item):
            nonlocal sequence
            require(not cancelled.is_set(), "cancelled")
            sequence += 1
            item = {
                **item,
                "sequence": sequence,
                "instance_id": owner.instance_id,
                "source_digest": expected["digest"],
            }
            soak.encode(item)
            mailbox.put(item, timeout=0.25)

        observer = Observer(emit, memory, capacity_smoke=capacity_smoke)
        stream = finalize = terminal = None
        source_clock = None
        previous_text = None
        previous_exports = None
        cuda_baseline = None
        try:
            require(
                "torch" not in sys.modules and "whisper" not in sys.modules,
                "baseline_native_already_imported",
            )
            if capacity_smoke:
                emit(
                    {
                        "type": "resource_capabilities",
                        "submitted_resource_contract": resource_contract(True),
                        "before_native_imports": True,
                        "torch_imported": False,
                        "whisper_imported": False,
                        "optional_guest_metrics": ["statm", "threads"],
                        "detailed_smaps_sampled": False,
                        "required_terminal_cuda_counters": True,
                        "guest_attests_provider_enforcement": False,
                    }
                )
            observer.capture("before_native_imports")
            for number, stage in enumerate(plan["phases"], 1):
                source_clock = None
                observer.session = stage["phase"]
                require(
                    time.monotonic() - started + stage["sample_count"] / 16000 + 40
                    < plan["outer_seconds"],
                    "insufficient_session_budget",
                )
                emit(
                    {
                        "type": "session_started",
                        "phase": stage["phase"],
                        "session_number": number,
                    }
                )
                with observer.factory_scope():
                    stream, finalize = owner.factory(stage["phase"])
                require(
                    soak.config_comparison(asdict(stream.config))
                    == soak.config_comparison(plan["profile"]["stream_config"]),
                    "stale_stream_config",
                )
                transcript = CommittedTranscript()
                counters = {
                    "samples": 0,
                    "chunks": 0,
                    "events": 0,
                    "final": 0,
                    "eof": False,
                    "last": None,
                }
                digest = hashlib.sha256()
                origin = time.monotonic()
                source_clock = soak.SourceClock(origin)
                first_commit = None

                def source():
                    for frame in soak.frames(seed, stage["sample_count"]):
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
                    nonlocal first_commit
                    caption = transcript.accept(value)
                    elapsed = time.monotonic() - origin
                    if caption is not None and caption.text.strip():
                        if first_commit is None:
                            first_commit = elapsed
                    counters["events"] += 1
                    counters["final"] += value.kind.value == "final"
                    counters["last"] = value.kind.value
                    require(counters["events"] <= 10000, "event_limit")
                    require(
                        time.monotonic() - origin <= stage["timeout_seconds"],
                        "session_timeout",
                    )
                    emit(
                        {
                            "type": "event",
                            "phase": stage["phase"],
                            "event": asdict(value),
                            "elapsed_seconds": elapsed,
                        }
                    )

                drive_stream(
                    stream,
                    source(),
                    on_event=event,
                    cancel=threading.Event(),
                    live=True,
                    drain_timeout=30,
                )
                metadata = finalize()
                terminal = {
                    "type": "session_done",
                    "phase": stage["phase"],
                    "status": "completed",
                    "profile": plan["profile"],
                    "source_eof_received": counters["eof"],
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
                    "first_commit_seconds": first_commit,
                }
                limit = plan.get("max_first_commit_seconds")
                require(
                    limit is None
                    or (first_commit is not None and first_commit <= limit),
                    "first_commit_deadline",
                )
                require(
                    transcript.final
                    and transcript.head == stage["sample_count"] == counters["samples"],
                    "coverage_gate",
                )
                require(
                    soak.stage_valid(
                        terminal,
                        stage,
                        expected,
                        plan["profile"],
                        owner.instance_id,
                        number,
                    ),
                    "stage_gate",
                )
                text = transcript.render("txt")
                require(len(text) <= 65536, "text_limit")
                difference = score or prior._corpus().b._word_difference
                terminal["score"] = difference(text, plan["smoke_reference"])
                rate = terminal["score"].get("word_edit_rate")
                require(
                    type(rate) in (float, int)
                    and 0 <= rate <= plan["max_smoke_word_edit_rate"],
                    "quality_gate",
                )
                require(
                    previous_text is None or text == previous_text,
                    "session_text_changed",
                )
                previous_text = text
                terminal["text"], terminal["text_sha256"] = (
                    text,
                    prior._sha(text.encode()),
                )
                if capacity_smoke:
                    exports = {
                        format: prior._sha(transcript.render(format).encode())
                        for format in ("txt", "srt", "vtt")
                    }
                    require(
                        previous_exports is None or exports == previous_exports,
                        "capacity_exports_changed",
                    )
                    terminal["export_sha256"] = previous_exports = exports
                stream = finalize = transcript = None
                gc.collect()
                sample = observer.capture(stage["phase"] + "_closed_gc")
                if capacity_smoke:
                    require(
                        capacity_cuda_valid(sample, cuda_baseline), "capacity_cuda_gate"
                    )
                    if cuda_baseline is None:
                        cuda_baseline = dict(sample["cuda"])
                emit(terminal)
                terminal = None
            require(
                set(REQUIRED_SNAPSHOTS) <= observer.seen, "missing_lifecycle_snapshot"
            )
            require(not observer.sampling_errors, "memory_sampling_incomplete")
            emit(
                {
                    "type": "diagnostic_done",
                    "status": "completed",
                    "sessions": owner.sessions,
                    "model_loads": owner.loads,
                    "dtw_calls": observer.counts,
                    "qualified": False,
                }
            )
        except BaseException as error:
            source_prefix = soak.source_prefix_receipt(
                source_clock, digest if source_clock else None, stream, terminal
            )
            cleanup = False
            terminal_after_cleanup = None
            if stream is not None:
                try:
                    stream.close()
                    terminal_after_cleanup = finalize()
                    cleanup = True
                except BaseException:
                    pass
                stream = finalize = None
            gc.collect()
            try:
                observer.capture("failure_closed_gc")
                emit(
                    {
                        "type": "diagnostic_done",
                        "status": "failed",
                        "qualified": False,
                        "error_code": soak.failure_code(
                            error, "native_memory_diagnostic_failed"
                        ),
                        "error_type": type(error).__name__,
                        "session": observer.session,
                        "cleanup_verified": cleanup,
                        "terminal_before_gate": terminal,
                        # Separate post-cleanup evidence never upgrades failure
                        # or replaces an earlier pre-gate terminal observation.
                        "terminal_after_cleanup": terminal_after_cleanup,
                        "source_clock": (
                            source_clock.receipt() if source_clock else None
                        ),
                        "source_prefix": source_prefix,
                        "initialization_stage": getattr(
                            owner, "initialization_stage", "scripted"
                        ),
                        "dtw_calls": observer.counts,
                    }
                )
            except (queue.Full, soak.SoakError):
                pass
        finally:
            observer.close()
            finished.set()

    thread = threading.Thread(
        target=work, name="memory-diagnostic-native-owner", daemon=True
    )
    thread.start()
    try:
        while not finished.is_set() or not mailbox.empty():
            require(
                time.monotonic() - started <= plan["outer_seconds"], "outer_timeout"
            )
            try:
                yield mailbox.get(timeout=0.1)
            except queue.Empty:
                continue
    finally:
        cancelled.set()
        thread.join(timeout=5)
        require(not thread.is_alive(), "native_owner_not_stopped")


def resources(modal, expected, plan, root):
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
    image = live.prepare_profile_image(
        image, plan.get("profile", {}).get("name", soak.PROFILE), expected, root
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
    app = modal.App("wr-memory-diagnostic-" + uuid.uuid4().hex[:12])

    @app.function(
        image=image,
        serialized=True,
        cpu=2,
        memory=(4096, 4096) if capacity_mode(plan) else 4096,
        gpu="T4",
        cloud="aws",
        region="us-west",
        timeout=TIMEOUT,
        startup_timeout=180,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=2,
        single_use_containers=True,
        block_network=True,
        restrict_modal_access=True,
        include_source=False,
        volumes={corpus.b.MODEL_CACHE_MOUNT: volume.with_mount_options(read_only=True)},
    )
    def endpoint(seed):
        module = importlib.import_module("infra.modal_memory_diagnostic")
        module.prior._verify_source(expected)
        yield from module.run_diagnostic(seed, plan, expected)

    return app, endpoint


def collect(messages, directory, plan, expected):
    from whisper_runtime.adapters import StreamEventKind, TranscriptEvent
    from whisper_runtime.captions import CommittedTranscript

    result = {"status": "failed", "sessions": [], "snapshots": [], "operations": []}
    capacity_smoke = capacity_mode(plan)
    profile_plan(plan)
    cuda_baseline = None
    previous_exports = None
    instance = active = projection = previous_text = None
    size = events = 0
    first_commit = None
    last_event_elapsed = 0.0
    digest = hashlib.sha256()
    try:
        with (directory / "observations.jsonl").open("xb") as handle:
            for sequence, item in enumerate(messages, 1):
                require("terminal" not in result, "message_after_terminal")
                raw = soak.encode(item)
                size += len(raw)
                require(size <= 8 * 1024 * 1024, "observation_limit")
                require(
                    item["sequence"] == sequence
                    and item["source_digest"] == expected["digest"],
                    "observation_identity",
                )
                require(
                    isinstance(item.get("instance_id"), str)
                    and bool(item["instance_id"].strip()),
                    "invalid_worker_identity",
                )
                instance = instance or item["instance_id"]
                require(item["instance_id"] == instance, "worker_changed")
                handle.write(raw)
                handle.flush()
                digest.update(raw)
                kind = item["type"]
                if kind == "resource_capabilities":
                    require(
                        capacity_smoke
                        and sequence == 1
                        and item["submitted_resource_contract"]
                        == resource_contract(True)
                        and item.get("before_native_imports") is True
                        and item.get("torch_imported") is False
                        and item.get("whisper_imported") is False
                        and item.get("guest_attests_provider_enforcement") is False,
                        "invalid_resource_receipt",
                    )
                    result["resource_contract_receipt"] = item
                elif kind == "memory":
                    require(
                        not any(
                            row["phase"] == item["phase"] for row in result["snapshots"]
                        ),
                        "duplicate_snapshot",
                    )
                    result["snapshots"].append(item)
                elif kind == "operation":
                    result["operations"].append(item)
                elif kind == "session_started":
                    require(
                        active is None and len(result["sessions"]) < 2, "session_replay"
                    )
                    stage = plan["phases"][len(result["sessions"])]
                    require(
                        item["phase"] == stage["phase"]
                        and item["session_number"] == len(result["sessions"]) + 1,
                        "session_order",
                    )
                    active, projection, events = (
                        stage["phase"],
                        CommittedTranscript(),
                        0,
                    )
                    first_commit, last_event_elapsed = None, 0.0
                elif kind == "event":
                    require(
                        active == item["phase"] and projection is not None,
                        "unexpected_event",
                    )
                    event = {
                        **item["event"],
                        "kind": StreamEventKind(item["event"]["kind"]),
                    }
                    caption = projection.accept(TranscriptEvent(**event))
                    if "elapsed_seconds" in item:
                        elapsed = item["elapsed_seconds"]
                        require(
                            type(elapsed) in (float, int)
                            and math.isfinite(elapsed)
                            and elapsed >= last_event_elapsed,
                            "invalid_event_clock",
                        )
                        last_event_elapsed = elapsed
                        if caption is not None and caption.text.strip():
                            if first_commit is None:
                                first_commit = elapsed
                    else:
                        require(
                            plan.get("max_first_commit_seconds") is None,
                            "missing_event_clock",
                        )
                    events += 1
                elif kind == "session_done":
                    limit = plan.get("max_first_commit_seconds")
                    require(
                        limit is None
                        or (
                            first_commit is not None
                            and first_commit <= limit
                            and item.get("first_commit_seconds") == first_commit
                        ),
                        "first_commit_deadline",
                    )
                    require(
                        active == item["phase"]
                        and projection is not None
                        and soak.stage_valid(
                            item,
                            stage,
                            expected,
                            plan["profile"],
                            instance,
                            len(result["sessions"]) + 1,
                        ),
                        "terminal_gate",
                    )
                    require(
                        projection.final
                        and projection.head
                        == item["offered_samples"]
                        == stage["sample_count"]
                        and events == item["event_count"],
                        "event_coverage",
                    )
                    text = projection.render("txt")
                    require(
                        text == item["text"]
                        and item["text_sha256"] == prior._sha(text.encode()),
                        "text_identity",
                    )
                    require(
                        previous_text is None or previous_text == text,
                        "session_text_changed",
                    )
                    rate = item.get("score", {}).get("word_edit_rate")
                    require(
                        type(rate) in (float, int)
                        and 0 <= rate <= plan["max_smoke_word_edit_rate"],
                        "quality_gate",
                    )
                    require(
                        any(
                            row["phase"] == active + "_closed_gc"
                            for row in result["snapshots"]
                        ),
                        "missing_cleanup_snapshot",
                    )
                    if capacity_smoke:
                        exports = {
                            format: prior._sha(projection.render(format).encode())
                            for format in ("txt", "srt", "vtt")
                        }
                        require(
                            item.get("export_sha256") == exports
                            and (
                                previous_exports is None or previous_exports == exports
                            ),
                            "capacity_exports_changed",
                        )
                        previous_exports = exports
                        sample = next(
                            row["sample"]
                            for row in result["snapshots"]
                            if row["phase"] == active + "_closed_gc"
                        )
                        require(
                            capacity_cuda_valid(sample, cuda_baseline),
                            "capacity_cuda_gate",
                        )
                        if cuda_baseline is None:
                            cuda_baseline = dict(sample["cuda"])
                    result["sessions"].append(item)
                    previous_text, active, projection = text, None, None
                elif kind == "diagnostic_done":
                    result["terminal"] = item
                    if item["status"] == "completed":
                        require(
                            not capacity_smoke or "resource_contract_receipt" in result,
                            "missing_resource_receipt",
                        )
                        require(
                            active is None
                            and len(result["sessions"]) == item["sessions"] == 2
                            and item["model_loads"] == 1,
                            "diagnostic_gate",
                        )
                        require(
                            set(REQUIRED_SNAPSHOTS)
                            <= {row["phase"] for row in result["snapshots"]},
                            "missing_lifecycle_snapshot",
                        )
                        require(
                            not any(
                                sampling_failed(row["sample"], capacity_smoke)
                                for row in result["snapshots"]
                            ),
                            "memory_sampling_incomplete",
                        )
                        result["status"] = "completed"
                else:
                    raise soak.SoakError("unexpected_message")
        if "terminal" not in result:
            result["error_code"] = "missing_terminal"
        result["observation_receipt"] = {
            "path": "observations.jsonl",
            "sha256": digest.hexdigest(),
            "size_bytes": size,
        }
        return result
    finally:
        if hasattr(messages, "close"):
            messages.close()


def run(
    *,
    replay_id,
    preflight=False,
    confirm_paid_gpu=False,
    capacity_smoke=False,
    profile=None,
    root=ROOT,
):
    require(preflight != confirm_paid_gpu, "choose_preflight_or_confirm_paid_gpu")
    require(type(capacity_smoke) is bool, "invalid_capacity_mode")
    profile = soak.select_profile(
        profile, capacity_limit=capacity_smoke, allow_standard=True
    )
    require(
        isinstance(replay_id, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", replay_id)
        and not re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", replay_id, re.I),
        "invalid_replay_id",
    )
    base = Path(root).resolve() / "artifacts/modal-memory-diagnostic"
    directory = base / replay_id
    require(directory.resolve().parent == base.resolve(), "artifact_path_redirected")
    output = directory / ("preflight.json" if preflight else "diagnostic.json")
    receipt, frozen = directory / "attempt.jsonl", directory / "source"
    require(not output.exists() and not receipt.exists(), "attempt_exists_no_retry")
    seed, plan = input_plan(root, capacity_smoke=capacity_smoke, profile=profile)
    expected = snapshot(root, profile=profile)
    specification = {
        "mode": CAPACITY_MODE if capacity_smoke else DIAGNOSTIC_MODE,
        "source": expected,
        "input": plan,
        "budget": budget_plan(capacity_smoke=capacity_smoke),
        "qualified": False,
    }
    if preflight:
        require(not directory.exists(), "namespace_exists_no_overwrite")
        directory.mkdir(parents=True, exist_ok=False)
        for target_root, entries in (
            (frozen, expected["files"]),
            (directory / "review", expected.get("review_files_not_uploaded", [])),
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
    previous = json.loads((directory / "preflight.json").read_text())
    require(
        previous == {"status": "local-preflight-passed", **specification},
        "preflight_no_longer_matches",
    )
    prior._verify_source(expected, frozen)
    if expected.get("review_files_not_uploaded"):
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
        app, endpoint = resources(modal, expected, plan, frozen)
        with app.run(detach=False):
            record["app_id"] = app.app_id
            prior._journal(receipt, "app-started", app_id=app.app_id)
            record.update(collect(endpoint.remote_gen(seed), directory, plan, expected))
        record["ephemeral_context_exit_completed"] = True
    except BaseException as error:
        record.update(
            status="failed",
            error_code=soak.failure_code(error, "memory_diagnostic_attempt_failed"),
            error_type=type(error).__name__,
        )
    record.update(
        specification=specification,
        qualified=False,
        diagnostic_only=not capacity_smoke,
        mode=CAPACITY_MODE if capacity_smoke else DIAGNOSTIC_MODE,
        claim_boundary=(
            "short operational capacity smoke under a submitted provider memory limit; guest counters do not attest enforcement; not release qualification or a change to the historical RSS gate"
            if capacity_smoke
            else "observational memory attribution; two repeated-input native sessions, not release qualification or a RAM policy change"
        ),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument(
        "--profile", choices=(soak.PROFILE, *soak.CAPACITY_PROFILES), default=None
    )
    parser.add_argument(
        "--capacity-smoke",
        action="store_true",
        help="separately registered short provider-capacity smoke; requests a 4096 MiB maximum",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--confirm-paid-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = run(**vars(args))
        record = json.loads(output.read_text())
        print(output)
        return (
            0
            if (
                (args.preflight and record["status"] == "local-preflight-passed")
                or (
                    args.confirm_paid_gpu
                    and record["status"] == "completed"
                    and record.get("ephemeral_context_exit_completed") is True
                )
            )
            else 1
        )
    except Exception as error:
        print(
            "memory-diagnostic: " + soak.failure_code(error, "local_diagnostic_failed"),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
