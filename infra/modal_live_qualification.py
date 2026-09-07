"""Opt-in, ephemeral live-v2 qualification: smoke gates a same-worker 30m run.

Imports and --preflight are CPU-local only. A paid attempt needs the matching
preflight plus --confirm-paid-gpu. No durable deployment, reconnect or retry.
--short-only permits registered context experiments; completed-short is not V0.1 qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from infra import modal_network_replay as prior

ROOT, REMOTE_ROOT = prior.ROOT, prior.REMOTE_ROOT
PRODUCER = "infra/modal_live_qualification.py"
TEST = "tools/test_modal_live_qualification.py"
LONG_SECONDS = 1800
SMOKE_TIMEOUT = 190
LONG_TIMEOUT = 1890
OUTER_SECONDS = 2200
SHORT_OUTER_SECONDS = 210
MAX_PLANNED_USD = 1.0
MAX_EVENT_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_TERMINAL_RECEIPT_BYTES = 524288
REUSE_PATCH = "patches/openai-whisper/experimental/0008-Add-optional-alignment-audio-features.patch"
REUSE_PATCH_SHA = "e665bca7abea5ab273c34ecf4d63c050f8c7120c7fdd9c411038b4f7511cd169"
SMOKE_MAX_WORD_EDIT_RATE = 0.10
PRICES = {
    "t4_per_second": 0.000164,
    "cpu_core_per_second": 0.0000131,
    "gib_per_second": 0.00000222,
    "regional_multiplier_ceiling": 1.75,
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def frame_source(seed: bytes, sample_count: int):
    """Repeat whole registered cycles, then silence; bounded 640-byte chunks."""
    _require(isinstance(seed, bytes) and seed and len(seed) % 2 == 0, "invalid seed")
    _require(
        type(sample_count) is int and 0 < sample_count <= LONG_SECONDS * 16000,
        "invalid registered sample count",
    )
    total_bytes = sample_count * 2
    repeated_bytes = total_bytes // len(seed) * len(seed)
    for offset in range(0, total_bytes, 640):
        count = min(640, total_bytes - offset)
        take = min(count, max(0, repeated_bytes - offset))
        start = offset % len(seed)
        content = seed[start : start + take]
        missing = take - len(content)
        if missing:
            content += (seed * ((missing + len(seed) - 1) // len(seed)))[:missing]
        yield content + bytes(count - take)


def input_plan(
    root=ROOT, *, short_only=False, left_context_ms=None, word_context_limit_ms=None
):
    overrides = {
        key: value
        for key, value in dict(
            left_context_ms=left_context_ms,
            word_context_limit_ms=word_context_limit_ms,
        ).items()
        if value is not None
    }
    _require(
        type(short_only) is bool and (short_only or not overrides),
        "context overrides require --short-only",
    )
    config = replace(
        importlib.import_module("whisper_runtime.native_setup").CLI_STREAM_CONFIG,
        **overrides,
    )
    corpus = prior._corpus()
    shared = importlib.import_module("infra.modal_acoustic_diagnostic")
    metadata, pcms = shared._cases(root, root / corpus.ASSET_PATH)
    inventory = {case["id"]: case for case in metadata["cases"]}
    pcm, noisy = (
        pcms["concatenated-no-added-pauses"],
        pcms["mixed-continuous-noise64"][: 174240 * 2],
    )
    _require(
        len(pcm) == 538560 * 2 and len(noisy) == 174240 * 2,
        "registered main/noisy prefix lengths differ",
    )
    # Every repeat ends in real digital silence, never a cut-off word.
    seed = pcm + bytes(32000) + noisy + bytes(32000)
    noisy_reference = " ".join(
        fixture["reference_text"]
        for fixture in corpus.read_registration(root)["fixtures"][:2]
    )
    stages = {}
    for phase, samples in (("smoke", len(seed) // 2), ("long", LONG_SECONDS * 16000)):
        if short_only and phase == "long":
            continue
        digest = hashlib.sha256()
        for frame in frame_source(seed, samples):
            digest.update(frame)
        stages[phase] = dict(
            sample_count=samples,
            sha256=digest.hexdigest(),
            timeout_seconds=SMOKE_TIMEOUT if phase == "smoke" else LONG_TIMEOUT,
            cycles=samples // (len(seed) // 2),
        )
        if short_only:
            stages[phase]["stream_config"] = asdict(config)
    return seed, dict(
        stages=stages,
        cycle_samples=len(seed) // 2,
        main_case="concatenated-no-added-pauses",
        main_sha256=prior._sha(pcm),
        noisy_source_case="mixed-continuous-noise64",
        noisy_prefix_samples=174240,
        noisy_prefix_sha256=prior._sha(noisy),
        smoke_reference=inventory["concatenated-no-added-pauses"]["reference_text"]
        + " "
        + noisy_reference,
        arms={
            "main": dict(
                start_sample=0,
                end_sample=len(pcm) // 2,
                reference=inventory["concatenated-no-added-pauses"]["reference_text"],
            ),
            "noisy": dict(
                start_sample=len(pcm) // 2 + 16000,
                end_sample=len(seed) // 2 - 16000,
                reference=noisy_reference,
            ),
        },
        max_smoke_word_edit_rate=SMOKE_MAX_WORD_EDIT_RATE,
        quality_calibration="preregistered before GPU: CPU main6/88 unchanged baseline, noisy1/26; 10% fixed development gate, not held-out accuracy",
        recipe="full33.66s registered concatenated speech + 1s silence + registered10.89s noisy prefix + 1s silence",
        long_tail="whole cycles then digital silence",
        **(dict(short_only=True, config_overrides=overrides) if short_only else {}),
    )


def budget_plan(*, short_only=False):
    outer_seconds = SHORT_OUTER_SECONDS if short_only else OUTER_SECONDS
    per_second = (
        PRICES["t4_per_second"]
        + 2 * PRICES["cpu_core_per_second"]
        + 4 * PRICES["gib_per_second"]
    )
    estimate = outer_seconds * per_second * PRICES["regional_multiplier_ceiling"]
    _require(
        estimate < MAX_PLANNED_USD,
        "registered compute estimate exceeds first-attempt target",
    )
    return dict(
        prices=PRICES,
        prices_source="https://modal.com/pricing",
        prices_checked="2026-09-06",
        maximum_gpu_containers=1,
        minimum_gpu_containers=0,
        outer_seconds=outer_seconds,
        planning_compute_usd=round(estimate, 6),
        first_attempt_target_usd=MAX_PLANNED_USD,
        total_pass_target_usd=2.0,
        physical_spending_cap=False,
        exclusions="image build, startup beyond active watchdog, storage, network, taxes, provider rescheduling and price changes",
        automatic_retries=0,
        long_requires_smoke=not short_only,
        persistent_deployment=False,
        **(
            dict(short_only=True, request_timeout_seconds=SMOKE_TIMEOUT)
            if short_only
            else {}
        ),
    )


def snapshot(root=ROOT):
    root = Path(root).resolve()
    names = set(prior._source_paths(root)) | {
        PRODUCER,
        TEST,
        "examples/pcm_live.py",
        "infra/modal_acoustic_diagnostic.py",
        "experiments/modal-acoustic-diagnostic-v2.json",
        "infra/modal_native_cuda_qualification.py",
        "infra/native_cuda_trace.py",
        "infra/modal-native-cuda-image-inputs.lock",
        "tools/prepare_acoustic_cases.py",
        "tools/prepare_speech_corpus.py",
    }
    files = [
        dict(path=name, size_bytes=len(data), sha256=prior._sha(data))
        for name in sorted(names)
        for data in [(root / name).read_bytes()]
    ]
    return dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        digest=prior._canonical(files),
        files=files,
        source_policy="exact reviewed working-byte overlay; not a clean-tree claim",
    )


def prepare_profile_image(image, profile, expected, root=ROOT):
    """Opt-in build-only reuse patch, after the frozen file overlay is attached.

    The caller first builds the unchanged standard image. Neither this helper
    nor NativeOwner applies a patch in a running inference worker. Imports and
    constructing this image description do not build or launch remote resources.
    """
    from whisper_runtime.native_setup import BACKEND_REUSE_TREE, BACKEND_TREE
    from whisper_runtime.profiles import get_profile

    if not get_profile(profile).reuse_alignment_features:
        return image
    manifest = str(Path(REUSE_PATCH).parent / "SHA256SUMS").replace("\\", "/")
    inventory = {item["path"]: item for item in expected["files"]}
    for name in (REUSE_PATCH, manifest):
        _require(name in inventory, "reuse image file not frozen")
        data = (Path(root) / name).read_bytes()
        _require(
            len(data) == inventory[name]["size_bytes"]
            and prior._sha(data) == inventory[name]["sha256"],
            "reuse image source changed",
        )
        if name == REUSE_PATCH:
            _require(prior._sha(data) == REUSE_PATCH_SHA, "reuse patch hash differs")
        else:
            _require(
                data.decode("ascii").splitlines()
                == [REUSE_PATCH_SHA + "  " + Path(REUSE_PATCH).name],
                "reuse patch manifest differs",
            )
    patch = (REMOTE_ROOT / REUSE_PATCH).as_posix()
    patch_directory = (REMOTE_ROOT / manifest).parent.as_posix()
    command = " && ".join(
        (
            f"cd {patch_directory} && sha256sum --check SHA256SUMS",
            "cd /opt/openai-whisper",
            'test "$(git rev-parse HEAD)" = a0b9695ae1cc52bad4b8626fe9fb6ea4ac0ee650',
            'test "$(git rev-parse HEAD^{tree})" = ' + BACKEND_TREE,
            'test -z "$(git status --porcelain --untracked-files=all)"',
            f"git apply --check {patch}",
            f"git apply --index {patch}",
            'test "$(git write-tree)" = ' + BACKEND_REUSE_TREE,
            "env GIT_AUTHOR_NAME='Whisper Runtime Bootstrap' "
            "GIT_AUTHOR_EMAIL=whisper-runtime@example.invalid "
            "GIT_COMMITTER_NAME='Whisper Runtime Bootstrap' "
            "GIT_COMMITTER_EMAIL=whisper-runtime@example.invalid "
            "GIT_AUTHOR_DATE=2000-01-01T00:00:00Z "
            "GIT_COMMITTER_DATE=2000-01-01T00:00:00Z "
            "git -c user.name='Whisper Runtime Bootstrap' "
            "-c user.email=whisper-runtime@example.invalid "
            "-c commit.gpgSign=false -c core.hooksPath=/dev/null "
            "commit --no-gpg-sign --no-verify "
            "-m 'Enable pinned same-window alignment feature reuse'",
            'test "$(git rev-parse HEAD^{tree})" = ' + BACKEND_REUSE_TREE,
            'test -z "$(git status --porcelain --untracked-files=all)"',
        )
    )
    return image.run_commands(command)


class NativeOwner:
    """One loaded model, successive closed single-stream owners, no warmup/control decode."""

    def __init__(self, expected, *, profile=None, config_overrides=None):
        from whisper_runtime.profiles import get_profile

        self.expected, self.model = expected, None
        self.execution_profile = get_profile(
            "conservative-v1" if profile is None else profile
        )
        self.explicit_profile = profile is not None
        self.config_overrides = dict(config_overrides or {})
        self._check_profile_overrides()
        self.instance_id, self.loads, self.sessions = uuid.uuid4().hex, 0, 0
        self.worker = None
        self.backend_setup = None
        self.backend_modules = None
        self.initialization_stage = "not_started"
        self.alignment_initialization = None

    def _check_profile_overrides(self):
        _require(
            not self.explicit_profile or not self.config_overrides,
            "named profile excludes config overrides",
        )
        _require(
            set(self.config_overrides) <= {"left_context_ms", "word_context_limit_ms"},
            "unregistered native context override",
        )

    def _check_backend_modules(self, path):
        """Reuse only the modules loaded by this owner's guarded source import."""
        if self.backend_modules is None:
            _require(
                not any(
                    name == "whisper" or name.startswith("whisper.")
                    for name in sys.modules
                ),
                "ambient Whisper module already imported",
            )
            return
        package = path / "whisper"
        for name, module in self.backend_modules.items():
            _require(sys.modules.get(name) is module, "backend module identity changed")
        for name, module in tuple(sys.modules.items()):
            if name != "whisper" and not name.startswith("whisper."):
                continue
            location = getattr(module, "__file__", None)
            _require(
                isinstance(location, str)
                and Path(location).resolve().is_relative_to(package),
                "backend module path differs",
            )
        for name, filename in (
            ("whisper", "__init__.py"),
            ("whisper.triton_ops", "triton_ops.py"),
        ):
            _require(
                Path(sys.modules[name].__file__).resolve() == package / filename,
                "backend module path differs",
            )

    def factory(self, phase):
        self.initialization_stage = "imports"
        runtime = importlib.import_module("whisper_runtime")
        adapters = importlib.import_module("whisper_runtime.adapters")
        setup = importlib.import_module("whisper_runtime.native_setup")
        self._check_profile_overrides()
        selected = self.execution_profile
        config = replace(selected.stream_config, **self.config_overrides)
        backend_tree = (
            setup.BACKEND_REUSE_TREE
            if selected.reuse_alignment_features
            else setup.BACKEND_TREE
        )
        corpus = prior._corpus()
        backend_path = Path(corpus.b.BACKEND_ROOT).resolve()
        self.initialization_stage = "backend_import_identity"
        self._check_backend_modules(backend_path)
        self.initialization_stage = "dependencies"
        setup.validate_dependencies()
        self.initialization_stage = "backend_tree"
        _require(
            Path(setup._git(backend_path, "rev-parse", "--show-toplevel")).resolve()
            == backend_path,
            "backend is not its own Git worktree",
        )
        revision = setup._git(backend_path, "rev-parse", "HEAD")
        _require(
            setup._git(backend_path, "rev-parse", "HEAD^{tree}") == backend_tree,
            "backend tree differs",
        )
        self.initialization_stage = "backend_clean"
        _require(
            not setup._git(
                backend_path, "status", "--porcelain", "--untracked-files=all"
            ),
            "backend dirty",
        )
        ignored = setup._git(
            backend_path,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "whisper",
        )
        for name in ignored.splitlines():
            extra = Path(name)
            _require(
                "__pycache__" in extra.parts and extra.suffix == ".pyc",
                "backend contains untracked importable files",
            )
        _require(
            (backend_path / "whisper/__init__.py").is_file(), "backend package missing"
        )
        backend_setup = setup.BackendSetup(backend_path, revision)
        _require(
            self.backend_setup is None or self.backend_setup == backend_setup,
            "backend identity changed between sessions",
        )
        self.initialization_stage = "checkpoint"
        setup.validate_checkpoint(corpus.b.MODEL_CHECKPOINT_PATH)
        torch, np = (importlib.import_module(name) for name in ("torch", "numpy"))
        self.initialization_stage = "device"
        _require(
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and torch.cuda.get_device_name(0) == "Tesla T4",
            "registered T4 unavailable",
        )
        if self.backend_modules is None:
            self.initialization_stage = "backend_import"
            # Match installed CLI readiness: explicit verified source and fresh
            # bytecode guard, including eager CUDA prerequisites. This imports
            # Triton declarations; it does not warm JIT kernels or run inference.
            whisper = setup._import_backend(backend_setup, eager_cuda=True)
            self.backend_modules = {
                name: module
                for name, module in tuple(sys.modules.items())
                if name == "whisper" or name.startswith("whisper.")
            }
            _require(
                self.backend_modules.get("whisper") is whisper
                and "whisper.triton_ops" in self.backend_modules,
                "guarded backend import incomplete",
            )
            self.backend_setup = backend_setup
            self._check_backend_modules(backend_path)
        whisper = self.backend_modules["whisper"]
        if self.model is None:
            self.initialization_stage = "model_load"
            torch.set_num_threads(1)
            self.model = setup._load_model(
                whisper, corpus.b.MODEL_CHECKPOINT_PATH, "cuda:0"
            )
            self.loads += 1
        model = self.model
        self.initialization_stage = "model_identity"
        initial = setup._fingerprint(model)
        _require(
            initial == setup.MODEL_FINGERPRINT and not model.training,
            "model identity differs",
        )
        _require(
            all(
                str(value.device) == "cuda:0"
                and (not value.is_floating_point() or value.dtype == torch.float32)
                for value in (*model.parameters(), *model.buffers())
            ),
            "model device/precision differs",
        )
        if self.alignment_initialization is None:
            self.initialization_stage = "alignment_backtrace"
            self.alignment_initialization = setup._prepare_alignment_backtrace()
        capacity = runtime.ResourceVector(
            memory_bytes=2147483648, compute_units=1, stream_slots=1
        )
        identity = runtime.ModelSnapshot(
            "tiny.en",
            revision,
            "pytorch-cuda-live-v01",
            initial,
        )
        self.initialization_stage = "worker"
        if self.worker is None:
            self.worker = runtime.Worker(
                "live-v01",
                identity,
                runtime.Budget(capacity),
                queue_capacity=1,
                transaction_ttl_seconds=180,
            )
        # A model binding owns exactly one Worker for the model's lifetime.
        worker, budget = self.worker, self.worker.budget
        _require(
            worker.model == identity
            and budget.available == capacity
            and budget.lease_count == 0
            and worker.queue_depth == 0,
            "native worker unavailable",
        )

        def probe(observed):
            _require(observed is model, "model binding changed")
            return identity

        self.initialization_stage = "adapter"
        adapter = adapters.NativeWhisperAdapter(
            worker,
            model,
            probe,
            adapters.NativeExecutionProfile(
                selected.native_profile_id,
                capacity,
                device="cuda:0",
                reuse_alignment_features=selected.reuse_alignment_features,
            ),
        )

        def mel(pcm):
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            return whisper.log_mel_spectrogram(
                whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
            ).contiguous()

        self.initialization_stage = "stream"
        self.sessions += 1
        session_number = self.sessions
        stream = adapters.ContinuousTranscriptStream(
            adapter,
            stream_id=f"live-v01:{phase}",
            mel_builder=mel,
            options=adapters.NativeDecodeOptions(
                language="en", without_timestamps=False
            ),
            rng_seed=7,
            config=config,
        )
        recent_traces = deque(maxlen=8)
        step = stream.step

        def observed_step():
            try:
                return step()
            finally:
                trace = stream.last_trace
                if trace is not None and (
                    not recent_traces
                    or recent_traces[-1]["decode_index"] != trace.decode_index
                ):
                    recent_traces.append(
                        {
                            key: getattr(trace, key)
                            for key in (
                                "decode_index",
                                "analysis_start_sample",
                                "analysis_end_sample",
                                "action",
                                "reason",
                                "eof",
                            )
                        }
                    )

        stream.step = observed_step
        self.initialization_stage = "cuda_synchronize"
        torch.cuda.synchronize(0)
        before = int(torch.cuda.memory_allocated(0))
        self.initialization_stage = "ready"

        def finalize():
            _require(
                budget.available == capacity
                and budget.lease_count == 0
                and worker.queue_depth == 0,
                "native capacity retained",
            )
            torch.cuda.synchronize(0)
            final = setup._fingerprint(model)
            _require(final == initial, "model state changed")
            last_trace = (
                corpus.b._plain(stream.last_trace)
                if stream.last_trace is not None
                else None
            )
            trace_bytes = json.dumps(last_trace, sort_keys=True).encode()
            return dict(
                instance_id=self.instance_id,
                model_loads=self.loads,
                session_number=session_number,
                capacity_restored=True,
                model_unchanged=True,
                model_initial_sha256=initial,
                model_final_sha256=final,
                alignment_heads="verified tiny.en named-model mask restored on path load",
                alignment_initialization=self.alignment_initialization,
                stream_config=asdict(config),
                config_overrides=self.config_overrides,
                execution_profile=(
                    asdict(selected) if not self.config_overrides else None
                ),
                native_profile_id=selected.native_profile_id,
                reuse_alignment_features=selected.reuse_alignment_features,
                backend_tree=backend_tree,
                backend_revision=revision,
                profile_id=stream.profile_id,
                allocated_before_bytes=before,
                allocated_after_bytes=int(torch.cuda.memory_allocated(0)),
                peak_buffered_samples=stream.metrics.peak_buffered_samples,
                max_buffer_samples=config.max_buffer_ms * 16,
                recent_traces=list(recent_traces),
                last_trace=last_trace if len(trace_bytes) <= 262144 else None,
                last_trace_sha256=prior._sha(trace_bytes),
                last_trace_omitted=len(trace_bytes) > 262144,
                source_digest=self.expected["digest"],
            )

        return stream, finalize


def stage_valid(done, stage, *, expected_digest):
    metrics, metadata = done.get("metrics", {}), done.get("metadata", {})
    return (
        done.get("status") == "completed"
        and done.get("source_eof_received") is True
        and metrics.get("accepted_samples")
        == metrics.get("committed_samples")
        == stage["sample_count"]
        and metrics.get("buffered_samples") == 0
        and metrics.get("accepted_sha256") == stage["sha256"]
        and metadata.get("capacity_restored") is True
        and metadata.get("model_unchanged") is True
        and metadata.get("model_loads") == 1
        and type(metadata.get("peak_buffered_samples")) is int
        and type(metadata.get("max_buffer_samples")) is int
        and 0 <= metadata["peak_buffered_samples"] <= metadata["max_buffer_samples"]
        and metadata.get("source_digest") == expected_digest
        and (
            "stream_config" not in stage
            or metadata.get("stream_config") == stage["stream_config"]
        )
    )


def make_gpu_app(expected, plan, *, owner=None):
    prior._verify_source(expected)
    server = importlib.import_module("examples.pcm_websocket")
    limits_module = importlib.import_module("examples.pcm_live")
    owner = owner or NativeOwner(
        expected, config_overrides=plan.get("config_overrides")
    )
    outer_seconds = SHORT_OUTER_SECONDS if plan.get("short_only") else OUTER_SECONDS
    guard, active, next_phase = threading.Lock(), False, "smoke"
    reports = {}
    created = time.monotonic()

    async def app(scope, receive, send):
        nonlocal active, next_phase
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                await send({"type": message["type"] + ".complete"})
                if message["type"] == "lifespan.shutdown":
                    return
        if scope["type"] == "http":
            key = scope.get("path", "").removeprefix("/receipt/")
            with guard:
                raw = reports.pop(key, None) if scope.get("method") == "GET" else None
            status, body = (
                (404, b'{"status":"missing_receipt"}') if raw is None else (200, raw)
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        phase = scope.get("path", "").strip("/")
        with guard:
            allowed = (
                scope["type"] == "websocket" and phase == next_phase and not active
            )
            allowed = allowed and time.monotonic() - created < outer_seconds
            if allowed:
                active, next_phase = True, None
        if not allowed:
            await send({"type": "websocket.close", "code": 1008})
            return
        stage, observed = plan["stages"][phase], None
        initialization_failure = None

        def checked_factory():
            nonlocal initialization_failure
            try:
                return owner.factory(phase)
            except Exception as error:
                initialization_failure = dict(
                    stage=str(getattr(owner, "initialization_stage", "factory"))[:64],
                    exception_type=type(error).__name__[:128],
                )
                # No exception message, traceback, environment, or credentials.
                print(
                    json.dumps({"initialization_failure": initialization_failure}),
                    flush=True,
                )
                raise

        async def checked_send(message):
            nonlocal observed
            if message.get("type") == "websocket.send" and message.get("text"):
                value = json.loads(message["text"])
                if value.get("type") == "done":
                    observed = value
                    if initialization_failure is not None:
                        value["metadata"] = dict(
                            value.get("metadata", {}),
                            initialization_failure=initialization_failure,
                            instance_id=getattr(owner, "instance_id", None),
                            source_digest=expected["digest"],
                        )
                    if value.get("status") == "completed" and not stage_valid(
                        value, stage, expected_digest=expected["digest"]
                    ):
                        value.update(
                            status="failed", error_code="qualification_gate_failed"
                        )
                    message = {**message, "text": json.dumps(value)}
                    raw = json.dumps(
                        dict(
                            phase=phase,
                            source_digest=expected["digest"],
                            done=value,
                            instance_id=value.get("metadata", {}).get("instance_id"),
                        ),
                        allow_nan=False,
                    ).encode()
                    with guard:
                        if len(raw) <= MAX_TERMINAL_RECEIPT_BYTES:
                            reports[phase] = raw
            await send(message)

        handler = server.make_app(
            checked_factory,
            live_limits=limits_module.LiveLimits(
                max_samples=stage["sample_count"],
                max_duration_s=stage["timeout_seconds"],
            ),
        )
        try:
            await handler(scope, receive, checked_send)
        finally:
            with guard:
                active = False
                if (
                    phase == "smoke"
                    and not plan.get("short_only")
                    and observed
                    and stage_valid(observed, stage, expected_digest=expected["digest"])
                ):
                    next_phase = "long"

    return app


async def read_receipt(url, headers, phase, expected_digest, instance_id):
    """One authenticated bounded read; no redirects/retries/inference on this route."""
    aiohttp = importlib.import_module("aiohttp")
    parsed = urlsplit(url)
    target = urlunsplit(
        ("https", parsed.netloc, parsed.path.rstrip("/") + "/receipt/" + phase, "", "")
    )
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5), trust_env=False
        ) as session:
            _require(
                hasattr(session, "_retry_connection"), "unsupported no-retry transport"
            )
            session._retry_connection = False
            async with session.get(
                target, headers=headers, allow_redirects=False
            ) as response:
                if response.status == 404:
                    return dict(status="missing_receipt")
                _require(
                    response.status == 200
                    and type(response.content_length) is int
                    and 0 < response.content_length <= MAX_TERMINAL_RECEIPT_BYTES,
                    "invalid receipt size/status",
                )
                raw = await response.content.readexactly(response.content_length)
        value = json.loads(raw)
        _require(
            value["phase"] == phase
            and value["source_digest"] == expected_digest
            and (instance_id is None or value["instance_id"] == instance_id),
            "receipt owner/source differs",
        )
        client = importlib.import_module("examples.replay_websocket")
        return dict(
            status="available",
            sha256=prior._sha(raw),
            record=client._redact(value, (target, *headers.values())),
        )
    except Exception:
        return dict(status="receipt_unavailable")


def resources(modal, expected, plan, root=ROOT):
    corpus = prior._corpus()
    q = corpus._helper("infra.modal_native_cuda_qualification")
    gpu_image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("ca-certificates", "ffmpeg", "git")
        .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
        .uv_pip_install(*q.DIRECT_IMAGE_PACKAGES)
        .run_commands(q._build_command(corpus.BASE_COMMIT))
    )

    def overlay(image):
        for item in expected["files"]:
            image = image.add_local_file(
                root / item["path"], (REMOTE_ROOT / item["path"]).as_posix(), copy=True
            )
        return image.env(
            {
                "PYTHONPATH": "/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "WHISPER_MODAL_ENABLE_WORD_CORPUS": "0",
                "WHISPER_MODAL_ENABLE_REMOTE_RESOURCES": "0",
            }
        )

    cpu_image, gpu_image = (
        overlay(modal.Image.debian_slim(python_version="3.13")),
        overlay(gpu_image),
    )
    label = "wr-live-v01-" + uuid.uuid4().hex[:12]
    app = modal.App(label)
    common = dict(
        serialized=True,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=2,
        startup_timeout=180,
        block_network=True,
        restrict_modal_access=True,
        include_source=False,
    )

    @app.function(
        **common,
        image=cpu_image,
        cpu=0.25,
        memory=256,
        timeout=30,
        single_use_containers=True,
    )
    @modal.asgi_app(requires_proxy_auth=True, label=label + "-echo")
    def echo():
        return prior.make_echo_app(expected, plan["echo_sha256"])

    volume = modal.Volume.from_name(corpus.b.MODEL_CACHE_NAME, create_if_missing=False)

    @app.function(
        **common,
        image=gpu_image,
        cpu=2,
        memory=4096,
        gpu="T4",
        cloud="aws",
        region="us-west",
        timeout=SMOKE_TIMEOUT if plan.get("short_only") else LONG_TIMEOUT,
        single_use_containers=False,
        volumes={corpus.b.MODEL_CACHE_MOUNT: volume.with_mount_options(read_only=True)},
    )
    @modal.asgi_app(requires_proxy_auth=True, label=label + "-gpu")
    def endpoint():
        return importlib.import_module("infra.modal_live_qualification").make_gpu_app(
            expected, plan
        )

    return app, echo, endpoint


async def run_stage(
    url, headers, seed, plan, phase, expected_digest, *, artifact_directory=None
):
    client = importlib.import_module("examples.replay_websocket")
    stage = plan["stages"][phase]
    revisions, words, commit_count = {}, [], 0
    arm_words, crossing = {"main": [], "noisy": []}, []
    event_file = (
        None
        if artifact_directory is None
        else artifact_directory / f"live-v01-{phase}.events.jsonl"
    )
    event_handle = event_file.open("xb") if event_file is not None else None
    event_digest, event_bytes = hashlib.sha256(), 0

    async def events(event, elapsed_ns):
        nonlocal commit_count, event_bytes
        if event_handle is not None:
            raw = (
                json.dumps(
                    dict(event=event, client_elapsed_ns=elapsed_ns),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode()
            _require(
                event_bytes + len(raw) <= MAX_EVENT_RECEIPT_BYTES,
                "event receipt byte limit",
            )
            event_handle.write(raw)
            event_handle.flush()
            event_digest.update(raw)
            event_bytes += len(raw)
        if event["kind"] in ("provisional", "replace"):
            revisions[event["segment_id"]] = event["text"]
        elif event["kind"] == "commit":
            text = revisions.pop(event["segment_id"], "")
            commit_count += 1
            if phase == "smoke":
                words.append(text)
                if text.strip() and "arms" in plan:
                    if event["end_sample"] <= plan["arms"]["noisy"]["start_sample"]:
                        arm_words["main"].append(text)
                    elif event["start_sample"] >= plan["arms"]["main"]["end_sample"]:
                        arm_words["noisy"].append(text)
                    else:
                        crossing.append(
                            dict(
                                start_sample=event["start_sample"],
                                end_sample=event["end_sample"],
                                text=text,
                            )
                        )

    def source_frames():
        if artifact_directory is None:
            yield from frame_source(seed, stage["sample_count"])
        else:
            with (artifact_directory / f"live-v01-{phase}.pcm").open(
                "rb"
            ) as source_file:
                yield from iter(lambda: source_file.read(640), b"")

    async def source():
        # This generator is first advanced only after server READY.
        origin = time.monotonic_ns()
        offered = 0
        for frame in source_frames():
            offered += len(frame) // 2
            deadline = origin + offered * 62500
            delay = deadline - time.monotonic_ns()
            if delay > 0:
                await asyncio.sleep(delay / 1e9)
            _require(
                time.monotonic_ns() - deadline <= 250_000_000,
                "source lag exceeds registration",
            )
            yield frame

    try:
        result = await client.stream_live(
            url.rstrip("/") + "/" + phase,
            source(),
            headers=headers,
            config=client.LiveConfig(
                max_samples=stage["sample_count"],
                total_timeout_s=stage["timeout_seconds"],
            ),
            on_event=events,
        )
    finally:
        if event_handle is not None:
            event_handle.close()
    done = result.get("server_done") or {}
    receipt = await read_receipt(
        url,
        headers,
        phase,
        expected_digest,
        done.get("metadata", {}).get("instance_id"),
    )
    result["qualification_receipt"] = receipt
    if receipt["status"] == "available":
        done = receipt["record"]["done"]
    valid = (
        stage_valid(done, stage, expected_digest=expected_digest)
        and result.get("status") == "completed"
        and receipt["status"] == "available"
    )
    score = None
    if phase == "smoke":
        score = prior._corpus().b._word_difference(
            " ".join(words), plan["smoke_reference"]
        )
        valid = (
            valid
            and score["word_edit_rate"] is not None
            and score["word_edit_rate"] <= plan["max_smoke_word_edit_rate"]
        )
    return dict(
        phase=phase,
        passed=bool(valid),
        result=result,
        committed_events=commit_count,
        smoke_score=score,
        smoke_text=" ".join(words) if phase == "smoke" else None,
        arm_scores=None
        if phase != "smoke" or "arms" not in plan
        else {
            name: dict(
                text=" ".join(arm_words[name]),
                score=None
                if crossing
                else prior._corpus().b._word_difference(
                    " ".join(arm_words[name]), arm["reference"]
                ),
            )
            for name, arm in plan["arms"].items()
        },
        cross_arm_commits=crossing,
        event_receipt=None
        if event_file is None
        else dict(
            path=event_file.name,
            size_bytes=event_bytes,
            sha256=event_digest.hexdigest(),
        ),
    )


async def staged_run(
    url,
    headers,
    seed,
    plan,
    expected_digest,
    *,
    runner=run_stage,
    artifact_directory=None,
):
    options = (
        {} if artifact_directory is None else {"artifact_directory": artifact_directory}
    )
    smoke = await runner(url, headers, seed, plan, "smoke", expected_digest, **options)
    result = dict(status="failed", smoke=smoke, long=None, long_gate_passed=False)
    if not smoke["passed"]:
        return result
    if plan.get("short_only"):
        result["status"] = "completed-short"
        return result
    result["long_gate_passed"] = True
    long = await runner(url, headers, seed, plan, "long", expected_digest, **options)
    result["long"] = long
    first = smoke["result"]["server_done"]["metadata"]
    second = long["result"].get("server_done", {}).get("metadata", {})
    same = (
        first.get("instance_id") == second.get("instance_id")
        and second.get("session_number") == 2
    )
    result.update(
        same_worker_model=same,
        status="completed" if long["passed"] and same else "failed",
    )
    return result


def attempt(
    modal, expected, plan, seed, *, resource_factory=resources, artifact_directory=None
):
    cleanup = dict(ephemeral_context_exit_completed=False, proxy_token_deleted=False)
    record = dict(status="failed", cleanup=cleanup)
    manager, token = modal.Workspace.from_context().proxy_tokens, None
    try:
        app, echo, endpoint = resource_factory(modal, expected, plan)
        environment = modal.Environment.from_context()
        environment.hydrate()
        _require(
            isinstance(environment.name, str) and environment.name,
            "Modal environment unavailable",
        )
        token = manager.create()
        info = next(item for item in manager.list() if item.token_id == token.token_id)
        if info.scoped:
            manager.allow(token.token_id, environment.name)
        headers = {"Modal-Key": token.token_id, "Modal-Secret": token.token_secret}
        with app.run(detach=False, environment_name=environment.name):
            record["app_id"] = app.app_id
            probe = asyncio.run(
                prior._cpu_preflight(
                    prior._websocket_url(echo.get_web_url()), seed, headers
                )
            )
            record["transport_preflight"] = probe
            _require(
                probe["status"] == "completed",
                "CPU authentication/transport preflight failed",
            )
            record["qualification"] = asyncio.run(
                asyncio.wait_for(
                    staged_run(
                        prior._websocket_url(endpoint.get_web_url()),
                        headers,
                        seed,
                        plan,
                        expected["digest"],
                        artifact_directory=artifact_directory,
                    ),
                    SHORT_OUTER_SECONDS if plan.get("short_only") else OUTER_SECONDS,
                )
            )
            record["status"] = record["qualification"]["status"]
        cleanup["ephemeral_context_exit_completed"] = True
    except BaseException as error:
        record.update(
            status="failed",
            error_code="qualification_attempt_failed",
            error_type=type(error).__name__,
        )
    finally:
        if token is not None:
            try:
                manager.delete(token.token_id)
                cleanup["proxy_token_deleted"] = True
            except BaseException as error:
                record.update(
                    status="failed",
                    error_code="proxy_token_cleanup_failed",
                    error_type=type(error).__name__,
                )
    if not all(cleanup.values()):
        record["status"] = "failed"
    return record


def run(
    *,
    replay_id,
    preflight=False,
    confirm_paid_gpu=False,
    root=ROOT,
    short_only=False,
    left_context_ms=None,
    word_context_limit_ms=None,
):
    directory = prior._paths(root, replay_id, True)[0].parent
    output = directory / ("live-v01-preflight.json" if preflight else "live-v01.json")
    receipt = directory / "live-v01.attempt.jsonl"
    _require(preflight or confirm_paid_gpu is True, "requires --confirm-paid-gpu")
    _require(
        not output.exists() and (preflight or not receipt.exists()),
        "attempt exists; no retry or overwrite",
    )
    seed, plan = input_plan(
        root,
        short_only=short_only,
        left_context_ms=left_context_ms,
        word_context_limit_ms=word_context_limit_ms,
    )
    plan["echo_sha256"] = prior._sha(seed[:32000])
    expected, budget = snapshot(root), budget_plan(short_only=short_only)
    specification = dict(source=expected, input=plan, budget=budget)
    if preflight:
        output.parent.mkdir(parents=True, exist_ok=True)
        for phase, stage in plan["stages"].items():
            with (directory / f"live-v01-{phase}.pcm").open("xb") as handle:
                for frame in frame_source(seed, stage["sample_count"]):
                    handle.write(frame)
        prior._write_result(
            output, dict(status="local-preflight-passed", **specification)
        )
        return output
    previous = json.loads(
        (directory / "live-v01-preflight.json").read_text(encoding="utf-8")
    )
    _require(
        previous == dict(status="local-preflight-passed", **specification),
        "local preflight no longer matches bytes/plan",
    )
    for phase, stage in plan["stages"].items():
        digest = hashlib.sha256()
        path = directory / f"live-v01-{phase}.pcm"
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
        _require(
            path.stat().st_size == stage["sample_count"] * 2
            and digest.hexdigest() == stage["sha256"],
            "preregistered raw source file differs",
        )
    prior._journal(
        receipt, "attempt-started", first=True, source_digest=expected["digest"]
    )
    modal = importlib.import_module("modal")
    _require(
        str(modal.__version__) == prior.SDK_VERSION, "registered Modal SDK required"
    )
    record = attempt(modal, expected, plan, seed, artifact_directory=directory)
    record.update(
        specification=specification,
        qualified=False,
        claim_boundary="one bounded development run; no universal accuracy, latency, availability or spending guarantee",
    )
    prior._write_result(output, record)
    prior._journal(
        receipt,
        "attempt-finished",
        status=record["status"],
        output_sha256=prior._sha(output.read_bytes()),
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--left-context-ms", type=int)
    parser.add_argument("--word-context-limit-ms", type=int)
    print(run(**vars(parser.parse_args())))
