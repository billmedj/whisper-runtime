"""Authenticated PC-to-Modal PCM replay. Importing creates no remote resources."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/whisper-runtime")
MANIFEST_PATH = "experiments/modal-network-replay-v1.json"
PRODUCER_PATH = "infra/modal_network_replay.py"
EXAMPLE_PATHS = ("examples/pcm_websocket.py", "examples/replay_websocket.py")
HEADER = struct.Struct("!IQ")
SAMPLES = 698560
PCM_SHA256 = "b6f8c541b55d2da4c9494dda2e5daebf8674c29286a3ea983206b9a99e9badad"
SDK_VERSION = "1.5.5"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical(value: object) -> str:
    return _sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def _corpus() -> Any:
    flag = "WHISPER_MODAL_ENABLE_WORD_CORPUS"
    old = os.environ.get(flag)
    try:
        os.environ[flag] = "0"
        return importlib.import_module("infra.modal_word_corpus")
    finally:
        if old is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = old


def read_registration(root: Path = ROOT) -> dict:
    registration = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1-diagnostic",
        "id": "modal-network-replay-v1",
        "protocol": "pcm-websocket/v1",
        "sample_rate_hz": 16000,
        "chunk_samples": 320,
        "sample_count": SAMPLES,
        "pcm_sha256": PCM_SHA256,
        "corpus_manifest": "experiments/modal-word-corpus-v1.json",
        "case_id": "three-speakers-repeat-pauses",
        "modal_sdk": SDK_VERSION,
        "proxy_auth_required": True,
        "preflight": {
            "gpu": False,
            "sample_count": 16000,
            "protocol": "pcm-echo/v1",
            "unauthorized_statuses": [401, 403],
        },
        "execution": {
            "gpu": "T4",
            "cpu": 2,
            "memory_mib": 4096,
            "cloud": "aws",
            "region": "us-west",
            "timeout_seconds": 180,
            "startup_timeout_seconds": 180,
            "max_containers": 1,
            "min_containers": 0,
            "buffer_containers": 0,
            "scaledown_seconds": 2,
            "configured_retries": 0,
            "client_connections": 1,
            "single_use_containers": True,
        },
        "claims": {
            "performance_benchmark": False,
            "production_readiness": False,
            "recognition_improvement": False,
            "one_way_network_latency": False,
            "physical_execution_or_spending_cap": False,
        },
    }
    for key, value in expected.items():
        if _canonical(registration.get(key)) != _canonical(value):
            raise ValueError(f"unregistered {key}")
    return registration


def _source_paths(root: Path) -> list[str]:
    corpus = _corpus()
    return sorted(
        [
            path.relative_to(root).as_posix()
            for path in (root / "src/whisper_runtime").rglob("*.py")
        ]
        + [PRODUCER_PATH, MANIFEST_PATH, *EXAMPLE_PATHS]
        + [corpus.PRODUCER_PATH, corpus.MANIFEST_PATH, *corpus.HELPER_PATHS]
    )


def source_snapshot(root: Path = ROOT) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("commit local work before a network diagnostic")
    commit = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid source commit")
    files = []
    for name in _source_paths(root):
        data = (root / name).read_bytes()
        files.append({"path": name, "size_bytes": len(data), "sha256": _sha(data)})
    return {"commit": commit, "files": files, "digest": _canonical(files)}


def _verify_source(snapshot: dict, root: Path = REMOTE_ROOT) -> None:
    if _canonical(snapshot["files"]) != snapshot["digest"]:
        raise ValueError("source manifest digest mismatch")
    for item in snapshot["files"]:
        path = root / item["path"]
        if path.resolve().is_relative_to(root.resolve()) is not True:
            raise ValueError("source path escapes root")
        data = path.read_bytes()
        if len(data) != item["size_bytes"] or _sha(data) != item["sha256"]:
            raise ValueError("source file mismatch")


def _input(root: Path = ROOT) -> bytes:
    corpus = _corpus()
    manifest = corpus.read_registration(root)
    case = next(
        item
        for item in manifest["cases"]
        if item["id"] == "three-speakers-repeat-pauses"
    )
    pcm = corpus.build_case(case, manifest["fixtures"], root / corpus.ASSET_PATH)
    if len(pcm) != SAMPLES * 2 or _sha(pcm) != PCM_SHA256:
        raise ValueError("registered PCM mismatch")
    return pcm


def _paths(root: Path, replay_id: str, preflight: bool) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", replay_id) or re.fullmatch(
        r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", replay_id, re.I
    ):
        raise ValueError("invalid replay id")
    directory = root / "artifacts/modal" / replay_id
    if directory.resolve().parent != (root / "artifacts/modal").resolve():
        raise ValueError("replay path redirects outside artifacts")
    stem = "network-preflight" if preflight else "network-replay"
    return directory / f"{stem}.json", directory / f"{stem}.attempt.jsonl"


def _journal(path: Path, event: str, *, first: bool = False, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if first else "a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "event": event,
                    "at": datetime.now(timezone.utc).isoformat(),
                    **fields,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def _write_result(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _websocket_url(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or not parts.hostname.endswith(".modal.run")
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("unexpected authenticated endpoint")
    return urlunsplit(("wss", parts.netloc, parts.path or "/", "", ""))


def make_echo_app(snapshot: dict, expected_sha256: str):
    """CPU-only binary transport check. It emits no transcription events."""

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                kind = event["type"].replace("lifespan.", "")
                await send({"type": f"lifespan.{kind}.complete"})
                if kind == "shutdown":
                    return
        if scope["type"] != "websocket":
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        _verify_source(snapshot)
        await receive()
        await send({"type": "websocket.accept"})
        try:
            initial = await asyncio.wait_for(receive(), 15)
            start = json.loads(initial.get("text", ""))
            if start != {
                "type": "start",
                "protocol": "pcm-echo/v1",
                "sample_count": 16000,
                "sha256": expected_sha256,
            }:
                raise ValueError("invalid echo start")
            await send(
                {
                    "type": "websocket.send",
                    "text": json.dumps({"type": "ready", "protocol": "pcm-echo/v1"}),
                }
            )
            sequence = samples = 0
            digest = hashlib.sha256()
            while True:
                message = await asyncio.wait_for(receive(), 15)
                frame = message.get("bytes")
                if frame is not None:
                    if (
                        not isinstance(frame, bytes)
                        or len(frame) != HEADER.size + 640
                        or samples + 320 > 16000
                    ):
                        raise ValueError("invalid echo frame")
                    if HEADER.unpack_from(frame) != (sequence, samples):
                        raise ValueError("invalid echo sequence")
                    digest.update(frame[HEADER.size :])
                    samples += 320
                    sequence += 1
                    await send({"type": "websocket.send", "bytes": frame})
                else:
                    eof = json.loads(message.get("text", ""))
                    if (
                        eof
                        != {
                            "type": "eof",
                            "chunks": 50,
                            "samples": 16000,
                            "sha256": expected_sha256,
                        }
                        or samples != 16000
                        or digest.hexdigest() != expected_sha256
                    ):
                        raise ValueError("invalid echo EOF")
                    await send(
                        {
                            "type": "websocket.send",
                            "text": json.dumps(
                                {
                                    "type": "done",
                                    "status": "completed",
                                    "samples": samples,
                                    "chunks": sequence,
                                    "sha256": digest.hexdigest(),
                                    "source_snapshot": snapshot,
                                }
                            ),
                        }
                    )
                    await send({"type": "websocket.close", "code": 1000})
                    return
        except (ValueError, KeyError, TypeError, asyncio.TimeoutError):
            await send({"type": "websocket.close", "code": 1008})

    return app


async def _cpu_preflight(url: str, pcm: bytes, headers: dict) -> dict:
    aiohttp = importlib.import_module("aiohttp")
    pcm = pcm[:32000]
    digest = _sha(pcm)

    async def reject_redirect(session, context, params):
        raise ValueError("proxy redirect rejected")

    trace = aiohttp.TraceConfig()
    trace.on_request_redirect.append(reject_redirect)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=90), trace_configs=[trace]
    ) as session:
        if not hasattr(session, "_retry_connection"):
            raise RuntimeError("HTTP client cannot disable hidden retries")
        session._retry_connection = False
        try:
            async with session.ws_connect(
                url, headers={}, compress=0, max_msg_size=4096
            ):
                raise ValueError("unauthorized WebSocket was accepted")
        except aiohttp.WSServerHandshakeError as error:
            unauthorized = error.status
            if unauthorized not in {401, 403}:
                raise ValueError("proxy did not reject unauthorized access") from None
        async with session.ws_connect(
            url, headers=headers, compress=0, max_msg_size=32768
        ) as ws:
            await ws.send_json(
                {
                    "type": "start",
                    "protocol": "pcm-echo/v1",
                    "sample_count": 16000,
                    "sha256": digest,
                }
            )
            ready = await ws.receive_json(timeout=30)
            if ready != {"type": "ready", "protocol": "pcm-echo/v1"}:
                raise ValueError("invalid echo ready")
            started = time.monotonic_ns()

            async def produce():
                max_lag = 0
                for sequence, offset in enumerate(range(0, len(pcm), 640)):
                    scheduled = (offset // 2 + 320) * 62500
                    await asyncio.sleep(
                        max(0, (started + scheduled - time.monotonic_ns()) / 1e9)
                    )
                    offered = time.monotonic_ns() - started
                    if offered - scheduled > 250_000_000:
                        raise ValueError("CPU source lag")
                    await ws.send_bytes(
                        HEADER.pack(sequence, offset // 2) + pcm[offset : offset + 640]
                    )
                    max_lag = max(max_lag, time.monotonic_ns() - started - scheduled)
                    if max_lag > 250_000_000:
                        raise ValueError("CPU send lag")
                await ws.send_json(
                    {"type": "eof", "chunks": 50, "samples": 16000, "sha256": digest}
                )
                return max_lag

            async def consume():
                for sequence, offset in enumerate(range(0, len(pcm), 640)):
                    message = await ws.receive(timeout=30)
                    if (
                        message.type != aiohttp.WSMsgType.BINARY
                        or message.data
                        != HEADER.pack(sequence, offset // 2)
                        + pcm[offset : offset + 640]
                    ):
                        raise ValueError("echo differs from sent PCM")
                return await ws.receive_json(timeout=30)

            tasks = [asyncio.create_task(produce()), asyncio.create_task(consume())]
            try:
                lag, done = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            if (
                done.get("status") != "completed"
                or done.get("sha256") != digest
                or done.get("samples") != 16000
                or done.get("chunks") != 50
            ):
                raise ValueError("incomplete echo")
            return {
                "status": "completed",
                "protocol": "pcm-echo/v1",
                "unauthorized_status": unauthorized,
                "sample_count": 16000,
                "chunks": 50,
                "sha256": digest,
                "maximum_send_lag_ns": lag,
                "server_done": done,
            }


def _native_factory(snapshot: dict, captured: bytearray):
    corpus = _corpus()
    manifest = corpus.read_registration(REMOTE_ROOT)
    q = corpus._helper("infra.modal_native_cuda_qualification")
    torch, np, whisper = (
        importlib.import_module(name) for name in ("torch", "numpy", "whisper")
    )
    runtime = importlib.import_module("whisper_runtime")
    adapters = importlib.import_module("whisper_runtime.adapters")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "Tesla T4"
    ):
        raise RuntimeError("registered T4 is unavailable")
    checkpoint = corpus.b.MODEL_CHECKPOINT_PATH
    if (
        not checkpoint.is_file()
        or _sha(checkpoint.read_bytes()) != manifest["model"]["checkpoint_sha256"]
    ):
        raise RuntimeError("cached checkpoint mismatch")
    setup = time.perf_counter_ns()
    # Keep the registered named model's alignment-head configuration. The exact
    # checkpoint was checked above; its cache is mounted read-only, with network
    # access disabled. A path-only load uses different default alignment heads.
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=corpus.b.MODEL_CACHE_MOUNT
    ).eval()
    fingerprint = q._model_fingerprint(model)
    if fingerprint != manifest["model"]["model_state_sha256"] or any(
        str(t.device) != "cuda:0"
        or (t.is_floating_point() and t.dtype != torch.float32)
        for t in (*model.parameters(), *model.buffers())
    ):
        raise RuntimeError("loaded model differs from registration")
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=corpus.c._command_output(corpus.b.BACKEND_ROOT, "rev-parse", "HEAD"),
        backend="pytorch-cuda-network-replay",
        fingerprint=fingerprint,
    )
    worker = runtime.Worker(
        "network-replay",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=180,
    )

    def probe(observed):
        if observed is not model:
            raise RuntimeError("model binding changed")
        return identity

    adapter = adapters.NativeWhisperAdapter(
        worker,
        model,
        probe,
        adapters.NativeExecutionProfile(
            "tiny.en/network-replay-v1", capacity, device="cuda:0"
        ),
    )
    options = adapters.NativeDecodeOptions(**manifest["decode_options"])

    def mel_builder(pcm):
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
        ).contiguous()

    warmup_start = time.perf_counter_ns()
    session = runtime.Session("network-warmup")
    with adapter.start_window(
        session=session,
        request=runtime.RequestState(
            "warmup", session.session_id, identity, rng_seed=7
        ),
        window_id="warmup",
        mel=mel_builder(bytes(32000)),
        start_ms=0,
        end_ms=1000,
        options=options,
    ) as run:
        for _ in range(100000):
            if run.complete:
                break
            run.step()
        run.prepare_result()
    torch.cuda.synchronize(0)
    warmup_ns = time.perf_counter_ns() - warmup_start
    if budget.available != capacity or budget.lease_count or worker.queue_depth:
        raise RuntimeError("warmup retained native capacity")
    stream_config = dict(corpus.AUTOMATIC_ENDPOINTS_PROFILE["stream_config"])
    endpoints = importlib.import_module("whisper_runtime.adapters.audio_endpoints")
    stream_config["endpointing"] = endpoints.QuietEndpointConfig(
        **stream_config["endpointing"]
    )
    stream = adapters.ContinuousTranscriptStream(
        adapter,
        stream_id="network-replay",
        mel_builder=mel_builder,
        options=options,
        rng_seed=7,
        config=adapters.ContinuousStreamConfig(**stream_config),
    )
    setup_ns = time.perf_counter_ns() - setup

    def finalize():
        if budget.available != capacity or budget.lease_count or worker.queue_depth:
            raise RuntimeError("native capacity remains retained")
        controls_started = time.perf_counter_ns()
        metadata = {
            "source_snapshot": snapshot,
            "profile_id": stream.profile_id,
            "model_initial_sha256": fingerprint,
            "model_backend_revision": identity.revision,
            "capacity_restored": True,
            "setup_ns": setup_ns,
            "warmup_ns": warmup_ns,
            "controls_status": "not_run_incomplete_input",
            "worker": {
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
                "modal": str(importlib.import_module("modal").__version__),
                "gpu_name": torch.cuda.get_device_name(0),
            },
        }
        if (
            stream.metrics.accepted_samples
            == stream.metrics.committed_samples
            == SAMPLES
            and len(captured) == SAMPLES * 2
            and _sha(captured) == PCM_SHA256
        ):

            def control(content):
                torch.manual_seed(7)
                torch.cuda.manual_seed_all(7)
                audio = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768.0
                return corpus.b._offline_result(
                    model.transcribe(audio, **manifest["offline_control_options"])
                )

            full = control(captured)
            unit_plan = corpus._source_unit_plan(manifest)
            fixtures = {}
            for unit in unit_plan:
                if unit["kind"] == "fixture" and unit["fixture_id"] not in fixtures:
                    content = captured[
                        unit["start_sample"] * 2 : unit["end_sample"] * 2
                    ]
                    fixtures[unit["fixture_id"]] = control(content)
            metadata.update(
                controls_status="completed",
                offline_control=full,
                fixture_controls=fixtures,
            )
        metadata["control_ns"] = time.perf_counter_ns() - controls_started
        verification = time.perf_counter_ns()
        final_hash = q._model_fingerprint(model)
        metadata["model_final_sha256"] = final_hash
        metadata["model_unchanged"] = final_hash == fingerprint
        metadata["final_hash_ns"] = time.perf_counter_ns() - verification
        if not metadata["model_unchanged"]:
            raise RuntimeError("model state changed")
        return metadata

    return stream, finalize


def make_gpu_app(snapshot: dict):
    _verify_source(snapshot)
    server = importlib.import_module("examples.pcm_websocket")
    guard = threading.Lock()
    used = False

    async def app(scope, receive, send):
        nonlocal used
        if scope["type"] == "websocket":
            with guard:
                reject = used
                used = True
            if reject:
                await send({"type": "websocket.close", "code": 1008})
                return
        captured = bytearray()
        next_sequence = 0

        async def capture_receive():
            nonlocal next_sequence
            message = await receive()
            frame = message.get("bytes")
            if (
                isinstance(frame, bytes)
                and HEADER.size < len(frame) <= HEADER.size + 640
                and (len(frame) - HEADER.size) % 2 == 0
            ):
                sequence, start = HEADER.unpack_from(frame)
                if (
                    sequence == next_sequence
                    and start == len(captured) // 2
                    and len(captured) + len(frame) - HEADER.size <= SAMPLES * 2
                ):
                    captured.extend(frame[HEADER.size :])
                    next_sequence += 1
            return message

        handler = server.make_app(
            lambda: _native_factory(snapshot, captured),
            expected_samples=SAMPLES,
            expected_sha256=PCM_SHA256,
        )
        await handler(scope, capture_receive, send)

    return app


def _resources(modal, snapshot: dict, *, preflight: bool, pcm: bytes):
    corpus = _corpus()
    if preflight:
        image = modal.Image.debian_slim(python_version="3.13")
    else:
        q = corpus._helper("infra.modal_native_cuda_qualification")
        image = (
            modal.Image.debian_slim(python_version="3.13")
            .apt_install("ca-certificates", "ffmpeg", "git")
            .pip_install(
                "torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124"
            )
            .uv_pip_install(*q.DIRECT_IMAGE_PACKAGES)
            .run_commands(q._build_command(corpus.BASE_COMMIT))
        )
    image = image.add_local_dir(
        ROOT / "src/whisper_runtime",
        (REMOTE_ROOT / "src/whisper_runtime").as_posix(),
        copy=True,
    )
    for item in snapshot["files"]:
        if not item["path"].startswith("src/whisper_runtime/"):
            image = image.add_local_file(
                ROOT / item["path"], (REMOTE_ROOT / item["path"]).as_posix(), copy=True
            )
    image = image.env(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime",
            "WHISPER_MODAL_ENABLE_WORD_CORPUS": "0",
            "WHISPER_MODAL_ENABLE_REMOTE_RESOURCES": "0",
        }
    )
    import uuid

    label = f"wr-net-{'cpu' if preflight else 'gpu'}-{uuid.uuid4().hex[:16]}"
    app = modal.App(label)
    # Modal 1.5.5 rejects the retries argument for ASGI, including retries=0.
    # Omit it; the client still makes one connection without a reconnect loop.
    options = dict(
        image=image,
        serialized=True,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=2,
        startup_timeout=180,
        single_use_containers=True,
        block_network=True,
        restrict_modal_access=True,
        include_source=False,
        timeout=30 if preflight else 180,
        cpu=0.25 if preflight else 2,
        memory=256 if preflight else 4096,
    )
    if not preflight:
        volume = modal.Volume.from_name(
            corpus.b.MODEL_CACHE_NAME, create_if_missing=False
        )
        options.update(
            gpu="T4",
            cloud="aws",
            region="us-west",
            volumes={
                corpus.b.MODEL_CACHE_MOUNT: volume.with_mount_options(read_only=True)
            },
        )
    expected_echo_sha = _sha(pcm[:32000])

    @app.function(**options)
    @modal.asgi_app(requires_proxy_auth=True, label=label)
    def endpoint():
        module = importlib.import_module("infra.modal_network_replay")
        return (
            module.make_echo_app(snapshot, expected_echo_sha)
            if preflight
            else module.make_gpu_app(snapshot)
        )

    return app, endpoint


def _attempt(
    *, modal, snapshot: dict, pcm: bytes, preflight: bool, resource_factory=_resources
) -> dict:
    cleanup = {"ephemeral_context_exit_completed": False, "proxy_token_deleted": False}
    record: dict = {"status": "failed", "cleanup": cleanup}
    manager = modal.Workspace.from_context().proxy_tokens
    token = None
    try:
        # ASGI resource validation is local. Do it before creating a credential.
        app, endpoint = resource_factory(modal, snapshot, preflight=preflight, pcm=pcm)
        environment = modal.Environment.from_context()
        environment.hydrate()
        if not isinstance(environment.name, str) or not environment.name:
            raise ValueError("environment unavailable")
        token = manager.create()
        own_info = next(
            item for item in manager.list() if item.token_id == token.token_id
        )
        record["proxy_token_scoped"] = bool(own_info.scoped)
        if own_info.scoped:
            manager.allow(token.token_id, environment.name)
        with app.run(detach=False, environment_name=environment.name):
            try:
                record["app_id"] = app.app_id
                url = _websocket_url(endpoint.get_web_url())
                headers = {
                    "Modal-Key": token.token_id,
                    "Modal-Secret": token.token_secret,
                }
                if preflight:
                    result = asyncio.run(_cpu_preflight(url, pcm, headers))
                else:
                    client = importlib.import_module("examples.replay_websocket")
                    result = asyncio.run(client.replay(url, pcm, headers=headers))
                record["result"] = result
                record["status"] = result.get("status", "failed")
            except BaseException as error:
                record["status"] = (
                    "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
                )
                record["error_code"] = "network_attempt_failed"
                record["error_type"] = type(error).__name__
        cleanup["ephemeral_context_exit_completed"] = True
    except BaseException as error:
        record["status"] = (
            "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        record["error_code"] = "network_attempt_failed"
        record["error_type"] = type(error).__name__
    finally:
        if token is not None:
            try:
                manager.delete(token.token_id)
                cleanup["proxy_token_deleted"] = True
            except BaseException as error:
                record["status"] = "failed"
                record["error_code"] = "proxy_token_cleanup_failed"
                record["cleanup_error_type"] = type(error).__name__
    if record["status"] == "completed" and not all(cleanup.values()):
        record["status"] = "failed"
        record["error_code"] = "incomplete_cleanup"
    return record


def _preflight_valid(record: dict, snapshot: dict, pcm: bytes) -> bool:
    result = record.get("result", {})
    cleanup = record.get("cleanup", {})
    return (
        record.get("status") == "completed"
        and record.get("preflight") is True
        and record.get("source") == snapshot
        and cleanup.get("ephemeral_context_exit_completed") is True
        and cleanup.get("proxy_token_deleted") is True
        and result.get("status") == "completed"
        and result.get("protocol") == "pcm-echo/v1"
        and type(result.get("unauthorized_status")) is int
        and result["unauthorized_status"] in {401, 403}
        and type(result.get("sample_count")) is int
        and result["sample_count"] == 16000
        and result.get("chunks") == 50
        and result.get("sha256") == _sha(pcm[:32000])
        and result.get("server_done", {}).get("source_snapshot") == snapshot
    )


def _validate_completed(record: dict, snapshot: dict, preflight: bool) -> None:
    result = record["result"]
    done = result["server_done"]
    metadata = done if preflight else done["metadata"]
    if metadata.get("source_snapshot") != snapshot:
        raise ValueError("server source differs from local snapshot")
    if not preflight:
        corpus = _corpus()
        manifest = corpus.read_registration()
        expected_hash = manifest["model"]["model_state_sha256"]
        metrics = done["metrics"]
        if (
            done.get("status") != "completed"
            or result.get("sample_count") != SAMPLES
            or result.get("sha256") != PCM_SHA256
            or metrics.get("accepted_samples") != SAMPLES
            or metrics.get("committed_samples") != SAMPLES
            or metrics.get("buffered_samples") != 0
            or metrics.get("accepted_sha256") != PCM_SHA256
            or metadata.get("capacity_restored") is not True
            or metadata.get("model_unchanged") is not True
            or metadata.get("model_initial_sha256") != expected_hash
            or metadata.get("model_final_sha256") != expected_hash
            or metadata.get("controls_status") != "completed"
            or metadata.get("profile_id")
            != corpus.AUTOMATIC_ENDPOINTS_PROFILE["profile_id"]
        ):
            raise ValueError("native validation did not complete")
        text = corpus.c._normalized_committed_text(
            [entry["event"] for entry in result["events"]]
        )
        case = next(
            item
            for item in manifest["cases"]
            if item["id"] == "three-speakers-repeat-pauses"
        )
        fixture_text = " ".join(
            metadata["fixture_controls"][unit["fixture_id"]]["text"]
            for unit in corpus._source_unit_plan(manifest)
            if unit["kind"] == "fixture"
        )
        record["recognition"] = {
            "text": text,
            "against_human_reference": corpus.b._word_difference(
                text, case["reference_text"]
            ),
            "against_offline_control": corpus.b._word_difference(
                text, metadata["offline_control"]["text"]
            ),
            "against_fixture_controls": corpus.b._word_difference(text, fixture_text),
        }


def run(
    *,
    replay_id: str,
    preflight: bool = False,
    confirm_paid_gpu: bool = False,
    root: Path = ROOT,
) -> Path:
    if not preflight and confirm_paid_gpu is not True:
        raise ValueError("GPU replay requires --confirm-paid-gpu")
    registration = read_registration(root)
    output, journal = _paths(root, replay_id, preflight)
    if output.exists() or journal.exists():
        raise FileExistsError("this network attempt already exists")
    snapshot, pcm = source_snapshot(root), _input(root)
    if not preflight:
        probe, _ = _paths(root, replay_id, True)
        previous = json.loads(probe.read_text(encoding="utf-8"))
        if not _preflight_valid(previous, snapshot, pcm):
            raise ValueError("successful matching CPU preflight is required")
    _journal(
        journal,
        "attempt-started",
        first=True,
        preflight=preflight,
        source_commit=snapshot["commit"],
        source_digest=snapshot["digest"],
    )
    try:
        modal = importlib.import_module("modal")
        if str(modal.__version__) != SDK_VERSION:
            raise ValueError("registered Modal SDK required")
        record = _attempt(modal=modal, snapshot=snapshot, pcm=pcm, preflight=preflight)
        record.update(
            schema_version="1-diagnostic",
            registration=registration,
            source=snapshot,
            preflight=preflight,
        )
        if record["status"] == "completed":
            try:
                _validate_completed(record, snapshot, preflight)
            except (ValueError, KeyError, TypeError) as error:
                record["status"] = "failed"
                record["error_code"] = "worker_validation_failed"
                record["validation_error_type"] = type(error).__name__
        _write_result(output, record)
    except BaseException as error:
        _journal(journal, "attempt-failed", error_type=type(error).__name__)
        raise RuntimeError(
            "network diagnostic failed; inspect the attempt receipt"
        ) from None
    _journal(
        journal,
        "record-written",
        status=record["status"],
        record_sha256=_sha(output.read_bytes()),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    args = parser.parse_args()
    try:
        output = run(**vars(args))
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"{type(error).__name__}: diagnostic did not complete\n")
    print(output)


if __name__ == "__main__":
    main()
