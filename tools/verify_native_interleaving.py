"""Verify early-cleanup isolation for two staged decodes on one Whisper model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from smoke_native_whisper import (
    fingerprint_loaded_model,
    verify_loaded_model_fingerprint,
    verify_source_revision,
)

ROOT = Path(__file__).resolve().parents[1]
PINNED_WHISPER_BASE = "86098128c0b4f24f0e2aa2994de830614b474227"
PINNED_WHISPER_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
EXPECTED_ASSERTIONS = frozenset(
    {
        "two_live_runs",
        "request_local_cache_path",
        "state_objects_distinct",
        "kv_cache_storage_disjoint",
        "both_runs_stepped_before_cancellation",
        "cancelled_cleanup_idempotent",
        "cancelled_state_released",
        "cancelled_run_rejects_step",
        "cancelled_run_rejects_finalize",
        "survivor_cache_unchanged_by_cancellation",
        "survivor_matches_isolated_baseline",
        "survivor_cache_cleanup_complete",
        "survivor_rejects_second_finalize",
        "model_reusable_after_cleanup",
        "model_state_unchanged",
        "model_hooks_unchanged",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two staged decodes on one model, cancel one, and verify the other."
        )
    )
    parser.add_argument("audio", help="Path to an audio file accepted by Whisper")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument(
        "--input-label",
        required=True,
        help="Portable repository-relative label for the audio input",
    )
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--download-root")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--expected-text")
    parser.add_argument(
        "--expected-model-fingerprint",
        help="Expected sha256:<hex> fingerprint of the loaded model state",
    )
    parser.add_argument(
        "--numeric-absolute-tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for scalar result comparisons",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(command: str) -> str:
    result = subprocess.run(
        [command, "-version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or not lines or not lines[0].strip():
        raise RuntimeError(f"{command} version information is unavailable")
    return lines[0].strip()


def tensor_fingerprint(tensor: object) -> str:
    """Hash a tensor or array together with its dtype and shape."""

    detach = getattr(tensor, "detach", None)
    if callable(detach):
        value = detach().cpu()
        layout = str(getattr(value, "layout", ""))
        if layout and layout != "torch.strided":
            to_dense = getattr(value, "to_dense", None)
            if not callable(to_dense):
                raise TypeError(f"the tensor layout cannot be hashed: {layout}")
            value = to_dense()
        value = value.contiguous()
    else:
        value = tensor
        layout = ""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    to_bytes = getattr(value, "tobytes", None)
    if callable(to_bytes):
        payload = to_bytes(order="C")
    else:
        numpy = getattr(value, "numpy", None)
        if not callable(numpy):
            raise TypeError("the value cannot be converted to contiguous bytes")
        payload = numpy().tobytes(order="C")

    digest = hashlib.sha256()
    digest.update(layout.encode("utf-8"))
    digest.update(str(dtype).encode("utf-8"))
    digest.update(repr(tuple(shape)).encode("utf-8"))
    digest.update(payload)
    return digest.hexdigest()


def tensor_storage_identity(tensor: object) -> tuple[str, int]:
    """Return the device and base storage address for a populated tensor."""

    storage = getattr(tensor, "untyped_storage", None)
    if not callable(storage):
        raise TypeError("the tensor does not expose untyped_storage()")
    data_ptr = storage().data_ptr()
    if not data_ptr:
        raise RuntimeError("a populated decoder cache has no storage address")
    return str(getattr(tensor, "device", "")), data_ptr


def model_execution_state_fingerprint(model: object) -> str:
    """Hash all parameters and buffers, including non-persistent buffers."""

    digest = hashlib.sha256()
    for category, attribute in (
        ("parameter", "named_parameters"),
        ("buffer", "named_buffers"),
    ):
        values = getattr(model, attribute, None)
        if not callable(values):
            raise TypeError(f"the loaded model must provide {attribute}()")
        for name, value in values():
            digest.update(category.encode("utf-8"))
            digest.update(str(name).encode("utf-8"))
            digest.update(tensor_fingerprint(value).encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def git_source(root: Path) -> dict[str, object]:
    """Return a clean, content-addressed Git source description."""

    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git metadata is unavailable for {root.name}")
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"the {root.name} source worktree is not clean")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def source_tree_for_module(module_file: str) -> str:
    source_root = Path(module_file).resolve().parent.parent
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD^{tree}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError("the imported Whisper source tree is unavailable")
    return result.stdout.strip()


def verify_base_revision(
    module_file: str,
    base_revision: str,
    applied_revision: str,
) -> None:
    """Require the declared base to be the pinned ancestor of the backend."""

    if base_revision != PINNED_WHISPER_BASE:
        raise RuntimeError(
            "the declared Whisper base does not match the pinned integration base"
        )
    source_root = Path(module_file).resolve().parent.parent
    result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "merge-base",
            "--is-ancestor",
            base_revision,
            applied_revision,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "the pinned Whisper base is not an ancestor of the applied revision"
        )


def count_model_hooks(model: object) -> dict[str, int]:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("the loaded model must provide named_modules()")
    counts = {"forward": 0, "forward_pre": 0, "backward": 0}
    for _name, module in named_modules():
        counts["forward"] += len(getattr(module, "_forward_hooks", {}))
        counts["forward_pre"] += len(getattr(module, "_forward_pre_hooks", {}))
        counts["backward"] += len(getattr(module, "_backward_hooks", {}))
    return counts


def model_hook_fingerprint(model: object) -> tuple[tuple[str, str, str, int], ...]:
    """Capture the identity of every registered module hook."""

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("the loaded model must provide named_modules()")
    records: list[tuple[str, str, str, int]] = []
    hook_sets = (
        ("forward", "_forward_hooks"),
        ("forward_pre", "_forward_pre_hooks"),
        ("backward", "_backward_hooks"),
    )
    for module_name, module in named_modules():
        for category, attribute in hook_sets:
            hooks = getattr(module, attribute, {})
            records.extend(
                (module_name, category, repr(key), id(callback))
                for key, callback in hooks.items()
            )
    return tuple(sorted(records))


def _equal_number(left: object, right: object, absolute_tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    if math.isnan(float(left)) or math.isnan(float(right)):
        return math.isnan(float(left)) and math.isnan(float(right))
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    )


def decode_results_match(
    baseline: object,
    candidate: object,
    *,
    absolute_tolerance: float,
) -> bool:
    """Compare every public result field used by the staged check."""

    for field in ("text", "tokens", "language"):
        if getattr(baseline, field, None) != getattr(candidate, field, None):
            return False
    for field in (
        "temperature",
        "compression_ratio",
        "avg_logprob",
        "no_speech_prob",
    ):
        if not _equal_number(
            getattr(baseline, field, None),
            getattr(candidate, field, None),
            absolute_tolerance,
        ):
            return False

    baseline_features = getattr(baseline, "audio_features", None)
    candidate_features = getattr(candidate, "audio_features", None)
    baseline_equal = getattr(baseline_features, "equal", None)
    if callable(baseline_equal):
        return bool(baseline_equal(candidate_features))
    return tensor_fingerprint(baseline_features) == tensor_fingerprint(
        candidate_features
    )


def _json_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("the decode result contains a non-numeric scalar")
    number = float(value)
    return None if math.isnan(number) else number


def result_record(result: object) -> dict[str, object]:
    tokens = getattr(result, "tokens", None)
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        raise TypeError("the decode result tokens must be a list of integers")
    text = getattr(result, "text", None)
    language = getattr(result, "language", None)
    if not isinstance(text, str) or not isinstance(language, str):
        raise TypeError("the decode result must contain text and language")

    token_payload = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return {
        "text": text,
        "language": language,
        "token_count": len(tokens),
        "tokens_sha256": hashlib.sha256(token_payload).hexdigest(),
        "audio_features_sha256": tensor_fingerprint(
            getattr(result, "audio_features", None)
        ),
        "temperature": _json_number(getattr(result, "temperature", None)),
        "compression_ratio": _json_number(getattr(result, "compression_ratio", None)),
        "avg_logprob": _json_number(getattr(result, "avg_logprob", None)),
        "no_speech_prob": _json_number(getattr(result, "no_speech_prob", None)),
    }


def verify_passed_assertions(assertions: Mapping[str, object]) -> None:
    missing = EXPECTED_ASSERTIONS - set(assertions)
    unexpected = set(assertions) - EXPECTED_ASSERTIONS
    if missing:
        raise RuntimeError(f"missing check assertions: {', '.join(sorted(missing))}")
    if unexpected:
        raise RuntimeError(
            f"unexpected check assertions: {', '.join(sorted(unexpected))}"
        )
    failed = sorted(name for name, passed in assertions.items() if passed is not True)
    if failed:
        raise RuntimeError(f"check assertions failed: {', '.join(failed)}")


def _expect_rejection(operation: Callable[[], object], phase: str) -> str:
    try:
        operation()
    except RuntimeError as error:
        message = str(error)
        if phase not in message:
            raise RuntimeError(
                f"terminal operation failed for the wrong reason: {message}"
            ) from error
        return message
    raise RuntimeError(f"a {phase} decode run accepted another operation")


def _finish(run: object) -> object:
    if not run.complete:
        run.prefill()
    while not run.complete:
        run.step()
    results = run.finalize()
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("the staged decoder did not return exactly one result")
    run.cleanup()
    return results[0]


def _checkpoint_path(model_name: str, download_root: str | None) -> Path:
    root = (
        Path(download_root).expanduser()
        if download_root is not None
        else Path.home() / ".cache" / "whisper"
    )
    path = root / f"{model_name}.pt"
    if not path.is_file():
        raise RuntimeError(f"the downloaded model checkpoint is unavailable: {path}")
    return path


def verify_audio_manifest_binding(
    *,
    fixture_id: str,
    file_sha256: str,
    size_bytes: int,
    sample_rate_hz: int,
    sample_count: int,
) -> None:
    """Bind executed audio bytes to the repository provenance manifest."""

    manifest_path = ROOT / "conformance" / "audio-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("the audio provenance manifest is unavailable") from error
    fixtures = manifest.get("fixtures", []) if isinstance(manifest, dict) else []
    record = next(
        (
            item
            for item in fixtures
            if isinstance(item, dict) and item.get("id") == fixture_id
        ),
        None,
    )
    if record is None:
        raise RuntimeError(f"the audio fixture is not declared: {fixture_id}")
    expected = {
        "sha256": file_sha256,
        "size_bytes": size_bytes,
        "decoded_sample_rate_hz": sample_rate_hz,
        "decoded_sample_count": sample_count,
    }
    mismatches = [name for name, value in expected.items() if record.get(name) != value]
    if mismatches:
        raise RuntimeError(
            "the audio input does not match its manifest record: "
            + ", ".join(mismatches)
        )


def main() -> int:
    args = parse_args()
    if (
        not math.isfinite(args.numeric_absolute_tolerance)
        or args.numeric_absolute_tolerance < 0
    ):
        raise ValueError("--numeric-absolute-tolerance must be finite and non-negative")
    input_label = Path(args.input_label)
    if (
        input_label.is_absolute()
        or input_label.drive
        or ".." in input_label.parts
        or not input_label.parts
    ):
        raise ValueError("--input-label must be a safe repository-relative path")

    import numpy
    import torch
    import whisper
    from whisper.audio import SAMPLE_RATE
    from whisper.decoding import DecodingOptions, DecodingTask, PyTorchInference

    torch.set_num_threads(1)
    revision = verify_source_revision(whisper.__file__, args.revision)
    verify_base_revision(whisper.__file__, args.base_revision, revision)
    observed_tree = source_tree_for_module(whisper.__file__)
    if observed_tree != PINNED_WHISPER_TREE:
        raise RuntimeError(
            "the imported Whisper tree does not match the pinned patch series: "
            f"expected {PINNED_WHISPER_TREE}, observed {observed_tree}"
        )
    runtime_source = git_source(ROOT)

    model = whisper.load_model(
        args.model,
        device="cpu",
        download_root=args.download_root,
    ).eval()
    model_fingerprint_before = fingerprint_loaded_model(model)
    execution_state_before = model_execution_state_fingerprint(model)
    verify_loaded_model_fingerprint(
        model_fingerprint_before, args.expected_model_fingerprint
    )
    hooks_before = count_model_hooks(model)
    hook_fingerprint_before = model_hook_fingerprint(model)

    audio_path = Path(args.audio)
    audio_file_sha256 = sha256_file(audio_path)
    audio_size_bytes = audio_path.stat().st_size
    audio = whisper.load_audio(str(audio_path))
    if len(audio) < 2:
        raise ValueError("the audio fixture must contain at least two samples")
    verify_audio_manifest_binding(
        fixture_id=args.fixture_id,
        file_sha256=audio_file_sha256,
        size_bytes=audio_size_bytes,
        sample_rate_hz=SAMPLE_RATE,
        sample_count=len(audio),
    )
    split = len(audio) // 2
    cancelled_audio = audio[:split]
    survivor_audio = audio
    cancelled_mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(cancelled_audio), n_mels=model.dims.n_mels
    )
    survivor_mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(survivor_audio), n_mels=model.dims.n_mels
    )
    options = DecodingOptions(
        language="en" if args.model.endswith(".en") else None,
        temperature=0.0,
        without_timestamps=True,
        fp16=False,
    )

    total_started = time.perf_counter()
    baseline_started = time.perf_counter()
    baseline_run = DecodingTask(model, options)._start_run(survivor_mel.unsqueeze(0))
    try:
        baseline = _finish(baseline_run)
    finally:
        baseline_run.cleanup()
    baseline_elapsed = time.perf_counter() - baseline_started

    cancelled_run = None
    survivor_run = None
    schedule: list[str] = []
    cancelled_step_rejection = ""
    cancelled_finalize_rejection = ""
    survivor_finalize_rejection = ""
    interleaving_started = time.perf_counter()
    try:
        shared_task = DecodingTask(model, options)
        cancelled_run = shared_task._start_run(cancelled_mel.unsqueeze(0))
        schedule.append("start:cancelled")
        survivor_run = shared_task._start_run(survivor_mel.unsqueeze(0))
        schedule.append("start:survivor")

        if cancelled_run.complete or survivor_run.complete:
            raise RuntimeError("both staged runs must be live before interleaving")
        if not isinstance(cancelled_run.inference, PyTorchInference) or not isinstance(
            survivor_run.inference, PyTorchInference
        ):
            raise RuntimeError("the check requires the built-in PyTorch decoder")
        if (
            cancelled_run.inference._use_legacy_cache
            or survivor_run.inference._use_legacy_cache
        ):
            raise RuntimeError("the check requires the request-local cache path")

        state_objects_distinct = all(
            (
                cancelled_run is not survivor_run,
                cancelled_run.inference is not survivor_run.inference,
                cancelled_run.decoder is not survivor_run.decoder,
                cancelled_run.inference.kv_cache is not survivor_run.inference.kv_cache,
                cancelled_run.audio_features.data_ptr()
                != survivor_run.audio_features.data_ptr(),
                cancelled_run.tokens.data_ptr() != survivor_run.tokens.data_ptr(),
                cancelled_run.sum_logprobs.data_ptr()
                != survivor_run.sum_logprobs.data_ptr(),
                cancelled_run.no_speech_probs is not survivor_run.no_speech_probs,
            )
        )
        if not state_objects_distinct:
            raise RuntimeError("the two staged runs share mutable state")

        cancelled_run.prefill()
        schedule.append("prefill:cancelled")
        survivor_run.prefill()
        schedule.append("prefill:survivor")
        if (
            cancelled_run._pending_logits is None
            or survivor_run._pending_logits is None
            or cancelled_run._pending_logits.data_ptr()
            == survivor_run._pending_logits.data_ptr()
        ):
            raise RuntimeError("the two staged runs share pending decoder logits")
        if cancelled_run.step():
            raise RuntimeError("the cancelled candidate completed before cancellation")
        schedule.append("step:cancelled:1")
        if survivor_run.step():
            raise RuntimeError(
                "the survivor completed before interleaving was observed"
            )
        schedule.append("step:survivor:1")

        cancelled_inference = cancelled_run.inference
        survivor_inference = survivor_run.inference
        if not cancelled_inference.kv_cache or not survivor_inference.kv_cache:
            raise RuntimeError("both request-local caches must be populated")
        cancelled_cache_storage = {
            tensor_storage_identity(value)
            for value in cancelled_inference.kv_cache.values()
        }
        survivor_cache_storage = {
            tensor_storage_identity(value)
            for value in survivor_inference.kv_cache.values()
        }
        kv_cache_storage_disjoint = cancelled_cache_storage.isdisjoint(
            survivor_cache_storage
        )
        if not kv_cache_storage_disjoint:
            raise RuntimeError("the two staged runs share decoder cache storage")
        survivor_cache_snapshot = {
            key: value.clone() for key, value in survivor_inference.kv_cache.items()
        }
        survivor_cache_keys = tuple(survivor_inference.kv_cache)

        cancelled_run.cleanup()
        schedule.append("cancel:cancelled")
        cancelled_run.cleanup()
        schedule.append("cleanup:cancelled:idempotent")
        if cancelled_inference.kv_cache:
            raise RuntimeError("the cancelled run retained cache entries")
        cancelled_state_released = all(
            getattr(cancelled_run, field, None) is None
            for field in (
                "audio_features",
                "tokens",
                "sum_logprobs",
                "inference",
                "decoder",
                "_pending_logits",
            )
        )
        if not cancelled_state_released:
            raise RuntimeError("the cancelled run retained request-owned state")
        survivor_cache_unchanged = tuple(
            survivor_inference.kv_cache
        ) == survivor_cache_keys and all(
            torch.equal(survivor_inference.kv_cache[key], value)
            for key, value in survivor_cache_snapshot.items()
        )
        if not survivor_cache_unchanged:
            raise RuntimeError("cancelling one run changed the survivor cache")

        cancelled_step_rejection = _expect_rejection(cancelled_run.step, "cancelled")
        cancelled_finalize_rejection = _expect_rejection(
            cancelled_run.finalize, "cancelled"
        )
        while not survivor_run.complete:
            survivor_run.step()
            schedule.append(f"step:survivor:{survivor_run.step_index}")
        survivor_results = survivor_run.finalize()
        schedule.append("finalize:survivor")
        if not isinstance(survivor_results, list) or len(survivor_results) != 1:
            raise RuntimeError("the survivor did not return exactly one result")
        survivor = survivor_results[0]
        if survivor_inference.kv_cache:
            raise RuntimeError("the survivor retained cache entries after completion")
        survivor_finalize_rejection = _expect_rejection(
            survivor_run.finalize, "finalized"
        )
        survivor_run.cleanup()
        schedule.append("cleanup:survivor:idempotent")
    finally:
        if cancelled_run is not None:
            cancelled_run.cleanup()
        if survivor_run is not None:
            survivor_run.cleanup()
    interleaving_elapsed = time.perf_counter() - interleaving_started

    if not decode_results_match(
        baseline,
        survivor,
        absolute_tolerance=args.numeric_absolute_tolerance,
    ):
        raise RuntimeError("the surviving run differs from its isolated baseline")
    if args.expected_text is not None and survivor.text != args.expected_text:
        raise RuntimeError(
            "the surviving transcript does not match --expected-text: "
            f"expected {args.expected_text!r}, observed {survivor.text!r}"
        )

    reuse_started = time.perf_counter()
    reuse_run = DecodingTask(model, options)._start_run(survivor_mel.unsqueeze(0))
    try:
        reuse = _finish(reuse_run)
    finally:
        reuse_run.cleanup()
    reuse_elapsed = time.perf_counter() - reuse_started
    reusable = decode_results_match(
        baseline,
        reuse,
        absolute_tolerance=args.numeric_absolute_tolerance,
    )
    if not reusable:
        raise RuntimeError("the model changed after staged run cleanup")

    model_fingerprint_after = fingerprint_loaded_model(model)
    execution_state_after = model_execution_state_fingerprint(model)
    hooks_after = count_model_hooks(model)
    hook_fingerprint_after = model_hook_fingerprint(model)
    assertions = {
        "two_live_runs": True,
        "request_local_cache_path": True,
        "state_objects_distinct": state_objects_distinct,
        "kv_cache_storage_disjoint": kv_cache_storage_disjoint,
        "both_runs_stepped_before_cancellation": True,
        "cancelled_cleanup_idempotent": True,
        "cancelled_state_released": cancelled_state_released,
        "cancelled_run_rejects_step": "cancelled" in cancelled_step_rejection,
        "cancelled_run_rejects_finalize": "cancelled" in cancelled_finalize_rejection,
        "survivor_cache_unchanged_by_cancellation": survivor_cache_unchanged,
        "survivor_matches_isolated_baseline": True,
        "survivor_cache_cleanup_complete": not survivor_inference.kv_cache,
        "survivor_rejects_second_finalize": "finalized" in survivor_finalize_rejection,
        "model_reusable_after_cleanup": reusable,
        "model_state_unchanged": (
            model_fingerprint_after == model_fingerprint_before
            and execution_state_after == execution_state_before
        ),
        "model_hooks_unchanged": (
            hooks_after == hooks_before
            and hook_fingerprint_after == hook_fingerprint_before
        ),
    }
    verify_passed_assertions(assertions)

    if (
        sha256_file(audio_path) != audio_file_sha256
        or audio_path.stat().st_size != audio_size_bytes
    ):
        raise RuntimeError("the audio input changed while the check was running")

    checkpoint = _checkpoint_path(args.model, args.download_root)
    manifest = ROOT / "patches" / "openai-whisper" / "SHA256SUMS"
    record = {
        "schema_version": "1",
        "recorded_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "passed",
        "scope": "patched_backend",
        "runtime": {
            "version": importlib.metadata.version("whisper-execution-runtime"),
            **runtime_source,
        },
        "backend": {
            "name": "openai-whisper-suspendable",
            "base_commit": args.base_revision,
            "applied_commit": revision,
            "git_tree": observed_tree,
            "clean": True,
            "patch_manifest": "patches/openai-whisper/SHA256SUMS",
            "patch_manifest_sha256": sha256_file(manifest),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "tiktoken": importlib.metadata.version("tiktoken"),
            "numba": importlib.metadata.version("numba"),
            "tqdm": importlib.metadata.version("tqdm"),
            "more_itertools": importlib.metadata.version("more-itertools"),
            "jsonschema": importlib.metadata.version("jsonschema"),
            "ffmpeg": command_version("ffmpeg"),
            "cpu_threads": torch.get_num_threads(),
        },
        "model": {
            "name": args.model,
            "device": "cpu",
            "checkpoint_sha256": sha256_file(checkpoint),
            "loaded_state_before": model_fingerprint_before,
            "loaded_state_after": model_fingerprint_after,
            "execution_state_before": execution_state_before,
            "execution_state_after": execution_state_after,
        },
        "input": {
            "fixture_id": args.fixture_id,
            "path": input_label.as_posix(),
            "file_sha256": audio_file_sha256,
            "size_bytes": audio_size_bytes,
            "sample_rate_hz": SAMPLE_RATE,
            "source_sample_count": len(audio),
            "cancelled": {
                "sample_start": 0,
                "sample_end": split,
                "pcm_sha256": tensor_fingerprint(cancelled_audio),
                "mel_sha256": tensor_fingerprint(cancelled_mel),
            },
            "survivor": {
                "sample_start": 0,
                "sample_end": len(audio),
                "pcm_sha256": tensor_fingerprint(survivor_audio),
                "mel_sha256": tensor_fingerprint(survivor_mel),
            },
        },
        "execution": {
            "mode": "deterministic_sequential_interleaving",
            "two_overlapping_run_lifetimes": True,
            "parallel_kernels": False,
            "cancellation": "explicit_cleanup_after_token_step",
            "schedule": schedule,
            "cancelled_steps": 1,
            "survivor_steps": survivor_run.step_index,
            "survivor_cache_entries_at_cancellation": len(survivor_cache_snapshot),
            "numeric_absolute_tolerance": args.numeric_absolute_tolerance,
            "elapsed_seconds": {
                "baseline": baseline_elapsed,
                "interleaving": interleaving_elapsed,
                "reuse_control": reuse_elapsed,
                "total": time.perf_counter() - total_started,
            },
            "timing_is_benchmark": False,
        },
        "assertions": assertions,
        "results": {
            "isolated_baseline": result_record(baseline),
            "survivor": result_record(survivor),
            "reuse_control": result_record(reuse),
        },
    }
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
