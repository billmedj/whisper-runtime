"""Compare independent CPU decodes with one same-window prompt re-decode.

Both audio and the model checkpoint must already exist. This diagnostic never
downloads a model and its single-pass timings are not a performance benchmark.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from native_backend_setup import (
    BACKEND_BASE_COMMIT,
    BACKEND_PATCHED_TREE,
    sha256_file,
)
from smoke_native_whisper import (
    fingerprint_loaded_model,
    verify_terminal_invariants,
)
from verify_native_interleaving import source_tree_for_module, verify_base_revision

from whisper_runtime import (
    Budget,
    ModelSnapshot,
    RequestState,
    ResourceVector,
    Session,
    Worker,
)
from whisper_runtime.adapters import (
    NativeDecodeOptions,
    NativeExecutionProfile,
    NativeWhisperAdapter,
    NativeWindowResult,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Existing audio, at most 30 seconds")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Applied Whisper Git commit")
    parser.add_argument("--prompt-a", default="An address to the American people.")
    parser.add_argument("--prompt-b", default="A speech about citizenship and service.")
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--sample-len", type=int, default=96)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--profile", choices=("greedy", "beam", "sample"), default="greedy"
    )
    parser.add_argument("--timestamps", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def decode_options(profile: str, *, prompt: str, sample_len: int, timestamps: bool):
    """Keep both contexts on one declared decode profile."""

    if profile not in ("greedy", "beam", "sample"):
        raise ValueError("unknown decode profile")
    return NativeDecodeOptions(
        language="en",
        prompt=prompt,
        sample_len=sample_len,
        temperature=0.4 if profile == "sample" else 0.0,
        beam_size=2 if profile == "beam" else None,
        best_of=2 if profile == "sample" else None,
        without_timestamps=not timestamps,
    )


def result_record(result: NativeWindowResult) -> dict[str, object]:
    """Require scalar provenance and copy every public result field."""

    metadata = result.metadata
    require(metadata is not None, "the real backend omitted decode metadata")
    require(metadata.avg_logprob is not None, "the backend omitted avg_logprob")
    require(metadata.no_speech_prob is not None, "the backend omitted no_speech_prob")
    require(isinstance(metadata.tokens, tuple), "result tokens are not immutable")
    require(result.__dataclass_params__.frozen, "the result is not frozen")
    require(metadata.__dataclass_params__.frozen, "result metadata is not frozen")
    return asdict(result)


def verify_revision(module_file: str, expected: str) -> str:
    """Verify tracked source while allowing unused, pre-existing bytecode caches."""

    module = Path(module_file).resolve()
    root = module.parent.parent

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    require(
        Path(git("rev-parse", "--show-toplevel")).resolve() == root,
        "the imported Whisper source is not a standalone Git worktree",
    )
    git("ls-files", "--error-unmatch", "--", module.relative_to(root).as_posix())
    revision = git("rev-parse", "HEAD")
    require(revision == expected, "the imported Whisper commit differs from --revision")
    require(
        not git("status", "--porcelain", "--untracked-files=all"),
        "the imported Whisper worktree has source changes",
    )
    ignored = git(
        "ls-files", "--others", "--ignored", "--exclude-standard", "--", "whisper"
    )
    for filename in ignored.splitlines():
        path = Path(filename)
        require(
            path.parent.name == "__pycache__" and path.suffix == ".pyc",
            "the Whisper package contains ignored non-bytecode files",
        )
    return revision


def verify(args: argparse.Namespace) -> dict[str, object]:
    if not 1 <= args.sample_len <= 224:
        raise ValueError("--sample-len must be between 1 and 224")
    if not 1 <= args.threads <= 8:
        raise ValueError("--threads must be between 1 and 8")
    if args.prompt_a == args.prompt_b:
        raise ValueError("the two prompts must differ")
    audio_path = args.audio.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    for label, path in (("audio", audio_path), ("checkpoint", checkpoint)):
        if not path.is_file():
            raise ValueError(f"the existing {label} file was not found: {path}")

    import torch

    # Do not execute or delete bytecode caches left by earlier diagnostics.
    # Whisper imports its model/audio/decoding modules eagerly from this scope.
    previous_prefix = sys.pycache_prefix
    with tempfile.TemporaryDirectory(prefix="whisper-prompt-reuse-") as cache:
        sys.pycache_prefix = cache
        try:
            import whisper
            from whisper.audio import SAMPLE_RATE
        finally:
            sys.pycache_prefix = previous_prefix

    torch.set_num_threads(args.threads)
    revision = verify_revision(whisper.__file__, args.revision)
    verify_base_revision(whisper.__file__, BACKEND_BASE_COMMIT, revision)
    tree = source_tree_for_module(whisper.__file__)
    require(tree == BACKEND_PATCHED_TREE, "the backend differs from the pinned tree")

    # An existing absolute path cannot take Whisper's named-model download branch.
    model = whisper.load_model(str(checkpoint), device="cpu").float().eval()
    fingerprint = fingerprint_loaded_model(model)
    identity = ModelSnapshot(checkpoint.stem, revision, "pytorch-cpu", fingerprint)

    def identity_probe(observed: object) -> ModelSnapshot:
        return ModelSnapshot(
            checkpoint.stem,
            revision,
            "pytorch-cpu",
            fingerprint_loaded_model(observed),
        )

    audio_digest = sha256_file(audio_path)
    audio = whisper.load_audio(str(audio_path))
    if not 0 < len(audio) <= 30 * SAMPLE_RATE:
        raise ValueError("audio must be nonempty and no longer than 30 seconds")
    duration_ms = max(1, round(len(audio) * 1_000 / SAMPLE_RATE))
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
    ).float()
    capacity = ResourceVector(
        memory_bytes=1_000_000_000, compute_units=1, stream_slots=1
    )
    budget = Budget(capacity)
    worker = Worker(
        "prompt-reuse-cpu",
        identity,
        budget,
        queue_capacity=1,
        transaction_ttl_seconds=180,
    )
    adapter = NativeWhisperAdapter(
        worker,
        model,
        identity_probe,
        NativeExecutionProfile(
            "prompt-reuse-cpu", capacity, reuse_decode_features=True
        ),
    )
    encoder_calls = 0

    def count_encoder(_module: object, _inputs: object, _output: object) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    def execute(label: str, prompt: str, *, second_prompt: str | None = None):
        session = Session(label)
        request = RequestState(label, label, identity, rng_seed=args.rng_seed)
        before_calls = encoder_calls
        started = time.perf_counter()
        with adapter.start_window(
            session=session,
            request=request,
            window_id="window-0",
            mel=mel,
            start_ms=0,
            end_ms=duration_ms,
            options=decode_options(
                args.profile,
                prompt=prompt,
                sample_len=args.sample_len,
                timestamps=args.timestamps,
            ),
        ) as run:
            while not run.complete:
                run.step()
            first = run.prepare_result()
            first_record = result_record(first)
            if second_prompt is not None:
                transaction = run._transaction
                require(run.decode_attempt == 1, "the first attempt was not recorded")
                run.redecode(prompt=second_prompt)
                require(run.decode_attempt == 2, "the replacement was not recorded")
                require(run._transaction is transaction, "re-decode changed lease")
                require(not run.capacity_released, "re-decode released capacity")
                require(worker.queue_depth == 1, "re-decode changed admission depth")
                require(budget.available == ResourceVector(), "lease is not held")
                require(session.snapshot().version == 0, "preview was committed")
                while not run.complete:
                    run.step()
                final = run.prepare_result()
                require(final is not first, "re-decode reused the old result object")
            else:
                final = first
            final_record = result_record(final)
            require(result_record(first) == first_record, "old provenance changed")
            state = run.finish(committed_through_ms=duration_ms)
        require(result_record(first) == first_record, "cleanup changed old provenance")
        require(state.windows[-1].result == final, "the wrong result was committed")
        require(run.capacity_released, "the completed run retained its lease")
        verify_terminal_invariants(
            request_status=request.status.value,
            session_version=state.version,
            queue_depth=worker.queue_depth,
            available=budget.available,
            capacity=capacity,
        )
        return {
            "encoder_forwards": encoder_calls - before_calls,
            "elapsed_seconds": time.perf_counter() - started,
            "first_result": first_record,
            "committed_result": final_record,
            "immutable_provenance": True,
            "resources_released": True,
        }

    hook = model.encoder.register_forward_hook(count_encoder)
    try:
        baseline_a = execute("baseline-a", args.prompt_a)
        baseline_b = execute("baseline-b", args.prompt_b)
        reused = execute("reuse-a-b", args.prompt_a, second_prompt=args.prompt_b)
    finally:
        hook.remove()

    checks = {
        "first_result_exact": baseline_a["committed_result"] == reused["first_result"],
        "second_result_exact": (
            baseline_b["committed_result"] == reused["committed_result"]
        ),
        "baseline_two_encoder_forwards": (
            baseline_a["encoder_forwards"] == baseline_b["encoder_forwards"] == 1
        ),
        "reuse_one_encoder_forward": reused["encoder_forwards"] == 1,
        "model_state_unchanged": fingerprint_loaded_model(model) == fingerprint,
        "audio_unchanged": sha256_file(audio_path) == audio_digest,
        "resources_released": budget.available == capacity and worker.queue_depth == 0,
    }
    return {
        "schema_version": "1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "cpu_fp32_exact_window_prompt_redecode",
        "backend": {
            "revision": revision,
            "tree": tree,
            "tracked_source_clean": True,
            "preexisting_bytecode_bypassed": True,
        },
        "model": {"checkpoint": str(checkpoint), "fingerprint": fingerprint},
        "audio": {"path": str(audio_path), "sha256": audio_digest},
        "configuration": {
            "prompt_a": args.prompt_a,
            "prompt_b": args.prompt_b,
            "rng_seed": args.rng_seed,
            "sample_len": args.sample_len,
            "cpu_threads": args.threads,
            "decode_options": asdict(
                decode_options(
                    args.profile,
                    prompt=args.prompt_a,
                    sample_len=args.sample_len,
                    timestamps=args.timestamps,
                )
            ),
            "profile": args.profile,
        },
        "checks": checks,
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "reuse_a_b": reused,
        "timing_is_benchmark": False,
        "model_downloaded": False,
    }


def main() -> int:
    args = parse_args()
    try:
        report = verify(args)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        report = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
