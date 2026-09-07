"""CPU-only proof of fresh-cache greedy draft verification, not a runtime feature.

Six bounded trajectories use existing local assets. The public backend prefill,
step, filters, timestamp rules and finalization remain unchanged. Only one run's
inference object is wrapped. No production/backend files or defaults are edited.
The parent enforces a 120-second subprocess timeout and writes a new receipt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def draft_variants(tokens, eot):
    """Raw prior-window IDs are proposals, never forced current context."""
    tokens = tuple(tokens)
    if len(tokens) < 32 or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("the source must provide at least 32 valid native tokens")
    # EOT is deliberately invalid at the first timestamp-required step.
    return {
        "draft16": tokens[:16],
        "draft32": tokens[:32],
        "corrupt_first": (eot, *tokens[1:16]),
        "truncated8": tokens[:8],
    }


def trim_self_cache(inference, length):
    """Trim only explicit self-attention modules; never trim audio K/V."""
    if type(length) is not int or length < 1:
        raise ValueError("cache prefix length must be a positive integer")
    modules = tuple(inference.kv_modules)
    if not modules or any(module not in inference.kv_cache for module in modules):
        raise RuntimeError("self-attention cache is incomplete")
    for module in modules:
        value = inference.kv_cache[module]
        if value.shape[1] < length:
            raise RuntimeError("cannot extend a cache by truncating it")
        inference.kv_cache[module] = value[:, :length].detach()


class DraftInference:
    """Serve causally verified rows to the unchanged one-token step loop.

    A full current-audio prefill owns all speculative KV. The next logits call
    observes the token that standard step() selected; a mismatch invalidates the
    unaccepted suffix before ordinary inference consumes the correcting token.
    """

    def __init__(self, original, draft):
        if original._use_legacy_cache:
            raise ValueError("draft verification requires request-local caching")
        self.original = original
        self.draft = tuple(draft)
        self.rows = None
        self.initial_length = None
        self.started = False
        self.audio_features = None
        self.stats = dict(
            proposed_tokens=len(draft),
            matched_tokens=0,
            mismatch_index=None,
            crop_length=None,
            parallel_prefills=0,
            saved_row_calls=0,
            ordinary_suffix_forwards=0,
            cleanup_calls=0,
        )

    def logits(self, tokens, audio_features):
        if not self.started:
            import torch

            if tokens.shape[0] != 1 or self.original.kv_cache:
                raise ValueError("draft prefill requires one fresh greedy sequence")
            self.started = True
            self.initial_length = int(tokens.shape[1])
            self.audio_features = audio_features
            draft = torch.tensor([self.draft], device=tokens.device, dtype=tokens.dtype)
            proposed = torch.cat((tokens, draft), dim=1)
            if proposed.shape[1] > self.original.model.dims.n_text_ctx:
                raise ValueError("draft exceeds the decoder context")
            self.rows = self.original.model.decoder(
                proposed,
                audio_features,
                kv_cache=self.original.kv_cache,
                _update_kv_cache=True,
            )
            self.stats["parallel_prefills"] += 1
            # Standard prefill computes no-speech at SOT and keeps the last row.
            return self.rows[:, : self.initial_length].clone()

        if audio_features is not self.audio_features:
            raise RuntimeError("current audio features changed during a decode")
        if self.rows is not None:
            generated = int(tokens.shape[1]) - self.initial_length
            if not 1 <= generated <= len(self.draft) + 1:
                raise RuntimeError("unexpected token cursor during draft verification")
            if generated <= len(self.draft):
                if int(tokens[0, -1]) == self.draft[generated - 1]:
                    self.stats["matched_tokens"] += 1
                    self.stats["saved_row_calls"] += 1
                    index = self.initial_length + generated - 1
                    return self.rows[:, index : index + 1].clone()
                self.stats["mismatch_index"] = generated - 1
                length = self.initial_length + generated - 1
                self.stats["crop_length"] = length
                trim_self_cache(self.original, length)
            # All proposals matched, or the actual next token differed. Future
            # proposals are never used after either transition to normal decode.
            self.rows = None
        self.stats["ordinary_suffix_forwards"] += 1
        return self.original.logits(tokens, audio_features)

    def cleanup_caching(self):
        self.stats["cleanup_calls"] += 1
        self.rows = None
        self.audio_features = None
        self.original.cleanup_caching()

    def rearrange_kv_cache(self, _indices):
        raise RuntimeError("beam search is outside this greedy diagnostic")


def result_record(result):
    return dict(
        tokens=list(result.tokens),
        text=result.text,
        language=result.language,
        avg_logprob=float(result.avg_logprob),
        no_speech_prob=float(result.no_speech_prob),
        temperature=float(result.temperature),
        compression_ratio=float(result.compression_ratio),
    )


def compare_results(reference, candidate):
    scalar_fields = ("avg_logprob", "no_speech_prob", "compression_ratio")
    return dict(
        exact_tokens=candidate["tokens"] == reference["tokens"],
        exact_text=candidate["text"] == reference["text"],
        exact_result=candidate == reference,
        numeric_delta={key: candidate[key] - reference[key] for key in scalar_fields},
        exact_numeric={key: candidate[key] == reference[key] for key in scalar_fields},
    )


def run_worker(model_path, manifest):
    from tools.prepare_acoustic_cases import build_cases
    from tools.verify_deferred_word_commits import _count_forwards
    from whisper_runtime import native_setup

    started = time.monotonic()
    initial = native_setup.create_stream(
        manifest=manifest,
        model=model_path,
        device="cpu",
        reuse_alignment_features=True,
    )
    adapter, mel_builder = initial._adapter, initial._mel_builder
    model = adapter._model
    initial.close()
    import torch
    from whisper.decoding import DecodingOptions, DecodingTask

    torch.set_num_threads(1)
    torch.manual_seed(7)
    if model.device.type != "cpu":
        raise RuntimeError("this diagnostic forbids non-CPU execution")
    options = DecodingOptions(
        language="en",
        temperature=0.0,
        fp16=False,
        without_timestamps=False,
        sample_len=96,
    )
    _, pcms = build_cases()
    pcm = pcms["concatenated-no-added-pauses"]
    window_pcm = {seconds: pcm[: seconds * 16000 * 2] for seconds in (8, 10)}
    before = native_setup._fingerprint(model)
    hooks_before = [
        (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
        for module in model.modules()
    ]
    trajectories = []

    def trajectory(name, input_value, draft=None):
        if len(trajectories) >= 6:
            raise RuntimeError("six-trajectory hard limit exceeded")
        task = DecodingTask(model, options)
        run = None
        wrapped = None
        beginning = time.monotonic()
        with _count_forwards(model) as counts, torch.no_grad():
            try:
                run = task._start_run(input_value)
                inference = run.inference
                if draft is not None:
                    wrapped = DraftInference(inference, draft)
                    run.inference = wrapped
                run.prefill()
                while not run.complete:
                    run.step()
                result = run.finalize()[0]
                record = result_record(result)
                if run.tokens[0, -1].item() != task.tokenizer.eot:
                    raise RuntimeError("sample limit reached before natural EOT")
            finally:
                if run is not None:
                    run.cleanup()
        if inference.kv_cache or inference.hooks:
            raise RuntimeError("backend cache or hook survived cleanup")
        trajectories.append(
            dict(
                name=name,
                result=record,
                forwards=dict(counts),
                token_steps=run.step_index,
                draft=list(draft) if draft is not None else None,
                draft_stats=dict(wrapped.stats) if wrapped is not None else None,
                cache_empty=True,
                wall_seconds=time.monotonic() - beginning,
            )
        )
        return result, task

    source, task = trajectory("source8", mel_builder(window_pcm[8]).unsqueeze(0))
    target, _ = trajectory("baseline10", mel_builder(window_pcm[10]).unsqueeze(0))
    current_features = target.audio_features.unsqueeze(0)
    feature_digest = sha256(current_features.numpy().tobytes()).hexdigest()
    variants = draft_variants(source.tokens, task.tokenizer.eot)
    for name, draft in variants.items():
        trajectory(name, current_features, draft)
        trajectories[-1]["comparison"] = compare_results(
            trajectories[1]["result"], trajectories[-1]["result"]
        )
        trajectories[-1]["decoder_forwards_saved"] = (
            trajectories[1]["forwards"]["decoder_forwards"]
            - trajectories[-1]["forwards"]["decoder_forwards"]
        )
    after = native_setup._fingerprint(model)
    hooks_after = [
        (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
        for module in model.modules()
    ]
    model_unchanged = before == after
    features_unchanged = (
        feature_digest == sha256(current_features.numpy().tobytes()).hexdigest()
    )
    hooks_unchanged = hooks_before == hooks_after
    if not model_unchanged or not features_unchanged or not hooks_unchanged:
        raise RuntimeError("model, features or hook-state integrity failed")
    return dict(
        schema="decoder-draft-cpu/v1",
        experiment_completed=True,
        production_feature=False,
        source_paced=False,
        gpu_used=False,
        threads=torch.get_num_threads(),
        trajectory_limit=6,
        trajectory_count=len(trajectories),
        options=asdict(options),
        model_identity=asdict(adapter.model_identity),
        model_unchanged=model_unchanged,
        current_features_unchanged=features_unchanged,
        current_features_sha256=feature_digest,
        hook_state_unchanged=hooks_unchanged,
        adapter_capacity_unchanged=(
            adapter.worker.budget.available == adapter.worker.budget.capacity
        ),
        adapter_transaction_qualification=False,
        alignment_or_publication_qualification=False,
        current_feature_reuse_scope="same target10 window only; no source8 KV or features reused",
        input_sha256={
            str(seconds): sha256(value).hexdigest()
            for seconds, value in window_pcm.items()
        },
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        trajectories=trajectories,
        all_target_tokens_exact=all(
            t["comparison"]["exact_tokens"] for t in trajectories[2:]
        ),
        all_target_results_exact=all(
            t["comparison"]["exact_result"] for t in trajectories[2:]
        ),
        wall_seconds=time.monotonic() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / ".tmp-composed-native/manifest.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        try:
            report = run_worker(args.model.resolve(), args.manifest.resolve())
        except BaseException as exc:
            print(
                json.dumps(
                    dict(
                        experiment_completed=False,
                        error=repr(exc),
                        traceback=traceback.format_exc(),
                    )
                )
            )
            return 1
        print(json.dumps(report, allow_nan=False))
        return 0
    if args.output is None:
        parser.error("the parent requires --output for a new no-clobber receipt")
    if args.output.exists():
        parser.error("receipt already exists; choose a new output path")
    beginning = time.monotonic()
    try:
        completed = subprocess.run(
            [
                str(ROOT / ".tmp-native/venv/Scripts/python.exe"),
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                "--model",
                str(args.model.resolve()),
                "--manifest",
                str(args.manifest.resolve()),
            ],
            cwd=ROOT,
            timeout=120,
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError:
            report = dict(experiment_completed=False, worker_stdout=completed.stdout)
        report["worker_returncode"] = completed.returncode
        report["worker_stderr"] = completed.stderr
    except subprocess.TimeoutExpired:
        report = dict(experiment_completed=False, timed_out=True)
    report["parent_timeout_seconds"] = 120
    report["parent_wall_seconds"] = time.monotonic() - beginning
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, allow_nan=False))
    return 0 if report.get("experiment_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
