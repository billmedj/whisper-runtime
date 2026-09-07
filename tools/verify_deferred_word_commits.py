"""Bounded CPU comparison using only the cached backend, model and exact PCM.

This is accelerated deterministic chunk admission, not source-paced latency.
The default parent process enforces a 240-second hard worker deadline. Stdout
contains one JSON report per cell; this tool never downloads or writes evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


@contextmanager
def _count_forwards(model):
    """Count real forward inputs, not unique audio frames or device execution time."""
    counts = dict(
        encoder_forwards=0,
        encoder_input_frames=0,
        decoder_forwards=0,
        decoder_input_frames=0,
        decoder_input_tokens=0,
        hooks_removed=False,
    )
    modules = model.encoder, model.decoder
    previous = [tuple(module._forward_pre_hooks) for module in modules]

    def encoder(_module, inputs, _kwargs):
        counts["encoder_forwards"] += 1
        counts["encoder_input_frames"] += int(inputs[0].shape[-1])

    def decoder(_module, inputs, _kwargs):
        counts["decoder_forwards"] += 1
        counts["decoder_input_frames"] += int(inputs[1].shape[-2])
        counts["decoder_input_tokens"] += int(inputs[0].shape[-1])

    try:
        with ExitStack() as cleanup:
            for module, observer in zip(modules, (encoder, decoder)):
                handle = module.register_forward_pre_hook(observer, with_kwargs=True)
                cleanup.callback(handle.remove)
            yield counts
    finally:
        counts["hooks_removed"] = previous == [
            tuple(module._forward_pre_hooks) for module in modules
        ]


def run_worker(
    model_path,
    arm,
    *,
    manifest=ROOT / ".tmp-native/manifest.json",
    reuse_alignment_features=False,
    legacy_alignment=False,
    case="both",
    cell_timeout_seconds=80,
    left_context_ms=2000,
    word_context_limit_ms=6000,
):
    if legacy_alignment and not reuse_alignment_features:
        raise ValueError("legacy alignment control requires the feature-reuse opt-in")
    if case not in ("both", "noisy", "continuous"):
        raise ValueError("case must be both, noisy, or continuous")
    if type(cell_timeout_seconds) is not int or not 1 <= cell_timeout_seconds <= 140:
        raise ValueError("cell timeout must be between 1 and 140 seconds")
    from infra.modal_noisy_context_prompt import source_case
    from infra.modal_paced_features import _MeasuredAdapter
    from tools.prepare_acoustic_cases import build_cases
    from whisper_runtime import native_setup
    from whisper_runtime.adapters import (
        ContinuousStreamConfig,
        ContinuousTranscriptStream,
        QuietEndpointConfig,
    )
    from whisper_runtime.adapters.audio_endpoints import QuietEndpointDetector
    from whisper_runtime.adapters.continuous_stream import StreamNeedsResolutionError

    source = source_case()
    metadata, pcms = build_cases()
    no_pause = "concatenated-no-added-pauses"
    config = ContinuousStreamConfig(
        preview_interval_ms=2000,
        max_window_ms=30000,
        max_buffer_ms=40000,
        holdback_ms=2000,
        left_context_ms=left_context_ms,
        coalesce_previews=False,
        source_units=True,
        input_evidence=True,
        endpointing=QuietEndpointConfig(),
        word_boundary_fallback=True,
        word_context_limit_ms=word_context_limit_ms,
        eof_context_retry=True,
    )
    initial = native_setup.create_stream(
        manifest=manifest,
        model=model_path,
        device="cpu",
        config=config,
        reuse_alignment_features=reuse_alignment_features,
    )
    adapter, mel_builder = initial._adapter, initial._mel_builder
    initial.close()
    measured = _MeasuredAdapter(adapter)
    measured.reuse = reuse_alignment_features and not legacy_alignment
    cases = (("noisy-prefix", source["pcm"]), (no_pause, pcms[no_pause]))
    if case != "both":
        cases = tuple(
            item
            for item in cases
            if item[0] == ("noisy-prefix" if case == "noisy" else no_pause)
        )
    references = {item["id"]: item["reference_text"] for item in metadata["cases"]}
    references["noisy-prefix"] = source["reference_text"]
    from whisper.normalizers import EnglishTextNormalizer

    from tools.analyze_stream_text_agreement import _commits

    normalizer = EnglishTextNormalizer()

    def edits(left, right):
        row = list(range(len(right) + 1))
        for i, word in enumerate(left, 1):
            nxt = [i]
            for j, other in enumerate(right, 1):
                nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j - 1] + (word != other)))
            row = nxt
        return row[-1]

    # Candidate first: obtain the new-policy result before spending the control.
    for enabled in (
        (True,)
        if arm == "candidate"
        else (False,)
        if arm == "baseline"
        else (True, False)
    ):
        for case_id, pcm in cases:
            current = replace(config, defer_word_commits=enabled)
            first_window = len(measured.records)
            stream = ContinuousTranscriptStream(
                measured,
                stream_id=f"deferred-cpu:{case_id}:{enabled}",
                config=current,
                mel_builder=mel_builder,
                options=initial._options,
                rng_seed=7,
            )
            events, traces, error = [], [], None
            timeout_error = None
            commit_input_samples = []
            started = time.monotonic()

            def drain():
                for _ in range(10000):
                    if not stream.ready:
                        return
                    if time.monotonic() - started > cell_timeout_seconds:
                        raise TimeoutError(
                            f"{cell_timeout_seconds}-second per-cell work limit"
                        )
                    emitted = stream.step()
                    events.extend(asdict(e) for e in emitted)
                    commit_input_samples.extend(
                        stream.accepted_samples
                        for e in emitted
                        if e.kind.value == "commit"
                    )
                    trace = stream.last_trace
                    if trace is not None and (
                        not traces or traces[-1]["decode_index"] != trace.decode_index
                    ):
                        traces.append(
                            dict(
                                decode_index=trace.decode_index,
                                start_sample=trace.analysis_start_sample,
                                end_sample=trace.analysis_end_sample,
                                reason=trace.reason,
                                action=trace.action,
                                alignment_prepared=trace.word_alignment is not None,
                                text=trace.result.text,
                            )
                        )
                raise RuntimeError("driver step guard exhausted")

            with _count_forwards(adapter._model) as forward_counters:
                try:
                    for sequence, offset in enumerate(range(0, len(pcm), 64000)):
                        stream.push(sequence, pcm[offset : offset + 64000])
                        if offset + 64000 >= len(pcm):
                            stream.finish_input()
                        drain()
                except (StreamNeedsResolutionError, TimeoutError) as exc:
                    error = str(exc)
                    if isinstance(exc, TimeoutError):
                        timeout_error = exc
                finally:
                    completed = stream.done
                    final_trace = stream.last_trace
                    stream.close()
            windows = measured.records[first_window:]
            committed = " ".join(text for _, text in _commits(events))
            expected_words = normalizer(references[case_id]).split()
            observed_words = normalizer(committed).split()
            print(
                json.dumps(
                    dict(
                        case=case_id,
                        candidate=enabled,
                        manifest=str(manifest.resolve()),
                        model_identity=asdict(adapter.model_identity),
                        factory_reuse_alignment_features=reuse_alignment_features,
                        legacy_alignment=legacy_alignment,
                        reuse_alignment_features=measured.reuse,
                        profile=stream.profile_id,
                        config=asdict(current),
                        pcm_sha256=sha256(pcm).hexdigest(),
                        input_samples=len(pcm) // 2,
                        endpoint_samples=[
                            p.end_sample
                            for p in QuietEndpointDetector().observe(0, pcm)
                        ],
                        done=completed,
                        error=error,
                        timed_out=timeout_error is not None,
                        cell_timeout_seconds=cell_timeout_seconds,
                        metrics=asdict(stream.metrics),
                        text=committed,
                        word_edits=edits(expected_words, observed_words),
                        reference_word_count=len(expected_words),
                        final_count=sum(e["kind"] == "final" for e in events),
                        commit_source_ends=[
                            e["end_sample"] for e in events if e["kind"] == "commit"
                        ],
                        commit_input_samples=commit_input_samples,
                        traces=traces,
                        final_trace=dict(
                            reason=final_trace.reason,
                            action=final_trace.action,
                            no_speech_prob=final_trace.result.metadata.no_speech_prob,
                        )
                        if final_trace is not None
                        and final_trace.result.metadata is not None
                        else None,
                        wall_seconds=time.monotonic() - started,
                        capacity_restored=adapter.worker.budget.available
                        == adapter.worker.budget.capacity,
                        queue_depth_after_close=adapter.worker.queue_depth,
                        budget_lease_count_after_close=adapter.worker.budget.lease_count,
                        native_windows=len(windows),
                        all_windows_closed=all(w["closed"] for w in windows),
                        all_window_capacity_restored=all(
                            w["capacity_restored"] for w in windows
                        ),
                        forward_counters=forward_counters,
                        source_paced=False,
                        gpu_used=False,
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if timeout_error is not None:
                raise timeout_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / ".tmp-native/manifest.json"
    )
    parser.add_argument("--reuse-alignment-features", action="store_true")
    parser.add_argument("--legacy-alignment", action="store_true")
    parser.add_argument(
        "--case", choices=("both", "noisy", "continuous"), default="both"
    )
    parser.add_argument("--cell-timeout-seconds", type=int, default=80)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--left-context-ms", type=int, default=2000)
    parser.add_argument("--word-context-limit-ms", type=int, default=6000)
    parser.add_argument(
        "--arm", choices=("both", "candidate", "baseline"), default="both"
    )
    args = parser.parse_args()
    if args.legacy_alignment and not args.reuse_alignment_features:
        parser.error("--legacy-alignment requires --reuse-alignment-features")
    if not 1 <= args.cell_timeout_seconds <= 140:
        parser.error("--cell-timeout-seconds must be between 1 and 140")
    if args.worker:
        run_worker(
            args.model,
            args.arm,
            manifest=args.manifest,
            reuse_alignment_features=args.reuse_alignment_features,
            legacy_alignment=args.legacy_alignment,
            case=args.case,
            cell_timeout_seconds=args.cell_timeout_seconds,
            left_context_ms=args.left_context_ms,
            word_context_limit_ms=args.word_context_limit_ms,
        )
    else:
        subprocess.run(
            [
                str(ROOT / ".tmp-native/venv/Scripts/python.exe"),
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                "--model",
                str(args.model.resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                *(
                    ["--reuse-alignment-features"]
                    if args.reuse_alignment_features
                    else []
                ),
                *(["--legacy-alignment"] if args.legacy_alignment else []),
                "--case",
                args.case,
                "--cell-timeout-seconds",
                str(args.cell_timeout_seconds),
                "--arm",
                args.arm,
                "--left-context-ms",
                str(args.left_context_ms),
                "--word-context-limit-ms",
                str(args.word_context_limit_ms),
            ],
            cwd=ROOT,
            timeout=240,
            check=True,
        )


if __name__ == "__main__":
    main()
