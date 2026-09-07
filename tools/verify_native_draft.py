"""Bounded CPU regression of the integrated draft option. No downloads or Modal."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SOURCES = (
    "src/whisper_runtime/adapters/_draft_inference.py",
    "src/whisper_runtime/adapters/native_whisper.py",
    "src/whisper_runtime/adapters/continuous_stream.py",
    "src/whisper_runtime/adapters/_checkpoint_io.py",
    "src/whisper_runtime/adapters/stream_checkpoint.py",
    "src/whisper_runtime/native_setup.py",
    "tools/verify_native_draft.py",
)


def worker(args):
    from tools.prepare_acoustic_cases import build_cases
    from tools.verify_deferred_word_commits import _count_forwards
    from tools.verify_draft_gpu import commits
    from whisper_runtime import (
        RequestCancelledError,
        RequestState,
        Session,
        native_setup,
    )
    from whisper_runtime.adapters import ContinuousTranscriptStream

    hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    pipeline = (
        "sha256:" + sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    )
    initial = native_setup.create_stream(
        manifest=args.manifest,
        model=args.model,
        device="cpu",
        reuse_alignment_features=True,
    )
    adapter, mel_builder = initial._adapter, initial._mel_builder
    options = initial._options
    initial.close()
    model = adapter._model

    def hooks():
        return [
            (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
            for m in model.modules()
        ]

    original_hooks = hooks()
    _, cases = build_cases()
    pcm = cases["concatenated-no-added-pauses"]
    records = []
    with tempfile.TemporaryDirectory(prefix="native-draft-checkpoint-") as temporary:
        for name, limit, resume in (
            ("control", 0, False),
            ("draft", 32, False),
            ("draft-resume", 32, True),
        ):
            config = replace(
                native_setup.CLI_STREAM_CONFIG,
                left_context_ms=20000,
                word_context_limit_ms=24000,
                max_draft_tokens=limit,
            )
            stream = ContinuousTranscriptStream(
                adapter,
                stream_id="native-draft-cpu",
                mel_builder=mel_builder,
                options=options,
                config=config,
                rng_seed=7,
            )
            events, traces = [], []
            restarted = False
            started = time.monotonic()
            with _count_forwards(model) as counters:
                try:
                    for sequence, offset in enumerate(range(0, len(pcm), 64000)):
                        stream.push(sequence, pcm[offset : offset + 64000])
                        if offset + 64000 >= len(pcm):
                            stream.finish_input()
                        for _ in range(10000):
                            if not stream.ready:
                                break
                            if time.monotonic() - started > 100:
                                raise TimeoutError("100-second CPU arm limit")
                            batch = stream.step()
                            events.extend(asdict(event) for event in batch)
                            trace = stream.last_trace
                            if trace is not None and (
                                not traces
                                or traces[-1]["decode_index"] != trace.decode_index
                            ):
                                traces.append(
                                    dict(
                                        decode_index=trace.decode_index,
                                        start=trace.analysis_start_sample,
                                        end=trace.analysis_end_sample,
                                        action=trace.action,
                                        reason=trace.reason,
                                        tokens=list(trace.result.metadata.tokens),
                                        avg_logprob=trace.result.metadata.avg_logprob,
                                        no_speech_prob=trace.result.metadata.no_speech_prob,
                                    )
                                )
                            if (
                                resume
                                and not restarted
                                and not stream.done
                                and any(e.kind.value == "commit" for e in batch)
                            ):
                                path = Path(temporary) / "savepoint.json"
                                digest = stream.save_checkpoint(
                                    path, pipeline_identity=pipeline
                                )
                                assert stream._draft_tokens
                                stream.close()
                                stream = ContinuousTranscriptStream.from_checkpoint(
                                    adapter,
                                    path=path,
                                    expected_sha256=digest,
                                    pipeline_identity=pipeline,
                                    mel_builder=mel_builder,
                                )
                                assert not stream._draft_tokens and not stream.active
                                restarted = True
                        else:
                            raise RuntimeError("driver step guard exhausted")
                    completed, metrics = stream.done, asdict(stream.metrics)
                finally:
                    stream.close()
            record = dict(
                arm=name,
                completed=completed,
                restarted=restarted,
                events=events,
                commits=commits(events),
                traces=traces,
                counts=counters,
                metrics=metrics,
                hints_cleared=not stream._draft_tokens,
                capacity_restored=adapter.worker.budget.available
                == adapter.worker.budget.capacity,
                leases=adapter.worker.budget.lease_count,
            )
            assert completed and (not resume or restarted)
            assert record["capacity_restored"] and record["leases"] == 0
            assert (
                metrics["accepted_samples"]
                == metrics["committed_samples"]
                == len(pcm) // 2
            )
            assert sum(e["kind"] == "final" for e in events) == 1
            records.append(record)

    # Cancel a real owned speculative prefill, then verify terminal cleanup.
    session = Session("draft-cancel")
    run = adapter.start_window(
        session=session,
        request=RequestState(
            "cancel", session.session_id, adapter.model_identity, rng_seed=7
        ),
        window_id="cancel",
        mel=mel_builder(pcm[:320000]),
        start_ms=0,
        end_ms=10000,
        options=options,
        draft_tokens=tuple(records[0]["traces"][0]["tokens"][:32]),
    )
    inference = run._backend_run.inference
    assert inference.rows is not None and inference.stats["parallel_prefills"] == 1
    run.cancel()
    try:
        run.step()
    except RequestCancelledError:
        pass
    else:
        raise AssertionError("cancelled decode advanced")
    finally:
        run.close()
    cancellation = dict(
        closed=run.closed,
        capacity_released=run.capacity_released,
        rows_cleared=inference.rows is None,
        audio_cleared=inference.audio_features is None,
        draft_cleared=not inference.draft,
        cache_cleared=not inference.original.kv_cache,
        session_unchanged=session.snapshot().version == 0,
    )

    # Compare decisions separately from floating-point diagnostic scores.
    def decisions(record):
        return [
            {k: v for k, v in t.items() if k not in ("avg_logprob", "no_speech_prob")}
            for t in record["traces"]
        ]

    reference = records[0]
    comparisons = [
        dict(
            arm=r["arm"],
            events_exact=r["events"] == reference["events"],
            commits_exact=r["commits"] == reference["commits"],
            decisions_and_tokens_exact=decisions(r) == decisions(reference),
            metrics_exact=r["metrics"] == reference["metrics"],
            fewer_decoder_forwards=r["counts"]["decoder_forwards"]
            < reference["counts"]["decoder_forwards"],
        )
        for r in records[1:]
    ]
    return dict(
        schema="native-draft-integration-cpu/v1",
        sources=hashes,
        model=asdict(adapter.model_identity),
        pcm_sha256=sha256(pcm).hexdigest(),
        samples=len(pcm) // 2,
        source_paced=False,
        gpu_used=False,
        records=records,
        comparisons=comparisons,
        cancellation=cancellation,
        hooks_unchanged=original_hooks == hooks(),
        model_unchanged=native_setup._fingerprint(model)
        == adapter.model_identity.fingerprint,
        claim="One fixed CPU input. Not a GPU timing or general checkpoint parity claim.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            print(json.dumps(worker(args)))
        except BaseException:
            print(json.dumps(dict(error=traceback.format_exc())))
            raise SystemExit(1) from None
        return
    if args.output is None or args.output.exists():
        parser.error("provide a new --output path; existing evidence is never replaced")
    command = [
        str(ROOT / ".tmp-native/venv/Scripts/python.exe"),
        "-B",
        str(Path(__file__)),
        "--worker",
        "--model",
        str(args.model),
        "--manifest",
        str(args.manifest),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=240,
            check=False,
        )
        report = dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )
        if result.returncode == 0:
            report = json.loads(result.stdout)
    except subprocess.TimeoutExpired as error:
        report = dict(
            error="240-second parent timeout",
            stdout=str(error.stdout),
            stderr=str(error.stderr),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(args.output)
    success = (
        report.get("comparisons")
        and all(
            all(v for k, v in comparison.items() if k != "arm")
            for comparison in report["comparisons"]
        )
        and all(report["cancellation"].values())
        and all(record["hints_cleared"] for record in report["records"])
        and report["hooks_unchanged"]
        and report["model_unchanged"]
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
