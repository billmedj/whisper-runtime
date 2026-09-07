"""Accelerated controller soak with scripted recognition, not an acoustic test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from whisper_runtime.adapters import ContinuousTranscriptStream, StreamEventKind
from whisper_runtime.native_setup import CLI_STREAM_CONFIG


def resident_memory():
    """Read process RSS without instrumenting every allocation in the soak."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in ("peak", "rss", "pp", "p", "np", "n", "page", "page_peak")
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "process memory query failed")
        return dict(rss_bytes=counters.rss, peak_rss_bytes=counters.peak)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        rss_bytes=None, peak_rss_bytes=peak * (1 if sys.platform == "darwin" else 1024)
    )


def run(*, seconds=14_400, chunk_ms=100, wall_limit=180):
    if type(seconds) is not int or not 1 <= seconds <= 14_400:
        raise ValueError("seconds must be an integer between 1 and 14400")
    if type(chunk_ms) is not int or chunk_ms not in (20, 100, 200):
        raise ValueError("chunk_ms must be 20, 100 or 200")
    if not 0 < wall_limit <= 180:
        raise ValueError("wall_limit must be between 0 and 180 seconds")
    fixture_path = str(Path(__file__).resolve().parents[1] / "tests")
    with patch.object(sys, "path", [fixture_path, *sys.path]):
        from test_continuous_evidence import EvidenceNativeAdapter, speech_result

    adapter = EvidenceNativeAdapter(speech_result)
    stream = ContinuousTranscriptStream(
        adapter,
        stream_id="accelerated-soak",
        mel_builder=lambda pcm: pcm,
        config=CLI_STREAM_CONFIG,
    )
    digest = hashlib.sha256()
    events = commits = finals = steps = committed = 0
    sequence = 0
    checkpoints = []
    started = time.monotonic()

    def drain():
        nonlocal events, commits, finals, steps, committed
        for _ in range(100):
            if not stream.ready:
                return
            if time.monotonic() - started > wall_limit:
                raise TimeoutError("accelerated soak wall deadline exceeded")
            steps += 1
            for event in stream.step():
                if event.sequence_number != events + 1:
                    raise AssertionError("event sequence gap")
                events += 1
                # A compact deterministic event receipt; recursive dataclass
                # serialization here measures the harness rather than the core.
                digest.update(
                    repr(
                        (
                            event.kind.value,
                            event.sequence_number,
                            event.segment_id,
                            event.revision,
                            event.start_sample,
                            event.end_sample,
                            event.text,
                            event.committed_through_sample,
                        )
                    ).encode()
                )
                if event.kind == StreamEventKind.COMMIT:
                    if event.start_sample != committed:
                        raise AssertionError("noncontiguous committed coverage")
                    committed = event.end_sample
                    commits += 1
                if event.kind == StreamEventKind.FINAL:
                    finals += 1
            # Test-fixture recording must not masquerade as runtime retention.
            if not stream.active:
                adapter.calls.clear()
                adapter.inputs.clear()
                adapter.options.clear()
                adapter.runs.clear()
        raise AssertionError("controller failed to yield within 100 steps")

    try:
        for offset_ms in range(0, seconds * 1000, chunk_ms):
            count = min(chunk_ms, seconds * 1000 - offset_ms) * 16
            # Alternate one second of nonzero PCM and one second of exact zero.
            # The scripted recognizer does not establish real speech detection.
            value = 1000 if (offset_ms // 1000) % 2 == 0 else 0
            content = value.to_bytes(2, "little", signed=True) * count
            stream.push(sequence, content)
            sequence += 1
            drain()
            if stream.metrics.buffered_samples > stream.config.max_buffer_ms * 16:
                raise AssertionError("PCM buffer bound exceeded")
            if len(stream.state.windows) > 4:
                raise AssertionError("session history bound exceeded")
            if offset_ms % 900_000 == 0:
                gc.collect()
                checkpoints.append(dict(source_ms=offset_ms, **resident_memory()))
        stream.finish_input()
        drain()
        if not stream.done or finals != 1 or committed != seconds * 16_000:
            raise AssertionError("missing final or incomplete source coverage")
        metrics = asdict(stream.metrics)
        stream.close()
        if adapter.budget.available != adapter.capacity or adapter.worker.queue_depth:
            raise AssertionError("worker capacity not restored")
        return dict(
            schema_version="1-accelerated-controller-soak",
            status="completed",
            simulated_audio_seconds=seconds,
            chunk_ms=chunk_ms,
            elapsed_seconds=time.monotonic() - started,
            profile=stream.profile_id,
            config=asdict(stream.config),
            metrics=metrics,
            driver_steps=steps,
            events=events,
            commits=commits,
            final_events=finals,
            event_sha256=digest.hexdigest(),
            retained_publications=len(stream.state.windows),
            process_memory=resident_memory(),
            memory_checkpoints=checkpoints,
            source_clock_paced=False,
            real_audio=False,
            native_inference=False,
            gpu_used=False,
            capacity_restored=True,
            full_transcript_retained_by_harness=False,
        )
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=14_400)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists")
    record = run(seconds=args.seconds, chunk_ms=args.chunk_ms)
    content = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(content)
        print(args.output)


if __name__ == "__main__":
    main()
