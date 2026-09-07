"""One local or remote transcript stream. Import/help require no ML stack."""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from pathlib import Path
from threading import Event, Thread
from typing import TextIO

from .adapters.continuous_stream import ContinuousTranscriptStream
from .adapters.native_stream import (
    AudioBufferFullError,
    StreamEventKind,
    TranscriptEvent,
)
from .audio_source import AudioSourceError, file_chunks, microphone_chunks
from .captions import CommittedTranscript, validate_outputs, write_exports
from .profiles import DEFAULT_PROFILE, list_profiles


def drive_stream(
    stream: ContinuousTranscriptStream,
    chunks: Iterator[bytes],
    *,
    on_event: Callable[[TranscriptEvent], None],
    cancel: Event,
    live: bool = False,
    drain_timeout: float = 90,
) -> None:
    """Keep all inference on the caller; producer only offers PCM and EOF.

    File mode retries the exact rejected chunk under backpressure. Live/paced
    mode stops on rejection. Cancellation is cooperative, including native work.
    The source is joined before the native stream is closed on every exit path.
    """
    if not math.isfinite(drain_timeout) or drain_timeout <= 0:
        raise ValueError("drain timeout must be positive and finite")
    changed, finished = Event(), Event()
    errors: list[BaseException] = []
    offered = 0
    eof_at: float | None = None

    def produce() -> None:
        nonlocal offered, eof_at
        try:
            for sequence, content in enumerate(chunks):
                if cancel.is_set():
                    return
                if not isinstance(content, bytes) or not content or len(content) % 2:
                    raise AudioSourceError(
                        "source returned empty or incomplete signed-16LE samples"
                    )
                if len(content) > stream.config.max_buffer_ms * 32:
                    raise AudioSourceError(
                        "one source chunk exceeds the runtime buffer capacity"
                    )
                offered += len(content) // 2
                while not cancel.is_set():
                    try:
                        stream.push(sequence, content)
                        changed.set()
                        break
                    except AudioBufferFullError as error:
                        if live:
                            raise AudioSourceError(
                                "live input exceeds the runtime buffer; stopping without silent loss or retry"
                            ) from error
                        changed.wait(0.01)
                        changed.clear()
            if not cancel.is_set():
                if not offered:
                    raise AudioSourceError("audio source is empty")
                eof_at = time.monotonic()
                stream.finish_input()
        except BaseException as error:
            errors.append(error)
        finally:
            try:
                if hasattr(chunks, "close"):
                    chunks.close()
            except BaseException as error:
                errors.append(error)
            finished.set()
            changed.set()

    def checkpoint() -> None:
        if cancel.is_set():
            raise KeyboardInterrupt
        if errors:
            raise errors[0]
        if eof_at is not None and time.monotonic() - eof_at > drain_timeout:
            raise TimeoutError(
                "EOF drain deadline exceeded; no final transcript was synthesized"
            )

    producer = Thread(target=produce, name="whisper-runtime-input", daemon=True)
    producer.start()
    try:
        while not stream.done:
            checkpoint()
            if stream.ready:
                events = stream.step()
                checkpoint()
                for event in events:
                    checkpoint()
                    on_event(event)
                changed.set()
            else:
                changed.wait(0.01)
                changed.clear()
        producer.join(timeout=2)
        if producer.is_alive() or not finished.is_set():
            raise RuntimeError("source did not stop after stream completion")
        checkpoint()
        if (
            offered != stream.accepted_samples
            or stream.metrics.committed_samples != offered
        ):
            raise RuntimeError(
                "final stream coverage does not account for all offered PCM"
            )
    finally:
        cancel.set()
        producer.join(timeout=2)
        stream.close()
        if producer.is_alive():
            raise RuntimeError(
                "audio source failed to stop; its input thread is still active"
            )


def display_event(event: TranscriptEvent, text: str | None, output: TextIO) -> None:
    if event.kind is StreamEventKind.FINAL:
        print("[final] complete", file=output, flush=True)
        return
    shown = " ".join(
        "".join(c for c in (text or "") if c.isprintable() or c.isspace()).split()
    )
    start = (event.start_sample or 0) / 16000
    end = (event.end_sample or 0) / 16000
    print(
        f"[{event.kind.value} r{event.revision} {start:.3f}-{end:.3f}s] {shown}",
        file=output,
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="whisper-runtime",
        description="Experimental single-stream transcription using a verified local backend or live-v2 WebSocket server. No model download or audio conversion.",
    )
    result.add_argument(
        "audio", nargs="?", type=Path, help="mono16kHz16-bit WAV, .pcm or .s16le file"
    )
    result.add_argument(
        "--setup-manifest",
        type=Path,
        help="verified native setup manifest produced by bootstrap_native_backend.py",
    )
    result.add_argument(
        "--model",
        type=Path,
        help="existing pinned tiny.en checkpoint path; never downloaded",
    )
    result.add_argument(
        "--device", choices=("cpu", "cuda:0"), help="local native device (default: cpu)"
    )
    result.add_argument(
        "--profile",
        choices=tuple(profile.name for profile in list_profiles()),
        help=f"versioned local settings (default: {DEFAULT_PROFILE}); experimental choices are not qualified; excludes --server",
    )
    result.add_argument(
        "--server",
        help="live-v2 server; wss:// remotely, ws:// only on loopback; files are paced",
    )
    result.add_argument(
        "--header-env",
        action="append",
        default=[],
        metavar="HEADER=ENV_NAME",
        help="server authentication header read from a named environment variable; never pass secret values here",
    )
    result.add_argument(
        "--paced",
        action="store_true",
        help="offer file chunks on their source clock; fail rather than conceal overload",
    )
    result.add_argument(
        "--microphone",
        action="store_true",
        help="capture with optional sounddevice/PortAudio (requires --duration)",
    )
    result.add_argument(
        "--duration",
        type=float,
        help="microphone capture duration in seconds (0 < value <= 3600)",
    )
    result.add_argument(
        "--input-device", help="optional sounddevice microphone name/query"
    )
    result.add_argument(
        "--drain-timeout",
        type=float,
        help="EOF drain deadline in seconds (local default: 90; remote default/cap: 30)",
    )
    for format in ("txt", "srt", "vtt"):
        result.add_argument(
            "--" + format,
            type=Path,
            help=f"create commit-only {format.upper()} after successful FINAL; never overwrite",
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    outputs = {
        format: path
        for format in ("txt", "srt", "vtt")
        if (path := getattr(args, format)) is not None
    }
    stream: ContinuousTranscriptStream | None = None
    exports_started = False
    try:
        if args.server:
            if (
                args.setup_manifest is not None
                or args.model is not None
                or args.device is not None
                or args.profile is not None
            ):
                raise ValueError(
                    "--server excludes --setup-manifest, --model, --device and --profile; live-v2 does not select or attest a server execution profile"
                )
            from .remote import environment_headers, validate_server

            validate_server(args.server)
            headers = environment_headers(args.header_env)
        else:
            if args.setup_manifest is None or args.model is None:
                raise ValueError(
                    "local mode requires --setup-manifest and --model; alternatively select --server"
                )
            if args.header_env:
                raise ValueError("--header-env requires --server")
            headers = {}
        if args.drain_timeout is None:
            args.drain_timeout = 30 if args.server else 90
        if bool(args.audio) == args.microphone:
            raise ValueError("choose exactly one audio file or --microphone")
        if not math.isfinite(args.drain_timeout) or args.drain_timeout <= 0:
            raise ValueError("--drain-timeout must be positive and finite")
        if args.server and args.drain_timeout > 30:
            raise ValueError("live-v2 caps --drain-timeout at 30 seconds")
        if args.microphone:
            if (
                args.duration is None
                or not math.isfinite(args.duration)
                or not 0 < args.duration <= 3600
            ):
                raise ValueError(
                    "--microphone requires --duration between 0 (exclusive) and 3600 seconds"
                )
            if args.paced:
                raise ValueError(
                    "--paced is for files; microphone capture already follows its source clock"
                )
        elif args.duration is not None or args.input_device is not None:
            raise ValueError("--duration and --input-device apply only to --microphone")
        validate_outputs(outputs)
        if args.audio is not None:
            with closing(file_chunks(args.audio)) as probe:
                next(probe)  # Validate the source before loading a model.
        cancel = Event()
        if args.microphone:
            assert args.duration is not None
            source = microphone_chunks(
                duration=args.duration, cancel=cancel, device=args.input_device
            )
        else:
            source = file_chunks(
                args.audio, paced=args.paced or bool(args.server), cancel=cancel
            )
        transcript = CommittedTranscript()
        pending_final: TranscriptEvent | None = None

        def receive(event: TranscriptEvent) -> None:
            nonlocal pending_final
            caption = transcript.accept(event)
            if event.kind is StreamEventKind.FINAL:
                # FINAL closes SDK history, not driver/transport verification.
                # Preserve it until the selected driver has returned successfully.
                pending_final = event
                return
            display_event(
                event, caption.text if caption is not None else event.text, sys.stdout
            )

        if args.server:
            from .remote import transcribe_remote
            from .remote_transport import LiveConfig

            print(
                "Experimental remote live-v2 stream. Server execution profile is not selected or attested by this client. File input is paced; no reconnect or audio retry.",
                file=sys.stderr,
            )
            accepted_samples = asyncio.run(
                transcribe_remote(
                    args.server,
                    source,
                    cancel=cancel,
                    on_event=receive,
                    headers=headers,
                    config=LiveConfig(drain_timeout_s=args.drain_timeout),
                )
            )
        else:
            from .native_setup import create_stream

            assert args.setup_manifest is not None and args.model is not None
            stream = create_stream(
                manifest=args.setup_manifest,
                model=args.model,
                device=args.device or "cpu",
                profile=args.profile or DEFAULT_PROFILE,
            )
            print(
                f"Local execution profile: {args.profile or DEFAULT_PROFILE}. Runtime remains experimental. Publication policy: {stream.profile_id}. Caption times are committed source coverage, not word-accurate alignment.",
                file=sys.stderr,
            )
            drive_stream(
                stream,
                source,
                on_event=receive,
                cancel=cancel,
                live=args.paced or args.microphone,
                drain_timeout=args.drain_timeout,
            )
            accepted_samples = stream.accepted_samples
        if not transcript.final or transcript.head != accepted_samples:
            raise RuntimeError(
                "missing real FINAL or incomplete committed event coverage"
            )
        if stream is not None:
            stream.close()
            stream = None
        assert pending_final is not None
        display_event(pending_final, None, sys.stdout)
        exports_started = True
        write_exports(transcript, outputs)
        for path in outputs.values():
            print(f"Created {path}", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print(
            "Cancelled during export; newly created files may be partial. Existing files were not overwritten."
            if exports_started
            else "Cancelled. No final exports were generated; printed commits remain valid. Cleanup waits cooperatively for active native work.",
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        print(f"whisper-runtime: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "No successful final export is claimed. If an export write failed, newly created files may be partial; existing files were not overwritten.",
            file=sys.stderr,
        )
        return 1
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception as error:
                print(
                    f"whisper-runtime: cleanup failed: {type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
