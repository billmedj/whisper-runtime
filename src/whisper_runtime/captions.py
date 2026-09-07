"""Exact committed-revision projection and conservative source-coverage captions."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from .adapters.native_stream import StreamEventKind, TranscriptEvent


@dataclass(frozen=True, slots=True)
class Caption:
    start_sample: int
    end_sample: int
    text: str


class CommittedTranscript:
    """Do not treat a preview, intent, or a FINAL marker as transcript text."""

    def __init__(self) -> None:
        self.captions: list[Caption] = []
        self.head = self.sequence = self.version = 0
        self.final = False
        self._pending: TranscriptEvent | None = None
        self._committed: set[str] = set()

    def accept(self, event: TranscriptEvent) -> Caption | None:
        if self.final or event.sequence_number != self.sequence + 1:
            raise ValueError(
                "transcript events must be consecutive and stop after FINAL"
            )
        if event.kind is StreamEventKind.FINAL:
            if self._pending is not None or event.session_version != self.version:
                raise ValueError("FINAL cannot resolve a pending text revision")
            self.final = True
            self.sequence = event.sequence_number
            return None
        if event.sample_rate_hz != 16000 or event.start_sample != self.head:
            raise ValueError("event source span must begin at the committed head")
        assert event.segment_id is not None and event.revision is not None
        if event.segment_id in self._committed:
            raise ValueError("a committed segment cannot be changed")
        old = self._pending
        if old is not None:
            assert old.revision is not None
        if event.kind in {StreamEventKind.PROVISIONAL, StreamEventKind.REPLACE}:
            if (
                old is None
                and (
                    event.kind is not StreamEventKind.PROVISIONAL or event.revision != 1
                )
            ) or (
                old is not None
                and (
                    event.kind is not StreamEventKind.REPLACE
                    or event.segment_id != old.segment_id
                    or event.revision != (old.revision or 0) + 1
                )
            ):
                raise ValueError("text revision chain differs")
            self._pending = event
            self.sequence = event.sequence_number
            return None
        if old is None or any(
            getattr(old, key) != getattr(event, key)
            for key in (
                "segment_id",
                "revision",
                "start_sample",
                "end_sample",
                "session_version",
            )
        ):
            raise ValueError("COMMIT lacks its exact preceding text revision")
        if (
            event.committed_through_sample != event.end_sample
            or event.session_version != self.version + 1
        ):
            raise ValueError("COMMIT coverage or session version differs")
        assert event.end_sample is not None and old.text is not None
        if event.end_sample <= self.head:
            raise ValueError("COMMIT must advance source coverage")
        caption = Caption(self.head, event.end_sample, old.text)
        self.captions.append(caption)
        self.head, self.version = event.end_sample, event.session_version
        self.sequence = event.sequence_number
        self._committed.add(event.segment_id)
        self._pending = None
        return caption

    def render(self, format: str) -> str:
        if not self.final:
            raise ValueError("exports require a real FINAL and committed revisions")
        if format == "txt":
            text = " ".join(c.text.strip() for c in self.captions if c.text.strip())
            return text + "\n" if text else ""
        if format not in {"srt", "vtt"}:
            raise ValueError("export format must be txt, srt or vtt")
        pieces = ["WEBVTT\n\n"] if format == "vtt" else []
        index = 0
        for caption in self.captions:
            text = " ".join(caption.text.split())
            if not text:
                continue
            index += 1
            start, end = caption.start_sample // 16, (caption.end_sample + 15) // 16
            if end <= start:
                end = start + 1
            separator = "," if format == "srt" else "."
            # Escape subtitle markup; timestamps denote committed source coverage,
            # not measured word boundaries. No heuristic cue splitting is applied.
            text = html.escape(text, quote=False)
            pieces.append(
                f"{index}\n{_timestamp(start, separator)} --> {_timestamp(end, separator)}\n{text}\n\n"
            )
        return "".join(pieces)


def _timestamp(ms: int, separator: str) -> str:
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}"


def validate_outputs(outputs: dict[str, Path]) -> None:
    paths = [path.resolve() for path in outputs.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("export paths must be distinct")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite export: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"export directory does not exist: {path.parent}")


def write_exports(transcript: CommittedTranscript, outputs: dict[str, Path]) -> None:
    """Exclusive creation protects existing paths, including a late path race."""
    validate_outputs(outputs)
    rendered = {format: transcript.render(format) for format in outputs}
    for format, path in outputs.items():
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered[format])
