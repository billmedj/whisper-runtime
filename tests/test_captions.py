"""Final exports never use speculative text or an unjoined revision."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from whisper_runtime.adapters.native_stream import StreamEventKind as K
from whisper_runtime.adapters.native_stream import TranscriptEvent
from whisper_runtime.captions import CommittedTranscript, write_exports


def events(text="Real text"):
    preview = TranscriptEvent(
        1, K.PROVISIONAL, "s", 1, 0, 32000, 16000, "Wrong preview", session_version=0
    )
    chosen = TranscriptEvent(
        2,
        K.REPLACE,
        "s",
        2,
        0,
        32000,
        16000,
        text,
        supersedes_revision=1,
        session_version=1,
    )
    commit = TranscriptEvent(
        3,
        K.COMMIT,
        "s",
        2,
        0,
        32000,
        16000,
        committed_through_sample=32000,
        committed_through_ms=2000,
        session_version=1,
    )
    final = TranscriptEvent(4, K.FINAL, session_version=1)
    return preview, chosen, commit, final


class CaptionTests(unittest.TestCase):
    def transcript(self, text="Real text"):
        result = CommittedTranscript()
        for event in events(text):
            result.accept(event)
        return result

    def test_only_exact_committed_revision_is_exported(self):
        result = self.transcript()
        self.assertEqual(result.render("txt"), "Real text\n")
        self.assertEqual(
            result.render("srt"), "1\n00:00:00,000 --> 00:00:02,000\nReal text\n\n"
        )
        self.assertEqual(
            result.render("vtt"),
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nReal text\n\n",
        )
        self.assertNotIn("Wrong preview", result.render("srt"))

    def test_missing_commit_final_revision_order_or_changed_commit_is_rejected(self):
        preview, chosen, commit, final = events()
        for sequence in (
            (preview, final),
            (preview, commit),
            (preview, chosen, replace(commit, revision=1)),
            (preview, chosen, commit, final, final),
        ):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                transcript = CommittedTranscript()
                for event in sequence:
                    transcript.accept(event)
        transcript = CommittedTranscript()
        transcript.accept(preview)
        with self.assertRaisesRegex(ValueError, "real FINAL"):
            transcript.render("txt")

    def test_subtitle_markup_is_escaped_and_silence_has_no_cue(self):
        self.assertIn("&lt;b&gt;&amp;", self.transcript("<b>&").render("vtt"))
        self.assertEqual(self.transcript("").render("vtt"), "WEBVTT\n\n")
        self.assertEqual(self.transcript("").head, 32000)

    def test_existing_and_duplicate_output_paths_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.txt"
            write_exports(self.transcript(), {"txt": path})
            with self.assertRaises(FileExistsError):
                write_exports(self.transcript("Other"), {"txt": path})
            self.assertEqual(path.read_text(), "Real text\n")
            other = Path(directory) / "new"
            with self.assertRaisesRegex(ValueError, "distinct"):
                write_exports(self.transcript(), {"txt": other, "srt": other})
            self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
