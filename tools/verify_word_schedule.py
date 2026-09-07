"""Test early word checks with shorter context on the cached CPU fixture.

This process-local override changes no runtime defaults or checkpoint format.
It reuses the existing bounded driver, model checks and forward counters. Input
is admitted in chunks without pacing; elapsed time is not live latency.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def early_schedule(base, start_ms=8000):
    """Advance the first word check; leave context selection and guards intact."""
    if type(start_ms) is not int or start_ms <= 0 or start_ms % 20:
        raise ValueError("start_ms must be a positive multiple of 20")

    class EarlyWordStream(base):
        def _word_fallback_start(self, origin):
            return min(super()._word_fallback_start(origin), origin + start_ms * 16)

    return EarlyWordStream


def worker(args):
    from tools.verify_deferred_word_commits import run_worker
    from whisper_runtime.adapters import ContinuousTranscriptStream

    captured = io.StringIO()
    try:
        with (
            patch(
                "whisper_runtime.adapters.ContinuousTranscriptStream",
                early_schedule(ContinuousTranscriptStream),
            ),
            contextlib.redirect_stdout(captured),
        ):
            run_worker(
                args.model,
                "candidate",
                manifest=args.manifest,
                reuse_alignment_features=True,
                case=args.case,
                cell_timeout_seconds=140,
                left_context_ms=args.left_context_ms,
                word_context_limit_ms=args.word_context_limit_ms,
            )
    finally:
        reports = [json.loads(line) for line in captured.getvalue().splitlines()]
        for report in reports:
            report["experiment"] = dict(
                first_word_check_ms=8000,
                context_selection_unchanged=True,
                production_defaults_changed=False,
                checkpoint_qualification=False,
                tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            )
        if reports:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as target:
                for report in reports:
                    target.write(json.dumps(report, ensure_ascii=False) + "\n")
            for report in reports:
                print(
                    json.dumps(
                        {
                            key: report[key]
                            for key in (
                                "done",
                                "error",
                                "word_edits",
                                "reference_word_count",
                                "text",
                                "commit_input_samples",
                                "commit_source_ends",
                                "forward_counters",
                                "wall_seconds",
                                "capacity_restored",
                                "all_windows_closed",
                                "final_count",
                            )
                        }
                    ),
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-context-ms", type=int, default=2000)
    parser.add_argument("--word-context-limit-ms", type=int, default=6000)
    parser.add_argument("--case", choices=("noisy", "continuous"), default="continuous")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new path")
    if args.worker:
        worker(args)
    else:
        subprocess.run(
            [
                str(ROOT / ".tmp-native/venv/Scripts/python.exe"),
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                *sys.argv[1:],
            ],
            cwd=ROOT,
            timeout=240,
            check=True,
        )


if __name__ == "__main__":
    main()
