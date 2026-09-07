"""Offline score-boundary audit; no model, Torch, GPU, network, or source edits.

Reconstructs native score records and calls the actual input-evidence policy.
The synthetic boundary examples are counterexamples, not new acoustic evidence.
Usage: python tools/audit_draft_thresholds.py [acoustic-diagnostic.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whisper_runtime import AudioSpan  # noqa: E402
from whisper_runtime.adapters.audio_evidence import (  # noqa: E402
    _NO_SPEECH_THRESHOLD,
    AudioObservation,
    assess_audio,
)
from whisper_runtime.adapters.native_result import (  # noqa: E402
    build_native_window_result,
)

DEFAULT = (
    ROOT / "artifacts/modal/integrated-draft-t4-20260907-v1/acoustic-diagnostic.json"
)
SCORE_FIELDS = ("avg_logprob", "no_speech_prob", "compression_ratio")


def classify(raw, observation=None):
    """Use the production policy without copying or softening its comparison."""
    observation = observation or AudioObservation.from_pcm(b"\x01\x00" * 16)
    result = build_native_window_result(
        SimpleNamespace(**raw), window_id="score-audit", analysis_span=AudioSpan(0, 1)
    )
    decision = assess_audio(observation, result)
    return {"state": decision.state, "reason": decision.reason}


def raw_result(score, *, avg_logprob=-1.0, compression_ratio=2.4, text="word"):
    return dict(
        language="en",
        tokens=[42],
        text=text,
        no_speech_prob=score,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        temperature=0.0,
    )


def boundary_cases():
    threshold = _NO_SPEECH_THRESHOLD
    return [
        dict(position=name, no_speech_prob=score, decision=classify(raw_result(score)))
        for name, score in (
            ("just_below", math.nextafter(threshold, -math.inf)),
            ("at", threshold),
            ("just_above", math.nextafter(threshold, math.inf)),
        )
    ]


def crossing_example(delta):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("the counterexample requires a positive finite delta")
    low = _NO_SPEECH_THRESHOLD - delta / 2
    high = _NO_SPEECH_THRESHOLD + delta / 2
    control, candidate = raw_result(low), raw_result(high)
    return dict(
        synthetic=True,
        tokens_exact=control["tokens"] == candidate["tokens"],
        control_score=low,
        candidate_score=high,
        observed_max_delta_used_only_as_example=delta,
        control_decision=classify(control),
        candidate_decision=classify(candidate),
        decision_parity=classify(control) == classify(candidate),
    )


def audit_record(record):
    cells = record["cells"]
    if [(c["position"], c["arm"]) for c in cells] != [
        ("a1", "fast-reuse"),
        ("b1", "fast-draft32"),
        ("b2", "fast-draft32"),
        ("a2", "fast-reuse"),
    ]:
        raise ValueError("expected the registered four-cell ABBA screen")
    pairs = []
    for control, candidate in ((cells[0], cells[1]), (cells[3], cells[2])):
        windows = list(zip(control["windows"], candidate["windows"], strict=True))
        traces = list(
            zip(control["decision_traces"], candidate["decision_traces"], strict=True)
        )
        rows = []
        for (a, b), (ta, tb) in zip(windows, traces, strict=True):
            if (a["start_ms"], a["end_ms"]) != (b["start_ms"], b["end_ms"]):
                raise ValueError("paired analysis spans differ")
            raw_a = a["decode_observation"]["result"]
            raw_b = b["decode_observation"]["result"]
            decisions = []
            margins = []
            for raw, window, trace in ((raw_a, a, ta), (raw_b, b, tb)):
                evidence = trace["audio_evidence"]
                if (
                    trace["analysis_start_sample"] // 16 != window["start_ms"]
                    or trace["analysis_end_sample"] // 16 != window["end_ms"]
                    or raw["no_speech_prob"] != evidence["no_speech_prob"]
                ):
                    raise ValueError("trace is not bound to its native observation")
                observation = AudioObservation(**evidence["observation"])
                decision = classify(raw, observation)
                decisions.append(decision)
                if not observation.digital_silence and any(
                    c.isalnum() for c in raw["text"]
                ):
                    margins.append(abs(raw["no_speech_prob"] - _NO_SPEECH_THRESHOLD))
                if decision != {k: evidence[k] for k in ("state", "reason")}:
                    raise ValueError(
                        "stored evidence differs from full-result gate replay"
                    )
            rows.append(
                dict(
                    start_ms=a["start_ms"],
                    end_ms=a["end_ms"],
                    tokens_exact=raw_a["tokens"] == raw_b["tokens"],
                    text_exact=raw_a["text"] == raw_b["text"],
                    gate_decisions_exact=decisions[0] == decisions[1],
                    score_deltas={k: abs(raw_a[k] - raw_b[k]) for k in SCORE_FIELDS},
                    active_threshold_margins=margins,
                )
            )
        pairs.append(
            dict(
                control=control["position"],
                candidate=candidate["position"],
                paired_windows=len(rows),
                reconstructed_gate_replays=2 * len(rows),
                events_exact=control["events"] == candidate["events"],
                tokens_exact=all(r["tokens_exact"] for r in rows),
                text_exact=all(r["text_exact"] for r in rows),
                gate_decisions_exact=all(r["gate_decisions_exact"] for r in rows),
                maximum_absolute_score_deltas={
                    k: max(r["score_deltas"][k] for r in rows) for k in SCORE_FIELDS
                },
                minimum_active_no_speech_margin=min(
                    m for r in rows for m in r["active_threshold_margins"]
                ),
                gate_crossings=[r for r in rows if not r["gate_decisions_exact"]],
            )
        )
    maximum = max(p["maximum_absolute_score_deltas"]["no_speech_prob"] for p in pairs)
    return dict(
        schema_version=1,
        offline=True,
        reconstructed_not_new_inference=True,
        active_score_gate={
            "field": "no_speech_prob",
            "threshold": _NO_SPEECH_THRESHOLD,
            "reject_comparison": ">=",
            "avg_logprob_override": False,
        },
        boundary_cases=boundary_cases(),
        pairs=pairs,
        synthetic_crossing=crossing_example(maximum) if maximum else None,
        exact_decision_parity_guaranteed_for_all_inputs=False,
        limitation="Observed deltas are not a proven error bound. Same-side scores and "
        "identical other inputs preserve this gate, but do not prove future "
        "token, timing, publication, or whole-event parity.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    raw = args.record.read_bytes()
    result = audit_record(json.loads(raw))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "tools/audit_draft_thresholds.py",
            "src/whisper_runtime/adapters/audio_evidence.py",
            "src/whisper_runtime/adapters/continuous_stream.py",
            "src/whisper_runtime/adapters/_draft_inference.py",
        )
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
