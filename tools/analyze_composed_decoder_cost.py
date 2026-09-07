"""Recompute composed-window decoder accounting offline; no inference or writes."""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/modal/composed-features-20260906-v1/acoustic-diagnostic.json"
)
ARMS = ("cli-legacy", "fast-reuse")
COUNTS = (
    "decode_forwards",
    "alignment_forwards",
    "lexical_tokens",
    "timestamp_tokens",
    "special_tokens",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def window_row(cell, cell_index, index):
    """Validate positional joins before counting raw forward records and tokens."""
    window = cell["windows"][index]
    trace = cell["decision_traces"][index]
    result = trace["result"]
    path = f"$.cells[{cell_index}]"
    label = f"{path}.windows[{index}]"
    start, end = window["start_ms"], window["end_ms"]
    require(
        type(start) is int and type(end) is int and 0 <= start < end,
        f"{label}: invalid input span",
    )
    require(trace["decode_index"] == index + 1, f"{label}: decode index mismatch")
    require(
        (trace["analysis_start_sample"], trace["analysis_end_sample"])
        == (start * 16, end * 16)
        and result["analysis_span"] == {"start_ms": start, "end_ms": end}
        and (result["start_ms"], result["end_ms"]) == (start, end),
        f"{label}: analysis span identity mismatch",
    )
    kind = "unit" if trace["source_unit"] is not None else "preview"
    expected_id = f"composed:{cell['arm']}:window:{start * 16}:{end * 16}:{kind}"
    require(
        window["window_id"] == result["window_id"] == expected_id,
        f"{label}: window identity mismatch",
    )
    committed = trace["committed_before_sample"]
    require(start * 16 <= committed <= end * 16, f"{label}: invalid committed head")
    tokens = result["metadata"]["tokens"]
    # The artifact declares tiny.en: EOT=50256, timestamp_begin=50363.
    require(
        all(type(token) is int and 0 <= token <= 51863 for token in tokens),
        f"{label}: invalid tiny.en token ID",
    )
    forwards = window["forwards"]
    require(
        all(
            f["kind"] in ("encoder", "decoder")
            and f["phase"] in ("decode", "alignment")
            for f in forwards
        ),
        f"{label}: unknown forward kind or phase",
    )
    counts = {
        f"{phase}_forwards": sum(
            f["kind"] == "decoder" and f["phase"] == phase for f in forwards
        )
        for phase in ("decode", "alignment")
    }
    require(
        counts["decode_forwards"] == len(tokens) + 1,
        f"{label}: decode forwards != generated token count + 1",
    )
    publication = trace["word_publication"]
    if publication is not None:
        require(
            publication["window_id"] == expected_id,
            f"{label}: publication window identity mismatch",
        )
    return {
        "window_path": label,
        "trace_path": f"{path}.decision_traces[{index}]",
        "window_id": expected_id,
        "start_ms": start,
        "end_ms": end,
        "committed_before_ms": committed / 16,
        "committed_context_ms": committed / 16 - start,
        **counts,
        "lexical_tokens": sum(token < 50256 for token in tokens),
        "timestamp_tokens": sum(token >= 50363 for token in tokens),
        "special_tokens": sum(50256 <= token < 50363 for token in tokens),
        "text": result["text"],
        "action": trace["action"],
        "reason": trace["reason"],
        "eof": trace["eof"],
        "word_publication": None
        if publication is None
        else {
            key: publication[key]
            for key in ("start_ms", "end_ms", "text", "word_start", "word_end")
        },
    }


def analyze(record):
    """Return count attribution, not a speed estimate or qualification verdict."""
    require(
        record["effective_identity"]["profile"]["profile_id"].startswith("tiny.en/"),
        "token split requires the declared tiny.en tokenizer identity",
    )
    cells = {}
    seen_arms = set()
    for cell_index, cell in enumerate(record["cells"]):
        arm = cell["arm"]
        require(arm not in seen_arms, f"duplicate arm: {arm}")
        seen_arms.add(arm)
        if arm in ARMS:
            cells[arm] = (cell_index, cell)
    require(set(cells) == set(ARMS), "missing comparison arm")
    count = len(cells[ARMS[0]][1]["windows"])
    require(count > 0, "empty window list")
    rows_by_arm = {}
    for arm, (cell_index, cell) in cells.items():
        require(
            cell["input_sha256"] == record["input"]["pcm_sha256"],
            f"{arm}: input identity mismatch",
        )
        require(
            len(cell["windows"]) == len(cell["decision_traces"]) == count,
            f"{arm}: window/trace count mismatch",
        )
        calls = [w["call_index"] for w in cell["windows"]]
        require(
            all(type(call) is int for call in calls)
            and calls == list(range(calls[0], calls[0] + count)),
            f"{arm}: native call indices not contiguous",
        )
        require(
            len({w["window_id"] for w in cell["windows"]}) == count,
            f"{arm}: duplicate window identity",
        )
        rows_by_arm[arm] = [window_row(cell, cell_index, i) for i in range(count)]
    rows = []
    for index, (control, candidate) in enumerate(
        zip(rows_by_arm[ARMS[0]], rows_by_arm[ARMS[1]])
    ):
        require(
            (control["end_ms"], control["eof"])
            == (candidate["end_ms"], candidate["eof"]),
            f"window {index}: comparison endpoint identity mismatch",
        )
        traces = [cells[arm][1]["decision_traces"][index] for arm in ARMS]
        rows.append(
            {
                "index": index,
                "control": control,
                "candidate": candidate,
                "delta": {key: candidate[key] - control[key] for key in COUNTS},
                "same_generated_text_and_tokens": traces[0]["result"]["text"]
                == traces[1]["result"]["text"]
                and traces[0]["result"]["metadata"]["tokens"]
                == traces[1]["result"]["metadata"]["tokens"],
            }
        )
    totals = {
        arm: {key: sum(row[key] for row in rows_by_arm[arm]) for key in COUNTS}
        for arm in ARMS
    }
    return {
        "claim_boundary": "Raw forward/token counts only; no speed or savings claim.",
        "control_arm": ARMS[0],
        "candidate_arm": ARMS[1],
        "window_count_per_arm": count,
        "totals": totals,
        "delta": {key: totals[ARMS[1]][key] - totals[ARMS[0]][key] for key in COUNTS},
        "decode_delta_groups": {
            name: {
                "indices": [i for i in indices if i < count],
                "decode_forwards_delta": sum(
                    rows[i]["delta"]["decode_forwards"] for i in indices if i < count
                ),
            }
            for name, indices in (
                ("windows_12_13", range(12, 14)),
                ("windows_14_17", range(14, 18)),
            )
        },
        "unchanged_decode_window_indices": [
            row["index"] for row in rows if row["delta"]["decode_forwards"] == 0
        ],
        "changed_alignment_window_indices": [
            row["index"] for row in rows if row["delta"]["alignment_forwards"] != 0
        ],
        "eof_window_indices": [row["index"] for row in rows if row["control"]["eof"]],
        "windows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args(argv)
    try:
        report = analyze(json.loads(args.artifact.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"decoder cost audit rejected: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
