"""One bounded T4 counterfactual experiment; imports start no remote work."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from infra import modal_acoustic_diagnostic as shared
from tools.analyze_stream_text_agreement import _commits
from tools.analyze_word_resolution import analyze_bytes
from whisper_runtime import AudioSpan
from whisper_runtime.adapters import NativeTimestampSegment

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = shared.REMOTE_ROOT
MANIFEST = "experiments/modal-word-resolution-v1.json"
INPUTS = "experiments/modal-word-resolution-inputs-v1.json"
EVIDENCE = "evidence/modal-t4-tiny-en-word-context-2026-09-06.json"
EVIDENCE_SHA = "f96e524e01574b6f2f8df3f9e09932b990313bba9354799a58703da043526b85"
DEVELOPMENT = [
    "context:concatenated-no-added-pauses",
    "context:mixed-continuous-noise64",
]
PARAMETERS = {
    "timestamp_tolerance_ms": 200,
    "max_end_extension_ms": 1000,
    "bootstrap_holdback_ms": 1000,
    "left_context_ms": 2000,
}
EXECUTION = {
    "gpu": "T4",
    "cpu": 2,
    "memory_mib": 4096,
    "timeout_seconds": 180,
    "max_containers": 1,
    "configured_retries": 0,
    "gpu_client_calls": 1,
    "maximum_native_windows": 10,
}
CLAIMS = dict.fromkeys(
    (
        "full_stream_recovery",
        "publication_authority",
        "general_efficiency",
        "acoustic_certification",
        "production_readiness",
        "physical_spending_cap",
    ),
    False,
)


def _corpus():
    return shared._corpus()


def registration(root=ROOT):
    value = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1-diagnostic",
        "id": "modal-word-resolution-v1",
        "modal_sdk": "1.5.5",
        "development_cells": DEVELOPMENT,
        "parameters": PARAMETERS,
        "execution": EXECUTION,
        "heldout_recipe": ["clean", "noise64"],
        "claim_boundary": CLAIMS,
    }
    for key, wanted in expected.items():
        if shared._hash(value.get(key)) != shared._hash(wanted):
            raise ValueError(f"unregistered {key}")
    return value


def _heldout(root):
    value = json.loads((root / INPUTS).read_text(encoding="utf-8"))
    fixtures = value["fixtures"]
    if len(fixtures) != 2:
        raise ValueError("exactly two held-out fixtures required")
    speakers = [item["source"]["speaker_id"] for item in fixtures]
    if len(set(speakers)) != 2 or set(speakers) & {6930, 1995, 672, 1272}:
        raise ValueError("held-out speakers overlap or were previously used")
    return fixtures


def build_cases(root=ROOT):
    """Build fixed past-only states; reference text is never a selector input."""
    raw = (root / EVIDENCE).read_bytes()
    if shared._sha(raw) != EVIDENCE_SHA:
        raise ValueError("development evidence changed")
    replay = analyze_bytes(raw)
    saved = json.loads(raw)
    corpus = _corpus()
    assets = corpus.REMOTE_ASSETS if root == REMOTE_ROOT else root / corpus.ASSET_PATH
    inputs, pcms = shared._cases(root, assets)
    metadata = {case["id"]: case for case in inputs["cases"]}
    states = {case["id"]: case for case in replay["cells"]}
    originals = {case["id"]: case for case in saved["cells"]}
    cases = []
    for name in DEVELOPMENT:
        state, original = states[name], originals[name]
        if state["terminal_kind"] != "eof" or not state["legacy_reason_matches"]:
            raise ValueError("development control must be a reproduced EOF failure")
        case_id = name.split(":", 1)[1]
        pcm = pcms[case_id]
        terminal = original["decision_traces"][-1]
        cases.append(
            {
                "id": name,
                "split": "development",
                "pcm": pcm,
                "reference_text": metadata[case_id]["reference_text"],
                "retained_ms": state["analysis_start_ms"],
                "head_ms": state["committed_through_ms"],
                "end_ms": state["analysis_end_ms"],
                "frozen_anchor": tuple(
                    NativeTimestampSegment(
                        AudioSpan(**word["span"]), word["text"], tuple(word["tokens"])
                    )
                    for word in state["retained_anchor"]
                ),
                "committed_text": " ".join(
                    text for _, text in _commits(original["events"])
                ),
                "expected_native_text": terminal["word_alignment"]["native"]["text"],
                "expected_words": terminal["word_alignment"]["words"],
                "expected_strict_reason": state["recorded_reason"],
            }
        )
    for index, item in enumerate(_heldout(root)):
        path = (root / item["pcm_path"]).resolve()
        if not path.is_relative_to((root / "artifacts/resolution-inputs-v1").resolve()):
            raise ValueError("held-out PCM path escapes its asset directory")
        pcm = path.read_bytes()
        count = item["sample_count"]
        if type(count) is not int or not 80000 <= count <= 192000:
            raise ValueError(
                "held-out duration must be between five and twelve seconds"
            )
        if len(pcm) != count * 2 or shared._sha(pcm) != item["pcm_sha256"]:
            raise ValueError("held-out PCM identity differs")
        if index:
            pcm, _ = importlib.import_module("tools.prepare_acoustic_cases")._add_noise(
                pcm
            )
        # End padding is digital zero and is part of the recorded PCM hash.
        pcm += b"\x00\x00" * (-count % 320)
        end = len(pcm) // 32
        cases.append(
            {
                "id": f"heldout:{item['id']}:{'noise64' if index else 'clean'}",
                "split": "heldout",
                "pcm": pcm,
                "reference_text": item["reference_text"],
                "end_ms": end,
                "bootstrap_ms": min(4000, end // 40 * 20),
            }
        )
    return cases


def input_records(root=ROOT):
    corpus = _corpus()
    return [
        {
            **{
                key: corpus.b._plain(value)
                for key, value in case.items()
                if key != "pcm"
            },
            "pcm_sha256": shared._sha(case["pcm"]),
            "sample_count": len(case["pcm"]) // 2,
        }
        for case in build_cases(root)
    ]


def snapshot(root=ROOT):
    expected = shared.snapshot(root)
    names = {item["path"] for item in expected["files"]}
    names.update(
        (
            MANIFEST,
            INPUTS,
            EVIDENCE,
            "infra/modal_word_resolution.py",
            "infra/word_resolution_worker.py",
            "tools/analyze_word_resolution.py",
            "tools/analyze_stream_text_agreement.py",
            "tools/word_anchor_reconciliation.py",
            "infra/modal_native_cuda_qualification.py",
        )
    )
    names.update(item["pcm_path"] for item in _heldout(root))
    files = [
        {
            "path": name,
            "size_bytes": (root / name).stat().st_size,
            "sha256": shared._sha((root / name).read_bytes()),
        }
        for name in sorted(names)
    ]
    return {"commit": expected["commit"], "digest": shared._hash(files), "files": files}


def verify_snapshot(expected, root=REMOTE_ROOT):
    shared.verify_snapshot(expected, root)


def experiment_complete(result, inputs):
    """Completion is not recognition improvement or stream qualification."""
    cells = result["cells"]
    if (
        [cell["id"] for cell in cells] != [item["id"] for item in inputs]
        or result["model"]["unchanged"] is not True
        or result["model"]["initial_sha256"] != result["model"]["final_sha256"]
        or result["capacity_restored"] is not True
        or result["native_window_count"] != 10
    ):
        return False
    calls = []
    for cell, item in zip(cells, inputs):
        if (
            cell["status"] != "evaluated"
            or cell["capacity_restored"] is not True
            or cell["outcome"]["comparison_valid"] is not True
            or cell["routed"]["uses_reference"] is not False
            or any(
                arm["publication_authority"] is not False
                for arm in cell["arms"].values()
            )
        ):
            return False
        if item["split"] == "development" and (
            cell["outcome"]["expected_native_text_reproduced"] is not True
            or any(
                cell["frozen_state"][key] != item[key]
                for key in ("head_ms", "retained_ms", "end_ms", "committed_text")
            )
        ):
            return False
        names = {"current", "alternative"}
        if item["split"] == "heldout":
            names.add("bootstrap")
        if set(cell["timings"]) != names:
            return False
        for timing in cell["timings"].values():
            if (
                timing["completed"] is not True
                or timing["capacity_restored"] is not True
            ):
                return False
            calls.append(timing["call_index"])
    return sorted(calls) == list(range(1, 11))


def run_worker(expected_snapshot):
    from infra.word_resolution_worker import run_worker as execute

    verify_snapshot(expected_snapshot)
    settings = registration(REMOTE_ROOT)
    result = execute(expected_snapshot)
    modal = importlib.import_module("modal")
    call_id = modal.current_function_call_id()
    inputs = input_records(REMOTE_ROOT)
    complete = experiment_complete(result, inputs)
    return {
        **result,
        "schema_version": "1-diagnostic",
        "experiment_id": settings["id"],
        "status": "completed" if complete else "failed",
        "qualified": False,
        "scope": settings["scope"],
        "claim_boundary": settings["claim_boundary"],
        "inputs": inputs,
        "source": {
            "snapshot": expected_snapshot,
            "registration_sha256": shared._sha((REMOTE_ROOT / MANIFEST).read_bytes()),
        },
        "worker": {
            "function_call_id": call_id,
            "function_call_id_sha256": shared._sha(call_id.encode()),
            "modal": str(modal.__version__),
        },
    }


def run(*, replay_id, preflight=False, confirm_paid_gpu=False, root=ROOT):
    if not preflight and confirm_paid_gpu is not True:
        raise ValueError("GPU work requires --confirm-paid-gpu")
    settings = registration(root)
    # Existing validated path/receipt mechanics, in a distinct replay directory.
    output, receipt, raw = shared._paths(root, replay_id, preflight)
    if any(path.exists() for path in (output, receipt, raw)):
        raise FileExistsError("attempt exists; will not overwrite or retry")
    expected, inputs = snapshot(root), input_records(root)
    corpus = _corpus()
    payload = corpus.b._transport_probe_payload()
    if not preflight:
        probe = json.loads(shared._paths(root, replay_id, True)[0].read_text())
        if (
            probe.get("status") != "completed"
            or probe.get("source_snapshot") != expected
            or probe.get("payload_sha256") != shared._sha(payload)
        ):
            raise ValueError("matching CPU preflight required")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "attempt-started",
                    "preflight": preflight,
                    "source_digest": expected["digest"],
                }
            )
            + "\n"
        )
    try:
        app, echo, execute = shared.resources(
            expected, root, worker_module="infra.modal_word_resolution"
        )
        with app.run(detach=False):
            if preflight:
                if echo.remote(payload) != payload:
                    raise ValueError("transport changed payload")
                record = {
                    "status": "completed",
                    "source_snapshot": expected,
                    "payload_sha256": shared._sha(payload),
                    "app_id": app.app_id,
                }
            else:
                payload = execute.remote()
                corpus.b._write_bytes_exclusive(raw, payload)
                record = corpus.b._decode_worker_record(
                    payload,
                    expected_snapshot=expected,
                    registration_sha256=shared._sha((root / MANIFEST).read_bytes()),
                    manifest=settings,
                )
                if (
                    record["source"]["snapshot"] != expected
                    or record["inputs"] != inputs
                ):
                    raise ValueError("worker inputs or source differ")
                status = (
                    "completed" if experiment_complete(record, inputs) else "failed"
                )
                if record["status"] != status or record["qualified"] is not False:
                    raise ValueError("worker completion claims differ")
                record["app_id"] = app.app_id
        corpus.c._write_json_exclusive(output, record)
        event = {
            "event": "record-written",
            "status": record["status"],
            "sha256": shared._sha(output.read_bytes()),
        }
    except BaseException as error:
        with receipt.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": "attempt-failed", "error_type": type(error).__name__}
                )
                + "\n"
            )
        raise
    with receipt.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
