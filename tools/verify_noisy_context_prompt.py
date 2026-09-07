"""Verify the noisy prompt diagnostic locally, without running inference.

One known receipt has an incorrect control-comparison flag: Python tuples were
compared with JSON lists. Preserve that receipt. Correct only an in-memory copy
for validation, identify it by its full canonical hash, and report the change.
No native result, alignment, score, model identity or execution claim is edited.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra import modal_noisy_context_prompt as p

KNOWN_CONTROL_ERROR_SHA = (
    "03dde4ce7468ec5904f6bab6fd0252b6585a87097b5d5345672cf4b8e672c92c"
)


def verify(record, root=p.ROOT):
    digest = p.shared._hash(record)
    source = p.source_case(root)
    replayed = p.control(record["cells"], source)
    checked = copy.deepcopy(record)
    correction = record["control"] != replayed
    if correction:
        p._require(
            digest == KNOWN_CONTROL_ERROR_SHA
            and replayed["retained_null_alignment_equal"] is True
            and record["control"]
            == {**replayed, "retained_null_alignment_equal": False},
            "unrecognized control mismatch; no receipt correction permitted",
        )
        checked["control"] = replayed
    p.validate_record(checked, record["source"]["snapshot"], root)
    p._require(p.shared._hash(record) == digest, "source receipt changed during replay")
    return dict(
        schema_version="1-offline-validation",
        observation_sha256=digest,
        dispatched_snapshot_digest=record["source"]["snapshot"]["digest"],
        verifier_sha256=p.shared._sha(Path(__file__).read_bytes()),
        current_producer_sha256=p.shared._sha((Path(root) / p.PRODUCER).read_bytes()),
        original_control=record["control"],
        replayed_control=replayed,
        report_only_correction=correction,
        corrected_fields=["control.retained_null_alignment_equal"]
        if correction
        else [],
        all_other_record_fields_unchanged=True,
        record_checks_pass=True,
        inference_run=False,
        publication_authorized=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipt", type=Path, help="JSON or original compressed .zlib receipt"
    )
    parser.add_argument(
        "--archive", type=Path, help="save the unchanged decoded receipt"
    )
    parser.add_argument(
        "--report", type=Path, help="save the separate offline validation"
    )
    args = parser.parse_args(argv)
    targets = [path for path in (args.archive, args.report) if path is not None]
    if len({path.resolve() for path in targets}) != len(targets) or any(
        path.exists() for path in targets
    ):
        raise FileExistsError("output paths must be distinct and must not exist")
    raw = args.receipt.read_bytes()
    record = json.loads(zlib.decompress(raw) if args.receipt.suffix == ".zlib" else raw)
    report = verify(record)
    report["input_file_sha256"] = p.shared._sha(raw)
    writer = p._corpus().c._write_json_exclusive
    if args.archive is not None:
        writer(args.archive, record)
    if args.report is not None:
        writer(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
