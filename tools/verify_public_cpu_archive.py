"""Verify the explicitly redacted CPU archive without native inference or secrets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
combine = importlib.import_module("tools.verify_draft_restore_process").combine

ORIGINAL_SHA256 = "34334f9ec91904f5ad230c6937cb78485e2d65d0f86c6c72a3a4158d17a6600a"
ORIGINAL_REPORT_SHA256 = (
    "248b03810db17e244141d6296fd08d1effdac1cd3a856ca51b7cd4c0568d24c4"
)
ORIGINAL_MANIFEST_SHA256 = (
    "e5039b4503cc85a24ef92f10c0f0b9b05a4c6b1513946f63c7fc3a0917e1fd4c"
)
BACKEND_MANIFEST = ".tmp-composed-native/manifest.json"
DERIVATION = "PUBLIC_DERIVATION.json"
PUBLIC_MANIFEST = "PUBLIC_MANIFEST.json"
PERSONAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|[/]Users[/]|[/]home[/])")
REDACTED_ENTRIES = {
    "report.json",
    BACKEND_MANIFEST,
    "supplementary-packaging-hashes.json",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def verify_payload(files):
    manifest = json.loads(files[PUBLIC_MANIFEST])
    expected = {
        name: {"sha256": sha(raw), "size_bytes": len(raw)}
        for name, raw in sorted(files.items())
        if name != PUBLIC_MANIFEST
    }
    require(
        manifest == {"schema": "public-cpu-archive-manifest/v1", "files": expected},
        "public payload manifest mismatch",
    )
    derivation = json.loads(files[DERIVATION])
    require(
        derivation["schema"] == "public-cpu-archive-derivative/v1"
        and derivation["original_archive_sha256"] == ORIGINAL_SHA256
        and derivation["original_report_sha256"] == ORIGINAL_REPORT_SHA256
        and derivation["raw_report_byte_identity_preserved"] is False
        and derivation["executed_source_bytes_unchanged"] is True,
        "public derivative provenance mismatch",
    )
    originals = derivation["original_payload_sha256"]
    require(
        len(originals) == 74 and set(originals) <= set(files),
        "original inventory mismatch",
    )
    changes = {name for name, digest in originals.items() if sha(files[name]) != digest}
    require(changes == REDACTED_ENTRIES, "unexpected original payload change")
    require(
        originals["report.json"] == ORIGINAL_REPORT_SHA256
        and originals[BACKEND_MANIFEST] == ORIGINAL_MANIFEST_SHA256,
        "original private metadata hashes differ",
    )
    for name, raw in files.items():
        relative = PurePosixPath(name)
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and ":" not in name
            and "\\" not in name,
            "unsafe archive member",
        )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        require(PERSONAL_PATH.search(text) is None, "personal path remains: " + name)
    report = json.loads(files["report.json"])
    records = report["records"]
    preflight_raw = files[
        "artifacts/modal/integrated-draft-t4-20260907-v1/integrated-draft-preflight.json"
    ]
    frozen = json.loads(preflight_raw)["source"]["files"]
    require(
        len(frozen) == 63
        and all(
            sha(files[item["path"]]) == item["sha256"]
            and len(files[item["path"]]) == item["size_bytes"]
            for item in frozen
        ),
        "prior frozen preflight bytes differ",
    )
    require(
        len(records) == 3
        and combine(*records, report["orchestration"]) == report
        and report["status"] == "passed",
        "CPU report recomputation failed",
    )
    for record in records:
        source = record["sources"]
        require(
            source["frozen_preflight_sha256"] == sha(preflight_raw),
            "during-phase preflight digest differs",
        )
        require(len(source["files"]) == 65, "during-phase source inventory differs")
        require(
            all(sha(files[name]) == digest for name, digest in source["files"].items()),
            "during-phase source bytes differ",
        )
        require(
            record["manifest_sha256"] == ORIGINAL_MANIFEST_SHA256,
            "original execution manifest hash was rewritten",
        )
        require(
            record["executable"] == "<native-python>", "unredacted executable metadata"
        )
    require(
        json.loads(files[BACKEND_MANIFEST])["backend"]["path"] == "<verified-backend>",
        "unredacted backend metadata",
    )
    supplementary = json.loads(files["supplementary-packaging-hashes.json"])
    require(
        all(
            sha(files[name]) == digest
            for name, digest in supplementary["files"].items()
        ),
        "supplementary public hashes differ",
    )
    checkpoint = json.loads(files["report.savepoint.json"])
    payload = json.dumps(
        checkpoint["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    require(sha(payload) == checkpoint["sha256"], "checkpoint digest differs")
    require(
        sha(files["report.savepoint.json"]) == originals["report.savepoint.json"],
        "checkpoint bytes changed",
    )
    return {
        "status": "passed",
        "public_derivative": True,
        "payload_files": len(files) - 1,
        "original_archive_sha256": ORIGINAL_SHA256,
        "report_sha256": sha(files["report.json"]),
        "during_phase_source_files": 65,
        "raw_report_byte_identity_preserved": False,
    }


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(
            len(names) == len(set(names)) and archive.testzip() is None,
            "archive CRC or duplicate member failure",
        )
        require(
            sum(item.file_size for item in archive.infolist()) < 10_000_000,
            "archive exceeds bounded size",
        )
        result = verify_payload({name: archive.read(name) for name in names})
    return {
        **result,
        "archive_sha256": sha(path.read_bytes()),
        "size_bytes": path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2))


if __name__ == "__main__":
    main()
