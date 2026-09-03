from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "conformance" / "audio-manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and verify a conformance audio fixture."
    )
    parser.add_argument("fixture_id")
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_record(fixture_id: str) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in manifest["fixtures"]:
        if record["id"] == fixture_id:
            return record
    raise ValueError(f"unknown fixture id: {fixture_id}")


def main() -> int:
    args = parse_args()
    record = load_record(args.fixture_id)
    with urllib.request.urlopen(record["source_url"], timeout=30) as response:
        content = response.read(record["size_bytes"] + 1)
    if len(content) > record["size_bytes"]:
        raise RuntimeError(
            f"fixture exceeds the declared size: expected {record['size_bytes']} bytes"
        )
    actual_digest = sha256_bytes(content)
    if actual_digest != record["sha256"]:
        raise RuntimeError(
            f"fixture digest mismatch: expected {record['sha256']}, got {actual_digest}"
        )
    if len(content) != record["size_bytes"]:
        raise RuntimeError(
            f"fixture size mismatch: expected {record['size_bytes']}, got {len(content)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(f"wrote verified fixture to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
