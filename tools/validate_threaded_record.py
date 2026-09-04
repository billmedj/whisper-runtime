"""Validate one native OS-thread isolation evidence record."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from check_repository import ROOT, validate_native_threaded_evidence


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _find_nonfinite(value: Any, location: str) -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{location} contains a non-finite number"]
    if isinstance(value, dict):
        failures: list[str] = []
        for key, item in value.items():
            failures.extend(_find_nonfinite(item, f"{location}.{key}"))
        return failures
    if isinstance(value, list):
        failures = []
        for index, item in enumerate(value):
            failures.extend(_find_nonfinite(item, f"{location}[{index}]"))
        return failures
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a native same-model OS-thread isolation record."
    )
    parser.add_argument("record", type=Path)
    return parser.parse_args()


def validate_threaded_record(record: Any, location: str) -> list[str]:
    failures = _find_nonfinite(record, location)
    schema_path = ROOT / "evidence" / "native-threaded.schema.json"
    try:
        schema = json.loads(
            schema_path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return [f"cannot read the evidence schema: {error}"]

    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import SchemaError
    except ImportError:
        return ["JSON Schema validation is unavailable; install the validation extra"]
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        return [f"the native threaded schema is invalid: {error}"]

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = list(validator.iter_errors(record))
    for error in schema_errors:
        field = ".".join(str(part) for part in error.absolute_path)
        suffix = f".{field}" if field else ""
        failures.append(f"{location}{suffix}: {error.message}")
    if not schema_errors:
        failures.extend(validate_native_threaded_evidence(record, location))
    return failures


def main() -> int:
    args = parse_args()
    try:
        record = json.loads(
            args.record.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"FAIL: cannot read {args.record}: {error}")
        return 1
    failures = validate_threaded_record(record, args.record.name)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"evidence record passed: {args.record.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
