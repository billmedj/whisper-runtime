from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXACT_IDENTITY_PATHS = (
    ("profile",),
    ("model", "name"),
    ("model", "checkpoint_sha256"),
    ("audio", "fixture_id"),
    ("audio", "sha256"),
    ("audio", "size_bytes"),
    ("audio", "sample_rate_hz"),
    ("audio", "sample_start"),
    ("audio", "sample_end"),
    ("options",),
)
TIMESTAMP_FIELDS = {"start", "end"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Whisper conformance fixtures."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--timestamp-tolerance", type=float)
    parser.add_argument("--numeric-tolerance", type=float)
    return parser.parse_args()


def _value_at(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def validate_conformance_document(document: Any, location: str) -> list[str]:
    """Validate one fixture and fail closed when validation is unavailable."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import SchemaError
    except ImportError:
        return [
            f"{location}: JSON Schema validation is unavailable; "
            "install the 'validation' extra"
        ]

    schema_path = ROOT / "conformance" / "fixture.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, json.JSONDecodeError, SchemaError) as error:
        return [f"{schema_path.relative_to(ROOT)} is not a valid schema: {error}"]

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    failures: list[str] = []
    for error in sorted(
        validator.iter_errors(document),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        field = ".".join(str(part) for part in error.absolute_path)
        suffix = f".{field}" if field else ""
        failures.append(f"{location}{suffix}: {error.message}")
    return failures


def _compare_value(
    expected: Any,
    actual: Any,
    path: str,
    failures: list[str],
    *,
    timestamp_tolerance: float,
    numeric_tolerance: float,
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            failures.append(f"{path} differs: {expected!r} != {actual!r}")
        return

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        field = path.rsplit(".", 1)[-1]
        if isinstance(expected, int) and isinstance(actual, int):
            if expected != actual:
                failures.append(f"{path} differs: {expected!r} != {actual!r}")
            return
        tolerance = (
            timestamp_tolerance if field in TIMESTAMP_FIELDS else numeric_tolerance
        )
        if not math.isclose(
            float(expected),
            float(actual),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            failures.append(f"{path} differs: {expected!r} != {actual!r}")
        return

    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            failures.append(f"{path}.{key} is missing from candidate")
        for key in sorted(actual_keys - expected_keys):
            failures.append(f"{path}.{key} is not present in reference")
        for key in sorted(expected_keys & actual_keys):
            _compare_value(
                expected[key],
                actual[key],
                f"{path}.{key}",
                failures,
                timestamp_tolerance=timestamp_tolerance,
                numeric_tolerance=numeric_tolerance,
            )
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            failures.append(f"{path} length differs: {len(expected)} != {len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare_value(
                left,
                right,
                f"{path}[{index}]",
                failures,
                timestamp_tolerance=timestamp_tolerance,
                numeric_tolerance=numeric_tolerance,
            )
        return

    if type(expected) is not type(actual) or expected != actual:
        failures.append(f"{path} differs: {expected!r} != {actual!r}")


def compare_fixtures(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    timestamp_tolerance: float | None = None,
    numeric_tolerance: float | None = None,
    validate_documents: bool = True,
) -> list[str]:
    failures: list[str] = []
    if validate_documents:
        failures.extend(validate_conformance_document(reference, "reference"))
        failures.extend(validate_conformance_document(candidate, "candidate"))
        if failures:
            return failures

    expected_outcome = reference.get("outcome")
    actual_outcome = candidate.get("outcome")
    if expected_outcome != actual_outcome:
        return [f"outcome differs: {expected_outcome!r} != {actual_outcome!r}"]

    comparison = reference.get("comparison", {})
    timestamp_abs_tol = (
        timestamp_tolerance
        if timestamp_tolerance is not None
        else float(comparison.get("timestamp_abs_tol", 0.0))
    )
    numeric_abs_tol = (
        numeric_tolerance
        if numeric_tolerance is not None
        else float(comparison.get("numeric_abs_tol", 0.0))
    )
    if timestamp_abs_tol < 0 or numeric_abs_tol < 0:
        return ["comparison tolerances must be non-negative"]

    for path in EXACT_IDENTITY_PATHS:
        expected = _value_at(reference, path)
        actual = _value_at(candidate, path)
        if expected != actual:
            failures.append(f"{'.'.join(path)} differs: {expected!r} != {actual!r}")

    payload_name = "result" if expected_outcome == "success" else "termination"
    _compare_value(
        reference.get(payload_name),
        candidate.get(payload_name),
        payload_name,
        failures,
        timestamp_tolerance=timestamp_abs_tol,
        numeric_tolerance=numeric_abs_tol,
    )
    return failures


def main() -> int:
    args = parse_args()
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"FAIL: cannot read a fixture: {error}")
        return 1
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        print("FAIL: each fixture must be a JSON object")
        return 1
    failures = compare_fixtures(
        reference,
        candidate,
        timestamp_tolerance=args.timestamp_tolerance,
        numeric_tolerance=args.numeric_tolerance,
    )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("PASS: conformance payloads match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
