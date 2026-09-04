"""Validate a Modal CUDA-readiness record and its claim boundaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "evidence/modal-cuda-readiness.schema.json"
EXPECTED_ASSERTIONS = frozenset(
    {
        "runtime_source_pinned",
        "backend_source_pinned",
        "patch_manifest_verified",
        "checkpoint_verified_before_load",
        "input_fixture_verified",
        "torch_cuda_available",
        "requested_t4_observed",
        "model_loaded_on_cuda",
        "staged_decode_completed",
        "decode_result_matches_expected",
        "cleanup_idempotent",
        "persistent_model_state_unchanged",
        "model_hooks_unchanged",
        "model_reusable_after_decode",
        "cuda_timing_synchronized",
        "torch_cuda_memory_measured",
        "native_runtime_adapter_rejected_before_admission",
        "network_block_configured",
        "outbound_network_probe_denied",
        "model_cache_read_only",
    }
)
SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api_key|credential|identity_token|password|secret|token)(?:$|_)",
    re.IGNORECASE,
)
SENSITIVE_VALUE = re.compile(
    r"(?:github_pat_|gh[opsu]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"MODAL_TOKEN_(?:ID|SECRET))"
)
ABSOLUTE_USER_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/(?:home|Users)/[^/\s]+/)",
    re.IGNORECASE,
)


def _reject_nonstandard_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key is not permitted: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _walk(value: Any, location: str = "record") -> list[tuple[str, Any]]:
    result = [(location, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(_walk(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_walk(child, f"{location}[{index}]"))
    return result


def validate_semantics(
    record: Any,
    *,
    expected_runtime_commit: str | None = None,
    expected_runtime_tree: str | None = None,
) -> list[str]:
    """Return failures not expressible as local JSON Schema constraints."""

    failures: list[str] = []
    if not isinstance(record, dict):
        return ["record must be an object"]

    if expected_runtime_commit is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", expected_runtime_commit):
            failures.append("expected runtime commit must be a full lowercase Git hash")
        elif record.get("runtime", {}).get("git_commit") != expected_runtime_commit:
            failures.append("runtime.git_commit does not match the requested commit")
    if expected_runtime_tree is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", expected_runtime_tree):
            failures.append("expected runtime tree must be a full lowercase Git hash")
        elif record.get("runtime", {}).get("git_tree") != expected_runtime_tree:
            failures.append("runtime.git_tree does not match the requested tree")

    modal_record = record.get("modal", {})
    if not isinstance(modal_record, dict) or not modal_record.get("function_call_id"):
        failures.append("modal.function_call_id must identify the remote call")

    claims = record.get("claims", {})
    if (
        not isinstance(claims, dict)
        or claims.get("runtime_adapter_exercised") is not False
    ):
        failures.append("the record must not claim runtime-adapter execution")
    scope = record.get("scope", {})
    if (
        not isinstance(scope, dict)
        or scope.get("runtime_adapter_exercised") is not False
    ):
        failures.append("scope.runtime_adapter_exercised must be false")

    assertions = record.get("assertions", {})
    if not isinstance(assertions, dict):
        failures.append("assertions must be an object")
    else:
        missing = sorted(EXPECTED_ASSERTIONS - set(assertions))
        unexpected = sorted(set(assertions) - EXPECTED_ASSERTIONS)
        if missing:
            failures.append(f"assertions are missing: {', '.join(missing)}")
        if unexpected:
            failures.append(f"assertions are unexpected: {', '.join(unexpected)}")
        failed = sorted(name for name, value in assertions.items() if value is not True)
        if failed:
            failures.append(f"assertions failed: {', '.join(failed)}")

    timing = record.get("timing", {})
    if isinstance(timing, dict):
        phases = [
            timing.get("staged_decode_seconds"),
            timing.get("reuse_decode_seconds"),
        ]
        total = timing.get("total_seconds")
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in (*phases, total)
        ) and total < sum(phases):
            failures.append(
                "timing.total_seconds is shorter than measured decode phases"
            )

    memory = record.get("memory", {})
    if isinstance(memory, dict):
        before = memory.get("allocated_before_bytes")
        after = memory.get("allocated_after_bytes")
        peak = memory.get("peak_allocated_bytes")
        reserved = memory.get("peak_reserved_bytes")
        if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (before, after, peak, reserved)
        ):
            if peak < max(before, after):
                failures.append("memory peak is below an observed allocation")
            if reserved < peak:
                failures.append("reserved CUDA memory is below allocated CUDA memory")

    boundary = record.get("adapter_boundary", {})
    if isinstance(boundary, dict) and boundary.get(
        "budget_available_before"
    ) != boundary.get("budget_available_after"):
        failures.append("the rejected adapter call changed the resource budget")

    for location, value in _walk(record):
        key = location.rsplit(".", 1)[-1]
        if SENSITIVE_KEY.search(key):
            failures.append(f"{location} uses a sensitive field name")
        if isinstance(value, str):
            if SENSITIVE_VALUE.search(value):
                failures.append(f"{location} appears to contain a secret")
            if ABSOLUTE_USER_PATH.search(value):
                failures.append(f"{location} contains an absolute user path")
            if any(
                ord(character) < 32 and character not in "\t\n\r" for character in value
            ):
                failures.append(f"{location} contains a control character")

    return failures


def validate_record(
    record: Any,
    schema: Any,
    *,
    expected_runtime_commit: str | None = None,
    expected_runtime_tree: str | None = None,
) -> list[str]:
    failures: list[str] = []
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import SchemaError
    except ImportError:
        return ["JSON Schema validation is unavailable; install the validation extra"]
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        return [f"the CUDA-readiness schema is invalid: {error}"]

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for error in sorted(
        validator.iter_errors(record), key=lambda item: list(item.path)
    ):
        field = ".".join(str(part) for part in error.absolute_path)
        location = f"record.{field}" if field else "record"
        failures.append(f"{location}: {error.message}")
    if not failures:
        failures.extend(
            validate_semantics(
                record,
                expected_runtime_commit=expected_runtime_commit,
                expected_runtime_tree=expected_runtime_tree,
            )
        )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--expected-runtime-commit")
    parser.add_argument("--expected-runtime-tree")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record = read_json(args.record)
        schema = read_json(args.schema)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"FAIL: cannot read validation input: {error}")
        return 1
    failures = validate_record(
        record,
        schema,
        expected_runtime_commit=args.expected_runtime_commit,
        expected_runtime_tree=args.expected_runtime_tree,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Modal CUDA-readiness record passed schema and semantic validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
