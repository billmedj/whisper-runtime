from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

from compare_whisper_fixtures import compare_fixtures, validate_conformance_document

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".git",
    ".lake",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "generated",
}
TEXT_SUFFIXES = {".json", ".lean", ".md", ".ps1", ".py", ".toml", ".yml", ".yaml"}
OUTCOMES = {"success", "error", "cancelled", "deadline_exceeded"}
MEASUREMENT_FIELDS = {
    "active_requests_peak",
    "admission_delay_seconds",
    "alignment_capture_bytes",
    "alignment_seconds",
    "cancellation_delay_seconds",
    "decoder_batch_size_peak",
    "decoder_steps",
    "encoder_batch_size_peak",
    "encoder_calls",
    "execution_seconds",
    "fallback_attempts",
    "peak_device_memory_bytes",
    "peak_host_memory_bytes",
    "peak_hypothesis_count",
    "queue_delay_seconds",
    "real_time_factor",
    "resources_held_after_bytes",
    "state_isolation_failures",
    "wall_seconds",
}
INTEGER_MEASUREMENT_FIELDS = {
    "active_requests_peak",
    "alignment_capture_bytes",
    "decoder_batch_size_peak",
    "decoder_steps",
    "encoder_batch_size_peak",
    "encoder_calls",
    "fallback_attempts",
    "peak_device_memory_bytes",
    "peak_host_memory_bytes",
    "peak_hypothesis_count",
    "resources_held_after_bytes",
    "state_isolation_failures",
}
SEGMENT_FIELDS = {
    "id",
    "seek",
    "start",
    "end",
    "text",
    "tokens",
    "temperature",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
}


def tracked_text_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and not any(
            part in IGNORED_PARTS or part.startswith(".tmp-") for part in path.parts
        )
    ]


def contains_absolute_user_path(text: str) -> bool:
    """Detect native paths, including backslashes escaped in JSON strings."""
    drive_user = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE)
    posix_user = re.compile(r"/(?:home|Users)/[^/\s]+/")
    unc = re.compile(r"\\{2,}[A-Za-z0-9_.-]+\\+[A-Za-z0-9$_.-]+\\+")
    return bool(drive_user.search(text) or posix_user.search(text) or unc.search(text))


def check_portability(files: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if contains_absolute_user_path(text):
            failures.append(f"{path.relative_to(ROOT)} contains an absolute user path")
    return failures


def check_lean_sources() -> list[str]:
    failures: list[str] = []
    placeholder = re.compile(r"\b(?:sorry|admit|axiom)\b")
    for path in (ROOT / "formal" / "lean").rglob("*.lean"):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            code = line.split("--", 1)[0]
            if placeholder.search(code):
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number} contains a proof placeholder"
                )
    return failures


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_fields(
    value: Any,
    fields: set[str],
    location: str,
    failures: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        failures.append(f"{location} must be an object")
        return None
    missing = sorted(fields - set(value))
    if missing:
        failures.append(f"{location} is missing fields: {', '.join(missing)}")
    return value


def validate_fixture(fixture: Any, location: str) -> list[str]:
    failures: list[str] = []
    top = _require_fields(
        fixture,
        {
            "schema_version",
            "created_at",
            "outcome",
            "profile",
            "source",
            "environment",
            "model",
            "audio",
            "options",
            "comparison",
            "measurement",
        },
        location,
        failures,
    )
    if top is None:
        return failures
    allowed_top = {
        "schema_version",
        "created_at",
        "outcome",
        "profile",
        "source",
        "environment",
        "model",
        "audio",
        "options",
        "comparison",
        "measurement",
        "result",
        "termination",
    }
    unknown_top = sorted(set(top) - allowed_top)
    if unknown_top:
        failures.append(f"{location} has unknown fields: {', '.join(unknown_top)}")
    if top.get("schema_version") != "1":
        failures.append(f"{location}.schema_version must be '1'")
    try:
        dt.datetime.fromisoformat(str(top.get("created_at", "")).replace("Z", "+00:00"))
    except ValueError:
        failures.append(f"{location}.created_at must be an ISO 8601 timestamp")

    outcome = top.get("outcome")
    if outcome not in OUTCOMES:
        failures.append(f"{location}.outcome is invalid")
    if top.get("profile") not in {"reference", "optimized"}:
        failures.append(f"{location}.profile is invalid")
    if outcome == "success":
        if "termination" in top:
            failures.append(f"{location} success fixture must not contain termination")
        result = _require_fields(
            top.get("result"),
            {"text", "segments", "language"},
            f"{location}.result",
            failures,
        )
        if result is not None:
            if not isinstance(result.get("text"), str) or not isinstance(
                result.get("language"), str
            ):
                failures.append(f"{location}.result text and language must be strings")
            segments = result.get("segments")
            if not isinstance(segments, list):
                failures.append(f"{location}.result.segments must be an array")
            else:
                for index, segment_value in enumerate(segments):
                    segment = _require_fields(
                        segment_value,
                        SEGMENT_FIELDS,
                        f"{location}.result.segments[{index}]",
                        failures,
                    )
                    if segment is None:
                        continue
                    if not isinstance(segment.get("tokens"), list) or not all(
                        isinstance(token, int) and not isinstance(token, bool)
                        for token in segment.get("tokens", [])
                    ):
                        failures.append(
                            f"{location}.result.segments[{index}].tokens must contain integers"
                        )
                    words = segment.get("words")
                    if words is not None:
                        if not isinstance(words, list):
                            failures.append(
                                f"{location}.result.segments[{index}].words must be an array"
                            )
                        else:
                            for word_index, word in enumerate(words):
                                _require_fields(
                                    word,
                                    {"word", "start", "end", "probability"},
                                    f"{location}.result.segments[{index}].words[{word_index}]",
                                    failures,
                                )
    else:
        if "result" in top:
            failures.append(f"{location} terminal fixture must not contain result")
        _require_fields(
            top.get("termination"),
            {"reason_code", "message"},
            f"{location}.termination",
            failures,
        )

    source = _require_fields(
        top.get("source"),
        {"root", "git_commit", "dirty", "tree_sha256"},
        f"{location}.source",
        failures,
    )
    if source is not None:
        if not isinstance(source.get("root"), str) or not source.get("root"):
            failures.append(f"{location}.source.root must be a non-empty string")
        elif any(separator in source["root"] for separator in ("/", "\\", ":")):
            failures.append(f"{location}.source.root must be a portable checkout label")
        commit = source.get("git_commit")
        if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
            failures.append(f"{location}.source.git_commit must be a full commit hash")
        if source.get("dirty") not in (True, False, None):
            failures.append(f"{location}.source.dirty must be boolean or null")
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("tree_sha256", ""))):
            failures.append(f"{location}.source.tree_sha256 must be a SHA-256 digest")

    environment = _require_fields(
        top.get("environment"),
        {"python", "platform", "torch", "whisper_module"},
        f"{location}.environment",
        failures,
    )
    if environment is not None:
        for field in ("python", "platform", "torch", "whisper_module"):
            if not isinstance(environment.get(field), str):
                failures.append(f"{location}.environment.{field} must be a string")
        module_path = str(environment.get("whisper_module", ""))
        if (
            Path(module_path).is_absolute()
            or re.match(r"^[A-Za-z]:", module_path)
            or module_path.startswith(("\\\\", "//"))
        ):
            failures.append(f"{location}.environment.whisper_module must be portable")
    model = _require_fields(
        top.get("model"),
        {"name", "device", "dtype", "checkpoint_sha256", "dimensions"},
        f"{location}.model",
        failures,
    )
    if model is not None:
        for field in ("name", "device", "dtype"):
            if not isinstance(model.get(field), str) or not model.get(field):
                failures.append(f"{location}.model.{field} must be a non-empty string")
        if not isinstance(model.get("dimensions"), dict):
            failures.append(f"{location}.model.dimensions must be an object")
        digest = model.get("checkpoint_sha256")
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            failures.append(
                f"{location}.model.checkpoint_sha256 must be a SHA-256 digest"
            )

    audio = _require_fields(
        top.get("audio"),
        {
            "fixture_id",
            "path",
            "sha256",
            "size_bytes",
            "sample_rate_hz",
            "sample_start",
            "sample_end",
        },
        f"{location}.audio",
        failures,
    )
    if audio is not None:
        if not isinstance(audio.get("fixture_id"), str) or not audio.get("fixture_id"):
            failures.append(f"{location}.audio.fixture_id must be a non-empty string")
        size_bytes = audio.get("size_bytes")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            failures.append(
                f"{location}.audio.size_bytes must be a non-negative integer"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", str(audio.get("sha256", ""))):
            failures.append(f"{location}.audio.sha256 must be a SHA-256 digest")
        sample_rate = audio.get("sample_rate_hz")
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
        ):
            failures.append(
                f"{location}.audio.sample_rate_hz must be a positive integer"
            )
        sample_start = audio.get("sample_start")
        sample_end = audio.get("sample_end")
        if (
            not isinstance(sample_start, int)
            or isinstance(sample_start, bool)
            or sample_start < 0
        ):
            failures.append(
                f"{location}.audio.sample_start must be a non-negative integer"
            )
        if sample_end is not None and (
            not isinstance(sample_end, int)
            or isinstance(sample_end, bool)
            or not isinstance(sample_start, int)
            or isinstance(sample_start, bool)
            or sample_end < sample_start
        ):
            failures.append(f"{location}.audio.sample_end must follow sample_start")
        audio_path = str(audio.get("path", ""))
        if (
            Path(audio_path).is_absolute()
            or re.match(r"^[A-Za-z]:", audio_path)
            or audio_path.startswith(("\\\\", "//"))
        ):
            failures.append(f"{location}.audio.path must be portable")

    if not isinstance(top.get("options"), dict):
        failures.append(f"{location}.options must be an object")

    comparison = _require_fields(
        top.get("comparison"),
        {"timestamp_abs_tol", "numeric_abs_tol"},
        f"{location}.comparison",
        failures,
    )
    if comparison is not None:
        for field in ("timestamp_abs_tol", "numeric_abs_tol"):
            value = comparison.get(field)
            if not _is_number(value) or value < 0:
                failures.append(f"{location}.comparison.{field} must be non-negative")

    measurement = _require_fields(
        top.get("measurement"), MEASUREMENT_FIELDS, f"{location}.measurement", failures
    )
    if measurement is not None:
        for field in MEASUREMENT_FIELDS:
            value = measurement.get(field)
            if value is not None and (not _is_number(value) or value < 0):
                failures.append(
                    f"{location}.measurement.{field} must be non-negative or null"
                )
        for field in INTEGER_MEASUREMENT_FIELDS:
            value = measurement.get(field)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                failures.append(
                    f"{location}.measurement.{field} must be an integer or null"
                )
    return failures


def _read_json(path: Path, failures: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"{path.relative_to(ROOT)} is not valid JSON: {error}")
        return None


def check_fixture_schema() -> list[str]:
    failures: list[str] = []
    path = ROOT / "conformance" / "fixture.schema.json"
    schema = _read_json(path, failures)
    if not isinstance(schema, dict):
        return failures
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        failures.append("conformance/fixture.schema.json must use JSON Schema 2020-12")
    required = schema.get("required")
    if not isinstance(required, list) or not {
        "outcome",
        "profile",
        "comparison",
        "measurement",
    }.issubset(required):
        failures.append(
            "conformance/fixture.schema.json omits required contract fields"
        )
    if not isinstance(schema.get("oneOf"), list) or len(schema["oneOf"]) != 2:
        failures.append(
            "conformance/fixture.schema.json must discriminate success and termination"
        )
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError:
        failures.append(
            "JSON Schema validation is unavailable; install the 'validation' extra"
        )
    else:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            failures.append(
                f"conformance/fixture.schema.json is not a valid schema: {error}"
            )
    return failures


def validate_against_json_schema(fixture: dict[str, Any], location: str) -> list[str]:
    return validate_conformance_document(fixture, location)


def validate_audio_binding(
    audio: Any,
    manifest_record: dict[str, Any] | None,
    location: str,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(audio, dict) or manifest_record is None:
        return failures
    for fixture_field, manifest_field in (
        ("sha256", "sha256"),
        ("size_bytes", "size_bytes"),
        ("sample_rate_hz", "decoded_sample_rate_hz"),
    ):
        if audio.get(fixture_field) != manifest_record.get(manifest_field):
            failures.append(
                f"{location}.{fixture_field} does not match the audio manifest"
            )
    sample_end = audio.get("sample_end")
    sample_count = manifest_record.get("decoded_sample_count")
    if (
        isinstance(sample_end, int)
        and isinstance(sample_count, int)
        and sample_end > sample_count
    ):
        failures.append(f"{location}.sample_end exceeds the decoded audio length")
    return failures


def check_conformance_cases() -> list[str]:
    failures: list[str] = []
    manifest_path = ROOT / "conformance" / "audio-manifest.json"
    manifest = _read_json(manifest_path, failures)
    manifest_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1":
        failures.append("conformance/audio-manifest.json has an invalid schema_version")
    elif not isinstance(manifest.get("fixtures"), list):
        failures.append("conformance/audio-manifest.json fixtures must be an array")
    else:
        for index, record in enumerate(manifest["fixtures"]):
            location = f"conformance/audio-manifest.json.fixtures[{index}]"
            item = _require_fields(
                record,
                {
                    "id",
                    "source_url",
                    "sha256",
                    "size_bytes",
                    "decoded_sample_rate_hz",
                    "decoded_sample_count",
                    "license_context",
                },
                location,
                failures,
            )
            if item is None:
                continue
            fixture_id = item.get("id")
            if not isinstance(fixture_id, str) or not fixture_id:
                failures.append(f"audio fixture {index} has no stable id")
            elif fixture_id in manifest_by_id:
                failures.append(f"duplicate audio fixture id: {fixture_id}")
            else:
                manifest_by_id[fixture_id] = item
            source_url = item.get("source_url")
            if not isinstance(source_url, str) or not source_url.startswith("https://"):
                failures.append(f"{location}.source_url must use HTTPS")
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                failures.append(
                    f"audio fixture {fixture_id!r} has an invalid SHA-256 digest"
                )
            for field in (
                "size_bytes",
                "decoded_sample_rate_hz",
                "decoded_sample_count",
            ):
                value = item.get(field)
                minimum = 0 if field == "size_bytes" else 1
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < minimum
                ):
                    failures.append(
                        f"{location}.{field} must be an integer >= {minimum}"
                    )
            if not isinstance(item.get("license_context"), str) or not item.get(
                "license_context"
            ):
                failures.append(
                    f"{location}.license_context must be a non-empty string"
                )

    cases_path = ROOT / "conformance" / "cases.json"
    document = _read_json(cases_path, failures)
    if not isinstance(document, dict) or document.get("schema_version") != "1":
        failures.append("conformance/cases.json has an invalid schema_version")
        return failures
    cases = document.get("cases")
    if not isinstance(cases, list):
        failures.append("conformance/cases.json cases must be an array")
        return failures

    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            failures.append(f"conformance case {index} must be an object")
            continue
        _require_fields(
            case,
            {
                "id",
                "model",
                "audio_fixture_class",
                "options",
                "required_profile",
                "expected_resource_measurements",
                "status",
            },
            f"conformance/cases.json.cases[{index}]",
            failures,
        )
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            failures.append(f"conformance case {index} has no stable id")
            continue
        if case_id in seen:
            failures.append(f"duplicate conformance case id: {case_id}")
        seen.add(case_id)
        if case.get("status") not in {"planned", "implemented"}:
            failures.append(f"case {case_id} has an invalid status")
        if case.get("required_profile") not in {"reference", "optimized"}:
            failures.append(f"case {case_id} has an invalid required_profile")
        if not isinstance(case.get("model"), str) or not case.get("model"):
            failures.append(f"case {case_id} has no model")
        if not isinstance(case.get("audio_fixture_class"), str) or not case.get(
            "audio_fixture_class"
        ):
            failures.append(f"case {case_id} has no audio_fixture_class")
        if not isinstance(case.get("options"), dict):
            failures.append(f"case {case_id} options must be an object")
        expected = case.get("expected_resource_measurements")
        if not isinstance(expected, list) or not expected:
            failures.append(f"case {case_id} has no expected resource measurements")
            expected = []
        elif all(isinstance(field, str) for field in expected) and len(expected) != len(
            set(expected)
        ):
            failures.append(f"case {case_id} has duplicate resource measurements")
        for field in expected:
            if not isinstance(field, str) or field not in MEASUREMENT_FIELDS:
                failures.append(
                    f"case {case_id} uses an unknown measurement: {field!r}"
                )
        if case.get("status") != "implemented":
            continue

        audio_fixture_id = case.get("audio_fixture_id")
        manifest_record = (
            manifest_by_id.get(audio_fixture_id)
            if isinstance(audio_fixture_id, str)
            else None
        )
        if manifest_record is None:
            failures.append(f"implemented case {case_id} has no known audio_fixture_id")

        records = case.get("fixture_records")
        if not isinstance(records, dict):
            failures.append(f"implemented case {case_id} has no fixture_records object")
            continue
        loaded: dict[str, dict[str, Any]] = {}
        for role in ("reference", "candidate"):
            relative = records.get(role)
            if not isinstance(relative, str) or not relative:
                failures.append(f"implemented case {case_id} has no {role} record")
                continue
            fixture_path = (ROOT / "conformance" / relative).resolve()
            conformance_root = (ROOT / "conformance").resolve()
            if conformance_root not in fixture_path.parents:
                failures.append(f"implemented case {case_id} has an unsafe {role} path")
                continue
            if not fixture_path.is_file():
                failures.append(
                    f"implemented case {case_id} has no {role} fixture: {relative}"
                )
                continue
            fixture = _read_json(fixture_path, failures)
            if not isinstance(fixture, dict):
                continue
            location = fixture_path.relative_to(ROOT).as_posix()
            failures.extend(validate_fixture(fixture, location))
            failures.extend(validate_against_json_schema(fixture, location))
            source = fixture.get("source", {})
            model = fixture.get("model", {})
            audio = fixture.get("audio", {})
            if source.get("dirty") is not False:
                failures.append(f"{location} must come from a clean source checkout")
            if role == "reference":
                if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit", ""))):
                    failures.append(f"{location} reference requires a full git commit")
                if not re.fullmatch(
                    r"[0-9a-f]{64}", str(model.get("checkpoint_sha256", ""))
                ):
                    failures.append(
                        f"{location} reference requires a checkpoint digest"
                    )
            if model.get("name") != case.get("model"):
                failures.append(f"{location} model does not match case {case_id}")
            if fixture.get("profile") != case.get("required_profile"):
                failures.append(f"{location} profile does not match case {case_id}")
            if audio.get("fixture_id") != audio_fixture_id:
                failures.append(
                    f"{location} audio fixture does not match case {case_id}"
                )
            failures.extend(
                validate_audio_binding(audio, manifest_record, f"{location}.audio")
            )
            if fixture.get("options") != case.get("options"):
                failures.append(f"{location} options do not match case {case_id}")
            measurement = fixture.get("measurement", {})
            for field in expected:
                if field not in measurement:
                    failures.append(f"{location} has no required measurement: {field}")
                elif measurement[field] is None:
                    failures.append(
                        f"{location} did not record required measurement: {field}"
                    )
            loaded[role] = fixture

        if set(loaded) == {"reference", "candidate"}:
            for failure in compare_fixtures(
                loaded["reference"],
                loaded["candidate"],
                validate_documents=False,
            ):
                failures.append(f"case {case_id}: {failure}")
    return failures


def main() -> int:
    files = tracked_text_files()
    failures = [
        *check_portability(files),
        *check_lean_sources(),
        *check_fixture_schema(),
        *check_conformance_cases(),
    ]
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"repository checks passed ({len(files)} text files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
