from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from compare_whisper_fixtures import compare_fixtures, validate_conformance_document
from validate_modal_native_cuda_qualification import validate_qualification_manifest

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
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"{path.relative_to(ROOT)} is not valid JSON: {error}")
        return None


def _same_recorded_result(
    left: Any,
    right: Any,
    *,
    absolute_tolerance: float,
) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    for field in (
        "text",
        "language",
        "token_count",
        "tokens_sha256",
        "audio_features_sha256",
    ):
        if left.get(field) != right.get(field):
            return False
    for field in (
        "temperature",
        "compression_ratio",
        "avg_logprob",
        "no_speech_prob",
    ):
        left_value = left.get(field)
        right_value = right.get(field)
        if left_value is None or right_value is None:
            if left_value is not right_value:
                return False
        elif not _is_number(left_value) or not _is_number(right_value):
            return False
        elif not math.isclose(
            left_value,
            right_value,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            return False
    return True


def validate_native_interleaving_evidence(
    record: Any,
    location: str,
) -> list[str]:
    """Check relations that JSON Schema cannot express by itself."""

    failures: list[str] = []
    if not isinstance(record, dict):
        return [f"{location} must be an object"]

    try:
        dt.datetime.fromisoformat(
            str(record.get("recorded_at", "")).replace("Z", "+00:00")
        )
    except ValueError:
        failures.append(f"{location}.recorded_at must be an ISO 8601 timestamp")

    input_record = record.get("input", {})
    if isinstance(input_record, dict):
        input_path = str(input_record.get("path", ""))
        input_parts = Path(input_path).parts
        if (
            Path(input_path).is_absolute()
            or re.match(r"^[A-Za-z]:", input_path)
            or input_path.startswith(("\\\\", "//"))
            or ".." in input_parts
        ):
            failures.append(f"{location}.input.path must be portable")
        cancelled = input_record.get("cancelled", {})
        survivor = input_record.get("survivor", {})
        if isinstance(cancelled, dict) and isinstance(survivor, dict):
            if cancelled.get("sample_start") != survivor.get("sample_start"):
                failures.append(
                    f"{location} derived inputs must use the same sample start"
                )
            cancelled_end = cancelled.get("sample_end")
            survivor_end = survivor.get("sample_end")
            if not (
                isinstance(cancelled_end, int)
                and isinstance(survivor_end, int)
                and cancelled_end < survivor_end
                and survivor_end == input_record.get("source_sample_count")
            ):
                failures.append(
                    f"{location} cancelled input must be shorter than the survivor"
                )
            for field in ("pcm_sha256", "mel_sha256"):
                if cancelled.get(field) == survivor.get(field):
                    failures.append(
                        f"{location} derived inputs must have distinct {field} values"
                    )

        manifest_path = ROOT / "conformance" / "audio-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        fixture_id = input_record.get("fixture_id")
        fixtures = manifest.get("fixtures", []) if isinstance(manifest, dict) else []
        manifest_record = next(
            (
                fixture
                for fixture in fixtures
                if isinstance(fixture, dict) and fixture.get("id") == fixture_id
            ),
            None,
        )
        if manifest_record is None:
            failures.append(f"{location} uses an unknown audio fixture")
        else:
            for evidence_field, manifest_field in (
                ("file_sha256", "sha256"),
                ("size_bytes", "size_bytes"),
                ("sample_rate_hz", "decoded_sample_rate_hz"),
                ("source_sample_count", "decoded_sample_count"),
            ):
                if input_record.get(evidence_field) != manifest_record.get(
                    manifest_field
                ):
                    failures.append(
                        f"{location}.input.{evidence_field} does not match "
                        "the audio manifest"
                    )

    model = record.get("model", {})
    if isinstance(model, dict) and model.get("loaded_state_before") != model.get(
        "loaded_state_after"
    ):
        failures.append(f"{location} loaded model state changed during the check")
    if isinstance(model, dict) and model.get("execution_state_before") != model.get(
        "execution_state_after"
    ):
        failures.append(f"{location} model execution state changed during the check")

    execution = record.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    survivor_steps = execution.get("survivor_steps")
    expected_schedule = [
        "start:cancelled",
        "start:survivor",
        "prefill:cancelled",
        "prefill:survivor",
        "step:cancelled:1",
        "step:survivor:1",
        "cancel:cancelled",
        "cleanup:cancelled:idempotent",
    ]
    if isinstance(survivor_steps, int) and not isinstance(survivor_steps, bool):
        expected_schedule.extend(
            f"step:survivor:{step}" for step in range(2, survivor_steps + 1)
        )
    expected_schedule.extend(["finalize:survivor", "cleanup:survivor:idempotent"])
    if execution.get("schedule") != expected_schedule:
        failures.append(f"{location}.execution.schedule is not the expected schedule")
    if execution.get("cancelled_steps") != 1:
        failures.append(f"{location} must cancel after exactly one token step")

    tolerance = execution.get("numeric_absolute_tolerance")
    if not _is_number(tolerance) or tolerance < 0:
        tolerance = 0.0
    results = record.get("results", {})
    if isinstance(results, dict):
        baseline = results.get("isolated_baseline")
        for role in ("survivor", "reuse_control"):
            if not _same_recorded_result(
                baseline,
                results.get(role),
                absolute_tolerance=tolerance,
            ):
                failures.append(
                    f"{location}.results.{role} differs from isolated_baseline"
                )

    assertions = record.get("assertions", {})
    if (
        not isinstance(assertions, dict)
        or not assertions
        or any(value is not True for value in assertions.values())
    ):
        failures.append(f"{location} contains a failed or missing assertion")

    manifest_path = ROOT / "patches" / "openai-whisper" / "SHA256SUMS"
    backend = record.get("backend", {})
    if isinstance(backend, dict) and manifest_path.is_file():
        observed_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if backend.get("patch_manifest_sha256") != observed_manifest:
            failures.append(f"{location} does not match the current patch manifest")
    return failures


def check_native_interleaving_evidence() -> list[str]:
    failures: list[str] = []
    schema_path = ROOT / "evidence" / "native-interleaving.schema.json"
    schema = _read_json(schema_path, failures)
    if not isinstance(schema, dict):
        return failures
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        failures.append(
            "evidence/native-interleaving.schema.json must use JSON Schema 2020-12"
        )
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import SchemaError
    except ImportError:
        failures.append(
            "JSON Schema validation is unavailable; install the 'validation' extra"
        )
        return failures

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        failures.append(
            f"evidence/native-interleaving.schema.json is not a valid schema: {error}"
        )
        return failures

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    records = sorted((ROOT / "evidence").glob("native*interleaving*.json"))
    for path in records:
        if path.name == "native-interleaving.schema.json":
            continue
        record = _read_json(path, failures)
        if record is None:
            continue
        location = path.relative_to(ROOT).as_posix()
        for error in validator.iter_errors(record):
            field = ".".join(str(part) for part in error.absolute_path)
            suffix = f".{field}" if field else ""
            failures.append(f"{location}{suffix}: {error.message}")
        failures.extend(validate_native_interleaving_evidence(record, location))
    return failures


def validate_native_threaded_evidence(record: Any, location: str) -> list[str]:
    """Check cross-field relations in one OS-thread isolation record."""

    failures: list[str] = []
    if not isinstance(record, dict):
        return [f"{location} must be an object"]
    try:
        recorded_at = dt.datetime.fromisoformat(
            str(record.get("recorded_at", "")).replace("Z", "+00:00")
        )
        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError("timestamp has no UTC offset")
    except ValueError:
        failures.append(
            f"{location}.recorded_at must be an ISO 8601 timestamp with an offset"
        )

    input_record = record.get("input", {})
    if isinstance(input_record, dict):
        input_path = str(input_record.get("path", ""))
        posix_path = PurePosixPath(input_path)
        path_parts = input_path.split("/")
        if (
            not input_path
            or "\\" in input_path
            or ":" in input_path
            or posix_path.is_absolute()
            or posix_path.as_posix() != input_path
            or any(part in {"", ".", ".."} for part in path_parts)
        ):
            failures.append(
                f"{location}.input.path must be a normalized relative POSIX path"
            )
        cancelled = input_record.get("cancelled", {})
        survivor = input_record.get("survivor", {})
        source_count = input_record.get("source_sample_count")
        if isinstance(cancelled, dict) and isinstance(survivor, dict):
            if cancelled.get("sample_start") != 0 or survivor.get("sample_start") != 0:
                failures.append(f"{location} derived inputs must start at sample zero")
            if (
                not isinstance(source_count, int)
                or isinstance(source_count, bool)
                or cancelled.get("sample_end") != source_count // 2
                or survivor.get("sample_end") != source_count
            ):
                failures.append(f"{location} derived input ranges are inconsistent")
            for field in ("pcm_sha256", "mel_sha256", "encoded_features_sha256"):
                if cancelled.get(field) == survivor.get(field):
                    failures.append(
                        f"{location} derived inputs must have distinct {field} values"
                    )

        manifest_path = ROOT / "conformance" / "audio-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        fixtures = manifest.get("fixtures", []) if isinstance(manifest, dict) else []
        manifest_record = next(
            (
                fixture
                for fixture in fixtures
                if isinstance(fixture, dict)
                and fixture.get("id") == input_record.get("fixture_id")
            ),
            None,
        )
        if manifest_record is None:
            failures.append(f"{location} uses an unknown audio fixture")
        else:
            for evidence_field, manifest_field in (
                ("file_sha256", "sha256"),
                ("size_bytes", "size_bytes"),
                ("sample_rate_hz", "decoded_sample_rate_hz"),
                ("source_sample_count", "decoded_sample_count"),
            ):
                if input_record.get(evidence_field) != manifest_record.get(
                    manifest_field
                ):
                    failures.append(
                        f"{location}.input.{evidence_field} does not match "
                        "the audio manifest"
                    )

    model = record.get("model", {})
    if isinstance(model, dict):
        if model.get("loaded_state_before") != model.get("loaded_state_after"):
            failures.append(f"{location} loaded model state changed during the check")
        if model.get("execution_state_before") != model.get("execution_state_after"):
            failures.append(
                f"{location} model execution state changed during the check"
            )

    execution = record.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    threads = execution.get("threads", [])
    calls = execution.get("controlled_forward_lifetimes", [])
    event_times: dict[str, dict[str, int]] = {}
    if isinstance(threads, list) and len(threads) == 2:
        by_role = {
            role: thread
            for thread in threads
            if isinstance(thread, dict)
            and isinstance((role := thread.get("role")), str)
        }
        if set(by_role) != {"cancelled", "survivor"}:
            failures.append(f"{location}.execution.threads must contain both roles")
        else:
            if len({item.get("python_thread_id") for item in by_role.values()}) != 2:
                failures.append(f"{location} Python worker thread IDs are not distinct")
            if len({item.get("native_thread_id") for item in by_role.values()}) != 2:
                failures.append(f"{location} native worker thread IDs are not distinct")
            survivor_steps = execution.get("survivor_steps")
            expected_events = {
                "cancelled": [
                    "start",
                    "prefill",
                    "step:1",
                    "cleanup",
                    "cleanup:idempotent",
                    "step:rejected",
                    "finalize:rejected",
                ],
                "survivor": [
                    "start",
                    "prefill",
                    *(
                        [f"step:{index}" for index in range(1, survivor_steps + 1)]
                        if isinstance(survivor_steps, int)
                        and not isinstance(survivor_steps, bool)
                        else []
                    ),
                    "finalize",
                    "finalize:rejected",
                    "cleanup:idempotent",
                ],
            }
            for role, thread in by_role.items():
                events = thread.get("events", [])
                started = thread.get("started_ns")
                finished = thread.get("finished_ns")
                if not (
                    isinstance(started, int)
                    and isinstance(finished, int)
                    and started < finished
                ):
                    failures.append(f"{location} {role} worker interval is invalid")
                    continue
                if not isinstance(events, list) or not all(
                    isinstance(event, dict) for event in events
                ):
                    failures.append(f"{location} {role} worker events are invalid")
                    continue
                names = [event.get("name") for event in events]
                times = [event.get("at_ns") for event in events]
                if names != expected_events[role]:
                    failures.append(
                        f"{location} {role} worker event sequence is invalid"
                    )
                if not all(
                    isinstance(at_ns, int) and not isinstance(at_ns, bool)
                    for at_ns in times
                ):
                    failures.append(
                        f"{location} {role} worker event timestamps are invalid"
                    )
                elif not (
                    all(started <= at_ns <= finished for at_ns in times)
                    and all(
                        earlier <= later
                        for earlier, later in zip(times, times[1:], strict=False)
                    )
                ):
                    failures.append(
                        f"{location} {role} worker event timestamps are unordered"
                    )
                elif all(isinstance(name, str) for name in names):
                    event_times[role] = dict(zip(names, times, strict=True))
    else:
        by_role = {}

    if set(event_times) == {"cancelled", "survivor"}:
        cleanup_ns = event_times["cancelled"].get("cleanup")
        cancelled_step_ns = event_times["cancelled"].get("step:1")
        survivor_step_ns = event_times["survivor"].get("step:1")
        if not (
            isinstance(cleanup_ns, int)
            and isinstance(cancelled_step_ns, int)
            and isinstance(survivor_step_ns, int)
            and max(cancelled_step_ns, survivor_step_ns) < cleanup_ns
        ):
            failures.append(
                f"{location} cancellation must follow both first token steps"
            )

    if isinstance(calls, list) and len(calls) == 2:
        call_by_role = {
            role: call
            for call in calls
            if isinstance(call, dict) and isinstance((role := call.get("role")), str)
        }
        if set(call_by_role) != {"cancelled", "survivor"}:
            failures.append(
                f"{location}.execution.controlled_forward_lifetimes must contain both roles"
            )
        else:
            barriers: list[int] = []
            started_calls: list[int] = []
            finished_calls: list[int] = []
            for role, call in call_by_role.items():
                values = (
                    call.get("inner_barrier_entered_ns"),
                    call.get("forward_started_ns"),
                    call.get("forward_finished_ns"),
                )
                if not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in values
                ):
                    failures.append(f"{location} {role} forward interval is invalid")
                    continue
                inner_barrier, call_started, call_finished = values
                if not call_started <= inner_barrier < call_finished:
                    failures.append(f"{location} {role} forward interval is unordered")
                barriers.append(inner_barrier)
                started_calls.append(call_started)
                finished_calls.append(call_finished)
                owner = by_role.get(role)
                if isinstance(owner, dict) and (
                    call.get("python_thread_id") != owner.get("python_thread_id")
                    or call.get("native_thread_id") != owner.get("native_thread_id")
                    or not owner.get("started_ns")
                    <= call_started
                    <= inner_barrier
                    < call_finished
                    <= owner.get("finished_ns")
                ):
                    failures.append(
                        f"{location} {role} forward call has the wrong owner"
                    )
                role_events = event_times.get(role)
                if isinstance(role_events, dict):
                    start_event = role_events.get("start")
                    prefill_event = role_events.get("prefill")
                    if not (
                        isinstance(start_event, int)
                        and isinstance(prefill_event, int)
                        and start_event
                        <= call_started
                        <= inner_barrier
                        < call_finished
                        <= prefill_event
                    ):
                        failures.append(
                            f"{location} {role} forward call is outside its prefill"
                        )
            if len(barriers) == 2 and not (
                max(started_calls) < min(finished_calls)
                and max(barriers) < min(finished_calls)
            ):
                failures.append(
                    f"{location} decoder forward-call bodies do not overlap"
                )

    if execution.get("cancelled_steps") != 1:
        failures.append(f"{location} must cancel after exactly one token step")
    elapsed = execution.get("elapsed_seconds", {})
    if isinstance(elapsed, dict) and all(
        _is_number(elapsed.get(field))
        for field in (
            "encoder_preparation",
            "baseline",
            "threaded",
            "reuse_control",
            "total",
        )
    ):
        measured_phases = sum(
            elapsed[field]
            for field in (
                "encoder_preparation",
                "baseline",
                "threaded",
                "reuse_control",
            )
        )
        if elapsed["total"] < measured_phases:
            failures.append(
                f"{location}.execution.elapsed_seconds.total is shorter than "
                "its measured phases"
            )
    tolerance = execution.get("numeric_absolute_tolerance")
    if not _is_number(tolerance) or tolerance < 0:
        tolerance = 0.0
    results = record.get("results", {})
    if isinstance(results, dict):
        baseline = results.get("isolated_baseline")
        for role in ("survivor", "reuse_control"):
            if not _same_recorded_result(
                baseline,
                results.get(role),
                absolute_tolerance=tolerance,
            ):
                failures.append(
                    f"{location}.results.{role} differs from isolated_baseline"
                )
        input_record = record.get("input", {})
        survivor_input = (
            input_record.get("survivor", {}) if isinstance(input_record, dict) else {}
        )
        prepared_survivor_sha256 = (
            survivor_input.get("encoded_features_sha256")
            if isinstance(survivor_input, dict)
            else None
        )
        for role in ("isolated_baseline", "survivor", "reuse_control"):
            result = results.get(role)
            if (
                not isinstance(result, dict)
                or result.get("audio_features_sha256") != prepared_survivor_sha256
            ):
                failures.append(
                    f"{location}.results.{role}.audio_features_sha256 does not "
                    "match input.survivor.encoded_features_sha256"
                )

    assertions = record.get("assertions", {})
    if (
        not isinstance(assertions, dict)
        or not assertions
        or any(value is not True for value in assertions.values())
    ):
        failures.append(f"{location} contains a failed or missing assertion")

    manifest_path = ROOT / "patches" / "openai-whisper" / "SHA256SUMS"
    backend = record.get("backend", {})
    if isinstance(backend, dict) and manifest_path.is_file():
        observed_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if backend.get("patch_manifest_sha256") != observed_manifest:
            failures.append(f"{location} does not match the current patch manifest")
    return failures


def check_native_threaded_evidence() -> list[str]:
    failures: list[str] = []
    schema_path = ROOT / "evidence" / "native-threaded.schema.json"
    schema = _read_json(schema_path, failures)
    if not isinstance(schema, dict):
        return failures
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        failures.append(
            "evidence/native-threaded.schema.json must use JSON Schema 2020-12"
        )
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import SchemaError
    except ImportError:
        failures.append(
            "JSON Schema validation is unavailable; install the 'validation' extra"
        )
        return failures
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        failures.append(
            f"evidence/native-threaded.schema.json is not a valid schema: {error}"
        )
        return failures
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for path in sorted((ROOT / "evidence").glob("native*threaded*.json")):
        if path.name == "native-threaded.schema.json":
            continue
        record = _read_json(path, failures)
        if record is None:
            continue
        location = path.relative_to(ROOT).as_posix()
        schema_errors = list(validator.iter_errors(record))
        for error in schema_errors:
            field = ".".join(str(part) for part in error.absolute_path)
            suffix = f".{field}" if field else ""
            failures.append(f"{location}{suffix}: {error.message}")
        if not schema_errors:
            failures.extend(validate_native_threaded_evidence(record, location))
    return failures


def _resource_components(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, dict):
        return None
    components = tuple(
        value.get(field) for field in ("memory_bytes", "compute_units", "stream_slots")
    )
    if not all(
        isinstance(item, int) and not isinstance(item, bool) for item in components
    ):
        return None
    return components


def validate_native_runtime_concurrency_evidence(
    record: Any,
    location: str,
) -> list[str]:
    """Check cross-field relations in one adapter concurrency record."""

    failures: list[str] = []
    if not isinstance(record, dict):
        return [f"{location} must be an object"]
    try:
        recorded_at = dt.datetime.fromisoformat(
            str(record.get("recorded_at", "")).replace("Z", "+00:00")
        )
        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError("timestamp has no UTC offset")
    except ValueError:
        failures.append(
            f"{location}.recorded_at must be an ISO 8601 timestamp with an offset"
        )

    input_record = record.get("input", {})
    if isinstance(input_record, dict):
        input_path = str(input_record.get("path", ""))
        posix_path = PurePosixPath(input_path)
        parts = input_path.split("/")
        if (
            not input_path
            or "\\" in input_path
            or ":" in input_path
            or posix_path.is_absolute()
            or posix_path.as_posix() != input_path
            or any(part in {"", ".", ".."} for part in parts)
        ):
            failures.append(
                f"{location}.input.path must be a normalized relative POSIX path"
            )
        cancelled = input_record.get("cancelled", {})
        survivor = input_record.get("survivor", {})
        source_count = input_record.get("source_sample_count")
        if isinstance(cancelled, dict) and isinstance(survivor, dict):
            if cancelled.get("sample_start") != 0 or survivor.get("sample_start") != 0:
                failures.append(f"{location} derived inputs must start at sample zero")
            if (
                not isinstance(source_count, int)
                or isinstance(source_count, bool)
                or cancelled.get("sample_end") != source_count // 2
                or survivor.get("sample_end") != source_count
            ):
                failures.append(f"{location} derived input ranges are inconsistent")
            for field in (
                "pcm_sha256",
                "mel_sha256",
                "observed_runtime_features_sha256",
            ):
                if cancelled.get(field) == survivor.get(field):
                    failures.append(
                        f"{location} derived inputs must have distinct {field} values"
                    )

        manifest_path = ROOT / "conformance" / "audio-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        fixtures = manifest.get("fixtures", []) if isinstance(manifest, dict) else []
        manifest_record = next(
            (
                fixture
                for fixture in fixtures
                if isinstance(fixture, dict)
                and fixture.get("id") == input_record.get("fixture_id")
            ),
            None,
        )
        if manifest_record is None:
            failures.append(f"{location} uses an unknown audio fixture")
        else:
            for evidence_field, manifest_field in (
                ("file_sha256", "sha256"),
                ("size_bytes", "size_bytes"),
                ("sample_rate_hz", "decoded_sample_rate_hz"),
                ("source_sample_count", "decoded_sample_count"),
            ):
                if input_record.get(evidence_field) != manifest_record.get(
                    manifest_field
                ):
                    failures.append(
                        f"{location}.input.{evidence_field} does not match "
                        "the audio manifest"
                    )

    model = record.get("model", {})
    if isinstance(model, dict):
        if model.get("loaded_state_before") != model.get("loaded_state_after"):
            failures.append(f"{location} loaded model state changed during the check")
        if model.get("execution_state_before") != model.get("execution_state_after"):
            failures.append(
                f"{location} model execution state changed during the check"
            )

    execution = record.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    threads = execution.get("threads", [])
    event_times: dict[str, dict[str, int]] = {}
    by_role: dict[str, dict[str, Any]] = {}
    survivor_steps = execution.get("survivor_steps")
    if isinstance(threads, list) and len(threads) == 2:
        by_role = {
            role: thread
            for thread in threads
            if isinstance(thread, dict)
            and isinstance((role := thread.get("role")), str)
        }
        if set(by_role) != {"cancelled", "survivor"}:
            failures.append(f"{location}.execution.threads must contain both roles")
        else:
            if len({item.get("python_thread_id") for item in by_role.values()}) != 2:
                failures.append(f"{location} Python caller thread IDs are not distinct")
            if len({item.get("native_thread_id") for item in by_role.values()}) != 2:
                failures.append(f"{location} native caller thread IDs are not distinct")
            expected = {
                "cancelled": [
                    "adapter:enter",
                    "start_run:begin",
                    "start_run:end",
                    "prefill:begin",
                    "prefill:end",
                    "step:1",
                    "cleanup:begin",
                    "cleanup:end",
                    "adapter:cancelled",
                ],
                "survivor": [
                    "adapter:enter",
                    "start_run:begin",
                    "start_run:end",
                    "prefill:begin",
                    "prefill:end",
                    *(
                        [f"step:{index}" for index in range(1, survivor_steps + 1)]
                        if isinstance(survivor_steps, int)
                        and not isinstance(survivor_steps, bool)
                        else []
                    ),
                    "finalize",
                    "cleanup:begin",
                    "cleanup:end",
                    "adapter:committed",
                ],
            }
            for role, thread in by_role.items():
                started = thread.get("started_ns")
                finished = thread.get("finished_ns")
                events = thread.get("events", [])
                if not (
                    isinstance(started, int)
                    and not isinstance(started, bool)
                    and isinstance(finished, int)
                    and not isinstance(finished, bool)
                    and started < finished
                ):
                    failures.append(f"{location} {role} caller interval is invalid")
                    continue
                if not isinstance(events, list) or not all(
                    isinstance(event, dict) for event in events
                ):
                    failures.append(f"{location} {role} caller events are invalid")
                    continue
                names = [event.get("name") for event in events]
                times = [event.get("at_ns") for event in events]
                if names != expected[role]:
                    failures.append(
                        f"{location} {role} caller event sequence is invalid"
                    )
                if not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in times
                ):
                    failures.append(
                        f"{location} {role} caller event timestamps are invalid"
                    )
                elif not (
                    all(started <= value <= finished for value in times)
                    and all(
                        earlier <= later
                        for earlier, later in zip(times, times[1:], strict=False)
                    )
                ):
                    failures.append(
                        f"{location} {role} caller event timestamps are unordered"
                    )
                elif all(isinstance(name, str) for name in names):
                    event_times[role] = dict(zip(names, times, strict=True))

    start_intervals = execution.get("start_run_intervals", [])
    if isinstance(start_intervals, list) and len(start_intervals) == 2:
        start_by_role = {
            role: interval
            for interval in start_intervals
            if isinstance(interval, dict)
            and isinstance((role := interval.get("role")), str)
        }
        if set(start_by_role) != {"cancelled", "survivor"}:
            failures.append(
                f"{location}.execution.start_run_intervals must contain both roles"
            )
        else:
            ordered: list[tuple[int, int]] = []
            for role, interval in start_by_role.items():
                started = interval.get("started_ns")
                finished = interval.get("finished_ns")
                owner = by_role.get(role)
                role_events = event_times.get(role, {})
                if not (
                    isinstance(started, int)
                    and not isinstance(started, bool)
                    and isinstance(finished, int)
                    and not isinstance(finished, bool)
                    and started < finished
                ):
                    failures.append(f"{location} {role} start-run interval is invalid")
                    continue
                ordered.append((started, finished))
                if isinstance(owner, dict) and (
                    interval.get("python_thread_id") != owner.get("python_thread_id")
                    or interval.get("native_thread_id") != owner.get("native_thread_id")
                    or not role_events.get("start_run:begin")
                    <= started
                    < finished
                    <= role_events.get("start_run:end")
                ):
                    failures.append(
                        f"{location} {role} start-run interval has the wrong owner"
                    )
            ordered.sort()
            if len(ordered) == 2 and ordered[0][1] > ordered[1][0]:
                failures.append(f"{location} start-run intervals overlap")

    calls = execution.get("controlled_forward_lifetimes", [])
    if isinstance(calls, list) and len(calls) == 2:
        call_by_role = {
            role: call
            for call in calls
            if isinstance(call, dict) and isinstance((role := call.get("role")), str)
        }
        if set(call_by_role) != {"cancelled", "survivor"}:
            failures.append(
                f"{location}.execution.controlled_forward_lifetimes must contain both roles"
            )
        else:
            started_calls: list[int] = []
            finished_calls: list[int] = []
            barriers: list[int] = []
            for role, call in call_by_role.items():
                started = call.get("forward_started_ns")
                barrier = call.get("inner_barrier_entered_ns")
                finished = call.get("forward_finished_ns")
                owner = by_role.get(role)
                role_events = event_times.get(role, {})
                if not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in (started, barrier, finished)
                ):
                    failures.append(f"{location} {role} forward interval is invalid")
                    continue
                if not started <= barrier < finished:
                    failures.append(f"{location} {role} forward interval is unordered")
                started_calls.append(started)
                barriers.append(barrier)
                finished_calls.append(finished)
                if isinstance(owner, dict) and (
                    call.get("python_thread_id") != owner.get("python_thread_id")
                    or call.get("native_thread_id") != owner.get("native_thread_id")
                    or not role_events.get("prefill:begin")
                    <= started
                    <= barrier
                    < finished
                    <= role_events.get("prefill:end")
                ):
                    failures.append(
                        f"{location} {role} forward call has the wrong owner or phase"
                    )
            if len(started_calls) == 2 and not (
                max(started_calls) < min(finished_calls)
                and max(barriers) < min(finished_calls)
            ):
                failures.append(
                    f"{location} recorded decoder forward-call lifetimes do not overlap"
                )

    controller = execution.get("controller", {})
    if isinstance(controller, dict) and set(event_times) == {"cancelled", "survivor"}:
        cancel_started = controller.get("cancel_started_ns")
        cancel_finished = controller.get("cancel_finished_ns")
        step_times = [
            event_times[role].get("step:1") for role in ("cancelled", "survivor")
        ]
        cleanup_started = event_times["cancelled"].get("cleanup:begin")
        if not (
            all(isinstance(value, int) for value in step_times)
            and isinstance(cancel_started, int)
            and isinstance(cancel_finished, int)
            and isinstance(cleanup_started, int)
            and max(step_times) <= cancel_started <= cancel_finished < cleanup_started
        ):
            failures.append(
                f"{location} cancellation is not between both first steps and cleanup"
            )

    resources = record.get("resources", {})
    snapshots: list[dict[str, Any]] = []
    if isinstance(resources, dict):
        per_transaction = _resource_components(resources.get("per_transaction"))
        capacity = _resource_components(resources.get("capacity"))
        raw_snapshots = resources.get("snapshots", [])
        if per_transaction is not None and capacity != tuple(
            component * 2 for component in per_transaction
        ):
            failures.append(f"{location} declared capacity is not two transactions")
        if isinstance(raw_snapshots, list) and all(
            isinstance(item, dict) for item in raw_snapshots
        ):
            snapshots = raw_snapshots
        expected_labels = ["initial", "both_admitted", "cancelled_released", "final"]
        if [item.get("label") for item in snapshots] != expected_labels:
            failures.append(f"{location} resource snapshot sequence is invalid")
        elif per_transaction is not None and capacity is not None:
            expected_values = [
                (0, 0, capacity, (0, 0, 0), "created", "created", 0, 0),
                (2, 2, (0, 0, 0), capacity, "running", "running", 0, 0),
                (
                    1,
                    1,
                    per_transaction,
                    per_transaction,
                    "cancelled",
                    "running",
                    0,
                    0,
                ),
                (0, 0, capacity, (0, 0, 0), "cancelled", "committed", 0, 1),
            ]
            for snapshot, expected in zip(snapshots, expected_values, strict=True):
                (
                    queue_depth,
                    lease_count,
                    available,
                    in_use,
                    cancelled_status,
                    survivor_status,
                    cancelled_version,
                    survivor_version,
                ) = expected
                cancelled_state = snapshot.get("cancelled", {})
                survivor_state = snapshot.get("survivor", {})
                if (
                    snapshot.get("queue_depth") != queue_depth
                    or snapshot.get("lease_count") != lease_count
                    or _resource_components(snapshot.get("capacity")) != capacity
                    or _resource_components(snapshot.get("available")) != available
                    or _resource_components(snapshot.get("in_use")) != in_use
                    or not isinstance(cancelled_state, dict)
                    or not isinstance(survivor_state, dict)
                    or cancelled_state.get("request_status") != cancelled_status
                    or survivor_state.get("request_status") != survivor_status
                    or cancelled_state.get("session_version") != cancelled_version
                    or survivor_state.get("session_version") != survivor_version
                    or cancelled_state.get("window_count") != 0
                    or survivor_state.get("window_count") != survivor_version
                ):
                    failures.append(
                        f"{location} {snapshot.get('label')} resource state is invalid"
                    )
            snapshot_times = [snapshot.get("at_ns") for snapshot in snapshots]
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in snapshot_times
            ) or not all(
                earlier <= later
                for earlier, later in zip(
                    snapshot_times, snapshot_times[1:], strict=False
                )
            ):
                failures.append(f"{location} resource snapshot times are unordered")
            elif set(event_times) == {"cancelled", "survivor"} and isinstance(
                controller, dict
            ):
                if not (
                    snapshot_times[0]
                    <= min(item.get("started_ns", 0) for item in by_role.values())
                    and max(
                        event_times["cancelled"]["step:1"],
                        event_times["survivor"]["step:1"],
                    )
                    <= snapshot_times[1]
                    <= controller.get("cancel_started_ns", 0)
                    and controller.get("cancel_finished_ns", 0) <= snapshot_times[2]
                    and event_times["cancelled"]["cleanup:end"] <= snapshot_times[2]
                    and max(item.get("finished_ns", 0) for item in by_role.values())
                    <= snapshot_times[3]
                ):
                    failures.append(
                        f"{location} resource snapshots do not match runtime events"
                    )

    elapsed = execution.get("elapsed_seconds", {})
    if isinstance(elapsed, dict) and all(
        _is_number(elapsed.get(field))
        for field in (
            "input_preparation",
            "baseline",
            "concurrent_adapter",
            "reuse_control",
            "total",
        )
    ):
        measured = sum(
            elapsed[field]
            for field in (
                "input_preparation",
                "baseline",
                "concurrent_adapter",
                "reuse_control",
            )
        )
        if elapsed["total"] < measured:
            failures.append(
                f"{location}.execution.elapsed_seconds.total is shorter than "
                "its measured phases"
            )

    results = record.get("results", {})
    if isinstance(results, dict):
        baseline = results.get("isolated_baseline", {})
        survivor = results.get("survivor", {})
        reuse = results.get("reuse_control", {})
        if not (
            isinstance(baseline, dict)
            and isinstance(survivor, dict)
            and isinstance(reuse, dict)
            and baseline.get("text") == survivor.get("text") == reuse.get("text")
        ):
            failures.append(f"{location} committed transcripts differ from baseline")

    assertions = record.get("assertions", {})
    if (
        not isinstance(assertions, dict)
        or not assertions
        or any(value is not True for value in assertions.values())
    ):
        failures.append(f"{location} contains a failed or missing assertion")

    manifest_path = ROOT / "patches" / "openai-whisper" / "SHA256SUMS"
    backend = record.get("backend", {})
    if isinstance(backend, dict) and manifest_path.is_file():
        observed_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if backend.get("patch_manifest_sha256") != observed_manifest:
            failures.append(f"{location} does not match the current patch manifest")
    return failures


def check_native_runtime_concurrency_evidence() -> list[str]:
    failures: list[str] = []
    schema_path = ROOT / "evidence" / "native-runtime-concurrency.schema.json"
    schema = _read_json(schema_path, failures)
    if not isinstance(schema, dict):
        return failures
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        failures.append(
            "evidence/native-runtime-concurrency.schema.json must use "
            "JSON Schema 2020-12"
        )
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import SchemaError
    except ImportError:
        failures.append(
            "JSON Schema validation is unavailable; install the 'validation' extra"
        )
        return failures
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        failures.append(
            "evidence/native-runtime-concurrency.schema.json is not a valid schema: "
            f"{error}"
        )
        return failures
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for path in sorted((ROOT / "evidence").glob("native*runtime-concurrency*.json")):
        if path.name == "native-runtime-concurrency.schema.json":
            continue
        record = _read_json(path, failures)
        if record is None:
            continue
        evidence_location = path.relative_to(ROOT).as_posix()
        schema_errors = list(validator.iter_errors(record))
        for error in schema_errors:
            field = ".".join(str(part) for part in error.absolute_path)
            suffix = f".{field}" if field else ""
            failures.append(f"{evidence_location}{suffix}: {error.message}")
        if not schema_errors:
            failures.extend(
                validate_native_runtime_concurrency_evidence(record, evidence_location)
            )
    return failures


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


def check_modal_cuda_schema() -> list[str]:
    """Require valid closed schemas for optional Modal evidence records."""

    failures: list[str] = []
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError:
        failures.append(
            "JSON Schema validation is unavailable; install the 'validation' extra"
        )
        return failures

    for name in (
        "modal-cuda-readiness.schema.json",
        "modal-native-cuda-qualification.schema.json",
        "modal-native-cuda-transaction.schema.json",
    ):
        path = ROOT / "evidence" / name
        schema = _read_json(path, failures)
        if not isinstance(schema, dict):
            continue
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            failures.append(f"evidence/{name} must use JSON Schema 2020-12")
        if schema.get("additionalProperties") is not False:
            failures.append(f"evidence/{name} must close the top-level object")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            failures.append(f"evidence/{name} is not a valid schema: {error}")

    for name in (
        "native-cuda-qualification-v1.json",
        "native-cuda-qualification-v2.json",
        "native-cuda-qualification-v3.json",
        "native-cuda-qualification-v4.json",
        "native-cuda-qualification-v5.json",
        "native-cuda-qualification-v6.json",
    ):
        manifest_path = ROOT / "experiments" / name
        manifest = _read_json(manifest_path, failures)
        if manifest is not None:
            failures.extend(
                validate_qualification_manifest(manifest, f"experiments/{name}")
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
                    "source_context",
                    "rights_notice",
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
            for field in ("source_context", "rights_notice"):
                if not isinstance(item.get(field), str) or not item.get(field):
                    failures.append(f"{location}.{field} must be a non-empty string")

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
        *check_modal_cuda_schema(),
        *check_conformance_cases(),
        *check_native_interleaving_evidence(),
        *check_native_threaded_evidence(),
        *check_native_runtime_concurrency_evidence(),
    ]
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"repository checks passed ({len(files)} text files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
