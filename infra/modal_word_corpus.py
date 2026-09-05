"""One opt-in T4 word-corpus diagnostic; importing allocates nothing."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "experiments/modal-word-corpus-v1.json"
PRODUCER_PATH = "infra/modal_word_corpus.py"
HELPER_PATHS = (
    "infra/modal_stream_boundary_diagnostic.py",
    "infra/modal_continuous_smoke.py",
)
ASSET_PATH = "artifacts/speech-corpus-v1"
REMOTE_ASSETS = Path("/opt/speech-corpus")
RUNTIME_ROOT = Path("/opt/whisper-runtime")
APP_NAME = "whisper-runtime-word-corpus-v1"
MANIFEST_ID = "modal-word-corpus-v1"
REMOTE_RESOURCES_ENV = "WHISPER_MODAL_ENABLE_WORD_CORPUS"
BASE_COMMIT = "9c2494234f08b24325d427ea422818b24f460c0c"
CLAIMS = dict.fromkeys(
    (
        "recognition_improvement",
        "performance_benchmark",
        "production_readiness",
        "public_commit_reproducibility",
        "generalization_beyond_this_input",
    ),
    False,
)
STREAM_CONFIG = {
    "preview_interval_ms": 2000,
    "max_window_ms": 30000,
    "max_buffer_ms": 40000,
    "holdback_ms": 2000,
    "timestamp_tolerance_ms": 200,
    "left_context_ms": 2000,
    "coalesce_previews": False,
    "word_alignment": True,
}
INPUT_EVIDENCE_PROFILE = {
    "profile_id": "word_agreement_stream/v1+input_evidence/v1",
    "stream_config": {**STREAM_CONFIG, "input_evidence": True},
}
SOURCE_UNITS_PROFILE = {
    "profile_id": "source_unit_stream/v1+input_evidence/v1",
    "stream_config": {
        **STREAM_CONFIG,
        "input_evidence": True,
        "source_units": True,
        "word_alignment": False,
        "left_context_ms": 0,
    },
}
QUIET_ENDPOINT_CONFIG = {
    "frame_ms": 20,
    "quiet_ms": 600,
    "min_unit_ms": 1000,
    "max_quiet_unit_ms": 10000,
    "quiet_peak": 32,
}
AUTOMATIC_ENDPOINTS_PROFILE = {
    "profile_id": "quiet_endpoint_stream/v1+input_evidence/v1",
    "stream_config": {
        **SOURCE_UNITS_PROFILE["stream_config"],
        "endpointing": QUIET_ENDPOINT_CONFIG,
    },
}
PACED_REPLAY = {
    "pacing_id": "wall-clock-pcm/v1",
    "config": {
        "chunk_ms": 20,
        "max_input_ms": 120000,
        "max_source_lag_ms": 250,
        "max_drain_ms": 15000,
        "idle_wait_ms": 2,
        "max_steps": 100000,
    },
}
SOURCE_UNIT_BOUNDARIES = {
    "case_id": "three-speakers-repeat-pauses",
    "authority": "caller-supplied-fixture-oracle",
    "acoustic_detection": False,
    "end_samples": [
        56080,
        88080,
        174240,
        206240,
        333280,
        365280,
        421360,
        453360,
        539520,
        571520,
        698560,
    ],
}


def _helper(name: str) -> Any:
    flags = (
        "WHISPER_MODAL_ENABLE_STREAM_BOUNDARY_DIAGNOSTIC",
        "WHISPER_MODAL_ENABLE_CONTINUOUS_SMOKE",
        "WHISPER_MODAL_ENABLE_REMOTE_RESOURCES",
    )
    old = {key: os.environ.get(key) for key in flags}
    try:
        os.environ.update(dict.fromkeys(flags, "0"))
        return importlib.import_module(name)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


b = _helper("infra.modal_stream_boundary_diagnostic")
c = b._common


def _integer(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("integer outside the registered bound")
    return value


def _name(value: object, *, filename: bool = False) -> str:
    pattern = (
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
        if filename
        else r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
    )
    if not isinstance(value, str) or not re.fullmatch(pattern, value) or ".." in value:
        raise ValueError("unsafe corpus identifier or filename")
    return value


def _equal(actual: object, expected: object, label: str) -> None:
    if c._canonical_sha256(actual) != c._canonical_sha256(expected):
        raise ValueError(f"unregistered {label}")


def read_registration(root: Path = ROOT) -> dict[str, Any]:
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("registration must be an object")
    for key, expected in (
        ("manifest_version", "1"),
        ("manifest_id", MANIFEST_ID),
        ("state", "diagnostic"),
        ("claim_boundary", CLAIMS),
        ("stream_config", STREAM_CONFIG),
        ("input_evidence_profile", INPUT_EVIDENCE_PROFILE),
        ("source_units_profile", SOURCE_UNITS_PROFILE),
        ("automatic_endpoints_profile", AUTOMATIC_ENDPOINTS_PROFILE),
        ("paced_replay", PACED_REPLAY),
        ("source_unit_boundaries", SOURCE_UNIT_BOUNDARIES),
        ("rng_seed", 7),
    ):
        _equal(manifest.get(key), expected, key)
    _equal(manifest["source_policy"]["image_base_commit"], BASE_COMMIT, "image base")
    model = manifest["model"]
    for key, expected in {
        "gpu": "T4",
        "cloud": "aws",
        "region": "us-west",
        "device": "cuda:0",
        "name": "tiny.en",
        "numeric_precision": "float32",
        "checkpoint_sha256": "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
        "model_state_sha256": "sha256:8041a80119a588f542472da35e97d0372fce1d9709ed9874475e9c03deac5de6",
    }.items():
        _equal(model.get(key), expected, key)
    _equal(
        manifest["decode_options"],
        {
            "language": "en",
            "task": "transcribe",
            "temperature": 0,
            "without_timestamps": False,
        },
        "decode options",
    )
    _equal(
        manifest["offline_control_options"],
        {
            "language": "en",
            "task": "transcribe",
            "temperature": 0,
            "fp16": False,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "initial_prompt": None,
            "carry_initial_prompt": False,
            "word_timestamps": False,
            "clip_timestamps": "0",
            "hallucination_silence_threshold": None,
            "without_timestamps": False,
            "verbose": None,
        },
        "offline options",
    )
    for key, expected in {
        "maximum_gpu_function_calls": 1,
        "gpu_seconds_per_call": 180,
        "maximum_gpu_seconds": 180,
        "automatic_retries": 0,
        "maximum_containers": 1,
        "minimum_containers": 0,
        "startup_timeout_seconds": 300,
        "scaledown_window_seconds": 2,
        "cpu_cores": 2,
        "memory_gib": 4,
    }.items():
        _equal(manifest["paid_budget"].get(key), expected, key)
    transport = manifest["result_transport"]
    for key, expected in {
        "encoding": b.WORKER_RESULT_ENCODING,
        "modal_sdk_version": b.MODAL_SDK_VERSION,
        "invocation_type": "sync",
        "modal_sync_serialized_limit_bytes": b.MODAL_SYNC_SERIALIZED_LIMIT_BYTES,
        "modal_async_serialized_limit_bytes": b.MODAL_ASYNC_SERIALIZED_LIMIT_BYTES,
        "modal_pickle_protocol": 4,
        "modal_pickle_bytes_overhead": b.MODAL_PICKLE4_BYTES_OVERHEAD,
        "inline_margin_bytes": b.INLINE_RESULT_MARGIN_BYTES,
        "maximum_inline_compressed_bytes": b.MAX_INLINE_COMPRESSED_BYTES,
        "maximum_decompressed_json_bytes": b.MAX_DECOMPRESSED_RESULT_BYTES,
        "oversize_behavior": "return-compact-transport-rejected-record",
    }.items():
        _equal(transport.get(key), expected, key)
    _equal(
        transport["preflight"],
        {
            "required_before_gpu": True,
            "gpu": None,
            "maximum_function_calls": 1,
            "timeout_seconds": 30,
            "payload_raw_bytes": 65536,
            "crosses_async_threshold": True,
        },
        "transport preflight",
    )
    fixtures, cases = manifest["fixtures"], manifest["cases"]
    if (
        not isinstance(fixtures, list)
        or not 1 <= len(fixtures) <= 6
        or not isinstance(cases, list)
        or not 1 <= len(cases) <= 6
    ):
        raise ValueError("require one to six fixtures and cases")
    for items in (fixtures, cases):
        if len({_name(item["id"]) for item in items}) != len(items):
            raise ValueError("duplicate corpus identifiers")
        for item in items:
            _integer(item["sample_count"], 1, 180 * 16000)
            if not isinstance(item["reference_text"], str) or not re.fullmatch(
                r"[0-9a-f]{64}", item["pcm_sha256"]
            ):
                raise ValueError("invalid reference or PCM fingerprint")
    if len({_name(item["filename"], filename=True) for item in fixtures}) != len(
        fixtures
    ):
        raise ValueError("duplicate fixture filenames")
    if sum(item["sample_count"] for item in cases) > 180 * 16000:
        raise ValueError("corpus exceeds 180 seconds of source audio")
    for case in cases:
        _equal(case["chunk_ms"], 1000, "chunk cadence")
    _source_unit_plan(manifest)
    return manifest


def source_snapshot(root: Path = ROOT) -> dict[str, object]:
    package_paths = list(root.joinpath("src/whisper_runtime").rglob("*.py"))
    if not package_paths:
        raise ValueError("source snapshot requires runtime files")
    paths = sorted(
        [
            *package_paths,
            *(root / item for item in (PRODUCER_PATH, *HELPER_PATHS, MANIFEST_PATH)),
        ],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("incomplete source snapshot")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": c._sha256_file(path),
        }
        for path in paths
    ]
    return {
        "algorithm": "sha256-file-manifest-v1",
        "digest": c._canonical_sha256(files),
        "files": files,
    }


def build_case(
    case: Mapping[str, Any], fixtures: list[dict[str, Any]], asset_dir: Path
) -> bytes:
    inventory = {_name(item["id"]): item for item in fixtures}
    parts = case["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 32:
        raise ValueError("case requires one to 32 explicit parts")
    chunks, references = [], []
    samples = 0
    for part in parts:
        if set(part) == {"fixture_id"}:
            fixture = inventory[_name(part["fixture_id"])]
            path = asset_dir / _name(fixture["filename"], filename=True)
            if path.resolve().parent != asset_dir.resolve():
                raise ValueError("fixture escapes asset directory")
            expected_size = _integer(fixture["sample_count"], 1, 180 * 16000) * 2
            if path.stat().st_size != expected_size:
                raise ValueError("fixture sample count differs from registration")
            content = path.read_bytes()
            if (
                len(content) != expected_size
                or hashlib.sha256(content).hexdigest() != fixture["pcm_sha256"]
            ):
                raise ValueError("fixture PCM differs from registration")
            references.append(fixture["reference_text"])
        elif set(part) == {"silence_ms"}:
            content = bytes(_integer(part["silence_ms"], 0, 180000) * 32)
        else:
            raise ValueError("unknown corpus part")
        samples += len(content) // 2
        if samples > 180 * 16000:
            raise ValueError("constructed case exceeds source limit")
        chunks.append(content)
    pcm = b"".join(chunks)
    reference = " ".join(" ".join(references).split())
    if (
        samples != case["sample_count"]
        or hashlib.sha256(pcm).hexdigest() != case["pcm_sha256"]
        or reference != " ".join(case["reference_text"].split())
    ):
        raise ValueError("composed PCM or reference differs from registration")
    return pcm


def _inputs(manifest: Mapping[str, Any], asset_dir: Path) -> list[bytes]:
    # Check every registered asset, including the one borrowed for warmup.
    for fixture in manifest["fixtures"]:
        build_case(
            {**fixture, "parts": [{"fixture_id": fixture["id"]}]},
            manifest["fixtures"],
            asset_dir,
        )
    return [
        build_case(case, manifest["fixtures"], asset_dir) for case in manifest["cases"]
    ]


def _source_progress(
    events: list[dict[str, Any]], traces: list[dict[str, Any]], total_samples: int
) -> dict[str, object]:
    """Join emitted commits to EOF intent; these are source spans, not latency."""
    eof_spans = {
        (trace["committed_before_sample"], trace["analysis_end_sample"])
        for trace in traces
        if trace.get("eof")
        and (
            trace.get("word_publication") is not None
            or trace.get("silence_publication") is not None
            or trace.get("source_unit") is not None
        )
    }
    commits = [event for event in events if event.get("kind") == "commit"]
    pre_eof = [
        event
        for event in commits
        if (event.get("start_sample"), event.get("end_sample")) not in eof_spans
    ]
    eof = [
        event
        for event in commits
        if (event.get("start_sample"), event.get("end_sample")) in eof_spans
    ]
    watermark = max((event["end_sample"] for event in pre_eof), default=0)
    return {
        "pre_eof_commit_count": len(pre_eof),
        "pre_eof_committed_samples": watermark,
        "source_ms_after_last_pre_eof_commit": (total_samples - watermark) / 16,
        "eof_commit_coverage_ms": sum(
            event["end_sample"] - event["start_sample"] for event in eof
        )
        / 16
        if eof
        else None,
        "wall_clock_latency": False,
    }


def _stream_profile(
    manifest: Mapping[str, Any],
    input_evidence: bool,
    source_units: bool = False,
    automatic_endpoints: bool = False,
    paced_replay: bool = False,
) -> dict[str, Any]:
    if any(
        type(flag) is not bool
        for flag in (input_evidence, source_units, automatic_endpoints, paced_replay)
    ):
        raise ValueError("profile flags must be booleans")
    if source_units and (automatic_endpoints or paced_replay):
        raise ValueError(
            "caller source units and automatic endpoints are mutually exclusive"
        )
    if automatic_endpoints or paced_replay:
        return manifest["automatic_endpoints_profile"]
    if source_units:
        return manifest["source_units_profile"]
    return (
        manifest["input_evidence_profile"]
        if input_evidence
        else {
            "profile_id": "word_agreement_stream/v1",
            "stream_config": manifest["stream_config"],
        }
    )


def _trace_checks(
    traces: list[dict[str, Any]],
    *,
    input_evidence: bool,
    word_alignment: bool = True,
    source_units: bool = False,
    registered_units: list[dict[str, Any]] | None = None,
    automatic_endpoints: bool = False,
) -> dict[str, bool]:
    normal = [
        trace
        for trace in traces
        if trace.get("audio_evidence") is None
        or (
            isinstance(trace["audio_evidence"], Mapping)
            and trace["audio_evidence"].get("state") == "speech_candidate"
        )
    ]
    checks = b._trace_profile_checks(normal, word_alignment=word_alignment)
    valid = True
    unit_valid = True
    expected_units = {
        (unit["start_sample"], unit["end_sample"]) for unit in registered_units or []
    }
    for trace in traces:
        unit = trace.get("source_unit")
        if not source_units:
            unit_valid &= unit is None
        elif unit is None:
            unit_valid &= trace.get("action") != "commit"
        else:
            unit_valid &= (
                isinstance(unit, Mapping)
                and unit.get("start_sample")
                == trace["analysis_start_sample"]
                == trace["committed_before_sample"]
                and unit.get("end_sample") == trace["analysis_end_sample"]
                and unit.get("end_sample") > unit.get("start_sample")
                and unit.get("origin")
                in (
                    {"quiet_run", "end_of_input"}
                    if automatic_endpoints
                    else {"caller", "end_of_input"}
                )
                and (unit["origin"] != "end_of_input" or trace.get("eof") is True)
                and (
                    registered_units is None
                    or (unit["start_sample"], unit["end_sample"]) in expected_units
                )
            )
        evidence = trace.get("audio_evidence")
        if evidence is None:
            valid &= not input_evidence and trace.get("silence_publication") is None
            continue
        if not isinstance(evidence, Mapping) or not isinstance(
            evidence.get("observation"), Mapping
        ):
            valid = False
            continue
        valid &= input_evidence
        result = trace.get("result")
        if not isinstance(result, Mapping) or not isinstance(
            result.get("metadata"), Mapping
        ):
            checks["trace_preserves_native_metadata"] = False
        analysis_span = (
            (
                result.get("analysis_span")
                or {"start_ms": result.get("start_ms"), "end_ms": result.get("end_ms")}
            )
            if isinstance(result, Mapping)
            else None
        )
        if source_units and unit is not None:
            unit_valid &= analysis_span == {
                "start_ms": trace["analysis_start_sample"] // 16,
                "end_ms": trace["analysis_end_sample"] // 16,
            }
        state = evidence.get("state")
        observation = evidence.get("observation", {})
        valid &= (
            state in {"speech_candidate", "uncertain", "non_speech"}
            and observation.get("sample_count")
            == trace["analysis_end_sample"] - trace["analysis_start_sample"]
            and isinstance(observation.get("pcm_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", observation["pcm_sha256"]) is not None
            and observation.get("digital_silence") is (state == "non_speech")
        )
        silence = trace.get("silence_publication")
        if state == "speech_candidate":
            valid &= silence is None
            continue
        valid &= (
            trace.get("word_alignment") is None
            and trace.get("word_publication") is None
            and trace.get("publication_span") is None
        )
        if silence is None:
            valid &= trace.get("action") in {"wait_for_input", "unresolved"} or (
                source_units
                and state == "non_speech"
                and trace.get("source_unit") is None
                and trace.get("reason") == "source_unit_open"
                and trace.get("action") == "preview"
            )
        else:
            valid &= (
                state == "non_speech"
                and isinstance(silence, Mapping)
                and silence.get("text") == ""
                and silence.get("native") == result
                and silence.get("observation") == observation
                and isinstance(result, Mapping)
                and silence.get("window_id") == result.get("window_id")
                and silence.get("analysis_span") == analysis_span
                and trace["analysis_start_sample"]
                <= trace["committed_before_sample"]
                < trace["analysis_end_sample"]
                and silence.get("start_ms") == trace["committed_before_sample"] // 16
                and silence.get("end_ms") == trace["analysis_end_sample"] // 16
                and analysis_span
                == {
                    "start_ms": trace["analysis_start_sample"] // 16,
                    "end_ms": trace["analysis_end_sample"] // 16,
                }
                and trace.get("action") == "commit"
            )
    checks["input_evidence_trace_contract"] = valid
    checks["source_unit_trace_contract"] = unit_valid
    return checks


def _unit_publication_checks(
    events: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> dict[str, bool]:
    """Bind each committed revision to its prepared native or typed result."""
    prepared = {}
    for trace in traces:
        if trace.get("action") == "commit" and trace.get("source_unit") is not None:
            span = (trace["committed_before_sample"], trace["analysis_end_sample"])
            publication = trace.get("silence_publication") or trace.get("result")
            prepared[span] = (
                publication.get("text") if isinstance(publication, Mapping) else None
            )
    revisions = {}
    published = set()
    valid = True
    for event in events:
        key = (event.get("segment_id"), event.get("revision"))
        if event.get("kind") in {"provisional", "replace"}:
            revisions[key] = event
        elif event.get("kind") == "commit":
            span = (event.get("start_sample"), event.get("end_sample"))
            revision = revisions.get(key, {})
            expected = prepared.get(span)
            valid &= (
                span not in published
                and isinstance(expected, str)
                and revision.get("text") == expected
                and (revision.get("start_sample"), revision.get("end_sample")) == span
            )
            published.add(span)
    # A failed prepared decision can have no event. Complete streams cannot.
    if any(event.get("kind") == "final" for event in events):
        valid &= published == set(prepared)
    return {"unit_publication_matches_trace": valid}


def _quiet_endpoint_reference(pcm: bytes) -> list[dict[str, int]]:
    """Independent batch calculation of the registered causal frame rule."""
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    config = QUIET_ENDPOINT_CONFIG
    frame = config["frame_ms"] * 16
    quiet_start = None
    quiet_peak = previous = 0
    nonquiet = False
    endpoints = []
    for start in range(0, len(samples) - frame + 1, frame):
        end = start + frame
        peak = max(abs(value) for value in samples[start:end])
        if peak > config["quiet_peak"]:
            quiet_start = None
            quiet_peak = 0
            nonquiet = True
            continue
        if quiet_start is None:
            quiet_start = start
        quiet_peak = max(quiet_peak, peak)
        if (
            end - quiet_start >= config["quiet_ms"] * 16
            and end - previous >= config["min_unit_ms"] * 16
            and (nonquiet or end - previous >= config["max_quiet_unit_ms"] * 16)
        ):
            endpoints.append(
                {
                    "end_sample": end,
                    "quiet_start_sample": quiet_start,
                    "peak": quiet_peak,
                }
            )
            previous = end
            quiet_start = None
            quiet_peak = 0
            nonquiet = False
    return endpoints


def _automatic_endpoint_checks(
    traces: list[dict[str, Any]], pcm: bytes
) -> dict[str, bool]:
    """Check observed bytes and endpoint proposals without fixture boundaries."""
    expected = {item["end_sample"]: item for item in _quiet_endpoint_reference(pcm)}
    pcm_valid = endpoint_valid = authority_valid = True
    observed_ends: list[int] = []
    for trace in traces:
        start, end = trace["analysis_start_sample"], trace["analysis_end_sample"]
        accepted = trace.get("accepted_through_sample")
        observation = trace.get("audio_evidence", {}).get("observation", {})
        content = pcm[start * 2 : end * 2]
        pcm_valid &= (
            0 <= start < end <= len(pcm) // 2
            and observation.get("sample_count") == end - start
            and observation.get("pcm_sha256") == hashlib.sha256(content).hexdigest()
            and observation.get("digital_silence") is (not any(content))
        )
        endpoint_valid &= type(accepted) is int and end <= accepted <= len(pcm) // 2
        unit = trace.get("source_unit")
        if unit is None:
            continue
        endpoint = unit.get("endpoint")
        if unit.get("origin") == "quiet_run":
            if not observed_ends or observed_ends[-1] != end:
                observed_ends.append(end)
            endpoint_valid &= (
                isinstance(endpoint, Mapping)
                and endpoint == expected.get(end)
                and endpoint.get("end_sample") == unit.get("end_sample")
            )
        else:
            endpoint_valid &= (
                unit.get("origin") == "end_of_input"
                and endpoint is None
                and trace.get("eof") is True
                and end == len(pcm) // 2
                and observed_ends == list(expected)
            )
        if trace.get("silence_publication") is not None:
            authority_valid &= observation.get("digital_silence") is True and not any(
                content
            )
    return {
        "observed_pcm_matches_input": pcm_valid,
        "automatic_endpoint_trace_contract": endpoint_valid
        and observed_ends == list(expected)[: len(observed_ends)],
        "quiet_proposal_has_no_silence_authority": authority_valid,
    }


def _source_unit_plan(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Frozen fixture boundaries are oracle input, never acoustic inference."""
    cases = [
        case
        for case in manifest["cases"]
        if case["id"] == SOURCE_UNIT_BOUNDARIES["case_id"]
    ]
    if len(cases) != 1:
        raise ValueError("source-unit diagnostic requires its registered mixed case")
    inventory = {fixture["id"]: fixture for fixture in manifest["fixtures"]}
    units, occurrences, start = [], {}, 0
    for index, part in enumerate(cases[0]["parts"]):
        unit = {"index": index, "start_sample": start, "origin": "caller"}
        if set(part) == {"fixture_id"} and part["fixture_id"] in inventory:
            fixture = inventory[part["fixture_id"]]
            occurrences[fixture["id"]] = occurrences.get(fixture["id"], 0) + 1
            count = fixture["sample_count"]
            unit.update(
                kind="fixture",
                fixture_id=fixture["id"],
                occurrence=occurrences[fixture["id"]],
                pcm_sha256=fixture["pcm_sha256"],
                reference_text=fixture["reference_text"],
            )
        elif set(part) == {"silence_ms"}:
            count = _integer(part["silence_ms"], 1, 180000) * 16
            unit.update(
                kind="digital_silence",
                reference_text="",
                pcm_sha256=hashlib.sha256(bytes(count * 2)).hexdigest(),
            )
        else:
            raise ValueError("unregistered source unit")
        start += count
        units.append({**unit, "end_sample": start})
    _equal(
        [unit["end_sample"] for unit in units],
        SOURCE_UNIT_BOUNDARIES["end_samples"],
        "source unit endpoints",
    )
    _equal(start, cases[0]["sample_count"], "source unit coverage")
    return units


def _drive_source_units(
    stream: Any, pcm: bytes, units: list[dict[str, Any]], *, chunk_bytes: int
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], int, int, dict[str, object] | None, int
]:
    """One controller owns all units, pauses, push sequence and event positions."""
    events, traces = [], []
    steps = chunks = accepted_samples = 0
    error_record = None
    if (
        chunk_bytes <= 0
        or chunk_bytes % 2
        or not units
        or units[-1]["end_sample"] * 2 != len(pcm)
    ):
        raise ValueError("source units must cover the exact complete PCM")
    cursor = 0
    for unit in units:
        end = unit["end_sample"] * 2
        if (
            unit["start_sample"] * 2 != cursor
            or end <= cursor
            or hashlib.sha256(pcm[cursor:end]).hexdigest() != unit["pcm_sha256"]
        ):
            raise ValueError(
                "source unit PCM or contiguous coverage differs from registration"
            )
        cursor = end
    try:
        with stream:
            offset = 0
            for unit in units:
                endpoint = unit["end_sample"] * 2
                while offset < endpoint:
                    end = min(offset + chunk_bytes, endpoint)
                    chunk = pcm[offset:end]
                    stream.push(chunks, chunk)
                    chunks += 1
                    accepted_samples += len(chunk) // 2
                    offset = end
                    if offset == endpoint:
                        stream.seal_unit(unit["end_sample"])
                        if offset == len(pcm):
                            stream.finish_input()
                    while stream.ready:
                        try:
                            batch = stream.step()
                        except Exception:
                            b._capture_trace(stream, traces)
                            raise
                        steps += 1
                        if steps > b.MAX_DRIVER_STEPS:
                            raise RuntimeError(
                                "the driver exceeded its fixed step bound"
                            )
                        b._capture_trace(stream, traces)
                        events.extend(b._plain(event) for event in batch)
    except Exception as error:
        error_record = b._safe_stream_error(error)
    return events, traces, steps, chunks, error_record, accepted_samples


def _source_unit_results(
    events: list[dict[str, Any]],
    units: list[dict[str, Any]],
    controls: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for unit in units:
        local = [
            event
            for event in events
            if event.get("start_sample") is not None
            and unit["start_sample"] <= event["start_sample"]
            and event.get("end_sample") is not None
            and event["end_sample"] <= unit["end_sample"]
        ]
        text = c._normalized_committed_text(local)
        committed = any(
            event.get("kind") == "commit" and event["end_sample"] == unit["end_sample"]
            for event in local
        )
        control = (
            controls[unit["fixture_id"]]["text"] if unit["kind"] == "fixture" else ""
        )
        results.append(
            {
                **unit,
                "completed": committed,
                "text": text,
                "comparison_scope": "complete_source_unit"
                if committed
                else "partial_source_unit",
                "offline_control_text": control,
                "against_human_reference": b._word_difference(
                    text, unit["reference_text"]
                ),
                "against_fixture_control": b._word_difference(text, control),
            }
        )
    return results


def _drive_paced(stream: Any, pcm: bytes) -> tuple[list, list, dict, dict | None]:
    """Keep native records unchanged and attach separate clock observations."""
    module = importlib.import_module("whisper_runtime.adapters.paced_replay")
    events, traces, event_times, trace_times = [], [], [], []

    def observe_event(event: object, elapsed_ns: int) -> None:
        events.append(b._plain(event))
        event_times.append(elapsed_ns)

    def observe_trace(trace: object, elapsed_ns: int) -> None:
        traces.append(b._plain(trace))
        trace_times.append(elapsed_ns)

    result = module.drive_paced(
        stream,
        pcm,
        config=module.PacedReplayConfig(**PACED_REPLAY["config"]),
        on_event=observe_event,
        on_trace=observe_trace,
    )
    error = b._safe_stream_error(result.error) if result.error is not None else None
    pacing = {
        **PACED_REPLAY,
        **{
            name: b._plain(getattr(result, name))
            for name in (
                "status",
                "elapsed_ns",
                "input_finished_ns",
                "offered_samples",
                "accepted_samples",
                "driver_steps",
                "max_source_lag_ns",
                "max_admission_lag_ns",
                "admissions",
                "undelivered_events",
                "undelivered_event_ns",
            )
        },
        "event_elapsed_ns": event_times,
        "trace_elapsed_ns": trace_times,
        "latency_scope": "same-worker-source-end-to-output; excludes endpoint quiet wait; not word or subtitle latency",
    }
    pacing["output_lags"] = _paced_output_lags(events, traces, pacing)
    pacing.update(_paced_milestones(events, pacing))
    return events, traces, pacing, error


def _paced_milestones(events: list, pacing: Mapping) -> dict:
    observed = list(zip(events, pacing["event_elapsed_ns"], strict=True))
    finished = pacing["input_finished_ns"]
    return {
        "first_nonempty_text_ns": next(
            (
                elapsed
                for event, elapsed in observed
                if event.get("kind") in {"provisional", "replace"}
                and isinstance(event.get("text"), str)
                and event["text"].strip()
            ),
            None,
        ),
        "first_commit_ns": next(
            (elapsed for event, elapsed in observed if event.get("kind") == "commit"),
            None,
        ),
        "drain_after_input_finished_ns": pacing["elapsed_ns"] - finished
        if finished is not None
        else None,
    }


def _paced_output_lags(events: list, traces: list, pacing: Mapping) -> dict:
    """Keep the full distributions; do not relabel source age as word latency."""
    groups: dict[str, list[int]] = {"provisional": [], "commit": [], "trace": []}
    for event, elapsed in zip(events, pacing["event_elapsed_ns"], strict=True):
        kind = event.get("kind")
        if kind in {"provisional", "replace", "commit"}:
            groups["commit" if kind == "commit" else "provisional"].append(
                elapsed - event["end_sample"] * 62500
            )
    groups["trace"] = [
        elapsed - trace["analysis_end_sample"] * 62500
        for trace, elapsed in zip(traces, pacing["trace_elapsed_ns"], strict=True)
    ]
    return {
        key: {
            "values_ns": values,
            "count": len(values),
            "min_ns": min(values) if values else None,
            "p50_ns": sorted(values)[(len(values) - 1) // 2] if values else None,
            "p95_ns": sorted(values)[(95 * len(values) + 99) // 100 - 1]
            if values
            else None,
            "max_ns": max(values) if values else None,
        }
        for key, values in groups.items()
    }


def _paced_checks(
    events: list, traces: list, pacing: Mapping, total_samples: int
) -> dict[str, bool]:
    checks = dict.fromkeys(
        (
            "paced_completed",
            "paced_admission_contract",
            "paced_source_not_early",
            "paced_source_lag_within_bound",
            "paced_admission_lag_within_bound",
            "paced_full_input_admitted",
            "paced_output_clock_contract",
            "paced_pre_eof_commit",
            "paced_output_delivered",
            "paced_terminal_coverage",
        ),
        False,
    )
    try:
        checks["paced_completed"] = pacing["status"] == "completed"
        _equal(pacing["pacing_id"], PACED_REPLAY["pacing_id"], "pacing identity")
        _equal(pacing["config"], PACED_REPLAY["config"], "pacing config")
        elapsed = _integer(pacing["elapsed_ns"], 0, 180_000_000_000)
        admissions = pacing["admissions"]
        if not isinstance(admissions, list):
            raise ValueError("admissions must be a list")
        head = last_accepted = 0
        no_early = valid = True
        max_lag = max_admission_lag = 0
        for index, item in enumerate(admissions):
            if not isinstance(item, Mapping):
                raise ValueError("admission must be an object")
            start = _integer(item["start_sample"], 0, total_samples)
            end = _integer(item["end_sample"], start + 1, total_samples)
            scheduled = _integer(item["scheduled_ns"], 0, elapsed)
            offered = _integer(item["offered_ns"], 0, elapsed)
            accepted = _integer(item["accepted_ns"], offered, elapsed)
            _integer(item["buffered_samples"], 0, STREAM_CONFIG["max_buffer_ms"] * 16)
            valid &= (
                type(item["sequence_number"]) is int
                and item["sequence_number"] == index
                and start == head
                and end
                == min(head + PACED_REPLAY["config"]["chunk_ms"] * 16, total_samples)
                and scheduled == end * 62500
                and offered >= last_accepted
            )
            no_early &= scheduled <= offered
            max_lag = max(max_lag, offered - scheduled)
            max_admission_lag = max(max_admission_lag, accepted - scheduled)
            head, last_accepted = end, accepted
        checks["paced_admission_contract"] = valid
        checks["paced_source_not_early"] = no_early
        observed_lag = _integer(pacing["max_source_lag_ns"], 0, elapsed)
        checks["paced_source_lag_within_bound"] = (
            observed_lag >= max_lag
            and (pacing["status"] != "completed" or observed_lag == max_lag)
            and observed_lag <= PACED_REPLAY["config"]["max_source_lag_ms"] * 1_000_000
        )
        offered_samples = _integer(pacing["offered_samples"], head, total_samples)
        observed_admission_lag = _integer(pacing["max_admission_lag_ns"], 0, elapsed)
        checks["paced_admission_lag_within_bound"] = (
            observed_admission_lag == max_admission_lag
            and observed_admission_lag
            <= PACED_REPLAY["config"]["max_source_lag_ms"] * 1_000_000
        )
        _integer(pacing["driver_steps"], 0, PACED_REPLAY["config"]["max_steps"])
        checks["paced_full_input_admitted"] = (
            type(pacing["accepted_samples"]) is int
            and pacing["accepted_samples"] == head == offered_samples == total_samples
        )
        clock_valid = True
        for records, key, end_key in (
            (events, "event_elapsed_ns", "end_sample"),
            (traces, "trace_elapsed_ns", "analysis_end_sample"),
        ):
            times = pacing[key]
            if not isinstance(times, list) or len(times) != len(records):
                raise ValueError("clock observations must match records")
            previous = 0
            for record, observed in zip(records, times, strict=True):
                if not isinstance(record, Mapping):
                    raise ValueError("output must be an object")
                observed = _integer(observed, previous, elapsed)
                previous = observed
                end = record.get(end_key)
                if end is not None:
                    end = _integer(end, 0, total_samples)
                    clock_valid &= observed >= end * 62500
        _equal(
            pacing["output_lags"],
            _paced_output_lags(events, traces, pacing),
            "output lags",
        )
        for key, value in _paced_milestones(events, pacing).items():
            _equal(pacing[key], value, key)
        checks["paced_output_clock_contract"] = clock_valid
        checks["paced_output_delivered"] = (
            pacing["undelivered_events"] == []
            and pacing["undelivered_event_ns"] is None
        )
        commit_head = 0
        coverage = True
        finals = []
        for index, event in enumerate(events):
            coverage &= _integer(event["sequence_number"], 1, len(events)) == index + 1
            if event.get("kind") == "final":
                finals.append(index)
            elif event.get("kind") == "commit":
                start = _integer(event["start_sample"], 0, total_samples)
                end = _integer(event["end_sample"], start + 1, total_samples)
                watermark = _integer(
                    event["committed_through_sample"], 0, total_samples
                )
                watermark_ms = _integer(
                    event["committed_through_ms"], 0, total_samples // 16
                )
                coverage &= (
                    start == commit_head
                    and end == watermark
                    and watermark_ms == end // 16
                )
                commit_head = end
        checks["paced_terminal_coverage"] = (
            coverage and commit_head == total_samples and finals == [len(events) - 1]
        )
        finished = pacing["input_finished_ns"]
        if finished is not None:
            finished = _integer(finished, last_accepted, elapsed)
            checks["paced_pre_eof_commit"] = any(
                event.get("kind") == "commit" and observed < finished
                for event, observed in zip(
                    events, pacing["event_elapsed_ns"], strict=True
                )
            )
    except (KeyError, TypeError, ValueError):
        return dict.fromkeys(checks, False)
    return checks


def _run_worker(
    expected_snapshot: Mapping[str, Any],
    *,
    registration_sha256: str,
    modal_module: Any,
    input_evidence: bool = False,
    source_units: bool = False,
    automatic_endpoints: bool = False,
    paced_replay: bool = False,
) -> dict[str, Any]:
    manifest = read_registration(RUNTIME_ROOT)
    profile = _stream_profile(
        manifest, input_evidence, source_units, automatic_endpoints, paced_replay
    )
    automatic_endpoints = automatic_endpoints or paced_replay
    input_evidence = input_evidence or source_units or automatic_endpoints
    unit_profile = source_units or automatic_endpoints
    _equal(source_snapshot(RUNTIME_ROOT), expected_snapshot, "remote source snapshot")
    _equal(
        c._sha256_file(RUNTIME_ROOT / MANIFEST_PATH),
        registration_sha256,
        "remote registration",
    )
    _equal(
        c._command_output(RUNTIME_ROOT, "rev-parse", "HEAD"),
        BASE_COMMIT,
        "image checkout",
    )
    pcms = _inputs(manifest, REMOTE_ASSETS)
    call_id = str(modal_module.current_function_call_id())
    if not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise RuntimeError("invalid function-call identity")
    print(f"MODAL_FUNCTION_CALL_ID={call_id}", flush=True)
    torch, np, whisper = (
        importlib.import_module(name) for name in ("torch", "numpy", "whisper")
    )
    runtime = importlib.import_module("whisper_runtime")
    adapters = importlib.import_module("whisper_runtime.adapters")
    q = _helper("infra.modal_native_cuda_qualification")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.current_device() != 0
        or torch.cuda.get_device_name(0) != "Tesla T4"
        or tuple(torch.cuda.get_device_capability(0)) != (7, 5)
    ):
        raise RuntimeError("one registered T4 is required")
    config = manifest["model"]
    _equal(
        c._sha256_file(b.MODEL_CHECKPOINT_PATH),
        config["checkpoint_sha256"],
        "checkpoint",
    )
    model = whisper.load_model(
        "tiny.en", device="cuda:0", download_root=b.MODEL_CACHE_MOUNT
    ).eval()
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if (
        model.training
        or not tensors
        or any(
            str(t.device) != "cuda:0"
            or (t.is_floating_point() and t.dtype != torch.float32)
            for t in tensors
        )
    ):
        raise RuntimeError("model device/precision differs from FP32 registration")
    actual_fingerprint = q._model_fingerprint(model)
    _equal(actual_fingerprint, config["model_state_sha256"], "loaded model state")
    capacity = runtime.ResourceVector(
        memory_bytes=2147483648, compute_units=1, stream_slots=1
    )
    budget = runtime.Budget(capacity)
    identity = runtime.ModelSnapshot(
        model_id="tiny.en",
        revision=c._command_output(b.BACKEND_ROOT, "rev-parse", "HEAD"),
        backend="pytorch-cuda-word-corpus",
        fingerprint=actual_fingerprint,
    )
    worker = runtime.Worker(
        MANIFEST_ID, identity, budget, queue_capacity=1, transaction_ttl_seconds=180
    )

    def identity_probe(observed: object) -> object:
        if observed is not model or str(getattr(observed, "device", "")) != "cuda:0":
            raise RuntimeError("model binding changed")
        return identity

    adapter = adapters.NativeWhisperAdapter(
        worker,
        model,
        identity_probe,
        adapters.NativeExecutionProfile(
            "tiny.en/cuda-word-corpus-v1", capacity, device="cuda:0"
        ),
    )
    options = adapters.NativeDecodeOptions(**manifest["decode_options"])

    def mel_builder(pcm: bytes) -> object:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
        ).contiguous()

    seed = manifest["rng_seed"]
    warm_pcm = (REMOTE_ASSETS / manifest["fixtures"][0]["filename"]).read_bytes()[
        : 10 * 16000 * 2
    ]
    session = runtime.Session(f"{MANIFEST_ID}:warmup")
    started = time.perf_counter_ns()
    with adapter.start_window(
        session=session,
        request=runtime.RequestState(
            "warmup", session.session_id, identity, rng_seed=seed
        ),
        window_id="warmup",
        mel=mel_builder(warm_pcm),
        start_ms=0,
        end_ms=len(warm_pcm) // 32,
        options=options,
    ) as run:
        for _ in range(b.MAX_DRIVER_STEPS):
            if run.complete:
                break
            run.step()
        alignment = run.prepare_word_alignment()
    torch.cuda.synchronize(0)
    warmup = {
        "wall_ns": time.perf_counter_ns() - started,
        "sample_count": len(warm_pcm) // 2,
        "word_count": len(alignment.words),
        "included_in_case_timing": False,
    }
    if budget.available != capacity or worker.queue_depth:
        raise RuntimeError("warmup retained capacity")
    cases, stopped = [], None
    units = _source_unit_plan(manifest) if unit_profile else []
    fixture_controls = {}
    for case, pcm in zip(manifest["cases"], pcms):
        if unit_profile and case["id"] != SOURCE_UNIT_BOUNDARIES["case_id"]:
            continue
        try:
            if unit_profile:
                for fixture in manifest["fixtures"]:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    fixture_audio = (
                        np.frombuffer(
                            (REMOTE_ASSETS / fixture["filename"]).read_bytes(),
                            dtype="<i2",
                        ).astype(np.float32)
                        / 32768.0
                    )
                    fixture_controls[fixture["id"]] = b._offline_result(
                        model.transcribe(
                            fixture_audio, **manifest["offline_control_options"]
                        )
                    )
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            offline = b._offline_result(
                model.transcribe(audio, **manifest["offline_control_options"])
            )
            offline["against_human_reference"] = b._word_difference(
                offline["text"], case["reference_text"]
            )
            torch.cuda.synchronize(0)
            stream_config = dict(profile["stream_config"])
            if automatic_endpoints:
                endpoint_module = importlib.import_module(
                    "whisper_runtime.adapters.audio_endpoints"
                )
                stream_config["endpointing"] = endpoint_module.QuietEndpointConfig(
                    **stream_config["endpointing"]
                )
            stream = adapters.ContinuousTranscriptStream(
                adapter,
                stream_id=f"{MANIFEST_ID}:{case['id']}",
                mel_builder=mel_builder,
                options=options,
                rng_seed=seed,
                config=adapters.ContinuousStreamConfig(**stream_config),
            )
            torch.cuda.reset_peak_memory_stats(0)
            allocated, reserved = (
                int(torch.cuda.memory_allocated(0)),
                int(torch.cuda.memory_reserved(0)),
            )
            started = time.perf_counter_ns()
            pacing = None
            if paced_replay:
                events, traces, pacing, error = _drive_paced(stream, pcm)
                steps = pacing["driver_steps"]
                accepted_chunks = len(pacing["admissions"])
                accepted_samples = pacing["accepted_samples"]
            elif source_units:
                events, traces, steps, accepted_chunks, error, accepted_samples = (
                    _drive_source_units(
                        stream, pcm, units, chunk_bytes=case["chunk_ms"] * 32
                    )
                )
            else:
                events, traces, steps, accepted_chunks, error = b._drive_stream(
                    stream, pcm, chunk_bytes=case["chunk_ms"] * 32
                )
                accepted_samples = min(
                    accepted_chunks * case["chunk_ms"] * 16, len(pcm) // 2
                )
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter_ns() - started
            checks = b._event_checks(
                events,
                traces,
                accepted_samples=accepted_samples,
                total_samples=len(pcm) // 2,
                state=stream.state,
                metrics=stream.metrics,
                budget=budget,
                worker=worker,
                capacity=capacity,
                retained_from_sample=stream.retained_from_sample,
                max_buffer_samples=STREAM_CONFIG["max_buffer_ms"] * 16,
                error_record=error,
            )
            checks.update(
                _trace_checks(
                    traces,
                    input_evidence=input_evidence,
                    word_alignment=profile["stream_config"]["word_alignment"],
                    source_units=unit_profile,
                    registered_units=units if source_units else None,
                    automatic_endpoints=automatic_endpoints,
                )
            )
            checks["profile_id_matches_config"] = (
                stream.profile_id == profile["profile_id"]
            )
            if unit_profile:
                checks.update(_unit_publication_checks(events, traces))
            if automatic_endpoints:
                checks.update(_automatic_endpoint_checks(traces, pcm))
            if pacing is not None:
                checks.update(_paced_checks(events, traces, pacing, len(pcm) // 2))
            policy_resolution = (
                not paced_replay
                and error is not None
                and error["category"] == "policy_resolution"
            )
            lifecycle = all(
                value
                for key, value in checks.items()
                if key
                not in {
                    "completion_required",
                    "final_event_once",
                    "full_input_committed_at_eof",
                }
            )
            complete = (
                checks["final_event_once"] and checks["full_input_committed_at_eof"]
            )
            unit_results = (
                _source_unit_results(events, units, fixture_controls)
                if source_units
                else []
            )
            if source_units:
                expected_spans = {
                    (unit["start_sample"], unit["end_sample"]) for unit in units
                }
                observed_spans = [
                    (event["start_sample"], event["end_sample"])
                    for event in events
                    if event.get("kind") == "commit"
                ]
                checks["source_unit_ownership"] = len(observed_spans) == len(
                    set(observed_spans)
                ) and all(span in expected_spans for span in observed_spans)
                checks["source_unit_pause_text_empty"] = all(
                    unit["text"] == ""
                    for unit in unit_results
                    if unit["kind"] == "digital_silence"
                )
                checks["source_unit_completion_accounted"] = not complete or all(
                    unit["completed"] for unit in unit_results
                )
                lifecycle &= all(
                    checks[key]
                    for key in (
                        "source_unit_ownership",
                        "source_unit_pause_text_empty",
                        "source_unit_completion_accounted",
                    )
                )
            passed = lifecycle and (policy_resolution or (error is None and complete))
            text = c._normalized_committed_text(events)
            cases.append(
                {
                    "case_id": case["id"],
                    "input": case,
                    "profile_id": stream.profile_id,
                    "status": "policy_resolution"
                    if passed and policy_resolution
                    else "completed"
                    if passed
                    else "lifecycle_failure",
                    "error": error,
                    "checks": checks,
                    "events": events,
                    "decision_traces": traces,
                    "source_progress": _source_progress(events, traces, len(pcm) // 2),
                    "state": b._plain(stream.state),
                    "metrics": b._plain(stream.metrics),
                    "driver_steps": steps,
                    "accepted_chunks": accepted_chunks,
                    "accepted_samples": accepted_samples,
                    **({"pacing": pacing} if pacing is not None else {}),
                    "offline_control": offline,
                    **(
                        {
                            "source_unit_boundaries": manifest[
                                "source_unit_boundaries"
                            ],
                            "source_units": unit_results,
                            "fixture_controls": fixture_controls,
                            "against_concatenated_fixture_controls": b._word_difference(
                                text,
                                " ".join(
                                    unit["offline_control_text"]
                                    for unit in unit_results
                                    if unit["kind"] == "fixture"
                                ),
                            ),
                        }
                        if source_units
                        else {}
                    ),
                    **(
                        {
                            "endpoint_policy": {
                                "kind": "online-quiet-run",
                                "config": QUIET_ENDPOINT_CONFIG,
                                "oracle_boundaries_used": False,
                                "silence_publication_authority": False,
                            },
                            "fixture_boundaries_for_evaluation": units,
                            "fixture_controls": fixture_controls,
                            "against_concatenated_fixture_controls": b._word_difference(
                                text,
                                " ".join(
                                    fixture_controls[unit["fixture_id"]]["text"]
                                    for unit in units
                                    if unit["kind"] == "fixture"
                                ),
                            ),
                        }
                        if automatic_endpoints
                        else {}
                    ),
                    "recognition": {
                        "comparison_scope": "complete_committed_transcript"
                        if complete
                        else "partial_committed_transcript",
                        "text": text,
                        "against_human_reference": b._word_difference(
                            text, case["reference_text"]
                        ),
                        "against_offline_control": b._word_difference(
                            text, offline["text"]
                        ),
                    },
                    "timing": {"wall_ns": elapsed, "benchmark": False},
                    "cuda_memory": {
                        "allocated_before_bytes": allocated,
                        "reserved_before_bytes": reserved,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    },
                }
            )
            if not passed:
                stopped = case["id"]
                break
        except Exception as error:
            cases.append(
                {
                    "case_id": case["id"],
                    "status": "infrastructure_failure",
                    "error": b._safe_stream_error(error),
                }
            )
            stopped = case["id"]
            break
    final_fingerprint = None
    fingerprint_error = None
    if budget.lease_count == 0 and worker.queue_depth == 0:
        try:
            final_fingerprint = q._model_fingerprint(model)
        except Exception as error:
            fingerprint_error = b._safe_stream_error(error)
    unchanged = final_fingerprint == actual_fingerprint
    status = (
        "stopped"
        if stopped or not unchanged
        else "unresolved"
        if any(item["status"] == "policy_resolution" for item in cases)
        else "completed"
    )
    return {
        "schema_version": "1-diagnostic",
        "recorded_at": c._utc_now(),
        "status": status,
        "claim_boundary": manifest["claim_boundary"],
        "replay_pacing": PACED_REPLAY["pacing_id"]
        if paced_replay
        else "unpaced-source-time",
        "paced_replay": paced_replay,
        "input_evidence": input_evidence,
        "source_units": source_units,
        "automatic_endpoints": automatic_endpoints,
        **profile,
        "source": {
            "image_base_commit": BASE_COMMIT,
            "snapshot": expected_snapshot,
            "registration_sha256": registration_sha256,
            "public_commit_reproducibility": False,
        },
        "worker": {
            "function_call_id": call_id,
            "function_call_id_sha256": c._sha256_text(call_id),
            "python": sys.version.split()[0],
            "modal": str(modal_module.__version__),
            "torch": str(torch.__version__),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "model": {
            **config,
            "observed_state_sha256": actual_fingerprint,
            "final_state_sha256": final_fingerprint,
            "unchanged": unchanged,
            "final_verification_error": fingerprint_error,
            "backend_revision": identity.revision,
        },
        "warmup": warmup,
        "cases": cases,
        "stopped_after_case": stopped,
    }


def _paths(root: Path, replay_id: str = "") -> tuple[Path, Path, Path, Path]:
    directory = root / "artifacts/modal"
    if replay_id != "":
        name = _name(replay_id)
        if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", name, re.IGNORECASE):
            raise ValueError("unsafe replay identifier")
        namespace = directory / name
        if namespace.resolve() != directory.resolve() / name:
            raise ValueError("replay namespace cannot redirect outside its path")
        directory = namespace
    stem = directory / "word-corpus-v1-attempt-1"
    return (
        stem.with_suffix(".json"),
        stem.with_suffix(".attempt.jsonl"),
        stem.with_suffix(".result.zlib"),
        directory / "word-corpus-v1-transport-preflight.attempt.jsonl",
    )


def _receipt(
    path: Path,
    event: str,
    sequence: int,
    *,
    replay_id: str = "",
    input_evidence: bool | None = None,
    source_units: bool | None = None,
    automatic_endpoints: bool | None = None,
    paced_replay: bool | None = None,
    **fields: object,
) -> None:
    c._append_receipt(
        path,
        {
            "receipt_version": "1",
            "manifest_id": MANIFEST_ID,
            "event": event,
            "sequence": sequence,
            "at": c._utc_now(),
            **({"replay_id": replay_id} if replay_id else {}),
            **(
                {"input_evidence": input_evidence} if input_evidence is not None else {}
            ),
            **({"source_units": source_units} if source_units is not None else {}),
            **(
                {"automatic_endpoints": automatic_endpoints}
                if automatic_endpoints is not None
                else {}
            ),
            **({"paced_replay": paced_replay} if paced_replay is not None else {}),
            **fields,
        },
        create=sequence == 0,
    )


def _execute_transport_probe(
    *, root: Path = ROOT, replay_id: str = "", remote_function: object
) -> Path:
    receipt = _paths(root, replay_id)[3]
    read_registration(root)
    if receipt.exists():
        raise FileExistsError("the one transport attempt already exists")
    payload = b._transport_probe_payload()
    _receipt(receipt, "attempt-started", 0, replay_id=replay_id)
    try:
        observed = getattr(remote_function, "remote")(payload)
        if type(observed) is not bytes or observed != payload:
            raise ValueError("transport changed the echo payload")
    except BaseException as error:
        _receipt(
            receipt,
            "attempt-failed",
            1,
            replay_id=replay_id,
            error_type=type(error).__name__,
        )
        raise
    _receipt(
        receipt,
        "transport-passed",
        1,
        replay_id=replay_id,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=len(payload),
    )
    return receipt


def _execute_local_attempt(
    *,
    root: Path = ROOT,
    attempt: int = 1,
    replay_id: str = "",
    input_evidence: bool = False,
    source_units: bool = False,
    automatic_endpoints: bool = False,
    paced_replay: bool = False,
    confirm_paid_gpu: bool = False,
    remote_function: object,
) -> Path:
    c._require_paid_confirmation(confirm_paid_gpu)
    _integer(attempt, 1, 1)
    manifest = read_registration(root)
    profile = _stream_profile(
        manifest, input_evidence, source_units, automatic_endpoints, paced_replay
    )
    automatic_endpoints = automatic_endpoints or paced_replay
    input_evidence = input_evidence or source_units or automatic_endpoints
    if input_evidence and not replay_id:
        raise ValueError("diagnostic variants require a named replay namespace")
    output, receipt, raw, probe = _paths(root, replay_id)
    if any(path.exists() for path in (output, receipt, raw)):
        raise FileExistsError("the one GPU attempt already exists")
    _inputs(manifest, root / ASSET_PATH)
    probe_events = [
        json.loads(line) for line in probe.read_text(encoding="utf-8").splitlines()
    ]
    if (
        len(probe_events) != 2
        or probe_events[-1].get("event") != "transport-passed"
        or probe_events[-1].get("manifest_id") != MANIFEST_ID
        or any(event.get("replay_id", "") != replay_id for event in probe_events)
        or probe_events[-1].get("payload_sha256")
        != hashlib.sha256(b._transport_probe_payload()).hexdigest()
    ):
        raise ValueError("a successful registered transport preflight is required")
    snapshot, registration_hash = (
        source_snapshot(root),
        c._sha256_file(root / MANIFEST_PATH),
    )
    _receipt(
        receipt,
        "attempt-started",
        0,
        replay_id=replay_id,
        input_evidence=input_evidence if replay_id else None,
        source_units=source_units if replay_id else None,
        automatic_endpoints=automatic_endpoints if replay_id else None,
        paced_replay=paced_replay if replay_id else None,
        **profile,
        source_snapshot_sha256=snapshot["digest"],
        registration_sha256=registration_hash,
    )
    sequence = 1
    try:
        _receipt(
            receipt,
            "synchronous-call-started",
            sequence,
            replay_id=replay_id,
            input_evidence=input_evidence if replay_id else None,
            source_units=source_units if replay_id else None,
            automatic_endpoints=automatic_endpoints if replay_id else None,
            paced_replay=paced_replay if replay_id else None,
        )
        sequence += 1
        args = (
            (snapshot, registration_hash, True, False, True, True)
            if paced_replay
            else (snapshot, registration_hash, True, False, True)
            if automatic_endpoints
            else (snapshot, registration_hash, True, True)
            if source_units
            else (snapshot, registration_hash, True)
            if input_evidence
            else (snapshot, registration_hash)
        )
        payload = getattr(remote_function, "remote")(*args)
        b._write_bytes_exclusive(raw, payload)
        _receipt(
            receipt,
            "compressed-result-written",
            sequence,
            replay_id=replay_id,
            input_evidence=input_evidence if replay_id else None,
            source_units=source_units if replay_id else None,
            automatic_endpoints=automatic_endpoints if replay_id else None,
            paced_replay=paced_replay if replay_id else None,
            sha256=c._sha256_file(raw),
            size_bytes=raw.stat().st_size,
        )
        sequence += 1
        record = b._decode_worker_record(
            payload,
            expected_snapshot=snapshot,
            registration_sha256=registration_hash,
            manifest=manifest,
        )
        _equal(record.get("input_evidence"), input_evidence, "worker input evidence")
        _equal(record.get("source_units"), source_units, "worker source units")
        _equal(
            record.get("automatic_endpoints", False),
            automatic_endpoints,
            "worker automatic endpoints",
        )
        _equal(record.get("paced_replay", False), paced_replay, "worker paced replay")
        if paced_replay:
            _equal(
                record.get("replay_pacing"), PACED_REPLAY["pacing_id"], "worker pacing"
            )
            cases = record.get("cases")
            if not isinstance(cases, list) or len(cases) != 1:
                raise ValueError("paced replay requires one case result")
            case = cases[0]
            _equal(case.get("case_id"), SOURCE_UNIT_BOUNDARIES["case_id"], "paced case")
            if record.get("status") == "unresolved" or (
                record.get("status") == "completed"
                and case.get("status") != "completed"
            ):
                raise ValueError("paced result status does not match its case")
            if case.get("status") != "infrastructure_failure":
                paced_checks = _paced_checks(
                    case["events"],
                    case["decision_traces"],
                    case["pacing"],
                    SOURCE_UNIT_BOUNDARIES["end_samples"][-1],
                )
                for key, expected in paced_checks.items():
                    _equal(case["checks"].get(key), expected, key)
                _equal(
                    case["accepted_samples"],
                    case["pacing"]["accepted_samples"],
                    "paced samples",
                )
                _equal(
                    case["accepted_chunks"],
                    len(case["pacing"]["admissions"]),
                    "paced chunks",
                )
                _equal(
                    case["driver_steps"], case["pacing"]["driver_steps"], "paced steps"
                )
                if case.get("status") == "policy_resolution" or (
                    case.get("status") == "completed" and not all(paced_checks.values())
                ):
                    raise ValueError("failed paced replay cannot pass")
        for key, expected in profile.items():
            _equal(record.get(key), expected, f"worker {key}")
        c._write_json_exclusive(output, record)
    except BaseException as error:
        _receipt(
            receipt,
            "attempt-failed",
            sequence,
            replay_id=replay_id,
            input_evidence=input_evidence if replay_id else None,
            source_units=source_units if replay_id else None,
            automatic_endpoints=automatic_endpoints if replay_id else None,
            paced_replay=paced_replay if replay_id else None,
            error_type=type(error).__name__,
            error_message_sha256=c._sha256_text(str(error)),
        )
        raise
    _receipt(
        receipt,
        "record-written",
        sequence,
        replay_id=replay_id,
        input_evidence=input_evidence if replay_id else None,
        source_units=source_units if replay_id else None,
        automatic_endpoints=automatic_endpoints if replay_id else None,
        paced_replay=paced_replay if replay_id else None,
        record_sha256=c._sha256_file(output),
        status=record["status"],
        function_call_id=record["worker"]["function_call_id"],
    )
    return output


def _modal_main(
    attempt: int = 1,
    confirm_paid_gpu: bool = False,
    transport_preflight_only: bool = False,
    replay_id: str = "",
    input_evidence: bool = False,
    source_units: bool = False,
    automatic_endpoints: bool = False,
    paced_replay: bool = False,
) -> None:
    if run_word_corpus is None or run_transport_probe is None:
        raise RuntimeError(f"set {REMOTE_RESOURCES_ENV}=1 before modal run")
    if transport_preflight_only:
        print(
            _execute_transport_probe(
                replay_id=replay_id, remote_function=run_transport_probe
            )
        )
    else:
        print(
            _execute_local_attempt(
                attempt=attempt,
                replay_id=replay_id,
                input_evidence=input_evidence,
                source_units=source_units,
                automatic_endpoints=automatic_endpoints,
                paced_replay=paced_replay,
                confirm_paid_gpu=confirm_paid_gpu,
                remote_function=run_word_corpus,
            )
        )


def _define_modal_resources() -> tuple[Any, Any, Any, Any]:
    manifest = read_registration()
    _inputs(manifest, ROOT / ASSET_PATH)
    modal = importlib.import_module("modal")
    _equal(str(modal.__version__), "1.5.5", "Modal SDK")
    q = _helper("infra.modal_native_cuda_qualification")
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("ca-certificates", "ffmpeg", "git")
        .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
        .uv_pip_install(*q.DIRECT_IMAGE_PACKAGES)
        .run_commands(q._build_command(BASE_COMMIT))
        .add_local_dir(
            ROOT / "src/whisper_runtime",
            "/opt/whisper-runtime/src/whisper_runtime",
            copy=True,
        )
    )
    for relative in (PRODUCER_PATH, *HELPER_PATHS, MANIFEST_PATH):
        image = image.add_local_file(
            ROOT / relative, f"/opt/whisper-runtime/{relative}", copy=True
        )
    for fixture in manifest["fixtures"]:
        image = image.add_local_file(
            ROOT / ASSET_PATH / fixture["filename"],
            (REMOTE_ASSETS / fixture["filename"]).as_posix(),
            copy=True,
        )
    image = image.env(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "/opt/openai-whisper:/opt/whisper-runtime/src:/opt/whisper-runtime",
            "PYTHONUTF8": "1",
            REMOTE_RESOURCES_ENV: "0",
            "WHISPER_MODAL_ENABLE_REMOTE_RESOURCES": "0",
        }
    )
    volume = modal.Volume.from_name(b.MODEL_CACHE_NAME, create_if_missing=False)
    app = modal.App(APP_NAME)
    common = dict(
        image=image,
        serialized=True,
        min_containers=0,
        max_containers=1,
        scaledown_window=2,
        retries=0,
        startup_timeout=300,
        block_network=True,
        restrict_modal_access=True,
        single_use_containers=True,
        include_source=False,
    )

    @app.function(**common, cpu=0.125, memory=128, timeout=30)
    def echo(payload: bytes) -> bytes:
        if type(payload) is not bytes:
            raise TypeError("transport requires bytes")
        return payload

    @app.function(
        **common,
        gpu="T4",
        cloud="aws",
        region="us-west",
        volumes={b.MODEL_CACHE_MOUNT: volume.with_mount_options(read_only=True)},
        cpu=2,
        memory=4096,
        timeout=180,
    )
    def execute(
        snapshot: Mapping[str, Any],
        registration_sha256: str,
        input_evidence: bool = False,
        source_units: bool = False,
        automatic_endpoints: bool = False,
        paced_replay: bool = False,
    ) -> bytes:
        producer = importlib.import_module("infra.modal_word_corpus")
        return producer.b._encode_worker_record(
            producer._run_worker(
                snapshot,
                registration_sha256=registration_sha256,
                modal_module=modal,
                input_evidence=input_evidence,
                source_units=source_units,
                automatic_endpoints=automatic_endpoints,
                paced_replay=paced_replay,
            )
        )

    return app, execute, echo, app.local_entrypoint(name="main")(_modal_main)


if os.environ.get(REMOTE_RESOURCES_ENV) == "1":
    app, run_word_corpus, run_transport_probe, main = _define_modal_resources()
else:
    app = run_word_corpus = run_transport_probe = main = None
