"""Build four bounded synthetic acoustic stress cases from frozen local PCM.

No downloads, model imports, remote resources, or writes occur here. These are
synthetic transformations of the same six registered utterance occurrences,
not recordings of naturally continuous/no-pause or naturally weak speech.
The independent reference text is unchanged; no recognition claim is implied.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from array import array
from pathlib import Path
from typing import Any

from tools.prepare_speech_corpus import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "experiments/modal-word-corpus-v1.json"
ASSET_PATH = "artifacts/speech-corpus-v1"
SOURCE_CASE_ID = "three-speakers-repeat-pauses"
SAMPLE_RATE_HZ = 16_000
MAX_CASE_SAMPLES = 120 * SAMPLE_RATE_HZ
MAX_TOTAL_SAMPLES = 175 * SAMPLE_RATE_HZ
CASE_IDS = (
    "mixed-control",
    "concatenated-no-added-pauses",
    "mixed-continuous-noise64",
    "mixed-attenuated32",
)
NOISE_SEED = 0x5EED
NOISE_AMPLITUDE = 64
ATTENUATION_DIVISOR = 32
QUIET_PEAK = 32
FRAME_SAMPLES = 320


def _digest(pcm: bytes) -> str:
    return hashlib.sha256(pcm).hexdigest()


def _samples(pcm: bytes) -> array:
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise ValueError("PCM must contain complete nonempty signed16LE samples")
    result = array("h")
    if result.itemsize != 2:
        raise ValueError("this platform does not provide 16-bit array samples")
    result.frombytes(pcm)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def _pcm(samples: array) -> bytes:
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def _add_noise(pcm: bytes) -> tuple[bytes, int]:
    """Independent two-level LCG noise, then saturating signed16 addition."""
    samples = _samples(pcm)
    state, clipped = NOISE_SEED, 0
    for index, sample in enumerate(samples):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        noise = NOISE_AMPLITUDE if state & 0x80000000 else -NOISE_AMPLITUDE
        value = sample + noise
        bounded = max(-32768, min(32767, value))
        clipped += bounded != value
        samples[index] = bounded
    return _pcm(samples), clipped


def _attenuate(pcm: bytes) -> bytes:
    """Integer division toward zero, with no gain normalization or dither."""
    samples = _samples(pcm)
    for index, sample in enumerate(samples):
        magnitude = abs(sample) // ATTENUATION_DIVISOR
        samples[index] = -magnitude if sample < 0 else magnitude
    return _pcm(samples)


def signal_stats(pcm: bytes) -> dict[str, Any]:
    """Exact integer moments plus derived RMS; frames include a partial tail."""
    samples = _samples(pcm)
    count = len(samples)
    squared_sum = sum(sample * sample for sample in samples)
    frame_count = (count + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    quiet_frames = sum(
        max(abs(sample) for sample in samples[offset : offset + FRAME_SAMPLES])
        <= QUIET_PEAK
        for offset in range(0, count, FRAME_SAMPLES)
    )
    rms = math.sqrt(squared_sum / count)
    return {
        "sample_count": count,
        "minimum_s16": min(samples),
        "maximum_s16": max(samples),
        "peak_abs_s16": max(abs(sample) for sample in samples),
        "sum_s16": sum(samples),
        "sum_squares_s16": squared_sum,
        "rms_s16": rms,
        "rms_dbfs": 20 * math.log10(rms / 32768) if rms else None,
        "zero_samples": samples.count(0),
        "full_scale_samples": samples.count(-32768) + samples.count(32767),
        "quiet_peak_s16": QUIET_PEAK,
        "quiet_frame_samples": FRAME_SAMPLES,
        "frame_count": frame_count,
        "quiet_frame_count": quiet_frames,
        "partial_tail_included": count % FRAME_SAMPLES != 0,
    }


def build_cases(
    root: Path = ROOT, asset_dir: Path | None = None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Return JSON-safe metadata and a separate ordered ID-to-PCM mapping.

    ``root`` owns the existing corpus manifest; ``asset_dir`` can point to the
    cached three PCM assets in another environment. No original registration
    is copied or changed. Every source byte is checked against its registration.
    """
    root = Path(root)
    asset_dir = Path(asset_dir) if asset_dir is not None else root / ASSET_PATH
    manifest_path = root / MANIFEST_PATH
    if not 0 < manifest_path.stat().st_size <= 256 * 1024:
        raise ValueError("corpus manifest exceeds the local metadata bound")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    # Reuse the preparation tool's side-effect-free source registration parser.
    # The older infra build_case module conditionally defines remote resources
    # at import time, so importing it is deliberately avoided in this helper.
    fixtures = load_manifest(manifest_path)
    if len(fixtures) != 3:
        raise ValueError("acoustic cases require exactly three registered fixtures")
    originals = [
        case
        for case in manifest.get("cases", [])
        if isinstance(case, dict) and case.get("id") == SOURCE_CASE_ID
    ]
    if len(originals) != 1:
        raise ValueError("registered mixed source case is missing or duplicated")
    original = originals[0]
    parts = original.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= 32:
        raise ValueError("registered mixture needs bounded explicit parts")
    inventory = {fixture.fixture_id: fixture for fixture in fixtures}
    source_pcm: dict[str, bytes] = {}
    for fixture in fixtures:
        if not 0 < fixture.sample_count <= MAX_CASE_SAMPLES:
            raise ValueError("registered fixture exceeds the acoustic input bound")
        path = asset_dir / fixture.filename
        if path.resolve().parent != asset_dir.resolve():
            raise ValueError("registered fixture escapes the asset directory")
        if path.stat().st_size != fixture.sample_count * 2:
            raise ValueError("registered fixture sample count mismatch")
        content = path.read_bytes()
        if (
            len(content) != fixture.sample_count * 2
            or _digest(content) != fixture.pcm_sha256
        ):
            raise ValueError("registered fixture PCM digest mismatch")
        source_pcm[fixture.fixture_id] = content
    chunks, speech_chunks, references, speech_parts = [], [], [], []
    sample_count = inserted_samples = 0
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("invalid acoustic source part")
        if set(part) == {"fixture_id"} and part["fixture_id"] in inventory:
            fixture = inventory[part["fixture_id"]]
            content = source_pcm[fixture.fixture_id]
            speech_chunks.append(content)
            speech_parts.append(copy.deepcopy(part))
            references.append(fixture.reference_text)
        elif set(part) == {"silence_ms"} and type(part["silence_ms"]) is int:
            silence_samples = part["silence_ms"] * 16
            if not 0 <= silence_samples <= MAX_CASE_SAMPLES - sample_count:
                raise ValueError("inserted silence exceeds the acoustic input bound")
            content = bytes(silence_samples * 2)
            inserted_samples += silence_samples
        else:
            raise ValueError("unknown acoustic source part")
        sample_count += len(content) // 2
        if sample_count > MAX_CASE_SAMPLES:
            raise ValueError("mixed case exceeds the acoustic input bound")
        chunks.append(content)
    if len(speech_chunks) != 6:
        raise ValueError("acoustic mixture requires six untrimmed fixture occurrences")
    if 4 * sample_count - inserted_samples > MAX_TOTAL_SAMPLES:
        raise ValueError("acoustic suite exceeds the 175-second total source bound")
    mixed = b"".join(chunks)
    reference = " ".join(" ".join(references).split())
    mixed_sha256 = _digest(mixed)
    if (
        type(original.get("sample_count")) is not int
        or original["sample_count"] != sample_count
        or original.get("pcm_sha256") != mixed_sha256
        or original.get("reference_text") != reference
    ):
        raise ValueError("mixed source PCM or reference differs from registration")
    noisy, clipped = _add_noise(mixed)
    variants = {
        CASE_IDS[0]: mixed,
        CASE_IDS[1]: b"".join(speech_chunks),
        CASE_IDS[2]: noisy,
        CASE_IDS[3]: _attenuate(mixed),
    }
    recipes = (
        {
            "operation": "registered-parts-concatenation/v1",
            "parts": copy.deepcopy(parts),
        },
        {
            "operation": "remove-only-inserted-silence/v1",
            "parts": speech_parts,
            "removed_silence_samples": inserted_samples,
            "fixture_internal_silence_preserved": True,
            "natural_continuous_speech": False,
        },
        {
            "operation": "add-two-level-lcg-noise/v1",
            "amplitude_s16": NOISE_AMPLITUDE,
            "seed_u32": NOISE_SEED,
            "lcg_multiplier": 1_664_525,
            "lcg_increment": 1_013_904_223,
            "lcg_modulus": 2**32,
            "state_update": "update before each sample",
            "noise_sign": "positive iff updated state bit31 is set",
            "addition": "saturate to [-32768,32767]",
        },
        {
            "operation": "integer-attenuation/v1",
            "divisor": ATTENUATION_DIVISOR,
            "rounding": "toward-zero",
            "dither": False,
            "normalization": False,
        },
    )
    cases = [
        {
            "id": case_id,
            "reference_text": reference,
            "sample_count": len(content) // 2,
            "duration_seconds": len(content) / (2 * SAMPLE_RATE_HZ),
            "pcm_sha256": _digest(content),
            "input_sha256": mixed_sha256,
            "recipe": recipe,
            "clipping_count": clipped if case_id == CASE_IDS[2] else 0,
            "signal_stats": signal_stats(content),
        }
        for (case_id, content), recipe in zip(variants.items(), recipes)
    ]
    total_samples = sum(case["sample_count"] for case in cases)
    record = {
        "schema": "acoustic-stress-cases/v1",
        "synthetic": True,
        "natural_no_pause_speech": False,
        "recognition_improvement_claim": False,
        "audio_format": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
            "encoding": "pcm_s16le",
        },
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "total_samples": total_samples,
        "total_duration_seconds": total_samples / SAMPLE_RATE_HZ,
        "maximum_total_duration_seconds": 175,
        "source_manifest": {
            "id": "modal-word-corpus-v1",
            "path": MANIFEST_PATH,
            "sha256": _digest(manifest_bytes),
            "case_id": SOURCE_CASE_ID,
        },
        "source_fixtures": [
            {
                "id": fixture.fixture_id,
                "filename": fixture.filename,
                "sample_count": fixture.sample_count,
                "pcm_sha256": fixture.pcm_sha256,
            }
            for fixture in fixtures
        ],
        "cases": cases,
    }
    return record, variants
