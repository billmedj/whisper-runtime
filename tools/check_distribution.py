from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

SDIST_REQUIRED_SUFFIXES = {
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "conformance/audio-manifest.json",
    "conformance/cases.json",
    "conformance/fixture.schema.json",
    "constraints/native-integration.txt",
    "constraints/modal-client.txt",
    "docs/CUDA_QUALIFICATION_CONTRACT.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/MODAL_GPU_VALIDATION.md",
    "docs/rfcs/0001-state-resource-execution.md",
    "evidence/README.md",
    "evidence/native-cpu-tiny-en-jfk-2026-09-03.json",
    "evidence/native-cpu-tiny-en-jfk-interleaving-2026-09-03.json",
    "evidence/native-cpu-tiny-en-jfk-runtime-concurrency-2026-09-04.json",
    "evidence/native-cpu-tiny-en-jfk-threaded-2026-09-04.json",
    "evidence/native-interleaving.schema.json",
    "evidence/native-runtime-concurrency.schema.json",
    "evidence/native-threaded.schema.json",
    "evidence/modal-cuda-readiness.schema.json",
    "evidence/modal-native-cuda-qualification-v1-attempt-2026-09-04.jsonl",
    "evidence/modal-native-cuda-qualification-v2-attempt-2026-09-04.jsonl",
    "evidence/modal-native-cuda-qualification-v3-attempt-2026-09-04.jsonl",
    "evidence/modal-native-cuda-qualification-v4-attempt-2026-09-04.jsonl",
    "evidence/modal-native-cuda-qualification-v5-attempt-2026-09-04.jsonl",
    "evidence/modal-native-cuda-qualification.schema.json",
    "experiments/native-cuda-qualification-v1.json",
    "experiments/native-cuda-qualification-v2.json",
    "experiments/native-cuda-qualification-v3.json",
    "experiments/native-cuda-qualification-v4.json",
    "experiments/native-cuda-qualification-v5.json",
    "experiments/native-cuda-qualification-v6.json",
    "examples/minimal_transaction.py",
    "formal/lean/WhisperRuntimeFormal.lean",
    "formal/lean/WhisperRuntimeFormal/CompletionFence.lean",
    "formal/lean/WhisperRuntimeFormal/StateResource.lean",
    "infra/__init__.py",
    "infra/modal_gpu_validation.py",
    "patches/openai-whisper/0001-Make-native-inference-state-request-local.patch",
    "patches/openai-whisper/0002-Make-decode-options-request-local.patch",
    "patches/openai-whisper/0003-Prototype-suspendable-token-step-decoding.patch",
    "patches/openai-whisper/0004-Harden-request-local-decode-options.patch",
    "patches/openai-whisper/0005-Fix-grouped-decoding-for-audio-batches.patch",
    "patches/openai-whisper/0006-Harden-suspendable-decode-lifecycle.patch",
    "patches/openai-whisper/0007-Serialize-legacy-cache-run-lifetimes.patch",
    "patches/openai-whisper/LICENSE",
    "patches/openai-whisper/README.md",
    "patches/openai-whisper/SHA256SUMS",
    "pyproject.toml",
    "src/whisper_runtime/py.typed",
    "tools/check_repository.py",
    "tools/compare_whisper_fixtures.py",
    "tools/smoke_native_whisper.py",
    "tools/test_modal_cuda_record.py",
    "tools/test_modal_native_cuda_qualification.py",
    "tools/validate_modal_cuda_record.py",
    "tools/validate_modal_native_cuda_qualification.py",
    "tools/validate_interleaving_record.py",
    "tools/validate_runtime_concurrency_record.py",
    "tools/validate_threaded_record.py",
    "tools/verify_native_interleaving.py",
    "tools/verify_native_runtime_concurrency.py",
    "tools/verify_native_threaded.py",
}
FORBIDDEN_PARTS = {
    ".lake",
    "__pycache__",
    "build",
    "conformance/cache",
    "dist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check built package contents.")
    parser.add_argument("distribution_directory", type=Path)
    return parser.parse_args()


def _single(paths: list[Path], label: str, failures: list[str]) -> Path | None:
    if len(paths) != 1:
        failures.append(f"expected one {label}; found {len(paths)}")
        return None
    return paths[0]


def _has_suffix(names: set[str], suffix: str) -> bool:
    return any(name == suffix or name.endswith(f"/{suffix}") for name in names)


def _check_forbidden(names: set[str], label: str) -> list[str]:
    failures: list[str] = []
    for name in names:
        normalized = f"/{name.strip('/')}"
        if any(f"/{part}/" in f"{normalized}/" for part in FORBIDDEN_PARTS):
            failures.append(f"{label} contains excluded path: {name}")
    return failures


def check_wheel(path: Path) -> list[str]:
    failures: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for required in ("whisper_runtime/__init__.py", "whisper_runtime/py.typed"):
            if required not in names:
                failures.append(f"wheel is missing {required}")
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            failures.append(f"wheel has {len(metadata_names)} METADATA files")
        else:
            metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
            if metadata.get("License-Expression") != "Apache-2.0 AND MIT":
                failures.append("wheel metadata has an unexpected License-Expression")
            if metadata.get("Requires-Python") != ">=3.10":
                failures.append(
                    "wheel metadata has an unexpected Requires-Python value"
                )
        required_licenses = (
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/NOTICE",
            ".dist-info/licenses/THIRD_PARTY_NOTICES.md",
            ".dist-info/licenses/patches/openai-whisper/LICENSE",
        )
        for required in required_licenses:
            if not any(name.endswith(required) for name in names):
                failures.append(f"wheel is missing packaged license file {required}")
        failures.extend(_check_forbidden(names, "wheel"))
    return failures


def check_sdist(path: Path) -> list[str]:
    failures: list[str] = []
    with tarfile.open(path, mode="r:gz") as archive:
        names = {member.name.replace("\\", "/") for member in archive.getmembers()}
    for suffix in sorted(SDIST_REQUIRED_SUFFIXES):
        if not _has_suffix(names, suffix):
            failures.append(f"source distribution is missing {suffix}")
    failures.extend(_check_forbidden(names, "source distribution"))
    return failures


def main() -> int:
    args = parse_args()
    directory = args.distribution_directory
    failures: list[str] = []
    wheel = _single(sorted(directory.glob("*.whl")), "wheel", failures)
    sdist = _single(sorted(directory.glob("*.tar.gz")), "source distribution", failures)
    if wheel is not None:
        failures.extend(check_wheel(wheel))
    if sdist is not None:
        failures.extend(check_sdist(sdist))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("distribution checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
