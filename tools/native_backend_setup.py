"""Build and verify the pinned native Whisper development environment."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIRECTORY = RUNTIME_ROOT / "patches" / "openai-whisper"
CONSTRAINTS_FILE = RUNTIME_ROOT / "constraints" / "native-integration.txt"

SCHEMA_VERSION = "1"
BACKEND_URL = "https://github.com/openai/whisper.git"
BACKEND_BASE_COMMIT = "86098128c0b4f24f0e2aa2994de830614b474227"
BACKEND_BASE_TREE = "f7b3cb8e12a2e84dccacc4c858c33d5a9c114688"
BACKEND_PATCHED_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
TORCH_VERSION = "2.6.0"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

REQUIRED_DISTRIBUTIONS = {
    "jsonschema": "4.25.1",
    "more-itertools": "11.1.0",
    "numba": "0.67.0",
    "numpy": "2.5.2",
    "tiktoken": "0.14.0",
    "torch": TORCH_VERSION,
    "tqdm": "4.70.0",
}

_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*\.patch)")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")


class NativeSetupError(RuntimeError):
    """Report a setup or provenance contract violation."""


@dataclass(frozen=True, slots=True)
class GitIdentity:
    commit: str
    tree: str
    clean: bool


@dataclass(frozen=True, slots=True)
class SetupPaths:
    root: Path
    backend: Path
    environment: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: Path) -> SetupPaths:
        resolved = root.expanduser().resolve()
        return cls(
            root=resolved,
            backend=resolved / "backend",
            environment=resolved / "venv",
            manifest=resolved / "manifest.json",
        )


@dataclass(frozen=True, slots=True)
class ValidatedSetup:
    paths: SetupPaths
    runtime: GitIdentity
    backend: GitIdentity
    python: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checksum_manifest(text: str) -> dict[str, str]:
    """Parse the repository's strict two-space SHA-256 format."""

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise NativeSetupError(
                f"invalid patch checksum entry on line {line_number}"
            )
        digest, name = match.groups()
        if name in entries:
            raise NativeSetupError(f"duplicate patch checksum entry: {name}")
        entries[name] = digest
    if not entries:
        raise NativeSetupError("the patch checksum manifest is empty")
    return entries


def verify_patch_manifest(patch_directory: Path = PATCH_DIRECTORY) -> dict[str, str]:
    manifest = patch_directory / "SHA256SUMS"
    try:
        entries = parse_checksum_manifest(manifest.read_text(encoding="utf-8"))
    except OSError as error:
        raise NativeSetupError(f"cannot read {manifest}: {error}") from error

    observed_names = {path.name for path in patch_directory.glob("*.patch")}
    expected_names = set(entries)
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unlisted: {', '.join(extra)}")
        raise NativeSetupError(
            "patch set does not match SHA256SUMS (" + "; ".join(details) + ")"
        )

    for name, expected in entries.items():
        observed = sha256_file(patch_directory / name)
        if observed != expected:
            raise NativeSetupError(
                f"patch checksum mismatch for {name}: expected {expected}, observed {observed}"
            )
    return entries


def environment_python(environment: Path, *, os_name: str | None = None) -> Path:
    selected_os = os.name if os_name is None else os_name
    if selected_os == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def dependency_install_commands(
    python: Path,
    *,
    platform: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Return commands that can only install into the selected environment."""

    selected_platform = sys.platform if platform is None else platform
    common = (
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--only-binary=:all:",
    )
    if selected_platform == "darwin":
        torch_command = common + (f"torch=={TORCH_VERSION}",)
    else:
        torch_command = common + (
            "--index-url",
            TORCH_CPU_INDEX,
            f"torch=={TORCH_VERSION}",
        )
    requirements_command = common + ("-r", str(CONSTRAINTS_FILE))
    return (
        torch_command,
        requirements_command,
        (str(python), "-m", "pip", "check"),
    )


def verify_dependency_versions(observed: object) -> dict[str, str]:
    if not isinstance(observed, dict) or not all(
        isinstance(name, str) and isinstance(version, str)
        for name, version in observed.items()
    ):
        raise NativeSetupError("dependency versions must be a string map")
    versions = dict(observed)
    if set(versions) != set(REQUIRED_DISTRIBUTIONS):
        raise NativeSetupError(
            f"installed dependency set differs from the contract: {versions!r}"
        )
    for name, expected in REQUIRED_DISTRIBUTIONS.items():
        value = versions[name]
        comparable = value.split("+", 1)[0] if name == "torch" else value
        if comparable != expected:
            raise NativeSetupError(
                f"installed {name} version differs from the contract: "
                f"expected {expected}, observed {value}"
            )
    return versions


def checkpoint_digest_from_url(url: str) -> str:
    """Read the content digest embedded in an official Whisper model URL."""

    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) < 2 or _DIGEST.fullmatch(parts[-2]) is None:
        raise NativeSetupError("the model URL does not contain a SHA-256 digest")
    return parts[-2]


def checkpoint_path(model_name: str, model_url: str, cache: Path) -> Path:
    file_name = Path(urlparse(model_url).path).name
    if not file_name or file_name in {".", ".."}:
        raise NativeSetupError(f"the {model_name!r} model URL has no file name")
    return cache.expanduser().resolve() / file_name


def require_cached_checkpoint(
    model_name: str,
    model_url: str,
    cache: Path,
    *,
    allow_download: bool,
) -> Path:
    """Reject a missing or changed checkpoint unless network use is explicit."""

    path = checkpoint_path(model_name, model_url, cache)
    if allow_download:
        return path
    if not path.is_file():
        raise NativeSetupError(
            f"model checkpoint not found at {path}. Rerun with "
            "--allow-model-download to let Whisper fetch it."
        )
    expected = checkpoint_digest_from_url(model_url)
    observed = sha256_file(path)
    if observed != expected:
        raise NativeSetupError(
            f"model checkpoint checksum mismatch at {path}: expected {expected}, "
            f"observed {observed}. No download was attempted."
        )
    return path


def _command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PIP_PREFIX", "PIP_TARGET", "PIP_USER", "PYTHONHOME", "PYTHONPATH"):
        environment.pop(name, None)
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
    )
    return environment


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=capture,
            text=True,
            encoding="utf-8",
            env=_command_environment(),
        )
    except OSError as error:
        raise NativeSetupError(f"cannot run {command[0]}: {error}") from error
    if completed.returncode != 0:
        detail = ""
        if capture:
            detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise NativeSetupError(
            f"command failed ({completed.returncode}): {' '.join(command)}{suffix}"
        )
    return completed


def _git_output(repository: Path, *arguments: str) -> str:
    result = _run(
        ("git", "-C", str(repository), *arguments),
        capture=True,
    )
    return result.stdout.strip()


def git_identity(repository: Path) -> GitIdentity:
    try:
        top_level = Path(
            _git_output(repository, "rev-parse", "--show-toplevel")
        ).resolve()
    except (OSError, NativeSetupError) as error:
        raise NativeSetupError(
            f"{repository} is not a readable Git worktree"
        ) from error
    if top_level != repository.resolve():
        raise NativeSetupError(
            f"Git reports a different worktree root for {repository}"
        )
    commit = _git_output(repository, "rev-parse", "HEAD")
    tree = _git_output(repository, "rev-parse", "HEAD^{tree}")
    status = _git_output(repository, "status", "--porcelain", "--untracked-files=all")
    if _GIT_OBJECT.fullmatch(commit) is None or _GIT_OBJECT.fullmatch(tree) is None:
        raise NativeSetupError(
            f"Git returned an invalid object identifier for {repository}"
        )
    return GitIdentity(commit=commit, tree=tree, clean=not status)


def require_runtime_identity(*, allow_dirty: bool = False) -> GitIdentity:
    identity = git_identity(RUNTIME_ROOT)
    if not identity.clean and not allow_dirty:
        raise NativeSetupError(
            "the runtime worktree has source changes; commit or stash them before setup"
        )
    return identity


def _backend_state(repository: Path) -> tuple[str, GitIdentity]:
    identity = git_identity(repository)
    if not identity.clean:
        raise NativeSetupError("the backend worktree has source changes")
    if _git_output(repository, "remote", "get-url", "origin") != BACKEND_URL:
        raise NativeSetupError("the backend origin does not match the pinned source")
    if identity.commit == BACKEND_BASE_COMMIT and identity.tree == BACKEND_BASE_TREE:
        return "base", identity
    if identity.tree != BACKEND_PATCHED_TREE:
        raise NativeSetupError(
            "the backend tree does not match the pinned base or patched tree"
        )
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            BACKEND_BASE_COMMIT,
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_command_environment(),
    )
    count = _git_output(
        repository, "rev-list", "--count", f"{BACKEND_BASE_COMMIT}..HEAD"
    )
    if ancestor.returncode != 0 or count != "7":
        raise NativeSetupError(
            "the patched backend does not contain the expected seven-commit series"
        )
    return "patched", identity


def ensure_backend(paths: SetupPaths, patches: dict[str, str]) -> GitIdentity:
    if not paths.backend.exists():
        _run(
            (
                "git",
                "clone",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.eol=lf",
                "--filter=blob:none",
                "--no-tags",
                BACKEND_URL,
                str(paths.backend),
            )
        )
        _run(
            (
                "git",
                "-C",
                str(paths.backend),
                "checkout",
                "--detach",
                BACKEND_BASE_COMMIT,
            )
        )
    elif not paths.backend.is_dir():
        raise NativeSetupError(f"backend path is not a directory: {paths.backend}")

    state, identity = _backend_state(paths.backend)
    if state == "patched":
        return identity

    patch_files = tuple(str(PATCH_DIRECTORY / name) for name in sorted(patches))
    _run(
        (
            "git",
            "-C",
            str(paths.backend),
            "-c",
            "user.name=Whisper Runtime Bootstrap",
            "-c",
            "user.email=whisper-runtime@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "-c",
            "core.hooksPath=",
            "am",
            "--committer-date-is-author-date",
            "--no-gpg-sign",
            "--no-verify",
            *patch_files,
        )
    )
    state, identity = _backend_state(paths.backend)
    if state != "patched":
        raise NativeSetupError(
            "the backend patch series did not produce the expected tree"
        )
    return identity


def ensure_environment(paths: SetupPaths) -> Path:
    python = environment_python(paths.environment)
    if not paths.environment.exists():
        _run((sys.executable, "-m", "venv", str(paths.environment)))
    elif not paths.environment.is_dir():
        raise NativeSetupError(
            f"environment path is not a directory: {paths.environment}"
        )
    if not (paths.environment / "pyvenv.cfg").is_file() or not python.is_file():
        raise NativeSetupError(
            f"an incomplete environment already exists at {paths.environment}; "
            "choose another --root"
        )
    return python


def install_dependencies(python: Path) -> dict[str, str]:
    for command in dependency_install_commands(python):
        _run(command)
    probe = (
        "import importlib.metadata as m,json; "
        f"names={sorted(REQUIRED_DISTRIBUTIONS)!r}; "
        "print(json.dumps({name:m.version(name) for name in names},sort_keys=True))"
    )
    observed = json.loads(_run((str(python), "-I", "-c", probe), capture=True).stdout)
    return verify_dependency_versions(observed)


def _tool_version(command: Sequence[str]) -> str:
    output = _run(command, capture=True).stdout.splitlines()
    return output[0].strip() if output else "unknown"


def require_prerequisites() -> dict[str, str]:
    if sys.version_info < (3, 10):
        raise NativeSetupError("Python 3.10 or later is required")
    for tool in ("git", "ffmpeg"):
        if shutil.which(tool) is None:
            raise NativeSetupError(f"required command not found on PATH: {tool}")
    return {
        "git": _tool_version(("git", "--version")),
        "ffmpeg": _tool_version(("ffmpeg", "-version")),
    }


def _runtime_record(identity: GitIdentity) -> dict[str, object]:
    return {
        "path": str(RUNTIME_ROOT),
        "git_commit": identity.commit,
        "git_tree": identity.tree,
        "clean": identity.clean,
    }


def build_manifest(
    *,
    paths: SetupPaths,
    runtime: GitIdentity,
    backend: GitIdentity,
    python: Path,
    patches: dict[str, str],
    dependencies: dict[str, str],
    tools: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "runtime": _runtime_record(runtime),
        "backend": {
            "path": str(paths.backend),
            "url": BACKEND_URL,
            "base_commit": BACKEND_BASE_COMMIT,
            "base_tree": BACKEND_BASE_TREE,
            "applied_commit": backend.commit,
            "tree": backend.tree,
            "clean": backend.clean,
        },
        "patches": {
            "manifest": "patches/openai-whisper/SHA256SUMS",
            "manifest_sha256": sha256_file(PATCH_DIRECTORY / "SHA256SUMS"),
            "files": patches,
        },
        "environment": {
            "path": str(paths.environment),
            "python": str(python),
            "python_version": _tool_version((str(python), "--version")),
            "constraints": "constraints/native-integration.txt",
            "constraints_sha256": sha256_file(CONSTRAINTS_FILE),
            "dependencies": dependencies,
        },
        "tools": tools,
        "bootstrap_downloaded_models": False,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NativeSetupError(f"{location} must be an object")
    return value


def load_validated_setup(manifest_path: Path) -> ValidatedSetup:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativeSetupError(
            f"cannot read setup manifest {manifest_path}: {error}"
        ) from error
    document = _require_object(manifest, "manifest")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise NativeSetupError("unsupported setup manifest schema")
    if document.get("bootstrap_downloaded_models") is not False:
        raise NativeSetupError("setup manifest must state that it downloaded no models")

    runtime_record = _require_object(document.get("runtime"), "manifest.runtime")
    backend_record = _require_object(document.get("backend"), "manifest.backend")
    environment_record = _require_object(
        document.get("environment"), "manifest.environment"
    )
    patch_record = _require_object(document.get("patches"), "manifest.patches")

    paths = SetupPaths.from_root(manifest_path.resolve().parent)
    if paths.manifest != manifest_path.resolve():
        raise NativeSetupError("setup manifest must be named manifest.json")
    expected_paths = {
        "runtime": (runtime_record.get("path"), RUNTIME_ROOT),
        "backend": (backend_record.get("path"), paths.backend),
        "environment": (environment_record.get("path"), paths.environment),
        "python": (
            environment_record.get("python"),
            environment_python(paths.environment),
        ),
    }
    for label, (recorded, expected) in expected_paths.items():
        if (
            not isinstance(recorded, str)
            or Path(recorded).resolve() != expected.resolve()
        ):
            raise NativeSetupError(
                f"manifest {label} path does not match the setup root"
            )

    if backend_record.get("url") != BACKEND_URL:
        raise NativeSetupError("manifest backend URL does not match the pinned source")
    if backend_record.get("base_commit") != BACKEND_BASE_COMMIT:
        raise NativeSetupError("manifest backend base commit does not match")
    if backend_record.get("base_tree") != BACKEND_BASE_TREE:
        raise NativeSetupError("manifest backend base tree does not match")
    if backend_record.get("tree") != BACKEND_PATCHED_TREE:
        raise NativeSetupError("manifest backend patched tree does not match")

    patches = verify_patch_manifest()
    if patch_record.get("files") != patches:
        raise NativeSetupError("manifest patch list does not match the repository")
    if patch_record.get("manifest_sha256") != sha256_file(
        PATCH_DIRECTORY / "SHA256SUMS"
    ):
        raise NativeSetupError("manifest patch checksum file does not match")
    if environment_record.get("constraints_sha256") != sha256_file(CONSTRAINTS_FILE):
        raise NativeSetupError("manifest dependency constraints do not match")
    verify_dependency_versions(environment_record.get("dependencies"))

    runtime = require_runtime_identity()
    if (
        runtime_record.get("git_commit") != runtime.commit
        or runtime_record.get("git_tree") != runtime.tree
        or runtime_record.get("clean") is not True
    ):
        raise NativeSetupError("runtime source no longer matches the setup manifest")

    state, backend = _backend_state(paths.backend)
    if state != "patched":
        raise NativeSetupError("backend source is not in the patched state")
    if (
        backend_record.get("applied_commit") != backend.commit
        or backend_record.get("clean") is not True
    ):
        raise NativeSetupError("backend source no longer matches the setup manifest")

    python = environment_python(paths.environment)
    if not python.is_file():
        raise NativeSetupError(f"setup Python executable not found: {python}")
    return ValidatedSetup(paths=paths, runtime=runtime, backend=backend, python=python)


def installed_versions() -> dict[str, str]:
    """Return required versions in the current interpreter without importing them."""

    return {name: importlib.metadata.version(name) for name in REQUIRED_DISTRIBUTIONS}
