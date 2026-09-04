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
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIRECTORY = RUNTIME_ROOT / "patches" / "openai-whisper"
CONSTRAINTS_FILE = RUNTIME_ROOT / "constraints" / "native-integration.txt"
DEFAULT_SETUP_ROOT = RUNTIME_ROOT / ".tmp-native"

SCHEMA_VERSION = "2"
BACKEND_URL = "https://github.com/openai/whisper.git"
BACKEND_BASE_COMMIT = "86098128c0b4f24f0e2aa2994de830614b474227"
BACKEND_BASE_TREE = "f7b3cb8e12a2e84dccacc4c858c33d5a9c114688"
BACKEND_PATCHED_TREE = "c011d2563c26763b5f147026e6b18ef85bccd4fb"
TORCH_VERSION = "2.6.0"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
NATIVE_PYTHON_MIN = (3, 12)
NATIVE_PYTHON_MAX = (3, 13)
PATCH_MANIFEST_PATH = "patches/openai-whisper/SHA256SUMS"
CONSTRAINTS_PATH = "constraints/native-integration.txt"

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
_DISTRIBUTION_SEPARATOR = re.compile(r"[-_.]+")


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


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    """Record the interpreter and every installed Python distribution."""

    python_version: str
    dependencies: dict[str, str]
    distributions: dict[str, str]


def require_safe_setup_root(
    root: Path,
    *,
    runtime_root: Path = RUNTIME_ROOT,
) -> Path:
    """Reject setup paths that can add generated files to the source tree."""

    resolved = root.expanduser().resolve()
    repository = runtime_root.resolve()
    default = repository / DEFAULT_SETUP_ROOT.name
    if resolved == repository or resolved in repository.parents:
        raise NativeSetupError("--root cannot be the repository or its parent")
    if repository in resolved.parents and resolved != default:
        raise NativeSetupError(
            "--root inside the repository must be .tmp-native; "
            "use a path outside the repository for another setup"
        )
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise NativeSetupError(f"cannot read {path}: {error}") from error
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
        "--isolated",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--only-binary=:all:",
    )
    if selected_platform == "darwin":
        torch_command = common + (
            "--index-url",
            PYPI_INDEX,
            f"torch=={TORCH_VERSION}",
        )
    else:
        torch_command = common + (
            "--index-url",
            TORCH_CPU_INDEX,
            f"torch=={TORCH_VERSION}",
        )
    requirements_command = common + (
        "--index-url",
        PYPI_INDEX,
        "-r",
        str(CONSTRAINTS_FILE),
    )
    return (
        torch_command,
        requirements_command,
        (str(python), "-m", "pip", "--isolated", "check"),
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


def _canonical_distribution_name(name: str) -> str:
    return _DISTRIBUTION_SEPARATOR.sub("-", name).lower()


def verify_distribution_inventory(observed: object) -> dict[str, str]:
    """Validate a complete, canonical distribution inventory."""

    if not isinstance(observed, dict) or not all(
        isinstance(name, str)
        and bool(name)
        and name == _canonical_distribution_name(name)
        and isinstance(version, str)
        and bool(version)
        for name, version in observed.items()
    ):
        raise NativeSetupError("resolved distributions must be a canonical string map")
    return dict(sorted(observed.items()))


def _reject_json_constant(value: str) -> None:
    raise NativeSetupError(f"JSON constant is not permitted: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for name, value in pairs:
        if name in document:
            raise NativeSetupError(f"duplicate JSON field is not permitted: {name}")
        document[name] = value
    return document


def _load_json(text: str, *, subject: str) -> object:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except json.JSONDecodeError as error:
        raise NativeSetupError(f"invalid JSON from {subject}: {error}") from error


def inspect_environment(python: Path) -> EnvironmentIdentity:
    """Read the actual interpreter and installed distributions in one process."""

    probe = """
import importlib.metadata as metadata
import json
import platform
import re

canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
distributions = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    version = distribution.version
    if name and version:
        distributions[canonical(name)] = version
required = {
    name: metadata.version(name)
    for name in %r
}
print(json.dumps({
    "implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "dependencies": required,
    "distributions": distributions,
}, sort_keys=True))
""" % sorted(REQUIRED_DISTRIBUTIONS)
    completed = _run((str(python), "-I", "-c", probe), capture=True)
    document = _require_object(
        _load_json(completed.stdout, subject="the setup Python environment"),
        "environment probe",
    )
    if document.get("implementation") != "CPython":
        raise NativeSetupError("the pinned native backend requires CPython")
    version = document.get("python_version")
    if not isinstance(version, str) or re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise NativeSetupError("environment probe returned an invalid Python version")
    major, minor, _ = (int(part) for part in version.split("."))
    require_supported_python((major, minor))
    dependencies = verify_dependency_versions(document.get("dependencies"))
    distributions = verify_distribution_inventory(document.get("distributions"))
    for name, dependency_version in dependencies.items():
        if distributions.get(_canonical_distribution_name(name)) != dependency_version:
            raise NativeSetupError(
                f"resolved distribution inventory differs for {name}"
            )
    return EnvironmentIdentity(
        python_version=f"Python {version}",
        dependencies=dependencies,
        distributions=distributions,
    )


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


def isolated_command_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Remove Python and pip injection settings from a child environment."""

    environment = dict(os.environ if source is None else source)
    for name in tuple(environment):
        if name.upper().startswith("PIP_") or name.upper().startswith("PYTHON"):
            environment.pop(name)
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
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
            env=isolated_command_environment(),
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
        env=isolated_command_environment(),
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


def install_dependencies(python: Path) -> EnvironmentIdentity:
    for command in dependency_install_commands(python):
        _run(command)
    return inspect_environment(python)


def _tool_version(command: Sequence[str]) -> str:
    completed = _run(command, capture=True)
    output = (completed.stdout or completed.stderr).splitlines()
    return output[0].strip() if output else "unknown"


def require_supported_python(version: tuple[int, int]) -> None:
    if not NATIVE_PYTHON_MIN <= version <= NATIVE_PYTHON_MAX:
        raise NativeSetupError(
            "the pinned native backend requires CPython 3.12 or 3.13"
        )


def current_tool_versions() -> dict[str, str]:
    for tool in ("git", "ffmpeg"):
        if shutil.which(tool) is None:
            raise NativeSetupError(f"required command not found on PATH: {tool}")
    return {
        "git": _tool_version(("git", "--version")),
        "ffmpeg": _tool_version(("ffmpeg", "-version")),
    }


def require_prerequisites() -> dict[str, str]:
    require_supported_python((sys.version_info.major, sys.version_info.minor))
    return current_tool_versions()


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
    environment: EnvironmentIdentity,
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
            "manifest": PATCH_MANIFEST_PATH,
            "manifest_sha256": sha256_file(PATCH_DIRECTORY / "SHA256SUMS"),
            "files": patches,
        },
        "environment": {
            "path": str(paths.environment),
            "python": str(python),
            "python_version": environment.python_version,
            "constraints": CONSTRAINTS_PATH,
            "constraints_sha256": sha256_file(CONSTRAINTS_FILE),
            "dependencies": environment.dependencies,
            "resolved_distributions": environment.distributions,
        },
        "tools": tools,
        "bootstrap_downloaded_models": False,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        raise NativeSetupError(
            f"cannot write setup manifest {path}: {error}"
        ) from error


def _require_object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NativeSetupError(f"{location} must be an object")
    return value


def _require_exact_keys(
    record: Mapping[str, object], expected: set[str], location: str
) -> None:
    if set(record) != expected:
        raise NativeSetupError(
            f"{location} fields do not match schema {SCHEMA_VERSION}"
        )


def _require_nonempty_strings(value: object, location: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(name, str)
        and bool(name)
        and isinstance(text, str)
        and bool(text.strip())
        for name, text in value.items()
    ):
        raise NativeSetupError(f"{location} must be a non-empty string map")
    return dict(value)


def _validate_created_at(value: object) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NativeSetupError("manifest.created_at must be a UTC timestamp")
    try:
        timestamp = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise NativeSetupError("manifest.created_at must be a UTC timestamp") from error
    if timestamp.utcoffset() != dt.timedelta(0):
        raise NativeSetupError("manifest.created_at must be a UTC timestamp")


def verify_recorded_environment(
    recorded: Mapping[str, object], python: Path
) -> EnvironmentIdentity:
    """Compare the manifest with the live isolated Python environment."""

    actual = inspect_environment(python)
    if recorded.get("python_version") != actual.python_version:
        raise NativeSetupError("setup Python version no longer matches the manifest")
    if recorded.get("dependencies") != actual.dependencies:
        raise NativeSetupError("setup dependency versions no longer match the manifest")
    if recorded.get("resolved_distributions") != actual.distributions:
        raise NativeSetupError(
            "setup distribution inventory no longer matches the manifest"
        )
    return actual


def load_validated_setup(manifest_path: Path) -> ValidatedSetup:
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise NativeSetupError(
            f"cannot read setup manifest {manifest_path}: {error}"
        ) from error
    manifest = _load_json(text, subject=f"setup manifest {manifest_path}")
    document = _require_object(manifest, "manifest")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise NativeSetupError("unsupported setup manifest schema")
    _require_exact_keys(
        document,
        {
            "schema_version",
            "created_at",
            "runtime",
            "backend",
            "patches",
            "environment",
            "tools",
            "bootstrap_downloaded_models",
        },
        "manifest",
    )
    _validate_created_at(document.get("created_at"))
    if document.get("bootstrap_downloaded_models") is not False:
        raise NativeSetupError("setup manifest must state that it downloaded no models")

    runtime_record = _require_object(document.get("runtime"), "manifest.runtime")
    backend_record = _require_object(document.get("backend"), "manifest.backend")
    environment_record = _require_object(
        document.get("environment"), "manifest.environment"
    )
    patch_record = _require_object(document.get("patches"), "manifest.patches")
    tool_record = _require_object(document.get("tools"), "manifest.tools")
    _require_exact_keys(
        runtime_record, {"path", "git_commit", "git_tree", "clean"}, "manifest.runtime"
    )
    _require_exact_keys(
        backend_record,
        {
            "path",
            "url",
            "base_commit",
            "base_tree",
            "applied_commit",
            "tree",
            "clean",
        },
        "manifest.backend",
    )
    _require_exact_keys(
        patch_record,
        {"manifest", "manifest_sha256", "files"},
        "manifest.patches",
    )
    _require_exact_keys(
        environment_record,
        {
            "path",
            "python",
            "python_version",
            "constraints",
            "constraints_sha256",
            "dependencies",
            "resolved_distributions",
        },
        "manifest.environment",
    )
    _require_exact_keys(tool_record, {"git", "ffmpeg"}, "manifest.tools")

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
    if patch_record.get("manifest") != PATCH_MANIFEST_PATH:
        raise NativeSetupError("manifest patch checksum path does not match")
    if patch_record.get("files") != patches:
        raise NativeSetupError("manifest patch list does not match the repository")
    if patch_record.get("manifest_sha256") != sha256_file(
        PATCH_DIRECTORY / "SHA256SUMS"
    ):
        raise NativeSetupError("manifest patch checksum file does not match")
    if environment_record.get("constraints") != CONSTRAINTS_PATH:
        raise NativeSetupError("manifest dependency constraints path does not match")
    if environment_record.get("constraints_sha256") != sha256_file(CONSTRAINTS_FILE):
        raise NativeSetupError("manifest dependency constraints do not match")
    verify_dependency_versions(environment_record.get("dependencies"))
    verify_distribution_inventory(environment_record.get("resolved_distributions"))

    recorded_tools = _require_nonempty_strings(tool_record, "manifest.tools")
    if recorded_tools != current_tool_versions():
        raise NativeSetupError("host tool versions no longer match the setup manifest")

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
    verify_recorded_environment(environment_record, python)
    return ValidatedSetup(paths=paths, runtime=runtime, backend=backend, python=python)


def installed_versions() -> dict[str, str]:
    """Return required versions in the current interpreter without importing them."""

    try:
        return {
            name: importlib.metadata.version(name) for name in REQUIRED_DISTRIBUTIONS
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise NativeSetupError(
            f"required distribution is not installed: {error}"
        ) from error
