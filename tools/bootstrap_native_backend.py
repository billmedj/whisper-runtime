"""Create an isolated environment for the pinned native Whisper backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from native_backend_setup import (
    RUNTIME_ROOT,
    NativeSetupError,
    SetupPaths,
    build_manifest,
    ensure_backend,
    ensure_environment,
    install_dependencies,
    load_validated_setup,
    require_prerequisites,
    require_runtime_identity,
    verify_patch_manifest,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a project-local Python environment and a verified patched "
            "OpenAI Whisper source tree. This command does not download a model."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=RUNTIME_ROOT / ".tmp-native",
        help="Setup directory (default: .tmp-native in the repository)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing setup without using the network or installing packages",
    )
    return parser.parse_args()


def run_setup(paths: SetupPaths) -> dict[str, object]:
    if paths.root == RUNTIME_ROOT or paths.root in RUNTIME_ROOT.parents:
        raise NativeSetupError("--root cannot be the repository or its parent")
    if paths.root.exists() and not paths.root.is_dir():
        raise NativeSetupError(f"setup root is not a directory: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)

    tools = require_prerequisites()
    runtime = require_runtime_identity()
    patches = verify_patch_manifest()
    backend = ensure_backend(paths, patches)
    python = ensure_environment(paths)
    dependencies = install_dependencies(python)
    manifest = build_manifest(
        paths=paths,
        runtime=runtime,
        backend=backend,
        python=python,
        patches=patches,
        dependencies=dependencies,
        tools=tools,
    )
    write_manifest(paths.manifest, manifest)
    load_validated_setup(paths.manifest)
    return manifest


def main() -> int:
    args = parse_args()
    paths = SetupPaths.from_root(args.root)
    try:
        if args.verify_only:
            setup = load_validated_setup(paths.manifest)
            result: dict[str, object] = {
                "status": "verified",
                "manifest": str(setup.paths.manifest),
                "runtime_commit": setup.runtime.commit,
                "backend_commit": setup.backend.commit,
                "backend_tree": setup.backend.tree,
                "python": str(setup.python),
                "bootstrap_downloaded_models": False,
            }
        else:
            manifest = run_setup(paths)
            backend = manifest["backend"]
            result = {
                "status": "ready",
                "manifest": str(paths.manifest),
                "backend_commit": backend["applied_commit"],
                "backend_tree": backend["tree"],
                "python": manifest["environment"]["python"],
                "bootstrap_downloaded_models": False,
                "next_command": "python tools/run_native_example.py",
            }
    except NativeSetupError as error:
        raise SystemExit(f"setup failed: {error}") from error

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
