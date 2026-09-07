"""Create an isolated environment for the pinned native Whisper backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from native_backend_setup import (
    BACKEND_REUSE_TREE,
    DEFAULT_REUSE_SETUP_ROOT,
    DEFAULT_SETUP_ROOT,
    NativeSetupError,
    SetupPaths,
    build_manifest,
    ensure_backend,
    ensure_environment,
    install_dependencies,
    load_validated_setup,
    require_prerequisites,
    require_runtime_identity,
    require_safe_setup_root,
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
        help="Setup directory (default: .tmp-native, or .tmp-native-reuse with reuse)",
    )
    parser.add_argument(
        "--alignment-feature-reuse",
        action="store_true",
        help="Apply the pinned optional alignment patch for low-latency/optimized profiles",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing setup without using the network or installing packages",
    )
    args = parser.parse_args()
    if args.root is None:
        args.root = (
            DEFAULT_REUSE_SETUP_ROOT
            if args.alignment_feature_reuse
            else DEFAULT_SETUP_ROOT
        )
    return args


def run_setup(
    paths: SetupPaths, *, alignment_feature_reuse: bool = False
) -> dict[str, object]:
    require_safe_setup_root(paths.root)
    if paths.root.exists() and not paths.root.is_dir():
        raise NativeSetupError(f"setup root is not a directory: {paths.root}")
    try:
        paths.root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise NativeSetupError(
            f"cannot create setup root {paths.root}: {error}"
        ) from error

    tools = require_prerequisites()
    runtime = require_runtime_identity()
    patches = verify_patch_manifest()
    backend = ensure_backend(
        paths, patches, alignment_feature_reuse=alignment_feature_reuse
    )
    python = ensure_environment(paths)
    environment = install_dependencies(python)
    manifest = build_manifest(
        paths=paths,
        runtime=runtime,
        backend=backend,
        python=python,
        patches=patches,
        environment=environment,
        tools=tools,
        alignment_feature_reuse=alignment_feature_reuse,
    )
    write_manifest(paths.manifest, manifest)
    load_validated_setup(paths.manifest)
    return manifest


def next_example_command(paths: SetupPaths) -> list[str]:
    """Return shell-independent arguments for the matching setup root."""

    return [
        "python",
        "tools/run_native_example.py",
        "--root",
        str(paths.root),
    ]


def main() -> int:
    args = parse_args()
    paths = SetupPaths.from_root(args.root)
    try:
        if args.verify_only:
            setup = load_validated_setup(paths.manifest)
            if (
                args.alignment_feature_reuse
                and setup.backend.tree != BACKEND_REUSE_TREE
            ):
                raise NativeSetupError(
                    "selected setup does not enable alignment feature reuse"
                )
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
            manifest = run_setup(
                paths, alignment_feature_reuse=args.alignment_feature_reuse
            )
            backend = manifest["backend"]
            result = {
                "status": "ready",
                "manifest": str(paths.manifest),
                "backend_commit": backend["applied_commit"],
                "backend_tree": backend["tree"],
                "python": manifest["environment"]["python"],
                "bootstrap_downloaded_models": False,
                "next_command": next_example_command(paths),
            }
    except NativeSetupError as error:
        raise SystemExit(f"setup failed: {error}") from error

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
