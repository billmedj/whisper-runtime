# Real backend quick start

This procedure builds the patched OpenAI Whisper backend used by
`NativeWhisperAdapter`. It keeps the backend, Python environment, and model
cache inside one ignored project directory.

## Requirements

- CPython 3.12 or 3.13
- Git
- FFmpeg on `PATH`
- network access for the first setup

The bootstrap does not install system software. Install FFmpeg with the package
manager for your operating system if the command is not available.

## Build the environment

Run this command from the repository root:

```sh
python tools/bootstrap_native_backend.py
```

The command creates `.tmp-native` and performs these checks:

1. The runtime worktree is clean and has a full Git identity.
2. The OpenAI Whisper source is checked out at the pinned commit.
3. Each integration patch matches its recorded SHA-256 digest.
4. The seven-patch series produces the expected Git tree.
5. Required dependencies and the current runtime package are installed only in
   `.tmp-native/venv`.
6. A local manifest records the runtime, backend, patches, tools, interpreter,
   and complete resolved distribution inventory.

The bootstrap does not download a Whisper model checkpoint or fetch a separate
audio fixture. The pinned source checkout includes `tests/jfk.flac`.

Verify an existing setup without network access or package installation:

```sh
python tools/bootstrap_native_backend.py --verify-only
```

Verification reads the actual interpreter and installed distributions from the
setup environment. It fails if they differ from the manifest. It does not run
pip, install packages, or use the network.

Verification fails on any mismatch. The bootstrap does not delete an existing
backend or environment that it cannot reuse safely. Use a different `--root`
when you need a separate setup. A custom root must be outside the repository:

```sh
python tools/bootstrap_native_backend.py --root /path/to/native-setup
python tools/run_native_example.py --root /path/to/native-setup --allow-model-download
```

The bootstrap result returns the second command as an argument array. This
keeps paths with spaces unambiguous in every shell.

## Run one native transaction

The example uses `tiny.en` on the CPU. The first run needs the model checkpoint.
Permit that download explicitly:

```sh
python tools/run_native_example.py --allow-model-download
```

Later runs use the verified file in `.tmp-native/models` and need no network:

```sh
python tools/run_native_example.py
```

To use a checkpoint that is already on disk:

```sh
python tools/run_native_example.py --model-cache /path/to/whisper-models
```

On Windows, the same command accepts a Windows path:

```powershell
python tools\run_native_example.py `
  --model-cache D:\models\whisper
```

Pass `--audio` to decode another local file. Without that option, the example
uses `backend/tests/jfk.flac` from the pinned source tree and checks its expected
transcript.

The runner verifies the setup manifest, both Git worktrees, the patched backend
tree, the dependency contract, and the checkpoint checksum before it starts the
transaction. It then launches the example with the Python executable from the
local environment. It does not use packages from the user site directory.

## Files created locally

```text
.tmp-native/
  backend/       pinned and patched OpenAI Whisper worktree
  venv/          isolated Python environment
  models/        model cache, created only after explicit permission
  manifest.json  source and environment identity
```

`.tmp-native` is excluded from Git. Do not commit model checkpoints, audio
files, or the generated manifest.

## Scope

This setup builds the tested default single-lane CPU profile. Native CI runs the
complete networked procedure on Ubuntu 24.04 x86-64 with CPython 3.13. Offline
tests cover command construction and path handling for Windows and macOS, but
the complete setup is not yet validated on those systems.

This is not a hermetic binary build. Python, package wheels, and FFmpeg remain
specific to the host. Required top-level package versions are pinned. Their
transitive dependencies and wheel files are not locked by hash. The manifest
records the complete environment that pip resolved, and later verification
rejects any change to that inventory. It does not attest installed package
file contents.

The example handles one unbatched window of at most 30 seconds with the default
single-lane profile. The runtime also contains an experimental two-lane CPU
profile. This quick start does not exercise it, and the repository does not yet
contain a real-model adapter-boundary record for that profile. CUDA, live audio
streaming, and durable resume remain outside this example.
