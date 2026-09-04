# Real backend quick start

This procedure builds the patched OpenAI Whisper backend used by
`NativeWhisperAdapter`. It keeps the backend, Python environment, and model
cache inside one ignored project directory.

## Requirements

- Python 3.10 or later
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
5. Python packages are installed only in `.tmp-native/venv` at the pinned
   versions used by native CI.
6. A local manifest records the runtime, backend, patches, tools, and package
   versions.

The bootstrap does not download a Whisper checkpoint or an audio file. The
default audio example already exists in the cloned Whisper test tree.

Verify an existing setup without network access or package installation:

```sh
python tools/bootstrap_native_backend.py --verify-only
```

The command fails instead of replacing an existing backend or environment that
does not match the contract. Use a different `--root` when you need a separate
setup.

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

This setup reproduces the tested CPU development profile. It is not a hermetic
binary build: Python and package wheels remain specific to the host platform,
and FFmpeg is supplied by the host. Package versions are pinned, but wheel
files are not locked by hash across all supported platforms.

The example handles one unbatched window of at most 30 seconds. It does not
enable CUDA, live audio streaming, concurrent adapter transactions, or durable
resume.
