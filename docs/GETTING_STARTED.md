# Getting started

This alpha provides a Python SDK and a single-stream transcript command. Start
with the core smoke test, then use the verified native setup for real audio.
The local transcription example below selects `low-latency-v2` explicitly.

## 1. Try the core without a model

The core requires Python 3.10 or later. If you also want native transcription,
use **CPython 3.12 or 3.13** from the start. Run these commands in a terminal:

```console
git clone https://github.com/billmedj/whisper-runtime.git
cd whisper-runtime
git checkout v0.1.0a1
python -m venv .venv
```

Activate the environment. In a POSIX shell:

```sh
. .venv/bin/activate
```

In Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install and run the example:

```console
python -m pip install .
python examples/minimal_transaction.py
whisper-runtime --help
```

The example prints `Example transcript` after committing one result and returning
the worker's capacity. It uses no Whisper, PyTorch, GPU, audio or model download.

## 2. Prepare the native environment

Stay at the repository root. You need CPython 3.12 or 3.13, Git, FFmpeg on `PATH`,
and network access for this setup step. The runtime checkout must be clean.

```console
python tools/bootstrap_native_backend.py --alignment-feature-reuse
python tools/bootstrap_native_backend.py --alignment-feature-reuse --verify-only
```

Bootstrap creates `.tmp-native-reuse/`: a pinned, patched Whisper checkout, an
isolated `venv`, the installed runtime and `manifest.json`. The reuse patch is
required by `low-latency-v2`; the ordinary `.tmp-native` backend is not compatible
with that profile. Do not install an arbitrary `openai-whisper` package instead.
Bootstrap fetches source and installs dependencies, but **does not download a
model**. `--verify-only` performs no downloads or installations.

## 3. Supply the checkpoint

If you already have the supported `tiny.en.pt`, keep it where it is and use its
full path in the transcription command below. The CLI verifies its checksum;
renaming another checkpoint does not make it compatible.

Otherwise, this explicit opt-in downloads `tiny.en` and runs a short CPU
transaction against the backend's included JFK recording:

```console
python tools/run_native_example.py --root .tmp-native-reuse --allow-model-download
```

The checkpoint is saved as `.tmp-native-reuse/models/tiny.en.pt`. Later runs of
that example omit `--allow-model-download` and use the verified cache. To use an
existing cache with the example, pass `--model-cache PATH/TO/DIRECTORY`; that
directory must contain `tiny.en.pt`. This helper is not a download-only command.

## 4. Transcribe a file

Switch to the environment created by bootstrap, which contains the native
dependencies and the installed command. In a POSIX shell:

```sh
. .tmp-native-reuse/venv/bin/activate
```

In Windows PowerShell:

```powershell
.\.tmp-native-reuse\venv\Scripts\Activate.ps1
```

Prepare uncompressed mono, 16 kHz, signed 16-bit PCM WAV. This command converts
the included recording; replace its input path with your own recording if needed:

```console
ffmpeg -n -i .tmp-native-reuse/backend/tests/jfk.flac -vn -ac 1 -ar 16000 -c:a pcm_s16le .tmp-native-reuse/input.wav
```

Already have a WAV in that format, or little-endian raw `.pcm`/`.s16le`? Use it
directly. The CLI does not invoke FFmpeg or convert input formats.

```console
whisper-runtime .tmp-native-reuse/input.wav --setup-manifest .tmp-native-reuse/manifest.json --model .tmp-native-reuse/models/tiny.en.pt --profile low-latency-v2 --txt .tmp-native-reuse/transcript.txt
```

For a checkpoint stored elsewhere, replace the `--model` path. CPU is the default.
The terminal shows provisional text, replacements, commits and verified final
completion. The TXT file contains committed text only and is written only after
successful completion and cleanup. Existing output files are not overwritten;
choose another filename when repeating the command. Add `--srt` or `--vtt` with
distinct paths for caption exports; their times describe committed audio spans,
not exact word timing.

## What to expect

`low-latency-v2` checks word agreement early and requires right context in both
observations. It does not promise a fixed caption delay or error-free text.
Omitting `--profile` still selects `conservative-v1`, whose confirmed text can
wait nearly 30 seconds during continuous speech.

File transcription runs as fast as the machine can process it. CPU is not
guaranteed to keep up with live audio. `--paced` is an optional source-clock test:
it fails on lateness or overload instead of slowing playback or dropping audio.

This bootstrap is a CPU setup. CUDA requires a separately provisioned compatible
CUDA PyTorch/Triton environment and NVIDIA driver; adding `--device cuda:0` does
not install them. Native Windows users should use CPU. CUDA startup also prepares
the CPU alignment backtrace before accepting audio; it does not warm Triton
kernels. See [CLI details](CLI.md#installation-and-verified-backend).

The supported local model is English `tiny.en`, FP32, one stream. Native endurance,
physical microphone reliability and clean-platform installation remain open
qualification work. There is no GUI, desktop installer or hosted service. Remote
use requires your own existing server and access controls.

See [CLI options and failure behavior](CLI.md), [release status](V01_STATUS.md),
and [backend setup details](REAL_BACKEND_QUICKSTART.md).
