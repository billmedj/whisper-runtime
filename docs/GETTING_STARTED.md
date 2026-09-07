# Getting started

The CLI transcribes one English audio stream. This guide uses CPU and
`low-latency-v2`.

## 1. Try the core without a model

Use CPython 3.12 or 3.13 for transcription. The core alone supports Python 3.10+.

```console
git clone https://github.com/billmedj/whisper-runtime.git
cd whisper-runtime
git checkout v0.1.0a1
python -m venv .venv
```

Activate the environment on macOS/Linux:

```sh
. .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install and run the example:

```console
python -m pip install .
python examples/minimal_transaction.py
whisper-runtime --help
```

A successful run prints `Example transcript`. No model is loaded.

## 2. Prepare the native environment

Stay at the repository root. Install Git and FFmpeg on `PATH` first. Setup needs
network access and a clean checkout.

```console
python tools/bootstrap_native_backend.py --alignment-feature-reuse
python tools/bootstrap_native_backend.py --alignment-feature-reuse --verify-only
```

This creates `.tmp-native-reuse/` with the patched Whisper source, a Python
environment, the installed command and `manifest.json`. `low-latency-v2` needs
this reuse backend, not the default `.tmp-native` setup or stock `openai-whisper`.
No model is downloaded. `--verify-only` checks the setup without installing or
downloading anything.

## 3. Supply the checkpoint

Already have the supported `tiny.en.pt`? Use its path in step 4. Otherwise, this
command downloads it and runs a short CPU test on the included JFK recording:

```console
python tools/run_native_example.py --root .tmp-native-reuse --allow-model-download
```

The model is saved at `.tmp-native-reuse/models/tiny.en.pt`. Omit
`--allow-model-download` on later runs. For another cache directory, use
`--model-cache PATH/TO/DIRECTORY`; it must contain `tiny.en.pt`.

## 4. Transcribe a file

Switch to the native environment. On macOS/Linux:

```sh
. .tmp-native-reuse/venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.tmp-native-reuse\venv\Scripts\Activate.ps1
```

Convert the recording to mono, 16 kHz, signed 16-bit PCM WAV. Replace the input
path to use your own audio:

```console
ffmpeg -n -i .tmp-native-reuse/backend/tests/jfk.flac -vn -ac 1 -ar 16000 -c:a pcm_s16le .tmp-native-reuse/input.wav
```

Skip conversion if your WAV already has that format, or use little-endian raw
`.pcm`/`.s16le`. The CLI does not convert audio itself.

```console
whisper-runtime .tmp-native-reuse/input.wav --setup-manifest .tmp-native-reuse/manifest.json --model .tmp-native-reuse/models/tiny.en.pt --profile low-latency-v2 --txt .tmp-native-reuse/transcript.txt
```

Replace `--model` if your checkpoint is elsewhere. The terminal shows changing
previews and confirmed text. TXT is written only after the run and cleanup
succeed; existing files are never overwritten. Add `--srt` or `--vtt` with other
output paths for captions. Caption times cover committed audio, not individual words.

## What to expect

`low-latency-v2` starts confirmation checks earlier. Recognition can still be
wrong, and timing varies. Omitting `--profile` selects `conservative-v1`, which
can wait nearly 30 seconds to confirm continuous speech.

Files run as fast as the machine can process them. CPU may not keep up with live
audio. Add `--paced` to test real-time input; lateness or overload stops the run.

Bootstrap installs a CPU setup. CUDA needs a compatible PyTorch/Triton environment
and NVIDIA driver; `--device cuda:0` does not install them. Use CPU on native Windows.

This alpha supports one English `tiny.en` stream in FP32. Long runs, microphones
and fresh-machine installs need more testing. There is no GUI or hosted service.

See [CLI options and failure behavior](CLI.md), [release status](V01_STATUS.md),
and [backend setup details](REAL_BACKEND_QUICKSTART.md).
