# Whisper Execution Runtime

Continuous transcription with explicit control over inference.

[![CI](https://github.com/billmedj/whisper-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/billmedj/whisper-runtime/actions/workflows/ci.yml)
[![Native integration](https://github.com/billmedj/whisper-runtime/actions/workflows/native-integration.yml/badge.svg)](https://github.com/billmedj/whisper-runtime/actions/workflows/native-integration.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20AND%20MIT-536879)](#license)

Feed audio in chunks. Read provisional text as it changes. Export confirmed text.
Use the Python API to control admission, decoder steps, cancellation and cleanup.

This project adds a runtime and a reproducible patch series to
[OpenAI Whisper](https://github.com/openai/whisper). It uses the existing model
weights. It is an independent project, not an OpenAI product.

**0.1.0a1 is a developer alpha.** The supported transcription path is one English
stream with the pinned `tiny.en` model. There is a Python package and a command,
but no desktop GUI. See the [alpha notes](docs/releases/0.1.0a1.md) for scope.

[Get started](docs/GETTING_STARTED.md) · [Command reference](docs/CLI.md) ·
[Results](#measured-results) · [Architecture](#how-it-works) ·
[Roadmap](docs/ROADMAP.md)

![Audio enters a bounded buffer, Whisper proposes text, and publication checks separate revisable text from confirmed output.](docs/assets/transcription-flow.svg)

## Try it

### Explore the runtime without a model

The core example needs no GPU, model download or API key.

```sh
git clone https://github.com/billmedj/whisper-runtime.git
cd whisper-runtime
git checkout v0.1.0a1
python -m venv .venv
```

Activate the environment, then install and run:

```text
macOS / Linux:      . .venv/bin/activate
Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

```sh
python -m pip install .
python examples/minimal_transaction.py
```

This scripted example commits `Example transcript` and returns its reserved
capacity. It demonstrates the runtime contract; it does not transcribe audio.

### Transcribe audio

Follow [Get started](docs/GETTING_STARTED.md) to install the verified native
backend, obtain the model and run your first file. Native inference requires
Python 3.12 or 3.13; the runtime core supports Python 3.10 and later.

The low-latency example uses this explicit selection:

```sh
whisper-runtime speech.wav --setup-manifest .tmp-native-reuse/manifest.json --model PATH/TO/tiny.en.pt --profile low-latency-v2 --txt speech.txt --srt speech.srt --vtt speech.vtt
```

Run it in the backend's Python environment after setup. Input is mono 16 kHz
16-bit PCM WAV or raw PCM. The command does not convert arbitrary media or
download a model. CPU operation is supported, but real-time CPU performance is
not guaranteed.

**Select `low-latency-v2` to try the tested earlier-confirmation behavior.**
The unchanged default, `conservative-v1`, can wait roughly one 30-second window
before confirming uninterrupted speech.

## What you can use

| Capability | In this alpha |
| --- | --- |
| Progressive transcription | Provisional text, revisions, confirmed prefixes and a final completion event |
| Input and exports | Mono WAV/PCM; committed TXT, SRT and VTT |
| Local or remote execution | Verified local backend, or a client for an explicitly configured authenticated WebSocket server |
| Inference control | Bounded admission, stepwise decoding, cooperative cancellation and cleanup fences |
| State and recovery | Versioned commits; logical savepoints that recompute decoder state when restored |
| Compute reuse | Same-window encoder features reused for alignment in the low-latency profile |
| Verification | Automated tests, recorded native runs, reproducible patches and an abstract Lean model |

The optional microphone path is implemented but has no physical-device
qualification. Translation, multi-channel product support and a desktop
application are outside this alpha. Experimental token drafts are separate
from the low-latency profile.

## Measured results

The latest registered test ran the same **46.55-second** speech/noise/silence
input twice on one Modal T4, using `tiny.en`, FP32 and `low-latency-v2`.

| Measurement | First session | Second session |
| --- | ---: | ---: |
| First confirmed text after audio starts | 7.358 s | 6.186 s |
| Largest gap between confirmed-text updates | 7.582 s | 7.679 s |
| Reference word edits | 7 / 114 | 7 / 114 |
| Complete input coverage and final event | Pass | Pass |
| Native analyses | 25 | 25 |

Both sessions produced identical TXT/SRT/VTT exports and returned runtime
capacity. Terminal PyTorch CUDA allocation was identical across sessions.
The provider reported the app stopped with zero tasks.

These are short, repeated-input results. They do not establish per-word
latency, four-hour endurance, general accuracy or performance on other hardware.
Timing starts after initialization and excludes deployment, network transport
and display.

Read the [result and limits](docs/research/2026-09-07-low-latency-v2-gpu-results.md)
or inspect the [exact evidence archive](evidence/modal-low-latency-v2-2026-09-07.zip).
The [evidence index](evidence/README.md) also retains earlier failures.

## How it works

There are two boundaries.

**Audio to text.** The stream retains bounded audio context while the backend
produces hypotheses. The low-latency policy checks agreement across observations,
word timing and audio evidence before extending the confirmed prefix. New
context can revise provisional text, but cannot rewrite confirmed text. An
unresolved boundary can stop the stream; confirmation is not a guarantee that
recognition is correct.

**Request to execution.** Each decode has an owner, a declared resource lease
and a transaction. Cleanup must close further submissions and establish that
registered backend work has completed before capacity can be returned.

![A request reserves capacity, runs backend work, closes submissions and waits for a completion fence. Valid results can then be committed. An unproven completion keeps the lease quarantined.](docs/assets/execution-lifecycle.svg)

The resource ledger limits admitted work; it is not an operating-system RAM
limit or a hardware sandbox. Cancellation is cooperative. A logical savepoint
is not a serialized GPU cache and does not provide transparent GPU migration.
The Lean proofs cover an abstract lifecycle model, not the entire Python,
PyTorch or CUDA implementation.

- [Runtime architecture](docs/rfcs/0001-state-resource-execution.md)
- [Native adapter and run handles](docs/NATIVE_ADAPTER.md)
- [Streaming and publication rules](docs/CONTINUOUS_STREAMING.md)
- [Savepoints and their limits](docs/CHECKPOINTS.md)
- [Formal assurance map](docs/ASSURANCE.md)

## What remains before stable release

The four-hour native endurance gate is still open. Clean-platform installation,
physical microphone capture and broader audio coverage need verification.
Neither the alpha label nor the passing short test closes those gates.

Report reproducible failures through [GitHub issues](https://github.com/billmedj/whisper-runtime/issues).
Include the version, selected profile, device, input format and error.
Share only audio you have permission to publish. See
[release status](docs/V01_STATUS.md) and [contributing](CONTRIBUTING.md).

## Find your way around

| Directory | Contents |
| --- | --- |
| `src/whisper_runtime/` | Engine, adapters, command and remote client |
| `examples/` | Small runnable integrations |
| `tests/` and `tools/` | Runtime tests, validation and packaging tools |
| `patches/openai-whisper/` | Pinned backend patches with digests |
| `formal/lean/` | Abstract lifecycle and resource proofs |
| `docs/` | Guides, contracts, results and roadmap |
| `evidence/` | Reviewed records and self-contained experiment archives |
| `infra/` | Explicitly authorized remote-test harnesses |

For development checks, see [CONTRIBUTING.md](CONTRIBUTING.md).
Upstream changes are proposed independently; the bundled patches are not
represented as merged into Whisper. See [upstream contributions](docs/UPSTREAM.md).

## License

Original runtime code is [Apache-2.0](LICENSE). Patches derived from OpenAI
Whisper retain its [MIT license](patches/openai-whisper/LICENSE).
The combined package license expression is `Apache-2.0 AND MIT`.

See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and audio
provenance, [CITATION.cff](CITATION.cff) for citation, and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.
