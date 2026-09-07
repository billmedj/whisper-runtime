# Single-stream transcript CLI

`whisper-runtime` runs one alpha local or remote stream. Confirmed text requires
the SDK's agreement and audio-evidence checks; this is not an omission-free
recognizer or a production streaming guarantee. Local inference supports the
pinned `tiny.en` checkpoint, English transcription, FP32, on `cpu` or an available
`cuda:0`. For a first local run, follow [Getting started](GETTING_STARTED.md).

Use `--profile low-latency-v2` for earlier confirmed text. It starts word-agreement
checks without waiting for the late-window schedule and requires two seconds of
right context in both observations. Confirmation still depends on supported
audio and agreement, not a fixed timer. This profile needs the separate reuse
backend described below; it does not use token drafts.

Omitting `--profile` retains `conservative-v1`. That default keeps text provisional
until a quiet endpoint, EOF, or a reserved pair of boundary checks before the
30-second window fills. Previews update every two seconds, but without a pause
confirmed text can take roughly one window to become available. See the
[standard-policy comparison](research/2026-09-06-deferred-word-commits.md) and
[two-observation results](research/2026-09-07-two-observation-holdback.md).
Remote mode sends audio to an explicitly selected live-v2 server and requires no
local model, setup manifest, PyTorch, Whisper or NumPy.

## Installation and verified backend

The package exposes a real console-script entrypoint:

```console
python -m pip install .
whisper-runtime --help
```

Installation of the runtime itself does not install PyTorch, Whisper, NumPy or
microphone support. Help and argument parsing work without those dependencies.
Use the interpreter/environment containing the verified native prerequisites for
transcription. `python -m whisper_runtime.cli` is an equivalent entrypoint.
Bootstrap installs the runtime into its own environment: activate
`.tmp-native-reuse/venv` for a reuse profile, or `.tmp-native/venv` for Standard.
The separate core-only `.venv` does not acquire those native dependencies.

First prepare or verify the repository's native setup using
`tools/bootstrap_native_backend.py` and the instructions in
[NATIVE_ADAPTER.md](NATIVE_ADAPTER.md). The bootstrap can install dependencies and
fetch backend source; that is a separate, explicit setup operation. The CLI does
not run bootstrap, install packages, fetch models, or accept an arbitrary stock
Whisper installation.

Every local transcription requires the setup manifest and a local checkpoint path.
The installed factory checks the backend checkout identity and patch artifacts,
dependency versions, checkpoint SHA-256 and loaded model fingerprint before
inference. A checkpoint path is passed directly to the backend loader; model-name
download fallback is not used. Because local-path loading omits the named-model
alignment mask, the factory restores the verified backend's `tiny.en` mask before
moving the model to the requested device. This preserves the named model's legacy
alignment-head selection; the mask is not part of the weight-state fingerprint.
The initial release deliberately supports only the recorded `tiny.en` checkpoint,
not custom or user-trained weights.

Selecting `cuda:0` also requires the verified backend's CUDA alignment
prerequisites, including importable Triton. Its otherwise-lazy alignment module is
loaded under the same fresh-bytecode-cache guard before checkpoint loading; a
missing prerequisite fails early. Before creating the worker or accepting audio,
CUDA startup compiles the alignment backtrace's two CPU Numba signatures:
two-dimensional int32 arrays with contiguous or general strided layout. Compilation
failure prevents readiness. It runs no model decode, allocates no GPU tensors for
warm-up, and leaves Triton kernels lazy. Startup precedes the source-pacing clock;
it does not remove later kernel-compilation costs or change any lateness limit.
CPU skips both CUDA-specific preparation steps. Neither preparation nor READY
alone qualifies GPU performance or changes alignment numerics.

The repository bootstrap installs a CPU environment. CUDA requires separately
provisioned compatible PyTorch/CUDA, Triton and driver prerequisites; changing
`--device` does not install them. Native Windows uses the CPU path, not a bundled
Windows CUDA setup. See [current platform and release limits](V01_STATUS.md).

## File transcription

```console
whisper-runtime speech.wav --setup-manifest .tmp-native-reuse/manifest.json --model "PATH/TO/EXISTING/tiny.en.pt" --profile low-latency-v2 --txt speech.txt --srt speech.srt --vtt speech.vtt
```

Replace the model placeholder with the existing checkpoint's actual path; no
particular model-cache directory is assumed or created. Use the actual manifest
path printed by bootstrap. Input must be uncompressed,
mono, 16 kHz, signed 16-bit PCM WAV (`.wav`/`.wave`) or little-endian raw PCM
(`.pcm`/`.s16le`). No resampling, channel mixing, compressed decoding, or implicit
ffmpeg invocation is performed. Empty, incomplete and truncated inputs fail
explicitly. Files are read in 20 ms chunks, including a partial final chunk.

In local mode a file is offered as fast as the SDK can accept it. Backpressure keeps
and retries the exact rejected chunk without losing samples. Add `--paced` to
offer chunks on their real source clock instead. A source-clock delay above
250 ms or a full runtime input buffer stops paced mode with an error; it never
slows the clock or drops audio to claim success. This option is a local replay,
not a device-latency or GPU benchmark.

Stdout shows `[provisional]`, `[replace]`, `[commit]`, and `[final]` events with
revision numbers and source intervals. Redirecting stdout captures this event
display, not a clean transcript. Use `--txt` for committed transcript text.

## Remote WebSocket transcription

Install the optional client dependency and select an existing live-v2 endpoint:

```console
python -m pip install ".[remote]"
whisper-runtime speech.wav --server wss://YOUR-SERVER/ --header-env Authorization=WHISPER_AUTHORIZATION --txt speech.txt --srt speech.srt --vtt speech.vtt
```

Set `WHISPER_AUTHORIZATION` securely in the process environment before running;
its value is the full header value, such as the authentication scheme followed
by its credential. Do not put credentials in command arguments or URL paths.
`--header-env HEADER=ENV_NAME` is repeatable for gateways needing several headers
(for example, `Modal-Key` and `Modal-Secret`). The command receives variable names,
not credential values. Missing variables, header injection, duplicate header
names and reserved WebSocket/connection headers fail before connecting. The CLI
does not print headers, the server URL, or raw transport/server exceptions, and
rejects transcript text reflecting supplied credentials before displaying it.

`--server` excludes `--setup-manifest`, `--model`, `--device` and `--profile`.
The live-v2 protocol does not negotiate or attest the server's execution profile;
the client neither chooses nor claims to verify one. Even an explicit
`--profile conservative-v1` is rejected remotely. Configure the server separately.
Non-loopback
servers require `wss://`; `ws://` is accepted only for `localhost`, `127.0.0.1` or
`::1` testing. URL user/password fields, query strings and fragments are rejected.
The client uses normal TLS verification, does not follow redirects or reconnect,
and does not use environment proxy/netrc credentials. Server deployment and access
control are separate operations; this command does not create or configure a server.

Remote files are always paced in 20 ms chunks, beginning only after the server's
READY message. The same optional bounded microphone source can be selected with
`--microphone --duration 30`; install `.[remote,microphone]` for that combination.
Capture likewise starts only after READY. Both sources use the existing
`pcm-websocket/live-v2` protocol, with unknown length at START and actual sample
count/hash at EOF. No legacy replay-v1 fallback or audio retry is attempted.

Bounds remain explicit: at most one hour of audio, 3,700 seconds for the whole
connection, 90 seconds for READY, 10 seconds without source data, and 250 ms for
a send or output callback. `--drain-timeout` defaults to 30 seconds remotely and
may only be lowered. File-clock lateness above 250 ms, a blocked send, a source or
server overload, protocol errors, premature completion or failed cleanup aborts;
none is converted into a successful partial export. A real FINAL is necessary
but insufficient: verified DONE, EOF hash/count and full committed coverage are
also required. Remote failures are intentionally shown as fixed local error codes.
The CLI retains SDK FINAL internally but prints `[final] complete` only after
those remote completion checks pass. FINAL followed by invalid DONE or a
disconnect therefore prints neither completion nor success exports.

The installed transport lives in `whisper_runtime.remote_transport`; old
`examples.replay_websocket` imports remain compatibility aliases. The synchronous
audio source and stdout callback are bridged with Python's
[`asyncio.to_thread`](https://docs.python.org/3.10/library/asyncio-task.html#asyncio.to_thread)
so they do not block network receiving. The callback deadline bounds protocol
waiting, not process exit: if stdout blocks, timing out its coroutine cannot stop
the underlying thread, and executor shutdown may wait for that write to return.
OS file/device/terminal calls likewise stop cooperatively; Python cannot forcibly
terminate a stuck I/O thread. Source/callback deadlines are not hard wall-clock
guarantees for CLI cancellation or exit. Loopback
scripted-backend tests establish integration behavior, not WAN/GPU capacity or
physical microphone qualification.

## Optional microphone capture

```console
python -m pip install ".[microphone]"
whisper-runtime --microphone --duration 30 --setup-manifest .tmp-native/manifest.json --model "PATH/TO/EXISTING/tiny.en.pt" --txt recording.txt
```

Microphone support is lazy and optional: `sounddevice` plus a functioning
PortAudio installation and device capable of mono 16 kHz int16 capture are
required. `--input-device "device name"` selects a sounddevice name/query.
`--duration` is mandatory (greater than zero, at most one hour); completion of
that finite capture closes input and drains the SDK. Ctrl+C cancels rather than
synthesizing EOF or a final transcript.

Capture uses a bounded 64-frame queue (1.28 seconds at the 20 ms callback size).
Device status errors, capture-queue overflow, runtime-buffer overflow, a stopped
device, or five seconds without a callback abort explicitly. There is no silent
drop, restart, reconnect, or conversion. This Python callback path is experimental,
not a hard-real-time implementation. Little-endian hosts are currently required.
Physical-device behavior has to be verified on the deployment machine; scripted
tests do not establish microphone reliability.

The capture API and error semantics follow the primary
[sounddevice raw-stream documentation](https://python-sounddevice.readthedocs.io/en/0.5.3/api/raw-streams.html),
[callback status documentation](https://python-sounddevice.readthedocs.io/en/0.5.3/api/misc.html),
and [stream lifecycle documentation](https://python-sounddevice.readthedocs.io/en/0.5.3/api/streams.html).

## Final exports and failures

TXT/SRT/VTT are created only after a real SDK `FINAL`, exact committed revisions,
full offered/accepted/committed sample coverage, and cleanup (plus verified DONE
and EOF hash/count remotely). Preview text
never appears in a final export. Empty committed silence advances coverage but
does not create an empty subtitle cue. SRT/VTT use one cue per nonempty commit;
their times describe committed source coverage, not acoustic word boundaries.
There is no heuristic cue splitting or line-length/reading-speed optimization.

The CLI prints `[final] complete` only after the driver and cleanup succeed.
An SDK FINAL alone does not confirm successful completion of the command.

Existing paths are never overwritten, and the CLI does not create missing parent
directories. Choose distinct output paths. Creation is exclusive even if a file
appears after preflight. Multiple exports are not one atomic filesystem transaction:
an I/O failure or interruption during writing can leave new files partial or some
exports complete. The error reports that possibility; existing files stay intact.

Exit status is `0` only on successful completion, `1` on input, native-policy,
network, execution, cleanup or export failure, `2` for argparse usage errors, and `130` for
Ctrl+C cancellation unless cleanup itself fails. An unresolved SDK boundary stays
an error; a prompt is not automatically injected and acceptance rules are not
weakened. On pre-export failure/cancellation no final files are generated; printed
commits are still committed history, not proof of complete recognition.

Locally, `--drain-timeout` defaults to 90 seconds. Its clock starts when the
source ends, before the driver publishes EOF to the stream. Cancellation and
deadlines are checked again after native work and before reporting completion.
They remain cooperative: a blocking native call cannot be forcibly preempted
by this CLI. It does not offer microphone recovery, resume/checkpoint
controls, multiple concurrent streams, diarization,
translation, or automatic model choice.

## Versioned local execution profiles

`--profile` selects an exact, versioned local configuration. Omitting it keeps
`conservative-v1` and all existing conservative defaults. Profile names are stable
identifiers, not model names or a promise of qualification; the runtime remains
experimental. A changed registered definition requires a new profile version.

| Profile | Display label | Left / retained word context | Alignment feature reuse | Draft token limit |
| --- | --- | --- | --- | --- |
| `conservative-v1` (default) | Standard | 2 / 6 seconds | Off | 0 |
| `low-latency-v1` | Low latency (experimental) | 20 / 24 seconds | On, same window only | 0 |
| `low-latency-v2` (recommended for earlier confirmation) | Low latency v2 (experimental) | 20 / 24 seconds | On, same window only | 0 |
| `experimental-optimized-v1` | Optimized (experimental) | 20 / 24 seconds | On, same window only | 32 |

All choices keep English `tiny.en`, FP32, one inference lane, the same decode
options and RNG seed, 2-second previews/holdback, 30-second maximum window,
40-second input buffer, endpoint settings, source-unit/evidence checks,
word-boundary fallback and EOF context retry. The low-latency profiles start word
agreement checks without waiting for the late-window schedule. Version 2 also
requires two seconds of right context in the earlier observation before a word
can be confirmed. Version 1 checks that holdback only in the later observation.
Standard and Optimized defer those checks. Publication still needs supported
audio and matching hypotheses; no clock alone can authorize text. The
optimized choice does not relax publication checks, change draft thresholds, or
reuse an encoder result across a changed audio window. Its name does not claim
universal speedup, accuracy, low final-caption latency, or sustained live capacity.
See the [composition experiment](research/2026-09-06-composed-inference.md) and
[integrated draft results](research/2026-09-07-integrated-draft-gpu-results.md).

```console
whisper-runtime speech.wav --setup-manifest PATH/TO/REUSE/manifest.json --model PATH/TO/tiny.en.pt --profile low-latency-v2 --txt speech.txt
```

The standard profile requires the original pinned backend tree
`c011d2563c26763b5f147026e6b18ef85bccd4fb`. The feature-reuse profiles require the
separate alignment-reuse tree `32163d5cdb87babc1cd415a86cc5a58116c86a16`.
Each rejects the other tree, even when the checkpoint is correct. Both check the
manifest, clean checkout, actual tree, dependency versions and checkpoint before
inference. The command never applies patches or downloads a model. Unknown names
are usage errors, with no fallback to another profile. Profiles are local only;
there is no corresponding live-v2 server-selection field.

Prepare the separate feature-reuse backend explicitly:

```console
python tools/bootstrap_native_backend.py --alignment-feature-reuse
python tools/bootstrap_native_backend.py --alignment-feature-reuse --verify-only
```

This uses `.tmp-native-reuse` by default and leaves `.tmp-native` unchanged. Use
the resulting `.tmp-native-reuse/manifest.json` with a feature-reuse profile.
The opt-in verifies and applies the pinned
[alignment patch](../patches/openai-whisper/experimental/README.md), then checks
the clean backend tree. The ordinary bootstrap still uses seven patches; the
opt-in uses eight. Setup can fetch source and install dependencies. Verification
does neither. A matching local checkpoint is still required separately.

### Python catalog and factory

The installed catalog is usable without Torch, NumPy, Whisper or device access:

```python
from pathlib import Path
from whisper_runtime.native_setup import create_stream
from whisper_runtime.profiles import DEFAULT_PROFILE, get_profile, list_profiles

choices = list_profiles()  # Immutable tuple of frozen ExecutionProfile records.
standard = get_profile(DEFAULT_PROFILE)
assert standard.name == "conservative-v1"

stream = create_stream(
    manifest=Path("PATH/TO/manifest.json"),
    model=Path("PATH/TO/tiny.en.pt"),
    profile=standard.name,
    device="cpu",
)
try:
    # Feed PCM and drive the stream using the existing SDK contract.
    ...
finally:
    stream.close()
```

Records expose `name`, `label`, `experimental`, `native_profile_id`,
`stream_config`, and `reuse_alignment_features`. The tuple, records, stream
configurations and nested endpoint configuration are immutable; there is no
registration API or implicit `latest` alias. `get_profile` accepts an exact name
and raises `ValueError` for unknown names or `TypeError` for non-string names.
The factory accepts names rather than caller-created profile records. These are
configuration identities, not cryptographic attestation of a running server or
of the installed runtime source. `experimental=False` for Standard only means it
is the conservative baseline, not that the release is production-qualified.

For compatibility, omitting `profile` in the Python factory still permits
`config=ContinuousStreamConfig(...)` and `reuse_alignment_features=True` for
custom experiments. Explicit named profiles cannot be combined with a custom
config or enabled reuse override; such combinations fail before backend setup.
Legacy custom calls retain their existing native profile ID, so that ID alone
does not attest custom settings. The named standard profile preserves the old
`tiny.en/cli-fp32-v1` native identity; the optimized choice has the distinct
`tiny.en/cli-experimental-optimized-fp32-v1` identity. `CLI_STREAM_CONFIG` remains
a compatibility alias of the standard stream config. `stream.profile_id`
continues to describe the SDK publication policy, not this execution selection.
