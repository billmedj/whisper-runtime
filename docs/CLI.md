# Single-stream transcript CLI

`whisper-runtime` transcribes one stream locally or through a live-v2 server.
Local mode uses the pinned English `tiny.en` model in FP32 on `cpu` or `cuda:0`.
For your first run, see [Getting started](GETTING_STARTED.md).

Choose `--profile low-latency-v2` for earlier confirmed text. Omitting it selects
`conservative-v1`, which can wait nearly 30 seconds to confirm continuous speech.
Both check agreement and audio evidence; neither guarantees correct recognition
or a fixed caption delay. See [profiles](#versioned-local-execution-profiles).

## Installation and verified backend

From the repository checkout:

```console
python -m pip install .
whisper-runtime --help
```

The package and help command need no ML dependencies.
`python -m whisper_runtime.cli` runs the same command.

For local transcription, use `tools/bootstrap_native_backend.py` as described in
[Getting started](GETTING_STARTED.md). It installs the runtime and dependencies
into its own environment. Activate `.tmp-native-reuse/venv` for reuse profiles,
or `.tmp-native/venv` for Standard—not the core-only `.venv`.
The CLI does not bootstrap, install packages or download models. Stock Whisper
and custom weights are not supported.

Supply a setup manifest and an existing checkpoint. Before inference, the
factory verifies the backend checkout, dependency versions, checkpoint SHA-256
and model fingerprint. It restores the `tiny.en` alignment-head mask when loading
by path; that mask is separate from the weight fingerprint.

Bootstrap installs CPU dependencies. CUDA needs compatible CUDA PyTorch, Triton
and an NVIDIA driver supplied separately. Use CPU on native Windows; changing
`--device` does not set up CUDA.

CUDA startup imports the alignment module under the fresh-bytecode guard before
loading the checkpoint. It then compiles the CPU backtrace for contiguous and
strided 2-D int32 arrays before creating a worker or accepting audio. An import
or compile failure stops startup. Backtrace compilation runs no model decode
and allocates no GPU tensors; Triton kernels stay lazy. CPU skips these
CUDA-specific steps. Startup is outside the source-pacing clock; later kernel
compilation still counts against the unchanged timing limits. See
[platform status](V01_STATUS.md).

## File transcription

```console
whisper-runtime speech.wav --setup-manifest .tmp-native-reuse/manifest.json --model "PATH/TO/EXISTING/tiny.en.pt" --profile low-latency-v2 --txt speech.txt --srt speech.srt --vtt speech.vtt
```

Use the manifest path printed by bootstrap and your checkpoint's path.
Input must be uncompressed mono, 16 kHz, signed 16-bit PCM WAV (`.wav`/`.wave`)
or little-endian raw PCM (`.pcm`/`.s16le`). The CLI does not resample, mix channels,
decode compressed formats or run FFmpeg. Empty or truncated input is rejected.
Reads use 20 ms chunks, including a partial final chunk.

Local files run as fast as the SDK can accept them; a full buffer delays and
retries the same chunk. With `--paced`, chunks follow the source clock instead.
Lateness above 250 ms or a full buffer stops paced mode. Audio is not dropped,
and pacing is not slowed to accommodate decoding. CPU may not keep up in this mode.

Stdout shows `[provisional]`, `[replace]`, `[commit]` and `[final]` events with
revisions and source intervals. Use `--txt` for a transcript; redirected stdout
contains the event display.

## Remote WebSocket transcription

Remote mode needs an existing live-v2 server and the client dependency, but no
local model or ML packages:

```console
python -m pip install ".[remote]"
whisper-runtime speech.wav --server wss://YOUR-SERVER/ --header-env Authorization=WHISPER_AUTHORIZATION --txt speech.txt --srt speech.srt --vtt speech.vtt
```

Set `WHISPER_AUTHORIZATION` in the environment to the full header value, including
its authentication scheme. Pass variable names, never credentials, on the command
line. Repeat `--header-env HEADER=ENV_NAME` for additional headers. Missing
variables, injected newlines, duplicate names and reserved connection headers are
rejected before connecting. The CLI hides headers, the server URL and raw remote
exceptions. It also rejects transcript text that repeats supplied credentials.

`--server` cannot be combined with `--setup-manifest`, `--model`, `--device` or
`--profile`. Configure the server's profile separately; the client cannot verify
it. You must also provide the server and its access controls.

Use `wss://` except for loopback tests: `ws://` allows only `localhost`,
`127.0.0.1` or `::1`. URL credentials, query strings and fragments are rejected.
TLS verification stays enabled. The client does not follow redirects, reconnect
or use environment proxy/netrc credentials.

Files are paced in 20 ms chunks after READY. For a microphone, install
`.[remote,microphone]` and use `--microphone --duration 30`; capture also starts
after READY. Both use `pcm-websocket/live-v2`, sending the sample count and hash
at EOF. There is no replay-v1 fallback or audio retry.

Limits are one hour of audio, 3,700 seconds per connection, 90 seconds for READY,
10 seconds without source data, and 250 ms per send or output callback.
Remote `--drain-timeout` defaults to 30 seconds and may only be lowered.
File-clock lateness above 250 ms, overload, blocked sends, protocol errors,
premature completion or failed cleanup stop the run. Errors use fixed local
codes, not raw server messages. FINAL alone cannot produce successful exports;
see [completion checks](#final-exports-and-failures).

Audio reads and output callbacks run through
[`asyncio.to_thread`](https://docs.python.org/3.10/library/asyncio-task.html#asyncio.to_thread)
to keep network receiving responsive. Timing out a callback does not kill its
thread: blocked stdout or device I/O can still delay process exit.

## Optional microphone capture

```console
python -m pip install ".[microphone]"
whisper-runtime --microphone --duration 30 --setup-manifest .tmp-native/manifest.json --model "PATH/TO/EXISTING/tiny.en.pt" --txt recording.txt
```

This example uses Standard and its `.tmp-native` setup. Capture needs
`sounddevice`, PortAudio and a device that supports mono 16 kHz int16 input on a
little-endian host. Select a device with `--input-device "device name"`.
`--duration` is required: greater than zero and at most one hour. At the end,
capture closes input and drains the stream. Ctrl+C cancels; it does not create EOF.

The queue holds 64 frames of 20 ms (1.28 seconds). Device errors, either input
buffer overflowing, a stopped device or five seconds without a callback stop
capture. It does not drop audio, restart or reconnect. Physical microphones still
need testing on the deployment machine; this is not a hard-real-time path.

See sounddevice's [raw streams](https://python-sounddevice.readthedocs.io/en/0.5.3/api/raw-streams.html),
[callback status](https://python-sounddevice.readthedocs.io/en/0.5.3/api/misc.html)
and [stream lifecycle](https://python-sounddevice.readthedocs.io/en/0.5.3/api/streams.html).

## Final exports and failures

Completion requires SDK FINAL, valid committed revisions, full sample coverage
from input through commit, and successful cleanup. Remote runs also need valid
DONE and matching EOF counts/hash. Only then does the CLI print `[final] complete`
and write exports. A disconnect or invalid DONE after FINAL is still a failure.

Exports contain no preview text. SRT/VTT create one cue per nonempty commit;
silence advances coverage without an empty cue. Times describe committed audio,
not word boundaries. Cue splitting and reading-speed formatting are not provided.

Choose distinct paths in existing directories. Output creation never overwrites
a file, even one created after preflight. The files are not written as one atomic
group: an export error can leave new files incomplete or only some formats saved.
Failures before export create no final files.

Exit codes: `0` for success, `1` for run or export failure, `2` for invalid
arguments, and `130` for Ctrl+C unless cleanup fails. Unresolved boundaries remain
errors; the CLI does not inject prompts or relax publication checks to finish.
Previously printed commits do not prove that the whole recording was transcribed.

Local `--drain-timeout` defaults to 90 seconds, starting when the source ends,
before EOF is sent to the stream. Cancellation and deadlines are checked after
native work and before completion. Blocking native calls and I/O cannot be
forcibly interrupted, so these deadlines do not guarantee process exit time.
The CLI has no microphone recovery, resume/checkpoint controls, concurrent
streams, diarization, translation or automatic model selection.

## Versioned local execution profiles

`--profile` selects a versioned local configuration, not a model.
The default remains `conservative-v1`. Changed settings require a new profile
version; unknown names are errors, with no fallback.

| Profile | Display label | Left / retained word context | Alignment feature reuse | Draft token limit |
| --- | --- | --- | --- | --- |
| `conservative-v1` (default) | Standard | 2 / 6 seconds | Off | 0 |
| `low-latency-v1` | Low latency (experimental) | 20 / 24 seconds | On, same window only | 0 |
| `low-latency-v2` (recommended for earlier confirmation) | Low latency v2 (experimental) | 20 / 24 seconds | On, same window only | 0 |
| `experimental-optimized-v1` | Optimized (experimental) | 20 / 24 seconds | On, same window only | 32 |

All profiles use one inference lane, the same decode options and RNG seed,
2-second preview intervals and holdback, a 30-second window and a 40-second input
buffer. Endpoint, audio-evidence, word-boundary and EOF-retry checks are shared.

Low-latency profiles start agreement checks early. During nonfinal agreement,
version 2 requires two seconds of audio after a word in both observations;
version 1 applies that holdback only to the later one. When a source unit closes
or the stream ends, these holdbacks do not apply; word anchors are still checked.
Standard and Optimized defer checks until a quiet endpoint,
EOF or the reserved checks near the window limit. Reuse stays within one audio
window. Optimized enables token drafts without weakening their acceptance or
publication checks.

See the [standard-policy comparison](research/2026-09-06-deferred-word-commits.md),
[two-observation results](research/2026-09-07-two-observation-holdback.md),
[reuse experiment](research/2026-09-06-composed-inference.md) and
[draft results](research/2026-09-07-integrated-draft-gpu-results.md).

```console
whisper-runtime speech.wav --setup-manifest PATH/TO/REUSE/manifest.json --model PATH/TO/tiny.en.pt --profile low-latency-v2 --txt speech.txt
```

Standard requires backend tree `c011d2563c26763b5f147026e6b18ef85bccd4fb`;
reuse profiles require `32163d5cdb87babc1cd415a86cc5a58116c86a16`. Each rejects
the other tree, even with the right checkpoint. Prepare the reuse backend with:

```console
python tools/bootstrap_native_backend.py --alignment-feature-reuse
python tools/bootstrap_native_backend.py --alignment-feature-reuse --verify-only
```

This creates `.tmp-native-reuse` without changing `.tmp-native`. Use its
`manifest.json` with a reuse profile. Bootstrap checks and applies the
[alignment patch](../patches/openai-whisper/experimental/README.md): eight patches
instead of Standard's seven. Setup can fetch source and install dependencies;
verification does neither. The checkpoint is still supplied separately.

### Python catalog and factory

The catalog needs no ML imports. Calling `create_stream` does need the native
environment, manifest and model:

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

Records expose `name`, `label`, `experimental`, `native_profile_id`, `stream_config`
and `reuse_alignment_features`. The catalog and nested settings are immutable;
there is no registration API or `latest` alias. `get_profile` raises `ValueError`
for an unknown name and `TypeError` for a non-string. The factory takes names,
not profile records. These IDs do not verify installed source or a remote server.
Standard's `experimental=False` labels the baseline, not production readiness.

For custom experiments, omit `profile` and pass `config=ContinuousStreamConfig(...)`
or `reuse_alignment_features=True`. A named profile cannot be combined with either
override; rejection happens before setup. Custom calls keep the legacy native
profile ID, so that ID alone does not describe their settings.

Standard keeps `tiny.en/cli-fp32-v1`; Optimized uses
`tiny.en/cli-experimental-optimized-fp32-v1`. `CLI_STREAM_CONFIG` remains an alias
for the standard config. `stream.profile_id` describes the publication policy,
not the execution selection.
