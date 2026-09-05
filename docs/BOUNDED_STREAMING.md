# Bounded native transcript previews

`NativeTranscriptStream` in
[`native_stream.py`](../src/whisper_runtime/adapters/native_stream.py) implements
the profile `bounded_prefix_preview/v1`. It accepts one bounded PCM slice,
decodes growing prefixes, and emits revision events. Text stays provisional
until a successful end-of-input decode. This is not continuous streaming or
Local Agreement.

## Input and scheduling

Input must be raw mono, 16 kHz, signed 16-bit little-endian PCM (`s16le`), not a
WAV file. The caller supplies `mel_builder(bytes)`, which must return a mel
tensor accepted by the configured native adapter. The stream does not resample,
validate audio provenance, or claim equivalence to offline transcription.

`NativeStreamConfig` defaults to a 1,000 ms preview interval and a 30,000 ms
audio limit. Both can be reduced; the interval must not exceed the limit. The
sample rate is fixed at 16,000 Hz. At the maximum, admitted PCM payload is
480,000 samples, or 960,000 bytes. This is an input bound, not a bound on total
process or device memory.

`push(sequence_number, pcm_s16le)` accepts nonempty `bytes` containing complete
16-bit samples. Chunk numbers start at zero and must increase by one. It returns
the cumulative accepted sample count and does no decoding. Invalid sequences
raise `AudioSequenceError`; a chunk exceeding the limit raises
`AudioBufferFullError`. Rejection leaves both audio and the expected sequence
unchanged. The whole chunk is rejected, not truncated.

Preview endpoints are multiples of the configured interval, measured from the
start of the slice. Each preview reprocesses the entire prefix from sample zero.
There is no rolling buffer or eviction: completing a preview does not make room
for more input. The caller must stop ingestion at the bound or end this slice.

## Driving the stream

Each `step()` returns a tuple of events and performs at most one of these
operations:

- Start a scheduled window, including preprocessing and native startup.
- Advance its active native run by at most one decoder token step.
- Finalize and publish a completed run.

Startup and token steps normally return an empty tuple. `ready` says whether a
step can make progress; `active` says whether the stream holds a native run.
`step()` is a synchronous call, not a latency bound. It does not hide a full
token-generation loop inside audio ingestion.

This example assumes `adapter`, `mel_builder`, and `pcm_chunks` already exist:

```python
from whisper_runtime.adapters import NativeTranscriptStream

with NativeTranscriptStream(
    adapter, stream_id="clip-1", mel_builder=mel_builder
) as stream:
    for sequence, pcm in enumerate(pcm_chunks):
        stream.push(sequence, pcm)
        while stream.ready:
            for event in stream.step():
                print(event)
    stream.finish_input()
    while stream.ready:
        for event in stream.step():
            print(event)
```

The loops above drain available work. A scheduler can instead call `step()` once
per turn and do other work between calls. Use one owner thread for ingestion,
stepping, and closing. Only `cancel_active()` and `stop_active()` support calls
from another thread through the native run controls.

`finish_input()` marks EOF once and rejects subsequent pushes. It does not
decode or publish by itself. Continue stepping until `done`. An already active
preview finishes first; the next window uses the complete accepted slice and
skips other pending preview endpoints. A final decode is still required when
the last preview already covered the same endpoint. An empty stream emits only
`final`, with session version zero.

## Event contract

`TranscriptEvent` instances are immutable. Event sequence numbers start at one.
There is one segment, named `<stream_id>:segment-0`, whose source span is
`[0, end_sample)`. These are input-window boundaries, not word or token times.
`start_ms` and `end_ms` preserve fractional milliseconds from sample coordinates.

- `provisional` introduces revision one with its text and source span.
- `replace` publishes a new revision and names the preceding revision in
  `supersedes_revision`. A changed span creates a revision even when the text
  is unchanged. Identical text and span do not create a duplicate revision.
- `commit` identifies the existing revision to freeze; it does not repeat its
  text. Resolve it using the stored `(segment_id, revision)` payload. It carries
  the committed source watermark and session version.
- `final` is a terminal marker containing the session version, not transcript
  text or a source span.

A revision identifies both text and span; consumers must not change either in
place. Even an empty hypothesis gets a revision before it can be committed.
Successful final publication emits any required hypothesis event, then
`commit`, then `final`. Preview publication updates runtime session state but
does not advance its committed-prefix watermark. A session version alone does
not mean the text is committed.

The caller owns delivery and storage of returned events. This in-process API
does not retain an event log or implement reconnection and replay after a
sequence number.

The runtime watermark uses whole milliseconds. At EOF, `end_sample` retains
the exact endpoint, while `committed_through_ms` is rounded down and
`committed_through_sample` corresponds to that whole-millisecond boundary.
Any fractional-millisecond tail remains inside the final source span; the
transaction watermark makes no finer-grained claim.

## Pause, cancellation, and cleanup

Withholding `step()` pauses token submission without restarting the active
decoder. The native transaction deadline still runs. This is not an indefinite
pause or a TTL renewal, and the paused run continues to own capacity.

`cancel_active()` requests cooperative cancellation. `stop_active()` delegates
fencing and recovery to the native run, including recovery when its owner has
departed. Their boolean return values report a change or delivered signal, not
proof that all resources have been released. Cancellation does not end input
or emit `final`; after cleanup, later steps can retry the same scheduled prefix.

Use the context manager, or call `close()` from the owner thread, to abandon an
unfinished stream. Closing is not successful EOF publication and emits no final
transcript event. Do not rely on garbage collection for native cleanup.

The stream controls have two limits:

- `mel_builder` runs before native admission. Native startup then performs
  encoder preparation and decoder prefill before returning the run handle.
  The stream has no active handle during either stage, so its active cancel
  and stop methods cannot reach that work. Preprocessing and startup are not
  subdivided into stream steps.
- A closed or failed run can still retain worker capacity if fencing or
  cleanup has not completed safely. The stream retains that handle instead
  of starting another run; stepping a closed retained run raises
  `NativeStreamError`. Follow the native runtime's retained-transaction
  recovery path. Do not treat `closed`, an exception, or a stop signal as
  evidence of resource release.

If a retained-transaction error contains an already committed result, the
stream keeps it. After recovery, the next `step()` emits its events without
repeating inference. Call `close()` to abandon delivery instead. The native
session remains available through `state`.

## Measurements and remaining work

### Local CPU smoke

The [2026-09-05 smoke record](../evidence/native-cpu-tiny-en-jfk-bounded-previews-2026-09-05.json)
contains three replays of the 11-second JFK fixture with `tiny.en` on CPU.
The tested runtime commit is `6e6cd9884753e530cf3fea567bac2ed28b79839d`.
Each run uses English, greedy decoding, no timestamp tokens, and final seed 7.
Each also runs a separate control decode with identical PCM and settings.

| Chunk size | Preview interval | Stream decodes | Decoded source duration | Observed total time |
| --- | --- | --- | --- | --- |
| 200 ms | 4 s | 3 | 23 s | 8.872 s |
| 137 ms | 4 s | 3 | 23 s | 6.237 s |
| 200 ms | 8 s | 2 | 19 s | 4.977 s |

All final texts match the known fixture text and their same-PCM controls.
The two 4-second schedules produce identical event sequences despite different
chunk boundaries. Each run ends with an empty queue and all declared capacity
returned. At 4 seconds, the first hypothesis ends with `asked`; the later
revision changes it to `ask not`. No preview is committed before EOF.

Total time includes stream replays and the separate control, but excludes model
loading and setup checks. Input is replayed without waiting for wall-clock audio
arrival. These are single observations, not a live-latency benchmark. Warm-up,
host load, and decode count can affect the times. They do not establish a speedup
or general accuracy equivalence.

To repeat these configurations with the verified backend and a cached model:

```sh
python tools/run_native_example.py --stream-preview-ms 4000 --stream-chunk-ms 200
python tools/run_native_example.py --stream-preview-ms 4000 --stream-chunk-ms 137
python tools/run_native_example.py --stream-preview-ms 8000 --stream-chunk-ms 200
```

Use `--model-cache PATH` if the checkpoint is outside the default cache. Add
`--output PATH` to save a JSON report. The commands do not permit downloads.

### Counter scope

`metrics` returns a snapshot of admitted chunks and samples, successfully
published decodes, cumulative decoded source samples, and emitted events.
`source_reprocessing_factor` divides cumulative decoded prefix length by unique
admitted audio length. It does not measure GPU work, latency, throughput, or
failed decode attempts.
Whisper pads each prefix to its full mel window, so prefix length does not
represent actual encoder work. The counters do not include that padding.

The stream retains the bounded PCM slice and session window history. It does
not reuse encoder state, discard committed audio, or support arbitrarily long
input with constant memory. The native result exposed here contains text and a
window span, not timed tokens. Implementing Local Agreement requires timestamped
hypotheses, overlap alignment, and a safe progressive commit boundary before
old audio can be removed. Those are next steps, not properties of this profile.
