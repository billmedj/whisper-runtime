# Local stream checkpoints

A continuous stream can save a publication boundary and resume in a new process.
The checkpoint keeps committed text and its native metadata, retained PCM,
stream settings, event cursors, and endpoint detector state. It contains no
model weights, device tensors, decoder cache, random generator, or resource lease.

This is a logical savepoint. The new process decodes the retained audio again.
It does not resume an unfinished token step without recomputation.

Pending or failed EOF context retries cannot be saved. A completed retry can
be saved after normal publication and resource release. Its diagnostic receipt
is not stored; committed results and native metadata are preserved. Restoring
a completed stream does not repeat its final event. Earlier v1 files that lack
the `eof_context_retry` setting load it as `False` without changing their profile.

## Save and resume

Create the stream with enough bounded history for the intended session:

```python
stream = ContinuousTranscriptStream(
    adapter,
    stream_id="meeting-1",
    mel_builder=build_mel,
    config=config,
    history_limit=1024,
)
```

The existing default remains four records. Saving refuses a session whose
earlier publications have already left that history. It never silently saves
only the last part of the transcript. The maximum history limit is 4096 records;
the encoded checkpoint has a separate 16 MiB limit.

After a commit, before starting the next analysis:

```python
digest = stream.save_checkpoint(
    "meeting-1-0001.checkpoint",
    pipeline_identity=pipeline_digest,
)
# Persist digest in trusted caller-owned state before transferring ownership.
stream.close()
```

Use a new filename for each save. Existing files are never overwritten.
`pipeline_digest` must be `sha256:` followed by 64 lowercase hexadecimal digits.
It identifies the caller's verified tokenizer, preprocessing and backend build.
The runtime stores and compares this declaration; it cannot inspect an arbitrary
`mel_builder` callback to verify its implementation.

In a new process, load a compatible model and adapter, then:

```python
restored = ContinuousTranscriptStream.from_checkpoint(
    fresh_adapter,
    path="meeting-1-0001.checkpoint",
    expected_sha256=digest,
    pipeline_identity=pipeline_digest,
    mel_builder=build_mel,
)
# For new input, use restored.expected_chunk as the next chunk number.
# Drive restored.step() as usual. Saved EOF needs no second finish_input().
```

Restore compares the exact declared model and execution profile. A different
device, precision, backend, or feature-reuse profile is not a qualified migration.
New model work uses the normal admission, deadline and publication checks.
Previously committed output is available in `restored.state`; it is not emitted
again. New events continue the saved sequence and segment counters. A completed
stream does not emit another final event.

## Supported boundary

Save before the first analysis or immediately after a completed publication.
Segment, word-aligned and automatic-endpoint profiles retain their respective
state. In particular, the detector's partial-frame and quiet-run counters must
survive even when the corresponding earlier PCM has been evicted.

Save refuses active or retained transactions, pending commits, provisional
hypotheses, unresolved boundaries and resolution probes. It also refuses a
stream abandoned with `close()`. It does not clear a refusal, force an audio cut,
or select a new transcript. Finish the current publication before saving; an
arbitrary token-step pause is a different operation.

Input admission is locked while the snapshot is written. Saving does not close
the original stream or stop its producer. The caller must stop that owner before
resuming. Two processes restoring the same snapshot are independent forks, not
coordinated writers. No distributed lease or rollback protection is provided.

## Storage and failure contract

- The file uses bounded JSON with a fixed dataclass allowlist. It contains no
  pickle data or executable object names. Unknown fields and types are refused.
- The payload checksum and externally supplied file digest detect changes.
  They are not signatures, encryption or authentication.
- Publication uses an exclusive temporary file, file flush, and atomic
  no-overwrite linking on the same local filesystem. Symlink and reparse paths
  are refused. A failed save does not replace an earlier checkpoint.
  On Windows, the caller must prevent concurrent replacement of ancestor
  directories while file operations run.
- A completed save can survive source-process termination. Unfinished temporary
  files are not checkpoints. Universal power-loss durability is not claimed,
  particularly for Windows directory metadata or remote filesystems.
- The file contains audio and transcript content. Protect it with local access
  controls. Do not commit personal recordings to a public repository.

Choose a trusted storage directory before constructing the checkpoint filename.
For an application-created temporary directory, resolve that directory first:
`Path(temporary_directory).resolve() / "session.checkpoint"`. This matters on
macOS, where temporary paths can start with the `/var` alias for `/private/var`.
Do not resolve arbitrary untrusted checkpoint paths to bypass link refusal.

Only input included in this explicit savepoint is recoverable. `push()` still
acknowledges in-memory admission; this API does not make each acknowledgement
durable. Nor does it replay an output-delivery journal. Those are separate D6a
gates. Exact decoder-state restoration and GPU-free mid-token suspension remain
D6b work.

## Recorded validation

The [native CPU record](../evidence/native-cpu-tiny-en-checkpoint-2026-09-06.json)
uses the existing FP32 `tiny.en` checkpoint and an 11-second JFK fixture repeated
as two explicitly delimited source units. Four greedy decodes compare an
uninterrupted control with a saved first-unit boundary and a fresh process:

- The source had already admitted all 22 seconds when it saved at 11 seconds.
- The source process exited before the restore process started.
- The restored process decoded only samples 176000 through 352000: the retained
  second unit, not the committed first unit.
- Final session state, concrete publication types, native tokens, scores, timing
  metadata and combined events match the uninterrupted control exactly.
- The checkpoint was 473593 bytes. Both processes restored all declared capacity
  and left zero resource leases and an empty admission queue.

The record binds executed source, model and input hashes. Its own SHA-256 is
`0d94faf718c9f0c5a1b07b5ea765aa84f99f9c55cd04432e081874a9660c251e`.
Local input paths were replaced by portable labels. No GPU or model download
was used. This is a known-boundary recovery smoke, not a noise, long-session,
live-latency, power-loss or mid-token continuation qualification.

Eighteen dependency-free integration tests also cover base and hybrid profiles,
separate producer/consumer processes, partial endpoint frames, pending quiet
boundaries, aligned and silence provenance, refusal cases and no event redelivery.
Storage and state tests cover integrity, bounds, file failures and detector
roundtrips at all 401 split points of a fixed synthetic signal.

With the pinned backend and existing local inputs, reproduce the CPU smoke:

```sh
python -B tools/verify_native_checkpoint.py path/to/jfk.flac \
  --checkpoint path/to/tiny.en.pt --revision APPLIED_BACKEND_COMMIT
```

The command uses the current interpreter for two sequential child processes.
Set `PYTHONPATH` to the runtime `src` directory and the verified patched Whisper
checkout. It refuses missing inputs and does not fetch weights.
