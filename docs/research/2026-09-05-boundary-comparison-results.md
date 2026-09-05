# Stream boundaries: four-configuration T4 diagnostic

Date: 2026-09-05.

## Scope and source

One T4 call compared four configurations on 33 seconds of audio: three exact
copies of Whisper's 11-second JFK fixture. All cells used `tiny.en`, FP32,
seed 7, and the same PCM. Input was not paced in real time. This is a diagnostic
on one synthetic input, not a latency benchmark or a corpus evaluation.

The [registration](../../experiments/modal-stream-boundary-diagnostic-v3.json),
[raw record](../../evidence/modal-t4-tiny-en-stream-boundaries-v3-2026-09-05.json),
and [receipt](../../evidence/modal-t4-tiny-en-stream-boundaries-v3-2026-09-05.attempt.jsonl)
are preserved. The record binds a 19-file source snapshot with SHA-256
`2ccc928f8d9cd1b7e00c4d21fad66ef2c6811538eb3f4494f51a19dbcddffda9`.
It does not claim that this snapshot was public before execution.

## Results

| Left context | Holdback | Outcome | Published through | Word edits against full-stream control |
| --- | --- | --- | --- | --- |
| 0 s | 1 s | Completed | 33 s | 2 / 66 |
| 2 s | 1 s | EOF needs resolution | 7.44 s | Not comparable: partial transcript |
| 0 s | 2 s | Completed | 33 s | 0 / 66 |
| 2 s | 2 s | EOF needs resolution | 7.44 s | Not comparable: partial transcript |

Word comparison removes case and punctuation. The two-second holdback result
does not match the control string exactly. The references are model outputs,
not independent human annotations. No general recognition improvement follows
from this single result, and the default holdback remains unchanged.

All four cells passed the registered lifecycle checks. They accepted 528,000
samples each, preserved committed revisions, accounted for buffered input, and
released runtime capacity. Only the two completed cells published all input
and emitted a final event. Each completed cell published before EOF.

The two context cells stopped with 119,040 samples committed. They retained
440,960 samples, including 32,000 already-committed context samples. Thus
`87,040 retained origin + 440,960 buffered = 528,000 accepted`.
Adding committed and buffered samples would double-count the overlap.

The generic error message in this record says that the bounded window reached
its limit. That wording is too broad. Trace 34 records `eof_unresolved`: the
analysis spans 5.44-33 seconds, which is 27.56 seconds, below the 30-second limit.
Its first segment crosses the committed boundary at 7.44 seconds. The current
policy cannot publish a suffix of that segment. The original record is unchanged.

## What the comparison establishes

On this input, longer holdback avoids the baseline's `am I` substitution for
`my`. Keeping left context does not solve the problem under the existing
whole-segment publication contract. It prevents completion in both tested cells.

Every cell performed 34 decodes. Decoded source coverage totals 188.84 seconds
for baseline, 224.94 seconds for longer holdback, and 463.44 seconds for each
context cell. These totals measure repeated source coverage, not GPU time:
Whisper pads encoder input to a fixed window. The small wall-time observations
in the record are not a speed comparison.

## Text agreement is not source alignment

The new offline analyzer compares closed-segment text tokens without requiring
unchanged segment boundaries. It checks actual COMMIT events before extending
its bounded token anchor. It does not authorize publication or audio eviction.

Run it without a model or GPU:

```shell
python tools/analyze_stream_text_agreement.py evidence/modal-t4-tiny-en-stream-boundaries-v3-2026-09-05.json
```

The context trace exposes a second failure mode. A previously committed phrase
can occur once in a later hypothesis, but refer to a later repetition in the
audio. An exact, unique token match does not establish the correct source
position. Token candidates from this analyzer must not be counted as recovered
publishable text. Shortening the anchor or taking the first match is not a fix.

Specifically, context traces 24 and 25 analyze 5.44-24 and 5.44-25 seconds.
The 14-token committed anchor matches token indices `[17,31)` in the later
11.52-20.36 second segment, not at the 7.44-second committed boundary. The
15-token candidate at `[31,46)` remains an analysis result, not a publication.
A local regression test preserves this case without loading the recording.

## Changes delivered with this diagnostic

- `resolve_text_prefix` isolates text-token agreement for analysis. It leaves
  timed publication and coverage unchanged.
- Optional preview coalescing skips obsolete analysis endpoints under backlog.
  It is off by default and has not been measured on GPU.
- An admitted coalesced analysis retains its input interval and EOF status
  across retries, including concurrent PCM arrival and resource recovery.

Verification: all 330 runtime tests and 249 repository-tool tests pass. The
rebuilt wheel also passes the 330 runtime tests. Type, lint, format, repository,
and distribution checks pass. None of these checks establishes ASR quality.

## Next implementation gate

Implement a separately named publication profile with distinct text and audio
progress. An anchor must refer to a source position, not just a matching string.
Keep raw model timestamps unchanged. Silence and unmatched audio need explicit
coverage rules before buffer eviction. Ambiguous alignment must remain visible.

First test repeated phrases, segment splits, timestamp drift, partial words,
silence, cancellation, and EOF locally. Then use matched, distinct recordings
to measure completion, word errors, revision rate, delay, and compute. Do not
start the 30-minute GPU replay until short cases complete under the new contract.
Do not change defaults based on this fixture.

## Transport and cost

The preceding v2 call computed but failed to return its result. Modal's
asynchronous return exceeded its 8 KiB inline limit; the required blob upload
was blocked by restricted Modal access. Its failed receipt is retained.

Before v3, a CPU-only probe returned 65,571 serialized bytes synchronously under
the same network restrictions. The payload digest matched. The v3 worker then
returned 21,935 compressed bytes through the synchronous path. The client saved
those bytes before decoding and wrote the complete 505,108-byte JSON record.
Its SHA-256 is
`220ef884ec52b8027dda1b1500be9af0517d09547efa5d270717ce1622c25328`.

This step used two T4 calls: the failed v2 transport and the recovered v3
comparison. Neither used automatic retries. The observed monthly metered total
rose from USD 0.07021114 to USD 0.09021114. This USD 0.02 change is not an exact
per-call invoice or remaining-credit balance. All listed apps were stopped with
zero tasks after the comparison.
