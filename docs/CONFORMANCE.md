# Conformance oracle

The conformance oracle separates output compatibility from execution quality.
A runtime change can preserve one and regress the other.

## Current coverage

The repository contains four implemented reference/candidate pairs for the
pinned JFK audio fixture on the CPU:

- `tiny.en` English transcription with greedy decoding;
- `tiny.en` English transcription with beam size 5;
- `tiny.en` English transcription with word timestamps;
- multilingual `tiny` with the translation task and English input.

All four cases use temperature `0.0` and `float32` tensors. The comparator
checks the complete recorded public result for each pair. The remaining cases
in `conformance/cases.json` are marked `planned`. The current fixtures record
wall time, but they do not constitute a latency benchmark. Several resource
fields are null because those measurements have not been implemented.

## Output contract

Each fixture records:

- checkpoint digest and model dimensions;
- audio digest and sample range;
- decoding options and random seed;
- device, data type, and software versions;
- text, tokens, segments, and language;
- segment timestamps and word timestamps when enabled;
- declared numerical tolerance.

Each record has one outcome: `success`, `error`, `cancelled`, or
`deadline_exceeded`. Successful records contain a public `result`. Other
outcomes contain a typed `termination` record. This distinction lets the same
oracle test output compatibility and terminal behavior without inventing a
successful result for failed work.

The oracle compares public output first. Internal tensors are diagnostic data,
not a stable public contract.

## Planned execution contract

The completed oracle is intended to record:

- wall time and real-time factor;
- peak host and device memory;
- queue delay and execution delay;
- encoder and decoder step counts;
- fallback attempts;
- cancellation delay;
- resources held after termination.

## Planned matrix

The first complete matrix is intended to cover:

- greedy, sampling, and beam search;
- one audio and a batch of audios;
- timestamps on and off;
- word alignment on and off;
- transcription and translation;
- silence, noise, short speech, and long speech;
- normal completion, failure, timeout, and cancellation;
- isolated, sequential, and forced concurrent execution.

## Compatibility profiles

`reference` targets a pinned OpenAI Whisper revision in a recorded environment.
`optimized` permits declared numerical differences but must retain public output
types, option semantics, and error behavior.

No cross-device bit-for-bit claim is made. Token, timestamp, and numerical
tolerances must be explicit for each profile.

All tolerances are absolute. The comparator sets relative tolerance to zero.
It compares the complete public payload, including segment text, token lists,
diagnostic values, and word timestamps when present. Environment and timing
measurements remain evidence; they are not output-equality fields.
