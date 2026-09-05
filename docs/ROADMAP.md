# Development roadmap

[RFC 0001](rfcs/0001-state-resource-execution.md) is the design proposal. This
page records implementation order. A target is incomplete until its stated
evidence is committed to the repository.

The single-worker cell in the
[versioned CUDA qualification registration](../experiments/native-cuda-qualification-v6.json)
has one passing
[record](../evidence/modal-t4-tiny-en-jfk-native-cuda-qualification-v6-2026-09-04.json).
Its evidence contract is qualification-only. The next Modal step requires a
separate registration for replication or a new schema for performance. A later
performance matrix follows the fixed comparison, cost order, metrics,
falsifiers, and publication rules in the
[CUDA experiment protocol](EXPERIMENT_PROTOCOL.md).

## Now: establish the reference boundary

- Expand the four-case CPU conformance corpus beyond one input.
- Add sampling, segment-timestamp, cancellation, and failure records. Extend
  the existing beam-search, translation, and word-timestamp coverage to more
  inputs and model sizes.
- Maintain the pinned CPU backend checks for deterministic interleaving and
  overlapping outer decoder calls in two operating-system threads.
- Maintain the adapter-level two-lane CPU record.
- Select and document one random-generator contract.
- Maintain the fixed single-worker T4 qualification record. Rerun it only under
  a new versioned registration.
- Extend the recorded single-lane T4 case to more inputs and model sizes only
  under separately versioned registrations.
- Derive measured resource profiles only after the expanded CUDA corpus is
  stable across repeated workers.

Exit condition: the recorded compatibility matrix covers the declared adapter
capabilities, and admitted work does not exceed its measured profile under the
tested load.

## Next: schedule useful units of work

- Reuse encoded audio where the selected compatibility profile permits it.
- Add compatible encoder and decoder batching.
- Represent each hypothesis-to-window relation explicitly.
- Add managed cache slots, quotas, and quiescence-aware release.
- Measure cancellation delay, queue delay, memory, latency, and throughput.

Exit condition: batching and cache reuse preserve the selected output contract,
isolate requests, and remain within the measured resource bound.

## Streaming: implemented bounded slice

[`NativeTranscriptStream`](BOUNDED_STREAMING.md) implements
`bounded_prefix_preview/v1`: sequence-checked 16 kHz mono s16le ingestion,
a configurable limit of at most 30 seconds, and full-prefix previews. `push()`
only admits audio; `step()` exposes native startup, individual token steps,
and publication separately. Immutable text-and-span revisions remain
provisional until a successful EOF decode emits commit and final events.

The stream supports owner-thread close and active-run cancel and stop controls.
Preprocessing and startup before handle return are not interruptible through
those stream controls. Pausing does not renew the transaction TTL or release
capacity; failed cleanup may retain capacity pending native recovery.

This slice is not continuous streaming or Local Agreement. It retains audio
and window history, reprocesses prefixes, and exposes no timed-token hypotheses.
The [local CPU smoke](BOUNDED_STREAMING.md#local-cpu-smoke) checks three JFK
replays against same-PCM controls, chunk-partition independence for two of those
replays, and capacity release. It makes no general performance or
offline-equivalence claim.

## Next streaming boundary: timestamped Local Agreement

The native adapter now retains language, tokens, available scores, and complete
timestamped segments. An optional publication span selects whole segments from
a larger analysis span. The session's committed-prefix guard still applies to
the selected output. See the [native result contract](NATIVE_ADAPTER.md#analysis-context-and-published-text).
These model-predicted segment times are not word alignments or stability scores.
The [local CPU smoke](../evidence/native-cpu-tiny-en-jfk-timed-publication-2026-09-05.json)
checks result inspection, two selected publications, fixed-analysis control
parity, and unchanged default and bounded-preview paths on one JFK fixture.

- Carry timed hypotheses into the stream policy without changing the existing
  bounded-preview event contract.
- Align overlapping hypotheses and define a versioned Local Agreement policy
  for progressive commits. Do not infer token times from window endpoints.
- Preserve immutable committed text and text-and-span revision identities
  across window shifts, cancellation, retries, and EOF.
- Add rolling audio retention, window-history compaction, and client flow
  control so session memory remains bounded as input duration grows.
- Keep offline-compatible, bounded-preview, and continuous stable-streaming
  profiles separate, with evidence for each declared output contract.
- Measure prefix reprocessing, cancellation delay, retained capacity, and
  long-session memory before making continuous-streaming performance claims.

Exit condition: committed text does not change, source commit boundaries are
supported by timed hypotheses, and measured session memory stays bounded as
input duration grows under the declared profile.

## Later: backend coverage

- Add compiled or exported backends behind declared capabilities.
- Apply the same ownership and close rules to every added backend.

Exit condition: each adapter passes the cases for every capability it declares.

## Release gate

The native adapter can become this project's default only after reproducible
compatibility and performance evidence, documented extension behavior, a
release with an opt-out path, and closure of all critical correctness issues.
