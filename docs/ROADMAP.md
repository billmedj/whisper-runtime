# Development roadmap

[RFC 0001](rfcs/0001-state-resource-execution.md) is the design proposal. This
page records implementation order. A target is incomplete until its stated
evidence is committed to the repository.

The next Modal step is the single-worker cell in the
[versioned CUDA qualification registration](../experiments/native-cuda-qualification-v3.json).
Its evidence contract is qualification-only. A later performance matrix follows
the fixed comparison, cost order, metrics, falsifiers, and publication rules in
the [CUDA experiment protocol](EXPERIMENT_PROTOCOL.md) under a new evidence
schema or version.

## Now: establish the reference boundary

- Expand the four-case CPU conformance corpus beyond one input.
- Add sampling, segment-timestamp, cancellation, and failure records. Extend
  the existing beam-search, translation, and word-timestamp coverage to more
  inputs and model sizes.
- Maintain the pinned CPU backend checks for deterministic interleaving and
  overlapping outer decoder calls in two operating-system threads.
- Maintain the adapter-level two-lane CPU record.
- Select and document one random-generator contract.
- Run and publish the fixed single-worker T4 qualification cell after its
  registration has an external public timestamp.
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

## Later: bounded streaming and backend coverage

- Add bounded audio ingestion and client flow control.
- Publish versioned provisional segments and immutable commit watermarks.
- Keep offline-compatible and stable-streaming profiles separate.
- Add compiled or exported backends behind declared capabilities.
- Apply the same ownership and close rules to every added backend.

Exit condition: committed text does not change, session memory stays bounded as
input duration grows, and each adapter passes the cases for every capability it
declares.

## Release gate

The native adapter can become this project's default only after reproducible
compatibility and performance evidence, documented extension behavior, a
release with an opt-out path, and closure of all critical correctness issues.
