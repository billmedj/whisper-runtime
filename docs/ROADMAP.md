# Development roadmap

[RFC 0001](rfcs/0001-state-resource-execution.md) is the design proposal. This
page records implementation order. A target is incomplete until its stated
evidence is committed to the repository.

## Now: establish the reference boundary

- Expand the conformance corpus beyond one greedy CPU case.
- Record sampling, beam search, translation, timestamps, word timestamps,
  multiple inputs, cancellation, and failure behavior.
- Maintain the CPU check for deterministic same-model interleaving, early
  cleanup of one run, and isolated-baseline equality for the survivor.
- Extend request-isolation evidence to two admitted runtime transactions,
  operating-system threads, and CUDA.
- Select and document one random-generator contract.
- Validate the native path on CUDA with source, model, input, environment,
  output, latency, and memory records.
- Add device completion fences and measured resource profiles.

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
- Validate a second Whisper model size and decode configuration against the
  same ownership and close rules.

Exit condition: committed text does not change, session memory stays bounded as
input duration grows, and each adapter passes the cases for every capability it
declares.

## Release gate

The native adapter can become this project's default only after reproducible
compatibility and performance evidence, documented extension behavior, a
release with an opt-out path, and closure of all critical correctness issues.
