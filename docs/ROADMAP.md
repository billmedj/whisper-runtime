# Development roadmap

RFC 0001 is the normative architecture and milestone definition. This page is
a short execution index. It does not define a second milestone scheme.

No milestone is complete. The repository currently provides a transactional
reference model, a formal lease model, 89 Python tests, one implemented Whisper
conformance pair, a serialized adapter for the historical synchronous API, and
an experimental native CPU adapter. The native adapter submits run creation,
prefill, each token step, and finalization separately. It checks cancellation
between stages but still reserves one complete worker. The transaction work
implements part of the M0, M1, M2, M3, and M7 foundations.

## M0 — Characterize the reference

Freeze the output, error, and resource contracts. Build the executable state
model and its formal safety model.

Exit: the conformance suite detects an intentional change to every recorded
behavior. Current status: one greedy CPU fixture pair is implemented; the exit
condition is not met.

## M1 — Extract request state

Move decode state, alignment state, options, cancellation, and random state out
of shared model objects.

Exit: forced concurrent interleavings match isolated executions.

Current status: the companion decoder exposes request-local suspendable run
state. The native CPU adapter owns one such run through a transaction fence.
Forced concurrent real-model interleavings remain untested.

## M2 — Separate and reuse kernels

Expose encode, prefill, decode-step, and alignment boundaries. Reuse encoded
audio where the reference algorithm permits it.

Exit: reference outputs pass and redundant encoder work is removed.

Current status: the native CPU adapter exposes create, prefill, token-step, and
finalize boundaries. It does not yet reuse encoded audio or schedule those
stages independently.

## M3 — Add the bounded worker

Add atomic resource admission, bounded stage queues, deadlines, cancellation,
and typed refusal results.

Exit: declared capacity bounds hold under sustained overload with real backend
adapters. Current status: bounded admission, exact leases, deadlines,
quarantine, and recovery are tested in the reference model. The legacy adapter
uses a conservative full-worker reservation; calibrated stage costs remain
unimplemented.

## M4 — Add safe batching

Batch compatible encoder inputs and decoder cohorts. Represent every
hypothesis-to-window relation explicitly.

Exit: batched results meet the selected compatibility profile and remain
isolated.

## M5 — Add managed cache memory

Add cache slots or pages, shared cross-attention state, copy-on-write beam
prefixes, quotas, and backend-aware release after quiescence.

Exit: measured peak memory stays within the admitted bound.

## M6 — Add streaming sessions

Add bounded audio ingestion, provisional revisions, commit watermarks, and
client credits. Keep offline-compatible and stable-streaming profiles distinct.

Exit: committed text never changes and memory does not grow with stream
duration.

## M7 — Add backend adapters

Keep the legacy backend and add optimized or exported backends behind explicit
capabilities.

Exit: each backend passes the conformance matrix for every capability it
declares. Current status: the historical synchronous adapter preserves the
legacy result mapping inside an immutable, versioned envelope. The native CPU
adapter adds cooperative token-step cancellation and a request-local generator.
Neither adapter declares batching, streaming, or device-fencing capability.

## M8 — Consider a default change

Consider a default runtime change only after reproducible compatibility and
performance evidence, documented extension behavior, an opt-out release, and
closure of every critical correctness issue.
