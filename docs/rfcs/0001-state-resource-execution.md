# RFC 0001: Explicit State and Resource-Aware Whisper Execution

- Status: Draft
- Date: 2026-09-03
- Scope: OpenAI Whisper inference runtime

## Summary

This RFC proposes an execution architecture for Whisper inference.

The architecture aims to preserve Whisper model weights, checkpoints, and
public results. It changes how the runtime owns mutable state and physical
resources.

The design has six primary objects:

- `Model`
- `Worker`
- `Session`
- `Request`
- `Window`
- `Hypothesis`

The `Model` is immutable after publication to a worker. All request data is private to a request or to one of its child objects. The `Worker` owns the device, memory pools, queues, and scheduler. The scheduler admits work only when the required resources are available.

The first implementation must preserve the existing synchronous Python API.
The new runtime remains opt-in until its compatibility suite covers each model
and adapter capability that the project declares.

## Requirements language

The words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** define requirements in this RFC.

## Implementation status

This RFC describes the target architecture. A requirement in this document is
not evidence that the current package implements it.

The current reference package implements `ModelSnapshot`, `Worker`, `Session`,
`RequestState`, `WindowTransaction`, `ExecutionScope`, and `SubmissionGate`. It
provides bounded admission, exact in-process leases, request-local random state,
versioned session commits, deadlines, cooperative stop, quarantine, recovery,
and owner-death takeover. A legacy adapter executes an unmodified synchronous
`model.transcribe()` call as one serialized transaction with a full-worker
reservation. A native CPU adapter exposes request-local run creation, prefill,
token steps, finalization, and cleanup through a pinned patched backend. Its
default profile admits one transaction. An experimental profile admits two
transactions while serializing run construction and encoder preparation. One
committed Windows CPU record exercises that profile through the runtime
adapter. Four CPU reference/candidate pairs cover greedy decoding, beam search,
word timestamps, and translation. A strict single-lane CUDA profile has two
committed T4 records for the same pinned `tiny.en` FP32 case on separate AWS and
GCP workers. They cover admission, a private stream, completion-fence ordering,
cooperative cancellation, retained capacity after an injected fence failure,
manual recovery, and reuse. The injected failure occurs in the harness before
the delegate synchronization call; it is not evidence of a physical CUDA
driver failure. Safe batching, CUDA concurrency, other models and devices,
streaming, managed cache slots, optimized backends, measured memory bounds,
performance, and the complete conformance matrix remain design targets.

The formal models cover abstract lease provenance, lifecycle, capacity, stale
commits, independent-session commits, and a small completion-boundary state
machine. The latter makes publication and capacity release depend on an
observed completion fence or explicit recovery. The models do not establish
Python thread behavior, backend execution, submission-gate refinement, CUDA
event validity, or correspondence with the adapter implementation.

## Problem

Whisper inference appears to be one function:

```text
audio -> text
```

The real operation is a sequence of state transitions:

```text
audio input
  -> feature extraction
  -> audio encoding
  -> language detection
  -> autoregressive decoding
  -> fallback attempts
  -> optional word alignment
  -> result assembly
```

Each stage uses mutable state and finite resources. Examples include:

- decoder key-value caches;
- beam search state;
- random number generator state;
- cross-attention data used for word alignment;
- audio buffers;
- model input and output tensors;
- CPU threads and decoder processes;
- RAM, pinned memory, and device memory.

The pinned OpenAI Whisper revision used by this project does not define one
owner for every state. Some state is carried through hooks, module attributes,
shared random number generators, or mutable Python objects. Its public API also
starts work without an admitted peak resource cost.

These properties limit safe concurrency. They also make cancellation, batching, streaming, and device admission difficult.

## Goals

This RFC has the following goals:

1. Give every mutable state one owner and one lifetime.
2. Keep model weights and published model configuration immutable during service.
3. Make each inference transition explicit and testable.
4. Reserve resources before a transition starts.
5. Keep all queues and buffers bounded.
6. Support cooperative cancellation with commit and release guarantees.
7. Reuse audio computation across language detection, decode attempts, and alignment.
8. Support encoder batching and incremental decoder scheduling.
9. Define honest streaming semantics.
10. Preserve the existing checkpoints and Python API.
11. Provide a stable boundary for PyTorch, compiled, and exported backends.

## Design rules

The runtime MUST follow these rules:

1. Shared data is immutable or owned by one worker.
2. Request state is never stored in the shared model.
3. A resource is reserved before it is used.
4. A device out-of-memory error is not an admission policy.
5. A cancelled request cannot commit a result.
6. A late retry cannot replace a newer result.
7. A beam can read only the window and cache assigned to its request.
8. The runtime does not reduce quality without explicit client permission.
9. A runtime result envelope identifies the model, runtime profile, and state revision that produced it.
10. Compatibility means behavior, not only function signatures.

## System model

For model weights `theta`, a kernel transition is:

```text
K(theta, request_state, input, grant)
    -> (next_request_state, output, usage)
```

`grant` is the resource reservation for the transition. `usage` is the measured cost.

The transition is logically pure. A backend MAY update a buffer in place for performance, but the buffer MUST have an exclusive owner for the full update. The next state MUST not contain an untracked alias to another request's mutable buffer.

The scheduler chooses a set of transitions whose combined reservations fit within the worker budget.

## State ownership

### Model

`Model` contains:

- model weights;
- model dimensions;
- tokenizer data;
- positional data;
- alignment-head selection;
- model and checkpoint identity;
- backend capability data.

The model becomes immutable when a worker publishes it for use.

While the model is published, callers MUST NOT:

- move it to another device;
- change its dtype;
- enter training mode;
- replace a module;
- change alignment heads;
- change an attention backend through shared state.

Such changes require a new model instance or a worker restart.

### Worker

`Worker` owns physical execution resources:

- one model replica;
- one compute device;
- device streams;
- memory pools;
- CPU worker limits;
- audio decoder process limits;
- stage queues;
- resource reservations;
- scheduling state;
- compiled graph caches;
- device health state.

The first implementation SHOULD use one scheduling thread per device. This gives one writer for worker state. Additional device streams are an optimization and require measurement.

### Session

`Session` owns the state that persists across requests or audio windows:

- session identifier;
- source clock;
- audio position;
- bounded input buffer;
- current language state;
- committed transcript;
- provisional transcript;
- decoder prompt history;
- next window position;
- current state version;
- streaming profile;
- session deadline and priority class.

A file transcription MAY use one short session. A live input can use one session for its full lifetime.

Session memory MUST remain bounded as input duration increases. Old audio, features, and caches MUST be released after the commit boundary unless a selected profile requires them.

### Request

`Request` owns one operation submitted to a worker. It contains:

- request identifier;
- parent session and input version;
- normalized, immutable options;
- priority and deadline;
- cancellation state;
- request-local random number generator;
- current stage;
- resource reservation;
- result revision;
- failure state.

Options supplied by a caller MUST be copied and normalized before admission. The runtime MUST NOT modify caller-owned lists or dictionaries.

### Window

`Window` owns data for one exact audio span:

- start and end time;
- source and normalization profile;
- Mel features;
- encoded audio features;
- optional cross-attention key-value data;
- active reference count;
- device and dtype;
- lifecycle state.

Encoded audio and cross-attention key-value data are read-only after creation. All hypotheses for the same window SHOULD share them.

Language detection, decode retries, and word alignment SHOULD reuse the same encoded audio when their numerical contract permits it.

### Hypothesis

`Hypothesis` owns one active decode path:

- token sequence;
- self-attention key-value state;
- cumulative log probability;
- completion state;
- beam ancestry;
- request and window identifiers;
- current text position;
- random generator subsequence when sampling is used.

Beam branches MAY share immutable prefix pages. A write after a fork MUST use copy-on-write or a new page.

The scheduler MUST treat all hypotheses in one beam group as one accounting unit. It MUST keep an explicit mapping from each hypothesis to its audio window.

## State machines

### Request states

```text
RECEIVED
  -> VALIDATED
  -> ADMITTED
  -> PREPROCESSING
  -> WAITING_FOR_ENCODER
  -> ENCODING
  -> WAITING_FOR_DECODER
  -> DECODING
  -> FALLBACK or ALIGNING
  -> COMMITTING
  -> COMPLETED
```

Terminal alternatives are:

```text
REJECTED
CANCELLED
DEADLINE_EXCEEDED
FAILED
```

Transitions to a terminal state are monotonic. Cancellation is idempotent.

### Session states

```text
OPEN -> DRAINING -> CLOSED
  \-> FAILED
  \-> CANCELLED
```

`DRAINING` rejects new audio and finishes admitted work. `CLOSED` has no live device allocation.

### Window states

```text
BUFFERING
  -> READY
  -> ENCODED
  -> ACTIVE
  -> COMMITTED
  -> RELEASED
```

A released window cannot become active again. The runtime must create a new window if it must recompute that span.

### Transactional commit

Each request records the session version that it read. A commit succeeds only if that version is still current.

The current reference transaction has these states:

```text
PREPARED -> RUNNING -> QUIESCING -> COMMITTED
                                 -> ABORTED
                                 -> EXPIRED
                                 -> QUARANTINED
QUARANTINED -> QUIESCING -> ABORTED or EXPIRED
```

`PREPARED` work has not started a backend scope and can release its lease
directly. Running work must complete this close protocol:

1. Seal its `SubmissionGate` to reject new submissions.
2. Drain every registration callback admitted before the seal.
3. For abort, cancellation, or expiry, deliver an idempotent stop request.
4. Create one final aggregate completion fence after the drain.
5. Wait until the fence proves that all registered backend work is quiescent.
6. Close the gate, apply the terminal transition, and release the lease.

The completion fence MUST NOT be created before the submission gate drains.
Otherwise, an admitted callback could register work after the fence snapshot.
The lease MUST remain held if drain, stop, or fence completion fails.

The commit record contains:

- session identifier;
- input state version;
- output state version;
- audio span;
- model digest;
- canonical options digest;
- selected result revision;
- measured resource use.

If the input version is stale, the runtime discards or rebases the result according to an explicit policy. It MUST NOT overwrite newer session state.

The current reference model rejects a stale commit. It does not implement a
rebase policy.

Cancellation, abort, and expiry remain effective while a commit waits for the
submission gate and backend fence. The runtime checks the selected outcome
again after quiescence. It then serializes request cancellation with one atomic
publication of request state, session state, and random state. Publication is
the revocation boundary. A stop accepted before it prevents publication; a stop
after it does not roll back the committed state.

This revocation rule is covered by executable concurrency tests. The Lean
completion-boundary model represents fence outcomes and publication at the
abstract state-machine level. It does not represent threads, cancellation
races, concrete backend fences, or the Python publication critical section.

## Functional kernel

The native backend SHOULD expose these operations:

```text
extract_features
encode
detect_language
prefill
decode_step
fork_hypothesis
reorder_hypotheses
align
release
```

The fast path MUST NOT require module hooks or class-level mode changes.

The attention mode is request-local. Alignment can request explicit attention capture without changing the mode of another request.

The internal cache format SHOULD use stable layer indexes and tensor tuples. It SHOULD NOT use module objects as serialized keys.

The compiler boundary contains tensor operations only. Scheduling, queue access, callbacks, and resource accounting stay outside compiled graphs.

### Trusted backend boundary

`ExecutionScope` is a trusted in-process interface. Each conforming scope MUST:

- be bound to only one live transaction at a time;
- register each backend operation before its submission callback returns;
- include every registered operation in the final aggregate completion fence;
- make `request_stop()` idempotent;
- return from the fence only after the registered work no longer uses the
  transaction lease.

The current runtime claims a scope object by identity when a transaction
starts. It rejects a second live transaction that uses the same object. It
releases the claim during terminal cleanup.

`SubmissionGate` prevents late registration after close begins. It cannot
detect work that an adapter starts without registration. The runtime does not
provide process isolation and does not defend against arbitrary writes by code
inside the host process.

A submission callback MUST NOT synchronously close its own transaction. Closing
waits for all admitted callbacks to return, so a reentrant close would wait for
itself. The current runtime rejects a direct commit from the callback. A stop
request made there seals the gate and defers completion to a later safe point.

## Resource model

Each worker has a budget vector:

```text
device memory
host memory
pinned memory
CPU threads
audio decoder processes
transfer bandwidth
spool storage
wall-clock time
```

For a decoder with:

- `L` layers;
- state width `D`;
- `A` encoded audio positions;
- `T` text positions;
- `G` active hypotheses;
- `p` bytes per element;

the main persistent cache estimates are:

```text
self_KV  ~= 2 * L * D * p * sum(G[r] * T[r])
cross_KV ~= 2 * L * A * D * p * active_windows
```

`cross_KV` MUST be counted once per active window when it is shared by its hypotheses.

Peak device memory is:

```text
model weights
+ persistent state
+ encoder workspace
+ decoder workspace
+ alignment workspace
+ allocator reserve
+ safety margin
```

The worker MUST calibrate workspace estimates for each model, device, dtype, backend, and batch shape. Static formulas alone are not sufficient.

### Reservations

Admission creates an atomic reservation. A stage can start only when it holds the required reservation.

Reservations MAY be released in stages. A long transcription does not need to reserve device memory for every future window. It needs a bounded window pipeline and the maximum allowed active state.

If actual use approaches a reservation limit, the worker MUST pause at a safe
transition boundary. It then releases any recomputable reservation and either
requeues for a new atomic grant or fails with a typed resource error. A stage
MUST NOT retain a partial grant while it waits for more capacity.

It MUST NOT silently change model, dtype, beam size, timestamp mode, or alignment mode.

### Memory manager

The device memory manager SHOULD provide:

- fixed-size cache pages or slots;
- per-request ownership records;
- shared read-only cross-attention cache;
- copy-on-write beam prefixes;
- release only after verified backend quiescence;
- high-water measurements;
- request and worker quotas;
- optional eviction of recomputable data.

Beam reorder should update page tables or slot maps. It should not copy complete cache tensors when an indirection is sufficient.

Alignment has a separate memory class. The backend SHOULD capture only selected alignment heads. It SHOULD reduce across heads before it retains more data than the alignment algorithm requires.

## Scheduler

The runtime uses separate bounded queues for:

1. audio ingestion and decoding;
2. feature extraction;
3. audio encoding;
4. decoder prefill;
5. decoder steps;
6. fallback attempts;
7. word alignment;
8. result commit.

These stages have different compute and memory profiles. A single queue would cause head-of-line blocking and weak resource estimates.

### Encoder scheduling

The encoder scheduler MAY form a microbatch when these values match:

- model identity;
- device;
- dtype;
- number of Mel bands;
- input shape;
- deterministic execution profile.

The batching delay MUST respect the earliest admitted deadline.

The runtime SHOULD encode one window once and reuse the result for language detection, decode fallbacks, and alignment.

### Decoder scheduling

The decoder scheduler operates at token-step boundaries.

The first version MAY batch only requests with:

- the same model and dtype;
- the same decode strategy;
- the same number of hypotheses;
- compatible token positions;
- compatible timestamp and logit filters.

General continuous batching requires:

- one position offset per hypothesis;
- explicit padding or ragged masks;
- an explicit hypothesis-to-window map;
- cache slots or pages;
- controlled gather and scatter operations.

The scheduler MUST remove a completed or cancelled hypothesis at the next safe boundary.

### Fairness

The scheduler SHOULD provide separate interactive and batch classes. Each class has a minimum share and MAY borrow unused capacity.

Within a class, the scheduler SHOULD use weighted fair service between tenants or sessions. It MAY use earliest deadline first within the fair share. Aging MUST prevent starvation.

Service cost includes:

- encoded audio duration;
- encoder passes;
- generated tokens;
- active hypotheses;
- fallback attempts;
- alignment work.

Token count alone is not a sufficient cost unit for Whisper.

### Admission results

Admission returns one of these results:

- `ADMITTED`
- `QUEUED`
- `RESOURCE_EXHAUSTED`
- `DEADLINE_UNATTAINABLE`
- `UNSUPPORTED_COMBINATION`

`QUEUED` includes a queue deadline. A request that cannot fit on the worker at any time returns `RESOURCE_EXHAUSTED` before model execution.

### Backpressure

Each queue has limits for:

- total estimated cost;
- reserved memory;
- item count;
- maximum wait time.

When a high-water mark is reached, the runtime stops admission. It resumes only after the low-water mark is reached. This hysteresis prevents rapid admission oscillation.

A slow streaming client uses a credit limit. It cannot retain unbounded result or device state.

### Cancellation and deadlines

The runtime checks cancellation:

- before admission;
- before and after each stage;
- at each decoder step;
- before commit.

A running device kernel may finish before cancellation takes effect. The documented cancellation bound is the current kernel time plus scheduler response time.

Audio decoder processes receive a normal termination request first. The worker force-stops a process only after a fixed grace period.

## Long-form transcription

A session normally decodes one active window at a time because predicted timestamps and previous text can determine the next window.

The runtime MAY prepare or encode a bounded number of later windows when this does not change the selected compatibility profile. It MUST NOT decode dependent windows out of order.

The window pipeline MUST have a fixed maximum depth. Processing a longer file must increase total work, not peak session memory.

## Streaming limits

This design does not claim both exact offline output and early immutable
streaming output for the current model and preprocessing.

There are two causes:

1. Current log-Mel preprocessing clips values relative to the maximum value in the processed spectrogram. Future audio can change this maximum.
2. The audio encoder uses bidirectional attention inside its input window. Future samples in that window can change past representations.

The runtime MUST expose the selected streaming contract.

### Offline-compatible profile

The `offline_compatible` profile preserves the current preprocessing and window algorithm.

- It can emit progress events.
- It can emit provisional text.
- It cannot promise an immutable prefix before enough input is available.
- An exact final pass can revise provisional text.

### Stable streaming profile

The `streaming_stable` profile uses a bounded normalization and window policy.

- It emits versioned provisional segments.
- It uses overlap and bounded future context.
- It commits a segment only after the profile's stability rule passes.
- It never changes a committed segment.
- Its final result is not guaranteed to equal the offline-compatible result.

The event protocol is:

```text
provisional(segment_id, revision, span, text)
replace(segment_id, old_revision, new_revision, span, text)
commit(segment_id, revision, watermark)
final(session_version)
```

Sequence numbers are monotonic. A client can resume after the last accepted sequence number.

If a caller requires an offline-exact final result, all text that can still change MUST remain provisional until the final pass.

### True causal streaming

A true causal mode can require a different model or additional training. Its
design can include:

- causal or calibrated feature normalization;
- chunked encoder attention;
- bounded right context;
- encoder cache;
- training on the same streaming regime.

This work is outside this RFC.

## Compatibility

### Existing Python API

The following behavior MUST remain available:

- existing checkpoint files and state dictionary keys;
- `whisper.load_model()`;
- `model.transcribe()`;
- `whisper.decode()`;
- `whisper.detect_language()`;
- current result dictionaries and `DecodingResult` fields;
- single-input and batch return conventions;
- existing decode options and validation errors;
- current CLI output formats.

The legacy synchronous call creates an internal session, drains it, and converts the result to the existing format.

Runtime metadata is available through an opt-in result envelope. The envelope
contains the model identity, execution profile, session revision, measurements,
and the legacy payload. It MUST NOT add fields to the historical result
dictionary returned by the existing Python API.

The first release keeps the historical path as the default. The new runtime is selected explicitly.

### Legacy backend

`LegacyBackend` preserves the current module and hook behavior.

The runtime MUST route an unknown subclass, replaced decoder, overridden cache hook, or unsupported extension to the legacy backend. The legacy backend MAY serialize calls on one model instance when shared hooks or shared module state make concurrency unsafe.

The native fast path MUST use explicit capability detection. It MUST NOT assume that any object with a similar method name has native state semantics.

### Backend contract

A backend publishes capabilities such as:

```text
batched_encode
incremental_decode
continuous_batching
beam_search
sampling
word_alignment
translation
supported_streaming_profiles
supported_dtypes
supported_export_formats
```

The runtime rejects an unsupported option before admission. It does not emulate a missing feature without an explicit compatibility rule.

### Compilation and export

The functional kernel gives compilers an explicit tensor boundary.

For decoder export, cache inputs and outputs SHOULD use a flat, stable order:

```text
past_self_key[layer]
past_self_value[layer]
cross_key[layer]
cross_value[layer]
present_self_key[layer]
present_self_value[layer]
```

The scheduler, Python callbacks, hooks, locks, and resource manager remain outside the exported graph.

The runtime MAY provide backends for eager PyTorch, `torch.compile`, `torch.export`, ONNX Runtime, or other engines. Each backend must pass the same conformance suite for the features it declares.

### Determinism profiles

The runtime defines at least two execution profiles:

- `reference`: preserves the historical algorithm and ordering;
- `throughput`: permits compatible batching and optimized kernels.

Each request has its own random number generator and seed record.

Bitwise identity across devices and backends is not required. Each profile MUST document its numerical and output compatibility rules. A backend MUST NOT claim reference compatibility only because its tensor shapes match.

## Failure handling

Validation failures before admission acquire no lease. An idle `PREPARED`
transaction can release directly because no backend execution scope has
started.

A running transaction releases its lease only after the close protocol proves
backend quiescence. Success, cancellation, expiry, and runtime failure select a
terminal outcome; they do not by themselves authorize release.

If drain, stop delivery, fence creation, or fence wait fails, the transaction
enters `QUARANTINED`. The worker retains its queue entry and resource lease.
Recovery retries stop delivery and obtains a new final aggregate fence. It
releases the lease only after that fence completes. Stop delivery is
idempotent, and concurrent recovery waits for an in-flight stop call instead of
delivering a duplicate call.

If lease release or worker retirement fails after a terminal outcome, the
terminal decision remains fixed and cleanup can be retried. A successful lease
release is single-use.

The thread that calls `start()` owns cooperative checkpoints and commit. A
supervisor may request stop while that thread is alive, but it must not take
over fence completion or release. After the owner thread exits, the supervisor
may take over a running, quiescing, or quarantined transaction and complete the
stop, fence, and release sequence.

A production backend adapter SHOULD mark its worker unhealthy after a device
failure that makes quiescence uncertain. Worker health management is not
implemented in the current reference package.

Partial output is returned only when the caller requested it. It includes the last committed span and a terminal reason.

## Observability

The runtime SHOULD measure:

- admission decisions and reasons;
- queue depth and age by cost;
- stage latency;
- time to first segment;
- real-time factor;
- encoder and decoder batch fill;
- generated tokens per second;
- device memory reserved, allocated, and peak;
- self- and cross-cache size;
- fallback count;
- alignment cost;
- cancellation response time;
- deadline misses;
- worker failures.

Traces MUST NOT contain audio or transcript text by default. High-cardinality identifiers and content logging require explicit configuration.

Model confidence values, compression ratio, and no-speech probability are runtime signals. They are not proof that a transcript is correct.

## Conformance tests

The implementation is not complete until it passes tests for:

1. isolated and concurrent output equivalence;
2. request-local cache, alignment, and random state;
3. single audio, multiple audio, sampling, and beam groups;
4. translation, timestamps, prompts, and silence handling;
5. cleanup after every stage failure;
6. cancellation at every transition boundary;
7. bounded queues under sustained overload;
8. stable device memory after repeated work;
9. no starvation under mixed long and short requests;
10. monotonic streaming sequence numbers;
11. immutable committed streaming segments;
12. legacy subclass and hook compatibility;
13. checkpoint and result-format compatibility;
14. backend capability rejection;
15. download and audio-ingestion fault handling.

The compatibility corpus SHOULD cover every official model family and both CPU and device execution where supported.

## Milestones

### Milestone 0: Characterize the reference

- Freeze a compatibility corpus.
- Record options, tokens, text, timestamps, errors, and resource use.
- Cover greedy decode, sampling, beam search, translation, long-form work, and alignment.

Exit condition: the suite detects an intentional change to each recorded behavior.

### Milestone 1: Extract request state

- Add immutable request options.
- Add request-local random state.
- Make cache and attention capture request-local.
- Keep the public API unchanged.

Exit condition: forced concurrent interleavings do not contaminate results.

### Milestone 2: Separate and reuse kernels

- Separate encode, prefill, decode step, and alignment.
- Add `EncodedWindow`.
- Reuse encoder output across language detection, fallbacks, and alignment.

Exit condition: reference outputs pass and redundant encoder work is removed.

### Milestone 3: Add the bounded worker

- Add resource estimates and reservations.
- Add bounded stage queues.
- Add cancellation and deadlines.
- Add typed admission results.

Exit condition: overload does not cause unbounded memory growth or device out-of-memory errors within the declared capacity.

### Milestone 4: Add safe batching

- Batch compatible encoder windows.
- Add explicit hypothesis-to-window ownership.
- Support correct multi-audio beam groups.
- Add token-step decoder scheduling for compatible cohorts.

Exit condition: batched results meet the selected compatibility profile and all beam groups remain isolated.

### Milestone 5: Add managed cache memory

- Add cache slots or pages.
- Share cross-attention state.
- Add copy-on-write beam prefixes.
- Add request quotas and quiescence-aware release.

Exit condition: measured peak memory stays within the admitted bound.

### Milestone 6: Add streaming sessions

- Add bounded audio ingestion.
- Add provisional and commit events.
- Add watermarks, revision rules, and client credits.
- Keep offline and streaming profiles separate.

Exit condition: committed text never changes and session memory does not grow with total stream duration.

### Milestone 7: Add backend adapters

- Add the capability interface.
- Keep the legacy backend.
- Add compiled or exported backends one at a time.

Exit condition: every backend passes the conformance tests for each declared capability.

### Milestone 8: Consider a default change

The native runtime can become the default only after:

- all compatibility tests pass;
- performance results are reproducible;
- third-party extension behavior is documented;
- one release provides an opt-out path;
- no open critical correctness issue remains.

## Non-goals

This RFC does not:

- change Whisper model weights;
- define a new training process;
- claim true causal streaming for current checkpoints;
- add speaker identification or diarization;
- add semantic interpretation of transcripts;
- add domain-specific applications;
- add arbitrary language-to-language translation;
- define a distributed multi-node scheduler;
- replace third-party Whisper implementations;
- guarantee bitwise identity across hardware;
- use output confidence as a correctness guarantee.

## Open questions

1. Which behaviors require exact token equality in the reference profile?
2. Which cache layout gives the best result for Whisper's large cross-attention state and short text context?
3. Which alignment heads can be captured directly without a material output change?
4. How much encoder look-ahead gives useful throughput without increasing long-form peak memory?
5. Which legacy extension patterns require permanent serialization?
6. What queue cost unit best predicts device time across model sizes?
7. Which stable-streaming normalization gives the best accuracy and revision rate?

These questions require measurement. They do not change the ownership and admission rules in this RFC.
