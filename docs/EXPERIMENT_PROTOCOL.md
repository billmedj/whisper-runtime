# CUDA experiment protocol

This protocol defines the staged Modal experiment program. It separates
functional qualification, fault testing, and performance measurement. Passing
one class does not imply that another class passes.

The current
[`modal-native-cuda-qualification.schema.json`](../evidence/modal-native-cuda-qualification.schema.json)
is qualification-only despite its historical filename. Its fixed cell is
registered in
[`experiments/native-cuda-qualification-v6.json`](../experiments/native-cuda-qualification-v6.json).
The local validator binds that tracked file by path, digest, and runtime commit,
but cannot prove that it was public before execution. A public Git host or
another independent service must supply that timestamp.

The [evidence index](../evidence/README.md) records all attempts and their
limits. Version six has one passing
[qualification record](../evidence/modal-t4-tiny-en-jfk-native-cuda-qualification-v6-2026-09-04.json)
and one append-only
[attempt receipt](../evidence/modal-native-cuda-qualification-v6-attempt-2026-09-04.jsonl).
This completes the first fixed T4 qualification cell. It does not complete the
performance matrix.

A performance campaign requires a new evidence schema or version. It must
require the full metrics in this protocol, matched probes on the control and
runtime paths, repeated fresh workers, raw observations, registered aggregation
rules, and uncertainty estimates.

## Fixed comparison

Each performance cell compares two paths from the same clean source revision:

1. the patched Whisper backend without the transaction wrapper; and
2. `NativeWhisperAdapter` with the strict single-lane CUDA profile.

Both paths MUST use the same checkpoint, decoded audio tensor, decode options,
random seed, numerical precision, device, and process image. Network access
MUST be disabled during measured work. A functional cell passes only when both
paths meet its declared output-compatibility rule.

Qualification and fault runs keep the full event instrumentation. Performance
runs use identical measurement hooks on both paths. Hooks MUST remain outside
the timed interval unless the same hook executes at the same boundary on both
paths. An asymmetrically instrumented run is qualification evidence, not a
performance comparison.

## Matrix

Run the matrix in this order and stop expanding it when a lower-cost stage
fails:

1. Local schema, validator, source-state, and controlled-fault tests.
2. One T4 qualification run with `tiny.en` and the pinned JFK fixture.
3. Repeated T4 runs with `tiny.en`, then `base.en`.
4. One independent L4 or A10G replication of each passing model cell.
5. `small.en`, then `turbo`, only after the smaller-model cells pass.
6. One A100 replication after the model and workload matrix is stable.
7. Fault and concurrency cells after the corresponding single-request cell
   passes. CUDA concurrency requires a separate profile and claim boundary.

The initial audio corpus is:

- the repository's pinned JFK fixture;
- a frozen, hashed subset of LibriSpeech `test-clean` for English speech;
- a frozen, hashed multilingual FLEURS subset before multilingual models are
  claimed;
- synthetic silence and exact boundary-duration fixtures.

Every subset MUST have an immutable manifest that records upstream identity,
license, selected item identifiers, original byte hashes, decoded PCM hashes,
sample rates, channel counts, and durations. Do not publish source audio when
its license does not permit redistribution.

## Repetitions and warm-up

A qualification cell uses one worker, two unmeasured warm-up pairs, five
measured pairs, three cancellation runs, and two repetitions at each of the
four registered fault points. Its p50 and p95 summaries are diagnostic. Its p99
is `not_estimated`. It establishes wiring and invariants, not performance.

A benchmark cell uses five unmeasured warm-up iterations per process and at
least 100 measured iterations on each of three fresh workers. Report p50 and
p95 from the raw samples. Report p99 only after an expanded cell contains at
least 1,000 measured iterations; otherwise report it as `not_estimated`.
Workers, failed allocations, retries, and excluded observations MUST remain in
the run ledger. An infrastructure failure may be excluded only by a rule fixed
before the run.

Alternate the control and runtime order between workers. Synchronize the device
before each measured interval. Do not include image build, model download, or
worker allocation time in inference latency; report them separately.

Confidence intervals use a two-level bootstrap: resample workers, then resample
runs within each selected worker. Use 10,000 resamples, a fixed seed derived
from the cell profile digest, and the 2.5th and 97.5th percentiles. Report the
worker count with every interval. These intervals describe the sampled worker
population; three workers cannot establish fleet-wide behavior.

## Measurements

Record raw observations before computing summaries. At minimum, record:

- admission, queue, encoder, prefill, decoder, finalization, cleanup, fence,
  publication, and end-to-end wall times;
- p50, p95, and eligible p99 latency with sample counts and bootstrap confidence
  intervals;
- audio seconds processed per wall second and generated tokens per second;
- cancellation request to acknowledgement, cancellation request to backend
  quiescence, and quiescence to lease release;
- CUDA memory allocated and reserved before the run, peak allocated and
  reserved memory, device free memory, and the post-fence and post-recovery
  baselines;
- admission result, terminal state, queue depth, lease state, retained
  capacity, quarantine duration, and recovery outcome;
- output text, tokens, timestamps, and model-state digest as required by the
  cell's compatibility rule.

CUDA memory readings MUST be taken after explicit synchronization at the
defined observation boundaries. The declared resource vector remains an
accounting value until a separate test demonstrates that it bounds observed
physical use.

## Fault matrix

The initial qualification contract requires four harness faults:

- decoder cleanup;
- completion-event creation;
- completion-event record; and
- completion-event synchronization.

For each fault, the record MUST show no publication or session-version change,
a held lease and retained budget while quiescence is unknown, rejection of new
work on the bound model, recovery to an aborted transaction after a new
completion fence, restored capacity, and one successful reuse transaction.

The extended matrix adds one named fault per run at these boundaries:

- after admission and before stream creation;
- after stream creation, run creation, encoder work, prefill, and a token step;
- during finalization and cleanup;
- during completion-event creation, record, and wait;
- after quiescence and before session publication;
- during session publication and lease release;
- during recovery; and
- by terminating the worker process while work is admitted.

Each fault record MUST state whether the fault occurred before or after a real
backend call. A harness fault MUST NOT be described as a CUDA driver, hardware,
or process failure. Process-loss claims require an external observer and a
durable recovery record.

The extended matrix is not part of the initial qualification contract. Process
termination remains disabled until durable recovery and an external observer
are implemented. A record MUST list every required fault point and MUST fail
validation if it omits one.

## Falsifiers

A single observation falsifies the current correctness claim for its exact
profile if it shows any of the following:

- publication before the completion fence or publication after acknowledged
  cancellation;
- release or reuse of capacity while backend quiescence is unknown;
- a missing, duplicated, or cross-owned lease;
- cross-request cache, random-state, hook, or model-state contamination;
- a committed output outside the declared compatibility rule;
- successful recovery without a new completed fence; or
- a validator accepting a record with an invalid event order or source
  identity.

Correctness gates have zero tolerance: one safety violation fails its exact
cell. The first performance matrix is descriptive. It measures paired runtime
and control ratios and estimates between-worker variation. It does not label a
cell acceptable from an arbitrary percentage.

After the pilot, register operational thresholds before confirmatory runs. The
registration MUST define the paired latency-ratio estimator, memory quantity,
allowed post-fence drift, admission margin, exclusion rules, and decision rule.
A failed threshold remains a reportable result.

## Provenance

Every record MUST bind:

- runtime commit and Git tree, clean-worktree result, and critical-file hashes;
- patched Whisper commit, tree, patch-manifest digest, and patch-file hashes;
- container image identity and the Python distribution metadata inventory;
- Python, PyTorch, CUDA runtime, CUDA driver, and cuDNN versions;
- provider, region, GPU type, GPU memory, and device capability;
- checkpoint name and SHA-256 digest;
- input manifest, original bytes where available, decoded PCM digest, and
  preprocessing options;
- decode options, seed, profile, script version, schema version, and validator
  version; and
- UTC timestamps plus a sanitized provider allocation identifier.

Credentials, user names, local paths, temporary URLs, and account identifiers
MUST NOT appear in a published record.

The dependency inventory stores one metadata record returned by
`importlib.metadata.distribution(name)` for each normalized name discovered by
`importlib.metadata.distributions()`. It does not prove the bytes of imported
modules. Source and build-input hashes are the integrity anchors for the
tracked implementation and image recipe. The producer MUST collect this
inventory before GPU work. It MUST record the platform-injected Modal module
version independently, even when the inventory contains no Modal distribution.
It MUST require exact Torch module and distribution equality because Torch
executes the workload.

## Publication rules

1. Freeze the matrix, thresholds, schemas, and exclusion rules before paid
   execution.
2. Write each run to a new artifact path. Never overwrite a raw record.
3. Validate schema, cross-field semantics, source identity, and secret/path
   sanitization before review.
4. Publish raw observations with derived summaries and the exact aggregation
   program. Do not publish summaries alone.
5. Publish all completed registered cells, including failures and negative
   results. Do not select only favorable workers or iterations.
6. Bind each claim to the exact models, inputs, devices, dependency image, and
   concurrency level that ran.
7. Use a new schema or evidence version when the harness, trace contract,
   workload, or claim boundary changes. Performance evidence must not use the
   current qualification-only schema.
8. Commit evidence only after independent review of the record and workflow
   log. A committed record is append-only; corrections require a superseding
   record with an explicit reason.

Passing this protocol does not establish production readiness, general CUDA
correctness, improved transcription accuracy, or performance on an untested
configuration.
