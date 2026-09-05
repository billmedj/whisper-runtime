# Delivery milestones

Updated: 2026-09-05.

The project has two goals: control inference execution, and make continuous
Whisper transcription practical. Lower compute cost is a target to measure,
not an established property of the current runtime.

This page defines delivery gates. The [architecture RFC](rfcs/0001-state-resource-execution.md)
remains the design reference; its numbered implementation steps are not release
status. D0-D7 below are delivery identifiers, not version tags or dates.
A gate closes only when its acceptance cases and results are committed.

## Status at a glance

| Gate | Deliverable | Status | Depends on |
| --- | --- | --- | --- |
| D0 | Governed decoding and timed publication | Validated within the recorded pre-alpha scope | None |
| D1 | Continuous transcription with progressive commits | Next; prerequisites implemented | D0 |
| D2 | A usable local live-transcription entry point | Planned | D1 |
| D3 | Broader quality and failure coverage | Initial coverage; expand alongside D1-D2 | D0; release gate for D2 |
| D4 | Measured compute and memory improvements | Not demonstrated | D1 and matched D3 baselines |
| D5 | A reproducible developer release | Package builds; release gates remain open | D2 and D3; D4 for efficiency claims |
| D6 | Durable recovery and finer resource scheduling | Later | D3 and a recovery contract |
| D7 | Multiple channels, translation, and a second backend | Later | D1-D3 and per-output contracts |

## D0. Governed decoding and timed publication

**Status: validated within the recorded pre-alpha scope.**

- [x] Isolate request options, random state, and decoder state.
- [x] Admit work under declared resource and queue limits.
- [x] Expose token-step progress, cancellation, cleanup, and recovery.
- [x] Preserve versioned commits and the committed-prefix boundary.
- [x] Accept bounded PCM input and emit whole-prefix preview revisions.
- [x] Retain immutable tokens, language, scores, and predicted segment times.
- [x] Inspect a result without repeating decoder finalization.
- [x] Select complete segments from a larger analysis span without rewriting
  already committed output.

Evidence includes 217 runtime tests, 198 repository-tool tests, 55 Lean theorem
declarations for the abstract protocol, built-package tests, CPU runs, and a
narrow T4 qualification. The [evidence index](../evidence/README.md) defines each
record's scope. The [timed-publication smoke](../evidence/native-cpu-tiny-en-jfk-timed-publication-2026-09-05.json)
compares two publications with the same full-window control on one JFK input.

Limits:

- Pause retains process state and reserved resources until the existing deadline.
  It is not a durable checkpoint or a way to release a GPU mid-run.
- The [bounded-preview profile](BOUNDED_STREAMING.md) accepts at most 30 seconds
  and commits at EOF. It is not continuous transcription.
- The caller chooses the publication span and asserts finality, including gaps.
  Predicted timestamps do not prove word accuracy or silence.
- Lean covers an abstract protocol, not all Python, PyTorch, or CUDA execution.
  The T4 records do not qualify every model or GPU.

## D1. Continuous transcription with progressive commits

**Status: next.**

Deliver a separately named continuous profile. Keep the existing
offline-compatible and bounded-preview paths.

Implementation order:

1. Carry timed native hypotheses into the stream policy.
2. Implement and version a reference agreement policy. Compare successive
   hypotheses and commit the prefix that meets the policy's conditions. Document
   the source method and any changes when the policy is selected.
3. Account for silence, incomplete words, and gaps before advancing the committed
   boundary. A timestamp alone is not a silence detector.
4. Add rolling audio retention and bounded text context. Evict audio only when
   the policy no longer needs it for uncommitted output.
5. Separate input admission from the model-work owner. Bound pending input and
   event queues, and return explicit backpressure when they are full.
6. Skip obsolete preview computations where useful; do not silently discard
   audio that the runtime has accepted.

Acceptance gate:

- [ ] Replay at least 30 minutes at wall-clock audio speed, including speech,
  pauses, and boundaries that cut through words.
- [ ] Commit text before EOF. Previously committed text and segment identities
  remain unchanged after window shifts, retries, and cancellation.
- [ ] Account for accepted samples in the input timeline, with no gaps caused by
  silent loss. Rejected chunks leave sequence state unchanged and can be retried.
- [ ] Record configured buffer limits and observed high-water marks. Application
  buffers stay within their limits; measure process memory separately.
- [ ] Exercise slow decoders and consumers. Report delay or backpressure without
  unbounded queues, silent loss, or deadlock.
- [ ] Repeat deterministic replays with different chunk partitions and test the
  output equivalence promised by this profile.

D1 establishes continuous operation, not a quality or speed advantage over
another streaming system.

## D2. A usable local live-transcription entry point

**Status: planned after D1.**

Deliver a small Python API and one local command, with microphone input and paced
file replay. Do not require a server deployment or a desktop application.

- [ ] Document input format, model selection, provisional and committed results,
  cancellation, and failure behavior.
- [ ] Show readable partial captions with explicit corrections and finality.
- [ ] Export committed subtitles with source times. Distinguish a final subtitle
  file from a provisional event stream.
- [ ] Handle microphone disconnection, EOF, cancellation, and device errors without
  claiming success or losing the committed transcript.
- [ ] Verify that a new user can install, run the example, transcribe a microphone
  session, and stop it with resources released.

Record the tested OS, Python version, backend, and model. Passing on Windows
alone is not cross-platform qualification.

## D3. Quality and failure coverage

**Status: initial coverage exists; expand alongside D1 and D2.**

Keep the four-case compatibility corpus and recorded CPU/T4 cases. Add licensed
inputs with reference transcripts. Record hashes, options, model identities,
seeds, environments, and outputs.

- [ ] Cover English and actual French and Spanish speech, multiple speakers,
  accents, noise, silence, names, and long sessions.
- [ ] Extend sampling, beam search, timestamps, cancellation, and failures beyond
  one input and model size.
- [ ] Measure recognition errors, revisions, time to first and stable output,
  queue delay, and memory. Report distributions and sample counts.
- [ ] Run a four-hour bounded-memory soak on a declared profile. Check retained
  history and queues as well as host/device memory.
- [ ] Inject failures around input admission, decoding, result preparation,
  publication, cleanup, and EOF. Reject stale or duplicate publication.
- [ ] Register acceptance thresholds and matched baselines before measured runs.
  Retain regressions and excluded cases in the report.

The current Whisper `translate` fixture uses English input. It checks mode
compatibility, not interlanguage or simultaneous translation.

The [CUDA registration](../experiments/native-cuda-qualification-v6.json) and its
[passing record](../evidence/modal-t4-tiny-en-jfk-native-cuda-qualification-v6-2026-09-04.json)
remain fixed. New GPU runs need the separate registration and scope required by
the [experiment protocol](EXPERIMENT_PROTOCOL.md). This roadmap update does not
authorize or start new GPU spending.

## D4. Measured compute and memory improvements

**Status: not demonstrated.**

Remove unnecessary work before adding scheduling or caching complexity.

- [ ] Coalesce obsolete previews and measure the work avoided.
- [ ] Reuse preprocessing or encoder output only when input identity and the
  chosen profile permit it. New audio does not make Whisper's noncausal encoder
  cache append-only.
- [ ] Schedule and batch compatible work without sharing mutable request state
  or mixing outputs.
- [ ] Compare fixed-window computations separately from live services. Match
  hardware, model, quality settings, and workload.
- [ ] Measure CPU/GPU time, peak memory, throughput, and caption delay over
  repeated runs, including cancellation and overload.

Close D4 only when a published workload shows a reproducible improvement within
predeclared quality and latency limits. State where it does not help. Declared
resource reservations are not measurements of actual hardware use.

## D5. A reproducible developer release

**Status: source and wheel builds pass; release gates remain open.**

- [ ] Publish a documented Python API and CLI with a small configuration surface.
- [ ] Verify clean installation on the supported platform matrix, including the
  pinned backend setup and an explicit model-download step.
- [ ] Pass D1-D3 and package checks in CI for the release commit. Publish the
  tested capability matrix and known limits.
- [ ] Document exact result-type and serialization changes. Keep the legacy path
  and provide an opt-out before changing defaults.
- [ ] Resolve critical correctness issues and document upstream extension
  compatibility, licenses, and third-party notices.

D4 is required for efficiency claims, not for an honestly labeled functional
release. A tag or package upload alone does not close this gate.

## D6. Durable recovery and finer resource scheduling

**Status: later research and engineering work.**

- [ ] Define what survives a process failure: accepted audio, committed output,
  provisional state, and decoder state need different recovery contracts.
- [ ] Add a durable input/event journal and reject publication from an obsolete
  worker after recovery.
- [ ] Kill and restart workers around commit boundaries. Verify the declared
  no-loss and no-duplicate guarantees against input and event identities.
- [ ] Distinguish resident pause, reconstruction by replay, and portable
  checkpoints. Test each capability before advertising it.
- [ ] Evaluate finer resource leases and cache quotas. Do not free ownership
  while backend work can still access the resources.

Acceptance requires crash tests for the declared recovery profile. Cross-device
bitwise equivalence is not assumed. This gate is separate from local live use.

## D7. Multiple channels, translation, and backend coverage

**Status: later; not required for the first single-stream release.**

- [ ] Run independent audio streams with bounded admission and explicit scheduling.
  Measure the effect of an overloaded client on other streams.
- [ ] Give transcription and translation separate revision and commit rules.
  Invalidate provisional translations when their source changes.
- [ ] Test real interlanguage inputs and each advertised target language. Declare
  any additional translation model and its resource requirements.
- [ ] Preserve source timing for captions and live or generated media. Add speaker
  attribution only under a separate tested contract.
- [ ] Implement a second backend against the ownership, lifecycle, and result
  contract. Reject capabilities it cannot support.

Acceptance requires reproducible multi-stream and translation cases and a
published capability matrix. A second backend must pass its own conformance
suite; a wrapper alone is not sufficient evidence.
