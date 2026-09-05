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
| D1 | Continuous transcription with progressive commits | Experimental rolling profile implemented; long-session gate open | D0 |
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

Evidence includes 330 runtime tests, 249 repository-tool tests, 55 Lean theorem
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

**Status: experimental profile implemented; acceptance gate remains open.**

The [timestamp agreement stream](CONTINUOUS_STREAMING.md) implements growing
native hypotheses, prefix publication, bounded rolling PCM, thread-safe input
admission, and explicit backpressure. Deterministic tests cover rolling windows,
chunk partitions, cancellation, and retained-resource recovery. Gaps and
unstable prefixes can still stop this conservative profile. These tests do not
close the real-audio 30-minute gate below.

An opt-in context profile now separates retained audio from committed output.
Its 18 additional scripted tests pass. The recovered four-cell T4 comparison
shows that longer holdback removes two word edits on one synthetic input, but
both retained-context cells stop at EOF after publishing only 7.44 seconds.
See the [results and next gate](research/2026-09-05-boundary-comparison-results.md).

Text-token agreement analysis and optional preview coalescing are implemented.
The analyzer has no publication authority. Coalescing preserves admitted input
across retries and is off by default. Its GPU benefit remains unmeasured.

An opt-in word-aligned profile now connects native word estimates, prefix
selection, transactional publication, and audio progress. Its 52 additional
tests cover boundary drift, repeated phrases, rolling input, and recovery.
The pre-repair runtime suite has 382 passing tests. Default segment profiles remain
unchanged. Alignment adds model work; matched GPU cost and long-session behavior
are still open gates. See [configuration and limits](CONTINUOUS_STREAMING.md#optional-word-alignment).

The [matched T4 diagnostic](research/2026-09-05-word-alignment-comparison.md)
does not complete: the segment profile publishes 8.00 seconds and the word
profile 5.70 seconds of 33 seconds. A partially retained anchor word blocks the
word profile. Keep this result distinct from the successful CPU smoke and
validate the anchor repair before a long-session run.
The repair is covered by two additional local tests (384 total). The
[T4 replay](research/2026-09-05-word-alignment-replay.md) confirms progress through
12.78 seconds, including the repaired boundary. A window-leading word start
estimate then prevents the next anchor match. EOF completion remains unproven
for this word-aligned configuration.
The window-edge matching repair is covered by four additional local tests
(388 total). It preserves the full anchor text, word ends, and neighboring
bounds. The [next T4 replay](research/2026-09-05-word-alignment-completion.md)
completes the word profile's 33 seconds with eight commits and restored capacity.
Its 66 normalized words match the model controls, but exact strings differ.
The segment profile still stops at 8 seconds. The last word commit covers
11.26 seconds at EOF; paced latency and the long-session gate remain open.

The [independent-reference diagnostic](research/2026-09-05-word-corpus-diagnostic.md)
completes three individual LibriSpeech clips. The mixed input and digital
silence expose two publication failures: punctuation can advance coverage past
untranscribed speech, and repeated hypotheses can agree on a hallucination.
The [punctuation replay](research/2026-09-05-input-evidence.md) confirms that
the invalid eviction is closed. The mixture still lacks a stable lexical suffix
and stops with audio retained. An opt-in input-evidence policy now distinguishes
digital silence from uncertain output and commits empty silence coverage through
the existing transaction. Its T4 comparison completes 32 seconds of digital
silence with empty text and preserves the three short speech transcripts.
The mixed input remains unresolved. General acoustic coverage is the next gate before
long-session qualification. Sample accounting alone does not prove
that speech content was preserved.

The [caller-delimited source-unit diagnostic](research/2026-09-05-source-units.md)
completes the same 43.660-second mixture under a separate profile. Eleven
source-owned units preserve all six speech occurrences and five empty pauses.
Normalized words match the model controls; six human-reference edits remain.
The boundaries come from fixture construction, not a detector. This closes the
known-boundary integration test, not the general acoustic or long-session gate.
That iteration passed 458 runtime and 287 repository-tool tests.

The [automatic-endpoint diagnostic](research/2026-09-05-automatic-endpoints.md)
then completes the same mixture without supplied boundaries. A small opt-in
quiet-run detector feeds the existing controller; it adds no model dependency
and cannot authorize empty publication. Five inferred endpoints and EOF cover
all input. Text matches both model controls exactly. The current suites pass
481 runtime and 291 repository-tool tests. Noisy pauses, weak speech, continuous
speech without gaps, paced latency and the 30-minute gate remain open.

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

Three licensed English references, a constructed speaker/pause sequence, and
digital silence now have frozen inputs and one T4 record. This small sample
does not cover multilingual speech, noise, natural conversation, or long sessions.

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

- [x] Add opt-in preview coalescing with bounded state and immutable retries.
- [ ] Measure avoided work and quality changes on matched paced input.
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
