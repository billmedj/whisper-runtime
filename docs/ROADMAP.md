# Delivery milestones

Updated: 2026-09-07.

The project has two goals: control inference execution, and make continuous
Whisper transcription practical. Same-window encoder reuse now avoids measured
work; lower live-service cost remains a target to validate.

This page defines delivery gates. The [architecture RFC](rfcs/0001-state-resource-execution.md)
remains the design reference; its numbered implementation steps are not release
status. D0-D7 below are delivery identifiers, not version tags or dates.
A gate closes only when its acceptance cases and results are committed.

## Status at a glance

The first [four-hour native endurance attempt](research/2026-09-07-release-soak-results.md)
stopped after the short smoke stage: transcription and cleanup passed, but
process RSS exceeded its registered limit. No hourly stage ran. The next release
check must first distinguish startup/framework memory from session growth; the
recorded failure cannot be changed by adjusting a threshold after the run.

The [two-session attribution diagnostic](research/2026-09-07-memory-attribution-results.md)
now completes both transcripts with identical exports and terminal CUDA allocation,
but its detailed accounting fails because smaps_rollup is unavailable. Most guest
RSS growth appears at imports/initial execution; the second session adds 4.55 MiB.
This is not verified host RAM or endurance. A separately registered short capacity
test with an enforced limit or suitable host telemetry must precede a longer run.

The [separate capped smoke](research/2026-09-07-capacity-smoke-results.md) then
stopped during its first session on source-clock lateness. It did not complete
capacity qualification. Cleanup passed and the app is stopped. The next local
step isolates source scheduling and cold alignment before another GPU attempt.

The subsequent [instrumented T4 replay](research/2026-09-07-source-clock-replay-results.md)
passes two short sessions under the same submitted 4 GiB limit. Coverage, exports
and terminal CUDA memory match. Source lateness is 222.77 ms in session 1 and
2.13 ms in session 2. The narrow initial margin and the earlier failure still
need attention; four-hour endurance and release-platform gates remain open.

The latest [local hardening](research/2026-09-07-completion-and-startup-hardening.md)
fixes completion races, preserves the final microphone frame during concurrent
shutdown, and aligns harness imports with the CLI's verified startup path.
Local suites and the installed CPU command pass. The separate
[verified-startup T4 replay](research/2026-09-07-verified-startup-results.md)
now passes two complete sessions; four-hour endurance is still open.

The [low-latency candidate](research/2026-09-07-low-latency-cpu-results.md)
confirms the first speech at 4/6 seconds of admitted audio on two CPU fixtures,
without token drafts or relaxed publication checks. Same-window reuse preserves
candidate text and halves encoder forwards. Its first T4 check confirms text at
7.746 seconds but fails at a final source boundary. The
[two-observation correction](research/2026-09-07-two-observation-holdback.md)
completes the same input on CPU with first confirmation after 6 seconds of
admitted audio and the standard control's word-edit count. The subsequent
[source-paced v2 T4 check](research/2026-09-07-low-latency-v2-gpu-results.md)
passes both sessions: first confirmation at 7.358/6.186 seconds, full coverage,
identical exports and unchanged terminal CUDA allocation. This closes the
short check, not endurance or default-profile promotion.

The installed SDK and CLI now share named execution profiles. `conservative-v1`
preserves the standard settings. `experimental-optimized-v1` exposes retained
context, same-window alignment reuse and verified token drafts; it requires the
matching patched backend. Remote servers configure their own profile.

The latest [native draft checks](research/2026-09-07-draft-qualification-results.md)
record 45.9% fewer decoder calls and 23.5% less decode-phase wall time on two
project-new short recordings on T4. Text and commit spans match. A separate
CPU test restores logical state in a fresh process, then recomputes decoder
state. Neither result qualifies durable GPU-state migration or general speed.
Drafts remain experimental because token parity does not guarantee score parity
at publication thresholds.

The earlier [matched T4 comparison](research/2026-09-07-composed-gpu-results.md)
connects earlier publication and same-window encoder reuse. First COMMIT arrives
at 10.245 seconds versus 28.439 seconds; peak allocated memory falls 22.77%.
Reuse preserves exact fast-arm commits and removes 18 encoder forwards. Its
combined forward interval sum falls 5.97% versus fast legacy alignment but
remains 10.92% above the conservative control. The longer combined-profile
qualification remains open; defaults are unchanged.

The earlier [short T4 candidate](research/2026-09-06-retained-context-commits.md)
separates text publication from audio-window movement. First commit arrives at
13.220 seconds versus 31.156 seconds in the earlier smoke run, with unchanged
word-edit count. Same-origin alignment reuse and checkpoint continuation are
implemented. The candidate context settings need a longer matched replay before
they replace the CLI defaults; this latency gain is not a compute-saving claim.

The [single-stream CLI](CLI.md) now accepts WAV/PCM or optional microphone input
and exports committed TXT, SRT and VTT. The same command connects to an
authenticated live-v2 server without installing PyTorch on the client. A clean
Windows Python environment, a separate backend checkout and the cached model
complete the 11-second JFK file outside the source checkout. This is not a clean
operating-system installation. Microphone tests use scripted capture.

The [deferred-commit candidate](research/2026-09-06-deferred-word-commits.md)
completes both the 33.66-second continuous clip and a 10.89-second noisy prefix
on CPU. A four-hour accelerated controller soak completes with bounded history
and buffers; recognition is scripted in that test. The separate
[30-minute source-paced T4 run now passes](research/2026-09-06-live-v01.md), with
90,000 chunks, 118 commits and verified terminal input coverage and cleanup.
The source repeats a short cycle; broader acoustic and release-platform checks
remain open. See [V0.1 status](V01_STATUS.md) for the current release checklist.

Earlier GPU work: the opt-in EOF context retry completes the previously
blocked stream [at source speed on T4](research/2026-09-06-paced-context-retry.md):
21.26 to 33.66 seconds committed, with one additional native window and unchanged
strict publication checks. Matched native observations and the published prefix
stay fixed. The final event arrives 326.527 ms after input EOF. The noisy-prefix
case still refuses and retains audio without a redundant decode. A subsequent
[fixed-audio prompt comparison](research/2026-09-06-noisy-context-prompt.md)
recovers its missing next utterance but not its publication boundary. Both
decoder-history and crop effects are now observed; a saved earlier joint
observation supplies the next local continuity test. That
[read-only replay](research/2026-09-06-continuity-witness.md) now finds the exact
published anchor and matching lexical tokens, but still refuses differences in
declared identity, punctuation, timing, and a word cut by the overlap boundary.
It does not authorize publication or audio retirement. These earlier diagnostics
remain historical evidence for their named profiles; the newer deferred-commit
result above does not change their outcomes.

| Gate | Deliverable | Status | Depends on |
| --- | --- | --- | --- |
| D0 | Governed decoding and timed publication | Validated within the recorded pre-alpha scope | None |
| D1 | Continuous transcription with progressive commits | Registered 30-minute T4 run passes; broader acceptance coverage remains open | D0 |
| D2 | A usable local live-transcription entry point | CLI and optional capture implemented; short CPU file run passes; physical microphone gate open | D1 |
| D3 | Broader quality and failure coverage | Initial coverage; expand alongside D1-D2 | D0; release gate for D2 |
| D4 | Measured compute and memory improvements | Encoder reuse and token drafts measured on short paced inputs; broad quality and efficiency gates open | D1 and matched D3 baselines |
| D5 | A reproducible developer release | Clean Windows Python native file run passes; platform and release gates remain open | D2 and D3; D4 for efficiency claims |
| D6 | Durable recovery, GPU-free suspension, and compatible migration | Logical savepoint restored in a fresh CPU process; durable delivery, crash recovery and token-state migration remain open | D0, a recovery contract, and D3 failure tests; stream recovery integrates with D1 |
| D7 | Multiple channels, translation, and a second backend | Later | D1-D3 and per-output contracts |

## Execution order

Keep the existing gate identifiers. Their numbers do not specify execution order.
Durable recovery is now a core requirement, not an optional follow-up. This
revision changes priorities; it does not qualify new runtime capabilities.

For **V0.1**, prioritize the installed single-stream path and its D1-D3/D5
qualification before extending durable token state. V0.2 targets independent
channels; V0.3 targets translation. Those version targets do not close the
broader D6 recovery requirements or imply untested language support.

GUI delivery uses the proposed [application boundary](rfcs/0002-application-boundary.md):
one engine, a versioned local application API, and a browser interface. It follows
engine qualification and does not require moving decoder logic into the frontend.

1. **Complete a bounded live recovery path and define durable session state
   (D1, D3, D6a).** Preserve the successful strict path. Test one alternative
   decode only for the retained unresolved span. Separate lexical failures from
   timing and representation differences. Use the same input identities,
   committed-prefix boundary, and retained audio in the session checkpoint.
   Develop CPU persistence and crash tests while the audio repair is validated.
   A saved session preserves unresolved work; it does not fix recognition.
2. **Resume saved computation in a new process (D6b, D6c).** First qualify CPU
   greedy decoding after prefill and after a token step. Then cover sampling
   and beam state. Follow with a short, registered T4 shutdown-and-restore test.
   Keep durable replay as a separate recovery path when exact-state restore is
   unavailable. Do not label replay as continuation without recomputation.
3. **Measure useful work per resource budget (D4, D3).** Reuse the recovery
   corpus for matched quality, delay, memory, and compute measurements. Measure
   checkpoint size and save/restore overhead before adding an offload policy.
   Compare a fixed policy with a bounded adaptive candidate on held-out inputs.
   Add no general scheduler until the measured workload justifies one.
4. **Qualify and package continuous use (D1-D3, D5).** Preserve the passing
   30-minute paced result and extend coverage. Include disconnects and backpressure;
   qualify pause/resume separately from uninterrupted caption latency. Deliver
   a small API and CLI before a GUI. Keep the broader four-hour soak gate.
5. **Extend the qualified contract (D7).** Add independent channels, translation,
   and a second backend through separate conformance cases. Do not put these
   features on the critical path for the first single-stream release.

Use licensed, fixed inputs and explicit baselines. Reuse recorded controls only
when their input and execution identities match; use new speakers or recordings
for held-out checks. A second observation from a tuning case is not a new test
case. Planned work here does not start GPU jobs or change the spending budget.

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

**Status: registered 30-minute profile passes; broader acceptance gate remains open.**

Current evidence is in the [V0.1 live result](research/2026-09-06-live-v01.md).
The following diagnostics describe earlier profiles and their recorded limits;
they are not instructions to rerun already completed work.

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
all input. Text matches both model controls exactly. That iteration passed
481 runtime and 291 repository-tool tests. Noisy pauses, weak speech, continuous
speech without gaps, paced latency and the 30-minute gate remain open.

The [paced T4 replay](research/2026-09-05-paced-replay.md) completes the same
43.660-second mixture at source speed. An independent producer supplies 20 ms
chunks while decoding continues. First text appears at 2.097 seconds; five
commits precede EOF. All samples are committed, the final buffer is empty and
text matches both model controls exactly. That iteration passed 500 runtime and
296 repository-tool tests. Local tests cover slow decoding and consumers,
input lag, overload, cancellation and delivery failures. This short same-worker
test does not close the acoustic, network or 30-minute gates below.

The [PC-to-Modal replay](research/2026-09-05-network-replay.md) then sends the
same audio over an authenticated WebSocket. First text returns to the Windows
client at 2.426 seconds; the native final event arrives 613.573 ms after EOF
send completion. Full input coverage and exact control text are preserved.
That iteration passed 514 runtime and 332 repository-tool tests. The short
network integration passes; microphone, acoustic and long-session gates remain
open. No GPU efficiency advantage is established.

The [hybrid acoustic diagnostic](research/2026-09-05-acoustic-boundaries.md)
combines word-prefix agreement with quiet endpoints in an opt-in profile.
The normal and attenuated cells complete, but both inputs without detected
pauses still fail. Concatenated speech commits 21.26 of 33.66 seconds; noisy
input commits 6.10 seconds before stopping. No accepted input is silently lost
and no false final event is emitted. The run used 534 passing runtime tests and
352 repository-tool tests. Next, make context retention preserve complete
published anchors and test whether this also avoids missing native continuation.
Do not advance to long-session qualification or claim general noise tolerance
from this failed diagnostic.

The [bounded word-context comparison](research/2026-09-05-word-context.md)
then preserves complete estimated words and multiple lexical anchor witnesses
before a partial commit. Normal and attenuated candidates complete. Noise
advances from 6.10 to 33.54 seconds but stops on a 540 ms anchor-end mismatch.
The no-added-pauses case remains at 21.26 seconds with a missing native suffix.
Matched controls reproduce their failures. The candidate remains unqualified
and opt-in. Distinguish timestamp uncertainty from missing recognition before
the next acoustic run; do not advance to long-session or efficiency claims.

The [local resolution work](research/2026-09-06-word-resolution.md) adds structured
anchor diagnostics and a read-only experimental matcher. It leaves publication,
PCM retention and model execution unchanged. This separates the two observed
failure types without enabling automatic recovery. Bounded counterfactuals are
the next stage before a live routing policy.

The [terminal-window experiment](research/2026-09-06-word-resolution.md#t4-result)
now records that comparison: two saved failures and two new speakers. The frozen
selector improves three candidate texts relative to incomplete strict baselines;
the fourth remains unresolved despite a better measured alternative. All ten
native windows release their leases, and model parameters remain unchanged.
No result is a complete live-stream recovery. Next, test fallback selection for
ineligible timing mismatches, then verify publication and retention in a paced
stream. The held-out failure is now development data, not an unseen test.

The opt-in [EOF resolution probe](CONTINUOUS_STREAMING.md#optional-eof-resolution-probe)
now connects one distinct retained-audio candidate to the controller. It records
the candidate after the refused original run releases its resources. It never
publishes the candidate or evicts audio. Scripted tests cover unchanged prefix,
attempt limits, cancellation and recovery. Full recovery still needs a justified
publication boundary and real-stream validation.

The [local handoff assessment](research/2026-09-06-resolution-handoff.md) now
specifies one anchor-bearing overlap and checks full continuation agreement,
frozen state and PCM correspondence. Adversarial tests cover repeated anchors,
stale state and omitted boundary words. It cannot authorize publication; all
four stored head-only cases initially lacked the required overlap observation.
The paced test also exposed a refusal path that omitted already computed word
alignment. A local repair preserves this evidence and connects aligned EOF
refusals to the same opt-in, single-attempt probe. Its scripted tests pass; the
repaired live path still needs a real-audio run.

The [fixed overlap experiment](research/2026-09-06-overlap-observations.md#recorded-result)
now records seven T4 windows across these four cases and the paced noisy state.
All five structural assessments reject. The new noisy head-only proposal reduces
word edit distance from 18 to 1, but its overlap omits the continuation. One
other refusal is a capitalization/token difference; four involve absent or
changed anchor words. All runs release capacity. The next change must separate
representation, lexical and timing diagnostics, then test boundary selection
with retained acoustic context. No live recovery or relaxed publication rule
follows from this result. Analysis receipts now retain declared model, options
and seed; missing effective identities remain unknown.

The [CPU disagreement replay](research/2026-09-06-resolution-disagreements.md)
now separates text, tokens and estimated timing without changing publication.
The earlier-start noisy control restores the anchor but still omits the next
utterance. A fixed context guard is therefore not a universal recovery rule.
That CPU replay identified two existing controls and three missing intervals.

The [fixed context-guard run](research/2026-09-06-context-guard-observations.md#recorded-result)
completed those three intervals on one T4 and reused the controls. Two missing
anchors return, but all five comparisons still reject. The no-pause crop retains
every word token while moving the first word's estimated start by 500 ms.
Speaker 2961 has exact suffix agreement but a 300 ms interior anchor shift and
a 20 ms boundary crossing. All new calls close and restore capacity.

The [group-boundary comparison](research/2026-09-06-group-correspondence.md)
traces the first-word shift to the raw alignment path and compares complete
anchor groups in a local, post-hoc replay. Four states still reject; speaker
2961 becomes structurally eligible only when the group rule is combined with
the live-sized 200 ms boundary tolerance. No text changes or live recovery
follow. The additional CPU validation scores 14 observations from eight recorded
anchor states: all match under both rules. It adds one source fixture outside
the tuning states, with no observed regression or further gain. Synthetic
repetitions expose a stricter ambiguity refusal in the group rule. Next, test
the complete fallback handoff while preserving the successful strict path, and
define what evidence permits audio-retention progress. The live policy and raw
records remain unchanged.

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

- [x] Replay 30 minutes at wall-clock audio speed on the registered repeated
  speech/noise/pause sequence. Full coverage and cleanup pass on one T4.
- [ ] Commit text before EOF. Previously committed text and segment identities
  remain unchanged after window shifts, retries, and cancellation.
- [ ] Account for accepted samples in the input timeline, with no gaps caused by
  silent loss. Rejected chunks leave sequence state unchanged and can be retried.
- [x] Record configured buffer limits and observed high-water marks. The long
  T4 stage peaks at 28.80 seconds of PCM within its 40-second cap. This is not
  a process-memory or VRAM bound.
- [ ] Exercise slow decoders and consumers. Report delay or backpressure without
  unbounded queues, silent loss, or deadlock.
- [ ] Repeat deterministic replays with different chunk partitions and test the
  output equivalence promised by this profile.

D1 establishes continuous operation, not a quality or speed advantage over
another streaming system.

## D2. A usable local live-transcription entry point

**Status: CLI implemented and short file run tested; physical microphone and live qualification remain open.**

Deliver a small Python API and one local command, with microphone input and paced
file replay. Do not require a server deployment or a desktop application.

The [paced PCM API](PACED_REPLAY.md) is implemented and tested locally and on one
T4. It accepts recorded bytes and emits timed events through callbacks. The
[installed CLI](CLI.md) now wraps the existing controller for one local stream,
with WAV/PCM input, optional bounded microphone capture and final text/subtitle
exports. A short real CPU file run completes; capture lifecycle and failure tests
use scripted devices. Subtitle times denote committed source coverage, not
word-accurate caption alignment. The
[network reference](NETWORK_REPLAY.md) connects a Windows file-replay client to
one Modal T4 and has a recorded successful short run. Its diagnostic command
is not an installed microphone transcription application. Its opt-in
`pcm-websocket/live-v2` protocol now accepts an initially unknown duration, checks
observed input at EOF, and enforces finite session and buffer limits. Local tests
include an actual WebSocket exchange and an accelerated input longer than 120
seconds. This is not a long-session or remote GPU qualification. The installed
CLI now connects both paced file input and optional microphone capture to that
remote example. Real localhost tests cover the installed command path through
the controller and final exports. The new Modal profile now passes its registered
short and 30-minute tests. Physical capture remains unqualified.

- [x] Document input format, model selection, provisional and committed results,
  cancellation, and failure behavior.
- [x] Show readable partial captions with explicit corrections and finality.
- [x] Export committed subtitles with source times. Distinguish a final subtitle
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

**Status: short memory and fixed-window compute improvements verified; full gate open.**

Remove unnecessary work before adding scheduling or caching complexity.

The [CUDA lifetime comparison](research/2026-09-06-cuda-lane-lifetime.md) verifies
a post-release allocation plateau on consecutive T4 analyses. The fresh-stream
control adds 8,519,680 bytes per new stream; the reused lane adds none after its
first call on this input. Words, tokens and times match exactly. This does not
close the long-session or live efficiency gates.

The runtime retains one fenced CUDA lane per model binding. An explicit
`reuse_alignment_features=True` execution profile can also borrow its own
decode features for alignment using the optional backend patch. The
[paired T4 comparison](research/2026-09-06-alignment-feature-handoff.md) removes
one encoder forward on all eight inputs while preserving exact words and times.
The default path and active backend patch manifest remain unchanged. This is
same-window reuse, not a cache across growing audio. No general latency or live
cost advantage follows from the fixed-window comparison.

The [paced comparison](research/2026-09-06-paced-feature-reuse.md) now completes
the normal and attenuated inputs in both arms with identical published text
and spans. The noisy prefix stops at the same boundary in both arms; the overall
record remains failed. On the normal mixture, encoder forwards fall from 47 to
25 at unchanged analysis count. Peak allocated memory falls by 34.4%; reserved
memory does not fall. Host-call timing improves in this single comparison, but
device-time, repeated-worker and long-session efficiency gates remain open.

- [x] Add opt-in preview coalescing with bounded state and immutable retries.
- [x] Reuse one CUDA lane under exact ownership and completion fences; verify
  short fixed-input allocation and output parity on T4.
- [x] Borrow same-window decode features for alignment behind an opt-in profile;
  verify encoder counts and exact output parity on eight T4 inputs.
- [x] Measure work, output parity and failures on three matched paced inputs;
  retain the unresolved noisy pair and single-worker timing limits.
- [x] Add one opt-in prompt re-decode within the existing transaction. Reuse
  exact-window encoder features with a fresh decoder, cache and generator.
  Local contract tests cover cancellation, stale publication and fence recovery.
- [x] Verify prompt re-decode with the real CPU backend on the JFK fixture.
  Greedy, beam-size-2 and sampled decoding match their independent controls
  exactly, with one encoder forward instead of two. See the
  [comparison and limits](research/2026-09-06-prompt-feature-reuse.md).
- [x] Verify the same-window prompt path on T4 for the JFK fixture, three decode
  profiles and cancellation after replacement. Exact results and scores match;
  three matched pairs use three encoder forwards instead of six. All leases
  and the CUDA lane are released. This does not qualify a general live policy.
- [ ] Connect an opt-in same-window context observation to recovery diagnostics.
  Keep its provenance separate and preserve the acoustic publication checks.
  Different audio windows cannot share this encoding.
- [ ] Replicate on unseen inputs and repeated workers before general live-cost
  or latency claims.
- [ ] Evaluate additional preprocessing reuse with explicit input identity and
  profile limits. New audio does not make Whisper's noncausal encoder cache
  append-only.
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

Local checks on 2026-09-06: the wheel installs into a new Windows CPython 3.13
environment. After installing the pinned native dependencies from the official
indexes, its installed command completes the 11-second JFK file outside the
source checkout, using a separate pinned backend copy and cached `tiny.en`
weights. Real FINAL and TXT/SRT/VTT exports are verified. Help also works without
PyTorch. This is not a virgin Windows machine or cross-platform qualification.
Local-path loading restores the named model's alignment-head mask;
the weight-state fingerprint alone does not cover that nonpersistent buffer.

The runtime suite runs 790 tests with one skip. The repository-tool suite runs
702 tests with three skips in the native environment. The 42 client/new-server
tests also pass in the host environment, including the real localhost WebSocket
test skipped where its optional dependencies are absent. Ruff and strict mypy
pass. These checks run locally; they are not CI results for a published release.

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

## D6. Durable recovery, GPU-free suspension, and compatible migration

**Status: local publication-boundary savepoints implemented; broader recovery gates remain open.**

The current resident pause and transaction `checkpoint()` do not save state to
storage. Implement explicit capabilities below without changing their meaning.
Use one versioned checkpoint contract with optional backend-state payloads, not
a new execution framework. Do not serialize Python object graphs or device
handles as the portable format.

### D6a. Durable session recovery with bounded replay

- [x] Add an explicit local publication-boundary savepoint with complete bounded
  committed history, retained PCM, event cursors, word anchors and exact endpoint
  detector counters. Restore into a fresh session and worker; do not deserialize
  native execution handles. Scripted base and hybrid profiles pass separate-process
  comparison with uninterrupted output. A real CPU `tiny.en` test also passes:
  save after 11 seconds of 22 admitted seconds, exit, restore in a second process,
  and decode the remaining 11 seconds with exact final state and event parity.
  See [contract, evidence and use](CHECKPOINTS.md).
  This savepoint refuses unresolved or in-flight work. It does not make `push()`
  acknowledgements durable, fence a second owner, or redeliver a historical event
  journal. The remaining items below retain their broader acceptance conditions.
- [ ] Define the failure model, storage durability, and input acknowledgement
  boundary. Persist audio before acknowledging it under the durable profile.
- [ ] Save model/tokenizer/profile identities, options, required audio and its
  sample offsets, committed output, context, pending policy state, and sequence
  numbers. A session snapshot alone does not contain all controller state.
- [ ] Commit journal/checkpoint updates atomically. Validate format, sizes,
  identities, and integrity before restore. Bound storage and apply backpressure
  while paused; never silently discard acknowledged input.
- [ ] Restore through a fresh worker and admission. Fence obsolete owners with
  a durable generation check. Preserve committed event identities; support
  consumer deduplication instead of promising exactly-once network delivery.
- [ ] Recompute only the declared unfinished range and required retained context.
  Preserve committed text and unresolved status. Record the recomputation bound.
- [ ] Kill the source process around input acknowledgement, storage updates, and
  publication. Verify recovery, stale-owner rejection, corrupt-record refusal,
  and bounded storage using local CPU tests.

Acceptance: a new process restores the acknowledged input timeline and committed
output under the declared failure model. Recognition accuracy is a separate
gate. Replay is not an exact mid-token checkpoint.

### D6b. Durable token-state checkpoint and GPU release

- [ ] Add a non-destructive quiesce/export path at an owner-controlled token
  boundary. Close admission, drain submissions, and wait for outstanding device
  work before copying state. The existing completion path cleans up the decoder
  and cannot serve as this export fence unchanged.
- [ ] Save decoder phase, position, every hypothesis, scores, pending logits,
  language state, encoder features, and attention caches. Include actual sampling
  generator and beam state, not just an initial seed. Retain inputs needed for
  later alignment. Encode cache entries by stable identifiers, with explicit
  position metadata rather than incidental dictionary order.
- [ ] Complete device-to-host copies and commit the durable record before
  destroying the source state. Handle copy, storage, cleanup, and fence failures
  without releasing uncertain ownership or publishing an incomplete checkpoint.
- [ ] Restore immutable model identity and fresh execution handles separately.
  Do not restore locks, leases, threads, CUDA streams, or events from a file.
- [ ] Qualify CPU greedy restore after prefill and after step N, in a fresh
  process, against uninterrupted tokens, scores, and final output. Extend to
  sampling, beam search, timestamps, and alignment as separate cases.
- [ ] Verify job-state release separately from full worker shutdown. Returning a
  transaction lease does not unload model weights or the retained CUDA lane.
  Measure live allocations, allocator reservations, and process/device use.
- [ ] Run a short registered T4 comparison: save, stop the source worker, verify
  resource release, restore in another process, and finish. Record checkpoint
  bytes, save/restore time, output parity, and work avoided or repeated.

Acceptance: the source process is absent during suspension; a compatible new
process continues from saved token state. The tested GPU-free profile specifies
whether it releases only job state or the whole worker. An in-process pause,
CPU-only copy, or successful cache-flush call is not this result.

### D6c. Compatible migration and scheduling policy

- [ ] Declare a compatibility matrix for model weights, tokenizer, backend patch,
  runtime versions, precision, kernels, and hardware. Refuse unsupported exact
  restores before execution; offer explicit replay only where qualified.
- [ ] Enforce a single current owner when two workers try to restore the same
  session. Reject stale publication after a replacement worker starts.
- [ ] Test another worker first, then another hardware profile. Distinguish
  state portability, committed-output preservation, and identical future tokens.
  Cross-device bitwise equivalence is not assumed.
- [ ] Measure whether offload pays for the tested pause duration and workload.
  Compare resident pause, durable replay, and exact-state restore before selecting
  a policy. Preserve resource limits and fresh admission on every resume.

Close D6 only for a published recovery profile and its failure tests. Durable
recovery is required for the full product, but a single-stream release can ship
earlier with its actual resident-pause limits stated. D4 is required for savings
claims; persistence alone does not make inference faster or cheaper.

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
