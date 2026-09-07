# Continuous transcription

`timestamp_agreement_stream/v1` is an experimental rolling-input profile. It
uses the existing native decoder, completion fences, and session commits.
It does not change the existing bounded-preview or offline adapters.

## Use

An opt-in `eof_context_retry=True` policy can try one shifted retained-context
window after an aligned EOF refusal. It requires word alignment and
`input_evidence=True`, and cannot be combined with `resolution_probe`.
It uses the unchanged publication checks and never revises committed output.
Inspect `context_retry_observation` for the original refusal and retry outcome.
The [incremental comparison](research/2026-09-06-eof-context-retry.md) and
[source-paced T4 comparison](research/2026-09-06-paced-context-retry.md) recover
one blocked development stream. Noisy inputs and long sessions remain unqualified.

For an independent source clock, use the [paced PCM replay driver](PACED_REPLAY.md).
It supplies recorded chunks while the model owner drives decoding and reports
overload instead of slowing the source to match the decoder.

An optional `input_evidence=True` policy checks admitted PCM before permitting
text publication. Exact digital silence produces empty, source-backed coverage;
uncertain output retains audio. It does not provide general voice activity
detection or resolve all pause boundaries. See the
[input-evidence policy and validation](research/2026-09-05-input-evidence.md).

Create `ContinuousTranscriptStream` with a native adapter and a PCM-to-mel
function. Enable timestamp tokens in `NativeDecodeOptions`. Input is mono,
16 kHz, signed 16-bit little-endian PCM.

- Call `push(sequence_number, pcm)` in order, starting at zero. This method may
  run on an input thread while the owner drives decoding.
- Call `step()` on the creating thread while `ready` is true. Consume each
  returned event batch before calling again. There is no internal event queue.
- Call `finish_input()` at EOF, then continue stepping until `done`.
- Call `close()` to abandon the stream. Use `cancel_active()` to cancel only the
  current decode; its accepted audio remains available for a retry.

`AudioBufferFullError` rejects the entire chunk without changing its sequence
number. The producer must retry, pause its source, or report overload. It must
not silently drop the rejected input. A microphone driver may have its own
overflow behavior outside this contract.

## Optional verified token drafts

`ContinuousStreamConfig(max_draft_tokens=32)` enables request-local draft
verification during greedy decoding. Values from 1 to 32 cap the proposal;
the default `0` disables it. Sampling and beam search are rejected when enabled.

For the existing local factory, keep its profile and change only this option:

```python
from dataclasses import replace
from pathlib import Path
from whisper_runtime.native_setup import CLI_STREAM_CONFIG, create_stream

stream = create_stream(
    manifest=Path(".tmp-composed-native/manifest.json"),
    model=Path("/absolute/path/to/tiny.en.pt"),
    reuse_alignment_features=True,
    config=replace(CLI_STREAM_CONFIG, max_draft_tokens=32),
)
# Feed PCM and consume events through the existing stream API. Always close it.
```

Use paths from your existing native setup; this example downloads nothing.
The draft option also works without alignment-feature reuse and does not
change the native backend pin. The 20/24-second context used in the recorded
comparison is a separate opt-in, not the factory default shown above.

Each stream keeps only the leading raw token IDs from its last completed
analysis. Those IDs are proposals, not published text, a prompt or a continuity
witness. Current-audio decoding still selects the output. Other streams share
neither hints nor decoder cache. Failure, cancellation, auxiliary recovery and
terminal cleanup drop the hint. Prompt re-decode uses ordinary decoding.

Savepoints retain the configuration but omit the disposable hint. The first
window after restore uses ordinary decoding and builds a new hint. Older v1
savepoints without the option restore with it disabled. This is logical
boundary restore, not a saved GPU cache or exact mid-token continuation.

See the [integrated CPU result](research/2026-09-07-native-draft-integration.md)
for the tested scope. Numeric scores may change; the policy thresholds do not.
The default stream profile does not enable drafts. The installed CLI exposes
the separate opt-in `experimental-optimized-v1` profile, which also changes
context and alignment reuse. See [execution profiles](CLI.md#versioned-local-execution-profiles)
for its settings and backend requirements.

## Publication rule

An offline [crop/prompt comparison](research/2026-09-06-noisy-context-prompt.md)
shows why better recognition alone cannot authorize a retry: published decoder
history restores a missing utterance, but its word alignment no longer contains
the frozen anchor. This diagnostic does not enable prompt retries in the live
profile or change its publication checks.

### Caller-delimited source units

Set `source_units=True, input_evidence=True` to select
`source_unit_stream/v1+input_evidence/v1`. This separate transcription profile
requires zero left context and disables word alignment. Existing profiles and
defaults are unchanged.
With preview coalescing enabled, the profile ID starts with
`coalesced_source_unit_stream/v1` instead.

The caller closes an admitted audio range with `stream.seal_unit(end_sample)`.
Use exact, absolute 16 kHz sample positions. Call it on the model-work owner
after pushing audio through that boundary and before processing beyond it.
Only one boundary can be pending. The next unit starts where the preceding
unit commits; admitted audio after that point remains in the same bounded buffer.
After `finish_input()`, the caller may still close remaining admitted ranges
before their analysis starts. This does not reopen audio admission.

For example, a caller that owns the boundaries can drive one session as follows:

```python
config = ContinuousStreamConfig(source_units=True, input_evidence=True)
# Construct stream with the same native adapter, mel builder, and config.
for sequence, pcm_unit in enumerate(bounded_audio_units):
    stream.push(sequence, pcm_unit)
    stream.seal_unit(stream.accepted_samples)
    while stream.ready:
        consume(stream.step())
stream.finish_input()
while stream.ready:
    consume(stream.step())
```

This example supplies whole units. To expose previews while audio arrives,
push smaller chunks and drive `step()` between them; seal before driving the
chunk that reaches the desired boundary. Handle overload and unresolved input
explicitly. Do not catch those errors and silently continue with later audio.

Within an open unit, native results are provisional. Closing the unit schedules
a full-range native result under the input-evidence policy. Publication still
waits for the existing transaction and resource-release path. It emits the same
revision and commit events as other profiles. Agreement history does not cross
unit boundaries. `finish_input()` closes the final range; an intermediate unit
never emits a final-session event.

A preview already in flight is not converted into a final analysis. Its retry
keeps the original PCM, operation identity, and EOF snapshot. A subsequent
closed-unit analysis has a distinct identity even at the same source endpoint.
`last_trace.source_unit` records exact range boundaries and `caller`,
`end_of_input`, or `quiet_run` as the origin. `last_trace.eof` means that this
analysis reaches
the global EOF observed at admission, not merely that input is closed.

This is a full-result recognition contract, not a word-agreement contract or a
proof of complete recognition. The boundary is external input, not detected
speech or evidence that a gap is silent. Low-confidence or nonlexical results
at a closed boundary stop with `StreamNeedsResolutionError` and retain admitted
PCM. Repeating `step()` does not spend more work on that unresolved boundary.
Exact digital zeros use the existing empty silence publication; this profile
still decodes them and does not yet bypass the model.

Each unit must fit the configured analysis window. The caller must handle
continuous speech that has no suitable boundary; the runtime does not split or
discard it automatically. Short units can lose context and increase fixed
model overhead. Provisional captions can appear before a pause, but final text
waits for closure. Automatic endpoint detection, overlap handling, and paced
latency remain separate acceptance tests.

### Automatic quiet-run endpoints

To infer unit boundaries from the admitted PCM, configure the same controller:

```python
from whisper_runtime.adapters import ContinuousStreamConfig, QuietEndpointConfig

config = ContinuousStreamConfig(
    source_units=True,
    input_evidence=True,
    endpointing=QuietEndpointConfig(),
)
```

The profile is `quiet_endpoint_stream/v1+input_evidence/v1`. Preview coalescing
uses the `coalesced_quiet_endpoint_stream/v1` prefix. Push chunks, drive `step()`
and call `finish_input()` as usual. Do not call `seal_unit()` in this mode.
No additional model, dependency, worker or audio queue is required.

The detector measures absolute peak amplitude in consecutive 20 ms frames.
A frame is quiet when every signed 16-bit sample has magnitude at most 32.
After a nonquiet frame, 600 ms of consecutive quiet can close a unit that is
at least one second long. Sustained quiet can close a unit every ten seconds,
including leading silence. Any nonquiet frame resets the quiet duration.
These configurable thresholds describe a heuristic, not voice activity detection.

The boundary is the detection endpoint, not the beginning of the quiet range.
All samples remain in the same stream buffer until the native transaction
commits and releases its resources. Admission scans PCM before model scheduling.
Rejected pushes change neither the detector nor the pending endpoints. Partial
frames survive chunk boundaries; EOF never pads the detector's input.

Pending endpoints are sample positions, not copies of audio. Their count is
bounded by the admitted buffer and minimum unit duration. Retries retain their
original endpoint. A commit removes only that endpoint; it does not reset the
detector, which may already have examined later admitted audio. EOF drains
pending endpoints before closing the actual remaining samples.

`source_unit.origin` is `quiet_run` for inferred boundaries. Its `endpoint`
records the observed quiet start, end and peak. `accepted_through_sample` records
the admission horizon when the decision trace is created. It is not the model's
original admission snapshot. The trace remains a prepared decision, not proof
that publication succeeded.

Low amplitude never authorizes empty publication. Only the existing exact-zero
rule does that. A closed nonzero range with uncertain output retains its PCM and
stops. Conversely, a model-supported result can still be wrong: this detector
does not prove the absence of speech or the correctness of recognition.

Quiet speech can satisfy the threshold; noise can prevent a boundary. Keeping
every sample does not guarantee that a split preserves word recognition. There
is no forced cut for continuous speech at the analysis limit. Such input stops
with explicit backpressure. Real microphones, background noise, faint voices,
cross-boundary words and paced latency need separate validation.

### Word boundaries with quiet endpoints

The experimental `word_boundary_fallback=True` option combines word-prefix
agreement during speech with closure at an observed quiet endpoint. It requires
`source_units=True`, `input_evidence=True`, a `QuietEndpointConfig`, and positive
`left_context_ms`. Keep the separate `word_alignment` flag false: this combined
profile selects alignment internally.

```python
config = ContinuousStreamConfig(
    preview_interval_ms=2000,
    holdback_ms=2000,
    left_context_ms=2000,
    input_evidence=True,
    source_units=True,
    endpointing=QuietEndpointConfig(),
    word_boundary_fallback=True,
)
```

Its profile identifier is
`word_boundary_quiet_endpoint_stream/v1+input_evidence/v1`.
Open windows can commit an agreed word prefix and retain context for the next
analysis. A queued quiet endpoint beyond the current window waits for supported
prefix progress. It cannot force a cut. At a closed unit, alignment must exclude
previously committed context before publishing the remaining suffix. A closed
unit does not emit the stream's final event; that still requires source EOF.

This option does not guarantee that quiet intervals contain no speech, that
word timing is correct, or that every anchor will resolve. Uncertain input and
unresolved anchors retain the existing explicit failure behavior. See the
[acoustic diagnostic plan and results](research/2026-09-05-acoustic-boundaries.md).
The first T4 diagnostic completes normal and attenuated speech with pauses.
Both inputs without detected pauses still stop before completion. This profile
is not yet suitable for unattended continuous speech.
All existing profile defaults remain unchanged.

#### Bounded word-aware context

Set `word_context_limit_ms=6000` on the configuration above to test a separate
`+word_context/v1` profile, before the `+input_evidence/v1` suffix. The default is
zero (disabled). `left_context_ms` remains the desired overlap; the new value
limits how far the retained origin may move backward from a proposed commit.

Before a partial commit, the controller preserves at least two lexical words
in the next anchor and snaps the desired origin backward to avoid cutting
through words in the current alignment. Leading standalone punctuation is
excluded from the next anchor only. The published text, internal punctuation,
native tokens and times stay unchanged. This does not loosen the matcher.

If context and growth limits cannot both be met, the decision stays provisional
with `context_unresolved`; no output is committed and no audio is evicted.
The chosen origin is fixed before native commit and applied only after cleanup.
Quiet-unit closure, digital silence and EOF keep their existing rules. A single
word at a closed boundary can still publish if the existing checks accept it.

The limit must cover the desired overlap, use a multiple of 20 ms, and leave
room for two preview intervals plus positive growth within the analysis window.
This heuristic uses estimated word boundaries, not a guarantee of sufficient
acoustic context. See the [registered comparison](research/2026-09-05-word-context.md).
That comparison fails: noisy-input coverage advances from 6.10 to 33.54 seconds,
but EOF remains unresolved. The no-added-pauses case still stops at 21.26 seconds.
Normal and attenuated cases complete. Keep this profile opt-in; it does not
qualify unattended continuous speech.

### Timestamp agreement

Two successive analyses must start at the same source position and the second
must contain more audio. Closed timestamp segments qualify when their text and
tokens match exactly, timestamps are within the configured tolerance, and the
new endpoint leaves the configured holdback after the selected segment.
Selected segments must form a contiguous prefix at the commit boundary.

This whole-segment rule is inspired by
[LocalAgreement](https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.html)
and [Whisper-Streaming](https://aclanthology.org/2023.ijcnlp-demo.3/). It is stricter
than their reference methods and is not a claim of a new agreement algorithm.
Agreement between predictions does not establish that the words are correct.

A provisional event may be replaced. A commit references its exact revision;
that segment identifier is never reused. Audio is evicted only after the native
transaction commits and releases its resources. In the default profile, each
new rolling window starts at the committed endpoint, without retained left
audio context. Neither profile carries an unbounded text prompt across windows.

## Optional retained context

Set `ContinuousStreamConfig(left_context_ms=2000)` to select the experimental
`context_agreement_stream/v1` profile. The default remains zero; the existing
offline and bounded-preview paths do not change.

This profile separates the committed position from `retained_from_sample`.
Previously published audio can remain in the buffer for later analyses. The
retained context counts toward both the input-buffer limit and the 30-second
analysis limit. Context must be a multiple of Whisper's 20 ms timestamp grid
and leave room for a growing analysis pair.

Only new, complete timestamp segments may appear in previews or commits. A
segment that crosses the committed boundary cannot be clipped or deduplicated
by guessing. Such a boundary can remain unresolved; retaining context alone
does not guarantee useful continuous recognition.

At EOF with retained context, the remaining text must have complete, contiguous
segments from the committed boundary to the analysis endpoint. Otherwise,
`StreamNeedsResolutionError` retains the input and prevents identical EOF
decodes from being retried. Close the stream before starting a different policy.
Previously committed text is never rewritten. With zero retained context, the
ordinary native EOF behavior described below remains unchanged.

## Coalescing pending previews

`ContinuousStreamConfig(coalesce_previews=True)` selects the optional
`coalesced_timestamp_agreement_stream/v1` profile, or
`coalesced_context_agreement_stream/v1` when left context is also enabled.
When audio arrives faster than the caller drives decoding, the controller
analyzes a more recent accepted endpoint instead of replaying every pending
preview. It retains the admitted PCM and reserves room for two growing
hypotheses before the analysis-window limit.

The default remains `False`. Coalescing changes which hypotheses are compared;
its transcripts and revisions need not match fixed-cadence output. Scripted
tests measure avoided decode calls, not GPU savings or recognition quality.
This option needs a matched paced-audio comparison before a performance claim.
It does not alter EOF, cancellation, or resource-recovery contracts.

## Optional word alignment

Set `ContinuousStreamConfig(word_alignment=True, left_context_ms=2000)` to use
`word_agreement_stream/v1`. Add `coalesce_previews=True` to use
`coalesced_word_agreement_stream/v1`. Both are opt-in transcription profiles;
translation is not supported by this publication rule.

The adapter runs Whisper's request-local word alignment once per completed
analysis. It preserves the raw result and attaches estimated word times. The
policy compares whole words, including their tokens and source positions,
instead of requiring matching native segment boundaries. A native segment may
cross the committed boundary while its new words remain publishable.

After a commit, at most four published words anchor the next analysis. A match
must remain near their original source positions. Matching text in a later
repetition cannot authorize publication. Missing or ambiguous anchors stop
progress. Only anchor words whose whole estimated span remains in retained audio
are required; a word cut by the left edge is excluded.

One window-edge exception applies to the start estimate of the first observed
word. It can extend left to the exact analysis origin if the matched anchor has
at least two words. All anchor text and tokens must still match. The first word's
end, and both bounds of every other anchor word, must remain within the original
timing tolerance. The exception does not apply to a later occurrence, an internal
word, or a rightward start shift. Original estimates remain unchanged.

After a validated anchor, a new word may start before the processed
boundary by at most `timestamp_tolerance_ms`. The publication records this
tolerance; the word's original estimate is unchanged. Larger overlaps remain
unresolved. The next anchor uses only newly published words from one analysis.

An `AlignedPublication` binds a contiguous word slice to its original alignment
and an explicit processed-audio range. The native transaction commits this
object. Processed coverage and estimated word bounds are distinct. The stream
advances its audio boundary and anchor only after the commit
and resource release succeed. The admitted input range stays fixed across
cancellation and recovery, even when more input arrives.

Processed coverage is not a claim that every sound was recognized. Natural
gaps between aligned words are permitted. At explicit EOF, the remaining word
suffix may commit without a second observation, and coverage extends to the
input endpoint. An empty suffix is allowed only at EOF. Before EOF, empty text
does not advance the stream. Word times are estimates, not evidence of silence
or correct recognition.

Alignment adds model work and may increase memory use. It runs under the same
execution scope, cancellation checks, model lock, and cleanup fence as decoding.
The legacy hook-based alignment path is not supported. This option needs a
matched real-audio run before any claim of quality, latency, or GPU savings.

## Optional EOF resolution probe

Set `ContinuousStreamConfig(resolution_probe=True)` alongside `word_alignment=True`
or a valid `word_boundary_fallback` configuration to connect one experimental
candidate observation to the stream. The default is `False`; enabling it adds
`+resolution_probe/v1` before any `+input_evidence/v1` profile suffix.

After an EOF word-policy refusal, the controller may decode one different
window from the frozen committed boundary to the original analysis endpoint.
That boundary must lie strictly inside the original retained analysis. The
controller freezes the exact retained PCM slice, anchor and session version;
it never retrieves evicted audio or uses a reference transcript. The extra PCM
copy is bounded by the analysis-window limit. Candidate admission waits for
the original run's successful fence, including explicit recovery if needed.
Non-EOF refusals and identical windows do not trigger this probe.

An input-evidence refusal after word alignment, such as a selected suffix with
no lexical text, retains that computed alignment and its anchor diagnostic in
the trace. At EOF it can use the same opt-in probe path. The unsupported
publication is never attached to the trace or committed. Generic score failures
that precede alignment and nonfinal source-unit refusals gain no extra work.

Continue calling `step()` to drive the probe. It prepares raw aligned words,
closes without calling `finish()`, and raises `StreamNeedsResolutionError`.
It emits no candidate transcript, commit or final event; accepted PCM, frozen
commits and the unresolved status remain unchanged. Startup failure or
cancellation consumes the single attempt. A retained native transaction still
requires exact recovery; repeated `step()` calls never launch another candidate.

Inspect `stream.resolution_observation` on the owner thread. This immutable
record retains the original refusal in `source`, the frozen anchor/version,
candidate sample bounds and PCM SHA-256, and a status of `scheduled`, `running`,
`observed`, `failed` or `unavailable`. `candidate` contains the unmodified
`NativeWordAlignment` when available. `last_trace` then describes the candidate
with action `resolution_observation`, not a publication. Native cleanup can
still fail after candidate preparation; inspect the observation status and
resource-release state, not just the presence of words.

`analysis_identity` records the adapter's declared model, the requested decode
options and seed, and a native execution profile when available. It adds no
model work. The current runtime cannot verify tokenizer artifacts, an arbitrary
PCM-to-mel callback, backend code, or the effective per-run alignment mode;
those fields remain `None`. A declared profile is not proof of the mode an
adapter actually used. Receipt equality does not authorize replay or publication.
The receipt contains no tensors and stays separate from session version and
audio identity. Older observations may omit it.

This is a connected probe, not recovered live continuation. A head-only result
can miss speech at the estimated cut, even when its text appears correct.
Publishing it needs a separately justified handoff rule. The probe does not
relax timestamp tolerance, apply the experimental local matcher, or claim
complete acoustic coverage, transcription recovery or inference savings.

## Decision trace

`last_trace` exposes one immutable `ContinuousDecodeTrace` on the owner thread.
It includes the actual analysis sample range, previous committed position,
retained origin, raw result metadata, selected publication span, and decision.
Its monotonic `decode_index` lets an external diagnostic collect each new record.
There is no internal trace history. Persisting a full trace is the caller's
choice and can include recognized speech.

The trace describes a prepared decision, not successful publication. A commit
can still fail or await resource recovery. Check transcript events and runtime
state for the outcome. Do not infer an analysis endpoint from a commit event:
the latter describes the selected output range.

Word profiles also expose `word_alignment` and `word_publication`. Their raw
`result` remains unchanged, and `publication_span` is `None`: the publication
selects words, not native segments. The segment-only trace analyzer does not
validate this profile.

`last_trace.anchor_diagnostic` adds detail when the word policy checks a
committed anchor. The existing top-level reasons and publication rules are
unchanged. The optional record distinguishes unavailable retained context,
missing lexical text, changed tokens, timestamp mismatch, relocated text and
multiple possible occurrences. It identifies the previous or current observation,
counts exact matches and records signed boundary differences. A matched anchor
does not establish that its continuation is correct.

`diagnose_word_anchor` exposes the same pure check for local analysis. It uses no
model, reference transcript or corrected timestamp. The separate
`tools/analyze_word_resolution.py` command replays saved failures and tests an
experimental terminal-end correspondence without publishing any text. See the
[local resolution report](research/2026-09-06-word-resolution.md).

## Text-only agreement analysis

`resolve_text_prefix` in `adapters.stream_policy` compares the text-token prefix
of two growing analyses independently of how Whisper divided them into timed
segments. A bounded committed-token anchor identifies the new suffix. A missing
or repeated anchor leaves the comparison unresolved. The caller must bind both
analyses to unchanged audio, tokenizer, model, and options.

The result identifies exact token indices in the current hypothesis. It does
not publish text, classify silence, or advance the audio watermark. A candidate
can end inside a byte-encoded character; a future publication path must check
text decoding and map source coverage before committing it. Original timestamps
remain unchanged. This analyzer is separate from the opt-in word alignment
path above and does not authorize that path's commits.

This comparison is intended for trace analysis and for testing a future text
publication contract. Agreement alone is not an acoustic alignment or accuracy
test. An empty anchor is valid only when no text has been committed in the block.

## End of input

In the segment profiles, EOF with zero retained context uses the native decode
contract. EOF finality does not assert two-hypothesis agreement. A final event
means all admitted input was processed, not that recognition was error-free.

## Bounds and unresolved input

The default analysis window is 30 seconds. The input buffer holds at most
40 seconds, including pending input. Session history retains four records.
Only one prior hypothesis is retained. The caller owns the full transcript and
any persisted event log. Preprocessing and backend tensors add to these buffers;
the input-buffer bound is not a process or GPU memory limit.

If no contiguous prefix can be committed before the analysis-window limit,
`StreamNeedsResolutionError` stops progress and retains the accepted PCM. The
producer eventually receives backpressure. The segment profiles do not interpret a
timestamp gap, silence probability, or empty text as proof that audio can be
discarded. Sustained silence and unstable timestamp boundaries therefore remain
important cases to resolve before general live use.

## Verification scope

The first two [recorded T4 diagnostics](../evidence/README.md) use `tiny.en`
FP32. The 11-second case matches its same-options control after whitespace
normalization. The 33-second synthetic concatenation exercises rolling commits
and releases all resources, but its text differs from the segmented reference.
Both were unpaced file replays. The larger test reached a 10.56-second peak PCM
buffer; this is not a total process-memory bound.

The later [four-configuration comparison](research/2026-09-05-boundary-comparison-results.md)
isolates holdback and left context. Longer holdback removes the two word edits
on this input, but both left-context cells stop at EOF with an unresolved
publication boundary. Defaults remain unchanged. Add real silence, distinct
recordings, and long paced input before widening the profile's claims.

The [boundary diagnosis](research/2026-09-05-stream-boundaries.md) traces the
repeated recognition error, separates publication from context retention, and
defines the next controlled comparisons. It is a plan, not a validated fix.

Deterministic tests exercise agreement and runtime lifecycle behavior without
requiring a model. Native GPU diagnostics must record the tested source snapshot,
model, input, configuration, events, and outcome separately. A short diagnostic
does not close the 30-minute continuous-operation gate in `ROADMAP.md`.

No speedup, lower GPU cost, durable crash recovery, or unrestricted continuous
transcription is claimed by this profile.

### Word-alignment CPU smoke, 2026-09-05

A local diagnostic used the cached `tiny.en` model on CPU with two Torch threads
and the patched backend at `70141b8b26acc09f5fe7fadee63d1604178309df`. It repeated
the backend's 11-second JFK fixture three times. PCM conversion used clipped
float samples, scaling by 32767, rounding, and signed 16-bit encoding. The
528,000-sample PCM had SHA-256
`73412abdfa7bc14c1967d0c55871145665b73254812eccfac8a1acd74452b103`.

Settings were English transcription, timestamp tokens, seed 7, two-second input
chunks, preview interval, holdback, and retained context. Word alignment was on;
coalescing was off. The caller drained each chunk before supplying the next,
then declared EOF. This was not a wall-clock-paced replay.

The first run stopped at EOF with 18 seconds committed: the next word started
80 ms before the processed boundary after realignment. Tests now cover this
case. The fix permits bounded start drift only after a valid timed anchor and
keeps original estimates intact. New anchors use one alignment observation.

The same replay then processed all 528,000 samples in 17 decodes, with eight
commits including EOF. The loop took 44.998 seconds on CPU. The queue was empty,
declared capacity was restored, and the model fingerprint was unchanged after
the run. This is a manual integration smoke, not a registered quality or speed
comparison; no reference-transcript error metric was computed. It does not
close the long-session gate. No models were downloaded and no GPU was used.

At source commit `93588fd`, the runtime suite passes 382 tests, including 52 new policy, adapter,
and stream tests. The same 382 tests pass when imported from the built wheel.
The repository-tool suite passes 249 tests. Type, lint, format, and distribution
checks also pass.

The later [T4 comparison](research/2026-09-05-word-alignment-comparison.md) stops
at a partially retained anchor. The [repair replay](research/2026-09-05-word-alignment-replay.md)
passes that boundary and commits through 12.78 seconds. It then stops on a
window-leading word start mismatch. The window-edge exception above was added
after that run. A [subsequent T4 replay](research/2026-09-05-word-alignment-completion.md)
completes the word profile with eight commits across 33 seconds, zero normalized
word edits against the model controls, and restored capacity. Exact strings
differ. This short unpaced test does not qualify long-session or live latency;
its final 11.26 seconds are committed at EOF. The segment profile still stalls.

The [independent-reference T4 diagnostic](research/2026-09-05-word-corpus-diagnostic.md)
completes three separate speech clips but exposes a punctuation-only coverage
advance and hallucinated text on digital silence. The later local guard requires
a nonfinal publication to end on a unit containing a Unicode letter or number.
Trailing standalone punctuation waits for a stable lexical unit or EOF; internal
punctuation and native alignment data stay intact. A punctuation-only anchor
cannot authorize nonfinal advancement. EOF authority is unchanged. This guard
does not detect silence or reject lexical hallucinations, and has not yet been
replayed on T4. The mixed-input and long-session gates remain open.
