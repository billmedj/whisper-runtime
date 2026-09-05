# Continuous transcription

`timestamp_agreement_stream/v1` is an experimental rolling-input profile. It
uses the existing native decoder, completion fences, and session commits.
It does not change the existing bounded-preview or offline adapters.

## Use

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

## Publication rule

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
