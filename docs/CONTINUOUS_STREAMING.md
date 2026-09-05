# Timestamp agreement stream

`timestamp_agreement_stream/v1` is an experimental rolling-input profile. It
uses the existing native decoder, completion fences, and session commits.
It does not change the existing bounded-preview or offline adapters.

## Use

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
remain unchanged. Current stream profiles retain their timed-segment rules.

This comparison is intended for trace analysis and for testing a future text
publication contract. Agreement alone is not an acoustic alignment or accuracy
test. An empty anchor is valid only when no text has been committed in the block.

## End of input

At EOF with zero retained context, the remaining window uses the native decode
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
producer eventually receives backpressure. The profile does not interpret a
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
