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
transaction commits and releases its resources. Each new rolling window starts
at the committed endpoint. This version does not retain left audio context or
carry an unbounded text prompt across windows.

At EOF, the remaining window is finalized under the ordinary native decode
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

Next, test the effect of window boundaries and retained context on recognition
with matched inputs. Do not tune the policy solely to one JFK phrase. Add real
silence, distinct recordings, and long paced input before widening the profile's
claims. Preserve the failed text comparison as a regression observation.

Deterministic tests exercise agreement and runtime lifecycle behavior without
requiring a model. Native GPU diagnostics must record the tested source snapshot,
model, input, configuration, events, and outcome separately. A short diagnostic
does not close the 30-minute continuous-operation gate in `ROADMAP.md`.

No speedup, lower GPU cost, durable crash recovery, or unrestricted continuous
transcription is claimed by this profile.
