# Fixed overlap observations

## Question

Does one anchor-bearing overlap supply the correspondence missing from a
head-only continuation? Test the five recorded failures before changing the
live selector. This experiment observes native outputs. It cannot publish,
evict retained audio, or declare a stream complete.

## Fixed workload

Use the existing hashed LibriSpeech PCM and transformations. No new input or
model download is needed. All five cases are development data. Human reference
text scores the complete proposed transcript after assessment; it does not
select a window, match anchors, or set a timestamp tolerance.

| State | New native windows, absolute milliseconds |
| --- | --- |
| Paced noisy prefix | Control 1680–10890; head-only 3680–10890; overlap 2020–10890 |
| Context, no added pauses | Overlap 20720–33660 |
| Context, continuous noise | Overlap 32400–43660 |
| Speaker 2961, clean | Overlap 1120–8220 |
| Speaker 8455, noise | Overlap 1100–7740 |

The last four head-only observations already exist in the word-resolution
archive. Reuse their records only with an explicit compatibility assessment.
Their old held-out designation does not apply to this follow-up. The two
speaker-specific prefixes were synthesized from past-only bootstrap outputs,
not emitted by an actual continuous stream.

There are seven native windows, including the new control, and no warmup or
automatic retry. Run one T4 function with the cached FP32 `tiny.en` model, one
model owner and CUDA lane, a 180-second function timeout, and at most one
container. These execution limits are not an invoice cap. Verify the frozen
source and CPU transport before the GPU function. Never overwrite an attempt.

## Comparison rules

Keep legacy alignment in every new window. Encoder reuse has a separate paired
record; it is not the variable in this boundary experiment. Record exact PCM
slices, model state and checkpoint, backend code, decode options, random seed,
tokenizer and preprocessing identities, precision, and alignment mode wherever
they are observed. Distinguish observed identity from a declaration recovered
from an old producer. A missing identity remains missing.

The paced archive did not retain the failing alignment. The new control is a
separate native observation with full alignment; it must not be inserted into
the historical trace as if it had been recorded there. Compare its native text,
tokens and other available outputs with the archived control. Keep differences.

Use the existing structural handoff assessor without changing its thresholds:
one unique raw text/token anchor, existing 200 ms word-boundary tolerance, and
agreement of the entire overlap continuation with the entire head-only result.
Do not normalize punctuation for matching, select a convenient subsequence,
move the frozen boundary, or search for another overlap after a refusal.

The assessor always returns `publication_authorized=False`. Matching words and
times can share an omission; they are not independent acoustic ground truth.
Compatibility checks and human-reference scores are separate from structural
eligibility. A failed decode or mismatch must remain in the report.

## Cost and lifetime checks

For each new window, record encoder forwards and padded frames, decoder work,
host phase times, allocated/reserved memory, and completion-fence/capacity
release. Count all seven windows, including the control and failed attempts.
Old inference was not free: distinguish fresh experiment work from reused
records when comparing a prospective policy's cost.

No device-time or energy claim follows from host durations. Shorter slices can
still use Whisper's full padded encoder input. Stop further native work if a
failed fence retains capacity. Preserve the partial record.

## Decision after measurement

If the fixed overlaps do not recover a unique anchor and a comparable complete
continuation, reject or narrow this recovery hypothesis on these states. If
structural agreement passes but words are still omitted, keep publication
blocked and investigate the missing acoustic coverage. Do not turn a diagnostic
success into a live-recovery claim.

Only then choose the smallest selector change supported by the observations.
It must fit the existing attempt budget and preserve the default profile.
Evaluate that selector on unseen speakers and complete paced streams before
claiming improved live quality or compute efficiency.
