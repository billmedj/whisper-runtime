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

## Recorded result

The [T4 record](../../evidence/modal-t4-tiny-en-resolution-handoff-2026-09-06.json)
contains all seven observations from source commit
`2dbf1dcb3c34cde109829f49d9697e33bf842788`. Its JSON SHA-256 is
`545e6e8d0d420539b5e94333e5aeeba2095886577f9daca0382cc183344b9a6c`.
The status `completed` means that the observations ran. **All five structural
handoff assessments rejected. None authorizes publication.** No stream was
resumed, no audio was evicted, and no selector rule changed in this experiment.

| State | Recorded refusal | Inspection of the raw words |
| --- | --- | --- |
| Paced noisy prefix | Anchor absent | `place amidst the tents` becomes `and it's the tense`; the overlap omits the next utterance. |
| Context, no added pauses | Complete suffix disagrees | The anchor matches. Only the first suffix unit differs: ` She` (token 1375) versus ` she` (673). |
| Context, continuous noise | Anchor absent | The initial `and` disappears from `and bird and tree`; later timings also disagree. |
| Speaker 2961, clean | Anchor absent | `danger` disappears from `danger of the modern`; the continuation also changes. |
| Speaker 8455, noise | Anchor absent | `Eva's` becomes `Eve's`. This is a name substitution, not a case change. |

On the new paced noisy case, the prefix plus head-only proposal has **1 word
edit in 26 reference words**, compared with 18 for the previously incomplete
transcript. The remaining substitution is `son` for `sun`. This is a scored
proposal, not recovered live output. The other four proposals reuse the old
head-only observations; their edit distances remain 6, 6, 4, and 0. A zero
normalized word score is not an exact raw-text match or acoustic proof.

The fresh noisy control reproduces every archived native field except its
window identifier. Its newly recorded alignment remains separate from the
old trace. The assessment joins these two observations explicitly; it is not
an archived live transition. All input and compatibility checks pass under the
declared comparison. Historical package identities remain unmeasured. The
old four observations use a source-checked bridge to the backend with the
unused optional feature-reuse patch; their backend trees are not identical.

### What the failures separate

In the no-added-pauses case, the overlap has one exact raw text/token anchor,
with boundary differences of at most 20 ms. Both suffixes have 35 units. Their
last 34 units match in raw text and tokens, including punctuation; all suffix
boundary differences are at most 60 ms. The initial capitalization changes
the token too. This is a specific representation disagreement. It does not
justify case-folding every comparison or accepting the other four cases.

The other failures involve missing or changed words. Even apparently matching
fragments can disagree in time: the noisy context places the next `And`
2,580 ms later. In the speaker-8455 case, the head-only candidate begins 100 ms
before the observed end of `husband`, independently of the name substitution.
Comparing these fragments is diagnostic only; they are not substitute anchors.

The noisy overlap assigns its final period the interval 3,600–10,860 ms while
its lexical content stops near 3,600 ms. A punctuation interval that reaches
the end of an input cannot establish coverage of the speech in that interval.
The head-only result supplies the missing utterance, but not the evidence
needed to join it safely to the frozen prefix.

### Work and resource lifetime

All seven runs close and restore capacity. The model fingerprint is unchanged.
There are 14 encoder forwards, each with 3,000 padded frames: legacy decode
and alignment each encode every window. This represents 42,000 padded input
frames, not GPU time or an energy measurement. The raw slices total 63.23 s.
Peak PyTorch allocation is 290,098,176 bytes; allocation after each result
handle is released is 160,720,896 bytes.

The recorded loop duration is 12.88 s. It excludes initial case reconstruction,
model setup, and initial identity checks. It is neither total remote invocation time
nor billed GPU time. The Modal application stopped after the one bounded run.
There were no automatic retries or further GPU experiments.

### Architectural consequence

The fixed hypothesis fails on these five states: starting an overlap at the
first estimated anchor-word onset does not reliably supply a usable handoff.
The record distinguishes a representation mismatch from absent lexical evidence
and timing disagreement. Treating every refusal as a reason to widen tolerance
would hide these different causes.

The next small change should expose these diagnostic differences while keeping
the original refusal. Then test an input-boundary choice with retained acoustic
context, rather than treating an estimated word onset as an exact cut. Whether
cutting context caused these omissions remains a hypothesis; these text outputs
alone cannot establish that mechanism. Any new choice needs a fixed comparison,
an attempt budget, and separate speech-coverage tests before live publication.

This step also records declared model, decode options, and seed in runtime
resolution observations. Unknown tokenizer, preprocessing, backend, and effective
alignment identities remain unknown unless measured. This receipt identifies
requested work; it is not replay authority or a cache across changed audio.

CPU regression checks: `PYTHONPATH=src python -B -m unittest tools.test_modal_resolution_handoff`.
The archive checks need no GPU. Full input reconstruction also needs the existing
local PCM fixtures; it does not download them.
