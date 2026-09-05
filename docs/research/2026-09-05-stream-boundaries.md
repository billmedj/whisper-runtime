# Continuous transcription: publication and context boundaries

Date: 2026-09-05. Inspected runtime commit: `5de3723`.

This is a diagnosis and an implementation plan. It does not record a new model
run or a validated recognition fix. No GPU work was started for this analysis.

## What the records establish

The [11-second diagnostic](../../evidence/modal-t4-tiny-en-continuous-smoke-v1-2026-09-05.json)
matches its same-options control after whitespace normalization. The
[33-second diagnostic](../../evidence/modal-t4-tiny-en-continuous-smoke-v2-2026-09-05.json)
processes three concatenated copies of that recording. It publishes four commits
before EOF, then finalizes the remainder. Its transcript differs from the
segmented control. Neither run used wall-clock pacing.

The second record reports 528,000 accepted and committed samples, 34 decodes,
40 ordered events, and a peak input buffer of 168,960 samples (10.56 seconds).
The resource-release and publication checks pass. These results establish input
accounting and lifecycle behavior on this case, not word accuracy or live latency.

Its cumulative decoded source spans total 188.84 seconds for 33 seconds of input:
5.72 times the input duration. This is a reprocessing indicator, not a GPU cost
ratio. The harness pads each analysis to Whisper's fixed input length before
computing its mel features. Short source spans therefore do not imply equally
short encoder work. Measure encoder time, decoder time, and preprocessing
separately before making an efficiency claim.

The record's `status: passed` refers to its registered lifecycle checks. The
recognition comparison is explicitly false. Future reports should present these
two outcomes separately, without changing the historical record.

## The useful failure

All positions below are source-audio times, not wall-clock emission times.
Event numbers refer to the second diagnostic.

| Events | Observation | Interpretation |
| --- | --- | --- |
| 3, 4 | At 3 seconds, the preview says `am I`; at 4 seconds, it changes to `my`. | The error can occur without a rolling crop. More input resolves this instance. |
| 15, 16 | At 14 seconds, the second copy also says `am I`; at 15 seconds, it changes to `my`. | The same transient error recurs with earlier acoustic context retained. |
| 27 | The publication boundary advances to 22.540 seconds. The third copy starts at 22.000 seconds. | The next analysis discards the first 540 ms of that copy. This does not identify which phonemes were lost. |
| 28-31 | The next window repeats the error and commits `And so am I fellow Americans.` | Agreement and timestamp holdback do not establish recognition accuracy. |

The last publication ends at 25.540 seconds. Its event endpoint is the selected
publication boundary, not necessarily the endpoint of the analysis that produced
it. The present record does not retain every raw hypothesis and decision.

**Established:** a transient recognition error became immutable in the third
copy. The code couples publication to removal of all preceding audio.

**Plausible:** the interaction between a shifted acoustic start and insufficient
future context prolongs a wrong hypothesis until it passes agreement.

**Not established:** the 540 ms crop is the sole cause, an overlap fixes the
problem, or a larger model cannot make the same class of error. A concatenation
seam, model behavior, timestamp shifts, and segmentation remain alternative
explanations. Agreement between two correlated decodes is not independent
corroboration.

## Internal recrossing

| New observation | Existing object | Relation | Consequence |
| --- | --- | --- | --- |
| Commit immediately crops the audio | `continuous_stream.py`, `_publish_commit` and `step` | Coupled state | `_head` is both the committed position and the retained-audio start. Split these meanings. |
| A published prefix can remain useful context | `native_whisper.py` publication API; `NATIVE_ADAPTER.md` | Existing capability | Analyze overlapping audio while publishing only new complete segments. No second transaction engine is needed. |
| Overlap can move a timestamp boundary | `stream_policy.py`, `compare_hypotheses` | Integration constraint | Retention alone is insufficient: a segment that straddles the commit boundary currently prevents progress. |
| Empty output cannot account for a silence interval | `CONTINUOUS_STREAMING.md`, unresolved input | Known release blocker | Add explicit interval handling. Do not treat missing words as proof of silence. |
| Many growing analyses revisit audio | RFC stable-streaming profile; roadmap D4 | Measurable optimization target | Coalesce obsolete previews before attempting encoder-cache reuse. |
| A useful pause need not mean a cancelled task | Native token-step run | Potential extension | Add a separately specified wait-for-input path if attention-guided decoding is adopted. Existing cancellation is not that path. |

## Research ledger and bridges

These are primary sources. The proposed transfers are engineering inferences,
not claims that the papers prove this implementation correct.

| Source | Supported result | Bridge to this repository | Limit |
| --- | --- | --- | --- |
| [Liu et al., Interspeech 2020](https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.html) | Partial-hypothesis selection supports low-latency recognition and translation. | Keep agreement as an attributed baseline. | Matching hypotheses do not certify truth. |
| [Machacek et al., IJCNLP 2023, section 3](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) | Whisper-Streaming retains some confirmed audio, matches already confirmed words, trims its buffer, and carries text context. | Our current rolling crop omits an important part of the comparison method. | Their word-level matching is not our stricter whole-segment policy. |
| [Papi et al., ACL 2024, sections 3.1-3.2](https://aclanthology.org/2024.acl-long.202.pdf) | StreamAtt separates hypothesis selection from history selection. It uses cross-attention for emission and audio-history decisions. | Treat publication and context retention as separate policy outputs; evaluate attention as an optional signal. | Speech-translation results on their models do not transfer automatically to native Whisper ASR. Attention is not a proof of acoustic dependence or word accuracy. |
| [SimulStreaming, author implementation](https://github.com/ufal/SimulStreaming) | Its Whisper path uses attention-guided stopping near the available-audio boundary and carries context between windows. | Compare an attention-guided policy with our agreement baseline. Token-step execution is a useful integration point. | Our API does not yet provide this policy or safe cross-input cache continuation. |
| [Akidau et al., VLDB 2015, section 2.3](https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf) | Dataflow separates window grouping from result triggers; estimated watermarks alone do not settle completeness. | Keep source progress, publication decisions, and retained state distinct. | Later audio changes an ASR interpretation; it is not literally an out-of-order database record. |
| [Naiad, Microsoft Research](https://www.microsoft.com/en-us/research/project/naiad/) | Progress tracking accounts for possible future records while computation proceeds asynchronously. | Resource release must respect outstanding work, not only visible output. | Exact runtime progress says nothing about statistical recognition correctness. |

Publication/history separation and attention-guided streaming already exist.
The opportunity here is a compact integration with explicit execution ownership,
bounded state, recovery, and reproducible quality measurements. A novelty claim
requires a comparison with those implementations, not just the offline baseline.

## Smallest coherent change

Keep one stream controller. Give it separate positions, in source samples:

```text
retained_from <= published_through <= received_through
```

`published_through` records an irreversible output decision. `retained_from`
records what remains available to the next analysis. Retained context counts
against the existing memory and analysis-window limits. Active work must keep
its input valid until cleanup completes. These are policy-relative guarantees;
no finite context proves that arbitrarily distant speech is irrelevant.

1. Retain a bounded left context after publication. Test small overlap values
   before choosing a default. Do not carry the entire session.
2. Reconcile newly decoded overlap with committed text. The existing complete-
   segment API is useful, but a straddling segment needs an explicit policy.
   Ambiguous repeated phrases must not trigger silent deduplication or timestamp
   rewriting. Keep them provisional, wait, or report an unresolved boundary.
3. Keep publication, silence classification, and input accounting distinct.
   An explicit endpointer can use pre-roll, post-roll, and hysteresis. VAD is a
   fallible classifier; test quiet speech and false negatives. It cannot justify
   a claim that no speech was lost.
4. Preserve fast provisional output while testing a stronger finality rule.
   Extra future context and a shifted analysis start are diagnostic controls.
   They are not independent votes or a universal correctness test.
5. Under load, skip superseded preview endpoints without dropping admitted audio.
   Do not cancel every in-flight preview: that can prevent any result completing.
   A live microphone cannot be paused indefinitely; report overload explicitly.

Do not reuse decoder or encoder caches across changed audio without a separate
validity contract. Keeping tokens or a checkpoint does not make Whisper's
noncausal encoder incremental. Attention-guided stopping is a later optimization
candidate; storing full attention matrices or disabling fast attention can cost
more than it saves.

## Action register

| Order | Inputs and work | Question and acceptance | Resource decision |
| --- | --- | --- | --- |
| 1 | Existing event record; compact per-decode trace fields | Retain analysis start/end, publication span, policy reason, segment times, and EOF mode. Can we reconstruct each decision without inferring analysis time from an output event? | Local first. No model required. |
| 2 | Stream controller and scripted backend | Separate retention/publication positions. Test overlap, straddling segments, repeated phrases, full buffers, EOF, cancellation, and post-commit release failure. No duplicate publication or silent input loss. | CPU tests before GPU. Preserve the existing profiles. |
| 3 | Same 33-second PCM, model, options, seed, precision | Compare baseline, left overlap only, more right context only, and both. Keep text prompting off initially. Add a same-input offline transcription control, not only the repeated 11-second control. | One bounded diagnostic job after local preflight and a new registered spending limit. Do not bypass the exhausted two-call guard. |
| 4 | At least two distinct licensed recordings and reference transcripts; shifted seams and quiet speech | Does the improvement extend beyond JFK? Measure substitutions, insertions, deletions, boundary errors, revisions, and delay. An improvement on one phrase alone is insufficient. | Small matched cases before a long run. |
| 5 | Corrected policy, silence/gap handling, paced replay | Close D1 with at least 30 minutes, measured host/device memory, source accounting, slow decoder and consumer cases. Then expose the small microphone/file CLI. | GPU only for model work. Stop on the first actionable failure. |
| 6 | Passing quality baseline, per-stage timings | Compare preview coalescing and then optional attention-guided stopping at matched quality and latency. | Defer a broader performance campaign until correctness gates pass. |

Falsifiers: if left overlap does not help but more future context does, prioritize
finality rather than cropping. If neither helps, investigate model/segmentation
behavior before adding retention complexity. If overlap creates omissions or
duplicates, reject that policy despite a better aggregate error rate. If a gain
comes only from more latency or compute, report the tradeoff, not a speedup.

The next milestone is reliable progressive captions under a declared context,
latency, and memory budget. Durable GPU-free suspension, unrestricted live
translation, and a general inference speedup remain separate, unclosed goals.
