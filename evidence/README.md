# Integration evidence

The [low-latency v2 T4 check](../docs/research/2026-09-07-low-latency-v2-gpu-results.md)
passes two full source-paced sessions. First confirmed text arrives at
7.358/6.186 seconds, with identical exports and 7/114 word edits. The
[verified archive](modal-low-latency-v2-2026-09-07.zip) includes exact source,
input, timing records, startup receipts, independent audit and stopped-app
observation. This is a short check, not endurance or a stable release.

The [two-observation CPU replay](../docs/research/2026-09-07-two-observation-holdback.md)
completes the full 46.55-second input with first confirmation after 6 seconds of
admitted audio and 7/114 word edits. Its
[event audit](two-observation-cpu-2026-09-07.json) checks final coverage, exports,
source identity and cleanup receipts. This is accelerated CPU evidence, not
source-paced GPU latency or endurance.

The [verified-startup replay](../docs/research/2026-09-07-verified-startup-results.md)
passes two short T4 sessions with the guarded backend importer and SourceClock
v2. Full coverage, identical exports and CUDA capacity return pass. The initial
maximum wake delay is 206.387 ms against the unchanged 250 ms bound; its interval
overlaps the first DTW operation, without establishing causality. The
[exact-byte archive](modal-verified-startup-2026-09-07.zip) includes frozen source,
raw timing/input/result records, an independent audit and a stopped/zero-task
receipt. This is not endurance, clean-install or physical-microphone qualification.

The [instrumented source-clock replay](../docs/research/2026-09-07-source-clock-replay-results.md)
passes two short T4 sessions under the submitted 4 GiB limit. Coverage and
exports match, and terminal CUDA allocation is unchanged. Initial maximum
source lateness is 222.77 ms against a 250 ms bound. The
[verified archive](modal-source-clock-replay-2026-09-07.zip) retains source,
input, raw timing records, audit and stopped-app observation. This does not
qualify endurance or repair the preceding failure.

The [4 GiB provider-capacity smoke](../docs/research/2026-09-07-capacity-smoke-results.md)
stopped on a source-clock delay during its first session. Cleanup passed; no
session completed and no endurance stage started. The
[failure archive](modal-capacity-smoke-2026-09-07.zip) preserves the exact source,
input, raw observations, independent audit and stopped-app observation. It is
not a passing capacity result or an observed out-of-memory termination.

The [native release endurance attempt](../docs/research/2026-09-07-release-soak-results.md)
stopped at its smoke-stage RAM check. Its 46.55-second source completed with
verified coverage, 7/114 word edits and released runtime capacity, but process
RSS exceeded the preregistered ceiling. No hourly stage ran. The
[independent audit](release-soak-smoke-audit-2026-09-07.json) and
[attempt archive](modal-release-soak-attempt-2026-09-07.zip) retain the failure,
events and exact executed source. This is not a passing endurance result.

The [new-audio and restart checks](../docs/research/2026-09-07-draft-qualification-results.md)
extend native draft testing to two new speakers and an actual CPU process
restart. On the 15.485-second T4 input, decoder forwards fall from 133 to 72
per stream and pooled decode-phase wall falls by 23.5%, with exact events and
tokens. The separate restart retains full logical output while dropping the
old draft hint. Threshold counterexamples remain: equal tokens do not prove
identical decisions for scores straddling 0.6. Drafts remain opt-in. See the
[GPU archive](modal-draft-holdout-2026-09-07.zip),
[independent GPU audit](draft-holdout-audit-2026-09-07.json),
[CPU restart audit](native-draft-fresh-process-cpu-audit-2026-09-07.json) and
[threshold audit](draft-threshold-audit-2026-09-07.json).
The [CPU restart archive](native-draft-fresh-process-cpu-2026-09-07.zip) is a
documented privacy derivative, not the original raw report bytes: 19 metadata
paths were replaced while executed source, audio, savepoint and numerical
evidence remain unchanged. Its original/public hashes and derivation manifest
are linked from the audit. Verify it with
`python -B tools/verify_public_cpu_archive.py evidence/native-draft-fresh-process-cpu-2026-09-07.zip`.

The [integrated draft T4 comparison](../docs/research/2026-09-07-integrated-draft-gpu-results.md)
tests the native opt-in in ABBA order on four 93.1-second source-paced streams.
Decoder forwards fall from 2,382 to 1,692 per stream, and pooled decode-phase
native wall time falls by 18.2%, with improvement in both orders. All 77 events,
27 commits and current-audio tokens match. The first control has extra alignment
wall time that is not credited to drafts. Known repeated audio, unchanged word
errors and small score differences limit the result. See the
[record archive](modal-integrated-draft-2026-09-07.zip) and
[offline audit](integrated-draft-gpu-audit-2026-09-07.json). Defaults are unchanged.

The [native draft integration](../docs/research/2026-09-07-native-draft-integration.md)
replaces diagnostic method patches with an opt-in request-local path. On a
33.66-second CPU fixture, decoder forwards fall from 1,009 to 810; a logical
checkpoint/restore arm uses 820. All 29 events, 17 token traces and 11 commits
match. A real cancellation probe releases native state and capacity. The
[archive](native-draft-integration-2026-09-07.zip) includes the executed source
and raw result. These are CPU correctness/count checks, not new GPU timings.

The [verified-draft T4 comparison](../docs/research/2026-09-07-draft-gpu-results.md)
reduces decoder forwards from 1,214 to 917 on the original 46.55-second mixed
input. The CUDA forward-interval sum falls by 21.1%; decode-phase native wall
time falls by 16.9%. All 25 raw token sequences and 14 commit texts and source
spans match, with unchanged peak memory. Submitted token positions increase
by 15.3%, and small score differences remain. Control-only alignment stalls
prevent attributing the full total-wall reduction to drafts. This is one
fixed-order test, not a general speed or production claim. See the
[exact archive](modal-draft-features-2026-09-07.zip) and
[offline audit](draft-gpu-audit-2026-09-07.json). Defaults are unchanged.

The [decoder-work investigation](../docs/research/2026-09-07-decoder-work.md)
accounts for the 216 extra T4 decoder forwards and tests two remedies locally.
The [verified-draft CPU stream](stream-draft-cpu-2026-09-07.json) reduces decoder
forwards from 1009 to 810 while preserving the archived text, commit boundaries,
admission cadence and policy trace summaries. Encoder calls stay at 17; decoder
input tokens increase. This does not establish lower GPU time or bit-identical
scores. The same report retains the short-context candidate's longer commit
gaps. Production defaults are unchanged.

The [composed T4 result](../docs/research/2026-09-07-composed-gpu-results.md)
completes three matched source-paced arms. Reuse removes 18 encoder forwards
with exact fast-arm commit parity. The combined profile publishes earlier and
uses less peak allocated memory than the conservative control, but retains a
10.92% higher forward interval sum. See the
[exact archive](modal-composed-features-2026-09-07.zip) and
[offline cross-arm audit](composed-gpu-audit-2026-09-07.json).

The [composed CPU comparison](composed-cpu-2026-09-07.json) connects earlier
commits with same-window alignment reuse. Encoder forwards fall from 11 to 7
and from 31 to 17 on two fixed inputs, with exact text, commit-boundary, trace
and metric parity. The initial CPU timeout is retained. The
[result and limits](../docs/research/2026-09-07-composed-inference-results.md)
distinguish these CPU counts from the subsequent matched T4 measurements above.

The [retained-context comparison](../docs/research/2026-09-06-retained-context-commits.md)
publishes its first commit at 13.220 seconds in a short T4 run, compared with
31.156 seconds in the earlier smoke test on the same input. Full-sequence word
edits remain 7/114; summed source-window duration increases, not a measured GPU
work total. The
[receipt archive](modal-early-commit-2026-09-06.zip) and
[offline event audit](retained-context-live-audit-2026-09-06.json) preserve this
short-only result. CLI defaults and the earlier long-run claims are unchanged.

The [single-stream live result](../docs/research/2026-09-06-live-v01.md) completes
46.55 seconds, then 30 source-paced minutes on one T4 with one loaded model.
Input hashes, full sample coverage, FINAL and cleanup pass. The long source
repeats a short licensed cycle; its 6.14% word-edit rate is not held-out accuracy.
The [exact record archive](modal-live-v01-2026-09-06.zip) includes both event logs
for offline verification. General release qualification remains open.

The [noisy crop/prompt diagnostic](modal-t4-tiny-en-noisy-context-prompt-2026-09-06.json)
records four observations in two native windows on T4. Published decoder history
restores a missing continuation on unchanged audio; the result loses the old
anchor, so no text is published. Six encoder forwards are measured. A known
tuple/list comparison error affects one control flag, not the observations;
the original receipt is preserved with a
[separate offline validation](noisy-context-prompt-offline-validation-2026-09-06.json).
See the [results and next local boundary test](../docs/research/2026-09-06-noisy-context-prompt.md).

The [additional group-anchor replay](group-validation-2026-09-06.json) validates
24 recorded traces before scoring 14 observations from eight anchor states.
Both the strict and group rules match all 14. No regression or additional gain
is observed. This set adds one source fixture outside the tuning states; it is
not 14 independent tests. Ten exclusions remain explicit. No inference runs or
live policy changes occur. See the
[results and repetition tradeoff](../docs/research/2026-09-06-group-correspondence.md#additional-saved-history-validation).

The [group-boundary replay](group-correspondence-2026-09-06.json) is a CPU-only,
post-hoc comparison on five saved states. Four still reject. Speaker 2961 has
an exact group and continuation, but becomes structurally eligible only with
the separate 200 ms boundary tolerance. The text is unchanged; this is not
live recovery or improved recognition. The
[source trace and limits](../docs/research/2026-09-06-group-correspondence.md)
explain the raw first-word timing behavior and the remaining evidence gap.

The [fixed context-guard observations](modal-t4-tiny-en-context-guard-2026-09-06.json)
add three T4 windows and reuse two existing controls. Two missing anchors are
restored, but all five comparisons still reject. One crop changes a word's
estimated start by 500 ms without changing any word text or word tokens.
Another has an exact complete suffix but an interior anchor boundary moves
300 ms. No publication rule or live profile changed. All three new calls close
and restore capacity. See the
[results and next test](../docs/research/2026-09-06-context-guard-observations.md#recorded-result)
and the [single-attempt receipt](modal-context-guard-attempt-2026-09-06.jsonl).

The [CPU disagreement replay](resolution-disagreements-2026-09-06.json) separates
complete-unit text, token and estimated-time differences in the fixed overlap
archive. All five original refusals remain unchanged. It identified two existing
controls and three missing intervals for the later context-guard comparison.
This CPU replay performs no GPU work. The
[report](../docs/research/2026-09-06-resolution-disagreements.md) explains why the
saved noisy control already refutes this guard as a universal recovery rule.

The [fixed overlap observations](modal-t4-tiny-en-resolution-handoff-2026-09-06.json)
complete seven native windows on one T4 at source commit `2dbf1dc`. All five
structural handoff assessments reject; none authorizes publication. The new
noisy head-only proposal reduces whole-text edit distance from 18 to 1, but its
overlap omits the continuation. One other refusal is only a `She`/`she` raw
text and token difference; the remaining cases lose or replace words. All
seven runs release capacity, with 14 legacy encoder forwards and an unchanged
model. See the [results and limits](../docs/research/2026-09-06-overlap-observations.md#recorded-result).
This is a completed diagnostic, not recovered live transcription.

The [paced encoder comparison](modal-t4-tiny-en-paced-features-2026-09-06.json)
runs all six registered cells at source commit `b1901c7`. The normal and
attenuated inputs complete in both arms with identical committed text and spans.
The noisy prefix remains unresolved at 3.68 of 10.89 seconds in both arms, so
the experiment is **failed** overall. On the full mixture, encoder forwards fall
from 47 to 25. Peak PyTorch allocations fall from 290,098,176 to 190,260,736 bytes;
reserved memory stays unchanged. All 76 native runs release capacity and the
model is unchanged. See the [paired results and limits](../docs/research/2026-09-06-paced-feature-reuse.md#recorded-result).
This is one-worker diagnostic evidence, not general performance qualification
or recovered noisy live transcription.

The [same-window feature comparison](modal-t4-tiny-en-alignment-features-2026-09-06.json)
completes eight paired inputs on one T4 at source commit `27ee020`. Reuse removes
one of two encoder forwards per analysis, with exact native and aligned-word
outputs on every pair. Decode and legacy alignment features are numerically
different; output equality is measured, not assumed. All sixteen runs close
and restore capacity. Native output on digital silence remains incorrect.
See the [report](../docs/research/2026-09-06-alignment-feature-handoff.md) for
identities, memory, timing regressions and scope. This validates opt-in
fixed-window reuse, not general speedup, live recovery or acoustic accuracy.

The [CUDA lane comparison](modal-t4-tiny-en-cuda-lane-2026-09-06.json) preserves
exact native output across twelve analyses. Alternating arms exposed a retained
result-handle measurement effect; its original per-call plateau criterion failed.
The [post-handle-release comparison](modal-t4-tiny-en-cuda-lane-blocked-2026-09-06.json)
then records exactly zero allocation change across six consecutive reused-lane
comparisons. Four fresh-stream controls each add 8,519,680 bytes. Both models
are unchanged and both applications stopped. The
[report](../docs/research/2026-09-06-cuda-lane-lifetime.md) retains both results,
their measurement scopes, source identities and limits. This is a short repeated
workload, not a general speedup, live recovery result or long-session memory test.

The [terminal-window experiment](modal-t4-tiny-en-word-resolution-2026-09-06.json)
completed ten native analyses on one T4 at source commit `1977c68`. Two saved
failure states reproduce exactly. Diagnostic-based selection reduces word edit
distance in three of four proposed full texts; the clean held-out state remains
unresolved. This is a counterfactual experiment, not stream qualification. The
[report](../docs/research/2026-09-06-word-resolution.md#t4-result) includes every
arm, cold-start effects, repeated encoder work and observed memory growth.
The JSON SHA-256 is
`6415da1704b84663780ddd1eda8e18b6daffaae72c5c2501207474ed81078b4f`.
Local tests replay all choices and scores from the stored alignments without a
model or audio download.

The [local word-resolution replay](word-resolution-replay-2026-09-06.json)
reproduces all four terminal failures in the word-context comparison below.
It separates absent lexical anchors from timing mismatches and finds one
structurally eligible terminal-end correspondence. It uses no GPU and cannot
publish, change qualification or establish acoustic correctness. The
[report](../docs/research/2026-09-06-word-resolution.md) documents the rules,
counterexamples and the subsequent T4 test.

The [word-context comparison](modal-t4-tiny-en-word-context-2026-09-06.json)
is **failed**, with matched controls. The candidate advances noisy-input
publication from 6.10 to 33.54 seconds, but does not finish its 43.66 seconds.
The no-added-pauses case remains at 21.26 seconds. Normal and attenuated
candidates complete within their matched quality gates. At the final noisy
boundary the lexical anchor exists but its last word's end shifts by 540 ms;
the no-added-pauses native result still omits the continuation. See the
[comparison and limits](../docs/research/2026-09-05-word-context.md).
No general acoustic qualification or performance advantage follows.

The [acoustic-boundary T4 diagnostic](modal-t4-tiny-en-acoustic-boundaries-2026-09-05.json)
is **failed**. The quiet control and the new hybrid profile complete the normal
43.66-second input; the attenuated hybrid also completes. Without detected
pauses, publication stops at 21.26 seconds on concatenated speech and 6.10
seconds on noisy input. Accepted, uncommitted audio is retained and no false
final event is emitted. The [report](../docs/research/2026-09-05-acoustic-boundaries.md)
separates missing native continuation from a fragile single-word anchor. One
T4 invocation ran the five cells. This unpaced diagnostic does not qualify
general live speech, noise tolerance or lower GPU cost.

The [PC-to-Modal T4 replay](modal-t4-tiny-en-network-replay-2026-09-05.json)
sends 43.660 seconds from Windows over an authenticated WebSocket. First text
returns to the PC at 2.426 seconds; five commits precede EOF. All 698,560 samples
are committed and the final text matches both offline controls. Six
human-reference edits remain. The application stopped and its temporary proxy
token was deleted. See the [network report](../docs/research/2026-09-05-network-replay.md),
including the retained failed CPU preflight. This short file replay does not
qualify microphones, noisy speech, long sessions or GPU efficiency.

The [paced T4 replay](modal-t4-tiny-en-paced-replay-2026-09-05.json) completes
43.660 seconds supplied in 20 ms chunks by an independent producer. First text
appears at 2.097 seconds, and five commits precede EOF. All samples are accounted
for; the final text matches both model controls exactly. Six human-reference
edits remain. All 32 checks pass. See the
[timing definitions and limits](../docs/research/2026-09-05-paced-replay.md).
This is a short same-worker replay, not PC-to-Modal streaming or a microphone
qualification. It demonstrates no GPU saving.

The [automatic-endpoint T4 diagnostic](modal-t4-tiny-en-automatic-endpoints-2026-09-05.json)
completes the 43.660-second mixture without supplied boundaries. Five quiet-run
endpoints and EOF produce six contiguous commits. Text matches both model
controls exactly; six human-reference edits remain. The detector preserves PCM
and does not authorize silence publication. See the
[report and limits](../docs/research/2026-09-05-automatic-endpoints.md). Clean speech
with constructed pauses does not qualify noisy microphones or live latency.

The [caller-delimited source-unit T4 diagnostic](modal-t4-tiny-en-source-units-2026-09-05.json)
completes the 43.660-second mixture through one continuous controller: six speech
occurrences, five empty pauses, and exact contiguous input accounting. Normalized
words match the model controls; six human-reference edits remain. Boundaries are
supplied from the fixture construction, not detected acoustically. See the
[report and limits](../docs/research/2026-09-05-source-units.md). This separate
profile does not establish automatic live transcription or a speed advantage.

The [input-evidence T4 comparison](modal-t4-tiny-en-input-evidence-2026-09-05.json)
completes 32 seconds of digital silence with no emitted words. The three
individual speech transcripts match the punctuation-repair baseline exactly.
The repeated-voice mixture remains unresolved with its uncommitted audio
retained. This opt-in policy is not general voice activity detection. See the
[results and limits](../docs/research/2026-09-05-input-evidence.md).

The [punctuation-repair T4 replay](modal-t4-tiny-en-punctuation-repair-2026-09-05.json)
confirms that standalone punctuation no longer discards the second utterance.
The mixture remains unresolved with audio retained; the three separate clips
still complete. Silence still hallucinates in that baseline. The
[follow-up report](../docs/research/2026-09-05-input-evidence.md) describes the
separate opt-in input-evidence policy and its limits.

The [three-speaker T4 diagnostic](modal-t4-tiny-en-word-corpus-v1-2026-09-05.json)
completes three individual LibriSpeech clips but fails the constructed mixed
input and digital silence. A punctuation-only commit advances audio coverage
past untranscribed speech; silence produces a hallucinated word. Lifecycle
checks pass, which does not establish recognition accuracy. The
[report](../docs/research/2026-09-05-word-corpus-diagnostic.md) separates these
GPU findings from the later local punctuation guard. No live or efficiency
claim follows from this unpaced diagnostic.

The [T4 window-edge repair replay](modal-t4-tiny-en-word-alignment-v6-2026-09-05.json)
completes all 33 seconds in the word-aligned cell, with eight commits, unchanged
committed revisions, and restored capacity. Its 66 words match the model controls
after case and punctuation normalization; exact strings differ. The segment cell
remains unresolved at 8 seconds, so the overall record remains `unresolved`.
This unpaced repeated-fixture result does not qualify live latency or efficiency.
See the [completion report](../docs/research/2026-09-05-word-alignment-completion.md).

The [T4 anchor-repair replay](modal-t4-tiny-en-word-alignment-v5-2026-09-05.json)
removes the first word-profile blockage. Committed coverage advances from
5.70 to 12.78 seconds; the segment profile remains at 8.00 seconds. Both still
stop before completing the 33-second input. Input accounting and lifecycle
checks pass. The next mismatch is a window-leading word start estimate, not a
missing word. See the [replay report](../docs/research/2026-09-05-word-alignment-replay.md).
This record binds source commit `212b293`; it does not validate later repairs.

The [matched segment/word T4 diagnostic](modal-t4-tiny-en-word-alignment-v4-2026-09-05.json)
is **unresolved**. The segment profile publishes 8.00 seconds and the word profile
5.70 seconds of the same 33-second input. Both pass lifecycle and input-accounting
checks; neither emits a final event. The word path requires an anchor word whose
audio was partly removed at rebase. See the [diagnosis, measurements, and limits](../docs/research/2026-09-05-word-alignment-comparison.md).
The record remains bound to source commit `05e83e1`, before the anchor repair.

The [four-configuration boundary diagnostic](modal-t4-tiny-en-stream-boundaries-v3-2026-09-05.json)
and [receipt](modal-t4-tiny-en-stream-boundaries-v3-2026-09-05.attempt.jsonl)
compare holdback and retained context on the same 33-second synthetic input.
Baseline completes with two word edits against the full-stream model control;
two-second holdback completes with zero word edits. Both context configurations
stop at EOF after publishing 7.44 seconds. All four pass lifecycle checks, which
do not imply completion or recognition accuracy. The generic error description
is imprecise; the raw `eof_unresolved` trace identifies the actual failure.
See the [interpretation and next gate](../docs/research/2026-09-05-boundary-comparison-results.md).

The [failed v2 receipt](modal-stream-boundaries-v2-failed-2026-09-05.attempt.jsonl)
records the preceding result-transport failure, not an ASR result. The
[CPU transport receipt](modal-stream-boundaries-v3-transport-2026-09-05.attempt.jsonl)
records a synchronous byte-payload check before the v3 GPU call.

The [33-second rolling-stream diagnostic](modal-t4-tiny-en-continuous-smoke-v2-2026-09-05.json)
and [receipt](modal-t4-tiny-en-continuous-smoke-v2-2026-09-05.attempt.jsonl) extend
the first T4 case with three exact concatenations of the converted JFK PCM.
The stream made four pre-EOF commits and one EOF commit. All 528,000 samples
were accounted for; the peak input buffer held 168,960 samples (10.56 seconds).
The 40 ordered events and resource-release checks passed. The 34 decodes were
not paced in real time. This is a synthetic rolling-input test, not a diverse
corpus or a 30-minute live qualification.

**Recognition quality did not match the segmented reference.** The final text
contains "am I fellow Americans" where the reference contains "my fellow
Americans". The record retains this difference and both complete transcripts.
Its `passed` status applies to the registered lifecycle checks, not recognition
quality. The reference is one same-options 11-second decode repeated three
times, not a full 33-second decode. Legacy `full_window_control` check aliases
remain in the record; `segmented_control.scope` defines the actual comparison.
GPU allocation and reservation peaks are observations, not general memory
bounds. Independent review checked source hashes, receipts, all event revisions,
input accounting, the quality difference, and sanitization before this copy.

The [rolling-stream T4 diagnostic](modal-t4-tiny-en-continuous-smoke-v1-2026-09-05.json)
and its [attempt receipt](modal-t4-tiny-en-continuous-smoke-v1-2026-09-05.attempt.jsonl)
record the first native `timestamp_agreement_stream/v1` case on Modal. The
11-second JFK input produced one pre-EOF commit at 7,440 ms and a final commit at
11,000 ms. Committed text matches a same-options full-window control after
whitespace normalization. All recorded lifecycle checks passed and capacity
was released. The exact 18-file source snapshot is preserved in local commit
`2f47111`; the record does not claim a public preregistration or public source
commit at execution time. The 1.76694553-second loop was not paced in real time.
It does not establish live latency, rolling operation beyond 30 seconds, or a
performance improvement. Independent review checked the source snapshot,
receipt digest, event sequence, scope, and sanitization before this copy.

The [timed-publication CPU smoke](native-cpu-tiny-en-jfk-timed-publication-2026-09-05.json)
records one full-window timestamp-enabled control and two selected publications
from that same audio. The decoded spans are 0–8 seconds and 8–11 seconds. Both
publications retain the full 0–11 second analysis provenance. The second leaves
the first record unchanged. Repeated pre-publication inspection reuses the
prepared result. Tokens and segment text match the same-options control.
The record also includes passing default and bounded-preview regression runs.
All runs release runtime capacity. This is one local fixture, not evidence of
rolling-window continuity, recognition accuracy, or a performance improvement.

The [bounded-preview CPU smoke](native-cpu-tiny-en-jfk-bounded-previews-2026-09-05.json)
records three local JFK replays, same-PCM control results, revision events, and
resource release. It is a diagnostic smoke record, not a registered CUDA
qualification or a live-performance benchmark. See the
[test scope](../docs/BOUNDED_STREAMING.md#local-cpu-smoke).

In addition to the rolling-stream diagnostics and two local smoke records above,
this directory contains nine earlier records from real backend runs:

- `native-cpu-tiny-en-jfk-2026-09-03.json` records one
  `NativeWhisperAdapter` transaction.
- `native-cpu-tiny-en-jfk-interleaving-2026-09-03.json` records two staged
  decode runs on one loaded model, early cleanup of one run, and completion of
  the other.
- `native-cpu-tiny-en-jfk-threaded-2026-09-04.json` records the same isolation
  case across two operating-system threads in the patched decoder backend.
- `native-cpu-tiny-en-jfk-runtime-concurrency-2026-09-04.json` records two
  transactions through the experimental two-lane runtime adapter.
- `modal-t4-tiny-en-jfk-cuda-readiness-gcp-2026-09-04.json` and
  `modal-t4-tiny-en-jfk-cuda-readiness-aws-2026-09-04.json` record two separate
  direct-backend CUDA executions on Modal T4 workers. The workers ran in GCP
  `asia-southeast2` and AWS `us-west-2`.
- `modal-t4-tiny-en-jfk-native-cuda-transaction-aws-2026-09-04.json` and
  `modal-t4-tiny-en-jfk-native-cuda-transaction-gcp-2026-09-04.json` record two
  native-adapter transactions on Modal T4 workers. The workers ran in AWS
  `us-west-2` and GCP `europe-west2`.
- `modal-t4-tiny-en-jfk-native-cuda-qualification-v6-2026-09-04.json` records
  the registered single-worker qualification on an AWS `us-west-2` T4.

Each record identifies the runtime revision, backend source tree, model
checkpoint, input, environment, and observed outcome.

`modal-native-cuda-qualification.schema.json` defines the draft record for one
fixed native CUDA qualification cell. The cell uses one worker, two warm-up pairs, five
measured pairs, three cancellation runs, and two repetitions at each of four
fault points. Its p50 and p95 summaries are diagnostic; p99 is
`not_estimated`. The version-six record satisfies this contract.

The active registration is
[`experiments/native-cuda-qualification-v6.json`](../experiments/native-cuda-qualification-v6.json).
A record binds its path, digest, and runtime commit. The local validator cannot
prove that the registration was public before execution; that requires an
external public timestamp. The same version-six manifest and digest were public
in commit `bf46687d6c0f837426d85a1f97c60dd64128f9ed` before execution. A future
performance campaign requires a new schema or version, full performance
metrics, and matched control/runtime probes. See
[`docs/CUDA_QUALIFICATION_CONTRACT.md`](../docs/CUDA_QUALIFICATION_CONTRACT.md) and
[`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

The descriptions of failed attempts below include operator observations from
Modal logs that are not committed here. Their detailed execution paths and
causes are not independently verifiable from the public receipts alone. The
receipts record only their stated fields; none is a passing qualification.

During the version-one dispatch, the operator observed repeated Modal
deserialization errors and stopped the run before any inference. Its
append-only
[`attempt-started` receipt](modal-native-cuda-qualification-v1-attempt-2026-09-04.jsonl)
is retained without a synthetic terminal event. It records the local
pre-dispatch step, not successful dispatch, remote worker state, or the cause.
The version-two producer required the canonical module invocation and rejected
a file-path invocation before dispatch.

For version two, the operator observed that the worker reached the registered
T4 cell and stopped during its first control decode because the transcript
digest did not match the registration. Its
[`attempt-failed` receipt](modal-native-cuda-qualification-v2-attempt-2026-09-04.jsonl)
is retained. The receipt itself records the `gpu-campaign` stage, exception
type, and message digest; it does not independently prove the allocated GPU or
preserve the observed transcript. The earlier passing AWS record used
timestamp-free decoding. Version two registered timestamp-token decoding with
the timestamp-free transcript digest. Version three aligned
`without_timestamps` with that digest and was executed once.

For version three, the operator reported that the worker completed the registered
warm-up, measured, and cancellation runs. The first `cleanup` fault scenario
then completed retention and recovery before an incorrect event-order
expectation stopped the harness. The
[`attempt-failed` receipt](modal-native-cuda-qualification-v3-attempt-2026-09-04.jsonl)
records the `gpu-campaign` stage, exception type, and message digest. It does not
contain the transient event trace, and no qualification record was published.
The reported diagnosis was a harness assertion failure; the receipt alone
cannot exclude a backend or hardware failure. Version four records `fault-armed` only after the injection plan exists
and before the protected operation acquires its lease. It also records that
harness faults occur before the named delegate call.

For version four, the operator reported that the single attempt completed the
registered GPU campaign on a T4 and reached the post-campaign dependency
inventory. Inventory then failed because the environment exposed more than one
visible metadata version for `idna`. The
[`attempt-failed` receipt](modal-native-cuda-qualification-v4-attempt-2026-09-04.jsonl)
records the `gpu-campaign` stage, exception type, and message digest. The
receipt does not contain the transient GPU observations, and no qualification
record was published. The attempt supports no passing qualification claim.

Version five changed the inventory to one metadata record returned by
`importlib.metadata.distribution(name)` for each normalized name discovered by
`importlib.metadata.distributions()`. The operator reported that its single T4
attempt completed the registered GPU campaign and constructed a schema-valid
record. Internal semantic validation then rejected the record because the
harness required distribution metadata for Modal to equal the
platform-injected module version. A separate CPU diagnostic inspected the same
Modal image (`im-GLEsEGZRFsSkRtNGoxP69W`) in run
`ap-u7s5B07rhiT7g276cf2856`. It observed Modal 1.5.5 at
`/pkg/modal/__init__.py` with no Modal distribution metadata. It also observed
Torch 2.6.0+cu124 from matching module and distribution metadata, and selected
idna 3.19 while idna 3.10 remained visible under `/__modal/deps`. This
diagnostic was not a qualification attempt. The
[`attempt-failed` receipt](modal-native-cuda-qualification-v5-attempt-2026-09-04.jsonl)
records only the failure stage, exception type, and message digest. The detailed
execution path and cause remain operator observations. No qualification record was
published, so the attempt supports no passing qualification claim.

Version six treats the platform-injected Modal module version and the
distribution inventory as separate provenance observations. It does not
require a Modal distribution entry. It retains exact Torch module and
distribution equality because Torch executed the workload. It collects the
dependency inventory and runs a CPU contract rehearsal before creating the GPU
attempt receipt. Its single attempt produced a
[`record-published` receipt](modal-native-cuda-qualification-v6-attempt-2026-09-04.jsonl)
and a
[passing qualification record](modal-t4-tiny-en-jfk-native-cuda-qualification-v6-2026-09-04.json).
The record binds runtime commit
`9c2494234f08b24325d427ea422818b24f460c0c`, an AWS `us-west-2` Tesla T4,
the `tiny.en` checkpoint, and the pinned JFK input. It passed the schema and
semantic validator, including exact output compatibility, cancellation,
retention, recovery, resource-ledger, completion-fence, and publication
relations. The record SHA-256 is
`e3374c6f39f0739336706ce161e3836396f440b01c6254210d90e806678476bb`.
The receipt SHA-256 is
`641c3cd248de75fc7c96096fee8aee6ff1000adb3ba06cbe916db9c096255d02`.
This is qualification evidence for one fixed cell, not a performance or
production-readiness result. The inventory describes distribution metadata; it
does not prove the bytes of imported modules. Source and build-input hashes
remain the integrity anchors.

`modal-cuda-readiness.schema.json` defines the version-one T4 record. Both
committed records passed its schema and semantic validator. They bind the same
runtime commit, backend tree, patch manifest, model checkpoint, decoded PCM,
and transcript. A passing record covers the direct patched backend only. The
CUDA rejection boundary was exercised; no runtime transaction was admitted or
executed. No record covers worker admission, the transaction lifecycle, a CUDA
completion fence, or a performance benchmark. See
[`docs/MODAL_GPU_VALIDATION.md`](../docs/MODAL_GPU_VALIDATION.md).

`modal-native-cuda-transaction.schema.json` defines the separate version-two
adapter transaction record. Both committed records bind runtime commit
`28415364d167f71d5b0cdf441b0738ae4689b683` and tree
`9b0c3f5788635bb4a8044307d3f13dfec5690131`. Each record covers one
instrumented successful transaction, cooperative cancellation, injected fence
failure and recovery, post-recovery reuse, and one unproxied native control
transaction. They also bind source, model, input, environment, trace order,
terminal state, and resource state. See
[`docs/MODAL_NATIVE_CUDA_VALIDATION.md`](../docs/MODAL_NATIVE_CUDA_VALIDATION.md).

The version-two records passed the closed schema and semantic validator. Their
trace event sequences, state snapshots, transcript, source identities, model
state, and resource outcomes match across the two providers. The injected
synchronization failure occurs in the harness before the delegate call. It does
not represent a physical CUDA driver failure. The records are not performance
or production readiness claims.

The native CI workflow repeats the same-model interleaving check and publishes
its record as a 30-day artifact. The check covers state separation, early
cleanup, rejection of cancelled-run reuse, and a survivor that matches an
isolated baseline within the recorded scalar tolerance, which is zero in CI.
Its format is defined by `native-interleaving.schema.json`.

The workflow also runs the decoder isolation case in two operating-system
threads after preparing both encoder outputs sequentially. Each thread enters
its first outer decoder call. A barrier in the first decoder block holds both
calls before either continues. The record identifies each
owner thread, captures the start and end of each outer call, and requires the
two intervals to overlap. It also records the explicit decode options. The
committed record and each 30-day CI artifact are validated against
`native-threaded.schema.json` and by cross-field checks in
`tools/validate_threaded_record.py`.

The workflow then sends two real-model transactions through the experimental
two-lane `NativeWhisperAdapter` profile. It verifies admission and declared
budget state while both calls are live, cancels one request through the runtime
transaction, and requires the other request to commit the isolated-baseline
text. The cancelled session stays empty. The queue and declared budget must be
fully restored, and a later adapter call must succeed. The CI artifact is
validated against `native-runtime-concurrency.schema.json` and by
`tools/validate_runtime_concurrency_record.py`. The committed record captures
the same contract on one Windows CPU configuration.

Each record applies only to its stated configuration. The records are not
performance benchmarks. The committed two-thread check exercises the patched
Whisper backend below the runtime adapter. The adapter-level CI check uses
caller threads; it does not exercise a runtime-owned scheduler. Encoder
preparation remains serialized. The declared resource vectors are
admission-ledger values, not measured RAM or device memory. No check establishes
kernel overlap, throughput, CUDA behavior, production readiness, or behavior
on other models, devices, operating systems, or dependency versions. Each
two-thread check covers one controlled case; it is not a general thread-safety
guarantee.
