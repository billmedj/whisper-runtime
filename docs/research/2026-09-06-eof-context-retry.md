# One retained-context retry at end of input

## Result

The real controller now completes the development input that previously stopped
at 21.26 of 33.66 seconds. One additional native window passes the existing
word-anchor and input-evidence checks. Published output before the retry stays
unchanged. The controller commits the remaining input, empties its buffer and
emits one final event.

This is an opt-in recovery path. It changes neither Whisper weights nor the
acceptance thresholds. It does not resolve every acoustic failure.

## Method

An earlier retained-context observation contained a usable continuation. The
diagnostic refused it because it also compared it with a head-only proposal.
That comparison was not needed to publish the retained-context result through
the original strict policy. This experiment connects that policy to a real
controller retry; it does not promote the head-only proposal.

After an aligned EOF refusal, `eof_context_retry=True` permits one new analysis.
Its start is 500 ms before the first published anchor word, rounded down to the
20 ms grid and bounded by retained audio. Its end stays at the same source EOF.
The window must differ from the refused window, contain at least two lexical
anchor words, start before the committed boundary and fit the native limit.

The controller freezes the refusal, anchor, session version, requested decode
configuration and PCM hash. It rechecks them before admission and resolution.
The original operation must release capacity before the retry starts. The retry
uses the normal word selector, input-evidence checks, native transaction and
publication path. No reference text, case folding or wider time tolerance
selects or accepts the result.

Startup failure, cancellation and cleanup failure consume the attempt. Failed
input stays retained. A committed result whose cleanup fails waits for resource
recovery before publication. `context_retry_observation` records the source
refusal and outcome; it is not a publication permit.

## T4 comparison

One Tesla T4 ran the existing `tiny.en` checkpoint in FP32 with seed 7 and legacy
word alignment. The actual controller received incremental one-second PCM
chunks, not wall-clock-paced input. Source hashes, audio recipes, execution
limits and the three-cell order were fixed before dispatch.

| Input and policy | Committed audio | Native windows | Encoder forwards | Word edits / reference words |
| --- | ---: | ---: | ---: | ---: |
| No added pauses, unchanged baseline | 21.26 / 33.66 s | 17 | 34 | 36 / 88 |
| Same input, one context retry | 33.66 / 33.66 s | 18 | 36 | 6 / 88 |
| Attenuated-prefix control, retry enabled | 10.89 / 10.89 s | 6 | 11 | 1 / 26 |

The baseline reproduces the archived failure, including native observations.
The retry arm matches all baseline observations up to that failure. Its one
new window is `[20.22, 33.66]` seconds, inside retained input. It recovers the
unpublished 12.40 seconds through one additional released transaction. All
earlier commits remain identical. The attenuated control completes without
scheduling a retry. All event and PCM checks pass for both completed cells.

All 41 native windows and 81 encoder forwards are recorded. Legacy alignment
accounts for a second encoder forward on aligned windows; this test does not
enable the separate feature-reuse optimization. All cells restore declared
capacity and the model fingerprint stays unchanged.

The worker reports 25.018 seconds elapsed. The Modal app stopped with zero
tasks. The function had a 120-second timeout, no automatic retries, at most
60 admissions and a 20-second cleanup reserve. No model was downloaded.
These bounds and timings are not a billing receipt. The first arm was cold;
the arm timings do not establish a speedup.

## Evidence and local checks

The [full record](../../evidence/modal-t4-tiny-en-eof-context-retry-2026-09-06.json)
contains events, native results, word alignments, input hashes, operation
counts, refusal and recovery receipts, and recomputed checks. It is an unchanged
copy of the local output JSON. SHA-256:

`4ab9914df3932a12909e21993f579c7b49d7e818fa7e94cc98ade7f4eae5a72e`

The executed 37-file working-source snapshot has digest:

`b63ec4fe32e783400c683fea24183235f6133149d9132e50fc6566d80aa3ac25`

Seventeen targeted controller tests cover exact PCM selection, immutable
prefixes, unchanged lexical and time checks, one-attempt bounds, startup
failure, cancellation, stale input and retained native capacity. Checkpoint
tests cover rejection of pending retries and save/restore after an actual
mock-native retry commit. Old v1 checkpoints lacking the new flag load it as
`False`; all other field and checksum checks remain strict.

## Limits and next gate

The development input joins known speech clips and was used to select the
fixed context rule. It is not a held-out quality test. Successful sample
coverage does not remove the remaining six word edits. Strict anchors and
a completed transaction do not prove absence of acoustic omissions.

This policy adds work only after an EOF refusal. It does not yet repair a
nonfinal source-unit failure, qualify a long live session or reduce GPU work.
Noisy and ambiguous cases that failed prior diagnostics remain unqualified.
The next comparison should use the same controller at source speed, include
those refusals and held-out speech, and report final-caption delay alongside
recognition quality. Do not add more retry windows before that comparison.
