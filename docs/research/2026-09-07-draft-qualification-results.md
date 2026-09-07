# Native drafts: new audio, restart and threshold checks

The new-audio GPU screen and the fresh-process CPU restart pass their registered
checks. Threshold tests identify a remaining boundary: exact token agreement
does not guarantee identical publication decisions when floating-point scores
cross a decision threshold. Drafts remain opt-in; runtime defaults are unchanged.

## New voices on T4

The [registered screen](2026-09-07-draft-holdout-plan.md) uses two LibriSpeech
speakers absent from prior project evidence. The unchanged clips total 11.485
seconds, with two seconds of digital silence after each: 15.485 seconds per
stream. Selection and independent references were frozen before recognition.
This is project-new material, not claimed absent from Whisper training.
The [manifest](../../experiments/draft-holdout-20260907.json) records selection,
source hashes, CC BY 4.0 attribution and the earlier duration-only inspection.

One cached `tiny.en`, English FP32 greedy, seed 7, runs four source-paced
streams in control/candidate/candidate/control order. Both arms retain the same
20/24-second context and alignment encoder reuse. Only `max_draft_tokens`
changes, from 0 to 32. Both paths are warmed before measurement.

| Per-stream measurement | Control A1 | Draft B1 | Draft B2 | Control A2 |
|---|---:|---:|---:|---:|
| Native windows / encoder forwards | 9 | 9 | 9 | 9 |
| Decoder forwards, including alignment | 133 | 72 | 72 | 133 |
| Decode-phase wall, milliseconds | 891.4 | 697.1 | 669.6 | 894.8 |
| Alignment wall, milliseconds | 489.8 | 19.3 | 18.8 | 17.5 |
| Events / commits | 13 / 3 | 13 / 3 | 13 / 3 | 13 / 3 |

Decoder forwards fall by **45.9%**. Pooled decode-phase native wall time falls
by **23.5%**; both adjacent comparisons improve. All forward CUDA intervals
combined fall by **28.2%**. Each candidate verifies 61 proposed tokens, removing
61 sequential calls. Submitted decoder token positions increase from 175 to
233; lower FLOPs or energy use is not established.

Tokens, events, selected decisions and commit source endpoints match exactly.
All samples have accounted-for coverage, FINAL occurs once, and native state,
capacity and hooks are released. Allocated memory returns to 160,720,896 bytes;
all four cells peak at 190,260,736 bytes. These are allocator measurements, not
total process memory.

The raw word-edit count is 1/31 for both arms. The difference is the equivalent
date spelling `fifteenth` versus `15th`, which the existing metric does not
normalize. No reference or metric was changed after observing this result.
Maximum score differences are `1.43e-6` for average log probability and
`6.56e-7` for no-speech probability; scores are not bit-identical.

The first control again has longer host-side alignment operations. Do not credit
that difference, or the larger total-wall reduction, to drafts. Decode-phase
wall includes startup/encoder work and token stepping; it is not end-to-end
latency. Two short clean clips and one ABBA order do not establish population
accuracy, general speed, noisy-audio robustness or sustained-live performance.

The [independent offline audit](../../evidence/draft-holdout-audit-2026-09-07.json)
verifies all 67 source hashes, PCM, raw transport, output parity, clocks and
cleanup. Its adjacent decode-wall reductions are 21.8% and 25.2%. No no-speech
gate crossing occurs across the 36 measured native results; the minimum
distance to 0.6 is 0.24936, so these inputs do not test the numerical boundary.

## Restart after process exit

Three distinct CPU processes run an uninterrupted control, a checkpoint
producer, and a restore consumer. The producer exits before the consumer starts.
The 16.83-second known fixture completes with the same 15 events, token sequences,
decisions, input admission and source coverage as the uninterrupted draft32 arm.
All 20 required checks pass.

The savepoint preserves logical state and retained PCM, not decoder tensors or
the old 32-token hint. The first restored analysis uses ordinary inference;
decoder forwards total 256 rather than 246, with nine encoder forwards in both
paths. Scores differ slightly in that first analysis. This is a graceful
publication-boundary CPU restart, not abrupt-crash recovery, GPU migration,
bitwise score identity or durable delivery.

The [CPU evidence](../../evidence/native-draft-fresh-process-cpu-audit-2026-09-07.json)
and [public archive](../../evidence/native-draft-fresh-process-cpu-2026-09-07.zip)
preserve the results, savepoint and executed source bytes. The public archive
is an explicitly labelled privacy derivative: 18 report path strings and one
setup metadata path were replaced, and public packaging hashes were regenerated.
The numerical records, events, traces, PCM, savepoint and 65 executed-source
files are unchanged. Original execution-manifest hashes remain labelled as
original; they are not hashes of the sanitized setup metadata.

The byte-identical original is retained privately, outside the publication set:
SHA-256 `34334f9ec91904f5ad230c6937cb78485e2d65d0f86c6c72a3a4158d17a6600a`.
The public derivative is 1,162,498 bytes, SHA-256
`b92826a7adfed5c7af227a808c280ec69f9857ff3f44bccfc435acca639d1c3a`.
`PUBLIC_DERIVATION.json` documents changed fields and original hashes;
`PUBLIC_MANIFEST.json` verifies every public payload member. Verify without
native inference using:

```text
python -B tools/verify_public_cpu_archive.py evidence/native-draft-fresh-process-cpu-2026-09-07.zip
```

## Threshold result and next boundary

The [threshold audit](2026-09-07-draft-threshold-audit.md) replays all 196 saved
native results from the preceding mixed-audio T4 screen. There are no crossings,
but those inputs are far from the active `no_speech_prob >= 0.6` boundary.
Deterministic tests with identical reconstructed tokens demonstrate that small
score changes across this threshold can change real EOF/publication decisions.
Refused streams retain all PCM and restore capacity; refusal is not data loss.

Keep draft32 opt-in. The next targeted experiment is a canonical ordinary-path
SOT score for the publication gate, with its ownership and compute cost measured.
Do not round scores, move the threshold or treat observed score deltas as a
proven numerical error bound. A new-audio pass does not resolve this boundary.

## Verification and cost record

The combined local selection passes **117 tests**, including the new input,
threshold and restart checks. Ruff passes. The previous frozen runtime source
is unchanged. No new production dependency was added.

One GPU worker ran for **74.470 seconds**, using 40 native windows including
warmup. Modal app `ap-eI6KIFhXdAMLJKSCwhIkAv` is stopped with zero tasks. No GPU
retry occurred. The earlier rejected launch created no process or job; execution
followed explicit approval of the 67-file payload. No invoice was retrieved.

The [GPU archive](../../evidence/modal-draft-holdout-2026-09-07.zip) contains the
four raw records, all 67 executed source files, two PCM fixtures and license
notices. All 76 entries match their originals. SHA-256:
`a5f1282404e8009192fd13c304bd1201dc953ee79d94117c5e0c0c25ecdd0f03`.
Frozen source digest:
`69326bfcb630460d6c33927eb0a759a5dddf93cb936a54cdb968bf6ed2a3eac3`.
