# Noisy continuation: crop and decoder history

Date: 2026-09-06. Status: completed diagnostic, not a live recovery.

On the same retained audio, supplying the model's published prefix changes an
incomplete decode into the missing next utterance. The new result omits the old
anchor and has unsuitable leading word times. Recognition changes; publication
remains refused. No runtime acceptance rule or default changed.

## Fixed comparison

The [paced receipt](../../evidence/modal-t4-tiny-en-paced-context-retry-2026-09-06.json)
fixes the noisy input, head, retained words, configuration and decoder history.
The source contains 174,240 samples at 16 kHz, or 10.89 seconds. The published
head is 3.68 seconds. Retained context starts at 1.68 seconds.

The prompt is the actual published text:
`Concord returned to its place amidst the tents`. It has no trailing
period. The human transcript is never a prompt or a selection input.

Two crops each receive two observations: null prompt, then published-prefix
prompt. Each replacement uses the existing `NativeWindowRun.redecode` API and
its original encoder features. It keeps the audio, seed and lease, and creates
fresh decoder state. Alignment stays on the legacy path for both observations.
All sessions close without `finish()`, at version zero.

Settings match the archived control: cached `tiny.en`, FP32, T4, seed 7,
English transcription, timestamp tokens, greedy decoding, and `sample_len=None`.
The frozen 200 ms timestamp tolerance is unchanged.

| Audio crop | Prompt | Observed text | Publication outcome |
| --- | --- | --- | --- |
| 1.68–10.89 s | None | `to its place amidst the tents.` | Proposed new suffix is only punctuation; refused |
| 1.68–10.89 s | Published prefix | Next utterance, through `bird and tree` | Old anchor absent; refused |
| 3.68–10.89 s | None | Same next utterance | Diagnostic crop has no join authority |
| 3.68–10.89 s | Published prefix | Same next utterance | Diagnostic crop has no join authority |

The retained/null result reproduces the archived result and alignment exactly,
excluding its window ID. Head-only results have equal text and word times;
their model scores differ. Hypothetically joining either head-only transcript
to the published prefix gives 1 normalized word edit in 26 reference words
(`son` versus `sun`). That join is not published. The retained/prompted result
has no selector-approved suffix, so no joined transcript score is assigned.

The prompt effect is therefore observable on fixed audio, not just a crop
effect. This is one tuned example, not evidence of general quality improvement.

## The remaining boundary problem

The retained/prompted alignment places `For` at 1.68–4.00 seconds, before the
3.68-second published head. `a` has zero duration. Both head-only observations
place `For` at 3.68–6.10 seconds. Equal text does not establish equal timing or
safe continuity with previous words.

There is useful evidence already in the paced archive. In the noisy/retry cell,
`decision_traces[3]` analyzed 0–8 seconds without a prompt. It observed the old
anchor and the next utterance together, but committed only through 3.68 seconds.
The continuation in that observation remains unpublished.

| Boundary item | Earlier joint observation | Head-only/null observation |
| --- | --- | --- |
| Period after `tents` | 3.68–4.10 s | Absent |
| `For` | 4.10–6.10 s | 3.68–6.10 s |
| `a` | 6.10–6.20 s | 6.10–6.20 s |
| `while` | 6.20–6.44 s | 6.20–6.44 s |
| Comma | 6.44–6.66 s | Absent |
| `she` | 6.66–6.74 s | 6.44–6.70 s |

The joint observation's PCM, overlapping PCM, model, backend, options and common
runtime source files match. But the first sequence mismatch is a period; the
first lexical onset mismatch is 420 ms, beyond the 200 ms gate.

The current nonfinal selector rejects the pair because their window origins
differ. Its EOF selection on the head-only result bypasses retained-anchor
matching because the crop starts at the head. That selection is not a bridge
certificate. Do not drop punctuation, clip times or promote unpublished words
into the committed anchor to make the pair pass.

## Next implementation gate

Evaluate an explicit continuity witness using the saved joint observation,
separately from decoder history and the published anchor. Start with local
counterexamples: missing words, changed punctuation, repeated phrases, moved
word boundaries and changed input. Preserve the refusals above. Decide which
publication contract can use that witness before connecting it to the live
controller. Additional model work must address a specific missing observation;
the present results do not justify another prompt sweep.

This separation is the useful architectural finding: decoder history can help
recognition without serving as evidence that the output joins safely. Reusing
the encoder makes this bounded second observation cheaper in model work, but
does not resolve its publication boundary.

## Resources and reproducibility

One T4 call completed four observations in 14.931 seconds of worker wall time.
Two decoder encodes and four alignment encodes were measured: six encoder
forwards, versus eight required for four separate decode-and-align runs on
this path. This is operation accounting, not a measured speedup or billing
comparison. The first alignment includes cold setup cost.

All capacity was released, the model fingerprint stayed unchanged, and app
`ap-JwMyOiL2afBNRGy8pUF8hQ` stopped with zero tasks. There was no GPU retry.
The configured worker timeout was 90 seconds, with a 15-second cleanup reserve.
No model was downloaded.

The [original decoded receipt](../../evidence/modal-t4-tiny-en-noisy-context-prompt-2026-09-06.json)
is unchanged. File SHA-256:

`ac48e91c068d5d57ee746a04cb3cecbdf9cea2ecc493775048aba430992e9832`

Executed 43-file snapshot digest:

`0badcd5eda5fb18ea25863139e834060d7f951d0322e841b1a3748a8e3958635`

The GPU returned all observations successfully, but the local exporter rejected
its control flag. The worker had compared Python tuples with JSON lists and
reported a false mismatch. Canonical comparison reproduces the old control.
The producer now normalizes both sides, with a regression test.

The [separate offline validation](../../evidence/noisy-context-prompt-offline-validation-2026-09-06.json)
records that one report-only correction. A verifier permits it only for this
exact receipt's canonical hash, checks an in-memory copy, and preserves every
native observation and the original flag. It does not rewrite the dispatched
source snapshot or claim that the original exporter succeeded.

Replay locally with `PYTHONPATH=src`:

```sh
python -B tools/verify_noisy_context_prompt.py evidence/modal-t4-tiny-en-noisy-context-prompt-2026-09-06.json
```
