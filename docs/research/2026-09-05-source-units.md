# Caller-delimited audio units: T4 diagnostic

## Result

The registered 43.660-second speech/pause mixture completes with caller-supplied
boundaries. One continuous controller admits all 698,560 samples and publishes
11 contiguous units: six speech occurrences and five digital-zero pauses.
The final PCM buffer is empty. Each repeated utterance remains a separate
occurrence, and each pause emits no text.

The complete transcript has zero normalized word edits against both the full
mixed-input Whisper control and the concatenated individual-fixture controls.
Exact strings differ in punctuation. Against the independent human reference,
there are six normalized word edits over 88 reference words: the same 0, 1,
and 2 edits per fixture, repeated twice. This is not error-free recognition.

The [record](../../evidence/modal-t4-tiny-en-source-units-2026-09-05.json)
reports all 21 lifecycle, ownership, evidence and completion checks passing.
The model fingerprint is unchanged and worker capacity is restored.

## What changed

Source commit: `44ba920dd206d48958e1ed5e319aed6ca6689be6`.

The optional `source_units=True` profile extends the existing continuous
controller. The caller closes an admitted range with `seal_unit(end_sample)`.
Open-range results remain provisional. A closed range uses the existing full
native result and transaction, subject to input-evidence checks. Exact zeros
use the existing typed empty publication. Resource release still precedes
output events and buffer advancement.

This profile does not use word alignment or retained left context. It avoids
requiring lexical anchors across unrelated utterances. It preserves source
positions, retry identity, native results and global session event order.
The legacy and word-agreement profiles remain unchanged.

Closing input and closing a unit are distinct operations. The caller can still
partition admitted audio after EOF, provided its analysis is not already frozen.
An in-flight preview cannot become a final result merely because its endpoint
has since been sealed. It completes or retries under its original identity;
the closed-unit analysis has a separate identity.

This is an explicit full-result recognition contract. A supplied endpoint does
not prove that every word was recognized or that any nonzero interval is silent.
Uncertain closed-unit results stop without discarding their PCM.

## Diagnostic design

The source is the same frozen LibriSpeech mixture used by the previous
[input-evidence comparison](2026-09-05-input-evidence.md). The model, checkpoint,
FP32 precision, seed 7, and T4 AWS region remain fixed.

The variant processes only the mixed case. It runs one full-mixture offline
control and one offline control per unique speech fixture. Those three fixture
controls are reused for both occurrences. No new corpus or checkpoint download
was required.

The eleven sample endpoints are supplied from the fixture construction:

```text
56080, 88080, 174240, 206240, 333280, 365280,
421360, 453360, 539520, 571520, 698560
```

These are oracle boundaries, not detected endpoints. The driver feeds up to
one second per chunk, splits a chunk at each boundary, and seals before processing
the endpoint. It uses one controller and one monotonically increasing input
sequence. Independent admission accounting totals the actual chunk lengths.

## Measurements and limits

| Measurement | Result |
| --- | --- |
| Accepted / committed input | 698,560 / 698,560 samples |
| Final buffered input | 0 samples |
| Peak buffered input | 127,040 samples, or 7.940 seconds |
| Input chunks | 46 |
| Native decodes | 23 |
| Word alignment in the measured stream | None |
| Summed source duration submitted to decoding | 83.660 seconds |
| Output events / commits / final events | 35 / 11 / 1 |
| Commits before global EOF | 10, including the five empty pauses |
| Measured stream loop | 2.018828540 seconds |

The loop measurement excludes loading, warmup and offline controls. The run is
unpaced. It does not measure caption latency or qualify a sustained live input.
The longest final unit covers 7.940 seconds; provisional text can appear within
that unit, but final text waits for the caller to close it.

The prior mixed-input policy stopped after publishing 3.480 seconds, with audio
retained. This variant completes with external boundary knowledge and different
publication settings. It is not a matched performance comparison. In particular,
23 decodes here exceed the prior mixed run's 16 decodes; summed source duration
does not measure fixed-size encoder work. No general GPU saving follows.

General acoustic endpoint detection, quiet speech, environmental non-speech,
cross-boundary words, continuous speech without pauses, and long-session latency
remain open. Small units can lose linguistic context or increase fixed overhead.
The test establishes that the existing transaction controller can join these
known source units without introducing word omissions or duplicate occurrences.
It does not establish that an automatic detector will supply suitable boundaries.

## Validation and evidence identity

- 458 runtime tests pass, including 13 source-unit tests.
- 287 repository-tool tests pass, including 22 corpus harness tests.
- Strict typing, Ruff lint/format and repository checks pass.
- No model, native backend or formal proof changes were made.
- This iteration did not rebuild wheel or source packages.

Local tests cover fractional sample boundaries, different chunk partitions,
repeated text, digital silence, unknown input, active previews, cancellation,
five preparation/execution failure points, post-EOF partitioning, and failure
before or after transaction commit. They use the existing transaction/fence
fixtures; native execution is separately recorded by this T4 diagnostic.

Record SHA-256:
`80cf5a0168ce256da44f06ace0deb1ec3597c6e462a7fd6efcee8cf5a0ab5f89`.

Source snapshot digest:
`43cffed2512450b22b01b7673b58834c607d7694a747ce150720852cf8f02422`.

A separate in-team read-only audit matched all 22 source hashes to the stated
commit and reconstructed all 23 PCM observations. Unit ranges and commits match
the registered source exactly. All five empty publications retain native `you`
results, no-speech scores near 0.948 and incomplete predicted timestamps. All
18 speech observations are candidates; none is classified as uncertain. This
audit checks the recorded diagnostic, not external security certification.

The [GPU receipt](../../evidence/modal-t4-tiny-en-source-units-2026-09-05.attempt.jsonl)
and [CPU transport receipt](../../evidence/modal-source-units-transport-2026-09-05.attempt.jsonl)
are separate from all preceding attempts. No previous evidence was overwritten.

## Reproduction and resources

Use Modal SDK 1.5.5, the registered cached model and local PCM assets. Set
`WHISPER_MODAL_ENABLE_WORD_CORPUS=1`. On Windows, also set `PYTHONUTF8=1` and
`PYTHONIOENCODING=utf-8`. Use a fresh replay name: existing receipts block reuse.

```sh
python -m modal run infra/modal_word_corpus.py --transport-preflight-only --replay-id source-units-20260905
python -m modal run infra/modal_word_corpus.py --confirm-paid-gpu --source-units --replay-id source-units-20260905
```

One CPU transport probe and one GPU function ran. The GPU function has a
180-second execution timeout, zero automatic retries and at most one container.
These execution limits are not a billing cap. Remaining account credit was not
measured. Both apps were verified stopped with zero tasks:

- CPU: `ap-vqoXM2KNrlQrhwWIHVUfoL`.
- GPU: `ap-BsmE2gL6057SxHgQeYNfGY`.
- GPU function call: `fc-01M1RPWR65P8FR93VYBA96FXAS`.

## Next gate

Replace oracle boundaries with conservative, independently tested acoustic
endpoint proposals. Keep this oracle test as the upper-control condition for
segmentation. Compare complete recognition and source ownership before paced
latency or compute claims. A proposed boundary must not authorize silent loss
of uncertain input.
