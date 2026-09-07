# Same-window prompt re-decode

## Question

Can one completed window provide encoder features to a second decoder with a
different prompt, with the same results as independent decoding?

The runtime's opt-in `reuse_decode_features` profile allows one replacement
attempt inside the original transaction. It changes only the prompt. Decoder
state and the random generator are new. The seed, model, audio span, resource
lease and deadline stay fixed. Only the active candidate can be published.

## CPU results

All three tests passed on the existing `tiny.en` checkpoint and the 11-second
JFK fixture from the pinned Whisper source. Each test compares independent
A and B decodes with one A-to-B re-decode. The inputs were already on disk;
the diagnostic did not download weights.

| Profile | Timestamp tokens | Independent encoder forwards | Shared encoder forwards | Results and scores |
| --- | --- | ---: | ---: | --- |
| Greedy | Disabled | 2 | 1 | Exact match |
| Beam, size 2 | Enabled | 2 | 1 | Exact match |
| Sampling, temperature 0.4, best-of 2 | Enabled | 2 | 1 | Exact match |

For both candidates, comparisons include text, tokens, timing metadata,
average log probability and no-speech probability. Model and audio hashes
match after each test. The original result remains immutable. All transactions
release capacity after completion.

Records:

- [Greedy CPU record](../../evidence/native-prompt-reuse-cpu-tiny-en-jfk-2026-09-06.json)
- [Beam CPU record](../../evidence/native-prompt-reuse-cpu-tiny-en-jfk-beam-2026-09-06.json)
- [Sampling CPU record](../../evidence/native-prompt-reuse-cpu-tiny-en-jfk-sample-2026-09-06.json)

Local paths in these records were replaced by portable labels. The records
retain input hashes, checkpoint fingerprints and hashes of the executed code.
The first model-loading attempt failed when the host disk was nearly full.
These successful runs followed removal of old Cargo build caches.

## T4 test plan

This plan is recorded before dispatch. Use one T4, one container, no automatic
retries and a 120-second function timeout. Use the existing read-only model
cache and block the worker's network access. Upload and verify hashes of the
current runtime source, including the uncommitted change under test.

Run matched controls and reuse for three profiles: greedy with timestamps,
beam size 2, and sampling with temperature 0.4 and best-of 2. Include removal
of the prompt in the greedy profile. Count decoder attempts separately from
encoder forwards. Allow at most 14 decoder attempts, including a cancellation
test after the replacement decoder has been registered.

Accept only if:

- Each reused result exactly matches its independent control, including scores.
- Every matched pair uses two encoder forwards independently and one with reuse.
- The first result remains immutable and is not silently published.
- Decoder work remains on the transaction's owned CUDA lane.
- Completion and cancellation restore capacity and leave no active borrower.
- Model identity and the dispatched source hashes remain unchanged.

## T4 result

The single registered call completed on a Tesla T4 with PyTorch 2.6.0 and
FP32 decoding. All three profiles passed exact parity, including removal of
the prompt in the greedy profile. All timestamp fields and scores matched.

- Independent controls: six encoder forwards for the three pairs.
- Reuse: three encoder forwards for the same pairs.
- One CUDA lane was used across all observations.
- The cancellation check started its replacement, then cancelled before the
  next token step. It published nothing, retained the first immutable result
  and restored all declared capacity. The CUDA lane had no remaining borrower.
- All 14 planned decoder attempts were accounted for. Model hashes matched
  before and after. The uploaded 33-file source snapshot matched the dispatched
  working tree.

The [T4 record](../../evidence/modal-t4-tiny-en-prompt-reuse-2026-09-06.json)
preserves the results, source hashes, execution bounds and recomputed summary.
Its SHA-256 is
`1dcdd5a4620cb10606c87082e9344bbe04c0067caaf4a495d7d0dc56f2afc195`.
The published JSON uses normalized whitespace; its data matches the local raw
record, whose SHA-256 is
`0ae5537e111e3b35735741e93b233c47f5f3b09544db714c34f524f65f523dee`.

The Modal app stopped with zero tasks. No retry or model download occurred.
The function had a 120-second timeout; this is a resource limit, not a measured
monetary spending cap. No cost-per-minute or energy-efficiency claim follows
from the encoder count.

## Boundaries and next integration

These comparisons test reuse, not recognition accuracy. Single-pass host
timings include validation overhead and are not a speed benchmark. This
experiment does not qualify other checkpoints, longer streams, arbitrary
decoding options, durable checkpoints or GPU offload.

The existing live recovery and word-resolution worker compare different audio
windows. They cannot substitute this API for their second encoder pass. A
future same-window context diagnostic must keep each candidate's provenance
and preserve the existing acoustic publication checks. Prompted agreement is
not independent corroboration: the observations share encoder features, and
the prompt can influence the words being tested.

The live policy and all default execution profiles remain unchanged.
