# Earlier publication with same-window feature reuse on T4

The registered three-arm test completes on one T4 with one loaded `tiny.en`
model. The combined profile keeps the earlier commits, removes redundant
alignment encoder calls and lowers peak allocated memory. Its measured forward
interval total remains higher than the conservative control.

## Matched result

Each arm receives the same 46.55-second speech/noise/silence sequence in 20 ms
chunks on its source clock. FP32, seed 7, weights, backend tree, stream ownership
and publication checks are shared. Warmup is separate. Both controls explicitly
request legacy alignment on the same reuse-capable adapter.

| Measure | Conservative | Earlier commits | Earlier commits + reuse |
| --- | --- | --- | --- |
| Left context / word context limit | 2 / 6 s | 20 / 24 s | 20 / 24 s |
| First COMMIT, from source-clock start | 28.439 s | 10.266 s | 10.245 s |
| Native analysis windows | 25 | 25 | 25 |
| Encoder forwards | 32 | 43 | 25 |
| Decoder forwards | 998 | 1,214 | 1,214 |
| Encoder CUDA-event interval sum | 575.534 ms | 635.101 ms | 376.730 ms |
| Decoder CUDA-event interval sum | 3,658.049 ms | 4,359.044 ms | 4,319.332 ms |
| Combined forward interval sum | 4,233.582 ms | 4,994.145 ms | 4,696.063 ms |
| Peak PyTorch allocated memory | 276.66 MiB | 276.66 MiB | 213.67 MiB |
| Peak PyTorch reserved memory | 342 MiB | 342 MiB | 342 MiB |
| Word edits / reference words | 7 / 114 | 7 / 114 | 7 / 114 |
| Commit events, including silence | 4 | 14 | 14 |

The fast arms have exactly the same 14 committed texts and source spans. The
conservative arm has different segmentation and joined punctuation spacing;
its equal word-edit count does not mean an identical output string. Every arm
accounts for all 744,800 samples, emits one FINAL, empties its buffer and restores
capacity. All publication and pacing checks pass. Model weights are unchanged.

## Gain and remaining cost

Against fast legacy alignment, reuse removes all 18 alignment encoder forwards
without adding decoder calls. Its combined forward interval sum is 5.97% lower
in this run. Peak allocated memory falls 22.77%; reserved memory is unchanged.

Against the conservative control, first COMMIT arrives 63.98% earlier, but the
combined forward interval sum is **10.92% higher**. Earlier publication is not
free. Conservative decoding uses 991 decoder forwards plus 7 for alignment;
both fast arms use 1,196 plus 18. The remaining 216 extra decoder forwards,
despite the same 25 analysis windows, identify the next work to inspect. This
test does not establish which of those forwards can safely be removed.

## Limits

This is one short run per arm, on fixed input and in fixed order, not a repeated
benchmark or held-out accuracy study. CUDA-event intervals bracket actual
forwards on the owned stream; they can include launch gaps and are not
kernel-only occupancy, energy use or total service cost. Identical frame counts
need not have identical measured intervals. Warmup is excluded from the table.

First COMMIT is not first provisional text or per-word latency. The source clock
excludes deployment, model startup and PC-to-server networking. Allocated memory
is not total device memory; unchanged reservations prevent a claim that the
saved amount was returned to other processes. Recognition errors remain. No
general hallucination-prevention or cross-window feature-reuse claim is made.

## Evidence and repeatable audit

The [archive](../../evidence/modal-composed-features-2026-09-07.zip) contains the
exact result, compressed transport payload, frozen preflight and attempt journal.
Its size is 658,933 bytes; SHA-256:
`c63f609444abcf7ed1472b2860675a4caef4743f75de6090ab1a9e543aa5deee`.
Result SHA-256:
`d712cc9232491b5a77feb0e7411f88783ee658d6401dd61e57e474e0b6e6946d`.
The [offline audit](../../evidence/composed-gpu-audit-2026-09-07.json) recomputes
cross-arm comparisons from raw events and forwards. After extracting the archive,
run from the repository root with the package installed or `PYTHONPATH=src`:

```console
python -m tools.verify_composed_features PATH/acoustic-diagnostic.json PATH/composed-preflight.json
```

The audit needs the cached licensed corpus for input/publication checks, but
runs no inference and contacts no service. It prints a report and exits nonzero
when cross-arm expectations fail. Thirteen focused producer/audit tests and
Ruff pass. The preceding full regressions ran 807 core and 727 tool tests; the
six new audit tests are additional, not part of those earlier counts.

The [registration](2026-09-06-composed-inference.md) stays unchanged. Its 54-file
source digest is `3ce3b09e733b85989f039648b76371769124d31bac7d03cfd0c0937785d83799`.
Backend tree: `32163d5cdb87babc1cd415a86cc5a58116c86a16`.

## Execution and next gate

After explicit upload/spend approval, one GPU call ran without application retry.
Worker time was 154.114 seconds within the 300-second bound, including warmup.
The 78 native windows comprise 3 warmup and 75 measured windows (limit: 128).
Modal reports application `ap-tlZe5HYuZEucJR2fUeZZNl` stopped at
2026-09-07 00:21:37 +08:00 with zero tasks. Image construction and GPU scheduling
added wall time. The $0.105 preregistered compute estimate for the full bound is
not an invoice or spending cap and excludes startup, build and other fees.

The opt-in Python path is implemented. CLI defaults remain unchanged. Before
promoting this profile, repeat the longer source-paced test and measure sustained
decoding work and memory. No further GPU job is scheduled here.

The [decoder-work follow-up](2026-09-07-decoder-work.md) attributes all 216
extra decoder forwards and tests shorter context and verified drafts on CPU.
Those local experiments do not replace or extend this T4 qualification.
