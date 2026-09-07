# Verified token drafts: matched T4 result

The registered test completed on 2026-09-07. Reusing prior-window tokens as
proposals reduced decoder forwards and measured decoding time without changing
the recorded tokens or committed text. Production defaults are unchanged.

## Comparison

One T4 worker loaded `tiny.en` once. Both arms used FP32, English, greedy
decoding, seed 7, 20/24-second retained context and same-window alignment
encoder reuse. Each arm received the same 46.55-second source-paced input:
speech, noise and digital silence, with 114 reference words.

The control ran first. The candidate verified up to 32 prior-window tokens
against current-audio features. A mismatch ended the proposal; ordinary
decoding continued. It reused no old-window encoder features or decoder cache,
loaded no assistant model and changed no model weights or publication rules.
Each arm had two warmup windows. Warmup and a cancellation probe are excluded
from the table.

| Measurement | Control | Verified draft |
| --- | ---: | ---: |
| Decoder forwards, including alignment | 1,214 | 917 |
| Decode-phase forwards | 1,196 | 899 |
| Alignment decoder forwards | 18 | 18 |
| Encoder forwards | 25 | 25 |
| Submitted decoder token positions | 2,197 | 2,533 |
| Sum of encoder and decoder CUDA forward intervals | 4,695.50 ms | 3,703.26 ms |
| Decode-phase native operation wall time | 5.729 s | 4.763 s |
| Alignment native operation wall time | 1.783 s | 0.183 s |
| Peak PyTorch allocated memory | 224,433,152 bytes | 224,433,152 bytes |
| Peak PyTorch reserved memory | 306,184,192 bytes | 306,184,192 bytes |
| Native windows | 25 | 25 |
| Commit events, including empty silence commits | 14 | 14 |
| Word edits / reference words | 7 / 114 | 7 / 114 |

The candidate accepts 297 of 633 proposed tokens. Decoder forwards fall by
**24.5%**; the CUDA forward-interval sum falls by **21.1%**. Decode-phase native
wall time falls by **16.9%**, including proposal verification and host-side
work inside the measured native operations. These intervals exclude waiting
for input, mel construction and outer stream-driver work. They are not
end-to-end latency.

## Output and cleanup

All 25 raw native token sequences and all 14 commit texts and source spans
match. Supporting analysis endpoints and the selected policy decisions also
match. Both arms account for all 744,800 samples, emit FINAL once and finish
with an empty buffer. Hooks, model state and native capacity are restored.
The separate cancellation probe also passes its recorded cleanup checks.

Scores are not bit-identical. Maximum absolute differences are
`9.5367431640625e-7` for average log probability and
`1.4156103134155273e-6` for no-speech probability. No threshold was relaxed.
Receiver input positions at publication differ with processing speed; the
analysis endpoints supporting publication do not. First commits occur at
10.539 s and 10.196 s. Time from final input admission to FINAL is 52.52 ms
and 60.78 ms; drain time does not improve. The current control reproduces the
prior T4 control's recorded commits, counts and selected decisions.

## Limits

Do not attribute the entire reduction in total native wall time to drafts.
The control has six alignment stalls absent from the second arm, although
alignment call counts and GPU intervals are similar. First-use compilation
or fixed-order effects are possible explanations, not established causes.
The decode-phase measurement above avoids attributing these alignment stalls
to the candidate. It remains a single fixed-order screening result.

Submitted token positions increase by 15.3%. Fewer calls do not establish
lower FLOPs or energy. Peak and post-close memory are unchanged between arms;
this test does not demonstrate GPU memory release. It does not qualify other
models, sampling, beam search, checkpoint/restart or sustained live use.

## Evidence and resources

The [plan](2026-09-07-draft-gpu-plan.md) was frozen before execution. The
[archive](../../evidence/modal-draft-features-2026-09-07.zip) contains the exact
JSON result, compressed worker response, preflight and attempt journal.
The [offline audit](../../evidence/draft-gpu-audit-2026-09-07.json) recomputes
the comparison from the saved record.

From the repository root, reproduce the audit without a model or Modal call:

```sh
python -B -m tools.verify_draft_gpu artifacts/modal/draft-t4-20260907 artifacts/modal/composed-features-20260906-v1/acoustic-diagnostic.json
python -B -m unittest tools.test_verify_draft_gpu -q
```

The first command reads the extracted records; it prints the audit and does
not overwrite evidence. The historical control is in the earlier
[composed archive](../../evidence/modal-composed-features-2026-09-07.zip).
The saved audit reproduces exactly. Thirty focused local tests and Ruff pass;
all 59 frozen source file hashes still match after documentation and archiving.

- Replay: `draft-t4-20260907`.
- Source: 59 files, 1,208,823 bytes; digest
  `98e9629766c54ab48015ae4a473c6140d6cd53238c9d3a4a0a822c110aff6741`.
- Result SHA-256:
  `8d1923e2fe79505a82c7a25cfc7abbac56752c9e7e848958185180b385ecf9a0`.
- Archive SHA-256:
  `8bd827de48f11cfcfd25b9860889467cc93d4b1407ea3f2278bb315526c15ae1`.
- One GPU invocation; worker duration 106.252 s, registered limit 240 s.
- Modal app `ap-r6Ziy9BMxF0wGUZEEdzdOq` stopped at
  `2026-09-07T01:44:07+08:00`; the subsequent status check showed zero tasks.

The preregistered $0.083614 planning compute estimate is not an invoice or
billing cap. It excludes startup, image work, storage, network, taxes and
provider rescheduling. No second GPU test is queued.

## Next boundary

Replace the diagnostic's process-level patching with a native, opt-in greedy
path. Verify mismatch, early EOT, score thresholds, cancellation and
checkpoint/restart before exposing it. Then test longer, held-out streams
with order-balanced timing. Preserve the ordinary path as the default until
those checks pass.
