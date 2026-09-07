# Decoder work: context retention and verified drafts

Date: 2026-09-07. Local CPU experiments only; no new GPU call.

The previous T4 result added 216 decoder forwards, not 216 native windows.
This investigation separates the cause from two possible remedies. Production
defaults, backend source and publication thresholds remain unchanged.

## 1. Account for the measured cost

The [offline cost audit](../../evidence/composed-decoder-cost-2026-09-07.json)
recounts the original raw forward records. It does not infer cost from audio
duration or from an estimated token count.

| Work | Conservative | Early commits with encoder reuse | Difference |
| --- | ---: | ---: | ---: |
| Decode forwards | 991 | 1196 | +205 |
| Alignment decoder forwards | 7 | 18 | +11 |
| Total decoder forwards | 998 | 1214 | +216 |

Only windows 12–17 differ in decode work, using zero-based indices. Windows
14–17 add 219 forwards; windows 12–13 save 14. All other decode counts match.
The net token difference is 177 lexical tokens and 28 timestamp tokens. For
each window, decode forwards equal generated native token count plus one.
There is no extra temperature or EOF retry behind this difference.

In the four expensive late windows, the early-commit profile retains about
20 seconds of already committed audio. The conservative profile retains about
2 seconds. Most extra generated tokens describe that old audio. They are not
published twice. Every encoder input still has 3000 frames, so shortening
source duration does not by itself shorten the encoder tensor.

Reproduce the accounting after extracting the original receipt archive:

```console
python tools/analyze_composed_decoder_cost.py PATH/acoustic-diagnostic.json
```

## 2. Separate the first word check from context retention

The existing first-check offset is:

```text
max_window - max(left_context, preview_interval) - preview_interval
```

With a 30-second window and 2-second previews, a 20-second retained context
starts checks at 8 seconds. A 2-second context starts them at 26 seconds.
One setting therefore controls two different decisions.

The [CPU probe](../../tools/verify_word_schedule.py) advances that check to
8 seconds while retaining the original earlier reserve when necessary. It
changes neither word matching nor context selection. This is a process-local
experimental subclass, not a new public configuration or checkpoint profile.

[Receipts](../../evidence/word-schedule-cpu-2026-09-07.json) compare three new
cells with two archived CPU controls on the same backend, model and PCM.
All use alignment encoder reuse, `tiny.en`, FP32 and seed 7. Input is admitted
in 2-second chunks without pacing.

| Input / context | Encoder forwards | Decoder forwards | Word edits | Largest gap between commit admissions |
| --- | ---: | ---: | ---: | ---: |
| Continuous 33.66 s / control 20/24 s | 17 | 1009 | 6/88 | 4 s |
| Continuous / early check, 2/6 s | 18 | 550 | 6/88 | 14 s |
| Continuous / early check, 6/10 s | 27 | 920 | 6/88 | 5.66 s |
| Noisy 10.89 s / control 20/24 s | 7 | 153 | 1/26 | 0.89 s |
| Noisy / early check, 2/6 s | 6 | 116 | 1/26 | 0.89 s |

All cells finish, cover the input, emit one FINAL and restore native capacity.
The first commit follows admission of 10 seconds of source audio in every
cell. These are admission positions, not source-paced or microphone latency.
Equal edit counts do not imply identical text, segmentation or timestamps.

**The 2/6-second profile is not a replacement for the early-commit profile.**
It cuts decoder forwards by 45.5% on the continuous fixture, but the frozen
anchor is absent from several later candidates. Publication waits until a
valid continuation appears. The 14-second commit gap is a real drawback for
live use, not an error to hide by weakening the matcher.

The 6/10-second middle setting also changes scheduling and adds analyses.
Neither candidate establishes lower GPU time or a general context optimum.
Earlier context experiments already showed that retaining whole estimated
words does not guarantee sufficient recognition context; see the
[word-context failure analysis](2026-09-05-word-context.md).

## 3. Retain context; verify an existing draft in parallel

The next candidate uses the preceding native token sequence as an untrusted
proposal. It is not a forced prompt and does not reuse old-window encoder
features, decoder K/V or confidence scores.

For the current audio, the prototype builds a fresh causal decoder cache over
the initial tokens and a short draft. Standard per-token steps still apply
the original suppression and timestamp rules. Matching proposed tokens can
use the precomputed prediction rows. At the first mismatch, the prototype
discards the unaccepted suffix, trims only current-run self-attention K/V and
continues ordinary decoding with the correcting token. Current-audio cross-
attention K/V remains valid. No second model or training is involved.

The [six-trajectory CPU diagnostic](../../tools/verify_decoder_draft.py) uses
an 8-second source window and a 10-second target window from the same fixture.
The [receipt](../../evidence/decoder-draft-cpu-2026-09-07.json) records:

| Target variant | Decoder forwards | Decoder input tokens | Same output tokens and text |
| --- | ---: | ---: | --- |
| Ordinary decoding | 39 | 39 | Reference |
| Prior 16-token draft | 24 | 40 | Yes |
| Prior 32-token draft | 24 | 56 | Yes |
| Deliberately wrong first token | 39 | 55 | Yes |
| Truncated 8-token draft | 31 | 39 | Yes |

The longer drafts both match 15 tokens before an old comma differs. The wrong
first token matches none. The truncated draft exercises full acceptance and
ordinary continuation. All caches and hooks are cleaned up; model and current
features remain unchanged.

**Numeric results are not bit-identical.** Average log-probability differs by
up to 2.690e-7; no-speech probability differs by up to 7.376e-7. Different
sequence shapes can change floating-point computation. Near a decision
threshold, even small differences may matter. Token equality on this example
does not prove universal greedy equivalence or publication equivalence.

Fewer forwards also do not mean proportionally less work. The wrong proposal
does extra token work without saving a forward. The 32-token draft costs more
than the 16-token draft here and accepts no additional token. GPU time, memory,
draft-size selection and cancellation boundaries remain separate test gates.

## 4. Full-stream CPU result

One additional [stream probe](../../tools/verify_stream_draft.py) runs the
unaltered fast 20/24-second profile with a maximum 32-token proposal from the
preceding completed decode. Each request still constructs its own cache from
current audio. Drafts remain untrusted across window growth and rebasing.

The [stream receipt](../../evidence/stream-draft-cpu-2026-09-07.json) compares
the same 33.66-second fixture with the archived fast CPU control:

| Measure | Control | Verified draft |
| --- | ---: | ---: |
| Decoder forwards | 1009 | 810 |
| Encoder forwards | 17 | 17 |
| Decoder input tokens | 1844 | 2105 |
| Native windows | 17 | 17 |
| Commit events | 11 | 11 |
| Word edits | 6/88 | 6/88 |

Joined text, all commit source boundaries, all commit admission positions,
metrics and the 17 archived policy trace summaries match exactly. The first
commit still follows admission of 10 seconds of audio; the largest admission
gap remains 4 seconds. All 538,560 samples are committed, the buffer is empty,
FINAL occurs once, and native capacity and hooks are restored. The model
fingerprint is unchanged.

This is 199 fewer decoder forwards (19.7%) without the delayed-publication
drawback of the short-context candidate on this fixture. It does **not** show
that the 216-forward T4 difference has disappeared: the earlier T4 input was
46.55 seconds and used source pacing.

Proposal work increases decoder input tokens by 261 (14.2%). This is a
reduction in sequential forward calls, not a demonstrated reduction in FLOPs,
energy or GPU time. The final no-speech score changes by -3.073e-8. The control
archive does not contain every raw per-token score or event payload; exact
comparison is limited to the listed recorded fields. No threshold was relaxed.

After the measurement, the comparison helper was tightened to reject missing
required fields. A read-only re-audit reproduces the recorded comparison;
inference was not repeated. The executed tool hash remains in the receipt.
Seventeen focused accounting, schedule and draft tests pass, as does Ruff.

## 5. Connections and limits

| New observation | Existing project evidence | Consequence |
| --- | --- | --- |
| Early timing is coupled to retention | Same-origin agreement witnesses already survive commits | Separate scheduling from evidence lifetime; do not discard witnesses to save work |
| Short context saves work but delays valid continuations | Earlier missing-anchor and timestamp-drift failures | Keep quality gates; shorter context alone is not the required solution |
| A previous transcript predicts part of the next one | Exact-window feature reuse rejects changed audio | Reuse tokens as proposals, not old tensors as current evidence |
| Block verification changes small numeric values | Publication uses native scores and aligned boundaries | Test full decisions, not just final text or aggregate WER |

The literature supports the decoding mechanism; it does not establish the
results of this implementation:

| Primary source | Relevant result | Connection and limit |
| --- | --- | --- |
| [Stern, Shazeer and Uszkoreit, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/c4127b9194fe8562c64dc0f5bf2c93bc-Abstract.html) | Blockwise predictions followed by longest-prefix verification | Established method. Here the draft comes from a preceding audio window rather than new prediction heads |
| [Gandhi, Hugging Face, 2023](https://huggingface.co/blog/whisper-speculative-decoding) | Speculative decoding applied to Whisper with an assistant model | Established Whisper use; this experiment does not load an assistant model or inherit that implementation's speed claims |
| [Wang, Xu and Lin, WhisperFlow, 2025 revision](https://arxiv.org/abs/2412.11272v2) | Streaming buffer alignment, beam pruning and CPU/GPU scheduling | Related work on repeated stream decoding; includes model/system techniques beyond this unchanged-model greedy experiment |

The bridge is specific: a provisional transcript can supply cheap proposals
while current-audio computation retains authority. This is an application of
known block-verification ideas, not a claim to invent speculative decoding.
The retained-context screen is an indirect scheduling improvement; replacing
sequential regeneration is the candidate aimed at preserving commit cadence.

## 6. Action register

1. Completed: full fast 20/24 CPU stream comparison. Listed observable fields
   match and decoder forward count falls. This is a development-fixture screen,
   not a held-out or sustained-live qualification.
2. Completed in the [matched T4 follow-up](2026-09-07-draft-gpu-results.md):
   decoder forwards fall from 1,214 to 917 on the original 46.55-second mixed
   input, with exact recorded token and commit parity. Decode-phase native
   wall time falls by 16.9%; peak memory is unchanged. See the follow-up's
   fixed-order and alignment timing caveats.
3. Before adding a public opt-in, test score-threshold boundaries, draft
   mismatches, early EOT, cancellation and checkpoint/restart. Keep beam search
   and sampling outside the initial greedy-only claim.
4. Consider a bounded draft-length policy only after measuring acceptance and
   wasted proposal tokens. Do not add a second model or a larger scheduler to
   solve an unmeasured problem.

At the time of this CPU investigation, no additional Modal job had run and
no CLI default had changed. The subsequent T4 test is recorded separately in
the linked follow-up. The preceding 54-file source snapshot remains unchanged.
