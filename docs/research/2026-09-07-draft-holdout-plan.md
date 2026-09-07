# Native drafts on new audio: registered screen

Registered before model inference. Keep runtime code and defaults unchanged.
Use the [frozen public-audio manifest](../../experiments/draft-holdout-20260907.json).
The two selected LibriSpeech speakers, 260 and 4446, were not used in prior
project evidence. This is new project test material, not a claim that Whisper
did not encounter it during training.

Concatenate the unchanged 7.04- and 4.445-second clips, with two seconds of
digital silence after each: 15.485 seconds, 247,760 samples. Freeze the PCM hash,
independent transcripts and source snapshot before the GPU call. No recognition
output was used to select these clips. Record the acquisition history rather
than implying that only two dataset rows were inspected.

Use the integrated native adapter, cached tiny.en, English FP32 greedy,
seed 7, 20/24-second retained context, and alignment encoder reuse. Only
max_draft_tokens changes: control 0, candidate 32. Warm both paths twice on
the first eight seconds, then run fresh source-paced streams in ABBA order.
Use the same pinned backend. No diagnostic decoder method replacements, new
thresholds or score tolerances.

Record complete tokens, events, source boundaries, selected decisions,
admission clocks, scores, native calls, CUDA intervals, phase wall time,
allocator memory and cleanup. Compare adjacent pairs and pooled totals.
Publication and token parity are required independently of speed. Report
transcript word edits against the frozen references even if both arms agree.
Treat failure or unchanged performance as a result, not a reason to replace
the clips or retry the GPU run.

A blocked publication may be recorded and followed by the other registered
arms, so that refusal behavior can be compared. An infrastructure or cleanup
error stops the worker. Full-completion and efficiency gates remain separate.
Do not credit an alignment startup difference to verified drafts.

Exactly one T4 call, 300-second limit, 20-second cleanup reserve, two CPU cores,
4 GiB host memory, at most 256 native windows. No automatic retry, deployment,
model download or writable model cache. Worker network access is blocked.
Source and two PCM fixtures are allowlisted and hash-verified. A small CPU
transport probe must pass before the GPU call.

Using the prices checked in the preceding registered plan, the 300-second
planning compute estimate is $0.104517 with the 1.75 regional multiplier.
This is not a bill or spending cap; build/startup/storage/network/taxes and
provider rescheduling are excluded. Preserve the compressed response before
validation. Local fresh-process restore and threshold tests are separate:
they must not be reported as GPU restart or population-level qualification.
