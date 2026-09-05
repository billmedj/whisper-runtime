# Retained context: implementation and diagnostic status

Date: 2026-09-05. Executed source commit: `9c10520`.

This page records the first attempt. The later
[four-cell comparison](2026-09-05-boundary-comparison-results.md) recovered the
results and supersedes the next-test status below. Historical observations and
failed receipts remain unchanged.

## Implementation

The stream controller now separates three sample positions: retained audio,
committed output, and accepted input. A caller can retain bounded acoustic
context after publishing a transcript prefix. The default retains no extra
context and keeps the existing profile.

The opt-in profile publishes only complete, contiguous timestamp segments after
the committed boundary. It does not splice tokens, rewrite timestamps, or alter
committed text. A segment that crosses that boundary can stop progress. An
unresolved EOF stops decoding rather than repeating the same failed decision.

The latest decode trace records the analysis interval, raw hypothesis, selected
publication interval, and policy decision. It describes a planned decision;
it does not certify that publication and resource cleanup succeeded.

Eighteen new scripted tests cover retained context, buffer limits, publication,
EOF, cancellation, and cleanup recovery. The 288 runtime tests also pass when
imported from the built wheel. These tests check runtime behavior, not speech
recognition quality.

## Modal attempt

The registered diagnostic compares four configurations on the same 33-second
input: baseline, two seconds of retained context, two seconds of publication
holdback, and both. It uses one T4 call, `tiny.en`, FP32, and no real-time pacing.

- Registration: `experiments/modal-stream-boundary-diagnostic-v1.json`.
- Source snapshot: `5e1bc2f8906966e7260949de79285c67eb3f0922ce2992854be12ffa7a149db6`.
- App: `ap-Zqa6dAuFt88Rt4zYAdCrD7`.
- Function call: `fc-01M1QWRSXZKEXWAHD3PMGVDCNM`.

The Modal dashboard reports that the function succeeded in 28.731 seconds.
The local client then failed to deserialize its return value. No comparison
record was saved. Function success does not establish that all four cells
completed or that their checks passed: the producer can return a failure record.

The cause was a string-valued enum that survived conversion to plain values.
Modal serialized that Python object by reference to `whisper_runtime`, which
was not importable in the local client environment. The SDK's `.remote()` path
cleared the output before local deserialization. Retrieval of the existing call
with the correct import path returned `NotFound`. No second GPU call was made.

The original started/failed receipt remains in the local artifact directory.
The failed attempt consumes this diagnostic's one-call allowance. Its missing
transcripts and traces must not be reconstructed from expected behavior.

## Cost observation

The billing API's metered total changed from USD 0.06021114 before the call to
USD 0.07021114 afterward. This is an observed USD 0.01 increase, not an exact
per-call invoice or remaining-credit balance. All listed apps were stopped with
zero tasks after the attempt.

## Transport correction and next gate

The worker now returns a JSON string. The client saves the function-call ID
before retrieving its result with `FunctionCall.get()`. Local tests cover
string-valued enums, invalid records, and retrieval failures. A failed retrieval
preserves the call ID and still blocks another dispatch under the same allowance.
This correction has not been exercised on another GPU call. The failed attempt
and its source identity remain unchanged. Any further GPU call needs a separate
registered allowance.

After the correction, all 219 repository-tool tests pass, including 11 tests for
this diagnostic. The 288 runtime tests also pass. Static type, lint, and
repository checks pass. No second GPU call was used for these checks.

Recognition improvement remains unmeasured for the new profile. Do not choose
an overlap default, launch a long-session campaign, or claim a speedup from this
attempt. The next model test is still the small four-configuration comparison.
