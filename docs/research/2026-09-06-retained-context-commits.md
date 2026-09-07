# Earlier commits with retained audio context

Status: short native test passed. The CLI defaults are unchanged.

## Change

A word commit publishes text. It need not move the audio window. Previously,
every commit cleared the comparison history and reset the observation cursor to
the committed position. When the window origin stayed fixed, this could repeat
analyses of shorter prefixes that the stream had already processed.

Deferred word commits now keep the accepted alignment when its audio origin is
unchanged. The next observation must extend it. The existing word, anchor, audio
evidence and transaction-release checks still apply. Moving the origin, closing
a speech unit, accepting silence or reaching FINAL clears the comparison history.
Other publication profiles keep their previous behavior.

Publication-boundary checkpoints reconstruct this alignment from the existing
committed record. No tensor cache or new checkpoint field is added. Tests compare
uninterrupted and restored events, state, metrics and native window calls.

## Candidate configuration

The test retains 20 seconds of left context and permits a 24-second word-context
span. All other CLI settings remain unchanged: 30-second analysis window,
two-second previews and holdback, fixed input buffer limit, strict word checks
and the existing quiet detector. The first comparison pair is at 8 and 10 seconds
of source audio. No model weights, silence thresholds or lexical checks change.

An earlier 24/25-second context trial is rejected: it publishes at six seconds
but cannot finish the continuous clip after moving the window. Its
[failure summary](../../evidence/retained-context-initial-probe-summary-2026-09-06.json)
is retained. The candidate below uses the scheduling correction and 20/24 seconds.

## CPU comparison

Both arms use the same updated runtime, cached `tiny.en`, FP32 and seed 7.
Input admission is accelerated; these first-commit positions are not elapsed
live latency. The control uses the unchanged 2/6-second CLI context settings.

| Input | First commit, control / candidate | Word edits, both arms | Complete |
| --- | --- | --- | --- |
| 33.66-second continuous speech | 28 / 10 seconds of admitted audio | 6 / 88 reference words | Both |
| 10.89-second noisy prefix | EOF / 10 seconds of admitted audio | 1 / 26 reference words | Both |

Full reports: [control](../../evidence/retained-context-cpu-control-2026-09-06.json)
and [candidate](../../evidence/retained-context-cpu-2026-09-06.json).
Both finish with an empty buffer and restored request capacity.

## Source-paced T4 test

Attempt `v01-early-commit-20260906-a1` sends the same 46.55-second registered
speech/noise/silence sequence as the earlier V0.1 smoke test. One T4 runs the
existing cached model. This is one short run per configuration, not a repeated
latency benchmark. Timings start after the server's READY message; they exclude
deployment and model startup.

| Measure | Earlier smoke test | Candidate |
| --- | --- | --- |
| First provisional text | 4.255 s | 4.098 s |
| First commit | 31.156 s | 13.220 s |
| Commit events, including silence | 4 | 14 |
| Full-sequence word edits | 7 / 114 | 7 / 114 |
| EOF to verified DONE | 0.892 s | 0.840 s |
| Peak buffered source audio | 30.76 s | 27.16 s |
| Native decode count | 25 | 25 |
| Sum of decoded window durations | 288.31 s | 349.07 s |

All 744,800 input samples are accounted for, the buffer is empty, FINAL is
present, the model fingerprint is unchanged and request capacity is restored.
The final silent span produces a native `you` hypothesis, which the digital-
silence rule suppresses. This case does not establish general hallucination
prevention. The independent event audit confirms full-sequence scoring; three
commits cross recipe boundaries and cannot support separate per-arm scores.

Summed source-window duration increases by about 21%, with the same number of
decodes. This is an audio-overlap measure. Each initial encoder input is still
padded to 3,000 frames; alignment forwards and decoded tokens also affect work.
This test did not measure their device execution. It establishes neither a
compute saving nor a 21% increase in GPU work. The
[matched composition experiment](2026-09-06-composed-inference.md) measures the
missing forward counts and device-event intervals.

## Evidence and limits

Local validation runs 800 core tests (one optional skip) and 704 repository-tool
tests (three optional skips). The updated offline audit passes 17 focused tests.
Ruff and strict mypy pass for all 30 source modules. The full tool suite uses
the existing native environment with JSON Schema support; an initial run in the
host interpreter lacked that dependency and is not counted as a passing run.

The [original receipt archive](../../evidence/modal-early-commit-2026-09-06.zip)
contains the preflight, result, attempt journal and full event log. Its SHA-256 is
`c373288b353e948ca246712a33e086b7b619b2b97281037330c55dd649b96bb7`.
The uploaded 48-file source digest is
`8020fe939a04104cb3c93f14835be59edabf8304b4611502eddb249c150e6979`.
The [offline audit](../../evidence/retained-context-live-audit-2026-09-06.json)
checks the configuration, receipt identities, event hash and committed coverage.
To repeat that audit after extracting the archive, run
`python tools/summarize_live_qualification.py <extracted-directory>`.

Modal reports application `ap-SJrcUDjLQmsej9VNz0R5RD` stopped at
2026-09-06 23:08:28 +08:00. Token cleanup passed. No long stage or automatic
retry ran. The preflight compute estimate was $0.073162, not a billing cap or
an invoice; startup and other listed fees are excluded.

Before making this configuration the default, repeat the longer source-paced
test and compare sustained work, memory and publication delay. The previous
30-minute result belongs to the old configuration and does not qualify this one.
Broader audio, physical microphone and release-platform checks remain open.
