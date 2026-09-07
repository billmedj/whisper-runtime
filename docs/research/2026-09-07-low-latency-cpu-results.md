# Earlier confirmed text: bounded CPU comparison

Disabling word-commit deferral moves first confirmed text from 10 seconds to
4 or 6 seconds of admitted source audio on the two tested clips. Adding
same-window feature reuse preserves those results exactly and halves encoder
forwards. This is a short CPU check, not a measured live-latency guarantee.

## Matched settings and result

Six fresh cells use cached `tiny.en`, FP32, seed 7 and the same two inputs.
All retain 20 seconds of left context with a 24-second word-context limit,
30-second windows, two-second previews and holdback, 200 ms timestamp tolerance
and zero draft tokens. Audio-evidence and publication thresholds are unchanged.
The non-deferred reuse arm matches the settings of `low-latency-v1`.

| Input | Setting | First COMMIT, admitted source | Encoder / decoder forwards |
| --- | --- | --- | --- |
| Noisy, 10.89 s | Deferred, legacy alignment | 10 s | 11 / 153 |
| Noisy, 10.89 s | Non-deferred, legacy alignment | 4 s | 12 / 132 |
| Noisy, 10.89 s | Non-deferred, reuse | 4 s | 6 / 132 |
| Continuous, 33.66 s | Deferred, legacy alignment | 10 s | 31 / 1,009 |
| Continuous, 33.66 s | Non-deferred, legacy alignment | 6 s | 34 / 1,023 |
| Continuous, 33.66 s | Non-deferred, reuse | 6 s | 17 / 1,023 |

Every cell completes full input coverage with one FINAL, an empty buffer,
closed native windows, restored capacity, empty queues/leases and removed
measurement hooks. Word edits remain 1/26 on the noisy clip and 6/88 on the
continuous clip across all three settings. Deferred and non-deferred joined
texts are not identical; matching error counts do not establish exact parity.

Reuse versus non-deferred legacy alignment does have exact parity: joined text,
commit source boundaries and admission positions, recorded decision traces,
final trace and all stream metrics match. Decoder forward counts and input
frame/token totals are unchanged. These are forward-input counts, not measured
kernel time, wall-clock speed or energy savings.

## Why the timer changes, and the cadence tradeoff

Deferred alignment starts at `window - max(left context, preview) - preview`.
The standard 2/6-second context settings reserve observations at 26 and 28
seconds; 20/24 reserves 8 and 10 seconds. Without deferral, alignment begins
at 2 and 4 seconds. The continuous clip's 4-second hypothesis is unstable, so
the unchanged agreement gate correctly waits until 6 seconds.

Non-deferred publication clears its comparison witness after each commit. It
therefore needs another agreeing pair: continuous commits occur at 6, 10, 14,
18, 22, 26, 30 and 33.66 seconds here. The deferred setting retains the
same-origin witness and usually commits every two seconds after its first
10-second commit, with two four-second gaps after moving the window. Earlier
first publication does not imply faster subsequent cadence or bounded per-word
latency. No engine layer or publication threshold was changed for these probes.

## Evidence and limits

The [checked summary](../../evidence/low-latency-cpu-2026-09-07.json) contains
all commit positions, source/runtime hashes, work counts and comparison checks,
without absolute local paths. Local raw receipts are
[deferred](../../artifacts/confirmed-latency-20260907/deferred-20s24s.json),
[non-deferred](../../artifacts/confirmed-latency-20260907/no-deferral-20s24s.json)
and [non-deferred reuse](../../artifacts/confirmed-latency-20260907/no-deferral-20s24s-reuse.json).
Their SHA-256 values are recorded in the summary; the reuse receipt is
`ab5e7eb5bf8f1fbac9ce87149a6460a6feb66c30810a04d7f19861c390f70c3d`.

Runtime hashes stayed unchanged across the cells. Legacy and reuse use their
separately pinned standard and optional patched backends; model fingerprints
match. The existing custom-config factory driver checks profile settings, not
installed CLI/profile routing. Five deterministic scheduling/refusal tests and
Ruff pass.

Admission is accelerated in two-second chunks. The reported seconds are input
positions when events appear, not elapsed CPU time, source-paced latency or
end-to-end microphone/network delay. This is one cell per setting/input, with
no new GPU work or downloads. Longer source-paced, broader-audio and release
qualification remain separate gates; the historical standard control and its
near-28-second first commit are preserved in the
[earlier study](2026-09-06-retained-context-commits.md).
