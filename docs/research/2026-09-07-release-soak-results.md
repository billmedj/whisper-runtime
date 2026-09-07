# Native soak attempt: stopped at smoke — 2026-09-07

The authorized four-hour attempt **failed at the registered smoke memory gate**.
No hour-one phase started; no four-hour endurance qualification was obtained.
The thresholds were not raised and no further GPU attempt was authorized.

The [portable audit](../../evidence/release-soak-smoke-audit-2026-09-07.json)
and [raw evidence bundle](../../evidence/modal-release-soak-attempt-2026-09-07.zip)
preserve both the failed local SDK-definition attempt and the subsequent actual
T4 run, including preflights, source archive, journals, 30 events and failed
terminal receipt. Earlier [candidate checks](2026-09-07-release-candidate-checks.md)
remain historical and unchanged.

## What happened

The first attempt ended locally after 81.864 ms: Modal 1.5.5 rejects `retries=0`
for generator functions during resource definition, before `app.run`. The
minimal correction omitted that unsupported argument and disclosed the SDK
capability in metadata; it did not add application retries. Fifteen focused
tests passed with the real pinned SDK, including a regression that forbids
network calls, hydration and remote execution while defining resources.

Only the harness changed among 50 uploaded files; review-only documentation and
tests were updated separately. All 53 recorded file hashes match the approved
source archive; the 32 runtime module hashes still match the installed CPU
candidate. Input, conservative profile, phase schedule and memory gates remained
unchanged.

The actual T4 smoke then processed the registered repeated speech/noise/silence
input, not four hours of diverse audio:

| Observation | Result |
|---|---|
| Source / elapsed time | 46.55 s / 47.103489543 s |
| Coverage | 744,800 accepted and committed samples; 2,328 chunks; no buffered remainder |
| Native decoding / events | 25 decodes; 30 events; four commits and one FINAL |
| Reconstructed word score | 7 edits / 114 reference words = 6.14035%, within the registered 10% smoke-quality gate |
| Process RSS | 5,292,425,216 bytes = 4.928955 GiB, above the fixed 3.5 GiB ceiling |
| CUDA allocated / peak allocated / reserved | 160,720,896 / 290,099,200 / 358,612,992 bytes |

The 47.103489543-second clock starts after `owner.factory`, excluding model
initialization and build time; it is not startup-inclusive end-to-end latency.

Independent replay through `CommittedTranscript` validates the exact revision
chain, complete coverage and FINAL. The event digest and final journal digest
also match. Cleanup and restored capacity are reported consistently. Nevertheless,
the overall result is failed and the event receipt remains `partial: true`:
valid transcript completion does not override the memory gate.

App `ap-pABNi973B2Un3GiEhrHvNB` exited its ephemeral context. Modal reports
`stopped_at` as 02:52:50 UTC (10:52:50 +08) on September 7; the root agent later
verified zero active tasks.
The independent local audit did not make another remote status request.

## What the memory result does not establish

Modal's `memory=4096` is a minimum reservation, not an enforced usage cap; higher
actual usage may be billed. It does not replace the experiment's stricter
registered RSS ceiling. [Modal resource documentation](https://modal.com/docs/guide/resources)

The harness sampled process RSS through `/proc/self/statm`. Its resident/shared
counts have documented approximation limits; `smaps` or `smaps_rollup` provide
more detailed accounting. This does **not** establish that measurement error
explains the observed excess. [statm documentation](https://man7.org/linux/man-pages/man5/proc_pid_statm.5.html)

No pre-model baseline, mapping breakdown or cgroup measurements were captured.
`smaps` distinguishes proportional, private and shared mappings; cgroup
`memory.current` covers the cgroup and descendants, not just process RSS or CUDA
allocator counters. [smaps documentation](https://man7.org/linux/man-pages/man5/proc_pid_smaps.5.html),
[cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)

The post-smoke baseline was never established because the absolute gate failed
first. Consequently, neither a leak nor a particular allocation/mapping source
is proven. The cause remains unresolved; the registered failure is preserved.
