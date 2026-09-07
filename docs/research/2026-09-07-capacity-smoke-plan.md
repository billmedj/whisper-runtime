# Provider memory-capacity smoke test

Registered: 2026-09-07, before remote execution.

## Question

Can one worker complete two source-paced transcription sessions under Modal's
documented 4 GiB container-memory limit?

The earlier 3.5 GiB guest-RSS test failed. Its record and threshold stay unchanged.
Guest `/proc` readings did not establish host physical memory use. This is a
separate capacity test, not a relabeling of that result.

## Fixed workload and execution bounds

- Mode: `provider-capacity-smoke-v1`, selected with `--capacity-smoke` for both
  preflight and execution.
- One T4, one container, one model owner; no configured retry or restart.
- Modal SDK 1.5.5; `memory=(4096, 4096)`. Check the SDK resource conversion and
  the actual function declaration before execution. Scalar `4096` is only a
  request. [Modal resource limits](https://modal.com/docs/guide/resources).
- Function execution timeout: 300 seconds. Startup timeout: 180 seconds.
- Two successive 46.55-second sessions, each with 744,800 mono 16 kHz samples.
  Use the existing public audio recipe and read-only cached `tiny.en` weights.
- Use `conservative-v1`, FP32 and the pinned native backend. Do not change the
  decoder, audio, publication thresholds or execution profile.
- Stop if the first session fails. Do not start hourly sessions in this test.

The prior planning estimate is USD 0.104517 for 300 seconds of requested
resources. It excludes image construction, startup and other charges. It is
not an invoice or a guaranteed billing ceiling. The frozen preflight records
the upload contents, source digest, input hash and budget assumptions.

## Acceptance

1. Emit a resource-contract and capability record before importing Torch. It
   records the submitted configuration, not an independent host attestation.
2. Require one source identity and one worker identity throughout the record.
3. Require both sessions to complete source coverage, emit FINAL, pass the
   registered word-edit check and close with capacity restored.
4. Require identical TXT, SRT and VTT output across the two sessions.
5. Require CUDA accounting after each closed session. Use the first closed
   session as the fixed baseline: no increase in allocated bytes; reserved
   growth at most 256 MiB and reserved memory at most 2 GiB. Do not call
   `empty_cache` to meet the checks.
6. Require a valid terminal record and successful collection. Missing terminal
   data, a changed worker, timeout or provider failure fails the test. Missing
   terminal data alone does not establish an out-of-memory kill.
7. Confirm that the app is stopped and has no active tasks after execution.

Guest memory readings are optional observations in this mode. Missing `smaps`
or `smaps_rollup` does not fail a capacity test; missing required CUDA accounting
does. Do not use guest RSS as a physical-memory ceiling or invent unavailable
values. The attribution-diagnostic mode retains its existing failure behavior.

## Interpretation and next step

A pass supports completion of this workload under the submitted provider limit.
It does not establish process RSS below 4 GiB, spare capacity, long-run stability,
or the absence of leaks. It does not repair the earlier failed RSS test.

After a pass, register a bounded endurance run with this accounting model.
After a failure, inspect the recorded cause before changing anything. Do not
raise the memory limit or launch another paid attempt automatically.

The package remains `0.1.0.dev0`. This test does not authorize a stable tag or
a GitHub push.
