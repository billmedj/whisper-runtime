# T4 memory attribution: two completed sessions, incomplete accounting

The authorized diagnostic ran once and stopped. Both source-paced transcription
sessions completed. The overall diagnostic **failed** with
`memory_sampling_incomplete`: `/proc/self/smaps_rollup` was unavailable in every
snapshot. Neither this result nor the earlier failed endurance gate is a pass.

[Portable audit](../../evidence/memory-attribution-audit-2026-09-07.json) and
[raw evidence with frozen source](../../evidence/modal-memory-diagnostic-2026-09-07.zip)
preserve the input, all observations, journal, source and review files.
The archive SHA-256 is
`0d17e1e89ffe5f82bcda8a7eb3a8030ac27c8a17bcca51b8f240326d1ed0dc79`.

## Transcription and lifecycle

One T4, one model load and one native owner processed two identical 46.55-second
inputs using `conservative-v1`. Each session accepted and committed 744,800
samples in 2,328 chunks, made 25 native decodes and emitted one FINAL. No input
remained buffered. The independent event replay reproduces identical TXT, SRT
and VTT exports across sessions. Both have 7 word edits against 114 reference
words (6.14%). This is repeated-input consistency, not held-out accuracy.

Source-paced session clocks measured 47.196 and 47.084 seconds. These clocks
exclude model initialization, image construction and provider startup.
Model fingerprints match before and after both sessions. Each session restores
its runtime capacity. Terminal PyTorch CUDA allocation is 160,720,896 bytes
after each session; reserved allocation also matches across these boundaries.

The failed diagnostic's `cleanup_verified: false` belongs to its additional
failure-handler path, reached after both sessions were already closed. It does
not negate the two verified session-cleanup receipts. Raw fields are unchanged.

Modal reports app `ap-G629ZxMxWkpe3mCZTA3Hxt` stopped at 04:27:09 UTC.
A separate read-only query completed at 04:27:47.2390229 UTC and showed zero
active tasks. The command exited with status 1. No retry, long test, persistent
deployment or subsequent paid job was launched. Actual billing was not retrieved.

## Where the guest RSS counter increased

These are `/proc/self/statm` readings inside the Modal guest, not verified host
physical-memory measurements.

| Observation | Reported RSS, bytes | Change from preceding row, MiB |
| --- | ---: | ---: |
| Before native imports | 64,372,736 | — |
| After Torch import | 3,389,759,488 | 3,171.34 |
| After remaining native imports | 3,573,563,392 | 175.29 |
| After model load and fingerprint | 3,826,253,824 | 240.98 |
| Before first CUDA alignment call | 5,266,440,192 | 1,373.47 |
| After that alignment call | 5,290,086,400 | 22.55 |
| After session 1 close and GC | 5,308,788,736 | 17.84 |
| After session 2 close and GC | 5,313,556,480 | 4.55 |

Most of the observed first increase precedes model loading and transcription.
The large later increase is already present before the first DTW call; that
call alone cannot explain it. Fourteen CUDA DTW calls and zero CPU DTW calls
were observed. Thread count stayed between three and seven. The proposed CPU
alignment fallback/thread-pool explanation is therefore unsupported in this run.

Only 4,767,744 additional guest RSS bytes remain after the second short session.
That does not prove a leak, absence of leaks, or long-run stability. Allocation
sites, file mappings, host memory and proportional/private/shared memory were
not captured. The exact cause of the full RSS reading remains unresolved.

## Why another Linux counter is not automatically a solution

Modal documents its use of gVisor's userspace kernel. gVisor documents that guest
`proc` memory statistics are approximate; host-level accounting is distinct.
[Modal runtime description](https://modal.com/blog/truly-serverless-gpus),
[gVisor resource model](https://gvisor.dev/docs/architecture_guide/resources/).

The inspected upstream gVisor implementation of `smaps` reports PSS equal to
RSS, zero shared pages and simplified private/dirty attribution. The exact
deployed Modal revision was not identified. A fallback to `smaps` can restore
compatible reporting where that file exists, but it cannot establish accurate
host-private memory merely by returning more fields.
[Upstream implementation](https://github.com/google/gvisor/blob/master/pkg/sentry/mm/procfs.go).

This limitation does not show that all excess RSS is spurious. The original
3.5 GiB guest-RSS gate remains failed. It must not be retrospectively relabeled
as passed or silently raised.

## Next qualification step

1. Check available accounting interfaces before loading the model. Preserve
   unavailable fields as unknown; never replace them with zero or a success.
2. Use a separately registered provider-enforced memory limit or verifiable
   host/container telemetry to test actual capacity. A scalar Modal memory
   request is not a hard limit; its documented tuple form distinguishes request
   from limit. Guest RSS can still reveal changes between repeated sessions,
   with its provenance recorded. [Modal resources](https://modal.com/docs/guide/resources).
3. Run a short capacity check before returning to four-hour endurance. Any
   revised acceptance policy must state which former metric it replaces and why;
   keep the failed records and obtain approval before new paid execution.

No decoder behavior, weights or existing release thresholds changed in this run.
The package remains `0.1.0.dev0`, with no push or stable release.

## Subsequent local sampler correction

The diagnostic now attempts bounded streaming `smaps` aggregation only when
`smaps_rollup` is missing. Permission failures and malformed records remain errors;
if both files are absent, the result is still incomplete. Sample schema v2 records
the metric source and explicitly marks physical/private/shared attribution as
unverified guest estimates. It does not claim to repair capacity accounting.

Twenty-two focused tests pass with the pinned Modal 1.5.5 SDK, including actual
resource declarations with remote execution blocked. This correction was not
run on T4. The source used for the completed attempt remains in the evidence ZIP;
the old preflight is intentionally no longer valid for a new invocation.
