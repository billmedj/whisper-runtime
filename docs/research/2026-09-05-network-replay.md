# PC-to-Modal transcription: paced WebSocket diagnostic

## Result

A Windows client sends the registered 43.660-second PCM mixture to one Modal T4
over an authenticated WebSocket. Transcript events return while input continues.
The first nonempty preview reaches the PC at 2.426 seconds; the first commit
reaches it at 4.682 seconds. Five commits precede source EOF. A sixth commit and
one native final event complete the transcript.

All 2,183 chunks, comprising 698,560 samples, are accepted and committed. Source
ranges are contiguous and the received PCM hash matches the local source. The
final controller buffer is empty. Text matches both the full-input Whisper
control and concatenated fixture controls exactly. The same six normalized
word edits remain against the 88-word independent human reference.

The [record](../../evidence/modal-t4-tiny-en-network-replay-2026-09-05.json) has
`completed` status, with model state unchanged, native capacity restored and
the temporary proxy token deleted. The application was separately checked as
stopped, with zero tasks.

This is a short, real PC-to-GPU network replay. It is not microphone capture,
a user interface or long-session qualification. The source contains clean
English speech and constructed digital-zero pauses, not natural conversation.

## Implementation and input

Source commit: `2aa06f892297c9897929bd9e43e5ee9d27e6a753`.

The [transport reference](../NETWORK_REPLAY.md) adds a paced client and an ASGI
adapter around the existing controller. It changes no runtime recognition,
publication, transaction or resource policy. The client starts its source clock
after `ready`, then uses separate send and receive tasks. It does not wait for
per-chunk acknowledgements. The server's creating thread owns all native work
and cleanup; its receiver admits whole chunks through the existing input lock.

The server does not replay a local fixture. Its diagnostic wrapper captures the
bounded PCM received over the connection. Only after the native stream finishes
does it run offline controls on those received bytes. A one-second synthetic
warmup precedes `ready`; warmup output is not published.

The profile remains `quiet_endpoint_stream/v1+input_evidence/v1`, using cached
`tiny.en` in FP32. Five quiet-run endpoints and EOF define the same six units as
the [same-worker paced test](2026-09-05-paced-replay.md). No boundary is supplied
by the client. The input SHA-256 is:

`b6f8c541b55d2da4c9494dda2e5daebf8674c29286a3ea983206b9a99e9badad`.

## Timing measured at the PC

All values in this table use the client's monotonic clock after `ready`.
They exclude connection setup, model loading and warmup. Event receipt is not
the time at which a graphical subtitle renderer displays a word.

| Measurement | Result |
| --- | --- |
| Source duration | 43.660 s |
| First nonempty preview received | 2.426080 s |
| First commit received | 4.681731 s |
| Last audio send and EOF completed | 43.669546 s |
| Native final event received | 44.283120 s |
| EOF send completion to native final receipt | 613.573 ms |
| Diagnostic report received | 46.844307 s |
| Maximum source-offer lateness | 16.105 ms |
| Maximum audio send wait | 1.966 ms |
| Configured lateness / send-wait limits | 250 / 250 ms |

The final report arrives after the native final event. It includes offline
controls, model hashing and the transfer of diagnostic traces. The server
records 854.949 ms for controls and 473.360 ms for final hashing. Do not report
the full 3.175-second EOF-to-report interval as transcription drain time.

On the server's own clock, EOF admission to the native final event takes
148.585 ms. Client and server monotonic clocks have different origins: do not
subtract their raw timestamps or claim a measured one-way network latency.

Client source-end-to-commit-receipt lags range from 499.716 to 630.855 ms across
six commits. The source endpoint includes the detector's quiet interval, so
these are not word latency or full end-of-speech latency. The complete commit
and event timestamps remain in the record.

## Coverage and work

| Measurement | Result |
| --- | --- |
| Sent / admitted / committed samples | 698,560 / 698,560 / 698,560 |
| Sent / admitted chunks | 2,183 / 2,183 |
| Pre-EOF / total commits | 5 / 6 |
| Provisional / commit / final events | 25 / 6 / 1 |
| Peak controller PCM buffer | 165,120 samples, or 10.320 s |
| Final buffered samples | 0 |
| Native decodes | 25 |
| Summed source submitted to decoding | 129.660 s |
| Exact match against both model controls | Yes |
| Human-reference word edits | 6 / 88 |

The decode and buffered-audio counters match the preceding same-worker paced
run on this input. They do not establish cheaper inference. Controller PCM
metrics exclude the diagnostic's received-audio copy, event histories, socket
buffers and process/device memory. No GPU saving is demonstrated.

## Validation and the failed preflight

Local verification passes: 514 runtime tests and 332 repository-tool tests,
plus strict typing and Ruff lint/format checks. This includes 14 server tests,
21 client tests and 15 Modal network-harness tests. The native owner tests use
the existing transaction/fence fixtures, not a fabricated transcript service.

A separate artifact review verified all 28 frozen source files against the
source commit, reconstructed the input PCM, checked every admission and all
25 analysis-window PCM hashes, and joined publications to native results.
It independently reproduced the timing and reference metrics. The six
reference edits use normalized regex tokens; apostrophe splitting contributes
to that count. They are not six distinct spoken-word errors. This review was
performed by a second coding agent, not an external auditor.

Tests cover source pacing, independent receiving, blocked native work, duplicate
chunks, partial final chunks, changed digests, missing or changed publications,
slow consumers, cancellation, disconnection and cleanup failure. EOF bookkeeping
is atomic with completion. Native cleanup does not hold the ingress guard.
An event-limit failure retains the final native batch instead of losing an
already committed result.

The [first preflight record](../../evidence/modal-network-preflight-failed-2026-09-05.json)
is retained. Modal rejected an unsupported `retries=0` argument on an ASGI
function during local resource definition. No application or GPU was created;
the temporary token was deleted. The repair omits this argument and validates
resource definitions before token creation. Tests now construct both CPU and
GPU definitions using the actual pinned SDK, with remote execution forbidden.

The [successful CPU preflight](../../evidence/modal-network-preflight-2026-09-05.json)
then rejected an unauthenticated WebSocket with HTTP 401 and echoed 50 paced
binary frames exactly. It produced no ASR output. The paid run required that
successful record and the same source snapshot.

## Access, resources and evidence identity

Proxy authentication protects the Modal ingress. Each attempt creates a
temporary token, keeps its credentials in memory and deletes it after teardown.
This workspace's proxy tokens are not environment-scoped; their scope is
workspace-wide until deletion. No credential or endpoint URL is stored in the
evidence. Redirects are rejected and client connection retries are disabled.

One GPU WebSocket connection was opened, with no reconnect or configured retry.
The function has a 180-second execution timeout and one-container concurrency
limit. These are not a spending cap: Modal can reschedule crashed containers,
and startup/build time is separate. Actual billed cost and account balance
were not queried. No model or corpus download was required.

- Successful CPU application: `ap-ekmJWhs4y8ybs5djbh5xup`.
- GPU application: `ap-es0pNs7GARJe83ljL2Tdob`.
- Both applications were verified stopped with zero tasks.
- Both successful records confirm temporary-token deletion.
- No backend or Lean proof was changed; packages were not rebuilt.

GPU record SHA-256:
`ca6fb41ea9d8621eff12ccc825d99e7a85345a071740960d3f8e441236ed52df`.

CPU record SHA-256:
`987dda1bd9532134863d838176c21dd88a049fa175a84f9fd85e4a58a59c0ea3`.

Source snapshot digest:
`d6e6e523613acba951e74d49bbce20f52df6d66c02facfda6c1099b00d50e89b`.

Records and their attempt journals were copied to `evidence` without overwrite;
each copy was checked against the source hash.

## Next gate

Keep this fixture as a transport regression. Next, exercise quiet speech, noisy
pauses and words near detected boundaries, then extend paced duration toward
the 30-minute gate. Add a small readable caption consumer without changing the
controller or making report assembly block native event delivery.

The current protocol declares total sample count and source hash in advance.
An open-ended microphone needs an explicit duration/budget and EOF contract.
Reconnection, durable recovery, multiple speakers speaking at once, translation
and multi-channel scheduling remain outside this diagnostic. Continuous speech
without a suitable endpoint can still reach the analysis limit and stop.
