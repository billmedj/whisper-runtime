# PC-to-Modal PCM replay

This reference integration sends recorded audio from a PC and returns transcript
events over one authenticated WebSocket. It uses the existing continuous
controller. It does not capture a microphone, reconnect a failed session or
provide a production service.

## Data flow

The client sends a source declaration. The server loads the model on its native
owner thread, then sends `ready`. Only then does the PC start its source clock.
One client task sends audio; another receives provisional text and commits.
Neither waits for per-chunk acknowledgements before doing its next operation.

The server admits PCM from its receive task. Its dedicated owner thread performs
all native steps and cleanup. A bounded mailbox carries results to the socket.
No network handler can authorize publication independently of the controller.

The implementation has two reference modules:

- `examples/replay_websocket.py`: paced client, optional `aiohttp` dependency.
- `examples/pcm_websocket.py`: ASGI handler, standard library only beyond the
  runtime. Authentication must be enforced by the hosting ingress.

No new dependency is added to the installed runtime package.

## Run the diagnostic

Use a source checkout with the `network-validation` extra (Modal 1.5.5 and
aiohttp 3.14.3). The Modal account, frozen corpus files and existing cached model
volume must already be configured. This command does not create or download a
model cache. Commit local source changes before a run.

```sh
python -m infra.modal_network_replay --preflight --replay-id network-replay-20260905
python -m infra.modal_network_replay --confirm-paid-gpu --replay-id network-replay-20260905
```

The first command checks an authenticated CPU-only binary echo, including
rejection of an unauthenticated connection. It produces no transcript. The
second requires a successful preflight bound to the same source snapshot.
Existing receipts prevent rerunning either attempt in that namespace. Inspect
the resulting JSON status, not just the process exit code.

The client disables aiohttp's internal connection retry through a checked
version-specific setting. An unsupported transport fails before connecting.
Redirects are rejected before authentication headers can be forwarded.

## Wire format

Protocol identifier: `pcm-websocket/v1`. This is an experimental local contract,
not an interoperability standard. Input is mono 16 kHz signed 16-bit
little-endian PCM. The current replay contract requires the final input length
and SHA-256 in advance. An open-ended microphone API needs a separate contract.

1. The client sends JSON with exactly `type: "start"`, `protocol`, `sample_count`
   and `sha256`. The sample count must be between 1 and 1,920,000 (120 seconds).
2. The server sends `ready` with the protocol, sample rate and chunk size.
3. Each binary message has a 12-byte network-order header: an unsigned 32-bit
   sequence number followed by an unsigned 64-bit starting sample index.
   The payload contains 320 samples, except for the final partial chunk.
   Sequence numbers start at zero; source spans must be contiguous.
4. After all sends complete, the client sends JSON `eof` with `chunks`, `samples`
   and `sha256`. Only exact counts and the digest of accepted bytes allow EOF.
5. Server `event` messages contain unmodified runtime events and a server-local
   observation timestamp. `done` records completion, metrics and diagnostic
   evidence. A successful result requires full committed coverage and exactly
   one native final event. A closed socket alone does not establish success.

JSON controls are limited to 1 KiB; binary input messages to 652 bytes. The
output mailbox holds 64 messages. Event and trace histories stop at 1,000 entries;
an event-limit failure retains the last native batch as well. Each serialized
output is limited to 1 MiB. The controller's PCM bound is separate from socket,
framework and diagnostic-history memory.

## Timing

The PC uses absolute monotonic deadlines. A chunk becomes available only at its
end sample divided by 16,000. Sending too late or blocking too long ends the
replay; the client cannot slow the source clock to accommodate the decoder.
Default maximum source lateness and send wait are each 250 ms.

Client send completion means the client transport accepted the bytes. It is not
proof of remote admission. The server independently records accepted ranges and
their digest. Client receive times measure results arriving back at the PC.
Never subtract monotonic timestamps taken on different machines.

Model setup precedes `ready`. The diagnostic may run offline controls after the
native final event, so `eof` to `done` includes those controls. Use the received
native final event to distinguish transcription completion from report assembly.
Neither measurement includes actual microphone capture or subtitle rendering.

## Failures and limits

Duplicate chunks, wrong spans, changed hashes, overflow and invalid EOF stop the
connection. Cancellation and disconnection do not manufacture source EOF. The
native owner closes its work; cleanup failure cannot be reported as success.
Provider exception messages are not sent to the client.

The client retains received output and send records when a connection fails.
There is no automatic reconnect, resend or inference retry. Server evidence can
retain an event that the client did not receive; delivery is not exactly-once
across a broken connection. There is no durable remote recovery in this example.

Native cancellation is cooperative. A blocked device step or callback cannot be
forcibly stopped by an asyncio deadline. The Modal function's outer timeout and
ephemeral application lifecycle are separate controls.

## Modal authentication and lifetime

The diagnostic uses `requires_proxy_auth=True`. Modal's ingress checks proxy
credentials before invoking the handler. A temporary proxy token is created for
the run, kept in memory and deleted in `finally`. On RBAC workspaces it is
associated only with the selected environment; on other workspaces its scope is
workspace-wide until deletion. Tokens and headers must never enter evidence or
logs. See [Modal proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth)
and [proxy token scope](https://modal.com/docs/guide/rbac#proxy-tokens).

The application is ephemeral. One client connection is opened, without a
health-check request or reconnect. Closing its context ends the application;
the runner records cleanup separately from transcription success. Container
limits restrict concurrent work, not total billing. Modal may reschedule a
crashed container even with application retries disabled. See
[WebSocket behavior](https://modal.com/docs/guide/webhooks#websockets) and
[failure semantics](https://modal.com/docs/guide/retries).

This integration does not close the acoustic or long-session gates in the
[roadmap](ROADMAP.md). Weak speech, noisy pauses, overlapping speakers and long
speech without a usable endpoint still require qualification.
