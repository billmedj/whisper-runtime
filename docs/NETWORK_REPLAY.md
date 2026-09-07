# PC-to-Modal PCM replay

This reference integration sends recorded or live PCM from a PC and returns
transcript events over one authenticated WebSocket. Both paths use the existing
continuous controller. The transport itself does not capture a microphone,
reconnect a failed session or provide a production service.

The [installed CLI](CLI.md) now connects file input and optional microphone
capture to this protocol. The [registered T4 result](research/2026-09-06-live-v01.md)
completes a short test followed by 30 minutes of source-paced, repeated audio.
Physical capture and broader acoustic/platform qualification remain open.

## Data flow

The client sends a source declaration. The server loads the model on its native
owner thread, then sends `ready`. Only then does the PC start its source clock.
One client task sends audio; another receives provisional text and commits.
Neither waits for per-chunk acknowledgements before doing its next operation.

The server admits PCM from its receive task. Its dedicated owner thread performs
all native steps and cleanup. A bounded mailbox carries results to the socket.
No network handler can authorize publication independently of the controller.

The implementation has two reference modules:

- `examples/replay_websocket.py`: paced replay and live clients, optional `aiohttp` dependency.
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

Prerecorded protocol identifier: `pcm-websocket/v1`. This is an experimental local contract,
not an interoperability standard. Input is mono 16 kHz signed 16-bit
little-endian PCM. The current replay contract requires the final input length
and SHA-256 in advance. Its length, framing and timing semantics are unchanged.

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

## Unknown-length live PCM: `pcm-websocket/live-v2`

The explicit live protocol does not require future audio metadata. It is an
open-length **bounded session**, not an unlimited connection. Authentication,
the native owner, controller publication rules, frame-size limits, output
mailbox, cancellation and cleanup are shared with v1.

1. START contains exactly `{"type":"start","protocol":"pcm-websocket/live-v2"}`.
   Supplying v1's length/hash fields is rejected, not silently ignored.
2. READY echoes the live protocol and includes sample rate, chunk size and
   server `limits`. Source iteration begins only after READY.
3. Binary headers and sequence/span rules are unchanged. Supply full 640-byte
   PCM frames, optionally one short final frame with a whole number of samples.
   A short frame closes the audio-frame sequence: only EOF or CANCEL may follow.
4. On normal source exhaustion, EOF supplies the final `chunks`, `samples` and
   `sha256`. The server compares all three with bytes actually accepted. Empty
   input, mismatch, duplicate EOF or audio after EOF fails without authorizing
   completion. The client independently verifies the returned cumulative
   counts/hash, full committed coverage and exactly one final event.

Use the same header-only authentication as replay:

```python
from examples.replay_websocket import LiveConfig, stream_live


# source is an AsyncIterable[bytes], e.g. an application-owned capture adapter.
# It supplies paced mono 16 kHz signed-16-bit little-endian frames.
async def receive_event(event, client_elapsed_ns):
    await application.publish(event)  # Must stay within callback_timeout_s.


record = await stream_live(
    url,
    source,
    headers=authentication_headers,
    config=LiveConfig(max_samples=30 * 60 * 16_000, total_timeout_s=1_900),
    on_event=receive_event,
)
```

`LiveConfig` defaults to at most 57,600,000 samples (one hour), a 3,700-second
connection-inclusive deadline, 10 seconds awaiting the next source frame,
250 ms per send/callback, 30 seconds draining after EOF, and 100,000 events.
Bounds may be lowered, not disabled or increased beyond those caps. The live
client uses the smaller of its sample cap and the server's advertised cap.
The source supplies capture pacing: live input is not replayed against a
synthetic clock. A blocked send, exhausted limit or stalled source fails; audio
is not silently discarded or automatically retried. The source iterator's
`aclose()` is called when available after consumption begins, including failure
and cancellation. A capture adapter must separately bound its own queue and
surface device overflow; this transport cannot detect audio lost before yield.

Server operators can lower limits with
`make_app(factory, live_limits=LiveLimits(...))`, importing `LiveLimits` from
`examples.pcm_live`. The default sample/session caps match the client, with a
10-second input-idle deadline, 15-second EOF drain and 100,000 native driver
steps. Idle polling does not consume the live native-step budget. The
controller's existing audio-buffer bound remains independent. Source-pinned
servers configured with `expected_samples` or `expected_sha256` reject live
START before native setup; the existing Modal diagnostic remains v1.

Live events are delivered through `on_event`; their history and per-frame send,
admission and native-trace histories are intentionally **not retained** in the
returned diagnostic (`diagnostic_history: "not_retained"`). Live DONE contains
compact cumulative sample/chunk/event/trace counters and the accepted PCM
digest. Output queue/event limits still fail explicitly. This avoids building
an hour-long diagnostic blob; it does not provide durable acknowledgements,
reconnect, resume, resend or exactly-once delivery after disconnect.

Live client timing starts after READY and reports transport observations only;
it has no capture timestamp or word/subtitle latency claim. A one-hour safety
cap is not acoustic or long-session quality qualification.

### CPU and local transport verification

With the checkout's `src` directory on `PYTHONPATH`:

```sh
python -m unittest tools.test_replay_websocket tools.test_pcm_websocket
python -m unittest discover -s tests -p test_pcm_websocket.py
```

These exercise framing, EOF verification, source/session bounds, cancellation,
cleanup and the real controller using a scripted native adapter. When installed,
`aiohttp` and `uvicorn` also exercise an actual loopback-only WebSocket; otherwise
that optional test is skipped. No model, GPU, credentials or external service is
needed. Local protocol success does not qualify a remote deployment.

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
