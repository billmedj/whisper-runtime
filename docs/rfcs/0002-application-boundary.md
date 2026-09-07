# RFC 0002: application boundary and desktop delivery

Status: proposed. No GUI or application host is installed by the current package.
This design follows V0.1 engine qualification; it is not a release-readiness claim.

## Decision

Keep one engine package. Developers use its Python API or installed command.
A small application host calls the same API and serves packaged browser assets
on loopback. The browser renders application state; it never drives decoder
steps or reads command-line output.

```text
Python SDK ───────────────┐
Installed CLI ────────────┼── Engine and supported execution profile
                         │
Browser UI ── App API v1 ─┘
                  └──────── Existing authenticated remote client, if selected
```

The first GUI is file-first and single-session. A native window wrapper may
reuse the frontend later. It is not required to test the application boundary.

## Existing parts to reuse

- `native_setup.create_stream` checks the local backend, dependencies and model.
- `remote.transcribe_remote` applies the installed client's URL, credential and
  completion policy. Use it with `environment_headers` for server configuration;
  do not bypass it by calling the lower-level `remote_transport` directly.
- `audio_source` provides bounded file and optional microphone input.
- `captions.CommittedTranscript` resolves exact committed revisions and renders
  TXT/SRT/VTT. Keep this projection in Python; do not implement a second version
  in JavaScript.
- Versioned execution profiles separate supported defaults from experiments.

The repository WebSocket server is a one-shot reference application, not a
multi-session desktop host. Do not ship it unchanged as one.

## Small application API

These endpoints are proposals, not current commands. Requests and responses
use explicit versioned schemas, bounded fields and opaque IDs.

| Operation | Contract |
|---|---|
| `GET /api/v1/capabilities` | App API version, engine version, available execution locations, input/export formats, named profiles, limits and feature availability |
| `POST /api/v1/jobs` | One input and execution location; returns a job ID or a specific validation error. Reject a second active job. |
| `GET /api/v1/jobs/{id}` | State, progress, immutable committed captions, current provisional revision, and a stable error code if present |
| `GET /api/v1/jobs/{id}/events` | Ordered application updates, with a revision number and a bounded history. Reconnection can fetch a state snapshot; it never restarts inference. |
| `POST /api/v1/jobs/{id}/cancel` | Requests cooperative cancellation. Show `stopping` until completion fences and cleanup succeed. |
| `GET /api/v1/jobs/{id}/exports/{format}` | Downloads a verified completed result through an opaque job ID, not an arbitrary filesystem path |
| `POST /api/v1/quit` | Stops admission, closes active work, and exits only after cleanup or an explicit failure outcome |

Keep raw `pcm-websocket/live-v2` separate and immutable. The current client
rejects unknown envelope fields; even adding an optional field there can break
compatibility. Capabilities belong to the application API, not injected into
the existing audio transport.

Application states are `ready`, `loading`, `running`, `draining`, `stopping`,
`completed`, `cancelled`, and `failed`. New states or changed meanings require
version negotiation. Unknown capability names may be ignored; required unknown
capabilities must stop setup with a clear error.

## What the user sees

One main screen contains a file picker/drop target, execution location, Start
and Stop, live text, and exports. Put device selection and server setup in
Settings. Show advanced profiles only after explicit selection; no decoder
parameters in the main screen.

Separate provisional text from committed text visually. The interface may
replace provisional text, never committed text. Display `Complete` only after
verified input coverage and cleanup. In remote mode, a FINAL event alone does
not suffice: the terminal DONE receipt must pass its existing checks.

Suggested copy is short and literal: `Choose a file`, `On this computer`,
`Remote server`, `Transcribe`, `Stop`, `Finishing`, `Download transcript`.
An overload message should say what happened and what the user can do; it must
not imply that rejected or incomplete audio was transcribed.

The current engine accepts mono 16 kHz PCM/WAV. The first usable file GUI must
handle ordinary WAV, MP3 and M4A input through one tested conversion adapter;
do not ask users to convert files themselves. Use an existing decoder rather
than implement codecs. Invoke it without a shell, stream bounded 16 kHz PCM,
cap duration and output size, and stop the decoder when the job stops. Keep the
original file unchanged. Record resampling and any explicit mono downmix in job
metadata so caption times and channel treatment remain clear.

This adapter is not implemented yet. Qualify truncated files, decoder failures,
duration limits, channel conversion and timestamp offsets. Video containers and
browser microphone audio need their own cases before they appear as supported
inputs. No automatic remote fallback is allowed when local execution fails.

## Security and lifecycle

Bind only to loopback. Check the exact Host and Origin, require per-launch
authentication before allocating a model, and protect state-changing requests
against CSRF. CORS alone is insufficient for WebSockets. Serve same-origin local
assets with a restrictive content security policy and render transcript text
as text, never HTML.

Remote credentials stay in the Python host and never enter URLs or frontend
storage. Authenticate local launch without permanent URL credentials. Bound
uploads, jobs, retained results and event queues. Closing a tab is not reliable
shutdown; provide explicit Quit and an idle/disconnect policy. A cancelled
Python coroutine does not prove that a native inference thread stopped.

## Installation and upgrades

Developers retain the small SDK/CLI installation with optional dependencies.
End users should receive one launcher that starts the host and opens the UI.
It selects a managed, versioned engine environment; model files are cached
separately and verified. Users should not manage Python environments themselves.

The local ML dependencies are much larger than the interface. A browser UI does
not remove that installation requirement. Remote mode can avoid the local ML
stack, but requires a configured service, network access and explicit consent
to send audio. A remote service is not provisioned by this design.

Do not replace Git/backend/checkpoint verification with an unverified bundled
binary. A relocatable native bundle requires its own versioned verification
manifest and supported-platform tests. Tauri or another native wrapper still
needs platform-specific sidecar packaging, signing and update tests.

Engine upgrades must pass the released frontend's contract tests unchanged.
Keep the previous engine environment until the replacement passes startup and
a small smoke test; preserve user files and model cache during rollback. Do
not upgrade a running session. New GUI-visible features use capability
discovery and a deliberate frontend change; internal optimizations do not.

## Delivery gates

1. Finish the engine release profile, installation and endurance checks.
2. Implement the application host and schema/contract tests using scripted
   results first, then one real local and remote file.
3. Add the bounded input conversion adapter and build the file-first interface
   against the contract. Test loading, progress, corrections, cancellation,
   failed cleanup, completion and exports.
4. Qualify launch, backend setup, update and rollback on each advertised OS.
5. Add microphone capture only with device-loss, overload and physical-device
   checks. Add a native wrapper only for a concrete distribution or OS feature.

References: [WebSocket interface](https://websockets.spec.whatwg.org/#the-websocket-interface),
[WebSocket security](https://www.rfc-editor.org/rfc/rfc6455#section-10.2),
[Tauri sidecars](https://v2.tauri.app/develop/sidecar/).
