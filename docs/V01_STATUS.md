# V0.1 release status

Updated: 2026-09-07. Package version: `0.1.0a1`. **Developer alpha; stable V0.1 is not released.**

This alpha targets one English audio stream with the existing `tiny.en`
weights. It provides a Python API and an installed command. A desktop GUI,
multiple channels and translation are separate deliverables.

## Available now

- Read mono 16 kHz WAV or raw PCM, optionally paced on its source clock.
- Show provisional text, corrections, committed text and verified completion.
- Export committed TXT, SRT and VTT without overwriting existing files.
- Select local CPU/CUDA inference or an authenticated WebSocket server. The
  remote client does not need PyTorch or a local model.
- Capture one microphone through the optional `sounddevice` dependency.
- Bound input buffering, reject overload, cancel cooperatively and release
  request resources. Logical savepoints remain a separate API.
- Select a versioned local execution profile from the SDK or installed command.

| Local profile | Settings and status |
|---|---|
| `conservative-v1` | Standard settings; no alignment reuse or token drafts. This remains the CLI default. |
| `low-latency-v1` | Earlier word confirmation, retained context and same-window alignment reuse; no token drafts. The first T4 test confirms text earlier but fails on the final source boundary. Not qualified. |
| `low-latency-v2` | Adds right-context holdback to the earlier word observation. Full CPU replay and two source-paced T4 sessions pass after compile-only startup preparation. First confirmed text at 7.358 s / 6.186 s. Endurance and release qualification remain open. |
| `experimental-optimized-v1` | Longer retained context, same-window alignment reuse and up to 32 verified draft tokens. Explicit opt-in; requires the matching patched backend. |

Remote mode does not select or verify the server's execution profile. Configure
the server separately. See [commands and limits](CLI.md).

Microphone capture has scripted tests but no physical-device qualification.
Caption times describe committed audio spans, not precise word timing. Stable
captions with the default profile can wait almost one 30-second window when
speech has no pause. The low-latency candidate removes that scheduling delay;
confirmed text still requires agreement and supported audio. No universal
accuracy or maximum-latency claim is made.

## Local release checks

The two-observation update passes 903 runtime tests (7 skipped) and 86 focused
Modal harness tests. Ruff, strict mypy and repository checks pass. The CUDA
startup change compiles two CPU backtrace signatures before accepting audio;
a separate pinned-Numba CPU test verifies identical paths and no new signatures
for dense and strided int32 traces. It runs no model decode. A fresh development
wheel builds, installs offline and opens console help. These checks do not
qualify source-paced GPU operation or endurance. See the
[two-observation report](research/2026-09-07-two-observation-holdback.md).

The earlier [completion and startup fixes](research/2026-09-07-completion-and-startup-hardening.md)
pass 877 runtime tests (7 skipped), 853 repository-tool tests (5 skipped),
and 66 focused harness tests with the pinned Modal SDK. Ruff, strict mypy and
wheel checks pass. All 32 installed modules match source and wheel bytes; the
installed CPU command produces the expected TXT/SRT/VTT for the cached JFK file.
These checks cover cancellation/deadline races, final microphone-frame admission
and verified native imports. The corrected harness has since passed the separate
[verified-startup replay](research/2026-09-07-verified-startup-results.md).

The historical candidate sweep on 2026-09-07 ran 871 runtime tests with 7 skips
and 815 repository-tool tests with 3 skips, including 14 scripted native-soak
harness tests. Both suites, Ruff, strict mypy, compilation, repository checks and
distribution checks passed at that checkpoint. These counts predate the later
SDK correction and are not a full-suite result for the latest working tree.
The [release-candidate report](research/2026-09-07-release-candidate-checks.md)
preserves its exact source hashes, skips and installed export evidence.

The subsequent SDK and failed-result exit-status corrections passed 16 focused
tests with Modal 1.5.5; the resource-definition regression forbids remote work.
That narrower result does not replace the full suite or qualify native endurance.
See the
[actual soak attempt](research/2026-09-07-release-soak-results.md).

That offline wheel was installed in a fresh Python environment. Console help
worked without Torch, NumPy, Whisper or aiohttp. At that checkpoint all 32
installed source modules matched the wheel and checkout. The installed standard
and optimized commands both transcribed the 11-second JFK fixture, completed its
committed source coverage, and produced byte-identical TXT/SRT/VTT exports.

Those native runs reuse existing verified ML dependencies, backend checkouts
and a cached model. They do not qualify a clean operating-system installation.
Optional loopback tests also pass with host aiohttp 3.13.5; they do not replace
the earlier pinned aiohttp 3.14.3 record. The existing candidate archives predate
later harness and evidence changes. Final source must be frozen, the full suite
rerun, and matching distributions rebuilt before release; the earlier archive
is not a clean committed release revision or approval to publish.

## Recorded native results

| Check | Result and scope |
|---|---|
| Earlier clean Windows Python environment, pinned native dependencies, separate backend copy, cached model | Installed command transcribes the 11-second JFK file outside the source checkout; TXT/SRT/VTT produced. Not a clean operating-system installation. |
| Installed remote client with pinned aiohttp 3.14.3 | Earlier 22-test record includes real localhost exchange, overload, cancellation and invalid terminal receipts. |
| Two difficult cached CPU inputs | Full coverage and FINAL on 33.66-second continuous speech and 10.89-second noisy prefix; reference word edits 6/88 and 1/26 respectively. |
| Accelerated controller soak | Four logical hours, 72,000 chunks, bounded retained history, one FINAL and restored capacity. Recognition is scripted. |
| Real source-paced T4 test | Short test and 30-minute run complete on repeated licensed audio; 90,000 long-stage chunks, 118 commits, verified input coverage and restored capacity. |
| New-audio draft comparison on T4 | Two short project-new speakers; 45.9% fewer decoder calls and 23.5% less decode-phase wall time, with exact text and commit-span agreement. Not an end-to-end latency or general accuracy result. |
| Fresh-process CPU restart | Logical savepoint restored after the producer exits; events, tokens and coverage match the uninterrupted run. Decoder state is recomputed. Not GPU-state migration or crash recovery. |

The [native draft report](research/2026-09-07-draft-qualification-results.md)
links frozen source, raw GPU records and the separate CPU restart audit.
Floating-point score differences can change publication decisions near a
threshold even when tokens match. Drafts therefore remain experimental.

The earlier [three-arm T4 comparison](research/2026-09-07-composed-gpu-results.md)
measures first commit at 10.245 seconds versus 28.439 seconds for the conservative
control and 22.77% lower peak allocated memory on one 46.55-second input. It
does not include drafts. Its costs and the newer draft measurements use different
baselines; do not add their percentages or combine them into one speed claim.

The [scripted soak](../evidence/scripted-controller-soak-4h-2026-09-06.json)
finishes in 95.48 seconds. It does not test four hours of native recognition,
real-time scheduling or GPU memory. The separate
[30-minute native test](research/2026-09-06-live-v01.md) uses a repeated short
speech/noise/silence cycle, not held-out accuracy coverage.

## Gates for stable V0.1

The [native endurance attempt](research/2026-09-07-release-soak-results.md)
stopped after its 46.55-second smoke input. Transcription, source coverage,
the registered word-edit check and cleanup passed, but process RSS was
5,292,425,216 bytes (4.93 GiB), above the registered 3.5 GiB ceiling. None of the
four hourly sessions started. The Modal app is stopped with zero active tasks.
This is a failed endurance attempt, not evidence of a memory leak or a passing
four-hour run. Memory attribution needs a separate short diagnostic.

The subsequent [two-session T4 diagnostic](research/2026-09-07-memory-attribution-results.md)
completes both transcripts with identical exports and restored capacity. Its
accounting result remains incomplete: smaps_rollup is absent. Guest RSS rises
mostly during imports/initial execution, then by 4.55 MiB over the second session;
the terminal CUDA allocation matches. Guest counters do not verify host physical
RAM. The original memory policy and failed result are unchanged. A short check
with a verified capacity limit must precede another endurance attempt.

That [separate 4 GiB capacity smoke](research/2026-09-07-capacity-smoke-results.md)
has now run. It stopped during session 1 on `source_late`, with cleanup confirmed.
No session completed; session 2 and endurance did not start. The app is stopped.
Source scheduling needs investigation before another paid run. The original
RSS failure and release gates remain unchanged.

The [separately authorized instrumented replay](research/2026-09-07-source-clock-replay-results.md)
passes both 46.55-second sessions under the same submitted 4 GiB limit, with
full coverage, identical exports and unchanged terminal CUDA allocation.
Maximum source lateness is 222.77 ms then 2.13 ms, below the unchanged 250 ms
bound. This passes the short capacity check, not cold-start reliability or
endurance; the earlier timing failure remains unexplained. The app is stopped.

The subsequent [verified-startup replay](research/2026-09-07-verified-startup-results.md)
passes two full sessions after those import fixes, with unchanged exports and
terminal CUDA allocation. The first source-delay peak (206.39 ms) overlaps the
first DTW operation; this does not prove its cause. The app is stopped.

The new [low-latency CPU comparison](research/2026-09-07-low-latency-cpu-results.md)
confirms speech after 4 seconds of admitted audio on the noisy fixture and
6 seconds on continuous speech, with unchanged word-edit counts. Same-window
reuse halves encoder forwards and preserves those candidate outputs exactly.
These are accelerated source positions, not live latency measurements. Its
separate source-paced T4 check confirms the first caption at 7.746 seconds, but
fails before completing the transcript: all 46.55 seconds of input were accepted;
only 38.10 seconds were committed. A quiet-boundary anchor timing mismatch raises
`StreamNeedsResolutionError`. No FINAL or verified exports were produced. Cleanup
passed and the app is stopped. This is a failed check, not a release result;
endurance has not started.

The [two-observation correction](research/2026-09-07-two-observation-holdback.md)
now completes that entire input on CPU: one FINAL, no buffered tail, restored
capacity and 7/114 word edits. First nonempty confirmation occurs after 6 seconds
of admitted audio. This is accelerated replay, not a wall-clock latency result.
The standard and earlier experimental profiles remain available unchanged.

The subsequent [v2 T4 check](research/2026-09-07-low-latency-v2-gpu-results.md)
passes both source-paced sessions after bounded backtrace compilation before
readiness. First confirmed text arrives at 7.358 and 6.186 seconds; both sessions
finish with identical exports, 7/114 word edits and restored capacity. The
maximum source delay is 210.950 ms, below the unchanged 250 ms gate. This is the
completed short check, not four-hour endurance or a general latency guarantee.
The app is stopped with zero tasks; the candidate remains experimental.

1. Complete the registered four-hour native endurance check and inspect memory,
   terminal coverage, failures and cleanup. A prepared test is not a result.
2. Test physical capture and clean installation on each advertised platform.
3. Run release CI against the exact source revision. Publish the package, limits
   and corresponding source only after those checks pass.

The alpha is intended for evaluation while these gates remain open. A stable
V0.1 release still requires them. Experimental profiles do not extend the
qualified scope of an individual test.

## Developer and GUI delivery

Keep one engine package. Developers use its Python API and command. The proposed
[application boundary](rfcs/0002-application-boundary.md) adds a small local host
and a browser interface through a versioned API. It reuses transcript projection,
exports and the existing remote client; it does not duplicate decoder logic.

An end-user launcher, managed backend installation, update/rollback checks and
the GUI are not implemented by this release-hardening work. They follow engine
qualification. The [broader roadmap](ROADMAP.md) tracks durable token-state
recovery and later research separately from V0.1 features.
