# Local release-candidate checks — 2026-09-07

The candidate remains `0.1.0.dev0`, unpublished and experimental. This is a local
Windows Python 3.13.1 check, not remote CI or qualification of a stable release.
The [portable receipt](../../evidence/release-candidate-local-2026-09-07.json)
records all 32 runtime module hashes, relevant release-tool hashes, exact skips,
local log digests, and installed transcript-export hashes.

## Final local checks

| Check | Result |
|---|---|
| Runtime suite | 871 tests run; 7 skipped; suite passes. |
| Repository-tool suite | 815 tests run; 3 skipped; suite passes, including 14 scripted soak-harness tests. |
| Ruff lint and formatting | Pass; 222 Python files formatted. |
| Strict mypy | Pass; 32 runtime source modules. |
| Compilation, minimal example and installed dependency consistency | Pass. |
| Repository and distribution contracts | Pass locally; generated run artifacts are excluded from published source. |
| Offline wheel/source packaging | Final candidate contains the checked source and explicit allowlist of all nine reviewed evidence archives. The wheel carries runtime code, not those archives. |

One POSIX FIFO check and two optional Modal SDK definition checks were not run
in this environment. Seven optional loopback checks skipped by the base suites
were exercised separately: 12 remote tests and 11 PCM WebSocket tests pass with
host aiohttp 3.13.5 and uvicorn 0.42.0. Those runs do **not** validate the declared
aiohttp 3.14.3 pin or replace earlier pinned-dependency evidence.

## Installed command evidence

A fresh package environment accepted the wheel with no dependency installation.
The real console entrypoint and in-process help both work while Torch, NumPy,
Whisper and aiohttp are absent. Native prerequisites were then reused from an
existing verified environment through an explicit dependency path. This is a
fresh package installation, not a clean operating system or fresh ML setup.

The default `conservative-v1` and explicit `experimental-optimized-v1` commands
both finish the same cached 11-second JFK WAV on CPU. Each displays contiguous
committed coverage from 0 to 11 seconds and a real FINAL, then creates nonempty
TXT/SRT/VTT exports. Corresponding exports are byte-identical. The receipt binds
their SHA-256 hashes and all 32 installed modules to the wheel and checkout.
Startup timings are not presented as a speed comparison.

Only documentation and release tooling changed after those native runs. Their
results are reused only after checking every candidate runtime module against
the recorded hashes; unchanged inference is not rerun to manufacture new evidence.

## Preservation and remaining gates

Before formatting nine diagnostic files, their exact original bytes were saved
in the [pre-format archive](../../evidence/pre-release-format-20260907.zip).
Its SHA-256 is
`3b77111df24d699138224bbfed8aee517c191b3682ce907bc8270362d09e8d5d`.
Original hashes and before/after AST equivalence were verified. Historical
diagnostics therefore retain their executed source independently of formatting.

Final candidate archive hashes and the archive-versus-checkout audit are kept in
the local validation summary, outside the source distribution, to avoid a
self-referential archive digest. The source receipt is portable; raw local logs
and environment paths are not published.

No GPU job, physical microphone test, dependency download, package publication,
version bump or four-hour native soak occurred in these checks. The new soak
harness tests are scripted controls, not endurance results. Clean-platform
installation, physical capture, the registered native endurance run and release
CI against a frozen revision remain open in [V0.1 status](../V01_STATUS.md).
