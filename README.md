# Whisper Execution Runtime

[![CI](https://github.com/billmedj/whisper-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/billmedj/whisper-runtime/actions/workflows/ci.yml)
[![Native integration](https://github.com/billmedj/whisper-runtime/actions/workflows/native-integration.yml/badge.svg)](https://github.com/billmedj/whisper-runtime/actions/workflows/native-integration.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![License: Apache-2.0 and MIT](https://img.shields.io/badge/license-Apache--2.0%20AND%20MIT-5B5B5B)](#license-and-attribution)

Whisper Execution Runtime is an experimental Python runtime for bounded
admission, cooperative cancellation, and controlled cleanup around OpenAI
Whisper inference.

It does not change the model weights or neural-network architecture. It makes
request state, declared resource claims, backend work, and result publication
explicit.

> **Status:** pre-alpha research implementation. The current native adapter is
> CPU-only, handles one unbatched 30-second mel window, and reserves one full
> worker. This repository is not a production transcription service.

## What it provides

| Component | Current implementation |
| --- | --- |
| Runtime core | Bounded admission, exact in-process leases, deadlines, versioned commits, cancellation, quarantine, and cleanup recovery |
| Legacy adapter | Runs the existing synchronous `model.transcribe()` call as one serialized transaction |
| Native adapter | Exposes run creation, prefill, token steps, finalization, and cleanup through a patched CPU decoder |
| Conformance data | Records one pinned `tiny.en` and JFK CPU comparison with source, input, model, and output identities |
| Isolation evidence | Runs two staged decodes on one loaded model, cleans one early, and checks the survivor against an isolated baseline |
| Formal model | Proves lease, capacity, lifecycle, and stale-commit properties within an abstract Lean model |

The runtime follows one execution path:

```text
request
  -> admit and reserve declared capacity
  -> create a transaction
  -> submit backend stages through a closing gate
  -> commit one versioned result, or abort
  -> wait for backend cleanup
  -> release the lease, or quarantine it if cleanup is unproven
```

The declared budget is an admission ledger. It does not enforce operating
system RAM or device-memory limits. Production enforcement requires a backend
adapter that measures its work and provides a completion fence for the target
device.

## Why this boundary exists

Whisper inference is a sequence of mutable operations: feature processing,
encoding, autoregressive decoding, fallback attempts, alignment, and result
assembly. A synchronous function call hides their ownership and lifetime.

This runtime assigns each mutable object to a request, session, transaction, or
worker. A transaction cannot return its lease until the submission gate is
closed and registered backend work has reached a completion fence. A failed
fence keeps the lease quarantined instead of reporting capacity that may still
be in use.

The detailed state and close rules are in
[RFC 0001](https://github.com/billmedj/whisper-runtime/blob/main/docs/rfcs/0001-state-resource-execution.md).

## Quick start

Clone the repository and create an environment:

```sh
git clone https://github.com/billmedj/whisper-runtime.git
cd whisper-runtime
python -m venv .venv
```

Activate the environment:

```text
POSIX:              . .venv/bin/activate
Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

Install the package and validation tools:

```sh
python -m pip install -e ".[validation,quality]" "build>=1.2,<2"
```

Run a minimal transaction. The example uses a synchronous completion fence and
does not load Whisper:

```sh
python examples/minimal_transaction.py
```

The command prints `Example transcript` after it commits session version 1,
returns the worker queue to zero, and restores the declared budget.

## Development validation

Run the repository checks:

```sh
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tools -p "test_*.py" -v
python -m ruff check src tests tools examples
python -m ruff format --check src tests tools examples
python -m mypy src
python -B tools/check_repository.py
python -m build
python -B tools/check_distribution.py dist
```

The Windows check command also compiles the Python sources and builds the Lean
model when the pinned Lean toolchain is available:

```powershell
.\tools\run_checks.ps1
```

To build the formal model directly:

```sh
cd formal/lean
lake build
```

## Adapters

### Historical API

`LegacyWhisperAdapter` places an existing blocking `transcribe()` call behind
the same admission, publication, and cleanup boundary. It is a migration path;
it cannot stop a call between decoder tokens.

See the [legacy adapter contract](https://github.com/billmedj/whisper-runtime/blob/main/docs/LEGACY_ADAPTER.md).

### Staged CPU decoder

`NativeWhisperAdapter` checks cancellation between prefill, token steps, and
finalization. It requires the pinned backend and seven-patch integration series
under [`patches/openai-whisper`](https://github.com/billmedj/whisper-runtime/tree/main/patches/openai-whisper). The wheel
contains only the runtime package; the source distribution also contains the
patch series.

See the [native adapter contract](https://github.com/billmedj/whisper-runtime/blob/main/docs/NATIVE_ADAPTER.md).

## Evidence and limits

| Evidence | Scope |
| --- | --- |
| 89 runtime tests | Resource accounting, queue bounds, commit races, deadlines, cancellation, quarantine, recovery, and adapter behavior |
| 20 repository-tool tests | Provenance, source state, fixtures, portability, evidence schemas, semantic validation, and native smoke contracts |
| 2,000-step deterministic state trace | State-machine transitions under generated operations |
| 35 Lean theorem declarations | Abstract lease provenance, capacity conservation, lifecycle, and stale-commit properties |
| One recorded native run | Patched `tiny.en` decoder, JFK fixture, CPU, exact transcript, queue returned to zero, declared budget restored |
| One conformance pair | Pinned greedy CPU reference and candidate records |

The recorded transaction identifies the imported source tree, checkpoint,
loaded model state, audio input, environment, and committed output. The native
CI workflow also runs two staged decodes on one loaded model under a fixed
token-step schedule. It cleans one run after a decoder step and requires the
survivor to match an isolated baseline. A third decode checks that the model is
still reusable after cleanup.

The interleaving check is a backend lifecycle test. It does not demonstrate
simultaneous kernel execution, multi-threaded same-model safety, or throughput.
The current `NativeWhisperAdapter` still admits one transaction at a time.

These results do not establish CUDA correctness, safe batching, live audio
streaming, durable mid-window resume, portable worker migration, latency,
throughput, or production readiness. The Lean model does not model Python
threads, PyTorch kernels, submission gates, or adapter code.

See the [integration record](https://github.com/billmedj/whisper-runtime/blob/main/evidence/native-cpu-tiny-en-jfk-2026-09-03.json),
[conformance contract](https://github.com/billmedj/whisper-runtime/blob/main/docs/CONFORMANCE.md), and
[development roadmap](https://github.com/billmedj/whisper-runtime/blob/main/docs/ROADMAP.md).

## Repository map

```text
conformance/     Fixture schema, case matrix, and recorded comparison
docs/            Architecture, adapter contracts, and roadmap
evidence/        Versioned integration-run records
examples/        Minimal executable use of the runtime core
formal/lean/     Abstract state, lease, and capacity model
patches/         Reproducible integration patches for the pinned backend
src/             Python reference implementation
tests/           Runtime and deterministic state-machine tests
tools/           Validation, fixture, packaging, and smoke commands
```

## Relationship to OpenAI Whisper

This repository contains the experimental runtime implementation and a
reproducible backend patch series. It does not claim that those patches are
accepted by, or part of, OpenAI Whisper.

Upstream contributions remain small and independent. Request-local cache state
is tracked in [openai/whisper#2842](https://github.com/openai/whisper/pull/2842).
Grouped multi-audio decoding is tracked in
[openai/whisper#2843](https://github.com/openai/whisper/pull/2843). See
[the upstream contribution policy](https://github.com/billmedj/whisper-runtime/blob/main/docs/UPSTREAM.md).

## Contributing and security

Read [CONTRIBUTING.md](https://github.com/billmedj/whisper-runtime/blob/main/CONTRIBUTING.md) before sending a change. Report a
security issue through GitHub private vulnerability reporting as described in
[SECURITY.md](https://github.com/billmedj/whisper-runtime/blob/main/SECURITY.md).

## License and attribution

The repository and source distribution contain original runtime code and
patches derived from OpenAI Whisper. The package license expression is
`Apache-2.0 AND MIT`:

- original repository code is under the [Apache License 2.0](https://github.com/billmedj/whisper-runtime/blob/main/LICENSE);
- files under `patches/openai-whisper/` modify MIT-licensed OpenAI Whisper
  source and retain the applicable [MIT license](https://github.com/billmedj/whisper-runtime/blob/main/patches/openai-whisper/LICENSE).

See [THIRD_PARTY_NOTICES.md](https://github.com/billmedj/whisper-runtime/blob/main/THIRD_PARTY_NOTICES.md) for provenance. This is an
independent project. It is not an OpenAI product and is not endorsed by OpenAI.
