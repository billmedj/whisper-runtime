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

> **Status:** pre-alpha research implementation. The native adapter handles one
> unbatched 30-second mel window per transaction. CPU remains the default. A
> strict single-lane CUDA path is implemented and recorded for one pinned
> `tiny.en` FP32 case on separate AWS and GCP T4 workers. This is not a
> production transcription service.

## What it provides

| Component | Current implementation |
| --- | --- |
| Runtime core | Bounded admission, exact in-process leases, deadlines, versioned commits, cancellation, quarantine, and cleanup recovery |
| Legacy adapter | Runs the existing synchronous `model.transcribe()` call as one serialized transaction |
| Native adapter | Exposes run creation, prefill, token steps, finalization, and cleanup through the patched decoder; CPU and one strict single-lane T4 profile have recorded integration cases |
| Conformance data | Records four pinned JFK CPU comparisons: greedy, beam search, word timestamps, and translation |
| Isolation checks | Exercises two staged decodes under a fixed schedule and in two operating-system threads, cleans one early, and checks the survivor against an isolated baseline |
| Formal model | Proves abstract lease, capacity, lifecycle, stale-commit, completion-fence, quarantine, and release properties in Lean |

The runtime follows one execution path:

```text
request
  -> admit and reserve declared capacity
  -> create a transaction
  -> submit backend stages through a closing gate
  -> seal the gate, clean up, and wait for the completion fence
  -> publish one versioned result, or discard it
  -> release the lease, or quarantine it if quiescence is unproven
```

The declared budget is an admission ledger. It does not enforce operating
system RAM or device-memory limits. The CUDA path has an event-backed completion
fence. The T4 records measure observed allocation, but the declared 1 GB value
remains a configured ledger value rather than an enforced device limit.

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

### Runtime core

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

### Patched Whisper backend

Build the pinned backend and its isolated Python environment with one command:

```sh
python tools/bootstrap_native_backend.py
```

The pinned native dependency set requires CPython 3.12 or 3.13. The runtime
core supports Python 3.10 and later.

The bootstrap verifies the backend commit, source trees, and patch digests. It
uses pip's isolated mode, installs packages only in `.tmp-native/venv`, and
records the complete resolved distribution inventory. It does not download a
model.

Run one real `tiny.en` CPU transaction. The flag gives explicit permission for
the first checkpoint download:

```sh
python tools/run_native_example.py --allow-model-download
```

Later runs use the verified local checkpoint and omit the flag. See the
[real backend quick start](https://github.com/billmedj/whisper-runtime/blob/main/docs/REAL_BACKEND_QUICKSTART.md)
for prerequisites, offline verification, custom paths, and limits.

## Development validation

Run the repository checks:

```sh
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tools -p "test_*.py" -v
python -m ruff check src tests tools examples infra
python -m ruff format --check src tests tools examples infra
python -m mypy src
python -B tools/check_repository.py
python -m build
python -B tools/check_distribution.py dist
```

The optional, paid Modal T4 transaction check has two confirmation guards. Run
its local definition and adversarial record tests first; see the
[native CUDA validation guide](https://github.com/billmedj/whisper-runtime/blob/main/docs/MODAL_NATIVE_CUDA_VALIDATION.md).
Two validated native-adapter records from AWS and GCP are committed under
[`evidence/`](https://github.com/billmedj/whisper-runtime/tree/main/evidence).
The earlier direct-backend records use the frozen
[historical version-one guide](https://github.com/billmedj/whisper-runtime/blob/main/docs/MODAL_GPU_VALIDATION.md).

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

### Staged native decoder

`NativeWhisperAdapter` checks cancellation between prefill, token steps, and
finalization. It requires the pinned backend and seven-patch integration series
under [`patches/openai-whisper`](https://github.com/billmedj/whisper-runtime/tree/main/patches/openai-whisper). The wheel
contains only the runtime package; the source distribution also contains the
patch series. The default profile admits one transaction. An experimental CPU
profile admits two transactions, serializes encoder preparation, and permits
only verified request-local decoder runs to overlap.

An explicit `cuda:N` profile admits one transaction, copies one CPU `float32`
mel tensor after admission, and waits on a CUDA event before commit or resource
release. This code path has unit coverage with a controlled CUDA double. Two
validated records also exercise the pinned `tiny.en` FP32 case on separate Modal
T4 workers in AWS and GCP.

See the [native adapter contract](https://github.com/billmedj/whisper-runtime/blob/main/docs/NATIVE_ADAPTER.md).

## Evidence and limits

| Evidence | Scope |
| --- | --- |
| 105 runtime tests | Resource accounting, queue bounds, commit races, deadlines, cancellation, quarantine, recovery, and adapter behavior |
| 125 repository-tool tests | Provenance, source state, fixtures, portability, setup contracts, evidence schemas, semantic validation, qualification relations, and native smoke contracts |
| 2,000-step deterministic state trace | State-machine transitions under generated operations |
| 51 Lean theorem declarations | Abstract lease provenance, capacity conservation, lifecycle, stale-commit properties, completion-fence publication, quarantine, and recovery |
| One recorded native run | Patched `tiny.en` decoder, JFK fixture, CPU, exact transcript, queue returned to zero, declared budget restored |
| One recorded staged-run isolation check | One loaded `tiny.en` model, two overlapping run lifetimes, early cleanup, unchanged survivor, and successful model reuse |
| One recorded OS-thread isolation check | Two native worker threads, overlapping outer decoder-call intervals, owner-thread cleanup, unchanged survivor, and model reuse |
| One recorded adapter-level concurrency check | Two runtime-admitted transactions, serialized encoder preparation, cooperative cancellation, isolated commit, and exact budget restoration |
| Two recorded Modal T4 readiness checks | Direct patched-backend CUDA decode on separate GCP and AWS workers, exact transcript and model reuse, verified source and model identities, blocked network, and read-only model cache |
| Two recorded native CUDA transaction checks | The same public runtime revision on separate AWS and GCP T4 workers: admission, one private stream, exact transcript, fence-before-publication, cooperative cancellation, retained failure, manual recovery, and native reuse |
| Four conformance pairs | Pinned greedy, beam-search, word-timestamp, and translation reference/candidate records |

The recorded transaction identifies the imported source tree, checkpoint,
loaded model state, audio input, environment, and committed output. The native
CI workflow also runs two staged decodes on one loaded model under a fixed
token-step schedule. It then repeats the decoder isolation case with two native
worker threads after preparing both encoder outputs sequentially. Each thread
enters its first outer decoder call. A barrier in the first decoder block holds
both calls before either continues. The evidence records the start and end of
each outer call and requires the two intervals to overlap.
It also records the explicit decode options. In both backend checks, one run is
cleaned after a decoder step, the survivor must match an isolated baseline, and
a final decode must show that the model remains usable.

`tools/verify_native_runtime_concurrency.py` adds an adapter-level check. Two
caller threads enter `NativeWhisperAdapter.decode_window` with independent
sessions. The worker admits both transactions and reserves both declared
resource vectors. The adapter serializes each `_start_run`, then the first
decoder calls have overlapping recorded lifetimes. The controller cancels one
request after both first token steps. That request does not commit, its lease is
released after cleanup, and the other request commits the isolated-baseline
text. A final adapter call checks reuse and complete budget restoration. Native
CI validates and publishes a fresh record as a 30-day artifact.

The adapter check does not exercise a runtime-owned thread scheduler or
concurrent encoder calls. The resource vectors are declared admission units,
not measured RAM or device memory. Recorded outer decoder-call overlap does not
establish simultaneous PyTorch kernel execution or higher throughput. The
checks cover one pinned CPU configuration and are not a general thread-safety
guarantee.

The version-one T4 records establish direct-backend CUDA execution. The
version-two records additionally exercise the runtime adapter and its
transaction boundary. All four records cover one `tiny.en` FP32 fixture. They
do not establish other models or devices, safe batching, live audio streaming,
durable mid-window resume, portable worker migration, latency, throughput, or
production readiness. The injected synchronization failure is a harness fault,
not a physical driver fault. The Lean completion-boundary model proves an
abstract release and publication protocol. It does not prove Python threads,
PyTorch kernels, the concrete submission gate, CUDA events, or adapter code.

The next fixed CUDA qualification cell is defined by the
[versioned registration](https://github.com/billmedj/whisper-runtime/blob/main/experiments/native-cuda-qualification-v3.json)
and the [qualification evidence contract](https://github.com/billmedj/whisper-runtime/blob/main/docs/CUDA_QUALIFICATION_CONTRACT.md).
The contract includes the exact Modal command, an append-only attempt receipt,
and an offline validator. The cell uses one worker and a small diagnostic
sample. It does not make a latency or throughput claim. The broader
[experiment protocol](https://github.com/billmedj/whisper-runtime/blob/main/docs/EXPERIMENT_PROTOCOL.md)
defines requirements for a future performance campaign, which will require a
new evidence schema or version.

See the [transaction record](https://github.com/billmedj/whisper-runtime/blob/main/evidence/native-cpu-tiny-en-jfk-2026-09-03.json),
[staged-run isolation record](https://github.com/billmedj/whisper-runtime/blob/main/evidence/native-cpu-tiny-en-jfk-interleaving-2026-09-03.json),
[adapter concurrency record](https://github.com/billmedj/whisper-runtime/blob/main/evidence/native-cpu-tiny-en-jfk-runtime-concurrency-2026-09-04.json),
[GCP T4 readiness record](https://github.com/billmedj/whisper-runtime/blob/main/evidence/modal-t4-tiny-en-jfk-cuda-readiness-gcp-2026-09-04.json),
[AWS T4 readiness record](https://github.com/billmedj/whisper-runtime/blob/main/evidence/modal-t4-tiny-en-jfk-cuda-readiness-aws-2026-09-04.json),
[AWS native CUDA transaction record](https://github.com/billmedj/whisper-runtime/blob/main/evidence/modal-t4-tiny-en-jfk-native-cuda-transaction-aws-2026-09-04.json),
[GCP native CUDA transaction record](https://github.com/billmedj/whisper-runtime/blob/main/evidence/modal-t4-tiny-en-jfk-native-cuda-transaction-gcp-2026-09-04.json),
[assurance map](https://github.com/billmedj/whisper-runtime/blob/main/docs/ASSURANCE.md),
[conformance contract](https://github.com/billmedj/whisper-runtime/blob/main/docs/CONFORMANCE.md), and
[development roadmap](https://github.com/billmedj/whisper-runtime/blob/main/docs/ROADMAP.md).

## Repository map

```text
conformance/     Fixture schema, case matrix, and recorded comparison
docs/            Architecture, adapter contracts, and roadmap
evidence/        Versioned integration-run records
experiments/     Versioned experiment registrations
examples/        Minimal runtime-core and native-backend programs
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
[openai/whisper#2843](https://github.com/openai/whisper/pull/2843). Alignment
hook cleanup after inference errors is tracked in
[openai/whisper#2844](https://github.com/openai/whisper/pull/2844). Thread-local
SDPA disable scopes are tracked in
[openai/whisper#2845](https://github.com/openai/whisper/pull/2845). See
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
