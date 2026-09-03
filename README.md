# Whisper execution runtime

This repository contains an experimental transaction and resource layer for
Whisper-compatible inference. It does not change the recognition model. It
defines how one process admits work, owns mutable state, stops backend work,
publishes a result, and returns reserved capacity.

The current package is a reference implementation. A conservative adapter can
run the historical `model.transcribe()` API as one transaction. An experimental
native CPU adapter can also expose run creation, prefill, individual token
steps, and finalization to the transaction boundary. Both adapters serialize
calls to one model object and reserve one complete worker for each call. This
package is not a production transcription service.

## Implemented model

The Python package defines these ownership boundaries:

- `ModelSnapshot` identifies one immutable model configuration.
- `Worker` owns a bounded admission queue and a resource budget.
- `Session` publishes immutable, versioned state with compare-and-swap commits.
- `RequestState` owns immutable identity, lifecycle, and random state.
- `WindowTransaction` owns one session transition and one exact resource lease.
- `ExecutionScope` represents backend work submitted for that transaction.
- `SubmissionGate` orders backend submission against transaction close.
- `LegacyWhisperAdapter` places one existing synchronous Whisper transcription
  behind the same admission, commit, and recovery boundary.
- `NativeWhisperAdapter` places each CPU decoder stage and token step behind the
  submission gate and checks cancellation between them.

`Budget.acquire()` creates leases. Lease construction is not public. A lease is
bound to its originating budget and ledger entry. The worker creates
transactions and checks the request, session, model, queue slot, and resource
reservation before admission. Each admitted transaction receives a fixed
expiry time.

## Close protocol

A running transaction does not return capacity only because its host callback
has returned. Close follows this order:

1. Seal the `SubmissionGate`. Later submissions fail.
2. Drain callbacks admitted before the seal. Each callback must register its
   backend operation in the transaction's `ExecutionScope` before it returns.
3. For abort, cancellation, or expiry, deliver `request_stop()`. The operation
   must be idempotent because recovery can retry it.
4. Ask the scope for one final aggregate `CompletionFence`. The fence is created
   after the gate drains and must cover all registered backend work.
5. Wait for the fence.
6. Close the gate, select the terminal outcome, and release the lease.

Commit uses the same seal, drain, and final-fence sequence without requesting a
stop. Cancellation, abort, and expiry can replace the pending commit while it
waits for backend quiescence. The runtime checks them again before one atomic
publication of the request state, session state, and random state. Publication
is the revocation boundary; a later stop cannot roll back a committed result.

If the runtime cannot prove backend quiescence, the transaction enters
`QUARANTINED`. Its lease and queue entry remain held. Recovery retries stop and
fence completion before it permits release. A cleanup failure after a terminal
decision also retains the transaction for explicit retry.

## Execution ownership

The thread that starts a transaction is its execution owner. Only that thread
can cross a cooperative checkpoint or commit. Helper threads can submit work
through the gate while it remains open. A supervisor can request stop while the
owner is alive, but it cannot fence or release the owner's work. If the owner
thread exits, `Worker.stop()` can take over the close protocol. The same
takeover rule applies to an orphaned quiescing transaction.

A submission callback cannot synchronously close its own transaction. Such a
close would wait for that callback during drain. A stop requested from the
callback seals the gate and completes at a later safe point. A direct commit
from the callback is rejected.

This is a trusted in-process boundary. Correctness depends on an
`ExecutionScope` adapter that registers all backend work before a submission
callback returns, provides an aggregate completion fence, and implements an
idempotent stop request. One scope object can belong to only one live
transaction. A second transaction cannot start with that object, and the claim
remains until terminal cleanup. The package does not isolate an untrusted
backend or protect against arbitrary memory writes in the host process.

The legacy adapter uses a stricter profile. One model object is bound to one
worker with a queue capacity of one. Each call reserves the worker's complete
resource vector. Adapter objects for that model share the same binding. If
cleanup cannot prove a safe release, the binding remains closed until the
retained transaction is recovered. The adapter cannot stop a blocking
historical `transcribe()` call between decoder tokens; that requires the native
staged backend described in RFC 0001.

The native CPU adapter uses the same full-worker restriction. It creates a
fresh PyTorch generator from a rollback-safe transaction seed, commits one
result for one exact 30-second mel window, and cleans request-local
decode state at the completion fence. The native and legacy adapters cannot
bind the same live model object. PyTorch and Whisper remain optional runtime
dependencies and are not imported with the package.

## Current evidence

The local suite currently contains 89 Python unit tests. They cover resource
accounting, queue bounds, stale commits, cancellation races, deadlines,
submission drain, stop retries, quarantine, cleanup retry, and owner-death
takeover. Adapter tests also cover cross-adapter serialization, fixed resource
profiles, immutable result payloads, retained-result recovery, and provenance
validation. Native adapter tests cover stage submission, token checkpoints,
rollback-safe generator seeds, cooperative cancellation, exception cleanup,
model identity changes, and the single-result commit boundary. One state-machine
test executes a deterministic 2,000-step trace. The native unit tests use
controlled backend doubles so they can force failure and race paths.

A separate local integration smoke test ran the patched decoder and the real
`tiny.en` checkpoint on the JFK fixture. It committed the expected transcript,
returned the worker queue to zero, and restored the complete resource budget.
The source revision and loaded model weights were verified before the result was
reported. The native integration workflow repeats this check in CI.

The Lean model currently contains 35 theorem declarations, including helper
lemmas. It starts from a canonical empty runtime and proves properties for
states produced by its modeled transitions. The proved properties include
active lease uniqueness, runtime ownership, exact prepare provenance, exact
capacity conservation, stale-commit rejection, terminal lease reuse rejection,
and order-independent commits for two independently prepared sessions.

The Lean model does not model Python threads, `ExecutionScope`,
`SubmissionGate`, backend kernels, or adapter behavior. Those properties are
covered by the executable tests, not by the current formal model.

The conformance corpus has one implemented pair: OpenAI Whisper `tiny.en`, JFK
audio, greedy decoding on CPU. The recorded public outputs match. The remaining
matrix is planned; no latency, memory, or throughput benchmark result is
published.

## Run the checks

The runtime package uses only the Python standard library. Repository checks
also validate JSON fixtures and require the optional `validation` extra:

```powershell
python -m pip install -e ".[validation,quality]"
```

Run the Python tests and repository checks from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
python -m ruff check src tests tools
python -m ruff format --check src tests tools
python -m mypy src
python -B tools/check_repository.py
```

On Windows, the repository check also compiles Python files and builds the Lean
model:

```powershell
.\tools\run_checks.ps1
```

To build only the formal model:

```powershell
cd formal/lean
lake build
```

## Repository map

```text
conformance/     Fixture schema, case matrix, and one implemented fixture pair
docs/            Architecture, conformance contract, and roadmap
formal/lean/     Abstract lease, lifecycle, and capacity model
patches/         Reviewable patch series for the tested Whisper backend
src/             Executable reference implementation
tests/           Unit and deterministic state-machine tests
tools/           Repository, fixture, and local check commands
```

The optional bridges are under `src/whisper_runtime/adapters/`. Importing the
package loads no Whisper, PyTorch, or NumPy module. Applications provide the
model object, its identity probe, and a fixed execution profile. See the
[legacy adapter contract](docs/LEGACY_ADAPTER.md) and the
[native CPU adapter contract](docs/NATIVE_ADAPTER.md).

The tested suspendable backend is reproducible from the pinned base and patch
series in [`patches/openai-whisper`](patches/openai-whisper/README.md).

The upstream request-local cache change that motivated this work remains
separate in [openai/whisper#2842](https://github.com/openai/whisper/pull/2842).

## Scope and license

This is an independent project. It is not an OpenAI product and is not endorsed
by OpenAI. The code is available under the Apache License 2.0. The external
audio fixture is identified by a pinned URL and digest and is not bundled.
