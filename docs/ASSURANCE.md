# Assurance map

This document separates the properties that the repository proves, tests, or
observes. It also states where each property stops. A passing check in one
layer is not evidence for an untested layer.

## Evidence classes

| Class | Meaning |
| --- | --- |
| Formal | A theorem holds for the abstract transition system in `formal/lean`. |
| Executable | A deterministic test checks the Python implementation. |
| Integration | A recorded run checks the pinned Whisper source, model, input, and environment. |

The Lean model is not a proof of the Python source or PyTorch execution. The
integration records are not general correctness or performance claims.

## Claim matrix

| Property | Formal evidence | Executable evidence | Integration evidence | Boundary |
| --- | --- | --- | --- | --- |
| An accepted reservation fits the available budget. | `accepted_prepare_fits`, `prepare_preserves_ledger_fit`, `valid_reservations_within_capacity` | Budget and worker admission tests in `tests/test_runtime.py`; generated lifecycle trace in `tests/test_state_machine.py` | Native transaction records require the declared budget to return to its full capacity. | The budget is a ledger. It does not measure or enforce operating-system or device memory. |
| Active lease identifiers are unique and owned by the issuing runtime. | `valid_active_ids_unique`, `valid_active_leases_owned` | Lease identity, construction-failure, and cross-worker ownership tests in `tests/test_runtime.py` | Not observed directly. | Process crashes and distributed ownership are outside the model. |
| A terminal lease cannot run or resolve a second time. | `valid_terminal_id_rejects_all_transitions`, `successful_commit_is_single_use`, `successful_abort_is_single_use`, `second_abort_rejected` | Repeated commit, abort, cancellation, and cleanup tests in `tests/test_runtime.py` | Native records require one terminal outcome and one clean release. | Durable recovery after process loss is not implemented. |
| A stale session revision cannot publish a result. | `stale_commit_rejected` | Stale-commit and commit-versus-cancel race tests in `tests/test_runtime.py` | The native smoke record binds the committed session version. | The Lean theorem models revisions, not storage-system transactions. |
| Independent session commits commute in the abstract model. | `active_pair_commits_commute`, `independently_prepared_commits_commute` | Independent-session concurrency test in `tests/test_runtime.py` | Not yet recorded through two concurrent `NativeWhisperAdapter` transactions. | Same-session transactions intentionally conflict on a stale revision. |
| A transaction does not release capacity before registered backend work reaches its completion fence. | Not modeled. | Submission-gate, fence, cancellation, quarantine, and recovery tests in `tests/test_runtime.py` | The single native transaction checks clean terminal state and restored capacity. | Device-specific asynchronous completion still needs a measured adapter fence. |
| Failure to prove backend quiescence retains the reservation. | Not modeled. | Persistent-fence and quarantine-recovery tests in `tests/test_runtime.py` and adapter tests | Not forced in a real PyTorch run. | Recovery is in-process and backend-specific. |
| Two patched decoder runs on one loaded model keep request cache state separate in the recorded CPU case. | Not modeled. | Backend contract tests and evidence validators in `tools/test_contract_tools.py` | Staged and operating-system-thread records compare the surviving run with an isolated baseline and verify model reuse. | One pinned `tiny.en` greedy CPU case. No CUDA, extension, word-timestamp, or throughput claim. |
| Two outer decoder calls can have overlapping lifetimes in two operating-system threads in the recorded case. | Not modeled. | The verifier requires two owner threads, controlled rendezvous, interval overlap, and cleanup ownership. | `native-cpu-tiny-en-jfk-threaded-2026-09-04.json` | Overlapping Python call lifetimes do not prove simultaneous kernels or improved throughput. |
| The experimental adapter profile admits two request-local decoder runs while serializing run construction and encoder preparation. | Not modeled. | Controlled adapter tests in `tests/test_native_adapter.py` require exact queue and budget capacity, overlap two fake decoder runs, isolate cancellation, and retain independent cleanup failures. | One pinned Windows `tiny.en` greedy CPU record checks two admitted adapter transactions, cancellation, one isolated survivor commit, and restored ledger capacity. | The caller threads belong to the verifier. The record does not establish runtime scheduling, PyTorch kernel overlap, throughput, or behavior beyond its stated configuration. |

## Properties not yet established

The repository does not yet establish:

- two concurrent successful `NativeWhisperAdapter` commits with the real
  Whisper backend;
- runtime-owned scheduling of concurrent native transactions;
- safe concurrent encoder calls;
- CUDA correctness or device-memory enforcement;
- batching, fairness, or bounded streaming;
- a throughput or latency improvement;
- equivalence across model sizes, languages, decode modes, or third-party
  extensions;
- correspondence between the Lean transition system and every Python
  execution path.

Each new public claim must add its acceptance test to this map and state the
smallest configuration for which the claim holds.

## How to verify the current evidence

Run the Python and repository checks from the repository root:

```sh
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tools -p "test_*.py" -v
python -B tools/check_repository.py
```

Build the formal model separately:

```sh
cd formal/lean
lake build
```

The native integration workflow rebuilds the pinned Whisper source, applies
the checked patch series, runs the real-model checks, validates each JSON
record, and publishes the records as workflow artifacts.
