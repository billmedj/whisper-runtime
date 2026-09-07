# Draft T4 test: completed

The user approved the frozen payload and one bounded GPU invocation. The
test completed, passed its registered comparison, and the app stopped with
zero tasks. See the [result and limits](2026-09-07-draft-gpu-results.md).
No further GPU invocation is authorized by this record.

## Preparation history

The local preflight and Modal SDK resource construction passed. Sixteen focused
tests and Ruff passed. The independent CUDA review found no remaining launch
blocker in the final implementation.

The first authorization review rejected the launch before process creation. Its
reason was that the new 59-file source upload and paid GPU invocation needed
explicit approval for this payload; the earlier 54-file approval did not cover
it. That rejected attempt started no app or GPU call and created no attempt
journal. The user then explicitly approved the expanded payload.

The next local launch stopped at Git's repository ownership check, before the
attempt journal or any Modal call. An exact-path `safe.directory` setting was
applied only to the test process. No global Git configuration was changed.
The following launch made the single authorized GPU invocation.

Frozen preflight:
`artifacts/modal/draft-t4-20260907/draft-preflight.json`.
Source: 59 files, 1,208,823 bytes, digest
`98e9629766c54ab48015ae4a473c6140d6cd53238c9d3a4a0a822c110aff6741`.
The allowlist excludes credentials, personal directories and research papers.
The existing public audio fixtures are uploaded separately.

Requested execution: one T4 invocation, 240-second worker limit, no automatic
retry. Planning compute estimate: $0.083614, excluding startup, image work,
storage, network, taxes and provider rescheduling. This is not a billing cap.

The frozen hashes were rechecked before execution and after archiving. The
producer and registration were not changed after approval.
The earlier `draft-features-20260907-v1` directory contains only a superseded
local preflight, not a failed GPU attempt.
