# Same-window alignment features

## Registered question

Can word alignment borrow the encoder output already produced by the exact
same native decode, without changing its text, tokens or word times?

Legacy alignment encodes mel again with SDPA disabled. Ordinary decoding can
use SDPA. Equivalent input does not imply bit-identical encoded features.
This comparison measures that difference instead of assuming interchangeability.

## Implementation

An explicit `reuse_alignment_features=True` native execution profile enables
the optional backend handoff. It captures features only from its own finalized
result. The same window can request legacy alignment for a control comparison.
A disabled profile cannot enable reuse per call. An already prepared alignment
cannot change modes. The default profile remains unchanged.

The tensor stays private and borrowed until the transaction's completion fence
succeeds. Failed cleanup keeps it available for exact recovery. Opt-in runs
drop their backend handle after capacity release, including the finalized
feature reference. There is no global cache or reuse across changed windows.
If a caller recovers a failed transaction directly through `Worker.recover`,
it must then close or stop any retained native handle to drop those references.
Recovery does not notify the handle.

## T4 comparison

`infra/modal_alignment_features.py` compares eight fixed inputs in sixteen
native runs: the two recorded failure windows and their two head-only variants,
two previously used full utterances, two seconds of digital silence, and one
utterance attenuated by integer division by 32. No input is newly held out.
Source spans and PCM hashes are recorded; reference text does not select modes.

One T4 worker uses the existing cached FP32 `tiny.en` checkpoint. Its disposable
Whisper checkout receives the optional patch before import. The original tree,
patch digest and patched tree must match the recorded identities. The checkpoint
volume stays read-only. The driver uses one model, lane, host thread and fixed
decode configuration. Arm order alternates by input; source state is fresh for
each run. Both arms use the opt-in profile's ownership rules, while the control
explicitly requests the unchanged legacy alignment calculation.

Record native outputs and complete aligned words separately. Exact word-time
equality is the parity criterion; no timing tolerance is fitted afterward.
For nonempty text, expect two encoder forwards in the control and one with
reuse. Empty output can skip alignment in both arms. Count each phase.

Encoder hooks copy observations to CPU. They record exact feature equality,
maximum absolute difference, RMS difference and a descriptive allclose check
(absolute tolerance 1e-6, relative tolerance 1e-5). These numerical summaries
do not authorize publication or replace exact output comparisons. Drop native
handles and diagnostic tensor references before interpreting GPU allocations.

The run is limited to sixteen windows, one paid function, one container, a
180-second function timeout and no automatic retry. The shared CPU transport
check precedes GPU work. These limits are not an invoice cap. Elapsed phase
times include instrumentation and cold-start effects, exclude preprocessing
and collection, and do not establish a general service speedup.

All failures and differences remain in the record. No runtime default changes
based on these eight inputs. Full-stream recovery, other models/devices and
long-session behavior require separate evidence.

## T4 result

The [stored record](../../evidence/modal-t4-tiny-en-alignment-features-2026-09-06.json)
contains all sixteen runs at runtime commit
`27ee020817c3b8f614b535cbf0fdf5bdca058f5b`. Its SHA-256 is
`85dde1179ae00c0ba9153f3ef52b51f22115081352c1409a0920a964c21d9782`.

| Check | Observed result |
| --- | --- |
| Paired inputs | 8 of 8 completed |
| Native text, tokens and metadata | Exact equality, excluding window ID |
| Aligned words, tokens and times | Exact equality |
| Encoder forwards | 16 control; 8 reuse |
| Encoder input length | 3,000 mel frames for every forward |
| Model state | Initial and final fingerprints match |
| Cleanup | All 16 runs closed and restored capacity |
| Post-handle allocated GPU memory | 160,720,896 bytes after every run |

Actual decode features match exactly between paired runs. They differ from
legacy alignment features on all eight inputs: maximum absolute differences
range from 2.924e-5 to 8.922e-4. All eight fail the descriptive allclose check.
The exact word outputs nevertheless match on this set. Keep this distinction:
output parity here does not prove numerical equivalence or universal parity.

Both paths produce `you` on digital silence. This is a native hallucination,
not a successful silence transcript. The experiment bypasses stream publication
and produces no commits. Feature reuse preserves this error; it does not replace
the input-evidence policy.

The measured native phases total 7.506 seconds across sixteen calls. This is
not billed GPU duration. Per-cell timing varies with startup and shape-specific
work. For example, reuse alignment takes 199.788 ms on the noisy alternative
where its control takes 23.329 ms. The first control takes 3,340.140 ms for
alignment. Retain both observations; do not turn these single comparisons into
a throughput multiplier. The established reduction is one encoder forward
per aligned analysis, not a halving of total inference cost.

Reserved allocator memory rises from 412 to 416 MiB as input shapes change;
allocated memory after release stays constant. This short test does not cover
external cleanup failures, long sessions or other devices/models.

Modal application `ap-ZFqgWMomF5q6jK2UigBSDj` stopped at
2026-09-06 09:23:33 UTC+08 with zero tasks. One GPU function was called; no retry
or model download was needed. The record is replayed by a repository-tool test
without a GPU or audio download.

Post-run review found that an already closed handle skipped reference cleanup
after an external `Worker.recover`. A later local fix makes repeated `close()`
release those references once capacity is restored, without repeating the
backend fence. Scripted cleanup, lease-release and CUDA-fence failure tests
cover this correction. The T4 record remains bound to the preceding commit;
it did not test this failure path.

## Next acceptance gate

Keep reuse opt-in and preserve the legacy path. Compare both on the same paced
stream, with identical publication decisions and input boundaries. Include
weak speech and noisy pauses. Measure completed audio, word errors, caption
delay and device time together; unchanged fixed-window results alone do not
establish a live-service benefit.
