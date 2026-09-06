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
