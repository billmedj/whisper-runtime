# Earlier commits and same-window feature reuse

Experiment registration. CLI defaults are unchanged. This file records the plan
before the matched GPU call; measured results belong in a separate report.

## Question

Can earlier publication and same-window encoder reuse work together without
changing committed text or weakening the existing publication checks?

The previous short test published earlier with the same word-edit count. Its
sum of decoded source durations increased by 21%. That is an audio-overlap
measure, not GPU work: each initial encoder call still receives a padded
3,000-frame input. We need actual forward counts and device-event measurements.

## Matched comparison

One cached `tiny.en` model, one T4 and one owned CUDA stream run three arms:

| Arm | Left context / word context limit | Alignment |
| --- | --- | --- |
| `cli-legacy` | 2 / 6 seconds | Re-encode |
| `fast-legacy` | 20 / 24 seconds | Re-encode |
| `fast-reuse` | 20 / 24 seconds | Reuse the same window's features |

All arms use the same optional backend patch and reuse-capable adapter profile.
The two controls explicitly select the legacy alignment path. No weights,
silence thresholds, word checks or publication rules change between arms.
Each arm receives the same 46.55-second speech/noise/silence sequence on its
source clock. Warmup is recorded separately. There is one run per arm, not a
statistical latency benchmark or a test of audio from a physical microphone.

Expected results, to be checked before accepting the combination:

- The two fast arms produce identical committed text and source spans.
- Reuse removes alignment encoder calls without adding decoder calls.
- Both fast arms retain the earlier first commit relative to the control.
- All arms account for every input sample, emit FINAL and release capacity.

CUDA events bracket actual encoder and decoder forwards on the owned stream.
Intervals are resolved after the native completion fence. They are not GPU
occupancy, energy use or an estimate of total service cost. Instrumentation is
the same in all arms. Allocated memory means PyTorch tensor allocations, not
total device memory.

The run has one paid call, no retry, a 300-second worker timeout and a cleanup
reserve. Its compute planning estimate is about $0.105; this is not a billing
cap and excludes startup, image build, storage and other provider charges.
Weights must already exist in the read-only model cache.

## Implementation boundary

`native_setup.create_stream` now accepts an explicit stream configuration and
an alignment-reuse option. The option requires its exact pinned backend tree;
the default still requires the original tree. No patch or download is implicit.

Feature reuse is valid only for the same audio window and matching execution
identity. A growing or shifted window must be encoded again. The existing
same-origin alignment can survive a text commit and a publication-boundary
checkpoint; it cannot turn a duplicate observation into fresh acoustic evidence.

The offline evidence-reuse check tests this distinction. It is a deterministic
diagnostic replay and scripted checkpoint test, not an implemented cache or a
validated policy for choosing the next inference operation.

## Acceptance boundary

Acceptance requires the CPU gate, matched T4 run and independent receipt audit.
A failed cell or changed word result must remain visible. This short experiment
cannot qualify the combined profile for a release or replace its longer
source-paced test.
