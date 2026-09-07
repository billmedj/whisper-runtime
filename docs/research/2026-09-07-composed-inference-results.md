# Composed inference: CPU result

The earlier-commit configuration and same-window encoder reuse work together
on the two tested CPU inputs. Text, commit boundaries, decision traces and
stream metrics match the legacy-alignment control exactly. Fewer encoder
forwards are measured. The subsequent [matched T4 test](2026-09-07-composed-gpu-results.md)
now measures the combined profile; this page preserves the CPU result.

## Comparison

Both arms use cached `tiny.en`, FP32, seed 7, 20-second left context and a
24-second word-context limit. Both use the same pinned optional backend and
reuse-capable adapter. The control explicitly requests legacy alignment.
Input admission is accelerated, not source-paced.

| Input | Encoder forwards, control / reuse | Decoder forwards, both | Word edits, both |
| --- | --- | --- | --- |
| 10.89-second noisy prefix | 11 / 7 | 153 | 1 / 26 |
| 33.66-second continuous speech | 31 / 17 | 1,009 | 6 / 88 |

Each initial encoder input has 3,000 frames. Encoder input-frame totals fall
from 33,000 to 21,000 and from 93,000 to 51,000. Decoder input-frame and token
totals are unchanged. These counters describe actual forward inputs, not unique
audio samples or device execution time.

Both arms first commit after 10 seconds of admitted source audio. The preserved
earlier-commit behavior is not a fresh measurement of live wall-clock latency.
Each successful cell emits one FINAL, accounts for all input, empties its
buffer, closes all native windows, restores capacity and removes its hooks.

The [comparison record](../../evidence/composed-cpu-2026-09-07.json) preserves
the four successful reports, exact parity checks and hashes of the original
outputs. Published reports omit local absolute manifest paths. The backend tree
is `32163d5cdb87babc1cd415a86cc5a58116c86a16`.

## Failure retained

The first reuse run completed the noisy case, then exceeded the continuous
case's 80-second limit. That original harness did not emit the timed-out cell;
we do not infer its progress or completion. The harness now records partial
timeouts and supports bounded case selection.

A single retry of the continuous case completed in 49.11 seconds under a
140-second cell limit and a 240-second parent deadline. The control took
65.97 seconds; the noisy reuse cell was slower than its control. This variability
does not establish a CPU elapsed-speed improvement. The original failure and
successful retry remain separate local files. These CPU runs did not freeze
their original harness source, unlike the registered GPU experiment.

## Validity checks

The [offline diagnostic](../../evidence/evidence-reuse-falsification-2026-09-06.json)
starts with a passing synthetic correspondence and changes one dependency at a
time. Changes to PCM, model identity, decoding options and window origin each
introduce the expected refusal. Repeating an archived diagnostic returns the
same result without adding an acoustic observation.

Existing scripted tests cover same-origin alignment retention, checkpoint
restoration without a new decode, and a fresh comparison after moving the
window. This is not a production deduplication cache or a learned policy for
choosing the next inference operation.

Final local regression checks: 807 core tests run with one Windows-inapplicable
skip; 727 repository-tool tests run with three optional skips. Ruff passes and
strict mypy passes for all 30 source modules. Full tool discovery exposed a
fixture-import collision; scoped imports fix it and a regression test checks
module/path restoration. The regenerated diagnostic JSON matches its archive.

## GPU test and release boundary

The [registered three-arm T4 experiment](2026-09-06-composed-inference.md) is
implemented and locally checked. Its frozen 54-file source digest is
`3ce3b09e733b85989f039648b76371769124d31bac7d03cfd0c0937785d83799`.
The initial launch was blocked before process creation pending explicit upload
and spend approval. The user then authorized it, and the
[T4 comparison completed](2026-09-07-composed-gpu-results.md).

The CPU result establishes fewer encoder forwards with exact tested output
parity; it makes no GPU timing claim by itself. Read the separate GPU result for
those measurements and the remaining decoding cost. CLI defaults are unchanged;
the combined profile still needs its longer source-paced qualification.
