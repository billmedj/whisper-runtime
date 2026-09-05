# Input evidence and silence publication

## Punctuation repair replay

The [T4 record](../../evidence/modal-t4-tiny-en-punctuation-repair-2026-09-05.json)
uses the registered five-case corpus and matches source `4e56cd3` across all
21 snapshot files. The three individual clips still complete, with the same
0, 1, and 2 normalized word edits as their offline controls.

The mixture remains unresolved, but the punctuation-only eviction is closed.
Only 3.480 seconds are committed; the retained origin stays at 1.480 seconds.
The runtime accepts 32 of 43.660 seconds before reporting the unresolved
boundary. It does not admit or silently drop the remaining input.

The raw alignments are identical to the original diagnostic through analysis
11. At 22 seconds, the original policy committed a standalone period through
11.340 seconds. The corrected policy declines that publication. Subsequent
matching cannot recover a stable lexical suffix. At 28 seconds, matching text
belongs to a later repeated utterance, not the original committed occurrence.
Increasing timestamp tolerance would not supply the missing observations.

Digital silence still commits `you` in this baseline. This replay therefore
validates the narrow punctuation repair, not complete streaming or acoustic
reliability. The record SHA-256 is
`d87c2be69bdb2015dc296ef93fb30463ece392b64b94c0d1330a57c68edeae54`.

## Opt-in input-evidence policy

`ContinuousStreamConfig(input_evidence=True)` adds a conservative policy before
segment or word selection, including EOF. Existing profiles remain unchanged
when the option is false. The enabled profile ID has the suffix
`+input_evidence/v1`.

The stream records a hash, exact sample count, and digital-zero test from the
admitted PCM before preprocessing. Retry boundaries remain fixed when more
input or EOF arrives. No amplitude threshold classifies quiet input as silence.

| Observation | Decision | Effect |
| --- | --- | --- |
| Every recorded sample is zero | `non_speech` | Commit empty coverage through a typed `SilencePublication`. |
| Nonzero input, lexical text, and native no-speech score below 0.6 | `speech_candidate` | Continue through the existing publication policy. |
| Missing score, score at least 0.6, or no lexical output | `uncertain` | Suppress unsupported text and retain audio; wait for input or report unresolved EOF. |

`speech_candidate` is a model-supported classification, not independent VAD
evidence or a guarantee of correct recognition. Model scores describe the
whole analysis. They cannot certify a gap between words or a silent suffix
after retained speech. The actual selected publication must also contain
lexical text; old words in context cannot authorize an empty nonzero EOF tail.

Silence publication keeps the native result intact, including any hallucinated
text and scores. The emitted transcript is empty. The publication also retains
the PCM observation and exact analysis start sample, so fractional-millisecond
origins do not lose sample accounting when timestamps are projected to integer
milliseconds.

The native adapter commits this result through the existing transaction and
cleanup path. Audio and output events advance only after capacity is released.
Certified-zero context is then released and the lexical anchor is cleared.
The next speech analysis starts a fresh agreement pair. Uncertain observations
advance the analysis schedule but cannot act as positive agreement witnesses.

This version still decodes zero input before publishing empty coverage. It
skips word alignment for that decision; it does not yet skip model execution.
It neither detects general environmental non-speech nor solves the mixed
utterance boundary failure.

## T4 comparison result

The [input-evidence record](../../evidence/modal-t4-tiny-en-input-evidence-2026-09-05.json)
uses the same five inputs, tiny.en, FP32, seed 7, T4, and offline controls as the
punctuation replay. The implementation is recorded at `0a6a8e4`.

| Case | Punctuation-repair baseline | Input evidence |
| --- | --- | --- |
| Three individual speech clips | Complete; 0, 1, 2 reference word edits | Complete; identical transcript strings and edit counts |
| Repeated voices with pauses | Unresolved; 3.480 s committed | Unresolved; 3.480 s committed |
| 32 s digital silence | Publishes `you`, then stops unresolved | Complete; empty transcript and all samples accounted for |

Silence uses 16 decodes with no word-alignment calls, versus 17 decodes with
alignment in the baseline. Summed source length submitted to decoding falls
from 199.980 to 32.000 seconds because resolved silence no longer accumulates
in growing windows. This is not a proportional reduction in GPU operations:
Whisper still pads encoder input. The measured silence loop is 1.443 seconds
in the baseline and 0.681 seconds in the candidate, excluding warmup, loading,
and offline controls. One unpaced pair is not a performance benchmark or live
latency qualification.

Every completed case passes its lifecycle/accounting checks. The mixture has
the expected open completion checks and retains unresolved audio. The final
model fingerprint is unchanged. The candidate JSON SHA-256 is
`39f2430774cb4763ebca601c7c085ccbdf6aa2d2c85da6f5a76cfea3d84fb4b2`.

A separate read-only audit reconstructed all 41 observed PCM slices and
verified their hashes, counts, and zero flags. All 22 source files match the
recorded snapshot. Each of the 16 silence publications retains the native
`you` hypothesis as provenance while emitting no words.

## Validation and reproduction

445 runtime tests and 284 repository-tool tests pass. Strict type, lint, format,
and repository checks also pass. The tool suite uses the validation extra;
running it in an interpreter without JSON Schema support fails its prerequisites.
Wheel and source archives were not rebuilt in this iteration: the inspected
environment lacked packaging tools, and dependency acquisition was stopped
when it made no progress. The recorded native run used the hashed source tree.

Local tests cover exact zeros, nonzero amplitudes of one PCM unit, seven-sample
tails, fractional origins, selected nonlexical EOF suffixes, coalescing, input
retry, silence-to-speech transitions, and pre/postcommit cleanup failures.
They do not qualify weak real voices or background noise.

The corpus harness registers a separate input-evidence profile. Use a fresh
named replay for both phases; existing evidence is never replaced. These
commands require Modal SDK 1.5.5, the registered local PCM assets, and
`WHISPER_MODAL_ENABLE_WORD_CORPUS=1`:

```sh
python -m modal run infra/modal_word_corpus.py --transport-preflight-only --replay-id input-evidence-20260905
python -m modal run infra/modal_word_corpus.py --confirm-paid-gpu --input-evidence --replay-id input-evidence-20260905
```

On Windows, set `PYTHONUTF8=1` before invoking the CLI. A named replay admits
one synchronous GPU call with a 180-second execution timeout, no automatic
retries, and one container. This is an execution bound, not a billing cap.
The selected policy and source hashes are bound in its receipt and result.

This iteration ran two GPU functions and two CPU transport probes. All related
apps were verified stopped with zero tasks afterward. An earlier CLI Unicode
display error created an app but no function receipt or worker task; that app
was explicitly stopped before continuing with UTF-8 enabled. No checkpoint or
corpus download was needed. Current remaining credit was not measured.

Remaining acceptance work: complete the mixed input without omissions or
duplicates; test quiet speech and natural non-speech; measure paced latency;
and compare compute under matched conditions. Explicit source-based utterance
boundaries are the next diagnostic candidate. They must not be inferred from
the unstable punctuation timestamps that caused the original failure.
