# Independent speech references expose publication failures

Date: 2026-09-05. GPU-tested source: `f9dbf6d`.

## Result

Three short LibriSpeech utterances complete. A constructed sequence with pauses
and repeated speakers, and a digital-silence case, remain unresolved. All five
cases release runtime capacity and pass the registered lifecycle checks. Those
checks do not establish recognition accuracy or justify audio eviction.

| Input | Source duration | Committed coverage | Pre-EOF commits | Outcome |
| --- | ---: | ---: | ---: | --- |
| `6930-75918-0000` | 3.505 s | 3.505 s | 0 | Completed |
| `1995-1837-0024` | 5.385 s | 5.385 s | 1 | Completed |
| `672-122797-0072` | 7.940 s | 7.940 s | 1 | Completed |
| Three speakers, repeated with pauses | 43.660 s | 11.340 s | 2 | Unresolved before EOF |
| Digital silence | 32.000 s | 16.860 s | 1 | Unresolved at EOF |

The three individual transcripts have 0, 1, and 2 normalized word edits against
their independent corpus references. The offline controls have the same edit
counts. Normalization uses case-folding and Python `\w+`; for example, it treats
`gardener's` and `gardeners` differently. Exact punctuation differs in one case.
These 44 reference words are a diagnostic sample, not a recognition benchmark.

The constructed input repeats the three utterances twice, with five two-second
zero-filled gaps. It is not natural conversation. The runtime accepts 40 of
43.66 seconds before stopping. The remaining input is not admitted or silently
dropped. Its transcript is partial; its 80 edits against the full 88-word
reference include unpublished speech and must not be presented as a standalone
recognition error estimate. The full offline control has six edits.

## Why the mixed input fails

The first commit ends at 3.480 seconds. At the 22-second analysis, the only newly
agreed unit is `.`. The model assigns that punctuation a span of 3.680-11.340
seconds. The publication policy accepts it, advances processed coverage to
11.340 seconds, and then moves the retained origin to 9.340 seconds.

The second utterance occupies 5.505-10.890 seconds. The runtime therefore evicts
part of its audio without having transcribed it. The numeric accounting checks
still pass: they count declared processed coverage, not preserved meaning.
Subsequent analyses cannot recover the punctuation-only anchor and stop.

This is a policy defect. Text agreement and an estimated punctuation timestamp
are insufficient authority to advance the audio boundary. A local regression
reconstructs the two recorded alignments; the correction is described below.

There is relevant precedent in Whisper itself: `timing.add_word_timestamps`
merges punctuation with neighboring words and adjusts some long durations.
This adapter deliberately freezes the raw `find_alignment` output, before
those presentation heuristics. The integration error was to let every raw
unit act as an audio-progress endpoint. The guard below preserves that raw
evidence instead of silently changing its times. It does not claim a new
punctuation-alignment method.

## Silence is a separate failure

The stream commits `you` over 0-16.860 seconds of zero-valued PCM. The offline
control emits `you you`. Neither output describes speech in the input.
`no_speech_prob` is about 0.948 in the committed native result, but its average
log probability is about -0.700. The offline control's configured confidence
exception can retain that text. Copying that threshold rule alone would not
fix this case. Word agreement is not an acoustic speech detector.

Silence needs an explicit input-evidence and coverage rule. It must preserve
quiet speech and partial-word boundaries, keep rejected audio recoverable, and
avoid treating a high model score as proof. The current test does not implement
or validate that rule. Do not launch long-session qualification yet.

## Local correction and scope

The post-run repair at `6aa6217` restricts a nonfinal publication to an endpoint containing
a Unicode letter or number. Trailing standalone punctuation waits for a later
stable lexical unit or EOF. Internal punctuation and all native text, tokens,
and alignment estimates remain intact. A punctuation-only retained anchor
cannot authorize further nonfinal progress.

This closes the recorded punctuation-only advancement path. It does not prove
that the complete mixed input now finishes, that lexical hallucinations are
removed, or that every estimated word boundary is correct. EOF authority is
unchanged. The repair has local regression coverage; this GPU record predates it.

The same iteration fixes two experiment defects: millisecond state must match
the floor of the exact sample watermark, and remote Modal asset paths must use
POSIX separators on Windows. Neither repair changes model weights or precision.

After the repair, all 393 runtime tests and 276 repository-tool tests pass.
Five new policy tests cover the recorded failure, nonlexical suffixes, Unicode
lexical text, retained anchors, and unchanged EOF behavior. Type checks, lint,
format checks, and repository checks also pass. These local checks do not
replace a new native-model replay of the repaired policy.

## Reproduction and evidence

The [manifest](../../experiments/modal-word-corpus-v1.json) fixes all PCM hashes,
sample counts, references, construction steps, options, and resource limits.
It attributes the corpus to Panayotov et al., LibriSpeech (ICASSP 2015), under
CC BY 4.0. The download endpoint is not revision-pinned: ID, transcript, and
content hash must match before a file is accepted. Signed URLs are not stored.
No corpus audio is committed to this repository.

With FFmpeg and FFprobe on PATH, run:

```shell
python tools/prepare_speech_corpus.py
```

This command is offline by default and never replaces an existing mismatched
file. Add `--download` to fetch missing registered inputs. The three original
FLACs total 326,353 bytes. Use the Modal SDK version registered in the manifest;
set `WHISPER_MODAL_ENABLE_WORD_CORPUS=1` before these explicit remote commands:

```shell
python -m modal run infra/modal_word_corpus.py --transport-preflight-only
python -m modal run infra/modal_word_corpus.py --confirm-paid-gpu
```

The harness refuses a second attempt in the same artifact directory. These
commands execute paid remote work. A new comparison needs a separate registered
attempt; do not erase old receipts to force a rerun. Use source `f9dbf6d` to
inspect the tested implementation; current source includes the later repair.

- [Raw GPU record](../../evidence/modal-t4-tiny-en-word-corpus-v1-2026-09-05.json).
- [GPU receipt](../../evidence/modal-t4-tiny-en-word-corpus-v1-2026-09-05.attempt.jsonl).
- [CPU transport receipt](../../evidence/modal-word-corpus-v1-transport-2026-09-05.attempt.jsonl).

The record is 516,686 bytes with SHA-256
`6c27715f06ec85e1c3591c6ef044adee783328d3e8f1be3451975a1a3b6b9fad`.
Its 21-file source snapshot digest is
`ef027e2838942c13e74049aaf0c9c25face37a1a5275e10329251d7b48a78d4d`.
The original compressed result is retained locally before decoding: 15,485
bytes, SHA-256
`9c4df33a7c3450f3715c4d39f2ef0e1bcf809363d657f82846df2aa7a3c9e28e`.

A separate read-only review verified the receipts, reconstructed inputs, all
21 source blobs against `f9dbf6d`, and event/accounting results. The record does
not itself name that commit; the association follows from matching its source
hashes. The review found no private user paths, signed URLs, or common secret
formats in the published evidence and manifest.

## Execution limits and cost

One CPU transport call and one T4 call ran. The GPU timeout was 180 seconds;
automatic retries were disabled. Both apps stopped with zero tasks. The initial
Windows path error occurred locally before app creation or a function call.

Modal's ephemeral-app meter changed from USD 0.11223549 to USD 0.11921395, an
observed increase of USD 0.00697846. The monthly total changed from USD
0.11021114 to USD 0.12021114; billed cost remained zero. These counters can lag
and are not an exact per-run invoice or a remaining-credit balance.

Replay was unpaced. The 7.754 seconds summed across stream loops exclude warmup
and offline controls. They are not live subtitle latency, a stock-Whisper speed
comparison, or evidence of reduced GPU use. D1, D3, and D4 remain open.
