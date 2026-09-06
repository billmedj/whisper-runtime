# Fixed CPU inventory for the experimental group-anchor diagnostic

This inventory is fixed before scoring the group rule. It selects complete saved
histories by source identity, not by whether their outputs pass. No new model,
audio, tokenizer, GPU, network, retries, or threshold selection is involved.

## Selected histories

| Archive | Selected cell | Complete trace count | Recorded session |
| --- | --- | ---: | --- |
| `modal-t4-tiny-en-word-alignment-v6-2026-09-05.json` | `word-left-context-2000` | 17 | `modal-stream-boundary-diagnostic-v6:word-left-context-2000:session` |
| `modal-t4-tiny-en-word-corpus-v1-2026-09-05.json` | `1995-1837-0024` | 3 | `modal-word-corpus-v1:1995-1837-0024:session` |
| `modal-t4-tiny-en-word-corpus-v1-2026-09-05.json` | `672-122797-0072` | 4 | `modal-word-corpus-v1:672-122797-0072:session` |

Both archives are under `evidence/`. Their fixed SHA-256 identities are:

- Word alignment v6: `ed5565fdcc86fabaad7d66d4122842a0ee99b6533a6d60535c9684c9a3346769`.
- Word corpus v1: `6c27715f06ec85e1c3591c6ef044adee783328d3e8f1be3451975a1a3b6b9fad`.

The first history uses the `openai-whisper-jfk-flac` fixture, repeated three times
to make 33 seconds. Its full PCM hash is
`a3fd62a5bb6caecc25585b9f81bc6f379726c723f5fec6f1557e01508a192372`.
It supplies one source fixture outside the five context-guard tuning states, not
three independent recordings. The archive has no separate numeric speaker ID.
The earlier v4 and v5 records use the same PCM and are not extra independent
inputs; the complete latest v6 history is selected without examining group scores.

The other two histories contain LibriSpeech speakers 1995 and 672, respectively.
Their exact PCM hashes are
`c8161a085b6871c73e775254d8e578451cce924609e08d0778323630db8c7a2b`
and `3bfeb4d05da42bcb00d1875e6dfe66e57ac1068afd957ab1851e74ad202becbc`.
These are the same utterances already included in the tuning mixtures, presented
as standalone inputs. They are related-condition checks, not statistically held-out
speech. Neither repeated windows nor separate sessions imply independent samples.

## Enrollment fixed from recorded state

All 24 traces must appear in the report. Reconstruct the watermark and last
committed anchor solely from exact, unique, ordered COMMIT/publication joins.
Use the recorded four-unit suffix, then the existing retained-audio filter. Do
not substitute words from a reference, choose a convenient matching phrase, or
update state from an experimental diagnostic result.

Seven traces have no previously committed anchor: JFK traces 1–2, speaker 1995
traces 1–2, and speaker 672 traces 1–3. JFK traces 15–17 have only the one-unit
anchor ` country`; retain them as explicit insufficient-anchor exclusions. The
remaining 14 observations comprise JFK traces 3–14 plus speaker 1995 trace 3
and speaker 672 trace 4. They represent eight recorded anchor states, not 14
independent validation cases.

Keep the experimental group rule opt-in and the timestamp tolerance at 200 ms.
Its existing origin-start exception must remain explicit; no new exception is
selected from this inventory. Report both the unchanged strict diagnostic and
the group result for every enrolled observation, with publication authority false.

## Why the broader inventory is not pooled

The corpus-v1 repeated-mixture history contains a historical nonfinal,
punctuation-only publication that the current `AlignedPublication` validator
rejects. It is not silently reinterpreted as a valid modern publication. Its
silence history contains only a one-word committed anchor, and the short speaker
6930 history has no observation after a nonfinal commit. Those complete histories
are not part of this small selected validation set.

The context and paced-feature records contain further related observations, but
reuse the same three source utterances and their transformed mixtures. Their
successful histories also include recorded `wait_for_input` actions without word
alignments, requiring an explicitly different trace path. The paced-replay record
has no raw word alignments. The word-resolution archive contains the original
tuning cases, and the CPU word-resolution replay contains derived diagnostics,
not new observations. These records are not added selectively after scoring.

## Interpretation boundary

This tests an anchor correspondence diagnostic against existing recorded states.
It does not evaluate a complete handoff: these histories do not supply a new
guard/head candidate pair for each state. Even a matched group does not establish
acoustic coverage, omission-free recognition, or publication authority. Preserve
all original archive outcomes. The CPU replay checks byte identity and recorded
joins, not actual PCM, historical package identities, GPU execution, or model
execution provenance anew. Missing historical measurements remain missing.
