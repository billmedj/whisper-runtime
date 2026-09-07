# Third-party notices

## OpenAI Whisper

The files under `patches/openai-whisper/` are patches against OpenAI Whisper:

- source: <https://github.com/openai/whisper>
- pinned base commit: `86098128c0b4f24f0e2aa2994de830614b474227`
- upstream copyright: Copyright (c) 2022 OpenAI
- upstream license: MIT

The patches contain modified and contextual portions of OpenAI Whisper source.
The MIT license in `patches/openai-whisper/LICENSE` applies to that material.
The patch authors' original changes are provided under the same MIT terms for
the patch series.

OpenAI and Whisper are names of their respective owners. Their use here
identifies the upstream software. It does not imply endorsement.

## JFK audio fixture

The repository identifies, but does not distribute, the `tests/jfk.flac` file
from the pinned OpenAI Whisper source tree. The manifest records its source URL,
size, and digest. Users who fetch the file are responsible for checking the
terms that apply to its source and underlying recording.

## LibriSpeech evidence audio and reference text

Reviewed archives under `evidence/` include selected LibriSpeech recordings,
reference text, and derived test inputs. These data retain their separate
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/);
they are not relicensed under the runtime's software licenses.

Source: [LibriSpeech, OpenSLR resource 12](https://www.openslr.org/12).
Prepared by Vassil Panayotov with assistance from Daniel Povey, from LibriVox
audiobook recordings. Corpus citation: Vassil Panayotov, Guoguo Chen, Daniel
Povey, and Sanjeev Khudanpur, *LibriSpeech: An ASR Corpus Based on Public Domain
Audio Books*, ICASSP 2015, DOI `10.1109/ICASSP.2015.7178964`.

The registered utterance IDs, source URLs, reference text and content hashes
are in `experiments/modal-word-corpus-v1.json`,
`experiments/modal-word-resolution-inputs-v1.json` and
`experiments/draft-holdout-20260907.json`, including frozen copies in archives.
Individual fixtures were decoded from FLAC to mono 16 kHz signed 16-bit
little-endian PCM. Constructed cases may concatenate or repeat recordings,
append digital silence, or mix registered synthetic noise; each archive's
input registration specifies those changes. Test model outputs are not the
independent reference transcripts.

Retain attribution, the license link and modification notices when sharing
these data. No endorsement by the corpus authors, OpenSLR or LibriVox is implied.
