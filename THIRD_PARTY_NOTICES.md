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
