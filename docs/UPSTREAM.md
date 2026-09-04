# Upstream contribution policy

Whisper Execution Runtime is independent of OpenAI Whisper. This repository
holds the runtime implementation and its pinned integration patch series.
OpenAI Whisper does not depend on this package.

Changes proposed to OpenAI Whisper must remain useful without this runtime.
Each upstream pull request must:

- reproduce one observable problem on the current upstream branch;
- contain the smallest change that solves that problem;
- keep the public API and existing extension paths unchanged unless maintainers
  agree to a new contract;
- include a deterministic regression test;
- separate correctness claims from performance claims;
- identify the tested model, device, dependency versions, and input when the
  result depends on them.

The first independent change is
[openai/whisper#2842](https://github.com/openai/whisper/pull/2842). It keeps the
built-in decoder key-value cache local to each decode request. It does not claim
that all Whisper integrations or model extensions are thread-safe.

The grouped multi-audio correction is
[openai/whisper#2843](https://github.com/openai/whisper/pull/2843). It preserves
audio-to-group pairing for beam search and best-of decoding. Shared alignment
capture remains a separate candidate.

The alignment error-path cleanup is
[openai/whisper#2844](https://github.com/openai/whisper/pull/2844). It removes
temporary word-alignment hooks when model inference raises. It does not change
alignment calculations or successful inference behavior.

The SDPA scope-isolation change is
[openai/whisper#2845](https://github.com/openai/whisper/pull/2845). It prevents
word-alignment work in one operating-system thread from changing SDPA selection
in another. It does not make concurrent calls on one model safe.

An incremental decode-session API needs maintainer design review before an
implementation pull request.

The seven patches in this repository are a reproducible integration series for
the native adapter. They are not the proposed upstream series and must not be
read as accepted OpenAI Whisper changes.
