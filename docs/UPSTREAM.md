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

Candidate corrections for grouped multi-audio decoding and shared alignment
capture must be submitted separately. An incremental decode-session API needs
maintainer design review before an implementation pull request.

The seven patches in this repository are a reproducible integration series for
the native adapter. They are not the proposed upstream series and must not be
read as accepted OpenAI Whisper changes.
