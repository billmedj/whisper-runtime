# Reproducible Whisper integration patches

This seven-patch integration series builds the staged Whisper backend used by
the native adapter. It applies to OpenAI Whisper commit
`86098128c0b4f24f0e2aa2994de830614b474227`.

The series is part of this repository's integration test. It is not an upstream
pull-request series and does not represent accepted OpenAI Whisper changes.

Apply the files in numeric order:

```text
0001-Make-native-inference-state-request-local.patch
0002-Make-decode-options-request-local.patch
0003-Prototype-suspendable-token-step-decoding.patch
0004-Harden-request-local-decode-options.patch
0005-Fix-grouped-decoding-for-audio-batches.patch
0006-Harden-suspendable-decode-lifecycle.patch
0007-Serialize-legacy-cache-run-lifetimes.patch
```

From a full repository checkout, the supported bootstrap verifies and applies
this series and installs its dependencies in a project-local environment:

```sh
python tools/bootstrap_native_backend.py
```

See the [real backend quick start](../../docs/REAL_BACKEND_QUICKSTART.md). The
manual procedure below remains useful for patch review and custom environments.

On a POSIX shell, verify the patch files before applying them:

```sh
cd /path/to/whisper-runtime/patches/openai-whisper
sha256sum --check SHA256SUMS

git clone https://github.com/openai/whisper.git whisper-runtime-backend
cd whisper-runtime-backend
git checkout 86098128c0b4f24f0e2aa2994de830614b474227
git am /path/to/whisper-runtime/patches/openai-whisper/*.patch
python -m pip install -r requirements.txt
```

The resulting source tree must have this Git tree identifier:

```text
c011d2563c26763b5f147026e6b18ef85bccd4fb
```

Verify it with `git rev-parse 'HEAD^{tree}'`. The resulting commit identifier
can differ because `git am` records a new committer timestamp. Pass that full
commit identifier to the native smoke test.

The patch files are reviewable mail patches. Their SHA-256 digests are:

```text
48c77a79e5ba289512e70c8df40fc911e5f0282fdbaae67f118641565433cc03  0001-Make-native-inference-state-request-local.patch
efe6dcd69581aa4c59dcc172d5661b7e2fb33dea64ee367ed6bd51df2f89f30d  0002-Make-decode-options-request-local.patch
5d9068509dc353f9320f53442237155f0a6e752214b61ca4e0f094b2f824a1e2  0003-Prototype-suspendable-token-step-decoding.patch
cd725795f0a0f2e5acf8c7d7ca20f19244b2808b00f8ade584ca565d03cec4ba  0004-Harden-request-local-decode-options.patch
d3c4ca1af1f7d3226c10dd99fe55b14ee02ac7ed34a54f8d301e83c3f5e50468  0005-Fix-grouped-decoding-for-audio-batches.patch
85982cfb95e7ad67b2d1c231e0281f3d268f107733bbd76f09c41a3625bb259a  0006-Harden-suspendable-decode-lifecycle.patch
19c3380c7d8c8a72849bf565b615f1fdc26cb639b83f75b488cf7a9217607460  0007-Serialize-legacy-cache-run-lifetimes.patch
```

## License and provenance

The patches contain modified and contextual portions of OpenAI Whisper source.
That source is Copyright (c) 2022 OpenAI and licensed under the MIT License.
The [MIT license](LICENSE) applies to the patch series. See
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) for full provenance.
