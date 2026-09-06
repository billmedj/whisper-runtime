# Experimental same-window alignment features

This optional patch adds `find_alignment(..., audio_features=None)` and seven
CPU tests to the existing seven-patch Whisper backend. It is **not** part of
the active patch manifest, bootstrap, or cached Modal image. The native adapter
now supports it through an explicit `reuse_alignment_features=True` execution
profile. The default profile still encodes mel with `use_sdpa=False` for
alignment; existing backend locks and archived experiments are unchanged.

Base tree: `c011d2563c26763b5f147026e6b18ef85bccd4fb`.
Patched tree: `32163d5cdb87babc1cd415a86cc5a58116c86a16`.
The patch digest is recorded in this directory's `SHA256SUMS`, separately from
the active series. The parent directory's MIT license and provenance apply.

## Contract and limits

The opt-in argument is the unbatched tensor already available as
`DecodingResult.audio_features`: shape `(n_audio_ctx, n_audio_state)`, model
device, and float16 or float32 dtype. The function borrows it synchronously,
does not mutate or cache it, and invokes the built-in decoder with request-local
attention capture and `use_sdpa=False`. It rejects custom model implementations
on this path. Model/encoder forward hooks are bypassed only on the feature path;
decoder hooks retain their normal behavior. Omitting the argument preserves the
existing native and custom-model paths.

Shape/device/dtype checks do not establish provenance. The caller must bind the
features to the exact immutable audio window, model identity, precision and
encoder settings, use tokens from that window, and retain ownership until the
operation and its device-completion fence finish. The opt-in path does not read
mel to verify that binding; doing so by re-encoding would defeat reuse. It is
not a cross-window cache and has no publication or timestamp acceptance authority.

In particular, decoding's default encoder can use SDPA, while legacy alignment
explicitly disables it. The two encodings are **not assumed numerically equal**.
The CPU tests show exact alignment equality only when supplied features were
produced by the matching explicit `use_sdpa=False` encoder. They also check
zero encoder calls on reuse, unchanged features/settings/hooks after success
and failure, malformed-input rejection, and unchanged default/custom behavior.
They do not establish GPU parity, acoustic correctness, or service latency
savings. Native adapter tests separately cover ownership and recovery. Its
opt-in profile permits `prepare_word_alignment(reuse_alignment_features=False)`
as a legacy control; it cannot change modes after alignment is cached. A paired
T4 comparison of actual decode-produced features is defined in the
[handoff report](../../../docs/research/2026-09-06-alignment-feature-handoff.md).

## Explicit local application

Use a disposable checkout of the pinned backend tree, not an active cached
backend or archived-run environment. Check this directory's SHA-256 manifest
first. From that checkout, replace the path below with this patch's location:

```sh
git rev-parse 'HEAD^{tree}'
git apply --check /path/to/experimental/0008-Add-optional-alignment-audio-features.patch
git apply /path/to/experimental/0008-Add-optional-alignment-audio-features.patch
python -m unittest discover -s tests -p test_alignment_audio_features.py -v
```

Use an already provisioned pinned CPU backend environment. No pretrained model,
network access, GPU, dependency installation, or new commit is needed. The
tests also collect under pytest if it is already installed. This patch is a
plain unified diff for `git apply`, not an addition to the default `git am`
series. Any later GPU experiment must explicitly record the new backend source
identity and patch digest; existing image/source locks must not silently change.
