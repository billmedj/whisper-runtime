# Conformance cases

`cases.json` is the initial conformance matrix for the runtime. Each case has:

- a stable ID;
- a model name;
- an audio fixture class;
- explicit decode or execution options;
- a required compatibility profile;
- resource measurements that the runner must record;
- an implementation status.

`implemented` means that the case has reference and candidate records. `planned`
means that the case definition exists but its fixture, runner support, or oracle is
not complete.

The paths in `fixture_records` are relative to this directory. They must stay
portable. Do not store machine-specific paths in this matrix.

`audio-manifest.json` identifies external audio by a stable ID, a pinned URL,
size, and SHA-256 digest. Fetch the current sample with:

```powershell
python tools/fetch_audio_fixture.py openai-whisper-jfk-flac conformance/cache/jfk.flac
```

The capture tool records the git commit and a SHA-256 digest of the source tree.
The digest covers tracked files and non-ignored untracked files. The input audio
path is the only explicit exclusion. Any other source change marks the capture
as dirty.

The values in `expected_resource_measurements` are required measurement names.
They are not performance limits. Add limits only after repeatable measurements
exist for a declared environment.

## Add a case

1. Add one entry with a new stable ID.
2. Use explicit options. Include the random seed when sampling can occur.
3. List each resource value that the runner must record.
4. Set `status` to `planned`.
5. Capture a reference record and a candidate record.
6. Set `status` to `implemented` only after the oracle can compare both records.

`tools/check_repository.py` validates every implemented record and compares
each reference/candidate pair. A missing field, dirty source checkout, output
difference, or undeclared fixture causes CI to fail.

The first implemented case uses the pinned Whisper JFK sample with `tiny.en`
and greedy decoding. Planned entries include sampling, beam search, timestamps,
word timestamps, batching, concurrency, errors, and cancellation.
