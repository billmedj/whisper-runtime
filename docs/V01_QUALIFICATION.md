# Bounded V0.1 live qualification

Result: [the short test and 30-minute T4 run pass](research/2026-09-06-live-v01.md)
under replay ID `v01-live-20260906-a2`. The original failed a1 attempt is retained.
Use a new explicit replay ID for any future run; do not overwrite these records.

`infra/modal_live_qualification.py` is an opt-in runnable harness, not a permanent
deployment. Import and `--preflight` do not start remote resources. The paid path
requires the exact matching preflight plus explicit confirmation; an exclusive
attempt receipt prevents reruns or overwrites in the same namespace.

## Registered sequence

1. A CPU-only authenticated Modal echo checks binary transport and verifies that
   missing proxy credentials are rejected before any GPU request.
2. One T4 loads only the existing checksum-verified `tiny.en` cache, read-only.
   Its pinned backend/tree/dependencies and FP32 model fingerprint are checked.
   Path loading explicitly restores the verified named-model alignment mask via
   `native_setup._load_model`. The stream uses the exact snapshotted
   `native_setup.CLI_STREAM_CONFIG`, including the preregistered deferred-commit
   and window-reserve candidate. There is no alternate inference fallback.
3. `/smoke`: full 33.66-second concatenated speech, one second digital silence,
   the 10.89-second registered noisy prefix, then one second silence (46.55 s).
   The input is sent on absolute source deadlines after READY. Full committed
   coverage, exact source hash, bounded buffering, unchanged weights and restored
   capacity are required. The human-reference word-edit-rate ceiling is **10%**,
   registered before GPU use against CPU calibration of 6/88 main edits
   (unchanged baseline) and 1/26 noisy-prefix edits. This is not held-out or
   general accuracy qualification. Raw per-arm text/scores are retained; commits
   spanning an arm boundary are explicitly reported as unattributable.
4. Only a successful smoke permits `/long`: exactly 1,800 seconds, repeating 38
   complete 46.55-second cycles then 31.1 seconds of silence. This is a source-
   paced file endurance test, not 30 minutes of independent acoustic material.
   Both stages must prove the same container instance, one model load and two
   successive cleaned-up sessions. Replacement containers reject `/long`;
   there is no reconnect, resend or retry.

The combined 46.55-second join is a new observation and may fail despite the
individual CPU cases passing. A failed smoke ends the attempt and blocks long
execution; its score threshold is not changed after seeing GPU output.

## Commands

Run from the repository root with `src` on `PYTHONPATH`. The available Modal SDK
environment is `../whisper-runtime-modal/.tmp-modal-sdk/Scripts/python.exe`
(Modal 1.5.5 and aiohttp are already present); no environment installation is
required. Freeze all source changes before preflight.

```powershell
$env:PYTHONPATH = 'src'
python -B -m unittest tools.test_modal_live_qualification -q
& '../whisper-runtime-modal/.tmp-modal-sdk/Scripts/python.exe' -B -m infra.modal_live_qualification --preflight --replay-id v01-live-20260906-a1
```

The preflight writes exact source hashes, input recipe, prices/bounds and two raw
PCM files (~59 MB) under `artifacts/modal/<replay-id>/`. It does not import Modal
or contact the service. Review the files and only then explicitly authorize:

```powershell
& '../whisper-runtime-modal/.tmp-modal-sdk/Scripts/python.exe' -B -m infra.modal_live_qualification --confirm-paid-gpu --replay-id v01-live-20260906-a1
```

Inspect `live-v01.json`, not just the process exit code. Local event JSONL files
are written incrementally and survive failures; each has an 8 MiB hard limit.
Raw source files are rehashed before the paid call. A harness-only authenticated
single-use GET `/receipt/<phase>` retrieves up to 512 KiB of terminal native
diagnostics without inference; the same worker/source identity is checked. No
redirect or retry is allowed. A cold container or lost report produces an
explicit missing/unavailable receipt and cannot pass the gate. The generic
client's error redaction remains unchanged.

## Runtime and budget boundaries

The smoke client/server deadline is 190 s; long is 1,890 s. The staged client has
a 2,200 s outer watchdog, and Modal enforces a 1,890 s maximum per GPU request.
There is one possible GPU container, `min_containers=0`, no buffer containers,
two-second scale-down and a non-detached ephemeral app context. The app rejects
duplicate/out-of-order source sessions. ASGI functions do not accept Modal's
`retries` setting, so it is omitted rather than falsely claiming it is set to
zero; the harness itself never retries. A scoped proxy token is deleted in
`finally`, and context exit/token deletion are recorded separately.

Using checked rates of $0.000164/T4-second, $0.0000131/CPU-core-second and
$0.00000222/GiB-second, two CPU cores, four GiB and the 1.75 regional multiplier
ceiling, 2,200 seconds estimates **$0.766458 compute**. The first-attempt target
is below $1 and this pass's total target is $2; neither is a physical billing
cap. Startup, image builds, storage, egress, taxes, provider rescheduling and
price changes can add charges. The image recipe reuses the pinned cached build;
no model-cache creation or model-download fallback is permitted.
[Modal pricing, checked 2026-09-06](https://modal.com/pricing).

## Accelerated local soak is a different result

The separate `tools/verify_live_soak.py` exercises four **logical** hours with a
scripted native controller in accelerated CPU time. It tests buffering,
state/history growth and lifecycle handling; it does not represent four hours
of wall-clock service, real GPU inference or acoustic quality. Keep its evidence
separate from the real source-paced 30-minute Modal result.
