# Low-latency v2: source-paced T4 check

Date: 2026-09-07. **Both short sessions pass. V0.1 is not released.**

One authorized T4 ran the same 46.55-second speech/noise/silence input twice,
in 20 ms source-paced frames. The input, 250 ms source-lateness limit,
eight-second first-commit target and word-edit gate were registered before
execution. No retry occurred.

## Results

| Measure | Session 1 | Session 2 |
| --- | ---: | ---: |
| First nonempty confirmed text | 7.358 s | 6.186 s |
| Text-growing confirmations | 11 | 11 |
| Largest gap between text-growing confirmations | 7.582 s | 7.679 s |
| Complete input coverage | 744,800 samples | 744,800 samples |
| Events / FINAL events | 38 / 1 | 38 / 1 |
| Native analyses | 25 | 25 |
| Reference word edits | 7 / 114 | 7 / 114 |
| Session time after readiness | 47.237 s | 47.062 s |
| Maximum source wake lateness | 210.950 ms | 1.605 ms |
| Peak buffered samples | 466,560 | 465,600 |
| Terminal CUDA allocated bytes | 160,720,896 | 160,720,896 |
| Terminal CUDA reserved bytes | 325,058,560 | 325,058,560 |

Both sessions use the same model and worker, restore runtime capacity, and
produce identical TXT/SRT/VTT hashes. No buffered tail remains. Peak allocated
CUDA memory is 229,245,952 bytes in session 1 and 230,071,296 bytes in session 2.
These are PyTorch counters, not total GPU memory. The repeated recording is not
independent accuracy coverage; seven word edits remain.

The update counts and gaps come from replaying the raw events through the frozen
canonical caption projection. The twelfth COMMIT covers trailing silence without
adding text. These update gaps are descriptive results, not a registered
per-word latency test or a universal eight-second bound.

## What changed

The [two-observation holdback](2026-09-07-two-observation-holdback.md) requires
right context in both word observations before confirmation. This avoids the
premature frozen boundary found in the earlier v1 run, without loosening the
timestamp tolerance. The earlier full CPU replay passed; this run now checks
that profile on paced GPU input.

The preceding v2 GPU attempt failed on 360.327 ms source lateness during its
first CUDA alignment. The new startup helper compiles that aligner's CPU
backtrace for dense and strided two-dimensional int32 arrays before accepting
audio. Compilation took 0.962 seconds here. It ran no model decode, consumed no
audio and did not warm Triton kernels. The installed CUDA command and the
harness use the same helper. The model, publication rules and timing limits
are unchanged.

The first observed CUDA DTW call took 0.665 seconds, versus 1.347 seconds in the
failed attempt. Its interval still overlaps the worst source wake delay.
This comparison supports further testing; it does not isolate the cause or
establish repeatable cold-start reliability. The initial timing margin is only
39.050 ms below the 250 ms gate. Startup cost has moved before capture, not
disappeared. First-commit timing excludes deployment, model setup, network
transport and display; it is not a per-word latency bound.

## Evidence

The [verified archive](../../evidence/modal-low-latency-v2-2026-09-07.zip)
contains the 53 uploaded files, six review files, raw PCM, all 92 observations,
results, execution journal, independent audit, verification helpers and provider
stop receipt: 69 payload files plus a manifest, 1,503,996 bytes.

Archive SHA-256:

```text
717e6fc87b4795b28f6d7807d52e1329050b76fbc26b306205072ed27cf44150
```

Uploaded-source digest:

```text
7af4c97c64938a39ee3b33e6a50ada7db11c47c748b448cf25ba3daca865a9a9
```

The independent audit reconstructs captions and exports, checks input hashes,
timing arithmetic, model/profile identity, startup receipts, memory parity and
cleanup. Its 33 local regression tests preserve earlier failed results and reject
changed or missing startup evidence. Archive-only verification passes.

Modal app `ap-XTp5swDLMT2wzQyo7RNdIz` is stopped with zero tasks. The provider
reports a stop time of 08:42:32 UTC; the read-only status query completed at
08:43:43 UTC. No endurance stage or GitHub publication ran.

## Release boundary

This passes the registered short check under the submitted 4 GiB memory limit.
Guest RSS is unchanged at 5,313,056,768 bytes across the two closed sessions,
but procfs estimates do not verify physical host RAM or provider enforcement.
The historical absolute 3.5 GiB guest-RSS gate remains failed and unchanged.

The next native gate is the separately registered endurance test. Physical
capture, clean-platform installation and exact-source release checks remain
open. The default stays `conservative-v1`; this candidate remains experimental.
The preregistered compute estimate was USD 0.104517, excluding startup/build and
other charges. No actual invoice or updated account balance was retrieved.
