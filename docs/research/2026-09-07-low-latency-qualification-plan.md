# Prospective low-latency qualification — not a result

This separately registered `provider-endurance-v1` experiment does not alter the
historical `release-soak-native-v1` plan, its failed absolute guest-RSS gate, or
any archived result. Preparation is local only; a paid attempt needs a fresh,
reviewed preflight and explicit authorization. No run or default promotion is
authorized by this document.

## Fixed candidate and workload

Use the immutable named `low-latency-v2` catalog record: English tiny.en, FP32,
seed 7, 20-second left context, 24-second word-context limit,
`defer_word_commits=False`, `previous_holdback_ms=2000`, zero draft tokens and
same-window alignment-feature reuse. The earlier observation must contain two
seconds of audio after a candidate word, in addition to the existing holdback
in the later observation. All remaining publication, input-evidence and acoustic checks are the
registered profile values. The profile remains experimental; the default is
still `conservative-v1`. The earlier `low-latency-v1` record and its failed T4
attempt remain unchanged. Its frozen plan is preserved with that attempt; this
revised plan does not reinterpret the failure.

The reuse backend must be a clean Git HEAD with tree
`32163d5cdb87babc1cd415a86cc5a58116c86a16`. The frozen optional patch and its
SHA256SUMS are checked while building an isolated image from the existing pinned
standard image. A deterministic build-only commit establishes the clean tree.
The inference worker never applies a patch. Its guarded import verifies the
selected source and eagerly imports CUDA prerequisites before capture; it does
not perform a warmup decode or change model weights.

Before opening the first stream, compile the CUDA aligner's CPU backtrace for
two-dimensional int32 arrays in dense and strided layouts. This bounded startup
step consumes no audio and runs no model decode. Record its duration separately
from the source clock. A compilation error prevents readiness. Triton kernels
remain lazy; this preparation is not a claim that all first-call costs are gone.
The installed CUDA command uses the same preparation function.

The earlier v2 short attempt stopped on 360 ms source lateness during the first
CUDA DTW call. It remains a failure. A new source-paced attempt must test this
initialization change with the same 250 ms tolerance and no clock reset.

One model, one Worker and one inference owner execute:

1. The existing registered 46.55-second speech/noise/silence seed, source-paced in
   20 ms frames. The first **nonempty committed caption** must reach the owner
   callback within **8.0 seconds** of the source-clock origin. This target is
   registered before observing this candidate on GPU. A provisional preview or
   empty/silence commit does not count. Canonical caption projection identifies
   committed text; COMMIT envelopes themselves need not contain text.
2. Only after every smoke gate passes, four sequential one-hour sessions using
   whole copies of the same seed and a final silence tail. These are four closed
   sessions, not a four-hour session or four hours of independent new speech.

Record the callback's monotonic elapsed nanoseconds in each event and the first
nonempty commit time in each terminal. The collector independently reconstructs
the caption stream and checks that time. Missing, nonfinite, forged or late
first-commit evidence cannot pass. Timing excludes deployment, model startup,
network transport and display; it is not per-word latency. Later default
promotion requires a separate reviewed release decision on this exact profile.

## Unchanged safety and prospective memory contract

Retain the existing 250 ms maximum source lateness, no clock reset or dropped
frames, 30-second EOF drain, smoke word-edit rate <=10%, exact sample/hash/EOF
coverage, ordered revisions and one FINAL. Verify the exact catalog record,
native execution ID, reuse flag, backend tree/revision, model fingerprint,
single model/worker identity and restored native capacity in every phase.
Any failed phase stops all later phases. No automatic retry or reconnect occurs.

The new contract requests one T4, scalar **2 CPU cores** (no CPU maximum), and
RAM **(4096,4096) MiB request/maximum**. Minimum/maximum containers remain 0/1,
warm buffer zero, startup timeout 180 seconds, execution timeout 14,800 seconds,
and per-phase timeouts 190 seconds for smoke / 3,690 seconds per hour. The model
volume remains read-only, runtime egress blocked and provider access restricted.
Pinned Modal SDK serialization must show memory request/max 4096/4096, CPU
request/max 2000/0 millicores. The initial receipt records the submitted contract;
guest counters do not attest provider enforcement. Missing terminal or worker
restart fails; missing terminal alone is not diagnosed as OOM.

After the smoke is closed, GC and a CUDA fence establish a fixed baseline:

| Measurement | Prospective gate |
| --- | --- |
| Terminal CUDA allocated | Exactly the post-smoke baseline |
| CUDA reserved | At most baseline +256 MiB and 2,048 MiB absolute |
| Guest current RSS | Required finite integer metric; at most baseline +512 MiB |
| Sampling | Existing event-boundary approximately 300-second checks and each terminal |

Guest RSS is a guest-reported growth sentinel, **not physical host RAM**. This
new mode has no absolute 3,584 MiB guest-RSS criterion. The historical criterion
remains unchanged and its failure remains a failure. Missing required guest or
CUDA measurements makes this prospective qualification incomplete/failed, not
zero usage. No allocator reset or `empty_cache` hides growth. These engineering
sentinels do not prove absence of leaks, and sampling is not hard real-time.

## Evidence and commands

Preflight freezes exact source, review/test/plan bytes, input seed, profile,
resource contract and policy into a new namespace. Paid dispatch rechecks both
the current registration and frozen copies, then uploads only frozen source.
No existing namespace or one-use attempt receipt can be overwritten. Full event
JSONL remains capped at 32 MiB and 100,000 events per phase; the 64-message
mailbox and 512 KiB message cap are unchanged. SourceClock v2 retains timing
argmax locations and failures distinguish offered input from native admission.

```console
python -m infra.modal_release_soak --replay-id CHOSEN-NEW-ID --profile low-latency-v2 --capacity-limit --preflight
# Only after review and separate paid authorization:
python -m infra.modal_release_soak --replay-id CHOSEN-NEW-ID --profile low-latency-v2 --capacity-limit --confirm-paid-gpu
```

The existing frozen price assumptions yield approximately **$5.16 compute** for
the 14,800-second request. This is not a spending cap and excludes image build,
startup, storage, egress, taxes, price changes and provider crash rescheduling.
Do not launch endurance simply because the two-session capacity diagnostic
passed: the chosen-profile smoke and this contract's additional gates still
apply. Completion supports only the registered repeated-input native workload,
not WebSocket/WAN, diverse speech, microphone hardware or production guarantees.
