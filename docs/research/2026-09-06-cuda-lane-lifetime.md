# CUDA lane lifetime

## Question

The word-resolution experiment recorded an 8,519,680-byte increase in peak
allocated memory per native window. PyTorch 2.6 retains a cuBLAS workspace for
each handle and stream pair. Its default non-Hopper workspace has exactly that
size. This is a testable explanation, not evidence of a decoder tensor leak.

Sources: [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.6/notes/cuda.html#cublas-workspaces)
and [the pinned handle implementation](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CublasHandlePool.cpp).

## Change

Keep one CUDA lane with the model binding. Each admitted window gets a fresh
execution scope and borrows the lane. It returns the lane only after cleanup
and a successful completion fence. Failed cleanup keeps the loan and blocks
new work until recovery. Cancellation belongs to the scope, not the lane.

The first lane creation retains the model-initialization barrier. Subsequent
windows use the same stream. This change does not share decoder state, remove
publication checks, or add GPU concurrency. A stable driving thread is needed
to bound thread-local library handles as well as streams.

The lane and its library workspaces are worker-level resources. Released
transaction capacity does not mean that all physical GPU memory is returned.

## Registered comparison

`infra/modal_cuda_lane.py` runs one paid T4 function. It uses the existing cached
FP32 `tiny.en` checkpoint and backend. The maximum is twelve native windows,
six per arm, with no automatic retry. The function timeout is 180 seconds and
the container limit is one. The shared CPU transport check runs before the GPU
call. These execution limits are not an invoice cap.

Both arms use the same LibriSpeech utterance, PCM, mel tensor, model, options,
and host thread. The order alternates: reused lane, fresh lane. The first pair
is cold. The control discards only an idle lane after a successful fence; the
candidate retains one lane between windows. This is a private test ablation,
not a public lane-reset API. The driver never clears allocator or cuBLAS caches.

Record for each call:

- CUDA stream identity, encoder calls, and complete word alignment;
- allocated and reserved bytes before execution and after close and collection;
- peak memory, elapsed time, capacity release, and stale cancellation outcome.

The cached backend retains finalized features while a closed run handle is
reachable. Each post-close sample retains one handle for the next stale-cancel
test. Warm deltas therefore compare the same fixed input and handle lifetime;
they do not represent zero decoder storage. A final sample drops that handle.
Partial cells are returned if a measurement fails. The first cold call also
establishes this retained-handle baseline. Report it separately.

The prediction is flat post-close allocated memory for warm reused-lane calls,
with identical words, tokens, and word times across both arms. Fresh-lane calls
should reproduce workspace growth. Reserved memory is reported separately.
Different results reject or narrow this explanation. Timing is descriptive:
one input and alternating calls do not establish a general speedup.

This experiment does not change the alignment encoder path or the live
publication policy. They require separate comparisons.
