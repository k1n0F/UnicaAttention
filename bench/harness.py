"""Speed and error of each attention kernel at one shape, against the FP32 reference.

M0 check: at 1 × 40 × 32760 × 128 this must reproduce rigsy.dev/bench (SDPA 511, SageAttention 2 76,
SageAttention 3 56 ms).

    uv run python bench/harness.py [--heads 40] [--tokens 32760] [--runs 10]

SageAttention 2 and 3 are local wheels built for sm_120, installed with `uv pip install`. A plain `uv sync`
removes them; use `uv sync --inexact`. A kernel whose package is missing is skipped.
"""

import argparse
import statistics

import torch
import torch.nn.functional as F
from emulate import errors, reference


def kernels():
    ks = {"PyTorch SDPA": F.scaled_dot_product_attention}
    try:
        from sageattention import sageattn

        ks["SageAttention 2"] = lambda q, k, v: sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    except ImportError as e:
        print(f"skip SageAttention 2: {e}")
    try:
        from sageattn3 import sageattn3_blackwell

        ks["SageAttention 3"] = sageattn3_blackwell
    except ImportError as e:
        print(f"skip SageAttention 3: {e}")
    return ks


def time_ms(fn, runs):
    """CUDA events: one warm-up, then the median of `runs`."""
    fn()
    times = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=32760)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name()}, {free / 2**30:.1f} of {total / 2**30:.1f} GB free")

    torch.manual_seed(0)
    shape = (1, args.heads, args.tokens, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = reference(q, k, v)
    flop = 4 * args.heads * args.tokens**2 * 128
    print(f"shape {shape}, bf16 random inputs, median of {args.runs}\n")
    print(f"{'kernel':18} {'time':>9} {'speed-up':>9} {'TFLOPS':>7} {'rel L2':>8} {'cos':>9}")

    base = None
    for name, fn in kernels().items():
        e = errors(fn(q, k, v).float(), ref)
        ms = time_ms(lambda fn=fn: fn(q, k, v), args.runs)
        base = base or ms
        print(f"{name:18} {ms:7.1f}ms {base / ms:8.1f}x {flop / ms / 1e9:7.0f} {e['rel_l2']:8.2%} {e['cos']:9.6f}")
