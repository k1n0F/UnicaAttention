"""
One variant of the kernel for Nsight Compute: quantizes once, then launches the kernel twice on the prepared
inputs (the first launch builds and loads the cubin). Profile the second launch:

    uv run ncu.bat -k regex:^attention_int8pv_async$ -s 1 -c 1 --metrics ... \
        python bench/ncu_launch.py attention_int8pv_async:FAST_EXP2=0 [--heads 4] [--queries 32760] [--keys 32760]

--heads 1 --queries 5376 gives 84 blocks of 64 queries, one per SM.
"""

from __future__ import annotations

import argparse

import torch
from compare import parse_variant

import unica

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", help="a kernel name, optionally with build settings, as in bench/compare.py")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--queries", type=int, default=32760)
    ap.add_argument("--keys", type=int, default=32760)
    args = ap.parse_args()

    kernel, tile = parse_variant(args.variant)
    torch.manual_seed(0)
    q = torch.randn(1, args.heads, args.queries, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(1, args.heads, args.keys, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    inputs = unica.kernel_inputs(q, k, v, kernel, **tile)
    for _ in range(2):
        unica.attention(q, k, v, kernel, inputs, **tile)
    torch.cuda.synchronize()
