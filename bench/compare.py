"""
Variants of the UnicaAttention kernel side by side, in one run.

Run-to-run drift (~10 %, all kernels together) is as large as many single optimizations, so a
variant is only ever compared with another one timed in the same process, in alternating rounds.

    uv run python bench/compare.py [attention attention_smem ...] [--heads 40] [--tokens 32760]
"""

from __future__ import annotations

import argparse
import statistics

import torch
from emulate import errors, reference
from harness import time_ms

import unica

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="*", default=list(unica.KERNELS))
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=32760)
    ap.add_argument("--runs", type=int, default=5, help="timed runs per round")
    ap.add_argument("--rounds", type=int, default=3, help="the variants take turns this many times")
    args = ap.parse_args()

    torch.manual_seed(0)
    shape = (1, args.heads, args.tokens, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = reference(q, k, v)
    print(f"shape {shape}, bf16 random inputs, {args.rounds} rounds × median of {args.runs}\n")

    times: dict[str, list[float]] = {name: [] for name in args.kernels}
    for _ in range(args.rounds):
        for name in args.kernels:
            times[name].append(time_ms(lambda name=name: unica.attention(q, k, v, kernel=name), args.runs))

    flop = 4 * args.heads * args.tokens**2 * 128
    base = statistics.median(times[args.kernels[0]])
    print(f"{'kernel':18} {'time':>9} {'spread':>8} {'vs first':>9} {'TFLOPS':>7} {'rel L2':>8}")
    for name in args.kernels:
        ms = statistics.median(times[name])
        spread = (max(times[name]) - min(times[name])) / ms
        e = errors(unica.attention(q, k, v, kernel=name).float(), ref)
        print(f"{name:18} {ms:7.1f}ms {spread:8.1%} {base / ms:8.2f}x {flop / ms / 1e9:7.0f} {e['rel_l2']:8.2%}")
