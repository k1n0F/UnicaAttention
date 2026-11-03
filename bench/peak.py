"""
Tensor-core peak of attention's mma mix (half bf16, half fp16 m16n8k16, FP32 accumulate), in TFLOPS
at whatever clock the card runs now. The ceiling bench/compare.py results are held against.

    uv run python bench/peak.py
"""

from __future__ import annotations

import torch
from harness import time_ms

from unica.cuda import module

SMS = 84
WARPS_PER_SM = 16  # enough warps that every tensor unit always has an mma ready
MMA_PER_ITERATION = 8
FLOP_PER_MMA = 2 * 16 * 8 * 16
ITERATIONS = 200_000

if __name__ == "__main__":
    blocks, threads = SMS * WARPS_PER_SM // 4, 128
    out = torch.empty(blocks * threads, device="cuda")
    kernel = module("flash")["mma_peak"]
    ms = time_ms(lambda: kernel(blocks, threads, out, ITERATIONS), runs=5)
    flop = blocks * threads // 32 * ITERATIONS * MMA_PER_ITERATION * FLOP_PER_MMA
    print(f"mma_peak: {ms:.1f} ms, {flop / ms / 1e9:.0f} TFLOPS")
