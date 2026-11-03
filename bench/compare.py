"""
Variants of the UnicaAttention kernel side by side, in one run.

Run-to-run drift (~10 %, all kernels together) is as large as many single optimizations, so a
variant is only ever compared with another one timed in the same process, in alternating rounds.

    uv run python bench/compare.py [attention attention_smem ...] [--heads 40] [--tokens 32760]

A variant is a kernel name, optionally with a tile shape (see unica.TILE):

    attention_smem:WARPS_PER_BLOCK=8,KEYS_PER_TILE=32
"""

from __future__ import annotations

import argparse
import statistics

import torch
from emulate import errors, reference
from harness import time_ms

import unica


def parse_variant(spec: str) -> tuple[str, dict[str, int]]:
    """"attention_smem:KEYS_PER_TILE=32" -> ("attention_smem", {"KEYS_PER_TILE": 32})."""
    kernel, _, settings = spec.partition(":")
    tile = {key: int(value) for key, value in (item.split("=") for item in settings.split(",") if item)}
    return kernel, tile


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

    # A quantized kernel is timed three ways: the whole call, its quantization alone, and the kernel alone on
    # inputs quantized once beforehand.
    times: dict[str, list[float]] = {name: [] for name in args.kernels}
    quant_times: dict[str, list[float]] = {name: [] for name in args.kernels}
    kernel_times: dict[str, list[float]] = {name: [] for name in args.kernels}
    for _ in range(args.rounds):
        for name in args.kernels:
            kernel, tile = parse_variant(name)
            run = lambda kernel=kernel, tile=tile: unica.attention(q, k, v, kernel=kernel, **tile)  # noqa: E731
            times[name].append(time_ms(run, args.runs))
            if kernel in unica.QUANTIZED:
                quantize = lambda kernel=kernel: unica.kernel_inputs(q, k, v, kernel)  # noqa: E731
                quant_times[name].append(time_ms(quantize, args.runs))
                inputs = quantize()
                run = lambda kernel=kernel, tile=tile: unica.attention(q, k, v, kernel, inputs, **tile)  # noqa: E731
                kernel_times[name].append(time_ms(run, args.runs))
                del inputs

    flop = 4 * args.heads * args.tokens**2 * 128
    base = statistics.median(times[args.kernels[0]])
    width = max(18, *map(len, args.kernels))
    print(f"{'kernel':{width}} {'time':>9} {'spread':>8} {'vs first':>9} {'TFLOPS':>7} {'rel L2':>8} "
          f"{'quantize':>9} {'kernel':>9}")
    for name in args.kernels:
        ms = statistics.median(times[name])
        spread = (max(times[name]) - min(times[name])) / ms
        kernel, tile = parse_variant(name)
        e = errors(unica.attention(q, k, v, kernel=kernel, **tile).float(), ref)
        split = (f" {statistics.median(quant_times[name]):7.2f}ms {statistics.median(kernel_times[name]):7.1f}ms"
                 if quant_times[name] else "")
        print(f"{name:{width}} {ms:7.1f}ms {spread:8.1%} {base / ms:8.2f}x {flop / ms / 1e9:7.0f} "
              f"{e['rel_l2']:8.2%}{split}")
