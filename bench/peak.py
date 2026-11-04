"""
Peak rates of the instructions attention is built from (csrc/peak.cu), at whatever clock the card runs
now: the ceilings bench/compare.py results are held against. All forms take turns in each round, so
they share the same clock.

    uv run python bench/peak.py [--rounds 3]
"""

from __future__ import annotations

import argparse
import statistics

import torch
from harness import time_ms

from unica.cuda import module

SMS = 84
WARPS_PER_SM = 16  # enough warps that every unit always has an instruction ready
CHAINS = 8
ITERATIONS = 20_000

# kernel -> (work per warp instruction, unit)
FORMS = {
    "peak_bf16_f16_f32": (2 * 16 * 8 * 16, "TFLOPS"),  # the exact path's mix, m16n8k16
    "peak_s8s8": (2 * 16 * 8 * 32, "TOPS"),  # m16n8k32 from here on
    "peak_u8s8": (2 * 16 * 8 * 32, "TOPS"),
    "peak_e4m3_f32": (2 * 16 * 8 * 32, "TFLOPS"),
    "peak_e4m3_f16": (2 * 16 * 8 * 32, "TFLOPS"),
    "peak_mix_s8s8_u8s8": (2 * 16 * 8 * 32, "TOPS"),
    "peak_mix_s8s8_e4m3_f16": (2 * 16 * 8 * 32, "TOPS"),
    "peak_i2f": (32, "T/s"),  # one conversion per lane
    "peak_ex2": (32, "T/s"),
}
# M4: FP4. k64 forms do 16384 flop per instruction; the pair kernels the same per chain and round
FORMS |= {name: (2 * 16 * 8 * 64, "TFLOPS") for name in ("peak_nvf4", "peak_mxf4", "peak_pair_nvf4_s8s8", "peak_pair_nvf4_u8s8")}
FORMS |= {name: (2 * 16 * 8 * 32, "TFLOPS") for name in ("peak_e2m1_e4m3_f32", "peak_e2m1_e4m3_f16", "peak_mxf8f6f4_e2m1_e4m3", "peak_mxf8f6f4_e4m3_e4m3")}
# M3.8: u8 × s8 with distinct operand registers and N FFMA per mma
FORMS |= {f"peak_u8s8_mixed_{n}": (2 * 16 * 8 * 32, "TOPS") for n in (0, 4, 8, 12, 16)}
FORMS |= {f"peak_u8s8_mixed_8_{kind}": (2 * 16 * 8 * 32, "TOPS") for kind in ("FFMA_IMM", "FMUL", "FMNMX", "IADD", "I2F", "EX2")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("forms", nargs="*", default=list(FORMS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warps-per-sm", type=int, default=WARPS_PER_SM, help="8 = 2 per scheduler, as attention runs")
    args = ap.parse_args()
    FORMS = {name: FORMS[name] for name in args.forms}

    blocks, threads = SMS * args.warps_per_sm // 4, 128
    out = torch.empty(blocks * threads, device="cuda")
    peak = module("peak")
    times: dict[str, list[float]] = {name: [] for name in FORMS}
    for _ in range(args.rounds):
        for name in FORMS:
            times[name].append(time_ms(lambda name=name: peak[name](blocks, threads, out, ITERATIONS), runs=5))

    warp_instructions = blocks * threads // 32 * ITERATIONS * CHAINS
    for name, (work, unit) in FORMS.items():
        ms = statistics.median(times[name])
        print(f"{name:24} {warp_instructions * work / ms / 1e9:7.1f} {unit}")
