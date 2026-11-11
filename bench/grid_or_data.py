"""M4.54: why 8 computing warps lose on Z-Image: the grid or the data. Two settings of attention_int8pv_tma (4 and 8
computing warps, both with the producer), the kernel alone, alternating rounds, on random inputs of three grids and on
Z-Image's own q, k, v as dumped, with the query rows shuffled and with the keys (and V) shuffled; Moss as the control.

    uv run python bench/grid_or_data.py [calls per dumped model=3]
"""

import os
import statistics
import sys
from pathlib import Path

import torch

import unica

DUMPS = Path(__file__).resolve().parents[1] / "build" / "dumps"
KERNEL = "attention_int8pv_tma"
SETTINGS = {"4 warps": {"PRODUCER": 1}, "8 warps": {"PRODUCER": 1, "WARPS_PER_BLOCK": 8}}
HEADS, COPIES = 24, 12  # random inputs have 24 heads; a dump's 2 heads are repeated to 24
ROUNDS, CALLS = 6, 20
PER_MODEL = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def median_ms(q, k, v, inputs, tile) -> float:
    times = []
    for _ in range(CALLS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        unica.attention(q, k, v, KERNEL, inputs, **tile)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def ratio(q, k, v) -> float:
    """8 warps / 4 warps, medians over alternating rounds; the two outputs must be equal bit for bit."""
    inputs = {name: unica.kernel_inputs(q, k, v, KERNEL, **tile) for name, tile in SETTINGS.items()}
    outs = [unica.attention(q, k, v, KERNEL, inputs[name], **tile) for name, tile in SETTINGS.items()]
    assert torch.equal(*outs)
    times = {name: [] for name in SETTINGS}
    for r in range(ROUNDS):
        for name in (list(SETTINGS) if r % 2 == 0 else list(SETTINGS)[::-1]):
            times[name].append(median_ms(q, k, v, inputs[name], SETTINGS[name]))
    four, eight = (statistics.median(times[name]) for name in SETTINGS)
    return eight / four, four, eight


def dumped(prefix):
    files = sorted(f for f in os.listdir(DUMPS) if f.startswith(prefix))
    for f in dict.fromkeys(files[round(i * (len(files) - 1) / max(1, PER_MODEL - 1))] for i in range(PER_MODEL)):
        dump = torch.load(DUMPS / f, map_location="cpu")
        yield f, [dump[x].cuda().bfloat16().repeat(1, COPIES, 1, 1) for x in "qkv"]


def report(label, q, k, v):
    r, four, eight = ratio(q, k, v)
    print(f"  {label:44} {q.shape[1]:3} × {q.shape[2]:6}  4 warps {four:7.3f} ms  8 warps {eight:7.3f} ms  {r:.3f}",
          flush=True)


def main():
    print(f"8 / 4 computing warps (PRODUCER), kernel alone, {ROUNDS} alternating rounds × median of {CALLS}")
    torch.manual_seed(0)
    for tokens in (4096, 4128, 3874):
        report(f"random, {tokens} tokens", *(torch.randn(1, HEADS, tokens, 128, device="cuda").bfloat16()
                                             for _ in range(3)))
    for f, (q, k, v) in dumped("zimage_call"):
        report(f"Z-Image {f[:18]}, as dumped", q, k, v)
        g = torch.Generator(device="cpu").manual_seed(1)
        rows = torch.randperm(q.shape[2], generator=g).cuda()
        report(f"Z-Image {f[:18]}, query rows shuffled", q[:, :, rows].contiguous(), k, v)
        keys = torch.randperm(k.shape[2], generator=g).cuda()
        report(f"Z-Image {f[:18]}, keys shuffled", q, k[:, :, keys].contiguous(), v[:, :, keys].contiguous())
    for f, (q, k, v) in dumped("moss_call"):
        report(f"Moss {f[:18]}, as dumped", q, k, v)


if __name__ == "__main__":
    main()
