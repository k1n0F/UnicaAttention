"""M4.44 H1: attention_int8pv_tma with SKIP_TILES=0 and =N on real q, k, v (build/dumps/), the kernel alone on its own
inputs, the two settings taking turns in one process. For each dumped call: the median time of each setting, their
ratio, and the rel L2 of each against FP32.

    uv run python bench/skip_tiles.py [N=10] [calls per model=6] [head copies=1]

The dumps hold 2 heads: a small model's call is then less than one wave of blocks, and a warp that skips frees issue
slots no other block can take. Head copies > 1 repeats them, for a real call's number of blocks; the error is taken on
the 2 heads alone.
"""

import os
import statistics
import sys
from pathlib import Path

import torch

import unica
from emulate import errors, reference

DUMPS = Path(__file__).resolve().parents[1] / "build" / "dumps"
MODELS = ["wan_call", "zimage_call", "ltx23_call", "qwen_edit_call", "moss_call"]
SKIP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PER_MODEL = int(sys.argv[2]) if len(sys.argv) > 2 else 6
COPIES = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ROUNDS, CALLS = 6, 20
KERNEL = "attention_int8pv_tma"


def timed(q, k, v, inputs, **tile) -> float:
    """Median of CALLS launches, in ms."""
    times = []
    for _ in range(CALLS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        unica.attention(q, k, v, KERNEL, inputs, **tile)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    files = sorted(os.listdir(DUMPS))
    print(f"SKIP_TILES={SKIP}, {ROUNDS} alternating rounds × median of {CALLS}, heads × {COPIES}")
    print(f"{'call':26} {'tokens':>7} {'off ms':>8} {'on ms':>8} {'on/off':>7} {'err off':>8} {'err on':>8}", flush=True)
    ratios = {}
    for prefix in MODELS:
        mine = [f for f in files if f.startswith(prefix)]
        for f in dict.fromkeys(mine[round(i * (len(mine) - 1) / max(1, PER_MODEL - 1))] for i in range(min(PER_MODEL, len(mine)))):
            dump = torch.load(DUMPS / f, map_location="cpu")
            q, k, v = (dump[x].cuda().bfloat16() for x in "qkv")
            ref = reference(q.float(), k.float(), v.float())
            err = {s: errors(unica.attention(q, k, v, KERNEL, SKIP_TILES=s).float(), ref)["rel_l2"] for s in (0, SKIP)}
            q, k, v = (x.repeat(1, COPIES, 1, 1) for x in (q, k, v))
            inputs = unica.kernel_inputs(q, k, v, KERNEL)
            times = {0: [], SKIP: []}
            for round_ in range(ROUNDS):
                for s in ((0, SKIP) if round_ % 2 == 0 else (SKIP, 0)):
                    times[s].append(timed(q, k, v, inputs, SKIP_TILES=s))
            off, on = statistics.median(times[0]), statistics.median(times[SKIP])
            ratios.setdefault(prefix, []).append(on / off)
            print(f"{f[:26]:26} {q.shape[2]:7} {off:8.3f} {on:8.3f} {on / off:7.3f} {100 * err[0]:8.3f} "
                  f"{100 * err[SKIP]:8.3f}", flush=True)
            del q, k, v, inputs, ref
            torch.cuda.empty_cache()
    print("\nper model, on / off: mean (min … max)")
    for prefix, r in ratios.items():
        print(f"{prefix:15} {statistics.mean(r):.3f} ({min(r):.3f} … {max(r):.3f})")


if __name__ == "__main__":
    main()
