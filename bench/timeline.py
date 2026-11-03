"""
When are the warps of one scheduler in an MMA phase, and do they get there together? (LOG.md, M3.8)

attention_int8pv_async built with TRACE=1 records clock64 at four points of every tile for the warps of
SM 0: tile start, last Q·Kᵀ IMMA, P packed, out flushed. The MMA phases are [start, Q·Kᵀ] and [P, flush].
For each scheduler (%warpid % 4) and each stretch of time in which the same warps are resident, this
compares the time no warp is in an MMA phase with what independent warps would give: the product of
(1 − each warp's MMA fraction) over that stretch.

    uv run python bench/timeline.py [attention_int8pv_async:KEYS_PER_TILE=32,MIN_BLOCKS_PER_SM=4] [--heads 1]
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import torch
from compare import parse_variant

import unica
from unica.cuda import module

TRACE_SLOTS, TRACE_TILES, TRACE_POINTS = 64, 1024, 4  # must match csrc/flash.cu
SCHEDULERS = 4


def timeline(variant: str, heads: int, tokens: int) -> tuple[torch.Tensor, list[int]]:
    """Runs the variant with TRACE=1; returns the clocks [slots, tiles, points] and each slot's %warpid."""
    kernel, tile = parse_variant(variant)
    tile["TRACE"] = 1
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, tokens, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    inputs = unica.kernel_inputs(q, k, v, kernel, **tile)
    defines = {f"UNICA_{key}": value for key, value in tile.items() if unica.TILE[key] != value}
    clocks = torch.empty(TRACE_SLOTS, TRACE_TILES, TRACE_POINTS, dtype=torch.int64, device="cuda")
    warps = torch.empty(TRACE_SLOTS, dtype=torch.int32, device="cuda")
    used = torch.empty(1, dtype=torch.int32, device="cuda")
    read = module("flash", **defines)["trace_read"]
    unica.attention(q, k, v, kernel, inputs, **tile)  # builds the cubin, warms up
    read(1, 1, clocks, warps, used)  # and clears the slots
    unica.attention(q, k, v, kernel, inputs, **tile)
    read(1, 1, clocks, warps, used)
    torch.cuda.synchronize()
    n = used.item()
    n_tiles = math.ceil(tokens / tile.get("KEYS_PER_TILE", unica.TILE["KEYS_PER_TILE"]))
    return clocks[:n, :n_tiles].cpu(), warps[:n].tolist()


def overlap(clocks: torch.Tensor, warps: list[int]) -> dict[str, float]:
    """Time per scheduler by how many resident warps are in an MMA phase, against independent warps."""
    totals = defaultdict(float)
    for scheduler in range(SCHEDULERS):
        events = []  # (time, slot, what): what 0 = MMA ends, 1 = leaves, 2 = arrives, 3 = MMA starts
        for slot in (s for s, w in enumerate(warps) if w % SCHEDULERS == scheduler):
            c = clocks[slot].tolist()
            events += [(c[0][0], slot, 2), (c[-1][3], slot, 1)]
            for start, qk, p, flushed in c:
                for begin, end in ((start, qk), (p, flushed)):
                    if end > begin:  # an empty phase would sort its end before its own start
                        events += [(begin, slot, 3), (end, slot, 0)]
        if not events:
            continue
        events.sort()
        resident, in_mma = set(), set()
        # per set of resident warps: time per subset of them in MMA
        stretches: dict[frozenset, dict[frozenset, float]] = defaultdict(lambda: defaultdict(float))
        for (time, slot, what), (next_time, *_) in zip(events, events[1:] + [events[-1]]):
            if what == 2:
                resident.add(slot)
            elif what == 1:
                resident.discard(slot)
                in_mma.discard(slot)
            elif what == 3:
                in_mma.add(slot)
            else:
                in_mma.discard(slot)
            if len(resident) >= 2:
                stretches[frozenset(resident)][frozenset(in_mma & resident)] += next_time - time
        for group, by_subset in stretches.items():
            span = sum(by_subset.values())
            fraction = {s: sum(t for sub, t in by_subset.items() if s in sub) / span for s in group}
            totals["time"] += span
            totals["none"] += by_subset.get(frozenset(), 0.0)
            totals["none, independent"] += span * math.prod(1 - f for f in fraction.values())
            totals["two or more"] += sum(t for sub, t in by_subset.items() if len(sub) >= 2)
            one = sum(f * math.prod(1 - g for s2, g in fraction.items() if s2 != s) for s, f in fraction.items())
            totals["two or more, independent"] += span * (1 - math.prod(1 - f for f in fraction.values()) - one)
            totals["warps"] += span * len(group)
            totals["mma fraction"] += span * sum(fraction.values())
    t = totals["time"]
    return {
        "resident warps per scheduler": totals["warps"] / t,
        "MMA fraction per warp": totals["mma fraction"] / totals["warps"],
        "no warp in MMA": totals["none"] / t,
        "no warp in MMA, if independent": totals["none, independent"] / t,
        "2+ warps in MMA": totals["two or more"] / t,
        "2+ warps in MMA, if independent": totals["two or more, independent"] / t,
    }


def test_overlap_tells_bunched_from_staggered_warps():
    # two warps on scheduler 0 (warp ids 0 and 4), both resident in [0, 100], each in MMA a quarter of it
    a = torch.tensor([[[0, 25, 50, 50], [50, 50, 100, 100]]])  # MMA in [0, 25]
    b = torch.tensor([[[0, 0, 50, 75], [100, 100, 100, 100]]])  # MMA in [50, 75]
    together = overlap(torch.cat([a, a]), [0, 4])
    assert together["no warp in MMA"] == 0.75 and together["2+ warps in MMA"] == 0.25, together
    assert together["no warp in MMA, if independent"] == 0.75**2, together
    apart = overlap(torch.cat([a, b]), [0, 4])
    assert apart["no warp in MMA"] == 0.5 and apart["2+ warps in MMA"] == 0, apart
    assert apart["no warp in MMA, if independent"] == 0.75**2, apart


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=["attention_int8pv_async"])
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=32760)
    args = ap.parse_args()
    test_overlap_tells_bunched_from_staggered_warps()
    for variant in args.variants:
        clocks, warps = timeline(variant, args.heads, args.tokens)
        print(f"\n{variant}: {len(warps)} warps traced on SM 0")
        for name, value in overlap(clocks, warps).items():
            print(f"  {name:34} {value:7.3f}")
