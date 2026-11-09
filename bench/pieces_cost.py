"""M4.31: what splitting a call by heads costs, at WAN's bench shape (34020 tokens, 40 heads, fp16 in ComfyUI's
layout), by preset: forced pieces of 40, 20, 10, 8 and 5 heads, alternating rounds in one process.
M4.39: any shape, whole against pieces of one head, e.g. an image model's call that the reserve used to split.

usage: uv run python bench/pieces_cost.py [tokens heads batch]
"""

import statistics
import sys

import torch

import unica
from unica.__main__ import time_ms

ROUNDS = 4


def main(tokens: int = 34020, heads: int = 40, batch: int = 1, pieces: tuple = (40, 20, 10, 8, 5)) -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(batch, tokens, heads, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
               for _ in range(3))
    real = unica.heads_per_call
    times = {(name, n): [] for name in unica.PRESETS for n in pieces}
    for r in range(ROUNDS):
        for name, n in times if r % 2 == 0 else reversed(list(times)):
            unica.heads_per_call = lambda *args, n=n: n
            times[name, n].append(time_ms(lambda name=name: unica.attention(q, k, v, **unica.PRESETS[name]), runs=5))
    unica.heads_per_call = real
    print(f"{batch} x {tokens} tokens, {heads} heads, fp16 in; median of {ROUNDS} rounds x median of 5")
    for name in unica.PRESETS:
        whole = statistics.median(times[name, pieces[0]])
        print(f"  {name:12}" + "".join(f"  {n:2} heads {statistics.median(times[name, n]):6.2f} ms "
                                       f"({statistics.median(times[name, n]) - whole:+6.2f} ms)" for n in pieces))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        tokens, heads, batch = map(int, sys.argv[1:])
        main(tokens, heads, batch, (heads, 1))
    else:
        main()
