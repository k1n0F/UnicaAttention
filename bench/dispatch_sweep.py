"""M4.55: T₈ / T₄ over a grid of shapes, 4 (as the preset) against 8 computing warps, in two power regimes,
with what the cost model predicts: F₈ = ⌈w₈⌉ / w₈, w₈ = blocks of 128 rows / SMs.

- sustained: a second of calls first, then alternating rounds of ≥ 250 ms each; the card sits at its power limit, as
  in a sampler's loop. Energy per call from NVML as well.
- burst: one call after 0.3 s idle, as a lone call sees it.

    uv run python bench/dispatch_sweep.py [mode=per_channel]
"""

import math
import statistics
import sys
import time

import torch
from energy_ab import Nvml

import unica

KERNEL = "attention_int8pv_tma"
MODE = sys.argv[1] if len(sys.argv) > 1 else "per_channel"
BASE = {"V_TOKEN": 1} if MODE == "per_token" else {}
# 4 warps as the preset runs them: per channel with the producer, per token without (it spills there, M4.49)
SETTINGS = {4: {**BASE, "PRODUCER": 0 if BASE else 1}, 8: {**BASE, "PRODUCER": 1, "WARPS_PER_BLOCK": 8}}
TOKENS = (3874, 4128, 8393, 14080, 34020)
HEADS = (24, 40, 60)
ROUNDS, ROUND_MS, WARM_S, IDLE_S = 6, 250, 1.0, 0.3


def calls(q, k, v, inputs, tile, n):
    for _ in range(n):
        unica.attention(q, k, v, KERNEL, inputs, **tile)


def sustained(q, k, v, inputs, nvml):
    one = {w: 0.0 for w in SETTINGS}
    for w, tile in SETTINGS.items():  # cubins, and a first estimate of a call's time
        torch.cuda.synchronize()
        start = time.perf_counter()
        calls(q, k, v, inputs[w], tile, 3)
        torch.cuda.synchronize()
        one[w] = (time.perf_counter() - start) / 3
    n = {w: max(3, math.ceil(ROUND_MS / 1e3 / one[w])) for w in SETTINGS}
    warm_end = time.perf_counter() + WARM_S
    while time.perf_counter() < warm_end:
        calls(q, k, v, inputs[4], SETTINGS[4], n[4])
        torch.cuda.synchronize()
    ms, joules = {w: [] for w in SETTINGS}, {w: [] for w in SETTINGS}
    for r in range(ROUNDS):
        for w in ((4, 8) if r % 2 == 0 else (8, 4)):
            torch.cuda.synchronize()
            energy = nvml.energy_mj()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            calls(q, k, v, inputs[w], SETTINGS[w], n[w])
            end.record()
            end.synchronize()
            ms[w].append(start.elapsed_time(end) / n[w])
            joules[w].append((nvml.energy_mj() - energy) / 1e3 / n[w])
    t = {w: statistics.median(ms[w]) for w in SETTINGS}
    e = {w: sum(joules[w]) for w in SETTINGS}  # sums: NVML's counter moves in steps that a short round can miss
    return t, e


def burst(q, k, v, inputs):
    ms = {w: [] for w in SETTINGS}
    for r in range(ROUNDS):
        for w in ((4, 8) if r % 2 == 0 else (8, 4)):
            torch.cuda.synchronize()
            time.sleep(IDLE_S)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            calls(q, k, v, inputs[w], SETTINGS[w], 1)
            end.record()
            end.synchronize()
            ms[w].append(start.elapsed_time(end))
    return {w: statistics.median(ms[w]) for w in SETTINGS}


def main():
    nvml = Nvml()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{MODE}, T₈ / T₄ (both PRODUCER), {sms} SMs; sustained: {ROUNDS} alternating rounds of ≥ {ROUND_MS} ms after "
          f"{WARM_S} s of calls; burst: one call after {IDLE_S} s idle")
    print(f"{'heads × tokens':>16} {'w₈':>7} {'⌈w₈⌉/w₈':>8} {'T₄ ms':>8} {'sust. T₈/T₄':>12} {'E₈/E₄':>7} "
          f"{'burst T₈/T₄':>12} {'model sust.':>12} {'model burst':>12}")
    torch.manual_seed(0)
    for tokens in TOKENS:
        for heads in HEADS:
            q, k, v = (torch.randn(1, heads, tokens, 128, device="cuda").bfloat16() for _ in range(3))
            inputs = {w: unica.kernel_inputs(q, k, v, KERNEL, **tile) for w, tile in SETTINGS.items()}
            w8 = heads * math.ceil(tokens / 128) / sms
            f8 = math.ceil(w8) / w8
            t, e = sustained(q, k, v, inputs, nvml)
            b = burst(q, k, v, inputs)
            energy = e[8] / e[4] if e[4] else math.nan
            print(f"{heads:7} × {tokens:6} {w8:7.2f} {f8:8.3f} {t[4]:8.3f} {t[8] / t[4]:12.3f} {energy:7.3f} "
                  f"{b[8] / b[4]:12.3f} {0.975 * f8:12.3f} {1.02 * f8:12.3f}", flush=True)
            del q, k, v, inputs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
