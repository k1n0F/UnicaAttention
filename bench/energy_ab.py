"""attention_int8pv_tma build settings side by side: the kernel alone on inputs quantized once, in alternating rounds,
with the energy of each variant's calls from NVML's counter, the mean power and the SM clock (M4.51). Under full
load the card runs at its power limit (M3.22): there less energy per call is what buys time.

    uv run python bench/energy_ab.py [--tokens 34020] [--heads 40] [--rounds 10] VARIANT [VARIANT ...]

A VARIANT is "" (the defaults) or settings as in bench/compare.py: "PRODUCER=1,WAIT_HINT=1000".
"""

import argparse
import ctypes
import os
import statistics

import torch
from compare import parse_variant

import unica

KERNEL = "attention_int8pv_tma"
CALLS = 10  # per variant and round
NVML_CLOCK_SM = 1


class Nvml:
    """The three NVML calls this needs, through ctypes: no pynvml dependency."""

    def __init__(self):
        self.lib = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(torch.cuda.current_device(), ctypes.byref(self.handle)) == 0

    def energy_mj(self) -> int:
        value = ctypes.c_ulonglong()
        assert self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(value)) == 0
        return value.value

    def sm_clock_mhz(self) -> int:
        value = ctypes.c_uint()
        assert self.lib.nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_SM, ctypes.byref(value)) == 0
        return value.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--tokens", type=int, default=34020)
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()

    nvml = Nvml()
    tiles = {spec: parse_variant(f"{KERNEL}:{spec}")[1] for spec in args.variants}
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, args.heads, args.tokens, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    inputs = {spec: unica.kernel_inputs(q, k, v, KERNEL, **tile) for spec, tile in tiles.items()}
    references = {}  # cubins load; and every variant must give the output of the first one with its V scale
    for spec, tile in tiles.items():
        out = unica.attention(q, k, v, KERNEL, inputs[spec], **tile)
        assert torch.equal(references.setdefault(tile.get("V_TOKEN", 0), out), out), spec

    ms, joules, clocks = ({spec: [] for spec in tiles} for _ in range(3))
    for r in range(args.rounds):
        for spec in (args.variants if r % 2 == 0 else args.variants[::-1]):
            tile = tiles[spec]
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            energy = nvml.energy_mj()
            start.record()
            for _ in range(CALLS):
                unica.attention(q, k, v, KERNEL, inputs[spec], **tile)
            end.record()
            clocks[spec].append(nvml.sm_clock_mhz())  # while the calls still run
            end.synchronize()
            joules[spec].append((nvml.energy_mj() - energy) / 1e3)
            ms[spec].append(start.elapsed_time(end) / CALLS)

    first = args.variants[0]
    # energy summed over all rounds: NVML's counter moves in steps that a short round can miss
    per_call = {spec: sum(joules[spec]) / (args.rounds * CALLS) for spec in args.variants}
    base_ms, base_j = statistics.median(ms[first]), per_call[first]
    print(f"{KERNEL}, 1 × {args.tokens} × {args.heads} × 128, kernel alone, {args.rounds} alternating rounds × "
          f"{CALLS} calls; medians per call")
    for spec in args.variants:
        t, j = statistics.median(ms[spec]), per_call[spec]
        print(f"  {spec or 'defaults':30} {t:7.2f} ms ({t / base_ms:.3f})  {j:6.2f} J ({j / base_j:.3f})  "
              f"{j / t * 1e3:5.0f} W  {statistics.median(clocks[spec]):5.0f} MHz")


if __name__ == "__main__":
    main()
