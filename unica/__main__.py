"""python -m unica: whether this machine can run unica, then each preset (PRESETS) against the exact attention on
random inputs, with its time and PyTorch's SDPA's for scale. Exit code 1 if it cannot run or a preset is off."""

import statistics
import sys

import torch

import unica
from unica.check import UNSUPPORTED, compatibility

SHAPE = (1, 16, 4096, 128)  # [batch, heads, tokens, 128]
# rel L2 against FP32 that a working kernel stays under on N(0, 1) inputs, the worst case for FP4: there the softmax is
# nearly flat, and the fast mode's P·V alone loses 13.7 % (LOG M4.2); on real WAN it is 7.45 % (M4.21). A broken
# kernel lands near 100 % or NaN. Measured on the RTX 5080 (LOG M4.27): 1.58, 1.67, 1.66, 19.26 %.
ERROR_LIMITS = {"per_token": 0.03, "per_channel": 0.03, "outliers": 0.03, "fast": 0.25}


def time_ms(function, runs: int = 10) -> float:
    function()
    times = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> int:
    status = compatibility()
    print(status)
    if status.status == UNSUPPORTED:
        return 1
    torch.manual_seed(0)
    q, k, v = (torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    exact = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    sdpa = time_ms(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    print(f"{list(SHAPE)} bf16, PyTorch SDPA {sdpa:.2f} ms (its backend depends on the build and the platform)")
    failed = 0
    for name, preset in unica.PRESETS.items():
        error = ((unica.attention(q, k, v, **preset).float() - exact).norm() / exact.norm()).item()
        ms = time_ms(lambda preset=preset: unica.attention(q, k, v, **preset))
        good = error < ERROR_LIMITS[name]  # also False for NaN
        failed += not good
        print(f"  {name:12} rel L2 {100 * error:5.2f} % (limit {100 * ERROR_LIMITS[name]:.0f} %)  {ms:6.2f} ms  "
              f"{'ok' if good else 'FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
