"""M4.45 A2: Moss's self-attention call (batch 1, 4131 tokens, 24 heads, fp16 in ComfyUI's [batch, tokens, heads, 128]),
eager against a CUDA graph replay of the same call, and SDPA. For each: the GPU time per call (a loop the CPU runs
ahead of, CUDA events around it) and the CPU time to issue one call (no sync in the loop). A call is CPU-bound where
the second is larger than the first; only there can a graph help.

    uv run python bench/graph_call.py [tokens=4131] [heads=24] [mode=per_token]
"""

import statistics
import sys
import time

import torch
import torch.nn.functional as F

import unica

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 4131
HEADS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
MODE = sys.argv[3] if len(sys.argv) > 3 else "per_token"
CALLS, ROUNDS = 200, 5


def gpu_and_cpu_ms(call) -> tuple[float, float]:
    """GPU time per call over CALLS back to back, and the CPU time spent issuing them, per call."""
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.perf_counter()
    for _ in range(CALLS):
        call()
    issue = (time.perf_counter() - t0) * 1e3 / CALLS
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / CALLS, issue


def graphed(call):
    """call captured once into a CUDA graph (its inputs and outputs stay where they are); returns the replay."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm-up outside the capture: cubins load, caches fill
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph.replay


def main():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, TOKENS, HEADS, 128, device="cuda", dtype=torch.float16) for _ in range(3))
    bhsd = [x.transpose(1, 2) for x in (q, k, v)]
    preset = unica.PRESETS[MODE]
    kernel = preset.get("kernel", unica.DEFAULT_KERNEL)
    tile = {key: value for key, value in preset.items() if key != "kernel"}
    inputs = unica.kernel_inputs(*(x.bfloat16() for x in bhsd), kernel, **tile)
    bf16 = [x.bfloat16() for x in bhsd]
    calls = {
        "unica eager": lambda: unica.attention(*bhsd, **preset),
        "unica graph": graphed(lambda: unica.attention(*bhsd, **preset)),
        "kernel alone (bf16, inputs given)": lambda: unica.attention(*bf16, kernel, inputs, **tile),
        "SDPA eager": lambda: F.scaled_dot_product_attention(*bhsd),
    }
    print(f"{TOKENS} tokens, {HEADS} heads, {MODE}; {ROUNDS} rounds × {CALLS} calls, medians per call")
    results = {name: [] for name in calls}
    for r in range(ROUNDS):
        for name in (calls if r % 2 == 0 else reversed(list(calls))):
            results[name].append(gpu_and_cpu_ms(calls[name]))
    for name, rows in results.items():
        gpu, cpu = statistics.median(g for g, _ in rows), statistics.median(c for _, c in rows)
        print(f"  {name:36} GPU {gpu:7.4f} ms   CPU issue {cpu:7.4f} ms   {'CPU-bound' if cpu > gpu else 'GPU-bound'}")


if __name__ == "__main__":
    main()
