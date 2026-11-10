"""M4.47: comfy_kitchen's INT8 attention (NVIDIA, in ComfyUI's venv) against unica, head to head. Run it with ComfyUI's
Python, which has comfy_kitchen; unica comes from this repository:

    <ComfyUI>/.venv-13/Scripts/python.exe bench/kitchen.py [time] [error]

time: Moss's call (1 × 4131 × 24 × 128) and WAN's (1 × 34,020 × 40), bf16 in ComfyUI's [batch, tokens, heads, 128]
seen as [batch, heads, tokens, 128], the kernels in alternating rounds. error: rel L2 against FP32 on the dumps
(build/dumps/, 6 calls per model), with SageAttention 2's error from the run that dumped them.
"""

# pyright: reportMissingImports=false
# comfy_kitchen exists only in ComfyUI's Python

import os
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "bench")]

import comfy_kitchen  # noqa: E402
from emulate import errors, reference  # noqa: E402

import unica  # noqa: E402

SIGNATURES = {"Moss": (4131, 24), "WAN": (34020, 40)}
MODELS = ["wan_call", "zimage_call", "ltx23_call", "qwen_edit_call", "moss_call"]
ROUNDS = 6
PATHS = {
    "comfy_kitchen int8": lambda q, k, v: comfy_kitchen.int8_attention(q, k, v),
    "unica per_token": lambda q, k, v: unica.attention(q, k, v, **unica.PRESETS["per_token"]),
    "unica per_channel": lambda q, k, v: unica.attention(q, k, v, **unica.PRESETS["per_channel"]),
}


def median_ms(call, runs: int) -> float:
    times = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def time_paths():
    for name, (tokens, heads) in SIGNATURES.items():
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                   for _ in range(3))
        runs = 50 if tokens < 10000 else 10
        for call in PATHS.values():  # warm-up: cubins, caches
            call(q, k, v)
        times = {p: [] for p in PATHS}
        for r in range(ROUNDS):
            for p in (PATHS if r % 2 == 0 else reversed(list(PATHS))):
                times[p].append(median_ms(lambda p=p, qkv=(q, k, v): PATHS[p](*qkv), runs))
        base = statistics.median(times["unica per_token"])
        print(f"{name}: 1 × {tokens} × {heads} × 128, bf16, {ROUNDS} rounds × median of {runs}", flush=True)
        for p, t in times.items():
            ms = statistics.median(t)
            print(f"  {p:20} {ms:8.3f} ms  ({ms / base:.3f} × unica per_token)", flush=True)
        del q, k, v
        torch.cuda.empty_cache()


def error_paths():
    files = sorted(os.listdir(REPO / "build" / "dumps"))
    print(f"{'call':26} {'kitchen':>8} {'per_token':>10} {'per_chan':>9} {'SA2':>7}  (rel L2 %, against FP32)", flush=True)
    rows = {}
    for prefix in MODELS:
        mine = [f for f in files if f.startswith(prefix)]
        for f in dict.fromkeys(mine[round(i * (len(mine) - 1) / 5)] for i in range(6)):
            dump = torch.load(REPO / "build" / "dumps" / f, map_location="cuda")
            q, k, v = (dump[x].bfloat16() for x in "qkv")
            ref = reference(q.float(), k.float(), v.float())
            e = [errors(PATHS[p](q, k, v).float(), ref)["rel_l2"] for p in PATHS] + [dump["row"]["sage2_rel_l2"]]
            rows.setdefault(prefix, []).append(e)
            print(f"{f[:26]:26} " + " ".join(f"{100 * x:8.3f}" for x in e), flush=True)
    print("\nmedians per model: kitchen / per_token / per_channel / SA2")
    for prefix, r in rows.items():
        print(f"  {prefix:15} " + " / ".join(f"{100 * statistics.median(x[i] for x in r):.3f}" for i in range(4)))


if __name__ == "__main__":
    parts = sys.argv[1:] or ["time", "error"]
    print(f"torch {torch.__version__}, comfy_kitchen {getattr(comfy_kitchen, '__version__', '?')}", flush=True)
    if "time" in parts:
        time_paths()
    if "error" in parts:
        error_paths()
