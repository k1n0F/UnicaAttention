"""M4.28: one WAN attention call as ComfyUI makes it, by each path the bench compares, and its parts.

usage: uv run python bench/wan_call.py                    times every path at the bench's 34020 tokens
       uv run python bench/wan_call.py --memory 118320   peak memory of one self-attention call per path (M4.31)

WAN 2.2 at the bench's 0.4 MP and 81 frames: self-attention q, k, v [1, 34020, 40 · 128], cross-attention to 512 text
tokens; fp16, as ComfyUI's WAN hands them over. Paths:
- sdpa: ComfyUI's attention_pytorch, with its backend priority (flash, cudnn, efficient);
- sage2, sage3: KJNodes' attention_sage (NHD views, SageAttention 2 "auto" / 3 on transposes);
- unica modes: the native node's override, as in a render; and the same kernel on bf16 input, without the conversion.
A render has 160 self- and 160 cross-attention calls (40 blocks, 4 steps). Alternating rounds in one process.
"""

import statistics
import sys
from pathlib import Path

import torch
from sageattention import sageattn
from sageattn3 import sageattn3_blackwell
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comfyui  # noqa: E402
import unica  # noqa: E402
from unica.__main__ import time_ms  # noqa: E402

TOKENS, TEXT, HEADS, DIM = 34020, 512, 40, 128
CALLS = 160  # per render, each of self- and cross-attention
ROUNDS = 3
PRIORITY = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def sdpa(q, k, v):  # comfy/ldm/modules/attention.py attention_pytorch
    b = q.shape[0]
    q, k, v = (x.view(b, -1, HEADS, DIM).transpose(1, 2) for x in (q, k, v))
    with sdpa_kernel(PRIORITY, set_priority=True):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, -1, HEADS * DIM)


def sage2(q, k, v):  # KJNodes attention_sage, "auto"
    b = q.shape[0]
    q, k, v = (x.view(b, -1, HEADS, DIM) for x in (q, k, v))
    return sageattn(q, k, v, tensor_layout="NHD").reshape(b, -1, HEADS * DIM)


def sage3(q, k, v):  # KJNodes attention_sage, "sageattn3"
    b = q.shape[0]
    q, k, v = (x.view(b, -1, HEADS, DIM).transpose(1, 2) for x in (q, k, v))
    return sageattn3_blackwell(q, k, v).transpose(1, 2).reshape(b, -1, HEADS * DIM)


def node(mode):  # the native node's override with its kernel chosen, as a render runs it
    settings = comfyui.choose_kernel("per_token" if mode == "fast" else mode, "fast" if mode == "fast" else "accurate",
                                     (unica.VRAM_RESERVE or 0) // 2**20)
    override = comfyui.native_override(None, settings)

    def call(q, k, v):
        return override(None, q, k, v, heads=HEADS)
    return call


def kernel_bf16(mode):  # the same kernel on bf16 views in the same layout: no conversion either way
    preset = unica.PRESETS[mode]

    def call(q, k, v):
        b = q.shape[0]
        q, k, v = (x.view(b, -1, HEADS, DIM).transpose(1, 2) for x in (q, k, v))
        return unica.attention(q, k, v, **preset).transpose(1, 2).reshape(b, -1, HEADS * DIM)
    return call


def memory(tokens: int) -> None:
    """One self-attention call per path: its peak memory above what was allocated before it (q, k, v themselves), and
    its time. At "wan stable work"'s hires pass (118320 tokens) one copy of q, k or v is 1.21 GB."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, tokens, HEADS * DIM, device="cuda", dtype=torch.float16) for _ in range(3))
    paths = {"sdpa": sdpa, "sage2": sage2, "sage3": sage3,
             **{f"unica {m}": node(m) for m in ("per_channel", "per_token", "fast")}}
    print(f"one self-attention call, {tokens} tokens, {HEADS} heads, fp16; one input {q.nbytes / 2**30:.2f} GiB")
    for name, call in paths.items():
        call(q, k, v)  # cubins, autotuning, the allocator's first growth
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = call(q, k, v)
        end.record()
        end.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        print(f"  {name:20} peak {peak / 2**30:5.2f} GiB above the inputs   {start.elapsed_time(end):8.1f} ms")
        del out


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--memory":
        return memory(int(sys.argv[2]))
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, TOKENS, HEADS * DIM, device="cuda", dtype=torch.float16) for _ in range(3))
    text_k, text_v = (torch.randn(1, TEXT, HEADS * DIM, device="cuda", dtype=torch.float16) for _ in range(2))
    qb, kb, vb = (x.bfloat16() for x in (q, k, v))
    calls = {
        "self: sdpa": lambda: sdpa(q, k, v),
        "self: sage2": lambda: sage2(q, k, v),
        "self: sage3": lambda: sage3(q, k, v),
        **{f"self: unica {m}": (lambda f=node(m): f(q, k, v)) for m in ("per_channel", "per_token", "fast")},
        **{f"self: kernel {m} on bf16": (lambda f=kernel_bf16(m): f(qb, kb, vb))
           for m in ("per_channel", "per_token", "fast")},
        "self: fp16 -> bf16, 3 inputs + output": lambda: [x.to(torch.bfloat16) for x in (q, k, v, q)],
        "cross: sdpa": lambda: sdpa(q, text_k, text_v),
        "cross: sage2": lambda: sage2(q, text_k, text_v),
        "cross: sage3": lambda: sage3(q, text_k, text_v),
    }
    times = {name: [] for name in calls}
    for r in range(ROUNDS):
        for name, call in calls.items() if r % 2 == 0 else reversed(calls.items()):
            times[name].append(time_ms(call, runs=5))
    print(f"WAN call: {TOKENS} tokens, {HEADS} heads, fp16; median of {ROUNDS} rounds x median of 5; "
          f"per render = {CALLS} calls")
    for name, t in times.items():
        ms = statistics.median(t)
        print(f"  {name:40} {ms:7.2f} ms   {ms * CALLS / 1000:5.2f} s per render")


if __name__ == "__main__":
    main()
