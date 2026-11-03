"""
UnicaAttention for ComfyUI-WanVideoWrapper: a node that runs the self-attention of a WAN model on unica.

    WanVideoModelLoader -> Unica Attention (WanVideo) -> WanVideoSampler

The node clones the model and sets WanVideoWrapper's attention-mode override to "unica". WanVideoWrapper
passes that mode down to the self-attention of every block; the `attention` function its model.py
imported is wrapped once, so "unica" goes to unica.attention and every other mode to the original. Cross-
attention to the text keeps the loader's attention_mode.

measure_every > 0 also compares, on every n-th self-attention call, unica and SageAttention 2 against FP32
attention on the first measure_heads heads of the real Q, K, V, and appends the errors to
build/wan_attention_errors.csv in the UnicaAttention repository.

Install: a directory junction from ComfyUI/custom_nodes/UnicaAttention to this folder. unica's cubins must
already be built (run the tests once): ComfyUI's shell has no nvcc.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

import unica  # noqa: E402
from emulate import errors, reference  # noqa: E402

MODE = "unica"
ERRORS_CSV = REPO / "build" / "wan_attention_errors.csv"
CSV_FIELDS = ["time", "call", "batch", "tokens", "keys", "heads_measured", "unica_rel_l2", "unica_cos",
              "sage2_rel_l2", "sage2_cos", "unica_ms", "sage2_ms"]
log = logging.getLogger("UnicaAttention")


def unica_nhd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """unica.attention on WanVideoWrapper's layout [batch, tokens, heads, 128], any float dtype in and out.
    One copy each way: to bf16 [batch, heads, tokens, 128] and back."""
    as_bhnd = lambda x: x.transpose(1, 2).to(torch.bfloat16, memory_format=torch.contiguous_format)  # noqa: E731
    out = unica.attention(as_bhnd(q), as_bhnd(k), as_bhnd(v))
    return out.transpose(1, 2).to(q.dtype).contiguous()


class Measurement:
    """Every `every`-th call: unica and SageAttention 2 against FP32 on the first `heads` heads, to the CSV."""

    def __init__(self) -> None:
        self.every, self.heads, self.calls = 0, 0, 0

    def __call__(self, q, k, v, out) -> None:
        self.calls += 1
        if self.every <= 0 or self.calls % self.every:
            return
        from sageattention import sageattn

        h = min(self.heads, q.shape[2])
        sub = [x[:, :, :h].contiguous() for x in (q, k, v)]
        ref = reference(*(x.transpose(1, 2) for x in sub))  # FP32, [batch, heads, tokens, 128]
        start = time.perf_counter()
        sage = sageattn(*(x.clone() for x in sub), tensor_layout="NHD")
        torch.cuda.synchronize()
        sage_ms = (time.perf_counter() - start) * 1e3
        start = time.perf_counter()
        mine = unica_nhd(*sub)
        torch.cuda.synchronize()
        unica_ms = (time.perf_counter() - start) * 1e3
        e_unica = errors(mine.transpose(1, 2).float(), ref)
        e_sage = errors(sage.transpose(1, 2).float(), ref)
        row = {"time": time.strftime("%H:%M:%S"), "call": self.calls, "batch": q.shape[0], "tokens": q.shape[1],
               "keys": k.shape[1], "heads_measured": h, "unica_rel_l2": e_unica["rel_l2"], "unica_cos": e_unica["cos"],
               "sage2_rel_l2": e_sage["rel_l2"], "sage2_cos": e_sage["cos"], "unica_ms": unica_ms, "sage2_ms": sage_ms}
        ERRORS_CSV.parent.mkdir(exist_ok=True)
        new = not ERRORS_CSV.exists()
        with ERRORS_CSV.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        log.info("call %d, %d tokens: rel L2 unica %.3f %%, SageAttention 2 %.3f %%", self.calls, q.shape[1],
                 100 * e_unica["rel_l2"], 100 * e_sage["rel_l2"])


measurement = Measurement()


def wrap_attention() -> int:
    """Wraps `attention` in every loaded WanVideoWrapper model module, once. Returns how many it wrapped."""
    wrapped = 0
    for name, module in list(sys.modules.items()):
        original = getattr(module, "attention", None)
        if not name.endswith("wanvideo.modules.model") or original is None or getattr(original, "unica", False):
            continue

        def attention(q, k, v, *args, attention_mode="sdpa", original=original, **kwargs):
            if attention_mode != MODE:
                return original(q, k, v, *args, attention_mode=attention_mode, **kwargs)
            out = unica_nhd(q, k, v)
            measurement(q, k, v, out)
            return out

        attention.unica = True
        module.attention = attention
        wrapped += 1
    return wrapped


class UnicaAttentionWanVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off: the model as it came, for A/B runs"}),
                "measure_every": ("INT", {"default": 0, "min": 0, "max": 100000,
                                          "tooltip": "Compare with SageAttention 2 and FP32 every n-th self-attention call (0: off)"}),
                "measure_heads": ("INT", {"default": 2, "min": 1, "max": 40,
                                          "tooltip": "Heads per measured call: the FP32 reference is slow"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "UnicaAttention"
    DESCRIPTION = "Self-attention of the WAN model on UnicaAttention (INT8 Q·Kᵀ, u8 × s8 P·V, sm_120)"

    def patch(self, model, enabled, measure_every, measure_heads):
        if not enabled:
            return (model,)
        if not wrap_attention() and not any(getattr(getattr(m, "attention", None), "unica", False)
                                            for n, m in sys.modules.items() if n.endswith("wanvideo.modules.model")):
            raise RuntimeError("ComfyUI-WanVideoWrapper's wanvideo.modules.model is not loaded")
        measurement.every, measurement.heads = measure_every, measure_heads
        patched = model.clone()
        patched.model_options["transformer_options"]["attention_mode_override"] = {
            "mode": MODE, "start_step": 0, "end_step": 10000, "verbose": False}
        return (patched,)


NODE_CLASS_MAPPINGS = {"UnicaAttentionWanVideo": UnicaAttentionWanVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"UnicaAttentionWanVideo": "Unica Attention (WanVideo)"}
