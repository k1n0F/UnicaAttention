"""
UnicaAttention for ComfyUI: nodes that run the self-attention of a WAN model on unica.

    WanVideoModelLoader -> Unica Attention (WanVideo) -> WanVideoSampler          (ComfyUI-WanVideoWrapper)
    model -> PathchSageAttentionKJ -> Unica Attention (native) -> sampler        (ComfyUI's own WAN model)

Unica Attention (WanVideo) clones the model and sets WanVideoWrapper's attention-mode override to "unica".
WanVideoWrapper passes that mode down to the self-attention of every block; the `attention` function its
model.py imported is wrapped once, so "unica" goes to unica.attention and every other mode to the original.
Cross-attention to the text keeps the loader's attention_mode.

Unica Attention (native) puts an `optimized_attention_override` on the model, the hook every attention call of
ComfyUI's own models goes through. Self-attention with head_dim 128 (as many keys as queries, no mask) goes to
unica; every other call goes to the override that was there before, so placed after PathchSageAttentionKJ the
cross-attention stays on SageAttention. Placed before it, it would be overwritten.

measure_every > 0 also compares, on every n-th self-attention call, unica, SageAttention 2 and (if installed)
SageAttention 3 against FP32 attention on the first measure_heads heads of the real Q, K, V, and appends the
errors to build/wan_attention_errors.csv in the UnicaAttention repository.

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
from emulate import TODAY, emulate, errors, reference  # noqa: E402

MODE = "unica"
ERRORS_CSV = REPO / "build" / "wan_attention_errors_m4.12.csv"  # each set of columns its own file (M5.3, M4.6, M4.7)
# attention_fp4pv as the emulator computes it (LOG M4.8, tests/test_fp4.py)
FP4PV = TODAY | dict(p_fmt="nvfp4", v_fmt="nvfp4", p_local=False, p_mul=1024.0, p_outer=1.0, v_outer="channel",
                     fp4_slots=True)
# Candidates, emulated on the same real Q, K, V (today's kernel is measured itself, as "unica"). M4.12 (H9): which
# layers tolerate FP4 in Q·Kᵀ on top of FP4 P·V
EMULATED = {
    "emu_hadamard": TODAY | dict(rotate_qk=128),
    "emu_fp4pv": FP4PV,
    "emu_fp4qk": TODAY | dict(qk="nvfp4"),
    "emu_fp4_both": FP4PV | dict(qk="nvfp4"),
}
CSV_FIELDS = ["time", "node", "call", "batch", "tokens", "keys", "heads_measured", "unica_rel_l2", "unica_cos",
              "sage2_rel_l2", "sage2_cos", "sage3_rel_l2", "sage3_cos",
              *(f"{name}_rel_l2" for name in EMULATED), "q_channel_spread", "k_channel_spread", "v_offset",
              "unica_ms", "sage2_ms", "sage3_ms"]
HEAD_DIM = 128
log = logging.getLogger("UnicaAttention")


def unica_nhd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """unica.attention on WanVideoWrapper's layout [batch, tokens, heads, 128], any float dtype in and out.
    One copy each way: to bf16 [batch, heads, tokens, 128] and back."""
    as_bhnd = lambda x: x.transpose(1, 2).to(torch.bfloat16, memory_format=torch.contiguous_format)  # noqa: E731
    out = unica.attention(as_bhnd(q), as_bhnd(k), as_bhnd(v))
    return out.transpose(1, 2).to(q.dtype).contiguous()


def timed(function, *args):
    """function(*args) and its time in ms, synchronized."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = function(*args)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1e3


def sage3_bhnd():
    """SageAttention 3 on [batch, heads, tokens, 128], or None if it is not installed. It modifies k in place, so
    it gets a copy."""
    try:
        from sageattn3 import sageattn3_blackwell
    except ImportError:
        return None
    return lambda q, k, v: sageattn3_blackwell(q, k.clone(), v, per_block_mean=False)


def channel_spread(x: torch.Tensor) -> float:
    """How uneven the channels of x [batch, heads, tokens, 128] are: the largest channel's RMS over the median
    channel's, per head, averaged. 1 = even; what makes a Hadamard rotation pay for INT8 (LOG M4.5)."""
    rms = x.float().pow(2).mean(-2).sqrt()  # [batch, heads, 128]
    return (rms.amax(-1) / rms.median(-1).values).mean().item()


def v_offset(v: torch.Tensor) -> float:
    """How much of V is a constant per channel: RMS of the channel means over RMS of V. 0 = centred; what makes
    centring V pay for FP4 (LOG M4.4)."""
    v = v.float()
    return (v.mean(-2).pow(2).mean().sqrt() / v.pow(2).mean().sqrt()).item()


class Measurement:
    """Every `every`-th call of a node: unica, SageAttention 2 and 3 against FP32 on the first `heads` heads."""

    def __init__(self, node: str) -> None:
        self.node, self.every, self.heads, self.calls = node, 0, 0, 0

    def __call__(self, q, k, v) -> None:
        """q, k, v in [batch, tokens, heads, 128]."""
        self.calls += 1
        if self.every <= 0 or self.calls % self.every:
            return
        from sageattention import sageattn

        h = min(self.heads, q.shape[2])
        sub = [x[:, :, :h].contiguous() for x in (q, k, v)]
        bhnd = [x.transpose(1, 2).contiguous() for x in sub]
        ref = reference(*bhnd)  # FP32, [batch, heads, tokens, 128]
        row = {"time": time.strftime("%H:%M:%S"), "node": self.node, "call": self.calls, "batch": q.shape[0],
               "tokens": q.shape[1], "keys": k.shape[1], "heads_measured": h}
        mine, row["unica_ms"] = timed(unica_nhd, *sub)
        sage2, row["sage2_ms"] = timed(lambda *x: sageattn(*(t.clone() for t in x), tensor_layout="NHD"), *sub)
        results = {"unica": mine.transpose(1, 2), "sage2": sage2.transpose(1, 2)}
        if (sage3 := sage3_bhnd()) is not None:
            results["sage3"], row["sage3_ms"] = timed(sage3, *bhnd)
        for name, out in results.items():
            e = errors(out.float(), ref)
            row[f"{name}_rel_l2"], row[f"{name}_cos"] = e["rel_l2"], e["cos"]
        for name, config in EMULATED.items():
            row[f"{name}_rel_l2"] = errors(emulate(*bhnd, **config), ref)["rel_l2"]
        row["q_channel_spread"], row["k_channel_spread"] = channel_spread(bhnd[0]), channel_spread(bhnd[1])
        row["v_offset"] = v_offset(bhnd[2])
        ERRORS_CSV.parent.mkdir(exist_ok=True)
        new = not ERRORS_CSV.exists()
        with ERRORS_CSV.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        log.info("%s call %d, %d tokens: rel L2 unica %.3f %%, SageAttention 2 %.3f %%%s; emulated %s; "
                 "channel spread q %.2f k %.2f, v offset %.3f", self.node, self.calls,
                 q.shape[1], 100 * row["unica_rel_l2"], 100 * row["sage2_rel_l2"],
                 f", SageAttention 3 {100 * row['sage3_rel_l2']:.3f} %" if "sage3_rel_l2" in row else "",
                 ", ".join(f"{name[4:]} {100 * row[f'{name}_rel_l2']:.3f} %" for name in EMULATED),
                 row["q_channel_spread"], row["k_channel_spread"], row["v_offset"])


measurement = Measurement("WanVideo")
native_measurement = Measurement("native")


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
            measurement(q, k, v)
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


def native_override(previous):
    """An optimized_attention_override for ComfyUI's own models. ComfyUI calls it as (func, q, k, v, heads=, ...):
    q, k, v are [batch, tokens, heads · dim], or [batch, heads, tokens, dim] with skip_reshape. Self-attention
    with head_dim 128, as many keys as queries, no mask and the default scale goes to unica; everything else
    to `previous` (the override that was on the model), or to func."""

    def override(func, q, k, v, *args, **kwargs):
        heads = kwargs.get("heads")
        if args or heads is None or kwargs.get("mask") is not None or not q.is_cuda:  # anything unusual: as before
            return previous(func, q, k, v, *args, **kwargs) if previous else func(q, k, v, *args, **kwargs)
        skip_reshape, skip_output_reshape = kwargs.get("skip_reshape", False), kwargs.get("skip_output_reshape", False)
        dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
        tokens = (lambda x: x.shape[2]) if skip_reshape else (lambda x: x.shape[1])
        if dim != HEAD_DIM or tokens(k) != tokens(q) or kwargs.get("scale") not in (None, HEAD_DIM**-0.5):
            return previous(func, q, k, v, **kwargs) if previous else func(q, k, v, **kwargs)
        # [batch, tokens, heads, 128], WanVideoWrapper's layout, as views
        nhd = [x.transpose(1, 2) if skip_reshape else x.view(x.shape[0], x.shape[1], heads, dim) for x in (q, k, v)]
        out = unica_nhd(*nhd)
        native_measurement(*nhd)
        return out.transpose(1, 2) if skip_output_reshape else out.view(out.shape[0], out.shape[1], heads * dim)

    override.unica = True
    return override


class UnicaAttentionNative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off: the model as it came, for A/B runs"}),
                "measure_every": ("INT", {"default": 0, "min": 0, "max": 100000,
                                          "tooltip": "Compare with SageAttention 2/3 and FP32 every n-th self-attention call (0: off)"}),
                "measure_heads": ("INT", {"default": 2, "min": 1, "max": 40,
                                          "tooltip": "Heads per measured call: the FP32 reference is slow"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "UnicaAttention"
    DESCRIPTION = ("Self-attention on UnicaAttention through optimized_attention_override. Put it after "
                   "PathchSageAttentionKJ: other calls (cross-attention) go to the override before it.")

    def patch(self, model, enabled, measure_every, measure_heads):
        if not enabled:
            return (model,)
        native_measurement.every, native_measurement.heads = measure_every, measure_heads
        patched = model.clone()
        options = patched.model_options["transformer_options"]
        previous = options.get("optimized_attention_override")
        if not getattr(previous, "unica", False):
            options["optimized_attention_override"] = native_override(previous)
        return (patched,)


NODE_CLASS_MAPPINGS = {"UnicaAttentionWanVideo": UnicaAttentionWanVideo, "UnicaAttentionNative": UnicaAttentionNative}
NODE_DISPLAY_NAME_MAPPINGS = {"UnicaAttentionWanVideo": "Unica Attention (WanVideo)",
                              "UnicaAttentionNative": "Unica Attention (native)"}
