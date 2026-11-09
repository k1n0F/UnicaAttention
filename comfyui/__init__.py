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

RIFE VFI (Unica) (comfyui/rife.py) is comfyui-frame-interpolation's RIFE node with a faster loop (LOG M4.37).

Install: a directory junction from ComfyUI/custom_nodes/UnicaAttention to this folder. unica's cubins must
already be built (run the tests once): ComfyUI's shell has no nvcc.
"""

# pyright: reportMissingImports=false
# ComfyUI's modules (comfy, cv2, the frame-interpolation node...) exist only in ComfyUI's Python, which runs this file

from __future__ import annotations

import collections
import functools
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
from .rife import UnicaRIFE  # noqa: E402
from emulate import TODAY, emulate, errors, reference  # noqa: E402

MODE = "unica"
ERRORS_CSV = REPO / "build" / "attention_errors_v_modes.csv"  # each set of columns its own file (M5.3, M4.6, M4.7)
# while this file exists, every measured call also saves its q, k, v (the measured heads) to build/dumps/
DUMP_FLAG = REPO / "build" / "dumps.flag"
OUTLIER_KEYS = unica.PRESETS["outliers"]["V_OUTLIERS"]  # M4.25: keys per head v_scale "outliers" takes out of INT8
# attention_fp4pv as the emulator computes it (LOG M4.8, tests/test_fp4.py)
FP4PV = TODAY | dict(p_fmt="nvfp4", v_fmt="nvfp4", p_local=False, p_mul=1024.0, p_outer=1.0, v_outer="channel",
                     fp4_slots=True)
# The fast mode's kernels, measured themselves on the same real Q, K, V (M4.21): attention_fp4, and with Q smoothed per
# 512 queries (M4.20 emulated it below SageAttention 3 in 22 of 22 blocks)
KERNELS = {
    "accurate_per_channel": dict(V_TOKEN=0),  # M4.23: both V scales, whatever v_scale the node runs with
    "accurate_per_token": dict(V_TOKEN=1),
    "accurate_outliers": dict(V_OUTLIERS=OUTLIER_KEYS),  # M4.25
    "fp4": dict(kernel="attention_fp4"),
    "fp4_smooth512": dict(kernel="attention_fp4", SMOOTH_Q=512),
}
# Emulated next to them, so that the kernel can be checked against the emulator on real data
EMULATED = {
    "emu_fp4_both_smoothq512": FP4PV | dict(qk="nvfp4", smooth_q=512),
}
CSV_FIELDS = ["time", "tag", "v_scale", "routes", "node", "call", "step", "block", "batch", "tokens", "keys", "heads_measured", "unica_rel_l2", "unica_cos",
              "sage2_rel_l2", "sage2_cos", "sage3_rel_l2", "sage3_cos",
              *(f"{name}_{e}" for name in KERNELS for e in ("rel_l2", "cos")),
              *(f"{name}_rel_l2" for name in EMULATED), "q_channel_spread", "k_channel_spread", "v_offset", "q_offset",
              "unica_ms", "sage2_ms", "sage3_ms"]
HEAD_DIM = 128
log = logging.getLogger("UnicaAttention")


# Build settings of the accurate mode's kernel, set by the native node's v_scale: per_token is V_TOKEN=1 (M4.23),
# outliers V_OUTLIERS=OUTLIER_KEYS (M4.25: per channel, a head's largest keys added exactly), per_channel neither
V_SCALES = {name: unica.PRESETS[name] for name in ("per_token", "outliers", "per_channel")}
# The fast mode (M4.21): Q·Kᵀ and P·V in NVFP4, Q smoothed per 512 queries; real WAN 7.45 % vs SageAttention 3 8.65 %
FAST = unica.PRESETS["fast"]
SETTINGS: dict = {}  # what unica.attention gets in the node: the accurate mode with one of V_SCALES, or FAST


def unica_nhd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """unica.attention on the layout [batch, tokens, heads, 128] (WanVideoWrapper's and ComfyUI's), any float dtype in
    and out. No copy for bf16 and fp16: the kernels read the tensors through a transpose and write their output in the
    same layout (M4.17), and unica.attention converts fp16 to bf16 a few heads at a time (M4.31). Float32 is converted
    once each way, keeping the layout."""
    half = q.dtype if q.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16  # float32: once each way
    out = unica.attention(*(x.transpose(1, 2).to(half) for x in (q, k, v)), **SETTINGS)
    return out.transpose(1, 2).to(q.dtype).contiguous()  # no-ops for bf16 and fp16


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
        self.sigma, self.step, self.block = None, 0, None
        self.tag = ""
        self.routes: collections.Counter = collections.Counter()  # "unica" or why a call went elsewhere
        self.numbered = False  # whether the model's calls carry a block index

    def route(self, reason: str | None, q: torch.Tensor, k: torch.Tensor) -> None:
        """Counts where a call went; the first call passed on for each reason is logged with its shapes, so a run on a
        new model shows what keeps it from unica."""
        key = reason or "unica"
        if reason is not None and not self.routes[key]:
            log.info("%s [%s]: passed on (%s), q %s, k %s", self.node, self.tag, reason, tuple(q.shape), tuple(k.shape))
        self.routes[key] += 1

    def due(self, options: dict | None = None) -> bool:
        """Advances the call count and says whether this call is measured. With ComfyUI's transformer_options the count
        is step · blocks + block + 1 within one run: the step advances when sigma falls and restarts when it rises (a
        new run), so every run measures the same (step, block) pairs (M4.20 found the plain count running on across
        runs). Without them (WanVideoWrapper), a plain count. Off (every 0), it does nothing: reading sigma is a
        device-to-host copy, which waits for the GPU on every call and leaves it idle while Python queues the rest of
        the block (M4.28: ~1 s per WAN step)."""
        if self.every == 0:
            return False
        block, sigmas = (options or {}).get("block_index"), (options or {}).get("sigmas")
        if sigmas is None:
            self.calls += 1
            self.step = self.block = None
        elif block is None:
            if self.numbered:  # a call outside the numbered blocks (Lumina's context refiners): not counted
                return False
            self.calls += 1  # a model that numbers no blocks at all (LTX): the plain count
            self.step = self.block = None
        else:
            self.numbered = True
            sigma = float(sigmas.max())
            if self.sigma is None or self.step is None or sigma > self.sigma:
                self.step = 0
            elif sigma < self.sigma:
                self.step += 1
            self.sigma, self.block = sigma, block
            self.calls = self.step * options.get("total_blocks", 40) + block + 1
        return self.every > 0 and self.calls % self.every == 0

    def __call__(self, q, k, v, options: dict | None = None) -> None:
        """q, k, v in [batch, tokens, heads, 128]; options: the call's transformer_options, if any."""
        if self.due(options):
            self.measure(q, k, v)

    def measure(self, q, k, v) -> None:
        from sageattention import sageattn

        h = min(self.heads, q.shape[2])
        sub = [x[:, :, :h].contiguous() for x in (q, k, v)]
        bhnd = [x.transpose(1, 2).contiguous() for x in sub]
        ref = reference(*bhnd)  # FP32, [batch, heads, tokens, 128]
        row = {"time": time.strftime("%H:%M:%S"), "tag": self.tag,
               "v_scale": "fast" if SETTINGS == FAST else next(n for n, s in V_SCALES.items() if s == SETTINGS),
               "routes": " ".join(f"{key}={n}" for key, n in sorted(self.routes.items())),
               "node": self.node, "call": self.calls, "step": self.step,
               "block": self.block, "batch": q.shape[0],
               "tokens": q.shape[1], "keys": k.shape[1], "heads_measured": h}
        mine, row["unica_ms"] = timed(unica_nhd, *sub)
        sage2, row["sage2_ms"] = timed(lambda *x: sageattn(*(t.clone() for t in x), tensor_layout="NHD"), *sub)
        results = {"unica": mine.transpose(1, 2), "sage2": sage2.transpose(1, 2)}
        if (sage3 := sage3_bhnd()) is not None:
            results["sage3"], row["sage3_ms"] = timed(sage3, *bhnd)
        for name, settings in KERNELS.items():
            results[name] = unica.attention(*(x.bfloat16() for x in bhnd), **settings)
        for name, out in results.items():
            e = errors(out.float(), ref)
            row[f"{name}_rel_l2"], row[f"{name}_cos"] = e["rel_l2"], e["cos"]
        for name, config in EMULATED.items():
            row[f"{name}_rel_l2"] = errors(emulate(*bhnd, **config), ref)["rel_l2"]
        row["q_channel_spread"], row["k_channel_spread"] = channel_spread(bhnd[0]), channel_spread(bhnd[1])
        row["v_offset"], row["q_offset"] = v_offset(bhnd[2]), v_offset(bhnd[0])
        if DUMP_FLAG.exists():  # the measured q, k, v themselves, to take an error apart offline
            DUMP_FLAG.with_suffix("").mkdir(exist_ok=True)
            name = f"{self.tag or self.node}_call{self.calls}_{time.monotonic_ns()}.pt"
            torch.save({"q": bhnd[0].cpu(), "k": bhnd[1].cpu(), "v": bhnd[2].cpu(), "row": row},
                       DUMP_FLAG.with_suffix("") / name)
        ERRORS_CSV.parent.mkdir(exist_ok=True)
        new = not ERRORS_CSV.exists()
        with ERRORS_CSV.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        log.info("%s call %d, %d tokens: rel L2 unica %.3f %%, SageAttention 2 %.3f %%%s; %s; emulated %s; "
                 "channel spread q %.2f k %.2f, v offset %.3f", self.node, self.calls,
                 q.shape[1], 100 * row["unica_rel_l2"], 100 * row["sage2_rel_l2"],
                 f", SageAttention 3 {100 * row['sage3_rel_l2']:.3f} %" if "sage3_rel_l2" in row else "",
                 ", ".join(f"{name} {100 * row[f'{name}_rel_l2']:.3f} %" for name in KERNELS),
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


# The kernel both nodes run, chosen by these optional inputs. Both set the one SETTINGS: in a workflow with both
# nodes, the one that ran last decides.
KERNEL_INPUTS = {
    "v_scale": (list(V_SCALES), {"default": "per_token",
                                 "tooltip": "V's INT8 scale: per token (folded into P, the most accurate; ~11 % slower "
                                            "kernel), outliers (experimental: per channel without each head's 8 largest "
                                            "keys, added exactly: Z-Image's two extreme tokens; ~1 ms more quantization) "
                                            "or per channel (the earlier path)"}),
    "vram_reserve_mb": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 64,
                                "tooltip": "VRAM left free for the desktop and other programs; 0: auto, the larger of "
                                           "1 GiB and a tenth of the card. A call that would reach it runs a few heads "
                                           "at a time, at most a few % slower. Past the VRAM, Windows moves memory to "
                                           "system RAM and a render slows down many times over"}),
    "mode": (["accurate", "fast"], {"default": "accurate",
                                    "tooltip": "accurate: INT8 Q·Kᵀ and P·V, below SageAttention 2's error. fast: NVFP4 "
                                               "(attention_fp4), SageAttention 3's error class and faster than it; "
                                               "v_scale does not apply"}),
}


def choose_kernel(v_scale: str, mode: str, vram_reserve_mb: int) -> None:
    SETTINGS.clear()
    SETTINGS.update(FAST if mode == "fast" else V_SCALES[v_scale])
    unica.VRAM_RESERVE = vram_reserve_mb * 2**20 if vram_reserve_mb else None


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
            },
            "optional": KERNEL_INPUTS,  # optional, so that workflows saved before it still validate
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "UnicaAttention"
    DESCRIPTION = "Self-attention of the WAN model on UnicaAttention (INT8 Q·Kᵀ, u8 × s8 P·V, sm_120)"

    def patch(self, model, enabled, measure_every, measure_heads, v_scale="per_token", mode="accurate",
              vram_reserve_mb=0):
        if not enabled or checked().status == unica.check.UNSUPPORTED:
            return (model,)
        choose_kernel(v_scale, mode, vram_reserve_mb)
        if not wrap_attention() and not any(getattr(getattr(m, "attention", None), "unica", False)
                                            for n, m in sys.modules.items() if n.endswith("wanvideo.modules.model")):
            raise RuntimeError("ComfyUI-WanVideoWrapper's wanvideo.modules.model is not loaded")
        measurement.every, measurement.heads = measure_every, measure_heads
        patched = model.clone()
        patched.model_options["transformer_options"]["attention_mode_override"] = {
            "mode": MODE, "start_step": 0, "end_step": 10000, "verbose": False}
        return (patched,)


# the positional parameters of ComfyUI's attention functions after q, k, v (comfy/ldm/modules/attention.py)
POSITIONAL = ("heads", "mask", "attn_precision", "skip_reshape", "skip_output_reshape")


FAILED: list = []  # the error, once a call to unica has failed: every call after it goes to the model's own attention


@functools.cache
def checked() -> unica.check.Compatibility:
    """unica.compatibility(), logged once: if unica cannot run here, the nodes leave the model as it is."""
    status = unica.compatibility()
    if status.status == unica.check.UNSUPPORTED:
        log.warning("%s\nThe Unica nodes leave the model's own attention in place.", status)
    elif status.status == unica.check.UNTESTED:
        log.warning("%s", status)
    return status


def native_override(previous):
    """An optimized_attention_override for ComfyUI's own models. ComfyUI calls it as (func, q, k, v, heads=, ...):
    q, k, v are [batch, tokens, heads · dim], or [batch, heads, tokens, dim] with skip_reshape. Self-attention
    with head_dim 128, as many keys as queries, no mask and the default scale goes to unica; everything else
    to `previous` (the override that was on the model), or to func."""

    def override(func, q, k, v, *args, **kwargs):
        # WAN passes heads by name, most other models by position (Flux, LTX, Qwen-Image, Lumina...): read both
        named = dict(zip(POSITIONAL, args)) | kwargs
        heads = named.get("heads")
        reason = ("unica failed" if FAILED else "positional arguments" if len(args) > len(POSITIONAL) else
                  "no heads" if heads is None else
                  "mask" if named.get("mask") is not None else "not on the GPU" if not q.is_cuda else None)
        if reason is None:
            skip_reshape = named.get("skip_reshape", False)
            dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
            tokens = (lambda x: x.shape[2]) if skip_reshape else (lambda x: x.shape[1])
            reason = (f"head size {dim}" if dim != HEAD_DIM else
                      "cross-attention" if tokens(k) != tokens(q) else
                      "scale" if named.get("scale") not in (None, HEAD_DIM**-0.5) else None)
        native_measurement.route(reason, q, k)
        if reason is not None:  # anything unusual: as before
            return previous(func, q, k, v, *args, **kwargs) if previous else func(q, k, v, *args, **kwargs)
        skip_reshape, skip_output_reshape = named.get("skip_reshape", False), named.get("skip_output_reshape", False)
        kwargs = named
        # [batch, tokens, heads, 128], WanVideoWrapper's layout, as views
        nhd = [x.transpose(1, 2) if skip_reshape else x.view(x.shape[0], x.shape[1], heads, dim) for x in (q, k, v)]
        try:
            out = unica_nhd(*nhd)
        except RuntimeError as error:  # e.g. a driver that cannot load the cubins: this call and every later one as before
            if not FAILED:
                log.error("unica failed, the model's own attention takes over: %s", error)
            FAILED.append(error)
            return previous(func, q, k, v, *args, **kwargs) if previous else func(q, k, v, *args, **kwargs)
        native_measurement(*nhd, options=kwargs.get("transformer_options"))
        return out.transpose(1, 2) if skip_output_reshape else out.view(out.shape[0], out.shape[1], heads * dim)

    override.unica = True
    return override


def fused_rope_forward(module, original):
    """M4.18–M4.19: a forward for ComfyUI's WanSelfAttention (comfy/ldm/wan/model.py, ComfyUI 0.42) that hands q and k
    to unica before RoPE, with the model's own table: the quantization rotates them on the way in, so apply_rope1's
    pass over q and k (~1.9 ms per call) goes. Everything else as the original: q = norm_q(q(x)), k = norm_k(k(x)),
    v = v(x), out = o(attention). An attn1_patch (it wants the rotated q and k), a head size other than 128 or a table
    that does not fit go to the original forward."""

    def forward(x, freqs, transformer_options={}):  # noqa: B006 (the original's signature)
        b, s, n, d = *x.shape[:2], module.num_heads, module.head_dim
        if (transformer_options.get("patches", {}).get("attn1_patch") or freqs is None or d != HEAD_DIM
                or not x.is_cuda or freqs.dim() != 6 or freqs.shape[1] != s):
            return original(x, freqs, transformer_options=transformer_options)
        q = module.norm_q(module.q(x)).view(b, s, n, d)
        k = module.norm_k(module.k(x)).view(b, s, n, d)
        v = module.v(x).view(b, s, n, d)
        # the measurement compares on the rotated q and k: rotate them only for a call it will measure
        if native_measurement.due(transformer_options):
            from comfy.ldm.flux.math import apply_rope1
            native_measurement.measure(apply_rope1(q, freqs), apply_rope1(k, freqs), v)
        out = unica.attention(*(t.transpose(1, 2).to(torch.bfloat16) for t in (q, k, v)), rope=freqs, **SETTINGS)
        return module.o(out.transpose(1, 2).reshape(b, s, n * d).to(x.dtype))

    forward.unica = True
    return forward


def patch_self_attention_forward(patched) -> int:
    """fused_rope_forward on every WanSelfAttention of the model, as object patches of this clone only. Returns how many."""
    blocks = patched.get_model_object("diffusion_model.blocks")
    count = 0
    for i in range(len(blocks)):
        name = f"diffusion_model.blocks.{i}.self_attn"
        module = patched.get_model_object(name)
        if type(module).__name__ != "WanSelfAttention":  # cross-attention classes derive from it: leave them
            continue
        original = patched.object_patches.get(f"{name}.forward", module.forward)
        if getattr(original, "unica", False):
            continue
        patched.add_object_patch(f"{name}.forward", fused_rope_forward(module, original))
        count += 1
    return count


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
            },
            "optional": {  # optional, so that workflows saved before it still validate
                "fuse_rope": ("BOOLEAN", {"default": False,
                                          "tooltip": "WAN: RoPE inside unica's quantization instead of ComfyUI's own pass "
                                                     "(~2 ms less per call). Patches WanSelfAttention.forward"}),
                "measure_tag": ("STRING", {"default": "",
                                           "tooltip": "Written into the measurement CSV, e.g. the model's name"}),
                **KERNEL_INPUTS,
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "UnicaAttention"
    DESCRIPTION = ("Self-attention on UnicaAttention through optimized_attention_override. Put it after "
                   "PathchSageAttentionKJ: other calls (cross-attention) go to the override before it.")

    def patch(self, model, enabled, measure_every, measure_heads, fuse_rope=False, measure_tag="",
              v_scale="per_token", mode="accurate", vram_reserve_mb=0):
        if not enabled or checked().status == unica.check.UNSUPPORTED:
            return (model,)
        choose_kernel(v_scale, mode, vram_reserve_mb)
        native_measurement.every, native_measurement.heads = measure_every, measure_heads
        native_measurement.tag = measure_tag
        native_measurement.routes.clear()
        native_measurement.sigma, native_measurement.numbered = None, False  # a new model starts its own count
        patched = model.clone()
        options = patched.model_options["transformer_options"]
        previous = options.get("optimized_attention_override")
        if not getattr(previous, "unica", False):
            options["optimized_attention_override"] = native_override(previous)
        if fuse_rope:
            log.info("fuse_rope: %d self-attention blocks patched", patch_self_attention_forward(patched))
        return (patched,)


NODE_CLASS_MAPPINGS = {"UnicaAttentionWanVideo": UnicaAttentionWanVideo, "UnicaAttentionNative": UnicaAttentionNative,
                       "UnicaRIFE": UnicaRIFE}
NODE_DISPLAY_NAME_MAPPINGS = {"UnicaAttentionWanVideo": "Unica Attention (WanVideo)",
                              "UnicaAttentionNative": "Unica Attention (native)", "UnicaRIFE": "RIFE VFI (Unica)"}
