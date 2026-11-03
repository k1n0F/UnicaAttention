"""
UnicaAttention: attention kernel made for the RTX 5080 (sm_120).

    out = unica.attention(q, k, v)  # [batch, heads, tokens, 128] bf16, like SDPA's default layout
"""

from __future__ import annotations

import ctypes
import math

import torch

from unica.cuda import module

HEAD_DIM = 128
WARP_SIZE = 32
ROWS_PER_WARP = 16  # these three must match csrc/flash.cu
Q_SCALE_BLOCK = 128
K_SCALE_BLOCK = 64
LOG2E = 1.4426950408889634

# Every variant in csrc/flash.cu, oldest first. All take the same grid and block; bench/compare.py
# times them side by side. The quantized ones trade exactness for speed: their error is measured
# against bench/emulate.py, which predicts it.
KERNELS = ("attention", "attention_smem", "attention_int8qk", "attention_int8pv")
QUANTIZED = {"attention_int8qk", "attention_int8pv"}
DEFAULT_KERNEL = "attention_smem"

# The tile shape csrc/flash.cu is built with when nothing overrides it (its UNICA_* macros).
TILE = {"WARPS_PER_BLOCK": 4, "KEYS_PER_TILE": 64, "MIN_BLOCKS_PER_SM": 1}


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel: str = DEFAULT_KERNEL,
    inputs: tuple | None = None,
    **tile: int,
) -> torch.Tensor:
    """softmax(q · kᵀ / sqrt(d)) · v, non-causal. q is [batch, heads, n_queries, 128]; k and v are
    [batch, heads, n_keys, 128], so cross-attention with a different n_keys works too.

    For tests and benchmarks: kernel picks a variant from KERNELS, and tile overrides entries of TILE,
    e.g. KEYS_PER_TILE=32; each combination is its own cubin. inputs is what kernel_inputs() returned for
    these q, k, v: the kernel then runs alone, without quantizing again."""
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}, expected one of {KERNELS}")
    if unknown := tile.keys() - TILE.keys():
        raise ValueError(f"unknown tile settings {sorted(unknown)}, expected some of {sorted(TILE)}")
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        raise TypeError(f"q, k, v must be bfloat16, got {q.dtype}, {k.dtype}, {v.dtype}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be on the GPU")
    if q.dim() != 4 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"q must be [batch, heads, tokens, {HEAD_DIM}], got {tuple(q.shape)}")
    if k.shape != v.shape or k.shape[:2] != q.shape[:2] or k.shape[-1] != HEAD_DIM:
        raise ValueError(f"k and v must be [batch, heads, n_keys, {HEAD_DIM}] like q, "
                         f"got q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    batch, heads, n_queries, _ = q.shape
    n_keys = k.shape[2]
    if n_keys == 0:
        raise ValueError("k and v must hold at least one token")

    # the kernel indexes plain [heads][tokens][128] arrays; for an already contiguous tensor this is free
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    defines = {f"UNICA_{key}": value for key, value in tile.items() if TILE[key] != value}
    warps = tile.get("WARPS_PER_BLOCK", TILE["WARPS_PER_BLOCK"])
    grid = (math.ceil(n_queries / (warps * ROWS_PER_WARP)), batch * heads)
    scale_log2 = HEAD_DIM**-0.5 * LOG2E
    if inputs is None:
        inputs = kernel_inputs(q, k, v, kernel)
    module("flash", **defines)[kernel](grid, warps * WARP_SIZE, *inputs, out, n_queries, n_keys, scale_log2)
    return out


def kernel_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel: str) -> tuple:
    """The arguments `kernel` takes before its output: q, k, v as they are, or quantized. q, k, v must be
    contiguous. Apart from attention() so that a benchmark can time quantization and kernel separately."""
    if kernel not in QUANTIZED:
        return q, k, v
    q8, q_scale = quantize_int8(q, None, Q_SCALE_BLOCK)
    k8, k_scale = quantize_int8(k, k.mean(-2, dtype=torch.float32), K_SCALE_BLOCK)  # softmax cancels the mean
    if kernel == "attention_int8qk":
        return q8, k8, q_scale, k_scale, v
    return q8, k8, q_scale, k_scale, *quantize_v(v)


QUANTIZE_THREADS = 256  # must match csrc/flash.cu
CHUNK = 8  # bf16 per thread in quantize_v


def quantize_int8(x: torch.Tensor, mean: torch.Tensor | None, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x − mean ≈ x8 · scale, one scale per `block` tokens over all of d. x is contiguous bf16
    [batch, heads, N, 128], mean float32 [batch, heads, 128] or None. Returns int8 [batch, heads, N, 128]
    and float32 scales [batch, heads, ceil(N / block)]. The scale never varies along d, the axis Q·Kᵀ sums
    over, so the int32 mma result is scaled once at the end. Bit for bit bench/emulate.py's quant()."""
    batch, heads, n, _ = x.shape
    x = x.contiguous()
    x8 = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(batch, heads, math.ceil(n / block), device=x.device)
    grid = (scale.shape[-1], batch * heads)
    mean_arg = mean.contiguous() if mean is not None else ctypes.c_void_p(0)
    module("flash")["quantize_int8"](grid, QUANTIZE_THREADS, x, mean_arg, x8, scale, n, block)
    return x8, scale


def quantize_v(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """v ≈ v8 · scale with one scale per channel over all tokens of a head: int8 [batch, heads, N, 128] and
    float32 [batch, heads, 128]. d is the axis P·V does not sum over, so the kernel applies the scale once,
    at the end. Bit for bit bench/emulate.py's V quantization."""
    v = v.contiguous()
    low, high = torch.aminmax(v, dim=-2)  # one pass, and |v| needs no copy of v
    scale = (torch.maximum(high, -low).float() / 127).clamp_min(1e-12)
    v8 = torch.empty(v.shape, dtype=torch.int8, device=v.device)
    n_chunks = v.numel() // CHUNK
    grid = math.ceil(n_chunks / QUANTIZE_THREADS)
    module("flash")["quantize_v"](grid, QUANTIZE_THREADS, v, scale, v8, v.shape[2], n_chunks)
    return v8, scale
