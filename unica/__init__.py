"""
UnicaAttention: attention kernel made for the RTX 5080 (sm_120).

    out = unica.attention(q, k, v)  # [batch, heads, tokens, 128] bf16, like SDPA's default layout
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

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
KERNELS = ("attention", "attention_smem", "attention_int8qk")
QUANTIZED = {"attention_int8qk"}
DEFAULT_KERNEL = "attention_smem"

# The tile shape csrc/flash.cu is built with when nothing overrides it (its UNICA_* macros).
TILE = {"WARPS_PER_BLOCK": 4, "KEYS_PER_TILE": 64, "MIN_BLOCKS_PER_SM": 1}


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel: str = DEFAULT_KERNEL, **tile: int
) -> torch.Tensor:
    """softmax(q · kᵀ / sqrt(d)) · v, non-causal. q is [batch, heads, n_queries, 128]; k and v are
    [batch, heads, n_keys, 128], so cross-attention with a different n_keys works too.

    For tests and benchmarks: kernel picks a variant from KERNELS, and tile overrides entries of TILE,
    e.g. KEYS_PER_TILE=32; each combination is its own cubin."""
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
    launch = module("flash", **defines)[kernel]
    if kernel == "attention_int8qk":
        # Quantized in torch, a separate pass before the kernel. Fusing it into the kernel is a later step.
        q8, q_scale = quantize_int8(q.float(), Q_SCALE_BLOCK)
        k = k.float()
        k8, k_scale = quantize_int8(k - k.mean(-2, keepdim=True), K_SCALE_BLOCK)  # softmax cancels the mean
        launch(grid, warps * WARP_SIZE, q8, k8, q_scale, k_scale, v, out, n_queries, n_keys, scale_log2)
    else:
        launch(grid, warps * WARP_SIZE, q, k, v, out, n_queries, n_keys, scale_log2)
    return out


def quantize_int8(x: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x ≈ x8 · scale, one scale per `block` tokens over all of d: [..., N, D] float -> int8 [..., N, D]
    and float32 [..., ceil(N / block)]. The scale never varies along d, the axis Q·Kᵀ sums over, so the
    int32 mma result is scaled once at the end. Same rounding as bench/emulate.py."""
    n = x.shape[-2]
    token_max = F.pad(x.abs().amax(-1), (0, -n % block))
    scale = (token_max.unflatten(-1, (-1, block)).amax(-1) / 127).clamp_min(1e-12)
    per_token = scale.repeat_interleave(block, -1)[..., :n, None]
    return (x / per_token).round().clamp(-127, 127).to(torch.int8), scale.contiguous()
