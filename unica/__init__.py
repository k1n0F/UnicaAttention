"""
UnicaAttention: attention kernel made for the RTX 5080 (sm_120).

    out = unica.attention(q, k, v)  # [batch, heads, tokens, 128] bf16, like SDPA's default layout
"""

from __future__ import annotations

import math

import torch

from unica.cuda import module

HEAD_DIM = 128
WARP_SIZE = 32
ROWS_PER_WARP = 16  # must match csrc/flash.cu
LOG2E = 1.4426950408889634

# Every variant in csrc/flash.cu with the attention() signature, oldest first. All take the same grid
# and block; bench/compare.py times them side by side.
KERNELS = ("attention", "attention_smem")
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
    module("flash", **defines)[kernel](grid, warps * WARP_SIZE, q, k, v, out, n_queries, n_keys, scale_log2)
    return out
