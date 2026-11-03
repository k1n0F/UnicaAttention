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
WARPS_PER_BLOCK = 4  # these three must match csrc/flash.cu
ROWS_PER_BLOCK = WARPS_PER_BLOCK * 16
LOG2E = 1.4426950408889634


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(q · kᵀ / sqrt(d)) · v, non-causal. q is [batch, heads, n_queries, 128]; k and v are
    [batch, heads, n_keys, 128], so cross-attention with a different n_keys works too."""
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
    grid = (math.ceil(n_queries / ROWS_PER_BLOCK), batch * heads)
    scale_log2 = HEAD_DIM**-0.5 * LOG2E
    module("flash")["attention"](grid, WARPS_PER_BLOCK * WARP_SIZE, q, k, v, out, n_queries, n_keys, scale_log2)
    return out
